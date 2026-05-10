/*
 * Brain Wallet v2 puzzle-only kernel -- Apple Metal port (Phase 11, v1.4.0).
 *
 * 1:1 port of src/gpu/v2/brain_wallet_v2.cu's puzzle-only path. Implements
 * SHA-256 + 4-limb 256-bit mask compare against the puzzle target list. The
 * Metal Shading Language (MSL) does not allow recursion or runtime constant-
 * memory (the MTLBuffer that backs `puzzle_targets` is the analogue), so the
 * structures are passed in as buffers.
 *
 * The CUDA kernel is the source of truth; this file MUST produce byte-equal
 * output for every passphrase in the test corpus. The CPU reference at
 * src/gpu/v2/address_derive_cpu.hpp::internal::sha256 is what BOTH kernels
 * are validated against.
 *
 * Design notes:
 *   - Uses 32-bit uints throughout for the SHA-256 state. Metal supports
 *     uint64_t in MSL 2.4+, used for the 4-limb mask compare.
 *   - Passphrase batch is two parallel buffers: `passphrases` (packed
 *     bytes) and `offsets[i]`/`lengths[i]` so each thread can locate
 *     its own passphrase without scanning.
 *   - V2MatchRecord layout matches the CUDA struct exactly (24 bytes).
 *   - Match counter is a single uint atomic; output buffer is bounded
 *     to V2_MAX_MATCHES_PER_BATCH on the host side (4096).
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Mirrors brain_wallet_v2.hpp::PuzzleTarget (72 bytes natural alignment).
// Field-for-field layout matches CUDA host (no pack pragma; explicit pad
// makes the layout identical across CUDA host / CUDA device / MSL).
struct PuzzleTarget {
    ushort   puzzle_n;          // 2
    uchar    reserved[2];       // 2
    uint     _pad;              // 4-byte explicit pad to 8-byte align ulongs
    ulong    low_mask[4];       // 32
    ulong    low_value[4];      // 32
};

// Mirrors brain_wallet_v2.hpp::V2MatchRecord (24 bytes).
struct V2MatchRecord {
    uint    pp_idx;
    ulong   weak_seed;
    ushort  puzzle_n;
    uchar   scheme_id;
    uchar   addr_type;
    uchar   kind;
    uchar   reserved[3];
};

constant uint kMaxMatches = 4096;

// --- SHA-256 ---------------------------------------------------------------

constant uint K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

inline uint rotr32_metal(uint x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void sha256_compress(thread uint state[8], thread const uint W_in[16]) {
    uint W[64];
    for (int i = 0; i < 16; ++i) W[i] = W_in[i];
    for (int i = 16; i < 64; ++i) {
        uint s0 = rotr32_metal(W[i - 15], 7) ^ rotr32_metal(W[i - 15], 18)
                ^ (W[i - 15] >> 3);
        uint s1 = rotr32_metal(W[i - 2], 17) ^ rotr32_metal(W[i - 2], 19)
                ^ (W[i - 2] >> 10);
        W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }
    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        uint S1 = rotr32_metal(e, 6) ^ rotr32_metal(e, 11) ^ rotr32_metal(e, 25);
        uint ch = (e & f) ^ ((~e) & g);
        uint T1 = h + S1 + ch + K256[i] + W[i];
        uint S0 = rotr32_metal(a, 2) ^ rotr32_metal(a, 13) ^ rotr32_metal(a, 22);
        uint mj = (a & b) ^ (a & c) ^ (b & c);
        uint T2 = S0 + mj;
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

// SHA-256 over a single block (passphrases capped at 56 bytes for the
// puzzle-only kernel; longer hashes use the multi-block path which falls
// outside the high-throughput path the GPU is optimized for).
static void sha256_short(device const uchar* msg, uint len,
                         thread uint state[8])
{
    state[0] = 0x6a09e667u; state[1] = 0xbb67ae85u;
    state[2] = 0x3c6ef372u; state[3] = 0xa54ff53au;
    state[4] = 0x510e527fu; state[5] = 0x9b05688cu;
    state[6] = 0x1f83d9abu; state[7] = 0x5be0cd19u;

    uchar block[64] = {0};
    for (uint i = 0; i < len; ++i) block[i] = msg[i];
    block[len] = 0x80u;
    ulong bit_len = (ulong)len * 8;
    block[56] = (uchar)(bit_len >> 56);
    block[57] = (uchar)(bit_len >> 48);
    block[58] = (uchar)(bit_len >> 40);
    block[59] = (uchar)(bit_len >> 32);
    block[60] = (uchar)(bit_len >> 24);
    block[61] = (uchar)(bit_len >> 16);
    block[62] = (uchar)(bit_len >>  8);
    block[63] = (uchar)(bit_len      );

    uint W[16];
    for (int i = 0; i < 16; ++i) {
        W[i] =  ((uint)block[i*4    ] << 24)
             |  ((uint)block[i*4 + 1] << 16)
             |  ((uint)block[i*4 + 2] <<  8)
             |  ((uint)block[i*4 + 3]      );
    }
    sha256_compress(state, W);
}

// Pack the 32-byte big-endian SHA-256 output into 4 little-endian-by-limb
// uint64s for the puzzle mask compare. Limb 0 is the lowest-index byte
// pair, matching the CUDA layout.
static void hash_to_limbs(const thread uint state[8], thread ulong limbs[4]) {
    // The big-endian word ordering: state[0]..state[7] = bytes [0..31].
    // limb[0] = bytes [24..31] = (state[6] << 32) | state[7] (BE)
    limbs[0] = (((ulong)state[6]) << 32) | (ulong)state[7];
    limbs[1] = (((ulong)state[4]) << 32) | (ulong)state[5];
    limbs[2] = (((ulong)state[2]) << 32) | (ulong)state[3];
    limbs[3] = (((ulong)state[0]) << 32) | (ulong)state[1];
}

// ---------------------------------------------------------------------------
// Multi-scheme derivations (Phase 3, ported to Metal in Phase 11 finish).
// Match the SHA-256-only schemes in brain_wallet_v2.cu byte-for-byte.
// ---------------------------------------------------------------------------

// SHA-256 over a multi-block message (up to 256 bytes; reasonable for
// brain-wallet inputs). Required for schemes that double-hash, append, or
// concatenate the passphrase.
static void sha256_long(thread const uchar* msg, uint len, thread uint state[8]) {
    state[0] = 0x6a09e667u; state[1] = 0xbb67ae85u;
    state[2] = 0x3c6ef372u; state[3] = 0xa54ff53au;
    state[4] = 0x510e527fu; state[5] = 0x9b05688cu;
    state[6] = 0x1f83d9abu; state[7] = 0x5be0cd19u;

    uint off = 0;
    uint W[16];
    while (off + 64 <= len) {
        for (int i = 0; i < 16; ++i) {
            W[i] =  ((uint)msg[off + i*4    ] << 24)
                 |  ((uint)msg[off + i*4 + 1] << 16)
                 |  ((uint)msg[off + i*4 + 2] <<  8)
                 |  ((uint)msg[off + i*4 + 3]      );
        }
        sha256_compress(state, W);
        off += 64;
    }
    uchar block[64];
    for (uint i = 0; i < 64; ++i) block[i] = 0;
    uint rem = len - off;
    for (uint i = 0; i < rem; ++i) block[i] = msg[off + i];
    block[rem] = 0x80u;
    ulong bit_len = (ulong)len * 8;
    if (rem + 1 + 8 > 64) {
        for (int i = 0; i < 16; ++i) {
            W[i] =  ((uint)block[i*4    ] << 24)
                 |  ((uint)block[i*4 + 1] << 16)
                 |  ((uint)block[i*4 + 2] <<  8)
                 |  ((uint)block[i*4 + 3]      );
        }
        sha256_compress(state, W);
        for (int i = 0; i < 56; ++i) block[i] = 0;
    }
    block[56] = (uchar)(bit_len >> 56);
    block[57] = (uchar)(bit_len >> 48);
    block[58] = (uchar)(bit_len >> 40);
    block[59] = (uchar)(bit_len >> 32);
    block[60] = (uchar)(bit_len >> 24);
    block[61] = (uchar)(bit_len >> 16);
    block[62] = (uchar)(bit_len >>  8);
    block[63] = (uchar)(bit_len      );
    for (int i = 0; i < 16; ++i) {
        W[i] =  ((uint)block[i*4    ] << 24)
             |  ((uint)block[i*4 + 1] << 16)
             |  ((uint)block[i*4 + 2] <<  8)
             |  ((uint)block[i*4 + 3]      );
    }
    sha256_compress(state, W);
}

static void state_to_bytes(thread const uint state[8], thread uchar out[32]) {
    for (int i = 0; i < 8; ++i) {
        out[i*4    ] = (uchar)(state[i] >> 24);
        out[i*4 + 1] = (uchar)(state[i] >> 16);
        out[i*4 + 2] = (uchar)(state[i] >>  8);
        out[i*4 + 3] = (uchar)(state[i]      );
    }
}

// Derive priv via the chosen scheme; writes 32 bytes to out_state[].
static void derive_scheme(uint scheme_id,
                          device const uchar* pw, uint pw_len,
                          thread uint out_state[8])
{
    thread uchar priv_buf[256];
    if (pw_len > 200) pw_len = 200;
    for (uint i = 0; i < pw_len; ++i) priv_buf[i] = pw[i];

    if (scheme_id == 0) {        // SHA256_PW
        sha256_long(priv_buf, pw_len, out_state);
    } else if (scheme_id == 1) { // SHA256_SHA256_PW
        sha256_long(priv_buf, pw_len, out_state);
        thread uchar inner[32];
        state_to_bytes(out_state, inner);
        sha256_long(inner, 32, out_state);
    } else if (scheme_id == 2) { // SHA256_PW_NEWLINE
        priv_buf[pw_len] = 0x0a;
        sha256_long(priv_buf, pw_len + 1, out_state);
    } else if (scheme_id == 3) { // SHA256_PW_PW
        // Cap at 128 to match CUDA. priv_buf is 256 bytes, so 128*2 fits exactly.
        if (pw_len > 128) pw_len = 128;
        for (uint i = 0; i < pw_len; ++i) priv_buf[pw_len + i] = pw[i];
        sha256_long(priv_buf, pw_len * 2, out_state);
    } else if (scheme_id == 4) { // SHA256_SHA256_PW_PW
        sha256_long(priv_buf, pw_len, out_state);
        thread uchar inner[32];
        state_to_bytes(out_state, inner);
        if (pw_len > 128) pw_len = 128;
        thread uchar combo[32 + 128];
        for (int i = 0; i < 32; ++i) combo[i] = inner[i];
        for (uint i = 0; i < pw_len; ++i) combo[32 + i] = pw[i];
        sha256_long(combo, 32 + pw_len, out_state);
    } else if (scheme_id == 5) { // SHA256_ITER_16
        sha256_long(priv_buf, pw_len, out_state);
        thread uchar buf[32];
        for (int i = 0; i < 15; ++i) {
            state_to_bytes(out_state, buf);
            sha256_long(buf, 32, out_state);
        }
    } else {
        // Unknown / SHA-512 schemes (S7/S8): the Metal port has not yet
        // implemented a device SHA-512. Rather than silently fall back to
        // SHA-256 (which would produce wrong priv keys and confuse the
        // user with apparent "no hits" when they ran HMAC_SHA512_PW), we
        // poison the output state to a value that cannot match any
        // realistic puzzle target. The host orchestrator MUST also reject
        // these scheme bits before dispatch and surface a clear error.
        out_state[0] = 0xFFFFFFFFu; out_state[1] = 0xFFFFFFFFu;
        out_state[2] = 0xFFFFFFFFu; out_state[3] = 0xFFFFFFFFu;
        out_state[4] = 0xFFFFFFFFu; out_state[5] = 0xFFFFFFFFu;
        out_state[6] = 0xFFFFFFFFu; out_state[7] = 0xFFFFFFFFu;
    }
}

// ---------------------------------------------------------------------------
// Public kernel: per-thread, derive via `scheme_id` and check puzzle targets.
// ---------------------------------------------------------------------------

kernel void v2_puzzle_only_multi_scheme(
    device const uchar*           passphrases       [[buffer(0)]],
    device const uint*            offsets           [[buffer(1)]],
    device const uint*            lengths           [[buffer(2)]],
    device const PuzzleTarget*    targets           [[buffer(3)]],
    constant uint&                target_count      [[buffer(4)]],
    device V2MatchRecord*         matches           [[buffer(5)]],
    device atomic_uint*           match_count       [[buffer(6)]],
    constant uint&                pw_count          [[buffer(7)]],
    constant uint&                scheme_id         [[buffer(8)]],
    uint                          gid               [[thread_position_in_grid]])
{
    if (gid >= pw_count) return;

    device const uchar* pw = passphrases + offsets[gid];
    uint len = lengths[gid];

    thread uint state[8];
    derive_scheme(scheme_id, pw, len, state);

    thread ulong limbs[4];
    hash_to_limbs(state, limbs);

    for (uint ti = 0; ti < target_count; ++ti) {
        bool ok = true;
        for (int j = 0; j < 4; ++j) {
            if ((limbs[j] & targets[ti].low_mask[j]) != targets[ti].low_value[j]) {
                ok = false;
                break;
            }
        }
        if (ok) {
            uint slot = atomic_fetch_add_explicit(match_count, 1u,
                                                   memory_order_relaxed);
            if (slot < kMaxMatches) {
                matches[slot].pp_idx    = gid;
                matches[slot].weak_seed = 0;
                matches[slot].puzzle_n  = targets[ti].puzzle_n;
                matches[slot].scheme_id = (uchar)scheme_id;
                matches[slot].addr_type = 0;
                matches[slot].kind      = 1;   // PUZZLE_KEY_HIT
            }
        }
    }
}

// Backwards-compatible single-scheme entry point (Phase 11 first cut).
kernel void v2_puzzle_only_sha256_pw(
    device const uchar*           passphrases       [[buffer(0)]],
    device const uint*            offsets           [[buffer(1)]],
    device const uint*            lengths           [[buffer(2)]],
    device const PuzzleTarget*    targets           [[buffer(3)]],
    constant uint&                target_count      [[buffer(4)]],
    device V2MatchRecord*         matches           [[buffer(5)]],
    device atomic_uint*           match_count       [[buffer(6)]],
    constant uint&                pw_count          [[buffer(7)]],
    uint                          gid               [[thread_position_in_grid]])
{
    if (gid >= pw_count) return;

    uint off = offsets[gid];
    uint len = lengths[gid];
    if (len > 55) return;

    thread uint state[8];
    sha256_short(passphrases + off, len, state);

    thread ulong limbs[4];
    hash_to_limbs(state, limbs);

    for (uint ti = 0; ti < target_count; ++ti) {
        bool ok = true;
        for (int j = 0; j < 4; ++j) {
            if ((limbs[j] & targets[ti].low_mask[j]) != targets[ti].low_value[j]) {
                ok = false;
                break;
            }
        }
        if (ok) {
            uint slot = atomic_fetch_add_explicit(match_count, 1u,
                                                   memory_order_relaxed);
            if (slot < kMaxMatches) {
                matches[slot].pp_idx    = gid;
                matches[slot].weak_seed = 0;
                matches[slot].puzzle_n  = targets[ti].puzzle_n;
                matches[slot].scheme_id = 0;
                matches[slot].addr_type = 0;
                matches[slot].kind      = 1;
            }
        }
    }
}
