/**
 * Device-side hash primitives for the Brain Wallet v2 GPU pipeline.
 *
 * All four hashes the v2 paths need (SHA-256, SHA-512, RIPEMD-160, MD5,
 * SHA-1) are header-only __device__ functions so each kernel TU can pull
 * in only what it uses. The CPU references at
 *   src/gpu/v2/address_derive_cpu.hpp  (sha256, ripemd160, hash160)
 *   src/gpu/v2/sha512_cpu.hpp          (sha512, hmac, pbkdf2)
 * are the authoritative byte-for-byte spec.
 *
 * No external libraries; no shared cudaMemcpy state; one input -> one
 * output buffer per call. All loops are unrolled or constant-bounded so
 * register usage stays predictable across kernels.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "../hash_rounds.cuh"

namespace collider {
namespace gpu {
namespace v2 {
namespace device {

// ===========================================================================
// SHA-256 (FIPS 180-4)
//
// One of three SHA-256 implementations (also in fused_pipeline.cu
// SHA256_K + sha256_compress, and mega_fused_kernel.cu MEGA_SHA256_K +
// mega_sha256_*). The K[64] tables are bit-identical FIPS values; the
// per-TU __constant__ memory regions on CUDA mean a shared anchor
// would require extern __constant__ + nvlink fixups without reducing
// runtime constant-cache use. The implementations differ in unroll
// strategy and inlining policy: this one is __noinline__ and
// variable-length for the v2 brain-wallet dispatcher (S1-S5 derivation
// schemes) that calls SHA-256 multiple times per passphrase.
// See fused_pipeline.cu for the audit context.
// ===========================================================================

#include "../sha256_k_constants.cuh"

__device__ __constant__ static const uint32_t kSha256_K[64] =
    COLLIDER_SHA256_K_INIT;

// SHA-256 right-rotate: forward to the canonical primitive in
// hash_rounds.cuh. Kept as a TU-local alias (rather than a using-decl)
// because this is a header -- a using inside the header would leak the
// name into every TU that includes it. The sha512_rotr below is a
// distinct 64-bit primitive and stays local.
__device__ __forceinline__ uint32_t sha256_rotr(uint32_t x, int n) {
    return ::collider::gpu::sha256::rotr(x, n);
}

// `__noinline__` for the same reason as sha512_compress below: the W[64]
// schedule is 256 bytes per call. Inlined into the v2 brain-wallet kernels
// (which already carry SHA-512 state, HMAC pads, and v2 match-record
// buffers) it spills to local memory and tanks occupancy on Ampere+. The
// non-v2 hot paths (kangaroo / pool DP detection) call SHA-256 from
// dedicated .cu files that don't include this header, so they keep their
// own inlined SHA-256 and aren't affected.
__device__ __noinline__ static void sha256_compress(uint32_t state[8], const uint32_t W_in[16]) {
    uint32_t W[64];
    #pragma unroll
    for (int i = 0; i < 16; ++i) W[i] = W_in[i];
    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = sha256_rotr(W[i-15], 7) ^ sha256_rotr(W[i-15], 18) ^ (W[i-15] >> 3);
        uint32_t s1 = sha256_rotr(W[i-2], 17) ^ sha256_rotr(W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t T1 = h + S1 + ch + kSha256_K[i] + W[i];
        uint32_t S0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t T2 = S0 + mj;
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ static void sha256(const uint8_t* msg, uint32_t len, uint8_t out[32]) {
    uint32_t state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    uint64_t bit_len = (uint64_t)len * 8;
    uint8_t  block[64];
    uint32_t off = 0;
    while (off + 64 <= len) {
        uint32_t W[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            W[i] =  ((uint32_t)msg[off + i*4    ] << 24)
                 |  ((uint32_t)msg[off + i*4 + 1] << 16)
                 |  ((uint32_t)msg[off + i*4 + 2] <<  8)
                 |  ((uint32_t)msg[off + i*4 + 3]      );
        }
        sha256_compress(state, W);
        off += 64;
    }
    uint32_t rem = len - off;
    for (uint32_t i = 0; i < rem; ++i) block[i] = msg[off + i];
    block[rem] = 0x80u;
    for (uint32_t i = rem + 1; i < 64; ++i) block[i] = 0;
    if (rem + 1 + 8 > 64) {
        uint32_t W[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            W[i] =  ((uint32_t)block[i*4    ] << 24)
                 |  ((uint32_t)block[i*4 + 1] << 16)
                 |  ((uint32_t)block[i*4 + 2] <<  8)
                 |  ((uint32_t)block[i*4 + 3]      );
        }
        sha256_compress(state, W);
        for (int i = 0; i < 56; ++i) block[i] = 0;
    }
    block[56] = (uint8_t)(bit_len >> 56);
    block[57] = (uint8_t)(bit_len >> 48);
    block[58] = (uint8_t)(bit_len >> 40);
    block[59] = (uint8_t)(bit_len >> 32);
    block[60] = (uint8_t)(bit_len >> 24);
    block[61] = (uint8_t)(bit_len >> 16);
    block[62] = (uint8_t)(bit_len >>  8);
    block[63] = (uint8_t)(bit_len      );
    {
        uint32_t W[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            W[i] =  ((uint32_t)block[i*4    ] << 24)
                 |  ((uint32_t)block[i*4 + 1] << 16)
                 |  ((uint32_t)block[i*4 + 2] <<  8)
                 |  ((uint32_t)block[i*4 + 3]      );
        }
        sha256_compress(state, W);
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        out[i*4    ] = (uint8_t)(state[i] >> 24);
        out[i*4 + 1] = (uint8_t)(state[i] >> 16);
        out[i*4 + 2] = (uint8_t)(state[i] >>  8);
        out[i*4 + 3] = (uint8_t)(state[i]      );
    }
}

// ===========================================================================
// SHA-512 (FIPS 180-4)
// ===========================================================================

__device__ __constant__ static const uint64_t kSha512_K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL,
    0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
    0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL,
    0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
    0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
    0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, 0x2de92c6f592b0275ULL,
    0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL,
    0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
    0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL,
    0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL,
    0x92722c851482353bULL, 0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
    0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
    0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL,
    0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
    0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL,
    0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL,
    0xc67178f2e372532bULL, 0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
    0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL,
    0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
    0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
    0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL,
};

__device__ __forceinline__ uint64_t sha512_rotr(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

// `__noinline__` is a deliberate trade-off: the W[80] schedule is 640
// bytes of stack per call, and inlining sha512_compress into the v2
// brain-wallet kernels (which already carry SHA-256 state, HMAC pads,
// and v2 match-record buffers) blows past the per-thread register
// budget and spills into local memory -- much slower than a real call
// (Gemini PR #15 review caught the inlined-cost regression).
__device__ __noinline__ static void sha512_compress(uint64_t state[8], const uint8_t block[128]) {
    uint64_t W[80];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        W[i] = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            W[i] |= (uint64_t)block[i * 8 + j] << (56 - 8 * j);
        }
    }
    #pragma unroll
    for (int i = 16; i < 80; ++i) {
        uint64_t s0 = sha512_rotr(W[i-15], 1) ^ sha512_rotr(W[i-15], 8) ^ (W[i-15] >> 7);
        uint64_t s1 = sha512_rotr(W[i-2], 19) ^ sha512_rotr(W[i-2], 61) ^ (W[i-2] >> 6);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    uint64_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint64_t e = state[4], f = state[5], g = state[6], h = state[7];
    #pragma unroll
    for (int i = 0; i < 80; ++i) {
        uint64_t S1 = sha512_rotr(e, 14) ^ sha512_rotr(e, 18) ^ sha512_rotr(e, 41);
        uint64_t ch = (e & f) ^ ((~e) & g);
        uint64_t T1 = h + S1 + ch + kSha512_K[i] + W[i];
        uint64_t S0 = sha512_rotr(a, 28) ^ sha512_rotr(a, 34) ^ sha512_rotr(a, 39);
        uint64_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t T2 = S0 + mj;
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ static void sha512(const uint8_t* msg, uint32_t len, uint8_t out[64]) {
    uint64_t state[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
        0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };
    uint64_t bit_len = (uint64_t)len * 8;
    uint8_t  block[128];
    uint32_t off = 0;
    while (off + 128 <= len) {
        for (int i = 0; i < 128; ++i) block[i] = msg[off + i];
        sha512_compress(state, block);
        off += 128;
    }
    uint32_t rem = len - off;
    for (uint32_t i = 0; i < rem; ++i) block[i] = msg[off + i];
    block[rem] = 0x80u;
    for (uint32_t i = rem + 1; i < 128; ++i) block[i] = 0;
    if (rem + 1 + 16 > 128) {
        sha512_compress(state, block);
        for (int i = 0; i < 112; ++i) block[i] = 0;
    }
    // 128-bit big-endian length: high 64 = 0
    for (int i = 0; i < 8; ++i) block[112 + i] = 0;
    block[120] = (uint8_t)(bit_len >> 56);
    block[121] = (uint8_t)(bit_len >> 48);
    block[122] = (uint8_t)(bit_len >> 40);
    block[123] = (uint8_t)(bit_len >> 32);
    block[124] = (uint8_t)(bit_len >> 24);
    block[125] = (uint8_t)(bit_len >> 16);
    block[126] = (uint8_t)(bit_len >>  8);
    block[127] = (uint8_t)(bit_len      );
    sha512_compress(state, block);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[i*8 + j] = (uint8_t)(state[i] >> (56 - 8 * j));
        }
    }
}

// HMAC-SHA512 -- block size 128.
// kHmacSha512MaxMsg bounds the i_pad stack allocation. CUDA threads have
// ~1024 bytes of default stack; allocating much more can blow the per-
// thread limit at runtime (Gemini PR #15 review pointed this out).
//
// Real-world call sites under the v2 path:
//   * Electrum v2 PBKDF2 inner block: msg = "electrum"(8) + passphrase(<=200)
//     + 4-byte counter = up to 212 bytes
//   * BIP-39 PBKDF2 inner block: similar shape, < 200 bytes
// 256 bytes is sufficient with 44 bytes headroom. Longer inputs return
// all-zero (which cannot match a real puzzle target), making overflow
// observable and non-silent.
constexpr uint32_t kHmacSha512MaxMsg = 256;

// Helper: write SHA-512 padding + 128-bit BE length into `block` and call
// sha512_compress. `bit_len` is the running total bits hashed so far.
// Caller has already filled the first `rem` bytes of `block` with the
// trailing partial message (rem < 128). On return, sha512_compress has
// been called once or twice (depending on whether rem+1+16 fits in 128).
__device__ __forceinline__ static void sha512_finish_block(
    uint64_t state[8], uint8_t block[128], uint32_t rem, uint64_t bit_len)
{
    block[rem] = 0x80u;
    for (uint32_t i = rem + 1; i < 128; ++i) block[i] = 0;
    if (rem + 1 + 16 > 128) {
        sha512_compress(state, block);
        for (int i = 0; i < 112; ++i) block[i] = 0;
    }
    // 128-bit big-endian length: high 64 always 0 in our use cases.
    for (int i = 0; i < 8; ++i) block[112 + i] = 0;
    block[120] = (uint8_t)(bit_len >> 56);
    block[121] = (uint8_t)(bit_len >> 48);
    block[122] = (uint8_t)(bit_len >> 40);
    block[123] = (uint8_t)(bit_len >> 32);
    block[124] = (uint8_t)(bit_len >> 24);
    block[125] = (uint8_t)(bit_len >> 16);
    block[126] = (uint8_t)(bit_len >>  8);
    block[127] = (uint8_t)(bit_len      );
    sha512_compress(state, block);
}

__device__ __forceinline__ static void sha512_state_to_bytes(
    const uint64_t state[8], uint8_t out[64])
{
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint64_t v = state[i];
        out[i*8 + 0] = (uint8_t)(v >> 56);
        out[i*8 + 1] = (uint8_t)(v >> 48);
        out[i*8 + 2] = (uint8_t)(v >> 40);
        out[i*8 + 3] = (uint8_t)(v >> 32);
        out[i*8 + 4] = (uint8_t)(v >> 24);
        out[i*8 + 5] = (uint8_t)(v >> 16);
        out[i*8 + 6] = (uint8_t)(v >>  8);
        out[i*8 + 7] = (uint8_t)(v      );
    }
}

// Streaming HMAC-SHA-512 to keep per-thread stack well under 1KB.
//
// The previous implementation built `i_pad[128 + kHmacSha512MaxMsg]` and
// `o_pad[128 + 64]` as contiguous buffers, then called sha512() twice.
// Total stack was 768 bytes -- safe under the 1KB CUDA default but heavy
// for kernels that already carry SHA state, intermediate hashes, and v2
// match-record buffers (Gemini PR #15 review flagged the cumulative
// pressure).
//
// The streaming form below feeds the K-XOR-ipad block directly into a
// fresh SHA-512 state, then walks the message in 128-byte chunks, then
// finalizes. The outer hash reuses the same `state` array and `block`
// buffer. Stack drops to k0[128] + block[128] + state[64] = 320 bytes.
__device__ static void hmac_sha512(
    const uint8_t* key, uint32_t key_len,
    const uint8_t* msg, uint32_t msg_len,
    uint8_t out[64])
{
    if (msg_len > kHmacSha512MaxMsg) {
        // Refuse silently rather than miscomputing. Caller's puzzle compare
        // will not match an all-zero priv on any realistic puzzle.
        for (int i = 0; i < 64; ++i) out[i] = 0;
        return;
    }

    // Derive K0 (key padded/hashed to 128 bytes).
    uint8_t k0[128];
    if (key_len > 128) {
        sha512(key, key_len, k0);
        for (int i = 64; i < 128; ++i) k0[i] = 0;
    } else {
        for (uint32_t i = 0; i < key_len; ++i) k0[i] = key[i];
        for (uint32_t i = key_len; i < 128; ++i) k0[i] = 0;
    }

    // ---- inner = SHA-512((K0 ^ ipad) || msg) ------------------------
    uint64_t state[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
        0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };
    uint8_t block[128];
    for (int i = 0; i < 128; ++i) block[i] = k0[i] ^ 0x36u;
    sha512_compress(state, block);

    uint32_t off = 0;
    while (off + 128 <= msg_len) {
        for (int i = 0; i < 128; ++i) block[i] = msg[off + i];
        sha512_compress(state, block);
        off += 128;
    }
    const uint32_t rem_inner = msg_len - off;
    for (uint32_t i = 0; i < rem_inner; ++i) block[i] = msg[off + i];
    sha512_finish_block(state, block,
                        rem_inner,
                        ((uint64_t)128 + (uint64_t)msg_len) * 8u);

    // Inner digest is now in `state`. Serialize it into the first 64
    // bytes of `block` -- which becomes the message payload for the
    // outer hash.
    uint8_t inner_digest[64];
    sha512_state_to_bytes(state, inner_digest);

    // ---- outer = SHA-512((K0 ^ opad) || inner_digest) ---------------
    state[0] = 0x6a09e667f3bcc908ULL;
    state[1] = 0xbb67ae8584caa73bULL;
    state[2] = 0x3c6ef372fe94f82bULL;
    state[3] = 0xa54ff53a5f1d36f1ULL;
    state[4] = 0x510e527fade682d1ULL;
    state[5] = 0x9b05688c2b3e6c1fULL;
    state[6] = 0x1f83d9abfb41bd6bULL;
    state[7] = 0x5be0cd19137e2179ULL;

    for (int i = 0; i < 128; ++i) block[i] = k0[i] ^ 0x5cu;
    sha512_compress(state, block);

    // 64-byte inner digest is the entirety of the outer message.
    for (int i = 0; i < 64; ++i) block[i] = inner_digest[i];
    sha512_finish_block(state, block, 64, ((uint64_t)128 + 64) * 8u);

    sha512_state_to_bytes(state, out);
}

// ===========================================================================
// MD5 (RFC 1321) -- needed for Phase 7 legacy KDFs
// ===========================================================================

__device__ __forceinline__ uint32_t md5_rol(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ static void md5_compress(uint32_t state[4], const uint8_t block[64]) {
    static const uint32_t T[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
        0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
        0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
        0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
        0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
    };
    static const int S[64] = {
        7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
        5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
        4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
        6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21
    };
    uint32_t M[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        M[i] =  ((uint32_t)block[i*4    ]      )
             |  ((uint32_t)block[i*4 + 1] <<  8)
             |  ((uint32_t)block[i*4 + 2] << 16)
             |  ((uint32_t)block[i*4 + 3] << 24);
    }
    uint32_t A = state[0], B = state[1], C = state[2], D = state[3];
    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t F, g;
        if      (i < 16) { F = (B & C) | ((~B) & D);    g = i; }
        else if (i < 32) { F = (D & B) | ((~D) & C);    g = (5*i + 1) % 16; }
        else if (i < 48) { F = B ^ C ^ D;               g = (3*i + 5) % 16; }
        else             { F = C ^ (B | (~D));          g = (7*i)     % 16; }
        F = F + A + T[i] + M[g];
        A = D; D = C; C = B; B = B + md5_rol(F, S[i]);
    }
    state[0] += A; state[1] += B; state[2] += C; state[3] += D;
}

__device__ static void md5(const uint8_t* msg, uint32_t len, uint8_t out[16]) {
    uint32_t state[4] = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u};
    uint64_t bit_len = (uint64_t)len * 8;
    uint8_t  block[64];
    uint32_t off = 0;
    while (off + 64 <= len) {
        for (int i = 0; i < 64; ++i) block[i] = msg[off + i];
        md5_compress(state, block);
        off += 64;
    }
    uint32_t rem = len - off;
    for (uint32_t i = 0; i < rem; ++i) block[i] = msg[off + i];
    block[rem] = 0x80u;
    for (uint32_t i = rem + 1; i < 64; ++i) block[i] = 0;
    if (rem + 1 + 8 > 64) {
        md5_compress(state, block);
        for (int i = 0; i < 56; ++i) block[i] = 0;
    }
    block[56] = (uint8_t)(bit_len      );
    block[57] = (uint8_t)(bit_len >>  8);
    block[58] = (uint8_t)(bit_len >> 16);
    block[59] = (uint8_t)(bit_len >> 24);
    block[60] = (uint8_t)(bit_len >> 32);
    block[61] = (uint8_t)(bit_len >> 40);
    block[62] = (uint8_t)(bit_len >> 48);
    block[63] = (uint8_t)(bit_len >> 56);
    md5_compress(state, block);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i*4    ] = (uint8_t)(state[i]      );
        out[i*4 + 1] = (uint8_t)(state[i] >>  8);
        out[i*4 + 2] = (uint8_t)(state[i] >> 16);
        out[i*4 + 3] = (uint8_t)(state[i] >> 24);
    }
}

// ===========================================================================
// SHA-1 (FIPS 180-4) -- needed for Phase 7 legacy KDFs
// ===========================================================================

__device__ __forceinline__ uint32_t sha1_rol(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ static void sha1_compress(uint32_t state[5], const uint8_t block[64]) {
    uint32_t W[80];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        W[i] =  ((uint32_t)block[i*4    ] << 24)
             |  ((uint32_t)block[i*4 + 1] << 16)
             |  ((uint32_t)block[i*4 + 2] <<  8)
             |  ((uint32_t)block[i*4 + 3]      );
    }
    #pragma unroll
    for (int i = 16; i < 80; ++i) {
        W[i] = sha1_rol(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], 1);
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3], e = state[4];
    #pragma unroll
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if      (i < 20) { f = (b & c) | ((~b) & d);  k = 0x5a827999u; }
        else if (i < 40) { f = b ^ c ^ d;             k = 0x6ed9eba1u; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8f1bbcdcu; }
        else             { f = b ^ c ^ d;             k = 0xca62c1d6u; }
        uint32_t T = sha1_rol(a, 5) + f + e + k + W[i];
        e = d; d = c; c = sha1_rol(b, 30); b = a; a = T;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d; state[4] += e;
}

__device__ static void sha1(const uint8_t* msg, uint32_t len, uint8_t out[20]) {
    uint32_t state[5] = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u};
    uint64_t bit_len = (uint64_t)len * 8;
    uint8_t  block[64];
    uint32_t off = 0;
    while (off + 64 <= len) {
        for (int i = 0; i < 64; ++i) block[i] = msg[off + i];
        sha1_compress(state, block);
        off += 64;
    }
    uint32_t rem = len - off;
    for (uint32_t i = 0; i < rem; ++i) block[i] = msg[off + i];
    block[rem] = 0x80u;
    for (uint32_t i = rem + 1; i < 64; ++i) block[i] = 0;
    if (rem + 1 + 8 > 64) {
        sha1_compress(state, block);
        for (int i = 0; i < 56; ++i) block[i] = 0;
    }
    block[56] = (uint8_t)(bit_len >> 56);
    block[57] = (uint8_t)(bit_len >> 48);
    block[58] = (uint8_t)(bit_len >> 40);
    block[59] = (uint8_t)(bit_len >> 32);
    block[60] = (uint8_t)(bit_len >> 24);
    block[61] = (uint8_t)(bit_len >> 16);
    block[62] = (uint8_t)(bit_len >>  8);
    block[63] = (uint8_t)(bit_len      );
    sha1_compress(state, block);
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        out[i*4    ] = (uint8_t)(state[i] >> 24);
        out[i*4 + 1] = (uint8_t)(state[i] >> 16);
        out[i*4 + 2] = (uint8_t)(state[i] >>  8);
        out[i*4 + 3] = (uint8_t)(state[i]      );
    }
}

// ===========================================================================
// RIPEMD-160 -- for hash160 used by every address derivation
// ===========================================================================

__device__ __forceinline__ uint32_t rmd_rol(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ static void ripemd160(const uint8_t* msg, uint32_t len, uint8_t out[20]) {
    static const int rL[80] = {
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
         7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
         3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
         1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
         4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13
    };
    static const int rR[80] = {
         5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12,
         6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2,
        15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13,
         8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14,
        12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11
    };
    static const int sL[80] = {
        11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
         7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
        11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
        11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
         9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6
    };
    static const int sR[80] = {
         8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
         9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
         9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
        15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
         8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11
    };
    static const uint32_t KL[5] = {0u, 0x5a827999u, 0x6ed9eba1u, 0x8f1bbcdcu, 0xa953fd4eu};
    static const uint32_t KR[5] = {0x50a28be6u, 0x5c4dd124u, 0x6d703ef3u, 0x7a6d76e9u, 0u};

    uint32_t state[5] = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u, 0xc3d2e1f0u};
    uint64_t bit_len = (uint64_t)len * 8;
    uint8_t  block[64];
    uint32_t off = 0;

    auto compress = [&](const uint8_t blk[64]) {
        uint32_t X[16];
        for (int i = 0; i < 16; ++i) {
            X[i] =  ((uint32_t)blk[i*4    ]      )
                 |  ((uint32_t)blk[i*4 + 1] <<  8)
                 |  ((uint32_t)blk[i*4 + 2] << 16)
                 |  ((uint32_t)blk[i*4 + 3] << 24);
        }
        uint32_t A = state[0], B = state[1], C = state[2], D = state[3], E = state[4];
        uint32_t Ar = A, Br = B, Cr = C, Dr = D, Er = E;
        for (int j = 0; j < 80; ++j) {
            int round = j / 16;
            uint32_t fL, fR;
            switch (round) {
                case 0: fL = B ^ C ^ D;                 fR = Br ^ (Cr | (~Dr));        break;
                case 1: fL = (B & C) | ((~B) & D);      fR = (Br & Dr) | (Cr & (~Dr)); break;
                case 2: fL = (B | (~C)) ^ D;            fR = (Br | (~Cr)) ^ Dr;        break;
                case 3: fL = (B & D) | (C & (~D));      fR = (Br & Cr) | ((~Br) & Dr); break;
                default: fL = B ^ (C | (~D));           fR = Br ^ Cr ^ Dr;             break;
            }
            uint32_t T = rmd_rol(A + fL + X[rL[j]] + KL[round], sL[j]) + E;
            A = E; E = D; D = rmd_rol(C, 10); C = B; B = T;
            T = rmd_rol(Ar + fR + X[rR[j]] + KR[round], sR[j]) + Er;
            Ar = Er; Er = Dr; Dr = rmd_rol(Cr, 10); Cr = Br; Br = T;
        }
        uint32_t T = state[1] + C + Dr;
        state[1] = state[2] + D + Er;
        state[2] = state[3] + E + Ar;
        state[3] = state[4] + A + Br;
        state[4] = state[0] + B + Cr;
        state[0] = T;
    };

    while (off + 64 <= len) {
        for (int i = 0; i < 64; ++i) block[i] = msg[off + i];
        compress(block);
        off += 64;
    }
    uint32_t rem = len - off;
    for (uint32_t i = 0; i < rem; ++i) block[i] = msg[off + i];
    block[rem] = 0x80u;
    for (uint32_t i = rem + 1; i < 64; ++i) block[i] = 0;
    if (rem + 1 + 8 > 64) {
        compress(block);
        for (int i = 0; i < 56; ++i) block[i] = 0;
    }
    for (int i = 0; i < 8; ++i) block[56 + i] = (uint8_t)(bit_len >> (i * 8));
    compress(block);
    for (int i = 0; i < 5; ++i) {
        out[i*4    ] = (uint8_t)(state[i]      );
        out[i*4 + 1] = (uint8_t)(state[i] >>  8);
        out[i*4 + 2] = (uint8_t)(state[i] >> 16);
        out[i*4 + 3] = (uint8_t)(state[i] >> 24);
    }
}

__device__ __forceinline__ void hash160(const uint8_t* msg, uint32_t len, uint8_t out[20]) {
    uint8_t s[32];
    sha256(msg, len, s);
    ripemd160(s, 32, out);
}

}  // namespace device
}  // namespace v2
}  // namespace gpu
}  // namespace collider
