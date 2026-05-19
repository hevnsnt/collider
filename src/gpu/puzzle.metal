/*
 * Apple Metal fused brute-force puzzle kernel (v1.4.1).
 *
 * Mirrors the CUDA puzzle_search_optimized pipeline (puzzle_optimized.cu)
 * but expressed for Apple Silicon. Per thread:
 *   1. priv = base + gid    (256-bit add; only the bottom 64 bits move)
 *   2. P    = priv * G      via an 8-bit-window precomputed table
 *   3. (px, py) = jac_to_affine(P)  (one mod_inv per thread)
 *   4. compressed pubkey = (0x02 | (py & 1)) || px_be
 *   5. sha256 of 33 bytes  (one 64-byte block, hand-padded)
 *   6. ripemd160 of 32 bytes (one 64-byte block, hand-padded)
 *   7. compare to target hash160; on match, atomic_compare_exchange the
 *      (match_lo, match_hi, match_found) trio.
 *
 * Field arithmetic conventions match src/gpu/kangaroo.metal: 4 ulong limbs
 * little-endian-by-limb, secp256k1 prime p = 2^256 - 2^32 - 977. The
 * implementations below are intentionally a near-duplicate of the kangaroo
 * helpers because Metal has no header-include mechanism (every kernel TU
 * is its own translation unit handed verbatim to newLibraryWithSource).
 * If you change a math primitive here, mirror the change in kangaroo.metal
 * to keep both kernels validated against the same KAT.
 *
 * Precomputed G table layout (passed as buffer 0):
 *   g_table[w * 256 + d] = (d * 2^(8*w)) * G in affine (X, Y) form, w in
 *   [0, 32), d in [0, 256). Each entry is 8 ulong = 64 bytes:
 *     ulong[0..3] = X (LE-by-limb)
 *     ulong[4..7] = Y (LE-by-limb)
 *   The d=0 row of every window is the point at infinity; we encode it as
 *   (X=0, Y=0) and the kernel skips windows where the scalar byte is 0.
 *   The host computes the table once at init via cpu::ec_mul / ec_to_affine
 *   and uploads it in MTLResourceStorageModeShared so the GPU can read
 *   directly from system memory on Apple Silicon.
 *
 * Thread layout: one MSL thread per candidate key. dispatchThreads
 * width = batch_size, threadsPerThreadgroup = 32 (matches Apple SIMD-group
 * width). No threadgroup-cooperative work in this kernel -- each thread is
 * fully independent. The threadgroup width of 32 is a perf hint for the
 * Apple GPU scheduler, not a correctness requirement.
 */

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// ---------------------------------------------------------------------------
// uint128 emulation via two ulongs.
// ---------------------------------------------------------------------------

inline ulong addc(ulong a, ulong b, thread ulong &carry) {
    ulong sum  = a + b;
    ulong c1   = (sum < a) ? 1ul : 0ul;
    ulong sum2 = sum + carry;
    ulong c2   = (sum2 < sum) ? 1ul : 0ul;
    carry = c1 + c2;
    return sum2;
}

inline ulong subb(ulong a, ulong b, thread ulong &borrow) {
    ulong diff  = a - b;
    ulong b1    = (a < b) ? 1ul : 0ul;
    ulong diff2 = diff - borrow;
    ulong b2    = (diff < borrow) ? 1ul : 0ul;
    borrow = b1 + b2;
    return diff2;
}

// 64x64 -> 128. Returns low; sets `hi` to the high 64.
inline ulong mul128(ulong a, ulong b, thread ulong &hi) {
    ulong al = a & 0xFFFFFFFFul, ah = a >> 32;
    ulong bl = b & 0xFFFFFFFFul, bh = b >> 32;
    ulong p0 = al * bl;
    ulong p1 = al * bh;
    ulong p2 = ah * bl;
    ulong p3 = ah * bh;
    ulong mid = (p0 >> 32) + (p1 & 0xFFFFFFFFul) + (p2 & 0xFFFFFFFFul);
    ulong lo  = (p0 & 0xFFFFFFFFul) | (mid << 32);
    hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    return lo;
}

// ---------------------------------------------------------------------------
// secp256k1 prime constants.
// ---------------------------------------------------------------------------

constant ulong P0 = 0xFFFFFFFEFFFFFC2Ful;
constant ulong P1 = 0xFFFFFFFFFFFFFFFFul;
constant ulong P2 = 0xFFFFFFFFFFFFFFFFul;
constant ulong P3 = 0xFFFFFFFFFFFFFFFFul;
// 2^32 + 977 = 0x1000003D1
constant ulong K0 = 0x00000001000003D1ul;

// ---------------------------------------------------------------------------
// 256-bit add / sub mod p.
// ---------------------------------------------------------------------------

inline void mod_add(thread ulong r[4], thread const ulong a[4], thread const ulong b[4]) {
    ulong c = 0;
    ulong t0 = addc(a[0], b[0], c);
    ulong t1 = addc(a[1], b[1], c);
    ulong t2 = addc(a[2], b[2], c);
    ulong t3 = addc(a[3], b[3], c);
    if (c) {
        ulong cc = 0;
        t0 = addc(t0, K0, cc);
        t1 = addc(t1, 0,  cc);
        t2 = addc(t2, 0,  cc);
        t3 = addc(t3, 0,  cc);
    }
    bool ge = (t3 > P3) ||
              (t3 == P3 && t2 > P2) ||
              (t3 == P3 && t2 == P2 && t1 > P1) ||
              (t3 == P3 && t2 == P2 && t1 == P1 && t0 >= P0);
    if (ge) {
        ulong b2 = 0;
        t0 = subb(t0, P0, b2);
        t1 = subb(t1, P1, b2);
        t2 = subb(t2, P2, b2);
        t3 = subb(t3, P3, b2);
    }
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

inline void mod_sub(thread ulong r[4], thread const ulong a[4], thread const ulong b[4]) {
    ulong bo = 0;
    ulong t0 = subb(a[0], b[0], bo);
    ulong t1 = subb(a[1], b[1], bo);
    ulong t2 = subb(a[2], b[2], bo);
    ulong t3 = subb(a[3], b[3], bo);
    if (bo) {
        ulong c = 0;
        t0 = addc(t0, P0, c);
        t1 = addc(t1, P1, c);
        t2 = addc(t2, P2, c);
        t3 = addc(t3, P3, c);
    }
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

// ---------------------------------------------------------------------------
// 256x256 -> 512-bit multiply, reduced via the 2^256 = 2^32 + 977 trick.
// ---------------------------------------------------------------------------

inline void mod_mul(thread ulong r[4], thread const ulong a[4], thread const ulong b[4]) {
    ulong w[8];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) w[i] = 0;

    for (int i = 0; i < 4; ++i) {
        ulong carry = 0;
        for (int j = 0; j < 4; ++j) {
            ulong hi;
            ulong lo = mul128(a[i], b[j], hi);
            ulong c1 = 0, c2 = 0;
            ulong sum0 = addc(w[i + j], lo,    c1);
            ulong sum1 = addc(sum0,     carry, c2);
            w[i + j] = sum1;
            carry = hi + c1 + c2;
        }
        w[i + 4] = carry;
    }

    ulong lo[4] = { w[0], w[1], w[2], w[3] };
    ulong hi[4] = { w[4], w[5], w[6], w[7] };

    ulong h[5] = { 0, 0, 0, 0, 0 };
    {
        ulong carry = 0;
        for (int i = 0; i < 4; ++i) {
            ulong hh;
            ulong ll = mul128(hi[i], K0, hh);
            ulong c1 = 0, c2 = 0;
            ulong s0 = addc(h[i], ll,    c1);
            ulong s1 = addc(s0,   carry, c2);
            h[i] = s1;
            carry = hh + c1 + c2;
        }
        h[4] = carry;
    }

    ulong c = 0;
    ulong t0 = addc(lo[0], h[0], c);
    ulong t1 = addc(lo[1], h[1], c);
    ulong t2 = addc(lo[2], h[2], c);
    ulong t3 = addc(lo[3], h[3], c);
    ulong overflow = c + h[4];
    if (overflow) {
        ulong oh, ol;
        ol = mul128(overflow, K0, oh);
        ulong c2 = 0;
        t0 = addc(t0, ol, c2);
        t1 = addc(t1, oh, c2);
        t2 = addc(t2, 0,  c2);
        t3 = addc(t3, 0,  c2);
        if (c2) {
            ulong c3 = 0;
            t0 = addc(t0, K0, c3);
            t1 = addc(t1, 0,  c3);
            t2 = addc(t2, 0,  c3);
            t3 = addc(t3, 0,  c3);
        }
    }
    bool ge = (t3 > P3) ||
              (t3 == P3 && t2 > P2) ||
              (t3 == P3 && t2 == P2 && t1 > P1) ||
              (t3 == P3 && t2 == P2 && t1 == P1 && t0 >= P0);
    if (ge) {
        ulong bo = 0;
        t0 = subb(t0, P0, bo);
        t1 = subb(t1, P1, bo);
        t2 = subb(t2, P2, bo);
        t3 = subb(t3, P3, bo);
    }
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3;
}

inline void mod_sqr(thread ulong r[4], thread const ulong a[4]) {
    mod_mul(r, a, a);
}

// ---------------------------------------------------------------------------
// Modular inverse via the libsecp256k1 addition chain (a^(p-2) mod p).
// Same chain used in kangaroo.metal mod_inv. ~256 squarings + 13 mults.
// ---------------------------------------------------------------------------

inline void inv_sqr(thread ulong r[4], thread const ulong x[4]) {
    ulong t[4]; mod_sqr(t, x);
    r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
}
inline void inv_mul(thread ulong r[4],
                    thread const ulong x[4],
                    thread const ulong y[4]) {
    ulong t[4]; mod_mul(t, x, y);
    r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
}

inline void mod_inv(thread ulong r[4], thread const ulong a[4]) {
    ulong x2[4], x3[4], x6[4], x9[4], x11[4], x22[4], x44[4];
    ulong x88[4], x176[4], x220[4], x223[4], t1[4];

    inv_sqr(x2, a); inv_mul(x2, x2, a);
    inv_sqr(x3, x2); inv_mul(x3, x3, a);

    inv_sqr(x6, x3); inv_sqr(x6, x6); inv_sqr(x6, x6); inv_mul(x6, x6, x3);
    inv_sqr(x9, x6); inv_sqr(x9, x9); inv_sqr(x9, x9); inv_mul(x9, x9, x3);
    inv_sqr(x11, x9); inv_sqr(x11, x11); inv_mul(x11, x11, x2);

    inv_sqr(x22, x11);
    for (int i = 1; i < 11; ++i) inv_sqr(x22, x22);
    inv_mul(x22, x22, x11);

    inv_sqr(x44, x22);
    for (int i = 1; i < 22; ++i) inv_sqr(x44, x44);
    inv_mul(x44, x44, x22);

    inv_sqr(x88, x44);
    for (int i = 1; i < 44; ++i) inv_sqr(x88, x88);
    inv_mul(x88, x88, x44);

    inv_sqr(x176, x88);
    for (int i = 1; i < 88; ++i) inv_sqr(x176, x176);
    inv_mul(x176, x176, x88);

    inv_sqr(x220, x176);
    for (int i = 1; i < 44; ++i) inv_sqr(x220, x220);
    inv_mul(x220, x220, x44);

    inv_sqr(x223, x220); inv_sqr(x223, x223); inv_sqr(x223, x223);
    inv_mul(x223, x223, x3);

    inv_sqr(t1, x223);
    for (int i = 1; i < 23; ++i) inv_sqr(t1, t1);
    inv_mul(t1, t1, x22);
    for (int i = 0; i < 5; ++i) inv_sqr(t1, t1);
    inv_mul(t1, t1, a);
    for (int i = 0; i < 3; ++i) inv_sqr(t1, t1);
    inv_mul(t1, t1, x2);
    inv_sqr(t1, t1); inv_sqr(t1, t1);
    inv_mul(t1, t1, a);

    r[0] = t1[0]; r[1] = t1[1]; r[2] = t1[2]; r[3] = t1[3];
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

inline bool limbs_zero(thread const ulong a[4]) {
    return (a[0] == 0) & (a[1] == 0) & (a[2] == 0) & (a[3] == 0);
}

inline void copy4(thread ulong dst[4], thread const ulong src[4]) {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
}

inline void set4(thread ulong dst[4], ulong v0, ulong v1, ulong v2, ulong v3) {
    dst[0] = v0; dst[1] = v1; dst[2] = v2; dst[3] = v3;
}

// ---------------------------------------------------------------------------
// Jacobian secp256k1 point operations. Same formulas as kangaroo.metal.
// ---------------------------------------------------------------------------

inline void jac_double(thread ulong rx[4], thread ulong ry[4], thread ulong rz[4],
                       thread const ulong px[4], thread const ulong py[4],
                       thread const ulong pz[4])
{
    if (limbs_zero(pz) || limbs_zero(py)) {
        set4(rx, 0, 0, 0, 0);
        set4(ry, 0, 0, 0, 0);
        set4(rz, 0, 0, 0, 0);
        return;
    }

    ulong y2[4], s[4], m[4], t[4];

    mod_sqr(y2, py);
    mod_mul(s, px, y2);
    mod_add(s, s, s);
    mod_add(s, s, s);
    mod_sqr(m, px);
    mod_add(t, m, m);
    mod_add(m, t, m);

    mod_sqr(rx, m);
    mod_sub(rx, rx, s);
    mod_sub(rx, rx, s);

    mod_sub(t, s, rx);
    mod_mul(t, m, t);
    ulong y4[4];
    mod_sqr(y4, y2);
    mod_add(y4, y4, y4);
    mod_add(y4, y4, y4);
    mod_add(y4, y4, y4);
    mod_sub(ry, t, y4);

    mod_mul(rz, py, pz);
    mod_add(rz, rz, rz);
}

// R = P + Q where P is Jacobian, Q is affine. Q = infinity is encoded as
// (0, 0); the kernel treats that as a no-op.
inline void jac_add_mixed(thread ulong rx[4], thread ulong ry[4], thread ulong rz[4],
                          thread const ulong px[4], thread const ulong py[4],
                          thread const ulong pz[4],
                          thread const ulong qx[4], thread const ulong qy[4])
{
    // Q = infinity sentinel -> R = P.
    if (limbs_zero(qx) && limbs_zero(qy)) {
        copy4(rx, px);
        copy4(ry, py);
        copy4(rz, pz);
        return;
    }
    // P = infinity -> R = Q (as affine, Z = 1).
    if (limbs_zero(pz)) {
        copy4(rx, qx);
        copy4(ry, qy);
        set4(rz, 1, 0, 0, 0);
        return;
    }

    ulong z2[4], z3[4], u2[4], s2[4];
    ulong h[4], rr[4], h2[4], h3[4], v[4], tmp[4];

    mod_sqr(z2, pz);
    mod_mul(z3, z2, pz);
    mod_mul(u2, qx, z2);
    mod_mul(s2, qy, z3);
    mod_sub(h,  u2, px);
    mod_sub(rr, s2, py);

    if (limbs_zero(h)) {
        if (limbs_zero(rr)) {
            jac_double(rx, ry, rz, px, py, pz);
        } else {
            set4(rx, 0, 0, 0, 0);
            set4(ry, 0, 0, 0, 0);
            set4(rz, 0, 0, 0, 0);
        }
        return;
    }

    mod_sqr(h2, h);
    mod_mul(h3, h2, h);
    mod_mul(v,  px, h2);

    mod_sqr(tmp, rr);
    mod_sub(tmp, tmp, h3);
    mod_sub(tmp, tmp, v);
    mod_sub(rx,  tmp, v);

    mod_sub(tmp, v, rx);
    mod_mul(tmp, rr, tmp);
    ulong yh3[4];
    mod_mul(yh3, py, h3);
    mod_sub(ry, tmp, yh3);

    mod_mul(rz, pz, h);
}

// ---------------------------------------------------------------------------
// SHA-256 of exactly 33 bytes (one block; hand-padded).
// Mirrors puzzle_optimized.cu sha256_33bytes_opt byte-for-byte. Inlined so
// the inner loop has no call boundary.
// ---------------------------------------------------------------------------

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

inline uint sha_rotr(uint x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint sha_ch (uint x, uint y, uint z) { return (x & y) ^ (~x & z); }
inline uint sha_maj(uint x, uint y, uint z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint sha_S0 (uint x) { return sha_rotr(x, 2)  ^ sha_rotr(x, 13) ^ sha_rotr(x, 22); }
inline uint sha_S1 (uint x) { return sha_rotr(x, 6)  ^ sha_rotr(x, 11) ^ sha_rotr(x, 25); }
inline uint sha_s0 (uint x) { return sha_rotr(x, 7)  ^ sha_rotr(x, 18) ^ (x >> 3); }
inline uint sha_s1 (uint x) { return sha_rotr(x, 17) ^ sha_rotr(x, 19) ^ (x >> 10); }

// pubkey: 33 bytes (compressed: prefix + X big-endian).
// Output: 32 bytes (big-endian SHA-256 digest).
inline void sha256_33bytes(thread const uchar pubkey[33], thread uchar hash[32]) {
    uint W[64];
    // First 32 bytes -> 8 big-endian words.
    for (int i = 0; i < 8; ++i) {
        W[i] = ((uint)pubkey[i*4]     << 24)
             | ((uint)pubkey[i*4 + 1] << 16)
             | ((uint)pubkey[i*4 + 2] <<  8)
             | ((uint)pubkey[i*4 + 3]      );
    }
    // Word 8 = byte 32 in the high byte, then 0x80 padding byte.
    W[8] = ((uint)pubkey[32] << 24) | (0x80u << 16);
    // Words 9..13: zeros.
    W[9] = 0; W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0;
    // Length: 33 * 8 = 264 bits.
    W[14] = 0;
    W[15] = 264;

    for (int i = 16; i < 64; ++i) {
        W[i] = sha_s1(W[i-2]) + W[i-7] + sha_s0(W[i-15]) + W[i-16];
    }

    uint a = 0x6a09e667u, b = 0xbb67ae85u, c = 0x3c6ef372u, d = 0xa54ff53au;
    uint e = 0x510e527fu, f = 0x9b05688cu, g = 0x1f83d9abu, h = 0x5be0cd19u;

    for (int i = 0; i < 64; ++i) {
        uint t1 = h + sha_S1(e) + sha_ch(e, f, g) + K256[i] + W[i];
        uint t2 = sha_S0(a) + sha_maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    a += 0x6a09e667u; b += 0xbb67ae85u; c += 0x3c6ef372u; d += 0xa54ff53au;
    e += 0x510e527fu; f += 0x9b05688cu; g += 0x1f83d9abu; h += 0x5be0cd19u;

    // Output big-endian.
    uint H[8] = { a, b, c, d, e, f, g, h };
    for (int i = 0; i < 8; ++i) {
        hash[i*4    ] = (uchar)(H[i] >> 24);
        hash[i*4 + 1] = (uchar)(H[i] >> 16);
        hash[i*4 + 2] = (uchar)(H[i] >>  8);
        hash[i*4 + 3] = (uchar)(H[i]      );
    }
}

// ---------------------------------------------------------------------------
// RIPEMD-160 of exactly 32 bytes (one 64-byte block; hand-padded).
// Mirrors puzzle_optimized.cu ripemd160_32bytes_opt byte-for-byte: input is
// loaded little-endian, padding is 0x80 at byte 32 + length 256 at the end,
// output is little-endian 5 words.
// ---------------------------------------------------------------------------

inline uint rmd_rotl(uint x, int n) { return (x << n) | (x >> (32 - n)); }
inline uint rmd_f0(uint x, uint y, uint z) { return x ^ y ^ z; }
inline uint rmd_f1(uint x, uint y, uint z) { return (x & y) | (~x & z); }
inline uint rmd_f2(uint x, uint y, uint z) { return (x | ~y) ^ z; }
inline uint rmd_f3(uint x, uint y, uint z) { return (x & z) | (y & ~z); }
inline uint rmd_f4(uint x, uint y, uint z) { return x ^ (y | ~z); }

constant uint RMD_KL[5] = { 0x00000000u, 0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xA953FD4Eu };
constant uint RMD_KR[5] = { 0x50A28BE6u, 0x5C4DD124u, 0x6D703EF3u, 0x7A6D76E9u, 0x00000000u };

constant int RMD_RL[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};
constant int RMD_RR[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};
constant int RMD_SL[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};
constant int RMD_SR[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

inline void ripemd160_32bytes(thread const uchar sha_out[32], thread uchar h160[20]) {
    uint X[16];
    for (int i = 0; i < 8; ++i) {
        X[i] = ((uint)sha_out[i*4    ]      )
             | ((uint)sha_out[i*4 + 1] <<  8)
             | ((uint)sha_out[i*4 + 2] << 16)
             | ((uint)sha_out[i*4 + 3] << 24);
    }
    X[8]  = 0x00000080u;     // 0x80 at byte 32 (little-endian word 8 byte 0)
    X[9]  = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0;
    X[14] = 256u;            // length in bits, low 32
    X[15] = 0;

    uint AL = 0x67452301u, BL = 0xEFCDAB89u, CL = 0x98BADCFEu,
         DL = 0x10325476u, EL = 0xC3D2E1F0u;
    uint AR = AL, BR = BL, CR = CL, DR = DL, ER = EL;

    // Round 0 (j=0..15)
    for (int j = 0; j < 16; ++j) {
        uint tL = rmd_rotl(AL + rmd_f0(BL, CL, DL) + X[RMD_RL[j]] + RMD_KL[0], RMD_SL[j]) + EL;
        AL = EL; EL = DL; DL = rmd_rotl(CL, 10); CL = BL; BL = tL;
        uint tR = rmd_rotl(AR + rmd_f4(BR, CR, DR) + X[RMD_RR[j]] + RMD_KR[0], RMD_SR[j]) + ER;
        AR = ER; ER = DR; DR = rmd_rotl(CR, 10); CR = BR; BR = tR;
    }
    // Round 1 (j=16..31)
    for (int j = 16; j < 32; ++j) {
        uint tL = rmd_rotl(AL + rmd_f1(BL, CL, DL) + X[RMD_RL[j]] + RMD_KL[1], RMD_SL[j]) + EL;
        AL = EL; EL = DL; DL = rmd_rotl(CL, 10); CL = BL; BL = tL;
        uint tR = rmd_rotl(AR + rmd_f3(BR, CR, DR) + X[RMD_RR[j]] + RMD_KR[1], RMD_SR[j]) + ER;
        AR = ER; ER = DR; DR = rmd_rotl(CR, 10); CR = BR; BR = tR;
    }
    // Round 2 (j=32..47)
    for (int j = 32; j < 48; ++j) {
        uint tL = rmd_rotl(AL + rmd_f2(BL, CL, DL) + X[RMD_RL[j]] + RMD_KL[2], RMD_SL[j]) + EL;
        AL = EL; EL = DL; DL = rmd_rotl(CL, 10); CL = BL; BL = tL;
        uint tR = rmd_rotl(AR + rmd_f2(BR, CR, DR) + X[RMD_RR[j]] + RMD_KR[2], RMD_SR[j]) + ER;
        AR = ER; ER = DR; DR = rmd_rotl(CR, 10); CR = BR; BR = tR;
    }
    // Round 3 (j=48..63)
    for (int j = 48; j < 64; ++j) {
        uint tL = rmd_rotl(AL + rmd_f3(BL, CL, DL) + X[RMD_RL[j]] + RMD_KL[3], RMD_SL[j]) + EL;
        AL = EL; EL = DL; DL = rmd_rotl(CL, 10); CL = BL; BL = tL;
        uint tR = rmd_rotl(AR + rmd_f1(BR, CR, DR) + X[RMD_RR[j]] + RMD_KR[3], RMD_SR[j]) + ER;
        AR = ER; ER = DR; DR = rmd_rotl(CR, 10); CR = BR; BR = tR;
    }
    // Round 4 (j=64..79)
    for (int j = 64; j < 80; ++j) {
        uint tL = rmd_rotl(AL + rmd_f4(BL, CL, DL) + X[RMD_RL[j]] + RMD_KL[4], RMD_SL[j]) + EL;
        AL = EL; EL = DL; DL = rmd_rotl(CL, 10); CL = BL; BL = tL;
        uint tR = rmd_rotl(AR + rmd_f0(BR, CR, DR) + X[RMD_RR[j]] + RMD_KR[4], RMD_SR[j]) + ER;
        AR = ER; ER = DR; DR = rmd_rotl(CR, 10); CR = BR; BR = tR;
    }

    // Single-block RIPEMD-160 final combine. Standard rotation:
    //   t  = H1 + CL + DR
    //   H1 = H2 + DL + ER
    //   H2 = H3 + EL + AR
    //   H3 = H4 + AL + BR
    //   H4 = H0 + BL + CR
    //   H0 = t
    uint H0 = 0x67452301u, H1 = 0xEFCDAB89u, H2 = 0x98BADCFEu,
         H3 = 0x10325476u, H4 = 0xC3D2E1F0u;
    uint t = H1 + CL + DR;
    H1 = H2 + DL + ER;
    H2 = H3 + EL + AR;
    H3 = H4 + AL + BR;
    H4 = H0 + BL + CR;
    H0 = t;

    uint Hf[5] = { H0, H1, H2, H3, H4 };
    for (int i = 0; i < 5; ++i) {
        h160[i*4    ] = (uchar)(Hf[i]      );
        h160[i*4 + 1] = (uchar)(Hf[i] >>  8);
        h160[i*4 + 2] = (uchar)(Hf[i] >> 16);
        h160[i*4 + 3] = (uchar)(Hf[i] >> 24);
    }
}

// ---------------------------------------------------------------------------
// Compress affine pubkey (X, Y) -> 33-byte big-endian (prefix || X_be).
// ---------------------------------------------------------------------------

inline void compress_pubkey(thread uchar out[33],
                            thread const ulong x[4],
                            thread const ulong y[4])
{
    out[0] = (uchar)(0x02u | (uint)(y[0] & 1ul));
    // limb[3] is the most-significant 64 bits.
    for (int i = 0; i < 4; ++i) {
        ulong l = x[3 - i];
        for (int j = 0; j < 8; ++j) {
            out[1 + i * 8 + j] = (uchar)(l >> (56 - 8 * j));
        }
    }
}

// ---------------------------------------------------------------------------
// Windowed scalar-by-G multiplication using the precomputed table.
// table layout: table[w * 256 + d] = (X[4], Y[4]) ulong (8 ulong per entry).
// table[w * 256 + 0] is the point at infinity, encoded as (0, 0). We skip
// windows where the byte is 0.
// scalar: 32 bytes, 4 limbs LE-by-limb.
// Output: Jacobian (rx, ry, rz). Caller affinizes via mod_inv.
// ---------------------------------------------------------------------------

inline void scalar_mul_g(thread ulong rx[4], thread ulong ry[4], thread ulong rz[4],
                         thread const ulong scalar[4],
                         device const ulong* g_table)
{
    set4(rx, 0, 0, 0, 0);
    set4(ry, 0, 0, 0, 0);
    set4(rz, 0, 0, 0, 0);            // start at infinity

    // 32 windows of 8 bits each = 32 bytes = 256 bits.
    for (uint w = 0; w < 32u; ++w) {
        const uint limb_idx = w >> 3;             // 0..3
        const uint byte_in_limb = w & 7u;         // 0..7
        const uint byte_val = (uint)((scalar[limb_idx] >> (8u * byte_in_limb)) & 0xFFu);

        if (byte_val == 0u) continue;

        const uint base = (w * 256u + byte_val) * 8u;
        ulong qx[4], qy[4];
        qx[0] = g_table[base + 0]; qx[1] = g_table[base + 1];
        qx[2] = g_table[base + 2]; qx[3] = g_table[base + 3];
        qy[0] = g_table[base + 4]; qy[1] = g_table[base + 5];
        qy[2] = g_table[base + 6]; qy[3] = g_table[base + 7];

        ulong nx[4], ny[4], nz[4];
        jac_add_mixed(nx, ny, nz, rx, ry, rz, qx, qy);
        copy4(rx, nx); copy4(ry, ny); copy4(rz, nz);
    }
}

// ---------------------------------------------------------------------------
// Fused brute-force kernel.
// One thread per candidate key. Thread global id `gid` selects the candidate
// k = (start_lo, start_hi) + gid (256-bit add restricted to the bottom 128).
// Puzzle ranges are < 2^160, so 128-bit math on the bottom limbs is enough;
// we still propagate the carry into limb 2 / 3 for correctness on hypothetical
// callers.
// Buffers (matching PuzzleMetalSolver dispatch order):
//   0: g_table             (precomputed table; const after setup)
//   1: target_h160         (20 bytes)
//   2: start_lo (constant)
//   3: start_hi (constant)
//   4: total_keys (constant; gid < total_keys)
//   5: match_lo  (plain ulong; written by the unique CAS winner)
//   6: match_hi  (plain ulong)
//   7: match_found (atomic uint; 0 = no match, 1 = match)
// Atomicity contract: match_found is a 32-bit CAS gate. Only the first
// thread to flip 0->1 owns match_lo / match_hi for the rest of the
// dispatch. Subsequent threads see match_found != 0 (their atomic_load
// races with the writer's atomic_store, but both states are valid) and
// bail out before reading or writing match_lo / match_hi. The host
// observes the final values only after waitUntilCompleted, which acts
// as a full memory barrier between GPU compute and host reads. No need
// for atomic_ulong here -- match_lo / match_hi have a single writer
// per dispatch.
// ---------------------------------------------------------------------------

kernel void puzzle_search(
    device const ulong*        g_table       [[buffer(0)]],
    device const uchar*        target_h160   [[buffer(1)]],
    constant     ulong&        start_lo      [[buffer(2)]],
    constant     ulong&        start_hi      [[buffer(3)]],
    constant     ulong&        total_keys    [[buffer(4)]],
    device       ulong*        match_lo      [[buffer(5)]],
    device       ulong*        match_hi      [[buffer(6)]],
    device       atomic_uint*  match_found   [[buffer(7)]],
    uint                       gid           [[thread_position_in_grid]])
{
    if ((ulong)gid >= total_keys) return;

    // Early exit if another thread already found the key. atomic_load with
    // relaxed order is fine -- we just want to bail out best-effort; the
    // CAS below is the actual correctness barrier.
    if (atomic_load_explicit(match_found, memory_order_relaxed) != 0u) return;

    // k = start + gid (256-bit add; gid is a 32-bit thread index, so the
    // top 192 bits of the offset are zero; only limbs 0 and 1 ever change).
    ulong k[4];
    ulong c = 0;
    k[0] = addc(start_lo, (ulong)gid, c);
    k[1] = addc(start_hi, 0,          c);
    k[2] = c;                                        // carry into limb 2
    k[3] = 0;
    // (carry past limb 2 cannot happen for any feasible puzzle range.)

    // P = k * G via 8-bit windowed table.
    ulong px[4], py[4], pz[4];
    scalar_mul_g(px, py, pz, k, g_table);

    // Affinize: x = X / Z^2, y = Y / Z^3.
    if (limbs_zero(pz)) return;     // k*G == infinity (k = 0 mod n); skip.
    ulong z_inv[4], z_inv2[4], z_inv3[4];
    mod_inv(z_inv,  pz);
    mod_sqr(z_inv2, z_inv);
    mod_mul(z_inv3, z_inv2, z_inv);

    ulong ax[4], ay[4];
    mod_mul(ax, px, z_inv2);
    mod_mul(ay, py, z_inv3);

    // Compress + hash + compare.
    uchar pubkey[33], sha[32], h160[20];
    compress_pubkey(pubkey, ax, ay);
    sha256_33bytes(pubkey, sha);
    ripemd160_32bytes(sha, h160);

    bool match = true;
    for (int i = 0; i < 20; ++i) {
        if (h160[i] != target_h160[i]) { match = false; break; }
    }

    if (match) {
        // Atomic-CAS the match_found flag; the first thread to flip 0->1
        // owns the (lo, hi) slots for the rest of the dispatch.
        // Subsequent matches (degenerate but possible if the user happens
        // to have multiple keys hashing to the same h160) are dropped.
        uint expected = 0u;
        if (atomic_compare_exchange_weak_explicit(
                match_found, &expected, 1u,
                memory_order_relaxed, memory_order_relaxed))
        {
            // Plain stores: the CAS guarantees we're the only writer.
            // Other threads bailed out at the atomic_load on match_found
            // before reaching this branch. The host's waitUntilCompleted
            // is the cross-domain memory barrier that lets it observe
            // these writes.
            match_lo[0] = k[0];
            match_hi[0] = k[1];
        }
    }
}
