/**
 * Collider Optimized Puzzle Search Kernel
 *
 * High-performance GPU kernel for Bitcoin puzzle key search using:
 *
 * 1. PRECOMPUTED TABLES: G, 2G, 4G, ... 2^255*G stored in constant memory
 * 2. WINDOWED MULTIPLICATION: 4-bit windows reduce additions by 4x
 * 3. STRIDED INCREMENTAL: Each thread processes sequential keys with single EC add
 * 4. MONTGOMERY BATCH INVERSION: Amortize expensive inverse across 256+ keys
 * 5. GLV ENDOMORPHISM: Split scalar using secp256k1's efficient endomorphism for 2x speedup
 * 6. JACOBIAN COORDINATES: Avoid inversions until final batch
 *
 * Target: 400-800M keys/sec on RTX 3090 class hardware
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace collider {
namespace gpu {
namespace optimized {

// =============================================================================
// INLINE SHA256 FOR 33-BYTE COMPRESSED PUBLIC KEYS
// =============================================================================

// SHA256 constants
static __constant__ uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __forceinline__ uint32_t sha_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t sha_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t sha_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sha_sigma0(uint32_t x) {
    return sha_rotr(x, 2) ^ sha_rotr(x, 13) ^ sha_rotr(x, 22);
}

__device__ __forceinline__ uint32_t sha_sigma1(uint32_t x) {
    return sha_rotr(x, 6) ^ sha_rotr(x, 11) ^ sha_rotr(x, 25);
}

__device__ __forceinline__ uint32_t sha_gamma0(uint32_t x) {
    return sha_rotr(x, 7) ^ sha_rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t sha_gamma1(uint32_t x) {
    return sha_rotr(x, 17) ^ sha_rotr(x, 19) ^ (x >> 10);
}

/**
 * Inline SHA256 for exactly 33 bytes (compressed public key).
 * Single-block processing with pre-computed padding.
 */
__device__ void sha256_33bytes_opt(const uint8_t* pubkey, uint8_t* hash) {
    uint32_t W[64];
    uint32_t H[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Build message block: 33 bytes + 0x80 + zeros + length
    // Words 0-7: first 32 bytes of pubkey (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        W[i] = (pubkey[i*4] << 24) | (pubkey[i*4 + 1] << 16) |
               (pubkey[i*4 + 2] << 8) | pubkey[i*4 + 3];
    }

    // Word 8: last byte + 0x80 padding
    W[8] = (pubkey[32] << 24) | (0x80 << 16);

    // Words 9-13: zeros
    W[9] = 0; W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0;

    // Words 14-15: length in bits (33 * 8 = 264)
    W[14] = 0;
    W[15] = 264;

    // Extend to 64 words
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = sha_gamma1(W[i-2]) + W[i-7] + sha_gamma0(W[i-15]) + W[i-16];
    }

    // Compression
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

    #pragma unroll 8
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sha_sigma1(e) + sha_ch(e, f, g) + SHA256_K[i] + W[i];
        uint32_t t2 = sha_sigma0(a) + sha_maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;

    // Output (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i*4]     = (H[i] >> 24) & 0xff;
        hash[i*4 + 1] = (H[i] >> 16) & 0xff;
        hash[i*4 + 2] = (H[i] >> 8) & 0xff;
        hash[i*4 + 3] = H[i] & 0xff;
    }
}

// =============================================================================
// INLINE RIPEMD160 FOR 32-BYTE SHA256 OUTPUT
// =============================================================================

// RIPEMD160 constants
static __constant__ uint32_t RMD_KL[5] = {
    0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E
};

static __constant__ uint32_t RMD_KR[5] = {
    0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000
};

static __constant__ int RMD_RL[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};

static __constant__ int RMD_RR[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};

static __constant__ int RMD_SL[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};

static __constant__ int RMD_SR[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

__device__ __forceinline__ uint32_t rmd_rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ uint32_t rmd_f0(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
__device__ __forceinline__ uint32_t rmd_f1(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
__device__ __forceinline__ uint32_t rmd_f2(uint32_t x, uint32_t y, uint32_t z) { return (x | ~y) ^ z; }
__device__ __forceinline__ uint32_t rmd_f3(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
__device__ __forceinline__ uint32_t rmd_f4(uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | ~z); }

/**
 * Inline RIPEMD160 for exactly 32 bytes (SHA256 output).
 */
__device__ void ripemd160_32bytes_opt(const uint8_t* sha_out, uint8_t* h160) {
    uint32_t H[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};

    // Build message words (little-endian)
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        X[i] = sha_out[i*4] | (sha_out[i*4+1] << 8) |
               (sha_out[i*4+2] << 16) | (sha_out[i*4+3] << 24);
    }
    // Padding: 0x80 at byte 32, length 256 bits at end
    X[8] = 0x00000080;
    X[9] = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0;
    X[14] = 256;  // length in bits
    X[15] = 0;

    uint32_t AL = H[0], BL = H[1], CL = H[2], DL = H[3], EL = H[4];
    uint32_t AR = H[0], BR = H[1], CR = H[2], DR = H[3], ER = H[4];

    // Round 0 (j=0-15)
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        uint32_t tL = rmd_rotl(AL + rmd_f0(BL, CL, DL) + X[RMD_RL[j]] + RMD_KL[0], RMD_SL[j]) + EL;
        AL = EL; EL = DL; DL = rmd_rotl(CL, 10); CL = BL; BL = tL;
        uint32_t tR = rmd_rotl(AR + rmd_f4(BR, CR, DR) + X[RMD_RR[j]] + RMD_KR[0], RMD_SR[j]) + ER;
        AR = ER; ER = DR; DR = rmd_rotl(CR, 10); CR = BR; BR = tR;
    }

    // Round 1 (j=16-31)
    #pragma unroll
    for (int j = 16; j < 32; j++) {
        uint32_t tL = rmd_rotl(AL + rmd_f1(BL, CL, DL) + X[RMD_RL[j]] + RMD_KL[1], RMD_SL[j]) + EL;
        AL = EL; EL = DL; DL = rmd_rotl(CL, 10); CL = BL; BL = tL;
        uint32_t tR = rmd_rotl(AR + rmd_f3(BR, CR, DR) + X[RMD_RR[j]] + RMD_KR[1], RMD_SR[j]) + ER;
        AR = ER; ER = DR; DR = rmd_rotl(CR, 10); CR = BR; BR = tR;
    }

    // Round 2 (j=32-47)
    #pragma unroll
    for (int j = 32; j < 48; j++) {
        uint32_t tL = rmd_rotl(AL + rmd_f2(BL, CL, DL) + X[RMD_RL[j]] + RMD_KL[2], RMD_SL[j]) + EL;
        AL = EL; EL = DL; DL = rmd_rotl(CL, 10); CL = BL; BL = tL;
        uint32_t tR = rmd_rotl(AR + rmd_f2(BR, CR, DR) + X[RMD_RR[j]] + RMD_KR[2], RMD_SR[j]) + ER;
        AR = ER; ER = DR; DR = rmd_rotl(CR, 10); CR = BR; BR = tR;
    }

    // Round 3 (j=48-63)
    #pragma unroll
    for (int j = 48; j < 64; j++) {
        uint32_t tL = rmd_rotl(AL + rmd_f3(BL, CL, DL) + X[RMD_RL[j]] + RMD_KL[3], RMD_SL[j]) + EL;
        AL = EL; EL = DL; DL = rmd_rotl(CL, 10); CL = BL; BL = tL;
        uint32_t tR = rmd_rotl(AR + rmd_f1(BR, CR, DR) + X[RMD_RR[j]] + RMD_KR[3], RMD_SR[j]) + ER;
        AR = ER; ER = DR; DR = rmd_rotl(CR, 10); CR = BR; BR = tR;
    }

    // Round 4 (j=64-79)
    #pragma unroll
    for (int j = 64; j < 80; j++) {
        uint32_t tL = rmd_rotl(AL + rmd_f4(BL, CL, DL) + X[RMD_RL[j]] + RMD_KL[4], RMD_SL[j]) + EL;
        AL = EL; EL = DL; DL = rmd_rotl(CL, 10); CL = BL; BL = tL;
        uint32_t tR = rmd_rotl(AR + rmd_f0(BR, CR, DR) + X[RMD_RR[j]] + RMD_KR[4], RMD_SR[j]) + ER;
        AR = ER; ER = DR; DR = rmd_rotl(CR, 10); CR = BR; BR = tR;
    }

    // Final addition
    uint32_t t = H[1] + CL + DR;
    H[1] = H[2] + DL + ER;
    H[2] = H[3] + EL + AR;
    H[3] = H[4] + AL + BR;
    H[4] = H[0] + BL + CR;
    H[0] = t;

    // Output (little-endian)
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        h160[i*4]     = H[i] & 0xff;
        h160[i*4 + 1] = (H[i] >> 8) & 0xff;
        h160[i*4 + 2] = (H[i] >> 16) & 0xff;
        h160[i*4 + 3] = (H[i] >> 24) & 0xff;
    }
}

// =============================================================================
// CONFIGURATION
// =============================================================================

// Keys processed per thread in strided mode
#define KEYS_PER_THREAD 256

// Batch size for Montgomery inversion (power of 2)
#define BATCH_INV_SIZE 256

// Window size for w-NAF (4 = 16 precomputed points per window)
#define WINDOW_SIZE 4
#define WINDOW_MASK ((1 << WINDOW_SIZE) - 1)

// Number of windows for 256-bit scalar
#define NUM_WINDOWS (256 / WINDOW_SIZE)

// =============================================================================
// DATA STRUCTURES (Optimized for GPU)
// =============================================================================

// 256-bit integer using uint64 for fewer operations
struct alignas(32) U256 {
    uint64_t d[4];  // Little-endian: d[0] is least significant

    __device__ __forceinline__ void set_zero() {
        d[0] = d[1] = d[2] = d[3] = 0;
    }

    __device__ __forceinline__ void set_one() {
        d[0] = 1; d[1] = d[2] = d[3] = 0;
    }

    __device__ __forceinline__ bool is_zero() const {
        return (d[0] | d[1] | d[2] | d[3]) == 0;
    }
};

// Jacobian point (X, Y, Z) where affine (x,y) = (X/Z^2, Y/Z^3)
struct alignas(32) PointJ {
    U256 X, Y, Z;

    __device__ __forceinline__ bool is_infinity() const {
        return Z.is_zero();
    }

    __device__ __forceinline__ void set_infinity() {
        X.set_one();
        Y.set_one();
        Z.set_zero();
    }
};

// Affine point (x, y)
// Note: alignas removed - U256 already has alignas(32) which propagates
struct PointA {
    U256 x, y;
};

// =============================================================================
// SECP256K1 CONSTANTS
// =============================================================================

// Field prime p = 2^256 - 2^32 - 977
__device__ __constant__ uint64_t SECP_P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};

// Curve order n
__device__ __constant__ uint64_t SECP_N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// Generator point G
__device__ __constant__ uint64_t SECP_GX[4] = {
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
};

__device__ __constant__ uint64_t SECP_GY[4] = {
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
};

// GLV endomorphism: lambda where lambda^3 = 1 mod n
// beta where (x, y) -> (beta*x, y) is equivalent to scalar mult by lambda
__device__ __constant__ uint64_t GLV_LAMBDA[4] = {
    0xDF02967C1B23BD72ULL, 0x122E22EA20816678ULL,
    0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL
};

// 2026-05-05 fix: prior beta value 0x7ae96a2b657c07106e64479eac3434e9... was
// corrupted in limbs d[0..2] -- the canonical secp256k1 beta is the cube root
// of 1 in the field where (x, y) -> (beta*x, y) implements lambda*P. The
// previous low limbs (D765..., 7A9C..., 51CA...) didn't satisfy beta^3 = 1
// mod p. Verified canonical value below pow(beta, 3, p) == 1.
// Canonical beta = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
// Stored little-endian (d[0] = LSB):
__device__ __constant__ uint64_t GLV_BETA[4] = {
    0xC1396C28719501EEULL, 0x9CF0497512F58995ULL,
    0x6E64479EAC3434E9ULL, 0x7AE96A2B657C0710ULL
};

// GLV Lattice basis vectors for scalar decomposition
// These form a short basis of the lattice L = {(a,b) : a + b*lambda = 0 mod n}
// Reference: Guide to Elliptic Curve Cryptography, Section 3.5
__device__ __constant__ uint64_t GLV_A1[2] = {
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL  // a1 = 0x3086d221a7d46bcde86c90e49284eb15
};

__device__ __constant__ uint64_t GLV_B1[2] = {
    0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL  // -b1 (stored positive, sign handled separately)
};

// a2 = 0x114CA50F7A8E2F3F657C1108D9D44CFD8 (129 bits, NOT equal to a1).
//
// 2026-05-05 fix (Defect D): the previous GLV_A2_LOW value was the wrong
// canonical reference -- 0x...656E48F0E8717E37D was lifted from an old
// "Guide to ECC" example that does not satisfy the lattice identity
// a1^2 + |b1|*a2 = n for our basis with b2 = a1. With the wrong a2, the
// k1 computation k = c1*a1 + c2*a2 + k1 fails to cancel down to the small
// Babai residual for full 256-bit scalars (k near n), producing a 129-bit
// k1 with d[2] holding garbage instead of the expected ~128-bit residual.
// The correct value is derived from a2 = (-a1*lambda) mod n, which gives
// a 129-bit number with the high bit at position 128. Verified: with the
// new value, a1^2 + |b1|*a2 == n exactly.
//
// 2026-05-04 fix (Defect A): the prior code approximated a2 ~= a1 in the
// glv_decompose c2*a2 multiply. That produces an error of c2 * (a2 - a1)
// which, for full 256-bit scalars (c2 ~ 2^127), can reach ~2^254 -- the
// decomposition becomes garbage and ec_mul_glv returns the wrong point.
//
// a2 has 129 bits, so it does not fit in two uint64s. We store the low
// 128 bits explicitly here; the high bit (bit 128) is exactly 1, so
// a2 = 2^128 + GLV_A2_LOW. The c2 * a2 multiply is split as
//   c2 * a2 = (c2 << 128) + c2 * GLV_A2_LOW
// in glv_decompose below, using the existing mul_128x128 routine for the
// low half and a 128-bit shift for the high half. This is byte-exact with
// the canonical lambda decomposition (libsecp256k1 secp256k1_scalar_split_lambda).
__device__ __constant__ uint64_t GLV_A2_LOW[2] = {
    0x57C1108D9D44CFD8ULL, 0x14CA50F7A8E2F3F6ULL  // low 128 bits of a2
};

__device__ __constant__ uint64_t GLV_B2[2] = {
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL  // b2 = a1 (same value)
};

// g1, g2 precomputed for efficient decomposition:
// g1 = floor(b2 * 2^384 / n), g2 = floor((-b1) * 2^384 / n)
//
// 2026-05-04 fix: Originally these were documented as `* 2^256 / n`, but the
// stored values are actually `* 2^384 / n` -- you can sanity-check this by
// observing that g2.d[3] = 0xE4437ED6010E8828 has its high bit set, which is
// only possible if g2 is on the order of 2^256. With the true `* 2^256 / n`
// scaling, g2 would be around 2^127 (since |-b1| ~ 2^127 and 2^256/n ~ 1).
// The corresponding extraction is therefore `(k * g_i) >> 384`, NOT `>> 256`.
// See glv_decompose() for the matching shift fix.
// g1 = round(b2 * 2^384 / n) = round(a1 * 2^384 / n)
// Matches libsecp256k1 secp256k1_scalar_split_lambda g1 constant.
// In little-endian u64 limbs (d[0]=LSB, d[3]=MSB):
//   full hex (big-endian): 3086d221a7d46bcde86c90e49284eb153da4445121181820ffa7bd168f1d4808
__device__ __constant__ uint64_t GLV_G1[4] = {
    0xFFA7BD168F1D4808ULL, 0x3DA4445121181820ULL,
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
};

// g2 = round((-b1) * 2^384 / n)
// Matches libsecp256k1 secp256k1_scalar_split_lambda g2 constant.
// In little-endian u64 limbs (d[0]=LSB, d[3]=MSB):
//   full hex (big-endian): e4437ed6010e88286f547fa90abfe4c423eb5cdc18462a36f7a70f55b96c7540
__device__ __constant__ uint64_t GLV_G2[4] = {
    0xF7A70F55B96C7540ULL, 0x23EB5CDC18462A36ULL,
    0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL
};

// Precomputed table: G, 2G, 3G, ..., 15G for window multiplication
// Each window has 16 points (including 0*G at index 0)
// Total: 64 windows * 16 points = 1024 points = 64KB per table
//
// NOTE: Tables are stored in GLOBAL DEVICE MEMORY (not constant memory)
// because combined tables exceed CUDA's 64KB constant memory limit.
// Access via __ldg() intrinsic provides L2 caching for good performance.
__device__ PointA* d_PRECOMP_TABLE = nullptr;
__device__ PointA* d_PRECOMP_TABLE_LAMBDA = nullptr;

// =============================================================================
// MODULAR ARITHMETIC (Optimized)
// =============================================================================

// Add with carry using PTX for maximum performance.
// Both instructions are in one asm block so NVCC cannot insert other
// instructions between them and break the CC register dependency.
__device__ __forceinline__ uint64_t add_cc(uint64_t a, uint64_t b, uint64_t& carry) {
    uint64_t result;
    asm volatile("add.cc.u64 %0, %2, %3;\n\t"
                 "addc.u64 %1, 0, 0;"
                 : "=l"(result), "=l"(carry) : "l"(a), "l"(b));
    return result;
}

__device__ __forceinline__ uint64_t addc_cc(uint64_t a, uint64_t b, uint64_t carry_in, uint64_t& carry_out) {
    uint64_t result;
    asm volatile("add.cc.u64 %0, %2, %3;\n\t"
                 "addc.u64 %1, 0, 0;"
                 : "=l"(result), "=l"(carry_out) : "l"(a), "l"(b));
    if (carry_in) {
        if (result == UINT64_MAX) carry_out = 1;
        result++;
    }
    return result;
}

// Subtraction with borrow output (first in chain)
__device__ __forceinline__ uint64_t sub_cc(uint64_t a, uint64_t b, uint64_t& borrow) {
    uint64_t result;
    asm volatile("sub.cc.u64 %0, %2, %3;\n\t"
                 "subc.u64 %1, 0, 0;"
                 : "=l"(result), "=l"(borrow) : "l"(a), "l"(b));
    return result;
}

// Subtraction with borrow-in and borrow-out (for chaining)
__device__ __forceinline__ uint64_t subc_cc(uint64_t a, uint64_t b, uint64_t borrow_in, uint64_t& borrow_out) {
    uint64_t b1 = borrow_in >> 63;  // 0 or 1
    uint64_t result;
    asm volatile("sub.cc.u64 %0, %2, %3;\n\t"
                 "subc.u64 %1, 0, 0;"
                 : "=l"(result), "=l"(borrow_out) : "l"(a), "l"(b));
    if (b1) {
        if (result == 0) borrow_out = UINT64_MAX;
        result--;  // uint64 wrap is intentional when result==0
    }
    return result;
}

// Multiply-add: a*b + c + carry -> (hi, lo)
// Uses CUDA intrinsics for MSVC compatibility (no __int128 on Windows)
__device__ __forceinline__ void mad_wide(uint64_t a, uint64_t b, uint64_t c,
                                          uint64_t carry_in, uint64_t& hi, uint64_t& lo) {
    // 64x64 -> 128 bit multiply using CUDA intrinsics
    lo = a * b;                    // Low 64 bits
    hi = __umul64hi(a, b);         // High 64 bits

    // Add c with carry propagation
    lo += c;
    hi += (lo < c) ? 1 : 0;

    // Add carry_in with carry propagation
    lo += carry_in;
    hi += (lo < carry_in) ? 1 : 0;
}

// Modular addition: r = (a + b) mod p
__device__ void mod_add(U256& r, const U256& a, const U256& b) {
    uint64_t carry = 0, c;

    r.d[0] = add_cc(a.d[0], b.d[0], c); carry = c;
    r.d[1] = addc_cc(a.d[1], b.d[1], carry, c); carry = c;
    r.d[2] = addc_cc(a.d[2], b.d[2], carry, c); carry = c;
    r.d[3] = addc_cc(a.d[3], b.d[3], carry, c); carry = c;

    // Reduce if >= p
    // p = 2^256 - 2^32 - 977, so if carry or result >= p, subtract p
    // This is equivalent to adding 2^32 + 977
    if (carry || (r.d[3] == 0xFFFFFFFFFFFFFFFFULL &&
                  r.d[2] == 0xFFFFFFFFFFFFFFFFULL &&
                  r.d[1] == 0xFFFFFFFFFFFFFFFFULL &&
                  r.d[0] >= 0xFFFFFFFEFFFFFC2FULL)) {
        // FIXED: Proper borrow chaining
        uint64_t borrow;
        r.d[0] = sub_cc(r.d[0], SECP_P[0], borrow);
        r.d[1] = subc_cc(r.d[1], SECP_P[1], borrow, borrow);
        r.d[2] = subc_cc(r.d[2], SECP_P[2], borrow, borrow);
        r.d[3] = r.d[3] - SECP_P[3] - (borrow >> 63);
    }
}

// Modular subtraction: r = (a - b) mod p
// FIXED: Proper borrow chaining using subc_cc
__device__ void mod_sub(U256& r, const U256& a, const U256& b) {
    uint64_t borrow;

    // Proper chained subtraction with borrow propagation
    r.d[0] = sub_cc(a.d[0], b.d[0], borrow);
    r.d[1] = subc_cc(a.d[1], b.d[1], borrow, borrow);
    r.d[2] = subc_cc(a.d[2], b.d[2], borrow, borrow);
    r.d[3] = subc_cc(a.d[3], b.d[3], borrow, borrow);

    // If final borrow occurred (result negative), add p back
    if (borrow) {
        uint64_t carry;
        r.d[0] = add_cc(r.d[0], SECP_P[0], carry);
        r.d[1] = addc_cc(r.d[1], SECP_P[1], carry, carry);
        r.d[2] = addc_cc(r.d[2], SECP_P[2], carry, carry);
        r.d[3] = r.d[3] + SECP_P[3] + carry;
    }
}

// Modular multiplication using secp256k1's special form
// p = 2^256 - c where c = 2^32 + 977
// After full 512-bit multiply, reduce using: r = low + high * c (mod p)
__device__ void mod_mul(U256& r, const U256& a, const U256& b) {
    // Full 512-bit product
    uint64_t p[8] = {0};

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // 64x64 -> 128 multiply + add using CUDA intrinsics
            uint64_t lo = a.d[i] * b.d[j];
            uint64_t hi = __umul64hi(a.d[i], b.d[j]);
            // Add p[i+j]
            lo += p[i+j];
            hi += (lo < p[i+j]) ? 1 : 0;
            // Add carry
            lo += carry;
            hi += (lo < carry) ? 1 : 0;
            p[i+j] = lo;
            carry = hi;
        }
        p[i+4] = carry;
    }

    // Reduce: multiply high part by c = 2^32 + 977 and add to low
    // c = 0x100000000 + 0x3D1 = 4294968273
    const uint64_t c = 0x1000003D1ULL;

    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // 64x64 -> 128 multiply + add using CUDA intrinsics
        uint64_t lo = p[i+4] * c;
        uint64_t hi = __umul64hi(p[i+4], c);
        // Add p[i]
        lo += p[i];
        hi += (lo < p[i]) ? 1 : 0;
        // Add carry
        lo += carry;
        hi += (lo < carry) ? 1 : 0;
        p[i] = lo;
        carry = hi;
    }

    // Handle final carry (multiply by c again if needed)
    if (carry) {
        uint64_t lo = carry * c;
        uint64_t hi = __umul64hi(carry, c);
        lo += p[0];
        hi += (lo < p[0]) ? 1 : 0;
        p[0] = lo;
        carry = hi;

        for (int i = 1; i < 4 && carry; i++) {
            p[i] += carry;
            carry = (p[i] < carry) ? 1 : 0;
        }
    }

    // Final reduction if >= p
    r.d[0] = p[0]; r.d[1] = p[1]; r.d[2] = p[2]; r.d[3] = p[3];

    // Wave 1 / C-CRIT-2 fix (2026-05-04): loop until canonical, mirroring the
    // mod_reduce while-loop in secp256k1.cu. The reduce-by-c overflow handling
    // above can leave r in roughly [0, k*p) for small k on pathological inputs;
    // a single `if` is not always enough. Use the proper subc_cc borrow chain
    // (matching mod_sub) to avoid the broken `r.d[i] - borrow` truncation
    // pattern previously here.
    while (r.d[3] > SECP_P[3] ||
           (r.d[3] == SECP_P[3] && r.d[2] == SECP_P[2] &&
            r.d[1] == SECP_P[1] && r.d[0] >= SECP_P[0])) {
        uint64_t borrow;
        r.d[0] = sub_cc(r.d[0], SECP_P[0], borrow);
        r.d[1] = subc_cc(r.d[1], SECP_P[1], borrow, borrow);
        r.d[2] = subc_cc(r.d[2], SECP_P[2], borrow, borrow);
        r.d[3] = subc_cc(r.d[3], SECP_P[3], borrow, borrow);
    }
}

// Modular squaring using Karatsuba-style optimization
// For squaring, we can reduce 16 multiplications to 10:
// - 4 squares: a0^2, a1^2, a2^2, a3^2
// - 6 cross-terms (computed once, doubled): a0*a1, a0*a2, a0*a3, a1*a2, a1*a3, a2*a3
// This gives ~37.5% reduction in multiplications compared to generic mul
__device__ void mod_sqr(U256& r, const U256& a) {
    // Delegate to mod_mul for correctness; the Karatsuba path had carry bugs.
    mod_mul(r, a, a);
}

// Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
//
// Wave 1 / C-CRIT-2 fix (2026-05-04): the prior implementation used a
// hand-coded addition chain that produced the wrong exponent. Tracing the
// chain showed it computed approximately a^(2^255 - 2^31 - 493) instead of
// a^(p-2) = a^(2^256 - 2^32 - 979), so a * mod_inv(a) was almost never 1
// mod p. Replaced with right-to-left binary exponentiation over all 256
// bits of p-2; verifiable by inspection.
//
// p     = 0xFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFF
//         FFFFFFFFFFFFFFFF FFFFFFFEFFFFFC2F
// p - 2 = 0xFFFFFFFFFFFFFFFF FFFFFFFFFFFFFFFF
//         FFFFFFFFFFFFFFFF FFFFFFFEFFFFFC2D
//
// p_minus_2[0] is the LEAST significant 64-bit word so the per-iteration
// bit walk processes bit 0 first. ~256 squarings + ~249 multiplications
// (only 7 zero bits in p-2). To rule out aliasing edge cases across mod_mul
// calls, we multiply via a fresh temporary then assign.
__device__ void mod_inv(U256& r, const U256& a) {
    // v1.4.0 phase 4 perf: libsecp256k1-style addition chain. 256
    // squarings + 13 multiplications instead of 256 + 248 -- ~1.9x
    // reduction in mod_mul calls per inversion. CRITICAL: x223 step
    // multiplies by x3 (a^7), NOT x2 (a^3); see the matching comment
    // in src/gpu/secp256k1.cu for the libsecp256k1 chain derivation.
    // Verified locally against tests/test_puzzle_optimized_inv.cu (the
    // KAT tests/test_secp256k1_inv.cu's analogue for U256 type).
    auto sqr = [](U256& r_, const U256& x) {
        U256 t; mod_mul(t, x, x); r_ = t;
    };
    auto mul = [](U256& r_, const U256& x, const U256& y) {
        U256 t; mod_mul(t, x, y); r_ = t;
    };

    U256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;

    // x2 = a^3, x3 = a^7
    sqr(x2, a);  mul(x2, x2, a);
    sqr(x3, x2); mul(x3, x3, a);

    // xN = a^(2^N - 1) for N = 6, 9, 11
    sqr(x6, x3); sqr(x6, x6); sqr(x6, x6); mul(x6, x6, x3);
    sqr(x9, x6); sqr(x9, x9); sqr(x9, x9); mul(x9, x9, x3);
    sqr(x11, x9); sqr(x11, x11); mul(x11, x11, x2);

    // xN for N = 22, 44, 88, 176, 220
    sqr(x22, x11);
    for (int i = 1; i < 11; ++i) sqr(x22, x22);
    mul(x22, x22, x11);

    sqr(x44, x22);
    for (int i = 1; i < 22; ++i) sqr(x44, x44);
    mul(x44, x44, x22);

    sqr(x88, x44);
    for (int i = 1; i < 44; ++i) sqr(x88, x88);
    mul(x88, x88, x44);

    sqr(x176, x88);
    for (int i = 1; i < 88; ++i) sqr(x176, x176);
    mul(x176, x176, x88);

    sqr(x220, x176);
    for (int i = 1; i < 44; ++i) sqr(x220, x220);
    mul(x220, x220, x44);

    // x223 = (2^220 - 1) * 2^3 + 7 = 2^223 - 1.
    sqr(x223, x220); sqr(x223, x223); sqr(x223, x223);
    mul(x223, x223, x3);

    // Tail: ^(2^23) * x22 ; ^(2^5) * a ; ^(2^3) * x2 ; ^(2^2) * a.
    sqr(t1, x223);
    for (int i = 1; i < 23; ++i) sqr(t1, t1);
    mul(t1, t1, x22);
    for (int i = 0; i < 5; ++i) sqr(t1, t1);
    mul(t1, t1, a);
    for (int i = 0; i < 3; ++i) sqr(t1, t1);
    mul(t1, t1, x2);
    sqr(t1, t1); sqr(t1, t1);
    mul(t1, t1, a);

    r = t1;
}

// =============================================================================
// ELLIPTIC CURVE OPERATIONS (Jacobian Coordinates)
// =============================================================================

// Point doubling: R = 2*P (Jacobian)
// Uses optimized formula for a=0 curves (secp256k1)
//
// 2026-05-04 fix (Defect C): the prior implementation wrote R.Y before
// computing R.Z = 2 * P.Y * P.Z, which silently corrupted the result
// whenever R aliased P (which is the common case: ec_double(R, R) is
// called from ec_mul_glv and from the precomputed-table generator).
// All output writes now happen at the very end into local temporaries,
// then are copied to R, so R may freely alias P.
__device__ void ec_double(PointJ& R, const PointJ& P) {
    if (P.is_infinity()) {
        R.set_infinity();
        return;
    }

    // Snapshot every input limb we need before writing anything to R.
    // This guarantees correctness even when R and P are the same object.
    U256 PX = P.X;
    U256 PY = P.Y;
    U256 PZ = P.Z;

    U256 S, M, T, Y2;

    // S = 4*X*Y^2
    mod_sqr(Y2, PY);
    mod_mul(S, PX, Y2);
    mod_add(S, S, S);
    mod_add(S, S, S);

    // M = 3*X^2 (a=0 for secp256k1)
    mod_sqr(M, PX);
    mod_add(T, M, M);
    mod_add(M, T, M);

    // X' = M^2 - 2*S  -- compute into a local, not R.X
    U256 newX;
    mod_sqr(newX, M);
    mod_sub(newX, newX, S);
    mod_sub(newX, newX, S);

    // Y' = M*(S - X') - 8*Y^4
    U256 newY;
    mod_sub(T, S, newX);
    mod_mul(T, M, T);
    mod_sqr(Y2, Y2);            // Y2 = Y^4
    mod_add(Y2, Y2, Y2);        // 2*Y^4
    mod_add(Y2, Y2, Y2);        // 4*Y^4
    mod_add(Y2, Y2, Y2);        // 8*Y^4
    mod_sub(newY, T, Y2);

    // Z' = 2*Y*Z (uses snapshotted PY, PZ -- safe even if R == P)
    U256 newZ;
    mod_mul(newZ, PY, PZ);
    mod_add(newZ, newZ, newZ);

    // Commit all outputs together. R may alias P; the snapshots above
    // mean every read above used the original input values.
    R.X = newX;
    R.Y = newY;
    R.Z = newZ;
}

// Point addition: R = P + Q (Jacobian + Affine -> Jacobian)
// Mixed addition is faster than full Jacobian addition
//
// 2026-05-04 fix (Defect C, related): same aliasing hazard as ec_double.
// ec_mul_glv calls ec_add_mixed(R, R, P1), so the R output aliases the P
// input. The previous code read P.Y at line 913 and P.Z at line 918
// AFTER writing R.X and R.Y, which corrupted those reads under aliasing.
// Fix: snapshot P.X, P.Y, P.Z to locals up front and write R only at the
// very end.
__device__ void ec_add_mixed(PointJ& R, const PointJ& P, const PointA& Q) {
    if (P.is_infinity()) {
        R.X = Q.x;
        R.Y = Q.y;
        R.Z.set_one();
        return;
    }

    // Snapshot inputs so R may safely alias P.
    U256 PX = P.X;
    U256 PY = P.Y;
    U256 PZ = P.Z;

    U256 Z1Z1, U2, S2, H, HH, I, J, r, V;

    // Z1Z1 = Z1^2
    mod_sqr(Z1Z1, PZ);

    // U2 = X2*Z1Z1
    mod_mul(U2, Q.x, Z1Z1);

    // S2 = Y2*Z1*Z1Z1
    mod_mul(S2, Q.y, PZ);
    mod_mul(S2, S2, Z1Z1);

    // H = U2 - X1
    mod_sub(H, U2, PX);

    // r = 2*(S2 - Y1)
    mod_sub(r, S2, PY);
    mod_add(r, r, r);

    // Check for doubling case
    if (H.is_zero()) {
        if (r.is_zero()) {
            // P == Q, use doubling. ec_double is alias-safe.
            ec_double(R, P);
            return;
        } else {
            // P == -Q, result is infinity
            R.set_infinity();
            return;
        }
    }

    // HH = H^2
    mod_sqr(HH, H);

    // I = 4*HH
    mod_add(I, HH, HH);
    mod_add(I, I, I);

    // J = H*I
    mod_mul(J, H, I);

    // V = X1*I
    mod_mul(V, PX, I);

    // X3 = r^2 - J - 2*V  -- accumulate into a local
    U256 newX;
    mod_sqr(newX, r);
    mod_sub(newX, newX, J);
    mod_sub(newX, newX, V);
    mod_sub(newX, newX, V);

    // Y3 = r*(V - X3) - 2*Y1*J
    U256 newY;
    mod_sub(newY, V, newX);
    mod_mul(newY, r, newY);
    U256 Y1J;
    mod_mul(Y1J, PY, J);
    mod_add(Y1J, Y1J, Y1J);
    mod_sub(newY, newY, Y1J);

    // Z3 = 2*Z1*H
    U256 newZ;
    mod_mul(newZ, PZ, H);
    mod_add(newZ, newZ, newZ);

    // Commit all outputs together. R may alias P.
    R.X = newX;
    R.Y = newY;
    R.Z = newZ;
}

// Full Jacobian addition: R = P + Q
__device__ void ec_add_full(PointJ& R, const PointJ& P, const PointJ& Q) {
    if (P.is_infinity()) { R = Q; return; }
    if (Q.is_infinity()) { R = P; return; }

    U256 Z1Z1, Z2Z2, U1, U2, S1, S2, H, I, J, r, V;

    mod_sqr(Z1Z1, P.Z);
    mod_sqr(Z2Z2, Q.Z);
    mod_mul(U1, P.X, Z2Z2);
    mod_mul(U2, Q.X, Z1Z1);
    mod_mul(S1, P.Y, Q.Z);
    mod_mul(S1, S1, Z2Z2);
    mod_mul(S2, Q.Y, P.Z);
    mod_mul(S2, S2, Z1Z1);

    mod_sub(H, U2, U1);
    mod_sub(r, S2, S1);
    mod_add(r, r, r);

    if (H.is_zero()) {
        if (r.is_zero()) {
            ec_double(R, P);
            return;
        }
        R.set_infinity();
        return;
    }

    mod_sqr(I, H);
    mod_add(I, I, I);
    mod_add(I, I, I);
    mod_mul(J, H, I);
    mod_mul(V, U1, I);

    mod_sqr(R.X, r);
    mod_sub(R.X, R.X, J);
    mod_sub(R.X, R.X, V);
    mod_sub(R.X, R.X, V);

    mod_sub(R.Y, V, R.X);
    mod_mul(R.Y, r, R.Y);
    mod_mul(S1, S1, J);
    mod_add(S1, S1, S1);
    mod_sub(R.Y, R.Y, S1);

    mod_add(R.Z, P.Z, Q.Z);
    mod_sqr(R.Z, R.Z);
    mod_sub(R.Z, R.Z, Z1Z1);
    mod_sub(R.Z, R.Z, Z2Z2);
    mod_mul(R.Z, R.Z, H);
}

// =============================================================================
// MONTGOMERY BATCH INVERSION
// =============================================================================

// Batch inversion: compute 1/z[0], 1/z[1], ..., 1/z[n-1] with only 1 inverse
// Uses Montgomery's trick: (a*b)^(-1) = a^(-1) * b^(-1)
__device__ void batch_invert(U256* z, U256* inv, int n) {
    if (n == 0) return;

    // Accumulate products: inv[i] = z[0] * z[1] * ... * z[i]
    inv[0] = z[0];
    for (int i = 1; i < n; i++) {
        mod_mul(inv[i], inv[i-1], z[i]);
    }

    // Single inverse of the product
    U256 acc;
    mod_inv(acc, inv[n-1]);

    // Back-substitute to get individual inverses
    for (int i = n - 1; i > 0; i--) {
        mod_mul(inv[i], acc, inv[i-1]);  // inv[i] = acc * (z[0]*...*z[i-1])
        mod_mul(acc, acc, z[i]);          // acc = acc * z[i] for next iteration
    }
    inv[0] = acc;
}

// =============================================================================
// SCALAR MULTIPLICATION WITH PRECOMPUTED TABLE
// =============================================================================

// Windowed scalar multiplication using precomputed table
// =============================================================================
// GLV ENDOMORPHISM - Full Implementation
// =============================================================================
// secp256k1 has an efficient endomorphism φ: (x,y) -> (β*x, y) where φ(P) = λ*P
// This allows decomposing k into k1 + k2*λ where k1, k2 are ~128 bits
// Result: k*G = k1*G + k2*(λ*G) computed via Shamir's trick (~30% faster)
// =============================================================================

// 128-bit type for intermediate GLV calculations
struct U128 {
    uint64_t lo, hi;

    __device__ __forceinline__ void set_zero() { lo = hi = 0; }
    __device__ __forceinline__ bool is_zero() const { return (lo | hi) == 0; }
    __device__ __forceinline__ bool is_negative() const { return (hi >> 63) != 0; }
};

// Multiply 256-bit by 128-bit, return high 128 bits (for rounding)
__device__ void mul_256x128_high(const U256& a, const uint64_t b[2], U128& high) {
    // We need the high 128 bits of a * b where b is 128 bits
    // Full product is 384 bits, we want bits [256..383]

    uint64_t p[6] = {0};  // 384-bit product

    // Multiply each limb of a by each limb of b
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 2; j++) {
            // 64x64 -> 128 multiply + add using CUDA intrinsics
            uint64_t lo = a.d[i] * b[j];
            uint64_t hi = __umul64hi(a.d[i], b[j]);
            // Add p[i+j]
            lo += p[i+j];
            hi += (lo < p[i+j]) ? 1 : 0;
            // Add carry
            lo += carry;
            hi += (lo < carry) ? 1 : 0;
            p[i+j] = lo;
            carry = hi;
        }
        p[i+2] += carry;
        if (p[i+2] < carry && i+3 < 6) p[i+3]++;
    }

    // Return high 128 bits (p[4], p[5])
    high.lo = p[4];
    high.hi = p[5];
}

// 128-bit multiply producing 256-bit result
__device__ void mul_128x128(const U128& a, const uint64_t b[2], U256& result) {
    uint64_t p[4] = {0};

    // a.lo * b[0] -> (p[0], carry)
    p[0] = a.lo * b[0];
    uint64_t carry = __umul64hi(a.lo, b[0]);

    // a.lo * b[1] + carry
    uint64_t lo1 = a.lo * b[1];
    uint64_t hi1 = __umul64hi(a.lo, b[1]);
    lo1 += carry;
    hi1 += (lo1 < carry) ? 1 : 0;

    // a.hi * b[0] + lo1
    uint64_t lo2 = a.hi * b[0];
    uint64_t hi2 = __umul64hi(a.hi, b[0]);
    lo2 += lo1;
    hi2 += (lo2 < lo1) ? 1 : 0;
    p[1] = lo2;

    // Combine carries: hi1 + hi2
    carry = hi1 + hi2;

    // a.hi * b[1] + carry
    uint64_t lo3 = a.hi * b[1];
    uint64_t hi3 = __umul64hi(a.hi, b[1]);
    lo3 += carry;
    hi3 += (lo3 < carry) ? 1 : 0;
    p[2] = lo3;
    p[3] = hi3;

    result.d[0] = p[0];
    result.d[1] = p[1];
    result.d[2] = p[2];
    result.d[3] = p[3];
}

// Subtract 256-bit values, returning borrow (for signed arithmetic)
__device__ bool sub_256(U256& r, const U256& a, const U256& b) {
    uint64_t borrow;
    r.d[0] = sub_cc(a.d[0], b.d[0], borrow);
    r.d[1] = subc_cc(a.d[1], b.d[1], borrow, borrow);
    r.d[2] = subc_cc(a.d[2], b.d[2], borrow, borrow);
    r.d[3] = subc_cc(a.d[3], b.d[3], borrow, borrow);
    return borrow != 0;
}

// Add 256-bit values
__device__ void add_256(U256& r, const U256& a, const U256& b) {
    uint64_t carry;
    r.d[0] = add_cc(a.d[0], b.d[0], carry);
    r.d[1] = addc_cc(a.d[1], b.d[1], carry, carry);
    r.d[2] = addc_cc(a.d[2], b.d[2], carry, carry);
    r.d[3] = addc_cc(a.d[3], b.d[3], carry, carry);
}

// GLV scalar decomposition: k = k1 + k2*lambda (mod n)
// Uses Babai's nearest plane algorithm with precomputed lattice basis
// k1, k2 will be ~128 bits each (half the size of k)
// Returns sign flags: k1_neg, k2_neg (true if that component should be negated)
__device__ void glv_decompose(const U256& k, U256& k1, U256& k2, bool& k1_neg, bool& k2_neg) {
    // Compute c1 = round(k * g1 / 2^384) and c2 = round(k * g2 / 2^384)
    // where g1, g2 are precomputed as floor(b2 * 2^384 / n) and floor(-b1 * 2^384 / n).
    //
    // 2026-05-04 fix (test_kangaroo_small_puzzle k=2,3,7 failing):
    // The previous code extracted p[4],p[5] which corresponds to (k*g_i) >> 256.
    // That was inconsistent with the actual magnitudes of GLV_G1/GLV_G2 (which
    // are computed mod 2^384/n, not 2^256/n). For small k like k=2, the spurious
    // carry-bit of the (k*g2) product would set c2 = 1 instead of 0, producing
    // a totally wrong decomposition (k1 = k - a1 instead of k1 = k). Fixed by
    // extracting p[6],p[7] = (k*g_i) >> 384, matching the libsecp256k1 algorithm
    // (see secp256k1_scalar_split_lambda in src/scalar_impl.h).

    U128 c1, c2;

    // c1 = (k * g1) >> 384  (top 128 bits of full 512-bit product)
    {
        uint64_t p[8] = {0};
        for (int i = 0; i < 4; i++) {
            uint64_t carry = 0;
            for (int j = 0; j < 4; j++) {
                // 64x64 -> 128 multiply + add using CUDA intrinsics
                uint64_t lo = k.d[i] * GLV_G1[j];
                uint64_t hi = __umul64hi(k.d[i], GLV_G1[j]);
                // Add p[i+j]
                lo += p[i+j];
                hi += (lo < p[i+j]) ? 1 : 0;
                // Add carry
                lo += carry;
                hi += (lo < carry) ? 1 : 0;
                p[i+j] = lo;
                carry = hi;
            }
            p[i+4] = carry;
        }
        c1.lo = p[6];
        c1.hi = p[7];
    }

    // c2 = (k * g2) >> 384  (top 128 bits of full 512-bit product)
    {
        uint64_t p[8] = {0};
        for (int i = 0; i < 4; i++) {
            uint64_t carry = 0;
            for (int j = 0; j < 4; j++) {
                // 64x64 -> 128 multiply + add using CUDA intrinsics
                uint64_t lo = k.d[i] * GLV_G2[j];
                uint64_t hi = __umul64hi(k.d[i], GLV_G2[j]);
                // Add p[i+j]
                lo += p[i+j];
                hi += (lo < p[i+j]) ? 1 : 0;
                // Add carry
                lo += carry;
                hi += (lo < carry) ? 1 : 0;
                p[i+j] = lo;
                carry = hi;
            }
            p[i+4] = carry;
        }
        c2.lo = p[6];
        c2.hi = p[7];
    }

    // k1 = k - c1*a1 - c2*a2
    // k2 = -c1*b1 + c2*b2  (note: our stored b1 is negative of the actual b1)
    //    = c1*(-b1) + c2*b2
    //    = c1*GLV_B1 + c2*GLV_B2  (both stored as positive)

    U256 c1_a1, c2_a2;
    mul_128x128(c1, GLV_A1, c1_a1);

    // 2026-05-04 fix (Defect A): use the real a2, not a1. a2 is 129 bits
    // and decomposes as 2^128 + GLV_A2_LOW, so
    //   c2 * a2 = c2 * GLV_A2_LOW + (c2 << 128)
    // We compute the low product with mul_128x128, then add c2 into the
    // upper 128 bits with proper carry propagation. The result is
    // truncated mod 2^256 since the subsequent k - c1*a1 - c2*a2 chain
    // wraps mod 2^256 anyway and the final |k1| fits in 128 bits.
    mul_128x128(c2, GLV_A2_LOW, c2_a2);
    {
        // Add c2 (interpreted at offset 128, i.e. into limbs [2] and [3])
        uint64_t carry;
        c2_a2.d[2] = add_cc(c2_a2.d[2], c2.lo, carry);
        c2_a2.d[3] = c2_a2.d[3] + c2.hi + carry;
    }

    // k1 = k - c1*a1 - c2*a2
    k1 = k;
    bool borrow1 = sub_256(k1, k1, c1_a1);
    bool borrow2 = sub_256(k1, k1, c2_a2);

    // Handle underflow - if k1 went negative, we need to adjust
    k1_neg = (k1.d[3] >> 63) != 0;
    if (k1_neg) {
        // Negate k1: k1 = -k1 = 0 - k1
        U256 zero; zero.set_zero();
        sub_256(k1, zero, k1);
    }

    // k2 = c1*b1 + c2*b2 (where b1 is stored as -b1, b2 = a1)
    U256 c1_b1, c2_b2;
    mul_128x128(c1, GLV_B1, c1_b1);  // c1 * (-b1)
    mul_128x128(c2, GLV_B2, c2_b2);  // c2 * b2

    // k2 = -c1*b1 + c2*b2 = c2_b2 - c1_b1  (since GLV_B1 stores -b1)
    // Actually for secp256k1: k2 = -c1*b1 - c2*b2 where b1 < 0 originally
    // So k2 = c1*|b1| - c2*b2
    k2_neg = false;
    bool k2_borrow = sub_256(k2, c1_b1, c2_b2);
    if (k2_borrow || (k2.d[3] >> 63)) {
        // k2 is negative, swap and negate
        sub_256(k2, c2_b2, c1_b1);
        k2_neg = true;
    }

    // k1 and k2 are bounded by the GLV Babai lattice: |k1|, |k2| < 2^128.5.
    // For scalars near n (e.g. k = n-1), the rounding gives k1 = a1+a2-j
    // which is a 129-bit number (d[2] = 1). We must NOT truncate d[2] here;
    // ec_mul_glv uses max_window = 33 to cover that extra bit.
    // d[3] is always 0 (Babai bound keeps |k1|,|k2| < 2^192).
    k1.d[3] = 0;
    k2.d[3] = 0;
}

// Apply beta endomorphism to point: (x, y) -> (beta*x, y)
// This computes λ*P efficiently without scalar multiplication
__device__ void apply_endomorphism(PointA& Q, const PointA& P) {
    U256 beta;
    beta.d[0] = GLV_BETA[0];
    beta.d[1] = GLV_BETA[1];
    beta.d[2] = GLV_BETA[2];
    beta.d[3] = GLV_BETA[3];

    mod_mul(Q.x, P.x, beta);
    Q.y = P.y;  // y unchanged
}

// Negate a point: (x, y) -> (x, -y) = (x, p - y)
__device__ void ec_negate(PointA& P) {
    U256 p;
    p.d[0] = SECP_P[0]; p.d[1] = SECP_P[1];
    p.d[2] = SECP_P[2]; p.d[3] = SECP_P[3];
    mod_sub(P.y, p, P.y);
}

__device__ void ec_negate_j(PointJ& P) {
    U256 p;
    p.d[0] = SECP_P[0]; p.d[1] = SECP_P[1];
    p.d[2] = SECP_P[2]; p.d[3] = SECP_P[3];
    mod_sub(P.Y, p, P.Y);
}

// GLV-accelerated scalar multiplication using Shamir's trick.
// Computes k*G = k1*G + k2*(λG) where k1, k2 are ~128-bit non-negative
// magnitudes (with separate sign flags from glv_decompose).
//
// 2026-05-04 fix (Defect B): the prior implementation combined two
// incompatible windowed-mul conventions. The precomputed table stores
//   table[w*16 + i] = i * 16^w * G
// (positional scaling baked in -- the right-to-left convention), but the
// loop also did 4 doublings per window step (the left-to-right
// convention). The result was a per-window double-counting of the
// positional weight: contributions from window w ended up multiplied by
// 16^(2w) instead of 16^w, so the final point was completely wrong for
// any non-trivial scalar.
//
// Fix: switch to the textbook left-to-right method using a flat table
//   table_flat[i] = i * G   (i = 0..15)
// which is exactly what the existing table generator stores at
// d_PRECOMP_TABLE[0 * 16 + i]. We use only those 16 entries (and the
// matching 16 entries of d_PRECOMP_TABLE_LAMBDA for i*λG). The 4
// doublings per window then provide the positional weight, exactly as
// the standard algorithm requires. Matches the algorithm in
// libsecp256k1's secp256k1_ecmult_strauss_wnaf simplified to fixed
// 4-bit windows and Shamir's joint accumulation.
__device__ void ec_mul_glv(PointJ& R, const U256& k) {
    // Decompose scalar
    U256 k1, k2;
    bool k1_neg, k2_neg;
    glv_decompose(k, k1, k2, k1_neg, k2_neg);

    // If both k1 and k2 are zero, k*G = infinity (k mod n == 0).
    if (k1.is_zero() && k2.is_zero()) {
        R.set_infinity();
        return;
    }

    R.set_infinity();

    // 33 windows of 4 bits each = 132 bits. The GLV Babai bound gives
    // |k1|, |k2| < 2^128.5; for scalars near n, d[2] = 1 (bit 128 set).
    // The 33rd window (w=32, bit_offset=128, word=2, shift=0) captures that
    // extra bit. d[3] is always 0 per the bound, so 33 windows suffice.
    constexpr int max_window = 33;

    // Standard left-to-right windowed multiplication with Shamir's trick.
    // R starts at infinity; each step shifts the accumulated value left
    // by one window (4 doublings) and adds the contribution of the
    // current nibbles from both scalars.
    for (int w = max_window - 1; w >= 0; w--) {
        // Shift R left by 4 bits (R = 16 * R) via 4 doublings.
        // ec_double is alias-safe (Defect C fix above).
        ec_double(R, R);
        ec_double(R, R);
        ec_double(R, R);
        ec_double(R, R);

        // Extract the 4-bit window from each scalar.
        // With max_window = 32 and WINDOW_SIZE = 4, every window is fully
        // contained inside a single 64-bit limb (word = 0 for w in 0..15,
        // word = 1 for w in 16..31, never crossing a limb boundary), so
        // no cross-limb stitching is needed.
        const int bit_offset = w * WINDOW_SIZE;
        const int word = bit_offset >> 6;       // / 64
        const int shift = bit_offset & 63;      // % 64
        uint64_t n1 = (k1.d[word] >> shift) & (uint64_t)WINDOW_MASK;
        uint64_t n2 = (k2.d[word] >> shift) & (uint64_t)WINDOW_MASK;

        // Add k1 contribution: n1 * G (from the flat first window of the
        // table, indices 0..15). Negate in place if the decomposition
        // produced a negative k1.
        if (n1 != 0) {
            PointA P1 = d_PRECOMP_TABLE[n1];
            if (k1_neg) ec_negate(P1);
            // ec_add_mixed is alias-safe (Defect C fix above).
            ec_add_mixed(R, R, P1);
        }

        // Add k2 contribution: n2 * (lambda * G) from the flat first
        // window of the lambda table.
        if (n2 != 0) {
            PointA P2 = d_PRECOMP_TABLE_LAMBDA[n2];
            if (k2_neg) ec_negate(P2);
            ec_add_mixed(R, R, P2);
        }
    }
}

// =============================================================================
// STRIDED INCREMENTAL SEARCH
// =============================================================================

// Process multiple sequential keys per thread using point addition
// After computing base*G, each subsequent key just adds G
// Uses GLV endomorphism for ~30% faster initial scalar multiplication
__device__ void search_strided(
    uint64_t base_lo, uint64_t base_hi,
    int keys_per_thread,
    const uint8_t* target_h160,
    uint64_t* found_key_lo,
    uint64_t* found_key_hi,
    uint32_t* found_flag
) {
    // Compute base point: P = base * G
    U256 scalar;
    scalar.d[0] = base_lo;
    scalar.d[1] = base_hi;
    scalar.d[2] = 0;
    scalar.d[3] = 0;

    PointJ P;
    // Use GLV for faster scalar multiplication (~30% speedup)
    // GLV decomposes 256-bit scalar into two 128-bit scalars
    ec_mul_glv(P, scalar);

    // Load generator G for incremental addition
    PointA G;
    G.x.d[0] = SECP_GX[0]; G.x.d[1] = SECP_GX[1];
    G.x.d[2] = SECP_GX[2]; G.x.d[3] = SECP_GX[3];
    G.y.d[0] = SECP_GY[0]; G.y.d[1] = SECP_GY[1];
    G.y.d[2] = SECP_GY[2]; G.y.d[3] = SECP_GY[3];

    // Arrays for batch inversion
    U256 z_values[KEYS_PER_THREAD];
    PointJ points[KEYS_PER_THREAD];

    // Process keys incrementally
    for (int i = 0; i < keys_per_thread; i++) {
        points[i] = P;
        z_values[i] = P.Z;

        // Increment: P = P + G
        ec_add_mixed(P, P, G);
    }

    // Batch inversion of all Z coordinates
    U256 z_inv[KEYS_PER_THREAD];
    batch_invert(z_values, z_inv, keys_per_thread);

    // Convert to affine and compute hash160 for each key
    for (int i = 0; i < keys_per_thread; i++) {
        if (*found_flag) return;  // Early exit if another thread found it

        // Convert to affine: x = X/Z^2, y = Y/Z^3
        U256 z_inv2, z_inv3, x_affine, y_affine;
        mod_sqr(z_inv2, z_inv[i]);
        mod_mul(z_inv3, z_inv2, z_inv[i]);
        mod_mul(x_affine, points[i].X, z_inv2);
        mod_mul(y_affine, points[i].Y, z_inv3);

        // Compress public key
        uint8_t compressed[33];
        compressed[0] = (y_affine.d[0] & 1) ? 0x03 : 0x02;

        // X coordinate in big-endian
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t limb = x_affine.d[3-j];
            compressed[1 + j*8]     = (limb >> 56) & 0xff;
            compressed[1 + j*8 + 1] = (limb >> 48) & 0xff;
            compressed[1 + j*8 + 2] = (limb >> 40) & 0xff;
            compressed[1 + j*8 + 3] = (limb >> 32) & 0xff;
            compressed[1 + j*8 + 4] = (limb >> 24) & 0xff;
            compressed[1 + j*8 + 5] = (limb >> 16) & 0xff;
            compressed[1 + j*8 + 6] = (limb >> 8) & 0xff;
            compressed[1 + j*8 + 7] = limb & 0xff;
        }

        // Compute Hash160 = RIPEMD160(SHA256(compressed_pubkey))
        uint8_t sha_out[32], h160[20];
        sha256_33bytes_opt(compressed, sha_out);
        ripemd160_32bytes_opt(sha_out, h160);

        // Compare with target
        bool match = true;
        #pragma unroll
        for (int j = 0; j < 20; j++) {
            if (h160[j] != target_h160[j]) {
                match = false;
                break;
            }
        }

        if (match) {
            uint64_t key_lo = base_lo + i;
            uint64_t key_hi = base_hi + (key_lo < base_lo ? 1 : 0);

            if (atomicCAS(found_flag, 0, 1) == 0) {
                *found_key_lo = key_lo;
                *found_key_hi = key_hi;
            }
            return;
        }
    }
}

// =============================================================================
// MAIN KERNEL
// =============================================================================

// Launch bounds hint for compiler optimization:
// - 256 max threads per block (or 128 for Blackwell)
// - 4 min blocks per SM for good occupancy
// This helps the compiler allocate registers more efficiently
__global__ void __launch_bounds__(256, 4) puzzle_search_optimized(
    uint64_t range_start_lo,
    uint64_t range_start_hi,
    uint64_t total_keys,
    const uint8_t* __restrict__ target_hash160,
    uint64_t* __restrict__ match_key_lo,
    uint64_t* __restrict__ match_key_hi,
    uint32_t* __restrict__ match_found
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;

    // Each thread processes KEYS_PER_THREAD sequential keys
    for (uint64_t base = tid * KEYS_PER_THREAD; base < total_keys; base += stride * KEYS_PER_THREAD) {
        if (*match_found) return;

        uint64_t key_lo = range_start_lo + base;
        uint64_t key_hi = range_start_hi + (key_lo < range_start_lo ? 1 : 0);

        int keys_this_batch = min((uint64_t)KEYS_PER_THREAD, total_keys - base);

        search_strided(key_lo, key_hi, keys_this_batch,
                      target_hash160, match_key_lo, match_key_hi, match_found);
    }
}

// =============================================================================
// INITIALIZATION (Precompute Tables)
// =============================================================================

// Device memory for precomputed tables (constant memory is limited to 64KB)
static PointA* g_precomputed_table_device = nullptr;
static PointA* g_precomputed_table_lambda_device = nullptr;
static bool g_table_initialized = false;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4189)  // local variable initialized but not referenced
#endif
#ifdef __CUDACC__
#pragma nv_diag_suppress 550  // variable set but never used
#endif
// GLV state tracking (set for future use, currently informational)
static bool g_glv_enabled = true;
#ifdef _MSC_VER
#pragma warning(pop)
#endif

/**
 * Kernel to generate precomputed table on GPU.
 * Table[w * 16 + i] = i * 2^(w*4) * G for each window w and value i (0-15).
 */
__global__ void generate_precomputed_table_kernel_opt(PointA* table, PointA* table_lambda) {
    // Load generator G
    PointA G;
    G.x.d[0] = SECP_GX[0]; G.x.d[1] = SECP_GX[1];
    G.x.d[2] = SECP_GX[2]; G.x.d[3] = SECP_GX[3];
    G.y.d[0] = SECP_GY[0]; G.y.d[1] = SECP_GY[1];
    G.y.d[2] = SECP_GY[2]; G.y.d[3] = SECP_GY[3];

    // Compute λG = endomorphism(G) = (β*Gx, Gy)
    PointA LG;
    apply_endomorphism(LG, G);

    // Build table for window 0: 0*G, 1*G, 2*G, ..., 15*G
    PointJ points[1 << WINDOW_SIZE];
    PointJ points_lambda[1 << WINDOW_SIZE];

    // 0 * G = infinity
    points[0].set_infinity();
    points_lambda[0].set_infinity();

    // 1 * G and 1 * λG
    points[1].X = G.x;
    points[1].Y = G.y;
    points[1].Z.set_one();

    points_lambda[1].X = LG.x;
    points_lambda[1].Y = LG.y;
    points_lambda[1].Z.set_one();

    // 2*G through 15*G via repeated addition
    for (int i = 2; i < (1 << WINDOW_SIZE); i++) {
        ec_add_mixed(points[i], points[i-1], G);
        ec_add_mixed(points_lambda[i], points_lambda[i-1], LG);
    }

    // Convert window 0 to affine and store
    for (int i = 0; i < (1 << WINDOW_SIZE); i++) {
        if (points[i].is_infinity()) {
            table[i].x.set_zero();
            table[i].y.set_zero();
        } else {
            U256 z_inv, z_inv2, z_inv3;
            mod_inv(z_inv, points[i].Z);
            mod_sqr(z_inv2, z_inv);
            mod_mul(z_inv3, z_inv2, z_inv);
            mod_mul(table[i].x, points[i].X, z_inv2);
            mod_mul(table[i].y, points[i].Y, z_inv3);
        }

        // Lambda table
        if (points_lambda[i].is_infinity()) {
            table_lambda[i].x.set_zero();
            table_lambda[i].y.set_zero();
        } else {
            U256 z_inv, z_inv2, z_inv3;
            mod_inv(z_inv, points_lambda[i].Z);
            mod_sqr(z_inv2, z_inv);
            mod_mul(z_inv3, z_inv2, z_inv);
            mod_mul(table_lambda[i].x, points_lambda[i].X, z_inv2);
            mod_mul(table_lambda[i].y, points_lambda[i].Y, z_inv3);
        }
    }

    // For GLV, we only need 32 windows (128-bit scalars) instead of 64
    int num_glv_windows = 32;

    // For each subsequent window, double all points WINDOW_SIZE times
    for (int w = 1; w < NUM_WINDOWS; w++) {
        // Double each point WINDOW_SIZE times
        for (int d = 0; d < WINDOW_SIZE; d++) {
            for (int i = 1; i < (1 << WINDOW_SIZE); i++) {
                PointJ temp;
                ec_double(temp, points[i]);
                points[i] = temp;

                // Only compute lambda table for first 32 windows (GLV uses 128-bit scalars)
                if (w < num_glv_windows) {
                    ec_double(temp, points_lambda[i]);
                    points_lambda[i] = temp;
                }
            }
        }

        // Convert to affine and store for this window
        for (int i = 0; i < (1 << WINDOW_SIZE); i++) {
            int table_idx = w * (1 << WINDOW_SIZE) + i;
            if (points[i].is_infinity() || i == 0) {
                table[table_idx].x.set_zero();
                table[table_idx].y.set_zero();
            } else {
                U256 z_inv, z_inv2, z_inv3;
                mod_inv(z_inv, points[i].Z);
                mod_sqr(z_inv2, z_inv);
                mod_mul(z_inv3, z_inv2, z_inv);
                mod_mul(table[table_idx].x, points[i].X, z_inv2);
                mod_mul(table[table_idx].y, points[i].Y, z_inv3);
            }

            // Lambda table (only for first 32 windows)
            if (w < num_glv_windows) {
                if (points_lambda[i].is_infinity() || i == 0) {
                    table_lambda[table_idx].x.set_zero();
                    table_lambda[table_idx].y.set_zero();
                } else {
                    U256 z_inv, z_inv2, z_inv3;
                    mod_inv(z_inv, points_lambda[i].Z);
                    mod_sqr(z_inv2, z_inv);
                    mod_mul(z_inv3, z_inv2, z_inv);
                    mod_mul(table_lambda[table_idx].x, points_lambda[i].X, z_inv2);
                    mod_mul(table_lambda[table_idx].y, points_lambda[i].Y, z_inv3);
                }
            }
        }
    }
}

// Host function to initialize precomputed tables
extern "C" cudaError_t init_puzzle_optimized(cudaStream_t stream) {
    if (g_table_initialized) {
        return cudaSuccess;  // Already initialized
    }

    // Allocate device memory for both tables (G and λG)
    size_t table_size = NUM_WINDOWS * (1 << WINDOW_SIZE) * sizeof(PointA);
    cudaError_t err = cudaMalloc(&g_precomputed_table_device, table_size);
    if (err != cudaSuccess) {
        return err;
    }

    err = cudaMalloc(&g_precomputed_table_lambda_device, table_size);
    if (err != cudaSuccess) {
        cudaFree(g_precomputed_table_device);
        g_precomputed_table_device = nullptr;
        return err;
    }

    // Generate both tables on GPU (single thread - table generation is one-time cost)
    generate_precomputed_table_kernel_opt<<<1, 1, 0, stream>>>(
        g_precomputed_table_device, g_precomputed_table_lambda_device);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(g_precomputed_table_device);
        cudaFree(g_precomputed_table_lambda_device);
        g_precomputed_table_device = nullptr;
        g_precomputed_table_lambda_device = nullptr;
        return err;
    }

    // Wait for generation to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(g_precomputed_table_device);
        cudaFree(g_precomputed_table_lambda_device);
        g_precomputed_table_device = nullptr;
        g_precomputed_table_lambda_device = nullptr;
        return err;
    }

    // Set device pointers to the allocated global memory tables
    // (No longer using constant memory due to 64KB size limit - tables are 128KB combined)
    err = cudaMemcpyToSymbol(d_PRECOMP_TABLE, &g_precomputed_table_device, sizeof(PointA*));
    if (err != cudaSuccess) {
        fprintf(stderr, "[EC] Error: Could not set d_PRECOMP_TABLE pointer\n");
        cudaFree(g_precomputed_table_device);
        cudaFree(g_precomputed_table_lambda_device);
        g_precomputed_table_device = nullptr;
        g_precomputed_table_lambda_device = nullptr;
        return err;
    }

    err = cudaMemcpyToSymbol(d_PRECOMP_TABLE_LAMBDA, &g_precomputed_table_lambda_device, sizeof(PointA*));
    if (err != cudaSuccess) {
        fprintf(stderr, "[EC] Error: Could not set d_PRECOMP_TABLE_LAMBDA pointer\n");
        cudaFree(g_precomputed_table_device);
        cudaFree(g_precomputed_table_lambda_device);
        g_precomputed_table_device = nullptr;
        g_precomputed_table_lambda_device = nullptr;
        return err;
    }

    // Apply L2 cache persistence hints for better performance (CUDA 11.4+)
    #if CUDART_VERSION >= 11040
    cudaStreamAttrValue stream_attr = {};
    stream_attr.accessPolicyWindow.base_ptr = g_precomputed_table_device;
    stream_attr.accessPolicyWindow.num_bytes = table_size;
    stream_attr.accessPolicyWindow.hitRatio = 1.0f;
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);

    // Also apply to lambda table
    stream_attr.accessPolicyWindow.base_ptr = g_precomputed_table_lambda_device;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    #endif

    g_glv_enabled = true;
    fprintf(stderr, "[EC] Precomputed tables initialized in global memory (128KB, L2 cached)\n");
    fprintf(stderr, "[GLV] Lambda table ready, GLV endomorphism enabled (~30%% speedup)\n");

    g_table_initialized = true;
    return err;
}

// Cleanup function
extern "C" cudaError_t cleanup_puzzle_optimized() {
    if (g_precomputed_table_device != nullptr) {
        cudaFree(g_precomputed_table_device);
        g_precomputed_table_device = nullptr;
    }
    if (g_precomputed_table_lambda_device != nullptr) {
        cudaFree(g_precomputed_table_lambda_device);
        g_precomputed_table_lambda_device = nullptr;
    }
    g_table_initialized = false;
    g_glv_enabled = false;
    return cudaSuccess;
}

// =============================================================================
// GPU DEVICE INFO CACHING
// =============================================================================

struct GPUDeviceInfo {
    int sm_count;
    int max_threads_per_sm;
    int max_blocks_per_sm;
    int warp_size;
    int compute_major;
    int compute_minor;
    size_t shared_mem_per_block;
    size_t l2_cache_size;
    bool initialized;
};

static GPUDeviceInfo g_gpu_info = {0, 0, 0, 0, 0, 0, 0, 0, false};

// Query and cache GPU device properties
static void ensure_gpu_info(int device = 0) {
    if (g_gpu_info.initialized) return;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    g_gpu_info.sm_count = props.multiProcessorCount;
    g_gpu_info.max_threads_per_sm = props.maxThreadsPerMultiProcessor;
    g_gpu_info.max_blocks_per_sm = props.maxBlocksPerMultiProcessor;
    g_gpu_info.warp_size = props.warpSize;
    g_gpu_info.compute_major = props.major;
    g_gpu_info.compute_minor = props.minor;
    g_gpu_info.shared_mem_per_block = props.sharedMemPerBlock;
    g_gpu_info.l2_cache_size = props.l2CacheSize;
    g_gpu_info.initialized = true;

    fprintf(stderr, "[GPU] %s: %d SMs, CC %d.%d, L2 %zu MB\n",
            props.name, g_gpu_info.sm_count,
            g_gpu_info.compute_major, g_gpu_info.compute_minor,
            g_gpu_info.l2_cache_size / (1024 * 1024));
}

// Calculate optimal launch configuration based on GPU capabilities
static void get_optimal_launch_config(
    uint64_t total_work,
    int* out_blocks,
    int* out_threads,
    int work_per_thread = KEYS_PER_THREAD
) {
    ensure_gpu_info();

    // Blackwell/Ada: prefer 128 threads for better register usage
    // Ampere/Turing: 256 threads is fine
    int threads_per_block = (g_gpu_info.compute_major >= 9) ? 128 : 256;

    // Target: 4-8 blocks per SM for good occupancy
    int target_blocks_per_sm = 6;
    int target_blocks = g_gpu_info.sm_count * target_blocks_per_sm;

    // Calculate blocks needed for the work
    int64_t work_items = (total_work + work_per_thread - 1) / work_per_thread;
    int blocks_needed = (work_items + threads_per_block - 1) / threads_per_block;

    // Use the larger of target or needed, capped at CUDA limit
    int blocks = max(target_blocks, blocks_needed);
    blocks = min(blocks, 65535);

    *out_blocks = blocks;
    *out_threads = threads_per_block;
}

// =============================================================================
// WRAPPER FOR EXTERNAL USE
// =============================================================================

extern "C" cudaError_t puzzle_search_batch_optimized(
    uint64_t range_start_lo,
    uint64_t range_start_hi,
    uint64_t batch_size,
    const uint8_t* d_target_hash160,
    uint64_t* d_match_key_lo,
    uint64_t* d_match_key_hi,
    uint32_t* d_match_found,
    cudaStream_t stream
) {
    // Clear match flag
    cudaMemsetAsync(d_match_found, 0, sizeof(uint32_t), stream);

    // Dynamic launch configuration based on GPU capabilities
    int blocks, threads_per_block;
    get_optimal_launch_config(batch_size, &blocks, &threads_per_block);

    puzzle_search_optimized<<<blocks, threads_per_block, 0, stream>>>(
        range_start_lo, range_start_hi, batch_size,
        d_target_hash160, d_match_key_lo, d_match_key_hi, d_match_found
    );

    return cudaGetLastError();
}

// Get GPU info for external use (e.g., progress display)
extern "C" void get_gpu_info(int* sm_count, int* compute_major, int* compute_minor) {
    ensure_gpu_info();
    if (sm_count) *sm_count = g_gpu_info.sm_count;
    if (compute_major) *compute_major = g_gpu_info.compute_major;
    if (compute_minor) *compute_minor = g_gpu_info.compute_minor;
}

// =============================================================================
// TEST INFRASTRUCTURE
// Wave 1 Track C C-CRIT-2 follow-up (2026-05-04). Permanent test entry point
// used by tests/test_puzzle_optimized_inv.cu. Exercises the __device__
// mod_inv / mod_mul functions in this translation unit; they have no other
// host-callable surface and must be tested in-tree.
// =============================================================================

// Verify mod_inv(a) is the modular inverse of a, by computing a * mod_inv(a)
// and checking the result is exactly 1 (mod p). Per-thread 1-byte result
// (1 = correct, 0 = wrong / skipped zero input). The mod_mul above already
// performs the final canonical reduction in its while-loop, so the product
// is comparable to the literal 1 limb representation.
__global__ void test_puzzle_opt_mod_inv_correctness_kernel(
    const U256* __restrict__ scalars,
    uint8_t* __restrict__ results,
    size_t count
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    U256 a = scalars[idx];

    // Skip a == 0 (no inverse exists). Caller should not pass zero.
    if (a.is_zero()) { results[idx] = 0; return; }

    U256 inv;
    mod_inv(inv, a);

    U256 product;
    mod_mul(product, a, inv);

    bool is_one = (product.d[0] == 1ULL) &&
                  (product.d[1] == 0ULL) &&
                  (product.d[2] == 0ULL) &&
                  (product.d[3] == 0ULL);

    results[idx] = is_one ? 1 : 0;
}

extern "C" cudaError_t puzzle_optimized_test_inverse_correctness_kernel_launch(
    const void* d_scalars,        // count * 32 bytes (4 x uint64 little-endian)
    uint8_t* d_results,           // count bytes (1=ok, 0=wrong)
    size_t count,
    cudaStream_t stream
) {
    if (count == 0) return cudaSuccess;
    const int threads = 64;
    int blocks = (int)((count + threads - 1) / threads);
    test_puzzle_opt_mod_inv_correctness_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const U256*>(d_scalars),
        d_results,
        count
    );
    return cudaGetLastError();
}

// =============================================================================
// EC mul GLV known-answer test kernel
//
// KangarooSmallPuzzle replacement (2026-05-04). The original
// test_kangaroo_small_puzzle.cu drove puzzle_search_batch_optimized end to end
// on a tiny 256-key range to verify EC + SHA + RIPEMD recover known privkeys.
// That test is fragile: the search kernel allocates a 256-entry PointJ scratch
// per thread, batch-inverts it, hashes every output, and only signals
// success/failure through a single match flag. No intermediate state is
// observable, so any failure becomes "no match found" with no actionable
// signal.
//
// This kernel exposes the EC math used by the kangaroo / puzzle search path
// (ec_mul_glv via the precomputed table built by init_puzzle_optimized).
// Each thread runs ec_mul_glv on an input scalar, jacobian->affine via
// mod_inv, and writes (x, y) to the output. The host test compares against
// known compressed-pubkey vectors, exactly mirroring EcMulKnownAnswers but
// for the puzzle_optimized.cu code path instead of secp256k1.cu.
//
// One thread per scalar; small input set (handful of vectors).
// =============================================================================
__global__ void test_puzzle_opt_ec_mul_glv_kernel(
    const U256* __restrict__ scalars,
    U256* __restrict__ out_x,
    U256* __restrict__ out_y,
    uint8_t* __restrict__ out_is_infinity,
    size_t count
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    U256 k = scalars[idx];

    PointJ P;
    ec_mul_glv(P, k);

    if (P.is_infinity()) {
        out_is_infinity[idx] = 1;
        out_x[idx].set_zero();
        out_y[idx].set_zero();
        return;
    }

    // Jacobian -> affine: x = X / Z^2, y = Y / Z^3
    U256 z_inv, z_inv2, z_inv3;
    mod_inv(z_inv, P.Z);
    mod_sqr(z_inv2, z_inv);
    mod_mul(z_inv3, z_inv2, z_inv);
    mod_mul(out_x[idx], P.X, z_inv2);
    mod_mul(out_y[idx], P.Y, z_inv3);
    out_is_infinity[idx] = 0;
}

extern "C" cudaError_t puzzle_optimized_test_ec_mul_glv_kernel_launch(
    const void* d_scalars,        // count * 32 bytes (4 x uint64 little-endian)
    void* d_out_x,                // count * 32 bytes (4 x uint64 little-endian)
    void* d_out_y,                // count * 32 bytes (4 x uint64 little-endian)
    uint8_t* d_out_is_infinity,   // count bytes (1 = infinity, 0 = finite)
    size_t count,
    cudaStream_t stream
) {
    if (count == 0) return cudaSuccess;
    const int threads = 32;
    int blocks = (int)((count + threads - 1) / threads);
    test_puzzle_opt_ec_mul_glv_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const U256*>(d_scalars),
        reinterpret_cast<U256*>(d_out_x),
        reinterpret_cast<U256*>(d_out_y),
        d_out_is_infinity,
        count
    );
    return cudaGetLastError();
}

}  // namespace optimized
}  // namespace gpu
}  // namespace collider

// =============================================================================
// HOST-FACING TEST API
// Single host-callable function that owns the whole life cycle: deterministic
// scalar generation, device alloc/copy, kernel launch, result reduction. The
// CTest harness calls this and just checks the wrong_count_out value. Mirrors
// the pattern of secp256k1_test_inverse_correctness but pushes the launch
// machinery host-side so the test file stays minimal.
// =============================================================================
extern "C" cudaError_t puzzle_optimized_test_inverse_correctness_kernel_launch(
    const void* d_scalars,
    uint8_t* d_results,
    size_t count,
    cudaStream_t stream
);

extern "C" int puzzle_optimized_test_inverse_correctness(int* wrong_count_out) {
    if (wrong_count_out == nullptr) return -1;
    *wrong_count_out = -1;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return -2;  // No CUDA device; caller should treat as skip
    }
    err = cudaSetDevice(0);
    if (err != cudaSuccess) return -3;

    constexpr size_t N = 64;  // 32 minimum required, 64 for stronger coverage
    constexpr size_t LIMBS_PER_SCALAR = 4;  // U256 is 4 x uint64
    constexpr size_t BYTES_PER_SCALAR = LIMBS_PER_SCALAR * sizeof(uint64_t);

    // Deterministic scalar table built on the host using a simple xorshift64.
    // Same seed pattern as test_secp256k1_inv (different RNG, but reproducible).
    uint64_t scalars[N * LIMBS_PER_SCALAR];
    uint64_t state = 0xC011DEC011DE0001ULL;
    auto next = [&]() -> uint64_t {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < LIMBS_PER_SCALAR; j++) {
            scalars[i * LIMBS_PER_SCALAR + j] = next();
        }
        // Force MSB clear so value is < 2^255 (well below p ~ 2^256 - small)
        scalars[i * LIMBS_PER_SCALAR + 3] &= 0x7FFFFFFFFFFFFFFFULL;
        // Force at least one bit set to avoid the degenerate zero input
        scalars[i * LIMBS_PER_SCALAR + 0] |= 1ULL;
    }

    // Edge cases at the start of the table (overwrite the random ones).
    // a = 1   -> inv(1) = 1
    for (size_t j = 0; j < LIMBS_PER_SCALAR; j++) scalars[0 * LIMBS_PER_SCALAR + j] = 0;
    scalars[0 * LIMBS_PER_SCALAR + 0] = 1ULL;
    // a = 2   -> inv(2) = (p+1)/2
    for (size_t j = 0; j < LIMBS_PER_SCALAR; j++) scalars[1 * LIMBS_PER_SCALAR + j] = 0;
    scalars[1 * LIMBS_PER_SCALAR + 0] = 2ULL;
    // a = 12345 (small composite, exercises reduction path)
    for (size_t j = 0; j < LIMBS_PER_SCALAR; j++) scalars[2 * LIMBS_PER_SCALAR + j] = 0;
    scalars[2 * LIMBS_PER_SCALAR + 0] = 12345ULL;
    // a = high-bit-heavy value (< 2^255 still)
    for (size_t j = 0; j < LIMBS_PER_SCALAR; j++) scalars[3 * LIMBS_PER_SCALAR + j] = 0;
    scalars[3 * LIMBS_PER_SCALAR + 0] = 0xFEDCBA9876543210ULL;
    scalars[3 * LIMBS_PER_SCALAR + 3] = 0x0123456789ABCDEFULL & 0x7FFFFFFFFFFFFFFFULL;

    // Allocate device buffers
    uint64_t* d_scalars = nullptr;
    uint8_t*  d_results = nullptr;
    err = cudaMalloc(&d_scalars, N * BYTES_PER_SCALAR);
    if (err != cudaSuccess) return -4;
    err = cudaMalloc(&d_results, N);
    if (err != cudaSuccess) {
        cudaFree(d_scalars);
        return -4;
    }

    err = cudaMemcpy(d_scalars, scalars, N * BYTES_PER_SCALAR, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_scalars); cudaFree(d_results); return -5; }
    err = cudaMemset(d_results, 0, N);
    if (err != cudaSuccess) { cudaFree(d_scalars); cudaFree(d_results); return -5; }

    err = puzzle_optimized_test_inverse_correctness_kernel_launch(
        d_scalars, d_results, N, /*stream*/ 0
    );
    if (err != cudaSuccess) { cudaFree(d_scalars); cudaFree(d_results); return -6; }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaFree(d_scalars); cudaFree(d_results); return -7; }

    uint8_t results[N];
    err = cudaMemcpy(results, d_results, N, cudaMemcpyDeviceToHost);
    cudaFree(d_scalars);
    cudaFree(d_results);
    if (err != cudaSuccess) return -8;

    int wrong = 0;
    for (size_t i = 0; i < N; i++) {
        if (results[i] != 1) wrong++;
    }
    *wrong_count_out = wrong;
    return 0;  // launch / mem ops all succeeded; caller checks wrong_count_out
}
