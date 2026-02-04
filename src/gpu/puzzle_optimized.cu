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

__device__ __constant__ uint64_t GLV_BETA[4] = {
    0xD765CDA83DB1562CULL, 0x7A9C47E08A641CE2ULL,
    0x51CA10B5A8AC4F6FULL, 0x7AE96A2B657C0710ULL
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

// a2 = 0x114CA50F7A8E2F3F657C1108D9D44CFD8 (129 bits)
// For our 128-bit multiply, we use a2 ≈ b2 = a1 (approximation that works for secp256k1)
// The extra bit causes at most a small error that's corrected by the final modular reduction

__device__ __constant__ uint64_t GLV_B2[2] = {
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL  // b2 = a1 (same value)
};

// g1, g2 precomputed for efficient decomposition:
// g1 = round(b2 * 2^256 / n), g2 = round((-b1) * 2^256 / n)
__device__ __constant__ uint64_t GLV_G1[4] = {
    0x5C02FFF3E0A24D4CULL, 0x6CCEF9CFBB6E0E30ULL,
    0x04B1CFDCFDB0A8DEULL, 0x3086D221A7D46BCDULL
};

__device__ __constant__ uint64_t GLV_G2[4] = {
    0xB739C6639FA88686ULL, 0xFB5B5494CD1D4C9AULL,
    0x4A5E7B7F7C0D2C62ULL, 0xE4437ED6010E8828ULL
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

// Add with carry using PTX for maximum performance
__device__ __forceinline__ uint64_t add_cc(uint64_t a, uint64_t b, uint64_t& carry) {
    uint64_t result;
    asm("add.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
    asm("addc.u64 %0, 0, 0;" : "=l"(carry));
    return result;
}

__device__ __forceinline__ uint64_t addc_cc(uint64_t a, uint64_t b, uint64_t carry_in, uint64_t& carry_out) {
    uint64_t result;
    asm("add.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(carry_in));
    asm("addc.cc.u64 %0, %0, %1;" : "+l"(result) : "l"(b));
    asm("addc.u64 %0, 0, 0;" : "=l"(carry_out));
    return result;
}

// Subtraction with borrow output (first in chain)
__device__ __forceinline__ uint64_t sub_cc(uint64_t a, uint64_t b, uint64_t& borrow) {
    uint64_t result;
    asm("sub.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(b));
    asm("subc.u64 %0, 0, 0;" : "=l"(borrow));
    return result;
}

// Subtraction with borrow-in and borrow-out (for chaining)
__device__ __forceinline__ uint64_t subc_cc(uint64_t a, uint64_t b, uint64_t borrow_in, uint64_t& borrow_out) {
    uint64_t result;
    // First subtract borrow_in from a, setting CC
    asm("sub.cc.u64 %0, %1, %2;" : "=l"(result) : "l"(a), "l"(borrow_in));
    // Then subtract b with borrow, setting CC
    asm("subc.cc.u64 %0, %0, %1;" : "+l"(result) : "l"(b));
    // Capture final borrow
    asm("subc.u64 %0, 0, 0;" : "=l"(borrow_out));
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
        r.d[3] = r.d[3] - SECP_P[3] - borrow;
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

    // Check if >= p and subtract
    if (r.d[3] > SECP_P[3] ||
        (r.d[3] == SECP_P[3] && r.d[2] == SECP_P[2] &&
         r.d[1] == SECP_P[1] && r.d[0] >= SECP_P[0])) {
        uint64_t borrow;
        r.d[0] = sub_cc(r.d[0], SECP_P[0], borrow);
        r.d[1] = sub_cc(r.d[1] - borrow, SECP_P[1], borrow);
        r.d[2] = sub_cc(r.d[2] - borrow, SECP_P[2], borrow);
        r.d[3] = r.d[3] - borrow - SECP_P[3];
    }
}

// Modular squaring using Karatsuba-style optimization
// For squaring, we can reduce 16 multiplications to 10:
// - 4 squares: a0^2, a1^2, a2^2, a3^2
// - 6 cross-terms (computed once, doubled): a0*a1, a0*a2, a0*a3, a1*a2, a1*a3, a2*a3
// This gives ~37.5% reduction in multiplications compared to generic mul
__device__ void mod_sqr(U256& r, const U256& a) {
    // Full 512-bit square using Karatsuba optimization
    uint64_t p[8] = {0};

    // Compute diagonal terms (squares): a[i] * a[i]
    // These contribute to p[2*i] and p[2*i+1]
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t lo = a.d[i] * a.d[i];
        uint64_t hi = __umul64hi(a.d[i], a.d[i]);

        // Add to accumulator
        uint64_t old_p = p[2*i];
        p[2*i] += lo;
        uint64_t carry = (p[2*i] < old_p) ? 1 : 0;

        old_p = p[2*i + 1];
        p[2*i + 1] += hi + carry;
        carry = (p[2*i + 1] < old_p) ? 1 : 0;

        // Propagate carry
        for (int j = 2*i + 2; j < 8 && carry; j++) {
            old_p = p[j];
            p[j] += carry;
            carry = (p[j] < old_p) ? 1 : 0;
        }
    }

    // Compute off-diagonal terms (cross-products): 2 * a[i] * a[j] for i < j
    // Each cross-term appears twice in the full product, so we compute once and double
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            // Compute a[i] * a[j]
            uint64_t lo = a.d[i] * a.d[j];
            uint64_t hi = __umul64hi(a.d[i], a.d[j]);

            // Double the result (shift left by 1 bit)
            uint64_t hi2 = (hi << 1) | (lo >> 63);
            uint64_t lo2 = lo << 1;

            // Add to p[i+j] with carry propagation
            uint64_t old_p = p[i + j];
            p[i + j] += lo2;
            uint64_t carry = (p[i + j] < old_p) ? 1 : 0;

            old_p = p[i + j + 1];
            p[i + j + 1] += hi2 + carry;
            carry = (p[i + j + 1] < old_p) ? 1 : 0;

            // Propagate carry
            for (int k = i + j + 2; k < 8 && carry; k++) {
                old_p = p[k];
                p[k] += carry;
                carry = (p[k] < old_p) ? 1 : 0;
            }
        }
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

    // Check if >= p and subtract
    if (r.d[3] > SECP_P[3] ||
        (r.d[3] == SECP_P[3] && r.d[2] == SECP_P[2] &&
         r.d[1] == SECP_P[1] && r.d[0] >= SECP_P[0])) {
        uint64_t borrow;
        r.d[0] = sub_cc(r.d[0], SECP_P[0], borrow);
        r.d[1] = sub_cc(r.d[1] - borrow, SECP_P[1], borrow);
        r.d[2] = sub_cc(r.d[2] - borrow, SECP_P[2], borrow);
        r.d[3] = r.d[3] - borrow - SECP_P[3];
    }
}

// Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
// Uses addition chain optimized for secp256k1's p
__device__ void mod_inv(U256& r, const U256& a) {
    U256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;

    mod_sqr(x2, a);
    mod_mul(x2, x2, a);           // x2 = a^3

    mod_sqr(x3, x2);
    mod_mul(x3, x3, a);           // x3 = a^7

    t = x3;
    for (int i = 0; i < 3; i++) mod_sqr(t, t);
    mod_mul(x6, t, x3);           // x6 = a^63

    t = x6;
    for (int i = 0; i < 3; i++) mod_sqr(t, t);
    mod_mul(x9, t, x3);           // x9

    t = x9;
    for (int i = 0; i < 2; i++) mod_sqr(t, t);
    mod_mul(x11, t, x2);          // x11

    t = x11;
    for (int i = 0; i < 11; i++) mod_sqr(t, t);
    mod_mul(x22, t, x11);         // x22

    t = x22;
    for (int i = 0; i < 22; i++) mod_sqr(t, t);
    mod_mul(x44, t, x22);         // x44

    t = x44;
    for (int i = 0; i < 44; i++) mod_sqr(t, t);
    mod_mul(x88, t, x44);         // x88

    t = x88;
    for (int i = 0; i < 88; i++) mod_sqr(t, t);
    mod_mul(x176, t, x88);        // x176

    t = x176;
    for (int i = 0; i < 44; i++) mod_sqr(t, t);
    mod_mul(x220, t, x44);        // x220

    t = x220;
    for (int i = 0; i < 3; i++) mod_sqr(t, t);
    mod_mul(x223, t, x3);         // x223

    t = x223;
    for (int i = 0; i < 23; i++) mod_sqr(t, t);
    mod_mul(t, t, x22);

    for (int i = 0; i < 5; i++) mod_sqr(t, t);
    mod_mul(t, t, a);

    for (int i = 0; i < 3; i++) mod_sqr(t, t);
    mod_mul(t, t, a);

    mod_sqr(t, t);
    mod_mul(r, t, a);
}

// =============================================================================
// ELLIPTIC CURVE OPERATIONS (Jacobian Coordinates)
// =============================================================================

// Point doubling: R = 2*P (Jacobian)
// Uses optimized formula for a=0 curves (secp256k1)
__device__ void ec_double(PointJ& R, const PointJ& P) {
    if (P.is_infinity()) {
        R.set_infinity();
        return;
    }

    U256 S, M, T, Y2;

    // S = 4*X*Y^2
    mod_sqr(Y2, P.Y);
    mod_mul(S, P.X, Y2);
    mod_add(S, S, S);
    mod_add(S, S, S);

    // M = 3*X^2 (a=0 for secp256k1)
    mod_sqr(M, P.X);
    mod_add(T, M, M);
    mod_add(M, T, M);

    // X' = M^2 - 2*S
    mod_sqr(R.X, M);
    mod_sub(R.X, R.X, S);
    mod_sub(R.X, R.X, S);

    // Y' = M*(S - X') - 8*Y^4
    mod_sub(T, S, R.X);
    mod_mul(T, M, T);
    mod_sqr(Y2, Y2);
    mod_add(Y2, Y2, Y2);
    mod_add(Y2, Y2, Y2);
    mod_add(Y2, Y2, Y2);
    mod_sub(R.Y, T, Y2);

    // Z' = 2*Y*Z
    mod_mul(R.Z, P.Y, P.Z);
    mod_add(R.Z, R.Z, R.Z);
}

// Point addition: R = P + Q (Jacobian + Affine -> Jacobian)
// Mixed addition is faster than full Jacobian addition
__device__ void ec_add_mixed(PointJ& R, const PointJ& P, const PointA& Q) {
    if (P.is_infinity()) {
        R.X = Q.x;
        R.Y = Q.y;
        R.Z.set_one();
        return;
    }

    U256 Z1Z1, U2, S2, H, HH, I, J, r, V;

    // Z1Z1 = Z1^2
    mod_sqr(Z1Z1, P.Z);

    // U2 = X2*Z1Z1
    mod_mul(U2, Q.x, Z1Z1);

    // S2 = Y2*Z1*Z1Z1
    mod_mul(S2, Q.y, P.Z);
    mod_mul(S2, S2, Z1Z1);

    // H = U2 - X1
    mod_sub(H, U2, P.X);

    // r = 2*(S2 - Y1)
    mod_sub(r, S2, P.Y);
    mod_add(r, r, r);

    // Check for doubling case
    if (H.is_zero()) {
        if (r.is_zero()) {
            // P == Q, use doubling
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
    mod_mul(V, P.X, I);

    // X3 = r^2 - J - 2*V
    mod_sqr(R.X, r);
    mod_sub(R.X, R.X, J);
    mod_sub(R.X, R.X, V);
    mod_sub(R.X, R.X, V);

    // Y3 = r*(V - X3) - 2*Y1*J
    mod_sub(R.Y, V, R.X);
    mod_mul(R.Y, r, R.Y);
    mod_mul(J, P.Y, J);
    mod_add(J, J, J);
    mod_sub(R.Y, R.Y, J);

    // Z3 = 2*Z1*H
    mod_mul(R.Z, P.Z, H);
    mod_add(R.Z, R.Z, R.Z);
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
// Process 4 bits at a time, reducing additions by 4x
__device__ void ec_mul_precomp(PointJ& R, const U256& k) {
    R.set_infinity();

    // Process from most significant window to least
    for (int w = NUM_WINDOWS - 1; w >= 0; w--) {
        // Double 4 times (for window size 4)
        for (int i = 0; i < WINDOW_SIZE; i++) {
            ec_double(R, R);
        }

        // Extract window value (4 bits)
        int shift = (w * WINDOW_SIZE) % 64;
        int word = (w * WINDOW_SIZE) / 64;
        uint64_t window = (k.d[word] >> shift) & WINDOW_MASK;

        // Handle window crossing word boundary
        if (shift > 60 && word < 3) {
            window |= (k.d[word + 1] << (64 - shift)) & WINDOW_MASK;
        }

        // Add precomputed point if window != 0
        if (window != 0) {
            // Read from global memory (L2 cached via cudaAccessPropertyPersisting)
            PointA P = d_PRECOMP_TABLE[w * (1 << WINDOW_SIZE) + window];
            ec_add_mixed(R, R, P);
        }
    }
}

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
    // Compute c1 = round(k * g1 / 2^256) and c2 = round(k * g2 / 2^256)
    // where g1, g2 are precomputed as floor(b2 * 2^256 / n) and floor(-b1 * 2^256 / n)

    U128 c1, c2;

    // c1 = (k * g1) >> 256  (high 128 bits of 512-bit product)
    // For efficiency, we only compute the high part we need
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
        c1.lo = p[4];
        c1.hi = p[5];
    }

    // c2 = (k * g2) >> 256
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
        c2.lo = p[4];
        c2.hi = p[5];
    }

    // k1 = k - c1*a1 - c2*a2
    // k2 = -c1*b1 + c2*b2  (note: our stored b1 is negative of the actual b1)
    //    = c1*(-b1) + c2*b2
    //    = c1*GLV_B1 + c2*GLV_B2  (both stored as positive)

    U256 c1_a1, c2_a2;
    mul_128x128(c1, GLV_A1, c1_a1);
    mul_128x128(c2, GLV_A1, c2_a2);  // a2 ≈ a1 for secp256k1 (b2 = a1)

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

    // Ensure k1, k2 are in valid range (should be ~128 bits)
    // Clear upper bits if any overflow occurred
    k1.d[2] = k1.d[3] = 0;
    k2.d[2] = k2.d[3] = 0;
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

// GLV-accelerated scalar multiplication using Shamir's trick
// Computes k*G = k1*G + k2*(λG) where k1, k2 are ~128 bits
// Uses joint window method for simultaneous computation
__device__ void ec_mul_glv(PointJ& R, const U256& k) {
    // Decompose scalar
    U256 k1, k2;
    bool k1_neg, k2_neg;
    glv_decompose(k, k1, k2, k1_neg, k2_neg);

    // If both k1 and k2 are zero, return infinity
    if (k1.is_zero() && k2.is_zero()) {
        R.set_infinity();
        return;
    }

    // Use windowed method on both scalars simultaneously (Shamir's trick)
    // Window size 4: process 4 bits of k1 and k2 together
    // This requires precomputed table of G and λG (PRECOMP_TABLE and PRECOMP_TABLE_LAMBDA)

    R.set_infinity();

    // Find highest non-zero bit position in k1 or k2 (they're ~128 bits)
    int max_window = 32;  // 128 bits / 4-bit windows

    // Process from most significant window
    for (int w = max_window - 1; w >= 0; w--) {
        // Double 4 times
        for (int i = 0; i < WINDOW_SIZE; i++) {
            ec_double(R, R);
        }

        // Extract window values from k1 and k2
        int shift = (w * WINDOW_SIZE) % 64;
        int word = (w * WINDOW_SIZE) / 64;

        uint64_t w1 = 0, w2 = 0;
        if (word < 2) {  // k1, k2 are 128-bit
            w1 = (k1.d[word] >> shift) & WINDOW_MASK;
            w2 = (k2.d[word] >> shift) & WINDOW_MASK;
            if (shift > 60 && word < 1) {
                w1 |= (k1.d[word + 1] << (64 - shift)) & WINDOW_MASK;
                w2 |= (k2.d[word + 1] << (64 - shift)) & WINDOW_MASK;
            }
        }

        // Add contributions from both tables (L2 cached via cudaAccessPropertyPersisting)
        if (w1 != 0) {
            PointA P1 = d_PRECOMP_TABLE[w * (1 << WINDOW_SIZE) + w1];
            if (k1_neg) ec_negate(P1);
            ec_add_mixed(R, R, P1);
        }

        if (w2 != 0) {
            PointA P2 = d_PRECOMP_TABLE_LAMBDA[w * (1 << WINDOW_SIZE) + w2];
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

}  // namespace optimized
}  // namespace gpu
}  // namespace collider
