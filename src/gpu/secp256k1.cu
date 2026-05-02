/**
 * Collider GPU secp256k1 Implementation - OPTIMIZED
 *
 * High-performance elliptic curve operations for brain wallet research.
 * Implements all critical optimizations:
 * - Precomputed table for windowed scalar multiplication (16x speedup)
 * - Schoolbook multiplication with secp256k1 special-form fast reduction
 *   (p = 2^256 - 2^32 - 977, NOT Montgomery form)
 * - Batch inversion using Montgomery's Trick (85x speedup on inversions)
 * - Optimized Jacobian coordinate arithmetic
 * - PTX inline assembly for carry chain optimization
 *
 * Target: 2.5B+ scalar multiplications per second per RTX 5090
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <mutex>

namespace collider {
namespace gpu {

// =============================================================================
// secp256k1 CURVE PARAMETERS
// =============================================================================

// Field prime: p = 2^256 - 2^32 - 977
static __constant__ uint32_t SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// p + 1 (for some reductions) - reserved for future Montgomery optimizations
// static __constant__ uint32_t SECP256K1_P_PLUS_1[8] = {
//     0xFFFFFC30, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
//     0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
// };

// Generator point Gx
static __constant__ uint32_t SECP256K1_GX[8] = {
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};

// Generator point Gy
static __constant__ uint32_t SECP256K1_GY[8] = {
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
};

// Montgomery constants - NOT USED. Field arithmetic uses schoolbook multiply
// with secp256k1 special-form reduction. Reserved for potential future use.
// static __constant__ uint32_t MONT_R[8] = {
//     0x000003D1, 0x00000001, 0x00000000, 0x00000000,
//     0x00000000, 0x00000000, 0x00000000, 0x00000000
// };
// static __constant__ uint32_t MONT_R2[8] = {
//     0x000E90A1, 0x000007A2, 0x00000001, 0x00000000,
//     0x00000000, 0x00000000, 0x00000000, 0x00000000
// };
// static __constant__ uint32_t MONT_N_PRIME = 0xD2253531;

static constexpr uint32_t SECP256K1_C_LO = 977;  // Low part of c = 2^32 + 977

#define MAX_TSIZE 4096  // Maximum table entries per window for batch inversion arrays

// =============================================================================
// GLV ENDOMORPHISM CONSTANTS (1.5x speedup for scalar multiplication)
// =============================================================================
// secp256k1 has efficient endomorphism: lambda * P = (beta * P.x, P.y)
// This allows decomposing k into k1 + k2*lambda where |k1|, |k2| ≈ sqrt(n)

// Beta: cube root of 1 mod p (for point transformation)
// beta^3 = 1 mod p
static __constant__ uint32_t GLV_BETA[8] = {
    0x719501EE, 0xC1396C28, 0x12F58995, 0x9CF04975,
    0xAC3434E9, 0x6E64479E, 0x657C0710, 0x7AE96A2B
};

// GLV constants - reserved for future GLV endomorphism implementation
// Lambda: cube root of 1 mod n (for scalar decomposition)
// static __constant__ uint32_t GLV_LAMBDA[8] = {
//     0x1B23BD72, 0xDF02967C, 0x20816678, 0x122E22EA,
//     0x8812645A, 0xA5261C02, 0xC05C30E0, 0x5363AD4C
// };
// Curve order n
// static __constant__ uint32_t SECP256K1_N[8] = {
//     0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
//     0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
// };
// GLV decomposition constants (a1, b1, a2, b2 from libsecp256k1)
// static __constant__ uint32_t GLV_A1[4] = { 0xE4437ED6, 0xEB03090F, 0x30A198A9, 0x3086D221 };
// static __constant__ uint32_t GLV_B1[4] = { 0xE86C90E4, 0x8B76EAAD, 0xF98FCFBF, 0x114CA50F };
// static __constant__ uint32_t GLV_A2[4] = { 0xE86C90E4, 0x8B76EAAD, 0xF98FCFBF, 0x114CA50F };
// static __constant__ uint32_t GLV_B2[4] = { 0x3AA1B14C, 0x8DAC0C6E, 0x0C3F3F2A, 0x1950B75F };

// =============================================================================
// DATA STRUCTURES
// =============================================================================

struct uint256 {
    uint32_t limbs[8];

    __device__ __forceinline__ bool is_zero() const {
        return (limbs[0] | limbs[1] | limbs[2] | limbs[3] |
                limbs[4] | limbs[5] | limbs[6] | limbs[7]) == 0;
    }

    __device__ __forceinline__ void set_zero() {
        #pragma unroll
        for (int i = 0; i < 8; i++) limbs[i] = 0;
    }

    __device__ __forceinline__ void set_one() {
        limbs[0] = 1;
        #pragma unroll
        for (int i = 1; i < 8; i++) limbs[i] = 0;
    }
};

// Jacobian coordinates: (X : Y : Z) represents affine (X/Z^2, Y/Z^3)
struct ECPointJacobian {
    uint256 X;
    uint256 Y;
    uint256 Z;

    __device__ __forceinline__ bool is_infinity() const {
        return Z.is_zero();
    }

    __device__ __forceinline__ void set_infinity() {
        X.set_one();
        Y.set_one();
        Z.set_zero();
    }
};

// Affine coordinates
struct ECPointAffine {
    uint256 x;
    uint256 y;
};

// Precomputed table entry (affine for memory efficiency)
struct PrecomputedPoint {
    uint256 x;
    uint256 y;
};

// =============================================================================
// 256-BIT ARITHMETIC (OPTIMIZED)
// =============================================================================

/**
 * Compare two 256-bit integers.
 */
__device__ __forceinline__ int uint256_cmp(const uint256& a, const uint256& b) {
    #pragma unroll
    for (int i = 7; i >= 0; i--) {
        if (a.limbs[i] < b.limbs[i]) return -1;
        if (a.limbs[i] > b.limbs[i]) return 1;
    }
    return 0;
}

/**
 * 256-bit addition: result = a + b, returns carry
 */
__device__ __forceinline__ uint32_t uint256_add(
    uint256& result, const uint256& a, const uint256& b
) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a.limbs[i] + b.limbs[i] + carry;
        result.limbs[i] = (uint32_t)sum;
        carry = sum >> 32;
    }
    return (uint32_t)carry;
}

/**
 * 256-bit subtraction: result = a - b, returns borrow
 */
__device__ __forceinline__ uint32_t uint256_sub(
    uint256& result, const uint256& a, const uint256& b
) {
    int64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int64_t diff = (int64_t)a.limbs[i] - b.limbs[i] - borrow;
        result.limbs[i] = (uint32_t)diff;
        borrow = (diff < 0) ? 1 : 0;
    }
    return (uint32_t)borrow;
}

/**
 * Load constant array into uint256
 */
__device__ __forceinline__ void uint256_load_const(uint256& a, const uint32_t* c) {
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        a.limbs[i] = c[i];
    }
}

// =============================================================================
// MODULAR ARITHMETIC (secp256k1 OPTIMIZED)
// =============================================================================

/**
 * PTX-optimized 256x256→512 bit multiplication using IMAD/IADD3.
 * Uses explicit carry chains for better instruction scheduling.
 */
__device__ void uint256_mul_512_ptx(
    uint32_t* result,       // 16 limbs output
    const uint256& a,
    const uint256& b
) {
    // Initialize result to zero
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        result[i] = 0;
    }

    // Schoolbook multiplication with PTX-optimized inner loop
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t carry = 0;

        // Inner loop uses PTX for optimal carry propagation
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint32_t lo, hi;

            // a[i] * b[j] + result[i+j] + carry
            asm volatile (
                "{\n\t"
                "  .reg .u32 t0, t1;\n\t"
                "  mul.lo.u32 t0, %2, %3;\n\t"       // t0 = a*b low
                "  mul.hi.u32 t1, %2, %3;\n\t"       // t1 = a*b high
                "  add.cc.u32 t0, t0, %4;\n\t"       // t0 += result[i+j]
                "  addc.u32 t1, t1, 0;\n\t"          // t1 += carry
                "  add.cc.u32 %0, t0, %5;\n\t"       // lo = t0 + carry_in
                "  addc.u32 %1, t1, 0;\n\t"          // hi = t1 + carry
                "}\n\t"
                : "=r"(lo), "=r"(hi)
                : "r"(a.limbs[i]), "r"(b.limbs[j]), "r"(result[i+j]), "r"(carry)
            );

            result[i+j] = lo;
            carry = hi;
        }

        // Propagate final carry with overflow protection
        {
            uint64_t sum = (uint64_t)result[i+8] + carry;
            result[i+8] = (uint32_t)sum;
            uint32_t prop = (uint32_t)(sum >> 32);
            for (int j = i+9; j < 16 && prop; j++) {
                sum = (uint64_t)result[j] + prop;
                result[j] = (uint32_t)sum;
                prop = (uint32_t)(sum >> 32);
            }
        }
    }
}

/**
 * Modular addition: result = (a + b) mod p
 */
__device__ __forceinline__ void mod_add(uint256& result, const uint256& a, const uint256& b) {
    uint32_t carry = uint256_add(result, a, b);

    uint256 p;
    uint256_load_const(p, SECP256K1_P);

    // Reduce if overflow or result >= p
    if (carry || uint256_cmp(result, p) >= 0) {
        uint256_sub(result, result, p);
    }
}

/**
 * Modular subtraction: result = (a - b) mod p
 */
__device__ __forceinline__ void mod_sub(uint256& result, const uint256& a, const uint256& b) {
    uint32_t borrow = uint256_sub(result, a, b);

    if (borrow) {
        uint256 p;
        uint256_load_const(p, SECP256K1_P);
        uint256_add(result, result, p);
    }
}

/**
 * Modular multiplication using secp256k1 fast reduction.
 * For a*b mod p where a,b < p.
 * OPTIMIZED: Uses PTX inline assembly for carry chain optimization.
 */
__device__ void mod_mul(uint256& result, const uint256& a, const uint256& b) {
    // 512-bit product using PTX-optimized multiplication
    uint32_t prod[16];
    uint256_mul_512_ptx(prod, a, b);

    // Fast reduction using p = 2^256 - c, where c = 2^32 + 977
    // For r = prod mod p:
    // Split prod = high * 2^256 + low
    // r = low + high * c (mod p)

    uint256 low;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        low.limbs[i] = prod[i];
    }

    // Compute high * c where c = 2^32 + 977
    // high * c = high * 2^32 + high * 977
    uint64_t carry = 0;
    uint32_t high_c[9] = {0};

    // high * SECP256K1_C_LO (977)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t term = (uint64_t)prod[i+8] * (uint64_t)SECP256K1_C_LO + carry;
        high_c[i] = (uint32_t)term;
        carry = term >> 32;
    }
    high_c[8] = (uint32_t)carry;

    // Add high * 2^32 (shift high by 1 limb and add)
    carry = 0;
    #pragma unroll
    for (int i = 1; i < 9; i++) {
        uint64_t term = (uint64_t)high_c[i] + prod[i+7] + carry;
        high_c[i] = (uint32_t)term;
        carry = term >> 32;
    }

    // Now add low + high_c[0..7]
    uint256 correction;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        correction.limbs[i] = high_c[i];
    }

    uint32_t c1 = uint256_add(result, low, correction);

    // Handle any overflow from high_c[8] and c1
    uint64_t extra = (uint64_t)high_c[8] + c1;
    if (extra) {
        // extra * c mod p where c = 2^32 + SECP256K1_C_LO (977)
        uint32_t e0 = (uint32_t)(extra * (uint64_t)SECP256K1_C_LO);
        uint32_t e1 = (uint32_t)(extra);

        carry = (uint64_t)result.limbs[0] + e0;
        result.limbs[0] = (uint32_t)carry;
        carry = (carry >> 32) + (uint64_t)result.limbs[1] + e1;
        result.limbs[1] = (uint32_t)carry;
        carry >>= 32;

        for (int i = 2; i < 8 && carry; i++) {
            carry = (uint64_t)result.limbs[i] + carry;
            result.limbs[i] = (uint32_t)carry;
            carry >>= 32;
        }
    }

    // Note: carry from reduction is bounded; conditional subtractions below handle any residual >= P

    // Final conditional subtraction if result >= P
    {
        uint256 p;
        uint256_load_const(p, SECP256K1_P);
        if (uint256_cmp(result, p) >= 0) {
            uint256_sub(result, result, p);
        }
        if (uint256_cmp(result, p) >= 0) {
            uint256_sub(result, result, p);
        }
    }
}

/**
 * Dedicated modular squaring exploiting symmetry of cross-terms.
 * For 8 limbs: 8 diagonal + 28 cross-term multiplies instead of 64.
 * ~30-40% faster than mod_mul(a, a).
 */
__device__ void mod_sqr(uint256& result, const uint256& a) {
    uint32_t prod[16];

    // Initialize 512-bit product to zero
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        prod[i] = 0;
    }

    // Step 1: Compute cross-terms a[i]*a[j] for i < j, accumulate once
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t carry = 0;
        #pragma unroll
        for (int j = i + 1; j < 8; j++) {
            uint32_t lo, hi;
            // a[i] * a[j] + prod[i+j] + carry
            asm volatile (
                "{\n\t"
                "  .reg .u32 t0, t1;\n\t"
                "  mul.lo.u32 t0, %2, %3;\n\t"
                "  mul.hi.u32 t1, %2, %3;\n\t"
                "  add.cc.u32 t0, t0, %4;\n\t"
                "  addc.u32 t1, t1, 0;\n\t"
                "  add.cc.u32 %0, t0, %5;\n\t"
                "  addc.u32 %1, t1, 0;\n\t"
                "}\n\t"
                : "=r"(lo), "=r"(hi)
                : "r"(a.limbs[i]), "r"(a.limbs[j]), "r"(prod[i+j]), "r"(carry)
            );
            prod[i+j] = lo;
            carry = hi;
        }
        uint64_t sum64 = (uint64_t)prod[i+8] + carry;
        prod[i+8] = (uint32_t)sum64;
        uint32_t prop = (uint32_t)(sum64 >> 32);
        for (int j = i+9; j < 16 && prop; j++) {
            sum64 = (uint64_t)prod[j] + prop;
            prod[j] = (uint32_t)sum64;
            prop = (uint32_t)(sum64 >> 32);
        }
    }

    // Step 2: Double the cross-terms (shift entire 512-bit result left by 1 bit)
    uint32_t top_bit = 0;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t new_top = prod[i] >> 31;
        prod[i] = (prod[i] << 1) | top_bit;
        top_bit = new_top;
    }

    // Step 3: Add diagonal terms a[i]*a[i] at positions [2i, 2i+1]
    uint64_t dcarry = 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t lo, hi;
        asm volatile (
            "mul.lo.u32 %0, %2, %2;\n\t"
            "mul.hi.u32 %1, %2, %2;\n\t"
            : "=r"(lo), "=r"(hi)
            : "r"(a.limbs[i])
        );

        uint64_t sum = (uint64_t)prod[2*i] + lo + dcarry;
        prod[2*i] = (uint32_t)sum;
        dcarry = sum >> 32;

        sum = (uint64_t)prod[2*i + 1] + hi + dcarry;
        prod[2*i + 1] = (uint32_t)sum;
        dcarry = sum >> 32;
    }

    // Step 4: secp256k1 fast reduction (identical to mod_mul)
    // p = 2^256 - c, where c = 2^32 + 977

    uint256 low;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        low.limbs[i] = prod[i];
    }

    // Compute high * c where c = 2^32 + 977
    // high * c = high * 2^32 + high * 977
    uint64_t carry = 0;
    uint32_t high_c[9] = {0};

    // high * SECP256K1_C_LO (977)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t term = (uint64_t)prod[i+8] * (uint64_t)SECP256K1_C_LO + carry;
        high_c[i] = (uint32_t)term;
        carry = term >> 32;
    }
    high_c[8] = (uint32_t)carry;

    // Add high * 2^32 (shift high by 1 limb and add)
    carry = 0;
    #pragma unroll
    for (int i = 1; i < 9; i++) {
        uint64_t term = (uint64_t)high_c[i] + prod[i+7] + carry;
        high_c[i] = (uint32_t)term;
        carry = term >> 32;
    }

    // Now add low + high_c[0..7]
    uint256 correction;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        correction.limbs[i] = high_c[i];
    }

    uint32_t c1 = uint256_add(result, low, correction);

    // Handle any overflow from high_c[8] and c1
    uint64_t extra = (uint64_t)high_c[8] + c1;
    if (extra) {
        // extra * c mod p where c = 2^32 + SECP256K1_C_LO (977)
        uint32_t e0 = (uint32_t)(extra * (uint64_t)SECP256K1_C_LO);
        uint32_t e1 = (uint32_t)(extra);

        carry = (uint64_t)result.limbs[0] + e0;
        result.limbs[0] = (uint32_t)carry;
        carry = (carry >> 32) + (uint64_t)result.limbs[1] + e1;
        result.limbs[1] = (uint32_t)carry;
        carry >>= 32;

        for (int i = 2; i < 8 && carry; i++) {
            carry = (uint64_t)result.limbs[i] + carry;
            result.limbs[i] = (uint32_t)carry;
            carry >>= 32;
        }
    }

    // Note: carry from reduction is bounded; conditional subtractions below handle any residual >= P

    // Final conditional subtraction if result >= P
    {
        uint256 p;
        uint256_load_const(p, SECP256K1_P);
        if (uint256_cmp(result, p) >= 0) {
            uint256_sub(result, result, p);
        }
        if (uint256_cmp(result, p) >= 0) {
            uint256_sub(result, result, p);
        }
    }
}

/**
 * Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
 *
 * OPTIMIZED: Uses an efficient addition chain specifically designed for secp256k1.
 * The exponent p-2 = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2D
 *
 * This addition chain reduces from ~256 squarings + ~128 multiplications (binary exp)
 * to ~255 squarings + ~13 multiplications, a significant speedup.
 *
 * Based on the libsecp256k1 addition chain which is near-optimal.
 */
__device__ void mod_inv(uint256& result, const uint256& a) {
    uint256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;

    // Start building up powers using addition chain
    // x2 = a^(2^2 - 1) = a^3  (xN convention: N-bit all-ones run)
    mod_sqr(x2, a);           // a^2
    mod_mul(x2, x2, a);       // a^3

    // x3 = a^(2^3 - 1) = a^7
    mod_sqr(x3, x2);          // a^6
    mod_mul(x3, x3, a);       // a^7

    // x6 = a^(2^6 - 1) = a^63
    mod_sqr(t1, x3);          // a^14
    mod_sqr(t1, t1);          // a^28
    mod_sqr(t1, t1);          // a^56
    mod_mul(x6, t1, x3);      // a^63

    // x9 = a^(2^9 - 1) = a^511
    mod_sqr(t1, x6);          // a^126
    mod_sqr(t1, t1);          // a^252
    mod_sqr(t1, t1);          // a^504
    mod_mul(x9, t1, x3);      // a^511

    // x11 = a^(2^11 - 1) = a^2047
    mod_sqr(t1, x9);          // a^1022
    mod_sqr(t1, t1);          // a^2044
    mod_mul(x11, t1, x2);     // a^2047

    // x22 = a^(2^22 - 1)
    mod_sqr(t1, x11);
    #pragma unroll
    for (int i = 0; i < 10; i++) {
        mod_sqr(t1, t1);      // a^(2^22 - 2^11)
    }
    mod_mul(x22, t1, x11);    // a^(2^22 - 1)

    // x44 = a^(2^44 - 1)
    mod_sqr(t1, x22);
    #pragma unroll
    for (int i = 0; i < 21; i++) {
        mod_sqr(t1, t1);      // a^(2^44 - 2^22)
    }
    mod_mul(x44, t1, x22);    // a^(2^44 - 1)

    // x88 = a^(2^88 - 1)
    mod_sqr(t1, x44);
    #pragma unroll
    for (int i = 0; i < 43; i++) {
        mod_sqr(t1, t1);      // a^(2^88 - 2^44)
    }
    mod_mul(x88, t1, x44);    // a^(2^88 - 1)

    // x176 = a^(2^176 - 1)
    mod_sqr(t1, x88);
    #pragma unroll
    for (int i = 0; i < 87; i++) {
        mod_sqr(t1, t1);      // a^(2^176 - 2^88)
    }
    mod_mul(x176, t1, x88);   // a^(2^176 - 1)

    // x220 = a^(2^220 - 1)
    mod_sqr(t1, x176);
    #pragma unroll
    for (int i = 0; i < 43; i++) {
        mod_sqr(t1, t1);      // a^(2^220 - 2^44)
    }
    mod_mul(x220, t1, x44);   // a^(2^220 - 1)

    // x223 = a^(2^223 - 1)
    // From x220, we square 3 times and multiply by x3 = a^7
    mod_sqr(t1, x220);        // a^(2^221 - 2)
    mod_sqr(t1, t1);          // a^(2^222 - 4)
    mod_sqr(t1, t1);          // a^(2^223 - 8)
    mod_mul(x223, t1, x3);    // a^(2^223 - 1) since x3 = a^7

    // p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    // Structure from bit 255 down: 223 ones, 1 zero (bit 32), 22 ones (bits 31..10),
    // then bits 9..0 = 0000101101

    mod_sqr(t1, x223);
    #pragma unroll
    for (int i = 0; i < 22; i++) {
        mod_sqr(t1, t1);
    }
    mod_mul(t1, t1, x22);
    // t1 = a^(2^246 - 2^22 - 1) -- covers bits 255..10

    // Remaining bits 9..0 = 0000101101
    mod_sqr(t1, t1);  // bit 9 = 0
    mod_sqr(t1, t1);  // bit 8 = 0
    mod_sqr(t1, t1);  // bit 7 = 0
    mod_sqr(t1, t1);  // bit 6 = 0
    mod_sqr(t1, t1);  // bit 5 = 1
    mod_mul(t1, t1, a);
    mod_sqr(t1, t1);  // bit 4 = 0
    mod_sqr(t1, t1);  // bits 3,2 = 11
    mod_sqr(t1, t1);
    mod_mul(t1, t1, x2);
    mod_sqr(t1, t1);  // bit 1 = 0
    mod_sqr(t1, t1);  // bit 0 = 1
    mod_mul(result, t1, a);
}

// =============================================================================
// ELLIPTIC CURVE POINT OPERATIONS (JACOBIAN COORDINATES)
// =============================================================================

/**
 * Point doubling in Jacobian coordinates.
 * Uses optimized formulas for a=0 (secp256k1).
 * Cost: 1M + 5S + 1*a + 7add + 2*2 + 1*3 + 1*8
 */
__device__ void ec_double_jacobian(ECPointJacobian& R, const ECPointJacobian& P) {
    if (P.is_infinity()) {
        R.set_infinity();
        return;
    }

    uint256 S, M, T, Y2;

    // Y^2
    mod_sqr(Y2, P.Y);

    // S = 4 * X * Y^2
    mod_mul(S, P.X, Y2);        // X * Y^2
    mod_add(S, S, S);           // 2 * X * Y^2
    mod_add(S, S, S);           // 4 * X * Y^2

    // M = 3 * X^2 (since a = 0 for secp256k1)
    mod_sqr(M, P.X);            // X^2
    mod_add(T, M, M);           // 2 * X^2
    mod_add(M, T, M);           // 3 * X^2

    // X' = M^2 - 2*S
    mod_sqr(R.X, M);            // M^2
    mod_sub(R.X, R.X, S);       // M^2 - S
    mod_sub(R.X, R.X, S);       // M^2 - 2*S

    // Y' = M * (S - X') - 8 * Y^4
    mod_sub(T, S, R.X);         // S - X'
    mod_mul(T, M, T);           // M * (S - X')

    mod_sqr(Y2, Y2);            // Y^4
    mod_add(Y2, Y2, Y2);        // 2 * Y^4
    mod_add(Y2, Y2, Y2);        // 4 * Y^4
    mod_add(Y2, Y2, Y2);        // 8 * Y^4

    mod_sub(R.Y, T, Y2);        // M * (S - X') - 8 * Y^4

    // Z' = 2 * Y * Z
    mod_mul(R.Z, P.Y, P.Z);
    mod_add(R.Z, R.Z, R.Z);
}

/**
 * Point addition: R = P + Q where Q is affine, P is Jacobian.
 * Mixed addition is more efficient.
 * Cost: 7M + 4S + 9add + 3*2 + 1*3
 */
__device__ void ec_add_mixed(ECPointJacobian& R, const ECPointJacobian& P, const ECPointAffine& Q) {
    // Handle special cases
    if (P.is_infinity()) {
        R.X = Q.x;
        R.Y = Q.y;
        R.Z.set_one();
        return;
    }

    uint256 Z1Z1, U2, S2, H, HH, I, J, r, V;

    // Z1Z1 = Z1^2
    mod_sqr(Z1Z1, P.Z);

    // U2 = X2 * Z1Z1
    mod_mul(U2, Q.x, Z1Z1);

    // S2 = Y2 * Z1 * Z1Z1
    mod_mul(S2, Q.y, P.Z);
    mod_mul(S2, S2, Z1Z1);

    // H = U2 - X1
    mod_sub(H, U2, P.X);

    // r = 2 * (S2 - Y1)
    mod_sub(r, S2, P.Y);
    mod_add(r, r, r);

    // Check if P == Q (need to double instead)
    if (H.is_zero()) {
        if (r.is_zero()) {
            // P == Q, need to double
            ec_double_jacobian(R, P);
            return;
        } else {
            // P == -Q, result is infinity
            R.set_infinity();
            return;
        }
    }

    // HH = H^2
    mod_sqr(HH, H);

    // I = 4 * HH
    mod_add(I, HH, HH);
    mod_add(I, I, I);

    // J = H * I
    mod_mul(J, H, I);

    // V = X1 * I
    mod_mul(V, P.X, I);

    // X3 = r^2 - J - 2*V
    mod_sqr(R.X, r);
    mod_sub(R.X, R.X, J);
    mod_sub(R.X, R.X, V);
    mod_sub(R.X, R.X, V);

    // Y3 = r * (V - X3) - 2 * Y1 * J
    mod_sub(V, V, R.X);         // V - X3
    mod_mul(R.Y, r, V);         // r * (V - X3)
    mod_mul(J, P.Y, J);         // Y1 * J
    mod_add(J, J, J);           // 2 * Y1 * J
    mod_sub(R.Y, R.Y, J);

    // Z3 = 2 * Z1 * H
    mod_mul(R.Z, P.Z, H);
    mod_add(R.Z, R.Z, R.Z);
}

/**
 * Convert Jacobian to Affine coordinates.
 * Requires modular inverse.
 */
__device__ void jacobian_to_affine(ECPointAffine& R, const ECPointJacobian& P) {
    if (P.is_infinity()) {
        R.x.set_zero();
        R.y.set_zero();
        return;
    }

    uint256 Z_inv, Z_inv2, Z_inv3;

    mod_inv(Z_inv, P.Z);
    mod_sqr(Z_inv2, Z_inv);
    mod_mul(Z_inv3, Z_inv2, Z_inv);

    mod_mul(R.x, P.X, Z_inv2);
    mod_mul(R.y, P.Y, Z_inv3);
}

// =============================================================================
// GLV ENDOMORPHISM FUNCTIONS (1.5x speedup for scalar multiplication)
// =============================================================================

/**
 * Apply GLV endomorphism to a point: lambda * P = (beta * P.x, P.y)
 * This is a very cheap operation - just a field multiplication on x.
 */
__device__ void glv_endomorphism(ECPointAffine& result, const ECPointAffine& P) {
    uint256 beta;
    #pragma unroll
    for (int i = 0; i < 8; i++) beta.limbs[i] = GLV_BETA[i];

    mod_mul(result.x, P.x, beta);
    result.y = P.y;
}

/**
 * Apply GLV endomorphism to a Jacobian point: lambda * P = (beta * P.x, P.y, P.z)
 * The Z coordinate doesn't change since beta only affects x.
 */
__device__ void glv_endomorphism_jacobian(ECPointJacobian& result, const ECPointJacobian& P) {
    uint256 beta;
    #pragma unroll
    for (int i = 0; i < 8; i++) beta.limbs[i] = GLV_BETA[i];

    mod_mul(result.X, P.X, beta);
    result.Y = P.Y;
    result.Z = P.Z;
}

// NOTE: GLV scalar multiplication lives in puzzle_optimized.cu which has the
// correct Babai's algorithm decomposition using precomputed lattice vectors.
// The glv_endomorphism() function above is the only GLV utility needed here.

// =============================================================================
// PRECOMPUTED TABLE FOR SCALAR MULTIPLICATION (SCALABLE)
// =============================================================================

// Runtime-configurable EC table parameters.
// Window size is selected at init based on available VRAM:
//   <8 GB  -> 5-bit (32 pts/window, 52 windows, ~106 KB)
//   8-16   -> 6-bit (64 pts/window, 43 windows, ~176 KB)
//   16-48  -> 8-bit (256 pts/window, 32 windows, ~524 KB)
//   48-64  -> 10-bit (1024 pts/window, 26 windows, ~1.7 MB)
//   64+    -> 12-bit (4096 pts/window, 22 windows, ~5.5 MB)

struct ECTableConfig {
    int window_bits;    // 5, 6, 8, 10, or 12
    int table_size;     // 1 << window_bits (points per window)
    int num_windows;    // ceil(256 / window_bits)
    size_t total_bytes; // num_windows * table_size * sizeof(PrecomputedPoint)
};

// Host-side config (static linkage, set once during init)
static ECTableConfig g_ec_config = {5, 32, 52, 52ULL * 32 * sizeof(PrecomputedPoint)};

// Device-side config (copied to device via cudaMemcpyToSymbol before table gen)
__device__ ECTableConfig d_ec_config;

static ECTableConfig make_ec_config(int window_bits) {
    ECTableConfig cfg;
    cfg.window_bits = window_bits;
    cfg.table_size  = 1 << window_bits;
    cfg.num_windows = (256 + window_bits - 1) / window_bits;
    cfg.total_bytes = (size_t)cfg.num_windows * cfg.table_size * sizeof(PrecomputedPoint);
    return cfg;
}

static int select_window_bits(size_t usable_vram) {
    constexpr size_t GB = 1024ULL * 1024 * 1024;
    if (usable_vram >= 64 * GB) return 12;
    if (usable_vram >= 48 * GB) return 10;
    if (usable_vram >= 16 * GB) return 8;
    if (usable_vram >=  8 * GB) return 6;
    return 5;
}

// Global precomputed table pointer (device memory, L2 cache persistence)
__device__ PrecomputedPoint* d_precomputed_table;

/**
 * Scalar multiplication using windowed method with precomputed table.
 * Much faster than naive double-and-add.
 * Reads window parameters from d_ec_config (set at init time).
 */
__device__ void ec_mul_windowed(
    ECPointAffine& result,
    const uint256& scalar,
    const PrecomputedPoint* table
) {
    const int wbits   = d_ec_config.window_bits;
    const int tsize   = d_ec_config.table_size;
    const int nwin    = d_ec_config.num_windows;
    const uint32_t mask = (uint32_t)tsize - 1;  // bitmask for window extraction

    ECPointJacobian R;
    R.set_infinity();

    // Accumulate positional table entries: T[w][v] = v * 2^(w*wbits) * G
    for (int w = nwin - 1; w >= 0; w--) {
        // Extract window value from scalar using bit-shift (handles limb spanning)
        int bit_start = w * wbits;
        int limb_idx  = bit_start / 32;
        int bit_off   = bit_start % 32;

        uint32_t window_val;
        if (bit_off + wbits <= 32) {
            // Window fits within a single limb
            window_val = (scalar.limbs[limb_idx] >> bit_off) & mask;
        } else {
            // Window spans two limbs
            uint32_t lo = scalar.limbs[limb_idx] >> bit_off;
            uint32_t hi = (limb_idx + 1 < 8) ? scalar.limbs[limb_idx + 1] : 0;
            window_val = (lo | (hi << (32 - bit_off))) & mask;
        }

        // Clamp: if this is the last (most significant) window, bits beyond 256
        // are zero, so window_val is already correct via the limb reads.

        // Add table[w][window_val] if window_val != 0
        if (window_val != 0) {
            ECPointAffine Q;
            int table_idx = w * tsize + window_val;
            Q.x = table[table_idx].x;
            Q.y = table[table_idx].y;

            ECPointJacobian temp;
            ec_add_mixed(temp, R, Q);
            R = temp;
        }
    }

    jacobian_to_affine(result, R);
}

/**
 * Simple double-and-add scalar multiplication (fallback).
 */
__device__ void ec_mul_simple(ECPointAffine& result, const uint256& scalar) {
    ECPointJacobian R;
    R.set_infinity();

    ECPointAffine G;
    uint256_load_const(G.x, SECP256K1_GX);
    uint256_load_const(G.y, SECP256K1_GY);

    for (int i = 255; i >= 0; i--) {
        ECPointJacobian temp;
        ec_double_jacobian(temp, R);
        R = temp;

        int limb = i / 32;
        int bit = i % 32;

        if ((scalar.limbs[limb] >> bit) & 1) {
            ec_add_mixed(temp, R, G);
            R = temp;
        }
    }

    jacobian_to_affine(result, R);
}

// =============================================================================
// BATCH INVERSION (MONTGOMERY'S TRICK)
// =============================================================================

/**
 * Batch inversion using Montgomery's trick.
 * Given z[0..n-1], compute z_inv[0..n-1] = z[i]^(-1) mod p
 * Using only ONE modular inversion (instead of n).
 *
 * Algorithm:
 * 1. products[i] = z[0] * z[1] * ... * z[i]
 * 2. inv_all = products[n-1]^(-1)
 * 3. Back-propagate: z_inv[i] = inv_all * products[i-1]
 *                   inv_all = inv_all * z[i]
 */
__device__ void batch_invert(
    uint256* z_inv,
    const uint256* z,
    int n,
    uint256* products  // Scratch space, size n
) {
    if (n == 0) return;
    if (n == 1) {
        mod_inv(z_inv[0], z[0]);
        return;
    }

    // Forward pass: compute cumulative products
    products[0] = z[0];
    for (int i = 1; i < n; i++) {
        mod_mul(products[i], products[i-1], z[i]);
    }

    // Invert the final product (only ONE inversion!)
    uint256 inv_all;
    mod_inv(inv_all, products[n-1]);

    // Backward pass: compute individual inverses
    for (int i = n - 1; i > 0; i--) {
        mod_mul(z_inv[i], inv_all, products[i-1]);
        mod_mul(inv_all, inv_all, z[i]);
    }
    z_inv[0] = inv_all;
}

/**
 * Batch Jacobian to Affine conversion using batch inversion.
 * Converts multiple Jacobian points to Affine with only one mod_inv.
 *
 * NOTE: scratch must have space for 3*n uint256 values to avoid aliasing.
 * Layout: z_vals[0..n-1], z_inv[0..n-1], products[0..n-1]
 */
__device__ void batch_jacobian_to_affine(
    ECPointAffine* affine,
    const ECPointJacobian* jacobian,
    int n,
    uint256* scratch  // Size 3*n (was 2*n -- increased to fix aliasing)
) {
    uint256* z_vals = scratch;
    uint256* z_inv = scratch + n;
    uint256* products = scratch + 2 * n;  // Separate scratch for prefix products

    // Extract Z coordinates
    for (int i = 0; i < n; i++) {
        z_vals[i] = jacobian[i].Z;
    }

    // Batch invert Z values using separate products array to avoid aliasing
    batch_invert(z_inv, z_vals, n, products);

    // Convert each point
    for (int i = 0; i < n; i++) {
        if (jacobian[i].is_infinity()) {
            affine[i].x.set_zero();
            affine[i].y.set_zero();
            continue;
        }

        uint256 z_inv2, z_inv3;
        mod_sqr(z_inv2, z_inv[i]);
        mod_mul(z_inv3, z_inv2, z_inv[i]);

        mod_mul(affine[i].x, jacobian[i].X, z_inv2);
        mod_mul(affine[i].y, jacobian[i].Y, z_inv3);
    }
}

// =============================================================================
// BATCH PROCESSING KERNELS
// =============================================================================

/**
 * Batch EC multiplication kernel.
 * Each thread computes one public key from a private key.
 */
__global__ void ec_mul_batch_kernel(
    const uint256* __restrict__ private_keys,
    ECPointAffine* __restrict__ public_keys,
    const PrecomputedPoint* __restrict__ table,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    if (table != nullptr) {
        ec_mul_windowed(public_keys[idx], private_keys[idx], table);
    } else {
        ec_mul_simple(public_keys[idx], private_keys[idx]);
    }
}

/**
 * Batch EC multiplication with PARALLEL batch inversion.
 * Processes BATCH_INV_SIZE keys together to share one inversion.
 * OPTIMIZED: All threads participate in batch conversion, not just thread 0.
 *
 * Shared memory budget at BATCH_INV_SIZE=128:
 *   products[128]     = 128*32 =  4096 bytes
 *   z_inv[128]        = 128*32 =  4096 bytes
 *   jac_points[128]   = 128*96 = 12288 bytes
 *   affine_points[128]= 128*64 =  8192 bytes
 *   Total             =          28672 bytes (~28KB, fits in 48KB default)
 *
 * Previous value of 32 yielded only 32 threads/block (~3% occupancy).
 * 128 threads/block gives 4 warps, a reasonable occupancy improvement.
 * 256 would need ~56KB shared memory, exceeding the 48KB default limit.
 */
// NOTE: BATCH_INV_SIZE=128 here vs 256 in puzzle_optimized.cu. The value here
// is constrained by shared memory (128 needs ~28KB, 256 would need ~56KB
// exceeding the 48KB default). puzzle_optimized.cu uses a different layout.
#define BATCH_INV_SIZE 128

/**
 * Parallel batch inversion using cooperative threading.
 * All threads participate to amortize the single modular inversion.
 *
 * Threading model:
 *   Steps 1 & 5: All threads participate (parallel Z extraction and affine conversion)
 *   Steps 2-4: Thread 0 only (prefix product scan, single inversion, back-substitution)
 *
 * The prefix product scan and back-substitution are inherently sequential due to
 * data dependencies in Montgomery's trick. The primary parallelism benefit comes
 * from Step 5 where all threads convert their points simultaneously, amortizing
 * the cost of the single mod_inv across all threads.
 *
 * NOTE: Each thread's original Z value is preserved in jacobian[].Z (read-only),
 * so overwriting products[] during the forward pass does not cause aliasing issues.
 */
__device__ void parallel_batch_jacobian_to_affine(
    ECPointAffine* affine,
    const ECPointJacobian* jacobian,
    int n,
    uint256* products,      // Shared memory: size n
    uint256* z_inv,         // Shared memory: size n
    int thread_idx
) {
    // Step 1: Each thread stores its Z coordinate (parallel)
    if (thread_idx < n) {
        products[thread_idx] = jacobian[thread_idx].Z;
    }
    __syncthreads();

    // Step 2: Thread 0 computes cumulative products (inherently sequential --
    // each products[i] depends on products[i-1])
    if (thread_idx == 0) {
        for (int i = 1; i < n; i++) {
            uint256 temp;
            mod_mul(temp, products[i-1], products[i]);
            products[i] = temp;
        }
    }
    __syncthreads();

    // Step 3: Thread 0 computes the single inversion
    uint256 inv_all;
    if (thread_idx == 0) {
        mod_inv(inv_all, products[n-1]);
        z_inv[n-1] = inv_all;
    }
    __syncthreads();

    // Step 4: Thread 0 computes individual inverses via back-substitution
    // (inherently sequential -- each step depends on the previous running_inv)
    // Original Z values are read from jacobian[i].Z, not from products[]
    if (thread_idx == 0) {
        uint256 running_inv = inv_all;
        for (int i = n - 1; i > 0; i--) {
            // z_inv[i] = running_inv * products[i-1]
            mod_mul(z_inv[i], running_inv, products[i-1]);
            // running_inv = running_inv * original_z[i] (from jacobian, not products)
            mod_mul(running_inv, running_inv, jacobian[i].Z);
        }
        z_inv[0] = running_inv;
    }
    __syncthreads();

    // Step 5: ALL THREADS convert their point in parallel (main optimization!)
    if (thread_idx < n) {
        if (jacobian[thread_idx].is_infinity()) {
            affine[thread_idx].x.set_zero();
            affine[thread_idx].y.set_zero();
        } else {
            uint256 z_inv2, z_inv3;
            mod_sqr(z_inv2, z_inv[thread_idx]);
            mod_mul(z_inv3, z_inv2, z_inv[thread_idx]);

            mod_mul(affine[thread_idx].x, jacobian[thread_idx].X, z_inv2);
            mod_mul(affine[thread_idx].y, jacobian[thread_idx].Y, z_inv3);
        }
    }
}

__global__ void ec_mul_batch_optimized_kernel(
    const uint256* __restrict__ private_keys,
    ECPointAffine* __restrict__ public_keys,
    const PrecomputedPoint* __restrict__ table,
    size_t count
) {
    // Shared memory for batch inversion
    __shared__ uint256 products[BATCH_INV_SIZE];
    __shared__ uint256 z_inv[BATCH_INV_SIZE];
    __shared__ ECPointJacobian jac_points[BATCH_INV_SIZE];
    __shared__ ECPointAffine affine_points[BATCH_INV_SIZE];

    size_t batch_idx = blockIdx.x;
    size_t batch_start = batch_idx * BATCH_INV_SIZE;
    size_t thread_idx = threadIdx.x;

    if (batch_start >= count) return;

    size_t batch_count = min((size_t)BATCH_INV_SIZE, count - batch_start);

    // Phase 1: Each thread in warp computes Jacobian result
    if (thread_idx < batch_count) {
        size_t global_idx = batch_start + thread_idx;

        // Compute scalar multiplication in Jacobian coords
        ECPointJacobian R;
        R.set_infinity();

        ECPointAffine G;
        uint256_load_const(G.x, SECP256K1_GX);
        uint256_load_const(G.y, SECP256K1_GY);

        const uint256& scalar = private_keys[global_idx];

        // Use windowed method if table available
        if (table != nullptr) {
            const int wbits = d_ec_config.window_bits;
            const int tsize = d_ec_config.table_size;
            const int nwin  = d_ec_config.num_windows;
            const uint32_t wmask = (uint32_t)tsize - 1;

            for (int w = nwin - 1; w >= 0; w--) {
                int bit_start = w * wbits;
                int limb_idx  = bit_start / 32;
                int bit_off   = bit_start % 32;

                uint32_t window_val;
                if (bit_off + wbits <= 32) {
                    window_val = (scalar.limbs[limb_idx] >> bit_off) & wmask;
                } else {
                    uint32_t lo = scalar.limbs[limb_idx] >> bit_off;
                    uint32_t hi = (limb_idx + 1 < 8) ? scalar.limbs[limb_idx + 1] : 0;
                    window_val = (lo | (hi << (32 - bit_off))) & wmask;
                }

                if (window_val != 0) {
                    ECPointAffine Q;
                    int table_idx = w * tsize + window_val;
                    Q.x = table[table_idx].x;
                    Q.y = table[table_idx].y;

                    ECPointJacobian temp;
                    ec_add_mixed(temp, R, Q);
                    R = temp;
                }
            }
        } else {
            // Simple double-and-add
            for (int i = 255; i >= 0; i--) {
                ECPointJacobian temp;
                ec_double_jacobian(temp, R);
                R = temp;

                int limb = i / 32;
                int bit = i % 32;

                if ((scalar.limbs[limb] >> bit) & 1) {
                    ec_add_mixed(temp, R, G);
                    R = temp;
                }
            }
        }

        jac_points[thread_idx] = R;
    }

    __syncthreads();

    // Phase 2: PARALLEL batch conversion - all threads participate!
    parallel_batch_jacobian_to_affine(
        affine_points, jac_points, batch_count,
        products, z_inv, thread_idx
    );

    __syncthreads();

    // Phase 3: All threads write their results in parallel
    if (thread_idx < batch_count) {
        public_keys[batch_start + thread_idx] = affine_points[thread_idx];
    }
}

/**
 * Generate precomputed table for windowed multiplication.
 * Each block generates one window's table entries sequentially.
 * Uses Montgomery batch inversion to avoid per-point mod_inv calls.
 *
 * Launch: <<<num_windows, 1>>>  (1 thread per block, inter-window parallelism)
 * Reads config from d_ec_config.
 */
__global__ void generate_precomputed_table_kernel(
    PrecomputedPoint* table
) {
    const int wbits = d_ec_config.window_bits;
    const int tsize = d_ec_config.table_size;
    const int w     = blockIdx.x;   // which window this block generates

    // table[w * tsize + i] = i * 2^(w * wbits) * G

    ECPointAffine G;
    uint256_load_const(G.x, SECP256K1_GX);
    uint256_load_const(G.y, SECP256K1_GY);

    ECPointJacobian base_jac;
    base_jac.X = G.x;
    base_jac.Y = G.y;
    base_jac.Z.set_one();

    // Double (w * wbits) times to get 2^(w*wbits) * G
    int total_doubles = w * wbits;
    for (int d = 0; d < total_doubles; d++) {
        ECPointJacobian temp;
        ec_double_jacobian(temp, base_jac);
        base_jac = temp;
    }

    // Convert base to affine for mixed additions
    ECPointAffine base_pt;
    jacobian_to_affine(base_pt, base_jac);

    // Accumulate all multiples in Jacobian form
    // Local memory arrays (spill to L2 -- acceptable for one-time init)
    ECPointJacobian jac_points[MAX_TSIZE];
    jac_points[0].set_infinity();
    jac_points[1].X = base_pt.x;
    jac_points[1].Y = base_pt.y;
    jac_points[1].Z.set_one();

    for (int i = 2; i < tsize; i++) {
        ec_add_mixed(jac_points[i], jac_points[i-1], base_pt);
    }

    // Montgomery batch inversion of Z coordinates
    // Forward pass: compute running products of Z values
    uint256 products[MAX_TSIZE];
    products[0] = jac_points[1].Z;  // skip infinity at index 0
    for (int i = 1; i < tsize - 1; i++) {
        mod_mul(products[i], products[i-1], jac_points[i+1].Z);
    }

    // Single mod_inv of the final product
    uint256 inv;
    mod_inv(inv, products[tsize - 2]);

    // Backward pass: recover individual Z^-1 values and convert to affine
    for (int i = tsize - 1; i > 1; i--) {
        uint256 z_inv;
        mod_mul(z_inv, inv, products[i - 2]);
        mod_mul(inv, inv, jac_points[i].Z);

        uint256 z_inv_sq;
        mod_sqr(z_inv_sq, z_inv);
        mod_mul(table[w * tsize + i].x, jac_points[i].X, z_inv_sq);
        mod_mul(z_inv_sq, z_inv_sq, z_inv);
        mod_mul(table[w * tsize + i].y, jac_points[i].Y, z_inv_sq);
    }

    // Handle index 1 (inv now holds Z_1^-1)
    {
        uint256 z_inv_sq;
        mod_sqr(z_inv_sq, inv);
        mod_mul(table[w * tsize + 1].x, jac_points[1].X, z_inv_sq);
        mod_mul(z_inv_sq, z_inv_sq, inv);
        mod_mul(table[w * tsize + 1].y, jac_points[1].Y, z_inv_sq);
    }

    // Index 0 is the point at infinity
    table[w * tsize].x.set_zero();
    table[w * tsize].y.set_zero();
}

// =============================================================================
// HOST API
// =============================================================================

extern "C" {

static PrecomputedPoint* g_precomputed_table = nullptr;
static std::once_flag g_table_init_flag;

cudaError_t secp256k1_init_table(cudaStream_t stream, int window_bits_override) {
    cudaError_t err = cudaSuccess;
    std::call_once(g_table_init_flag, [&]() {
        // Determine window bits: use override if positive, else auto-detect from VRAM
        int window_bits;
        if (window_bits_override > 0) {
            window_bits = window_bits_override;
        } else {
            int device;
            cudaGetDevice(&device);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device);
            constexpr size_t GB = 1024ULL * 1024 * 1024;
            size_t reserved = (prop.totalGlobalMem >= 16 * GB) ? 4 * GB
                            : (prop.totalGlobalMem >= 8 * GB)  ? 2 * GB
                            :                                     1 * GB;
            window_bits = select_window_bits(prop.totalGlobalMem - reserved);
        }

        // Build and store config
        g_ec_config = make_ec_config(window_bits);

        fprintf(stderr, "[EC] Precomputed table: %d-bit windows, %d windows, %zu KB\n",
                g_ec_config.window_bits, g_ec_config.num_windows,
                g_ec_config.total_bytes / 1024);

        // Copy config to device constant memory
        err = cudaMemcpyToSymbol(d_ec_config, &g_ec_config, sizeof(ECTableConfig));
        if (err != cudaSuccess) return;

        // Allocate table on device
        err = cudaMalloc(&g_precomputed_table, g_ec_config.total_bytes);
        if (err != cudaSuccess) return;

        // Copy table pointer to device symbol so kernels can access it
        err = cudaMemcpyToSymbol(d_precomputed_table, &g_precomputed_table, sizeof(PrecomputedPoint*));
        if (err != cudaSuccess) return;

        // Generate table: one block per window, sequential within each window
        generate_precomputed_table_kernel<<<g_ec_config.num_windows, 1, 0, stream>>>(
            g_precomputed_table
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) return;

        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) return;

        // OPTIMIZATION: Enable L2 cache persistence for EC precomputed table
        // This keeps the table hot in L2 cache across kernel launches
        #if CUDART_VERSION >= 11040
        {
            // Check if device supports L2 persistence before setting the policy
            int persisting_l2_max = 0;
            cudaDeviceGetAttribute(&persisting_l2_max,
                                   cudaDevAttrMaxPersistingL2CacheSize, 0);
            if (persisting_l2_max > 0) {
                cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)persisting_l2_max);

                // Free edition: EC table gets full L2 persistence budget
                // (Pro edition uses 25% for EC table, 75% for bloom filter)
                size_t persist_bytes = g_ec_config.total_bytes;
                if (persist_bytes > (size_t)persisting_l2_max) {
                    persist_bytes = (size_t)persisting_l2_max;
                }
                cudaStreamAttrValue stream_attr = {};
                stream_attr.accessPolicyWindow.base_ptr  = g_precomputed_table;
                stream_attr.accessPolicyWindow.num_bytes  = persist_bytes;
                stream_attr.accessPolicyWindow.hitRatio   = 1.0f;
                stream_attr.accessPolicyWindow.hitProp    = cudaAccessPropertyPersisting;
                stream_attr.accessPolicyWindow.missProp   = cudaAccessPropertyStreaming;
                err = cudaStreamSetAttribute(stream,
                                             cudaStreamAttributeAccessPolicyWindow,
                                             &stream_attr);
                // Non-fatal if this fails (older GPUs may not support it)
                if (err != cudaSuccess) {
                    err = cudaSuccess;
                }
            }
        }
        #endif
    });

    return err;
}

cudaError_t secp256k1_cleanup() {
    if (g_precomputed_table != nullptr) {
        cudaFree(g_precomputed_table);
        g_precomputed_table = nullptr;
    }
    return cudaSuccess;
}

cudaError_t secp256k1_batch_mul(
    const void* d_private_keys,
    void* d_public_keys,
    size_t count,
    cudaStream_t stream
) {
    if (count == 0) return cudaSuccess;

    // Use optimized kernel with batch inversion
    const int batch_size = BATCH_INV_SIZE;
    const int num_batches = (count + batch_size - 1) / batch_size;

    ec_mul_batch_optimized_kernel<<<num_batches, batch_size, 0, stream>>>(
        reinterpret_cast<const uint256*>(d_private_keys),
        reinterpret_cast<ECPointAffine*>(d_public_keys),
        g_precomputed_table,
        count
    );

    return cudaGetLastError();
}

cudaError_t secp256k1_batch_mul_simple(
    const void* d_private_keys,
    void* d_public_keys,
    size_t count,
    cudaStream_t stream
) {
    if (count == 0) return cudaSuccess;

    const int threads_per_block = 64;  // Lower due to register pressure
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    ec_mul_batch_kernel<<<blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<const uint256*>(d_private_keys),
        reinterpret_cast<ECPointAffine*>(d_public_keys),
        g_precomputed_table,
        count
    );

    return cudaGetLastError();
}

}  // extern "C"

}  // namespace gpu
}  // namespace collider
