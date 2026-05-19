/**
 * glv_decompose.cuh -- Shared GLV scalar decomposition for secp256k1.
 *
 * Decomposes a 256-bit scalar k into (k1, k2) such that:
 *     k = k1 + k2 * lambda (mod n)
 * where lambda is the secp256k1 endomorphism eigenvalue. The magnitudes
 * satisfy |k1|, |k2| < 2^128.5 (Babai bound). Sign flags k1_neg / k2_neg
 * indicate whether the magnitude must be negated in the EC mul.
 *
 * Algorithm: libsecp256k1 secp256k1_scalar_split_lambda
 * (algorithm 3.74 of "Guide to Elliptic Curve Cryptography").
 *
 *     c1 = round((k * g1) / 2^384)
 *     c2 = round((k * g2) / 2^384)
 *     k1 = k - c1 * a1 - c2 * a2  (mod n)
 *     k2 = -c1 * b1 - c2 * b2     (mod n)
 *
 * where (a1, b1, a2, b2) is a short basis of the endomorphism lattice
 * L = { (a, b) : a + b * lambda = 0 (mod n) }, and (g1, g2) are the
 * precomputed rounding constants:
 *     g1 = round(b2 * 2^384 / n)
 *     g2 = round(-b1 * 2^384 / n)
 *
 * Storage convention (libsecp256k1 compatible):
 *   - A1 = 0x3086D221A7D46BCDE86C90E49284EB15           (128-bit, positive)
 *   - B1_NEG = 0xE4437ED6010E88286F547FA90ABFE4C3       (128-bit, |b1|; sign tracked separately)
 *   - A2 = 2^128 + A2_LOW                                (129-bit, positive)
 *     A2_LOW = 0x14CA50F7A8E2F3F657C1108D9D44CFD8
 *   - B2 = A1                                            (128-bit, positive)
 *   - G1, G2 = 256-bit
 *
 * This header is included from multiple translation units. The constants
 * use `static __device__ __constant__` to give each including TU its own
 * private copy (~256 bytes total per TU). The decomposition function uses
 * `__device__ static inline` so each TU gets a private instantiation
 * (modest code-size impact, no linker conflicts).
 *
 * Phase C.1 of v1.4.2 -- extracted from puzzle_optimized.cu after audit.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace collider {
namespace gpu {
namespace glv {

// =============================================================================
// LATTICE CONSTANTS (libsecp256k1 secp256k1_scalar_split_lambda)
// =============================================================================
// All multi-limb values are little-endian: d[0] = LSB.

// a1 = 0x3086D221A7D46BCDE86C90E49284EB15 (128-bit)
static __device__ __constant__ uint64_t A1[2] = {
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
};

// |b1| = 0xE4437ED6010E88286F547FA90ABFE4C3 (128-bit; signed b1 is negative)
static __device__ __constant__ uint64_t B1_NEG[2] = {
    0x6F547FA90ABFE4C3ULL, 0xE4437ED6010E8828ULL
};

// a2 = 0x114CA50F7A8E2F3F657C1108D9D44CFD8 (129-bit, positive).
// a2 = 2^128 + A2_LOW.
static __device__ __constant__ uint64_t A2_LOW[2] = {
    0x57C1108D9D44CFD8ULL, 0x14CA50F7A8E2F3F6ULL
};

// b2 = a1 (same value)
static __device__ __constant__ uint64_t B2[2] = {
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
};

// g1 = round(b2 * 2^384 / n) = round(a1 * 2^384 / n)
//   full hex (big-endian):
//   3086d221a7d46bcde86c90e49284eb153da4445121181820ffa7bd168f1d4808
static __device__ __constant__ uint64_t G1[4] = {
    0xFFA7BD168F1D4808ULL, 0x3DA4445121181820ULL,
    0xE86C90E49284EB15ULL, 0x3086D221A7D46BCDULL
};

// g2 = round((-b1) * 2^384 / n)
//   full hex (big-endian):
//   e4437ed6010e88286f547fa90abfe4c423eb5cdc18462a36f7a70f55b96c7540
static __device__ __constant__ uint64_t G2[4] = {
    0xF7A70F55B96C7540ULL, 0x23EB5CDC18462A36ULL,
    0x6F547FA90ABFE4C4ULL, 0xE4437ED6010E8828ULL
};

// Lambda = secp256k1 endomorphism eigenvalue (256-bit scalar in Z_n)
//   full hex (big-endian):
//   5363AD4CC05C30E0A5261C028812645A122E22EA20816678DF02967C1B23BD72
static __device__ __constant__ uint64_t LAMBDA[4] = {
    0xDF02967C1B23BD72ULL, 0x122E22EA20816678ULL,
    0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL
};

// secp256k1 group order n (256-bit)
//   full hex (big-endian):
//   FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
static __device__ __constant__ uint64_t N[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// =============================================================================
// PRIMITIVES (header-local, naive carry-propagation; no PTX dependency)
// =============================================================================

// 256-bit subtract r = a - b. Returns true on borrow-out.
__device__ static inline bool sub_256(uint64_t r[4],
                                       const uint64_t a[4],
                                       const uint64_t b[4]) {
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t ai = a[i];
        uint64_t bi = b[i];
        uint64_t diff = ai - bi - borrow;
        // borrow out = 1 iff ai < bi (regardless of borrow_in), or
        //              ai == bi and borrow_in == 1 (propagation).
        uint64_t new_borrow = (ai < bi) ? 1ULL
                                        : ((ai == bi) ? borrow : 0ULL);
        r[i] = diff;
        borrow = new_borrow;
    }
    return borrow != 0;
}

// 128 x 128 -> 256-bit multiply. a, b: 2-limb LE. r: 4-limb LE.
__device__ static inline void mul_128x128(const uint64_t a[2],
                                           const uint64_t b[2],
                                           uint64_t r[4]) {
    // Partial products:
    //   p00 = a0 * b0  -> contributes to r[0..1]
    //   p01 = a0 * b1  -> contributes to r[1..2]
    //   p10 = a1 * b0  -> contributes to r[1..2]
    //   p11 = a1 * b1  -> contributes to r[2..3]

    uint64_t p00_lo = a[0] * b[0];
    uint64_t p00_hi = __umul64hi(a[0], b[0]);

    uint64_t p01_lo = a[0] * b[1];
    uint64_t p01_hi = __umul64hi(a[0], b[1]);

    uint64_t p10_lo = a[1] * b[0];
    uint64_t p10_hi = __umul64hi(a[1], b[0]);

    uint64_t p11_lo = a[1] * b[1];
    uint64_t p11_hi = __umul64hi(a[1], b[1]);

    // r[0] = p00_lo
    r[0] = p00_lo;

    // r[1] = p00_hi + p01_lo + p10_lo  (carries propagate to r[2])
    uint64_t s1 = p00_hi + p01_lo;
    uint64_t c1 = (s1 < p00_hi) ? 1ULL : 0ULL;
    s1 += p10_lo;
    uint64_t c2 = (s1 < p10_lo) ? 1ULL : 0ULL;
    r[1] = s1;

    // r[2] = p01_hi + p10_hi + p11_lo + (c1 + c2)  (carries propagate to r[3])
    uint64_t s2 = p01_hi + p10_hi;
    uint64_t c3 = (s2 < p01_hi) ? 1ULL : 0ULL;
    s2 += p11_lo;
    uint64_t c4 = (s2 < p11_lo) ? 1ULL : 0ULL;
    uint64_t carry_in_to_r2 = c1 + c2;  // <= 2, no overflow
    s2 += carry_in_to_r2;
    uint64_t c5 = (s2 < carry_in_to_r2) ? 1ULL : 0ULL;
    r[2] = s2;

    // r[3] = p11_hi + (c3 + c4 + c5)  (<= 3 added, never overflows)
    r[3] = p11_hi + (c3 + c4 + c5);
}

// =============================================================================
// DECOMPOSITION
// =============================================================================

/**
 * Decompose k in Z_n into magnitudes (k1, k2) and sign flags so that
 *     k = +/-k1 + (+/-k2) * lambda  (mod n).
 *
 * @param[in]  k       Scalar (LE, 4 x u64). Caller is responsible for k < n.
 * @param[out] k1      Magnitude of k1 (LE, 4 x u64). k1[3] == 0 always.
 * @param[out] k2      Magnitude of k2 (LE, 4 x u64). k2[3] == 0 always.
 * @param[out] k1_neg  True if the decomposed k1 is negative.
 * @param[out] k2_neg  True if the decomposed k2 is negative.
 *
 * Babai bound: |k1|, |k2| < 2^128.5 ~ 2^128 * sqrt(2). For scalars near n,
 * k1[2] may equal 1 (bit 128 set). k1[3] / k2[3] are always zero.
 */
__device__ static inline void decompose(const uint64_t k[4],
                                         uint64_t k1[4],
                                         uint64_t k2[4],
                                         bool& k1_neg,
                                         bool& k2_neg) {
    // ------------------------------------------------------------------------
    // Step 1: c1 = (k * G1) >> 384, c2 = (k * G2) >> 384
    // The full 256x256 = 512-bit product is computed limb-by-limb. We keep
    // only the top 128 bits (p[6], p[7]), which is the floor of the
    // rounding division x = round(k * g_i / 2^384). For libsecp256k1's
    // chosen g_i (with G_i ~ 2^256 / n exactly), the resulting c_i is the
    // 128-bit Babai rounding multiplier.
    // ------------------------------------------------------------------------

    uint64_t c1[2], c2[2];

    {
        uint64_t p[8] = {0};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t carry = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t lo = k[i] * G1[j];
                uint64_t hi = __umul64hi(k[i], G1[j]);
                lo += p[i + j];
                hi += (lo < p[i + j]) ? 1ULL : 0ULL;
                lo += carry;
                hi += (lo < carry) ? 1ULL : 0ULL;
                p[i + j] = lo;
                carry = hi;
            }
            p[i + 4] = carry;
        }
        c1[0] = p[6];
        c1[1] = p[7];
    }

    {
        uint64_t p[8] = {0};
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            uint64_t carry = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t lo = k[i] * G2[j];
                uint64_t hi = __umul64hi(k[i], G2[j]);
                lo += p[i + j];
                hi += (lo < p[i + j]) ? 1ULL : 0ULL;
                lo += carry;
                hi += (lo < carry) ? 1ULL : 0ULL;
                p[i + j] = lo;
                carry = hi;
            }
            p[i + 4] = carry;
        }
        c2[0] = p[6];
        c2[1] = p[7];
    }

    // ------------------------------------------------------------------------
    // Step 2: k1 = k - c1*A1 - c2*A2
    // A2 is 129 bits (= 2^128 + A2_LOW). Compute c2*A2 as
    //     c2 * A2 = c2 * A2_LOW + (c2 << 128)
    // which is exact modulo 2^256. The final reduction modulo n is implicit:
    // by the Babai bound, |k1| < n, so a single signed wrap captures the
    // correct value -- if the modular subtraction produces a negative result
    // (top bit set), we negate it and flip k1_neg.
    // ------------------------------------------------------------------------

    uint64_t c1_a1[4];
    mul_128x128(c1, A1, c1_a1);

    uint64_t c2_a2[4];
    mul_128x128(c2, A2_LOW, c2_a2);
    {
        // Add c2 at limb offset 2 (i.e., add c2 * 2^128).
        uint64_t s = c2_a2[2] + c2[0];
        uint64_t carry = (s < c2_a2[2]) ? 1ULL : 0ULL;
        c2_a2[2] = s;
        c2_a2[3] = c2_a2[3] + c2[1] + carry;
    }

    uint64_t tmp[4];
    sub_256(tmp, k, c1_a1);
    sub_256(k1, tmp, c2_a2);

    k1_neg = (k1[3] >> 63) != 0;
    if (k1_neg) {
        uint64_t zero[4] = {0, 0, 0, 0};
        sub_256(k1, zero, k1);
    }

    // ------------------------------------------------------------------------
    // Step 3: k2 = -c1*b1 - c2*b2
    //         = c1 * |b1| - c2 * b2     (since b1 is stored as negated)
    //         = c1_b1 - c2_b2           (256-bit signed)
    // If c1_b1 < c2_b2, swap operands and flip sign.
    // ------------------------------------------------------------------------

    uint64_t c1_b1[4], c2_b2[4];
    mul_128x128(c1, B1_NEG, c1_b1);
    mul_128x128(c2, B2, c2_b2);

    bool k2_borrow = sub_256(k2, c1_b1, c2_b2);
    k2_neg = false;
    if (k2_borrow || (k2[3] >> 63)) {
        sub_256(k2, c2_b2, c1_b1);
        k2_neg = true;
    }

    // Babai guarantee: top limb is always zero. The window-mul loop in
    // ec_mul_glv must cover up to bit 128 (33 four-bit windows) since
    // k1[2] / k2[2] may have the low bit set for scalars near n.
    k1[3] = 0;
    k2[3] = 0;
}

}  // namespace glv
}  // namespace gpu
}  // namespace collider
