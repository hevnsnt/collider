/**
 * Collider GPU secp256k1 Implementation - OPTIMIZED
 *
 * High-performance elliptic curve operations for brain wallet research.
 * Implements all critical optimizations:
 * - Precomputed table for windowed scalar multiplication (16x speedup)
 * - Montgomery arithmetic for field operations
 * - Batch inversion using Montgomery's Trick (85x speedup on inversions)
 * - Optimized Jacobian coordinate arithmetic
 *
 * Target: 2.5B+ scalar multiplications per second per RTX 5090
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Tier 1 perf (F-inv): shared addition-chain mod_inv (see header for
// rationale; also consumed by fused_pipeline.cu against its own
// TU-local uint256 / mod_mul).
#include "secp256k1_field.cuh"

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

// GLV endomorphism constants. secp256k1 has the efficient endomorphism
// lambda * P = (beta * P.x, P.y), allowing decomposition of k into
// k1 + k2*lambda where |k1|, |k2| are roughly sqrt(n). Beta below is the
// cube root of 1 mod p (beta^3 = 1 mod p) used for the point transform.
static __constant__ uint32_t GLV_BETA[8] = {
    0x719501EE, 0xC1396C28, 0x12F58995, 0x9CF04975,
    0xAC3434E9, 0x6E64479E, 0x657C0710, 0x7AE96A2B
};

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
 * PTX-optimized multiply-add with carry using IADD3 instruction.
 * IADD3 is a 3-input adder available on Volta+ (SM 7.0+) that's faster
 * than chained IMAD for carry propagation in multi-precision arithmetic.
 *
 * Computes: result = a * b + c + carry_in, returns carry_out
 */
__device__ __forceinline__ uint32_t mul_add_carry_ptx(
    uint32_t a, uint32_t b, uint32_t c, uint32_t carry_in, uint32_t* result_lo
) {
    uint32_t lo, hi;

    // Use PTX inline assembly for optimal instruction selection
    // mad.lo.cc.u32: multiply a*b, add c, with carry out
    // madc.hi.u32: get high 32 bits with carry in
    asm volatile (
        "{\n\t"
        "  .reg .u32 tmp;\n\t"
        "  mul.lo.u32 %0, %2, %3;\n\t"       // lo = a * b (low 32 bits)
        "  mul.hi.u32 %1, %2, %3;\n\t"       // hi = a * b (high 32 bits)
        "  add.cc.u32 %0, %0, %4;\n\t"       // lo += c with carry
        "  addc.u32 %1, %1, 0;\n\t"          // hi += carry
        "  add.cc.u32 %0, %0, %5;\n\t"       // lo += carry_in with carry
        "  addc.u32 %1, %1, 0;\n\t"          // hi += carry
        "}\n\t"
        : "=r"(lo), "=r"(hi)
        : "r"(a), "r"(b), "r"(c), "r"(carry_in)
    );

    *result_lo = lo;
    return hi;
}

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

        // Propagate final carry
        result[i+8] += carry;
    }
}

/**
 * Fast reduction mod p using secp256k1's special prime form.
 * p = 2^256 - 2^32 - 977 = 2^256 - c where c = 0x1000003D1
 *
 * Tier 2 perf F-modr (v1.4.2 brainwallet-tui-overhaul): replace the
 * while-loop with a single conditional sub. After mod_mul's carry-folding,
 * the input is provably in [0, 2p):
 *
 *   1. lo + hi*c is at most lo + hi*(2^32 + 977) where lo,hi < 2^256, so
 *      < 2^256 + 2^288 + 977 * 2^256 < 2^289.
 *   2. The first extra-fold adds extra*c with extra < 2^33, contributing
 *      at most 2^33 * (2^32+977) < 2^66, leaving the result < 2^256 + 2^66.
 *   3. The second fold (when a carry escapes the top limb) adds carry*c
 *      with carry <= 1 (the only way a carry escapes is if step-2 produced
 *      a value >= 2^256, and the residual after wrapping is < 2^66 + 2^33),
 *      contributing at most c < 2^33.
 *
 *   Final r < 2^256 + 2^66 + 2^33 < 2^257 - 2*(2^32 + 977) = 2p.
 *
 * Therefore one conditional sub is sufficient. Callers MUST pass mod_mul
 * output (or another value already known to be < 2p); raw 256-bit values
 * could be < 4p and would need the older while-loop. Every in-tree caller
 * (mod_mul tail + ec_add_mixed/ec_double/jac_to_affine pipelines) feeds
 * mod_mul-derived values, satisfying the contract.
 *
 * KAT validation: this change is bit-equivalent on the input domain
 * { x : 0 <= x < 2p } and tests/test_secp256k1_inv.cu + tests/test_ec_mul_known_answers.cu
 * exercise that domain through the addition chain, ec_mul, and jac_to_affine.
 */
__device__ __forceinline__ void mod_reduce(uint256& a) {
    uint256 p;
    uint256_load_const(p, SECP256K1_P);

    if (uint256_cmp(a, p) >= 0) {
        uint256_sub(a, a, p);
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
 * Modular negation: result = -a mod p = p - a
 */
__device__ __forceinline__ void mod_neg(uint256& result, const uint256& a) {
    if (a.is_zero()) {
        result.set_zero();
        return;
    }
    uint256 p;
    uint256_load_const(p, SECP256K1_P);
    uint256_sub(result, p, a);
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
    // Result occupies up to 9 limbs (288 bits) plus a possible 1-bit overflow.
    // We MUST capture every carry; dropping any of them produces wrong results
    // for inputs near p (which intermediate values during mod_inv routinely are).
    uint64_t carry = 0;
    uint32_t high_c[10] = {0};  // 10 limbs (320 bits) to safely hold high*c

    // high * 977
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint64_t term = (uint64_t)prod[i+8] * 977ULL + carry;
        high_c[i] = (uint32_t)term;
        carry = term >> 32;
    }
    high_c[8] = (uint32_t)carry;
    // (high_c[9] stays 0 here; high * 977 fits in 9 limbs since 977 < 2^10.)

    // Add high * 2^32 (shift high by 1 limb and add)
    // CRITICAL: capture the carry-out of the i=8 step into high_c[9].
    // For random inputs near p, prod[15] can be close to 2^32-1, so the
    // i=8 addition can overflow and produce a carry that MUST be retained.
    carry = 0;
    #pragma unroll
    for (int i = 1; i < 9; i++) {
        uint64_t term = (uint64_t)high_c[i] + prod[i+7] + carry;
        high_c[i] = (uint32_t)term;
        carry = term >> 32;
    }
    high_c[9] = (uint32_t)carry;  // Bug fix (2026-05-04): was previously dropped.

    // Now add low + high_c[0..7]
    uint256 correction;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        correction.limbs[i] = high_c[i];
    }

    uint32_t c1 = uint256_add(result, low, correction);

    // Handle any overflow from high_c[8..9] and c1.
    // Combine into a 64-bit `extra` representing the overflow above 2^256.
    // We then need to add extra * c (mod p) back into result.
    // extra * c = extra * 2^32 + extra * 977.
    // For random inputs near p, extra fits comfortably in 64 bits:
    //   high_c[9] <= 1, high_c[8] <= ~977 + 1, c1 <= 1, so extra <= ~2^32+979.
    uint64_t extra = ((uint64_t)high_c[9] << 32)
                   + (uint64_t)high_c[8]
                   + (uint64_t)c1;
    if (extra) {
        // Compute extra * c = extra * 2^32 + extra * 977 as up to 96 bits.
        // We need to add that into the low 256 bits of `result` and propagate
        // any carry through ALL eight limbs. Capturing every carry is critical.
        uint64_t e_x_977 = extra * 977ULL;            // up to ~ 2^42
        // bits  0..31 of extra*977 land at result[0]
        // bits 32..63 of extra*977 land at result[1] (added with extra's low32 below)
        uint32_t add0 = (uint32_t)(e_x_977);
        uint32_t add1 = (uint32_t)(e_x_977 >> 32);
        // bits  0..31 of extra (= extra * 2^32 low) land at result[1]
        // bits 32..63 of extra land at result[2]
        uint32_t shf1 = (uint32_t)(extra);
        uint32_t shf2 = (uint32_t)(extra >> 32);

        // 96-bit addend [add0, add1+shf1, shf2] added at limbs [0,1,2]
        uint64_t s0 = (uint64_t)result.limbs[0] + add0;
        result.limbs[0] = (uint32_t)s0;
        uint64_t s1 = (uint64_t)result.limbs[1] + add1 + shf1 + (s0 >> 32);
        result.limbs[1] = (uint32_t)s1;
        uint64_t s2 = (uint64_t)result.limbs[2] + shf2 + (s1 >> 32);
        result.limbs[2] = (uint32_t)s2;
        carry = s2 >> 32;

        // Propagate the remaining carry through the upper limbs.
        #pragma unroll
        for (int i = 3; i < 8; i++) {
            uint64_t s = (uint64_t)result.limbs[i] + carry;
            result.limbs[i] = (uint32_t)s;
            carry = s >> 32;
        }

        // If a carry escapes the top limb, fold it back as one more `c`.
        // (This is rare but must be handled to keep result bounded for the
        // canonical reduction below.)
        if (carry) {
            uint64_t ec = carry * 0x1000003D1ULL;  // c = 2^32 + 977
            uint64_t s = (uint64_t)result.limbs[0] + (uint32_t)ec;
            result.limbs[0] = (uint32_t)s;
            uint64_t hi = (ec >> 32) + (s >> 32);
            for (int i = 1; i < 8; i++) {
                uint64_t s2 = (uint64_t)result.limbs[i] + (uint32_t)hi;
                result.limbs[i] = (uint32_t)s2;
                hi = (hi >> 32) + (s2 >> 32);
                if (hi == 0) break;
            }
        }
    }

    // Final canonical reduction. v1.4.2 G-B6: mod_reduce performs a SINGLE
    // conditional sub. The caller MUST pass `result` already reduced to
    // [0, 2p); the carry-fold logic above (lines ~440-486) is the proof
    // that this mod_mul tail satisfies the precondition. Raw 256-bit
    // values that have not been folded through mod_reduce_512 / the
    // mod_add / mod_sub paths do NOT satisfy the precondition and must
    // not be passed to mod_reduce directly.
    mod_reduce(result);
}

/**
 * Modular squaring (slightly optimized over general multiplication)
 */
__device__ __forceinline__ void mod_sqr(uint256& result, const uint256& a) {
    mod_mul(result, a, a);  // Could optimize further with squaring-specific code
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
 *
 * Tier 1 perf (v1.4.2 brainwallet-tui-overhaul, F-inv): the algorithm
 * body now lives in secp256k1_field.cuh so fused_pipeline.cu can share
 * the addition-chain implementation instead of carrying its own slow
 * binary exponentiation of (p-2). The wrapper here keeps the public
 * symbol stable for tests/test_secp256k1_inv.cu and every existing
 * caller; the template instantiates against THIS TU's uint256 and
 * mod_mul, so behaviour is bit-equal to the prior in-line definition.
 */
__device__ void mod_inv(uint256& r, const uint256& a) {
    mod_inv_addition_chain<uint256>(r, a);
}

// =============================================================================
// ELLIPTIC CURVE POINT OPERATIONS (JACOBIAN COORDINATES)
// =============================================================================

/**
 * Point doubling in Jacobian coordinates.
 * Uses optimized formulas for a=0 (secp256k1).
 * Cost: 1M + 5S + 1*a + 7add + 2*2 + 1*3 + 1*8
 */
// T2.1 (A-tier wave 1, 2026-05-17): R may alias P. The prior code wrote
// R.X (line 522 in the pre-T2.1 file) and R.Y (line 535) before reading
// P.Y and P.Z (line 538) for the Z' computation, which silently
// corrupted the result whenever R == P. ec_double(R, R) is the common
// shape inside scalar mul. Sibling kernel puzzle_optimized.cu::ec_double
// was patched for the same Defect C in May 2026 (snapshot P.X/P.Y/P.Z
// into locals, commit R only after all reads); this is the matching
// fix for the secp256k1.cu copy. The fused_pipeline.cu copy gets the
// same treatment.
__device__ void ec_double_jacobian(ECPointJacobian& R, const ECPointJacobian& P) {
    if (P.is_infinity()) {
        R.set_infinity();
        return;
    }

    // Snapshot every input limb we need before writing anything to R.
    // This guarantees correctness even when R and P are the same object.
    const uint256 PX = P.X;
    const uint256 PY = P.Y;
    const uint256 PZ = P.Z;

    uint256 S, M, T, Y2;

    // Y^2
    mod_sqr(Y2, PY);

    // S = 4 * X * Y^2
    mod_mul(S, PX, Y2);         // X * Y^2
    mod_add(S, S, S);           // 2 * X * Y^2
    mod_add(S, S, S);           // 4 * X * Y^2

    // M = 3 * X^2 (since a = 0 for secp256k1)
    mod_sqr(M, PX);             // X^2
    mod_add(T, M, M);           // 2 * X^2
    mod_add(M, T, M);           // 3 * X^2

    // X' = M^2 - 2*S  (into a local, not R.X)
    uint256 newX;
    mod_sqr(newX, M);           // M^2
    mod_sub(newX, newX, S);     // M^2 - S
    mod_sub(newX, newX, S);     // M^2 - 2*S

    // Y' = M * (S - X') - 8 * Y^4
    uint256 newY;
    mod_sub(T, S, newX);        // S - X'
    mod_mul(T, M, T);           // M * (S - X')

    mod_sqr(Y2, Y2);            // Y^4
    mod_add(Y2, Y2, Y2);        // 2 * Y^4
    mod_add(Y2, Y2, Y2);        // 4 * Y^4
    mod_add(Y2, Y2, Y2);        // 8 * Y^4

    mod_sub(newY, T, Y2);       // M * (S - X') - 8 * Y^4

    // Z' = 2 * Y * Z  (uses snapshotted PY, PZ; safe even if R == P)
    uint256 newZ;
    mod_mul(newZ, PY, PZ);
    mod_add(newZ, newZ, newZ);

    // Commit all outputs together. R may alias P; the snapshots above
    // mean every read above used the original input values.
    R.X = newX;
    R.Y = newY;
    R.Z = newZ;
}

/**
 * Point addition: R = P + Q where Q is affine, P is Jacobian.
 * Mixed addition is more efficient.
 * Cost: 7M + 4S + 9add + 3*2 + 1*3
 *
 * B1 (2026-05-15): the body now lives in
 * secp256k1_field.cuh::ec_add_mixed_shared so this TU and
 * fused_pipeline.cu share one canonical implementation. The
 * fused-pipeline copy was previously missing the H==0 special
 * case and silently produced wrong public keys when triggered;
 * extracting the canonical version into the shared header
 * forces both TUs to honour the case identically. The doubler
 * differs by name across the two TUs (this file:
 * ec_double_jacobian; fused_pipeline.cu: ec_double_jac), so the
 * template takes it as a callable parameter rather than relying
 * on ADL on the bare name.
 *
 * KAT validation: tests/test_secp256k1_inv,
 * tests/test_ec_mul_known_answers, tests/test_gpu_hash160,
 * tests/test_multi_gpu_brain_wallet must all stay green.
 */
__device__ void ec_add_mixed(ECPointJacobian& R, const ECPointJacobian& P, const ECPointAffine& Q) {
    auto doubler = [](ECPointJacobian& Rd, const ECPointJacobian& Pd) {
        ec_double_jacobian(Rd, Pd);
    };
    ec_add_mixed_shared(R, P, Q, doubler);
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

/**
 * 128-bit structure for GLV decomposition results
 */
struct uint128 {
    uint32_t limbs[4];

    __device__ __forceinline__ bool is_zero() const {
        return (limbs[0] | limbs[1] | limbs[2] | limbs[3]) == 0;
    }

    __device__ __forceinline__ int get_bit(int idx) const {
        return (limbs[idx / 32] >> (idx % 32)) & 1;
    }
};

/**
 * GLV scalar decomposition: k = k1 + k2 * lambda (mod n)
 * Splits a 256-bit scalar into two ~128-bit scalars for faster multiplication.
 * Uses the extended Euclidean algorithm basis from libsecp256k1.
 */
// Wave 1 / C-CRIT-3 (2026-05-04): the prior glv_decompose, ec_mul_glv, and
// ec_add_glv_affine were removed.
// Reason: glv_decompose was not a real GLV decomposition. It split k at the
// 128-bit boundary (k1 = low128(k), k2 = high128(k)) instead of solving the
// lattice problem k = k1 + k2*lambda (mod n). Since 2^128 != lambda (mod n),
// the resulting k1*G + k2*(lambda*G) != k*G for almost any k. ec_mul_glv was
// therefore a silent landmine: anyone wiring it up for the "30% speedup"
// would have computed wrong pubkeys.
// Currently no caller uses these (ec_mul_optimized -> ec_mul_windowed). The
// GLV constants in this file remain available for a future correct
// implementation. To re-enable GLV, port libsecp256k1's
// secp256k1_scalar_split_lambda (lattice-reduction-based decomposition).
// glv_endomorphism (point P -> (beta*x, y)) and glv_endomorphism_jacobian
// remain; they are mathematically correct and harmless if invoked.

// =============================================================================
// PRECOMPUTED TABLE FOR SCALAR MULTIPLICATION
// =============================================================================

// Window size for precomputation (w=5 means 32 points per window)
// OPTIMIZED: 5-bit windows reduce main loop iterations from 64 to 52 (18% fewer)
// Trade-off: Table grows from 64KB to 103KB, still fits easily in L2 cache
#define EC_WINDOW_SIZE 5
#define EC_TABLE_SIZE (1 << EC_WINDOW_SIZE)  // 32 points
#define EC_NUM_WINDOWS ((256 + EC_WINDOW_SIZE - 1) / EC_WINDOW_SIZE)  // 52 windows

// Global precomputed table: G, 2G, 3G, ..., 31G, then 32G, 64G, etc.
// Actually, we store: [0, G, 2G, 3G, ..., 31G] for each window
// Table[w][i] = i * 2^(w*5) * G
__device__ PrecomputedPoint* d_precomputed_table;
// NOTE: Removed __constant__ c_precomputed_table - exceeds 64KB constant memory limit
// Using g_precomputed_table (device memory) with L2 cache persistence instead

/**
 * Scalar multiplication using windowed method with precomputed table.
 * Much faster than naive double-and-add.
 *
 * Wave 1 / C-CRIT-1 fix (2026-05-05): the precomputed table layout is
 *   table[w * EC_TABLE_SIZE + v] = v * 2^(w * EC_WINDOW_SIZE) * G
 * (see generate_precomputed_table_kernel above, which doubles the accumulating
 * points EC_WINDOW_SIZE times between windows). The per-window power of two is
 * therefore already baked into every table entry, so the caller just sums
 * table[w][window_val_w] across windows -- NO extra inter-window doubling.
 *
 * The previous version doubled R by EC_WINDOW_SIZE between iterations, which
 * multiplied every already-completed window's contribution by another 2^5
 * each pass. With 52 windows this drove the running point completely off the
 * scalar; the bug was masked by test_ec_mul_known_answers (k=1,2,3,7) because
 * those scalars have only window 0 non-zero, and the overshoot is a no-op when
 * the accumulator is still infinity. Any multi-window scalar (e.g. a SHA256
 * brain-wallet hash) lit up the bug -- producing wrong public keys downstream.
 *
 * The same defect was fixed in fused_pipeline.cu's ec_mul_windowed in commit
 * 81f3f58/5fac796; this is the matching fix for the standalone library copy.
 */
__device__ void ec_mul_windowed(
    ECPointAffine& result,
    const uint256& scalar,
    const PrecomputedPoint* table
) {
    ECPointJacobian R;
    R.set_infinity();

    // Each table entry already carries its window's power of 2; just add them.
    for (int w = 0; w < EC_NUM_WINDOWS; w++) {
        int bit_start = w * EC_WINDOW_SIZE;
        uint32_t window_val = 0;

        #pragma unroll
        for (int i = 0; i < EC_WINDOW_SIZE && (bit_start + i) < 256; i++) {
            int limb = (bit_start + i) / 32;
            int bit = (bit_start + i) % 32;
            window_val |= ((scalar.limbs[limb] >> bit) & 1) << i;
        }

        if (window_val != 0) {
            ECPointAffine Q;
            int table_idx = w * EC_TABLE_SIZE + window_val;
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

// B4 (2026-05-15): the prior `batch_invert(z_inv, z, n, products)`
// helper and its only caller `batch_jacobian_to_affine` were removed.
// The forward pass overwrote products[i] (which was, in the only caller,
// aliased with the input z[]) by storing the cumulative product through
// it; the backward pass then needed the original z[i] to multiply
// `inv_all` against, but read garbage because that slot had already
// been overwritten by the forward pass. The bug manifested as silently
// wrong inverses for n >= 2 whenever the products buffer aliased the
// input, which is exactly how `batch_jacobian_to_affine` invoked it
// (passing `z_vals` for both z and products). No production code path
// reaches either function: the brain-wallet pipeline's batch path uses
// the kangaroo solver's `warp_batch_jacobian_to_affine` in
// kangaroo_kernel.cu (or the per-thread mod_inv inside
// `jacobian_to_affine`). Removing the two latent helpers prevents a
// future caller from re-discovering the aliasing landmine.
// If a future kernel needs CPU-style sequential batch inversion, the
// correct shape is to make a local copy of z[i] before overwriting
// products[i] in the forward pass, OR to require non-aliased buffers
// and document the contract loudly. The structurally better path is
// to lane-cooperate the way kangaroo_kernel.cu's
// `warp_batch_jacobian_to_affine` already does.
//
// T0.3.a (A-tier wave 1, 2026-05-17): `parallel_batch_jacobian_to_affine`
// (the helper this comment used to cite as the model) and its only
// caller `ec_mul_batch_optimized_kernel` were deleted as a same-class
// crypto-correctness landmine. The "optimized" windowed-mul doubled R
// by EC_WINDOW_SIZE between iterations AND added entries from the
// precomputed table that already carry the per-window 2^(w*WINDOW_SIZE)
// factor. Result: 2^(5*(NW-1)) * sum_w win_w * G instead of k*G. The
// matching live windowed path lives in `ec_mul_windowed` above (no
// extra inter-window doubling). The host wrapper that fed the deleted
// kernel (`secp256k1_batch_mul`) is also gone; production callers (v2
// orchestrator multi-address bridge, bench_pipeline stage 2,
// pipeline.cu orchestrator) were all dead-in-current-production paths
// and their dead-branch surface is removed alongside the kernel.
// tests/test_secp256k1_batch_mul_kat.cu (T3.1) is a link-fail trap:
// the test references `secp256k1_batch_mul` so reintroducing that
// symbol without supplying KAT vectors fails the build.

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
 * Generate precomputed table for windowed multiplication.
 * Call once at initialization.
 * OPTIMIZED: 5-bit windows with 32 points per window.
 */
__global__ void generate_precomputed_table_kernel(
    PrecomputedPoint* table
) {
    // This kernel generates the table on GPU
    // table[w * EC_TABLE_SIZE + i] = i * 2^(w*EC_WINDOW_SIZE) * G

    ECPointAffine G;
    uint256_load_const(G.x, SECP256K1_GX);
    uint256_load_const(G.y, SECP256K1_GY);

    // Compute 1G, 2G, 3G, ..., 31G (EC_TABLE_SIZE-1 points)
    ECPointJacobian points[EC_TABLE_SIZE];
    points[0].set_infinity();  // 0 * G

    points[1].X = G.x;
    points[1].Y = G.y;
    points[1].Z.set_one();

    for (int i = 2; i < EC_TABLE_SIZE; i++) {
        ec_add_mixed(points[i], points[i-1], G);
    }

    // Store window 0
    for (int i = 0; i < EC_TABLE_SIZE; i++) {
        jacobian_to_affine(*(ECPointAffine*)&table[i], points[i]);
    }

    // For each subsequent window, multiply by 2^EC_WINDOW_SIZE
    for (int w = 1; w < EC_NUM_WINDOWS; w++) {
        // Double EC_WINDOW_SIZE times
        for (int d = 0; d < EC_WINDOW_SIZE; d++) {
            for (int i = 1; i < EC_TABLE_SIZE; i++) {
                ECPointJacobian temp;
                ec_double_jacobian(temp, points[i]);
                points[i] = temp;
            }
        }

        // Store window w
        for (int i = 0; i < EC_TABLE_SIZE; i++) {
            jacobian_to_affine(*(ECPointAffine*)&table[w * EC_TABLE_SIZE + i], points[i]);
        }
    }
}

// =============================================================================
// HOST API
// =============================================================================

extern "C" {

// Per-GPU precomputed tables (support up to 16 GPUs)
#define MAX_GPU_DEVICES 16
static PrecomputedPoint* g_precomputed_tables[MAX_GPU_DEVICES] = {nullptr};

cudaError_t secp256k1_init_table(cudaStream_t stream) {
    // Get current device ID
    int device_id = 0;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) return err;

    if (device_id >= MAX_GPU_DEVICES) {
        fprintf(stderr, "[EC] Device ID %d exceeds max supported devices (%d)\n", device_id, MAX_GPU_DEVICES);
        return cudaErrorInvalidDevice;
    }

    // Check if already initialized for this device
    if (g_precomputed_tables[device_id] != nullptr) {
        return cudaSuccess;  // Already initialized for this device
    }

    // Allocate table on current device
    size_t table_size = EC_NUM_WINDOWS * EC_TABLE_SIZE * sizeof(PrecomputedPoint);
    err = cudaMalloc(&g_precomputed_tables[device_id], table_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[EC] GPU %d: Failed to allocate precomputed table: %s\n", device_id, cudaGetErrorString(err));
        return err;
    }

    // Generate table on this device
    generate_precomputed_table_kernel<<<1, 1, 0, stream>>>(g_precomputed_tables[device_id]);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[EC] GPU %d: Failed to generate table: %s\n", device_id, cudaGetErrorString(err));
        cudaFree(g_precomputed_tables[device_id]);
        g_precomputed_tables[device_id] = nullptr;
        return err;
    }

    // Wait for generation to complete
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[EC] GPU %d: Table generation sync failed: %s\n", device_id, cudaGetErrorString(err));
        return err;
    }

    // OPTIMIZATION: Enable L2 cache persistence for EC precomputed table
    // This keeps the table hot in L2 cache across kernel launches
    // RTX 5090 has 96MB L2, RTX 4090 has 72MB L2 - table is ~103KB
    #if CUDART_VERSION >= 11040
    cudaStreamAttrValue stream_attr = {};
    stream_attr.accessPolicyWindow.base_ptr = g_precomputed_tables[device_id];
    stream_attr.accessPolicyWindow.num_bytes = table_size;
    stream_attr.accessPolicyWindow.hitRatio = 1.0f;  // Always persist
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    // Non-fatal if this fails (older GPUs may not support it)
    if (err != cudaSuccess) {
        err = cudaSuccess;  // Reset error, not critical
    }
    #endif

    return err;
}

cudaError_t secp256k1_cleanup() {
    // Clean up all device tables
    for (int i = 0; i < MAX_GPU_DEVICES; i++) {
        if (g_precomputed_tables[i] != nullptr) {
            // Need to set device before freeing
            cudaSetDevice(i);
            cudaFree(g_precomputed_tables[i]);
            g_precomputed_tables[i] = nullptr;
        }
    }
    return cudaSuccess;
}

// T0.3.a (A-tier wave 1): `secp256k1_batch_mul` host wrapper deleted
// along with its `ec_mul_batch_optimized_kernel` backend. Production
// callers (v2 orchestrator multi-address bridge, bench_pipeline stage
// 2, pipeline.cu orchestrator) were dead-in-current-production paths;
// the correct sibling `secp256k1_batch_mul_simple` (which dispatches
// `ec_mul_batch_kernel` -> `ec_mul_windowed`, the no-inter-window-
// double path) remains as the single source of truth for batch EC
// multiplication. Bench callers were migrated to the `_simple` API.
// tests/test_secp256k1_batch_mul_kat.cu is a link-fail trap: it
// declares (but does not call) the deleted wrapper, so reintroducing
// the symbol without supplying KAT vectors fails the test build.

cudaError_t secp256k1_batch_mul_simple(
    const void* d_private_keys,
    void* d_public_keys,
    size_t count,
    cudaStream_t stream
) {
    if (count == 0) return cudaSuccess;

    // Get table for current device
    int device_id = 0;
    cudaGetDevice(&device_id);
    PrecomputedPoint* table = (device_id < MAX_GPU_DEVICES) ? g_precomputed_tables[device_id] : nullptr;

    const int threads_per_block = 64;  // Lower due to register pressure
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    ec_mul_batch_kernel<<<blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<const uint256*>(d_private_keys),
        reinterpret_cast<ECPointAffine*>(d_public_keys),
        table,
        count
    );

    return cudaGetLastError();
}

// Get precomputed table pointer for current device (for use by other kernels)
void* secp256k1_get_precomputed_table() {
    int device_id = 0;
    cudaGetDevice(&device_id);
    if (device_id >= MAX_GPU_DEVICES) return nullptr;
    return g_precomputed_tables[device_id];
}

}  // extern "C"

// =============================================================================
// TEST INFRASTRUCTURE
// Wave 0 of the 2026-05-04 review. Permanent test entry points used by
// tests/test_secp256k1_inv.cu and tests/test_ec_table_consistency.cu.
// These exercise __device__ functions (mod_inv, mod_mul, EC table) that
// have no other host-callable surface.
// =============================================================================

// Verify mod_inv(a) is the modular inverse of a, by computing a * mod_inv(a)
// and checking the result reduces to 1 (mod p). Per-thread 1-byte result.
__global__ void test_mod_inv_correctness_kernel(
    const uint256* __restrict__ scalars,
    uint8_t* __restrict__ results,
    size_t count
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint256 a = scalars[idx];

    // Skip a == 0 (no inverse exists). Caller should not pass zero, but be safe.
    if (a.is_zero()) { results[idx] = 0; return; }

    uint256 inv;
    mod_inv(inv, a);

    uint256 product;
    mod_mul(product, a, inv);
    mod_reduce(product);

    bool is_one = (product.limbs[0] == 1);
    #pragma unroll
    for (int i = 1; i < 8; i++) {
        if (product.limbs[i] != 0) is_one = false;
    }

    results[idx] = is_one ? 1 : 0;
}

// Verify each entry in the per-GPU EC table is on the secp256k1 curve.
// Curve: y^2 = x^3 + 7 (mod p). Increments off_curve_count for each bad entry.
__global__ void test_table_on_curve_kernel(
    const PrecomputedPoint* __restrict__ table,
    uint32_t* __restrict__ off_curve_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = EC_NUM_WINDOWS * EC_TABLE_SIZE;
    if (idx >= total) return;

    const PrecomputedPoint& p = table[idx];

    // Skip the implicit identity-encoding entry at i=0 of each window if present
    // (i=0 means 0*G; some impls store as (0,0). Treat (0,0) as on-curve sentinel.)
    bool x_zero = true, y_zero = true;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (p.x.limbs[i] != 0) x_zero = false;
        if (p.y.limbs[i] != 0) y_zero = false;
    }
    if (x_zero && y_zero) return;  // sentinel for identity

    uint256 x_sq, x_cubed, y_sq, rhs;
    mod_sqr(x_sq, p.x);
    mod_mul(x_cubed, x_sq, p.x);

    uint256 seven;
    seven.limbs[0] = 7;
    #pragma unroll
    for (int i = 1; i < 8; i++) seven.limbs[i] = 0;

    mod_add(rhs, x_cubed, seven);
    mod_sqr(y_sq, p.y);

    mod_reduce(rhs);
    mod_reduce(y_sq);

    bool on_curve = true;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        if (y_sq.limbs[i] != rhs.limbs[i]) on_curve = false;
    }

    if (!on_curve) atomicAdd(off_curve_count, 1u);
}

extern "C" {

cudaError_t secp256k1_test_inverse_correctness(
    const void* d_scalars,        // count * 32 bytes
    uint8_t* d_results,           // count bytes (1=ok, 0=wrong)
    size_t count,
    cudaStream_t stream
) {
    if (count == 0) return cudaSuccess;
    const int threads = 64;
    int blocks = (int)((count + threads - 1) / threads);
    test_mod_inv_correctness_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint256*>(d_scalars),
        d_results,
        count
    );
    return cudaGetLastError();
}

cudaError_t secp256k1_test_table_on_curve(
    uint32_t* d_off_curve_count,  // single uint32, must be zeroed by caller
    cudaStream_t stream
) {
    int device_id = 0;
    cudaGetDevice(&device_id);
    if (device_id >= MAX_GPU_DEVICES) return cudaErrorInvalidValue;
    PrecomputedPoint* table = g_precomputed_tables[device_id];
    if (table == nullptr) return cudaErrorInvalidValue;

    int total = EC_NUM_WINDOWS * EC_TABLE_SIZE;
    const int threads = 128;
    int blocks = (total + threads - 1) / threads;
    test_table_on_curve_kernel<<<blocks, threads, 0, stream>>>(table, d_off_curve_count);
    return cudaGetLastError();
}

}  // extern "C" (test infrastructure)

}  // namespace gpu
}  // namespace collider
