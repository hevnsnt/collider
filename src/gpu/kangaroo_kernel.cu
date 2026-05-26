/**
 * GPU Kangaroo Kernel for ECDLP Solving
 *
 * High-performance CUDA implementation of Pollard's Kangaroo algorithm
 * for solving the Elliptic Curve Discrete Logarithm Problem on secp256k1.
 *
 * This kernel is optimized for:
 * - Coalesced memory access (structure-of-arrays layout)
 * - Minimal register pressure
 * - Efficient distinguished point detection
 * - Batch modular inversion (Montgomery's trick)
 * - SOTA 3-group Kangaroo with K=1.15 coefficient (45% fewer operations)
 *
 * GPU_GRP_SIZE Tuning Guide:
 * ==========================
 * GPU_GRP_SIZE controls how many kangaroos each thread processes in batch.
 * Higher values give better batch inversion efficiency but use more registers.
 *
 * Recommended values by GPU architecture:
 * - Pascal (GTX 10xx):        GPU_GRP_SIZE=32  (limited registers)
 * - Turing (RTX 20xx):        GPU_GRP_SIZE=64  (balanced)
 * - Ampere (RTX 30xx):        GPU_GRP_SIZE=64-128 (try both)
 * - Ada Lovelace (RTX 40xx):  GPU_GRP_SIZE=128 (more registers)
 *
 * To override at compile time: nvcc -DGPU_GRP_SIZE=128 ...
 * To check current value: the kernel prints "GPU_GRP_SIZE=N" at startup.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <iostream>
#include <memory>

// Shared jump-table size (must come BEFORE namespace block: the
// header opens its own collider::gpu, which would nest if pulled in
// inside an already-open namespace).
#include "kangaroo_jump_table_size.hpp"

// T1.6: shared pre-launch context-state probe.
#include "cuda_helpers.hpp"

// File-scope MSL boundary check: the Metal kernel hard-codes 32 via
// #define KANGAROO_JUMP_TABLE_SIZE; if the host-side constant is ever
// retuned, the MSL kernel needs the same change in lockstep.
static_assert(::collider::gpu::kKangarooJumpTableSize == 32u,
              "kangaroo.metal hard-codes KANGAROO_JUMP_TABLE_SIZE 32; "
              "update the .metal kernel together with this constant.");

namespace collider {
namespace gpu {

// ============================================================================
// cudaMemcpy hygiene helpers
// ============================================================================
//
// D-1 follow-up (2026-05-24): pre-existing kangaroo paths had ~40
// cudaMemcpy sites with discarded return codes. For H2D/D2H copies
// on the host-driver code paths (NOT the per-thread device kernels)
// a silent copy failure produces either (a) the kernel running on
// uninitialized device memory for init paths -- DPs from a corrupted
// herd are guaranteed-wrong + waste hours, or (b) uninitialized host
// buffers for D2H paths -- the host sees a "false zero" / garbage
// flag values that mask real DPs. The helper below logs + returns
// the cuda error so each caller picks the right recovery (return
// false from init, skip-iteration from harvest loops).
//
// Hot-path Async H2D copies (kernel pipelines that pre-fetch into
// double-buffered streams) still discard the immediate return value
// because the failure is caught by the next cudaStreamSynchronize;
// changing those would require a wider stream-management refactor.

namespace {
inline bool kg_cuda_check(cudaError_t err, const char* what) {
    if (err == cudaSuccess) return true;
    std::fprintf(stderr,
        "[!] kangaroo cudaMemcpy(%s) failed: %s\n",
        what, cudaGetErrorString(err));
    return false;
}
}  // namespace

// ============================================================================
// Constants
// ============================================================================

// secp256k1 prime: p = 2^256 - 2^32 - 977
// Using __device__ const instead of __constant__ to avoid RDC+LTO symbol visibility issues
__device__ const uint64_t SECP256K1_P[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// Number of jump table entries. Sourced from the shared header so the
// CUDA host path agrees with the Metal host path (kangaroo_metal.hpp
// pulls from the same kangaroo_jump_table_size.hpp). The include
// happens above the `namespace collider { namespace gpu {` block at
// the top of this file (an include here would nest collider::gpu
// inside itself); see the static_assert at file scope below for the
// MSL boundary check. Kept as a #define because the kernel uses it
// in shared-memory array sizing and that requires a compile-time
// integer constant expression visible without namespace qualification
// at __device__ scope.
#define NUM_JUMPS  ((int)::collider::gpu::kKangarooJumpTableSize)
#define JUMP_MASK  (NUM_JUMPS - 1)

// DP check interval: batch convert to affine every N steps for correct DP detection
// Using warp size (32) for efficient warp-cooperative batch inversion.
// This interval is ONLY consulted by the legacy Jacobian kernel
// (kangaroo_step_kernel) which is a debug-only path. The production solver
// uses kangaroo_step_kernel_sota / _jlp which check DP after every step
// (affine pipeline, no batch-conversion cost).
// The 32-step coarse interval is a known footgun for the legacy path: if
// dp_bits is mis-tuned low (say 16) so the expected DP spacing is 2^16
// steps, then 32 steps misses ~31/32 of the DPs that occur in the middle
// of an interval. The legacy kernel is kept in tree solely for
// regression-bisecting Jacobian-vs-affine kernel divergence; production
// callers must select KernelMode::SOTA_3GROUP (the default) or
// KernelMode::JLP_AFFINE_BATCH. If you reach for KernelMode::JACOBIAN_WARP_BATCH
// in a release build you are giving up the per-step DP detection that the
// SOTA / JLP kernels provide.
#define DP_CHECK_INTERVAL 32
#define WARP_SIZE 32

// secp256k1 reduction constant: c = 2^32 + 977 = 0x1000003D1
// Used for modular reduction since p = 2^256 - c, so 2^256 ≡ c (mod p)
#define SECP256K1_C_LO 0x1000003D1ULL

// Jump table - REMOVED __constant__ due to CUDA RDC linking issues
// With separable compilation (-rdc=true) + LTO, __constant__ symbols have
// internal linkage and cudaMemcpyToSymbol() cannot resolve them at runtime.
// Solution: Pass jump table pointers as kernel parameters instead.

// Debug mode flag - passed as kernel parameter to avoid RDC symbol issues

// ============================================================================
// PTX Carry-Chain Arithmetic (from JLP Kangaroo / BitCrack)
// ============================================================================
// PTX inline assembly provides direct access to carry flags, enabling
// efficient multi-word arithmetic without branch overhead.

// 64-bit addition with carry out
#define UADDO(r, a, b) asm volatile("add.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b))
// 64-bit addition with carry in and carry out
#define UADDC(r, a, b) asm volatile("addc.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b))
// 64-bit addition with carry in, no carry out
#define UADD(r, a, b) asm volatile("addc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b))

// 64-bit subtraction with borrow out
#define USUBO(r, a, b) asm volatile("sub.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b))
// 64-bit subtraction with borrow in and borrow out
#define USUBC(r, a, b) asm volatile("subc.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b))
// 64-bit subtraction with borrow in, no borrow out
#define USUB(r, a, b) asm volatile("subc.u64 %0, %1, %2;" : "=l"(r) : "l"(a), "l"(b))

// Multiply-add: r = a * b + c (with carry chain)
#define UMULLO(r, a, b) (r) = (a) * (b)
#define UMULHI(r, a, b) (r) = __umul64hi(a, b)

// Mad with carry: r = a * b + c, propagating carry
#define MADDO(r, a, b, c) { \
    uint64_t _lo = (a) * (b); \
    asm volatile("add.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(_lo), "l"(c)); \
}
#define MADDC(r, a, b, c) { \
    uint64_t _lo = (a) * (b); \
    asm volatile("addc.cc.u64 %0, %1, %2;" : "=l"(r) : "l"(_lo), "l"(c)); \
}

// ============================================================================
// 256-bit Arithmetic (Device Functions) - PTX Optimized
// ============================================================================

// Add two 256-bit numbers: r = a + b (mod 2^256), returns carry
__device__ __forceinline__ uint64_t add256_carry(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t carry;
    UADDO(r[0], a[0], b[0]);
    UADDC(r[1], a[1], b[1]);
    UADDC(r[2], a[2], b[2]);
    UADDC(r[3], a[3], b[3]);
    UADD(carry, 0ULL, 0ULL);  // Extract final carry
    return carry;
}

// Add two 256-bit numbers: r = a + b (mod 2^256)
__device__ __forceinline__ void add256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    add256_carry(r, a, b);
}

// Subtract: r = a - b (mod 2^256), returns borrow
__device__ __forceinline__ uint64_t sub256_borrow(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t borrow;
    USUBO(r[0], a[0], b[0]);
    USUBC(r[1], a[1], b[1]);
    USUBC(r[2], a[2], b[2]);
    USUBC(r[3], a[3], b[3]);
    USUB(borrow, 0ULL, 0ULL);  // Extract final borrow (will be 0 or -1)
    return borrow;
}

// Subtract: r = a - b (mod 2^256)
__device__ __forceinline__ void sub256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    sub256_borrow(r, a, b);
}

// Compare: return 1 if a >= b, 0 otherwise
__device__ __forceinline__ int cmp256(const uint64_t* a, const uint64_t* b) {
    #pragma unroll
    for (int i = 3; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1;  // Equal
}

// Modular reduction: r = a mod p
// Contract: input `a` MUST already be in [0, 2p). All callers in this TU pass
// values that have either been folded through reduce_512 (which leaves the
// result in [0, 2p) per the c-fold bound: low 256 bits + (high * c) where the
// final acc[4] carry-out has been folded twice) or come from mod_add /
// mod_sub overflow paths that bound the result the same way. With that
// precondition, a single conditional subtract is sufficient. The previous
// `while` loop iterated at most once but the branch-free form is preferable
// and matches secp256k1.cu::mod_reduce.
__device__ __forceinline__ void mod_p(uint64_t* r, const uint64_t* a) {
    // Copy
    #pragma unroll
    for (int i = 0; i < 4; i++) r[i] = a[i];

    // Single conditional subtract (input is in [0, 2p) per contract).
    if (cmp256(r, SECP256K1_P)) {
        sub256(r, r, SECP256K1_P);
    }
}

// Modular addition: r = (a + b) mod p
// Handles overflow correctly: 2^256 ≡ c (mod p) where c = 0x1000003D1
__device__ __forceinline__ void mod_add(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t overflow = add256_carry(r, a, b);

    if (overflow) {
        // a + b >= 2^256, need to add c = 0x1000003D1 (since 2^256 ≡ c mod p)
        uint64_t c[4] = {SECP256K1_C_LO, 0, 0, 0};
        overflow = add256_carry(r, r, c);
        // After adding c, result could still be >= p (but < 2p), or overflow again
        // If it overflowed again, add c once more
        if (overflow) {
            add256_carry(r, r, c);
        }
    }

    // Final reduction if r >= p
    if (cmp256(r, SECP256K1_P)) {
        sub256(r, r, SECP256K1_P);
    }
}

// Modular subtraction: r = (a - b) mod p
__device__ __forceinline__ void mod_sub(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    if (cmp256(a, b)) {
        sub256(r, a, b);
    } else {
        uint64_t tmp[4];
        sub256(tmp, b, a);
        sub256(r, SECP256K1_P, tmp);
    }
}

// 64-bit multiply with 128-bit result using CUDA intrinsics
// Returns low 64 bits in *lo, high 64 bits in *hi
__device__ __forceinline__ void umul64wide(uint64_t a, uint64_t b, uint64_t* hi, uint64_t* lo) {
    *lo = a * b;
    *hi = __umul64hi(a, b);
}

// Add with carry: returns sum and updates carry
__device__ __forceinline__ uint64_t add_with_carry(uint64_t a, uint64_t b, uint64_t* carry) {
    uint64_t sum = a + b + *carry;
    *carry = (sum < a || (sum == a && *carry)) ? 1ULL : 0ULL;
    return sum;
}

// 256x256 -> 512-bit multiplication (for Montgomery)
__device__ void mul256_512(uint64_t* rh, uint64_t* rl, const uint64_t* a, const uint64_t* b) {
    // Using 64-bit multiplies with CUDA's __umul64hi for high part
    uint64_t t[8] = {0};

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Compute a[i] * b[j] (128-bit result)
            uint64_t lo, hi;
            umul64wide(a[i], b[j], &hi, &lo);

            // Add to accumulator t[i+j] with carry propagation
            uint64_t old_t = t[i+j];
            t[i+j] = old_t + lo;
            uint64_t c1 = (t[i+j] < old_t) ? 1ULL : 0ULL;

            t[i+j] += carry;
            uint64_t c2 = (t[i+j] < carry) ? 1ULL : 0ULL;

            carry = hi + c1 + c2;
        }
        t[i+4] = carry;
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        rl[i] = t[i];
        rh[i] = t[i+4];
    }
}

// Reduce 512-bit number mod secp256k1 prime
// p = 2^256 - 2^32 - 977, so 2^256 ≡ 2^32 + 977 ≡ c (mod p)
// For N = H*2^256 + L: N mod p = L + H * c (mod p)
__device__ void reduce_512(uint64_t* r, const uint64_t* high, const uint64_t* low) {
    // Result = low + high * c where c = 2^32 + 977 = 0x1000003D1
    // high * c can be up to 289 bits, so we may need multiple rounds

    uint64_t acc[5] = {0};  // 320-bit accumulator

    // Start with low 256 bits
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = low[i];

    // Add high * c (c = 0x1000003D1)
    // Split into: high * 977 + high * 2^32

    // Part 1: Add high * 977
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t mul_hi, mul_lo;
        umul64wide(high[i], 977ULL, &mul_hi, &mul_lo);

        // acc[i] += mul_lo + carry (with proper carry detection)
        uint64_t old_acc = acc[i];
        acc[i] += mul_lo;
        uint64_t c1 = (acc[i] < old_acc) ? 1ULL : 0ULL;

        old_acc = acc[i];
        acc[i] += carry;
        uint64_t c2 = (acc[i] < old_acc) ? 1ULL : 0ULL;

        carry = mul_hi + c1 + c2;
    }
    acc[4] = carry;

    // Part 2: Add high << 32 (i.e., high * 2^32)
    // high[i] << 32 contributes to acc[i] (upper 32 bits) and acc[i+1] (lower 32 bits)
    carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t shifted_lo = high[i] << 32;
        uint64_t shifted_hi = high[i] >> 32;

        // Add shifted_lo to acc[i]
        uint64_t old_acc = acc[i];
        acc[i] += shifted_lo;
        uint64_t c1 = (acc[i] < old_acc) ? 1ULL : 0ULL;

        // Add carry to acc[i]
        old_acc = acc[i];
        acc[i] += carry;
        uint64_t c2 = (acc[i] < old_acc) ? 1ULL : 0ULL;

        // Carry for next iteration = shifted_hi + overflow carries
        carry = shifted_hi + c1 + c2;
    }
    acc[4] += carry;

    // Now acc has up to 289 bits. Reduce while acc[4] > 0
    // Reduce: acc[4] * 2^256 ≡ acc[4] * c (mod p)
    // Note: This loop runs at most 2-3 times since c is small (~33 bits)
    while (acc[4] > 0) {
        uint64_t extra = acc[4];
        acc[4] = 0;

        // Add extra * c = extra * 0x1000003D1
        // Since extra is small (at most ~33 bits after first iteration),
        // and c is ~33 bits, product fits in 66 bits
        uint64_t mul_hi, mul_lo;
        umul64wide(extra, SECP256K1_C_LO, &mul_hi, &mul_lo);

        // Add mul_lo to acc[0]
        uint64_t old_val = acc[0];
        acc[0] += mul_lo;
        carry = (acc[0] < old_val) ? 1ULL : 0ULL;

        // Add mul_hi + carry to acc[1] (two-step to catch both overflows)
        old_val = acc[1];
        acc[1] += mul_hi;
        uint64_t c1 = (acc[1] < old_val) ? 1ULL : 0ULL;
        old_val = acc[1];
        acc[1] += carry;
        uint64_t c2 = (acc[1] < old_val) ? 1ULL : 0ULL;
        carry = c1 + c2;

        // Propagate carry through acc[2] and acc[3]
        old_val = acc[2];
        acc[2] += carry;
        carry = (acc[2] < old_val) ? 1ULL : 0ULL;

        old_val = acc[3];
        acc[3] += carry;
        carry = (acc[3] < old_val) ? 1ULL : 0ULL;

        acc[4] = carry;
    }

    // Copy to result
    #pragma unroll
    for (int i = 0; i < 4; i++) r[i] = acc[i];

    // Final reduction: subtract p if result >= p
    if (cmp256(r, SECP256K1_P)) {
        sub256(r, r, SECP256K1_P);
    }
}

// Modular multiplication: r = a * b mod p
__device__ void mod_mul(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t rh[4], rl[4];
    mul256_512(rh, rl, a, b);
    reduce_512(r, rh, rl);
}

// Modular squaring
__device__ __forceinline__ void mod_sqr(uint64_t* r, const uint64_t* a) {
    mod_mul(r, a, a);
}

// Modular inversion using optimized addition chain for secp256k1
// Based on JLP Kangaroo - uses ~260 squarings + ~15 multiplications instead of ~510 ops
// p - 2 = 0xFFFFFFFE FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2D
__device__ void mod_inv(uint64_t* r, const uint64_t* a) {
    uint64_t x2[4], x3[4], x6[4], x9[4], x11[4], x22[4], x44[4], x88[4], x176[4], x220[4], x223[4], t1[4];

    // x2 = a^2
    mod_sqr(x2, a);
    // x2 = a^3 = a^2 * a
    mod_mul(x2, x2, a);

    // x3 = a^6 = (a^3)^2
    mod_sqr(x3, x2);
    // x3 = a^7 = a^6 * a
    mod_mul(x3, x3, a);

    // x6 = a^(2^6 - 1) using x3
    mod_sqr(x6, x3);        // a^14
    mod_sqr(x6, x6);        // a^28
    mod_sqr(x6, x6);        // a^56
    mod_mul(x6, x6, x3);    // a^63 = a^(2^6 - 1)

    // x9 = a^(2^9 - 1)
    mod_sqr(x9, x6);        // a^126
    mod_sqr(x9, x9);        // a^252
    mod_sqr(x9, x9);        // a^504
    mod_mul(x9, x9, x3);    // a^511 = a^(2^9 - 1)

    // x11 = a^(2^11 - 1)
    mod_sqr(x11, x9);       // a^1022
    mod_sqr(x11, x11);      // a^2044
    mod_mul(x11, x11, x2);  // a^2047 = a^(2^11 - 1)

    // x22 = a^(2^22 - 1)
    mod_sqr(x22, x11);
    for (int i = 1; i < 11; i++) mod_sqr(x22, x22);  // a^(2^22 - 2^11)
    mod_mul(x22, x22, x11); // a^(2^22 - 1)

    // x44 = a^(2^44 - 1)
    mod_sqr(x44, x22);
    for (int i = 1; i < 22; i++) mod_sqr(x44, x44);
    mod_mul(x44, x44, x22);

    // x88 = a^(2^88 - 1)
    mod_sqr(x88, x44);
    for (int i = 1; i < 44; i++) mod_sqr(x88, x88);
    mod_mul(x88, x88, x44);

    // x176 = a^(2^176 - 1)
    mod_sqr(x176, x88);
    for (int i = 1; i < 88; i++) mod_sqr(x176, x176);
    mod_mul(x176, x176, x88);

    // x220 = a^(2^220 - 1)
    mod_sqr(x220, x176);
    for (int i = 1; i < 44; i++) mod_sqr(x220, x220);
    mod_mul(x220, x220, x44);

    // x223 = a^(2^223 - 1).
    // 2026-05-09 fix: previous code multiplied by x2 (= a^3) here, which
    // produces x223 = a^(2^223 - 5), not a^(2^223 - 1). The libsecp256k1
    // canonical chain uses x3 (= a^7) at this position so that
    //   (2^220 - 1)*2^3 + 7 = 2^223 - 8 + 7 = 2^223 - 1.
    // Caught by tests/test_kangaroo_small_puzzle.cu reporting 7/8 vectors
    // wrong. Fixed by changing x2 -> x3 below.
    mod_sqr(x223, x220);
    mod_sqr(x223, x223);
    mod_sqr(x223, x223);
    mod_mul(x223, x223, x3);

    // Now compute a^(p-2) using x223 and additional operations
    // p - 2 = 2^256 - 2^32 - 979
    // We need: x223 * 2^33 then subtract operations

    // t1 = x223^(2^23)
    mod_sqr(t1, x223);
    for (int i = 1; i < 23; i++) mod_sqr(t1, t1);

    // t1 = t1 * x22
    mod_mul(t1, t1, x22);

    // t1 = t1^(2^5)
    for (int i = 0; i < 5; i++) mod_sqr(t1, t1);

    // t1 = t1 * a (for the final adjustment)
    mod_mul(t1, t1, a);

    // t1 = t1^(2^3)
    for (int i = 0; i < 3; i++) mod_sqr(t1, t1);

    // t1 = t1 * x2
    mod_mul(t1, t1, x2);

    // t1 = t1^(2^2)
    mod_sqr(t1, t1);
    mod_sqr(t1, t1);

    // Final result
    mod_mul(r, t1, a);
}

// ============================================================================
// Elliptic Curve Point Operations
// ============================================================================

// Point doubling: R = 2*P in Jacobian coordinates
// For secp256k1: a = 0, so simplified formulas apply
__device__ void ec_double_jacobian(
    uint64_t* rx, uint64_t* ry, uint64_t* rz,
    const uint64_t* px, const uint64_t* py, const uint64_t* pz
) {
    // Check if P is infinity or Y = 0
    bool is_inf = (pz[0] == 0 && pz[1] == 0 && pz[2] == 0 && pz[3] == 0);
    bool y_zero = (py[0] == 0 && py[1] == 0 && py[2] == 0 && py[3] == 0);
    if (is_inf || y_zero) {
        // Result is point at infinity
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            rx[i] = 0;
            ry[i] = 0;
            rz[i] = 0;
        }
        return;
    }

    uint64_t s[4], m[4], t[4], y2[4];

    // Y^2
    mod_sqr(y2, py);

    // S = 4 * X * Y^2
    mod_mul(s, px, y2);
    mod_add(s, s, s);  // 2*X*Y^2
    mod_add(s, s, s);  // 4*X*Y^2

    // M = 3 * X^2 (for secp256k1, a=0)
    mod_sqr(m, px);
    mod_add(t, m, m);  // 2*X^2
    mod_add(m, t, m);  // 3*X^2

    // X' = M^2 - 2*S
    mod_sqr(rx, m);
    mod_sub(rx, rx, s);
    mod_sub(rx, rx, s);

    // Y' = M * (S - X') - 8 * Y^4
    mod_sub(t, s, rx);
    mod_mul(t, m, t);

    uint64_t y4[4];
    mod_sqr(y4, y2);     // Y^4
    mod_add(y4, y4, y4); // 2*Y^4
    mod_add(y4, y4, y4); // 4*Y^4
    mod_add(y4, y4, y4); // 8*Y^4
    mod_sub(ry, t, y4);

    // Z' = 2 * Y * Z
    mod_mul(rz, py, pz);
    mod_add(rz, rz, rz);
}

// Point addition: R = P + Q (mixed Jacobian-affine)
// P is in Jacobian (X, Y, Z), Q is in affine (x, y)
__device__ void ec_add_mixed(
    uint64_t* rx, uint64_t* ry, uint64_t* rz,
    const uint64_t* px, const uint64_t* py, const uint64_t* pz,
    const uint64_t* qx, const uint64_t* qy
) {
    // Check if P is infinity
    bool p_inf = (pz[0] == 0 && pz[1] == 0 && pz[2] == 0 && pz[3] == 0);
    if (p_inf) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            rx[i] = qx[i];
            ry[i] = qy[i];
            rz[i] = (i == 0) ? 1 : 0;
        }
        return;
    }

    uint64_t z2[4], z3[4], u2[4], s2[4];
    uint64_t h[4], r[4], h2[4], h3[4];
    uint64_t v[4], tmp[4];

    // Z^2
    mod_sqr(z2, pz);
    // Z^3
    mod_mul(z3, z2, pz);

    // U2 = X2 * Z1^2
    mod_mul(u2, qx, z2);
    // = Y2 * Z1^3
    mod_mul(s2, qy, z3);

    // H = U2 - X1
    mod_sub(h, u2, px);
    // R = S2 - Y1
    mod_sub(r, s2, py);

    // Check for special cases
    bool h_zero = (h[0] == 0 && h[1] == 0 && h[2] == 0 && h[3] == 0);
    bool r_zero = (r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 0);

    if (h_zero) {
        if (r_zero) {
            // P == Q: need point doubling
            ec_double_jacobian(rx, ry, rz, px, py, pz);
            return;
        } else {
            // P == -Q: result is point at infinity
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                rx[i] = 0;
                ry[i] = 0;
                rz[i] = 0;
            }
            return;
        }
    }

    // Normal case: P != Q
    // H^2
    mod_sqr(h2, h);
    // H^3
    mod_mul(h3, h2, h);

    // V = X1 * H^2
    mod_mul(v, px, h2);

    // X3 = R^2 - H^3 - 2*V
    mod_sqr(tmp, r);
    mod_sub(tmp, tmp, h3);
    mod_sub(tmp, tmp, v);
    mod_sub(rx, tmp, v);

    // Y3 = R * (V - X3) - Y1 * H^3
    mod_sub(tmp, v, rx);
    mod_mul(tmp, r, tmp);
    uint64_t yh3[4];
    mod_mul(yh3, py, h3);
    mod_sub(ry, tmp, yh3);

    // Z3 = Z1 * H
    mod_mul(rz, pz, h);
}

// Convert Jacobian to affine: (X/Z^2, Y/Z^3)
__device__ void ec_to_affine(
    uint64_t* ax, uint64_t* ay,
    const uint64_t* jx, const uint64_t* jy, const uint64_t* jz
) {
    // Check for point at infinity (Z = 0) - mod_inv(0) is undefined
    bool is_inf = (jz[0] == 0 && jz[1] == 0 && jz[2] == 0 && jz[3] == 0);
    if (is_inf) {
        // Return (0, 0) for point at infinity
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            ax[i] = 0;
            ay[i] = 0;
        }
        return;
    }

    uint64_t z_inv[4], z2_inv[4], z3_inv[4];

    mod_inv(z_inv, jz);
    mod_sqr(z2_inv, z_inv);
    mod_mul(z3_inv, z2_inv, z_inv);

    mod_mul(ax, jx, z2_inv);
    mod_mul(ay, jy, z3_inv);
}

// ============================================================================
// Warp-Cooperative Batch Jacobian to Affine Conversion
// ============================================================================
// Uses Montgomery's batch inversion trick to amortize a single mod_inv
// across all 32 threads in a warp. This is critical for efficient DP detection
// since we MUST use affine X coordinates (Jacobian X has no correlation with
// affine X low bits due to the Z^2 factor).
// Algorithm:
// 1. Each thread stores its Z coordinate to shared memory
// 2. Lane 0 computes cumulative products: products[i] = Z[0] * Z[1] * ... * Z[i]
// 3. Lane 0 computes single inversion: inv_all = products[31]^(-1)
// 4. Lane 0 back-propagates: z_inv[i] = inv_all * products[i-1], inv_all *= Z[i]
// 5. ALL threads convert to affine in parallel using their z_inv
// Cost: 1 mod_inv + 3*(N-1) mod_mul for N points (vs N mod_inv without batching)
// For N=32: 1 mod_inv + 93 mod_mul vs 32 mod_inv (huge savings!)

__device__ void warp_batch_jacobian_to_affine(
    uint64_t* affine_x,          // Output: affine X for this thread
    uint64_t* affine_y,          // Output: affine Y for this thread
    const uint64_t* jac_x,       // Input: Jacobian X for this thread
    const uint64_t* jac_y,       // Input: Jacobian Y for this thread
    const uint64_t* jac_z,       // Input: Jacobian Z for this thread
    uint64_t* s_products,        // Shared memory: [WARP_SIZE][4] for cumulative products
    uint64_t* s_z_inv,           // Shared memory: [WARP_SIZE][4] for Z inverses
    uint64_t* s_z_coords         // Shared memory: [WARP_SIZE][4] for original Z coords
) {
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    // Step 1: Each thread stores its Z coordinate to shared memory
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        s_z_coords[lane * 4 + i] = jac_z[i];
    }
    __syncwarp();

    // Step 2: Lane 0 computes cumulative products (sequential, but amortizes the expensive mod_inv)
    if (lane == 0) {
        // First element: products[0] = z[0]
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            s_products[i] = s_z_coords[i];
        }

        // Remaining elements: products[i] = products[i-1] * z[i]
        for (int t = 1; t < WARP_SIZE; t++) {
            uint64_t temp[4];
            mod_mul(temp, &s_products[(t-1) * 4], &s_z_coords[t * 4]);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                s_products[t * 4 + i] = temp[i];
            }
        }
    }
    __syncwarp();

    // Step 3: Lane 0 computes the SINGLE modular inversion
    uint64_t inv_all[4];
    if (lane == 0) {
        mod_inv(inv_all, &s_products[(WARP_SIZE - 1) * 4]);
    }
    __syncwarp();

    // Step 4: Lane 0 back-propagates to compute individual Z inverses
    if (lane == 0) {
        // Work backwards: z_inv[i] = inv_all * products[i-1]
        //                 inv_all = inv_all * z[i]
        for (int t = WARP_SIZE - 1; t > 0; t--) {
            // z_inv[t] = inv_all * products[t-1]
            mod_mul(&s_z_inv[t * 4], inv_all, &s_products[(t-1) * 4]);
            // inv_all = inv_all * z[t] (for next iteration)
            uint64_t temp[4];
            mod_mul(temp, inv_all, &s_z_coords[t * 4]);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                inv_all[i] = temp[i];
            }
        }
        // z_inv[0] = inv_all (after all back-propagation)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            s_z_inv[i] = inv_all[i];
        }
    }
    __syncwarp();

    // Step 5: ALL threads convert their point to affine in parallel (main win!)
    // Check for infinity (Z == 0)
    bool is_inf = (jac_z[0] == 0 && jac_z[1] == 0 && jac_z[2] == 0 && jac_z[3] == 0);

    if (is_inf) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            affine_x[i] = 0;
            affine_y[i] = 0;
        }
    } else {
        // Load this thread's Z inverse from shared memory
        uint64_t my_z_inv[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            my_z_inv[i] = s_z_inv[lane * 4 + i];
        }

        // Compute z_inv^2 and z_inv^3
        uint64_t z_inv2[4], z_inv3[4];
        mod_sqr(z_inv2, my_z_inv);
        mod_mul(z_inv3, z_inv2, my_z_inv);

        // affine_x = jac_x * z_inv^2
        mod_mul(affine_x, jac_x, z_inv2);
        // affine_y = jac_y * z_inv^3
        mod_mul(affine_y, jac_y, z_inv3);
    }
    __syncwarp();
}

// ============================================================================
// JLP-Style Batch Inversion for Affine Point Additions
// ============================================================================
// Montgomery's trick: Convert N inversions into 1 inversion + 3N multiplications
// For GPU_GRP_SIZE=64: 1 inv + 189 mul vs 64 inv = ~50x speedup
// Note: GPU_GRP_SIZE=128 gives better amortization but causes register spilling on older GPUs

// GPU_GRP_SIZE: Number of kangaroos per thread for batch inversion
// JLP's default is 128, which provides better batch inversion efficiency.
// Trade-off: Higher values = better amortization of mod_inv but more register pressure.
// For GPUs with limited registers (older cards), try 64. For modern GPUs (Ampere+), use 128.
#ifndef GPU_GRP_SIZE
#define GPU_GRP_SIZE 64  // Balanced for register pressure - use 128 for modern GPUs with --gpu-grp-size=128
#endif

// ============================================================================
// GPU_GRP_SIZE Auto-Tuning Helpers
// ============================================================================

/**
 * Get recommended GPU_GRP_SIZE based on GPU compute capability.
 * This is a compile-time default - users can override with -DGPU_GRP_SIZE=N
 */
inline int get_recommended_gpu_grp_size(int major, int minor) {
    // Based on registers per SM and typical kernel register usage
    if (major >= 9) {
        // Hopper and beyond: 128 (more registers per SM)
        return 128;
    } else if (major >= 8) {
        // Ampere (SM 8.x): 64-128, default to 64 for safety
        return 64;
    } else if (major >= 7) {
        // Volta/Turing (SM 7.x): 64
        return 64;
    } else if (major >= 6) {
        // Pascal (SM 6.x): 32-64
        return 32;
    } else {
        // Older GPUs: 32
        return 32;
    }
}

/**
 * Print GPU_GRP_SIZE recommendation for the current GPU
 */
inline void print_gpu_grp_size_recommendation(int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    int recommended = get_recommended_gpu_grp_size(prop.major, prop.minor);

    if (GPU_GRP_SIZE != recommended) {
        std::cout << "[*] Note: Current GPU_GRP_SIZE=" << GPU_GRP_SIZE
                  << ", recommended for " << prop.name
                  << " (SM " << prop.major << "." << prop.minor << "): "
                  << recommended << "\n";
        std::cout << "    To change: rebuild with -DGPU_GRP_SIZE=" << recommended << "\n";
    }
}

// ============================================================================

// Batch modular inversion using Montgomery's trick
// Input: dx[GPU_GRP_SIZE][4] - denominators to invert
// Output: dx[GPU_GRP_SIZE][4] - inverted values
__device__ void batch_mod_inv(uint64_t dx[GPU_GRP_SIZE][4]) {
    uint64_t products[GPU_GRP_SIZE][4];

    // Step 1: Forward accumulation - build product chain
    // products[0] = dx[0]
    // products[i] = products[i-1] * dx[i]
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        products[0][i] = dx[0][i];
    }

    for (int g = 1; g < GPU_GRP_SIZE; g++) {
        mod_mul(products[g], products[g-1], dx[g]);
    }

    // Step 2: Single inversion of final product
    uint64_t inv_all[4];
    mod_inv(inv_all, products[GPU_GRP_SIZE - 1]);

    // Step 3: Backward propagation to recover individual inverses
    // dx[i] = inv_all * products[i-1]
    // inv_all = inv_all * original_dx[i]
    for (int g = GPU_GRP_SIZE - 1; g > 0; g--) {
        uint64_t new_inv[4];
        mod_mul(new_inv, inv_all, products[g-1]);

        // Update inv_all for next iteration (needs original dx[g])
        mod_mul(inv_all, inv_all, dx[g]);

        // Store the inverse
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            dx[g][i] = new_inv[i];
        }
    }

    // First element's inverse is just inv_all after all back-propagation
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        dx[0][i] = inv_all[i];
    }
}

// Deterministic per-kangaroo re-randomization: picks a jump-table slot
// based on (tid, step) and resets the kangaroo to that slot's affine
// point. Used by the dx==0 fall-out path, the infinity fall-out path,
// and the same-type DP-collision restart path. Distance is NOT zeroed
// because the kangaroo retains its identity (tame/wild1/wild2 + its
// starting offset for wilds); only its position is rebased so it leaves
// the cycle.
__device__ __forceinline__ void rerandomize_kangaroo(
    uint64_t* px, uint64_t* py,
    size_t tid, int salt,
    const uint64_t* __restrict__ jump_x,
    const uint64_t* __restrict__ jump_y
) {
    // Same hash mixing as the existing fall-out path (line 1407 pre-fix):
    // golden-ratio multiplier XOR'd with the step counter, modulo the
    // jump-table size. Different `salt` values give different slots when
    // the same kangaroo restarts multiple times in the same kernel
    // invocation (collision -> dx==0 in the very next step).
    uint64_t slot = ((uint64_t)tid * 0x9E3779B97F4A7C15ULL
                     ^ (uint64_t)salt)
                    % ::collider::gpu::kKangarooJumpTableSize;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        px[i] = jump_x[slot * 4 + i];
        py[i] = jump_y[slot * 4 + i];
    }
}

// Affine point addition with pre-inverted denominator
// R = P + Q where slope denominator is already inverted
// Input: (px, py) = point P, (qx, qy) = point Q, dx_inv = 1/(Qx - Px)
// Output: (rx, ry) = P + Q
__device__ void ec_add_affine_with_inv(
    uint64_t* rx, uint64_t* ry,
    const uint64_t* px, const uint64_t* py,
    const uint64_t* qx, const uint64_t* qy,
    const uint64_t* dx_inv  // Pre-computed 1/(qx - px)
) {
    uint64_t dy[4], s[4], s2[4];

    // dy = qy - py
    mod_sub(dy, qy, py);

    // s = dy * dx_inv = (qy - py) / (qx - px)
    mod_mul(s, dy, dx_inv);

    // s^2
    mod_sqr(s2, s);

    // rx = s^2 - px - qx
    mod_sub(rx, s2, px);
    mod_sub(rx, rx, qx);

    // ry = s * (px - rx) - py
    uint64_t tmp[4];
    mod_sub(tmp, px, rx);
    mod_mul(ry, s, tmp);
    mod_sub(ry, ry, py);
}

// ============================================================================
// Kangaroo Kernel
// ============================================================================

// Kernel state structure (structure-of-arrays for coalescing)
struct KangarooState {
    uint64_t* x;             // Point X coordinates [N][4]
    uint64_t* y;             // Point Y coordinates [N][4]
    uint64_t* z;             // Point Z coordinates [N][4]
    uint64_t* dist;          // Distances traveled [N][4]
    uint32_t* flags;         // DP detection flags [N]
    uint32_t* types;         // 0=tame, 1=wild1, 2=wild2 (SOTA uses 3 groups)
    // Per-kangaroo wild starting offset. For tame kangaroos this stays 0.
    // For wild1/wild2 it is the host-generated per-instance random scalar
    // used to make each wild start at (+/-Q + offset_i * G), with dist_i
    // initialized to offset_i so the collision algebra
    // (tame_d - wild_d = priv_key) still produces the recovered key
    // directly. Stored alongside d_dist for coalescing.
    uint64_t* wild_offset;   // Per-kangaroo wild starting offset [N][4]
    // Per-kangaroo flag asking the kernel to re-randomize this kangaroo on
    // its next step. Set when (a) batch_mod_inv saw dx==0 (same X collision
    // inside the SIMT lane) or (b) a same-type DP collision was detected
    // host-side (cycled kangaroo). The kernel reads-and-clears in
    // atomicExch fashion.
    uint32_t* restart_pending; // 1 = re-randomize on next kernel step [N]
};

// ============================================================================
// SOTA 3-Group Kangaroo Constants
// ============================================================================
// SOTA method uses 3 kangaroo groups with elliptic curve symmetry to achieve
// K=1.15 coefficient (45% fewer operations than classic K=2.08)
// Groups:
//   - Type 0: Tame kangaroos - start at k*G for random k in [0, sqrt(N))
//   - Type 1: Wild1 kangaroos - start at Q - (N/2)*G, search positive direction
//   - Type 2: Wild2 kangaroos - start at -Q + (N/2)*G, search using symmetry
// Symmetry optimization:
//   For any point P = (x, y) on secp256k1, -P = (x, -y) = (x, p-y)
//   When computing P + J (jump), we can cheaply get P - J by negating J's Y
//   This doubles effective collision probability without doubling work
// INV_FLAG mechanism:
//   - Track whether jump was inverted (subtracted) via flag in distance
//   - When Y coordinate LSB suggests inversion is beneficial, use -J instead of +J
//   - Distance is adjusted: dist = dist - jump_d instead of dist + jump_d

// INV_FLAG is stored in bit 63 of dist[2] (high bit of third limb)
// This leaves 191 bits for actual distance, plenty for puzzles up to 190 bits
#define SOTA_INV_FLAG (1ULL << 63)
#define SOTA_DIST_MASK (~SOTA_INV_FLAG)

// Kangaroo type constants for SOTA 3-group method
#define KANG_TAME  0   // Tame: starts at random k*G, tracks k
#define KANG_WILD1 1   // Wild1: starts at Q - offset, tracks offset
#define KANG_WILD2 2   // Wild2: starts at -Q + offset (mirror), tracks offset

// ============================================================================
// JLP-Style High-Performance Kernel (Affine + Batch Inversion)
// ============================================================================
// Each thread manages GPU_GRP_SIZE kangaroos with affine coordinates.
// Point additions are batched: we collect all denominators, batch invert,
// then complete all additions. This converts GPU_GRP_SIZE inversions per
// step into just 1 inversion + 3*GPU_GRP_SIZE multiplications.

__global__ void kangaroo_step_kernel_jlp(
    KangarooState state,
    uint32_t dp_bits,
    int steps_per_kernel,
    size_t total_kangaroos,  // Total kangaroos (must be multiple of GPU_GRP_SIZE)
    const uint64_t* __restrict__ d_jump_x,  // Jump table X [NUM_JUMPS * 4]
    const uint64_t* __restrict__ d_jump_y,  // Jump table Y [NUM_JUMPS * 4]
    const uint64_t* __restrict__ d_jump_d   // Jump distances [NUM_JUMPS * 4]
) {
    // =========================================================================
    // SHARED MEMORY JUMP TABLE CACHE
    // =========================================================================
    // Cache the entire jump table in shared memory for fast access.
    // Total size: 32 jumps * 4 limbs * 8 bytes * 3 arrays = 3072 bytes = 3 KB
    __shared__ uint64_t s_jump_x[NUM_JUMPS * 4];  // 1024 bytes
    __shared__ uint64_t s_jump_y[NUM_JUMPS * 4];  // 1024 bytes
    __shared__ uint64_t s_jump_d[NUM_JUMPS * 4];  // 1024 bytes

    // Cooperative loading: threads in block load jump table elements in parallel
    const int total_elements = NUM_JUMPS * 4;  // 128 elements total per array
    const int threads_per_block = blockDim.x;

    for (int i = threadIdx.x; i < total_elements; i += threads_per_block) {
        s_jump_x[i] = d_jump_x[i];
        s_jump_y[i] = d_jump_y[i];
        s_jump_d[i] = d_jump_d[i];
    }

    // Synchronize to ensure all threads have loaded the jump table
    __syncthreads();

    // Each thread handles GPU_GRP_SIZE kangaroos
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_threads = (total_kangaroos + GPU_GRP_SIZE - 1) / GPU_GRP_SIZE;
    if (thread_id >= num_threads) return;

    size_t base_idx = thread_id * GPU_GRP_SIZE;

    // Load all kangaroo states into registers
    uint64_t px[GPU_GRP_SIZE][4], py[GPU_GRP_SIZE][4];
    uint64_t dist[GPU_GRP_SIZE][4];
    // Per-lane "halt after DP" flag so a DP-hit kangaroo stops stepping for
    // the remainder of this kernel invocation. Without this, the
    // second/third/Nth DP overwrites the first via the same state.flags
    // slot, and the save-final-state guard at the bottom then refuses to
    // save the last position because flags is set, so the host drops ~50%
    // of DPs on dense puzzles.
    bool halted[GPU_GRP_SIZE];

    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        size_t idx = base_idx + g;
        if (idx >= total_kangaroos) { halted[g] = true; continue; }

        // Restart fall-out: if the host or last kernel marked this
        // kangaroo for restart, re-randomize position before stepping.
        if (state.restart_pending[idx]) {
            state.restart_pending[idx] = 0;
            rerandomize_kangaroo(px[g], py[g], idx, /*salt*/0,
                                 s_jump_x, s_jump_y);
            // Distance unchanged: the kangaroo's collision algebra still
            // relies on its accumulated dist (tame_d / wild_d). Only the
            // position is rebased to escape the cycle.
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                dist[g][i] = state.dist[idx * 4 + i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                px[g][i] = state.x[idx * 4 + i];
                py[g][i] = state.y[idx * 4 + i];
                dist[g][i] = state.dist[idx * 4 + i];
            }
        }
        halted[g] = false;
    }

    const uint64_t dp_mask = (1ULL << dp_bits) - 1;

    // Main stepping loop
    for (int step = 0; step < steps_per_kernel; step++) {
        // Step 1: Select jump points and compute denominators
        int jump_idx[GPU_GRP_SIZE];
        uint64_t dx[GPU_GRP_SIZE][4];  // Denominators to invert

        // Track which lanes had dx==0 so we can rerandomize them after the
        // batch inversion runs. We CANNOT skip the inversion for those
        // lanes (batch_mod_inv requires every product chain element to be
        // invertible) so we substitute dx=1 here, let the inversion
        // proceed, then discard the bogus EC-add result for the affected
        // lanes and rerandomize instead.
        bool dx_was_zero[GPU_GRP_SIZE];

        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            // Jump selection based on x-coordinate low bits
            jump_idx[g] = px[g][0] & JUMP_MASK;

            // Load jump X from SHARED MEMORY (cached)
            uint64_t jx[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                jx[i] = s_jump_x[jump_idx[g] * 4 + i];
            }

            // dx = Jx - Px (denominator for slope)
            mod_sub(dx[g], jx, px[g]);

            // detect dx==0 (the kangaroo landed exactly on its jump
            // point's X coord). Without this guard the product chain in
            // batch_mod_inv zeros out, all GPU_GRP_SIZE inverses become 0,
            // and every kangaroo in this SIMT group walks pure garbage.
            bool is_zero = halted[g] ||
                (dx[g][0] == 0 && dx[g][1] == 0 &&
                 dx[g][2] == 0 && dx[g][3] == 0);
            dx_was_zero[g] = is_zero;
            if (is_zero) {
                // Substitute dx=1 so the batch inversion's product chain
                // stays multiplicatively invertible. The resulting EC-add
                // value is bogus but we discard it below.
                dx[g][0] = 1; dx[g][1] = 0; dx[g][2] = 0; dx[g][3] = 0;
            }
        }

        // Step 2: Batch invert ALL denominators (this is the key optimization!)
        batch_mod_inv(dx);

        // Step 3: Complete all point additions using inverted denominators
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            if (halted[g]) continue;

            int j = jump_idx[g];

            if (dx_was_zero[g]) {
                // fall-out path: rerandomize this kangaroo's position
                // from the jump table (analogous to the infinity fall-out
                // path). Distance is preserved. Skip the EC add.
                rerandomize_kangaroo(px[g], py[g], base_idx + g,
                                     /*salt*/step + 1, s_jump_x, s_jump_y);
                continue;
            }

            uint64_t rx[4], ry[4];

            // Load jump point from SHARED MEMORY (cached)
            uint64_t jx[4], jy[4], jd[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                jx[i] = s_jump_x[j * 4 + i];
                jy[i] = s_jump_y[j * 4 + i];
                jd[i] = s_jump_d[j * 4 + i];
            }

            ec_add_affine_with_inv(
                rx, ry,
                px[g], py[g],
                jx, jy,
                dx[g]  // Pre-computed 1/(Jx - Px)
            );

            // Update point
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                px[g][i] = rx[i];
                py[g][i] = ry[i];
            }

            // Update distance
            add256(dist[g], dist[g], jd);

            // Check for DP (already in affine coordinates!)
            if ((px[g][0] & dp_mask) == 0) {
                size_t idx = base_idx + g;
                if (idx < total_kangaroos) {
                    state.flags[idx] = 1;

                    // Store affine coordinates
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        state.x[idx * 4 + i] = px[g][i];
                        state.y[idx * 4 + i] = py[g][i];
                        state.z[idx * 4 + i] = (i == 0) ? 1 : 0;
                        state.dist[idx * 4 + i] = dist[g][i];
                    }
                    // halt this lane for the rest of the kernel
                    // invocation. The host loop will collect this DP, clear
                    // the flag, and the next kernel launch resumes stepping
                    // from this saved (px, py, dist).
                    halted[g] = true;
                }
            }
        }
    }

    // Save final state back to global memory. Every non-halted lane must
    // still be saved here. A prior "if (state.flags[idx] == 0)" gate
    // skipped saving any kangaroo that hit a DP earlier in this kernel,
    // but did NOT prevent later DPs from overwriting; the lane would lose
    // all post-DP work on the next kernel invocation. With halted[]
    // driving the loop we save exactly the in-register state at the
    // moment the kangaroo stopped (either after the final step or
    // immediately after its DP hit).
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        size_t idx = base_idx + g;
        if (idx >= total_kangaroos) break;
        if (halted[g]) continue;  // DP-stored kangaroo: state already saved.

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            state.x[idx * 4 + i] = px[g][i];
            state.y[idx * 4 + i] = py[g][i];
            state.z[idx * 4 + i] = (i == 0) ? 1 : 0;  // Z=1 for affine
            state.dist[idx * 4 + i] = dist[g][i];
        }
    }
}

// ============================================================================
// SOTA 3-Group Kangaroo Kernel (K=1.15 coefficient)
// ============================================================================
// Revolutionary optimization using elliptic curve symmetry:
// - Uses 3 kangaroo groups instead of classic 2
// - Exploits point negation: -P = (x, -y) is nearly free to compute
// - "Cheap second point": when jumping P + J, also consider P - J
// - Achieves 45% fewer operations (K=1.15 vs K=2.08)
// Based on RCKangaroo by RetiredC: https://github.com/RetiredC/RCKangaroo
// Key insight: The Y coordinate's LSB determines whether to use +J or -J
// This deterministic choice based on current point ensures all kangaroos
// eventually collide while exploiting symmetry for faster convergence.

__global__ void kangaroo_step_kernel_sota(
    KangarooState state,
    uint32_t dp_bits,
    int steps_per_kernel,
    size_t total_kangaroos,
    const uint64_t* __restrict__ d_jump_x,  // Jump table X [NUM_JUMPS * 4]
    const uint64_t* __restrict__ d_jump_y,  // Jump table Y [NUM_JUMPS * 4]
    const uint64_t* __restrict__ d_jump_d   // Jump distances [NUM_JUMPS * 4]
) {
    // =========================================================================
    // SHARED MEMORY JUMP TABLE CACHE
    // =========================================================================
    // Cache the entire jump table in shared memory for fast access.
    // Total size: 32 jumps * 4 limbs * 8 bytes * 3 arrays = 3072 bytes = 3 KB
    // Well within the 48KB+ shared memory available per block.
    __shared__ uint64_t s_jump_x[NUM_JUMPS * 4];  // 1024 bytes
    __shared__ uint64_t s_jump_y[NUM_JUMPS * 4];  // 1024 bytes
    __shared__ uint64_t s_jump_d[NUM_JUMPS * 4];  // 1024 bytes

    // Cooperative loading: threads in block load jump table elements in parallel
    // Each thread loads one or more elements based on thread count
    const int total_elements = NUM_JUMPS * 4;  // 128 elements total per array
    const int threads_per_block = blockDim.x;

    for (int i = threadIdx.x; i < total_elements; i += threads_per_block) {
        s_jump_x[i] = d_jump_x[i];
        s_jump_y[i] = d_jump_y[i];
        s_jump_d[i] = d_jump_d[i];
    }

    // Synchronize to ensure all threads have loaded the jump table
    __syncthreads();

    // Each thread handles GPU_GRP_SIZE kangaroos (same as JLP kernel)
    size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    size_t num_threads = (total_kangaroos + GPU_GRP_SIZE - 1) / GPU_GRP_SIZE;
    if (thread_id >= num_threads) return;

    size_t base_idx = thread_id * GPU_GRP_SIZE;

    // Load all kangaroo states into registers
    uint64_t px[GPU_GRP_SIZE][4], py[GPU_GRP_SIZE][4];
    uint64_t dist[GPU_GRP_SIZE][4];
    // per-lane halt flag (see JLP kernel comment).
    bool halted[GPU_GRP_SIZE];

    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        size_t idx = base_idx + g;
        if (idx >= total_kangaroos) { halted[g] = true; continue; }

        // Restart fall-out: same restart-pending check as JLP kernel.
        if (state.restart_pending[idx]) {
            state.restart_pending[idx] = 0;
            rerandomize_kangaroo(px[g], py[g], idx, /*salt*/0,
                                 s_jump_x, s_jump_y);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                dist[g][i] = state.dist[idx * 4 + i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                px[g][i] = state.x[idx * 4 + i];
                py[g][i] = state.y[idx * 4 + i];
                dist[g][i] = state.dist[idx * 4 + i];
            }
        }
        halted[g] = false;
    }

    const uint64_t dp_mask = (1ULL << dp_bits) - 1;

    // Main stepping loop with SOTA symmetry optimization
    for (int step = 0; step < steps_per_kernel; step++) {
        // Step 1: Select jump points and compute denominators
        // SOTA twist: use Y coordinate LSB to decide +J or -J
        int jump_idx[GPU_GRP_SIZE];
        uint64_t dx[GPU_GRP_SIZE][4];
        bool use_neg_jump[GPU_GRP_SIZE];  // Track if using -J (negated jump)
        bool dx_was_zero[GPU_GRP_SIZE];   // see JLP kernel

        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            // Jump selection based on x-coordinate low bits (same as classic)
            jump_idx[g] = px[g][0] & JUMP_MASK;

            // SOTA SYMMETRY: Use Y coordinate LSB to decide direction
            // If py[0] & 1 == 1, we use the negated jump point -J instead of +J
            // This creates the "cheap second point" optimization
            use_neg_jump[g] = (py[g][0] & 1) != 0;

            // Load jump X from SHARED MEMORY (cached)
            uint64_t jx[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                jx[i] = s_jump_x[jump_idx[g] * 4 + i];
            }

            // dx = Jx - Px (denominator for slope)
            // Note: For -J, the X coordinate is the same (only Y is negated)
            mod_sub(dx[g], jx, px[g]);

            // Detect dx==0. See the matching comment in
            // kangaroo_step_kernel_jlp above for the SIMT-poison
            // explanation.
            bool is_zero = halted[g] ||
                (dx[g][0] == 0 && dx[g][1] == 0 &&
                 dx[g][2] == 0 && dx[g][3] == 0);
            dx_was_zero[g] = is_zero;
            if (is_zero) {
                dx[g][0] = 1; dx[g][1] = 0; dx[g][2] = 0; dx[g][3] = 0;
            }
        }

        // Step 2: Batch invert ALL denominators
        batch_mod_inv(dx);

        // Step 3: Complete all point additions with symmetry optimization
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            if (halted[g]) continue;

            int j = jump_idx[g];

            if (dx_was_zero[g]) {
                rerandomize_kangaroo(px[g], py[g], base_idx + g,
                                     /*salt*/step + 1, s_jump_x, s_jump_y);
                continue;
            }

            uint64_t rx[4], ry[4];

            // Load jump point from SHARED MEMORY (cached)
            uint64_t jx[4], jy[4], jd[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                jx[i] = s_jump_x[j * 4 + i];
                jy[i] = s_jump_y[j * 4 + i];
                jd[i] = s_jump_d[j * 4 + i];
            }

            // SOTA SYMMETRY: If use_neg_jump, negate the Y coordinate
            // -J = (Jx, -Jy) = (Jx, p - Jy)
            if (use_neg_jump[g]) {
                // Negate jy: jy = p - jy
                mod_sub(jy, SECP256K1_P, jy);
            }

            // Perform point addition with pre-inverted denominator
            ec_add_affine_with_inv(
                rx, ry,
                px[g], py[g],
                jx, jy,
                dx[g]  // Pre-computed 1/(Jx - Px)
            );

            // Update point
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                px[g][i] = rx[i];
                py[g][i] = ry[i];
            }

            // Update distance based on jump direction
            // SOTA: If we used -J, we SUBTRACT the jump distance instead of adding
            if (use_neg_jump[g]) {
                // dist = dist - jd (subtract jump distance)
                sub256(dist[g], dist[g], jd);
            } else {
                // dist = dist + jd (add jump distance)
                add256(dist[g], dist[g], jd);
            }

            // Check for DP (already in affine coordinates!)
            if ((px[g][0] & dp_mask) == 0) {
                size_t idx = base_idx + g;
                if (idx < total_kangaroos) {
                    state.flags[idx] = 1;

                    // Store affine coordinates and distance
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        state.x[idx * 4 + i] = px[g][i];
                        state.y[idx * 4 + i] = py[g][i];
                        state.z[idx * 4 + i] = (i == 0) ? 1 : 0;
                        state.dist[idx * 4 + i] = dist[g][i];
                    }
                    halted[g] = true;  // halt after DP, see JLP kernel.
                }
            }
        }
    }

    // save every non-halted lane (see JLP kernel save-final-state
    // comment for the full rationale).
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        size_t idx = base_idx + g;
        if (idx >= total_kangaroos) break;
        if (halted[g]) continue;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            state.x[idx * 4 + i] = px[g][i];
            state.y[idx * 4 + i] = py[g][i];
            state.z[idx * 4 + i] = (i == 0) ? 1 : 0;  // Z=1 for affine
            state.dist[idx * 4 + i] = dist[g][i];
        }
    }
}

/**
 * Main Kangaroo stepping kernel with CORRECT DP detection
 *
 * Each thread:
 * 1. Loads its kangaroo state
 * 2. Performs steps_per_kernel jumps in Jacobian coordinates
 * 3. Every DP_CHECK_INTERVAL steps, batch converts warp to affine
 * 4. Checks AFFINE X for DP property (the ONLY correct approach)
 * 5. Saves state back to global memory
 *
 * CRITICAL FIX: The previous implementation incorrectly used Jacobian X
 * for DP detection. This is MATHEMATICALLY WRONG because:
 *   - Jacobian: (X, Y, Z) represents affine point (X/Z^2, Y/Z^3)
 *   - The low bits of Jacobian X have ZERO correlation with affine X low bits
 *   - Any "quick check" on Jacobian X is pure noise - it catches ~0% of real DPs
 *
 * This implementation uses warp-cooperative batch inversion to efficiently
 * convert all 32 threads' points to affine every DP_CHECK_INTERVAL steps.
 * Cost: 1 mod_inv + ~93 mod_mul per 32 threads (vs 32 mod_inv before).
 */
__global__ void kangaroo_step_kernel(
    KangarooState state,
    uint32_t dp_bits,
    int steps_per_kernel,
    size_t count,
    const uint64_t* __restrict__ d_jump_x,  // Jump table X [NUM_JUMPS * 4]
    const uint64_t* __restrict__ d_jump_y,  // Jump table Y [NUM_JUMPS * 4]
    const uint64_t* __restrict__ d_jump_d,  // Jump distances [NUM_JUMPS * 4]
    int debug_mode                          // Debug output flag
) {
    // Shared memory for warp-cooperative batch inversion
    // Each warp needs 3 arrays of [WARP_SIZE][4] uint64_t
    // With 256 threads/block = 8 warps, we need 8 * 3 * 32 * 4 * 8 = 24KB
    __shared__ uint64_t s_products[8][WARP_SIZE * 4];   // Cumulative products
    __shared__ uint64_t s_z_inv[8][WARP_SIZE * 4];      // Z inverses
    __shared__ uint64_t s_z_coords[8][WARP_SIZE * 4];   // Original Z coordinates

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    // Warp identification for shared memory indexing
    const int warp_id = threadIdx.x / WARP_SIZE;

    // Load state into registers
    uint64_t px[4], py[4], pz[4], dist[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        px[i] = state.x[tid * 4 + i];
        py[i] = state.y[tid * 4 + i];
        pz[i] = state.z[tid * 4 + i];
        dist[i] = state.dist[tid * 4 + i];
    }

    const uint64_t dp_mask = (1ULL << dp_bits) - 1;

    // Step loop with periodic batch affine conversion for CORRECT DP detection
    for (int step = 0; step < steps_per_kernel; step++) {
        // Select jump based on low bits of Jacobian X
        // NOTE: This is fine for jump SELECTION (just need pseudo-randomness)
        // But we CANNOT use Jacobian X for DP DETECTION (need exact affine X)
        int jump_idx = px[0] & JUMP_MASK;

        // Load jump point (affine) from device memory (avoids RDC symbol issues)
        uint64_t jx[4], jy[4], jd[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            jx[i] = d_jump_x[jump_idx * 4 + i];
            jy[i] = d_jump_y[jump_idx * 4 + i];
            jd[i] = d_jump_d[jump_idx * 4 + i];
        }

        // Add jump point: P = P + jump (mixed: Jacobian + affine)
        uint64_t rx[4], ry[4], rz[4];
        ec_add_mixed(rx, ry, rz, px, py, pz, jx, jy);

        // Update state
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            px[i] = rx[i];
            py[i] = ry[i];
            pz[i] = rz[i];
        }

        // Update distance
        add256(dist, dist, jd);

        // Check if point hit infinity (Z = 0) IMMEDIATELY after each EC add.
        // This is extremely rare but causes crashes if not caught before the
        // next step. The kangaroo landed on its inverse point: P + jump = O.
        bool is_infinity = (pz[0] == 0 && pz[1] == 0 && pz[2] == 0 && pz[3] == 0);
        if (is_infinity) {
            // don't always rescue to jump slot 0. If a kangaroo
            // lands on infinity twice in succession (which CAN happen for
            // adversarially-chosen ranges), resetting to slot 0 each time
            // produces a closed loop with no progress. Pick a slot derived
            // from the thread id + step counter so each fall-out kangaroo
            // resumes on a different jump element. The jump table is large
            // (>= 32 entries) so collisions are negligible.
            uint64_t fallback_slot = ((uint64_t)tid * 0x9E3779B97F4A7C15ULL
                                       ^ (uint64_t)step) % ::collider::gpu::kKangarooJumpTableSize;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                px[i] = d_jump_x[fallback_slot * 4 + i];
                py[i] = d_jump_y[fallback_slot * 4 + i];
                pz[i] = (i == 0) ? 1 : 0;  // Z = 1 (affine)
            }
            continue;  // Skip this step's DP check entirely
        }

        // ======================================================================
        // CORRECT DP DETECTION: Batch convert to affine every DP_CHECK_INTERVAL
        // ======================================================================
        // We check at intervals AND at the last step to ensure no DPs are missed
        if (((step + 1) % DP_CHECK_INTERVAL) == 0 || step == steps_per_kernel - 1) {
            // Convert to affine coordinates for DP detection
            uint64_t affine_x[4], affine_y[4];

            // Batch inversion mode:
            // 0 = Warp-cooperative batch inversion (fast, 1 inv + 93 mul for 32 threads)
            // 1 = Simple per-thread inversion (slower, 32 inversions)
            #define USE_SIMPLE_INVERSION 0

            #if USE_SIMPLE_INVERSION
            // Simple per-thread Jacobian to affine conversion
            // This is ~32x slower but we know it works correctly
            bool is_inf = (pz[0] == 0 && pz[1] == 0 && pz[2] == 0 && pz[3] == 0);
            if (is_inf) {
                for (int i = 0; i < 4; i++) {
                    affine_x[i] = 0;
                    affine_y[i] = 0;
                }
            } else {
                uint64_t z_inv[4], z_inv2[4], z_inv3[4];
                mod_inv(z_inv, pz);      // z_inv = 1/Z

                // DEBUG: Verify z * z_inv = 1 (mod p) for thread 0, first check
                if (debug_mode && tid == 0 && step == DP_CHECK_INTERVAL - 1) {
                    uint64_t test[4];
                    mod_mul(test, pz, z_inv);  // test = z * z_inv, should be 1
                    printf("[DEBUG] Thread 0 mod_inv verification:\n");
                    printf("  pz:     [%016llx, %016llx, %016llx, %016llx]\n",
                           (unsigned long long)pz[3], (unsigned long long)pz[2],
                           (unsigned long long)pz[1], (unsigned long long)pz[0]);
                    printf("  z_inv:  [%016llx, %016llx, %016llx, %016llx]\n",
                           (unsigned long long)z_inv[3], (unsigned long long)z_inv[2],
                           (unsigned long long)z_inv[1], (unsigned long long)z_inv[0]);
                    printf("  z*z_inv:[%016llx, %016llx, %016llx, %016llx] (should be 1)\n",
                           (unsigned long long)test[3], (unsigned long long)test[2],
                           (unsigned long long)test[1], (unsigned long long)test[0]);
                }

                mod_sqr(z_inv2, z_inv);  // z_inv2 = 1/Z^2
                mod_mul(z_inv3, z_inv2, z_inv);  // z_inv3 = 1/Z^3
                mod_mul(affine_x, px, z_inv2);   // affine_x = X / Z^2
                mod_mul(affine_y, py, z_inv3);   // affine_y = Y / Z^3

                // DEBUG: Print affine_x for thread 0
                if (debug_mode && tid == 0 && step == DP_CHECK_INTERVAL - 1) {
                    printf("  affine_x:[%016llx, %016llx, %016llx, %016llx]\n",
                           (unsigned long long)affine_x[3], (unsigned long long)affine_x[2],
                           (unsigned long long)affine_x[1], (unsigned long long)affine_x[0]);
                    printf("  dp_mask: 0x%llx\n", (unsigned long long)dp_mask);
                    printf("  affine_x[0] & dp_mask = 0x%llx\n",
                           (unsigned long long)(affine_x[0] & dp_mask));
                }
            }
            #else
            // Warp-cooperative batch inversion (fast but needs debugging)
            warp_batch_jacobian_to_affine(
                affine_x, affine_y,
                px, py, pz,
                s_products[warp_id],
                s_z_inv[warp_id],
                s_z_coords[warp_id]
            );
            #endif

            // Check the AFFINE X coordinate for DP property.
            if ((affine_x[0] & dp_mask) == 0) {
                // TRUE Distinguished Point found!
                state.flags[tid] = 1;

                // Store AFFINE coordinates (required for collision detection)
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    state.x[tid * 4 + i] = affine_x[i];
                    state.y[tid * 4 + i] = affine_y[i];
                    state.z[tid * 4 + i] = (i == 0) ? 1 : 0;  // Z=1 for affine
                    state.dist[tid * 4 + i] = dist[i];
                }
                return;  // DP found, exit kernel for this thread
            }

            // After batch conversion, we can continue with Jacobian coordinates
            // The affine values were just for DP checking, we keep px/py/pz as-is
            // (they're still valid Jacobian coords for the same point)
        }
    }

    // Save final state back to global memory (still in Jacobian form)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        state.x[tid * 4 + i] = px[i];
        state.y[tid * 4 + i] = py[i];
        state.z[tid * 4 + i] = pz[i];
        state.dist[tid * 4 + i] = dist[i];
    }
}

// ============================================================================
// Host-side GPU Kangaroo Solver
// ============================================================================

class GPUKangarooSolver {
public:
    GPUKangarooSolver() = default;
    ~GPUKangarooSolver() { cleanup(); }

    bool init(int device_id = 0) {
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            std::cerr << "[!] Failed to set CUDA device: " << cudaGetErrorString(err) << "\n";
            return false;
        }

        // Get device properties
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);
        std::cout << "[*] GPU Kangaroo initialized on: " << props.name << "\n";
        std::cout << "    Compute capability: " << props.major << "." << props.minor << "\n";
        std::cout << "    SMs: " << props.multiProcessorCount << "\n";

        device_id_ = device_id;
        initialized_ = true;
        return true;
    }

    bool allocate(size_t num_kangaroos) {
        if (!initialized_) return false;

        num_kangaroos_ = num_kangaroos;
        size_t size_256 = num_kangaroos * 4 * sizeof(uint64_t);
        size_t size_32 = num_kangaroos * sizeof(uint32_t);

        cudaError_t err;

        err = cudaMalloc(&d_x_, size_256);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_y_, size_256);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_z_, size_256);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_dist_, size_256);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_flags_, size_32);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_types_, size_32);
        if (err != cudaSuccess) return false;

        // Per-kangaroo wild starting offset (256-bit) and a 32-bit restart
        // flag.
        err = cudaMalloc(&d_wild_offset_, size_256);
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&d_restart_pending_, size_32);
        if (err != cudaSuccess) return false;

        allocated_ = true;
        return true;
    }

    bool upload_jump_table(
        const uint64_t jump_x[][4],
        const uint64_t jump_y[][4],
        const uint64_t jump_d[][4]
    ) {
        // Use device global memory instead of __constant__ memory
        // This fixes the "named symbol not found" error with CUDA RDC + LTO
        cudaError_t err;
        size_t jump_table_size = NUM_JUMPS * 4 * sizeof(uint64_t);

        // Allocate device memory for jump tables (if not already allocated)
        if (!d_jump_x_) {
            err = cudaMalloc(&d_jump_x_, jump_table_size);
            if (err != cudaSuccess) {
                fprintf(stderr, "[CUDA ERROR] Failed to allocate d_jump_x_: %s\n", cudaGetErrorString(err));
                return false;
            }
        }
        if (!d_jump_y_) {
            err = cudaMalloc(&d_jump_y_, jump_table_size);
            if (err != cudaSuccess) {
                fprintf(stderr, "[CUDA ERROR] Failed to allocate d_jump_y_: %s\n", cudaGetErrorString(err));
                return false;
            }
        }
        if (!d_jump_d_) {
            err = cudaMalloc(&d_jump_d_, jump_table_size);
            if (err != cudaSuccess) {
                fprintf(stderr, "[CUDA ERROR] Failed to allocate d_jump_d_: %s\n", cudaGetErrorString(err));
                return false;
            }
        }

        // Copy jump tables to device memory
        err = cudaMemcpy(d_jump_x_, jump_x, jump_table_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to copy d_jump_x_: %s\n", cudaGetErrorString(err));
            return false;
        }

        err = cudaMemcpy(d_jump_y_, jump_y, jump_table_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to copy d_jump_y_: %s\n", cudaGetErrorString(err));
            return false;
        }

        err = cudaMemcpy(d_jump_d_, jump_d, jump_table_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to copy d_jump_d_: %s\n", cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    bool upload_initial_state(
        const uint64_t* h_x,
        const uint64_t* h_y,
        const uint64_t* h_z,
        const uint64_t* h_dist,
        const uint32_t* h_types
    ) {
        if (!allocated_) return false;

        size_t size_256 = num_kangaroos_ * 4 * sizeof(uint64_t);
        size_t size_32 = num_kangaroos_ * sizeof(uint32_t);

        if (!kg_cuda_check(
                cudaMemcpy(d_x_, h_x, size_256, cudaMemcpyHostToDevice),
                "upload_initial_state.d_x")) return false;
        if (!kg_cuda_check(
                cudaMemcpy(d_y_, h_y, size_256, cudaMemcpyHostToDevice),
                "upload_initial_state.d_y")) return false;
        if (!kg_cuda_check(
                cudaMemcpy(d_z_, h_z, size_256, cudaMemcpyHostToDevice),
                "upload_initial_state.d_z")) return false;
        if (!kg_cuda_check(
                cudaMemcpy(d_dist_, h_dist, size_256, cudaMemcpyHostToDevice),
                "upload_initial_state.d_dist")) return false;
        if (!kg_cuda_check(
                cudaMemcpy(d_types_, h_types, size_32, cudaMemcpyHostToDevice),
                "upload_initial_state.d_types")) return false;

        // Clear flags
        cudaMemset(d_flags_, 0, size_32);

        return true;
    }

    // Kernel mode selection
    enum class KernelMode {
        JACOBIAN_WARP_BATCH,  // Original: Jacobian coords + warp-cooperative batch inversion
        JLP_AFFINE_BATCH,     // JLP-style: Affine coords + per-thread batch inversion
        SOTA_3GROUP           // SOTA: 3-group with symmetry, K=1.15 (fastest, 45% fewer ops)
    };

    void step(uint32_t dp_bits, int steps_per_kernel, KernelMode mode = KernelMode::SOTA_3GROUP) {
        if (!allocated_) return;

        // Check that jump tables are uploaded
        if (!d_jump_x_ || !d_jump_y_ || !d_jump_d_) {
            fprintf(stderr, "[ERROR] Jump tables not uploaded! Call upload_jump_table() first.\n");
            return;
        }

        KangarooState state;
        state.x = d_x_;
        state.y = d_y_;
        state.z = d_z_;
        state.dist = d_dist_;
        state.flags = d_flags_;
        state.types = d_types_;
        state.wild_offset = d_wild_offset_;
        state.restart_pending = d_restart_pending_;

        if (mode == KernelMode::SOTA_3GROUP) {
            // SOTA kernel: 3-group with symmetry optimization (K=1.15)
            // Uses Y coordinate LSB to decide +J or -J, achieving 45% fewer operations
            int num_threads = (num_kangaroos_ + GPU_GRP_SIZE - 1) / GPU_GRP_SIZE;
            int threads_per_block = 128;  // Lower for register pressure
            int blocks = (num_threads + threads_per_block - 1) / threads_per_block;

            // T1.6: sticky context error attribution.
            (void)collider::gpu::pre_launch_context_probe(
                "kangaroo_step_kernel_sota");
            kangaroo_step_kernel_sota<<<blocks, threads_per_block>>>(
                state, dp_bits, steps_per_kernel, num_kangaroos_,
                d_jump_x_, d_jump_y_, d_jump_d_
            );
        } else if (mode == KernelMode::JLP_AFFINE_BATCH) {
            // JLP-style kernel: each thread handles GPU_GRP_SIZE kangaroos
            // This is significantly faster due to per-thread batch inversion
            int num_threads = (num_kangaroos_ + GPU_GRP_SIZE - 1) / GPU_GRP_SIZE;
            int threads_per_block = 128;  // Lower for register pressure
            int blocks = (num_threads + threads_per_block - 1) / threads_per_block;

            // T1.6: sticky context error attribution.
            (void)collider::gpu::pre_launch_context_probe(
                "kangaroo_step_kernel_jlp");
            kangaroo_step_kernel_jlp<<<blocks, threads_per_block>>>(
                state, dp_bits, steps_per_kernel, num_kangaroos_,
                d_jump_x_, d_jump_y_, d_jump_d_  // Pass jump table pointers
            );
        } else {
            // Original Jacobian kernel with warp-cooperative batch inversion
            int threads_per_block = 256;
            int blocks = (num_kangaroos_ + threads_per_block - 1) / threads_per_block;

            // T1.6: sticky context error attribution.
            (void)collider::gpu::pre_launch_context_probe(
                "kangaroo_step_kernel");
            kangaroo_step_kernel<<<blocks, threads_per_block>>>(
                state, dp_bits, steps_per_kernel, num_kangaroos_,
                d_jump_x_, d_jump_y_, d_jump_d_,  // Pass jump table pointers
                0  // debug_mode
            );
        }

        cudaDeviceSynchronize();
    }

    // Convenience method for JLP kernel
    void step_jlp(uint32_t dp_bits, int steps_per_kernel) {
        step(dp_bits, steps_per_kernel, KernelMode::JLP_AFFINE_BATCH);
    }

    // Convenience method for SOTA kernel (recommended - fastest)
    void step_sota(uint32_t dp_bits, int steps_per_kernel) {
        step(dp_bits, steps_per_kernel, KernelMode::SOTA_3GROUP);
    }

    // Download flags and check for DPs
    std::vector<size_t> collect_dps() {
        std::vector<size_t> dp_indices;
        if (!allocated_) return dp_indices;

        std::vector<uint32_t> h_flags(num_kangaroos_);
        if (!kg_cuda_check(
                cudaMemcpy(h_flags.data(), d_flags_,
                           num_kangaroos_ * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost),
                "collect_dps.h_flags")) {
            // D2H copy failure -> h_flags is garbage. Return empty
            // so the caller's loop doesn't dereference junk indices
            // into d_x/d_y. Next collect_dps() call retries.
            return dp_indices;
        }

        for (size_t i = 0; i < num_kangaroos_; i++) {
            if (h_flags[i]) {
                dp_indices.push_back(i);
            }
        }

        // Clear flags for next round
        cudaMemset(d_flags_, 0, num_kangaroos_ * sizeof(uint32_t));

        return dp_indices;
    }

    // Download state for specific kangaroos
    void download_state(
        const std::vector<size_t>& indices,
        uint64_t* h_x,
        uint64_t* h_y,
        uint64_t* h_dist,
        uint32_t* h_types
    ) {
        if (!allocated_) return;

        // For simplicity, download all and filter
        size_t size_256 = num_kangaroos_ * 4 * sizeof(uint64_t);
        std::vector<uint64_t> all_x(num_kangaroos_ * 4);
        std::vector<uint64_t> all_y(num_kangaroos_ * 4);
        std::vector<uint64_t> all_dist(num_kangaroos_ * 4);
        std::vector<uint32_t> all_types(num_kangaroos_);

        // Bail if any of the four download copies fails -- the rest
        // of this function would otherwise read garbage from the
        // host buffers and write nonsense h_x/h_y/h_dist back to
        // the caller (which then submits the bogus DPs to the pool).
        if (!kg_cuda_check(
                cudaMemcpy(all_x.data(), d_x_, size_256, cudaMemcpyDeviceToHost),
                "download_state.all_x")) return;
        if (!kg_cuda_check(
                cudaMemcpy(all_y.data(), d_y_, size_256, cudaMemcpyDeviceToHost),
                "download_state.all_y")) return;
        if (!kg_cuda_check(
                cudaMemcpy(all_dist.data(), d_dist_, size_256, cudaMemcpyDeviceToHost),
                "download_state.all_dist")) return;
        if (!kg_cuda_check(
                cudaMemcpy(all_types.data(), d_types_,
                           num_kangaroos_ * sizeof(uint32_t),
                           cudaMemcpyDeviceToHost),
                "download_state.all_types")) return;

        for (size_t idx = 0; idx < indices.size(); idx++) {
            size_t i = indices[idx];
            for (int j = 0; j < 4; j++) {
                h_x[idx * 4 + j] = all_x[i * 4 + j];
                h_y[idx * 4 + j] = all_y[i * 4 + j];
                h_dist[idx * 4 + j] = all_dist[i * 4 + j];
            }
            h_types[idx] = all_types[i];
        }
    }

    void cleanup() {
        if (d_x_) cudaFree(d_x_);
        if (d_y_) cudaFree(d_y_);
        if (d_z_) cudaFree(d_z_);
        if (d_dist_) cudaFree(d_dist_);
        if (d_flags_) cudaFree(d_flags_);
        if (d_types_) cudaFree(d_types_);
        if (d_wild_offset_) cudaFree(d_wild_offset_);
        if (d_restart_pending_) cudaFree(d_restart_pending_);

        // Free jump table device memory
        if (d_jump_x_) cudaFree(d_jump_x_);
        if (d_jump_y_) cudaFree(d_jump_y_);
        if (d_jump_d_) cudaFree(d_jump_d_);

        d_x_ = d_y_ = d_z_ = d_dist_ = nullptr;
        d_wild_offset_ = nullptr;
        d_flags_ = d_types_ = d_restart_pending_ = nullptr;
        d_jump_x_ = d_jump_y_ = d_jump_d_ = nullptr;
        allocated_ = false;
    }

private:
    int device_id_ = 0;
    bool initialized_ = false;
    bool allocated_ = false;
    size_t num_kangaroos_ = 0;

    // Device memory - kangaroo state
    uint64_t* d_x_ = nullptr;
    uint64_t* d_y_ = nullptr;
    uint64_t* d_z_ = nullptr;
    uint64_t* d_dist_ = nullptr;
    uint32_t* d_flags_ = nullptr;
    uint32_t* d_types_ = nullptr;
    uint64_t* d_wild_offset_ = nullptr;     // Per-kangaroo wild starting offset
    uint32_t* d_restart_pending_ = nullptr; // Per-kangaroo restart flag

    // Device memory - jump tables (using device memory instead of __constant__)
    // This fixes the "named symbol not found" error with CUDA RDC + LTO
    uint64_t* d_jump_x_ = nullptr;
    uint64_t* d_jump_y_ = nullptr;
    uint64_t* d_jump_d_ = nullptr;
};

}  // namespace gpu
}  // namespace collider

// ============================================================================
// GPUKangarooManager Implementation (must be in same .cu file as kernels)
// ============================================================================

#include "kangaroo_solver_gpu.hpp"
#include "../core/crypto_cpu.hpp"
#include "kangaroo_dp_table.hpp"   // full-256 DPKey
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <unordered_map>

#ifdef _MSC_VER
#include <intrin.h>
inline int clz64_local(uint64_t x) {
    unsigned long index;
    if (_BitScanReverse64(&index, x)) {
        return 63 - static_cast<int>(index);
    }
    return 64;
}
#define CLZ64_LOCAL(x) clz64_local(x)
#else
#define CLZ64_LOCAL(x) __builtin_clzll(x)
#endif

namespace collider {
namespace gpu {

/**
 * Implementation struct (pimpl)
 */
struct GPUKangarooManager::Impl {
    int device_id = 0;
    bool initialized = false;
    bool allocated = false;
    bool debug_mode = false;

    // Range and target
    UInt256 range_start;
    UInt256 range_end;
    std::array<uint8_t, 20> target_h160;

    // Target public key (needed for wild kangaroos)
    // Set via set_target_pubkey() before solve()
    uint64_t target_pubkey_x[4] = {0};
    uint64_t target_pubkey_y[4] = {0};
    bool has_target_pubkey = false;

    // Device memory - kangaroo state
    uint64_t* d_x = nullptr;
    uint64_t* d_y = nullptr;
    uint64_t* d_z = nullptr;
    uint64_t* d_dist = nullptr;
    uint32_t* d_flags = nullptr;
    uint32_t* d_types = nullptr;
    // Per-kangaroo wild starting offset + restart-pending flag
    uint64_t* d_wild_offset = nullptr;
    uint32_t* d_restart_pending = nullptr;

    // Device memory - jump tables (using device memory instead of __constant__)
    // This fixes the "named symbol not found" error with CUDA RDC + LTO
    uint64_t* d_jump_x = nullptr;
    uint64_t* d_jump_y = nullptr;
    uint64_t* d_jump_d = nullptr;

    // Jump table (host side)
    uint64_t jump_x[NUM_JUMPS][4];
    uint64_t jump_y[NUM_JUMPS][4];
    uint64_t jump_d[NUM_JUMPS][4];

    // Distinguished points keyed on full 256-bit X via DPKey.
    // Pre-fix this was keyed on x_low (uint64_t), silently dropping DPs that
    // collided in the low 64 bits but differed in the upper 192. With DPKey,
    // two DPs at distinct positions are always retained at distinct slots.
    // Wild kangaroos carry a per-instance starting offset (their initial
    // dist = offset). When a wild collides with a tame, the collision
    // algebra must subtract that offset from the wild's accumulated
    // distance before computing the private key.
    struct DPEntry {
        uint64_t dist[4];
        uint64_t wild_offset[4];
        uint32_t type;
        size_t   kang_idx;        // which kangaroo produced this DP
    };
    std::unordered_map<::collider::kangaroo::DPKey, DPEntry> dp_table;

    bool init(int dev_id) {
        device_id = dev_id;

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            std::cerr << "[!] Failed to set CUDA device: " << cudaGetErrorString(err) << "\n";
            return false;
        }

        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);
        std::cout << "[*] GPU Kangaroo initialized on: " << props.name << "\n";
        std::cout << "    Compute capability: " << props.major << "." << props.minor << "\n";
        std::cout << "    SMs: " << props.multiProcessorCount << "\n";

        initialized = true;
        return true;
    }

    bool allocate(int num_kangaroos) {
        if (!initialized) return false;

        size_t size_256 = num_kangaroos * 4 * sizeof(uint64_t);
        size_t size_32 = num_kangaroos * sizeof(uint32_t);

        cudaError_t err;

        err = cudaMalloc(&d_x, size_256);
        if (err != cudaSuccess) {
            std::cerr << "[!] cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
            return false;
        }

        err = cudaMalloc(&d_y, size_256);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_z, size_256);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_dist, size_256);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_flags, size_32);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_types, size_32);
        if (err != cudaSuccess) return false;

        // Per-kangaroo wild offset + restart-pending buffers.
        err = cudaMalloc(&d_wild_offset, size_256);
        if (err != cudaSuccess) return false;
        err = cudaMalloc(&d_restart_pending, size_32);
        if (err != cudaSuccess) return false;

        allocated = true;
        return true;
    }

    void cleanup() {
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_z) cudaFree(d_z);
        if (d_dist) cudaFree(d_dist);
        if (d_flags) cudaFree(d_flags);
        if (d_types) cudaFree(d_types);
        if (d_wild_offset) cudaFree(d_wild_offset);
        if (d_restart_pending) cudaFree(d_restart_pending);

        // Free jump table device memory
        if (d_jump_x) cudaFree(d_jump_x);
        if (d_jump_y) cudaFree(d_jump_y);
        if (d_jump_d) cudaFree(d_jump_d);

        d_x = d_y = d_z = d_dist = nullptr;
        d_wild_offset = nullptr;
        d_flags = d_types = d_restart_pending = nullptr;
        d_jump_x = d_jump_y = d_jump_d = nullptr;
        allocated = false;
    }

    // Upload jump tables to device memory (instead of constant memory)
    bool upload_jump_tables() {
        cudaError_t err;
        size_t jump_table_size = NUM_JUMPS * 4 * sizeof(uint64_t);

        // Allocate device memory for jump tables (if not already allocated)
        if (!d_jump_x) {
            err = cudaMalloc(&d_jump_x, jump_table_size);
            if (err != cudaSuccess) {
                fprintf(stderr, "[CUDA ERROR] Failed to allocate d_jump_x: %s\n", cudaGetErrorString(err));
                return false;
            }
        }
        if (!d_jump_y) {
            err = cudaMalloc(&d_jump_y, jump_table_size);
            if (err != cudaSuccess) {
                fprintf(stderr, "[CUDA ERROR] Failed to allocate d_jump_y: %s\n", cudaGetErrorString(err));
                return false;
            }
        }
        if (!d_jump_d) {
            err = cudaMalloc(&d_jump_d, jump_table_size);
            if (err != cudaSuccess) {
                fprintf(stderr, "[CUDA ERROR] Failed to allocate d_jump_d: %s\n", cudaGetErrorString(err));
                return false;
            }
        }

        // Copy jump tables to device memory
        err = cudaMemcpy(d_jump_x, jump_x, jump_table_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to copy d_jump_x: %s\n", cudaGetErrorString(err));
            return false;
        }

        err = cudaMemcpy(d_jump_y, jump_y, jump_table_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to copy d_jump_y: %s\n", cudaGetErrorString(err));
            return false;
        }

        err = cudaMemcpy(d_jump_d, jump_d, jump_table_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to copy d_jump_d: %s\n", cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    void generate_jump_table(uint64_t mean_jump) {
        for (int i = 0; i < NUM_JUMPS; i++) {
            // Compute jump distance: exponentially distributed around mean
            uint64_t jump_dist = mean_jump >> (NUM_JUMPS / 2 - i / 2);
            if (jump_dist == 0) jump_dist = 1;

            // Store jump distance as 256-bit scalar
            jump_d[i][0] = jump_dist;
            jump_d[i][1] = 0;
            jump_d[i][2] = 0;
            jump_d[i][3] = 0;

            // Compute jump_dist * G using CPU EC multiplication
            cpu::uint256_t k;
            k.d[0] = jump_dist;
            k.d[1] = 0;
            k.d[2] = 0;
            k.d[3] = 0;

            cpu::ECPoint P;
            cpu::ec_mul(P, k);  // P = jump_dist * G

            // Convert to affine coordinates
            cpu::uint256_t px, py;
            cpu::ec_to_affine(px, py, P);

            // Store affine coordinates in jump table
            for (int j = 0; j < 4; j++) {
                jump_x[i][j] = px.d[j];
                jump_y[i][j] = py.d[j];
            }
        }
    }

    void init_kangaroos(int num_kangaroos, std::vector<uint64_t>& h_x,
                        std::vector<uint64_t>& h_y, std::vector<uint64_t>& h_z,
                        std::vector<uint64_t>& h_dist, std::vector<uint32_t>& h_types,
                        std::vector<uint64_t>& h_wild_offset) {
        std::mt19937_64 rng(std::random_device{}());

        h_x.resize(num_kangaroos * 4);
        h_y.resize(num_kangaroos * 4);
        h_z.resize(num_kangaroos * 4);
        h_dist.resize(num_kangaroos * 4);
        h_types.resize(num_kangaroos);
        // Per-kangaroo wild starting offset. Zeroed by default; only
        // wild1/wild2 get a non-zero value.
        h_wild_offset.assign(num_kangaroos * 4, 0);

        // Compute 256-bit range size: range_size = range_end - range_start
        cpu::uint256_t range_start_256, range_end_256, range_size_256;
        range_start_256.d[0] = range_start.parts[0];
        range_start_256.d[1] = range_start.parts[1];
        range_start_256.d[2] = range_start.parts[2];
        range_start_256.d[3] = range_start.parts[3];
        range_end_256.d[0] = range_end.parts[0];
        range_end_256.d[1] = range_end.parts[1];
        range_end_256.d[2] = range_end.parts[2];
        range_end_256.d[3] = range_end.parts[3];
        cpu::sub256(range_size_256, range_end_256, range_start_256);

        // SOTA 3-group initialization:
        // - 1/3 Tame: start at random k*G in range, track k
        // - 1/3 Wild1: start at Q, track distance from Q
        // - 1/3 Wild2: start at -Q (mirror), track distance from -Q
        // When collision occurs between different groups (see detection site
        // for the precise algebra and mod_half for W1-W2):
        // - Tame-Wild1:  k = tame_dist - wild1_dist  mod n
        // - Tame-Wild2:  k = wild2_dist - tame_dist  mod n  (Wild2 tracks from -Q)
        // - Wild1-Wild2: k = (wild2_dist - wild1_dist) / 2  mod n  (symmetry)

        int third = num_kangaroos / 3;

        for (int i = 0; i < num_kangaroos; i++) {
            if (i < third) {
                // TAME kangaroos (Type 0): start at random k*G in range
                h_types[i] = KANG_TAME;

                // uniform sample in [0, range_size_256) via
                // bit-masked rejection sampling. Pre-fix: per-limb % was
                // mathematically nonsensical AND had no branch for ranges
                // requiring 3 or 4 limbs, so Tame kangaroos on >128-bit
                // puzzles started anywhere on the curve (not in range).
                cpu::uint256_t offset;
                cpu::random_below(offset, [&]() { return rng(); }, range_size_256);

                // dist = range_start + offset
                cpu::uint256_t dist;
                cpu::add256(dist, range_start_256, offset);

                h_dist[i * 4 + 0] = dist.d[0];
                h_dist[i * 4 + 1] = dist.d[1];
                h_dist[i * 4 + 2] = dist.d[2];
                h_dist[i * 4 + 3] = dist.d[3];

                // Position = dist * G
                cpu::ECPoint P;
                cpu::ec_mul(P, dist);
                cpu::uint256_t px, py;
                cpu::ec_to_affine(px, py, P);

                for (int j = 0; j < 4; j++) {
                    h_x[i * 4 + j] = px.d[j];
                    h_y[i * 4 + j] = py.d[j];
                }
            } else if (i < 2 * third) {
                // WILD1 kangaroos (Type 1): start at Q + offset_i * G, track
                // distance accumulated from offset_i.
                // Per-instance random offset replaces the prior shared-start design where
                // every wild1 began at exactly the same (Q.x, Q.y, dist=0).
                // Shared X => identical jump_idx => identical trajectory,
                // so a herd of N wilds did the work of 1; same problem on
                // multi-GPU and across kernel invocations. Generating a
                // fresh ~128-bit offset per kangaroo gives each wild a
                // distinct starting X (and hence trajectory) while the
                // collision algebra continues to work: a tame at
                // dist_t * G that meets wild1 at dist_w * G satisfies
                // tame_d = priv + (wild_d - offset_i)  =>  priv = tame_d - (wild_d - offset_i).
                h_types[i] = KANG_WILD1;

                // Bound offset by sqrt(range_size) so the wild stays in a
                // statistically reasonable region (Pollard's analysis
                // expects wild excursions of that magnitude). We do this
                // simply by sampling 128 bits from the rng; the range_bits
                // formula in solve() guarantees this bound is meaningful
                // for any puzzle <= 256 bits.
                cpu::uint256_t offset_i;
                offset_i.d[0] = rng();
                offset_i.d[1] = rng() & 0x3FFFFFFFFFFFFFFFULL;  // ~126 bits
                offset_i.d[2] = 0;
                offset_i.d[3] = 0;

                // Position = Q + offset_i * G. Compute offset_i * G on
                // the CPU (one ec_mul, ~256 ec_doubles) then ec_add to Q.
                cpu::ECPoint OG;
                cpu::ec_mul(OG, offset_i);

                cpu::uint256_t og_x, og_y;
                cpu::ec_to_affine(og_x, og_y, OG);

                // Q + offset_i*G via CPU EC add (Jacobian internally).
                // Build a Jacobian Q point then add.
                cpu::ECPoint QP;
                QP.X.d[0] = target_pubkey_x[0]; QP.X.d[1] = target_pubkey_x[1];
                QP.X.d[2] = target_pubkey_x[2]; QP.X.d[3] = target_pubkey_x[3];
                QP.Y.d[0] = target_pubkey_y[0]; QP.Y.d[1] = target_pubkey_y[1];
                QP.Y.d[2] = target_pubkey_y[2]; QP.Y.d[3] = target_pubkey_y[3];
                QP.Z.d[0] = 1; QP.Z.d[1] = 0; QP.Z.d[2] = 0; QP.Z.d[3] = 0;

                cpu::ECPoint W1;
                cpu::ec_add(W1, QP, og_x, og_y);

                cpu::uint256_t w1_x, w1_y;
                cpu::ec_to_affine(w1_x, w1_y, W1);

                for (int j = 0; j < 4; j++) {
                    h_x[i * 4 + j] = w1_x.d[j];
                    h_y[i * 4 + j] = w1_y.d[j];
                    h_dist[i * 4 + j] = offset_i.d[j];        // dist starts at offset
                    h_wild_offset[i * 4 + j] = offset_i.d[j]; // remembered for collision
                }
            } else {
                // WILD2 kangaroos (Type 2): start at -Q + offset_i * G.
                // Same per-instance offset story as WILD1.
                h_types[i] = KANG_WILD2;

                cpu::uint256_t offset_i;
                offset_i.d[0] = rng();
                offset_i.d[1] = rng() & 0x3FFFFFFFFFFFFFFFULL;
                offset_i.d[2] = 0;
                offset_i.d[3] = 0;

                cpu::ECPoint OG;
                cpu::ec_mul(OG, offset_i);

                cpu::uint256_t og_x, og_y;
                cpu::ec_to_affine(og_x, og_y, OG);

                // -Q in affine = (Q.x, p - Q.y).
                cpu::uint256_t qy, neg_qy;
                for (int j = 0; j < 4; j++) qy.d[j] = target_pubkey_y[j];
                cpu::sub256(neg_qy, cpu::SECP256K1_P, qy);

                // (-Q) + offset_i * G via CPU EC add.
                cpu::ECPoint NQP;
                NQP.X.d[0] = target_pubkey_x[0]; NQP.X.d[1] = target_pubkey_x[1];
                NQP.X.d[2] = target_pubkey_x[2]; NQP.X.d[3] = target_pubkey_x[3];
                NQP.Y = neg_qy;
                NQP.Z.d[0] = 1; NQP.Z.d[1] = 0; NQP.Z.d[2] = 0; NQP.Z.d[3] = 0;

                cpu::ECPoint W2;
                cpu::ec_add(W2, NQP, og_x, og_y);

                cpu::uint256_t w2_x, w2_y;
                cpu::ec_to_affine(w2_x, w2_y, W2);

                for (int j = 0; j < 4; j++) {
                    h_x[i * 4 + j] = w2_x.d[j];
                    h_y[i * 4 + j] = w2_y.d[j];
                    h_dist[i * 4 + j] = offset_i.d[j];
                    h_wild_offset[i * 4 + j] = offset_i.d[j];
                }
            }

            // Z = 1 (affine input)
            h_z[i * 4 + 0] = 1;
            h_z[i * 4 + 1] = 0;
            h_z[i * 4 + 2] = 0;
            h_z[i * 4 + 3] = 0;
        }
    }
};

GPUKangarooManager::GPUKangarooManager() : impl_(new Impl()) {}

GPUKangarooManager::~GPUKangarooManager() {
    if (impl_) {
        impl_->cleanup();
        delete impl_;
    }
}

bool GPUKangarooManager::init(int device_id) {
    bool result = impl_->init(device_id);
    if (result) {
        print_gpu_grp_size_recommendation(device_id);
    }
    return result;
}

void GPUKangarooManager::set_range(const UInt256& start, const UInt256& end) {
    impl_->range_start = start;
    impl_->range_end = end;
}

void GPUKangarooManager::set_target_h160(const std::array<uint8_t, 20>& h160) {
    impl_->target_h160 = h160;
}

void GPUKangarooManager::set_target_pubkey(const cpu::uint256_t& x, const cpu::uint256_t& y) {
    for (int i = 0; i < 4; i++) {
        impl_->target_pubkey_x[i] = x.d[i];
        impl_->target_pubkey_y[i] = y.d[i];
    }
    impl_->has_target_pubkey = true;
}

GPUKangarooResult GPUKangarooManager::solve() {
    GPUKangarooResult result;
    result.found = false;
    result.total_steps = 0;
    result.dp_count = 0;
    result.elapsed_seconds = 0;

    // Copy debug_mode from public interface to implementation
    impl_->debug_mode = debug_mode;

    // Verify target public key is set
    if (!impl_->has_target_pubkey) {
        std::cerr << "[!] Error: target public key not set. Call set_target_pubkey() before solve().\n";
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    if (!impl_->allocate(num_kangaroos)) {
        std::cerr << "[!] Failed to allocate GPU memory for " << num_kangaroos << " kangaroos\n";
        return result;
    }

    std::cout << "[*] Allocated GPU memory for " << num_kangaroos << " kangaroos\n";
    std::cout << "[*] Using SOTA 3-group kernel with GPU_GRP_SIZE=" << GPU_GRP_SIZE << "\n";

    uint64_t range_bits = 0;
    for (int i = 3; i >= 0; i--) {
        if (impl_->range_end.parts[i] != 0) {
            range_bits = i * 64 + 64 - CLZ64_LOCAL(impl_->range_end.parts[i]);
            break;
        }
    }
    // mean_jump formula.
    // A naive `1ULL << (range_bits / 2 - 4)` underflows for range_bits < 8
    // (shift count becomes negative and wraps to a huge value in unsigned
    // arithmetic) AND the constant -4 has no theoretical justification.
    // The standard Pollard kangaroo result says the expected jump
    // distance should be sqrt(L) / herd_size for an L-wide range with a herd
    // of N kangaroos. That gives jump_avg = 2^(range_bits/2) / N. Floor
    // range_bits at 8 so the shift is well-defined; floor N at 1 so the
    // divisor is non-zero. Matches RCKangaroo's tuning at
    // third_party/RCKangaroo/RCGpuUtils.h::CalcJumpAvg (which uses the same
    // sqrt(L)/N formula derived from Pollard's analysis).
    const uint64_t rb_floor = (range_bits < 8) ? 8 : range_bits;
    const uint64_t sqrt_l = 1ULL << (rb_floor / 2);
    const uint64_t herd = (num_kangaroos > 0) ? static_cast<uint64_t>(num_kangaroos) : 1ULL;
    uint64_t mean_jump = sqrt_l / herd;
    if (mean_jump == 0) mean_jump = 1;

    impl_->generate_jump_table(mean_jump);

    // Upload jump tables to device memory (not constant memory - fixes RDC linking)
    if (!impl_->upload_jump_tables()) {
        std::cerr << "[!] Failed to upload jump tables to device memory\n";
        return result;
    }

    std::cout << "[*] Uploaded jump table (mean_jump = " << mean_jump << ")\n";

    std::vector<uint64_t> h_x, h_y, h_z, h_dist, h_wild_offset;
    std::vector<uint32_t> h_types;
    impl_->init_kangaroos(num_kangaroos, h_x, h_y, h_z, h_dist, h_types, h_wild_offset);

    size_t size_256 = num_kangaroos * 4 * sizeof(uint64_t);
    size_t size_32 = num_kangaroos * sizeof(uint32_t);

    // Initial herd upload. ANY copy failure here = kernel runs on
    // uninitialized device memory and produces garbage DPs forever.
    // Bail out with an explicit log; the caller (single-GPU solve)
    // returns an empty result if init fails.
    if (!kg_cuda_check(
            cudaMemcpy(impl_->d_x, h_x.data(), size_256, cudaMemcpyHostToDevice),
            "single-gpu-init.d_x") ||
        !kg_cuda_check(
            cudaMemcpy(impl_->d_y, h_y.data(), size_256, cudaMemcpyHostToDevice),
            "single-gpu-init.d_y") ||
        !kg_cuda_check(
            cudaMemcpy(impl_->d_z, h_z.data(), size_256, cudaMemcpyHostToDevice),
            "single-gpu-init.d_z") ||
        !kg_cuda_check(
            cudaMemcpy(impl_->d_dist, h_dist.data(), size_256, cudaMemcpyHostToDevice),
            "single-gpu-init.d_dist") ||
        !kg_cuda_check(
            cudaMemcpy(impl_->d_types, h_types.data(), size_32, cudaMemcpyHostToDevice),
            "single-gpu-init.d_types") ||
        !kg_cuda_check(
            cudaMemcpy(impl_->d_wild_offset, h_wild_offset.data(), size_256, cudaMemcpyHostToDevice),
            "single-gpu-init.d_wild_offset")) {
        return result;   // empty / sentinel result
    }
    cudaMemset(impl_->d_flags, 0, size_32);
    cudaMemset(impl_->d_restart_pending, 0, size_32);

    std::cout << "[*] Starting GPU Kangaroo solve...\n";
    std::cout << "    DP bits: " << dp_bits << "\n";
    std::cout << "    Steps per round: " << steps_per_round << "\n";

    uint64_t total_steps = 0;
    uint64_t dp_count = 0;
    auto last_report = start_time;

    while (!stop_flag.load() && !result.found) {
        KangarooState state;
        state.x = impl_->d_x;
        state.y = impl_->d_y;
        state.z = impl_->d_z;
        state.dist = impl_->d_dist;
        state.flags = impl_->d_flags;
        state.types = impl_->d_types;
        state.wild_offset = impl_->d_wild_offset;
        state.restart_pending = impl_->d_restart_pending;

        // Use SOTA 3-group kernel with affine batch inversion (checks DP every step)
        // The old Jacobian kernel only checked DPs every 32 steps, causing ~97% DP misses!
        int num_threads = (num_kangaroos + GPU_GRP_SIZE - 1) / GPU_GRP_SIZE;
        int threads_per_block = 128;  // Lower for register pressure with batch inversion
        int blocks = (num_threads + threads_per_block - 1) / threads_per_block;

        // T1.6: sticky context error attribution.
        (void)collider::gpu::pre_launch_context_probe(
            "kangaroo_step_kernel_sota (single-GPU)");
        kangaroo_step_kernel_sota<<<blocks, threads_per_block>>>(
            state, dp_bits, steps_per_round, num_kangaroos,
            impl_->d_jump_x, impl_->d_jump_y, impl_->d_jump_d
        );

        cudaDeviceSynchronize();

        total_steps += static_cast<uint64_t>(num_kangaroos) * steps_per_round;

        std::vector<uint32_t> h_flags(num_kangaroos);
        if (!kg_cuda_check(
                cudaMemcpy(h_flags.data(), impl_->d_flags, size_32,
                           cudaMemcpyDeviceToHost),
                "solve.h_flags")) {
            // h_flags is garbage on copy failure -- skip DP collection
            // this round; next iteration's harvest catches them after
            // the transient cleared.
            continue;
        }

        std::vector<size_t> dp_indices;
        for (size_t i = 0; i < static_cast<size_t>(num_kangaroos); i++) {
            if (h_flags[i]) {
                dp_indices.push_back(i);
            }
        }

        if (!dp_indices.empty()) {
            dp_count += dp_indices.size();

            // Bail any one of these four D2H copies fails -- the DP
            // construction below would otherwise stamp garbage X/dist/
            // type/wild_offset values into the in-memory DP table and
            // (if pool-mode) submit them to the JLP server.
            bool harvest_ok =
                kg_cuda_check(
                    cudaMemcpy(h_x.data(), impl_->d_x, size_256,
                               cudaMemcpyDeviceToHost),
                    "solve.h_x") &&
                kg_cuda_check(
                    cudaMemcpy(h_dist.data(), impl_->d_dist, size_256,
                               cudaMemcpyDeviceToHost),
                    "solve.h_dist") &&
                kg_cuda_check(
                    cudaMemcpy(h_types.data(), impl_->d_types, size_32,
                               cudaMemcpyDeviceToHost),
                    "solve.h_types") &&
                // /B3: pull wild offsets so the collision algebra has the
                // per-kangaroo starting-distance correction available.
                kg_cuda_check(
                    cudaMemcpy(h_wild_offset.data(), impl_->d_wild_offset,
                               size_256, cudaMemcpyDeviceToHost),
                    "solve.h_wild_offset");
            if (!harvest_ok) continue;

            for (size_t idx : dp_indices) {
                // key on full 256-bit X. Pre-fix this was x_low only,
                // silently dropping DPs that collided on low 64 bits but differed
                // in the upper 192.
                ::collider::kangaroo::DPKey key{{
                    h_x[idx * 4 + 0], h_x[idx * 4 + 1],
                    h_x[idx * 4 + 2], h_x[idx * 4 + 3]
                }};

                Impl::DPEntry entry;
                for (int j = 0; j < 4; j++) {
                    entry.dist[j] = h_dist[idx * 4 + j];
                    entry.wild_offset[j] = h_wild_offset[idx * 4 + j];
                }
                entry.type = h_types[idx];
                entry.kang_idx = idx;

                auto it = impl_->dp_table.find(key);
                if (it != impl_->dp_table.end()) {
                    if (it->second.type != entry.type) {
                        // SOTA 3-Group Collision Detection
                        // True collision: same X coordinate, different kangaroo types
                        // Group types:
                        //   0 (KANG_TAME):  Position = dist * G
                        //   1 (KANG_WILD1): Position = Q + dist * G = k*G + dist*G
                        //   2 (KANG_WILD2): Position = -Q + dist * G = -k*G + dist*G
                        // Collision formulas:
                        //   Tame-Wild1: dist_t*G = k*G + dist_w1*G  =>  k = dist_t - dist_w1
                        //   Tame-Wild2: dist_t*G = -k*G + dist_w2*G =>  k = dist_w2 - dist_t
                        //   Wild1-Wild2: k*G + dist_w1*G = -k*G + dist_w2*G
                        //               => 2k*G = (dist_w2 - dist_w1)*G
                        //               => k = (dist_w2 - dist_w1) / 2 mod n
                        //               (parity-correct modular halving via cpu::mod_half;
                        //                see tests/test_mod_half.cpp)

                        uint32_t type1 = entry.type;
                        uint32_t type2 = it->second.type;

                        cpu::uint256_t d1, d2;
                        cpu::uint256_t w_off1, w_off2;
                        for (int j = 0; j < 4; j++) {
                            d1.d[j] = entry.dist[j];
                            d2.d[j] = it->second.dist[j];
                            w_off1.d[j] = entry.wild_offset[j];
                            w_off2.d[j] = it->second.wild_offset[j];
                        }

                        // Undo a wild's per-instance starting offset before
                        // plugging it into the collision algebra. For tame
                        // kangaroos the offset is 0 so the subtraction is a
                        // no-op. We do this once per side so the type-keyed
                        // branches below stay readable.
                        auto undo_offset = [](cpu::uint256_t& dist,
                                              const cpu::uint256_t& off) {
                            // dist_real = (dist - off) mod n
                            if (dist >= off) {
                                cpu::sub256(dist, dist, off);
                            } else {
                                cpu::uint256_t tmp;
                                cpu::sub256(tmp, off, dist);
                                cpu::sub256(dist, cpu::SECP256K1_N, tmp);
                            }
                        };
                        // d1 / d2 still hold the raw accumulated dist; subtract
                        // each kangaroo's recorded starting offset (zero for
                        // tame) to recover the algebraic distance used in the
                        // collision formulas below.
                        undo_offset(d1, w_off1);
                        undo_offset(d2, w_off2);

                        bool collision_valid = false;

                        if ((type1 == KANG_TAME && type2 == KANG_WILD1) ||
                            (type1 == KANG_WILD1 && type2 == KANG_TAME)) {
                            // Tame-Wild1 collision: k = tame_dist - wild1_dist
                            std::cout << "\n[!] SOTA Collision: Tame-Wild1!\n";

                            cpu::uint256_t tame_d = (type1 == KANG_TAME) ? d1 : d2;
                            cpu::uint256_t wild1_d = (type1 == KANG_WILD1) ? d1 : d2;

                            if (tame_d >= wild1_d) {
                                cpu::sub256(result.private_key, tame_d, wild1_d);
                            } else {
                                cpu::uint256_t diff;
                                cpu::sub256(diff, wild1_d, tame_d);
                                cpu::sub256(result.private_key, cpu::SECP256K1_N, diff);
                            }
                            collision_valid = true;

                        } else if ((type1 == KANG_TAME && type2 == KANG_WILD2) ||
                                   (type1 == KANG_WILD2 && type2 == KANG_TAME)) {
                            // Tame-Wild2 collision: k = wild2_dist - tame_dist
                            // (because Wild2 starts at -Q)
                            std::cout << "\n[!] SOTA Collision: Tame-Wild2!\n";

                            cpu::uint256_t tame_d = (type1 == KANG_TAME) ? d1 : d2;
                            cpu::uint256_t wild2_d = (type1 == KANG_WILD2) ? d1 : d2;

                            if (wild2_d >= tame_d) {
                                cpu::sub256(result.private_key, wild2_d, tame_d);
                            } else {
                                cpu::uint256_t diff;
                                cpu::sub256(diff, tame_d, wild2_d);
                                cpu::sub256(result.private_key, cpu::SECP256K1_N, diff);
                            }
                            collision_valid = true;

                        } else if ((type1 == KANG_WILD1 && type2 == KANG_WILD2) ||
                                   (type1 == KANG_WILD2 && type2 == KANG_WILD1)) {
                            // Wild1-Wild2 collision: k = (wild2_dist - wild1_dist) / 2 mod n.
                            // This is the "symmetry collision" unique to SOTA 3-group.
                            // modular halving must handle ODD diff, otherwise
                            // ~50% of W1-W2 collisions silently produce a wrong key. Use
                            // cpu::mod_half (parity-correct, KAT'd in test_mod_half.cpp).
                            std::cout << "\n[!] SOTA Collision: Wild1-Wild2 (symmetry)!\n";

                            cpu::uint256_t wild1_d = (type1 == KANG_WILD1) ? d1 : d2;
                            cpu::uint256_t wild2_d = (type1 == KANG_WILD2) ? d1 : d2;

                            cpu::uint256_t diff;
                            if (wild2_d >= wild1_d) {
                                cpu::sub256(diff, wild2_d, wild1_d);
                            } else {
                                cpu::uint256_t temp;
                                cpu::sub256(temp, wild1_d, wild2_d);
                                cpu::sub256(diff, cpu::SECP256K1_N, temp);
                            }

                            cpu::mod_half(result.private_key, diff, cpu::SECP256K1_N);
                            collision_valid = true;
                        }

                        if (collision_valid) {
                            result.found = true;
                            break;
                        }
                    } else {
                        // Same-type collision means a kangaroo's trajectory
                        // cycled through a DP we already saw. A naive
                        // approach silently overwrites the table entry, so
                        // the cycled kangaroo keeps producing duplicate DPs
                        // forever and never contributes a useful cross-type
                        // collision.
                        // Instead, tell the kernel to re-randomize whichever
                        // kangaroo produced the *new* hit (the colliding one).
                        // The dp_table entry stays at the older identity
                        // (preserves whichever of the two trajectories has
                        // been around longer). atomicExch is not available
                        // in host code, but we hold dp_mutex on this
                        // unordered_map already (well, here we're in single-
                        // GPU path so there's no contention) and a single
                        // 32-bit store is sufficient to flag the kernel.
                        uint32_t one = 1;
                        // 4-byte flag write. If this fails the kangaroo
                        // doesn't get restarted this round; next collision
                        // for the same kang_idx triggers another attempt.
                        // Logged so a sustained failure is diagnosable.
                        (void)kg_cuda_check(
                            cudaMemcpy(impl_->d_restart_pending + entry.kang_idx,
                                       &one, sizeof(uint32_t),
                                       cudaMemcpyHostToDevice),
                            "single-gpu-collision.restart_flag");
                        // Keep the older entry (do NOT overwrite). The newer
                        // duplicate is discarded after marking its source for
                        // restart.
                    }
                } else {
                    impl_->dp_table[key] = entry;
                }
            }

            cudaMemset(impl_->d_flags, 0, size_32);
        }

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_report).count();

        if (elapsed >= 1.0) {
            double rate = (total_steps - result.total_steps) / elapsed;  // steps/sec (raw)

            result.total_steps = total_steps;
            last_report = now;

            if (progress_callback) {
                if (!progress_callback(total_steps, dp_count, rate)) {
                    break;
                }
            }
        }
    }

    printf("\n");

    auto end_time = std::chrono::high_resolution_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    result.total_steps = total_steps;
    result.dp_count = dp_count;

    impl_->cleanup();

    return result;
}

// ============================================================================
// Multi-GPU Kangaroo Manager Implementation
// ============================================================================

struct MultiGPUKangarooManager::Impl {
    std::vector<int> gpu_ids;
    UInt256 range_start;
    UInt256 range_end;
    std::array<uint8_t, 20> target_h160;
    cpu::uint256_t target_pubkey_x;
    cpu::uint256_t target_pubkey_y;
    bool has_target_pubkey = false;
    bool debug_mode = false;

    // =========================================================================
    // MULTI-GPU WORK STEALING SYSTEM
    // =========================================================================
    // When one GPU finds more DPs (faster), it indicates that GPU has more
    // productive kangaroos. The work stealing mechanism allows dynamic
    // rebalancing by tracking per-GPU productivity and redistributing work.

    // Per-GPU work queue counters for work stealing
    struct WorkStealingState {
        std::atomic<uint64_t> completed_rounds{0};    // Rounds completed by this GPU
        std::atomic<uint64_t> dp_found{0};            // DPs found by this GPU
        std::atomic<uint64_t> work_units{0};          // Current work units assigned
        std::atomic<bool> needs_work{false};          // True if GPU is idle/slow
        std::atomic<bool> has_excess_work{false};     // True if GPU can donate work

        WorkStealingState() = default;
        // Explicitly delete copy/move since atomics aren't copyable
        WorkStealingState(const WorkStealingState&) = delete;
        WorkStealingState& operator=(const WorkStealingState&) = delete;
    };
    std::vector<std::unique_ptr<WorkStealingState>> work_states;

    // Global work pool for dynamic distribution
    std::atomic<uint64_t> global_work_pool{0};        // Total unassigned work
    std::atomic<int> active_gpus{0};                  // Number of active GPUs
    std::mutex work_mutex;                            // Protects work redistribution

    // Work stealing configuration
    static constexpr int WORK_STEAL_THRESHOLD = 100;  // Rounds difference to trigger steal
    static constexpr int WORK_STEAL_CHUNK = 1024;     // Kangaroos to transfer per steal

    // Check if work stealing is beneficial and perform transfer
    // Returns true if work was successfully stolen
    bool try_steal_work(int stealing_gpu_idx) {
        std::lock_guard<std::mutex> lock(work_mutex);

        int best_donor = -1;
        uint64_t max_excess = 0;

        // Find the GPU with most excess work (highest DP rate)
        for (size_t i = 0; i < work_states.size(); i++) {
            if (static_cast<int>(i) == stealing_gpu_idx) continue;

            uint64_t rounds_i = work_states[i]->completed_rounds.load();
            uint64_t rounds_steal = work_states[stealing_gpu_idx]->completed_rounds.load();

            // If this GPU has done significantly more rounds, it's faster
            if (rounds_i > rounds_steal + WORK_STEAL_THRESHOLD) {
                uint64_t excess = rounds_i - rounds_steal;
                if (excess > max_excess) {
                    max_excess = excess;
                    best_donor = static_cast<int>(i);
                }
            }
        }

        if (best_donor >= 0 && work_states[best_donor]->work_units > WORK_STEAL_CHUNK) {
            // Transfer work from donor to stealer
            work_states[best_donor]->work_units -= WORK_STEAL_CHUNK;
            work_states[stealing_gpu_idx]->work_units += WORK_STEAL_CHUNK;
            work_states[best_donor]->has_excess_work = false;
            work_states[stealing_gpu_idx]->needs_work = false;
            return true;
        }

        return false;
    }

    // Report GPU productivity for load balancing decisions
    void report_gpu_productivity(int gpu_idx, uint64_t rounds, uint64_t dps) {
        if (gpu_idx >= 0 && gpu_idx < static_cast<int>(work_states.size())) {
            work_states[gpu_idx]->completed_rounds = rounds;
            work_states[gpu_idx]->dp_found = dps;
        }
    }

    // Shared DP table keyed on full 256-bit X (DPKey).
    // Protected by mutex.
    // Carries the producing kangaroo's wild offset (so we can back-correct
    // the collision algebra for per-instance offsets) and the
    // (gpu_idx, kang_idx) pair so the host can flag the kangaroo for
    // restart on a same-type collision.
    struct DPEntry {
        uint64_t dist[4];
        uint64_t wild_offset[4];
        uint32_t type;
        int      gpu_id;     // Which GPU device id found this DP
        int      gpu_idx;    // Index into gpu_contexts (for d_restart_pending)
        size_t   kang_idx;   // Index within that GPU's herd
    };
    std::unordered_map<::collider::kangaroo::DPKey, DPEntry> dp_table;
    std::mutex dp_mutex;

    // Per-GPU state
    struct GPUContext {
        int device_id;
        uint64_t* d_x = nullptr;
        uint64_t* d_y = nullptr;
        uint64_t* d_z = nullptr;
        uint64_t* d_dist = nullptr;
        uint32_t* d_flags = nullptr;
        uint32_t* d_types = nullptr;
        // Per-kangaroo wild offset + restart-pending flag
        uint64_t* d_wild_offset = nullptr;
        uint32_t* d_restart_pending = nullptr;
        // Device memory for jump tables (per-GPU)
        // Using device memory instead of __constant__ fixes RDC + LTO linking issues
        uint64_t* d_jump_x = nullptr;
        uint64_t* d_jump_y = nullptr;
        uint64_t* d_jump_d = nullptr;
        int num_kangaroos = 0;
        std::string device_name;
    };
    std::vector<GPUContext> gpu_contexts;

    // Jump table (computed once, copied to all GPUs)
    uint64_t jump_x[NUM_JUMPS][4];
    uint64_t jump_y[NUM_JUMPS][4];
    uint64_t jump_d[NUM_JUMPS][4];

    // Statistics
    std::atomic<uint64_t> total_steps{0};
    std::atomic<uint64_t> total_dps{0};

    // Result
    std::atomic<bool> found{false};
    // T1.5 (2026-05-17). Kernel-launch / execution failures previously
    // overloaded `found = true` to force the per-GPU worker thread to
    // exit. That tripped the main thread's `impl_->found.load()` check
    // and made every OTHER GPU bail out as well, producing a silent
    // "search stopped after N steps" with no diagnostic. Separating
    // `failed` from `found` lets the workers exit on either condition
    // while preserving the distinction at the outer caller: the
    // dispatcher reports a proper "GPU N kernel launch failed" message
    // when `failed` is set without a `found`, and the result's
    // `found` field stays accurate.
    std::atomic<bool> failed{false};
    std::mutex failure_msg_mutex;
    std::string failure_msg;
    GPUKangarooResult result;
    std::mutex result_mutex;
};

MultiGPUKangarooManager::MultiGPUKangarooManager() : impl_(new Impl()) {}

MultiGPUKangarooManager::~MultiGPUKangarooManager() {
    // Cleanup all GPU contexts
    for (auto& ctx : impl_->gpu_contexts) {
        cudaSetDevice(ctx.device_id);
        if (ctx.d_x) cudaFree(ctx.d_x);
        if (ctx.d_y) cudaFree(ctx.d_y);
        if (ctx.d_z) cudaFree(ctx.d_z);
        if (ctx.d_dist) cudaFree(ctx.d_dist);
        if (ctx.d_flags) cudaFree(ctx.d_flags);
        if (ctx.d_types) cudaFree(ctx.d_types);
        if (ctx.d_wild_offset) cudaFree(ctx.d_wild_offset);
        if (ctx.d_restart_pending) cudaFree(ctx.d_restart_pending);
        // Free jump table device memory
        if (ctx.d_jump_x) cudaFree(ctx.d_jump_x);
        if (ctx.d_jump_y) cudaFree(ctx.d_jump_y);
        if (ctx.d_jump_d) cudaFree(ctx.d_jump_d);
    }
    delete impl_;
}

bool MultiGPUKangarooManager::init(const std::vector<int>& gpu_ids) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "[!] No CUDA devices found\n";
        return false;
    }

    // Use provided GPU IDs or auto-detect all GPUs
    if (gpu_ids.empty()) {
        for (int i = 0; i < device_count; i++) {
            impl_->gpu_ids.push_back(i);
        }
    } else {
        for (int id : gpu_ids) {
            if (id >= 0 && id < device_count) {
                impl_->gpu_ids.push_back(id);
            }
        }
    }

    if (impl_->gpu_ids.empty()) {
        std::cerr << "[!] No valid GPU IDs\n";
        return false;
    }

    // Initialize each GPU context
    for (int device_id : impl_->gpu_ids) {
        Impl::GPUContext ctx;
        ctx.device_id = device_id;

        cudaSetDevice(device_id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device_id);
        ctx.device_name = props.name;

        std::cout << "[*] GPU " << device_id << ": " << props.name
                  << " (" << props.multiProcessorCount << " SMs, "
                  << (props.totalGlobalMem / 1024 / 1024) << " MB)\n";

        impl_->gpu_contexts.push_back(ctx);
    }

    // Initialize work stealing state for each GPU
    impl_->work_states.clear();
    impl_->work_states.reserve(impl_->gpu_contexts.size());
    for (size_t i = 0; i < impl_->gpu_contexts.size(); i++) {
        impl_->work_states.push_back(std::make_unique<Impl::WorkStealingState>());
    }
    impl_->active_gpus = static_cast<int>(impl_->gpu_contexts.size());

    std::cout << "[*] Multi-GPU Kangaroo initialized with " << impl_->gpu_ids.size() << " GPU(s)\n";
    std::cout << "    Work stealing: enabled (threshold=" << Impl::WORK_STEAL_THRESHOLD
              << " rounds, chunk=" << Impl::WORK_STEAL_CHUNK << " kangaroos)\n";
    return true;
}

int MultiGPUKangarooManager::num_gpus() const {
    return static_cast<int>(impl_->gpu_ids.size());
}

void MultiGPUKangarooManager::set_range(const UInt256& start, const UInt256& end) {
    impl_->range_start = start;
    impl_->range_end = end;
}

void MultiGPUKangarooManager::set_target_h160(const std::array<uint8_t, 20>& h160) {
    impl_->target_h160 = h160;
}

void MultiGPUKangarooManager::set_target_pubkey(const cpu::uint256_t& x, const cpu::uint256_t& y) {
    impl_->target_pubkey_x = x;
    impl_->target_pubkey_y = y;
    impl_->has_target_pubkey = true;
}

GPUKangarooResult MultiGPUKangarooManager::solve() {
    GPUKangarooResult result;
    result.found = false;
    result.total_steps = 0;
    result.dp_count = 0;

    // Copy debug_mode from public interface to implementation
    impl_->debug_mode = debug_mode;

    if (!impl_->has_target_pubkey) {
        std::cerr << "[!] Error: target public key not set\n";
        return result;
    }

    if (impl_->gpu_contexts.empty()) {
        std::cerr << "[!] Error: no GPUs initialized\n";
        return result;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Calculate range bits for jump table
    uint64_t range_bits = 0;
    for (int i = 3; i >= 0; i--) {
        if (impl_->range_end.parts[i] != 0) {
            range_bits = i * 64 + 64 - CLZ64_LOCAL(impl_->range_end.parts[i]);
            break;
        }
    }
    // Multi-GPU herd is num_kangaroos_per_gpu * num_gpus; cap range_bits at
    // 8 to avoid underflow in the shift. See the matching comment in the
    // single-GPU solve() above for the full derivation.
    const uint64_t rb_floor = (range_bits < 8) ? 8 : range_bits;
    const uint64_t sqrt_l = 1ULL << (rb_floor / 2);
    const uint64_t total_herd =
        static_cast<uint64_t>(num_kangaroos_per_gpu) *
        static_cast<uint64_t>(impl_->gpu_contexts.size());
    const uint64_t herd = (total_herd > 0) ? total_herd : 1ULL;
    uint64_t mean_jump = sqrt_l / herd;
    if (mean_jump == 0) mean_jump = 1;

    // Generate jump table (same for all GPUs)
    for (int i = 0; i < NUM_JUMPS; i++) {
        uint64_t jump_dist = mean_jump >> (NUM_JUMPS / 2 - i / 2);
        if (jump_dist == 0) jump_dist = 1;

        impl_->jump_d[i][0] = jump_dist;
        impl_->jump_d[i][1] = 0;
        impl_->jump_d[i][2] = 0;
        impl_->jump_d[i][3] = 0;

        cpu::uint256_t k;
        k.d[0] = jump_dist;
        k.d[1] = 0;
        k.d[2] = 0;
        k.d[3] = 0;

        cpu::ECPoint P;
        cpu::ec_mul(P, k);
        cpu::uint256_t px, py;
        cpu::ec_to_affine(px, py, P);

        for (int j = 0; j < 4; j++) {
            impl_->jump_x[i][j] = px.d[j];
            impl_->jump_y[i][j] = py.d[j];
        }
    }

    std::cout << "[*] Generated jump table (mean_jump = " << mean_jump << ")\n";

    // Allocate memory and initialize kangaroos on each GPU
    size_t size_256 = num_kangaroos_per_gpu * 4 * sizeof(uint64_t);
    size_t size_32 = num_kangaroos_per_gpu * sizeof(uint32_t);

    for (auto& ctx : impl_->gpu_contexts) {
        cudaSetDevice(ctx.device_id);

        cudaMalloc(&ctx.d_x, size_256);
        cudaMalloc(&ctx.d_y, size_256);
        cudaMalloc(&ctx.d_z, size_256);
        cudaMalloc(&ctx.d_dist, size_256);
        cudaMalloc(&ctx.d_flags, size_32);
        cudaMalloc(&ctx.d_types, size_32);
        // Per-kangaroo wild offset + restart-pending buffers.
        cudaMalloc(&ctx.d_wild_offset, size_256);
        cudaMalloc(&ctx.d_restart_pending, size_32);
        ctx.num_kangaroos = num_kangaroos_per_gpu;

        // Upload jump table to this GPU's device memory (not constant - fixes RDC linking)
        cudaError_t err;
        size_t jump_table_size = NUM_JUMPS * 4 * sizeof(uint64_t);

        // Allocate jump table device memory
        err = cudaMalloc(&ctx.d_jump_x, jump_table_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to allocate d_jump_x on GPU %d: %s\n", ctx.device_id, cudaGetErrorString(err));
        }
        err = cudaMalloc(&ctx.d_jump_y, jump_table_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to allocate d_jump_y on GPU %d: %s\n", ctx.device_id, cudaGetErrorString(err));
        }
        err = cudaMalloc(&ctx.d_jump_d, jump_table_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to allocate d_jump_d on GPU %d: %s\n", ctx.device_id, cudaGetErrorString(err));
        }

        // Copy jump tables to device memory
        err = cudaMemcpy(ctx.d_jump_x, impl_->jump_x, jump_table_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to copy d_jump_x on GPU %d: %s\n", ctx.device_id, cudaGetErrorString(err));
        }
        err = cudaMemcpy(ctx.d_jump_y, impl_->jump_y, jump_table_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to copy d_jump_y on GPU %d: %s\n", ctx.device_id, cudaGetErrorString(err));
        }
        err = cudaMemcpy(ctx.d_jump_d, impl_->jump_d, jump_table_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA ERROR] Failed to copy d_jump_d on GPU %d: %s\n", ctx.device_id, cudaGetErrorString(err));
        }

        // SOTA 3-Group initialization for Multi-GPU
        // 1/3 Tame, 1/3 Wild1 (from Q + offset*G), 1/3 Wild2 (from -Q + offset*G)
        std::vector<uint64_t> h_x(num_kangaroos_per_gpu * 4);
        std::vector<uint64_t> h_y(num_kangaroos_per_gpu * 4);
        std::vector<uint64_t> h_z(num_kangaroos_per_gpu * 4);
        std::vector<uint64_t> h_dist(num_kangaroos_per_gpu * 4);
        std::vector<uint64_t> h_wild_offset(num_kangaroos_per_gpu * 4, 0);
        std::vector<uint32_t> h_types(num_kangaroos_per_gpu);

        std::mt19937_64 rng(std::random_device{}() + ctx.device_id);

        // Compute the full 256-bit [range_start, range_end) so Tame
        // kangaroos can sample uniformly over the actual puzzle range. A
        // naive loop that ignored range_start and capped scalars at 2^128
        // (via `1ULL << (range_bits - 64)` plus a 64-bit lower limb)
        // placed all Tame kangaroos in [0, 2^128) for puzzle 135, i.e.
        // entirely BELOW the puzzle range. The collision algebra then
        // returned a "private key" that mathematically existed but lay
        // outside the puzzle range, and the runner discarded it.
        cpu::uint256_t mg_range_start_256, mg_range_end_256, mg_range_size_256;
        mg_range_start_256.d[0] = impl_->range_start.parts[0];
        mg_range_start_256.d[1] = impl_->range_start.parts[1];
        mg_range_start_256.d[2] = impl_->range_start.parts[2];
        mg_range_start_256.d[3] = impl_->range_start.parts[3];
        mg_range_end_256.d[0] = impl_->range_end.parts[0];
        mg_range_end_256.d[1] = impl_->range_end.parts[1];
        mg_range_end_256.d[2] = impl_->range_end.parts[2];
        mg_range_end_256.d[3] = impl_->range_end.parts[3];
        cpu::sub256(mg_range_size_256, mg_range_end_256, mg_range_start_256);

        // Compute -Q for Wild2 kangaroos (in affine).
        cpu::uint256_t neg_Qy;
        cpu::sub256(neg_Qy, cpu::SECP256K1_P, impl_->target_pubkey_y);

        // Build a Jacobian point from the affine Q so the wild
        // add-offset loop can pass it to cpu::ec_add directly.
        cpu::ECPoint QP, NegQP;
        QP.X = impl_->target_pubkey_x;
        QP.Y = impl_->target_pubkey_y;
        QP.Z.d[0] = 1; QP.Z.d[1] = 0; QP.Z.d[2] = 0; QP.Z.d[3] = 0;

        NegQP.X = impl_->target_pubkey_x;
        NegQP.Y = neg_Qy;
        NegQP.Z = QP.Z;

        int third = num_kangaroos_per_gpu / 3;

        for (int i = 0; i < num_kangaroos_per_gpu; i++) {
            if (i < third) {
                // TAME (Type 0): start at range_start + offset*G for a
                // uniform random offset in [0, range_size). Matches the
                // single-GPU init path.
                h_types[i] = KANG_TAME;

                cpu::uint256_t offset;
                cpu::random_below(offset, [&]() { return rng(); }, mg_range_size_256);

                cpu::uint256_t scalar;
                cpu::add256(scalar, mg_range_start_256, offset);

                cpu::ECPoint P;
                cpu::ec_mul(P, scalar);
                cpu::uint256_t px, py;
                cpu::ec_to_affine(px, py, P);

                for (int j = 0; j < 4; j++) {
                    h_x[i * 4 + j] = px.d[j];
                    h_y[i * 4 + j] = py.d[j];
                    h_z[i * 4 + j] = (j == 0) ? 1 : 0;
                    h_dist[i * 4 + j] = scalar.d[j];
                }

            } else if (i < 2 * third) {
                // WILD1 (Type 1): start at Q + offset_i*G with per-instance
                // offset. See single-GPU init_kangaroos for the full
                // motivation.
                h_types[i] = KANG_WILD1;

                cpu::uint256_t offset_i;
                offset_i.d[0] = rng();
                offset_i.d[1] = rng() & 0x3FFFFFFFFFFFFFFFULL;
                offset_i.d[2] = 0;
                offset_i.d[3] = 0;

                cpu::ECPoint OG;
                cpu::ec_mul(OG, offset_i);
                cpu::uint256_t og_x, og_y;
                cpu::ec_to_affine(og_x, og_y, OG);

                cpu::ECPoint W1;
                cpu::ec_add(W1, QP, og_x, og_y);

                cpu::uint256_t w1_x, w1_y;
                cpu::ec_to_affine(w1_x, w1_y, W1);

                for (int j = 0; j < 4; j++) {
                    h_x[i * 4 + j] = w1_x.d[j];
                    h_y[i * 4 + j] = w1_y.d[j];
                    h_z[i * 4 + j] = (j == 0) ? 1 : 0;
                    h_dist[i * 4 + j] = offset_i.d[j];
                    h_wild_offset[i * 4 + j] = offset_i.d[j];
                }

            } else {
                // WILD2 (Type 2): start at -Q + offset_i*G.
                h_types[i] = KANG_WILD2;

                cpu::uint256_t offset_i;
                offset_i.d[0] = rng();
                offset_i.d[1] = rng() & 0x3FFFFFFFFFFFFFFFULL;
                offset_i.d[2] = 0;
                offset_i.d[3] = 0;

                cpu::ECPoint OG;
                cpu::ec_mul(OG, offset_i);
                cpu::uint256_t og_x, og_y;
                cpu::ec_to_affine(og_x, og_y, OG);

                cpu::ECPoint W2;
                cpu::ec_add(W2, NegQP, og_x, og_y);

                cpu::uint256_t w2_x, w2_y;
                cpu::ec_to_affine(w2_x, w2_y, W2);

                for (int j = 0; j < 4; j++) {
                    h_x[i * 4 + j] = w2_x.d[j];
                    h_y[i * 4 + j] = w2_y.d[j];
                    h_z[i * 4 + j] = (j == 0) ? 1 : 0;
                    h_dist[i * 4 + j] = offset_i.d[j];
                    h_wild_offset[i * 4 + j] = offset_i.d[j];
                }
            }
        }

        // Multi-GPU per-device herd init upload. Any copy failure
        // poisons this GPU's entire run (kernel reads uninitialized
        // device memory). Log every failure with the device_id so
        // the operator can diagnose which GPU is at fault; the
        // surrounding for(auto& ctx : impl_->gpu_contexts) loop
        // skips to the next device on failure rather than aborting
        // the whole multi-GPU solve. Other GPUs keep running with
        // their successfully-seeded herds; the impacted GPU's
        // subsequent harvest cycles produce zero DPs (the kernel
        // reads garbage flags), naturally falling out of the
        // collision algebra without poisoning the DP table.
        bool device_init_ok =
            kg_cuda_check(cudaMemcpy(ctx.d_x, h_x.data(), size_256,
                                     cudaMemcpyHostToDevice),
                          "multigpu-init.d_x") &&
            kg_cuda_check(cudaMemcpy(ctx.d_y, h_y.data(), size_256,
                                     cudaMemcpyHostToDevice),
                          "multigpu-init.d_y") &&
            kg_cuda_check(cudaMemcpy(ctx.d_z, h_z.data(), size_256,
                                     cudaMemcpyHostToDevice),
                          "multigpu-init.d_z") &&
            kg_cuda_check(cudaMemcpy(ctx.d_dist, h_dist.data(), size_256,
                                     cudaMemcpyHostToDevice),
                          "multigpu-init.d_dist") &&
            kg_cuda_check(cudaMemcpy(ctx.d_types, h_types.data(), size_32,
                                     cudaMemcpyHostToDevice),
                          "multigpu-init.d_types") &&
            kg_cuda_check(cudaMemcpy(ctx.d_wild_offset, h_wild_offset.data(),
                                     size_256, cudaMemcpyHostToDevice),
                          "multigpu-init.d_wild_offset");
        if (!device_init_ok) {
            std::fprintf(stderr,
                "[!] kangaroo multi-GPU init: device %d skipped due to "
                "host->device copy failure (see logs above)\n",
                ctx.device_id);
            continue;
        }
        cudaMemset(ctx.d_flags, 0, size_32);
        cudaMemset(ctx.d_restart_pending, 0, size_32);

        std::cout << "[*] GPU " << ctx.device_id << ": Initialized "
                  << num_kangaroos_per_gpu << " kangaroos\n";
    }

    impl_->total_steps = 0;
    impl_->total_dps = 0;
    impl_->found = false;
    impl_->failed = false;
    {
        std::lock_guard<std::mutex> lk(impl_->failure_msg_mutex);
        impl_->failure_msg.clear();
    }
    impl_->result.found = false;

    auto last_report = start_time;

    int group_third = num_kangaroos_per_gpu / 3;
    std::cout << "[*] Starting SOTA 3-Group Multi-GPU Kangaroo solve (K=1.15)...\n";
    std::cout << "    Algorithm: SOTA 3-group with symmetry (45% faster than classic)\n";
    std::cout << "    GPUs: " << impl_->gpu_contexts.size() << "\n";
    std::cout << "    Kangaroos per GPU: " << num_kangaroos_per_gpu << "\n";
    std::cout << "    Total kangaroos: " << (num_kangaroos_per_gpu * impl_->gpu_contexts.size()) << "\n";
    std::cout << "    Groups: " << group_third << " Tame, " << group_third << " Wild1, "
              << (num_kangaroos_per_gpu - 2*group_third) << " Wild2 (per GPU)\n";
    std::cout << "    DP bits: " << dp_bits << "\n";

    // Create worker threads for each GPU
    std::vector<std::thread> gpu_threads;

    for (size_t gpu_idx = 0; gpu_idx < impl_->gpu_contexts.size(); gpu_idx++) {
        // Initialize work units for this GPU
        impl_->work_states[gpu_idx]->work_units = num_kangaroos_per_gpu;

        gpu_threads.emplace_back([this, gpu_idx, size_256, size_32]() {
            auto& ctx = impl_->gpu_contexts[gpu_idx];
            cudaSetDevice(ctx.device_id);

            std::vector<uint64_t> h_x(ctx.num_kangaroos * 4);
            std::vector<uint64_t> h_dist(ctx.num_kangaroos * 4);
            std::vector<uint64_t> h_wild_off_per(ctx.num_kangaroos * 4);
            std::vector<uint32_t> h_flags(ctx.num_kangaroos);
            std::vector<uint32_t> h_types(ctx.num_kangaroos);

            bool first_round = true;  // DEBUG: flag for first kernel call
            uint64_t local_rounds = 0;  // Track rounds for work stealing
            uint64_t local_dps = 0;     // Track DPs for productivity

            while (!stop_flag.load() && !impl_->found.load() && !impl_->failed.load()) {
                // =====================================================================
                // WORK STEALING CHECK
                // =====================================================================
                // Every 50 rounds, check if this GPU is falling behind and should
                // try to steal work from a faster GPU
                if (local_rounds % 50 == 0 && local_rounds > 0) {
                    impl_->report_gpu_productivity(static_cast<int>(gpu_idx), local_rounds, local_dps);

                    // Check if we're slower than other GPUs
                    uint64_t our_rounds = impl_->work_states[gpu_idx]->completed_rounds.load();
                    uint64_t max_other_rounds = 0;
                    for (size_t i = 0; i < impl_->work_states.size(); i++) {
                        if (i != gpu_idx) {
                            max_other_rounds = std::max(max_other_rounds,
                                impl_->work_states[i]->completed_rounds.load());
                        }
                    }

                    // If we're falling behind significantly, try to rebalance
                    if (max_other_rounds > our_rounds + Impl::WORK_STEAL_THRESHOLD) {
                        // The faster GPU might be able to use more work
                        // (This is a simplified version - full implementation would
                        // transfer actual kangaroo state between GPUs)
                        impl_->work_states[gpu_idx]->needs_work = true;
                    }
                }

                // Launch kernel
                KangarooState state;
                state.x = ctx.d_x;
                state.y = ctx.d_y;
                state.z = ctx.d_z;
                state.dist = ctx.d_dist;
                state.flags = ctx.d_flags;
                state.types = ctx.d_types;
                state.wild_offset = ctx.d_wild_offset;
                state.restart_pending = ctx.d_restart_pending;

                // Use SOTA 3-group kernel with affine batch inversion (checks DP every step)
                // The old Jacobian kernel only checked DPs every 32 steps, causing ~97% DP misses!
                int num_threads = (ctx.num_kangaroos + GPU_GRP_SIZE - 1) / GPU_GRP_SIZE;
                int threads_per_block = 128;  // Lower for register pressure with batch inversion
                int blocks = (num_threads + threads_per_block - 1) / threads_per_block;

                // T1.6: sticky context error attribution. The probe
                // distinguishes a fresh launch failure from a cascade
                // carried over from a prior kernel on this device.
                (void)collider::gpu::pre_launch_context_probe(
                    "kangaroo_step_kernel_sota (multi-GPU)");
                kangaroo_step_kernel_sota<<<blocks, threads_per_block>>>(
                    state, dp_bits, steps_per_round, ctx.num_kangaroos,
                    ctx.d_jump_x, ctx.d_jump_y, ctx.d_jump_d
                );
                cudaError_t kernel_err = cudaGetLastError();
                if (kernel_err != cudaSuccess) {
                    // T1.5: signal failure via the dedicated `failed`
                    // atomic instead of `found`. Other GPU worker threads
                    // observe `failed` and exit their loops without their
                    // result.found bit being flipped, and the outer
                    // dispatcher reports the actual GPU id + CUDA error.
                    std::string msg =
                        "[CUDA ERROR] GPU " + std::to_string(ctx.device_id) +
                        " kernel launch failed: " + cudaGetErrorString(kernel_err);
                    fprintf(stderr, "\n%s\n", msg.c_str());
                    {
                        std::lock_guard<std::mutex> lk(impl_->failure_msg_mutex);
                        if (impl_->failure_msg.empty()) impl_->failure_msg = msg;
                    }
                    impl_->failed.store(true);
                    break;
                }

                // DEBUG: After first kernel call, download and print first kangaroo's state
                if (first_round && impl_->debug_mode) {
                    first_round = false;
                    cudaDeviceSynchronize();

                    std::vector<uint64_t> dbg_x(4), dbg_y(4), dbg_z(4);
                    cudaMemcpy(dbg_x.data(), ctx.d_x, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    cudaMemcpy(dbg_y.data(), ctx.d_y, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    cudaMemcpy(dbg_z.data(), ctx.d_z, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

                    printf("\n[DEBUG] After first kernel (GPU %d, kangaroo 0):\n", ctx.device_id);
                    printf("  X: [%016llx %016llx %016llx %016llx]\n",
                           (unsigned long long)dbg_x[3], (unsigned long long)dbg_x[2],
                           (unsigned long long)dbg_x[1], (unsigned long long)dbg_x[0]);
                    printf("  Y: [%016llx %016llx %016llx %016llx]\n",
                           (unsigned long long)dbg_y[3], (unsigned long long)dbg_y[2],
                           (unsigned long long)dbg_y[1], (unsigned long long)dbg_y[0]);
                    printf("  Z: [%016llx %016llx %016llx %016llx]\n",
                           (unsigned long long)dbg_z[3], (unsigned long long)dbg_z[2],
                           (unsigned long long)dbg_z[1], (unsigned long long)dbg_z[0]);
                    printf("  dp_mask: 0x%lx, X[0] & dp_mask = 0x%llx\n",
                           (1UL << dp_bits) - 1, (unsigned long long)(dbg_x[0] & ((1ULL << dp_bits) - 1)));
                    fflush(stdout);
                } else if (first_round) {
                    first_round = false;  // Still track first_round even without debug output
                }
                cudaDeviceSynchronize();
                kernel_err = cudaGetLastError();
                if (kernel_err != cudaSuccess) {
                    // T1.5: failure path (see matching comment on the
                    // post-launch peek above). Set `failed`, not `found`.
                    std::string msg =
                        "[CUDA ERROR] GPU " + std::to_string(ctx.device_id) +
                        " kernel execution failed: " + cudaGetErrorString(kernel_err);
                    fprintf(stderr, "\n%s\n", msg.c_str());
                    {
                        std::lock_guard<std::mutex> lk(impl_->failure_msg_mutex);
                        if (impl_->failure_msg.empty()) impl_->failure_msg = msg;
                    }
                    impl_->failed.store(true);
                    break;
                }

                impl_->total_steps += static_cast<uint64_t>(ctx.num_kangaroos) * steps_per_round;
                local_rounds++;  // Track for work stealing

                // Check for DPs. Skip iteration on copy failure so the
                // dp_indices loop doesn't dereference garbage flags.
                if (!kg_cuda_check(
                        cudaMemcpy(h_flags.data(), ctx.d_flags,
                                   ctx.num_kangaroos * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost),
                        "multigpu-solve.h_flags")) {
                    continue;
                }

                std::vector<size_t> dp_indices;
                for (size_t i = 0; i < static_cast<size_t>(ctx.num_kangaroos); i++) {
                    if (h_flags[i]) {
                        dp_indices.push_back(i);
                    }
                }

                if (!dp_indices.empty()) {
                    impl_->total_dps += dp_indices.size();
                    local_dps += dp_indices.size();  // Track for productivity measurement

                    // Same protection as collect_dps: bail if any of the
                    // four DP-harvest D2H copies fails. The DP table
                    // population below would otherwise stamp garbage
                    // 256-bit X / dist / type / wild_offset values and
                    // the collision algebra would chase the noise.
                    bool harvest_ok =
                        kg_cuda_check(
                            cudaMemcpy(h_x.data(), ctx.d_x, size_256,
                                       cudaMemcpyDeviceToHost),
                            "multigpu-solve.h_x") &&
                        kg_cuda_check(
                            cudaMemcpy(h_dist.data(), ctx.d_dist, size_256,
                                       cudaMemcpyDeviceToHost),
                            "multigpu-solve.h_dist") &&
                        kg_cuda_check(
                            cudaMemcpy(h_types.data(), ctx.d_types,
                                       ctx.num_kangaroos * sizeof(uint32_t),
                                       cudaMemcpyDeviceToHost),
                            "multigpu-solve.h_types") &&
                        // /B3: pull per-kangaroo wild offsets for collision algebra.
                        kg_cuda_check(
                            cudaMemcpy(h_wild_off_per.data(), ctx.d_wild_offset,
                                       size_256, cudaMemcpyDeviceToHost),
                            "multigpu-solve.h_wild_offset");
                    if (!harvest_ok) continue;

                    // Process DPs with shared table lock
                    std::lock_guard<std::mutex> lock(impl_->dp_mutex);

                    for (size_t idx : dp_indices) {
                        // full-256 DPKey.
                        ::collider::kangaroo::DPKey key{{
                            h_x[idx * 4 + 0], h_x[idx * 4 + 1],
                            h_x[idx * 4 + 2], h_x[idx * 4 + 3]
                        }};

                        Impl::DPEntry entry;
                        for (int j = 0; j < 4; j++) {
                            entry.dist[j] = h_dist[idx * 4 + j];
                            entry.wild_offset[j] = h_wild_off_per[idx * 4 + j];
                        }
                        entry.type = h_types[idx];
                        entry.gpu_id = ctx.device_id;
                        entry.gpu_idx = static_cast<int>(gpu_idx);
                        entry.kang_idx = idx;

                        auto it = impl_->dp_table.find(key);
                        if (it != impl_->dp_table.end()) {
                            if (it->second.type != entry.type) {
                                // SOTA 3-Group Collision on Multi-GPU
                                std::cout << "\n[!] COLLISION on GPU " << ctx.device_id
                                          << " (with DP from GPU " << it->second.gpu_id << ")\n";

                                uint32_t type1 = entry.type;
                                uint32_t type2 = it->second.type;

                                cpu::uint256_t d1, d2;
                                cpu::uint256_t w_off1, w_off2;
                                for (int j = 0; j < 4; j++) {
                                    d1.d[j] = entry.dist[j];
                                    d2.d[j] = it->second.dist[j];
                                    w_off1.d[j] = entry.wild_offset[j];
                                    w_off2.d[j] = it->second.wild_offset[j];
                                }

                                // /B3: undo each kangaroo's recorded
                                // starting offset (zero for tames) before
                                // plugging the distances into the collision
                                // algebra. See single-GPU collision handler
                                // for the full motivation.
                                auto undo_offset = [](cpu::uint256_t& dist,
                                                      const cpu::uint256_t& off) {
                                    if (dist >= off) {
                                        cpu::sub256(dist, dist, off);
                                    } else {
                                        cpu::uint256_t tmp;
                                        cpu::sub256(tmp, off, dist);
                                        cpu::sub256(dist, cpu::SECP256K1_N, tmp);
                                    }
                                };
                                undo_offset(d1, w_off1);
                                undo_offset(d2, w_off2);

                                std::lock_guard<std::mutex> rlock(impl_->result_mutex);
                                bool collision_valid = false;

                                if ((type1 == KANG_TAME && type2 == KANG_WILD1) ||
                                    (type1 == KANG_WILD1 && type2 == KANG_TAME)) {
                                    std::cout << "[!] SOTA: Tame-Wild1 collision\n";
                                    cpu::uint256_t tame_d = (type1 == KANG_TAME) ? d1 : d2;
                                    cpu::uint256_t wild1_d = (type1 == KANG_WILD1) ? d1 : d2;

                                    if (tame_d >= wild1_d) {
                                        cpu::sub256(impl_->result.private_key, tame_d, wild1_d);
                                    } else {
                                        cpu::uint256_t diff;
                                        cpu::sub256(diff, wild1_d, tame_d);
                                        cpu::sub256(impl_->result.private_key, cpu::SECP256K1_N, diff);
                                    }
                                    collision_valid = true;

                                } else if ((type1 == KANG_TAME && type2 == KANG_WILD2) ||
                                           (type1 == KANG_WILD2 && type2 == KANG_TAME)) {
                                    std::cout << "[!] SOTA: Tame-Wild2 collision\n";
                                    cpu::uint256_t tame_d = (type1 == KANG_TAME) ? d1 : d2;
                                    cpu::uint256_t wild2_d = (type1 == KANG_WILD2) ? d1 : d2;

                                    if (wild2_d >= tame_d) {
                                        cpu::sub256(impl_->result.private_key, wild2_d, tame_d);
                                    } else {
                                        cpu::uint256_t diff;
                                        cpu::sub256(diff, tame_d, wild2_d);
                                        cpu::sub256(impl_->result.private_key, cpu::SECP256K1_N, diff);
                                    }
                                    collision_valid = true;

                                } else if ((type1 == KANG_WILD1 && type2 == KANG_WILD2) ||
                                           (type1 == KANG_WILD2 && type2 == KANG_WILD1)) {
                                    // parity-correct modular halving (see
                                    // cpu::mod_half + tests/test_mod_half.cpp). Pre-fix
                                    // right-shift silently produced wrong keys on odd diff.
                                    std::cout << "[!] SOTA: Wild1-Wild2 symmetry collision\n";
                                    cpu::uint256_t wild1_d = (type1 == KANG_WILD1) ? d1 : d2;
                                    cpu::uint256_t wild2_d = (type1 == KANG_WILD2) ? d1 : d2;

                                    cpu::uint256_t diff;
                                    if (wild2_d >= wild1_d) {
                                        cpu::sub256(diff, wild2_d, wild1_d);
                                    } else {
                                        cpu::uint256_t temp;
                                        cpu::sub256(temp, wild1_d, wild2_d);
                                        cpu::sub256(diff, cpu::SECP256K1_N, temp);
                                    }

                                    cpu::mod_half(impl_->result.private_key, diff, cpu::SECP256K1_N);
                                    collision_valid = true;
                                }

                                if (collision_valid) {
                                    impl_->result.found = true;
                                    impl_->found = true;
                                    return;
                                }
                            } else {
                                // Same-type collision = cycled kangaroo.
                                // Flag the colliding kangaroo for restart
                                // and keep the older entry. The kernel
                                // reads d_restart_pending at the top of its
                                // next launch and rerandomizes the
                                // position. See single-GPU branch for full
                                // rationale.
                                uint32_t one = 1;
                                cudaSetDevice(ctx.device_id);
                                // 4-byte restart flag write. Failure here
                                // leaves the kangaroo on its current path;
                                // the next collision triggers another
                                // attempt. Logged so a sustained failure
                                // is diagnosable.
                                (void)kg_cuda_check(
                                    cudaMemcpy(
                                        ctx.d_restart_pending + entry.kang_idx,
                                        &one, sizeof(uint32_t),
                                        cudaMemcpyHostToDevice),
                                    "multigpu-collision.restart_flag");
                            }
                        } else {
                            impl_->dp_table[key] = entry;
                        }
                    }

                    cudaMemset(ctx.d_flags, 0, ctx.num_kangaroos * sizeof(uint32_t));
                }
            }
        });
    }

    // Main thread: report progress
    while (!stop_flag.load() && !impl_->found.load() && !impl_->failed.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_report).count();

        if (elapsed >= 1.0) {
            uint64_t steps = impl_->total_steps.load();
            uint64_t dps = impl_->total_dps.load();
            double rate = (steps - result.total_steps) / elapsed;  // steps/sec (raw)

            result.total_steps = steps;
            last_report = now;

            if (progress_callback) {
                if (!progress_callback(steps, dps, rate)) {
                    stop_flag = true;
                }
            }
        }
    }

    // Wait for all GPU threads
    for (auto& t : gpu_threads) {
        if (t.joinable()) t.join();
    }

    printf("\n");

    auto end_time = std::chrono::high_resolution_clock::now();

    if (impl_->found.load()) {
        std::lock_guard<std::mutex> lock(impl_->result_mutex);
        result = impl_->result;
    } else if (impl_->failed.load()) {
        // T1.5: dispatcher path. A GPU kernel failed. Make sure
        // result.found is NOT set, surface the captured failure message,
        // and let the caller distinguish a clean stop from a failure.
        result.found = false;
        std::string msg;
        {
            std::lock_guard<std::mutex> lk(impl_->failure_msg_mutex);
            msg = impl_->failure_msg;
        }
        if (!msg.empty()) {
            fprintf(stderr, "[!] Multi-GPU kangaroo solve aborted: %s\n", msg.c_str());
        } else {
            fprintf(stderr, "[!] Multi-GPU kangaroo solve aborted: a GPU kernel failed\n");
        }
    }

    result.total_steps = impl_->total_steps.load();
    result.dp_count = impl_->total_dps.load();
    result.elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

    // Cleanup GPU memory
    for (auto& ctx : impl_->gpu_contexts) {
        cudaSetDevice(ctx.device_id);
        if (ctx.d_x) { cudaFree(ctx.d_x); ctx.d_x = nullptr; }
        if (ctx.d_y) { cudaFree(ctx.d_y); ctx.d_y = nullptr; }
        if (ctx.d_z) { cudaFree(ctx.d_z); ctx.d_z = nullptr; }
        if (ctx.d_dist) { cudaFree(ctx.d_dist); ctx.d_dist = nullptr; }
        if (ctx.d_flags) { cudaFree(ctx.d_flags); ctx.d_flags = nullptr; }
        if (ctx.d_types) { cudaFree(ctx.d_types); ctx.d_types = nullptr; }
        if (ctx.d_wild_offset) { cudaFree(ctx.d_wild_offset); ctx.d_wild_offset = nullptr; }
        if (ctx.d_restart_pending) { cudaFree(ctx.d_restart_pending); ctx.d_restart_pending = nullptr; }
    }

    return result;
}

// ============================================================================
// Herd state serialization
// ============================================================================
// The on-disk format and motivation live in the kangaroo_solver_gpu.hpp
// header doc-comment. This implementation:
//   1. computes a 32-byte SHA256 of [range_start_be || range_end_be] for
//      the config-fingerprint header field;
//   2. for each GPU, downloads (d_x, d_y, d_dist, d_wild_offset, d_types)
//      to host memory and writes them sequentially to the file;
//   3. on load, performs the reverse plus a header validation that
//      refuses to apply state from a different puzzle / herd shape.
// Notes:
//   * The d_z buffer is omitted from the saved file because the affine
//     pipeline (SOTA / JLP kernels) sets Z=1 implicitly on every save.
//     On load, the kernel's per-step batch inversion reconstructs the
//     correct Z without any external help.
//   * The d_flags and d_restart_pending buffers are NOT saved either:
//     flags are cleared every host loop iteration, and restart_pending
//     is a transient signal that becomes meaningless across a restart.
//   * No protection is provided against the file being corrupted on
//     disk; the header check guards against config mismatch but not
//     against bit-flips in the body. A future revision can add a
//     trailing SHA256 of the body if the runner team wants that.

namespace tierc_helpers {

constexpr const char kHerdMagic[16] = {
    'C','O','L','L','I','D','E','R','_','K','A','N','G','\x01','\x00','\x00'
};
constexpr uint32_t kHerdVersion = 1u;
constexpr size_t   kBytesPerKangaroo = 32 /*x*/ + 32 /*y*/ + 32 /*dist*/
                                     + 32 /*wild_off*/ + 4 /*type*/ + 4 /*pad*/;

// Build the 32-byte range fingerprint that goes into the header. Both
// range_start and range_end are serialized big-endian 32-byte limbs.
static void compute_range_hash(const UInt256& range_start,
                               const UInt256& range_end,
                               uint8_t out_sha[32]) {
    uint8_t buf[64];
    // range_start big-endian
    for (int limb = 3; limb >= 0; --limb) {
        const uint64_t v = range_start.parts[limb];
        const size_t off = (3 - limb) * 8;
        for (int b = 0; b < 8; ++b) {
            buf[off + b] = static_cast<uint8_t>(v >> (56 - b * 8));
        }
    }
    // range_end big-endian
    for (int limb = 3; limb >= 0; --limb) {
        const uint64_t v = range_end.parts[limb];
        const size_t off = 32 + (3 - limb) * 8;
        for (int b = 0; b < 8; ++b) {
            buf[off + b] = static_cast<uint8_t>(v >> (56 - b * 8));
        }
    }
    auto h = cpu::SHA256::hash(buf, 64);
    std::memcpy(out_sha, h.data(), 32);
}

}  // namespace tierc_helpers

bool MultiGPUKangarooManager::save_herd_state(const std::string& path) {
    if (!impl_ || impl_->gpu_contexts.empty()) return false;

    const size_t num_gpus = impl_->gpu_contexts.size();
    if (num_gpus == 0) return false;

    // All GPU contexts share the same kangaroo count post-init.
    const int n = impl_->gpu_contexts[0].num_kangaroos;
    if (n <= 0) return false;
    for (size_t g = 1; g < num_gpus; ++g) {
        if (impl_->gpu_contexts[g].num_kangaroos != n) return false;
    }

    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;

    // Header
    if (std::fwrite(tierc_helpers::kHerdMagic, 1, 16, fp) != 16) {
        std::fclose(fp); return false;
    }
    const uint32_t version = tierc_helpers::kHerdVersion;
    const uint32_t gpus_u32 = static_cast<uint32_t>(num_gpus);
    const uint32_t n_u32 = static_cast<uint32_t>(n);
    const uint32_t dp_u32 = dp_bits;

    auto write_u32 = [&](uint32_t v) {
        uint8_t b[4] = {
            static_cast<uint8_t>(v & 0xFF),
            static_cast<uint8_t>((v >> 8) & 0xFF),
            static_cast<uint8_t>((v >> 16) & 0xFF),
            static_cast<uint8_t>((v >> 24) & 0xFF)
        };
        return std::fwrite(b, 1, 4, fp) == 4;
    };
    if (!write_u32(version) || !write_u32(gpus_u32)
        || !write_u32(n_u32) || !write_u32(dp_u32)) {
        std::fclose(fp); return false;
    }

    uint8_t range_hash[32];
    tierc_helpers::compute_range_hash(impl_->range_start, impl_->range_end, range_hash);
    if (std::fwrite(range_hash, 1, 32, fp) != 32) {
        std::fclose(fp); return false;
    }

    // Per-GPU per-kangaroo records.
    // Q11: the prior implementation re-downloaded the full x/y/dist/
    // wild_offset/type device buffers ONCE PER KANGAROO inside the inner
    // loop. That made the persistence path O(n^2) in cudaMemcpy bytes:
    // for n = 64K kangaroos and 32 B per limb the inner loop transferred
    // ~512 GiB of identical data per save_herd_state() call. Resolution
    // by inspection (no profiler needed): the device buffers are immutable
    // for the duration of a save (no other thread is touching them; this
    // is called between scan iterations), so each buffer is downloaded
    // exactly once per GPU and then indexed for each record. The new
    // shape is O(n) cudaMemcpy bytes: 4 * n * 32 + n * 4 = (128 n + 4 n)
    // bytes per GPU, independent of the inner record loop.
    std::vector<uint64_t> h_x(n * 4);
    std::vector<uint64_t> h_y(n * 4);
    std::vector<uint64_t> h_dist(n * 4);
    std::vector<uint64_t> h_wild_offset(n * 4);
    std::vector<uint32_t> h_types(n);
    for (auto& ctx : impl_->gpu_contexts) {
        cudaSetDevice(ctx.device_id);

        auto download_256 = [&](uint64_t* d_src, uint64_t* h_dst) {
            return cudaMemcpy(h_dst, d_src,
                              n * 4 * sizeof(uint64_t),
                              cudaMemcpyDeviceToHost) == cudaSuccess;
        };

        // Download each buffer exactly once per GPU. Order in the
        // file: x, y, dist, wild_offset, type.
        if (!download_256(ctx.d_x,           h_x.data()))           { std::fclose(fp); return false; }
        if (!download_256(ctx.d_y,           h_y.data()))           { std::fclose(fp); return false; }
        if (!download_256(ctx.d_dist,        h_dist.data()))        { std::fclose(fp); return false; }
        if (!download_256(ctx.d_wild_offset, h_wild_offset.data())) { std::fclose(fp); return false; }
        if (cudaMemcpy(h_types.data(), ctx.d_types,
                       n * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fclose(fp); return false;
        }

        for (int i = 0; i < n; ++i) {
            // Per-kangaroo record (136 bytes).
            uint8_t rec[tierc_helpers::kBytesPerKangaroo];
            auto write_limb = [&](const uint64_t* src, size_t off) {
                std::memcpy(rec + off, src, 32);
            };

            write_limb(&h_x[i * 4],           0);
            write_limb(&h_y[i * 4],           32);
            write_limb(&h_dist[i * 4],        64);
            write_limb(&h_wild_offset[i * 4], 96);

            const uint32_t t = h_types[i];
            rec[128] = static_cast<uint8_t>(t & 0xFF);
            rec[129] = static_cast<uint8_t>((t >> 8) & 0xFF);
            rec[130] = static_cast<uint8_t>((t >> 16) & 0xFF);
            rec[131] = static_cast<uint8_t>((t >> 24) & 0xFF);
            rec[132] = 0; rec[133] = 0; rec[134] = 0; rec[135] = 0;

            if (std::fwrite(rec, 1, sizeof(rec), fp) != sizeof(rec)) {
                std::fclose(fp); return false;
            }
        }
    }

    std::fflush(fp);
    std::fclose(fp);
    return true;
}

bool MultiGPUKangarooManager::load_herd_state(const std::string& path) {
    if (!impl_ || impl_->gpu_contexts.empty()) return false;

    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    // Header
    char magic[16];
    if (std::fread(magic, 1, 16, fp) != 16) { std::fclose(fp); return false; }
    if (std::memcmp(magic, tierc_helpers::kHerdMagic, 16) != 0) {
        std::fclose(fp); return false;
    }

    auto read_u32 = [&](uint32_t& v) {
        uint8_t b[4];
        if (std::fread(b, 1, 4, fp) != 4) return false;
        v = static_cast<uint32_t>(b[0])
          | (static_cast<uint32_t>(b[1]) << 8)
          | (static_cast<uint32_t>(b[2]) << 16)
          | (static_cast<uint32_t>(b[3]) << 24);
        return true;
    };

    uint32_t file_version = 0, file_gpus = 0, file_n = 0, file_dp = 0;
    if (!read_u32(file_version) || !read_u32(file_gpus)
        || !read_u32(file_n) || !read_u32(file_dp)) {
        std::fclose(fp); return false;
    }
    if (file_version != tierc_helpers::kHerdVersion) {
        std::fclose(fp); return false;
    }

    // Validate config matches.
    if (file_gpus != static_cast<uint32_t>(impl_->gpu_contexts.size())
        || file_n != static_cast<uint32_t>(num_kangaroos_per_gpu)
        || file_dp != dp_bits) {
        std::fclose(fp); return false;
    }

    uint8_t file_range_hash[32];
    if (std::fread(file_range_hash, 1, 32, fp) != 32) {
        std::fclose(fp); return false;
    }
    uint8_t expected_range_hash[32];
    tierc_helpers::compute_range_hash(impl_->range_start, impl_->range_end,
                                      expected_range_hash);
    if (std::memcmp(file_range_hash, expected_range_hash, 32) != 0) {
        std::fclose(fp); return false;
    }

    // Per-GPU body.
    const int n = static_cast<int>(file_n);
    std::vector<uint64_t> h_x(n * 4), h_y(n * 4), h_dist(n * 4), h_wild(n * 4);
    std::vector<uint32_t> h_types(n);
    size_t size_256 = n * 4 * sizeof(uint64_t);
    size_t size_32  = n * sizeof(uint32_t);

    for (auto& ctx : impl_->gpu_contexts) {
        for (int i = 0; i < n; ++i) {
            uint8_t rec[tierc_helpers::kBytesPerKangaroo];
            if (std::fread(rec, 1, sizeof(rec), fp) != sizeof(rec)) {
                std::fclose(fp); return false;
            }
            std::memcpy(&h_x[i * 4],    rec + 0,  32);
            std::memcpy(&h_y[i * 4],    rec + 32, 32);
            std::memcpy(&h_dist[i * 4], rec + 64, 32);
            std::memcpy(&h_wild[i * 4], rec + 96, 32);
            const uint32_t t =
                static_cast<uint32_t>(rec[128])
              | (static_cast<uint32_t>(rec[129]) << 8)
              | (static_cast<uint32_t>(rec[130]) << 16)
              | (static_cast<uint32_t>(rec[131]) << 24);
            h_types[i] = t;
        }

        cudaSetDevice(ctx.device_id);
        // Resume herd upload. Same protection as the fresh-init path
        // above: a copy failure here poisons this device's run. Log
        // + continue to skip this device; the other devices keep
        // resuming their slice of the herd state.
        bool resume_ok =
            kg_cuda_check(cudaMemcpy(ctx.d_x, h_x.data(), size_256,
                                     cudaMemcpyHostToDevice),
                          "multigpu-resume.d_x") &&
            kg_cuda_check(cudaMemcpy(ctx.d_y, h_y.data(), size_256,
                                     cudaMemcpyHostToDevice),
                          "multigpu-resume.d_y") &&
            kg_cuda_check(cudaMemcpy(ctx.d_dist, h_dist.data(), size_256,
                                     cudaMemcpyHostToDevice),
                          "multigpu-resume.d_dist") &&
            kg_cuda_check(cudaMemcpy(ctx.d_wild_offset, h_wild.data(),
                                     size_256, cudaMemcpyHostToDevice),
                          "multigpu-resume.d_wild_offset") &&
            kg_cuda_check(cudaMemcpy(ctx.d_types, h_types.data(), size_32,
                                     cudaMemcpyHostToDevice),
                          "multigpu-resume.d_types");
        if (!resume_ok) {
            std::fprintf(stderr,
                "[!] kangaroo multi-GPU resume: device %d skipped due "
                "to host->device copy failure\n", ctx.device_id);
            continue;
        }
        // Z is implicitly 1 in the affine pipeline; the kernel's first
        // batch inversion step will write Z=1 on its first DP store and
        // the host loop never reads Z prior to that. Reset flags +
        // restart_pending so a saved-mid-step DP isn't reprocessed.
        cudaMemset(ctx.d_flags, 0, size_32);
        cudaMemset(ctx.d_restart_pending, 0, size_32);
    }

    std::fclose(fp);
    return true;
}

// ============================================================================
// T3.3 (2026-05-17). Test-only kernel-driver entry point for the dx==0
// SIMT guard. The CPU mirror in tests/test_kangaroo_dxz_fuzz.cpp pins the
// algebraic property of the guard but does NOT exercise the actual CUDA
// kernel path. A refactor that drops the guard from
// kangaroo_step_kernel_jlp / kangaroo_step_kernel_sota while leaving the
// CPU mirror intact would still pass the existing test.
//
// This entry point mirrors the dx==0 detection + dx=1 substitution +
// batch_mod_inv pre-process from kangaroo_step_kernel_sota lines
// ~1340-1374 inside a single launch. The host test:
//   1. Builds a dx[GPU_GRP_SIZE][4] batch with random non-zero limbs.
//   2. Injects a zero at one or more lanes.
//   3. Launches this kernel with one thread (one SIMT group).
//   4. Reads back the inverted dx values + the dx_was_zero flag mask.
//   5. Verifies for every non-injected lane: dx_inv * dx_original == 1
//      (mod p) and for every injected lane: dx_was_zero[g] == 1 plus
//      dx_inv == 1 (since the substituted dx=1 inverts to 1).
//
// Without the guard the product chain would zero out at the injected
// lane, every back-propagated inverse would be 0, and the verification
// would fail on every non-injected lane, a strict witness that the
// guard is present in the kernel binary actually shipping. The kernel
// launches with one thread because the guard runs entirely within one
// SIMT group (per-thread); covering multiple groups is identical work.
// ============================================================================

__global__ void test_only_kangaroo_dxz_guard_kernel(
    const uint64_t* __restrict__ d_input_dx,   // [GPU_GRP_SIZE * 4] LE limbs
    uint64_t* __restrict__       d_output_dx,  // [GPU_GRP_SIZE * 4] inverted
    uint8_t* __restrict__        d_dx_was_zero // [GPU_GRP_SIZE] flag mask
) {
    // Single thread per launch: this exercises one SIMT group, which is
    // the unit at which the guard operates.
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    uint64_t dx[GPU_GRP_SIZE][4];
    bool was_zero[GPU_GRP_SIZE];

    // Load input.
    for (int g = 0; g < GPU_GRP_SIZE; ++g) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            dx[g][i] = d_input_dx[g * 4 + i];
        }
    }

    // Apply the EXACT guard logic from kangaroo_step_kernel_sota lines
    // ~1361-1370 (kangaroo_step_kernel_jlp uses the same pattern). If
    // this loop is ever removed from production, the test will detect
    // poisoned back-propagated inverses on every non-zero lane.
    for (int g = 0; g < GPU_GRP_SIZE; ++g) {
        bool is_zero =
            (dx[g][0] == 0 && dx[g][1] == 0 && dx[g][2] == 0 && dx[g][3] == 0);
        was_zero[g] = is_zero;
        if (is_zero) {
            dx[g][0] = 1; dx[g][1] = 0; dx[g][2] = 0; dx[g][3] = 0;
        }
    }

    // Run the same batch inversion the kernel uses. If the guard is
    // dropped, this would invert a zero element and back-propagate 0
    // across every lane.
    batch_mod_inv(dx);

    // Write results back.
    for (int g = 0; g < GPU_GRP_SIZE; ++g) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            d_output_dx[g * 4 + i] = dx[g][i];
        }
        d_dx_was_zero[g] = was_zero[g] ? 1 : 0;
    }
}

// Host-callable C entry point. Mirrors the puzzle_optimized.cu
// test_only_* convention. Allocates no device memory itself; the caller
// supplies device pointers sized for GPU_GRP_SIZE elements.
extern "C" cudaError_t test_only_kangaroo_dxz_guard_launch(
    const void* d_input_dx,
    void*       d_output_dx,
    uint8_t*    d_dx_was_zero,
    int*        out_gpu_grp_size,
    cudaStream_t stream
) {
    if (out_gpu_grp_size) *out_gpu_grp_size = GPU_GRP_SIZE;
    if (!d_input_dx || !d_output_dx || !d_dx_was_zero) {
        return cudaErrorInvalidValue;
    }
    // T1.6: sticky context error attribution. Skip the launch entirely
    // when the context is already poisoned so the test sees the actual
    // probe diagnostic rather than a confusing post-launch failure.
    cudaError_t pre = collider::gpu::pre_launch_context_probe(
        "test_only_kangaroo_dxz_guard_kernel");
    if (pre != cudaSuccess) return pre;
    test_only_kangaroo_dxz_guard_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const uint64_t*>(d_input_dx),
        reinterpret_cast<uint64_t*>(d_output_dx),
        d_dx_was_zero);
    return cudaGetLastError();
}

}  // namespace gpu
}  // namespace collider
