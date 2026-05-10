/**
 * GPU hash round primitives -- single source of truth.
 *
 * The SHA-256 (and -512) round functions defined in FIPS 180-4 are
 * direct bit operations with no implementation flexibility: rotr,
 * ch, maj, big-sigma, and small-sigma each have exactly one correct
 * formula. Pre-1.4.1 the codebase had 6 separate translation units
 * (sha256.cu, fused_pipeline.cu, mega_fused_kernel.cu,
 * puzzle_optimized.cu, rckangaroo_wrapper.cu, v2/device_hashes.cuh)
 * each defining its own rotr/ch/maj/sigma0/sigma1/gamma0/gamma1 with
 * a different per-TU prefix (rotr / mega_rotr / sha_rotr / rotr32 /
 * sha256_rotr). The duplication had nothing to load-bearing per-kernel
 * specialisation; the per-kernel comment in mega_fused_kernel.cu
 * about register-budget tuning is about the compression LOOP, not
 * the primitives.
 *
 * v1.4.1 D.2 (partial): one canonical header for the primitives.
 * Each consuming TU includes this header and uses the canonical
 * names directly. The compression LOOP variations stay per-TU
 * because their unroll factor / intermediate-state layout is in
 * fact register-budget-tuned per kernel.
 *
 * Usage:
 *
 *   #include "hash_rounds.cuh"
 *   using collider::gpu::sha256::ch;
 *   using collider::gpu::sha256::sigma0;
 *   ...
 *
 * or fully qualified inline.
 *
 * The primitives carry no state and have no side effects, so
 * __forceinline__ is safe; they compile to the same PTX every
 * consumer used to emit, and __constant__ memory footprint is
 * unchanged (each TU still allocates its own K[64] via
 * sha256_k_constants.cuh).
 */

#pragma once

#include <cstdint>

namespace collider {
namespace gpu {
namespace sha256 {

// Right-rotate (FIPS 180-4 Sec. 4.2.2).
__host__ __device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// Choose: y if x else z. Sec. 4.1.2.
__host__ __device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

// Majority. Sec. 4.1.2.
__host__ __device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

// Big-sigma_0. Sec. 4.1.2.
__host__ __device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

// Big-sigma_1. Sec. 4.1.2.
__host__ __device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

// Small-sigma_0. Sec. 4.1.2.
__host__ __device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

// Small-sigma_1. Sec. 4.1.2.
__host__ __device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

}  // namespace sha256

namespace sha512 {

// Right-rotate, 64-bit. Sec. 4.2.3.
__host__ __device__ __forceinline__ uint64_t rotr(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

}  // namespace sha512
}  // namespace gpu
}  // namespace collider
