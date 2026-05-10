/**
 * GPU RIPEMD-160 round primitives -- single source of truth.
 *
 * Pre-1.4.1 the RIPEMD-160 round functions were defined in 3 .cu
 * files (ripemd160.cu / mega_fused_kernel.cu / fused_pipeline.cu)
 * with TU-local prefixes (`f0`, `mega_f0`, `ripemd_f0`). The body
 * of each definition is identical to RFC 8092 / ISO/IEC 10118-3
 * specification of RIPEMD-160; there is no implementation
 * flexibility that would justify the duplication.
 *
 * v1.4.1 D.2: one canonical header, primitives live under
 * `collider::gpu::ripemd160::*`. Each consuming TU includes this
 * header and brings the names in via using-decls. The round LOOPS
 * stay per-TU because their unroll factor and intermediate-state
 * layout are register-budget-tuned per kernel.
 *
 * The primitives are `__host__ __device__` so the host-side
 * RIPEMD-160 in rckangaroo_wrapper.cu (precomputed-table generator)
 * can pull from the same header.
 */

#pragma once

#include <cstdint>

namespace collider {
namespace gpu {
namespace ripemd160 {

// Left-rotate.
__host__ __device__ __forceinline__ uint32_t rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// f0: rounds 1-16 left, rounds 65-80 right.
__host__ __device__ __forceinline__ uint32_t f0(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ y ^ z;
}

// f1: rounds 17-32 left, rounds 49-64 right.
__host__ __device__ __forceinline__ uint32_t f1(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) | (~x & z);
}

// f2: rounds 33-48 left, rounds 33-48 right.
__host__ __device__ __forceinline__ uint32_t f2(uint32_t x, uint32_t y, uint32_t z) {
    return (x | ~y) ^ z;
}

// f3: rounds 49-64 left, rounds 17-32 right.
__host__ __device__ __forceinline__ uint32_t f3(uint32_t x, uint32_t y, uint32_t z) {
    return (x & z) | (y & ~z);
}

// f4: rounds 65-80 left, rounds 1-16 right.
__host__ __device__ __forceinline__ uint32_t f4(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ (y | ~z);
}

}  // namespace ripemd160
}  // namespace gpu
}  // namespace collider
