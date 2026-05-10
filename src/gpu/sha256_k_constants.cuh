/**
 * SHA-256 K-table constants -- single source of truth.
 *
 * The 64 round constants from FIPS 180-4 Section 4.2.2 (sqrt(p) where p is
 * the first 64 primes; ASCII initials of "Beethoven, Mozart, ...etc"
 * is a common mnemonic). Three GPU SHA-256 implementations in this
 * codebase used to define their own copy of this 64-element array
 * (fused_pipeline.cu / mega_fused_kernel.cu / v2/device_hashes.cuh);
 * v1.4.0 phase 4 audit folded them into this single header.
 *
 * CUDA __constant__ memory is per-translation-unit (each PTX module
 * has its own constant cache region), so each consumer still emits
 * its own __constant__ array. The deduplication is at the SOURCE level,
 * not the runtime level: the actual constant memory footprint is
 * unchanged. What this header buys us:
 *
 *   1. A single canonical place to verify the constants against
 *      FIPS 180-4 (or against any future test vector).
 *   2. Compile-time guarantee that all three GPU SHA-256 paths use
 *      bit-identical K values (no risk of one diverging via typo).
 *   3. A grep target for "where did the SHA-256 constants come from".
 *
 * Usage in a .cu file:
 *
 *   #include "sha256_k_constants.cuh"
 *   __device__ __constant__ uint32_t SHA256_K[64] = COLLIDER_SHA256_K_INIT;
 *
 * The macro expands to a brace-enclosed 64-element initializer list
 * the same way each TU used to write inline.
 */

#pragma once

// FIPS 180-4 SHA-256 round constants. Do not edit without re-deriving
// from sqrt(p) for p = first 64 primes; any change here invalidates
// every SHA-256 hash the GPU produces, with no compile-time signal.
#define COLLIDER_SHA256_K_INIT {                                              \
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,                       \
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,                       \
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,                       \
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,                       \
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,                       \
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,                       \
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,                       \
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,                       \
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,                       \
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,                       \
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,                       \
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,                       \
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,                       \
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,                       \
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,                       \
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u                        \
}
