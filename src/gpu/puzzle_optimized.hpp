/**
 * Collider Optimized Puzzle Search Kernel - Public API
 *
 * High-performance GPU kernel for Bitcoin puzzle key search.
 * Targets 400-800M keys/sec on RTX 3090 class hardware.
 */

#pragma once

#include <cstdint>

#ifdef COLLIDER_USE_CUDA
#include <cuda_runtime.h>

extern "C" {

/**
 * Initialize the optimized puzzle search kernel.
 * Must be called once before puzzle_search_batch_optimized.
 *
 * This generates precomputed EC tables on the GPU and stores them
 * in constant memory (or device memory with L2 persistence).
 *
 * @param stream CUDA stream for async operations
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t init_puzzle_optimized(cudaStream_t stream);

/**
 * Cleanup optimized puzzle resources.
 * Call when done with puzzle search.
 */
cudaError_t cleanup_puzzle_optimized();

/**
 * Execute optimized puzzle search batch.
 *
 * Searches for a private key whose corresponding compressed public key
 * hashes to the target hash160.
 *
 * Features:
 * - Precomputed EC tables (16x speedup)
 * - Strided incremental search (256 keys per thread)
 * - Montgomery batch inversion (amortize mod_inv across batch)
 * - Inline SHA256/RIPEMD160 for Hash160
 *
 * @param range_start_lo    Lower 64 bits of search range start
 * @param range_start_hi    Upper 64 bits of search range start (for >64-bit puzzles)
 * @param batch_size        Number of keys to check in this batch
 * @param d_target_hash160  Device pointer to 20-byte target hash160
 * @param d_match_key_lo    Device pointer to store found key (lower 64 bits)
 * @param d_match_key_hi    Device pointer to store found key (upper 64 bits)
 * @param d_match_found     Device pointer to match flag (0=not found, 1=found)
 * @param stream            CUDA stream for async execution
 * @return cudaSuccess on success, error code on failure
 */
cudaError_t puzzle_search_batch_optimized(
    uint64_t range_start_lo,
    uint64_t range_start_hi,
    uint64_t batch_size,
    const uint8_t* d_target_hash160,
    uint64_t* d_match_key_lo,
    uint64_t* d_match_key_hi,
    uint32_t* d_match_found,
    cudaStream_t stream
);

}  // extern "C"

#endif  // COLLIDER_USE_CUDA
