/**
 * Collider GPU Bloom Filter
 *
 * High-performance bloom filter for Bitcoin address matching.
 * Designed to hold ~1 billion funded addresses with low false positive rate.
 *
 * Key insight from brainflayer: hash160 is already a hash, so we can
 * just bitslice it for bloom filter indices instead of rehashing.
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace collider {
namespace gpu {

// Bloom filter configuration
// For 1 billion addresses with 0.0000001 (1 in 10 million) FP rate:
// m = -n * ln(p) / (ln(2)^2) ≈ 48 billion bits ≈ 6 GB
// k = (m/n) * ln(2) ≈ 33 hash functions

constexpr size_t BLOOM_BITS = 48ULL * 1024 * 1024 * 1024;  // 48 Gbit = 6 GB
constexpr size_t BLOOM_BYTES = BLOOM_BITS / 8;
constexpr int NUM_HASHES = 20;  // Using 20 positions from hash160

/**
 * GPU Bloom Filter for address matching.
 *
 * Stored in GPU global memory with L2 cache hints for hot path.
 */
class GPUBloomFilter {
public:
    uint8_t* d_bits;
    size_t num_bits;
    int num_hashes;

    GPUBloomFilter() : d_bits(nullptr), num_bits(BLOOM_BITS), num_hashes(NUM_HASHES) {}

    /**
     * Allocate bloom filter on GPU.
     */
    cudaError_t allocate() {
        cudaError_t err = cudaMalloc(&d_bits, BLOOM_BYTES);
        if (err != cudaSuccess) return err;

        return cudaMemset(d_bits, 0, BLOOM_BYTES);
    }

    /**
     * Free GPU memory.
     */
    void free() {
        if (d_bits) {
            cudaFree(d_bits);
            d_bits = nullptr;
        }
    }
};

/**
 * Compute bloom filter indices from hash160.
 *
 * Since hash160 is already cryptographically hashed, we can use
 * bit slicing to extract multiple indices without rehashing.
 * OPTIMIZED: Uses vectorized uint64_t loads instead of byte-by-byte.
 *
 * @param hash160 The 20-byte address hash
 * @param indices Output array of indices (at least NUM_HASHES)
 * @param num_bits Total bits in bloom filter
 * @param num_hashes Number of hash functions to use
 */
__device__ __forceinline__ void compute_bloom_indices(
    const uint8_t* hash160,
    uint64_t* indices,
    size_t num_bits,
    int num_hashes
) {
    // Use double hashing: h(i) = h1 + i*h2 mod m
    // OPTIMIZED: Direct uint64_t loads (assumes little-endian, valid on CUDA)
    const uint64_t* hash64 = reinterpret_cast<const uint64_t*>(hash160);
    uint64_t h1 = hash64[0];  // First 8 bytes
    uint64_t h2 = hash64[1];  // Next 8 bytes

    // Ensure h2 is odd for better distribution
    h2 |= 1;

    // Unroll for performance
    #pragma unroll
    for (int i = 0; i < num_hashes; i++) {
        indices[i] = (h1 + (uint64_t)i * h2) % num_bits;
    }
}

/**
 * Check if a hash160 might be in the bloom filter.
 *
 * @param hash160 The 20-byte address hash to check
 * @param bits The bloom filter bit array
 * @param num_bits Total bits in filter
 * @param num_hashes Number of hash functions
 * @return true if possibly in set, false if definitely not
 */
__device__ bool bloom_check(
    const uint8_t* hash160,
    const uint8_t* bits,
    size_t num_bits,
    int num_hashes
) {
    uint64_t indices[32];  // Max hash functions
    compute_bloom_indices(hash160, indices, num_bits, num_hashes);

    for (int i = 0; i < num_hashes; i++) {
        uint64_t byte_idx = indices[i] / 8;
        int bit_idx = indices[i] % 8;

        if (!(bits[byte_idx] & (1 << bit_idx))) {
            return false;  // Definitely not in set
        }
    }

    return true;  // Possibly in set
}

/**
 * Batch bloom filter check kernel.
 *
 * @param hash160s Array of 20-byte hashes to check
 * @param bits Bloom filter bit array
 * @param results Output: 1 if possibly in set, 0 if definitely not
 * @param count Number of hashes to check
 * @param num_bits Total bits in filter
 * @param num_hashes Number of hash functions
 */
__global__ void bloom_check_batch_kernel(
    const uint8_t* __restrict__ hash160s,
    const uint8_t* __restrict__ bits,
    uint8_t* __restrict__ results,
    size_t count,
    size_t num_bits,
    int num_hashes
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* hash = hash160s + idx * 20;
    results[idx] = bloom_check(hash, bits, num_bits, num_hashes) ? 1 : 0;
}

/**
 * Optimized batch check using shared memory for bloom filter caching.
 *
 * OPTIMIZATION: Uses shared memory to cache frequently accessed bloom filter
 * bytes within a warp. Since hash160 is well-distributed, threads in a warp
 * may access overlapping regions of the bloom filter, especially for smaller
 * bloom filters or specific index patterns.
 *
 * Strategy: Use a small shared memory cache with a simple hash-based mapping.
 * Cache misses fall back to global memory (L2 cached).
 */
#define BLOOM_CACHE_SIZE 4096  // 4KB shared memory cache per block

__global__ void bloom_check_batch_optimized_kernel(
    const uint8_t* __restrict__ hash160s,
    const uint8_t* __restrict__ bits,
    uint8_t* __restrict__ results,
    size_t count,
    size_t num_bits,
    int num_hashes
) {
    // Shared memory cache for bloom filter bytes
    __shared__ uint8_t shared_cache[BLOOM_CACHE_SIZE];
    __shared__ uint64_t cache_tags[BLOOM_CACHE_SIZE / 64];  // Tag per cache line

    // Initialize cache tags to invalid (cooperative initialization)
    const int cache_lines = BLOOM_CACHE_SIZE / 64;
    for (int i = threadIdx.x; i < cache_lines; i += blockDim.x) {
        cache_tags[i] = ~0ULL;  // Invalid tag
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread checks one hash
    if (idx >= count) {
        return;
    }

    const uint8_t* hash = hash160s + idx * 20;

    uint64_t indices[32];
    compute_bloom_indices(hash, indices, num_bits, num_hashes);

    bool possibly_in_set = true;

    for (int i = 0; i < num_hashes && possibly_in_set; i++) {
        uint64_t byte_idx = indices[i] / 8;
        int bit_idx = indices[i] % 8;

        // Try shared memory cache first (simple direct-mapped cache)
        size_t cache_line = (byte_idx / 64) % cache_lines;
        size_t line_offset = byte_idx % 64;
        uint64_t tag = byte_idx / 64;

        uint8_t byte_val;

        // Check if cache line is valid and matches
        if (cache_tags[cache_line] == tag) {
            // Cache hit - read from shared memory
            byte_val = shared_cache[cache_line * 64 + line_offset];
        } else {
            // Cache miss - read from global memory (L2 will help)
            byte_val = bits[byte_idx];

            // Opportunistically populate cache using atomic CAS to avoid race conditions
            // Only the thread that wins the CAS populates the cache line
            uint64_t expected = ~0ULL;
            uint64_t* tag_ptr = &cache_tags[cache_line];
            if (atomicCAS((unsigned long long*)tag_ptr, expected, tag) == expected) {
                // This thread won - populate the cache line with bounds checking
                uint64_t base_byte = (byte_idx / 64) * 64;
                size_t bytes_in_filter = num_bits / 8;
                for (int j = 0; j < 64 && (base_byte + j) < bytes_in_filter; j++) {
                    shared_cache[cache_line * 64 + j] = bits[base_byte + j];
                }
            }
        }

        if (!(byte_val & (1 << bit_idx))) {
            possibly_in_set = false;
        }
    }

    results[idx] = possibly_in_set ? 1 : 0;
}

/**
 * Insert addresses into bloom filter (CPU-side preparation).
 */
__global__ void bloom_insert_kernel(
    const uint8_t* __restrict__ hash160s,
    uint8_t* __restrict__ bits,
    size_t count,
    size_t num_bits,
    int num_hashes
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* hash = hash160s + idx * 20;

    uint64_t indices[32];
    compute_bloom_indices(hash, indices, num_bits, num_hashes);

    for (int i = 0; i < num_hashes; i++) {
        uint64_t byte_idx = indices[i] / 8;
        int bit_idx = indices[i] % 8;

        // Atomic OR to set bit
        atomicOr((unsigned int*)(bits + (byte_idx & ~3)),
                 (1 << bit_idx) << ((byte_idx & 3) * 8));
    }
}

// -----------------------------------------------------------------------------
// Integrated Pipeline: Hash160 Generation + Bloom Check
// -----------------------------------------------------------------------------

/**
 * Combined kernel: compute hash160 and check bloom filter.
 *
 * @param sha256_inputs Array of 32-byte SHA256 outputs (private keys)
 * @param public_keys_x Array of 32-byte public key X coordinates
 * @param public_keys_y Array of 32-byte public key Y coordinates
 * @param bloom_bits Bloom filter
 * @param match_flags Output: 1 if bloom filter match, 0 otherwise
 * @param count Number of keys to process
 */
__global__ void hash160_and_bloom_kernel(
    const uint8_t* __restrict__ pubkey_hashes,  // Already SHA256'd pubkeys
    const uint8_t* __restrict__ bloom_bits,
    uint8_t* __restrict__ match_flags,
    size_t count,
    size_t num_bits,
    int num_hashes
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Input is already SHA256(pubkey), need RIPEMD160
    const uint8_t* sha256_pubkey = pubkey_hashes + idx * 32;

    // Compute RIPEMD160
    uint8_t hash160[20];
    // (Would call ripemd160_hash here - inline for performance)

    // For now, just use first 20 bytes as placeholder
    for (int i = 0; i < 20; i++) {
        hash160[i] = sha256_pubkey[i];
    }

    // Check bloom filter
    match_flags[idx] = bloom_check(hash160, bloom_bits, num_bits, num_hashes) ? 1 : 0;
}

// Host wrappers
extern "C" {

cudaError_t bloom_filter_check_batch(
    const uint8_t* d_hash160s,
    const uint8_t* d_bits,
    uint8_t* d_results,
    size_t count,
    size_t num_bits,
    int num_hashes,
    cudaStream_t stream
) {
    const int threads_per_block = 256;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    bloom_check_batch_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_hash160s,
        d_bits,
        d_results,
        count,
        num_bits,
        num_hashes
    );

    return cudaGetLastError();
}

cudaError_t bloom_filter_insert_batch(
    const uint8_t* d_hash160s,
    uint8_t* d_bits,
    size_t count,
    size_t num_bits,
    int num_hashes,
    cudaStream_t stream
) {
    const int threads_per_block = 256;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    bloom_insert_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_hash160s,
        d_bits,
        count,
        num_bits,
        num_hashes
    );

    return cudaGetLastError();
}

}  // extern "C"

}  // namespace gpu
}  // namespace collider
