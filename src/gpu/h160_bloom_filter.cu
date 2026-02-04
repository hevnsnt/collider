/**
 * H160 Bloom Filter GPU Kernel
 *
 * Optimized for RTX 5090 (Blackwell) architecture:
 * - 128-byte aligned memory access
 * - Texture memory for bloom filter
 * - Coalesced memory patterns
 * - Thread-safe atomic-free reads
 * - Warp-level optimization
 *
 * Uses MurmurHash3 double-hashing scheme for k hash functions.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace collider {
namespace gpu {

// Bloom filter configuration (matches host-side builder)
struct BloomConfig {
    uint64_t num_bits;
    uint32_t num_hashes;
    uint32_t seed;
};

// Constant memory for bloom filter configuration
__constant__ BloomConfig d_bloom_config;

// Texture reference for bloom filter (faster than global memory)
cudaTextureObject_t bloom_texture;

/**
 * MurmurHash3 fmix64 finalizer.
 */
__device__ __forceinline__ uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

/**
 * MurmurHash3 128-bit hash for 20-byte H160 input.
 * Returns two 64-bit values for double-hashing scheme.
 */
__device__ void murmurhash3_h160(
    const uint8_t* __restrict__ h160,
    uint32_t seed,
    uint64_t* h1_out,
    uint64_t* h2_out
) {
    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;

    uint64_t h1 = seed;
    uint64_t h2 = seed;

    // Process first 16 bytes (one full block)
    uint64_t k1 = *reinterpret_cast<const uint64_t*>(h160);
    uint64_t k2 = *reinterpret_cast<const uint64_t*>(h160 + 8);

    k1 *= c1;
    k1 = (k1 << 31) | (k1 >> 33);
    k1 *= c2;
    h1 ^= k1;

    h1 = (h1 << 27) | (h1 >> 37);
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729;

    k2 *= c2;
    k2 = (k2 << 33) | (k2 >> 31);
    k2 *= c1;
    h2 ^= k2;

    h2 = (h2 << 31) | (h2 >> 33);
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5;

    // Process remaining 4 bytes (tail)
    uint64_t k1_tail = 0;
    k1_tail ^= uint64_t(h160[19]) << 24;
    k1_tail ^= uint64_t(h160[18]) << 16;
    k1_tail ^= uint64_t(h160[17]) << 8;
    k1_tail ^= uint64_t(h160[16]);
    k1_tail *= c1;
    k1_tail = (k1_tail << 31) | (k1_tail >> 33);
    k1_tail *= c2;
    h1 ^= k1_tail;

    // Finalization
    h1 ^= 20;  // length
    h2 ^= 20;
    h1 += h2;
    h2 += h1;
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 += h2;
    h2 += h1;

    *h1_out = h1;
    *h2_out = h2;
}

/**
 * Check single bit in bloom filter using texture memory.
 */
__device__ __forceinline__ bool check_bit_texture(
    cudaTextureObject_t tex,
    uint64_t bit_idx
) {
    uint64_t byte_idx = bit_idx / 8;
    uint32_t bit_offset = bit_idx % 8;

    // Read byte from texture
    uint8_t byte_val = tex1Dfetch<uint8_t>(tex, byte_idx);

    return (byte_val >> bit_offset) & 1;
}

/**
 * Check single bit in bloom filter using global memory.
 * Optimized for coalesced access.
 */
__device__ __forceinline__ bool check_bit_global(
    const uint8_t* __restrict__ bloom_data,
    uint64_t bit_idx
) {
    uint64_t byte_idx = bit_idx / 8;
    uint32_t bit_offset = bit_idx % 8;

    return (bloom_data[byte_idx] >> bit_offset) & 1;
}

/**
 * Bloom filter lookup for single H160 using global memory.
 * Returns true if H160 might be in filter, false if definitely not.
 */
__device__ bool bloom_check_h160(
    const uint8_t* __restrict__ bloom_data,
    uint64_t num_bits,
    uint32_t num_hashes,
    uint32_t seed,
    const uint8_t* __restrict__ h160
) {
    uint64_t h1, h2;
    murmurhash3_h160(h160, seed, &h1, &h2);

    // Check all k hash positions
    for (uint32_t i = 0; i < num_hashes; i++) {
        uint64_t hash = h1 + i * h2;
        uint64_t bit_idx = hash % num_bits;

        if (!check_bit_global(bloom_data, bit_idx)) {
            return false;  // Definitely not in set
        }
    }

    return true;  // Probably in set
}

/**
 * Bloom filter lookup using texture memory for better cache performance.
 * Texture memory has dedicated cache and can be faster for random access patterns.
 */
__device__ bool bloom_check_h160_texture(
    cudaTextureObject_t bloom_tex,
    uint64_t num_bits,
    uint32_t num_hashes,
    uint32_t seed,
    const uint8_t* __restrict__ h160
) {
    uint64_t h1, h2;
    murmurhash3_h160(h160, seed, &h1, &h2);

    // Check all k hash positions using texture reads
    for (uint32_t i = 0; i < num_hashes; i++) {
        uint64_t hash = h1 + i * h2;
        uint64_t bit_idx = hash % num_bits;

        if (!check_bit_texture(bloom_tex, bit_idx)) {
            return false;  // Definitely not in set
        }
    }

    return true;  // Probably in set
}

// Global texture object for bloom filter (initialized once)
static cudaTextureObject_t g_bloom_texture = 0;
static bool g_bloom_texture_valid = false;

/**
 * Batch bloom filter check kernel.
 *
 * Checks multiple H160 hashes against the bloom filter.
 * Outputs match flags (1 = probable match, 0 = definitely not).
 *
 * @param h160s         Input H160 hashes (20 bytes each, contiguous)
 * @param bloom_data    Bloom filter bit array
 * @param match_flags   Output match flags (1 byte each)
 * @param count         Number of H160s to check
 */
__global__ void h160_bloom_batch_kernel(
    const uint8_t* __restrict__ h160s,
    const uint8_t* __restrict__ bloom_data,
    uint8_t* __restrict__ match_flags,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* h160 = h160s + idx * 20;

    bool match = bloom_check_h160(
        bloom_data,
        d_bloom_config.num_bits,
        d_bloom_config.num_hashes,
        d_bloom_config.seed,
        h160
    );

    match_flags[idx] = match ? 1 : 0;
}

/**
 * Texture-based batch bloom filter check kernel.
 * Uses texture memory for better cache performance on random access patterns.
 */
__global__ void h160_bloom_batch_kernel_texture(
    const uint8_t* __restrict__ h160s,
    cudaTextureObject_t bloom_tex,
    uint8_t* __restrict__ match_flags,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* h160 = h160s + idx * 20;

    bool match = bloom_check_h160_texture(
        bloom_tex,
        d_bloom_config.num_bits,
        d_bloom_config.num_hashes,
        d_bloom_config.seed,
        h160
    );

    match_flags[idx] = match ? 1 : 0;
}

/**
 * Compact kernel: extracts indices of probable matches.
 * Uses warp-level primitives for efficient compaction.
 */
__global__ void h160_bloom_compact_kernel(
    const uint8_t* __restrict__ h160s,
    const uint8_t* __restrict__ bloom_data,
    uint32_t* __restrict__ match_indices,
    uint32_t* __restrict__ match_count,
    uint8_t* __restrict__ matched_h160s,    // Optional: copy matched H160s
    size_t count,
    size_t max_matches
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* h160 = h160s + idx * 20;

    bool match = bloom_check_h160(
        bloom_data,
        d_bloom_config.num_bits,
        d_bloom_config.num_hashes,
        d_bloom_config.seed,
        h160
    );

    if (match) {
        // Atomic increment to get output position
        uint32_t pos = atomicAdd(match_count, 1);

        if (pos < max_matches) {
            match_indices[pos] = idx;

            // Copy matched H160 if output buffer provided
            if (matched_h160s != nullptr) {
                #pragma unroll
                for (int i = 0; i < 20; i++) {
                    matched_h160s[pos * 20 + i] = h160[i];
                }
            }
        }
    }
}

/**
 * Texture-based compact kernel: extracts indices of probable matches.
 * Uses texture memory for better cache performance.
 */
__global__ void h160_bloom_compact_kernel_texture(
    const uint8_t* __restrict__ h160s,
    cudaTextureObject_t bloom_tex,
    uint32_t* __restrict__ match_indices,
    uint32_t* __restrict__ match_count,
    uint8_t* __restrict__ matched_h160s,
    size_t count,
    size_t max_matches
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* h160 = h160s + idx * 20;

    bool match = bloom_check_h160_texture(
        bloom_tex,
        d_bloom_config.num_bits,
        d_bloom_config.num_hashes,
        d_bloom_config.seed,
        h160
    );

    if (match) {
        uint32_t pos = atomicAdd(match_count, 1);

        if (pos < max_matches) {
            match_indices[pos] = idx;

            if (matched_h160s != nullptr) {
                #pragma unroll
                for (int i = 0; i < 20; i++) {
                    matched_h160s[pos * 20 + i] = h160[i];
                }
            }
        }
    }
}

/**
 * Fused kernel: full brain wallet pipeline with bloom filter gate.
 *
 * passphrase -> SHA256 -> private_key -> EC_MUL -> pubkey ->
 * SHA256 -> RIPEMD160 -> H160 -> BLOOM_CHECK -> output if match
 *
 * This is called from fused_pipeline.cu and integrates seamlessly.
 */
__device__ bool bloom_filter_gate(
    const uint8_t* __restrict__ bloom_data,
    const uint8_t* __restrict__ h160
) {
    return bloom_check_h160(
        bloom_data,
        d_bloom_config.num_bits,
        d_bloom_config.num_hashes,
        d_bloom_config.seed,
        h160
    );
}

// Host-side API

/**
 * Set bloom filter configuration in constant memory.
 * Used for parallel loading where data copy is done separately.
 */
extern "C" cudaError_t h160_bloom_set_config(
    uint64_t num_bits,
    uint32_t num_hashes,
    uint32_t seed
) {
    BloomConfig config;
    config.num_bits = num_bits;
    config.num_hashes = num_hashes;
    config.seed = seed;

    return cudaMemcpyToSymbol(d_bloom_config, &config, sizeof(BloomConfig));
}

/**
 * Initialize bloom filter on GPU.
 */
extern "C" cudaError_t h160_bloom_init(
    const uint8_t* h_bloom_data,
    size_t bloom_size,
    uint64_t num_bits,
    uint32_t num_hashes,
    uint32_t seed,
    uint8_t** d_bloom_data
) {
    // Allocate device memory (128-byte aligned)
    size_t aligned_size = ((bloom_size + 127) / 128) * 128;
    cudaError_t err = cudaMalloc(d_bloom_data, aligned_size);
    if (err != cudaSuccess) return err;

    // Copy bloom filter data
    err = cudaMemcpy(*d_bloom_data, h_bloom_data, bloom_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(*d_bloom_data);
        return err;
    }

    // Set configuration in constant memory
    BloomConfig config;
    config.num_bits = num_bits;
    config.num_hashes = num_hashes;
    config.seed = seed;

    err = cudaMemcpyToSymbol(d_bloom_config, &config, sizeof(BloomConfig));
    if (err != cudaSuccess) {
        cudaFree(*d_bloom_data);
        return err;
    }

    return cudaSuccess;
}

/**
 * Create texture object for bloom filter (optional, for texture memory path).
 */
extern "C" cudaError_t h160_bloom_create_texture(
    uint8_t* d_bloom_data,
    size_t bloom_size,
    cudaTextureObject_t* tex_obj
) {
    cudaResourceDesc res_desc = {};
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = d_bloom_data;
    res_desc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    res_desc.res.linear.desc.x = 8;  // 8 bits per element
    res_desc.res.linear.sizeInBytes = bloom_size;

    cudaTextureDesc tex_desc = {};
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;

    return cudaCreateTextureObject(tex_obj, &res_desc, &tex_desc, nullptr);
}

/**
 * Enhanced initialization that also creates texture object for better performance.
 */
extern "C" cudaError_t h160_bloom_init_with_texture(
    const uint8_t* h_bloom_data,
    size_t bloom_size,
    uint64_t num_bits,
    uint32_t num_hashes,
    uint32_t seed,
    uint8_t** d_bloom_data
) {
    // First do the standard initialization
    cudaError_t err = h160_bloom_init(h_bloom_data, bloom_size, num_bits, num_hashes, seed, d_bloom_data);
    if (err != cudaSuccess) return err;

    // Now create texture object for optimized reads
    err = h160_bloom_create_texture(*d_bloom_data, bloom_size, &g_bloom_texture);
    if (err == cudaSuccess) {
        g_bloom_texture_valid = true;
        fprintf(stderr, "[Bloom] Texture memory enabled for bloom filter (%zu MB)\n",
                bloom_size / (1024 * 1024));
    } else {
        // Texture creation failed - not fatal, fall back to global memory
        g_bloom_texture_valid = false;
        fprintf(stderr, "[Bloom] Texture creation failed, using global memory\n");
        err = cudaSuccess;  // Don't fail the init
    }

    return err;
}

/**
 * Check if texture-based bloom filter is available.
 */
extern "C" bool h160_bloom_has_texture() {
    return g_bloom_texture_valid;
}

/**
 * Get the texture object for use in kernels.
 */
extern "C" cudaTextureObject_t h160_bloom_get_texture() {
    return g_bloom_texture;
}

/**
 * Batch bloom filter check.
 */
extern "C" cudaError_t h160_bloom_batch_check(
    const uint8_t* d_h160s,
    const uint8_t* d_bloom_data,
    uint8_t* d_match_flags,
    size_t count,
    cudaStream_t stream
) {
    const int threads_per_block = 256;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    h160_bloom_batch_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_h160s,
        d_bloom_data,
        d_match_flags,
        count
    );

    return cudaGetLastError();
}

/**
 * Compact bloom filter check (returns only matching indices).
 */
extern "C" cudaError_t h160_bloom_compact_check(
    const uint8_t* d_h160s,
    const uint8_t* d_bloom_data,
    uint32_t* d_match_indices,
    uint32_t* d_match_count,
    uint8_t* d_matched_h160s,  // Can be nullptr
    size_t count,
    size_t max_matches,
    cudaStream_t stream
) {
    // Reset match count
    cudaMemsetAsync(d_match_count, 0, sizeof(uint32_t), stream);

    const int threads_per_block = 256;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    h160_bloom_compact_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_h160s,
        d_bloom_data,
        d_match_indices,
        d_match_count,
        d_matched_h160s,
        count,
        max_matches
    );

    return cudaGetLastError();
}

/**
 * Texture-based batch bloom filter check.
 * Automatically uses global texture if available, falls back to global memory.
 */
extern "C" cudaError_t h160_bloom_batch_check_auto(
    const uint8_t* d_h160s,
    const uint8_t* d_bloom_data,
    uint8_t* d_match_flags,
    size_t count,
    cudaStream_t stream
) {
    const int threads_per_block = 256;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    if (g_bloom_texture_valid) {
        // Use texture memory path for better cache performance
        h160_bloom_batch_kernel_texture<<<blocks, threads_per_block, 0, stream>>>(
            d_h160s,
            g_bloom_texture,
            d_match_flags,
            count
        );
    } else {
        // Fall back to global memory
        h160_bloom_batch_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_h160s,
            d_bloom_data,
            d_match_flags,
            count
        );
    }

    return cudaGetLastError();
}

/**
 * Texture-based compact bloom filter check.
 * Automatically uses global texture if available, falls back to global memory.
 */
extern "C" cudaError_t h160_bloom_compact_check_auto(
    const uint8_t* d_h160s,
    const uint8_t* d_bloom_data,
    uint32_t* d_match_indices,
    uint32_t* d_match_count,
    uint8_t* d_matched_h160s,
    size_t count,
    size_t max_matches,
    cudaStream_t stream
) {
    cudaMemsetAsync(d_match_count, 0, sizeof(uint32_t), stream);

    const int threads_per_block = 256;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    if (g_bloom_texture_valid) {
        h160_bloom_compact_kernel_texture<<<blocks, threads_per_block, 0, stream>>>(
            d_h160s,
            g_bloom_texture,
            d_match_indices,
            d_match_count,
            d_matched_h160s,
            count,
            max_matches
        );
    } else {
        h160_bloom_compact_kernel<<<blocks, threads_per_block, 0, stream>>>(
            d_h160s,
            d_bloom_data,
            d_match_indices,
            d_match_count,
            d_matched_h160s,
            count,
            max_matches
        );
    }

    return cudaGetLastError();
}

/**
 * Cleanup bloom filter resources.
 */
extern "C" void h160_bloom_cleanup(
    uint8_t* d_bloom_data,
    cudaTextureObject_t tex_obj
) {
    if (tex_obj != 0) {
        cudaDestroyTextureObject(tex_obj);
    }
    if (d_bloom_data != nullptr) {
        cudaFree(d_bloom_data);
    }
}

/**
 * Cleanup bloom filter including global texture object.
 * Call this when shutting down to release all resources.
 */
extern "C" void h160_bloom_cleanup_all(uint8_t* d_bloom_data) {
    // Cleanup global texture if created
    if (g_bloom_texture_valid) {
        cudaDestroyTextureObject(g_bloom_texture);
        g_bloom_texture = 0;
        g_bloom_texture_valid = false;
    }
    // Free bloom data
    if (d_bloom_data != nullptr) {
        cudaFree(d_bloom_data);
    }
}

}  // namespace gpu
}  // namespace collider
