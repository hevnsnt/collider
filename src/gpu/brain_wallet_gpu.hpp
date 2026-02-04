/**
 * Brain Wallet GPU Pipeline Interface
 *
 * Declares GPU functions for the fused brain wallet pipeline:
 *   passphrase → SHA256 → EC_MUL → pubkey → SHA256 → RIPEMD160 → bloom check
 *
 * These are implemented in fused_pipeline.cu and h160_bloom_filter.cu
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <array>
#include <chrono>
#include <string>

#ifdef COLLIDER_USE_CUDA
#include <cuda_runtime.h>
#else
// Stub types when CUDA is not available
typedef int cudaStream_t;
typedef int cudaError_t;
typedef uint64_t cudaTextureObject_t;
#define cudaSuccess 0
#endif

namespace collider {
namespace gpu {

// Forward declare CUDA functions from .cu files
extern "C" {

/**
 * Initialize the fused pipeline (precomputes EC table).
 * Call once before any batch processing.
 */
cudaError_t fused_pipeline_init(cudaStream_t stream);

/**
 * Run the full fused brain wallet pipeline on a batch.
 *
 * For each passphrase:
 *   SHA256(passphrase) → private_key
 *   EC_MUL(private_key, G) → public_key
 *   SHA256(pubkey) → intermediate
 *   RIPEMD160(intermediate) → hash160
 *   bloom_check(hash160) → match?
 *
 * @param d_passphrases   Device: packed passphrase bytes (contiguous)
 * @param d_offsets       Device: offset into d_passphrases for each candidate
 * @param d_lengths       Device: length of each passphrase
 * @param d_bloom_filter  Device: bloom filter bit array
 * @param bloom_bits      Number of bits in bloom filter
 * @param bloom_hashes    Number of hash functions (k)
 * @param d_match_indices Device: output array for match indices (1024 max)
 * @param d_match_count   Device: output counter for matches
 * @param d_private_keys  Device: optional output for private keys (or nullptr)
 * @param count           Number of passphrases in batch
 * @param stream          CUDA stream
 *
 * @return cudaSuccess or error
 */
cudaError_t fused_brain_wallet_batch(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    const uint8_t* d_bloom_filter,
    uint64_t bloom_bits,
    int bloom_hashes,
    uint32_t* d_match_indices,
    uint32_t* d_match_count,
    uint8_t* d_private_keys,
    size_t count,
    cudaStream_t stream
);

/**
 * Run the fused brain wallet pipeline with fixed-stride input.
 * Used with GPU rule engine output (avoids GPU→CPU→GPU roundtrip).
 *
 * @param d_passphrases   Device: passphrase buffer (fixed stride, idx * stride)
 * @param d_lengths       Device: length of each passphrase
 * @param stride          Bytes between consecutive passphrases (typically 256)
 * @param d_bloom_filter  Device: bloom filter bit array
 * @param bloom_bits      Number of bits in bloom filter
 * @param bloom_hashes    Number of hash functions (k)
 * @param d_match_indices Device: output array for match indices
 * @param d_match_count   Device: output counter for matches
 * @param d_private_keys  Device: optional output for private keys (or nullptr)
 * @param count           Number of passphrases in batch
 * @param stream          CUDA stream
 */
cudaError_t fused_brain_wallet_batch_fixed_stride(
    const uint8_t* d_passphrases,
    const uint32_t* d_lengths,
    uint32_t stride,
    const uint8_t* d_bloom_filter,
    uint64_t bloom_bits,
    int bloom_hashes,
    uint32_t* d_match_indices,
    uint32_t* d_match_count,
    uint8_t* d_private_keys,
    size_t count,
    cudaStream_t stream
);

/**
 * Initialize bloom filter on GPU.
 *
 * @param h_bloom_data    Host: bloom filter bytes
 * @param bloom_size      Size in bytes
 * @param num_bits        Number of bits in filter
 * @param num_hashes      Number of hash functions
 * @param seed            MurmurHash3 seed
 * @param d_bloom_data    Output: device pointer to bloom data
 *
 * @return cudaSuccess or error
 */
cudaError_t h160_bloom_init(
    const uint8_t* h_bloom_data,
    size_t bloom_size,
    uint64_t num_bits,
    uint32_t num_hashes,
    uint32_t seed,
    uint8_t** d_bloom_data
);

/**
 * Create texture object for bloom filter (optional optimization).
 */
cudaError_t h160_bloom_create_texture(
    uint8_t* d_bloom_data,
    size_t bloom_size,
    cudaTextureObject_t* tex_obj
);

/**
 * Cleanup bloom filter resources.
 */
void h160_bloom_cleanup(
    uint8_t* d_bloom_data,
    cudaTextureObject_t tex_obj
);

}  // extern "C"

/**
 * GPU context for brain wallet processing.
 * Manages device memory allocations for a single GPU.
 */
struct BrainWalletGPUContext {
    int device_id = 0;
    cudaStream_t stream = 0;

    // Bloom filter
    uint8_t* d_bloom_filter = nullptr;
    uint64_t bloom_bits = 0;
    uint32_t bloom_hashes = 0;

    // Batch buffers (double-buffered for async)
    struct BatchBuffers {
        uint8_t* d_passphrases = nullptr;      // Packed passphrase bytes
        uint32_t* d_offsets = nullptr;          // Offset per passphrase
        uint32_t* d_lengths = nullptr;          // Length per passphrase
        uint32_t* d_match_indices = nullptr;    // Output match indices
        uint32_t* d_match_count = nullptr;      // Output match count
        uint8_t* d_private_keys = nullptr;      // Output private keys (optional)

        size_t max_passphrases = 0;
        size_t max_passphrase_bytes = 0;
    };

    BatchBuffers buffers[2];  // Double buffer for overlap
    int current_buffer = 0;

    // Host pinned memory for fast transfers
    uint8_t* h_passphrases = nullptr;
    uint32_t* h_offsets = nullptr;
    uint32_t* h_lengths = nullptr;
    uint32_t* h_match_indices = nullptr;
    uint32_t* h_match_count = nullptr;

    bool initialized = false;

    // Initialize context for a GPU
    bool init(int device, size_t max_batch_size, size_t max_passphrase_bytes);

    // Cleanup
    void cleanup();

    // Get current buffer
    BatchBuffers& get_buffer() { return buffers[current_buffer]; }

    // Swap buffers
    void swap_buffers() { current_buffer = 1 - current_buffer; }
};

/**
 * Multi-GPU brain wallet manager.
 * Distributes work across multiple GPUs.
 */
class MultiGPUBrainWallet {
public:
    struct Config {
        std::vector<int> gpu_ids = {0};
        size_t batch_size = 4'000'000;
        size_t max_passphrase_length = 256;
        bool store_private_keys = true;  // Store for verification
    };

    MultiGPUBrainWallet() = default;
    explicit MultiGPUBrainWallet(const Config& config) : config_(config) {}

    // Initialize all GPUs
    bool init();

    // Load bloom filter to all GPUs
    bool load_bloom_filter(const uint8_t* data, size_t size,
                          uint64_t num_bits, uint32_t num_hashes, uint32_t seed);

    // Process a batch of passphrases
    // Returns indices of matches (into original passphrase array)
    struct BatchResult {
        std::vector<uint32_t> match_indices;
        std::vector<std::array<uint8_t, 32>> private_keys;
        size_t processed = 0;
    };

    BatchResult process_batch(
        const std::vector<std::string>& passphrases
    );

    /**
     * Process passphrases directly from GPU rule engine output.
     * This avoids the GPU→CPU→GPU roundtrip for maximum throughput.
     *
     * @param d_passphrases   Device pointer: rule output buffer (fixed stride)
     * @param d_lengths       Device pointer: length of each passphrase
     * @param stride          Bytes between consecutive passphrases (typically 256)
     * @param count           Number of passphrases to process
     * @param gpu_index       Which GPU context to use (default 0)
     * @return BatchResult with match indices and private keys
     */
    BatchResult process_batch_from_gpu(
        const uint8_t* d_passphrases,
        const uint32_t* d_lengths,
        uint32_t stride,
        size_t count,
        int gpu_index = 0
    );

    // Get GPU context for direct access (needed for GPU rules integration)
    BrainWalletGPUContext& get_context(int index = 0) { return contexts_[index]; }
    size_t num_gpus() const { return contexts_.size(); }

    // Cleanup
    void cleanup();

    // Statistics
    uint64_t total_processed() const { return total_processed_; }
    double keys_per_second() const;

private:
    Config config_;
    std::vector<BrainWalletGPUContext> contexts_;
    uint64_t total_processed_ = 0;
    std::chrono::steady_clock::time_point start_time_;
    bool initialized_ = false;

    // Internal single-GPU processing (used by multi-GPU distribution)
    BatchResult process_batch_single_gpu(
        const std::vector<std::string>& passphrases,
        BrainWalletGPUContext& ctx,
        size_t index_offset = 0
    );
};

}  // namespace gpu
}  // namespace collider
