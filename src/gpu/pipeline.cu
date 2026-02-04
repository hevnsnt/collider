/**
 * Collider GPU Pipeline Coordinator
 *
 * Orchestrates the full cracking pipeline across 4x RTX 5090 GPUs:
 * Passphrase → SHA256 → EC Multiply → Address Hash → Bloom Check → Results
 *
 * Key optimizations:
 * - Double-buffered async transfers
 * - Multi-GPU work distribution
 * - Fused kernels where beneficial
 * - L2 cache persistence for bloom filter
 * - CUDA Graphs for minimal kernel launch overhead
 */

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace collider {
namespace gpu {

// ============================================================================
// Pinned Memory Pool - Pre-allocated for zero-overhead transfers
// ============================================================================

/**
 * Thread-safe pool of pre-allocated pinned memory buffers.
 *
 * Pinned (page-locked) memory enables faster DMA transfers between
 * host and device, but cudaMallocHost/cudaFreeHost are expensive.
 * This pool pre-allocates buffers at startup for zero runtime overhead.
 */
class PinnedMemoryPool {
public:
    struct Buffer {
        uint8_t* data;
        uint32_t* offsets;
        uint32_t* lengths;
        size_t data_capacity;
        size_t count_capacity;
        bool in_use;
    };

    PinnedMemoryPool(size_t num_buffers, size_t data_size, size_t max_count)
        : buffers_(num_buffers), data_size_(data_size), max_count_(max_count) {

        for (size_t i = 0; i < num_buffers; i++) {
            Buffer& buf = buffers_[i];
            cudaMallocHost(&buf.data, data_size);
            cudaMallocHost(&buf.offsets, max_count * sizeof(uint32_t));
            cudaMallocHost(&buf.lengths, max_count * sizeof(uint32_t));
            buf.data_capacity = data_size;
            buf.count_capacity = max_count;
            buf.in_use = false;
        }
    }

    ~PinnedMemoryPool() {
        for (auto& buf : buffers_) {
            if (buf.data) cudaFreeHost(buf.data);
            if (buf.offsets) cudaFreeHost(buf.offsets);
            if (buf.lengths) cudaFreeHost(buf.lengths);
        }
    }

    /**
     * Acquire a buffer from the pool.
     * Returns nullptr if all buffers are in use.
     */
    Buffer* acquire() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& buf : buffers_) {
            if (!buf.in_use) {
                buf.in_use = true;
                return &buf;
            }
        }
        return nullptr;  // All buffers in use
    }

    /**
     * Release a buffer back to the pool.
     */
    void release(Buffer* buf) {
        std::lock_guard<std::mutex> lock(mutex_);
        buf->in_use = false;
    }

    /**
     * Wait for any buffer to become available.
     */
    Buffer* acquire_blocking() {
        while (true) {
            if (Buffer* buf = acquire()) {
                return buf;
            }
            std::this_thread::yield();
        }
    }

private:
    std::vector<Buffer> buffers_;
    std::mutex mutex_;
    size_t data_size_;
    size_t max_count_;
};

// Forward declarations for kernel wrappers
extern "C" {
    cudaError_t sha256_batch(const uint8_t*, const uint32_t*, const uint32_t*,
                             uint8_t*, size_t, cudaStream_t);
    cudaError_t sha256_pubkey33_batch(const uint8_t*, uint8_t*, size_t, cudaStream_t);
    cudaError_t sha256_pubkey65_batch(const uint8_t*, uint8_t*, size_t, cudaStream_t);
    cudaError_t secp256k1_batch_mul(const void*, void*, size_t, cudaStream_t);
    cudaError_t ripemd160_batch(const uint8_t*, uint8_t*, size_t, cudaStream_t);
    cudaError_t bloom_filter_check_batch(const uint8_t*, const uint8_t*, uint8_t*,
                                         size_t, size_t, int, cudaStream_t);
}

/**
 * GPU device context for one GPU.
 */
struct GPUContext {
    int device_id;
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;

    // Device memory buffers (double-buffered)
    struct BufferSet {
        // Input
        uint8_t* d_passphrases;      // Concatenated passphrases
        uint32_t* d_offsets;          // Start offset per passphrase
        uint32_t* d_lengths;          // Length per passphrase
        size_t passphrase_buffer_size;

        // Intermediate
        uint8_t* d_private_keys;      // SHA256 outputs (32 bytes each)
        uint8_t* d_public_keys;       // EC mul outputs (33 bytes each, compressed)
        uint8_t* d_sha256_pubkeys;    // SHA256(pubkey) (32 bytes each)
        uint8_t* d_hash160s;          // RIPEMD160 outputs (20 bytes each)

        // Output
        uint8_t* d_match_flags;       // Bloom filter results

        // Synchronization
        cudaEvent_t transfer_complete;
        cudaEvent_t compute_complete;
        bool in_use;

        // CUDA Graph for pipeline execution (reduces kernel launch overhead)
        cudaGraph_t graph;
        cudaGraphExec_t graph_exec;
        bool graph_captured;
        size_t graph_batch_size;  // Batch size the graph was captured for
    };

    BufferSet buffers[2];  // Double buffer
    int current_buffer;

    // Bloom filter (shared across batches)
    uint8_t* d_bloom_filter;
    size_t bloom_size;
    size_t bloom_bits;
    int bloom_hashes;

    // Statistics
    std::atomic<uint64_t> keys_processed{0};
    std::atomic<uint64_t> matches_found{0};
};

/**
 * Result from GPU processing.
 */
struct GPUResult {
    std::vector<size_t> match_indices;  // Indices of candidates that matched bloom filter
    size_t batch_size;
    int gpu_id;
};

/**
 * Pipeline configuration.
 */
struct PipelineConfig {
    std::vector<int> gpu_device_ids = {0, 1, 2, 3};
    size_t batch_size = 4'000'000;            // Candidates per batch per GPU
    size_t max_passphrase_length = 128;       // Max passphrase bytes
    size_t passphrase_buffer_size = 256 * 1024 * 1024;  // 256 MB per buffer
    std::string bloom_filter_path;
    size_t bloom_bits = 48ULL * 1024 * 1024 * 1024;  // 48 Gbit
    int bloom_hashes = 20;

    // Pinned memory pool configuration
    size_t pinned_pool_buffers = 8;  // Pre-allocate 8 pinned buffers
};

/**
 * Multi-GPU Pipeline Coordinator.
 */
class GPUPipeline {
public:
    explicit GPUPipeline(const PipelineConfig& config)
        : config_(config), running_(false), pinned_pool_(nullptr) {}

    ~GPUPipeline() {
        shutdown();
    }

    /**
     * Initialize all GPUs and allocate memory.
     */
    bool initialize() {
        int device_count;
        cudaGetDeviceCount(&device_count);

        for (int device_id : config_.gpu_device_ids) {
            if (device_id >= device_count) {
                fprintf(stderr, "GPU %d not available\n", device_id);
                return false;
            }

            auto ctx = std::make_unique<GPUContext>();
            ctx->device_id = device_id;

            cudaSetDevice(device_id);

            // Create streams
            cudaStreamCreate(&ctx->compute_stream);
            cudaStreamCreate(&ctx->transfer_stream);

            // Allocate double buffers with events for async overlapping
            for (int i = 0; i < 2; i++) {
                auto& buf = ctx->buffers[i];

                buf.passphrase_buffer_size = config_.passphrase_buffer_size;

                cudaMalloc(&buf.d_passphrases, buf.passphrase_buffer_size);
                cudaMalloc(&buf.d_offsets, config_.batch_size * sizeof(uint32_t));
                cudaMalloc(&buf.d_lengths, config_.batch_size * sizeof(uint32_t));
                cudaMalloc(&buf.d_private_keys, config_.batch_size * 32);
                cudaMalloc(&buf.d_public_keys, config_.batch_size * 33);  // Compressed pubkeys
                cudaMalloc(&buf.d_sha256_pubkeys, config_.batch_size * 32);
                cudaMalloc(&buf.d_hash160s, config_.batch_size * 20);
                cudaMalloc(&buf.d_match_flags, config_.batch_size);

                // Create events for double-buffer synchronization
                cudaEventCreate(&buf.transfer_complete);
                cudaEventCreate(&buf.compute_complete);
                buf.in_use = false;

                // Initialize CUDA Graph state
                buf.graph = nullptr;
                buf.graph_exec = nullptr;
                buf.graph_captured = false;
                buf.graph_batch_size = 0;
            }

            ctx->current_buffer = 0;

            // Allocate bloom filter
            ctx->bloom_bits = config_.bloom_bits;
            ctx->bloom_size = config_.bloom_bits / 8;
            ctx->bloom_hashes = config_.bloom_hashes;
            cudaMalloc(&ctx->d_bloom_filter, ctx->bloom_size);
            cudaMemset(ctx->d_bloom_filter, 0, ctx->bloom_size);

            // Enable L2 cache persistence for bloom filter on Blackwell
            #if CUDART_VERSION >= 11040
            cudaStreamAttrValue stream_attr = {};
            stream_attr.accessPolicyWindow.base_ptr = ctx->d_bloom_filter;
            stream_attr.accessPolicyWindow.num_bytes = ctx->bloom_size;
            stream_attr.accessPolicyWindow.hitRatio = 1.0f;
            stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(ctx->compute_stream,
                                   cudaStreamAttributeAccessPolicyWindow,
                                   &stream_attr);
            #endif

            gpu_contexts_.push_back(std::move(ctx));
        }

        // Initialize pinned memory pool for zero-overhead host memory allocation
        // Pool size: one buffer per GPU × 2 (double buffering) + extras
        pinned_pool_ = std::make_unique<PinnedMemoryPool>(
            config_.pinned_pool_buffers,
            config_.passphrase_buffer_size,
            config_.batch_size
        );

        return true;
    }

    /**
     * Load bloom filter from file.
     */
    bool load_bloom_filter(const std::string& path) {
        FILE* f = fopen(path.c_str(), "rb");
        if (!f) return false;

        std::vector<uint8_t> host_bloom(config_.bloom_bits / 8);
        size_t read = fread(host_bloom.data(), 1, host_bloom.size(), f);
        fclose(f);

        if (read != host_bloom.size()) return false;

        // Copy to all GPUs
        for (auto& ctx : gpu_contexts_) {
            cudaSetDevice(ctx->device_id);
            cudaMemcpy(ctx->d_bloom_filter, host_bloom.data(),
                       host_bloom.size(), cudaMemcpyHostToDevice);
        }

        return true;
    }

    /**
     * Submit a batch of passphrases for processing.
     * True double-buffered: overlaps transfer of next batch with compute of current.
     *
     * @param passphrases Vector of passphrase strings
     * @param gpu_id Which GPU to use (-1 for auto-select)
     * @return Batch ID for tracking
     */
    uint64_t submit_batch(
        const std::vector<std::string>& passphrases,
        int gpu_id = -1
    ) {
        if (gpu_id < 0) {
            // Round-robin GPU selection
            gpu_id = next_gpu_.fetch_add(1) % gpu_contexts_.size();
        }

        auto& ctx = gpu_contexts_[gpu_id];
        cudaSetDevice(ctx->device_id);

        // Get buffer for this batch
        int buf_idx = ctx->current_buffer;
        auto& buf = ctx->buffers[buf_idx];
        ctx->current_buffer = 1 - buf_idx;

        // Wait for this buffer to be free (previous compute on it completed)
        if (buf.in_use) {
            cudaEventSynchronize(buf.compute_complete);
        }
        buf.in_use = true;

        // Prepare host data
        std::vector<uint8_t> concat_passphrases;
        std::vector<uint32_t> offsets(passphrases.size());
        std::vector<uint32_t> lengths(passphrases.size());

        uint32_t offset = 0;
        for (size_t i = 0; i < passphrases.size(); i++) {
            offsets[i] = offset;
            lengths[i] = passphrases[i].size();
            concat_passphrases.insert(concat_passphrases.end(),
                                      passphrases[i].begin(),
                                      passphrases[i].end());
            offset += lengths[i];
        }

        // Async transfer to GPU (can overlap with compute on other buffer)
        cudaMemcpyAsync(buf.d_passphrases, concat_passphrases.data(),
                        concat_passphrases.size(), cudaMemcpyHostToDevice,
                        ctx->transfer_stream);
        cudaMemcpyAsync(buf.d_offsets, offsets.data(),
                        offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice,
                        ctx->transfer_stream);
        cudaMemcpyAsync(buf.d_lengths, lengths.data(),
                        lengths.size() * sizeof(uint32_t), cudaMemcpyHostToDevice,
                        ctx->transfer_stream);

        // Record transfer complete event
        cudaEventRecord(buf.transfer_complete, ctx->transfer_stream);

        // Compute stream waits for transfer to complete (not host sync!)
        cudaStreamWaitEvent(ctx->compute_stream, buf.transfer_complete, 0);

        size_t count = passphrases.size();

        // Stage 1: SHA256 (passphrase → private key)
        sha256_batch(buf.d_passphrases, buf.d_offsets, buf.d_lengths,
                     buf.d_private_keys, count, ctx->compute_stream);

        // Stage 2: EC Multiply (private key → compressed public key, 33 bytes)
        secp256k1_batch_mul(buf.d_private_keys, buf.d_public_keys,
                            count, ctx->compute_stream);

        // Stage 3: SHA256 (public key → intermediate hash)
        // Using compressed public keys (33 bytes) for modern Bitcoin addresses
        sha256_pubkey33_batch(buf.d_public_keys, buf.d_sha256_pubkeys,
                              count, ctx->compute_stream);

        // Stage 4: RIPEMD160 (intermediate → hash160)
        ripemd160_batch(buf.d_sha256_pubkeys, buf.d_hash160s,
                        count, ctx->compute_stream);

        // Stage 5: Bloom filter check
        bloom_filter_check_batch(buf.d_hash160s, ctx->d_bloom_filter,
                                 buf.d_match_flags, count,
                                 ctx->bloom_bits, ctx->bloom_hashes,
                                 ctx->compute_stream);

        // Record compute complete event
        cudaEventRecord(buf.compute_complete, ctx->compute_stream);

        // Increment statistics
        ctx->keys_processed += count;

        return batch_counter_.fetch_add(1);
    }

    /**
     * Submit batch with pinned host memory for maximum transfer speed.
     * Use this for sustained high-throughput processing.
     */
    uint64_t submit_batch_pinned(
        const uint8_t* h_passphrases,       // Pinned memory
        const uint32_t* h_offsets,
        const uint32_t* h_lengths,
        size_t passphrase_total_bytes,
        size_t count,
        int gpu_id = -1
    ) {
        if (gpu_id < 0) {
            gpu_id = next_gpu_.fetch_add(1) % gpu_contexts_.size();
        }

        auto& ctx = gpu_contexts_[gpu_id];
        cudaSetDevice(ctx->device_id);

        int buf_idx = ctx->current_buffer;
        auto& buf = ctx->buffers[buf_idx];
        ctx->current_buffer = 1 - buf_idx;

        if (buf.in_use) {
            cudaEventSynchronize(buf.compute_complete);
        }
        buf.in_use = true;

        // Async transfer from pinned memory (faster than pageable)
        cudaMemcpyAsync(buf.d_passphrases, h_passphrases,
                        passphrase_total_bytes, cudaMemcpyHostToDevice,
                        ctx->transfer_stream);
        cudaMemcpyAsync(buf.d_offsets, h_offsets,
                        count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                        ctx->transfer_stream);
        cudaMemcpyAsync(buf.d_lengths, h_lengths,
                        count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                        ctx->transfer_stream);

        cudaEventRecord(buf.transfer_complete, ctx->transfer_stream);
        cudaStreamWaitEvent(ctx->compute_stream, buf.transfer_complete, 0);

        // Execute pipeline stages
        sha256_batch(buf.d_passphrases, buf.d_offsets, buf.d_lengths,
                     buf.d_private_keys, count, ctx->compute_stream);
        secp256k1_batch_mul(buf.d_private_keys, buf.d_public_keys,
                            count, ctx->compute_stream);
        sha256_pubkey33_batch(buf.d_public_keys, buf.d_sha256_pubkeys,
                              count, ctx->compute_stream);
        ripemd160_batch(buf.d_sha256_pubkeys, buf.d_hash160s,
                        count, ctx->compute_stream);
        bloom_filter_check_batch(buf.d_hash160s, ctx->d_bloom_filter,
                                 buf.d_match_flags, count,
                                 ctx->bloom_bits, ctx->bloom_hashes,
                                 ctx->compute_stream);

        cudaEventRecord(buf.compute_complete, ctx->compute_stream);
        ctx->keys_processed += count;

        return batch_counter_.fetch_add(1);
    }

    /**
     * Execute pipeline using CUDA Graphs for minimal kernel launch overhead.
     *
     * CUDA Graphs capture the sequence of kernel launches and replay them
     * with ~10x lower CPU overhead compared to individual launches.
     *
     * First call captures the graph; subsequent calls replay it.
     */
    void execute_pipeline_graph(
        GPUContext& ctx,
        GPUContext::BufferSet& buf,
        size_t count
    ) {
        // Check if we need to (re)capture the graph
        // Graph must be recaptured if batch size changed
        if (!buf.graph_captured || buf.graph_batch_size != count) {
            // Destroy old graph if exists
            if (buf.graph_exec) {
                cudaGraphExecDestroy(buf.graph_exec);
                buf.graph_exec = nullptr;
            }
            if (buf.graph) {
                cudaGraphDestroy(buf.graph);
                buf.graph = nullptr;
            }

            // Begin stream capture
            cudaStreamBeginCapture(ctx.compute_stream, cudaStreamCaptureModeGlobal);

            // Execute pipeline stages (these get captured, not executed)
            sha256_batch(buf.d_passphrases, buf.d_offsets, buf.d_lengths,
                         buf.d_private_keys, count, ctx.compute_stream);
            secp256k1_batch_mul(buf.d_private_keys, buf.d_public_keys,
                                count, ctx.compute_stream);
            sha256_pubkey33_batch(buf.d_public_keys, buf.d_sha256_pubkeys,
                                  count, ctx.compute_stream);
            ripemd160_batch(buf.d_sha256_pubkeys, buf.d_hash160s,
                            count, ctx.compute_stream);
            bloom_filter_check_batch(buf.d_hash160s, ctx.d_bloom_filter,
                                     buf.d_match_flags, count,
                                     ctx.bloom_bits, ctx.bloom_hashes,
                                     ctx.compute_stream);

            // End capture and create executable graph
            cudaStreamEndCapture(ctx.compute_stream, &buf.graph);

            // Instantiate executable graph
            cudaGraphInstantiate(&buf.graph_exec, buf.graph, nullptr, nullptr, 0);

            buf.graph_captured = true;
            buf.graph_batch_size = count;
        }

        // Launch the graph (much faster than individual kernel launches!)
        // This replaces 5 kernel launches with a single graph launch
        cudaGraphLaunch(buf.graph_exec, ctx.compute_stream);
    }

    /**
     * Submit batch using CUDA Graphs for maximum throughput.
     * Best for steady-state processing where batch size is constant.
     */
    uint64_t submit_batch_graph(
        const uint8_t* h_passphrases,       // Pinned memory
        const uint32_t* h_offsets,
        const uint32_t* h_lengths,
        size_t passphrase_total_bytes,
        size_t count,
        int gpu_id = -1
    ) {
        if (gpu_id < 0) {
            gpu_id = next_gpu_.fetch_add(1) % gpu_contexts_.size();
        }

        auto& ctx = gpu_contexts_[gpu_id];
        cudaSetDevice(ctx->device_id);

        int buf_idx = ctx->current_buffer;
        auto& buf = ctx->buffers[buf_idx];
        ctx->current_buffer = 1 - buf_idx;

        if (buf.in_use) {
            cudaEventSynchronize(buf.compute_complete);
        }
        buf.in_use = true;

        // Async transfer from pinned memory
        cudaMemcpyAsync(buf.d_passphrases, h_passphrases,
                        passphrase_total_bytes, cudaMemcpyHostToDevice,
                        ctx->transfer_stream);
        cudaMemcpyAsync(buf.d_offsets, h_offsets,
                        count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                        ctx->transfer_stream);
        cudaMemcpyAsync(buf.d_lengths, h_lengths,
                        count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                        ctx->transfer_stream);

        cudaEventRecord(buf.transfer_complete, ctx->transfer_stream);
        cudaStreamWaitEvent(ctx->compute_stream, buf.transfer_complete, 0);

        // Execute pipeline via CUDA Graph (10x lower launch overhead!)
        execute_pipeline_graph(*ctx, buf, count);

        cudaEventRecord(buf.compute_complete, ctx->compute_stream);
        ctx->keys_processed += count;

        return batch_counter_.fetch_add(1);
    }

    /**
     * Get results from a completed batch.
     * OPTIMIZED: Uses async copy overlapped with CPU work.
     */
    GPUResult get_results(int gpu_id, size_t batch_size) {
        auto& ctx = gpu_contexts_[gpu_id];
        cudaSetDevice(ctx->device_id);

        // Wait only for compute (not synchronously - event-based)
        cudaEventSynchronize(ctx->buffers[1 - ctx->current_buffer].compute_complete);

        // Get previous buffer (the one we just computed on)
        int buf_idx = 1 - ctx->current_buffer;
        auto& buf = ctx->buffers[buf_idx];

        // Copy match flags back using transfer stream (overlaps with next compute)
        std::vector<uint8_t> match_flags(batch_size);
        cudaMemcpyAsync(match_flags.data(), buf.d_match_flags,
                        batch_size, cudaMemcpyDeviceToHost, ctx->transfer_stream);

        // Wait for transfer to complete
        cudaStreamSynchronize(ctx->transfer_stream);

        // Collect matches
        GPUResult result;
        result.batch_size = batch_size;
        result.gpu_id = gpu_id;

        for (size_t i = 0; i < batch_size; i++) {
            if (match_flags[i]) {
                result.match_indices.push_back(i);
                ctx->matches_found++;
            }
        }

        buf.in_use = false;  // Mark buffer as available for reuse
        return result;
    }

    /**
     * Non-blocking check if results are ready.
     */
    bool results_ready(int gpu_id) {
        auto& ctx = gpu_contexts_[gpu_id];
        int buf_idx = 1 - ctx->current_buffer;
        auto& buf = ctx->buffers[buf_idx];

        cudaError_t status = cudaEventQuery(buf.compute_complete);
        return status == cudaSuccess;
    }

    /**
     * Callback type for async result processing.
     */
    using ResultCallback = std::function<void(GPUResult&&)>;

    /**
     * Register a callback to be invoked when batch results are ready.
     * The callback runs on a background thread, non-blocking to GPU work.
     *
     * This is the preferred API for high-throughput processing as it:
     * 1. Doesn't block the calling thread
     * 2. Allows overlapping result processing with next batch
     * 3. Enables efficient CPU utilization
     */
    void get_results_async(int gpu_id, size_t batch_size, ResultCallback callback) {
        // Launch async result collection on a background thread
        std::thread([this, gpu_id, batch_size, callback = std::move(callback)]() {
            auto& ctx = gpu_contexts_[gpu_id];
            cudaSetDevice(ctx->device_id);

            int buf_idx = 1 - ctx->current_buffer;
            auto& buf = ctx->buffers[buf_idx];

            // Wait for compute to complete (on this thread, not main thread)
            cudaEventSynchronize(buf.compute_complete);

            // Allocate pinned host memory for faster D2H transfer
            uint8_t* h_match_flags;
            cudaMallocHost(&h_match_flags, batch_size);

            // Async copy with immediate sync (we're already on background thread)
            cudaMemcpyAsync(h_match_flags, buf.d_match_flags,
                            batch_size, cudaMemcpyDeviceToHost, ctx->transfer_stream);
            cudaStreamSynchronize(ctx->transfer_stream);

            // Build result
            GPUResult result;
            result.batch_size = batch_size;
            result.gpu_id = gpu_id;

            for (size_t i = 0; i < batch_size; i++) {
                if (h_match_flags[i]) {
                    result.match_indices.push_back(i);
                    ctx->matches_found++;
                }
            }

            cudaFreeHost(h_match_flags);
            buf.in_use = false;

            // Invoke callback with results
            callback(std::move(result));
        }).detach();
    }

    /**
     * Process results with callback when ready, without blocking.
     * Returns immediately. Callback is invoked when results are available.
     *
     * @param gpu_id GPU to get results from
     * @param batch_size Size of the batch
     * @param callback Function to call with results
     * @param poll_interval_us Polling interval in microseconds (default 100)
     */
    void on_results_ready(
        int gpu_id,
        size_t batch_size,
        ResultCallback callback,
        int poll_interval_us = 100
    ) {
        std::thread([this, gpu_id, batch_size, callback = std::move(callback), poll_interval_us]() {
            // Poll until results are ready
            while (!results_ready(gpu_id)) {
                std::this_thread::sleep_for(std::chrono::microseconds(poll_interval_us));
            }

            // Get results and invoke callback
            GPUResult result = get_results(gpu_id, batch_size);
            callback(std::move(result));
        }).detach();
    }

    /**
     * Get aggregate statistics.
     */
    struct Stats {
        uint64_t total_keys_processed;
        uint64_t total_matches_found;
        double keys_per_second;
        std::vector<double> gpu_utilizations;
    };

    Stats get_stats() const {
        Stats s = {};
        for (const auto& ctx : gpu_contexts_) {
            s.total_keys_processed += ctx->keys_processed;
            s.total_matches_found += ctx->matches_found;
        }
        return s;
    }

    /**
     * Shutdown pipeline and free resources.
     */
    void shutdown() {
        for (auto& ctx : gpu_contexts_) {
            cudaSetDevice(ctx->device_id);

            // Wait for all pending work to complete
            cudaStreamSynchronize(ctx->compute_stream);
            cudaStreamSynchronize(ctx->transfer_stream);

            for (int i = 0; i < 2; i++) {
                auto& buf = ctx->buffers[i];
                cudaFree(buf.d_passphrases);
                cudaFree(buf.d_offsets);
                cudaFree(buf.d_lengths);
                cudaFree(buf.d_private_keys);
                cudaFree(buf.d_public_keys);
                cudaFree(buf.d_sha256_pubkeys);
                cudaFree(buf.d_hash160s);
                cudaFree(buf.d_match_flags);

                // Destroy events
                cudaEventDestroy(buf.transfer_complete);
                cudaEventDestroy(buf.compute_complete);

                // Destroy CUDA Graphs
                if (buf.graph_exec) {
                    cudaGraphExecDestroy(buf.graph_exec);
                }
                if (buf.graph) {
                    cudaGraphDestroy(buf.graph);
                }
            }

            cudaFree(ctx->d_bloom_filter);
            cudaStreamDestroy(ctx->compute_stream);
            cudaStreamDestroy(ctx->transfer_stream);
        }

        gpu_contexts_.clear();
    }

    /**
     * Get the pinned memory pool for zero-overhead host memory allocation.
     */
    PinnedMemoryPool* get_pinned_pool() { return pinned_pool_.get(); }

private:
    PipelineConfig config_;
    std::vector<std::unique_ptr<GPUContext>> gpu_contexts_;
    std::atomic<bool> running_;
    std::atomic<uint64_t> batch_counter_{0};
    std::atomic<int> next_gpu_{0};

    // Pre-allocated pinned memory pool for fast host→device transfers
    std::unique_ptr<PinnedMemoryPool> pinned_pool_;
};

/**
 * High-level async pipeline runner.
 *
 * Continuously pulls candidates from generator and feeds to GPUs.
 */
class AsyncPipelineRunner {
public:
    using MatchCallback = std::function<void(const std::string& passphrase,
                                             size_t batch_idx, int gpu_id)>;

    AsyncPipelineRunner(
        GPUPipeline& pipeline,
        MatchCallback on_match
    ) : pipeline_(pipeline), on_match_(on_match), running_(false) {}

    /**
     * Start processing loop.
     */
    void start() {
        running_ = true;

        // Worker thread per GPU
        for (size_t i = 0; i < 4; i++) {
            workers_.emplace_back([this, i]() {
                worker_loop(i);
            });
        }
    }

    /**
     * Submit batch to processing queue.
     */
    void submit(std::vector<std::string>&& batch) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            pending_batches_.push(std::move(batch));
        }
        queue_cv_.notify_one();
    }

    /**
     * Stop processing.
     */
    void stop() {
        running_ = false;
        queue_cv_.notify_all();

        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

private:
    GPUPipeline& pipeline_;
    MatchCallback on_match_;
    std::atomic<bool> running_;
    std::vector<std::thread> workers_;

    std::queue<std::vector<std::string>> pending_batches_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    void worker_loop(int gpu_id) {
        while (running_) {
            std::vector<std::string> batch;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this]() {
                    return !pending_batches_.empty() || !running_;
                });

                if (!running_ && pending_batches_.empty()) break;

                batch = std::move(pending_batches_.front());
                pending_batches_.pop();
            }

            // Process batch
            pipeline_.submit_batch(batch, gpu_id);
            auto results = pipeline_.get_results(gpu_id, batch.size());

            // Handle matches
            for (size_t idx : results.match_indices) {
                on_match_(batch[idx], idx, gpu_id);
            }
        }
    }
};

}  // namespace gpu
}  // namespace collider
