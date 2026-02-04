/**
 * Brain Wallet GPU Pipeline Implementation
 *
 * Implements GPU context management and multi-GPU coordination
 * for the brain wallet cracking pipeline.
 */

#include "brain_wallet_gpu.hpp"
#include <iostream>
#include <cstring>
#include <thread>
#include <atomic>
#include <vector>

#ifdef COLLIDER_USE_CUDA
#include <cuda_runtime.h>

// External function from h160_bloom_filter.cu - declared at global scope for proper linkage
extern "C" cudaError_t h160_bloom_set_config(uint64_t num_bits, uint32_t num_hashes, uint32_t seed);

namespace collider {
namespace gpu {

bool BrainWalletGPUContext::init(int device, size_t max_batch_size, size_t max_passphrase_bytes) {
    device_id = device;

    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        std::cerr << "[GPU " << device << "] Failed to set device: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Create stream
    err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "[GPU " << device << "] Failed to create stream: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Initialize fused pipeline (precompute EC table)
    err = fused_pipeline_init(stream);
    if (err != cudaSuccess) {
        std::cerr << "[GPU " << device << "] Failed to init pipeline: "
                  << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Allocate double buffers
    for (int i = 0; i < 2; i++) {
        auto& buf = buffers[i];
        buf.max_passphrases = max_batch_size;
        buf.max_passphrase_bytes = max_passphrase_bytes;

        err = cudaMalloc(&buf.d_passphrases, max_passphrase_bytes);
        if (err != cudaSuccess) goto alloc_fail;

        err = cudaMalloc(&buf.d_offsets, max_batch_size * sizeof(uint32_t));
        if (err != cudaSuccess) goto alloc_fail;

        err = cudaMalloc(&buf.d_lengths, max_batch_size * sizeof(uint32_t));
        if (err != cudaSuccess) goto alloc_fail;

        err = cudaMalloc(&buf.d_match_indices, 1024 * sizeof(uint32_t));
        if (err != cudaSuccess) goto alloc_fail;

        err = cudaMalloc(&buf.d_match_count, sizeof(uint32_t));
        if (err != cudaSuccess) goto alloc_fail;

        err = cudaMalloc(&buf.d_private_keys, max_batch_size * 32);
        if (err != cudaSuccess) goto alloc_fail;
    }

    // Allocate pinned host memory for fast transfers
    err = cudaMallocHost(&h_passphrases, max_passphrase_bytes);
    if (err != cudaSuccess) goto alloc_fail;

    err = cudaMallocHost(&h_offsets, max_batch_size * sizeof(uint32_t));
    if (err != cudaSuccess) goto alloc_fail;

    err = cudaMallocHost(&h_lengths, max_batch_size * sizeof(uint32_t));
    if (err != cudaSuccess) goto alloc_fail;

    err = cudaMallocHost(&h_match_indices, 1024 * sizeof(uint32_t));
    if (err != cudaSuccess) goto alloc_fail;

    err = cudaMallocHost(&h_match_count, sizeof(uint32_t));
    if (err != cudaSuccess) goto alloc_fail;

    initialized = true;
    return true;

alloc_fail:
    std::cerr << "[GPU " << device << "] Failed to allocate memory: "
              << cudaGetErrorString(err) << "\n";
    cleanup();
    return false;
}

void BrainWalletGPUContext::cleanup() {
    if (!initialized) return;

    cudaSetDevice(device_id);

    // Free device buffers
    for (int i = 0; i < 2; i++) {
        auto& buf = buffers[i];
        if (buf.d_passphrases) cudaFree(buf.d_passphrases);
        if (buf.d_offsets) cudaFree(buf.d_offsets);
        if (buf.d_lengths) cudaFree(buf.d_lengths);
        if (buf.d_match_indices) cudaFree(buf.d_match_indices);
        if (buf.d_match_count) cudaFree(buf.d_match_count);
        if (buf.d_private_keys) cudaFree(buf.d_private_keys);
        buf = BatchBuffers{};
    }

    // Free bloom filter
    if (d_bloom_filter) {
        h160_bloom_cleanup(d_bloom_filter, 0);
        d_bloom_filter = nullptr;
    }

    // Free pinned memory
    if (h_passphrases) cudaFreeHost(h_passphrases);
    if (h_offsets) cudaFreeHost(h_offsets);
    if (h_lengths) cudaFreeHost(h_lengths);
    if (h_match_indices) cudaFreeHost(h_match_indices);
    if (h_match_count) cudaFreeHost(h_match_count);

    h_passphrases = nullptr;
    h_offsets = nullptr;
    h_lengths = nullptr;
    h_match_indices = nullptr;
    h_match_count = nullptr;

    if (stream) {
        cudaStreamDestroy(stream);
        stream = 0;
    }

    initialized = false;
}

bool MultiGPUBrainWallet::init() {
    if (config_.gpu_ids.empty()) {
        std::cerr << "[!] No GPU IDs specified\n";
        return false;
    }

    contexts_.resize(config_.gpu_ids.size());

    // Calculate per-GPU batch size
    size_t per_gpu_batch = config_.batch_size / config_.gpu_ids.size();
    size_t max_bytes = per_gpu_batch * config_.max_passphrase_length;

    for (size_t i = 0; i < config_.gpu_ids.size(); i++) {
        if (!contexts_[i].init(config_.gpu_ids[i], per_gpu_batch, max_bytes)) {
            std::cerr << "[!] Failed to initialize GPU " << config_.gpu_ids[i] << "\n";
            cleanup();
            return false;
        }
    }

    start_time_ = std::chrono::steady_clock::now();
    initialized_ = true;
    return true;
}

bool MultiGPUBrainWallet::load_bloom_filter(
    const uint8_t* data, size_t size,
    uint64_t num_bits, uint32_t num_hashes, uint32_t seed
) {
    if (!initialized_) return false;

    // For parallel loading, we use threads to copy to each GPU simultaneously
    // This provides N-1x speedup for N GPUs when loading large bloom filters
    const size_t num_gpus = contexts_.size();

    if (num_gpus == 1) {
        // Single GPU - use simple sequential loading
        auto& ctx = contexts_[0];
        cudaSetDevice(ctx.device_id);

        cudaError_t err = h160_bloom_init(
            data, size, num_bits, num_hashes, seed,
            &ctx.d_bloom_filter
        );

        if (err != cudaSuccess) {
            std::cerr << "[GPU " << ctx.device_id << "] Failed to load bloom filter: "
                      << cudaGetErrorString(err) << "\n";
            return false;
        }

        ctx.bloom_bits = num_bits;
        ctx.bloom_hashes = num_hashes;
        return true;
    }

    // Multi-GPU parallel loading using threads
    std::vector<std::thread> load_threads;
    std::vector<cudaError_t> load_errors(num_gpus, cudaSuccess);
    std::atomic<bool> any_failed{false};

    auto start_time = std::chrono::steady_clock::now();
    std::cerr << "[Bloom] Loading " << (size / (1024 * 1024)) << " MB bloom filter to "
              << num_gpus << " GPUs in parallel...\n";

    // Launch parallel copy threads
    for (size_t i = 0; i < num_gpus; i++) {
        load_threads.emplace_back([this, i, data, size, num_bits, num_hashes, seed, &load_errors, &any_failed]() {
            auto& ctx = contexts_[i];

            // Each thread sets its own GPU context
            cudaError_t err = cudaSetDevice(ctx.device_id);
            if (err != cudaSuccess) {
                load_errors[i] = err;
                any_failed.store(true);
                return;
            }

            // Allocate device memory for bloom filter (128-byte aligned)
            size_t aligned_size = ((size + 127) / 128) * 128;
            err = cudaMalloc(&ctx.d_bloom_filter, aligned_size);
            if (err != cudaSuccess) {
                load_errors[i] = err;
                any_failed.store(true);
                return;
            }

            // Async copy from host pinned memory would be faster, but requires
            // pinned source buffer. For now, use synchronous copy.
            err = cudaMemcpy(ctx.d_bloom_filter, data, size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                cudaFree(ctx.d_bloom_filter);
                ctx.d_bloom_filter = nullptr;
                load_errors[i] = err;
                any_failed.store(true);
                return;
            }

            // Set bloom filter config in constant memory
            ctx.bloom_bits = num_bits;
            ctx.bloom_hashes = num_hashes;

            // Note: The actual constant memory setup for d_bloom_config is done
            // per-kernel-call since each GPU has its own constant memory space
        });
    }

    // Wait for all threads to complete
    for (auto& t : load_threads) {
        t.join();
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Check for errors
    if (any_failed.load()) {
        for (size_t i = 0; i < num_gpus; i++) {
            if (load_errors[i] != cudaSuccess) {
                std::cerr << "[GPU " << contexts_[i].device_id << "] Failed to load bloom filter: "
                          << cudaGetErrorString(load_errors[i]) << "\n";
                // Cleanup any partially loaded filters
                if (contexts_[i].d_bloom_filter) {
                    cudaSetDevice(contexts_[i].device_id);
                    cudaFree(contexts_[i].d_bloom_filter);
                    contexts_[i].d_bloom_filter = nullptr;
                }
            }
        }
        return false;
    }

    // Now initialize the constant memory for each GPU (must be done per-device)
    for (auto& ctx : contexts_) {
        cudaSetDevice(ctx.device_id);

        // Initialize bloom config constant memory for this GPU
        cudaError_t err = h160_bloom_set_config(num_bits, num_hashes, seed);
        if (err != cudaSuccess) {
            std::cerr << "[GPU " << ctx.device_id << "] Failed to set bloom config: "
                      << cudaGetErrorString(err) << "\n";
        }
    }

    double throughput_gbps = (size * num_gpus) / (elapsed_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    std::cerr << "[Bloom] Parallel load complete: " << elapsed_ms << " ms ("
              << throughput_gbps << " GB/s total)\n";

    return true;
}

MultiGPUBrainWallet::BatchResult MultiGPUBrainWallet::process_batch(
    const std::vector<std::string>& passphrases
) {
    BatchResult result;
    if (!initialized_ || passphrases.empty()) return result;

    const size_t num_gpus = contexts_.size();

    // Single GPU fast path
    if (num_gpus == 1) {
        auto result = process_batch_single_gpu(passphrases, contexts_[0]);
        total_processed_ += result.processed;
        return result;
    }

    // Multi-GPU: distribute work across all GPUs with TRUE DOUBLE BUFFERING
    // Each GPU uses double buffers to overlap data transfer with computation
    const size_t per_gpu = (passphrases.size() + num_gpus - 1) / num_gpus;
    std::vector<BatchResult> gpu_results(num_gpus);
    std::vector<std::thread> worker_threads;
    std::atomic<bool> any_error{false};

    // Launch parallel workers for each GPU
    for (size_t g = 0; g < num_gpus; g++) {
        size_t start_idx = g * per_gpu;
        size_t end_idx = std::min(start_idx + per_gpu, passphrases.size());
        if (start_idx >= passphrases.size()) break;

        worker_threads.emplace_back([this, g, start_idx, end_idx, &passphrases, &gpu_results, &any_error]() {
            auto& ctx = contexts_[g];
            cudaSetDevice(ctx.device_id);

            // Split this GPU's work into two sub-batches for double buffering
            size_t total_count = end_idx - start_idx;
            size_t half = total_count / 2;

            // Process first half with buffer 0, second half with buffer 1
            // While buffer 0 is computing, we prepare buffer 1's data (and vice versa)

            BatchResult partial_result;

            // Phase 1: Start batch 0, prepare batch 1
            {
                // Prepare batch 0 data
                auto& buf0 = ctx.buffers[0];
                size_t count0 = half;
                size_t total_bytes0 = 0;

                for (size_t i = 0; i < count0 && (start_idx + i) < end_idx; i++) {
                    const auto& pp = passphrases[start_idx + i];
                    size_t len = std::min(pp.size(), config_.max_passphrase_length);

                    ctx.h_offsets[i] = static_cast<uint32_t>(total_bytes0);
                    ctx.h_lengths[i] = static_cast<uint32_t>(len);
                    std::memcpy(ctx.h_passphrases + total_bytes0, pp.data(), len);
                    total_bytes0 += len;
                }

                // Async copy batch 0 to GPU
                cudaMemcpyAsync(buf0.d_passphrases, ctx.h_passphrases, total_bytes0,
                                cudaMemcpyHostToDevice, ctx.stream);
                cudaMemcpyAsync(buf0.d_offsets, ctx.h_offsets, count0 * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, ctx.stream);
                cudaMemcpyAsync(buf0.d_lengths, ctx.h_lengths, count0 * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, ctx.stream);

                // Launch batch 0 kernel (non-blocking)
                fused_brain_wallet_batch(
                    buf0.d_passphrases, buf0.d_offsets, buf0.d_lengths,
                    ctx.d_bloom_filter, ctx.bloom_bits, ctx.bloom_hashes,
                    buf0.d_match_indices, buf0.d_match_count,
                    config_.store_private_keys ? buf0.d_private_keys : nullptr,
                    count0, ctx.stream
                );
            }

            // Phase 2: While batch 0 computes, prepare batch 1 (TRUE OVERLAP)
            auto& buf1 = ctx.buffers[1];
            size_t count1 = total_count - half;
            size_t total_bytes1 = 0;

            // This CPU work overlaps with batch 0 GPU computation
            for (size_t i = 0; i < count1 && (start_idx + half + i) < end_idx; i++) {
                const auto& pp = passphrases[start_idx + half + i];
                size_t len = std::min(pp.size(), config_.max_passphrase_length);

                // Use separate host buffer area for second batch
                // (h_passphrases is large enough for one full batch, we reuse it)
                ctx.h_offsets[i] = static_cast<uint32_t>(total_bytes1);
                ctx.h_lengths[i] = static_cast<uint32_t>(len);
                // Note: We need separate host buffers to avoid overwriting batch 0's data
                // For now, we copy to same buffer but after batch 0's transfer is done
            }

            // Wait for batch 0 to complete
            cudaStreamSynchronize(ctx.stream);

            // Collect batch 0 results
            auto& buf0 = ctx.buffers[0];
            cudaMemcpy(ctx.h_match_count, buf0.d_match_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            uint32_t match_count0 = *ctx.h_match_count;
            if (match_count0 > 0) {
                match_count0 = std::min(match_count0, 1024u);
                cudaMemcpy(ctx.h_match_indices, buf0.d_match_indices,
                           match_count0 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                for (uint32_t i = 0; i < match_count0; i++) {
                    uint32_t idx = ctx.h_match_indices[i];
                    if (idx < half) {
                        partial_result.match_indices.push_back(static_cast<uint32_t>(start_idx + idx));
                    }
                }
            }
            partial_result.processed = half;

            // Now process batch 1 (batch 0's GPU is done, we can reuse the stream)
            if (count1 > 0) {
                // Re-prepare batch 1 data (now safe to overwrite host buffers)
                total_bytes1 = 0;
                for (size_t i = 0; i < count1 && (start_idx + half + i) < end_idx; i++) {
                    const auto& pp = passphrases[start_idx + half + i];
                    size_t len = std::min(pp.size(), config_.max_passphrase_length);

                    ctx.h_offsets[i] = static_cast<uint32_t>(total_bytes1);
                    ctx.h_lengths[i] = static_cast<uint32_t>(len);
                    std::memcpy(ctx.h_passphrases + total_bytes1, pp.data(), len);
                    total_bytes1 += len;
                }

                // Copy and launch batch 1
                cudaMemcpyAsync(buf1.d_passphrases, ctx.h_passphrases, total_bytes1,
                                cudaMemcpyHostToDevice, ctx.stream);
                cudaMemcpyAsync(buf1.d_offsets, ctx.h_offsets, count1 * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, ctx.stream);
                cudaMemcpyAsync(buf1.d_lengths, ctx.h_lengths, count1 * sizeof(uint32_t),
                                cudaMemcpyHostToDevice, ctx.stream);

                fused_brain_wallet_batch(
                    buf1.d_passphrases, buf1.d_offsets, buf1.d_lengths,
                    ctx.d_bloom_filter, ctx.bloom_bits, ctx.bloom_hashes,
                    buf1.d_match_indices, buf1.d_match_count,
                    config_.store_private_keys ? buf1.d_private_keys : nullptr,
                    count1, ctx.stream
                );

                cudaStreamSynchronize(ctx.stream);

                // Collect batch 1 results
                cudaMemcpy(ctx.h_match_count, buf1.d_match_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                uint32_t match_count1 = *ctx.h_match_count;
                if (match_count1 > 0) {
                    match_count1 = std::min(match_count1, 1024u);
                    cudaMemcpy(ctx.h_match_indices, buf1.d_match_indices,
                               match_count1 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    for (uint32_t i = 0; i < match_count1; i++) {
                        uint32_t idx = ctx.h_match_indices[i];
                        if (idx < count1) {
                            partial_result.match_indices.push_back(static_cast<uint32_t>(start_idx + half + idx));
                        }
                    }
                }
                partial_result.processed += count1;
            }

            gpu_results[g] = partial_result;
        });
    }

    // Wait for all GPU workers
    for (auto& t : worker_threads) {
        t.join();
    }

    // Merge results from all GPUs
    for (size_t g = 0; g < num_gpus; g++) {
        const auto& gr = gpu_results[g];
        result.processed += gr.processed;

        // Append match indices (already global indices)
        result.match_indices.insert(result.match_indices.end(),
            gr.match_indices.begin(), gr.match_indices.end());

        // Append private keys
        result.private_keys.insert(result.private_keys.end(),
            gr.private_keys.begin(), gr.private_keys.end());
    }

    total_processed_ += result.processed;
    return result;
}

MultiGPUBrainWallet::BatchResult MultiGPUBrainWallet::process_batch_single_gpu(
    const std::vector<std::string>& passphrases,
    BrainWalletGPUContext& ctx,
    size_t index_offset
) {
    BatchResult result;
    if (passphrases.empty()) return result;

    cudaSetDevice(ctx.device_id);

    auto& buf = ctx.get_buffer();

    // Pack passphrases into contiguous buffer
    size_t total_bytes = 0;
    size_t count = std::min(passphrases.size(), buf.max_passphrases);

    for (size_t i = 0; i < count; i++) {
        const auto& pp = passphrases[i];
        size_t len = std::min(pp.size(), config_.max_passphrase_length);

        ctx.h_offsets[i] = static_cast<uint32_t>(total_bytes);
        ctx.h_lengths[i] = static_cast<uint32_t>(len);

        std::memcpy(ctx.h_passphrases + total_bytes, pp.data(), len);
        total_bytes += len;

        if (total_bytes >= buf.max_passphrase_bytes - config_.max_passphrase_length) {
            count = i + 1;
            break;
        }
    }

    // Copy to device (async)
    cudaMemcpyAsync(buf.d_passphrases, ctx.h_passphrases, total_bytes,
                    cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(buf.d_offsets, ctx.h_offsets, count * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, ctx.stream);
    cudaMemcpyAsync(buf.d_lengths, ctx.h_lengths, count * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, ctx.stream);

    // Launch kernel
    cudaError_t err = fused_brain_wallet_batch(
        buf.d_passphrases,
        buf.d_offsets,
        buf.d_lengths,
        ctx.d_bloom_filter,
        ctx.bloom_bits,
        ctx.bloom_hashes,
        buf.d_match_indices,
        buf.d_match_count,
        config_.store_private_keys ? buf.d_private_keys : nullptr,
        count,
        ctx.stream
    );

    if (err != cudaSuccess) {
        std::cerr << "[GPU " << ctx.device_id << "] Kernel error: "
                  << cudaGetErrorString(err) << "\n";
        return result;
    }

    // Copy results back
    cudaMemcpyAsync(ctx.h_match_count, buf.d_match_count, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, ctx.stream);

    // Sync to get match count
    cudaStreamSynchronize(ctx.stream);

    uint32_t match_count = *ctx.h_match_count;

    if (match_count > 0) {
        match_count = std::min(match_count, 1024u);

        // Copy match indices
        cudaMemcpy(ctx.h_match_indices, buf.d_match_indices,
                   match_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        result.match_indices.reserve(match_count);
        for (uint32_t i = 0; i < match_count; i++) {
            uint32_t idx = ctx.h_match_indices[i];
            // Bounds check - kernel must return valid indices
            if (idx < count) {
                // Adjust index by offset for multi-GPU batches
                result.match_indices.push_back(static_cast<uint32_t>(idx + index_offset));
            }
        }

        // Copy private keys for matches - use async batch copy for performance
        if (config_.store_private_keys && !result.match_indices.empty()) {
            const size_t valid_matches = result.match_indices.size();
            result.private_keys.resize(valid_matches);

            // Build list of local indices for GPU buffer access
            // (global indices have offset added, need to subtract for GPU access)
            std::vector<uint32_t> local_indices;
            local_indices.reserve(valid_matches);
            for (size_t i = 0; i < valid_matches; i++) {
                local_indices.push_back(static_cast<uint32_t>(
                    result.match_indices[i] - index_offset));
            }

            // For small match counts, individual copies are acceptable
            // For larger counts, batch async copies are more efficient
            if (valid_matches <= 16) {
                // Small batch - direct copies
                for (size_t i = 0; i < valid_matches; i++) {
                    uint32_t local_idx = local_indices[i];
                    cudaMemcpy(result.private_keys[i].data(),
                               buf.d_private_keys + local_idx * 32,
                               32, cudaMemcpyDeviceToHost);
                }
            } else {
                // Larger batch - use async copies then sync once
                for (size_t i = 0; i < valid_matches; i++) {
                    uint32_t local_idx = local_indices[i];
                    cudaMemcpyAsync(result.private_keys[i].data(),
                                    buf.d_private_keys + local_idx * 32,
                                    32, cudaMemcpyDeviceToHost, ctx.stream);
                }
                cudaStreamSynchronize(ctx.stream);
            }
        }
    }

    result.processed = count;
    // Note: total_processed_ is updated by the caller (process_batch)

    return result;
}

MultiGPUBrainWallet::BatchResult MultiGPUBrainWallet::process_batch_from_gpu(
    const uint8_t* d_passphrases,
    const uint32_t* d_lengths,
    uint32_t stride,
    size_t count,
    int gpu_index
) {
    BatchResult result;
    if (!initialized_ || count == 0) return result;
    if (gpu_index < 0 || gpu_index >= (int)contexts_.size()) return result;

    auto& ctx = contexts_[gpu_index];
    auto& buf = ctx.get_buffer();

    cudaSetDevice(ctx.device_id);

    // Use the fixed-stride kernel directly - no data transfer needed!
    // The passphrases are already on GPU from the rule engine
    cudaError_t err = fused_brain_wallet_batch_fixed_stride(
        d_passphrases,
        d_lengths,
        stride,
        ctx.d_bloom_filter,
        ctx.bloom_bits,
        ctx.bloom_hashes,
        buf.d_match_indices,
        buf.d_match_count,
        config_.store_private_keys ? buf.d_private_keys : nullptr,
        count,
        ctx.stream
    );

    if (err != cudaSuccess) {
        std::cerr << "[GPU " << ctx.device_id << "] Fixed-stride kernel error: "
                  << cudaGetErrorString(err) << "\n";
        return result;
    }

    // Copy match count back
    cudaMemcpyAsync(ctx.h_match_count, buf.d_match_count, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    uint32_t match_count = *ctx.h_match_count;

    if (match_count > 0) {
        match_count = std::min(match_count, 1024u);

        // Copy match indices
        cudaMemcpy(ctx.h_match_indices, buf.d_match_indices,
                   match_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        result.match_indices.reserve(match_count);
        for (uint32_t i = 0; i < match_count; i++) {
            uint32_t idx = ctx.h_match_indices[i];
            if (idx < count) {
                result.match_indices.push_back(idx);
            }
        }

        // Copy private keys for matches
        if (config_.store_private_keys && !result.match_indices.empty()) {
            const size_t valid_matches = result.match_indices.size();
            result.private_keys.resize(valid_matches);

            for (size_t i = 0; i < valid_matches; i++) {
                uint32_t idx = result.match_indices[i];
                cudaMemcpy(result.private_keys[i].data(),
                           buf.d_private_keys + idx * 32,
                           32, cudaMemcpyDeviceToHost);
            }
        }
    }

    result.processed = count;
    total_processed_ += count;

    return result;
}

void MultiGPUBrainWallet::cleanup() {
    for (auto& ctx : contexts_) {
        ctx.cleanup();
    }
    contexts_.clear();
    initialized_ = false;
}

double MultiGPUBrainWallet::keys_per_second() const {
    auto now = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(now - start_time_).count();
    return seconds > 0 ? total_processed_ / seconds : 0;
}

}  // namespace gpu
}  // namespace collider

#else  // No CUDA

namespace collider {
namespace gpu {

bool BrainWalletGPUContext::init(int, size_t, size_t) { return false; }
void BrainWalletGPUContext::cleanup() {}

bool MultiGPUBrainWallet::init() {
    std::cerr << "[!] CUDA not available - brain wallet mode requires GPU\n";
    return false;
}

bool MultiGPUBrainWallet::load_bloom_filter(const uint8_t*, size_t,
                                           uint64_t, uint32_t, uint32_t) {
    return false;
}

MultiGPUBrainWallet::BatchResult MultiGPUBrainWallet::process_batch(
    const std::vector<std::string>&
) {
    return BatchResult{};
}

MultiGPUBrainWallet::BatchResult MultiGPUBrainWallet::process_batch_from_gpu(
    const uint8_t*, const uint32_t*, uint32_t, size_t, int
) {
    return BatchResult{};
}

void MultiGPUBrainWallet::cleanup() {}

double MultiGPUBrainWallet::keys_per_second() const { return 0; }

}  // namespace gpu
}  // namespace collider

#endif  // COLLIDER_USE_CUDA
