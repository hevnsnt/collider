/**
 * Collider GPU Puzzle Solver
 *
 * Multi-GPU implementation using optimized kernels with:
 * - Precomputed EC tables (16x speedup)
 * - Strided incremental search (256 keys/thread)
 * - Inline SHA256/RIPEMD160
 * - Montgomery batch inversion
 *
 * Target: 400-800M keys/sec per RTX 3090 class GPU
 */

#include "puzzle_gpu.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <algorithm>

// =============================================================================
// OPTIMIZED KERNEL DECLARATIONS (outside namespace for proper C linkage)
// =============================================================================

// These are implemented in puzzle_optimized.cu
extern "C" {
    ::cudaError_t init_puzzle_optimized(::cudaStream_t stream);
    ::cudaError_t cleanup_puzzle_optimized();
    ::cudaError_t puzzle_search_batch_optimized(
        uint64_t range_start_lo,
        uint64_t range_start_hi,
        uint64_t batch_size,
        const uint8_t* d_target_hash160,
        uint64_t* d_match_key_lo,
        uint64_t* d_match_key_hi,
        uint32_t* d_match_found,
        ::cudaStream_t stream
    );
}

namespace collider {
namespace gpu {

// =============================================================================
// SINGLE GPU SOLVER IMPLEMENTATION
// =============================================================================

GPUPuzzleSolver::~GPUPuzzleSolver() {
    if (initialized_) {
        cudaSetDevice(device_id_);
        cleanup_puzzle_optimized();
        if (d_target_hash160_) cudaFree(d_target_hash160_);
        if (d_match_key_lo_) cudaFree(d_match_key_lo_);
        if (d_match_key_hi_) cudaFree(d_match_key_hi_);
        if (d_match_found_) cudaFree(d_match_found_);
        if (stream_) cudaStreamDestroy(stream_);
    }
}

GPUPuzzleSolver& GPUPuzzleSolver::operator=(GPUPuzzleSolver&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources first
        if (initialized_) {
            cudaSetDevice(device_id_);
            cleanup_puzzle_optimized();
            if (d_target_hash160_) cudaFree(d_target_hash160_);
            if (d_match_key_lo_) cudaFree(d_match_key_lo_);
            if (d_match_key_hi_) cudaFree(d_match_key_hi_);
            if (d_match_found_) cudaFree(d_match_found_);
            if (stream_) cudaStreamDestroy(stream_);
        }

        // Take ownership of other's resources
        initialized_ = other.initialized_;
        device_id_ = other.device_id_;
        stream_ = other.stream_;
        d_target_hash160_ = other.d_target_hash160_;
        d_match_key_lo_ = other.d_match_key_lo_;
        d_match_key_hi_ = other.d_match_key_hi_;
        d_match_found_ = other.d_match_found_;

        // Nullify moved-from object
        other.initialized_ = false;
        other.stream_ = nullptr;
        other.d_target_hash160_ = nullptr;
        other.d_match_key_lo_ = nullptr;
        other.d_match_key_hi_ = nullptr;
        other.d_match_found_ = nullptr;
    }
    return *this;
}

bool GPUPuzzleSolver::init(int device_id) {
    if (initialized_) return true;

    device_id_ = device_id;
    cudaError_t err;

    // Set device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU %d] Failed to set device: %s\n", device_id, cudaGetErrorString(err));
        return false;
    }

    // Create stream for async operations
    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU %d] Failed to create stream: %s\n", device_id, cudaGetErrorString(err));
        return false;
    }

    // Initialize optimized kernel (generates precomputed tables)
    err = init_puzzle_optimized(stream_);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU %d] Failed to init optimized kernel: %s\n", device_id, cudaGetErrorString(err));
        cudaStreamDestroy(stream_);
        return false;
    }

    // Allocate device memory for results
    err = cudaMalloc(&d_target_hash160_, 20);
    if (err != cudaSuccess) goto alloc_failed;

    err = cudaMalloc(&d_match_key_lo_, sizeof(uint64_t));
    if (err != cudaSuccess) goto alloc_failed;

    err = cudaMalloc(&d_match_key_hi_, sizeof(uint64_t));
    if (err != cudaSuccess) goto alloc_failed;

    err = cudaMalloc(&d_match_found_, sizeof(uint32_t));
    if (err != cudaSuccess) goto alloc_failed;

    initialized_ = true;
    return true;

alloc_failed:
    fprintf(stderr, "[GPU %d] Memory allocation failed: %s\n", device_id, cudaGetErrorString(err));
    cleanup_puzzle_optimized();
    if (d_target_hash160_) cudaFree(d_target_hash160_);
    if (d_match_key_lo_) cudaFree(d_match_key_lo_);
    if (d_match_key_hi_) cudaFree(d_match_key_hi_);
    if (d_match_found_) cudaFree(d_match_found_);
    cudaStreamDestroy(stream_);
    return false;
}

bool GPUPuzzleSolver::set_target(const std::array<uint8_t, 20>& hash160) {
    if (!initialized_) return false;

    cudaSetDevice(device_id_);
    cudaError_t err = cudaMemcpyAsync(
        d_target_hash160_,
        hash160.data(),
        20,
        cudaMemcpyHostToDevice,
        stream_
    );

    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU %d] Failed to copy target hash: %s\n", device_id_, cudaGetErrorString(err));
        return false;
    }

    cudaStreamSynchronize(stream_);
    return true;
}

bool GPUPuzzleSolver::search_batch(
    uint64_t start_lo, uint64_t start_hi,
    uint64_t batch_size,
    uint64_t& found_lo, uint64_t& found_hi
) {
    if (!initialized_) return false;

    cudaSetDevice(device_id_);
    cudaError_t err;

    // Reset match flag
    err = cudaMemsetAsync(d_match_found_, 0, sizeof(uint32_t), stream_);
    if (err != cudaSuccess) return false;

    // Execute optimized search kernel
    err = puzzle_search_batch_optimized(
        start_lo,
        start_hi,
        batch_size,
        d_target_hash160_,
        d_match_key_lo_,
        d_match_key_hi_,
        d_match_found_,
        stream_
    );

    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU %d] Kernel failed: %s\n", device_id_, cudaGetErrorString(err));
        return false;
    }

    // Synchronize and check results
    cudaStreamSynchronize(stream_);

    uint32_t match_found = 0;
    cudaMemcpy(&match_found, d_match_found_, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (match_found) {
        cudaMemcpy(&found_lo, d_match_key_lo_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&found_hi, d_match_key_hi_, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        return true;
    }

    return false;
}

// =============================================================================
// GPU BATCH SIZE CALIBRATION
// =============================================================================

uint64_t GPUPuzzleSolver::calibrate_batch_size(int iterations_per_test) {
    if (!initialized_) return 0;

    cudaSetDevice(device_id_);

    // Get device info for context
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id_);

    printf("[GPU %d] Calibrating batch size for %s (%d SMs)...\n",
           device_id_, props.name, props.multiProcessorCount);

    // Test batch sizes from 1M to 128M
    const uint64_t test_sizes[] = {
        1'000'000,    // 1M
        2'000'000,    // 2M
        4'000'000,    // 4M
        8'000'000,    // 8M
        16'000'000,   // 16M
        32'000'000,   // 32M
        64'000'000,   // 64M
        128'000'000   // 128M
    };
    const int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    uint64_t best_batch_size = 4'000'000;  // Default
    double best_rate = 0.0;

    // Use a dummy target hash (we don't care about finding a match)
    std::array<uint8_t, 20> dummy_hash = {0};
    set_target(dummy_hash);

    for (int s = 0; s < num_sizes; s++) {
        uint64_t batch_size = test_sizes[s];

        // Skip sizes that would use too much memory
        // Each key needs ~32 bytes for EC point, rough estimate
        size_t estimated_mem = batch_size * 32;
        if (estimated_mem > props.totalGlobalMem / 2) {
            printf("[GPU %d]   Skipping %lluM (would exceed memory)\n",
                   device_id_, (unsigned long long)(batch_size / 1'000'000));
            continue;
        }

        // Warmup run
        uint64_t found_lo, found_hi;
        search_batch(0x1000000000000000ULL, 0, batch_size, found_lo, found_hi);
        cudaStreamSynchronize(stream_);

        // Timed runs
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < iterations_per_test; i++) {
            // Use different start positions to avoid any caching effects
            uint64_t start_offset = (uint64_t)i * batch_size;
            search_batch(0x1000000000000000ULL + start_offset, 0, batch_size, found_lo, found_hi);
        }

        cudaStreamSynchronize(stream_);
        auto end = std::chrono::steady_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        uint64_t total_keys = (uint64_t)iterations_per_test * batch_size;
        double rate = (total_keys / elapsed_ms) * 1000.0;  // keys/sec

        printf("[GPU %d]   Batch %3lluM: %.1f M/s\n",
               device_id_,
               (unsigned long long)(batch_size / 1'000'000),
               rate / 1'000'000.0);

        if (rate > best_rate) {
            best_rate = rate;
            best_batch_size = batch_size;
        }
    }

    printf("[GPU %d] Optimal batch size: %lluM (%.1f M/s)\n",
           device_id_,
           (unsigned long long)(best_batch_size / 1'000'000),
           best_rate / 1'000'000.0);

    return best_batch_size;
}

// =============================================================================
// MULTI-GPU SOLVER IMPLEMENTATION
// =============================================================================

MultiGPUPuzzleSolver::~MultiGPUPuzzleSolver() {
    solvers_.clear();
}

bool MultiGPUPuzzleSolver::init(const Config& config) {
    config_ = config;

    // Get total GPU count
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    if (device_count == 0) {
        fprintf(stderr, "[MultiGPU] No CUDA devices found\n");
        return false;
    }

    // Initialize each requested GPU
    for (int gpu_id : config.gpu_ids) {
        if (gpu_id >= device_count) {
            fprintf(stderr, "[MultiGPU] GPU %d not available (only %d GPUs)\n", gpu_id, device_count);
            continue;
        }

        GPUPuzzleSolver solver;
        if (solver.init(gpu_id)) {
            // Get device name for logging
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, gpu_id);
            printf("[MultiGPU] Initialized GPU %d: %s (%d SMs, %.1f GB)\n",
                   gpu_id, props.name, props.multiProcessorCount,
                   props.totalGlobalMem / (1024.0 * 1024 * 1024));

            solvers_.push_back(std::move(solver));
        } else {
            fprintf(stderr, "[MultiGPU] Failed to initialize GPU %d\n", gpu_id);
        }
    }

    if (solvers_.empty()) {
        fprintf(stderr, "[MultiGPU] No GPUs initialized successfully\n");
        return false;
    }

    printf("[MultiGPU] Initialized %d GPUs for parallel search\n", (int)solvers_.size());
    return true;
}

bool MultiGPUPuzzleSolver::init(const std::vector<int>& gpu_ids) {
    Config config;
    config.gpu_ids = gpu_ids;
    return init(config);
}

bool MultiGPUPuzzleSolver::set_target(const std::array<uint8_t, 20>& hash160) {
    target_hash160_ = hash160;

    for (auto& solver : solvers_) {
        if (!solver.set_target(hash160)) {
            return false;
        }
    }
    return true;
}

MultiGPUPuzzleSolver::Result MultiGPUPuzzleSolver::search_range(
    uint64_t start_lo, uint64_t start_hi,
    uint64_t end_lo, uint64_t end_hi
) {
    Result result;
    result.found = false;
    result.total_checked = 0;

    if (solvers_.empty()) return result;

    const int num_gpus = static_cast<int>(solvers_.size());
    const uint64_t batch_size = config_.batch_size_per_gpu;

    // Reset state
    found_.store(false);
    found_key_lo_.store(0);
    found_key_hi_.store(0);
    found_gpu_id_.store(-1);

    // Tracking for progress
    std::atomic<uint64_t> total_checked{0};
    auto start_time = std::chrono::steady_clock::now();
    auto last_progress = start_time;

    // Current position for each GPU (they work on interleaved ranges)
    std::vector<uint64_t> gpu_pos_lo(num_gpus);
    std::vector<uint64_t> gpu_pos_hi(num_gpus);

    // Initialize starting positions (GPUs work on interleaved blocks)
    for (int i = 0; i < num_gpus; i++) {
        gpu_pos_lo[i] = start_lo + (batch_size * i);
        gpu_pos_hi[i] = start_hi;
        if (gpu_pos_lo[i] < start_lo) gpu_pos_hi[i]++;  // Handle overflow
    }

    // Stride for advancing all GPUs together
    uint64_t stride = batch_size * num_gpus;

    while (!found_.load()) {
        // Check if all GPUs have exceeded the range
        bool all_done = true;
        for (int i = 0; i < num_gpus; i++) {
            if (gpu_pos_hi[i] < end_hi ||
                (gpu_pos_hi[i] == end_hi && gpu_pos_lo[i] < end_lo)) {
                all_done = false;
                break;
            }
        }
        if (all_done) break;

        // Launch batches on all GPUs in parallel
        std::vector<std::thread> threads;
        for (int i = 0; i < num_gpus; i++) {
            // Skip if this GPU is past end
            if (gpu_pos_hi[i] > end_hi ||
                (gpu_pos_hi[i] == end_hi && gpu_pos_lo[i] >= end_lo)) {
                continue;
            }

            threads.emplace_back([this, i, batch_size,
                                  pos_lo = gpu_pos_lo[i],
                                  pos_hi = gpu_pos_hi[i],
                                  &total_checked]() {
                uint64_t found_lo, found_hi;
                if (solvers_[i].search_batch(pos_lo, pos_hi, batch_size, found_lo, found_hi)) {
                    // FIXED: Use compare_exchange to prevent race condition
                    // Only the first GPU to find stores its result
                    bool expected = false;
                    if (found_.compare_exchange_strong(expected, true)) {
                        found_key_lo_.store(found_lo, std::memory_order_relaxed);
                        found_key_hi_.store(found_hi, std::memory_order_relaxed);
                        found_gpu_id_.store(i, std::memory_order_relaxed);
                    }
                }
                total_checked.fetch_add(batch_size);
            });
        }

        // Wait for all GPUs
        for (auto& t : threads) {
            t.join();
        }

        // Advance all GPU positions
        for (int i = 0; i < num_gpus; i++) {
            uint64_t new_lo = gpu_pos_lo[i] + stride;
            if (new_lo < gpu_pos_lo[i]) gpu_pos_hi[i]++;  // Handle overflow
            gpu_pos_lo[i] = new_lo;
        }

        // Progress callback
        auto now = std::chrono::steady_clock::now();
        if (progress_callback &&
            std::chrono::duration_cast<std::chrono::seconds>(now - last_progress).count() >= 1) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            double rate = (elapsed_ms > 0) ? (total_checked.load() * 1000.0 / elapsed_ms) : 0;

            if (!progress_callback(total_checked.load(), rate)) {
                break;  // User requested stop
            }
            last_progress = now;
        }
    }

    result.total_checked = total_checked.load();

    if (found_.load()) {
        result.found = true;
        result.key_lo = found_key_lo_.load();
        result.key_hi = found_key_hi_.load();
        result.gpu_id = found_gpu_id_.load();
    }

    return result;
}

bool MultiGPUPuzzleSolver::search_batch(
    uint64_t start_lo, uint64_t start_hi,
    uint64_t batch_size,
    uint64_t& found_lo, uint64_t& found_hi
) {
    if (solvers_.empty()) return false;

    const int num_gpus = static_cast<int>(solvers_.size());

    // For a single batch call, divide the batch among GPUs
    uint64_t keys_per_gpu = batch_size / num_gpus;
    if (keys_per_gpu == 0) keys_per_gpu = batch_size;  // At least 1 key per GPU

    std::atomic<bool> found{false};
    std::atomic<uint64_t> result_lo{0};
    std::atomic<uint64_t> result_hi{0};

    std::vector<std::thread> threads;

    for (int i = 0; i < num_gpus; i++) {
        // Calculate start position for this GPU
        uint64_t gpu_start_lo = start_lo + (keys_per_gpu * i);
        uint64_t gpu_start_hi = start_hi;
        if (gpu_start_lo < start_lo) gpu_start_hi++;  // Handle overflow

        // Last GPU gets remaining keys
        uint64_t gpu_batch = (i == num_gpus - 1) ?
            (batch_size - (keys_per_gpu * i)) : keys_per_gpu;

        threads.emplace_back([this, i, gpu_start_lo, gpu_start_hi, gpu_batch,
                              &found, &result_lo, &result_hi]() {
            uint64_t key_lo, key_hi;
            if (solvers_[i].search_batch(gpu_start_lo, gpu_start_hi, gpu_batch, key_lo, key_hi)) {
                // FIXED: Use compare_exchange to prevent race condition
                // Only the first GPU to find stores its result
                bool expected = false;
                if (found.compare_exchange_strong(expected, true)) {
                    result_lo.store(key_lo, std::memory_order_relaxed);
                    result_hi.store(key_hi, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    if (found.load()) {
        found_lo = result_lo.load();
        found_hi = result_hi.load();
        return true;
    }

    return false;
}

std::map<int, uint64_t> MultiGPUPuzzleSolver::calibrate_all(int iterations_per_test) {
    std::map<int, uint64_t> results;

    if (solvers_.empty()) {
        fprintf(stderr, "[MultiGPU] No GPUs initialized for calibration\n");
        return results;
    }

    printf("\n[MultiGPU] Starting batch size calibration for %d GPU(s)...\n",
           (int)solvers_.size());
    printf("[MultiGPU] Testing batch sizes: 1M, 2M, 4M, 8M, 16M, 32M, 64M, 128M\n");
    printf("[MultiGPU] Iterations per test: %d\n\n", iterations_per_test);

    // Calibrate each GPU (sequentially to avoid interference)
    uint64_t min_optimal = UINT64_MAX;

    for (auto& solver : solvers_) {
        int device_id = solver.device_id();
        uint64_t optimal = solver.calibrate_batch_size(iterations_per_test);
        results[device_id] = optimal;

        if (optimal < min_optimal) {
            min_optimal = optimal;
        }

        printf("\n");
    }

    // Use the minimum optimal batch size across all GPUs to ensure balanced load
    // (faster GPUs would otherwise wait for slower ones)
    if (min_optimal != UINT64_MAX) {
        config_.batch_size_per_gpu = min_optimal;
        printf("[MultiGPU] Using batch size %lluM (minimum across all GPUs for balanced load)\n",
               (unsigned long long)(min_optimal / 1'000'000));
    }

    return results;
}

}  // namespace gpu
}  // namespace collider
