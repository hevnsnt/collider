/**
 * GPU Puzzle Solver Interface
 *
 * Header for GPU-accelerated Bitcoin puzzle key search.
 * Supports multi-GPU with optimized kernels.
 */

#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <map>

#ifdef COLLIDER_USE_CUDA
// Forward declare CUDA types OUTSIDE namespace to match global ::cudaStream_t
struct CUstream_st;
typedef CUstream_st* cudaStream_t;
#endif

namespace collider {
namespace gpu {

#ifdef COLLIDER_USE_CUDA

// Note: Optimized kernel API (init_puzzle_optimized, cleanup_puzzle_optimized,
// puzzle_search_batch_optimized) are declared in puzzle_gpu.cu with proper
// CUDA types. They are implementation details not exposed in this header.

// =============================================================================
// SINGLE GPU PUZZLE SOLVER (uses optimized kernel)
// =============================================================================
class GPUPuzzleSolver {
public:
    GPUPuzzleSolver()
        : initialized_(false)
        , device_id_(0)
        , stream_(nullptr)
        , d_target_hash160_(nullptr)
        , d_match_key_lo_(nullptr)
        , d_match_key_hi_(nullptr)
        , d_match_found_(nullptr) {}
    ~GPUPuzzleSolver();

    // Delete copy operations (CUDA resources cannot be copied)
    GPUPuzzleSolver(const GPUPuzzleSolver&) = delete;
    GPUPuzzleSolver& operator=(const GPUPuzzleSolver&) = delete;

    // Move constructor - takes ownership of CUDA resources
    GPUPuzzleSolver(GPUPuzzleSolver&& other) noexcept
        : initialized_(other.initialized_)
        , device_id_(other.device_id_)
        , stream_(other.stream_)
        , d_target_hash160_(other.d_target_hash160_)
        , d_match_key_lo_(other.d_match_key_lo_)
        , d_match_key_hi_(other.d_match_key_hi_)
        , d_match_found_(other.d_match_found_)
    {
        // Nullify moved-from object so its destructor does not free resources
        other.initialized_ = false;
        other.stream_ = nullptr;
        other.d_target_hash160_ = nullptr;
        other.d_match_key_lo_ = nullptr;
        other.d_match_key_hi_ = nullptr;
        other.d_match_found_ = nullptr;
    }

    // Move assignment operator
    GPUPuzzleSolver& operator=(GPUPuzzleSolver&& other) noexcept;

    bool init(int device_id = 0);
    bool set_target(const std::array<uint8_t, 20>& hash160);
    bool search_batch(
        uint64_t start_lo, uint64_t start_hi,
        uint64_t batch_size,
        uint64_t& found_lo, uint64_t& found_hi
    );
    bool is_initialized() const { return initialized_; }
    int device_id() const { return device_id_; }

    // Calibration: find optimal batch size for this GPU
    // Tests various batch sizes and returns the one with highest throughput
    // iterations_per_test: how many batches to run for each size (more = more accurate)
    uint64_t calibrate_batch_size(int iterations_per_test = 5);

private:
    bool initialized_;
    int device_id_;
    cudaStream_t stream_;
    uint8_t* d_target_hash160_;
    uint64_t* d_match_key_lo_;
    uint64_t* d_match_key_hi_;
    uint32_t* d_match_found_;
};

// =============================================================================
// MULTI-GPU PUZZLE SOLVER
// =============================================================================
class MultiGPUPuzzleSolver {
public:
    struct Config {
        std::vector<int> gpu_ids = {0};
        uint64_t batch_size_per_gpu = 4'000'000;  // 4M keys per GPU per batch
    };

    struct Result {
        bool found = false;
        uint64_t key_lo = 0;
        uint64_t key_hi = 0;
        uint64_t total_checked = 0;
        int gpu_id = -1;  // Which GPU found it
    };

    using ProgressCallback = std::function<bool(uint64_t total_checked, double rate)>;

    MultiGPUPuzzleSolver() = default;
    ~MultiGPUPuzzleSolver();

    // Initialize all GPUs
    bool init(const Config& config);
    bool init(const std::vector<int>& gpu_ids);

    // Set target hash160
    bool set_target(const std::array<uint8_t, 20>& hash160);

    // Search range across all GPUs
    // Returns when match found or range exhausted
    Result search_range(
        uint64_t start_lo, uint64_t start_hi,
        uint64_t end_lo, uint64_t end_hi
    );

    // Single batch search (for compatibility with zone-based scanning)
    // Returns true if match found, with key stored in found_lo/found_hi
    bool search_batch(
        uint64_t start_lo, uint64_t start_hi,
        uint64_t batch_size,
        uint64_t& found_lo, uint64_t& found_hi
    );

    // Get number of active GPUs
    int num_gpus() const { return static_cast<int>(solvers_.size()); }

    // Calibration: find optimal batch sizes for all GPUs
    // Returns map of device_id -> optimal_batch_size
    std::map<int, uint64_t> calibrate_all(int iterations_per_test = 5);

    // Set batch size for all GPUs (call after calibration or loading from config)
    void set_batch_size(uint64_t batch_size) { config_.batch_size_per_gpu = batch_size; }

    // Get current batch size
    uint64_t get_batch_size() const { return config_.batch_size_per_gpu; }

    // Progress callback
    ProgressCallback progress_callback;

private:
    std::vector<GPUPuzzleSolver> solvers_;
    std::array<uint8_t, 20> target_hash160_;
    Config config_;
    std::atomic<bool> found_{false};
    std::atomic<uint64_t> found_key_lo_{0};
    std::atomic<uint64_t> found_key_hi_{0};
    std::atomic<int> found_gpu_id_{-1};
};

#else

// =============================================================================
// CPU FALLBACK - stub implementations
// =============================================================================
class GPUPuzzleSolver {
public:
    GPUPuzzleSolver() : initialized_(false) {}
    bool init(int device_id = 0) { (void)device_id; return false; }
    bool set_target(const std::array<uint8_t, 20>& /*hash160*/) { return false; }
    bool search_batch(uint64_t, uint64_t, uint64_t, uint64_t&, uint64_t&) { return false; }
    bool is_initialized() const { return initialized_; }
    int device_id() const { return 0; }
private:
    bool initialized_;
};

class MultiGPUPuzzleSolver {
public:
    struct Config {
        std::vector<int> gpu_ids = {0};
        uint64_t batch_size_per_gpu = 4'000'000;
    };
    struct Result {
        bool found = false;
        uint64_t key_lo = 0;
        uint64_t key_hi = 0;
        uint64_t total_checked = 0;
        int gpu_id = -1;
    };
    using ProgressCallback = std::function<bool(uint64_t, double)>;

    bool init(const Config&) { return false; }
    bool init(const std::vector<int>&) { return false; }
    bool set_target(const std::array<uint8_t, 20>&) { return false; }
    Result search_range(uint64_t, uint64_t, uint64_t, uint64_t) { return {}; }
    bool search_batch(uint64_t, uint64_t, uint64_t, uint64_t&, uint64_t&) { return false; }
    int num_gpus() const { return 0; }
    ProgressCallback progress_callback;
};

#endif  // COLLIDER_USE_CUDA

}  // namespace gpu
}  // namespace collider
