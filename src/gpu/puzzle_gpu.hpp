/**
 * GPU Puzzle Solver Interface
 *
 * Header for GPU-accelerated Bitcoin puzzle key search.
 * Supports multi-GPU on CUDA and the unified GPU on Apple Silicon (Metal).
 *
 * Backend split:
 *   - COLLIDER_USE_CUDA: full multi-GPU path. Implementation in puzzle_gpu.cu
 *     and puzzle_optimized.cu.
 *   - COLLIDER_USE_METAL: single-GPU path on Apple silicon. Implementation
 *     in metal_multi_gpu_puzzle.mm and puzzle_metal.{hpp,mm}. The
 *     GPUPuzzleSolver class is intentionally still a stub here -- the
 *     CUDA single-GPU class was a v1.0-era convenience that the multi-GPU
 *     class supersedes; the Mac path only implements the multi-GPU surface
 *     that puzzle_solver.cpp actually calls.
 *   - else: CPU-only stubs that always return false from init().
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

#elif defined(COLLIDER_USE_METAL)

// =============================================================================
// MULTI-GPU PUZZLE SOLVER (Metal, macOS)
//
// Pre-1.4.1 the Mac standalone puzzle path hit the no-CUDA stub below and
// silently fell back to the CPU reference at ~30 KKeys/s. This branch
// implements the full surface on top of PuzzleMetalSolver. Apple silicon
// has one unified GPU per device, so num_gpus() always reports 1; gpu_ids
// in Config is honored for API parity but the solver only ever uses [0].
//
// GPUPuzzleSolver is intentionally a stub on Metal -- the CUDA single-GPU
// class is a v1.0-era convenience that the multi-GPU class supersedes;
// puzzle_solver.cpp uses MultiGPUPuzzleSolver exclusively. Implementation
// of MultiGPUPuzzleSolver is in src/gpu/metal_multi_gpu_puzzle.mm and is
// built when APPLE && COLLIDER_USE_METAL.
// =============================================================================
class GPUPuzzleSolver {
public:
    GPUPuzzleSolver() = default;
    ~GPUPuzzleSolver() = default;
    bool init(int /*device_id*/ = 0) { return false; }
    bool set_target(const std::array<uint8_t, 20>& /*hash160*/) { return false; }
    bool search_batch(uint64_t, uint64_t, uint64_t, uint64_t&, uint64_t&) { return false; }
    bool is_initialized() const { return false; }
    int device_id() const { return 0; }
};

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
        int gpu_id = -1;
    };
    using ProgressCallback = std::function<bool(uint64_t total_checked, double rate)>;

    MultiGPUPuzzleSolver();
    ~MultiGPUPuzzleSolver();

    MultiGPUPuzzleSolver(const MultiGPUPuzzleSolver&) = delete;
    MultiGPUPuzzleSolver& operator=(const MultiGPUPuzzleSolver&) = delete;

    bool init(const Config& config);
    bool init(const std::vector<int>& gpu_ids);
    bool set_target(const std::array<uint8_t, 20>& hash160);
    Result search_range(
        uint64_t start_lo, uint64_t start_hi,
        uint64_t end_lo, uint64_t end_hi
    );
    bool search_batch(
        uint64_t start_lo, uint64_t start_hi,
        uint64_t batch_size,
        uint64_t& found_lo, uint64_t& found_hi
    );

    int num_gpus() const;
    std::map<int, uint64_t> calibrate_all(int iterations_per_test = 5);
    void set_batch_size(uint64_t batch_size);
    uint64_t get_batch_size() const;

    ProgressCallback progress_callback;

private:
    struct Impl;
    Impl* impl_ = nullptr;

    // Search-loop state shared with progress callbacks. Atomic so a
    // future multi-threaded variant (e.g., dispatching while a callback
    // runs on a host thread) can write without races. The Metal path
    // is single-threaded today; the atomics are cheap and the contract
    // is the safe one to ship.
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
    std::map<int, uint64_t> calibrate_all(int /*iters*/ = 5) { return {}; }
    void set_batch_size(uint64_t) {}
    uint64_t get_batch_size() const { return 0; }
    ProgressCallback progress_callback;
};

#endif  // COLLIDER_USE_CUDA / COLLIDER_USE_METAL / stub

}  // namespace gpu
}  // namespace collider
