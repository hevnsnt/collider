/**
 * GPU Kangaroo Solver Interface
 *
 * Pure C++ interface - CUDA implementation is in kangaroo_solver_gpu.cu
 * Falls back gracefully when CUDA is not available.
 */

#pragma once

#include <cstdint>
#include <array>
#include <functional>
#include <atomic>
#include "../core/puzzle_config.hpp"
#include "../core/crypto_cpu.hpp"

namespace collider {
namespace gpu {

/**
 * Result from GPU Kangaroo solve
 */
struct GPUKangarooResult {
    bool found;
    cpu::uint256_t private_key;
    uint64_t total_steps;
    uint64_t dp_count;
    double elapsed_seconds;
};

#ifdef COLLIDER_USE_CUDA

/**
 * High-level GPU Kangaroo Solver (CUDA)
 * Implementation in kangaroo_solver_gpu.cu
 */
class GPUKangarooManager {
public:
    // Configuration
    uint32_t dp_bits = 20;
    int num_kangaroos = 1 << 18;
    int steps_per_round = 256;
    bool debug_mode = false;
    std::atomic<bool> stop_flag{false};

    // Progress callback: (steps, dp_count, rate) -> continue?
    std::function<bool(uint64_t, uint64_t, double)> progress_callback;

    GPUKangarooManager();
    ~GPUKangarooManager();

    bool init(int device_id = 0);
    void set_range(const UInt256& start, const UInt256& end);
    void set_target_h160(const std::array<uint8_t, 20>& h160);
    void set_target_pubkey(const cpu::uint256_t& x, const cpu::uint256_t& y);
    GPUKangarooResult solve();

private:
    struct Impl;
    Impl* impl_;
};

/**
 * Multi-GPU Kangaroo Solver
 * Coordinates multiple GPUs running Kangaroo in parallel
 */
class MultiGPUKangarooManager {
public:
    // dp_bits acceptance window for this in-house multi-GPU backend.
    // Narrower than RCKangaroo's [14, 60] because the kernel's DP-buffer
    // sizing assumes <= 28 bits to avoid stalling kangaroos on
    // unreachable DP densities, and below 16 the host->device DP traffic
    // saturates PCIe with non-distinguishing points. Validate against
    // these in the standalone CLI.
    static constexpr int kMinDpBits = 16;
    static constexpr int kMaxDpBits = 28;

    // Configuration (applied to all GPUs)
    uint32_t dp_bits = 20;
    int num_kangaroos_per_gpu = 1 << 18;  // 262K per GPU
    int steps_per_round = 256;
    bool debug_mode = false;
    std::atomic<bool> stop_flag{false};

    // Progress callback: (total_steps, dp_count, rate) -> continue?
    std::function<bool(uint64_t, uint64_t, double)> progress_callback;

    MultiGPUKangarooManager();
    ~MultiGPUKangarooManager();

    // Initialize with specific GPU IDs, or empty vector for auto-detect
    bool init(const std::vector<int>& gpu_ids = {});
    int num_gpus() const;
    void set_range(const UInt256& start, const UInt256& end);
    void set_target_h160(const std::array<uint8_t, 20>& h160);
    void set_target_pubkey(const cpu::uint256_t& x, const cpu::uint256_t& y);
    GPUKangarooResult solve();

private:
    struct Impl;
    Impl* impl_;
};

#else

/**
 * Stub GPU Kangaroo Solver (no CUDA)
 * Always fails init() to trigger CPU fallback
 */
class GPUKangarooManager {
public:
    uint32_t dp_bits = 20;
    int num_kangaroos = 1 << 18;
    int steps_per_round = 256;
    bool debug_mode = false;
    std::atomic<bool> stop_flag{false};
    std::function<bool(uint64_t, uint64_t, double)> progress_callback;

    GPUKangarooManager() = default;
    ~GPUKangarooManager() = default;

    bool init(int /*device_id*/ = 0) { return false; }
    void set_range(const UInt256& /*start*/, const UInt256& /*end*/) {}
    void set_target_h160(const std::array<uint8_t, 20>& /*h160*/) {}
    void set_target_pubkey(const cpu::uint256_t& /*x*/, const cpu::uint256_t& /*y*/) {}
    GPUKangarooResult solve() { return GPUKangarooResult{false, {}, 0, 0, 0}; }
};

class MultiGPUKangarooManager {
public:
    // Mirror the dp_bits acceptance window from the CUDA-enabled
    // declaration so callers (main.cpp standalone path) compile on
    // non-CUDA builds even though init() will refuse. The stub returns
    // false from init, so these constants are decorative on Mac, but
    // their absence breaks the unconditional reference at the call site.
    static constexpr int kMinDpBits = 16;
    static constexpr int kMaxDpBits = 28;

    uint32_t dp_bits = 20;
    int num_kangaroos_per_gpu = 1 << 18;
    int steps_per_round = 256;
    bool debug_mode = false;
    std::atomic<bool> stop_flag{false};
    std::function<bool(uint64_t, uint64_t, double)> progress_callback;

    MultiGPUKangarooManager() = default;
    ~MultiGPUKangarooManager() = default;

    bool init(const std::vector<int>& /*gpu_ids*/ = {}) { return false; }
    int num_gpus() const { return 0; }
    void set_range(const UInt256& /*start*/, const UInt256& /*end*/) {}
    void set_target_h160(const std::array<uint8_t, 20>& /*h160*/) {}
    void set_target_pubkey(const cpu::uint256_t& /*x*/, const cpu::uint256_t& /*y*/) {}
    GPUKangarooResult solve() { return GPUKangarooResult{false, {}, 0, 0, 0}; }
};

#endif  // COLLIDER_USE_CUDA

}  // namespace gpu
}  // namespace collider
