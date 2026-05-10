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

#elif defined(COLLIDER_USE_METAL)

/**
 * GPU Kangaroo Solver (Metal, macOS).
 *
 * v1.4.1: Standalone puzzle solving on Mac now uses the Metal Jacobian
 * kangaroo (D.1 rewrite) via MultiGPUKangarooManager. Pre-1.4.1 the
 * Metal-only build hit the no-CUDA stub which returned false from
 * init() and silently fell back to CPU kangaroo -- meaning Mac
 * users got no GPU acceleration on standalone puzzle work, even
 * though the Metal kernel was already shipping for pool mode.
 *
 * GPUKangarooManager is intentionally still a stub here -- the
 * CUDA single-GPU class was a v1.0-era convenience that the
 * MultiGPU class supersedes. The Mac path only implements the
 * MultiGPU surface that puzzle_solver.cpp actually calls.
 *
 * Implementation lives in src/gpu/metal_multi_gpu_kangaroo.mm and
 * is built when APPLE && COLLIDER_USE_METAL.
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
    // dp_bits acceptance window, mirroring the CUDA path. The Metal
    // kernel itself accepts a wider range (kMetalMinDpBits=14 ..
    // kMetalMaxDpBits=60 in kangaroo_metal.hpp) but for standalone
    // host-side collision detection the practical window is the same
    // as CUDA's: below 16 floods the host hashmap with non-DP traffic,
    // above 28 stalls collisions on M-series sized batches.
    static constexpr int kMinDpBits = 16;
    static constexpr int kMaxDpBits = 28;

    uint32_t dp_bits = 20;
    // 1024 matches kangaroo_metal.hpp::kDefaultKangaroos (tuned for
    // M1/M2/M3 occupancy; the threadgroup batch-inversion trick
    // requires a multiple of 32, and 1024 = 32 * 32 is the sweet
    // spot for L1 / threadgroup memory pressure).
    int num_kangaroos_per_gpu = 1024;
    int steps_per_round = 1024;
    bool debug_mode = false;
    std::atomic<bool> stop_flag{false};
    std::function<bool(uint64_t, uint64_t, double)> progress_callback;

    MultiGPUKangarooManager();
    ~MultiGPUKangarooManager();

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
 * Stub GPU Kangaroo Solver (no CUDA, no Metal -- e.g. pure CPU build)
 * Always fails init() to trigger CPU fallback.
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

#endif  // COLLIDER_USE_CUDA / COLLIDER_USE_METAL / stub

}  // namespace gpu
}  // namespace collider
