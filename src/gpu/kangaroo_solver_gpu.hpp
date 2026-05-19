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

    // Tier C (v1.4.2 builder-kangaroo): herd state serialization.
    // save_herd_state writes the current per-kangaroo (px, py, dist, type,
    // wild_offset) tuples for every GPU into a single host-side file.
    // load_herd_state restores them, validating that the file's header
    // matches the current solver configuration (num_kangaroos_per_gpu,
    // num_gpus, dp_bits, range hash). Both functions are usable only
    // after init() has succeeded but before/between solve() invocations.
    // Returns false on any I/O failure or configuration mismatch.
    // Intended use: the runner-side checkpoint loop persists state every
    // N seconds so a crashed / SIGINT'd solver can resume without losing
    // its accumulated DP traversal. The runner wires this; the backend
    // only exposes the read/write primitives.
    // File format:
    //   - 16-byte magic "COLLIDER_KANG\x01\x00\x00" (NUL-padded to 16 bytes)
    //   - 4-byte uint32 version (= 1)
    //   - 4-byte uint32 num_gpus
    //   - 4-byte uint32 num_kangaroos_per_gpu
    //   - 4-byte uint32 dp_bits
    //   - 32-byte SHA256 of [range_start_be || range_end_be]
    //   - Per-GPU per-kangaroo records (little-endian):
    //       32 bytes px (4 x uint64 LE)
    //       32 bytes py
    //       32 bytes dist
    //       32 bytes wild_offset
    //       4 bytes type (uint32)
    //       4 bytes padding
    //     -> 136 bytes per kangaroo.
    bool save_herd_state(const std::string& path);
    bool load_herd_state(const std::string& path);

private:
    struct Impl;
    Impl* impl_;
};

#elif defined(COLLIDER_USE_METAL)

/**
 * GPU Kangaroo Solver (Metal, macOS).
 *
 * Standalone puzzle solving on Mac now uses the Metal Jacobian
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
    // 0 = auto-tune at init() time from the Metal device name. The
    // threadgroup batch-inversion contract requires this to be a
    // multiple of 32; the auto-tuner picks 1024 / 2048 / 4096 / 8192
    // depending on whether the chip is base / Pro / Max / Ultra
    // (counts that empirically saturate the corresponding GPU core
    // ranges of 8-10 / 14-20 / 24-40 / 48-76). Callers who want a
    // specific value can still set this field directly before init().
    // The 1024 baseline was tuned on M1 (8 GPU cores) and starved the
    // wider GPUs on M2 Pro / M3 Pro / M4 Pro and up; that miss-tuning
    // surfaced as DoctorNigel's 24 MKeys/s M4 mac mini measurement
    // in issue #4 (https://github.com/hevnsnt/collider/issues/4).
    int num_kangaroos_per_gpu = 0;
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

    // Tier C (v1.4.2 builder-kangaroo): herd state serialization stubs.
    // Metal backend will track CUDA's API surface once Tier C is wired
    // to the Mac runner; until then these return false so the runner
    // checkpoint loop falls back to "no checkpoint persistence on Mac".
    bool save_herd_state(const std::string& /*path*/) { return false; }
    bool load_herd_state(const std::string& /*path*/) { return false; }

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

    // no-op stubs for the no-GPU build.
    bool save_herd_state(const std::string& /*path*/) { return false; }
    bool load_herd_state(const std::string& /*path*/) { return false; }
};

#endif  // COLLIDER_USE_CUDA / COLLIDER_USE_METAL / stub

}  // namespace gpu
}  // namespace collider
