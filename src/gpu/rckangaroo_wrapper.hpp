/**
 * RCKangaroo Wrapper for theCollider
 *
 * Integrates RetiredCoder's RCKangaroo (GPLv3) as the Kangaroo solver backend.
 * RCKangaroo achieves ~8 GKeys/s on RTX 4090, ~4 GKeys/s on RTX 3090.
 *
 * Optional bloom filter integration (Pro only): checks each DP against a
 * bloom filter of funded Bitcoin addresses (opportunistic collision
 * detection). The feature lives entirely in the Pro-only
 * rckangaroo_bloom.{hpp,cu}; this wrapper reaches it through the
 * collider::gpu::bloom hook under #ifdef COLLIDER_PRO so Free builds carry
 * zero bloom symbols.
 *
 * Original software: (c) 2024, RetiredCoder (RC)
 * https://github.com/RetiredC/RCKangaroo
 */

#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <functional>
#include <atomic>

#ifdef COLLIDER_PRO
// BloomHit + the opportunistic-bloom hook API live in this Pro-only header.
// Including it here (under COLLIDER_PRO only) lets the gated result/config
// members below name collider::gpu::bloom::BloomHit. Free builds never see
// the type, the members, or the bloom source.
#include "rckangaroo_bloom.hpp"
#endif

namespace collider {
namespace gpu {

/**
 * Result from RCKangaroo solve
 */
struct RCKangarooResult {
    bool found;
    std::array<uint64_t, 4> private_key;  // 256-bit private key
    uint64_t total_ops;
    uint64_t dp_count;
    double elapsed_seconds;
    double k_value;  // K coefficient (target is 1.15)
    uint32_t error_count;

#ifdef COLLIDER_PRO
    // Opportunistic bloom results (Pro only). BloomHit is defined in
    // rckangaroo_bloom.hpp; the field is absent from Free builds.
    uint64_t bloom_checks = 0;                  // Total bloom probes performed
    std::vector<bloom::BloomHit> bloom_hits;    // Potential matches (verify externally)
#endif
};

/**
 * RCKangaroo Solver Manager
 *
 * High-performance GPU Kangaroo solver using RetiredCoder's implementation.
 * Supports ranges up to 170 bits, achieves K=1.15 (optimal).
 */
class RCKangarooManager {
public:
    // Per-backend dp_bits and range_bits acceptance windows. RCKangaroo's
    // command-line parser rejects anything outside these ranges (see
    // third_party/RCKangaroo/RCKangaroo.cpp:319 and :534), so we mirror
    // them here and validate at standalone call sites before invoking
    // the backend. Centralizing the constants keeps the standalone CLI,
    // the pool backend (cuda_rckangaroo_backend.cpp), and any future
    // tooling consistent.
    static constexpr int kMinDpBits    = 14;
    static constexpr int kMaxDpBits    = 60;
    static constexpr int kMinRangeBits = 32;
    static constexpr int kMaxRangeBits = 170;

    // Configuration
    uint32_t dp_bits = 20;      // Distinguished point bits (kMinDpBits..kMaxDpBits)
    int range_bits = 135;       // Search range in bits (kMinRangeBits..kMaxRangeBits)
    bool benchmark_mode = false;
    std::atomic<bool> stop_flag{false};

    // v1.5: asymmetric kangaroo mode (KANG_MODE_BOTH / TAME_ONLY / WILD_ONLY
    // from third_party/RCKangaroo/defs.h). Default BOTH preserves standalone
    // behavior; pool mode must set TAME_ONLY or WILD_ONLY before solve() so
    // the worker can only ever hold half of the tame/wild trail data
    // (theft-resistance guarantee). Applied to every per-GPU RCGpuKang in
    // solve() before Prepare(); takes effect for the next solve() call only.
    int mode = 0;               // 0 = KANG_MODE_BOTH (defs.h)

#ifdef COLLIDER_PRO
    // Bloom filter configuration (Pro only). The whole opportunistic-bloom
    // feature compiles out of Free builds; these fields, the hit callback,
    // and load_bloom_filter / get_bloom_checks below are absent there.
    bool bloom_enabled = false;         // Enable bloom filter checking
    std::string bloom_file;             // Path to .blf bloom filter file
#endif

    // Progress callback: (ops, dp_count, speed_mkeys) -> continue?
    std::function<bool(uint64_t, uint64_t, int)> progress_callback;

#ifdef COLLIDER_PRO
    // Bloom hit callback: (hit) -> called when bloom filter match found.
    std::function<void(const bloom::BloomHit&)> bloom_hit_callback;
#endif

    // DP callback for pool mode: (x[32], d[32], type) -> called for each new DP
    // Used to submit DPs to pool server
    std::function<void(const uint8_t*, const uint8_t*, uint8_t)> dp_callback;

    RCKangarooManager();
    ~RCKangarooManager();

    /**
     * Initialize with specific GPU IDs, or empty vector for auto-detect
     * Returns number of GPUs initialized
     */
    int init(const std::vector<int>& gpu_ids = {});

    /**
     * Get number of initialized GPUs
     */
    int num_gpus() const;

    /**
     * Set the target public key to solve
     * @param compressed_hex Compressed public key (33 bytes hex, e.g., "02abc...")
     * @return true if valid public key
     */
    bool set_target_pubkey(const std::string& compressed_hex);

    /**
     * Set the target public key from X,Y coordinates
     */
    bool set_target_pubkey(const std::array<uint64_t, 4>& x, const std::array<uint64_t, 4>& y);

    /**
     * Set the search range offset (starting point)
     * @param start_hex Hexadecimal offset (e.g., "40000000000000000000000000000000000")
     */
    void set_start_offset(const std::string& start_hex);

    /**
     * Load precomputed tame kangaroos from file (speeds up solving)
     * @param filename Path to tames file
     * @return true if loaded successfully
     */
    bool load_tames(const std::string& filename);

    /**
     * Generate and save tame kangaroos to file
     * @param filename Path to save tames
     * @param max_ops Maximum operations multiplier (e.g., 0.5 for half of expected ops)
     * @return true if generated successfully
     */
    bool generate_tames(const std::string& filename, double max_ops = 0.5);

    /**
     * Solve for the private key
     * @return Result containing private key if found
     */
    RCKangarooResult solve();

    /**
     * Run benchmark mode (solve random keys to measure K value)
     * @param num_points Number of random points to solve
     * @return Average K value achieved
     */
    double benchmark(int num_points = 10);

    /**
     * Get current speed in MKeys/s
     */
    int get_speed() const;

#ifdef COLLIDER_PRO
    /**
     * Load bloom filter from .blf file for opportunistic address checking
     * (Pro only). Absent from Free builds.
     * @param filename Path to .blf bloom filter file
     * @return true if loaded successfully
     */
    bool load_bloom_filter(const std::string& filename);

    /**
     * Get bloom filter statistics (Pro only).
     * @return Number of bloom filter checks performed so far
     */
    uint64_t get_bloom_checks() const;
#endif

    /**
     * kangaroo herd save / load.
     *
     * save_herd_state arms the RCGpuKang per-GPU SaveKangsHost hooks
     * before solve() runs. When solve() returns, the host buffers are
     * populated; this call serializes them to `path` using the format
     * documented in third_party/RCKangaroo/.patches/save-load-state.patch.
     * Returns false if no solve has occurred yet OR if the save buffers
     * are empty (e.g. init() never called).
     *
     * load_herd_state reads the file and arms InitKangsHost on each
     * RCGpuKang BEFORE solve(). The next solve()'s Start() consumes the
     * buffer in place of RCKangaroo's random-point generation. Returns
     * false on any file I/O error, magic mismatch, version mismatch,
     * GPU-count mismatch, KangCnt mismatch, or config-hash mismatch
     * (different target pubkey or range_start).
     *
     * Lifecycle: load BEFORE solve(); call request_save_on_stop()
     * BEFORE solve(); write the file AFTER solve() returns by calling
     * save_herd_state(path).
     *
     * These methods are no-ops (return false) if init() has not run.
     */
    bool request_save_on_stop();
    bool save_herd_state(const std::string& path);
    bool load_herd_state(const std::string& path);

private:
    struct Impl;
    Impl* impl_;
};

/**
 * Convert private key array to hex string
 */
std::string private_key_to_hex(const std::array<uint64_t, 4>& key);

/**
 * Convert hex string to private key array
 */
bool hex_to_private_key(const std::string& hex, std::array<uint64_t, 4>& key);

}  // namespace gpu
}  // namespace collider
