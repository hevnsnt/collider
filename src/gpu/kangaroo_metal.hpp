/**
 * Apple Metal Pollard's Kangaroo dispatcher (v1.4.0).
 *
 * Sits between the pool client (or solo solver) and the .metal kernel
 * at src/gpu/kangaroo.metal. Mirrors the CUDA kangaroo_step_kernel
 * dispatch (driven by MultiGPUKangarooManager::step in
 * kangaroo_kernel.cu) but in C++/Obj-C++ for Mac builds.
 *
 * Compiled only on APPLE && COLLIDER_USE_METAL.
 */

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Shared jump-table size header. The header opens its own
// `namespace collider { namespace gpu {` block, so it must be included
// BEFORE this file's namespace block to avoid nesting collider::gpu
// inside itself (Mac CI Pro pass on 9637518 caught the nested form
// when this include lived inside the open namespace).
#include "kangaroo_jump_table_size.hpp"

namespace collider {
namespace gpu {

// Jump-table size for the kangaroo walk. The canonical definition
// lives in kangaroo_jump_table_size.hpp -- this alias preserves the
// existing kJumpTableSize identifier for callers but binds it to the
// shared constant so a future change can't drift between Metal and
// CUDA host paths. The MSL kernel at src/gpu/kangaroo.metal still
// has its own #define KANGAROO_JUMP_TABLE_SIZE because Metal Shading
// Language can't include host headers; the boundary is checked via
// static_assert in KangarooMetalSolver.
inline constexpr uint32_t kJumpTableSize     = kKangarooJumpTableSize;
inline constexpr uint32_t kJumpSelectorMask  = kJumpTableSize - 1;

// Defaults tuned for Apple M1 / M2 / M3. Override per-call in main.cpp
// or via a future config-file knob if a future GPU benefits from
// different occupancy. See KangarooMetalConfig::auto_size_dp_buffer
// for the relationship between these and dp_max_per_round.
inline constexpr uint32_t kDefaultKangaroos      = 1024;   // 512 tame + 512 wild
inline constexpr uint32_t kDefaultStepsPerRound  = 1024;   // walk steps per kernel launch
inline constexpr uint32_t kDefaultDpBits         = 24;     // pool-protocol-typical
inline constexpr uint32_t kMinDpMaxPerRound      = 256;    // floor for tiny configs

// dp_bits acceptance window for the Metal kernel. The kernel itself
// rejects dp_bits >= 128 (kangaroo.metal:446), but anything above 60 is
// statistically unfindable on M-series GPUs in reasonable wall time,
// and anything below 14 floods the pool with non-distinguished points.
// Standalone callers (main.cpp Mac path) must validate against this
// window before constructing a KangarooMetalConfig.
inline constexpr uint32_t kMetalMinDpBits = 14;
inline constexpr uint32_t kMetalMaxDpBits = 60;

struct KangarooMetalConfig {
    uint32_t num_kangaroos     = kDefaultKangaroos;
    uint32_t steps_per_round   = kDefaultStepsPerRound;
    uint32_t dp_bits           = kDefaultDpBits;
    uint64_t work_id           = 0;
    // 0 -> auto-size from (num_kangaroos * steps_per_round) at the
    // configured dp_bits. Set explicitly only if you have a reason
    // to override; the auto-sized buffer is sized for ~4x the
    // expected DPs/round to absorb statistical bursts.
    uint32_t dp_max_per_round  = 0;
};

// One distinguished point coming out of the kernel. Layout mirrors
// JLPDistinguishedPointV2 wire format so we can copy directly into a
// DP_BATCH_V2 frame without re-marshalling.
struct KangarooMetalDP {
    uint64_t  work_id;
    uint8_t   x_be[32];
    uint8_t   d_be[32];
    uint8_t   type;       // 0 = tame, 1 = wild
    uint8_t   dp_bits;
};

// Initial state for a single kangaroo: starting point (x, y) in big-endian
// 32-byte form, walked distance d (also 32-byte BE), and type tag. The
// dispatcher will encode these into the device-side limb format before
// kicking the kernel.
struct KangarooSeed {
    std::array<uint8_t, 32> x;
    std::array<uint8_t, 32> y;
    std::array<uint8_t, 32> d;
    uint8_t                 type;
};

class KangarooMetalSolver {
public:
    KangarooMetalSolver();
    ~KangarooMetalSolver();

    // One-time setup. Loads kangaroo.metal at runtime, builds the
    // compute pipelines, allocates state buffers for `cfg.num_kangaroos`
    // kangaroos. Returns false + populates `error()` on any failure.
    bool init(const KangarooMetalConfig& cfg);

    // Upload a 32-entry jump table. Each entry is (jx, jy, jd) in BE
    // 32-byte form. cumulative_distances[i] is added to a kangaroo's
    // distance whenever the kernel selects jump i.
    bool set_jump_table(const std::array<KangarooSeed, kJumpTableSize>& jumps);

    // Seed N kangaroos with starting state. N must equal cfg.num_kangaroos.
    bool seed_kangaroos(const std::vector<KangarooSeed>& seeds);

    // Run a single round (cfg.steps_per_round walk steps). Returns the
    // DPs that surfaced this round. The kernel runs synchronously
    // (waitUntilCompleted) so the caller can drain DPs immediately.
    std::vector<KangarooMetalDP> step_round();

    // Update the assigned work_id (e.g., when a new chunk arrives).
    // The next round's DPs will carry this work_id.
    void set_work_id(uint64_t work_id);

    // Replace a single kangaroo's state with a fresh seed. Used by the
    // host-side identity guard: kangaroos that hit P == -Q have their
    // (x, y) zeroed by the kernel (kangaroo.metal point_op identity
    // arm). Without re-seeding, those threads stay at (0, 0) and emit
    // garbage DPs forever. The host scans state after each round, finds
    // dead kangaroos, picks a fresh offset, and calls this to revive
    // them. Index must be < cfg.num_kangaroos.
    bool replace_seed(uint32_t index, const KangarooSeed& seed);

    // Detect kangaroos whose (x, y) state is all-zero (the identity-
    // signal the kernel produces on P == -Q). Returns the indices in
    // out_dead so the caller can re-seed them via replace_seed().
    bool find_dead_kangaroos(std::vector<uint32_t>& out_dead);

    // Diagnostic info.
    std::string device_name() const;
    const std::string& error() const { return error_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;
    std::string error_;
};

}  // namespace gpu
}  // namespace collider
