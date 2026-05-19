/**
 * RCKangaroo backend adapter for the standalone puzzle solver.
 *
 * this is the standalone-side counterpart to
 * CudaRCKangarooBackend (which wraps the same RCKangarooManager for
 * pool mode). The two adapters intentionally coexist:
 *
 *   - CudaRCKangarooBackend lives in cuda_rckangaroo_backend.{hpp,cpp}
 *     and is registered with the pool-mode kangaroo_backend_factory.
 *     Its initialize() takes a pool WorkAssignment. It does NOT
 *     implement save/load (the pool driver knows the RC backend is
 *     opaque and skips persistence when save returns false).
 *
 *   - RCKangarooBackend (this file) is for the standalone puzzle
 *     solver, where the input shape is a (pubkey_hex, range_start_hex,
 *     range_bits, dp_bits) tuple. It DOES implement save/load via the
 *     final patch to third_party/RCKangaroo/. The
 *     puzzle_solver.cpp runner uses this adapter so the same
 *     IKangarooBackend::save_herd_state / load_herd_state contract
 *     covers both the JLP-based multi-GPU backend (pool path) and
 *     RCKangaroo (standalone path).
 *
 * The two adapters share the same underlying RCKangarooManager
 * singleton via the existing g_active_rckangaroo guard in
 * rckangaroo_wrapper.cu; only one instance of either adapter can be
 * alive at a time. Standalone code never runs the pool driver and
 * vice versa, so this is not a constraint in practice.
 */

#pragma once

#include "../core/kangaroo_backend.hpp"
#include "rckangaroo_wrapper.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace collider {
namespace kangaroo {

class RCKangarooBackend final : public IKangarooBackend {
public:
    explicit RCKangarooBackend(std::vector<int> gpu_ids);

    // Standalone-flavored initialize. Unlike CudaRCKangarooBackend which
    // takes a pool WorkAssignment, this one takes the puzzle's natural
    // inputs directly (pubkey hex + range_start hex + bits). The
    // IKangarooBackend base's initialize(WorkAssignment) override is
    // also implemented for callers that want to use a synthesized
    // assignment, but standalone code uses initialize_standalone for
    // legibility.
    bool initialize(const collider::pool::WorkAssignment& work) override;
    bool initialize_standalone(const std::string& pubkey_hex_compressed,
                               const std::string& range_start_hex,
                               int range_bits,
                               int dp_bits);

    bool try_set_bloom_filter(const std::string& path) override;
    void solve(BackendCallbacks cb) override;

    std::string name()           const override { return "RCKangaroo"; }
    std::string device_summary() const override;
    const std::string& error()   const override { return error_; }

    // Tier C herd save/load wired through the v1.4.2 builder-final
    // patch to third_party/RCKangaroo/. Both methods route to
    // RCKangarooManager's save_herd_state / load_herd_state.
    // Save semantics: call save_herd_state(path) BEFORE solve() to arm
    // the patch's per-GPU SaveKangsHost hooks. After solve() returns
    // (e.g. on SIGINT), the saved buffer is flushed to disk
    // automatically as part of the same call's bookkeeping. The
    // public method here therefore both arms and finalizes; the
    // arming on first call sets a sticky flag, and the finalize on
    // second call (after solve) writes the file.
    bool save_herd_state(const std::string& path) override;
    bool load_herd_state(const std::string& path) override;

private:
    gpu::RCKangarooManager rc_;
    std::vector<int>       gpu_ids_;
    int                    num_gpus_ = 0;
    std::string            error_;
    bool                   initialized_ = false;

    // save_herd_state has a two-phase lifecycle: first call (pre-solve)
    // arms the hook; second call (post-solve) writes the file. This
    // flag tracks which phase we are in.
    std::string            pending_save_path_;
};

}  // namespace kangaroo
}  // namespace collider
