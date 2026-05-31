/**
 * CUDA RCKangaroo backend adapter.
 *
 * Wraps the third-party RCKangarooManager (RetiredCoder's RCKangaroo)
 * behind the IKangarooBackend interface. Only built when
 * COLLIDER_USE_RCKANGAROO is defined; the factory in
 * kangaroo_backend_factory.cpp picks this on Windows/Linux + CUDA
 * builds.
 */

#pragma once

#include "../core/kangaroo_backend.hpp"
#include "rckangaroo_wrapper.hpp"

#include <vector>

namespace collider {
namespace kangaroo {

class CudaRCKangarooBackend final : public IKangarooBackend {
public:
    explicit CudaRCKangarooBackend(std::vector<int> gpu_ids);

    bool initialize(const collider::pool::WorkAssignment& work) override;
#ifdef COLLIDER_PRO
    // Opportunistic bloom checking is a Pro-only feature; Free builds fall
    // back to IKangarooBackend::try_set_bloom_filter (returns false) /
    // try_get_bloom_checks (returns 0).
    bool try_set_bloom_filter(const std::string& path) override;
    uint64_t try_get_bloom_checks() const override;
#endif
    void solve(BackendCallbacks cb) override;

    std::string name()           const override { return "RCKangaroo"; }
    std::string device_summary() const override;
    const std::string& error()   const override { return error_; }

    // Tier C herd save/load, wired through the v1.4.2 builder-final patch
    // to third_party/RCKangaroo/ (SaveKangsHost / InitKangsHost hooks
    // exposed via RCKangarooManager::request_save_on_stop / save_herd_state
    // / load_herd_state). This mirrors the standalone RCKangarooBackend
    // adapter (rc_kangaroo_backend.cpp), which already drives the same
    // RCKangarooManager API; the earlier "return false" stub predated the
    // wrapper gaining a real implementation.
    //
    // Lifecycle (RCKangaroo requires arming BEFORE solve): solve() arms
    // rc_.request_save_on_stop() and records pending_save_path_ so the
    // post-solve write has a destination, then flushes the populated host
    // buffers to disk after rc_.solve() returns. The pool runner
    // (pool_solver.cpp) calls save_herd_state(path) ONCE at teardown
    // (after solve) and load_herd_state(path) ONCE before solve; both are
    // satisfied here. load arms InitKangsHost so the next solve() resumes
    // the herd instead of regenerating it.
    bool save_herd_state(const std::string& path) override;
    bool load_herd_state(const std::string& path) override;

private:
    gpu::RCKangarooManager rc_;
    std::vector<int>       gpu_ids_;
    int                    num_gpus_ = 0;
    std::string            error_;
    bool                   initialized_ = false;

    // Destination for the post-solve herd write. Set when solve() arms the
    // save (or when save_herd_state is called pre-solve). Empty means no
    // save is pending. See solve() / save_herd_state() in the .cpp.
    std::string            pending_save_path_;
};

}  // namespace kangaroo
}  // namespace collider
