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
    bool try_set_bloom_filter(const std::string& path) override;
    void solve(BackendCallbacks cb) override;

    std::string name()           const override { return "RCKangaroo"; }
    std::string device_summary() const override;
    const std::string& error()   const override { return error_; }

    // Tier C (v1.4.2 builder-kangaroo): the third-party RCKangaroo
    // implementation owns its own per-GPU state buffers behind its own
    // API surface (RCKangarooManager) which does NOT expose a herd
    // dump / restore today. Adding it would require either upstream
    // changes to RCKangaroo or a sidecar host-side mirror that
    // round-trips through RCKangaroo's existing checkpoint format.
    // Until one of those lands, the runner checkpoint loop sees these
    // return false and skips persistence for the RCKangaroo backend
    // (graceful fallback: the run just doesn't resume from a saved
    // point). The in-house MultiGPUKangarooManager backend (see
    // kangaroo_solver_gpu.hpp) does implement the API and is the
    // intended path for users who need resumable solves.
    bool save_herd_state(const std::string& /*path*/) override { return false; }
    bool load_herd_state(const std::string& /*path*/) override { return false; }

private:
    gpu::RCKangarooManager rc_;
    std::vector<int>       gpu_ids_;
    int                    num_gpus_ = 0;
    std::string            error_;
    bool                   initialized_ = false;
};

}  // namespace kangaroo
}  // namespace collider
