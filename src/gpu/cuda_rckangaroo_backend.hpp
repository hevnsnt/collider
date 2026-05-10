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

private:
    gpu::RCKangarooManager rc_;
    std::vector<int>       gpu_ids_;
    int                    num_gpus_ = 0;
    std::string            error_;
    bool                   initialized_ = false;
};

}  // namespace kangaroo
}  // namespace collider
