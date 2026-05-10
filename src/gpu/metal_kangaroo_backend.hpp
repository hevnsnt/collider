/**
 * Apple Metal Kangaroo backend adapter.
 *
 * Wraps KangarooMetalSolver behind the IKangarooBackend interface.
 * Only built when APPLE && COLLIDER_USE_METAL; the factory in
 * kangaroo_backend_factory.cpp picks this on Mac builds.
 *
 * Unlike the CUDA backend (which has a third-party blocking solve()),
 * Metal needs the host to drive the round loop -- this adapter holds
 * that loop and pumps DPs into BackendCallbacks::on_dp every round.
 */

#pragma once

#include "../core/kangaroo_backend.hpp"
#include "../core/crypto_cpu.hpp"
#include "kangaroo_metal.hpp"

namespace collider {
namespace kangaroo {

class MetalKangarooBackend final : public IKangarooBackend {
public:
    bool initialize(const collider::pool::WorkAssignment& work) override;
    void solve(BackendCallbacks cb) override;

    std::string name()           const override { return "Metal Kangaroo"; }
    std::string device_summary() const override;
    const std::string& error()   const override { return error_; }

private:
    gpu::KangarooMetalSolver solver_;
    gpu::KangarooMetalConfig cfg_;
    uint64_t                  work_id_ = 0;
    std::string               error_;
    bool                      initialized_ = false;

    // Cached for the host-side identity-recovery loop in solve(): when
    // a kangaroo hits P == -Q the kernel zeros its (x, y); the host
    // detects this and re-seeds with a fresh offset using these inputs.
    cpu::uint256_t            range_start_{};
    cpu::uint256_t            target_x_{};
    cpu::uint256_t            target_y_{};
};

}  // namespace kangaroo
}  // namespace collider
