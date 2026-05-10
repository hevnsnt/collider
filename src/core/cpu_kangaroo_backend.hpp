/**
 * CPU Kangaroo backend adapter.
 *
 * Wraps KangarooSolver (the portable single-threaded reference solver)
 * behind the IKangarooBackend interface. This is the deepest fallback:
 * always built, no GPU required. Performance is sub-MKey/s on Apple
 * Silicon and similar on x86, suitable for testing and the rare
 * machines without CUDA or Metal.
 */

#pragma once

#include "kangaroo_backend.hpp"
#include "kangaroo.hpp"
#include "crypto_cpu.hpp"

namespace collider {
namespace kangaroo {

class CpuKangarooBackend final : public IKangarooBackend {
public:
    bool initialize(const collider::pool::WorkAssignment& work) override;
    void solve(BackendCallbacks cb) override;

    std::string name()           const override { return "CPU Kangaroo"; }
    std::string device_summary() const override;
    const std::string& error()   const override { return error_; }

private:
    KangarooSolver solver_;
    std::string    error_;
    bool           initialized_ = false;
};

}  // namespace kangaroo
}  // namespace collider
