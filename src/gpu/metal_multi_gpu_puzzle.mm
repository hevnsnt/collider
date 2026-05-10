/**
 * MultiGPUPuzzleSolver -- Metal backend for standalone puzzle brute-force.
 *
 * Pre-1.4.1 the Mac build of `collider --puzzle N` (without --kangaroo)
 * hit the no-CUDA stub of MultiGPUPuzzleSolver (puzzle_gpu.hpp) which
 * returned false from init() and silently fell back to the CPU reference
 * implementation at ~30 KKeys/s. This file closes that gap by implementing
 * the same surface (init / set_target / search_batch / search_range /
 * num_gpus / calibrate_all / set_batch_size / get_batch_size) on top of
 * PuzzleMetalSolver.
 *
 * Architecture:
 *   - One PuzzleMetalSolver instance. M-series machines have one unified
 *     GPU; gpu_ids is honored for API parity but the solver only ever
 *     uses [0]. The kangaroo Metal adapter follows the same convention.
 *   - search_range loops over search_batch, mirroring the CUDA path's
 *     orchestration in puzzle_gpu.cu. Progress callbacks fire once per
 *     batch; on `false` from the callback or a match, the loop exits.
 *   - calibrate_all returns the configured batch size for this single
 *     GPU. The CUDA implementation actually probes a sweep of batch
 *     sizes; on Metal the dispatch overhead is low enough that the
 *     default (4M) is already in the sweet spot for M1/M2/M3, and the
 *     runtime puzzle solver only invokes calibration under
 *     COLLIDER_USE_CUDA anyway. Returning the current batch keeps any
 *     hypothetical Mac caller well-behaved without burning startup
 *     cycles on a no-op probe.
 *
 * Compiled only on APPLE && COLLIDER_USE_METAL.
 */

#import <Foundation/Foundation.h>

#include "puzzle_gpu.hpp"
#include "puzzle_metal.hpp"

#include <chrono>
#include <iostream>

namespace collider {
namespace gpu {

// ---------------------------------------------------------------------------
// MultiGPUPuzzleSolver::Impl -- Metal backend.
// ---------------------------------------------------------------------------

struct MultiGPUPuzzleSolver::Impl {
    PuzzleMetalSolver solver;
    Config config;
    std::array<uint8_t, 20> target_h160{};
    bool target_set = false;
    bool initialized = false;
};

MultiGPUPuzzleSolver::MultiGPUPuzzleSolver() : impl_(new Impl()) {}

MultiGPUPuzzleSolver::~MultiGPUPuzzleSolver() {
    delete impl_;
}

bool MultiGPUPuzzleSolver::init(const Config& config) {
    impl_->config = config;
    if (!impl_->solver.init()) {
        std::cerr << "[Metal] PuzzleMetalSolver init failed: "
                  << impl_->solver.error() << "\n";
        return false;
    }
    if (config.batch_size_per_gpu > 0) {
        impl_->solver.set_batch_size(config.batch_size_per_gpu);
        impl_->config.batch_size_per_gpu = impl_->solver.batch_size();
    }
    impl_->initialized = true;
    return true;
}

bool MultiGPUPuzzleSolver::init(const std::vector<int>& gpu_ids) {
    Config cfg;
    cfg.gpu_ids = gpu_ids;
    return init(cfg);
}

bool MultiGPUPuzzleSolver::set_target(const std::array<uint8_t, 20>& hash160) {
    if (!impl_->initialized) return false;
    impl_->target_h160 = hash160;
    impl_->target_set  = impl_->solver.set_target(hash160);
    return impl_->target_set;
}

int MultiGPUPuzzleSolver::num_gpus() const {
    // Always 1 on Apple silicon -- one unified GPU per device. The CUDA
    // path returns the size of the solvers_ vector (one per CUDA device).
    return impl_->initialized ? 1 : 0;
}

void MultiGPUPuzzleSolver::set_batch_size(uint64_t batch_size) {
    impl_->solver.set_batch_size(batch_size);
    impl_->config.batch_size_per_gpu = impl_->solver.batch_size();
}

uint64_t MultiGPUPuzzleSolver::get_batch_size() const {
    return impl_->solver.batch_size();
}

std::map<int, uint64_t> MultiGPUPuzzleSolver::calibrate_all(int /*iters*/) {
    // The runtime puzzle solver only calls calibrate_all under
    // COLLIDER_USE_CUDA today, so this Metal implementation is a stable
    // baseline rather than a probing routine. Return the current batch
    // size for the single Apple GPU. If a future caller decides to run
    // calibration on Metal, a real probe (vary batch size, time
    // search_batch on a non-matching range, pick the highest rate) can
    // slot in here without changing the API.
    std::map<int, uint64_t> out;
    if (!impl_->initialized) return out;
    out[0] = impl_->solver.batch_size();
    return out;
}

bool MultiGPUPuzzleSolver::search_batch(
    uint64_t start_lo, uint64_t start_hi,
    uint64_t batch_size,
    uint64_t& found_lo, uint64_t& found_hi)
{
    if (!impl_->initialized || !impl_->target_set) {
        found_lo = 0; found_hi = 0;
        return false;
    }
    return impl_->solver.search_batch(start_lo, start_hi, batch_size,
                                      found_lo, found_hi);
}

MultiGPUPuzzleSolver::Result MultiGPUPuzzleSolver::search_range(
    uint64_t start_lo, uint64_t start_hi,
    uint64_t end_lo, uint64_t end_hi)
{
    Result r{};
    if (!impl_->initialized || !impl_->target_set) return r;

    const uint64_t batch = impl_->solver.batch_size();
    if (batch == 0) return r;

    uint64_t cur_lo = start_lo;
    uint64_t cur_hi = start_hi;
    const auto t0 = std::chrono::steady_clock::now();

    // Range comparison treats (hi, lo) as a 128-bit big-endian-by-half
    // integer: cur < end iff (cur_hi < end_hi) || (cur_hi == end_hi &&
    // cur_lo < end_lo). Matches the CUDA puzzle_gpu.cu loop.
    auto less_than_end = [&](uint64_t lo, uint64_t hi) {
        if (hi != end_hi) return hi < end_hi;
        return lo < end_lo;
    };

    while (less_than_end(cur_lo, cur_hi) && !found_.load(std::memory_order_relaxed)) {
        // Cap the batch at the remaining range so we don't overshoot.
        uint64_t remaining_lo = (cur_hi == end_hi)
            ? (end_lo - cur_lo)
            : UINT64_MAX;
        uint64_t this_batch = (batch < remaining_lo) ? batch : remaining_lo;
        if (this_batch == 0) break;

        uint64_t flo = 0, fhi = 0;
        const bool hit = impl_->solver.search_batch(cur_lo, cur_hi, this_batch, flo, fhi);
        r.total_checked += this_batch;

        if (hit) {
            r.found  = true;
            r.key_lo = flo;
            r.key_hi = fhi;
            r.gpu_id = 0;
            found_.store(true, std::memory_order_relaxed);
            found_key_lo_.store(flo, std::memory_order_relaxed);
            found_key_hi_.store(fhi, std::memory_order_relaxed);
            found_gpu_id_.store(0,   std::memory_order_relaxed);
            return r;
        }

        // Advance the cursor by this_batch (128-bit add).
        const uint64_t new_lo = cur_lo + this_batch;
        const uint64_t carry  = (new_lo < cur_lo) ? 1ull : 0ull;
        cur_lo = new_lo;
        cur_hi = cur_hi + carry;

        if (progress_callback) {
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - t0).count();
            const double rate = (elapsed > 0.0)
                ? static_cast<double>(r.total_checked) / elapsed
                : 0.0;
            if (!progress_callback(r.total_checked, rate)) {
                break;
            }
        }
    }

    return r;
}

}  // namespace gpu
}  // namespace collider
