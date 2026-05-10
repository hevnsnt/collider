/**
 * Kangaroo solver backend interface.
 *
 * Three solver implementations exist (CUDA RCKangaroo on Windows/Linux,
 * Apple Metal Kangaroo on macOS, portable CPU KangarooSolver as a
 * fallback). Each has a fundamentally different control flow:
 *
 *   - CUDA: third-party RCKangarooManager owns the loop, invokes host
 *           callbacks for DPs and progress.
 *   - Metal: host owns the loop, polls KangarooMetalSolver::step_round()
 *            and drains DPs synchronously each round.
 *   - CPU:  KangarooSolver owns the loop, invokes host callbacks.
 *
 * This header defines the contract that hides those differences: every
 * backend exposes initialize() + solve(BackendCallbacks). The pool
 * driver in main.cpp picks one via create_kangaroo_backend() (the only
 * remaining #ifdef in the pool flow) and runs it. ~470 lines of
 * branched setup-and-loop code in run_pool_mode() collapses into the
 * factory + three thin adapters under this interface.
 *
 * Solution reporting is via callback rather than a return value so the
 * solve() exit reason (shutdown / chunk boundary / unrecoverable error)
 * stays decoupled from "did we find a key this run."
 */

#pragma once

#include "../pool/jlp_pool_client.hpp"  // collider::pool::WorkAssignment

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace collider {
namespace kangaroo {

// Callback surface invoked by IKangarooBackend::solve() while the
// solver is running. All callbacks must be non-null; the factory
// guarantees that. Backends call them from the worker thread that
// owns the solve loop, NOT from a kernel callback, so they may
// freely take locks / write to stdout / call into the pool client.
struct BackendCallbacks {
    // Distinguished point surfaced. Bytes are in BE wire format ready
    // for direct submission to PoolManager::submit_dp().
    std::function<void(const uint8_t* x_be,
                       const uint8_t* d_be,
                       uint8_t type,
                       uint32_t dp_bits)> on_dp;

    // Periodic progress tick. Backend chooses tick frequency (~1-10 Hz
    // typical). Return false to request the solve loop exits cleanly.
    std::function<bool(double ops_per_sec,
                       uint64_t local_dps)> on_progress;

    // Called when a private key is recovered (32 bytes BE). The pool
    // driver decides what to do with it (report to pool, log, etc.).
    // Backends that never produce a solution locally (e.g., Metal,
    // which only contributes DPs) simply never invoke this.
    std::function<void(const uint8_t key[32])> on_solution;

    // Polled by the backend's loop body to ask "should I keep going?".
    // Independent of on_progress's return so backends without a
    // periodic-progress hook still have a stop signal.
    std::function<bool()> should_continue;
};

class IKangarooBackend {
public:
    virtual ~IKangarooBackend() = default;

    // One-time setup for the assigned chunk. Returns false on any
    // unrecoverable failure; error() carries the reason. Multiple
    // initialize() calls on the same instance are allowed (and used
    // when the pool reassigns work mid-run, though the v1.4 driver
    // restarts the worker on chunk reassignment).
    virtual bool initialize(const collider::pool::WorkAssignment& work) = 0;

    // Optionally load a bloom filter for opportunistic address
    // scanning. Pro-only / CUDA-only feature today; non-CUDA and Free
    // backends return false silently. Call before solve().
    virtual bool try_set_bloom_filter(const std::string& path) {
        (void)path;
        return false;
    }

    // Run the solve loop until BackendCallbacks::should_continue returns
    // false, on_progress returns false, or the chunk completes. Blocks
    // for the lifetime of the worker session; the pool driver loops
    // around solve() to handle reassignment.
    virtual void solve(BackendCallbacks cb) = 0;

    // Short backend name for status-line headers ("RCKangaroo",
    // "Metal Kangaroo", "CPU Kangaroo").
    virtual std::string name() const = 0;

    // Human-readable device summary ("2 x NVIDIA GeForce RTX 3060",
    // "Apple M1", "1 thread").
    virtual std::string device_summary() const = 0;

    // Last error message, or empty if no error has occurred.
    virtual const std::string& error() const = 0;
};

// Factory: returns the appropriate backend for this build configuration.
// The compile-time choice (CUDA / Metal / CPU) is the only #ifdef left
// in the pool driver. Caller owns the returned pointer.
//
// gpu_ids is consumed by the CUDA backend (which initializes the
// nominated devices); ignored by Metal and CPU backends.
std::unique_ptr<IKangarooBackend> create_kangaroo_backend(
    const std::vector<int>& gpu_ids);

}  // namespace kangaroo
}  // namespace collider
