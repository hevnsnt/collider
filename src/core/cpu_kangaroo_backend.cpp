/**
 * CPU Kangaroo backend adapter -- implementation.
 *
 * See cpu_kangaroo_backend.hpp for the public surface. This file
 * adapts KangarooSolver's blocking solve() + lambda callbacks into
 * the IKangarooBackend interface.
 */

#include "cpu_kangaroo_backend.hpp"

#include "byte_codec.hpp"

#include <cstdio>
#include <cstring>
#include <thread>

namespace collider {
namespace kangaroo {

namespace {

UInt256 bytes_to_uint256(const uint8_t* b) {
    UInt256 v;
    ::collider::be32_to_limbs_le(b, v.parts);
    return v;
}

}  // namespace

bool CpuKangarooBackend::initialize(const collider::pool::WorkAssignment& work) {
    error_.clear();

    // Range from work assignment.
    solver_.set_range(bytes_to_uint256(work.range_start),
                      bytes_to_uint256(work.range_end));
    solver_.dp_bits = static_cast<int>(work.dp_bits);
    // Pool mode: this worker contributes to a *global* solve, not its
    // own chunk-local discrete-log. KangarooSolver::set_range() defaults
    // max_steps to expected_steps*4 (~16M for a 2^40 chunk), which trips
    // after ~80s of CPU grinding -- before any DP is statistically
    // likely at dp_bits=35. Run indefinitely; the receiver/server
    // controls termination.
    solver_.max_steps = UINT64_MAX;

    // Decompress the 33-byte pool pubkey into (X, Y).
    char hex[67];
    ::collider::hex_encode_lower(work.public_key, 33, hex);
    cpu::uint256_t target_x, target_y;
    if (!cpu::decompress_pubkey(target_x, target_y, std::string(hex))) {
        error_ = std::string("failed to decompress pool target pubkey: ") + hex;
        return false;
    }
    solver_.set_target_pubkey(target_x, target_y);
    initialized_ = true;
    return true;
}

void CpuKangarooBackend::solve(BackendCallbacks cb) {
    if (!initialized_) {
        error_ = "CpuKangarooBackend::solve called before initialize";
        return;
    }
    const uint32_t dp_bits_u32 = solver_.dp_bits;  // already uint32_t in v1.4.0+

    // Wire DP submissions through the shared codec into the wire-format
    // bytes the pool expects. Encapsulated setters in v1.4.0 phase 1.9
    // (set_dp_callback / set_progress_callback) replaced the public
    // std::function fields so the solver controls invocation lifecycle.
    solver_.set_dp_callback([&cb, dp_bits_u32](const cpu::uint256_t& x,
                                                const cpu::uint256_t& dist,
                                                bool is_tame) {
        uint8_t x_be[32], d_be[32];
        ::collider::limbs_le_to_be32(x.d,    x_be);
        ::collider::limbs_le_to_be32(dist.d, d_be);
        // v1.5.5 (task #9): the CPU backend never captures a checkpoint chain,
        // so the trailing committable-chain pointers are always nullptr and the
        // pool client emits DP_BATCH_V2 for CPU-produced DPs.
        cb.on_dp(x_be, d_be, is_tame ? 0u : 1u, dp_bits_u32, nullptr, nullptr);
    });

    // KangarooSolver invokes the progress callback as
    // (tame_steps, wild_steps, dp_count, rate). Adapt to our (rate,
    // local_dps) tick.
    solver_.set_progress_callback([&cb](uint64_t /*tame*/, uint64_t /*wild*/,
                                         uint64_t dp_count, double rate) -> bool {
        return cb.on_progress(rate, dp_count);
    });

    // wire the fast shutdown probe so the per-thread
    // worker loops poll BackendCallbacks::should_continue every ~10k
    // steps. Without this, a Ctrl+C / pool disconnect waits for the
    // next 1Hz progress callback to flip stop_flag, leaving each
    // worker thread mid-loop for up to a second.
    solver_.set_should_continue_callback([&cb]() -> bool {
        return cb.should_continue ? cb.should_continue() : true;
    });

    KangarooResult result = solver_.solve();
    if (result.found) {
        // Convert little-endian limbs to BE 32-byte key for the pool.
        uint8_t key[32];
        ::collider::limbs_le_to_be32(result.private_key.d, key);
        if (cb.on_solution) cb.on_solution(key);
    }
}

std::string CpuKangarooBackend::device_summary() const {
    const unsigned int n = std::thread::hardware_concurrency();
    return "CPU (" + std::to_string(n ? n : 1u) + " thread"
         + (n == 1 ? "" : "s") + ")";
}

}  // namespace kangaroo
}  // namespace collider
