// puzzle_solver_bsgs.cpp -- Phase F2 standalone-challenge BSGS path.
//
// Sits next to run_kangaroo_solve in the puzzle solver dispatch.
// When args.solver == "bsgs" the kangaroo dispatcher delegates here;
// we wrap the GPU bsgs::bsgs_solve driver, format the result, and
// translate it into a PuzzleStepResult the existing puzzle_solver.cpp
// loop already handles (SolvedExitOrContinue / StoppedExitOrContinue
// / FallThrough / FatalError).
//
// Current implementation:
//   * Bits cap: bsgs::kMaxBits (48). Larger puzzles return FallThrough
//     so the loop tries Kangaroo.
//   * Multi-GPU via range partition. When args.gpu_ids has more than
//     one entry the runner splits the puzzle range into K equal
//     sub-ranges and runs K parallel bsgs_solve calls (one per GPU,
//     one thread per device). First-to-find cancels the others via
//     the progress callback returning false. Total work is roughly
//     sqrt(K)x the single-GPU compute (each sub-range needs its own
//     baby table); wall-clock speedup is K / sqrt(K) = sqrt(K).
//     2 GPUs -> ~1.41x faster wall-clock, 3 GPUs -> ~1.73x, etc.
//   * Host-side baby-table sort and lookup. Fine for bits<=40; the
//     bound is the host's sort throughput, not the GPU.
#include "runtime/puzzle_solver_helpers.hpp"
#include "runtime/runtime_control.hpp"

#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>

#include "core/crypto_cpu.hpp"
#include "core/secure_write.hpp"
#include "core/session_log.hpp"
#include "gpu/bsgs_solver_gpu.hpp"
#include "runtime/runtime_globals.hpp"
#include "ui/banner.hpp"
#include "ui/box_render.hpp"
#include "ui/tui/tui_app.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace collider::runtime::detail {

namespace {

// 32 BE bytes -> "0x" + lowercase hex.
std::string hex32(const uint8_t* be32) {
    static const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(66);
    for (int i = 0; i < 32; ++i) {
        out.push_back(hex[(be32[i] >> 4) & 0xF]);
        out.push_back(hex[(be32[i] >> 0) & 0xF]);
    }
    return out;
}

// Cross-thread progress hook: BSGS calls back into here every ~0.5 s
// with the count of giant steps it has completed. We push that to the
// TUI's chunk-progress field so the unified Phase Progress row shows
// the bar advancing. Returning false cancels BSGS cleanly.
struct BsgsProgressBridge {
    ::collider::ui::tui::TuiApp* tui_app;
    uint64_t total_giants;
};
bool progress_thunk(uint64_t giant_steps_done, void* user) {
    auto* bridge = static_cast<BsgsProgressBridge*>(user);
    if (!bridge) return true;
    if (g_shutdown.load()) return false;
    if (bridge->tui_app) {
        // Cap to INT_MAX as set_chunk_progress takes int.
        const uint64_t t = bridge->total_giants > 0
                               ? bridge->total_giants
                               : (giant_steps_done + 1);
        const int cur = static_cast<int>(std::min<uint64_t>(
            giant_steps_done, static_cast<uint64_t>(INT_MAX)));
        const int tot = static_cast<int>(std::min<uint64_t>(
            t, static_cast<uint64_t>(INT_MAX)));
        bridge->tui_app->set_chunk_progress(cur, tot);
        if (bridge->tui_app->requested_quit() && !g_shutdown.load()) {
            g_shutdown.store(true);
            return false;
        }
    }
    return true;
}

// Read the low 64 bits of a BE 32-byte buffer.
inline uint64_t be_low64(const uint8_t be[32]) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v = (v << 8) | be[24 + i];
    }
    return v;
}

// Write a 64-bit value into the low 8 bytes of a BE 32-byte buffer
// (the upper 24 bytes are caller-provided context).
inline void be_low64_set(uint8_t be[32], uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        be[24 + i] = static_cast<uint8_t>((v >> ((7 - i) * 8)) & 0xff);
    }
}

// Per-worker progress bridge for the multi-GPU dispatch. Captures
// (a) the shared cancel flag so any worker's first-find stops the
// others, (b) the TUI app pointer so the dashboard's chunk-progress
// reflects the maximum across all workers, (c) the per-worker
// device id for diagnostic logging.
struct MultiGpuBsgsBridge {
    std::atomic<bool>*     cancel;
    ::collider::ui::tui::TuiApp* tui_app;
    int                    device_id;
    uint64_t               sub_range_giants;
};
bool mg_progress_thunk(uint64_t giant_steps_done, void* user) {
    auto* bridge = static_cast<MultiGpuBsgsBridge*>(user);
    if (!bridge) return true;
    if (g_shutdown.load()) return false;
    if (bridge->cancel->load(std::memory_order_acquire)) return false;
    if (bridge->tui_app) {
        const uint64_t t = bridge->sub_range_giants > 0
                               ? bridge->sub_range_giants
                               : (giant_steps_done + 1);
        const int cur = static_cast<int>(std::min<uint64_t>(
            giant_steps_done, static_cast<uint64_t>(INT_MAX)));
        const int tot = static_cast<int>(std::min<uint64_t>(
            t, static_cast<uint64_t>(INT_MAX)));
        // chunk_progress is single-valued for the panel; per-worker
        // values would race. Whichever device updates last wins; the
        // dashboard's THROUGHPUT row is the better signal for
        // multi-worker progress.
        bridge->tui_app->set_chunk_progress(cur, tot);
        if (bridge->tui_app->requested_quit() && !g_shutdown.load()) {
            g_shutdown.store(true);
            return false;
        }
    }
    return true;
}

// Run BSGS in parallel across `gpu_ids`. Each worker gets a disjoint
// sub-range of [base.range_start_be, base.range_end_be). First worker
// to Found wins; others are cancelled via the shared cancel flag the
// progress callback checks. Returns the aggregate BsgsResult (Found
// with the recovered key if any worker found, NotInRange if all
// exhausted their sub-range cleanly, GpuError if any device errored).
::collider::gpu::bsgs::BsgsResult run_multi_gpu_bsgs(
    const ::collider::gpu::bsgs::BsgsConfig& base,
    const std::vector<int>& gpu_ids,
    ::collider::ui::tui::TuiApp* tui_app)
{
    using ::collider::gpu::bsgs::BsgsConfig;
    using ::collider::gpu::bsgs::BsgsResult;
    using ::collider::gpu::bsgs::BsgsResultKind;

    const uint64_t b_low = be_low64(base.range_start_be);
    const uint64_t e_low = be_low64(base.range_end_be);
    if (e_low <= b_low) {
        BsgsResult r{};
        r.kind = BsgsResultKind::NotInRange;
        r.error_message = "range_end <= range_start";
        return r;
    }
    const uint64_t N = e_low - b_low;
    const size_t   k = std::max<size_t>(1, gpu_ids.size());
    const uint64_t chunk = N / k;

    std::atomic<bool> cancel{false};
    std::atomic<bool> any_found{false};
    BsgsResult winning_result{};
    std::mutex result_mu;

    std::vector<std::thread> workers;
    std::vector<BsgsResult>  per_device_results(gpu_ids.size());
    std::vector<MultiGpuBsgsBridge> bridges(gpu_ids.size());
    workers.reserve(gpu_ids.size());

    for (size_t i = 0; i < gpu_ids.size(); ++i) {
        const uint64_t sub_start = b_low + i * chunk;
        const uint64_t sub_end   =
            (i == gpu_ids.size() - 1) ? e_low : (b_low + (i + 1) * chunk);
        const uint64_t sub_N     = sub_end - sub_start;

        // Build the per-worker config: same target pubkey, sub-range
        // start/end (preserving the upper 24 bytes of context),
        // dedicated device id.
        BsgsConfig cfg = base;
        cfg.device_id = gpu_ids[i];
        std::memcpy(cfg.range_start_be, base.range_start_be, 32);
        std::memcpy(cfg.range_end_be,   base.range_start_be, 32);
        be_low64_set(cfg.range_start_be, sub_start);
        be_low64_set(cfg.range_end_be,   sub_end);

        bridges[i] = MultiGpuBsgsBridge{
            &cancel, tui_app, gpu_ids[i],
            /*sub_range_giants=*/0  // filled below
        };
        // Estimate giants for the dashboard's progress bar (m = sqrt(N_i)
        // and giants = N_i / m). The bsgs driver computes its own value;
        // ours is just a TUI display approximation.
        if (sub_N > 0) {
            uint64_t m = 1;
            while ((m * m) < sub_N) m <<= 1;
            bridges[i].sub_range_giants =
                (m > 0) ? (sub_N + m - 1) / m : 0;
        }
        cfg.progress = &mg_progress_thunk;
        cfg.progress_user = &bridges[i];

        workers.emplace_back([&, i, cfg]() mutable {
            BsgsResult r = ::collider::gpu::bsgs::bsgs_solve(cfg);
            per_device_results[i] = r;
            if (r.kind == BsgsResultKind::Found) {
                std::lock_guard<std::mutex> lk(result_mu);
                if (!any_found.exchange(true)) {
                    winning_result = r;
                    cancel.store(true, std::memory_order_release);
                }
            }
        });
    }

    for (auto& t : workers) t.join();

    if (any_found.load()) return winning_result;

    // No worker found the key. Aggregate stats; surface any GpuError
    // ahead of NotInRange so the operator sees the real fault.
    for (const auto& r : per_device_results) {
        if (r.kind == BsgsResultKind::GpuError) return r;
    }
    BsgsResult agg{};
    agg.kind = BsgsResultKind::NotInRange;
    for (const auto& r : per_device_results) {
        agg.baby_table_size       += r.baby_table_size;
        agg.giant_steps_completed += r.giant_steps_completed;
    }
    return agg;
}

}  // namespace

PuzzleStepResult run_bsgs_solve(PuzzleIterContext& ctx) {
    Arguments& args = ctx.args;
    const PuzzleInfo* puzzle = ctx.puzzle;
    PuzzleTarget& tgt = ctx.tgt;
    const int bits = tgt.bits;
    const int current_puzzle = ctx.current_puzzle;

    std::cout << "[*] --solver bsgs: GPU BSGS (Phase F2 first cut, "
                 "single-GPU, bits<="
              << ::collider::gpu::bsgs::kMaxBits << ")\n";

    if (bits > ::collider::gpu::bsgs::kMaxBits) {
        std::cout << "[!] Puzzle bits=" << bits
                  << " exceeds BSGS cap of "
                  << ::collider::gpu::bsgs::kMaxBits
                  << " (baby table would not fit). Falling back to "
                     "Kangaroo.\n";
        return PuzzleStepResult::FallThrough;
    }

    // Resolve pubkey: --pubkey overrides puzzle->public_key_hex
    // (matches run_kangaroo_solve's selection).
    const std::string& pubkey_hex =
        !args.puzzle_pubkey.empty() ? args.puzzle_pubkey
                                    : puzzle->public_key_hex;
    if (pubkey_hex.empty()) {
        std::cerr << "[!] BSGS requires a known target pubkey but "
                     "none is set for puzzle "
                  << current_puzzle << ".\n";
        return PuzzleStepResult::FatalError;
    }

    cpu::uint256_t Hx, Hy;
    if (!cpu::decompress_pubkey(Hx, Hy, pubkey_hex)) {
        std::cerr << "[!] Failed to decompress pubkey for BSGS: "
                  << pubkey_hex << "\n";
        return PuzzleStepResult::FatalError;
    }

    ::collider::gpu::bsgs::BsgsConfig bcfg{};
    bcfg.bits = bits;
    // 32 BE bytes from cpu::uint256_t.
    for (int i = 0; i < 4; ++i) {
        const uint64_t vx = Hx.d[3 - i];
        const uint64_t vy = Hy.d[3 - i];
        for (int b = 0; b < 8; ++b) {
            bcfg.target_pubkey_x_be[i * 8 + b] =
                static_cast<uint8_t>((vx >> ((7 - b) * 8)) & 0xFF);
            bcfg.target_pubkey_y_be[i * 8 + b] =
                static_cast<uint8_t>((vy >> ((7 - b) * 8)) & 0xFF);
        }
        const uint64_t vs = tgt.range_start.parts[3 - i];
        const uint64_t ve = tgt.range_end.parts[3 - i];
        for (int b = 0; b < 8; ++b) {
            bcfg.range_start_be[i * 8 + b] =
                static_cast<uint8_t>((vs >> ((7 - b) * 8)) & 0xFF);
            bcfg.range_end_be[i * 8 + b] =
                static_cast<uint8_t>((ve >> ((7 - b) * 8)) & 0xFF);
        }
    }
    bcfg.device_id = args.gpu_ids.empty() ? 0 : args.gpu_ids[0];

    BsgsProgressBridge bridge{
        static_cast<::collider::ui::tui::TuiApp*>(ctx.tui_app),
        0  // total filled by the driver before the first callback
    };
    bcfg.progress = &progress_thunk;
    bcfg.progress_user = &bridge;

    // Multi-GPU surface honesty + ACTUAL multi-GPU dispatch. When the
    // operator passed more than one --gpus id, partition the puzzle
    // range across them; each worker runs an independent BSGS over
    // its sub-range. First-to-find cancels the others via the
    // progress callback. The single-GPU code path below still runs
    // when args.gpu_ids.size() <= 1 (preserves the existing fast
    // path for default invocations).
    const bool multi_gpu = args.gpu_ids.size() > 1;
    if (multi_gpu) {
        auto& rc = ::collider::runtime::global_runtime_control();
        {
            std::lock_guard<std::mutex> lk(rc.banner_mu);
            std::ostringstream b;
            b << "BSGS multi-GPU: partitioning range across "
              << args.gpu_ids.size() << " GPUs (~sqrt("
              << args.gpu_ids.size() << ")x wall-clock speedup)";
            rc.banner_text = b.str();
            rc.banner_set_at = std::chrono::steady_clock::now();
        }
        if (auto* tui_app =
                static_cast<::collider::ui::tui::TuiApp*>(ctx.tui_app)) {
            std::ostringstream phase;
            phase << "Puzzle #" << current_puzzle
                  << " BSGS multi-GPU (" << args.gpu_ids.size()
                  << " device(s))";
            tui_app->set_current_phase_name(phase.str());
        }
    }

    // Session-log entry: bsgs path is its own algorithm so it gets
    // its own milestone tag for log-line greps.
    {
        std::ostringstream d;
        d << "puzzle=" << current_puzzle
          << " backend=BSGS_GPU"
          << " puzzle_bits=" << bits
          << " device=" << bcfg.device_id;
        ::collider::log::milestone("puzzle_start", d.str());

        ::collider::log::SessionState s;
        s.mode = "puzzle";
        s.puzzle_number = current_puzzle;
        s.puzzle_algorithm = "BSGS_GPU";
        ::collider::log::update_session_state(s);
    }

    auto t0 = std::chrono::steady_clock::now();
    ::collider::gpu::bsgs::BsgsResult res =
        multi_gpu
            ? run_multi_gpu_bsgs(
                  bcfg, args.gpu_ids,
                  static_cast<::collider::ui::tui::TuiApp*>(ctx.tui_app))
            : ::collider::gpu::bsgs::bsgs_solve(bcfg);
    auto t1 = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(t1 - t0).count();

    switch (res.kind) {
        case ::collider::gpu::bsgs::BsgsResultKind::Found: {
            std::string khex = hex32(res.recovered_key_be);
            // Dashboard phase flip BEFORE the cout banner so the
            // operator sees the solve on the live TUI (the cout
            // banner that follows is captured to boot log; the
            // puzzle_<N>_solved_bsgs.txt file written below is the
            // durable on-disk record).
            if (auto* tui_solved =
                    static_cast<::collider::ui::tui::TuiApp*>(ctx.tui_app)) {
                std::ostringstream solved_phase;
                solved_phase << "Puzzle #" << current_puzzle
                             << " SOLVED (BSGS)";
                tui_solved->set_current_phase_name(solved_phase.str());
            }
            std::cout << "\n";
            ::collider::ui::box::top(std::cout);
            ::collider::ui::box::centered(std::cout,
                "PUZZLE SOLVED (BSGS)",
                ::collider::ui::box::ansi::BRIGHT_GREEN);
            ::collider::ui::box::bottom(std::cout);
            std::cout << "  Private key (hex): 0x" << khex << "\n";
            std::cout << "  Baby table size:   " << res.baby_table_size
                      << "\n";
            std::cout << "  Giant steps:       "
                      << res.giant_steps_completed << "\n";
            std::cout << "  Elapsed:           " << std::fixed
                      << std::setprecision(2) << elapsed_s << " s\n\n";

            std::ostringstream md;
            md << "puzzle=" << current_puzzle
               << " backend=BSGS_GPU"
               << " elapsed_s=" << elapsed_s
               << " baby_table=" << res.baby_table_size
               << " giants=" << res.giant_steps_completed;
            ::collider::log::milestone("puzzle_solved", md.str());

            // Owner-only solution file. Mirrors run_kangaroo_solve's
            // file convention so the harvest tooling picks it up
            // without a special case.
            std::ostringstream fname;
            fname << "puzzle_" << current_puzzle << "_solved_bsgs.txt";
            std::ofstream f = collider::secure_open_ofstream(
                fname.str(),
                std::ios::out | std::ios::trunc,
                collider::SecureWriteOnFailure::FallbackLoud);
            if (f.is_open()) {
                f << "puzzle=" << current_puzzle << "\n";
                f << "algorithm=BSGS_GPU\n";
                f << "private_key=0x" << khex << "\n";
                f << "pubkey=" << pubkey_hex << "\n";
                f.close();
                std::cout << "  Written to: " << fname.str() << "\n";
            }
            return PuzzleStepResult::SolvedExitOrContinue;
        }
        case ::collider::gpu::bsgs::BsgsResultKind::NotInRange:
            std::cout << "[*] BSGS exhausted range without finding "
                         "the key (elapsed=" << elapsed_s
                      << "s, baby_table=" << res.baby_table_size
                      << ", giants=" << res.giant_steps_completed
                      << ").\n";
            return PuzzleStepResult::StoppedExitOrContinue;
        case ::collider::gpu::bsgs::BsgsResultKind::OutOfRange:
            std::cout << "[!] BSGS out of range: "
                      << res.error_message
                      << ". Falling back to Kangaroo.\n";
            return PuzzleStepResult::FallThrough;
        case ::collider::gpu::bsgs::BsgsResultKind::GpuError:
            std::cerr << "[!] BSGS GPU error: " << res.error_message
                      << "\n";
            return PuzzleStepResult::FatalError;
        case ::collider::gpu::bsgs::BsgsResultKind::Cancelled:
            std::cout << "[*] BSGS cancelled by operator.\n";
            return PuzzleStepResult::StoppedExitOrContinue;
    }
    return PuzzleStepResult::FatalError;
}

}  // namespace collider::runtime::detail
