/**
 * puzzle_solver_bruteforce.cpp - Brute-force solve-path implementation.
 *
 * Extracted from src/runtime/puzzle_solver.cpp during the v1.4.2
 * structural decomposition. Hosts run_bruteforce_solve(), which mirrors
 * the original inline "MULTI-GPU PUZZLE SEARCH" + "CPU FALLBACK" blocks
 * verbatim. Includes:
 *   - >128-bit brute reject (R-B7).
 *   - GPU brute-force scan (MultiGPUPuzzleSolver, uniform random or
 *     sequential).
 *   - CPU brute-force fallback (per-batch hash160 compare).
 *   - Final stats banner (was reached via fall-through to the
 *     `search_done:` label in the inline body).
 *
 * v1.4.2 T3.11: the previous monolithic body (~700 lines) was split into
 * the public dispatcher plus two TU-local backend helpers. The GPU
 * helper returns FallThrough on init failure so the dispatcher can route
 * to the CPU path; only the dispatcher knows the chain order.
 *
 * Hard-preserve invariants (see puzzle_solver_helpers.hpp):
 *   - secure_wipe on every recovered private key (GPU brute + CPU brute).
 *   - secure_open_ofstream for puzzle_found.txt (2 sites here) with
 *     SecureWriteOnFailure::FailHard (T2.4: recovered-key sinks).
 *   - SearchState v4 format with full UInt256 position_full[0..3].
 *   - Brute-force rejection > 128 bits (R-B7).
 *   - SIGINT save path via emit_shutdown_message_from_main + state save.
 */
#include "runtime/puzzle_solver_helpers.hpp"
#include "ui/tui/tui_app.hpp"   // TuiApp::set_current_phase_name on solve

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/crypto_cpu.hpp"
#include "core/logger.hpp"
#include "core/puzzle_config.hpp"
#include "core/search_state.hpp"
#include "core/secure_buffer.hpp"
#include "core/secure_write.hpp"
#include "core/session_log.hpp"  // milestone() for puzzle_start /
                                 // puzzle_solved / puzzle_stopped /
                                 // state_saved (puzzle mode)
#include "core/types.hpp"
#include "gpu/puzzle_gpu.hpp"
#include "runtime/balance.hpp"
#include "runtime/format.hpp"
#include "runtime/runtime_globals.hpp"
#include "ui/banner.hpp"
#include "ui/box_render.hpp"
#include "ui/btc_balance.hpp"
#include "ui/interactive.hpp"

using namespace collider;
using collider::ui::format_rate;
using collider::runtime::format_number_human;

// Forward decl: defined at namespace-global scope in puzzle_solver.cpp.
std::string format_number(uint64_t n);

namespace collider::runtime::detail {

namespace {

// R-B8: persist GPU brute-force position + total_checked to disk. Used
// both by the 30-second periodic checkpoint and by the SIGINT save path.
// The v3 mirror fields (position_lo/hi) are populated alongside the
// canonical v4 position_full[0..3] so a v1.5 multi-limb reader can load
// the file without an intermediate migration step. The GPU brute path
// is 128-bit, so limbs [2..3] are always zero here.
void save_gpu_brute_state(int current_puzzle,
                          uint64_t current_lo,
                          uint64_t current_hi,
                          uint64_t total_checked) {
    collider::PuzzleSearchState state;
    state.puzzle_number = current_puzzle;
    state.position_lo = current_lo;
    state.position_hi = current_hi;
    state.position_full[0] = current_lo;
    state.position_full[1] = current_hi;
    state.position_full[2] = 0;
    state.position_full[3] = 0;
    state.total_checked = total_checked;
    collider::SearchStateManager::save_puzzle_state(state);
}

// Re-audit 5/7 follow-up (2026-05-17): per-batch telemetry/save/log
// heartbeat for the GPU brute scan. Owns the three throttled side-
// effects emitted after every batch advance:
//   (1) ~1s rate-line refresh on stdout
//   (2) ~30s SearchState checkpoint (save_gpu_brute_state)
//   (3) ~60s file-log progress emit (logger.log_progress)
// All clocks come in via mutable references so the helper can update
// them in place; the loop body just calls this once per iteration and
// reads no other state from the helper. Extracted to keep
// run_bruteforce_gpu under the 200-line ceiling.
struct GpuBruteTelemetryState {
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point& last_update;
    std::chrono::steady_clock::time_point& last_state_save;
    std::chrono::steady_clock::time_point& last_log_time;
    bool   is_sequential;
    int    current_puzzle;
};

inline void emit_gpu_brute_periodic_telemetry(
    GpuBruteTelemetryState& st,
    uint64_t total_checked,
    uint64_t session_checked,
    uint64_t current_lo,
    uint64_t current_hi)
{
    auto& logger = Logger::instance();
    auto now = std::chrono::steady_clock::now();

    if (std::chrono::duration_cast<std::chrono::seconds>(now - st.last_update).count() >= 1) {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - st.start_time).count();
        // Use session_checked for rate (not total_checked) to avoid
        // inflated rate on resume.
        double rate = (elapsed_ms > 0) ? (session_checked * 1000.0 / elapsed_ms) : 0;

        std::cout << "\r[*] "
                  << (st.is_sequential ? "Sequential" : "Random")
                  << " | Checked: " << std::setw(12) << format_number_human(total_checked)
                  << " | Rate: " << std::setw(10) << format_rate(rate)
                  << "   " << std::flush;

        st.last_update = now;
    }

    // Save state periodically (every 30 seconds). v1.4.2 R-B8:
    // populate the canonical position_full[0..3] in addition to
    // the v3 mirror fields. The GPU brute path is 128-bit, so
    // limbs [2..3] are always zero here; the v1.5 multi-limb
    // path will populate them.
    if (std::chrono::duration_cast<std::chrono::seconds>(now - st.last_state_save).count() >= 30) {
        save_gpu_brute_state(st.current_puzzle, current_lo, current_hi, total_checked);
        st.last_state_save = now;
    }

    // File logging for crash diagnosis (every 60 seconds).
    if (std::chrono::duration_cast<std::chrono::seconds>(now - st.last_log_time).count() >= 60) {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - st.start_time).count();
        double rate = (elapsed_ms > 0) ? (session_checked * 1000.0 / elapsed_ms) : 0;
        logger.log_progress(total_checked, rate);
        st.last_log_time = now;
    }
}

// Print the GPU solved banner + write puzzle_found.txt + secure_wipe.
// Extracted to keep run_bruteforce_gpu under the 200-line ceiling.
// All bookkeeping (logger.log_found, SearchStateManager::clear_puzzle_state,
// next-puzzle hint) stays inline at the call site since it is wired into
// the per-puzzle continuation logic. Returns the now-zeroed key_hex
// buffer length so the caller can confirm wipe semantics.
void report_gpu_brute_hit(const PuzzleIterContext& ctx,
                          uint64_t found_key_lo,
                          uint64_t found_key_hi,
                          uint64_t total_checked,
                          double elapsed,
                          const gpu::MultiGPUPuzzleSolver& gpu_solver) {
    const PuzzleInfo* puzzle = ctx.puzzle;
    const PuzzleTarget& tgt = ctx.tgt;
    const int bits = tgt.bits;
    const int current_puzzle = ctx.current_puzzle;
    const std::string& h160_hex = tgt.h160_hex;
    const std::string& target_address = tgt.target_address;
    auto& logger = Logger::instance();

    auto solve_time = std::chrono::system_clock::now();
    auto solve_time_t = std::chrono::system_clock::to_time_t(solve_time);
    char timestamp[64];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&solve_time_t));

    char key_hex[67];
    if (found_key_hi > 0) {
        snprintf(key_hex, sizeof(key_hex), "0x%llx%016llx",
                 (unsigned long long)found_key_hi, (unsigned long long)found_key_lo);
    } else {
        snprintf(key_hex, sizeof(key_hex), "0x%llx", (unsigned long long)found_key_lo);
    }

    // Flip the dashboard phase to SOLVED so the operator sees the
    // success on the TUI dashboard. The cout boxed banner below
    // executes too (captured to boot log; puzzle_found.txt is the
    // durable file record).
    if (auto* tui_solved =
            static_cast<::collider::ui::tui::TuiApp*>(ctx.tui_app)) {
        std::ostringstream solved_phase;
        solved_phase << "Puzzle #" << current_puzzle
                     << " SOLVED (GPU brute force)";
        tui_solved->set_current_phase_name(solved_phase.str());
    }
    {
        namespace boxui = ::collider::ui::box;
        std::cout << "\n\n";
        std::cout << boxui::ansi::BRIGHT_GREEN;
        boxui::top(std::cout);
        boxui::centered(std::cout, "PUZZLE SOLVED! (GPU Accelerated)");
        boxui::top(std::cout);
        std::cout << boxui::ansi::RESET;

        std::ostringstream pz; pz << "#" << current_puzzle << " (" << bits << "-bit)";
        std::ostringstream dur; dur << std::fixed << std::setprecision(3) << elapsed << " sec";
        std::ostringstream acc; acc << gpu_solver.num_gpus() << "x CUDA GPUs";
        boxui::kv(std::cout, "Puzzle",       pz.str(),                       boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Private Key",  key_hex,                        boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Address",      target_address,                 boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Balance",
                  ::collider::ui::format_balance(
                      ::collider::ui::fetch_balance_btc(target_address)),
                  boxui::ansi::BRIGHT_MAGENTA);
        boxui::sep(std::cout);
        boxui::kv(std::cout, "Solved At",    timestamp,                      boxui::ansi::BRIGHT_CYAN);
        boxui::kv(std::cout, "Duration",     dur.str(),                      boxui::ansi::BRIGHT_CYAN);
        boxui::kv(std::cout, "Keys Checked", format_number_human(total_checked), boxui::ansi::BRIGHT_CYAN);
        boxui::kv(std::cout, "Accelerator",  acc.str(),                      boxui::ansi::BRIGHT_CYAN);
        if (puzzle && puzzle->btc_reward > 0) {
            std::ostringstream rw; rw << std::fixed << std::setprecision(1)
                                      << puzzle->btc_reward << " BTC";
            boxui::kv(std::cout, "BTC Reward", rw.str(), boxui::ansi::BRIGHT_MAGENTA);
        }
        boxui::bottom(std::cout);
        std::cout << "\n";
    }

    // Save to file
    // Owner-only permissions; see secure_open_ofstream notes.
    // FailHard: a world-readable puzzle_found.txt with a
    // recovered private key is a worse outcome than no file
    // at all (the user can re-run; they cannot un-leak a key).
    std::ofstream found_file =
        ::collider::secure_open_ofstream(
            "puzzle_found.txt", std::ios::app,
            ::collider::SecureWriteOnFailure::FailHard);
    if (found_file) {
        found_file << "================================================================================\n";
        found_file << "                    PUZZLE SOLVED! (GPU Accelerated)\n";
        found_file << "================================================================================\n";
        found_file << "Timestamp:    " << timestamp << "\n";
        found_file << "Puzzle:       #" << current_puzzle << " (" << bits << "-bit)\n";
        found_file << "Private Key:  " << key_hex << "\n";
        found_file << "Address:      " << target_address << "\n";
        found_file << "Hash160:      " << h160_hex << "\n";
        found_file << "Keys Checked: " << total_checked << "\n";
        found_file << "Duration:     " << std::fixed << std::setprecision(3) << elapsed << " seconds\n";
        found_file << "Accelerator:  " << gpu_solver.num_gpus() << "x CUDA GPUs (Optimized)\n";
        found_file << "================================================================================\n\n";
        found_file.close();
        std::cout << "[*] Solution saved to: puzzle_found.txt\n";
    }

    // Log the discovery!
    logger.log_found(found_key_lo, found_key_hi, target_address);

    // Session log: redact the private key per spec. The recovered
    // key is persisted to puzzle_found.txt via secure_open_ofstream
    // (FailHard, owner-only DACL); duplicating it in the session log
    // would create a second on-disk copy with a less restrictive
    // permission set.
    {
        std::ostringstream d;
        d << "puzzle=" << current_puzzle
          << " algorithm=GPU_brute"
          << " address=" << target_address;
        ::collider::log::milestone("puzzle_solved", d.str());
    }

    // Wipe the recovered key locals once they have hit
    // disk + the persistent logger. The logger holds its
    // own copy in its append buffer; the locals here only
    // matter for this stack frame.
    ::collider::secure_wipe(&found_key_lo, sizeof(found_key_lo));
    ::collider::secure_wipe(&found_key_hi, sizeof(found_key_hi));
    ::collider::secure_wipe(key_hex, sizeof(key_hex));
}

// GPU brute scan cursor. Owns the current scan position + the RNG +
// the limb-decomposed [start, end] range. advance() picks the next
// position per mode (sequential bump vs random rejection sample).
// Extracted from run_bruteforce_gpu so the function body stays under
// the 200-line ceiling and so the cursor logic is testable in isolation
// (the GPU loop just calls advance(batch) and reads current_lo/hi).
struct BruteScanCursor {
    uint64_t current_lo = 0;
    uint64_t current_hi = 0;
    uint64_t start_lo = 0;
    uint64_t start_hi = 0;
    uint64_t range_size_lo = 0;
    uint64_t range_size_hi = 0;
    bool     is_sequential = true;
    std::mt19937_64 rng;

    BruteScanCursor(uint64_t s_lo, uint64_t s_hi,
                    uint64_t e_lo, uint64_t e_hi,
                    bool sequential,
                    std::mt19937_64&& seeded_rng)
        : current_lo(s_lo), current_hi(s_hi),
          start_lo(s_lo), start_hi(s_hi),
          is_sequential(sequential),
          rng(std::move(seeded_rng)) {
        // Range size = end - start (128-bit subtract with borrow).
        // Compatible with MSVC (no __int128 there); explicit limb math.
        range_size_lo = e_lo - s_lo;
        range_size_hi = e_hi - s_hi - (e_lo < s_lo ? 1ULL : 0ULL);
    }

    // Advance current_(lo,hi) to a new sample. For sequential mode this
    // is current += advance_count (with cross-limb carry). For random
    // mode this is a fresh uniform sample in [start, end]. Random
    // rejection sampling: draw a 128-bit value and accept only when it
    // falls strictly below range_size. The expected loop count is < 2
    // because range_size occupies the top bit of its 128-bit
    // representation for any non-trivial puzzle.
    void advance(uint64_t advance_count) {
        if (is_sequential) {
            uint64_t new_lo = current_lo + advance_count;
            if (new_lo < current_lo) current_hi++;  // carry
            current_lo = new_lo;
            return;
        }
        std::uniform_int_distribution<uint64_t> dlo(0, UINT64_MAX);
        uint64_t off_hi, off_lo;
        for (;;) {
            off_hi = dlo(rng);
            off_lo = dlo(rng);
            // Accept when (off_hi, off_lo) < (range_size_hi, range_size_lo).
            if (off_hi < range_size_hi) break;
            if (off_hi == range_size_hi && off_lo < range_size_lo) break;
            if (range_size_hi == 0 && range_size_lo == 0) {
                off_hi = 0;
                off_lo = 0;
                break;
            }
        }
        // pos = start + offset (128-bit add with carry).
        uint64_t sum_lo = start_lo + off_lo;
        uint64_t carry  = (sum_lo < start_lo) ? 1ULL : 0ULL;
        uint64_t sum_hi = start_hi + off_hi + carry;
        current_lo = sum_lo;
        current_hi = sum_hi;
    }
};

// Persist final state on SIGINT and emit the resume-hint line. Used
// only by the GPU brute path; kept TU-local because save_gpu_brute_state
// is local too.
void do_gpu_brute_shutdown_save(int current_puzzle,
                                uint64_t current_lo,
                                uint64_t current_hi,
                                uint64_t total_checked) {
    auto& logger = Logger::instance();
    save_gpu_brute_state(current_puzzle, current_lo, current_hi, total_checked);
    std::cout << "\n[*] State saved - run again to resume from "
              << format_number(total_checked) << " keys\n";
    logger.log_state_save(current_puzzle, current_lo, current_hi);

    // Session log: shutdown-time state save. The position is the
    // 128-bit (hi:lo) window the next resume will start from; the
    // total_checked is the per-puzzle progress. We log only the
    // low-order hex of the position; the full 256-bit position is
    // available in ~/.collider/state/ for resume.
    {
        char pos_hex[33];
        std::snprintf(pos_hex, sizeof(pos_hex), "%016llx%016llx",
                      static_cast<unsigned long long>(current_hi),
                      static_cast<unsigned long long>(current_lo));
        std::ostringstream d;
        d << "puzzle=" << current_puzzle
          << " total_checked=" << total_checked
          << " position_lo128=" << pos_hex;
        ::collider::log::milestone("state_saved", d.str());

        ::collider::log::SessionState s;
        s.mode = "puzzle";
        s.puzzle_number = current_puzzle;
        s.puzzle_algorithm = "GPU brute";
        s.total_steps = total_checked;
        s.position_full_hex = std::string(pos_hex);
        ::collider::log::update_session_state(s);
    }
}

// Render the GPU brute path's post-scan summary banner. Mirrors the
// inline block that used to live just before the StoppedExitOrContinue
// return.
void render_gpu_brute_final_stats(uint64_t session_checked,
                                  uint64_t total_checked,
                                  double elapsed_sec,
                                  bool found) {
    auto& logger = Logger::instance();
    double session_rate = session_checked / std::max(0.001, elapsed_sec);

    // Log shutdown with reason.
    std::string shutdown_reason = g_shutdown ? "User interrupt (Ctrl+C)" :
                                  (found ? "Key found!" : "Range exhausted (sequential mode)");
    logger.log_shutdown(shutdown_reason, total_checked, elapsed_sec);

    // Session log: shutdown reason for the puzzle path. The "shutdown"
    // milestone in main.cpp covers SIGINT only; this one fires from
    // natural exit (key found / range exhausted) AND from SIGINT (so
    // both events land in the session log, with the SIGINT case
    // double-tagged once by main.cpp's emit_shutdown_message_from_main
    // and once here with the per-puzzle reason context).
    ::collider::log::milestone("puzzle_stopped", "reason=" + shutdown_reason);

    namespace boxui = ::collider::ui::box;
    std::cout << "\n\n";
    boxui::top(std::cout);
    boxui::centered(std::cout, "GPU PUZZLE SEARCH RESULTS");
    boxui::top(std::cout);
    {
        std::ostringstream dur;
        dur << std::fixed << std::setprecision(2) << elapsed_sec << " seconds";
        boxui::kv(std::cout, "Session Duration", dur.str());
    }
    boxui::kv(std::cout, "Session Checked", format_number_human(session_checked));
    boxui::kv(std::cout, "Total Checked",   format_number_human(total_checked));
    boxui::kv(std::cout, "Session Rate",    format_rate(session_rate));
    boxui::bottom(std::cout);
}

// Post-hit bookkeeping: clears saved puzzle state and prints the
// continuation hint (auto-progression banner vs next-puzzle suggestion).
// The actual banner/file/secure_wipe is in report_gpu_brute_hit; this
// helper only handles the "what happens after we have written the key
// to disk" side of the path.
void finalize_gpu_brute_hit(int current_puzzle, bool is_multi_puzzle) {
    collider::SearchStateManager::clear_puzzle_state(current_puzzle);
    if (is_multi_puzzle) {
        std::cout << "[*] Puzzle solved! Continuing to next puzzle...\n";
        return;
    }
    auto unsolved = PuzzleDatabase::get_unsolved();
    if (!unsolved.empty()) {
        std::cout << "\n[*] Next unsolved puzzle: #" << unsolved[0]->number
                  << " (" << unsolved[0]->bits << "-bit, "
                  << std::fixed << std::setprecision(1) << unsolved[0]->btc_reward << " BTC)\n";
    }
}

// GPU brute-force backend. Returns FallThrough when gpu_solver.init or
// set_target fails OR when have_target_hash / force_sequential gate is
// not satisfied. The caller (run_bruteforce_solve) then routes to the
// CPU fallback path.
PuzzleStepResult run_bruteforce_gpu(PuzzleIterContext& ctx) {
    Arguments& args = ctx.args;
    const PuzzleTarget& tgt = ctx.tgt;
    const int bits = tgt.bits;
    const int current_puzzle = ctx.current_puzzle;
    const bool is_multi_puzzle = ctx.is_multi_puzzle;
    const std::array<uint8_t, 20>& target_hash160 = tgt.target_hash160;
    const bool have_target_hash = tgt.have_target_hash;
    const bool force_sequential = tgt.force_sequential;
    const uint64_t start_lo = tgt.start_lo;
    const uint64_t start_hi = tgt.start_hi;
    const uint64_t end_lo = tgt.end_lo;
    const uint64_t end_hi = tgt.end_hi;

    auto& logger = Logger::instance();

    // GPU is only attempted when we have a target hash160 AND the puzzle
    // is not gated to force_sequential (small puzzles cracked end-to-end
    // by the CPU path).
    if (!have_target_hash || force_sequential) {
        return PuzzleStepResult::FallThrough;
    }

    gpu::MultiGPUPuzzleSolver gpu_solver;

    // Initialize multi-GPU solver with user-specified GPUs
    gpu::MultiGPUPuzzleSolver::Config gpu_config;
    gpu_config.gpu_ids = args.gpu_ids;
    gpu_config.batch_size_per_gpu = args.batch_size;  // 4M keys per GPU per batch

    if (!gpu_solver.init(gpu_config) || !gpu_solver.set_target(target_hash160)) {
        // Reviewer 5/7 follow-up: the kangaroo sibling
        // (puzzle_solver_kangaroo.cpp:358-366) fires gpu_faulted on any
        // backend-reported error_count so post-mortem log forensics can
        // tell a clean fallback from a real GPU fault. The brute path
        // had puzzle_start / puzzle_solved / puzzle_stopped wired but
        // no gpu_faulted equivalent at its fault entry point. Wire it
        // here so an operator scanning ~/.collider/logs after a "why
        // did this run silently demote to CPU?" complaint sees the
        // event with enough context to find the underlying cuda /
        // memory failure in collider.log alongside.
        //
        // Note: a FallThrough return without this milestone is the
        // legitimate path for "GPU not eligible" (no target hash or
        // force_sequential), but those gates were already enforced by
        // the early-return above this block. Reaching here means we
        // entered the init / set_target path and it failed; that is a
        // GPU fault, not an eligibility miss.
        std::ostringstream d;
        d << "backend=GPU_brute"
          << " puzzle=" << current_puzzle
          << " stage=init_or_set_target"
          << " gpus_requested=" << args.gpu_ids.size()
          << " batch_size=" << args.batch_size;
        ::collider::log::milestone("gpu_faulted", d.str());
        return PuzzleStepResult::FallThrough;
    }

    std::cout << "\n[*] Starting MULTI-GPU optimized puzzle search...\n";
    std::cout << "    Pipeline: PrivKey -> EC Mul (precomp) -> Compress -> SHA256 -> RIPEMD160 -> Compare\n";
    std::cout << "    GPUs: " << gpu_solver.num_gpus() << " x " << ctx.gpu_info.backend << "\n";
    std::cout << "    Optimizations: Precomputed tables, inline hashes, batch inversion\n";
    std::cout << "    Log: " << logger.get_log_path() << "\n";
    std::cout << "    Press Ctrl+C to stop\n\n";

    // Log startup info for crash diagnosis
    logger.log_startup(current_puzzle, gpu_solver.num_gpus(), ctx.gpu_info.gpu_names,
                       args.batch_size, args.puzzle_random ? "Random" : "Zone-Based");

    // Session log: record the puzzle solve start. The algorithm tag
    // is "GPU brute" (this function); a separate sibling helper in
    // puzzle_solver_kangaroo.cpp handles "RCKangaroo" / "MultiGPU" /
    // "CPU" paths and wires its own milestone.
    {
        std::ostringstream d;
        d << "puzzle=" << current_puzzle
          << " algorithm=GPU_brute"
          << " gpus=" << gpu_solver.num_gpus()
          << " batch_size=" << args.batch_size;
        ::collider::log::milestone("puzzle_start", d.str());

        ::collider::log::SessionState s;
        s.mode = "puzzle";
        s.puzzle_number = current_puzzle;
        s.puzzle_algorithm = "GPU brute";
        ::collider::log::update_session_state(s);
    }

    auto start_time = std::chrono::steady_clock::now();
    uint64_t total_checked = 0;
    uint64_t session_checked = 0;  // Keys checked in THIS session only (for accurate rate)
    auto last_update = start_time;
    auto last_state_save = start_time;
    auto last_log_time = start_time;  // For periodic file logging
    bool found = false;
    uint64_t found_key_lo = 0, found_key_hi = 0;

    // GPU batch size - much larger than CPU
    size_t gpu_batch_size = args.batch_size;  // 4M keys per batch

    // ================================================================
    // GPU BRUTE-FORCE SCAN (v1.4.2 C.9: uniform, not Center-Heavy)
    // ================================================================
    // Without a verifiable prior on key location, the unbiased strategy
    // is uniform random sampling over the puzzle's [start, end] range.
    // The default mode is random; --sequential-search forces a linear
    // scan from start (useful for cracking small puzzles end-to-end
    // and for deterministic test replays).

    const bool is_sequential = !args.puzzle_random || force_sequential;
    std::cout << "[*] Using "
              << (is_sequential ? "Sequential" : "Random")
              << " scanning over the full puzzle range\n\n";

    // RNG for random mode. mt19937_64 seeded from system entropy.
    std::random_device rd;
    std::mt19937_64 scan_rng(((uint64_t)rd() << 32) ^ (uint64_t)rd());

    // Scan cursor: owns current position + RNG + range-size limbs +
    // sampling mode. advance(n) picks the next batch position.
    BruteScanCursor cursor(start_lo, start_hi, end_lo, end_hi,
                           is_sequential, std::move(scan_rng));

    // Saved-state resume only meaningful for sequential mode; in random
    // mode each batch picks an independent uniform sample so resume is
    // a no-op semantic-wise; we still load total_checked for the
    // human-readable cumulative counter.
    auto saved_state = SearchStateManager::load_puzzle_state(current_puzzle);
    if (saved_state.valid && saved_state.total_checked > 0) {
        std::cout << "[*] Resuming saved counters:\n";
        std::cout << "    Last saved: " << saved_state.timestamp << "\n";
        std::cout << "    Keys checked: "
                  << format_number(saved_state.total_checked) << "\n\n";
        cursor.current_lo = saved_state.position_lo;
        cursor.current_hi = saved_state.position_hi;
        total_checked = saved_state.total_checked;
    }

    while (!g_shutdown && !found) {
        // For sequential mode, stop once we have covered the range.
        if (is_sequential) {
            if (cursor.current_hi > end_hi ||
                (cursor.current_hi == end_hi && cursor.current_lo >= end_lo)) {
                std::cout << "\n[!] GPU search complete - full range scanned.\n";
                if (have_target_hash) std::cout << "[!] No match found.\n";
                break;
            }
        }

        // Search this batch on GPU
        if (gpu_solver.search_batch(cursor.current_lo, cursor.current_hi, gpu_batch_size,
                                    found_key_lo, found_key_hi)) {
            found = true;

            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count() / 1000.0;

            // Banner + puzzle_found.txt + secure_wipe of the recovered
            // key locals, all in one helper to keep this function short.
            report_gpu_brute_hit(ctx, found_key_lo, found_key_hi,
                                 total_checked, elapsed, gpu_solver);

            // Clear saved state + print continuation hint.
            finalize_gpu_brute_hit(current_puzzle, is_multi_puzzle);

            return PuzzleStepResult::SolvedExitOrContinue;
        }

        total_checked += gpu_batch_size;
        session_checked += gpu_batch_size;  // Track session-only for accurate rate

        // Advance to next batch (sequential bump or fresh random sample).
        cursor.advance(gpu_batch_size);

        GpuBruteTelemetryState telemetry_state{
            /*start_time=*/      start_time,
            /*last_update=*/     last_update,
            /*last_state_save=*/ last_state_save,
            /*last_log_time=*/   last_log_time,
            /*is_sequential=*/   is_sequential,
            /*current_puzzle=*/  current_puzzle,
        };
        emit_gpu_brute_periodic_telemetry(
            telemetry_state, total_checked, session_checked,
            cursor.current_lo, cursor.current_hi);
    }

    // emit the deferred shutdown message from
    // main-thread context (idempotent; no-op on the !g_shutdown path
    // and on the second-or-later call).
    if (g_shutdown.load(std::memory_order_acquire)) {
        emit_shutdown_message_from_main();
    }

    // Save state on shutdown for resume. v1.4.2 R-B8: populate
    // the canonical position_full[0..3] alongside the mirror
    // fields so the file is loadable by a v1.5 multi-limb
    // reader without an intermediate migration step.
    if (g_shutdown) {
        do_gpu_brute_shutdown_save(current_puzzle, cursor.current_lo, cursor.current_hi, total_checked);
    }

    // GPU search completed (or interrupted).
    auto end_time = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

    render_gpu_brute_final_stats(session_checked, total_checked, elapsed_sec, found);

    // Search loop exited without a solve (interrupted, exhausted, or
    // completed): honor --all-unsolved / --auto-next so the next puzzle
    // in the worklist gets a chance. Solved hits return earlier from
    // the inline solve path inside this function.
    return PuzzleStepResult::StoppedExitOrContinue;
}

// Print the CPU solved banner + write puzzle_found.txt + secure_wipe.
// Extracted from run_bruteforce_cpu to keep that helper short and to
// mirror report_gpu_brute_hit. The next-puzzle hint stays inline so the
// outer function controls per-puzzle continuation.
void report_cpu_brute_hit(const PuzzleIterContext& ctx,
                          uint64_t found_key_lo,
                          uint64_t found_key_hi,
                          uint64_t total_checked,
                          double elapsed) {
    const PuzzleInfo* puzzle = ctx.puzzle;
    const PuzzleTarget& tgt = ctx.tgt;
    const int bits = tgt.bits;
    const int current_puzzle = ctx.current_puzzle;
    const std::string& h160_hex = tgt.h160_hex;
    const std::string& target_address = tgt.target_address;

    auto solve_time = std::chrono::system_clock::now();
    auto solve_time_t = std::chrono::system_clock::to_time_t(solve_time);
    char timestamp[64];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&solve_time_t));

    char key_hex[67];
    if (found_key_hi > 0) {
        snprintf(key_hex, sizeof(key_hex), "0x%llx%016llx",
                 (unsigned long long)found_key_hi, (unsigned long long)found_key_lo);
    } else {
        snprintf(key_hex, sizeof(key_hex), "0x%llx", (unsigned long long)found_key_lo);
    }

    // Flip dashboard phase to SOLVED (CPU brute force path).
    if (auto* tui_solved =
            static_cast<::collider::ui::tui::TuiApp*>(ctx.tui_app)) {
        std::ostringstream solved_phase;
        solved_phase << "Puzzle #" << current_puzzle
                     << " SOLVED (CPU brute force)";
        tui_solved->set_current_phase_name(solved_phase.str());
    }
    {
        namespace boxui = ::collider::ui::box;
        std::cout << "\n\n";
        std::cout << boxui::ansi::BRIGHT_GREEN;
        boxui::top(std::cout);
        // Emoji widths break visible-length math; use ASCII.
        boxui::centered(std::cout, "PUZZLE SOLVED!");
        boxui::top(std::cout);
        std::cout << boxui::ansi::RESET;

        std::ostringstream pz; pz << "#" << current_puzzle << " (" << bits << "-bit)";
        std::ostringstream dur; dur << std::fixed << std::setprecision(3) << elapsed << " sec";
        boxui::kv(std::cout, "Puzzle",       pz.str(),                               boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Private Key",  key_hex,                                boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Address",      target_address,                         boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Balance",
                  ::collider::ui::format_balance(
                      ::collider::ui::fetch_balance_btc(target_address)),
                  boxui::ansi::BRIGHT_MAGENTA);
        boxui::sep(std::cout);
        boxui::kv(std::cout, "Solved At",    timestamp,                               boxui::ansi::BRIGHT_CYAN);
        boxui::kv(std::cout, "Duration",     dur.str(),                               boxui::ansi::BRIGHT_CYAN);
        boxui::kv(std::cout, "Keys Checked", format_number_human(total_checked),      boxui::ansi::BRIGHT_CYAN);
        if (puzzle && puzzle->btc_reward > 0) {
            std::ostringstream rw; rw << std::fixed << std::setprecision(1)
                                       << puzzle->btc_reward << " BTC";
            boxui::kv(std::cout, "BTC Reward", rw.str(), boxui::ansi::BRIGHT_MAGENTA);
        }
        boxui::bottom(std::cout);
        std::cout << "\n";
    }

    // Save to file with full details
    // Owner-only permissions; see secure_open_ofstream notes.
    // FailHard: same rationale as the GPU-accelerated sibling
    // above. Recovered private key sink: world-readable file
    // would be worse than no file.
    std::ofstream found_file =
        ::collider::secure_open_ofstream(
            "puzzle_found.txt", std::ios::app,
            ::collider::SecureWriteOnFailure::FailHard);
    if (found_file) {
        found_file << "================================================================================\n";
        found_file << "                         PUZZLE SOLVED!\n";
        found_file << "================================================================================\n";
        found_file << "Timestamp:    " << timestamp << "\n";
        found_file << "Puzzle:       #" << current_puzzle << " (" << bits << "-bit)\n";
        found_file << "Private Key:  " << key_hex << "\n";
        found_file << "Address:      " << target_address << "\n";
        found_file << "Hash160:      " << h160_hex << "\n";
        found_file << "Keys Checked: " << total_checked << "\n";
        found_file << "Duration:     " << std::fixed << std::setprecision(3) << elapsed << " seconds\n";
        if (puzzle && puzzle->btc_reward > 0) {
            found_file << "BTC Reward:   " << std::fixed << std::setprecision(1)
                       << puzzle->btc_reward << " BTC\n";
        }
        found_file << "================================================================================\n\n";
        found_file.close();
        std::cout << "[*] Solution saved to: puzzle_found.txt\n";
    }

    // Wipe the recovered key locals once they have hit
    // disk. The persistent SearchStateManager / logger
    // do their own zero-on-clear (see search_state.hpp).
    ::collider::secure_wipe(&found_key_lo, sizeof(found_key_lo));
    ::collider::secure_wipe(&found_key_hi, sizeof(found_key_hi));
    ::collider::secure_wipe(key_hex, sizeof(key_hex));
}

// Print the post-scan summary banner used by the CPU fallback path.
// Mirrors the original inline `search_done:` fallthrough block.
void render_cpu_brute_final_stats(int bits,
                                  uint64_t total_checked,
                                  double elapsed_sec) {
    double final_rate = total_checked / std::max(0.001, elapsed_sec);

    // Time to complete estimates (for puzzles we know the range size)
    double range_size_approx = std::pow(2.0, bits - 1);  // 2^(N-1) keys
    double remaining_approx = range_size_approx - total_checked;
    double time_to_complete_sec = remaining_approx / std::max(1.0, final_rate);
    double days_to_complete = time_to_complete_sec / 86400;

    {
        namespace boxui = ::collider::ui::box;
        std::cout << "\n\n";
        boxui::top(std::cout);
        boxui::centered(std::cout, "PUZZLE SEARCH RESULTS");
        boxui::top(std::cout);

        std::ostringstream dur; dur << std::fixed << std::setprecision(2) << elapsed_sec << " seconds";
        std::ostringstream kc;  kc  << format_number_human(total_checked);
        std::ostringstream rt;  rt  << format_rate(final_rate);
        boxui::kv(std::cout, "Duration",     dur.str());
        boxui::kv(std::cout, "Keys Checked", kc.str());
        boxui::kv(std::cout, "Average Rate", rt.str());
        boxui::sep(std::cout);

        std::ostringstream rs; rs << "2^" << (bits - 1) << " keys";
        boxui::kv(std::cout, "Range Size", rs.str());

        std::ostringstream eta;
        eta << std::fixed << std::setprecision(1);
        if (days_to_complete < 1) {
            eta << (time_to_complete_sec / 3600) << " hours";
        } else if (days_to_complete < 365) {
            eta << days_to_complete << " days";
        } else if (days_to_complete < 1e6) {
            eta << (days_to_complete / 365) << " years";
        } else {
            eta.str("");
            eta << std::scientific << std::setprecision(2)
                << (days_to_complete / 365) << " years";
        }
        boxui::kv(std::cout, "ETA (current)", eta.str());
        boxui::bottom(std::cout);
        std::cout << "\n";
    }

    std::cout << "[!] Note: Puzzle mode is using CPU simulation.\n";
    std::cout << "    Real GPU performance will be significantly higher.\n";
    std::cout << "    Once GPU pipeline is integrated, expect ~1B+ keys/sec per GPU.\n";
}

// CPU brute-force fallback. Last-resort path when the GPU helper
// reported FallThrough (init failure or have_target_hash/force_sequential
// gate not satisfied). Always terminal: either SolvedExitOrContinue or
// StoppedExitOrContinue.
PuzzleStepResult run_bruteforce_cpu(PuzzleIterContext& ctx) {
    Arguments& args = ctx.args;
    const PuzzleTarget& tgt = ctx.tgt;
    const int bits = tgt.bits;
    const int current_puzzle = ctx.current_puzzle;
    const bool is_multi_puzzle = ctx.is_multi_puzzle;
    const std::array<uint8_t, 20>& target_hash160 = tgt.target_hash160;
    const bool have_target_hash = tgt.have_target_hash;
    const bool force_sequential = tgt.force_sequential;
    const uint64_t start_lo = tgt.start_lo;
    const uint64_t start_hi = tgt.start_hi;
    const uint64_t end_lo = tgt.end_lo;
    const uint64_t end_hi = tgt.end_hi;

    std::cout << "\n[*] Starting puzzle search...\n";
    std::cout << "    Pipeline: PrivKey -> secp256k1 -> PubKey -> SHA256 -> RIPEMD160 -> Compare\n";
    std::cout << "    Using: CPU reference implementation\n";
    std::cout << "    Press Ctrl+C to stop\n\n";

    // RNG for CPU random mode (lives only here, distinct from GPU's scan_rng).
    std::random_device rd_cpu;
    std::mt19937_64 rng(rd_cpu());
    std::uniform_int_distribution<uint64_t> dist_lo(0, UINT64_MAX);
    std::uniform_int_distribution<uint64_t> dist_hi(start_hi, end_hi);

    auto start_time = std::chrono::steady_clock::now();
    uint64_t total_checked = 0;
    uint64_t batch_count = 0;
    auto last_update = start_time;
    bool found = false;
    uint64_t found_key_lo = 0, found_key_hi = 0;

    // For sequential search, track current position
    uint64_t seq_lo = start_lo;
    uint64_t seq_hi = start_hi;

    // Main puzzle search loop (CPU)
    while (!g_shutdown && !found) {
        // Generate batch of keys within range
        std::vector<std::pair<uint64_t, uint64_t>> key_batch;

        // Limit batch size for CPU (much slower than GPU)
        size_t cpu_batch_size = std::min(args.batch_size, (size_t)10000);
        key_batch.reserve(cpu_batch_size);

        bool range_exhausted = false;
        if (!args.puzzle_random || force_sequential) {
            // Sequential search - exhaustive for small puzzles
            for (size_t i = 0; i < cpu_batch_size; i++) {
                // Check if we've exceeded range
                if (seq_hi > end_hi || (seq_hi == end_hi && seq_lo > end_lo)) {
                    range_exhausted = true;
                    break;  // Break to process remaining keys in batch
                }
                key_batch.emplace_back(seq_lo, seq_hi);

                // Increment
                seq_lo++;
                if (seq_lo == 0) seq_hi++;  // Carry
            }
        } else {
            // Random search - generate keys uniformly in [start, end]
            for (size_t i = 0; i < cpu_batch_size; i++) {
                uint64_t hi = dist_hi(rng);
                uint64_t lo;

                if (hi == start_hi && hi == end_hi) {
                    std::uniform_int_distribution<uint64_t> dist_constrained(start_lo, end_lo);
                    lo = dist_constrained(rng);
                } else if (hi == start_hi) {
                    std::uniform_int_distribution<uint64_t> dist_above(start_lo, UINT64_MAX);
                    lo = dist_above(rng);
                } else if (hi == end_hi) {
                    std::uniform_int_distribution<uint64_t> dist_below(0, end_lo);
                    lo = dist_below(rng);
                } else {
                    lo = dist_lo(rng);
                }
                key_batch.emplace_back(lo, hi);
            }
        }

        // Process batch - compute hash160 for each key and check
        for (const auto& [key_lo, key_hi] : key_batch) {
            // Convert key to bytes
            uint8_t privkey_bytes[32];
            cpu::key_to_bytes(privkey_bytes, key_lo, key_hi);

            // Compute hash160
            auto hash160 = cpu::compute_hash160(privkey_bytes);

            // Compare with target
            if (have_target_hash && hash160 == target_hash160) {
                found = true;
                found_key_lo = key_lo;
                found_key_hi = key_hi;

                double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start_time).count() / 1000.0;

                // Banner + puzzle_found.txt + secure_wipe of the recovered
                // key locals, all in one helper to mirror the GPU sibling.
                report_cpu_brute_hit(ctx, found_key_lo, found_key_hi,
                                     total_checked, elapsed);

                // Show next puzzle suggestion for manual mode (i.e. when
                // we're not already chained to the next puzzle by
                // --all-unsolved or --auto-next)
                if (!is_multi_puzzle) {
                    auto unsolved = PuzzleDatabase::get_unsolved();
                    if (!unsolved.empty()) {
                        std::cout << "\n[*] Next unsolved puzzle: #" << unsolved[0]->number
                                  << " (" << unsolved[0]->bits << "-bit, "
                                  << std::fixed << std::setprecision(1) << unsolved[0]->btc_reward << " BTC)\n";
                        std::cout << "    Run: collider --puzzle " << unsolved[0]->number << "\n";
                    }
                }

                break;
            }

            total_checked++;
        }

        batch_count++;

        // Status update every second
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_update).count() >= 1) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            double rate = (elapsed_ms > 0) ? (total_checked * 1000.0 / elapsed_ms) : 0;

            // Calculate progress for sequential search
            std::string progress_str = "";
            if (!args.puzzle_random || force_sequential) {
                if (bits <= 40) {
                    uint64_t total_keys = 1ULL << (bits - 1);
                    double pct = (total_checked * 100.0) / total_keys;
                    progress_str = " | Progress: " + std::to_string(static_cast<int>(pct)) + "%";
                }
            }

            std::cout << "\r[*] Checked: " << std::setw(15) << format_number_human(total_checked)
                      << " | Rate: " << std::setw(10) << format_rate(rate)
                      << progress_str
                      << "     " << std::flush;

            last_update = now;
        }

        // Check if range was exhausted after processing batch
        if (range_exhausted) {
            if (!found) {
                std::cout << "\n[!] Sequential search complete - entire range checked.\n";
                if (have_target_hash) {
                    std::cout << "[!] No match found - verify target hash160 is correct.\n";
                }
            }
            break;
        }
    }

    // If we found the puzzle in auto-progression mode, skip final stats and continue
    if (found && is_multi_puzzle) {
        std::cout << "\n[*] Puzzle #" << current_puzzle << " solved! Continuing to next puzzle...\n";
        return PuzzleStepResult::SolvedExitOrContinue;
    }

    // emit deferred shutdown print/log if we got here via
    // a SIGINT/SIGTERM. Idempotent - no-op if no signal arrived.
    if (g_shutdown.load(std::memory_order_acquire)) {
        emit_shutdown_message_from_main();
    }

    // Final stats
    auto end_time = std::chrono::steady_clock::now();
    double elapsed_sec = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    render_cpu_brute_final_stats(bits, total_checked, elapsed_sec);

    return found ? PuzzleStepResult::SolvedExitOrContinue
                 : PuzzleStepResult::StoppedExitOrContinue;
}

}  // namespace

// T3.11: Dispatcher. Enforces the >128-bit brute reject (R-B7), then
// walks the backend chain in order:
//   GPU brute-force -> CPU brute-force fallback
// The GPU helper returns FallThrough when init fails or the puzzle is
// not eligible for GPU acceleration (no target hash160 / force_sequential);
// every other return is terminal for this puzzle.
PuzzleStepResult run_bruteforce_solve(PuzzleIterContext& ctx) {
    const Arguments& args = ctx.args;
    const int bits = ctx.tgt.bits;
    const int current_puzzle = ctx.current_puzzle;
    const bool is_multi_puzzle = ctx.is_multi_puzzle;

    // ======================================================================
    // R-B7: reject brute force for puzzles > 128 bits.
    // ======================================================================
    // The GPU brute path below tracks position state with two 64-bit
    // limbs (start_lo/hi, end_lo/hi, current_lo/hi, range_size_lo/hi)
    // and pick_next_position() does plain 128-bit arithmetic. For any
    // puzzle whose range exceeds 2^128 (i.e. bits > 128), the high
    // 128 bits of every candidate position are silently zeroed and
    // the scan only ever touches the bottom of the keyspace --
    // mathematically guaranteed to miss the target.
    // The kangaroo path does NOT have this limit (RCKangaroo takes a
    // hex string of arbitrary length for range_bits up to 170). So
    // the right answer here is: if the operator has demoted to brute
    // force AND the puzzle is > 128 bits, refuse rather than burn GPU
    // cycles searching the wrong slice.
    // Auto-progression worklists hit this when they sweep into the
    // 129+ band with no pubkey known. Per the audit comment we go
    // with rejection (honest about the limit) over extending the
    // arithmetic to UInt256; the latter is the v1.5 multi-limb
    // brute-force path, not a v1.4 backport.
    if (!args.puzzle_kangaroo && bits > 128) {
        std::cerr << "\n[!] Brute-force GPU search is not supported for "
                     "puzzles > 128 bits (this is puzzle #"
                  << current_puzzle << " at " << bits << " bits).\n"
                  << "    The GPU position arithmetic is 128-bit; a "
                     "brute scan would silently cover only the bottom "
                     "2^128 of the range.\n"
                  << "    Use kangaroo (default when a pubkey is "
                     "known) or supply --pubkey to enable kangaroo "
                     "on this puzzle.\n"
                  << "    A multi-limb brute-force path is on the "
                     "v1.5 roadmap.\n";
        if (is_multi_puzzle) {
            // Multi-puzzle worklist: don't kill the whole batch; skip
            // this puzzle and continue. The user's intent ("run
            // through all unsolved") is preserved.
            std::cerr << "    Multi-puzzle worklist: skipping puzzle #"
                      << current_puzzle << " and continuing.\n";
            return PuzzleStepResult::SkipPuzzle;
        }
        return PuzzleStepResult::UsageError;
    }

    // GPU brute-force (CUDA optimized). Falls through to CPU when init
    // fails or the puzzle is not GPU-eligible.
    PuzzleStepResult gpu_result = run_bruteforce_gpu(ctx);
    if (gpu_result != PuzzleStepResult::FallThrough) return gpu_result;

    // CPU brute-force fallback: last-resort path. Always terminal.
    return run_bruteforce_cpu(ctx);
}

}  // namespace collider::runtime::detail
