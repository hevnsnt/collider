/**
 * puzzle_solver_kangaroo.cpp - Kangaroo solve-path implementation.
 *
 * Extracted from src/runtime/puzzle_solver.cpp during the v1.4.2
 * structural decomposition. Hosts run_kangaroo_solve(), which dispatches
 * to one of three backends:
 *   - RCKangaroo (COLLIDER_USE_RCKANGAROO, when --use-rckangaroo set)
 *   - MultiGPU Kangaroo (CUDA / Metal MultiGPUKangarooManager)
 *   - CPU Kangaroo (KangarooSolver fallback)
 *
 * v1.4.2 T3.10: the previous monolithic body (~700 lines) was split into
 * the public dispatcher plus three TU-local backend helpers. Each helper
 * may return PuzzleStepResult::FallThrough to indicate "init failed, try
 * the next backend"; only the dispatcher knows the chain order.
 *
 * Hard-preserve invariants (see puzzle_solver_helpers.hpp):
 *   - secure_wipe on every recovered private key (4 sites here).
 *   - secure_open_ofstream for puzzle_found.txt (3 sites here) with
 *     SecureWriteOnFailure::FailHard (T2.4: recovered-key sinks must
 *     never silently fall back to a world-readable file).
 *   - paths::state_dir() for kangaroo_herd_puzzle_<N>.kang.
 *   - --resume-kangaroo InitKangsHost / SaveKangsHost wiring.
 *   - SIGINT save path (request_save_on_stop + save_herd_state on exit).
 */
#include "runtime/puzzle_solver_helpers.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "core/settings_sidecar.hpp"          // TR-5: persist settings
#include "ui/tui/panels/settings_panel.hpp"  // TP-1: live settings poll
#include "ui/tui/tui_app.hpp"  // Phase C: TuiApp setter calls from progress cb

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>

#include "core/crypto_cpu.hpp"
#include "core/kangaroo.hpp"
#include "core/paths.hpp"
#include "core/puzzle_config.hpp"
#include "core/secure_buffer.hpp"
#include "core/secure_write.hpp"
#include "core/session_log.hpp"  // milestone() for puzzle_start /
                                 // puzzle_solved / puzzle_stopped /
                                 // gpu_faulted (kangaroo backends).
#include "core/types.hpp"
#include "gpu/kangaroo_solver_gpu.hpp"
#include "gpu/puzzle_gpu.hpp"
#ifdef COLLIDER_USE_RCKANGAROO
#include "gpu/rckangaroo_wrapper.hpp"
#endif
#include "runtime/balance.hpp"
#include "runtime/runtime_globals.hpp"
#include "ui/banner.hpp"
#include "ui/box_render.hpp"
#include "ui/btc_balance.hpp"
#include "ui/interactive.hpp"

using namespace collider;
using collider::ui::format_rate;

// Forward decls: defined at namespace-global scope in puzzle_solver.cpp.
std::string format_number(uint64_t n);

// T3.12: calculate_optimal_dp_bits used to live both here (as an
// "intentional twin" private mirror) and in puzzle_solver.cpp's
// anonymous namespace. It is now the single header-inline in
// runtime/puzzle_solver_helpers.hpp (already included above). The
// using-declaration below keeps the call sites in this TU readable
// without an explicit collider::runtime::detail:: qualifier.
using collider::runtime::detail::calculate_optimal_dp_bits;

namespace collider::runtime::detail {

namespace {

#ifdef COLLIDER_USE_RCKANGAROO
// Configure rc_kangaroo.dp_bits from --dp-bits (with range check) or the
// auto-heuristic. Extracted from run_kangaroo_rckangaroo for clarity; the
// RCKangaroo accepted window is wider than MultiGPU's, so out-of-range
// user values must be rejected explicitly rather than silently clamped.
// Returns true on success, false (with stderr message) on UsageError.
bool configure_rckangaroo_dp_bits(gpu::RCKangarooManager& rc_kangaroo,
                                  const Arguments& args,
                                  int bits) {
    using gpu::RCKangarooManager;
    if (args.dp_bits > 0) {
        if (args.dp_bits < RCKangarooManager::kMinDpBits ||
            args.dp_bits > RCKangarooManager::kMaxDpBits) {
            std::cerr << "[!] --dp-bits=" << args.dp_bits
                      << " is outside RCKangaroo's accepted range ["
                      << RCKangarooManager::kMinDpBits << ".."
                      << RCKangarooManager::kMaxDpBits << "]. Aborting.\n";
            return false;
        }
        rc_kangaroo.dp_bits = args.dp_bits;
        std::cout << "\033[36m[*] DP Configuration\033[0m\n";
        std::cout << "    dp_bits: " << rc_kangaroo.dp_bits << " (user override)\n";
        std::cout << "    1 in " << format_number(1ULL << rc_kangaroo.dp_bits) << " points marked as DP\n";
    } else {
        // single calculate_optimal_dp_bits() source.
        // num_kangaroos isn't known until rc_kangaroo.init()
        // selects a GPU; pass 0 to take the bits/3 fallback
        // that historically lived inline here.
        rc_kangaroo.dp_bits = calculate_optimal_dp_bits(bits, /*num_kangaroos=*/0);
        std::cout << "\033[36m[*] DP Configuration (auto)\033[0m\n";
        std::cout << "    dp_bits: " << rc_kangaroo.dp_bits << " (optimal for " << bits << "-bit puzzle)\n";
        std::cout << "    1 in " << format_number(1ULL << rc_kangaroo.dp_bits) << " points marked as DP\n";
    }
    return true;
}

// Load Pro bloom filter into rc_kangaroo and install the hit callback that
// appends to bloom_hits.txt. Free builds clear args.bloom_file at config
// merge so this is a no-op; the COLLIDER_PRO guard is defense-in-depth.
void maybe_load_rckangaroo_bloom_filter(gpu::RCKangarooManager& rc_kangaroo,
                                        const Arguments& args) {
#ifdef COLLIDER_PRO
    if (args.bloom_file.empty()) return;
    if (rc_kangaroo.load_bloom_filter(args.bloom_file)) {
        std::cout << "[*] Bloom filter loaded - opportunistic address checking enabled\n";
        // Optional: Set hit callback for real-time notifications
        rc_kangaroo.bloom_hit_callback = [](const gpu::BloomHit& hit) {
            // Owner-only file permissions: bloom_hits.txt
            // logs (hash160, ops_at_hit) pairs for the
            // operator to follow up on; even though no
            // private key is written here, the hash160
            // alone is enough to spot a wallet that the
            // operator is hunting.
            std::ofstream hitlog =
                ::collider::secure_open_ofstream(
                    "bloom_hits.txt", std::ios::app);
            if (hitlog) {
                char h160_hex[41];
                for (int i = 0; i < 20; i++) {
                    snprintf(h160_hex + i*2, 3, "%02x", hit.hash160[i]);
                }
                hitlog << "H160: " << h160_hex << " at ops " << hit.ops_at_hit << "\n";
            }
        };
    } else {
        std::cerr << "[!] WARNING: Failed to load bloom filter: " << args.bloom_file << "\n";
    }
#else
    (void)rc_kangaroo;
    (void)args;
#endif
}

// Wire the rc_kangaroo checkpoint save/resume. Always arms the
// save-on-stop hook so a SIGINT mid-solve drops a checkpoint before
// process exit. Loads an existing checkpoint only when --resume-kangaroo
// was passed AND a file exists for this puzzle. Path convention:
//   ~/.collider/state/kangaroo_herd_puzzle_<N>.kang
// mirroring pool_solver.cpp's kangaroo_herd_<work_id>.kang shape.
void wire_rckangaroo_checkpoint(gpu::RCKangarooManager& rc_kangaroo,
                                const Arguments& args,
                                int current_puzzle,
                                std::string& out_checkpoint_path) {
    auto kangaroo_state_dir = []() -> std::string {
        return collider::paths::state_dir().string();
    };
    out_checkpoint_path =
        kangaroo_state_dir() + "/kangaroo_herd_puzzle_"
        + std::to_string(current_puzzle) + ".kang";
    try {
        std::filesystem::create_directories(kangaroo_state_dir());
    } catch (const std::exception& e) {
        std::cerr << "[!] Failed to create kangaroo state dir: "
                  << e.what()
                  << "\n    (Save-on-stop will be skipped.)\n";
    }

    // Arm the save-on-stop hook BEFORE solve(). The
    // patched RCGpuKang::Execute() reads SaveKangsHost
    // when its inner loop exits.
    rc_kangaroo.request_save_on_stop();

    // Load existing checkpoint if --resume-kangaroo and
    // a file exists for this puzzle.
    if (args.resume_kangaroo) {
        std::error_code ec;
        if (std::filesystem::exists(out_checkpoint_path, ec) && !ec) {
            if (rc_kangaroo.load_herd_state(out_checkpoint_path)) {
                std::cout << "[*] Resumed kangaroo herd from "
                          << out_checkpoint_path << "\n";
            } else {
                std::cerr << "[!] load_herd_state(" << out_checkpoint_path
                          << ") failed; starting with a fresh herd.\n";
            }
        } else {
            std::cout << "[*] --resume-kangaroo set but no checkpoint "
                         "exists yet at " << out_checkpoint_path
                      << "; starting fresh.\n";
        }
    }
}

// RCKangaroo backend. Returns FallThrough when init reports 0 GPUs (the
// caller then tries MultiGPU Kangaroo). All other returns are terminal
// for this puzzle.
PuzzleStepResult run_kangaroo_rckangaroo(PuzzleIterContext& ctx) {
    Arguments& args = ctx.args;
    const PuzzleInfo* puzzle = ctx.puzzle;
    const PuzzleTarget& tgt = ctx.tgt;
    const int bits = tgt.bits;
    const int current_puzzle = ctx.current_puzzle;
    UInt256& range_start = ctx.tgt.range_start;
    const std::string& target_address = tgt.target_address;

    std::cout << "[*] Using RCKangaroo (RetiredCoder's high-performance solver)\n";

    // v1.5.x: --kangaroos N is informational for RCKangaroo. The kangaroo
    // count is derived from the kernel grid in third_party/RCKangaroo
    // (mpCnt * BLOCK_SIZE * PNT_GROUP_CNT, all compile-time except mpCnt
    // which is the runtime GPU MP count). Changing K would require
    // modifying GPLv3 kernel source. Surface the no-op so the operator
    // sees their flag was acknowledged but won't take effect here; the
    // actual count emitted by RCKangaroo at init is "GPU X: allocated
    // ... kangaroos" a few lines below this.
    if (args.num_kangaroos > 0) {
        std::cout << "[*] --kangaroos=" << args.num_kangaroos
                  << " requested but RCKangaroo's K is kernel-grid-driven; "
                     "the value below is the actual per-GPU count\n";
    }

    gpu::RCKangarooManager rc_kangaroo;
    rc_kangaroo.range_bits = bits;

    // Set DP bits. RCKangaroo's documented acceptance window
    // is [kMinDpBits, kMaxDpBits] = [14, 60]. Reject explicitly
    // out-of-range user values rather than silently clamping;
    // a misconfigured CLI invocation should surface a clear
    // error so the operator can correct intent.
    if (!configure_rckangaroo_dp_bits(rc_kangaroo, args, bits)) {
        return PuzzleStepResult::UsageError;
    }

    // Initialize GPUs
    int num_gpus = rc_kangaroo.init(args.gpu_ids);
    if (num_gpus <= 0) {
        std::cout << "[!] RCKangaroo initialization failed, falling back to standard solver\n";
        return PuzzleStepResult::FallThrough;
    }

    // Load bloom filter if specified (Pro only).
    maybe_load_rckangaroo_bloom_filter(rc_kangaroo, args);

    // Set target public key. v1.4.1: prefer the --pubkey
    // CLI / config override when set; fall back to the
    // bundled puzzle->public_key_hex.
    std::string pubkey_hex = !args.puzzle_pubkey.empty()
                                 ? args.puzzle_pubkey
                                 : puzzle->public_key_hex;
    if (!rc_kangaroo.set_target_pubkey(pubkey_hex)) {
        std::cerr << "[!] ERROR: Failed to set target pubkey\n";
        return PuzzleStepResult::FatalError;
    }

    // Set start offset (range_start)
    char start_hex[100];
    snprintf(start_hex, sizeof(start_hex), "%llx%016llx%016llx%016llx",
             (unsigned long long)range_start.parts[3],
             (unsigned long long)range_start.parts[2],
             (unsigned long long)range_start.parts[1],
             (unsigned long long)range_start.parts[0]);
    rc_kangaroo.set_start_offset(start_hex);

    // Calculate expected operations for ETA
    double expected_ops_bits = (bits - 1) / 2.0 + 1;
    uint64_t expected_ops = (expected_ops_bits < 63) ? (1ULL << (int)expected_ops_bits) : 0;

    // Progress callback. Updates both the legacy cout-based progress line
    // (visible in piped / non-TUI runs) AND the unified TUI panels when
    // ctx.tui_app is non-null. The TUI panels (header throughput, status
    // phase name, performance keys/s history) are the authoritative live
    // surface; the cout line stays so headless invocations (CI, --logs,
    // SSH-no-tty) still see textual progress.
    auto* tui_for_rc = static_cast<::collider::ui::tui::TuiApp*>(ctx.tui_app);
    rc_kangaroo.progress_callback = [&, expected_ops, expected_ops_bits, tui_for_rc](uint64_t ops, uint64_t dp_count, int speed) -> bool {
        if (g_shutdown) return false;

        // Calculate progress percentage and ETA
        double progress_pct = (expected_ops > 0) ? (100.0 * ops / expected_ops) : 0;
        if (progress_pct > 100.0) progress_pct = 100.0;

        std::string eta_str = "calculating...";
        if (speed > 0 && expected_ops > ops) {
            double remaining_ops = expected_ops - ops;
            double remaining_secs = remaining_ops / (speed * 1e6);
            eta_str = ui::ProfessionalUI::format_duration(remaining_secs);
        }

        // Professional single-line progress
        std::cout << "\r\033[K";
        std::cout << "\033[36mProgress:\033[0m "
                  << std::fixed << std::setprecision(4) << progress_pct << "% | "
                  << "\033[33mOps:\033[0m " << ui::ProfessionalUI::format_number_short(ops) << " | "
                  << "\033[32mSpeed:\033[0m " << ui::format_rate(static_cast<double>(speed) * 1e6) << " | "
                  << "\033[35mDPs:\033[0m " << ui::ProfessionalUI::format_number_short(dp_count) << " | "
                  << "\033[34mETA:\033[0m " << eta_str
                  << "  " << std::flush;

        // Phase C TUI push: keys/s is speed (MKeys/s) * 1e6; chunk is
        // ops / expected_ops scaled to a (cur, total) pair the panel
        // can render as a bar. Cap total at INT_MAX for safety.
        if (tui_for_rc) {
            // TP-1 settings live-apply (RCKangaroo branch): poll
            // snapshot_and_clear; theme + refresh apply instantly,
            // backend change exits solve() so the dispatcher rebuilds.
            if (auto* st = tui_for_rc->settings_state()) {
                auto snap = ::collider::ui::tui::panels::snapshot_and_clear(*st);
                const bool any_change =
                    snap.dirty.num_kangaroos || snap.dirty.batch_size ||
                    snap.dirty.dp_bits || snap.dirty.refresh_hz ||
                    snap.dirty.theme || snap.dirty.verbose ||
                    snap.dirty.backend_kind || snap.dirty.solver;
                if (snap.dirty.refresh_hz) {
                    tui_for_rc->set_refresh_hz(snap.values.refresh_hz);
                }
                if (any_change) {
                    ::collider::settings_sidecar::save(snap.values);  // TR-5
                }
                if (snap.restart_requested) {
                    return false;  // rckangaroo exits; outer loop re-init
                }
            }
            tui_for_rc->set_keys_per_sec_current(static_cast<double>(speed) * 1e6);
            // Mode-aware overlay: puzzle ops counter for the status
            // panel's KANGAROO OPS row. Pre-overlay this data was lost
            // behind the alt-screen; the operator-visible text "Ops:"
            // line below us is invisible while the TUI is up.
            ::collider::ui::tui::ChallengeInfo ci;
            ci.puzzle_number = current_puzzle;
            ci.puzzle_bits   = bits;
            ci.ops_completed = ops;
            ci.expected_ops  = expected_ops;
            ci.dps_found     = dp_count;
            ci.backend_name  = "RCKangaroo";
            tui_for_rc->set_challenge_info(ci);
            const uint64_t exp_clip =
                expected_ops > 0 ? expected_ops : (ops + 1);
            const uint64_t cur_clip =
                ops > exp_clip ? exp_clip : ops;
            const int cur_chunk =
                cur_clip > static_cast<uint64_t>(INT_MAX)
                    ? INT_MAX
                    : static_cast<int>(cur_clip);
            const int tot_chunks =
                exp_clip > static_cast<uint64_t>(INT_MAX)
                    ? INT_MAX
                    : static_cast<int>(exp_clip);
            tui_for_rc->set_chunk_progress(cur_chunk, tot_chunks);
            if (tui_for_rc->requested_quit() && !g_shutdown) {
                g_shutdown.store(true);
                return false;
            }
        }

        return !g_shutdown;
    };

    // Display professional search header
    std::cout << "\n";
    ui::ProfessionalUI::render_section("RCKangaroo High-Performance Search");
    ui::ProfessionalUI::render_kv("Method", "RCKangaroo (K=1.15 optimal)");
    ui::ProfessionalUI::render_kv("GPUs", std::to_string(num_gpus) + " detected");
    ui::ProfessionalUI::render_kv("Range", std::to_string(bits) + " bits");
    ui::ProfessionalUI::render_kv("DP Bits", std::to_string(rc_kangaroo.dp_bits));
    ui::ProfessionalUI::render_kv("Expected Ops", "~2^" + std::to_string((int)expected_ops_bits));
    std::cout << "\n";
    ui::ProfessionalUI::render_footer("Press Ctrl+C to stop and save checkpoint");

    // Wire checkpoint save/resume (always arms save-on-stop; load is
    // gated on --resume-kangaroo).
    std::string checkpoint_path;
    wire_rckangaroo_checkpoint(rc_kangaroo, args, current_puzzle, checkpoint_path);

    // Session log: record entry into the RCKangaroo backend. dp_bits is
    // finalized above (configure_rckangaroo_dp_bits); num_kangaroos is
    // backend-internal to RCKangaroo and not exposed on this surface, so
    // we log num_gpus as the closest external proxy. No key material is
    // observable at this site.
    {
        std::ostringstream d;
        d << "puzzle=" << current_puzzle
          << " backend=RCKangaroo"
          << " puzzle_bits=" << bits
          << " dp_bits=" << rc_kangaroo.dp_bits
          << " num_gpus=" << num_gpus;
        ::collider::log::milestone("puzzle_start", d.str());

        ::collider::log::SessionState s;
        s.mode = "puzzle";
        s.puzzle_number = current_puzzle;
        s.puzzle_algorithm = "RCKangaroo";
        ::collider::log::update_session_state(s);
    }

    auto start_time = std::chrono::steady_clock::now();
    auto rc_result = rc_kangaroo.solve();
    auto end_time = std::chrono::steady_clock::now();

    // After solve(): flush the save buffer to disk. We
    // do this unconditionally on stop (whether or not
    // a solution was found) so a SIGINT during a long
    // search lands a usable checkpoint. If the solve
    // found a solution, the checkpoint is harmless;
    // operators clean it up manually or it is overwritten
    // by the next puzzle's run.
    if (rc_kangaroo.save_herd_state(checkpoint_path)) {
        std::cout << "[*] Saved kangaroo herd to "
                  << checkpoint_path << "\n";
    }
    double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

    if (!rc_result.found) {
        std::cout << "\n\n[!] RCKangaroo search stopped after "
                  << format_number(rc_result.total_ops) << " ops\n";
        std::cout << "    Duration: " << std::fixed << std::setprecision(1) << total_seconds << " seconds\n";
        if (rc_result.error_count > 0) {
            std::cout << "    Errors: " << rc_result.error_count << "\n";
            // Session log: surface backend-reported errors as a
            // gpu_faulted milestone. RCKangaroo does not expose
            // per-device GpuPhase, but a non-zero error_count from
            // its inner loop is the closest signal we have that a
            // GPU went off the rails during this solve.
            std::ostringstream d;
            d << "backend=RCKangaroo"
              << " puzzle=" << current_puzzle
              << " error_count=" << rc_result.error_count;
            ::collider::log::milestone("gpu_faulted", d.str());
        }
        // Session log: terminal stop without a solve. Reason carries
        // shutdown vs natural-exit context for downstream tooling.
        const char* reason = g_shutdown.load(std::memory_order_acquire)
                                 ? "user_interrupt"
                                 : "search_stopped";
        std::ostringstream sd;
        sd << "backend=RCKangaroo"
           << " puzzle=" << current_puzzle
           << " reason=" << reason
           << " total_ops=" << rc_result.total_ops;
        ::collider::log::milestone("puzzle_stopped", sd.str());
        return PuzzleStepResult::StoppedExitOrContinue;
    }

    std::string key_hex = gpu::private_key_to_hex(rc_result.private_key);

    std::cout << "\n\n";
    ui::ProfessionalUI::render_found_banner("PUZZLE #" + std::to_string(current_puzzle) + " SOLVED!");
    std::cout << "\n";
    ui::ProfessionalUI::render_kv("Private Key", "0x" + key_hex);
    ui::ProfessionalUI::render_kv("Address", target_address);
    ui::ProfessionalUI::render_kv("Balance",
        ::collider::ui::format_balance(
            ::collider::ui::fetch_balance_btc(target_address)));
    ui::ProfessionalUI::render_kv("Algorithm", "RCKangaroo (K=" + std::to_string(rc_result.k_value).substr(0,5) + ")");
    ui::ProfessionalUI::render_kv("Duration", ui::ProfessionalUI::format_duration(total_seconds));
    ui::ProfessionalUI::render_kv("Total Ops", format_number(rc_result.total_ops));
    std::cout << "\n";

    // Save to file
    // Owner-only file permissions: puzzle_found.txt
    // contains the recovered private key in plain hex.
    // FailHard: recovered private key sink, world-readable
    // file is worse than no file.
    std::ofstream found_file =
        ::collider::secure_open_ofstream(
            "puzzle_found.txt", std::ios::app,
            ::collider::SecureWriteOnFailure::FailHard);
    if (found_file) {
        found_file << "================================================================================\n";
        found_file << "                    PUZZLE SOLVED (RCKangaroo)\n";
        found_file << "================================================================================\n";
        found_file << "Puzzle:       #" << current_puzzle << " (" << bits << "-bit)\n";
        found_file << "Private Key:  0x" << key_hex << "\n";
        found_file << "Address:      " << target_address << "\n";
        found_file << "Algorithm:    RCKangaroo (K=" << rc_result.k_value << ")\n";
        found_file << "Duration:     " << std::fixed << std::setprecision(3) << total_seconds << " seconds\n";
        found_file << "================================================================================\n\n";
    }
    // Session log: solve event. ADDRESS ONLY by policy. The recovered
    // private key is persisted to puzzle_found.txt via secure_open_ofstream
    // (FailHard, owner-only DACL); duplicating the key in the session log
    // would create a second on-disk copy under a less restrictive
    // permission set. The mirror brute-force path in
    // puzzle_solver_bruteforce.cpp uses the same redaction rule.
    {
        std::ostringstream d;
        d << "puzzle=" << current_puzzle
          << " backend=RCKangaroo"
          << " address=" << target_address
          << " total_ops=" << rc_result.total_ops;
        ::collider::log::milestone("puzzle_solved", d.str());
    }

    // Wipe the recovered private key from heap + the
    // four-limb std::array<uint64_t, 4> in the result
    // struct. The string's backing storage may be SBO'd
    // inside key_hex itself, so cleanse the chars in
    // place before the local goes out of scope.
    ::collider::secure_wipe(
        rc_result.private_key.data(),
        rc_result.private_key.size() * sizeof(uint64_t));
    if (!key_hex.empty()) {
        ::collider::secure_wipe(&key_hex[0], key_hex.size());
    }
    // Honor --all-unsolved / --auto-next: solved is a positive
    // result, dispatcher decides whether to continue or exit.
    return PuzzleStepResult::SolvedExitOrContinue;
}
#endif  // COLLIDER_USE_RCKANGAROO

// MultiGPU Kangaroo backend (CUDA / Metal). Returns FallThrough when
// init reports failure (caller then tries CPU Kangaroo).
PuzzleStepResult run_kangaroo_multigpu(PuzzleIterContext& ctx,
                                       const cpu::uint256_t& target_pubkey_x,
                                       const cpu::uint256_t& target_pubkey_y) {
    Arguments& args = ctx.args;
    const PuzzleTarget& tgt = ctx.tgt;
    const int bits = tgt.bits;
    const int current_puzzle = ctx.current_puzzle;
    UInt256& range_start = ctx.tgt.range_start;
    UInt256& range_end = ctx.tgt.range_end;
    const std::string& target_address = tgt.target_address;

    gpu::MultiGPUKangarooManager gpu_kangaroo;
    int dp_bits_to_use = 20;  // Default, will be set properly below

    // v1.5.x: honor --kangaroos N (Arguments::num_kangaroos) as the
    // per-GPU kangaroo count. Default of 0 leaves the manager's built-in
    // 1<<18 in place. MultiGPUKangarooManager exposes the field as a
    // public member; the assignment must happen BEFORE init() so the
    // value flows into device buffer sizing.
    if (args.num_kangaroos > 0) {
        std::cout << "[*] --kangaroos=" << args.num_kangaroos
                  << " per GPU (overriding default "
                  << gpu_kangaroo.num_kangaroos_per_gpu << ")\n";
        gpu_kangaroo.num_kangaroos_per_gpu = args.num_kangaroos;
    }

    // Initialize with all available GPUs (or specific ones from args.gpu_ids if set)
    if (!gpu_kangaroo.init(args.gpu_ids)) {
        return PuzzleStepResult::FallThrough;
    }
    gpu_kangaroo.set_range(range_start, range_end);
    gpu_kangaroo.set_target_pubkey(target_pubkey_x, target_pubkey_y);

    // Calculate and set optimal dp_bits
    int num_gpus = gpu_kangaroo.num_gpus();
    int total_kangaroos = gpu_kangaroo.num_kangaroos_per_gpu * num_gpus;

    if (args.dp_bits > 0) {
        // User specified dp_bits manually. The MultiGPU
        // backend's documented window is [kMinDpBits,
        // kMaxDpBits] = [16, 28]. Out-of-range values used
        // to be silently clamped, hiding configuration
        // mistakes. Reject explicitly so the operator sees
        // and corrects the intent.
        using gpu::MultiGPUKangarooManager;
        if (args.dp_bits < MultiGPUKangarooManager::kMinDpBits ||
            args.dp_bits > MultiGPUKangarooManager::kMaxDpBits) {
            std::cerr << "[!] --dp-bits=" << args.dp_bits
                      << " is outside MultiGPU Kangaroo's accepted range ["
                      << MultiGPUKangarooManager::kMinDpBits << ".."
                      << MultiGPUKangarooManager::kMaxDpBits
                      << "]. Use --use-rckangaroo for the wider "
                      << "[14, 60] range. Aborting.\n";
            return PuzzleStepResult::UsageError;
        }
        dp_bits_to_use = args.dp_bits;
        std::cout << "\033[36m[*] DP Configuration\033[0m\n";
        std::cout << "    dp_bits: " << dp_bits_to_use << " (user override)\n";
        std::cout << "    1 in " << format_number(1ULL << dp_bits_to_use) << " points marked as DP\n";
    } else {
        // Auto-calculate optimal dp_bits
        dp_bits_to_use = calculate_optimal_dp_bits(bits, total_kangaroos);
        std::cout << "\033[36m[*] DP Configuration (auto)\033[0m\n";
        std::cout << "    dp_bits: " << dp_bits_to_use << " (optimal for " << bits << "-bit puzzle)\n";
        std::cout << "    Kangaroos: " << format_number(total_kangaroos) << " across " << num_gpus << " GPU(s)\n";
        std::cout << "    1 in " << format_number(1ULL << dp_bits_to_use) << " points marked as DP\n";
    }

    gpu_kangaroo.dp_bits = dp_bits_to_use;
    gpu_kangaroo.debug_mode = args.debug;

    // Session log: record entry into the MultiGPU Kangaroo backend.
    // total_kangaroos is known here (num_kangaroos_per_gpu * num_gpus).
    {
        std::ostringstream d;
        d << "puzzle=" << current_puzzle
          << " backend=MultiGPU"
          << " puzzle_bits=" << bits
          << " dp_bits=" << dp_bits_to_use
          << " num_gpus=" << num_gpus
          << " num_kangaroos=" << total_kangaroos;
        ::collider::log::milestone("puzzle_start", d.str());

        ::collider::log::SessionState s;
        s.mode = "puzzle";
        s.puzzle_number = current_puzzle;
        s.puzzle_algorithm = "MultiGPU";
        ::collider::log::update_session_state(s);
    }

    auto start_time = std::chrono::steady_clock::now();

    // Calculate expected operations for this puzzle
    double expected_ops_bits = (bits - 1) / 2.0 + 1;  // sqrt(2^(bits-1)) ~= 2^((bits-1)/2)
    uint64_t expected_ops = (expected_ops_bits < 63) ? (1ULL << (int)expected_ops_bits) : 0;

    auto* tui_for_mg = static_cast<::collider::ui::tui::TuiApp*>(ctx.tui_app);
    gpu_kangaroo.progress_callback = [&, expected_ops, expected_ops_bits, tui_for_mg](uint64_t steps, uint64_t dp_count, double rate) -> bool {
        if (g_shutdown) return false;
        // Phase C TUI push (mirrors the RCKangaroo path above; rate is
        // already keys/s, no MKey scaling needed).
        if (tui_for_mg) {
            // TP-1 settings live-apply (MultiGPU branch).
            if (auto* st = tui_for_mg->settings_state()) {
                auto snap = ::collider::ui::tui::panels::snapshot_and_clear(*st);
                const bool any_change =
                    snap.dirty.num_kangaroos || snap.dirty.batch_size ||
                    snap.dirty.dp_bits || snap.dirty.refresh_hz ||
                    snap.dirty.theme || snap.dirty.verbose ||
                    snap.dirty.backend_kind || snap.dirty.solver;
                if (snap.dirty.refresh_hz) {
                    tui_for_mg->set_refresh_hz(snap.values.refresh_hz);
                }
                if (any_change) {
                    ::collider::settings_sidecar::save(snap.values);  // TR-5
                }
                if (snap.restart_requested) return false;
            }
            tui_for_mg->set_keys_per_sec_current(rate);
            // Mode-aware overlay: same shape as the RCKangaroo branch
            // above, but the per-step counter is named `steps` here
            // (semantically equivalent to ops for the panel display).
            ::collider::ui::tui::ChallengeInfo ci;
            ci.puzzle_number = current_puzzle;
            ci.puzzle_bits   = bits;
            ci.ops_completed = steps;
            ci.expected_ops  = expected_ops;
            ci.dps_found     = dp_count;
            ci.backend_name  = "MultiGPU";
            tui_for_mg->set_challenge_info(ci);
            const uint64_t exp_clip =
                expected_ops > 0 ? expected_ops : (steps + 1);
            const uint64_t cur_clip =
                steps > exp_clip ? exp_clip : steps;
            const int cur_chunk =
                cur_clip > static_cast<uint64_t>(INT_MAX)
                    ? INT_MAX
                    : static_cast<int>(cur_clip);
            const int tot_chunks =
                exp_clip > static_cast<uint64_t>(INT_MAX)
                    ? INT_MAX
                    : static_cast<int>(exp_clip);
            tui_for_mg->set_chunk_progress(cur_chunk, tot_chunks);
            if (tui_for_mg->requested_quit() && !g_shutdown) {
                g_shutdown.store(true);
                return false;
            }
        }

        // Calculate expected DPs and progress
        double expected_dps = static_cast<double>(steps) / (1ULL << dp_bits_to_use);
        double progress_pct = (expected_ops > 0) ? (100.0 * steps / expected_ops) : 0;
        if (progress_pct > 100.0) progress_pct = 100.0;

        // Calculate ETA based on current rate
        std::string eta_str = "calculating...";
        if (rate > 0 && expected_ops > steps) {
            double remaining_ops = expected_ops - steps;
            double remaining_secs = remaining_ops / rate;
            eta_str = ui::ProfessionalUI::format_duration(remaining_secs);
        }

        // Professional single-line progress (updates in place)
        std::cout << "\r\033[K";  // Clear line
        std::cout << "\033[36mProgress:\033[0m "
                  << std::fixed << std::setprecision(4) << progress_pct << "% | "
                  << "\033[33mOps:\033[0m " << ui::ProfessionalUI::format_number_short(steps) << " | "
                  << "\033[32mSpeed:\033[0m " << format_rate(rate) << " | "
                  << "\033[35mDPs:\033[0m " << format_number(dp_count)
                  << " (exp ~" << static_cast<int>(expected_dps) << ") | "
                  << "\033[34mETA:\033[0m " << eta_str
                  << "  " << std::flush;

        return !g_shutdown;
    };

    // Display professional header for search
    std::cout << "\n";
    ui::ProfessionalUI::render_section("GPU Kangaroo Search");
    ui::ProfessionalUI::render_kv("Method", "Pollard's Kangaroo (K=1.15)");
    ui::ProfessionalUI::render_kv("GPUs", std::to_string(num_gpus) + "x " + ctx.gpu_info.gpu_names);
    ui::ProfessionalUI::render_kv("Range", std::to_string(bits) + " bits");
    ui::ProfessionalUI::render_kv("DP Bits", std::to_string(dp_bits_to_use));
    ui::ProfessionalUI::render_kv("Expected Ops", "~2^" + std::to_string((int)expected_ops_bits));
    std::cout << "\n";
    ui::ProfessionalUI::render_footer("Press Ctrl+C to stop and save checkpoint");

    auto gpu_result = gpu_kangaroo.solve();

    auto end_time = std::chrono::steady_clock::now();
    double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

    if (!gpu_result.found) {
        std::cout << "\n\n[!] GPU Kangaroo search stopped after "
                  << format_number(gpu_result.total_steps) << " steps\n";
        std::cout << "    Duration: " << std::fixed << std::setprecision(1) << total_seconds << " seconds\n";
        // Session log: terminal stop without a solve. Reason carries
        // shutdown vs natural-exit context for downstream tooling.
        const char* reason = g_shutdown.load(std::memory_order_acquire)
                                 ? "user_interrupt"
                                 : "search_stopped";
        std::ostringstream sd;
        sd << "backend=MultiGPU"
           << " puzzle=" << current_puzzle
           << " reason=" << reason
           << " total_steps=" << gpu_result.total_steps;
        ::collider::log::milestone("puzzle_stopped", sd.str());
        return PuzzleStepResult::StoppedExitOrContinue;
    }

    // Format key
    char key_hex[67];
    if (gpu_result.private_key.d[3] > 0 || gpu_result.private_key.d[2] > 0) {
        snprintf(key_hex, sizeof(key_hex), "0x%llx%016llx%016llx%016llx",
                 (unsigned long long)gpu_result.private_key.d[3],
                 (unsigned long long)gpu_result.private_key.d[2],
                 (unsigned long long)gpu_result.private_key.d[1],
                 (unsigned long long)gpu_result.private_key.d[0]);
    } else if (gpu_result.private_key.d[1] > 0) {
        snprintf(key_hex, sizeof(key_hex), "0x%llx%016llx",
                 (unsigned long long)gpu_result.private_key.d[1],
                 (unsigned long long)gpu_result.private_key.d[0]);
    } else {
        snprintf(key_hex, sizeof(key_hex), "0x%llx",
                 (unsigned long long)gpu_result.private_key.d[0]);
    }

    // Flip the dashboard phase name to the operator-visible SOLVED
    // state BEFORE the cout banner runs. The cout below still
    // executes (captured to boot log; puzzle_found.txt is the
    // durable record) but the dashboard now reflects the solve.
    if (auto* tui_solved =
            static_cast<::collider::ui::tui::TuiApp*>(ctx.tui_app)) {
        std::ostringstream solved_phase;
        solved_phase << "Puzzle #" << current_puzzle
                     << " SOLVED (GPU Kangaroo)";
        tui_solved->set_current_phase_name(solved_phase.str());
    }
    {
        namespace boxui = ::collider::ui::box;
        std::cout << "\n\n";
        std::cout << boxui::ansi::BRIGHT_GREEN;
        boxui::top(std::cout);
        boxui::centered(std::cout, "PUZZLE SOLVED! (GPU Kangaroo Algorithm)");
        boxui::top(std::cout);
        std::cout << boxui::ansi::RESET;

        std::ostringstream pz; pz << "#" << current_puzzle << " (" << bits << "-bit)";
        std::ostringstream dur; dur << std::fixed << std::setprecision(3) << total_seconds << " sec";
        boxui::kv(std::cout, "Puzzle",      pz.str(),                 boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Private Key", key_hex,                  boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Address",     target_address,           boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Balance",
                  ::collider::ui::format_balance(
                      ::collider::ui::fetch_balance_btc(target_address)),
                  boxui::ansi::BRIGHT_MAGENTA);
        boxui::sep(std::cout);
        boxui::kv(std::cout, "Duration",    dur.str(),                            boxui::ansi::BRIGHT_CYAN);
        boxui::kv(std::cout, "Total Steps", format_number(gpu_result.total_steps), boxui::ansi::BRIGHT_CYAN);
        boxui::bottom(std::cout);
        std::cout << "\n";
    }

    // Save to file
    // Owner-only permissions; see secure_open_ofstream notes.
    // FailHard: recovered private key sink, world-readable file
    // is worse than no file.
    std::ofstream found_file =
        ::collider::secure_open_ofstream(
            "puzzle_found.txt", std::ios::app,
            ::collider::SecureWriteOnFailure::FailHard);
    if (found_file) {
        found_file << "================================================================================\n";
        found_file << "                    PUZZLE SOLVED (GPU Kangaroo)\n";
        found_file << "================================================================================\n";
        found_file << "Puzzle:       #" << current_puzzle << " (" << bits << "-bit)\n";
        found_file << "Private Key:  " << key_hex << "\n";
        found_file << "Address:      " << target_address << "\n";
        found_file << "Algorithm:    GPU Kangaroo\n";
        found_file << "Duration:     " << std::fixed << std::setprecision(3) << total_seconds << " seconds\n";
        found_file << "================================================================================\n\n";
    }
    // Session log: solve event. ADDRESS ONLY by policy. The recovered
    // private key is persisted to puzzle_found.txt (FailHard, owner-only);
    // duplicating it in the session log would widen the leak surface.
    {
        std::ostringstream d;
        d << "puzzle=" << current_puzzle
          << " backend=MultiGPU"
          << " address=" << target_address
          << " total_steps=" << gpu_result.total_steps;
        ::collider::log::milestone("puzzle_solved", d.str());
    }

    // Wipe the recovered key bytes once they have hit disk.
    ::collider::secure_wipe(gpu_result.private_key.d,
                            sizeof(gpu_result.private_key.d));
    ::collider::secure_wipe(key_hex, sizeof(key_hex));
    // Honor --all-unsolved / --auto-next on the GPU Kangaroo path
    // the same as the CPU Kangaroo path does.
    return PuzzleStepResult::SolvedExitOrContinue;
}

// CPU Kangaroo backend. Last-resort fallback when neither GPU backend
// initialized. Returns SolvedExitOrContinue or StoppedExitOrContinue.
PuzzleStepResult run_kangaroo_cpu(PuzzleIterContext& ctx) {
    Arguments& args = ctx.args;
    const PuzzleTarget& tgt = ctx.tgt;
    const int bits = tgt.bits;
    const int current_puzzle = ctx.current_puzzle;
    const bool is_multi_puzzle = ctx.is_multi_puzzle;
    UInt256& range_start = ctx.tgt.range_start;
    UInt256& range_end = ctx.tgt.range_end;
    const std::string& target_address = tgt.target_address;
    const std::array<uint8_t, 20>& target_hash160 = tgt.target_hash160;
    const bool have_target_hash = tgt.have_target_hash;

    std::cout << "[*] Falling back to CPU Kangaroo...\n";

    KangarooSolver solver;
    solver.set_range(range_start, range_end);

    // Configure dp_bits for CPU solver
    // CPU solver uses fewer kangaroos, so adjust calculation
    int cpu_kangaroos = 2;  // CPU uses 1 tame + 1 wild kangaroo
    int dp_bits_to_use = 20;

    if (args.dp_bits > 0) {
        dp_bits_to_use = std::max(16, std::min(28, args.dp_bits));
        std::cout << "[*] Using dp_bits=" << dp_bits_to_use << " (user-specified)\n";
    } else {
        dp_bits_to_use = calculate_optimal_dp_bits(bits, cpu_kangaroos);
        std::cout << "[*] Using dp_bits=" << dp_bits_to_use
                  << " (auto-calculated for CPU with " << cpu_kangaroos << " kangaroos)\n";
    }
    solver.dp_bits = dp_bits_to_use;

    if (have_target_hash) {
        solver.set_target_h160(target_hash160);
    }

    solver.set_progress_callback([&](uint64_t tame_steps, uint64_t wild_steps, uint64_t dp_count, double rate) -> bool {
        if (g_shutdown) return false;

        uint64_t total = tame_steps + wild_steps;

        std::cout << "\r[*] Kangaroo: " << format_number(total) << " steps, "
                  << format_number(dp_count) << " DPs, "
                  << format_rate(rate) << "        " << std::flush;

        return !g_shutdown;
    });

    std::cout << "[*] Starting CPU kangaroo search...\n";
    std::cout << "    Press Ctrl+C to stop\n\n";

    // Session log: record entry into the CPU Kangaroo backend.
    // num_kangaroos is the cpu_kangaroos constant defined above (2:
    // 1 tame + 1 wild).
    {
        std::ostringstream d;
        d << "puzzle=" << current_puzzle
          << " backend=CPU"
          << " puzzle_bits=" << bits
          << " dp_bits=" << dp_bits_to_use
          << " num_kangaroos=" << cpu_kangaroos;
        ::collider::log::milestone("puzzle_start", d.str());

        ::collider::log::SessionState s;
        s.mode = "puzzle";
        s.puzzle_number = current_puzzle;
        s.puzzle_algorithm = "CPU";
        ::collider::log::update_session_state(s);
    }

    auto start_time = std::chrono::steady_clock::now();
    auto result = solver.solve();

    auto end_time = std::chrono::steady_clock::now();
    double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

    if (!result.found) {
        std::cout << "\n\n[!] Kangaroo search stopped after "
                  << format_number(result.tame_steps + result.wild_steps) << " steps\n";
        std::cout << "    Duration: " << std::fixed << std::setprecision(1) << total_seconds << " seconds\n";
        // Session log: terminal stop without a solve.
        const char* reason = g_shutdown.load(std::memory_order_acquire)
                                 ? "user_interrupt"
                                 : "search_stopped";
        std::ostringstream sd;
        sd << "backend=CPU"
           << " puzzle=" << current_puzzle
           << " reason=" << reason
           << " total_steps=" << (result.tame_steps + result.wild_steps);
        ::collider::log::milestone("puzzle_stopped", sd.str());
        // Stopped without a solve: dispatcher decides whether to skip to
        // the next puzzle (--all-unsolved) or exit cleanly (single-puzzle).
        return PuzzleStepResult::StoppedExitOrContinue;
    }

    // Format key as hex
    char key_hex[67];
    if (result.private_key.d[3] > 0 || result.private_key.d[2] > 0) {
        snprintf(key_hex, sizeof(key_hex), "0x%llx%016llx%016llx%016llx",
                 (unsigned long long)result.private_key.d[3],
                 (unsigned long long)result.private_key.d[2],
                 (unsigned long long)result.private_key.d[1],
                 (unsigned long long)result.private_key.d[0]);
    } else if (result.private_key.d[1] > 0) {
        snprintf(key_hex, sizeof(key_hex), "0x%llx%016llx",
                 (unsigned long long)result.private_key.d[1],
                 (unsigned long long)result.private_key.d[0]);
    } else {
        snprintf(key_hex, sizeof(key_hex), "0x%llx",
                 (unsigned long long)result.private_key.d[0]);
    }

    // Get solve time
    auto solve_time = std::chrono::system_clock::now();
    auto solve_time_t = std::chrono::system_clock::to_time_t(solve_time);
    char timestamp[64];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&solve_time_t));

    // Flip dashboard phase to SOLVED (CPU/multi-GPU kangaroo path).
    if (auto* tui_solved =
            static_cast<::collider::ui::tui::TuiApp*>(ctx.tui_app)) {
        std::ostringstream solved_phase;
        solved_phase << "Puzzle #" << current_puzzle << " SOLVED (Kangaroo)";
        tui_solved->set_current_phase_name(solved_phase.str());
    }
    {
        namespace boxui = ::collider::ui::box;
        std::cout << "\n\n";
        std::cout << boxui::ansi::BRIGHT_GREEN;
        boxui::top(std::cout);
        boxui::centered(std::cout, "PUZZLE SOLVED! (Kangaroo Algorithm)");
        boxui::top(std::cout);
        std::cout << boxui::ansi::RESET;

        std::ostringstream pz; pz << "#" << current_puzzle << " (" << bits << "-bit)";
        std::ostringstream dur; dur << std::fixed << std::setprecision(3) << total_seconds << " sec";
        boxui::kv(std::cout, "Puzzle",      pz.str(),                       boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Private Key", key_hex,                        boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Address",     target_address,                 boxui::ansi::BRIGHT_YELLOW);
        boxui::kv(std::cout, "Balance",
                  ::collider::ui::format_balance(
                      ::collider::ui::fetch_balance_btc(target_address)),
                  boxui::ansi::BRIGHT_MAGENTA);
        boxui::sep(std::cout);
        boxui::kv(std::cout, "Solved At",   timestamp,                      boxui::ansi::BRIGHT_CYAN);
        boxui::kv(std::cout, "Duration",    dur.str(),                      boxui::ansi::BRIGHT_CYAN);
        boxui::kv(std::cout, "Total Steps",
                  format_number(result.tame_steps + result.wild_steps),     boxui::ansi::BRIGHT_CYAN);
        boxui::kv(std::cout, "Algorithm",   "Pollard's Kangaroo",           boxui::ansi::BRIGHT_CYAN);
        boxui::bottom(std::cout);
        std::cout << "\n";
    }

    // Save to file
    // Owner-only permissions; see secure_open_ofstream notes.
    // FailHard: recovered private key sink, world-readable file
    // is worse than no file.
    std::ofstream found_file =
        ::collider::secure_open_ofstream(
            "puzzle_found.txt", std::ios::app,
            ::collider::SecureWriteOnFailure::FailHard);
    if (found_file) {
        found_file << "================================================================================\n";
        found_file << "                    PUZZLE SOLVED (Kangaroo Algorithm)\n";
        found_file << "================================================================================\n";
        found_file << "Timestamp:    " << timestamp << "\n";
        found_file << "Puzzle:       #" << current_puzzle << " (" << bits << "-bit)\n";
        found_file << "Private Key:  " << key_hex << "\n";
        found_file << "Address:      " << target_address << "\n";
        found_file << "Algorithm:    Pollard's Kangaroo\n";
        found_file << "Duration:     " << std::fixed << std::setprecision(3) << total_seconds << " seconds\n";
        found_file << "Total Steps:  " << result.tame_steps + result.wild_steps << "\n";
        found_file << "================================================================================\n\n";
    }

    // Session log: solve event. ADDRESS ONLY by policy.
    {
        std::ostringstream d;
        d << "puzzle=" << current_puzzle
          << " backend=CPU"
          << " address=" << target_address
          << " total_steps=" << (result.tame_steps + result.wild_steps);
        ::collider::log::milestone("puzzle_solved", d.str());
    }

    // Wipe the recovered key bytes once they have hit disk.
    ::collider::secure_wipe(result.private_key.d,
                            sizeof(result.private_key.d));
    ::collider::secure_wipe(key_hex, sizeof(key_hex));

    if (is_multi_puzzle) {
        std::cout << "[*] Puzzle solved! Continuing to next puzzle...\n";
    }
    return PuzzleStepResult::SolvedExitOrContinue;
}

}  // namespace

// T3.10: Dispatcher. Prints the kangaroo banner, decompresses the
// target pubkey, then walks the backend chain in order:
//   RCKangaroo -> MultiGPU Kangaroo -> CPU Kangaroo
// A helper returning FallThrough means "init failed, try the next
// backend in the chain". Every other return is terminal for this puzzle.
PuzzleStepResult run_kangaroo_solve(PuzzleIterContext& ctx) {
    Arguments& args = ctx.args;
    const PuzzleInfo* puzzle = ctx.puzzle;
    const int bits = ctx.tgt.bits;

    std::cout << "\n[*] Using Pollard's Kangaroo Algorithm (O(sqrt(n)))\n";
    std::cout << "    Search complexity reduced from 2^" << (bits-1) << " to ~2^" << ((bits-1)/2) << "\n";
    int expected_bits = (bits - 1) / 2 + 1;
    if (expected_bits < 63) {
        std::cout << "    Expected operations: ~" << format_number(1ULL << expected_bits) << "\n";
    } else {
        std::cout << "    Expected operations: ~2^" << expected_bits << " (still large, but tractable)\n";
    }
    std::cout << "\n";
    std::cout << "    NOTE: Kangaroo step rate may appear similar to brute force key rate.\n";
    std::cout << "    The advantage is ALGORITHMIC: sqrt(n) steps vs n keys.\n";
    std::cout << "    For " << bits << "-bit puzzle: Kangaroo is 2^" << ((bits-1)/2) << "x faster to solve!\n\n";

    // Kangaroo requires a known public key. The selection block in the
    // caller (search for "ALGORITHM SELECTION") guarantees this is non-
    // empty by the time we get here. A missing pubkey demotes
    // args.puzzle_kangaroo to false, so we wouldn't enter this path.
    // CLI / config --pubkey wins over the bundled value when both are set.
    const std::string& kangaroo_pubkey =
        !args.puzzle_pubkey.empty() ? args.puzzle_pubkey
                                    : puzzle->public_key_hex;
    if (!args.puzzle_pubkey.empty()) {
        std::cout << "[*] Using --pubkey override: " << kangaroo_pubkey << "\n";
    }

    // Decompress the public key. Required by the MultiGPU + CPU backends;
    // RCKangaroo re-decompresses internally from its own hex string, but
    // we still need the decompression here as the early failure guard.
    cpu::uint256_t target_pubkey_x, target_pubkey_y;
    if (!cpu::decompress_pubkey(target_pubkey_x, target_pubkey_y, kangaroo_pubkey)) {
        std::cerr << "[!] ERROR: Failed to decompress public key: " << kangaroo_pubkey << "\n";
        return PuzzleStepResult::FatalError;
    }
    std::cout << "[*] Target public key decompressed successfully\n";

    // v1.5.x: --backend cpu|cuda|metal explicit override.
    //
    // Pre-1.5.x the standalone path tried RCKangaroo first (when
    // --use-rckangaroo), fell through to MultiGPU (CUDA/Metal), and
    // ultimately to CPU. The operator had no way to PICK a backend
    // for an A/B test or to deliberately exercise the CPU path on a
    // GPU box; the fall-through always landed on the most-capable
    // available.
    //
    // With --backend set, the dispatch below short-circuits to the
    // requested backend only:
    //   cpu   -> straight to run_kangaroo_cpu (skip both GPU paths).
    //   cuda  -> RCKangaroo first (if compiled), then MultiGPU CUDA.
    //           Never falls to CPU.
    //   metal -> MultiGPU Metal only (RCKangaroo is CUDA-only).
    //           Never falls to CPU.
    //   ""    -> pre-1.5.x fall-through behavior.
    //
    // Unknown values are caught at CLI parse time (apply_backend in
    // cli_parser.cpp); reaching here with an unknown string would
    // already have failed --backend's validator.
    const bool backend_explicit = !args.backend_kind.empty();
    const bool want_cpu   = backend_explicit && args.backend_kind == "cpu";
    const bool want_cuda  = backend_explicit && args.backend_kind == "cuda";
    const bool want_metal = backend_explicit && args.backend_kind == "metal";

    // Phase F2: --solver bsgs short-circuits to the GPU BSGS path
    // BEFORE any kangaroo backend dispatch. The BSGS driver returns
    // FallThrough when the puzzle exceeds its bit-cap so the kangaroo
    // dispatch below still runs for large ranges.
    if (args.solver == "bsgs") {
        std::cout << "[*] --solver bsgs: routing to GPU BSGS solver\n";
        PuzzleStepResult br = run_bsgs_solve(ctx);
        if (br != PuzzleStepResult::FallThrough) return br;
        // Fall-through means BSGS rejected the range (too large).
        // Drop into the kangaroo dispatch below.
    }

    if (want_cpu) {
        std::cout << "[*] --backend cpu: routing to CPU kangaroo "
                     "(skipping RCKangaroo and MultiGPU)\n";
        return run_kangaroo_cpu(ctx);
    }

#ifdef COLLIDER_USE_RCKANGAROO
    // RCKangaroo - High-performance Kangaroo solver (8 GKeys/s on 4090).
    // Gated on --use-rckangaroo OR --backend cuda; on init failure falls
    // through to MultiGPU below unless --backend cuda was explicit (in
    // which case the operator wanted CUDA specifically; we don't auto-
    // demote to Metal).
    if (args.use_rckangaroo || want_cuda) {
        PuzzleStepResult rc_result = run_kangaroo_rckangaroo(ctx);
        if (rc_result != PuzzleStepResult::FallThrough) return rc_result;
    }
#else
    if (want_cuda) {
        std::cerr << "[!] --backend cuda requested but this binary was "
                     "built without RCKangaroo (-DCOLLIDER_USE_RCKANGAROO=OFF).\n";
        return PuzzleStepResult::FatalError;
    }
#endif

    // MultiGPU Kangaroo (CUDA/Metal). With --backend cuda or --backend
    // metal still set, do NOT fall through to CPU on init failure --
    // the operator asked for a specific GPU backend and falling to CPU
    // would be a silent contract break. Surface the failure instead.
    PuzzleStepResult gpu_result =
        run_kangaroo_multigpu(ctx, target_pubkey_x, target_pubkey_y);
    if (gpu_result != PuzzleStepResult::FallThrough) return gpu_result;

    if (want_cuda || want_metal) {
        std::cerr << "[!] --backend " << args.backend_kind
                  << " requested but no compatible GPU initialized. "
                     "Refusing to silently fall back to CPU.\n";
        return PuzzleStepResult::FatalError;
    }

    // CPU Kangaroo: last-resort fallback (auto-dispatch only; --backend
    // gates above prevent this for explicit GPU requests).
    return run_kangaroo_cpu(ctx);
}

}  // namespace collider::runtime::detail
