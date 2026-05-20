/**
 * theCollider - Bitcoin Puzzle Solver
 *
 * GPU-accelerated solver for the Bitcoin Puzzle Challenge.
 *
 * Usage:
 *   collider --puzzle <number>
 *
 * Options:
 *   --puzzle, -P     Puzzle number to solve (66-160)
 *   --all-unsolved   Auto-progress through all unsolved puzzles
 *   --random         Use random search instead of sequential
 *   --gpus, -g       GPU device IDs to use (default: auto-detect)
 *   --benchmark      Run GPU performance benchmark
 *   --verbose, -v    Verbose output
 *   --help, -h       Show this help message
 *
 * Example:
 *   collider --puzzle 71
 *   collider --all-unsolved --gpus 0,1
 */

// Prevent Windows min/max macros from breaking std::min/std::max
#ifndef NOMINMAX
#define NOMINMAX
#endif
// Prevent Windows.h from including winsock.h (conflicts with winsock2.h)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <io.h>      // _write (async-signal-safe stderr write)
#else
#include <unistd.h>  // write (async-signal-safe stderr write)
#endif

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "cli/cli_parser.hpp"
#include "core/config.hpp"
#include "core/logger.hpp"
#include "core/puzzle_analysis.hpp"
#include "core/puzzle_config.hpp"          // PuzzleDatabase (auto-enable smart pick)
#include "core/session_log.hpp"             // collider::log::{init_session_log,
                                            //   write_startup_banner,
                                            //   install_crash_handler,
                                            //   write_hardware_enum}
#include "core/version.hpp"                 // collider::kVersion
#include "core/yaml_config.hpp"
#include "platform/platform.hpp"            // platform::DeviceInfo for hardware enum
#include "runtime/gpu_detection.hpp"        // detect_gpus
#include "runtime/pool_solver.hpp"
#include "runtime/puzzle_solver.hpp"
#include "runtime/runtime_globals.hpp"
#include "ui/interactive.hpp"               // Interactive::display_header
#include "ui/interactive_ui.hpp"
#ifdef COLLIDER_PRO
#include "runtime/brain_wallet_runner.hpp"
#include "runtime/license_gate.hpp"
#include "ui/brainwallet_setup.hpp"
#endif

// ============================================================================
// A.3 refactor (commits 1/6 - 6/6) summary
// ============================================================================
// theCollider's mode runners and helpers all live in dedicated runtime
// modules. main.cpp is the thin dispatcher: parse args -> license gate
// (Pro) -> signal handler / logger init -> GPU detect -> mode dispatch.
//   - cli_parser.{hpp,cpp}       Arguments / parse_args / print_usage  (1)
//   - ui/interactive_ui.{hpp,cpp} run_interactive_mode + submenus     (2)
//   - runtime/pool_solver.{hpp,cpp}        run_pool_mode               (3)
//   - runtime/brain_wallet_runner.{hpp,cpp} run_brain_wallet_mode (PRO)(4)
//   - runtime/puzzle_solver.{hpp,cpp}      run_benchmark / run_puzzle  (5)
//   - runtime/gpu_detection.{hpp,cpp}      detect_gpus                 (6)
//   - runtime/license_gate.{hpp,cpp}       process_activate_flag /
//                                          validate_startup_license   (6 PRO)
// ============================================================================

using namespace collider;

// Global shutdown flag set ONLY by signal_handler (must be async-signal-
// safe per POSIX signal-safety(7)). All logging/printing of the shutdown
// event happens from the main thread once it observes g_shutdown == true.
std::atomic<bool> g_shutdown{false};

// captured signal number for delayed reporting from main
// thread. Set inside signal_handler (atomic store is async-signal-safe).
// Read once after the main loop exits so the LOG_INFO is emitted from
// non-signal context.
std::atomic<int> g_shutdown_signal{0};
std::atomic<bool> g_shutdown_logged{false};

// invoked from main-thread context after the loop sees
// g_shutdown==true. Performs the print + log emission that USED to live
// in signal_handler (and self-deadlocked on Logger::mutex_ if SIGINT
// arrived while a worker thread held the lock).
void emit_shutdown_message_from_main() {
    if (g_shutdown_logged.exchange(true)) return;
    int signum = g_shutdown_signal.load(std::memory_order_acquire);
    if (signum == 0) return;  // shutdown set without a signal (e.g. found key)
    // The signal handler already printed the immediate acknowledgment via
    // write(). Print the follow-up from main-thread context where cout is safe.
    std::cout << "[*] Saving checkpoint state and draining in-flight DPs...\n";
    LOG_INFO("Signal received: " + std::to_string(signum) +
             " (SIGINT=" + std::to_string(SIGINT) +
             ", SIGTERM=" + std::to_string(SIGTERM) + ")");
    // Session log: record the signal-driven shutdown alongside the
    // legacy collider.log entry. Calling milestone() here is safe
    // because emit_shutdown_message_from_main is documented to run
    // from MAIN-thread context (NOT a signal handler); the runtime
    // drivers all defer the call until they exit their solve loop.
    collider::log::milestone(
        "shutdown",
        std::string("signal=") + std::to_string(signum));
}

// fix: async-signal-safe handler.
// Per POSIX signal-safety(7), almost nothing in the C++ runtime is safe
// to call from a signal handler. Atomic stores are safe. write()/_write()
// is safe. cout/cerr are NOT safe (may hold their mutex at signal time).
// The full shutdown message is emitted from main-thread context via
// emit_shutdown_message_from_main() once the solve loop exits.
void signal_handler(int signum) {
    g_shutdown_signal.store(signum, std::memory_order_release);
    g_shutdown.store(true, std::memory_order_release);
    // Immediate acknowledgment so the user knows the keypress landed.
    // write()/_write() is async-signal-safe; the literal size avoids strlen().
    static const char kMsg[] = "\n[!] Ctrl-C caught - stopping...\n";
#ifdef _WIN32
    (void)_write(2, kMsg, static_cast<unsigned>(sizeof(kMsg) - 1));
#else
    (void)write(2, kMsg, sizeof(kMsg) - 1);
#endif
}

/**
 * If neither --benchmark nor --puzzle was supplied, auto-enable puzzle
 * mode and pick a default puzzle. Mirrors the legacy behavior of the
 * pre-A.3 main(). The smart-selection branch is identical to the one
 * inside run_puzzle_mode's banner; we apply it here as well so the
 * banner header in run_puzzle_mode sees the final puzzle number rather
 * than 0 (its smart-selection branch only fires on puzzle_number==0).
 */
static void auto_enable_puzzle_mode_if_needed(Arguments& args) {
    if (args.benchmark || args.puzzle_mode) return;

    args.puzzle_mode = true;
    if (args.puzzle_number != 0) return;

    if (args.smart_select) {
        int best = get_best_puzzle(400.0);
        if (best > 0) {
            args.puzzle_number = best;
            auto puzzle = PuzzleDatabase::get_puzzle(best);
            std::cout << "\n[*] Smart Selection: Puzzle #" << best;
            if (puzzle && !puzzle->public_key_hex.empty()) {
                std::cout << " (Kangaroo - pubkey known)";
                args.puzzle_kangaroo = true;
            } else {
                std::cout << " (Brute Force - no pubkey)";
            }
            std::cout << "\n";
            std::cout << "    Use --no-smart to disable smart selection\n";
            std::cout << "    Use --analyze to see full puzzle ranking\n\n";
        } else {
            args.puzzle_number = 71;
        }
    } else {
        // Legacy: sequential selection (easiest first by puzzle number)
        auto unsolved = PuzzleDatabase::get_unsolved();
        args.puzzle_number = unsolved.empty() ? 71 : unsolved[0]->number;
    }
}

/**
 * Main entry point.
 */
int main(int argc, char* argv[]) {
    // Enable ANSI colors on Windows
    ui::enable_windows_ansi();

    // Per-session log: opened BEFORE arg parsing so a parse_args crash
    // or a malformed argv that triggers the fail-hard path inside the
    // CLI still produces a session log entry. paths::collider_home()
    // resolves to ~/.collider/ via env vars; the call is allocation-
    // only at this point (no file I/O), so it is safe to invoke before
    // signal handlers / loggers are installed.
    collider::log::init_session_log();

    // Parse arguments. CLIFlags carries explicit per-arg "was set" bits,
    // populated inside parse_args at the moment argv supplied each value
    // (track-e refactor, Wave 5).
    collider::CLIFlags cli_flags;
    Arguments args = parse_args(argc, argv, &cli_flags);

    // Load config file (config.yml in cwd or ~/.collider/config.yml).
    // Command-line arguments take precedence over config file.
    collider::AppConfig app_config;
    if (app_config.load(args.config_file)) {
        collider::apply_config_to_args(args, app_config, cli_flags);
    }

    if (args.help) {
        print_usage();
        return 0;
    }

#ifndef COLLIDER_PRO
    // Free build: brain-wallet code is not compiled in. Reject early
    // with a pointer to the Pro upgrade. Both --brainwallet (CLI) and
    // brainwallet_enabled (config.yml) end up setting brainwallet_mode.
    if (args.brainwallet_mode) {
        std::cerr << "[PRO] Brain wallet scanning requires theCollider Pro.\n"
                  << "      Purchase at: https://collisionprotocol.com/pro\n";
        return 1;
    }
#endif

#ifdef COLLIDER_PRO
    // --activate KEY: save and validate a new license key, then exit.
    {
        int activate_exit = 0;
        if (collider::runtime::process_activate_flag(argc, argv, activate_exit)) {
            return activate_exit;
        }
    }
    // Startup license check (cache-backed, 24h).
    {
        int license_exit = 0;
        if (!collider::runtime::validate_startup_license(license_exit)) {
            return license_exit;
        }
    }
#endif

    // Run brainwallet setup wizard if requested (Pro-only feature; the
    // free build rejected --brainwallet-setup at parse time via the
    // brainwallet_mode early-exit above when applicable).
    if (args.brainwallet_setup) {
#ifdef COLLIDER_PRO
        ui::enable_windows_ansi();
        ui::BrainwalletSetup::run_wizard();
        return 0;
#else
        std::cerr << "[PRO] Brain wallet scanning requires theCollider Pro.\n"
                  << "      Purchase at: https://collisionprotocol.com/pro\n";
        return 1;
#endif
    }

    if (args.debug) std::cout << "[DEBUG] Starting collider...\n" << std::flush;

    // Setup signal handler.
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    if (args.debug) std::cout << "[DEBUG] Signal handlers set\n" << std::flush;

    // Initialize logger for crash diagnosis.
    if (args.debug) std::cout << "[DEBUG] Initializing logger...\n" << std::flush;
    auto& logger = Logger::instance();
    try {
        if (logger.init()) {
            LOG_INFO(std::string("Starting theCollider v") + collider::kVersion);
        }
    } catch (const std::exception& e) {
        if (args.debug) std::cerr << "[DEBUG] Logger exception: " << e.what() << "\n" << std::flush;
    } catch (...) {
        if (args.debug) std::cerr << "[DEBUG] Logger unknown exception\n" << std::flush;
    }
    if (args.debug) std::cout << "[DEBUG] Logger initialized\n" << std::flush;

    // Session log: write the startup banner (version, build flags, argv
    // with credentials redacted, cwd, config path), then arm the OS
    // crash handler so a SEGV / SEH from this point on lands in
    // ~/.collider/crash-<ts>.log with the latest session_state.json
    // attached. Order matters: banner first because the crash handler
    // pre-allocates its paths against the boot timestamp the sink
    // captured at init_session_log().
    collider::log::write_startup_banner(argc, argv, args);
    collider::log::install_crash_handler();

    // Detect real GPU hardware. Body lives in runtime/gpu_detection.cpp.
    if (args.debug) std::cout << "[DEBUG] Detecting GPUs...\n" << std::flush;
    auto gpu_info = detect_gpus(args.gpu_ids);
    if (args.debug) std::cout << "[DEBUG] GPUs detected: " << gpu_info.device_count << "\n" << std::flush;

    // Session log: capture the post-detection hardware enumeration. We
    // re-query the platform layer directly rather than reusing the
    // formatted strings inside GPUDetectionResult because the session
    // log wants the structured per-device fields (SM major.minor, VRAM
    // breakdown, multiprocessor count) that the human-facing summary
    // collapses into a single "Nx Foo GPU" line. Best-effort: if
    // platform init failed we skip silently; the banner already noted
    // backend=CPU + device_count=1 in that case.
    {
        std::vector<collider::platform::DeviceInfo> devices;
        try {
            auto& plat = collider::platform::get_platform();
            int total = plat.get_device_count();
            for (int id : args.gpu_ids) {
                if (id >= 0 && id < total) {
                    devices.push_back(plat.get_device_info(id));
                }
            }
        } catch (...) {
            // Platform not initialized (CPU fallback path) -- skip.
        }
        collider::log::write_hardware_enum(devices);
    }

    // INTERACTIVE MODE: when no command-line arguments provided.
    if (argc == 1) {
        double gpu_speed_mkeys = gpu_info.estimated_speed > 0
            ? gpu_info.estimated_speed / 1e6
            : 400.0;
        args = ui::run_interactive_mode(args, gpu_speed_mkeys);

        if (args.exit_program) return 0;
        if (args.help) { print_usage(); return 0; }
    } else {
        // Direct CLI invocation: render the brand header so every mode shows
        // the version banner, not just the interactive menu.
        ui::Interactive::display_header("theCollider", collider::kVersion);
        std::cout << "\n";
    }

    // ---- Mode dispatch -----------------------------------------------------
    // POOL MODE first (its banner/header comes from the runtime driver).
    if (args.pool_mode) {
        return collider::runtime::run_pool_mode(args, gpu_info);
    }

    // BRAINWALLET MODE / --puzzle-only-v2 (Pro-only).
#ifdef COLLIDER_PRO
    if (args.puzzle_only_v2 || args.brainwallet_mode) {
        return collider::runtime::run_brain_wallet_mode(args);
    }
#endif

    // If neither benchmark nor puzzle is set, auto-enable puzzle mode.
    auto_enable_puzzle_mode_if_needed(args);

    // BENCHMARK MODE.
    if (args.benchmark) {
        return collider::runtime::run_benchmark(args, gpu_info);
    }

    // PUZZLE MODE (default).
    if (args.puzzle_mode) {
        return collider::runtime::run_puzzle_mode(args, gpu_info);
    }

    // No mode resolved (shouldn't be reachable after auto_enable).
    std::cerr << "[!] Error: No mode specified. Use --puzzle <number> or --benchmark.\n";
    print_usage();
    return 1;
}
