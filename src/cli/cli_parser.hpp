/**
 * cli_parser.hpp - Command-line argument parsing for theCollider.
 *
 * Extracted from src/main.cpp during the v1.4.1 A.3 refactor. The struct
 * `Arguments`, the parser entry points (`parse_args`, `parse_args_for_test`),
 * the mode-mutex validator, and `print_usage` all live here so that the
 * runtime drivers (puzzle / pool / brain wallet) and the standalone CLI
 * parser test can share a single source of truth without copying the field
 * list.
 *
 * Behavior is intentionally unchanged from the in-place definitions; this
 * is a pure move, not a redesign.
 */
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "core/secure_buffer.hpp"  // SecureString for the pool password
#include "core/yaml_config.hpp"  // CLIFlags

namespace collider::cli {

/**
 * SecurePassword -- copyable wrapper around SecureString for use as a
 * struct member where the enclosing struct (Arguments) must remain
 * copyable for legacy reasons.
 *
 * Several runners do `Arguments args = args_in;` to get a mutable local
 * copy they can tune (smart-puzzle-selection rewrites, batch-size auto-
 * sizing, etc.) without surfacing the change back to the caller.
 * SecureString is intentionally move-only ("a copied secret is a leaked
 * secret"), which makes Arguments uncopyable by default. Adding a 50-
 * member-line explicit copy constructor to Arguments would solve the
 * compile problem but it would be fragile: every new Arguments field
 * adds a maintenance hazard.
 *
 * Instead this small wrapper:
 *   * On copy: byte-copies the password into a fresh SecureString. The
 *     copy still wipes on its own destruction; the operator visibility
 *     window is the same as the original.
 *   * On move: transfers the SecureString without copying.
 *
 * The data() / size() / empty() / wipe() / assign() surface mirrors
 * SecureString so call sites can treat this type as a drop-in
 * replacement.
 */
class SecurePassword {
public:
    SecurePassword() = default;
    ~SecurePassword() = default;

    // Copy: deep-copy bytes through SecureString::assign so the copy
    // owns wiped-on-free storage of its own. This DOES create a second
    // in-process copy of the secret; that is the tradeoff for keeping
    // Arguments trivially copyable. The window between the copy and the
    // copy's destruction is bounded by the runner's call frame, which
    // typically lives no longer than the original Arguments.
    SecurePassword(const SecurePassword& other) {
        sec_.assign(other.sec_.data(), other.sec_.size());
    }
    SecurePassword& operator=(const SecurePassword& other) {
        if (this != &other) {
            sec_.assign(other.sec_.data(), other.sec_.size());
        }
        return *this;
    }

    SecurePassword(SecurePassword&&) noexcept = default;
    SecurePassword& operator=(SecurePassword&&) noexcept = default;

    void assign(const char* src, std::size_t n) { sec_.assign(src, n); }
    void wipe() noexcept { sec_.wipe(); }

    const char* data() const noexcept { return sec_.data(); }
    std::size_t size() const noexcept { return sec_.size(); }
    bool empty() const noexcept { return sec_.empty(); }

private:
    ::collider::SecureString sec_;
};

}  // namespace collider::cli

/**
 * Command-line arguments. Populated by parse_args; consumed by every mode
 * runner.
 */
struct Arguments {
    std::vector<int> gpu_ids = {};  // Empty = auto-detect available GPUs
    size_t batch_size = 4'000'000;
    // When true, the runner auto-sizes batch_size to fit the configured rule
    // engine's worst-case per-GPU passphrase fan-out before allocating GPU
    // buffers. Set to false in cli_parser when --batch-size is supplied OR
    // in apply_config_to_args when config.yml pins gpu.batch_size, so an
    // explicit value (CLI or yaml) is never silently replaced.
    bool batch_size_auto = true;
    bool verbose = false;
    bool help = false;

    // Benchmark mode
    bool benchmark = false;
    int benchmark_seconds = 30;  // Run benchmark for N seconds

    // Puzzle mode (Bitcoin Puzzle Challenge)
    bool puzzle_mode = true;              // Puzzle mode is now the default
    int puzzle_number = 0;                // Target puzzle number (0 = auto-select easiest unsolved)
    std::string puzzle_target;            // Override target address
    std::string puzzle_range_start;       // Override range start (hex)
    std::string puzzle_range_end;         // Override range end (hex)
    // 33-byte compressed public key (02/03 + 32B hex). Overrides
    // puzzle->public_key_hex when set; needed only when scanning a target
    // whose pubkey isn't in puzzle_history.json (rare; see README).
    std::string puzzle_pubkey;
    bool puzzle_random = true;            // Random search (vs sequential)
    std::string puzzle_checkpoint;        // Checkpoint file for resume
    // standalone kangaroo herd save/resume. When
    // true, load_herd_state is attempted at the standalone puzzle
    // path's checkpoint file (default
    // ~/.collider/state/kangaroo_herd_puzzle_<N>.kang) AFTER backend
    // init() and BEFORE solve(). The SIGINT save path is unconditional
    // and mirrors pool_solver.cpp's behavior. Default false: existing
    // workflows that did not opt in to resume see no behavior change.
    bool resume_kangaroo = false;
    bool puzzle_auto_next = false;        // Auto-progress to next puzzle after solving
    bool puzzle_all_unsolved = false;     // Test all unsolved puzzles (in order)
    int puzzle_min_bits = 0;              // Minimum bit size for multi-puzzle (0 = no limit)
    int puzzle_max_bits = 160;            // Maximum bit size for multi-puzzle
    bool puzzle_kangaroo = false;         // Use Pollard's Kangaroo algorithm (O(sqrt(n)))
    bool use_rckangaroo = true;           // Use RCKangaroo as backend (default: true if available)
    int dp_bits = -1;                     // Distinguished point bits (-1 = auto-calculate)
    std::string bloom_file;               // Bloom filter file for opportunistic address checking
    std::string bloom_tight_file;         // tight CPU-side bloom for dual-bloom empty-hit re-probe
    std::string verify_set_file;          // UVRF file for HitVerifier (rejects bloom false positives)
    std::string verify_set_csv;           // CSV-format UTXO set fallback when no .uvrf is available
    bool track_empty_hits = false;        // log bloom-hit + UVRF-miss as "real but empty" wallets (requires seen-ever bloom)
    // True only when the user explicitly set --track-empty-hits or
    // --no-track-empty-hits (or pinned bloom.track_empty_hits in config.yml).
    // Lets the runner auto-enable the flag when both seen_tight.blf and
    // funded_addresses.uvrf resolve from auto-detection without stomping an
    // explicit operator preference.
    bool track_empty_hits_user_set = false;
    bool use_texture_bloom = false;       // opt-in texture-memory bloom (single-GPU only)

    // WarpWallet brainwallet (v1.4.2 C.5).
    // When non-empty, the brainwallet runner uses scrypt(N=2^18)+PBKDF2(c=2^16)
    // key derivation per the Keybase WarpWallet spec instead of plain SHA-256.
    // The string is the WarpWallet salt (the Keybase reference uses the user's
    // email address).
    std::string warpwallet_salt;

    // Brainwallet mode
    bool brainwallet_mode = false;        // Brainwallet-only mode (requires bloom filter)
    bool brainwallet_v2_mode = false;     // route --brainwallet through v2 (multi-scheme + multi-address + encoding-munge)
    bool brainwallet_setup = false;       // Run brainwallet setup wizard
    std::string wordlist_file;            // Wordlist file for brainwallet scanning
    bool resume = false;                  // Resume from saved state
    size_t save_interval = 1000000;       // Save state every N passphrases checked
    bool cpu_rules = false;               // Force CPU rule processing (enables multi-GPU)

    // --brute: incremental bruteforce by length. When non-empty,
    // replaces the wordlist+rules+PCFG+Markov pipeline with a deterministic
    // alphanumeric sweep of all 62^N strings for each N in this list, shortest
    // first. Implies --brainwallet. Mutually exclusive with --brainwallet-v2
    // and --brainwallet-warpwallet (different candidate-source assumptions).
    std::vector<int> brute_lengths;

    // Brain Wallet v2 (Phase 9, restructure plan v1.4.0).
    // --puzzle-only-v2 enables the multi-scheme puzzle-mode bloom check that
    // short-circuits before EC_MUL when no puzzle target hits. Requires
    // COLLIDER_PRO=ON.
    bool puzzle_only_v2 = false;
    std::string puzzle_keys_file;         // Override path to puzzle_history.json
    std::string schemes_csv;              // comma list of derivation schemes
    // addr_types_csv removed in Q10 cleanup: the v2 orchestrator -> kernel
    // bridge for multi-address scanning was never wired through; the legacy
    // --brainwallet --bloom path remains supported. Deferred to v1.5.0+.

    // Calibration mode
    bool calibrate = false;               // Run batch size calibration
    bool force_calibrate = false;         // Force re-calibration even if already done

    // Smart puzzle selection
    bool analyze_puzzles = false;         // Show puzzle analysis without running
    bool smart_select = true;             // Auto-select best puzzle by ROI (default ON)

    // Debug mode
    bool debug = false;                   // Show debug output

    // Diagnostic instrumentation. When false (default), the per-kernel
    // CUDA-event timing collector (perf::PerfCollector) stays disabled
    // for the lifetime of the scan, and instrument_start / instrument_stop
    // in the fused brain-wallet pipeline are single-load + early-return
    // no-ops. When true, the TUI performance panel populates and the
    // hot path pays one cudaEventRecord per kernel launch per side.
    //
    // Default-off (vs the prior unconditional set_enabled(true) in the
    // brain-wallet runner) is the safer posture for two reasons:
    //   1) Production scans without an operator watching the perf panel
    //      pay zero cuda-event cost (one relaxed load + predicted-not-
    //      taken branch on every launch).
    //   2) cudaEvent slots in the collector ring are created lazily on
    //      first touch and bound to the CURRENT device. Under multi-GPU
    //      dispatch with uneven per-GPU active state (e.g. one GPU
    //      toggled out mid-scan via the RuntimeControl GpuPhase machine)
    //      the ring's slot-to-device pairing can drift, so cross-device
    //      record attempts return cudaErrorInvalidResourceHandle. The
    //      collector handles this by dropping the sample, but the
    //      structurally cleanest avoidance is to leave the instrumentation
    //      off unless the operator is actively inspecting the panel.
    bool perf_instrument = false;

    // Pool mode (distributed solving)
    bool pool_mode = false;               // Connect to pool for distributed solving
    std::string pool_url;                 // Pool URL (jlp://host:port or http://host:port)
    std::string pool_worker;              // Worker name (Bitcoin address for rewards)
    // Pool password (optional). Reading the secret from argv leaks it via
    // ps / Task Manager; --pool-password-file is the supported replacement.
    // Held in a SecurePassword wrapper (around SecureString) so the bytes
    // get wiped on Arguments destruction and can be wiped explicitly the
    // moment the credential is handed off to PoolConfig in run_pool_mode
    // (cuts the upstream heap window down from "lifetime of Arguments"
    // to "until handoff completes"). The wrapper exists because Arguments
    // is copied in several runner entry points to obtain a mutable local;
    // SecureString itself is move-only, which would break those copies.
    //
    // Declared `mutable` so the run_pool_mode handoff site (which takes
    // `const Arguments&`) can wipe the secret as soon as it has been
    // copied into PoolConfig, without a const_cast or a signature change
    // on the runtime driver entry point.
    mutable ::collider::cli::SecurePassword pool_password;
    std::string pool_password_file;       // Path to a file whose first line is
                                          // the pool password. Strips trailing
                                          // CR/LF. Wins over --pool-password
                                          // when both are supplied.
    std::string pool_api_key;             // API key for HTTP pools (optional)

    // Config file
    std::string config_file;              // Custom config file path (default: ./config.yml)

    // TUI integration. When --no-tui is supplied (or --tui is supplied
    // to force it on, or tui.enabled is pinned in config.yml) the runner uses
    // an explicit setting; otherwise no_tui defaults to "auto-detect TTY at
    // run time" (the runner consults isatty on stdout). The user_set sentinel
    // mirrors the track_empty_hits_user_set pattern so config.yml can pin a
    // value without stomping an explicit CLI choice.
    bool no_tui = false;
    bool no_tui_user_set = false;

    // Menu navigation
    bool go_back = false;                 // Signal to return to main menu (not exit)
    bool exit_program = false;            // Signal to exit program cleanly
};

/**
 * Mutually-exclusive search-mode validation (track-e fix E.1).
 *
 * Returns 0 if no conflict; -1 with `msg` set if more than one of
 * {--brainwallet, --pool, --kangaroo (with no other puzzle mode), --puzzle N
 * (when combined with brainwallet/pool)} is active.
 *
 * `--puzzle N --kangaroo` is the standard "solve puzzle N with kangaroo
 * algorithm" combination and is NOT a conflict by itself.
 */
int validate_mode_mutex(const Arguments& args, std::string& msg);

/**
 * Internal core argv parser. No exit, no stderr; pure transformation.
 * Returns 0 on success, -1 if mode mutex violated (msg set in that case).
 *
 * Both parse_args() and parse_args_for_test() delegate here. Keeps a single
 * source of truth for argv -> Arguments / CLIFlags.
 */
int parse_args_core(int argc, char* argv[], Arguments& args,
                    collider::CLIFlags& cli, std::string& err_msg);

/**
 * Production entry point: parses argv, exits the process with a clear error
 * on mode-mutex violation. Fills `cli_out` (if non-null) with which CLI flags
 * were explicitly set.
 */
Arguments parse_args(int argc, char* argv[], collider::CLIFlags* cli_out = nullptr);

/**
 * Test-only entry point: returns -1 instead of exiting on mode-mutex
 * violation, and reports the rejection message in `err_out`. Used by
 * tests/test_cli_parser.cpp.
 */
int parse_args_for_test(int argc, char* argv[], Arguments& args,
                        collider::CLIFlags& cli, std::string* err_out = nullptr);

/**
 * Print usage information.
 */
void print_usage();
