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

#include "core/yaml_config.hpp"  // CLIFlags

/**
 * Command-line arguments. Populated by parse_args; consumed by every mode
 * runner.
 */
struct Arguments {
    std::vector<int> gpu_ids = {};  // Empty = auto-detect available GPUs
    size_t batch_size = 4'000'000;
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
    // v1.4.1: 33-byte compressed public key (02/03 + 32B hex). Overrides
    // puzzle->public_key_hex when set; needed only when scanning a target
    // whose pubkey isn't in puzzle_history.json (rare; see README).
    std::string puzzle_pubkey;
    bool puzzle_random = true;            // Random search (vs sequential)
    std::string puzzle_checkpoint;        // Checkpoint file for resume
    bool puzzle_auto_next = false;        // Auto-progress to next puzzle after solving
    bool puzzle_all_unsolved = false;     // Test all unsolved puzzles (in order)
    int puzzle_min_bits = 0;              // Minimum bit size for multi-puzzle (0 = no limit)
    int puzzle_max_bits = 160;            // Maximum bit size for multi-puzzle
    bool puzzle_kangaroo = false;         // Use Pollard's Kangaroo algorithm (O(sqrt(n)))
    bool use_rckangaroo = true;           // Use RCKangaroo as backend (default: true if available)
    int dp_bits = -1;                     // Distinguished point bits (-1 = auto-calculate)
    std::string bloom_file;               // Bloom filter file for opportunistic address checking

    // Brainwallet mode
    bool brainwallet_mode = false;        // Brainwallet-only mode (requires bloom filter)
    bool brainwallet_setup = false;       // Run brainwallet setup wizard
    std::string wordlist_file;            // Wordlist file for brainwallet scanning
    bool resume = false;                  // Resume from saved state
    size_t save_interval = 1000000;       // Save state every N passphrases checked
    bool cpu_rules = false;               // Force CPU rule processing (enables multi-GPU)

    // Brain Wallet v2 (Phase 9, restructure plan v1.4.0).
    // --puzzle-only-v2 enables the multi-scheme puzzle-mode bloom check that
    // short-circuits before EC_MUL when no puzzle target hits. Requires
    // COLLIDER_PRO=ON.
    bool puzzle_only_v2 = false;
    std::string puzzle_keys_file;         // Override path to puzzle_history.json
    std::string schemes_csv;              // Phase 3: comma list of derivation schemes
    std::string addr_types_csv;           // Phase 4: comma list of address types

    // Calibration mode
    bool calibrate = false;               // Run batch size calibration
    bool force_calibrate = false;         // Force re-calibration even if already done

    // Smart puzzle selection
    bool analyze_puzzles = false;         // Show puzzle analysis without running
    bool smart_select = true;             // Auto-select best puzzle by ROI (default ON)

    // Debug mode
    bool debug = false;                   // Show debug output

    // Pool mode (distributed solving)
    bool pool_mode = false;               // Connect to pool for distributed solving
    std::string pool_url;                 // Pool URL (jlp://host:port or http://host:port)
    std::string pool_worker;              // Worker name (Bitcoin address for rewards)
    std::string pool_password;            // Pool password (optional)
    std::string pool_api_key;             // API key for HTTP pools (optional)

    // Config file
    std::string config_file;              // Custom config file path (default: ./config.yml)

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
 * Internal core argv parser. No exit, no stderr -- pure transformation.
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
