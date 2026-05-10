/**
 * puzzle_solver.hpp - Puzzle-mode + benchmark runtime drivers.
 *
 * Extracted out of src/main.cpp during the v1.4.1 A.3 refactor (commit 5/6).
 * Hosts the two dispatch-time entry points used by main():
 *
 *   - run_benchmark(args, gpu_info)   --benchmark mode
 *   - run_puzzle_mode(args, gpu_info) --puzzle / default mode
 *
 * Both return the process exit code. Available in BOTH Free and Pro builds:
 * puzzle mode is a free feature, and the benchmark is gated internally to
 * the appropriate code path per edition (free path uses CPU/GPU SHA-256
 * benchmarks; Pro path exercises the brain-wallet pipeline).
 *
 * Also hosts the small helpers that previously lived as TU-local statics
 * in main.cpp but are now consumed by other runtime modules
 * (interactive_ui.cpp, brain_wallet_runner.cpp) via forward declarations:
 *
 *   - format_number, format_number_human  -- defined here, declared in
 *     callers as free functions for backwards compatibility with the
 *     existing forward-decl pattern set up by the previous A.3 commits.
 *
 *   - normalize_path                      -- ditto.
 *
 *   - check_balance_async                 -- ditto.
 *
 * The function bodies for analyze_puzzle / print_puzzle_analysis /
 * get_best_puzzle (declared in core/puzzle_analysis.hpp) also live in
 * puzzle_solver.cpp now (the puzzle_analysis.hpp comment that says "still
 * in main.cpp for now" is updated in this commit).
 */
#pragma once

#include "cli/cli_parser.hpp"           // Arguments
#include "runtime/runtime_globals.hpp"  // GPUDetectionResult

namespace collider::runtime {

/**
 * Run the GPU performance benchmark (--benchmark). In the Pro build this
 * exercises the full SHA256 -> EC -> hash160 -> bloom pipeline; in Free it
 * runs CPU + GPU SHA-256 throughput tests only. Returns the process exit
 * code.
 */
int run_benchmark(const Arguments& args, const GPUDetectionResult& gpu_info);

/**
 * Run puzzle mode - the default Bitcoin Puzzle Challenge solver. Handles
 * RCKangaroo, MultiGPU Kangaroo, CPU Kangaroo, multi-GPU brute force, and
 * CPU brute force fallbacks; supports --all-unsolved auto-progression and
 * SearchStateManager checkpoint resume. Returns the process exit code.
 */
int run_puzzle_mode(const Arguments& args, const GPUDetectionResult& gpu_info);

/**
 * Print the solved-puzzle zone distribution analysis to stdout. Used by
 * main.cpp's verbose-mode pre-banner output and the interactive UI.
 */
void analyze_solved_puzzles();

}  // namespace collider::runtime
