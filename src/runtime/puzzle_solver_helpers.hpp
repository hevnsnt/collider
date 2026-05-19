/**
 * puzzle_solver_helpers.hpp - Internal helpers carved out of
 * src/runtime/puzzle_solver.cpp during the v1.4.2 structural decomposition.
 *
 * This header is intended for consumption ONLY by puzzle_solver.cpp and
 * puzzle_solver_helpers.cpp. It does not widen the public runtime API; the
 * single public entry point in puzzle_solver.hpp (run_puzzle_mode /
 * run_benchmark) stays unchanged.
 *
 * Each helper preserves the original behavior verbatim. No contract changes
 * to ScanState, KangarooSolver, puzzle_optimized, RCKangarooManager,
 * bench_pipeline, secure_open_ofstream, secure_wipe, or paths::state_dir().
 *
 * Hard-preserve invariants the helpers must keep alive:
 *   1. secure_wipe at every solve path (recovered private key bytes).
 *   2. secure_open_ofstream for puzzle_found.txt and bloom_hits.txt.
 *   3. bench_pipeline::run_pipeline_benchmark + print_result_table delegation.
 *   4. runtime/format.hpp + runtime/balance.hpp as canonical homes for
 *      format_number_human / normalize_path / check_balance_async.
 *   5. paths::state_dir() for kangaroo herd checkpoint paths.
 *   6. SearchState v4 format with full UInt256 position fields.
 *   7. --resume-kangaroo InitKangsHost/SaveKangsHost wiring.
 *   8. SIGINT save path (g_shutdown atomic + R-B9/R-B10).
 *   9. WORK_ASN epoch rebuild (R-B4) is pool-only and not in this module.
 *  10. Brute-force rejection > 128 bits (R-B7).
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "cli/cli_parser.hpp"           // Arguments
#include "core/config.hpp"              // UserConfig
#include "core/puzzle_config.hpp"       // PuzzleInfo
#include "core/types.hpp"               // UInt256
#include "runtime/runtime_globals.hpp"  // GPUDetectionResult

namespace collider::runtime::detail {

// T3.12: Calculate optimal dp_bits for the Kangaroo algorithm. Single
// source of truth. Pre-fix this lived in BOTH puzzle_solver.cpp (file-
// scope anonymous namespace) and puzzle_solver_kangaroo.cpp (a private
// mirror flagged as the "intentional twin"). Two copies of a numeric
// heuristic with no test coverage is exactly the configuration drift
// trap that already bit the dp_bits calculation once. Header-inline so
// every TU that needs it picks the same body.
//
// num_kangaroos <= 0 means "unknown at this call site" and selects the
// puzzle_bits/3 fast path that the RCKangaroo dispatch always used.
//
// @param puzzle_bits   bit size of the puzzle (e.g. 135 for puzzle #135)
// @param num_kangaroos total kangaroos across all GPUs, or <= 0 if not
//                       yet known (RCKangaroo dispatch case)
// @return dp_bits value, clamped to [16, 28]
inline int calculate_optimal_dp_bits(int puzzle_bits, int num_kangaroos) {
    if (num_kangaroos <= 0) {
        // Pre-init dispatch path: use the puzzle_bits/3 heuristic that
        // RCKangaroo historically picked. Stays in [16, 28].
        return std::max(16, std::min(28, puzzle_bits / 3));
    }
    // Expected steps per kangaroo: sqrt(2^puzzle_bits) / num_kangaroos
    //                            = 2^(puzzle_bits/2) / num_kangaroos
    //                            = 2^(puzzle_bits/2 - log2(num_kangaroos))
    int sqrt_bits = puzzle_bits / 2;
    int kang_bits = static_cast<int>(std::log2(static_cast<double>(num_kangaroos)));

    // We want roughly 2^8 to 2^12 DPs per kangaroo for collision detection.
    // dp_bits = sqrt_bits - kang_bits - (8 to 12). +6 headroom yields ~64
    // DPs per kangaroo minimum, a good memory / collision balance.
    int optimal = sqrt_bits - kang_bits + 6;

    // Clamp:
    //  - Min 16: avoids flooding memory with DPs (1 in 65K points).
    //  - Max 28: keeps enough DPs for collision detection at high SM count.
    return std::max(16, std::min(28, optimal));
}

// Resolved puzzle target information. Returned by resolve_puzzle_target
// and consumed by every per-puzzle solve path. All fields mirror the
// locals that the original inline loop populated; nothing more.
struct PuzzleTarget {
    UInt256 range_start;
    UInt256 range_end;
    std::string target_address;
    int bits = 0;
    std::array<uint8_t, 20> target_hash160{};
    bool have_target_hash = false;
    std::string h160_hex;
    bool force_sequential = false;   // bits <= 40 implies exhaustive sequential
    // Convenience limb decomposition that the inline body uses everywhere.
    uint64_t start_lo = 0;
    uint64_t start_hi = 0;
    uint64_t end_lo = 0;
    uint64_t end_hi = 0;
};

// Step disposition returned by each per-puzzle solve helper. The outer
// per-puzzle loop in run_puzzle_mode translates this into the matching
// control-flow action (continue / break / return). Match the original
// inline semantics exactly:
//   - SolvedExitOrContinue: print solved banner already done, file written,
//     key wiped. Single-puzzle: return 0 from run_puzzle_mode. Multi-puzzle
//     (--all-unsolved / --auto-next): continue to next puzzle.
//   - StoppedExitOrContinue: no solution this run (Ctrl+C, exhausted, etc.).
//     Single-puzzle: return 0. Multi-puzzle: continue.
//   - FallThrough: helper did NOT consume the puzzle. Outer loop must keep
//     running its current phase (used by GPU init failures that fall back
//     to CPU paths, and by the multi-puzzle skip-this-puzzle path).
//   - FatalError: return 1 from run_puzzle_mode immediately.
//   - UsageError: return 64 (EX_USAGE) from run_puzzle_mode immediately.
//   - SkipPuzzle: multi-puzzle worklist skip-this-puzzle-and-continue.
//     Single-puzzle callers translate this to UsageError exit code 64
//     (mirrors original behavior at the brute-force >128-bit rejection).
enum class PuzzleStepResult {
    SolvedExitOrContinue,
    StoppedExitOrContinue,
    FallThrough,
    FatalError,
    UsageError,
    SkipPuzzle,
};

// Build the puzzle worklist from the parsed CLI args. Implements the
// --all-unsolved / --auto-next / --puzzle N selection. Mirrors the
// inline block that lived inside run_puzzle_mode after calibration.
// Returns empty vector when --all-unsolved is set and there are no
// unsolved puzzles left in the database (caller prints + returns 0).
std::vector<int> build_puzzle_worklist(const Arguments& args);

// Run GPU batch-size calibration on first run or when --calibrate /
// --force-calibrate was passed. Mutates args.batch_size in place when
// calibration succeeds or a saved value is loaded from config. No-op on
// non-CUDA builds. Mirrors the inline block in the original loop verbatim.
void maybe_run_calibration(Arguments& args, UserConfig& config);

// Resolve the per-puzzle search target (range, address, hash160). Mirrors
// the inline block in the original loop verbatim. Returns false when the
// puzzle number is unknown AND --puzzle-start / --puzzle-end were not
// given (caller logs + returns 1).
bool resolve_puzzle_target(const Arguments& args,
                           int current_puzzle,
                           const PuzzleInfo* puzzle,
                           PuzzleTarget& out);

// Print the search-space size + estimated wall-clock at 1B keys/sec.
// Mirrors the inline block in the original loop verbatim.
void print_search_space_analysis(int bits);

// Apply algorithm-selection rules: kangaroo demotion when no pubkey is
// known, auto-pick kangaroo when pubkey is known, brute-force selection
// otherwise. Mutates args.puzzle_kangaroo and args.puzzle_pubkey in place
// matching the original inline block. The `is_multi_puzzle` flag controls
// whether the TTY-prompt fallback is allowed.
void select_algorithm(Arguments& args,
                      const PuzzleInfo* puzzle,
                      int bits,
                      bool is_multi_puzzle);

// Per-iteration context shared by every solve-path helper. Carries the
// post-resolution state that the original inline body kept as a flat list
// of locals inside the per-puzzle for-loop. Construction is free; the
// solve helpers read/write fields through references and we pass the
// struct by reference to keep the call sites lightweight.
struct PuzzleIterContext {
    Arguments& args;
    const GPUDetectionResult& gpu_info;
    int current_puzzle;
    const PuzzleInfo* puzzle;     // may be null for --puzzle-start/--puzzle-end mode
    PuzzleTarget& tgt;
    bool is_multi_puzzle;
};

// Print the free SHA-256-only benchmark (CPU + GPU SHA256 throughput).
// Used by run_benchmark in the Free build. Returns the process exit code
// (always 0 today; reserved for future error surfacing).
int run_sha256_only_benchmark(const Arguments& args,
                              const GPUDetectionResult& gpu_info);

// Per-puzzle KANGAROO solve dispatcher. Walks RCKangaroo -> MultiGPU
// Kangaroo -> CPU Kangaroo in that order, mirroring the original inline
// block. Returns:
//   SolvedExitOrContinue when a solution was found (banner + file +
//     secure_wipe already done).
//   StoppedExitOrContinue when no solution (Ctrl+C, exhausted, etc.).
//   FatalError when pubkey decompression failed.
//   UsageError when --dp-bits was outside the backend's accepted range.
PuzzleStepResult run_kangaroo_solve(PuzzleIterContext& ctx);

// Per-puzzle BRUTE FORCE solve dispatcher. Routes to the multi-GPU path
// when available; falls through to CPU when not. Mirrors the original
// inline block. Returns disposition matching run_kangaroo_solve.
PuzzleStepResult run_bruteforce_solve(PuzzleIterContext& ctx);

}  // namespace collider::runtime::detail
