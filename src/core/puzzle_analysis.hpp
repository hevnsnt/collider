/**
 * puzzle_analysis.hpp - Smart puzzle selection / ROI ranking helpers.
 *
 * Extracted out of src/main.cpp during the v1.4.1 A.3 refactor so the
 * interactive UI flow (src/ui/interactive_ui.{hpp,cpp}) and the puzzle
 * runtime driver (src/runtime/puzzle_solver.{hpp,cpp}) can both reference
 * the PuzzleAnalysis struct without pulling in main.cpp.
 *
 * The function bodies still live in main.cpp for now (will migrate to
 * runtime/puzzle_solver.cpp in a follow-up commit of the same refactor).
 * What matters here is the struct shape + signatures so callers compile.
 */
#pragma once

#include <string>

#include "core/puzzle_config.hpp"  // collider::PuzzleInfo

/**
 * Puzzle analysis result for ROI comparison.
 */
struct PuzzleAnalysis {
    int number;
    int bits;
    double btc_reward;
    bool has_pubkey;
    std::string algorithm;     // "Kangaroo" or "BruteForce"
    double complexity_bits;    // Log2 of operations required
    double roi_score;          // Higher = better (reward / complexity)
    std::string feasibility;   // "RECOMMENDED", "VIABLE", "DIFFICULT", "INFEASIBLE"
    double estimated_gpu_years;  // Very rough estimate
};

/**
 * Analyze a single puzzle for ROI. Pure function over the static puzzle
 * database; no side effects.
 */
PuzzleAnalysis analyze_puzzle(const collider::PuzzleInfo* puzzle,
                              double gpu_speed_mkeys);

/**
 * Print puzzle analysis ranking to stdout (used by --analyze).
 */
void print_puzzle_analysis(double gpu_speed_mkeys);

/**
 * Pick the best ROI puzzle for the current hardware. Returns the puzzle
 * number, or 0 if no unsolved puzzles exist.
 */
int get_best_puzzle(double gpu_speed_mkeys);
