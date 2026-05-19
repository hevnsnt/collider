/**
 * puzzle_solver.cpp - Implementation of theCollider's puzzle-mode and
 * benchmark runtime drivers.
 *
 * Available in BOTH Free and Pro builds. The Pro-only branches inside
 * the benchmark and puzzle paths remain gated by #ifdef COLLIDER_PRO.
 */
#include "runtime/puzzle_solver.hpp"

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
#include <io.h>      // _isatty
#include <stdio.h>   // _fileno
#else
#include <unistd.h>  // isatty
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef COLLIDER_HAS_OPENSSL
#include <openssl/sha.h>
#endif

#ifdef __APPLE__
// CommonCrypto uses the ARMv8 SHA crypto extensions on Apple Silicon,
// roughly 100-300x faster than OpenSSL's software fallback when OpenSSL
// is built without crypto-extension support (Homebrew default).
#include <CommonCrypto/CommonDigest.h>
#endif

#if defined(__APPLE__) && defined(COLLIDER_USE_METAL)
#include "gpu/sha256_metal_bench.hpp"
#endif

#ifdef COLLIDER_USE_CUDA
#include <cuda_runtime.h>
// Forward decl from src/gpu/sha256.cu (no header for that one yet).
extern "C" cudaError_t sha256_batch(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    uint8_t* d_hashes,
    size_t count,
    cudaStream_t stream);
#endif

#include "core/byte_codec.hpp"
#include "core/config.hpp"
#include "core/crypto_cpu.hpp"
#include "core/kangaroo.hpp"
#include "core/logger.hpp"
#include "core/paths.hpp"
#include "core/puzzle_analysis.hpp"
#include "core/puzzle_config.hpp"
#include "core/search_state.hpp"
#include "core/secure_buffer.hpp"      // secure_wipe for key buffers
#include "core/secure_write.hpp"       // owner-only file open for key/hit logs
#include "core/types.hpp"
#include "gpu/kangaroo_solver_gpu.hpp"
#include "gpu/puzzle_gpu.hpp"
#ifdef COLLIDER_USE_RCKANGAROO
#include "gpu/rckangaroo_wrapper.hpp"
#endif
#ifdef COLLIDER_PRO
#include "gpu/brain_wallet_gpu.hpp"
#include "runtime/bench_pipeline.hpp"
#endif
#include "runtime/runtime_globals.hpp"
#include "ui/banner.hpp"
#include "ui/box_render.hpp"
#include "ui/btc_balance.hpp"
#include "ui/interactive.hpp"
#include "runtime/balance.hpp"
#include "runtime/format.hpp"
#include "runtime/puzzle_solver_helpers.hpp"

// File-scope using directives that mirror main.cpp's behavior so the
// extracted code resolves the same names. The puzzle-mode body relies on
// unqualified PuzzleInfo, PuzzleDatabase, UInt256, KangarooSolver, gpu::,
// cpu::, ui::, Logger, etc., which all live in namespace collider.
using namespace collider;

// Pull collider::ui::format_rate plus the shared text/balance helpers
// (now in collider::runtime) into the global namespace so the puzzle-mode
// body can keep calling them unqualified.
using collider::ui::format_rate;
using collider::runtime::format_number_human;
using collider::runtime::normalize_path;
using collider::runtime::check_balance_async;

// ============================================================================
// Solved Puzzle Keys Database (for reference)
// ============================================================================
// Puzzles 1-67 have known solutions; their private keys are recorded here as
// a sanity reference and for self-test of the brute-force / kangaroo pipeline.
// C.9 deleted the Center-Heavy "zone" scanning strategy that used
// this database to bias a brute-force scan toward "high-probability" zones.
// That strategy was pseudoscience built on selection bias (~67 data points
// drawn from a one-puzzle-per-bit-range distribution that mathematically
// must skew toward 0.5-1.0 of each range). The standalone GPU search now
// scans uniformly (random by default, sequential with --sequential-search).
// ============================================================================

namespace {

struct SolvedPuzzle {
    int number;           // Puzzle number (1-160)
    uint64_t key_lo;      // Lower 64 bits of private key
    uint64_t key_hi;      // Upper 64 bits (0 for puzzles <= 64)
    double position_pct;  // Position within range as percentage
};

// Known solved puzzle keys (puzzles 1-67 confirmed solved per Bitcoin
// Puzzle transaction records). Puzzles 68 and above remain unsolved as of
// 2026-05-11. Prior fabricated 68/69/70 placeholder rows (bogus keys +
// speculative "solved" dates) were removed. Add new rows ONLY when the
// solving transaction is on-chain and the key is recoverable from public
// sources.
const SolvedPuzzle SOLVED_PUZZLES[] = {
    // Early puzzles (trivial, 1-20)
    {  1, 0x0000000000000001ULL, 0, 100.0 },
    {  2, 0x0000000000000003ULL, 0, 100.0 },
    {  3, 0x0000000000000007ULL, 0, 87.5 },
    {  4, 0x0000000000000008ULL, 0, 50.0 },
    {  5, 0x0000000000000015ULL, 0, 65.6 },
    {  6, 0x0000000000000031ULL, 0, 60.9 },
    {  7, 0x000000000000004CULL, 0, 59.4 },
    {  8, 0x00000000000000E0ULL, 0, 87.5 },
    {  9, 0x00000000000001D3ULL, 0, 91.0 },
    { 10, 0x0000000000000202ULL, 0, 50.2 },
    { 11, 0x0000000000000483ULL, 0, 56.4 },
    { 12, 0x0000000000000A7BULL, 0, 66.4 },
    { 13, 0x0000000000001460ULL, 0, 63.5 },
    { 14, 0x0000000000002930ULL, 0, 64.5 },
    { 15, 0x00000000000068F3ULL, 0, 82.0 },
    { 16, 0x000000000000C936ULL, 0, 78.6 },
    { 17, 0x000000000001764FULL, 0, 73.1 },
    { 18, 0x000000000003080DULL, 0, 75.8 },
    { 19, 0x00000000000559BAULL, 0, 66.8 },
    { 20, 0x00000000000D2C55ULL, 0, 82.4 },
    // Medium puzzles (21-50)
    { 21, 0x00000000001BA534ULL, 0, 86.4 },
    { 22, 0x0000000000346532ULL, 0, 81.5 },
    { 23, 0x0000000000688CF6ULL, 0, 81.8 },
    { 24, 0x00000000009D0A0DULL, 0, 61.4 },
    { 25, 0x000000000137C5D3ULL, 0, 60.9 },
    { 26, 0x0000000002B47C0AULL, 0, 67.5 },
    { 27, 0x00000000051E88D5ULL, 0, 63.7 },
    { 28, 0x000000000C5B9C7FULL, 0, 77.2 },
    { 29, 0x0000000016BF8A26ULL, 0, 71.2 },
    { 30, 0x000000003A5E8E17ULL, 0, 91.4 },
    { 31, 0x000000007ABBC8A3ULL, 0, 95.9 },
    { 32, 0x00000000E9AE4933ULL, 0, 91.3 },
    { 33, 0x00000001A88C0C95ULL, 0, 83.0 },
    { 34, 0x0000000340326E96ULL, 0, 81.2 },
    { 35, 0x00000006AC3875A9ULL, 0, 83.5 },
    { 36, 0x0000000D916CE8A1ULL, 0, 84.7 },
    { 37, 0x0000001757756A93ULL, 0, 72.9 },
    { 38, 0x0000002DB46D0753ULL, 0, 71.6 },
    { 39, 0x000000685A8C3E89ULL, 0, 81.4 },
    { 40, 0x000000D2C55C00E3ULL, 0, 82.4 },
    { 41, 0x000001A96CA8D8BFULL, 0, 83.0 },
    { 42, 0x000003D94CD64D04ULL, 0, 96.1 },
    { 43, 0x00000735FB1829DAULL, 0, 90.2 },
    { 44, 0x00000E7A54A8C1B1ULL, 0, 90.5 },
    { 45, 0x00001A8B1F2F3E79ULL, 0, 82.8 },
    { 46, 0x000034A942DC4E9DULL, 0, 82.3 },
    { 47, 0x00006B1E91A44B3EULL, 0, 83.7 },
    { 48, 0x0000E140F0F14CB4ULL, 0, 87.9 },
    { 49, 0x0001A87F90BD4E8DULL, 0, 83.0 },
    { 50, 0x00034A65911FA071ULL, 0, 82.3 },
    // Harder puzzles (51-70) - solved 2019-2024
    { 51, 0x000693E219C88E27ULL, 0, 82.5 },
    { 52, 0x000E57B66EB0E33CULL, 0, 89.4 },
    { 53, 0x001A979E7C76BAFEULL, 0, 82.8 },
    { 54, 0x0035E66BD52F8EDEULL, 0, 83.7 },
    { 55, 0x006FD0A8B3E90F9BULL, 0, 87.1 },
    { 56, 0x00EB2C5513FBE04DULL, 0, 91.9 },
    { 57, 0x01A838B13505B26DULL, 0, 82.8 },
    { 58, 0x0340326E610B7D79ULL, 0, 81.2 },
    { 59, 0x068A52C7D45FF8C7ULL, 0, 81.6 },
    { 60, 0x0D916CE8A63E1A59ULL, 0, 84.7 },
    { 61, 0x1A96CA8D8BF31BA9ULL, 0, 83.0 },
    { 62, 0x34A65911FA070A65ULL, 0, 82.3 },
    { 63, 0x6AC3875A936C0595ULL, 0, 83.5 },
    { 64, 0xD2C55C00E3A2C889ULL, 0, 82.4 },
    { 65, 0x122FCA143C05E495ULL, 1, 71.7 },  // First >64 bit puzzle
    { 66, 0x2EC18388D544004AULL, 2, 73.3 },  // Solved Sept 2024
    { 67, 0x6CD610B53CBA1AEBULL, 5, 85.5 },  // Solved Nov 2024
};
const size_t NUM_SOLVED_PUZZLES = sizeof(SOLVED_PUZZLES) / sizeof(SOLVED_PUZZLES[0]);

// ============================================================================
// Optimal DP Bits Calculation for Kangaroo Algorithm
// ============================================================================
// For Pollard's Kangaroo, distinguished points (DPs) are used to detect
// collisions between tame and wild kangaroos. The dp_bits parameter controls
// how many trailing zero bits define a DP.
// Key relationships:
// - Expected steps to solve: O(sqrt(range)) = 2^(puzzle_bits/2)
// - Expected steps between DPs: 2^dp_bits
// - Number of DPs each kangaroo finds: steps / 2^dp_bits
// - Total DPs stored: num_kangaroos * steps_per_kangaroo / 2^dp_bits
// Optimal dp_bits balances:
// - Too low: Excessive DP storage, memory pressure, lookup overhead
// - Too high: Too few DPs, may miss collision, extra steps needed
// Formula: dp_bits = log2(sqrt(range) / num_kangaroos) + headroom
//        = (puzzle_bits / 2) - log2(num_kangaroos) + headroom
// The headroom constant (typically 4-8) accounts for:
// - Ensuring sufficient DP density for reliable collision detection
// - Memory efficiency vs collision probability trade-off
// ============================================================================
//
// T3.12: the calculate_optimal_dp_bits body that used to live in this TU's
// anonymous namespace has moved to runtime/puzzle_solver_helpers.hpp as a
// header-inline so puzzle_solver.cpp and puzzle_solver_kangaroo.cpp share
// one source of truth. Nothing in this TU referenced the file-scope copy,
// so the move is a pure dedup.

}  // namespace

// =============================================================================
// SHARED FREE-FUNCTION HELPERS (extern linkage)
// =============================================================================
// format_number / format_number_human / normalize_path / check_balance_async
// were file-scope helpers in main.cpp. The previous A.3 commits forward-
// declared them at the top of src/ui/interactive_ui.cpp and
// src/runtime/brain_wallet_runner.cpp on the understanding that the
// definitions would migrate here in the puzzle_solver extraction. This is
// that commit.
// They are intentionally kept at namespace global scope (not in
// collider::runtime::) to match the existing forward-decls without
// touching three other translation units.

/**
 * Format large numbers with commas.
 * OPTIMIZED: O(n) implementation instead of O(n²) string insertions.
 */
std::string format_number(uint64_t n) {
    if (n == 0) return "0";

    // Build digits in reverse order, then reverse
    std::string result;
    result.reserve(26);  // Max uint64 is 20 digits + 6 commas

    int digit_count = 0;
    while (n > 0) {
        if (digit_count > 0 && digit_count % 3 == 0) {
            result.push_back(',');
        }
        result.push_back('0' + (n % 10));
        n /= 10;
        digit_count++;
    }

    std::reverse(result.begin(), result.end());
    return result;
}

// format_number_human, normalize_path now live in runtime/format.hpp.
// check_balance_async lives in runtime/balance.{hpp,cpp}.

// =============================================================================
// PUZZLE ANALYSIS (declared in core/puzzle_analysis.hpp)
// =============================================================================
// analyze_puzzle / print_puzzle_analysis / get_best_puzzle were declared in
// core/puzzle_analysis.hpp as a transitional step in earlier A.3 commits.
// Their bodies live here now (they used to live in main.cpp).

/**
 * Analyze a single puzzle for ROI.
 */
PuzzleAnalysis analyze_puzzle(const collider::PuzzleInfo* puzzle, double gpu_speed_mkeys) {
    PuzzleAnalysis result;
    result.number = puzzle->number;
    result.bits = puzzle->bits;
    result.btc_reward = puzzle->btc_reward;
    result.has_pubkey = !puzzle->public_key_hex.empty();

    if (result.has_pubkey) {
        // Kangaroo: O(sqrt(2^N)) = O(2^(N/2))
        result.algorithm = "Kangaroo";
        result.complexity_bits = puzzle->bits / 2.0;
    } else {
        // Brute Force: O(2^N) - but we search 50% on average
        result.algorithm = "BruteForce";
        result.complexity_bits = puzzle->bits - 1.0;  // Average case
    }

    // ROI score: higher reward / lower complexity = better
    // Using log scale: reward / 2^complexity_bits
    // But for ranking we can use: log2(reward) - complexity_bits
    double log2_reward = log2(result.btc_reward);
    result.roi_score = log2_reward - result.complexity_bits;

    // Estimate GPU years (very rough)
    // Operations = 2^complexity_bits
    // Speed = gpu_speed_mkeys * 1e6 keys/sec
    // Years = operations / (speed * 86400 * 365)
    double operations = pow(2.0, result.complexity_bits);
    double keys_per_year = gpu_speed_mkeys * 1e6 * 86400.0 * 365.0;
    result.estimated_gpu_years = operations / keys_per_year;

    // Classify feasibility
    if (result.estimated_gpu_years < 0.1) {
        result.feasibility = "RECOMMENDED";
    } else if (result.estimated_gpu_years < 1.0) {
        result.feasibility = "VIABLE";
    } else if (result.estimated_gpu_years < 100.0) {
        result.feasibility = "DIFFICULT";
    } else {
        result.feasibility = "INFEASIBLE";
    }

    return result;
}

/**
 * Analyze all unsolved puzzles and print ranking.
 */
void print_puzzle_analysis(double gpu_speed_mkeys) {
    auto unsolved = collider::PuzzleDatabase::get_unsolved();
    std::vector<PuzzleAnalysis> analyses;

    for (const auto* puzzle : unsolved) {
        analyses.push_back(analyze_puzzle(puzzle, gpu_speed_mkeys));
    }

    // Sort by ROI score (higher = better)
    std::sort(analyses.begin(), analyses.end(),
              [](const PuzzleAnalysis& a, const PuzzleAnalysis& b) {
                  return a.roi_score > b.roi_score;
              });

    std::cout << "\n";
    std::cout << "+============================================================================+\n";
    std::cout << "|                    PUZZLE ANALYSIS - RANKED BY ROI                        |\n";
    std::cout << "+============================================================================+\n";
    std::cout << "| Rank | Puzzle | Bits | BTC    | Algorithm  | Complexity | Est.Time  | Status      |\n";
    std::cout << "+------+--------+------+--------+------------+------------+-----------+-------------+\n";

    int rank = 1;
    for (const auto& a : analyses) {
        std::string time_str;
        if (a.estimated_gpu_years < 0.01) {
            time_str = "<1 week";
        } else if (a.estimated_gpu_years < 0.1) {
            time_str = std::to_string((int)(a.estimated_gpu_years * 52)) + " weeks";
        } else if (a.estimated_gpu_years < 1.0) {
            time_str = std::to_string((int)(a.estimated_gpu_years * 12)) + " months";
        } else if (a.estimated_gpu_years < 1000) {
            time_str = std::to_string((int)a.estimated_gpu_years) + " years";
        } else {
            time_str = ">1000 yrs";
        }

        // Truncate to fit columns
        if (time_str.length() > 9) time_str = time_str.substr(0, 9);

        std::cout << "| " << std::setw(4) << rank << " | "
                  << std::setw(6) << a.number << " | "
                  << std::setw(4) << a.bits << " | "
                  << std::setw(6) << std::fixed << std::setprecision(1) << a.btc_reward << " | "
                  << std::setw(10) << a.algorithm << " | "
                  << "2^" << std::setw(7) << std::setprecision(1) << a.complexity_bits << " | "
                  << std::setw(9) << time_str << " | "
                  << std::setw(11) << a.feasibility << " |\n";
        rank++;

        // Only show top 20
        if (rank > 20) {
            std::cout << "| ...  | (showing top 20 of " << analyses.size() << " unsolved puzzles)                             |\n";
            break;
        }
    }

    std::cout << "+============================================================================+\n";
    std::cout << "\nNotes:\n";
    std::cout << "  - Kangaroo algorithm is O(sqrt(n)) and requires known public key\n";
    std::cout << "  - BruteForce is O(n) for unknown public keys\n";
    std::cout << "  - Time estimates assume " << (int)gpu_speed_mkeys << " MKeys/sec GPU speed\n";
    std::cout << "  - Puzzles #135, #140, #145, #150, #155, #160 have exposed public keys\n";
    std::cout << "\n";

    if (!analyses.empty()) {
        std::cout << "RECOMMENDATION: Puzzle #" << analyses[0].number
                  << " (" << analyses[0].algorithm << ", " << analyses[0].feasibility << ")\n\n";
    }
}

/**
 * Get the best puzzle to solve based on ROI analysis.
 * Returns puzzle number, or 0 if none suitable.
 */
int get_best_puzzle(double gpu_speed_mkeys) {
    auto unsolved = collider::PuzzleDatabase::get_unsolved();
    if (unsolved.empty()) return 0;

    double best_roi = -1e9;
    int best_puzzle = 0;

    for (const auto* puzzle : unsolved) {
        PuzzleAnalysis a = analyze_puzzle(puzzle, gpu_speed_mkeys);

        // Select puzzle with best ROI score (highest reward / lowest complexity)
        // Note: All remaining puzzles are extremely difficult, but some have
        // better odds than others (especially those with known public keys)
        if (a.roi_score > best_roi) {
            best_roi = a.roi_score;
            best_puzzle = puzzle->number;
        }
    }

    return best_puzzle > 0 ? best_puzzle : unsolved[0]->number;
}

// =============================================================================
// PUZZLE-MODE / BENCHMARK RUNTIME ENTRY POINTS
// =============================================================================

namespace collider::runtime {

// Display the project banner with mode-aware stats. Pre-dispatch UI
// shared by run_benchmark and run_puzzle_mode. For puzzle mode we also
// do smart-selection here (mutating args.puzzle_kangaroo and
// args.puzzle_number when --no-smart was not specified) since the
// banner stats need the final puzzle number to display reward/bits.
namespace {

void display_dispatch_banner(Arguments& args, const GPUDetectionResult& gpu_info) {
    ::collider::ui::BannerConfig banner_config;
    banner_config.enable_animation = !args.verbose;
    banner_config.enable_color = true;
    banner_config.animation_frames = 2;
    banner_config.frame_delay_ms = 100;

    // Set operation mode for context-aware display
    if (args.puzzle_mode) {
        banner_config.mode = ::collider::ui::OperationMode::PUZZLE_SEARCH;
    } else if (args.benchmark) {
        banner_config.mode = ::collider::ui::OperationMode::BENCHMARK;
    } else {
        banner_config.mode = ::collider::ui::OperationMode::PUZZLE_SEARCH;  // Default
    }

    ::collider::ui::BannerStats banner_stats;
    banner_stats.gpu_count = gpu_info.device_count;
    banner_stats.gpu_names = gpu_info.gpu_names;
    banner_stats.backend = gpu_info.backend;
    banner_stats.estimated_speed = gpu_info.estimated_speed;
    banner_stats.version = "1.4.0";

    // Smart-selection + puzzle-info banner population happen only in
    // puzzle mode. Both used to mutate args in place; preserve that.
    if (args.puzzle_mode) {
        const PuzzleInfo* puzzle = nullptr;
        if (args.puzzle_all_unsolved) {
            // Show first unsolved puzzle in banner when in auto-progression mode
            auto unsolved = PuzzleDatabase::get_unsolved();
            if (!unsolved.empty()) {
                puzzle = unsolved[0];
            }
        } else if (args.puzzle_number == 0) {
            // Smart puzzle selection: choose best ROI puzzle
            if (args.smart_select) {
                int best = ::get_best_puzzle(400.0);
                if (best > 0) {
                    args.puzzle_number = best;
                    puzzle = PuzzleDatabase::get_puzzle(best);
                    std::cout << "[*] Smart Selection: Puzzle #" << best;
                    // --pubkey override is also a "pubkey known" signal.
                    bool pubkey_known = !args.puzzle_pubkey.empty()
                                        || (puzzle && !puzzle->public_key_hex.empty());
                    if (pubkey_known) {
                        std::cout << " (Kangaroo - pubkey known)";
                        args.puzzle_kangaroo = true;  // Auto-enable Kangaroo for pubkey puzzles
                    } else {
                        std::cout << " (Brute Force - no pubkey)";
                    }
                    std::cout << "\n";
                    std::cout << "    Use --no-smart to disable smart selection\n";
                    std::cout << "    Use --analyze to see full puzzle ranking\n\n";
                } else {
                    // Fallback to first unsolved
                    auto unsolved = PuzzleDatabase::get_unsolved();
                    if (!unsolved.empty()) {
                        puzzle = unsolved[0];
                        args.puzzle_number = puzzle->number;
                    } else {
                        std::cerr << "[!] Error: No unsolved puzzles!\n";
                        // Caller (run_puzzle_mode) will see puzzle_number==0
                        // and bail; we let it through rather than throw.
                    }
                }
            } else {
                // Legacy: sequential selection (easiest first)
                auto unsolved = PuzzleDatabase::get_unsolved();
                if (!unsolved.empty()) {
                    puzzle = unsolved[0];
                    args.puzzle_number = puzzle->number;
                    std::cout << "[*] Auto-selected puzzle: #" << puzzle->number
                              << " (" << puzzle->bits << "-bit)\n\n";
                } else {
                    std::cerr << "[!] Error: No unsolved puzzles!\n";
                }
            }
        } else {
            puzzle = PuzzleDatabase::get_puzzle(args.puzzle_number);
            if (puzzle && puzzle->solved) {
                std::cout << "\n";
                std::cout << "\033[33m[*] Testing Mode\033[0m"
                          << " - Puzzle #" << args.puzzle_number << " is already SOLVED\n";
                std::cout << "    Known solution: " << puzzle->solution_hex << "\n";
                std::cout << "    Use this mode to verify implementation correctness.\n\n";
            }
        }
        if (puzzle) {
            banner_stats.puzzle_number = puzzle->number;
            banner_stats.puzzle_bits = puzzle->bits;
            banner_stats.puzzle_reward = puzzle->btc_reward;
        }
    }

    if (args.debug) std::cout << "[DEBUG] Displaying banner...\n" << std::flush;
    ::collider::ui::display_banner(banner_stats, banner_config);
    if (args.debug) std::cout << "[DEBUG] Banner displayed.\n" << std::flush;
}

}  // namespace

int run_benchmark(const Arguments& args_in, const GPUDetectionResult& gpu_info) {
    // display_dispatch_banner takes a mutable Arguments& so the puzzle-mode
    // smart-selection branch can populate puzzle_number/puzzle_kangaroo.
    // For --benchmark mode that branch is never entered, so the copy is
    // structurally inert; we keep it to avoid two near-identical helpers.
    Arguments args = args_in;
    display_dispatch_banner(args, gpu_info);

#ifndef COLLIDER_PRO
        // Free benchmark: SHA-256 throughput on CPU + GPU/backend info.
        // Body lives in puzzle_solver_benchmark.cpp; numbers are
        // identical, only the call site changed.
        return detail::run_sha256_only_benchmark(args, gpu_info);
#else
        // PRO --benchmark: delegate to the shared bench_pipeline core so
        // the output table is byte-identical to benchmarks/bench_gpu_pipeline.
        // This is the source of truth for the throughput numbers referenced
        // in docs/PRO.md and README.md. The benchmark drives the production
        // fused brain-wallet kernel directly on a pre-loaded fixed-stride
        // device buffer, isolating each pipeline stage (SHA-256, secp256k1
        // mul, hash160, bloom probe) for stage-specific rates and reporting
        // the end-to-end fused rate that operators see in live scans.
        {
            namespace boxui = ::collider::ui::box;
            std::cout << "\n";
            boxui::top(std::cout);
            boxui::centered(std::cout, "GPU PIPELINE BENCHMARK");
            boxui::top(std::cout);
            {
                std::ostringstream dur;
                dur << args.benchmark_seconds << " seconds per stage";
                boxui::kv(std::cout, "Duration", dur.str());
            }
            {
                std::ostringstream gpu_list;
                for (size_t i = 0; i < args.gpu_ids.size() && i < 8; i++) {
                    gpu_list << args.gpu_ids[i];
                    if (i < args.gpu_ids.size() - 1) gpu_list << ",";
                }
                boxui::kv(std::cout, "GPUs", gpu_list.str());
            }
            boxui::kv(std::cout, "Batch", format_number(args.batch_size));
            boxui::bottom(std::cout);
            std::cout << "\n";
        }

#ifdef COLLIDER_USE_CUDA
        // Run the shared pipeline benchmark per requested GPU. Stage-by-
        // stage timings come from sha256_batch, secp256k1_batch_mul_simple,
        // ripemd160_batch, and h160_bloom_batch_check_auto driven against
        // their own on-device buffers; the end-to-end number comes from
        // fused_brain_wallet_batch_fixed_stride, which is the kernel the
        // brain-wallet runner uses in production.
        ::collider::runtime::bench::PipelineBenchConfig bcfg;
        bcfg.bench_seconds = args.benchmark_seconds > 0 ? args.benchmark_seconds : 30;
        bcfg.batch_size = args.batch_size > 0 ? args.batch_size : 4'000'000;
        bcfg.measure_stages = true;

        double aggregate_keys_per_sec = 0.0;
        bool any_ok = false;
        for (int gpu_id : args.gpu_ids) {
            bcfg.gpu_id = gpu_id;
            auto result = ::collider::runtime::bench::run_pipeline_benchmark(
                bcfg, /*verbose=*/true);
            ::collider::runtime::bench::print_result_table(result);
            if (result.ok) {
                aggregate_keys_per_sec += result.fused_keys_per_sec;
                any_ok = true;
            }
        }

        if (args.gpu_ids.size() > 1 && any_ok) {
            std::cout << "\nAggregate (sum of fused rates across "
                      << args.gpu_ids.size() << " GPUs): "
                      << format_rate(aggregate_keys_per_sec) << "\n";
        }
        std::cout << "\nThese numbers reflect the production fused kernel on the\n"
                  << "configured hardware. Reproduce with the standalone driver:\n"
                  << "  bench_gpu_pipeline --time " << bcfg.bench_seconds
                  << " --gpu " << (args.gpu_ids.empty() ? 0 : args.gpu_ids[0]) << "\n";
        return any_ok ? 0 : 1;
#else
        std::cout << "[!] Built without CUDA: full-pipeline benchmark unavailable.\n"
                     "    Rebuild with -DCOLLIDER_USE_CUDA=ON to enable.\n";
        return 1;
#endif  // COLLIDER_USE_CUDA
#endif // COLLIDER_PRO (benchmark)
}

int run_puzzle_mode(const Arguments& args_in, const GPUDetectionResult& gpu_info) {
    // Local mutable copy: original main.cpp body modifies args.batch_size
    // (calibration result) and args.puzzle_kangaroo (auto-select). Match
    // that behavior here without forcing the caller to pass a non-const
    // ref through dispatch.
    Arguments args = args_in;

    // Display banner + run puzzle smart-selection (mutates args.puzzle_*).
    display_dispatch_banner(args, gpu_info);

    // removed verbose-mode solved-puzzle-zone-distribution
    // printout. The Center-Heavy strategy that consumed it is gone; the
    // distribution itself was selection-bias artifact, not signal.
    if (args.analyze_puzzles) {
        ::print_puzzle_analysis(400.0);  // Assume 400 MKeys/sec GPU speed
        return 0;
    }

    // Logger singleton is consumed by the extracted brute-force helper.
    // run_puzzle_mode itself no longer touches it directly.

        // Load config for progress saving
        UserConfig config;
        config.load();

        // GPU batch-size calibration (first run or --calibrate / --force-calibrate).
        // Mutates args.batch_size in place; no-op on non-CUDA builds.
        detail::maybe_run_calibration(args, config);

        // Build list of puzzles to solve (--all-unsolved / --auto-next / --puzzle N).
        std::vector<int> puzzles_to_solve = detail::build_puzzle_worklist(args);
        if (args.puzzle_all_unsolved && puzzles_to_solve.empty()) {
            // build_puzzle_worklist returns empty when --all-unsolved is set
            // and there are no unsolved puzzles left. Original behavior:
            // print + return 0.
            std::cout << "[*] No unsolved puzzles found - all puzzles have been solved!\n";
            return 0;
        }

        // Loop through each puzzle. Single iteration unless --all-unsolved
        // OR --auto-next built a multi-puzzle worklist above.
        const bool is_multi_puzzle = args.puzzle_all_unsolved || args.puzzle_auto_next;
        for (size_t puzzle_idx = 0; puzzle_idx < puzzles_to_solve.size() && !g_shutdown; puzzle_idx++) {
            int current_puzzle = puzzles_to_solve[puzzle_idx];

            // Show progress for auto-progression mode
            if (is_multi_puzzle && puzzle_idx > 0) {
                namespace boxui = ::collider::ui::box;
                std::cout << "\n\n";
                boxui::top(std::cout);
                {
                    std::ostringstream label;
                    label << "AUTO-PROGRESSION: Moving to puzzle #" << current_puzzle
                          << " (" << (puzzle_idx + 1) << "/" << puzzles_to_solve.size() << ")";
                    boxui::centered(std::cout, label.str());
                }
                boxui::bottom(std::cout);
            }

        // Get puzzle info
        const PuzzleInfo* puzzle = PuzzleDatabase::get_puzzle(current_puzzle);

        std::cout << "\n";
        ::collider::ui::box::top(std::cout);
        ::collider::ui::box::centered(std::cout, "BITCOIN PUZZLE CHALLENGE (1000 BTC)");
        ::collider::ui::box::top(std::cout);

        // Resolve range / target / hash160 for this puzzle. Mirrors the
        // original inline block; on unknown puzzle + no custom range it
        // logs and returns 1 just like before.
        detail::PuzzleTarget tgt;
        if (!detail::resolve_puzzle_target(args, current_puzzle, puzzle, tgt)) {
            return 1;
        }
        const int bits = tgt.bits;

        {
            std::ostringstream bits_str;
            bits_str << bits << " (2^" << (bits - 1) << " keys in range)";
            ::collider::ui::box::kv(std::cout, "Bits", bits_str.str());
        }
        ::collider::ui::box::kv(std::cout, "Target",  tgt.target_address);
        ::collider::ui::box::kv(std::cout, "Search",  args.puzzle_random ? "Random" : "Sequential");
        ::collider::ui::box::kv(std::cout, "Backend", gpu_info.backend);
        ::collider::ui::box::top(std::cout);
        ::collider::ui::box::kv(std::cout, "Range Start", tgt.range_start.to_hex());
        ::collider::ui::box::kv(std::cout, "Range End",   tgt.range_end.to_hex());
        ::collider::ui::box::top(std::cout);
        std::cout << "\n";

        detail::print_search_space_analysis(bits);

        // ======================================================================
        // ALGORITHM SELECTION: pick the right thing for this puzzle, even
        // if the user is mid-way through a multi-puzzle worklist.
        // ======================================================================
        // A puzzle is kangaroo-able iff we have the target's compressed
        // public key from one of: (a) the bundled puzzle_history.json,
        // (b) the --pubkey CLI override, (c) the puzzle.pubkey config.yml
        // field. Non-multiples of 5 in 71-160 (and certain unsolved gaps)
        // have NEVER had a spending tx, so the pubkey is not knowable to
        // anyone -- forcing --kangaroo on those is impossible, only brute
        // force can run. Auto-progression worklists hit those mid-stream,
        // so --kangaroo is gracefully demoted to brute force per-puzzle
        // instead of failing the whole batch.
        detail::select_algorithm(args, puzzle, bits, is_multi_puzzle);

        // ======================================================================
        // SOLVE DISPATCH
        // ======================================================================
        // The KANGAROO / BRUTE FORCE inline blocks (~1290 lines) live in
        // puzzle_solver_kangaroo.cpp and puzzle_solver_bruteforce.cpp.
        // The helpers return PuzzleStepResult so this dispatcher can
        // translate the original's mix of `return 0/1/64`, `continue`,
        // and fall-through into a single control-flow pattern. See
        // puzzle_solver_helpers.hpp for the enum semantics. Hard-preserve
        // invariants (secure_wipe, secure_open ofstream, SearchState v4,
        // --resume-kangaroo, SIGINT save, range-reject) live inside the
        // extracted helpers and are individually documented there.
        detail::PuzzleIterContext ctx{
            args, gpu_info, current_puzzle, puzzle, tgt, is_multi_puzzle
        };
        detail::PuzzleStepResult step;
        if (args.puzzle_kangaroo && bits > 40) {
            step = detail::run_kangaroo_solve(ctx);
        } else {
            step = detail::run_bruteforce_solve(ctx);
        }

        switch (step) {
            case detail::PuzzleStepResult::FatalError:
                return 1;
            case detail::PuzzleStepResult::UsageError:
                return 64;  // EX_USAGE
            case detail::PuzzleStepResult::SkipPuzzle:
                // > 128-bit brute-force reject in multi-puzzle mode: skip
                // and continue. (Single-puzzle mode never reaches here;
                // run_bruteforce_solve returns UsageError instead.)
                continue;
            case detail::PuzzleStepResult::SolvedExitOrContinue:
            case detail::PuzzleStepResult::StoppedExitOrContinue:
                // Every per-puzzle solve path lands here. Single-puzzle
                // mode: return 0. Multi-puzzle mode (--all-unsolved /
                // --auto-next): continue to the next puzzle in the
                // worklist. Previously the RCKangaroo / MultiGPU Kangaroo
                // / GPU brute paths returned unconditionally instead of
                // continuing, which terminated --all-unsolved scans
                // prematurely after the first puzzle on those code paths.
                if (is_multi_puzzle) continue;
                return 0;
            case detail::PuzzleStepResult::FallThrough:
                // Reserved for future helpers that hand control back to
                // the per-puzzle loop. Today no helper returns this; if
                // one does, the loop naturally iterates to the next puzzle.
                break;
        }

        } // End of puzzle iteration for loop

        // All puzzles completed (auto-progression) or single puzzle done
        if (is_multi_puzzle && puzzles_to_solve.size() > 1) {
            namespace boxui = ::collider::ui::box;
            std::cout << "\n";
            boxui::top(std::cout);
            boxui::centered(std::cout, "AUTO-PROGRESSION COMPLETE");
            boxui::top(std::cout);
            {
                std::ostringstream msg;
                msg << "  All " << puzzles_to_solve.size() << " puzzles have been processed.";
                boxui::line(std::cout, msg.str());
            }
            boxui::bottom(std::cout);
            std::cout << "\n";
        }

        return 0;
}

}  // namespace collider::runtime
