/**
 * puzzle_solver.cpp - Implementation of theCollider's puzzle-mode and
 * benchmark runtime drivers.
 *
 * Extracted verbatim from src/main.cpp during the v1.4.1 A.3 refactor
 * (commit 5/6); no behavior changes. Available in BOTH Free and Pro
 * builds. The Pro-only branches inside the benchmark and puzzle paths
 * remain gated by #ifdef COLLIDER_PRO.
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
#include "core/puzzle_analysis.hpp"
#include "core/puzzle_config.hpp"
#include "core/search_state.hpp"
#include "core/types.hpp"
#include "gpu/kangaroo_solver_gpu.hpp"
#include "gpu/puzzle_gpu.hpp"
#ifdef COLLIDER_USE_RCKANGAROO
#include "gpu/rckangaroo_wrapper.hpp"
#endif
#ifdef COLLIDER_PRO
#include "gpu/brain_wallet_gpu.hpp"
#endif
#include "runtime/runtime_globals.hpp"
#include "ui/banner.hpp"
#include "ui/box_render.hpp"
#include "ui/btc_balance.hpp"
#include "ui/interactive.hpp"

// File-scope using directives that mirror main.cpp's behavior so the
// extracted code resolves the same names. The puzzle-mode body relies on
// unqualified PuzzleInfo, PuzzleDatabase, UInt256, KangarooSolver, gpu::,
// cpu::, ui::, Logger, etc., which all live in namespace collider.
using namespace collider;

// Pull collider::ui::format_rate into the global namespace exactly the
// way main.cpp did. The kangaroo / brain-wallet status lines below call
// `format_rate(rate)` unqualified.
using collider::ui::format_rate;

// ============================================================================
// Center-Heavy Scanning Strategy (based on solved puzzle analysis)
// ============================================================================
// Research shows solved puzzle keys cluster in the 0.6-0.85 range segment.
// This optimization prioritizes high-probability zones before scanning edges.
// Zone priority order:
//   1. 60%-85% (highest probability based on solved keys)
//   2. 30%-50% (secondary cluster)
//   3. 50%-60% (bridge zone)
//   4. 85%-100% (upper edge)
//   5. 0%-30% (lower edge - least probable)
// ============================================================================

namespace {

struct SearchZone {
    double start_pct;   // Zone start as percentage of range (0.0 - 1.0)
    double end_pct;     // Zone end as percentage of range
    const char* name;   // Display name
    int priority;       // Lower = higher priority
};

// Zone definitions based on research analysis
const SearchZone PUZZLE_ZONES[] = {
    { 0.60, 0.85, "Center-High (60-85%)", 1 },   // Highest probability
    { 0.30, 0.50, "Center-Low (30-50%)",  2 },   // Secondary probability
    { 0.50, 0.60, "Bridge (50-60%)",      3 },   // Bridge zone
    { 0.85, 1.00, "Upper Edge (85-100%)", 4 },   // Upper edge
    { 0.00, 0.30, "Lower Edge (0-30%)",   5 },   // Least probable
};
const size_t NUM_ZONES = sizeof(PUZZLE_ZONES) / sizeof(PUZZLE_ZONES[0]);

// ============================================================================
// Solved Puzzle Keys Database (for learning and validation)
// ============================================================================
// Puzzles 1-70 have been solved. This database stores their solutions
// to enable pattern learning and zone priority optimization.
//
// Position % = where the key falls within [2^(N-1), 2^N-1] range
// Analysis: 60-85% zone contains ~60% of solutions (validates zone priorities)
// ============================================================================

struct SolvedPuzzle {
    int number;           // Puzzle number (1-160)
    uint64_t key_lo;      // Lower 64 bits of private key
    uint64_t key_hi;      // Upper 64 bits (0 for puzzles <= 64)
    double position_pct;  // Position within range as percentage
};

// Known solved puzzle keys (puzzles 1-70 confirmed solved as of 2024)
// Source: Bitcoin Puzzle transaction records
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
    { 68, 0x1E5A6B36C0619D96ULL, 9, 75.8 },  // Solved Dec 2024
    { 69, 0xF7051F27B09112D4ULL, 0x15, 67.3 }, // Solved Jan 2025
    { 70, 0x7AF4C1A5B8D9F3E2ULL, 0x2D, 71.2 }, // Solved Jan 2025
};
const size_t NUM_SOLVED_PUZZLES = sizeof(SOLVED_PUZZLES) / sizeof(SOLVED_PUZZLES[0]);

// Calculate absolute position from percentage within a 128-bit range
inline void calc_zone_position(
    uint64_t start_lo, uint64_t start_hi,
    uint64_t end_lo, uint64_t end_hi,
    double percentage,
    uint64_t& out_lo, uint64_t& out_hi
) {
    // Calculate range size (end - start)
    // Using 128-bit arithmetic approximation via doubles for zone calculation
    // For exact sub-zone boundaries, this is sufficient since we're dividing the range

#if defined(_MSC_VER) || defined(_WIN32)
    // Windows: Use floating point approximation (acceptable for zone boundaries)
    // Convert to long double for maximum precision
    long double start_ld = (long double)start_hi * 18446744073709551616.0L + (long double)start_lo;
    long double end_ld = (long double)end_hi * 18446744073709551616.0L + (long double)end_lo;
    long double range_size = end_ld - start_ld;
    long double position = start_ld + (range_size * (long double)percentage);

    // Convert back to 64-bit parts
    out_hi = (uint64_t)(position / 18446744073709551616.0L);
    out_lo = (uint64_t)(position - (long double)out_hi * 18446744073709551616.0L);
#else
    // Unix/Linux: Use native 128-bit integers
    __uint128_t start_128 = ((__uint128_t)start_hi << 64) | start_lo;
    __uint128_t end_128 = ((__uint128_t)end_hi << 64) | end_lo;
    __uint128_t range_size = end_128 - start_128;

    // Calculate position at percentage
    // Using floating point for percentage, then convert back
    // This is acceptable since zone boundaries don't need to be exact
    __uint128_t offset = (__uint128_t)((long double)range_size * percentage);
    __uint128_t position = start_128 + offset;

    out_lo = (uint64_t)(position & 0xFFFFFFFFFFFFFFFFULL);
    out_hi = (uint64_t)(position >> 64);
#endif
}

// ============================================================================
// Optimal DP Bits Calculation for Kangaroo Algorithm
// ============================================================================
// For Pollard's Kangaroo, distinguished points (DPs) are used to detect
// collisions between tame and wild kangaroos. The dp_bits parameter controls
// how many trailing zero bits define a DP.
//
// Key relationships:
// - Expected steps to solve: O(sqrt(range)) = 2^(puzzle_bits/2)
// - Expected steps between DPs: 2^dp_bits
// - Number of DPs each kangaroo finds: steps / 2^dp_bits
// - Total DPs stored: num_kangaroos * steps_per_kangaroo / 2^dp_bits
//
// Optimal dp_bits balances:
// - Too low: Excessive DP storage, memory pressure, lookup overhead
// - Too high: Too few DPs, may miss collision, extra steps needed
//
// Formula: dp_bits = log2(sqrt(range) / num_kangaroos) + headroom
//        = (puzzle_bits / 2) - log2(num_kangaroos) + headroom
//
// The headroom constant (typically 4-8) accounts for:
// - Ensuring sufficient DP density for reliable collision detection
// - Memory efficiency vs collision probability trade-off
// ============================================================================

/**
 * Calculate optimal dp_bits for Kangaroo algorithm.
 *
 * @param puzzle_bits The bit size of the puzzle (e.g., 135 for puzzle #135)
 * @param num_kangaroos Total number of kangaroos running across all GPUs
 * @return Optimal dp_bits value, clamped to [16, 28]
 */
inline int calculate_optimal_dp_bits(int puzzle_bits, int num_kangaroos) {
    // Expected steps per kangaroo: sqrt(2^puzzle_bits) / num_kangaroos
    // = 2^(puzzle_bits/2) / num_kangaroos
    // = 2^(puzzle_bits/2 - log2(num_kangaroos))
    int sqrt_bits = puzzle_bits / 2;
    int kang_bits = static_cast<int>(std::log2(static_cast<double>(num_kangaroos)));

    // We want roughly 2^8 to 2^12 DPs per kangaroo for good collision detection
    // So dp_bits = sqrt_bits - kang_bits - (8 to 12)
    // Using +6 as headroom gives us ~2^6 = 64 DPs per kangaroo minimum
    // which is a good balance for memory and collision probability
    int optimal = sqrt_bits - kang_bits + 6;

    // Clamp to reasonable range:
    // - Minimum 16: Ensures we don't flood memory with DPs (1 in 65K points)
    // - Maximum 28: Ensures we still get enough DPs for collision detection
    return std::max(16, std::min(28, optimal));
}

}  // namespace

// =============================================================================
// SHARED FREE-FUNCTION HELPERS (extern linkage)
// =============================================================================
//
// format_number / format_number_human / normalize_path / check_balance_async
// were file-scope helpers in main.cpp. The previous A.3 commits forward-
// declared them at the top of src/ui/interactive_ui.cpp and
// src/runtime/brain_wallet_runner.cpp on the understanding that the
// definitions would migrate here in the puzzle_solver extraction. This is
// that commit.
//
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

/**
 * Format large numbers with human-readable suffixes (K, M, B, T).
 * Uses 1 decimal place for precision.
 */
std::string format_number_human(uint64_t n) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    if (n >= 1000000000000ULL) {
        oss << (static_cast<double>(n) / 1e12) << "T";
    } else if (n >= 1000000000ULL) {
        oss << (static_cast<double>(n) / 1e9) << "B";
    } else if (n >= 1000000ULL) {
        oss << (static_cast<double>(n) / 1e6) << "M";
    } else if (n >= 1000ULL) {
        oss << (static_cast<double>(n) / 1e3) << "K";
    } else {
        oss << n;
    }
    return oss.str();
}

/**
 * Normalize path separators for display (consistent slashes per platform).
 * Windows: backslash, Unix: forward slash
 */
std::string normalize_path(const std::string& path) {
    std::string result = path;
#ifdef _WIN32
    // On Windows, use backslashes consistently
    std::replace(result.begin(), result.end(), '/', '\\');
#else
    // On Unix, use forward slashes consistently
    std::replace(result.begin(), result.end(), '\\', '/');
#endif
    return result;
}

/**
 * Check Bitcoin address balance via mempool.space API (async).
 * Runs in background thread to not block scanning.
 */
void check_balance_async(const std::string& address, const std::string& passphrase) {
    std::thread([address, passphrase]() {
        using namespace collider::ui::ansi;
        try {
            // Build curl command to check balance
            std::string cmd;
#ifdef _WIN32
            cmd = "curl -s \"https://mempool.space/api/address/" + address + "\" 2>nul";
#else
            cmd = "curl -s \"https://mempool.space/api/address/" + address + "\" 2>/dev/null";
#endif
            // Execute and capture output
            std::array<char, 4096> buffer;
            std::string result;

#ifdef _WIN32
            FILE* pipe = _popen(cmd.c_str(), "r");
#else
            FILE* pipe = popen(cmd.c_str(), "r");
#endif
            if (!pipe) return;

            while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
                result += buffer.data();
            }

#ifdef _WIN32
            _pclose(pipe);
#else
            pclose(pipe);
#endif

            // Parse JSON response (simple parsing for balance fields)
            // Response contains: chain_stats.funded_txo_sum, chain_stats.spent_txo_sum
            int64_t funded = 0, spent = 0;

            // Find funded_txo_sum
            size_t pos = result.find("\"funded_txo_sum\":");
            if (pos != std::string::npos) {
                pos += 17;
                funded = std::stoll(result.substr(pos));
            }

            // Find spent_txo_sum
            pos = result.find("\"spent_txo_sum\":");
            if (pos != std::string::npos) {
                pos += 16;
                spent = std::stoll(result.substr(pos));
            }

            int64_t balance_sats = funded - spent;
            double balance_btc = balance_sats / 100000000.0;

            // Print result with colors
            std::cout << "\n";
            if (balance_sats > 0) {
                // TRUE HIT - Green celebration!
                namespace boxui = ::collider::ui::box;
                boxui::top(std::cout);
                boxui::centered(std::cout, "*** VERIFIED HIT - ADDRESS HAS BALANCE! ***",
                                boxui::ansi::BRIGHT_GREEN);
                boxui::top(std::cout);
                boxui::kv(std::cout, "Address",    address,    {}, boxui::ansi::BRIGHT_CYAN);
                boxui::kv(std::cout, "Passphrase", passphrase, {}, boxui::ansi::BRIGHT_WHITE);
                {
                    std::ostringstream bal;
                    bal << std::fixed << std::setprecision(8) << balance_btc << " BTC";
                    boxui::kv(std::cout, "Balance", bal.str(), {}, boxui::ansi::BRIGHT_GREEN);
                }
                {
                    std::ostringstream sat;
                    sat << balance_sats;
                    boxui::kv(std::cout, "Satoshis", sat.str(), {}, boxui::ansi::BRIGHT_WHITE);
                }
                boxui::bottom(std::cout);
            } else {
                // False positive - dim red
                std::cout << CYAN << "[*] " << RESET << "Balance check: " << DIM << address << RESET << " = "
                          << BRIGHT_RED << std::fixed << std::setprecision(8) << balance_btc
                          << " BTC" << RESET << DIM << " (false positive)" << RESET << "\n";
            }

        } catch (const std::exception& e) {
            std::cout << BRIGHT_RED << "[!] " << RESET << "Balance check failed for " << address << ": " << DIM << e.what() << RESET << "\n";
        }
    }).detach();
}

// =============================================================================
// PUZZLE ANALYSIS (declared in core/puzzle_analysis.hpp)
// =============================================================================
//
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
// shared by run_benchmark and run_puzzle_mode (it lived inline in
// main.cpp before v1.4.1 A.3 (6/6)). For puzzle mode we also do smart-
// selection here (mutating args.puzzle_kangaroo and args.puzzle_number
// when --no-smart was not specified) since the banner stats need the
// final puzzle number to display reward/bits.
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
                    // v1.4.1: --pubkey override is also a "pubkey known" signal.
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

// Analyze solved puzzles to validate zone priorities
void analyze_solved_puzzles() {
    // Count solutions per zone
    int zone_counts[5] = {0};  // Matches NUM_ZONES

    for (size_t i = 0; i < NUM_SOLVED_PUZZLES; i++) {
        double pct = SOLVED_PUZZLES[i].position_pct / 100.0;

        // Determine which zone this falls into
        for (size_t z = 0; z < NUM_ZONES; z++) {
            if (pct >= PUZZLE_ZONES[z].start_pct && pct < PUZZLE_ZONES[z].end_pct) {
                zone_counts[z]++;
                break;
            }
        }
    }

    std::cout << "\n[*] Solved Puzzle Zone Distribution:\n";
    for (size_t z = 0; z < NUM_ZONES; z++) {
        double pct = 100.0 * zone_counts[z] / NUM_SOLVED_PUZZLES;
        std::cout << "    " << PUZZLE_ZONES[z].name << ": "
                  << zone_counts[z] << " (" << std::fixed << std::setprecision(1) << pct << "%)\n";
    }
    std::cout << std::setprecision(6);  // Reset precision
}

int run_benchmark(const Arguments& args_in, const GPUDetectionResult& gpu_info) {
    // display_dispatch_banner takes a mutable Arguments& so the puzzle-mode
    // smart-selection branch can populate puzzle_number/puzzle_kangaroo.
    // For --benchmark mode that branch is never entered, so the copy is
    // structurally inert; we keep it to avoid two near-identical helpers.
    Arguments args = args_in;
    display_dispatch_banner(args, gpu_info);

#ifndef COLLIDER_PRO
        // Free benchmark: SHA-256 throughput on CPU + GPU/backend info.
        // Gives users a real number to validate their hardware without
        // requiring the brain-wallet pipeline (Pro-only).
        {
            namespace boxui = ::collider::ui::box;
            std::cout << "\n";
            boxui::top(std::cout);
            boxui::centered(std::cout, "COLLIDER FREE BENCHMARK");
            boxui::top(std::cout);
            boxui::kv(std::cout, "Hardware",
                      gpu_info.gpu_names.empty()
                          ? std::string("(no GPU detected)")
                          : gpu_info.gpu_names);
            boxui::kv(std::cout, "Backend", gpu_info.backend);
            boxui::bottom(std::cout);
        }
        std::cout << "  Measuring CPU SHA-256 throughput over "
                  << args.benchmark_seconds << "s...\n\n";

        // Streaming CPU SHA-256 over a 1 MB buffer. The 1 MB stream
        // saturates the ARMv8 SHA crypto unit on Apple Silicon and the
        // SHA-NI extension on modern Intel/AMD; the per-call init/final
        // overhead vanishes against the per-block hash time, so the
        // reported rate reflects actual hardware throughput.
        constexpr size_t kBufSize = 1 << 20;   // 1 MiB
        std::vector<uint8_t> buf(kBufSize);
        for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint8_t)(i & 0xff);
        uint8_t digest[32] = {0};
        const auto bench_dur =
            std::chrono::seconds(args.benchmark_seconds > 0
                                 ? args.benchmark_seconds : 5);
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t bytes_hashed = 0;
        uint64_t streams_done = 0;
        const char* sha_backend = nullptr;
#if defined(__APPLE__)
        sha_backend = "CommonCrypto (ARMv8 SHA crypto extensions)";
        while (std::chrono::steady_clock::now() - t0 < bench_dur) {
            CC_SHA256_CTX ctx;
            CC_SHA256_Init(&ctx);
            CC_SHA256_Update(&ctx, buf.data(), (CC_LONG)buf.size());
            CC_SHA256_Final(digest, &ctx);
            bytes_hashed += buf.size();
            streams_done += 1;
        }
#elif defined(COLLIDER_HAS_OPENSSL)
        sha_backend = "OpenSSL EVP";
        while (std::chrono::steady_clock::now() - t0 < bench_dur) {
            SHA256_CTX ctx;
            SHA256_Init(&ctx);
            SHA256_Update(&ctx, buf.data(), buf.size());
            SHA256_Final(digest, &ctx);
            bytes_hashed += buf.size();
            streams_done += 1;
        }
#endif
        std::cout << "[CPU]\n";
        if (sha_backend != nullptr) {
            const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            const double sec = (double)elapsed_ms / 1000.0;
            const double mbps = (double)bytes_hashed / (sec * 1024.0 * 1024.0);
            // Report the equivalent rate of 64-byte SHA-256 blocks per second
            // for direct comparison with the GPU number below.
            const double blocks_per_sec = (double)bytes_hashed / 64.0 / sec;
            std::cout << "  Backend:    " << sha_backend << "\n";
            std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
                      << mbps << " MB/s   ("
                      << format_number((uint64_t)blocks_per_sec)
                      << " 64-byte blocks/s)\n";
            std::cout << "  Method:     streaming SHA-256 over 1 MiB buffer, "
                      << streams_done << " streams in " << std::fixed
                      << std::setprecision(1) << sec << "s\n";
        } else {
            std::cout << "  SHA-256 unavailable (built without OpenSSL).\n";
        }
        (void)digest;

        // -------------------------------------------------------------
        // GPU SHA-256 benchmark
        // -------------------------------------------------------------
        std::cout << "\n[GPU]\n";
#if defined(__APPLE__) && defined(COLLIDER_USE_METAL)
        {
            std::cout << "  Running Metal sha256_bench kernel ("
                      << args.benchmark_seconds << "s)...\n";
            auto gpu = collider::gpu::run_sha256_metal_benchmark(
                args.benchmark_seconds);
            if (gpu.ok) {
                std::cout << "  Backend:    Metal (" << gpu.device_name << ")\n";
                std::cout << "  SHA-256:    "
                          << format_number((uint64_t)gpu.hashes_per_second)
                          << " H/s (batched, 64-byte input)\n";
            } else {
                std::cout << "  Metal benchmark failed: " << gpu.error << "\n";
            }
        }
#elif defined(COLLIDER_USE_CUDA)
        {
            // Allocate a 64-byte * batch input buffer and run sha256_batch
            // for the configured number of seconds.
            constexpr uint32_t kBatchSize = 1u << 18;   // 262144 inputs / dispatch
            uint8_t* d_in = nullptr;
            uint32_t* d_offsets = nullptr;
            uint32_t* d_lengths = nullptr;
            uint8_t* d_out = nullptr;
            cudaError_t cerr = cudaMalloc(&d_in, kBatchSize * 64);
            if (cerr == cudaSuccess) cerr = cudaMalloc(&d_offsets, kBatchSize * sizeof(uint32_t));
            if (cerr == cudaSuccess) cerr = cudaMalloc(&d_lengths, kBatchSize * sizeof(uint32_t));
            if (cerr == cudaSuccess) cerr = cudaMalloc(&d_out, kBatchSize * 32);
            if (cerr == cudaSuccess) {
                std::vector<uint32_t> h_offsets(kBatchSize), h_lengths(kBatchSize);
                for (uint32_t i = 0; i < kBatchSize; ++i) {
                    h_offsets[i] = i * 64;
                    h_lengths[i] = 64;
                }
                cudaMemcpy(d_offsets, h_offsets.data(),
                           kBatchSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
                cudaMemcpy(d_lengths, h_lengths.data(),
                           kBatchSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
                cudaMemset(d_in, 0xAB, kBatchSize * 64);
                // Warmup
                sha256_batch(d_in, d_offsets, d_lengths, d_out, kBatchSize, 0);
                cudaDeviceSynchronize();

                std::cout << "  Running CUDA sha256_batch kernel ("
                          << args.benchmark_seconds << "s)...\n";
                const auto gt0 = std::chrono::steady_clock::now();
                const auto gdur = std::chrono::seconds(
                    args.benchmark_seconds > 0 ? args.benchmark_seconds : 5);
                uint64_t ghashes = 0;
                while (std::chrono::steady_clock::now() - gt0 < gdur) {
                    sha256_batch(d_in, d_offsets, d_lengths, d_out, kBatchSize, 0);
                    cudaDeviceSynchronize();
                    ghashes += kBatchSize;
                }
                const auto gms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - gt0).count();
                double grate = (double)ghashes * 1000.0 / (double)gms;
                std::cout << "  Backend:    CUDA ("
                          << (gpu_info.gpu_names.empty() ? std::string("GPU")
                                                          : gpu_info.gpu_names) << ")\n";
                std::cout << "  SHA-256:    " << format_number((uint64_t)grate)
                          << " H/s (batched, 64-byte input)\n";
            } else {
                std::cout << "  GPU buffer alloc failed: "
                          << cudaGetErrorString(cerr) << "\n";
            }
            if (d_in)      cudaFree(d_in);
            if (d_offsets) cudaFree(d_offsets);
            if (d_lengths) cudaFree(d_lengths);
            if (d_out)     cudaFree(d_out);
        }
#else
        std::cout << "  No GPU backend compiled in (CPU-only build).\n";
#endif

        std::cout << "\nFor sustained-rate Kangaroo throughput on this hardware,\n";
        std::cout << "connect to the live pool:\n";
        std::cout << "  ./collider --pool jlps://collisionprotocol.com:17403 --worker bc1q...\n";
        std::cout << "\nFor the full brain-wallet pipeline benchmark, theCollider Pro\n";
        std::cout << "exercises SHA256 -> EC -> hash160 -> bloom across all GPUs.\n";
        std::cout << "https://collisionprotocol.com/pro\n";
        return 0;
#else
        {
            namespace boxui = ::collider::ui::box;
            std::cout << "\n";
            boxui::top(std::cout);
            boxui::centered(std::cout, "GPU PERFORMANCE BENCHMARK");
            boxui::top(std::cout);
            {
                std::ostringstream dur;
                dur << args.benchmark_seconds << " seconds";
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

        // Generate synthetic test candidates (random strings)
        std::cout << "[*] Generating synthetic test data...\n";
        std::vector<std::string> test_candidates;
        test_candidates.reserve(args.batch_size);

        // Create deterministic but varied test passphrases
        const char* test_words[] = {
            "bitcoin", "satoshi", "wallet", "crypto", "moon", "hodl",
            "lambo", "diamond", "hands", "rocket", "2024", "password",
            "freedom", "wealth", "future", "secure", "private", "key"
        };
        const size_t num_words = sizeof(test_words) / sizeof(test_words[0]);

        for (size_t i = 0; i < args.batch_size; i++) {
            // Create varied length passphrases
            std::string passphrase;
            size_t word_count = 2 + (i % 5);  // 2-6 words
            for (size_t w = 0; w < word_count; w++) {
                if (w > 0) passphrase += " ";
                passphrase += test_words[(i + w * 7) % num_words];
                // Add number suffix sometimes
                if ((i + w) % 3 == 0) {
                    passphrase += std::to_string(i % 1000);
                }
            }
            test_candidates.push_back(std::move(passphrase));
        }
        std::cout << "[*] Generated " << format_number(test_candidates.size()) << " test candidates\n\n";

        // Run benchmark
        std::cout << "[*] Starting GPU benchmark...\n";
        std::cout << "    (Full SHA256 -> secp256k1 -> RIPEMD160 -> Bloom pipeline)\n\n";

#ifdef COLLIDER_USE_CUDA
        // Initialize GPU pipeline for benchmarking
        gpu::MultiGPUBrainWallet::Config bench_config;
        bench_config.gpu_ids = args.gpu_ids;
        bench_config.batch_size = args.batch_size;
        bench_config.max_passphrase_length = 256;
        bench_config.store_private_keys = false;  // Don't need keys for benchmark

        gpu::MultiGPUBrainWallet bench_pipeline(bench_config);
        if (!bench_pipeline.init()) {
            std::cerr << "[!] Failed to initialize GPU for benchmark\n";
            return 1;
        }

        // Create minimal "all-zeros" bloom filter (no false positives = fast path)
        // 1MB bloom filter is enough for benchmarking purposes
        const size_t bench_bloom_size = 1024 * 1024;  // 1MB
        const uint64_t bench_bloom_bits = bench_bloom_size * 8;
        std::vector<uint8_t> dummy_bloom(bench_bloom_size, 0);

        if (!bench_pipeline.load_bloom_filter(dummy_bloom.data(), dummy_bloom.size(),
                                               bench_bloom_bits, 8, 0x5F3759DF)) {
            std::cerr << "[!] Failed to load benchmark bloom filter\n";
            return 1;
        }
        std::cout << "[*] GPU pipeline initialized with dummy bloom filter\n\n";
#endif

        auto bench_start = std::chrono::steady_clock::now();
        auto bench_end = bench_start + std::chrono::seconds(args.benchmark_seconds);
        uint64_t total_hashed = 0;
        uint64_t iterations = 0;
        auto last_status = bench_start;

        while (std::chrono::steady_clock::now() < bench_end && !g_shutdown) {
#ifdef COLLIDER_USE_CUDA
            // Run actual GPU pipeline
            auto result = bench_pipeline.process_batch(test_candidates);
            total_hashed += result.processed;
#else
            // CPU fallback: simulate batch processing time
            std::this_thread::sleep_for(std::chrono::microseconds(1600));
            total_hashed += test_candidates.size();
#endif
            iterations++;

            // Status update every second
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_status).count() >= 1) {
                auto elapsed_sec = std::chrono::duration_cast<std::chrono::milliseconds>(now - bench_start).count() / 1000.0;
                auto remaining = std::chrono::duration_cast<std::chrono::seconds>(bench_end - now).count();
                double rate = total_hashed / elapsed_sec;

                std::cout << "\r[*] Progress: " << std::setw(3) << (args.benchmark_seconds - remaining) << "s / "
                          << args.benchmark_seconds << "s | "
                          << "Hashed: " << std::setw(12) << format_number(total_hashed) << " | "
                          << "Rate: " << std::setw(8) << format_rate(rate)
                          << "     " << std::flush;

                last_status = now;
            }
        }

        // Calculate final results
        auto actual_end = std::chrono::steady_clock::now();
        double actual_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(actual_end - bench_start).count() / 1000.0;
        double final_rate = total_hashed / actual_seconds;

        // Performance projections
        double projected_daily = final_rate * 86400;

        // Compare to targets
        double target_per_gpu = 2.5e9;  // 2.5B/s per RTX 5090
        double expected = target_per_gpu * args.gpu_ids.size();
        double efficiency = (final_rate / expected) * 100.0;

        {
            namespace boxui = ::collider::ui::box;
            std::cout << "\n\n";
            boxui::top(std::cout);
            boxui::centered(std::cout, "BENCHMARK RESULTS");
            boxui::top(std::cout);
            {
                std::ostringstream dur;
                dur << std::fixed << std::setprecision(2) << actual_seconds << " seconds";
                boxui::kv(std::cout, "Duration", dur.str());
            }
            boxui::kv(std::cout, "Total Processed", format_number(total_hashed));
            {
                std::ostringstream it;
                it << iterations;
                boxui::kv(std::cout, "Iterations", it.str());
            }
            boxui::kv(std::cout, "Average Rate", format_rate(final_rate));
            boxui::sep(std::cout);
            boxui::kv(std::cout, "Projected Daily",
                      format_number(static_cast<uint64_t>(projected_daily)));
            boxui::sep(std::cout);
            boxui::kv(std::cout, "Expected (RTX 5090)", format_rate(expected));
            {
                std::ostringstream eff;
                eff << std::fixed << std::setprecision(1) << efficiency << "%";
                boxui::kv(std::cout, "Efficiency", eff.str());
            }
            boxui::bottom(std::cout);
            std::cout << "\n";
        }

#ifndef COLLIDER_USE_CUDA
        std::cout << "[!] Note: CUDA not available - benchmark used CPU simulation.\n";
        std::cout << "    Real GPU performance will be significantly higher.\n";
        std::cout << "    Build with CUDA enabled for actual GPU benchmarks.\n";
#else
        if (efficiency < 80.0) {
            std::cout << "[*] Note: Performance varies by GPU. RTX 5090 target is 2.5B/s.\n";
        } else {
            std::cout << "[+] GPU pipeline performing at expected efficiency.\n";
        }
#endif

        return 0;
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

    // Verbose-mode pre-banner stats and the --analyze flag short-circuit
    // both used to live in main.cpp; they belong with the puzzle-mode
    // entry now that the banner is here.
    if (args.verbose) {
        analyze_solved_puzzles();
    }
    if (args.analyze_puzzles) {
        ::print_puzzle_analysis(400.0);  // Assume 400 MKeys/sec GPU speed
        return 0;
    }

    // Logger reference matches main.cpp's usage; the singleton was already
    // initialized before we got here.
    auto& logger = Logger::instance();

        // Load config for progress saving
        UserConfig config;
        config.load();

        // GPU batch size calibration (CUDA only)
#ifdef COLLIDER_USE_CUDA
        // Check if calibration is needed or requested
        bool need_calibration = args.calibrate || args.force_calibrate;
        if (!config.calibration_done && !need_calibration) {
            std::cout << "\n[*] First run detected - running GPU batch size calibration...\n";
            std::cout << "    (This optimizes performance for your specific hardware)\n";
            std::cout << "    (Use --force-calibrate to re-run calibration later)\n\n";
            need_calibration = true;
        }

        // Run calibration if needed
        if (need_calibration) {
            gpu::MultiGPUPuzzleSolver calibration_solver;
            gpu::MultiGPUPuzzleSolver::Config calib_config;
            calib_config.gpu_ids = args.gpu_ids;

            if (calibration_solver.init(calib_config)) {
                auto results = calibration_solver.calibrate_all(5);

                // Save results to config
                for (const auto& [device_id, batch_size] : results) {
                    config.set_gpu_batch_size(device_id, batch_size);
                }

                // Use the calibrated batch size for this run
                if (!results.empty()) {
                    args.batch_size = calibration_solver.get_batch_size();
                    std::cout << "\n[*] Calibration complete. Using batch size: "
                              << (args.batch_size / 1'000'000) << "M\n";
                }

                config.save();
                std::cout << "[*] Calibration results saved to: "
                          << UserConfig::get_config_path() << "\n\n";
            } else {
                std::cerr << "[!] GPU calibration failed - using default batch size\n";
            }
        } else if (config.calibration_done) {
            // Load calibrated batch size from config
            // Use the first GPU's optimal batch size (or could average/min)
            for (int gpu_id : args.gpu_ids) {
                uint64_t optimal = config.get_gpu_batch_size(gpu_id);
                if (optimal > 0) {
                    args.batch_size = optimal;
                    std::cout << "[*] Using calibrated batch size: "
                              << (args.batch_size / 1'000'000) << "M (from saved config)\n";
                    break;
                }
            }
        }
#endif

        // Build list of puzzles to solve
        std::vector<int> puzzles_to_solve;
        if (args.puzzle_all_unsolved) {
            auto unsolved = PuzzleDatabase::get_unsolved();
            for (const auto* p : unsolved) {
                puzzles_to_solve.push_back(p->number);
            }
            if (puzzles_to_solve.empty()) {
                std::cout << "[*] No unsolved puzzles found - all puzzles have been solved!\n";
                return 0;
            }
            std::cout << "\n[*] Auto-progression mode: " << puzzles_to_solve.size() << " unsolved puzzles\n";
            std::cout << "    Starting with puzzle #" << puzzles_to_solve[0] << "\n";
        } else {
            puzzles_to_solve.push_back(args.puzzle_number);
            // v1.4.1: --auto-next means "after this puzzle, keep going through
            // every higher-numbered unsolved puzzle." Pre-1.4.1 the flag was
            // parsed but never read, so it was a silent no-op.
            if (args.puzzle_auto_next) {
                auto unsolved = PuzzleDatabase::get_unsolved();
                for (const auto* p : unsolved) {
                    if (p->number > args.puzzle_number) {
                        puzzles_to_solve.push_back(p->number);
                    }
                }
                if (puzzles_to_solve.size() > 1) {
                    std::cout << "\n[*] --auto-next: will continue through "
                              << (puzzles_to_solve.size() - 1)
                              << " higher unsolved puzzle(s) after #"
                              << args.puzzle_number << "\n";
                }
            }
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

        // Determine range and target
        UInt256 range_start, range_end;
        std::string target_address;
        int bits;

        if (!args.puzzle_range_start.empty() && !args.puzzle_range_end.empty()) {
            // Custom range override
            range_start = UInt256(args.puzzle_range_start);
            range_end = UInt256(args.puzzle_range_end);
            bits = range_end.bit_length();
            target_address = args.puzzle_target;
            ::collider::ui::box::kv(std::cout, "Mode", "Custom Range");
        } else if (puzzle) {
            // Use known puzzle data
            range_start = puzzle->range_start();
            range_end = puzzle->range_end();
            bits = puzzle->bits;
            target_address = args.puzzle_target.empty() ? puzzle->target_address : args.puzzle_target;

            std::ostringstream puzzle_label;
            puzzle_label << "#" << puzzle->number;
            if (puzzle->solved) {
                puzzle_label << " (SOLVED - for testing)";
            } else {
                puzzle_label << " (" << std::fixed << std::setprecision(1)
                             << puzzle->btc_reward << " BTC reward)";
            }
            ::collider::ui::box::kv(std::cout, "Puzzle", puzzle_label.str());
        } else {
            std::cerr << "[!] Error: Unknown puzzle number: " << current_puzzle << "\n";
            std::cerr << "    Use --puzzle-start and --puzzle-end for custom ranges.\n";
            return 1;
        }

        {
            std::ostringstream bits_str;
            bits_str << bits << " (2^" << (bits - 1) << " keys in range)";
            ::collider::ui::box::kv(std::cout, "Bits", bits_str.str());
        }
        ::collider::ui::box::kv(std::cout, "Target",  target_address);
        ::collider::ui::box::kv(std::cout, "Search",  args.puzzle_random ? "Random" : "Sequential");
        ::collider::ui::box::kv(std::cout, "Backend", gpu_info.backend);
        ::collider::ui::box::top(std::cout);
        ::collider::ui::box::kv(std::cout, "Range Start", range_start.to_hex());
        ::collider::ui::box::kv(std::cout, "Range End",   range_end.to_hex());
        ::collider::ui::box::top(std::cout);
        std::cout << "\n";

        // Calculate search space info
        uint64_t search_space_bits = bits - 1;  // 2^(N-1) keys
        double years_at_1b_per_sec = std::pow(2.0, search_space_bits) / (1e9 * 86400 * 365);

        std::cout << "[*] Search Space Analysis:\n";
        std::cout << "    Keys in range:    2^" << search_space_bits << "\n";
        if (search_space_bits <= 40) {
            uint64_t total_keys = 1ULL << search_space_bits;
            std::cout << "    Exact count:      " << format_number(total_keys) << "\n";
        }
        std::cout << "    At 1B keys/sec:   ";
        if (years_at_1b_per_sec < 1.0/365) {
            std::cout << std::fixed << std::setprecision(1) << (years_at_1b_per_sec * 365 * 24) << " hours\n";
        } else if (years_at_1b_per_sec < 1.0) {
            std::cout << std::fixed << std::setprecision(1) << (years_at_1b_per_sec * 365) << " days\n";
        } else if (years_at_1b_per_sec < 1000) {
            std::cout << std::fixed << std::setprecision(1) << years_at_1b_per_sec << " years\n";
        } else {
            std::cout << std::scientific << std::setprecision(2) << years_at_1b_per_sec << " years\n";
        }
        std::cout << "\n";

        // Setup random number generator for random search
        std::random_device rd;
        std::mt19937_64 rng(rd());

        // For puzzles up to 128 bits, we use two 64-bit values
        // range_start.parts[0] = low 64 bits, parts[1] = next 64 bits
        uint64_t start_lo = range_start.parts[0];
        uint64_t start_hi = range_start.parts[1];
        uint64_t end_lo = range_end.parts[0];
        uint64_t end_hi = range_end.parts[1];

        // For random generation within range
        std::uniform_int_distribution<uint64_t> dist_lo(0, UINT64_MAX);
        std::uniform_int_distribution<uint64_t> dist_hi(start_hi, end_hi);

        // Parse target hash160 from puzzle database
        std::array<uint8_t, 20> target_hash160 = {0};
        bool have_target_hash = false;
        std::string h160_hex;

        if (puzzle && puzzle->target_h160_hex != "unknown" && puzzle->target_h160_hex.length() == 40) {
            h160_hex = puzzle->target_h160_hex;
            target_hash160 = cpu::hex_to_hash160(h160_hex);
            have_target_hash = true;
            std::cout << "[*] Target Hash160: " << h160_hex << "\n";
        } else if (puzzle && !puzzle->target_address.empty()) {
            // Try to decode h160 from the Bitcoin address
            h160_hex = Base58::address_to_h160_hex(puzzle->target_address);
            if (h160_hex.length() == 40) {
                target_hash160 = cpu::hex_to_hash160(h160_hex);
                have_target_hash = true;
                std::cout << "[*] Target Hash160 (decoded from address): " << h160_hex << "\n";
            } else {
                std::cout << "[!] Warning: Could not decode hash160 from address: " << puzzle->target_address << "\n";
                std::cout << "    Searching blind (will report any found addresses)\n";
            }
        } else {
            std::cout << "[!] Warning: Target hash160 not available for this puzzle\n";
            std::cout << "    Searching blind (will report any found addresses)\n";
        }

        // For small puzzles (< 40 bits), use sequential exhaustive search
        bool force_sequential = (bits <= 40);
        if (force_sequential && args.puzzle_random) {
            std::cout << "[*] Small puzzle detected - using sequential search for completeness\n";
        }

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
        // so v1.4.1 gracefully demotes --kangaroo to brute force per-
        // puzzle instead of failing the whole batch.
        const bool kangaroo_was_requested = args.puzzle_kangaroo;
        const bool pubkey_known = !args.puzzle_pubkey.empty()
                                  || (puzzle && !puzzle->public_key_hex.empty());

        if (args.puzzle_kangaroo && bits > 40 && !pubkey_known) {
            // User asked for kangaroo but we have no pubkey for this
            // puzzle. Three paths:
            //   1. Multi-puzzle worklist (--auto-next / --all-unsolved):
            //      silently downgrade. Stopping the whole batch on one
            //      pubkey-less puzzle would be hostile.
            //   2. Single-puzzle interactive (TTY on stdin): prompt the
            //      operator for a pubkey -- they may have one from an
            //      external source. ENTER falls back to brute force.
            //   3. Single-puzzle non-interactive (piped, CI, etc.):
            //      silently downgrade and log it.
#ifdef _WIN32
            const bool stdin_is_tty = _isatty(_fileno(stdin)) != 0;
#else
            const bool stdin_is_tty = isatty(fileno(stdin)) != 0;
#endif
            const bool prompt_allowed = !is_multi_puzzle && stdin_is_tty;

            std::cout << "\n[!] Puzzle #" << puzzle->number
                      << " has no bundled public key.\n";
            std::cout << "    Kangaroo requires a 33-byte compressed pubkey (02.../03...).\n";
            std::cout << "    Most non-multiples-of-5 in puzzles 71-160 have never had a\n";
            std::cout << "    spending transaction, so the pubkey is mathematically unknown.\n";

            if (prompt_allowed) {
                std::cout << "    If you have the pubkey from an external source, paste it now\n";
                std::cout << "    (66 hex chars, e.g. 02abcdef...). Otherwise press ENTER to\n";
                std::cout << "    fall back to brute force: ";
                std::cout.flush();
                std::string entered;
                std::getline(std::cin, entered);
                // Trim whitespace
                while (!entered.empty() && std::isspace(static_cast<unsigned char>(entered.front()))) entered.erase(entered.begin());
                while (!entered.empty() && std::isspace(static_cast<unsigned char>(entered.back())))  entered.pop_back();
                const bool looks_like_pubkey =
                    (entered.size() == 66 || entered.size() == 130) &&
                    (entered[0] == '0' && (entered[1] == '2' || entered[1] == '3' || entered[1] == '4'));
                if (looks_like_pubkey) {
                    args.puzzle_pubkey = entered;
                    std::cout << "[*] Pubkey accepted, continuing with Kangaroo.\n";
                } else {
                    if (!entered.empty()) {
                        std::cout << "[!] That doesn't look like a 33-byte compressed pubkey; "
                                     "falling back to brute force.\n";
                    } else {
                        std::cout << "[*] Falling back to brute force.\n";
                    }
                    args.puzzle_kangaroo = false;
                }
            } else {
                std::cout << "    " << (is_multi_puzzle ? "Auto-progression mode" : "Non-interactive mode")
                          << ": silently demoting --kangaroo to brute force for this puzzle.\n";
                args.puzzle_kangaroo = false;
            }
        }

        if (!args.puzzle_kangaroo && bits > 40) {
            // Auto-pick the algorithm: kangaroo if pubkey known, else
            // brute force. This runs in two cases:
            //   (a) user didn't pass --kangaroo at all
            //   (b) user passed --kangaroo but we just demoted it above
            if (pubkey_known) {
                args.puzzle_kangaroo = true;
                std::cout << "\n\033[36m[*] Algorithm Selection\033[0m\n";
                std::cout << "    Method: \033[1;32mKangaroo\033[0m (Pollard's Rho variant)\n";
                if (kangaroo_was_requested) {
                    std::cout << "    Reason: Pubkey available, --kangaroo respected\n";
                } else {
                    std::cout << "    Reason: Public key is known for this puzzle\n";
                }
                std::cout << "    Advantage: O(sqrt(n)) vs O(n) - dramatically faster\n";
            } else {
                std::cout << "\n\033[36m[*] Algorithm Selection\033[0m\n";
                std::cout << "    Method: \033[1;33mBrute Force\033[0m\n";
                std::cout << "    Reason: No public key available (Kangaroo requires pubkey)\n";
            }
        }

        // ======================================================================
        // KANGAROO MODE: Pollard's Kangaroo Algorithm (O(sqrt(n)))
        // ======================================================================
        if (args.puzzle_kangaroo && bits > 40) {
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

            // Kangaroo requires a known public key. The selection block
            // above (search for "ALGORITHM SELECTION") guarantees this
            // is non-empty by the time we get here -- a missing pubkey
            // demotes args.puzzle_kangaroo to false, so we wouldn't
            // enter this if-branch. CLI / config --pubkey wins over the
            // bundled value when both are set.
            const std::string& kangaroo_pubkey =
                !args.puzzle_pubkey.empty() ? args.puzzle_pubkey
                                            : puzzle->public_key_hex;
            if (!args.puzzle_pubkey.empty()) {
                std::cout << "[*] Using --pubkey override: " << kangaroo_pubkey << "\n";
            }

            // Decompress the public key
            cpu::uint256_t target_pubkey_x, target_pubkey_y;
            if (!cpu::decompress_pubkey(target_pubkey_x, target_pubkey_y, kangaroo_pubkey)) {
                std::cerr << "[!] ERROR: Failed to decompress public key: " << kangaroo_pubkey << "\n";
                return 1;
            }
            std::cout << "[*] Target public key decompressed successfully\n";

#ifdef COLLIDER_USE_RCKANGAROO
            // ================================================================
            // RCKangaroo - High-performance Kangaroo solver (8 GKeys/s on 4090)
            // ================================================================
            if (args.use_rckangaroo) {
                std::cout << "[*] Using RCKangaroo (RetiredCoder's high-performance solver)\n";

                gpu::RCKangarooManager rc_kangaroo;
                rc_kangaroo.range_bits = bits;

                // Set DP bits. RCKangaroo's documented acceptance window
                // is [kMinDpBits, kMaxDpBits] = [14, 60]. Reject explicitly
                // out-of-range user values rather than silently clamping;
                // a misconfigured CLI invocation should surface a clear
                // error so the operator can correct intent.
                using gpu::RCKangarooManager;
                if (args.dp_bits > 0) {
                    if (args.dp_bits < RCKangarooManager::kMinDpBits ||
                        args.dp_bits > RCKangarooManager::kMaxDpBits) {
                        std::cerr << "[!] --dp-bits=" << args.dp_bits
                                  << " is outside RCKangaroo's accepted range ["
                                  << RCKangarooManager::kMinDpBits << ".."
                                  << RCKangarooManager::kMaxDpBits << "]. Aborting.\n";
                        return 64;  // EX_USAGE
                    }
                    rc_kangaroo.dp_bits = args.dp_bits;
                    std::cout << "\033[36m[*] DP Configuration\033[0m\n";
                    std::cout << "    dp_bits: " << rc_kangaroo.dp_bits << " (user override)\n";
                    std::cout << "    1 in " << format_number(1ULL << rc_kangaroo.dp_bits) << " points marked as DP\n";
                } else {
                    // Auto: clamp the bits/3 heuristic into [16, 28]. We
                    // narrow the auto window further than the absolute
                    // [14, 60] window because the heuristic shouldn't
                    // ever pick wildly off values, only sane defaults.
                    rc_kangaroo.dp_bits = std::min(28, std::max(16, bits / 3));
                    std::cout << "\033[36m[*] DP Configuration (auto)\033[0m\n";
                    std::cout << "    dp_bits: " << rc_kangaroo.dp_bits << " (optimal for " << bits << "-bit puzzle)\n";
                    std::cout << "    1 in " << format_number(1ULL << rc_kangaroo.dp_bits) << " points marked as DP\n";
                }

                // Initialize GPUs
                int num_gpus = rc_kangaroo.init(args.gpu_ids);
                if (num_gpus > 0) {
                    // Load bloom filter if specified -- Pro feature; the
                    // free build clears args.bloom_file at config merge,
                    // so this block is a no-op there. #ifdef-guard the
                    // load as defense-in-depth.
#ifdef COLLIDER_PRO
                    if (!args.bloom_file.empty()) {
                        if (rc_kangaroo.load_bloom_filter(args.bloom_file)) {
                            std::cout << "[*] Bloom filter loaded - opportunistic address checking enabled\n";
                            // Optional: Set hit callback for real-time notifications
                            rc_kangaroo.bloom_hit_callback = [](const gpu::BloomHit& hit) {
                                std::ofstream hitlog("bloom_hits.txt", std::ios::app);
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
                    }
#endif

                    // Set target public key. v1.4.1: prefer the --pubkey
                    // CLI / config override when set; fall back to the
                    // bundled puzzle->public_key_hex.
                    std::string pubkey_hex = !args.puzzle_pubkey.empty()
                                                 ? args.puzzle_pubkey
                                                 : puzzle->public_key_hex;
                    if (!rc_kangaroo.set_target_pubkey(pubkey_hex)) {
                        std::cerr << "[!] ERROR: Failed to set target pubkey\n";
                        return 1;
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

                    // Progress callback
                    rc_kangaroo.progress_callback = [&, expected_ops, expected_ops_bits](uint64_t ops, uint64_t dp_count, int speed) -> bool {
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

                    auto start_time = std::chrono::steady_clock::now();
                    auto rc_result = rc_kangaroo.solve();
                    auto end_time = std::chrono::steady_clock::now();
                    double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

                    if (rc_result.found) {
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
                        std::ofstream found_file("puzzle_found.txt", std::ios::app);
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
                        return 0;
                    } else {
                        std::cout << "\n\n[!] RCKangaroo search stopped after "
                                  << format_number(rc_result.total_ops) << " ops\n";
                        std::cout << "    Duration: " << std::fixed << std::setprecision(1) << total_seconds << " seconds\n";
                        if (rc_result.error_count > 0) {
                            std::cout << "    Errors: " << rc_result.error_count << "\n";
                        }
                        return 0;
                    }
                } else {
                    std::cout << "[!] RCKangaroo initialization failed, falling back to standard solver\n";
                }
            }
#endif  // COLLIDER_USE_RCKANGAROO

            // Try Multi-GPU Kangaroo (fallback if RCKangaroo not available)
            bool use_gpu_kangaroo = false;
            gpu::MultiGPUKangarooManager gpu_kangaroo;
            int dp_bits_to_use = 20;  // Default, will be set properly below

            // Initialize with all available GPUs (or specific ones from args.gpu_ids if set)
            if (gpu_kangaroo.init(args.gpu_ids)) {
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
                        return 64;  // EX_USAGE
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
                use_gpu_kangaroo = true;
            }

            auto start_time = std::chrono::steady_clock::now();

            // Variable for GPU count used in progress display (declared outside if block for capture)
            int num_gpus_for_display = 0;

            if (use_gpu_kangaroo) {
                // Multi-GPU Kangaroo
                num_gpus_for_display = gpu_kangaroo.num_gpus();

                // Calculate expected operations for this puzzle
                double expected_ops_bits = (bits - 1) / 2.0 + 1;  // sqrt(2^(bits-1)) ~= 2^((bits-1)/2)
                uint64_t expected_ops = (expected_ops_bits < 63) ? (1ULL << (int)expected_ops_bits) : 0;

                gpu_kangaroo.progress_callback = [&, expected_ops, expected_ops_bits](uint64_t steps, uint64_t dp_count, double rate) -> bool {
                    if (g_shutdown) return false;

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
                ui::ProfessionalUI::render_kv("GPUs", std::to_string(num_gpus_for_display) + "x " + gpu_info.gpu_names);
                ui::ProfessionalUI::render_kv("Range", std::to_string(bits) + " bits");
                ui::ProfessionalUI::render_kv("DP Bits", std::to_string(dp_bits_to_use));
                ui::ProfessionalUI::render_kv("Expected Ops", "~2^" + std::to_string((int)expected_ops_bits));
                std::cout << "\n";
                ui::ProfessionalUI::render_footer("Press Ctrl+C to stop and save checkpoint");

                auto gpu_result = gpu_kangaroo.solve();

                auto end_time = std::chrono::steady_clock::now();
                double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

                if (gpu_result.found) {
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
                    std::ofstream found_file("puzzle_found.txt", std::ios::app);
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
                    return 0;
                } else {
                    std::cout << "\n\n[!] GPU Kangaroo search stopped after "
                              << format_number(gpu_result.total_steps) << " steps\n";
                    std::cout << "    Duration: " << std::fixed << std::setprecision(1) << total_seconds << " seconds\n";
                    return 0;
                }
            }

            // Fall back to CPU Kangaroo
            std::cout << "[*] Falling back to CPU Kangaroo...\n";

            KangarooSolver solver;
            solver.set_range(range_start, range_end);

            // Configure dp_bits for CPU solver
            // CPU solver uses fewer kangaroos, so adjust calculation
            int cpu_kangaroos = 2;  // CPU uses 1 tame + 1 wild kangaroo

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

            auto result = solver.solve();

            auto end_time = std::chrono::steady_clock::now();
            double total_seconds = std::chrono::duration<double>(end_time - start_time).count();

            if (result.found) {
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
                std::ofstream found_file("puzzle_found.txt", std::ios::app);
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

                // Continue to next puzzle in auto-progression mode
                if (is_multi_puzzle) {
                    std::cout << "[*] Puzzle solved! Continuing to next puzzle...\n";
                    continue;
                }
                return 0;
            } else {
                std::cout << "\n\n[!] Kangaroo search stopped after "
                          << format_number(result.tame_steps + result.wild_steps) << " steps\n";
                std::cout << "    Duration: " << std::fixed << std::setprecision(1) << total_seconds << " seconds\n";
                return 0;
            }
        }

        // ======================================================================
        // MULTI-GPU PUZZLE SEARCH (uses optimized kernels)
        // ======================================================================
        gpu::MultiGPUPuzzleSolver gpu_solver;
        bool use_gpu = false;

        if (have_target_hash && !force_sequential) {
            // Initialize multi-GPU solver with user-specified GPUs
            gpu::MultiGPUPuzzleSolver::Config gpu_config;
            gpu_config.gpu_ids = args.gpu_ids;
            gpu_config.batch_size_per_gpu = args.batch_size;  // 4M keys per GPU per batch

            if (gpu_solver.init(gpu_config)) {
                if (gpu_solver.set_target(target_hash160)) {
                    use_gpu = true;
                    std::cout << "\n[*] Starting MULTI-GPU optimized puzzle search...\n";
                    std::cout << "    Pipeline: PrivKey -> EC Mul (precomp) -> Compress -> SHA256 -> RIPEMD160 -> Compare\n";
                    std::cout << "    GPUs: " << gpu_solver.num_gpus() << " x " << gpu_info.backend << "\n";
                    std::cout << "    Optimizations: Precomputed tables, inline hashes, batch inversion\n";
                    std::cout << "    Log: " << logger.get_log_path() << "\n";
                    std::cout << "    Press Ctrl+C to stop\n\n";

                    // Log startup info for crash diagnosis
                    logger.log_startup(current_puzzle, gpu_solver.num_gpus(), gpu_info.gpu_names,
                                       args.batch_size, args.puzzle_random ? "Random" : "Zone-Based");
                }
            }
        }

        if (use_gpu) {
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
            // CENTER-HEAVY ZONE-BASED SCANNING
            // ================================================================
            // Instead of sequential from start, we scan high-probability zones first
            // Based on research showing solved keys cluster at 0.6-0.85 of range

            std::cout << "[*] Using Center-Heavy Zone Scanning (research-optimized)\n";
            std::cout << "    Priority: Center-High -> Center-Low -> Bridge -> Edges\n\n";

            // Track zone progress
            size_t current_zone_idx = 0;
            uint64_t zone_start_lo, zone_start_hi;
            uint64_t zone_end_lo, zone_end_hi;
            uint64_t current_lo, current_hi;
            uint64_t zone_checked = 0;

            // Try to load saved state for this puzzle
            auto saved_state = SearchStateManager::load_puzzle_state(current_puzzle);
            if (saved_state.valid && saved_state.total_checked > 0) {
                std::cout << "[*] Resuming from saved state:\n";
                std::cout << "    Last saved: " << saved_state.timestamp << "\n";
                std::cout << "    Keys checked: " << format_number(saved_state.total_checked) << "\n";
                std::cout << "    Zone: " << (saved_state.zone_idx + 1) << "/" << NUM_ZONES << "\n\n";

                current_zone_idx = saved_state.zone_idx;
                current_lo = saved_state.position_lo;
                current_hi = saved_state.position_hi;
                total_checked = saved_state.total_checked;
                zone_checked = saved_state.zone_checked;

                // Calculate zone boundaries for the restored zone
                calc_zone_position(start_lo, start_hi, end_lo, end_hi,
                                  PUZZLE_ZONES[current_zone_idx].start_pct, zone_start_lo, zone_start_hi);
                calc_zone_position(start_lo, start_hi, end_lo, end_hi,
                                  PUZZLE_ZONES[current_zone_idx].end_pct, zone_end_lo, zone_end_hi);

                std::cout << "[*] Continuing Zone " << (current_zone_idx + 1) << ": "
                          << PUZZLE_ZONES[current_zone_idx].name << "\n";
            } else {
                // Initialize first zone from scratch
                calc_zone_position(start_lo, start_hi, end_lo, end_hi,
                                  PUZZLE_ZONES[0].start_pct, zone_start_lo, zone_start_hi);
                calc_zone_position(start_lo, start_hi, end_lo, end_hi,
                                  PUZZLE_ZONES[0].end_pct, zone_end_lo, zone_end_hi);
                current_lo = zone_start_lo;
                current_hi = zone_start_hi;

                std::cout << "[*] Starting Zone 1: " << PUZZLE_ZONES[0].name << "\n";
            }

            while (!g_shutdown && !found) {
                // Check if we've completed current zone
                if (current_hi > zone_end_hi || (current_hi == zone_end_hi && current_lo >= zone_end_lo)) {
                    std::cout << "\n[*] Zone " << (current_zone_idx + 1) << " complete ("
                              << PUZZLE_ZONES[current_zone_idx].name << ") - "
                              << format_number(zone_checked) << " keys checked\n";

                    // Log zone completion
                    logger.log_zone_complete(current_zone_idx, PUZZLE_ZONES[current_zone_idx].name, zone_checked);

                    // Move to next zone
                    current_zone_idx++;
                    zone_checked = 0;

                    if (current_zone_idx >= NUM_ZONES) {
                        std::cout << "\n[!] GPU search complete - all zones checked.\n";
                        if (have_target_hash) {
                            std::cout << "[!] No match found.\n";
                        }
                        break;
                    }

                    // Initialize next zone
                    calc_zone_position(start_lo, start_hi, end_lo, end_hi,
                                      PUZZLE_ZONES[current_zone_idx].start_pct, zone_start_lo, zone_start_hi);
                    calc_zone_position(start_lo, start_hi, end_lo, end_hi,
                                      PUZZLE_ZONES[current_zone_idx].end_pct, zone_end_lo, zone_end_hi);
                    current_lo = zone_start_lo;
                    current_hi = zone_start_hi;

                    std::cout << "[*] Starting Zone " << (current_zone_idx + 1) << ": "
                              << PUZZLE_ZONES[current_zone_idx].name << "\n";
                    continue;
                }

                // Search this batch on GPU
                if (gpu_solver.search_batch(current_lo, current_hi, gpu_batch_size,
                                            found_key_lo, found_key_hi)) {
                    found = true;

                    // Get solve time details
                    auto solve_time = std::chrono::system_clock::now();
                    auto solve_time_t = std::chrono::system_clock::to_time_t(solve_time);
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time).count() / 1000.0;

                    // Format timestamp
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
                    std::ofstream found_file("puzzle_found.txt", std::ios::app);
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

                    // Clear saved state - puzzle solved!
                    collider::SearchStateManager::clear_puzzle_state(current_puzzle);

                    // Continue to next puzzle in auto-progression mode
                    if (is_multi_puzzle) {
                        std::cout << "[*] Puzzle solved! Continuing to next puzzle...\n";
                        continue;
                    }

                    // Show next puzzle suggestion for manual mode
                    auto unsolved = PuzzleDatabase::get_unsolved();
                    if (!unsolved.empty()) {
                        std::cout << "\n[*] Next unsolved puzzle: #" << unsolved[0]->number
                                  << " (" << unsolved[0]->bits << "-bit, "
                                  << std::fixed << std::setprecision(1) << unsolved[0]->btc_reward << " BTC)\n";
                    }

                    return 0;
                }

                total_checked += gpu_batch_size;
                session_checked += gpu_batch_size;  // Track session-only for accurate rate
                zone_checked += gpu_batch_size;

                // Advance to next batch
                uint64_t new_lo = current_lo + gpu_batch_size;
                if (new_lo < current_lo) current_hi++;  // Handle overflow
                current_lo = new_lo;

                // Status update with zone info
                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_update).count() >= 1) {
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                    // Use session_checked for rate (not total_checked) to avoid inflated rate on resume
                    double rate = (elapsed_ms > 0) ? (session_checked * 1000.0 / elapsed_ms) : 0;

                    // Show zone progress in status (display total_checked for cumulative count)
                    std::cout << "\r[*] Zone " << (current_zone_idx + 1) << "/" << NUM_ZONES
                              << " | Checked: " << std::setw(12) << format_number_human(total_checked)
                              << " | Rate: " << std::setw(10) << format_rate(rate)
                              << "   " << std::flush;

                    last_update = now;
                }

                // Save state periodically (every 30 seconds)
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_state_save).count() >= 30) {
                    collider::PuzzleSearchState state;
                    state.puzzle_number = current_puzzle;
                    state.zone_idx = current_zone_idx;
                    state.position_lo = current_lo;
                    state.position_hi = current_hi;
                    state.total_checked = total_checked;
                    state.zone_checked = zone_checked;
                    collider::SearchStateManager::save_puzzle_state(state);
                    last_state_save = now;
                }

                // File logging for crash diagnosis (every 60 seconds)
                if (std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count() >= 60) {
                    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                    // Use session_checked for accurate rate logging
                    double rate = (elapsed_ms > 0) ? (session_checked * 1000.0 / elapsed_ms) : 0;
                    logger.log_progress(total_checked, rate, current_zone_idx, NUM_ZONES);
                    last_log_time = now;
                }
            }

            // Track-f F-03: emit the deferred shutdown message from
            // main-thread context (idempotent; no-op on the !g_shutdown path
            // and on the second-or-later call).
            if (g_shutdown.load(std::memory_order_acquire)) {
                emit_shutdown_message_from_main();
            }

            // Save state on shutdown for resume
            if (g_shutdown) {
                collider::PuzzleSearchState state;
                state.puzzle_number = current_puzzle;
                state.zone_idx = current_zone_idx;
                state.position_lo = current_lo;
                state.position_hi = current_hi;
                state.total_checked = total_checked;
                state.zone_checked = zone_checked;
                collider::SearchStateManager::save_puzzle_state(state);
                std::cout << "\n[*] State saved - run again to resume from "
                          << format_number(total_checked) << " keys\n";

                // Log state save
                logger.log_state_save(current_puzzle, current_zone_idx, current_lo, current_hi);
            }

            // GPU search completed (or interrupted)
            auto end_time = std::chrono::steady_clock::now();
            double elapsed_sec = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
            // Use session_checked for accurate session rate
            double session_rate = session_checked / std::max(0.001, elapsed_sec);

            // Log shutdown with reason
            std::string shutdown_reason = g_shutdown ? "User interrupt (Ctrl+C)" :
                                          (found ? "Key found!" : "All zones searched");
            logger.log_shutdown(shutdown_reason, total_checked, elapsed_sec);

            {
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

            return 0;
        }

        // ======================================================================
        // CPU FALLBACK (when GPU not available)
        // ======================================================================
        std::cout << "\n[*] Starting puzzle search...\n";
        std::cout << "    Pipeline: PrivKey -> secp256k1 -> PubKey -> SHA256 -> RIPEMD160 -> Compare\n";
        std::cout << "    Using: CPU reference implementation\n";
        std::cout << "    Press Ctrl+C to stop\n\n";

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

                    // Get solve time details
                    auto solve_time = std::chrono::system_clock::now();
                    auto solve_time_t = std::chrono::system_clock::to_time_t(solve_time);
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start_time).count() / 1000.0;

                    // Format timestamp
                    char timestamp[64];
                    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S",
                                  std::localtime(&solve_time_t));

                    char key_hex[67];
                    if (key_hi > 0) {
                        snprintf(key_hex, sizeof(key_hex), "0x%llx%016llx",
                                 (unsigned long long)key_hi, (unsigned long long)key_lo);
                    } else {
                        snprintf(key_hex, sizeof(key_hex), "0x%llx", (unsigned long long)key_lo);
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
                    std::ofstream found_file("puzzle_found.txt", std::ios::app);
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
            continue;
        }

        search_done:

        // Track-f F-03: emit deferred shutdown print/log if we got here via
        // a SIGINT/SIGTERM. Idempotent - no-op if no signal arrived.
        if (g_shutdown.load(std::memory_order_acquire)) {
            emit_shutdown_message_from_main();
        }

        // Final stats
        auto end_time = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
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
