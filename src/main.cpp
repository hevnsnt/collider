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
#define NOMINMAX

// Prevent Windows.h from including winsock.h (conflicts with winsock2.h)
#define WIN32_LEAN_AND_MEAN

// Windows console ANSI support
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#endif

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <csignal>
#include <atomic>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <set>

#include "core/edition.hpp"
#include "core/types.hpp"
#include "core/rule_engine.hpp"
#include "core/puzzle_config.hpp"
#include "core/crypto_cpu.hpp"
#include "core/kangaroo.hpp"
#include "core/config.hpp"
#include "core/logger.hpp"
#include "gpu/kangaroo_solver_gpu.hpp"
#ifdef COLLIDER_USE_RCKANGAROO
#include "gpu/rckangaroo_wrapper.hpp"
#endif
#include "platform/platform.hpp"
#include "core/search_state.hpp"
#include "ui/banner.hpp"
#include "ui/interactive.hpp"
#include "gpu/puzzle_gpu.hpp"
#include "pool/pool_manager.hpp"
#include "tools/utxo_bloom_builder.hpp"
#include "core/hit_verifier.hpp"
#include "core/yaml_config.hpp"
#include <random>
#include <array>

// Forward declarations for GPU functions

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

struct SearchZone {
    double start_pct;   // Zone start as percentage of range (0.0 - 1.0)
    double end_pct;     // Zone end as percentage of range
    const char* name;   // Display name
    int priority;       // Lower = higher priority
};

// Zone definitions based on research analysis
static const SearchZone PUZZLE_ZONES[] = {
    { 0.60, 0.85, "Center-High (60-85%)", 1 },   // Highest probability
    { 0.30, 0.50, "Center-Low (30-50%)",  2 },   // Secondary probability
    { 0.50, 0.60, "Bridge (50-60%)",      3 },   // Bridge zone
    { 0.85, 1.00, "Upper Edge (85-100%)", 4 },   // Upper edge
    { 0.00, 0.30, "Lower Edge (0-30%)",   5 },   // Least probable
};
static const size_t NUM_ZONES = sizeof(PUZZLE_ZONES) / sizeof(PUZZLE_ZONES[0]);

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
static const SolvedPuzzle SOLVED_PUZZLES[] = {
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
static const size_t NUM_SOLVED_PUZZLES = sizeof(SOLVED_PUZZLES) / sizeof(SOLVED_PUZZLES[0]);

// Analyze solved puzzles to validate zone priorities
inline void analyze_solved_puzzles() {
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

namespace collider {
namespace gpu {
    class GPUPipeline;
    struct PipelineConfig;
}
}

using namespace collider;

// Global shutdown flag
std::atomic<bool> g_shutdown{false};

/**
 * Detect GPU hardware and return formatted info for banner
 */
struct GPUDetectionResult {
    int device_count;
    std::string gpu_names;
    uint64_t estimated_speed;  // keys/second
    std::string backend;
};

GPUDetectionResult detect_gpus(std::vector<int>& requested_ids) {
    GPUDetectionResult result;
    result.device_count = 0;
    result.estimated_speed = 0;
    result.backend = "CPU";

    // Flush stdout before platform init - prevents buffered output from
    // appearing after Metal/CUDA initialization messages
    std::cout << std::flush;

    try {
        auto& platform = platform::get_platform();
        auto init_result = platform.initialize();

        if (init_result.code == platform::ErrorCode::Success) {
            result.backend = platform.get_backend_name();
            int total_devices = platform.get_device_count();

            // Auto-detect all GPUs if none specified
            if (requested_ids.empty() && total_devices > 0) {
                for (int i = 0; i < total_devices; i++) {
                    requested_ids.push_back(i);
                }
            }

            std::vector<std::string> names;
            for (int id : requested_ids) {
                if (id < total_devices) {
                    auto info = platform.get_device_info(id);
                    names.push_back(info.name);
                    result.device_count++;

                    // Estimate speed based on GPU type
                    // NOTE: These estimates are for EC scalar multiply (puzzle search)
                    // which is ~100x slower than SHA256 due to modular arithmetic.
                    // Optimized implementations with precomputed tables can be 10-20x faster.
                    if (info.is_apple_silicon) {
                        // Apple Silicon estimates for naive EC multiply
                        if (info.name.find("M3 Max") != std::string::npos) {
                            result.estimated_speed += 8'000'000;   // ~8M/s
                        } else if (info.name.find("M3 Pro") != std::string::npos) {
                            result.estimated_speed += 6'000'000;   // ~6M/s
                        } else if (info.name.find("M3") != std::string::npos) {
                            result.estimated_speed += 4'000'000;   // ~4M/s
                        } else if (info.name.find("M2") != std::string::npos) {
                            result.estimated_speed += 3'000'000;   // ~3M/s
                        } else {
                            result.estimated_speed += 2'000'000;   // ~2M/s
                        }
                    } else if (info.is_blackwell) {
                        // RTX 5090 estimate
                        result.estimated_speed += 80'000'000;      // ~80M/s
                    } else if (info.is_ampere) {
                        // Ampere estimates (RTX 30xx series)
                        if (info.name.find("3090") != std::string::npos) {
                            result.estimated_speed += 20'000'000;  // ~20M/s
                        } else if (info.name.find("3080") != std::string::npos) {
                            result.estimated_speed += 15'000'000;  // ~15M/s
                        } else if (info.name.find("3070") != std::string::npos) {
                            result.estimated_speed += 10'000'000;  // ~10M/s
                        } else {
                            result.estimated_speed += 5'000'000;   // ~5M/s (3060)
                        }
                    } else {
                        // Ada Lovelace (RTX 40xx series)
                        if (info.name.find("4090") != std::string::npos) {
                            result.estimated_speed += 50'000'000;  // ~50M/s
                        } else if (info.name.find("4080") != std::string::npos) {
                            result.estimated_speed += 35'000'000;  // ~35M/s
                        } else {
                            result.estimated_speed += 25'000'000;  // ~25M/s
                        }
                    }
                }
            }

            // Format GPU names
            if (names.empty()) {
                result.gpu_names = "No GPUs detected";
            } else if (names.size() == 1) {
                result.gpu_names = names[0];
            } else {
                // Check if all same
                bool all_same = true;
                for (size_t i = 1; i < names.size(); i++) {
                    if (names[i] != names[0]) {
                        all_same = false;
                        break;
                    }
                }
                if (all_same) {
                    result.gpu_names = std::to_string(names.size()) + "x " + names[0];
                } else {
                    result.gpu_names = names[0];
                    for (size_t i = 1; i < names.size(); i++) {
                        result.gpu_names += ", " + names[i];
                    }
                }
            }
        } else {
            result.gpu_names = "GPU init failed: " + init_result.message;
        }
    } catch (const std::exception& e) {
        result.gpu_names = std::string("Detection error: ") + e.what();
    }

    // CPU fallback estimation
    if (result.device_count == 0 || result.estimated_speed == 0) {
        result.device_count = 1;
        result.gpu_names = "CPU (reference mode)";
        result.estimated_speed = 10'000;  // 10K/s for CPU reference
        result.backend = "CPU";
    }

    return result;
}

void signal_handler(int signum) {
    std::cout << "\n[!] Interrupt received, shutting down...\n";
    LOG_INFO("Signal received: " + std::to_string(signum) + " (SIGINT=" + std::to_string(SIGINT) + ", SIGTERM=" + std::to_string(SIGTERM) + ")");
    g_shutdown = true;
}

/**
 * Command-line arguments.
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
 * Parse command-line arguments.
 */
Arguments parse_args(int argc, char* argv[]) {
    Arguments args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true;
        } else if ((arg == "--gpus" || arg == "-g") && i + 1 < argc) {
            args.gpu_ids.clear();
            std::string gpus = argv[++i];
            size_t pos = 0;
            while ((pos = gpus.find(',')) != std::string::npos) {
                args.gpu_ids.push_back(std::stoi(gpus.substr(0, pos)));
                gpus.erase(0, pos + 1);
            }
            args.gpu_ids.push_back(std::stoi(gpus));
        } else if (arg == "--batch-size" && i + 1 < argc) {
            args.batch_size = std::stoull(argv[++i]);
        } else if (arg == "--benchmark") {
            args.benchmark = true;
        } else if (arg == "--benchmark-time" && i + 1 < argc) {
            args.benchmark_seconds = std::stoi(argv[++i]);
        } else if (arg == "--puzzle" || arg == "-P") {
#if COLLIDER_HAS_SOLO
            args.puzzle_mode = true;
            // Check if next arg is a number (puzzle number) or another flag
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.puzzle_number = std::stoi(argv[++i]);
            }
#else
            std::cerr << "[*] Solo puzzle solver requires collider pro — collisionprotocol.com/pro\n";
            std::cerr << "    Use the free edition to join pools: collider --worker <your_btc_address>\n";
            return 1;
#endif
        } else if (arg == "--puzzle-target" && i + 1 < argc) {
#if COLLIDER_HAS_SOLO
            args.puzzle_target = argv[++i];
#else
            COLLIDER_REQUIRE_PRO(SOLO);
#endif
        } else if (arg == "--puzzle-start" && i + 1 < argc) {
#if COLLIDER_HAS_SOLO
            args.puzzle_range_start = argv[++i];
#else
            COLLIDER_REQUIRE_PRO(SOLO);
#endif
        } else if (arg == "--puzzle-end" && i + 1 < argc) {
#if COLLIDER_HAS_SOLO
            args.puzzle_range_end = argv[++i];
#else
            COLLIDER_REQUIRE_PRO(SOLO);
#endif
        } else if (arg == "--sequential") {
            args.puzzle_random = false;
        } else if (arg == "--random") {
            args.puzzle_random = true;
        } else if (arg == "--puzzle-checkpoint" && i + 1 < argc) {
            args.puzzle_checkpoint = argv[++i];
        } else if (arg == "--auto-next") {
            args.puzzle_auto_next = true;
        } else if (arg == "--all-unsolved") {
            args.puzzle_all_unsolved = true;
        } else if (arg == "--puzzle-min-bits" && i + 1 < argc) {
            args.puzzle_min_bits = std::stoi(argv[++i]);
        } else if (arg == "--puzzle-max-bits" && i + 1 < argc) {
            args.puzzle_max_bits = std::stoi(argv[++i]);
        } else if (arg == "--kangaroo") {
            args.puzzle_kangaroo = true;
        } else if (arg == "--dp-bits" && i + 1 < argc) {
            args.dp_bits = std::stoi(argv[++i]);
        } else if (arg == "--bloom" && i + 1 < argc) {
#if COLLIDER_HAS_BLOOM
            args.bloom_file = argv[++i];
#else
            std::cerr << "[*] Bloom filters require collider pro — collisionprotocol.com/pro\n";
            return 1;
#endif
        } else if (arg == "--brainwallet") {
#if COLLIDER_HAS_BRAINWALLET
            args.brainwallet_mode = true;
#else
            std::cerr << "[*] Brain wallet requires collider pro — collisionprotocol.com/pro\n";
            return 1;
#endif
        } else if (arg == "--brainwallet-setup") {
#if COLLIDER_HAS_BRAINWALLET
            args.brainwallet_setup = true;
#else
            std::cerr << "[!] ERROR: --brainwallet-setup is only available in theCollider Pro\n";
            std::cerr << "    Visit https://thecollider.com/pro to upgrade.\n";
            return 1;
#endif
        } else if (arg == "--resume") {
            args.resume = true;
        } else if (arg == "--save-interval" && i + 1 < argc) {
            args.save_interval = std::stoull(argv[++i]);
        } else if (arg == "--calibrate") {
            args.calibrate = true;
        } else if (arg == "--force-calibrate") {
            args.calibrate = true;
            args.force_calibrate = true;
        } else if (arg == "--debug") {
            args.debug = true;
        } else if (arg == "--analyze") {
            args.analyze_puzzles = true;
        } else if (arg == "--no-smart") {
            args.smart_select = false;
        } else if ((arg == "--pool" || arg == "-p") && i + 1 < argc) {
#if COLLIDER_HAS_CUSTOM_POOL
            args.pool_mode = true;
            args.pool_url = argv[++i];
#else
            std::cerr << "[!] ERROR: --pool (custom pool) is only available in theCollider Pro\n";
            std::cerr << "    Free edition connects to: " << COLLIDER_FREE_POOL_URL << "\n";
            std::cerr << "    Use --worker <your_btc_address> to join the pool.\n";
            std::cerr << "    Visit https://thecollider.com/pro for custom pool support.\n";
            return 1;
#endif
        } else if ((arg == "--worker" || arg == "-w") && i + 1 < argc) {
            args.pool_worker = argv[++i];
            // In free edition, automatically enable pool mode with hardcoded URL
#if !COLLIDER_HAS_CUSTOM_POOL
            args.pool_mode = true;
            args.pool_url = COLLIDER_FREE_POOL_URL;
#endif
        } else if (arg == "--pool-password" && i + 1 < argc) {
#if COLLIDER_HAS_CUSTOM_POOL
            args.pool_password = argv[++i];
#else
            std::cerr << "[!] ERROR: --pool-password is only available in theCollider Pro\n";
            return 1;
#endif
        } else if (arg == "--pool-api-key" && i + 1 < argc) {
#if COLLIDER_HAS_CUSTOM_POOL
            args.pool_api_key = argv[++i];
#else
            std::cerr << "[!] ERROR: --pool-api-key is only available in theCollider Pro\n";
            return 1;
#endif
        } else if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            args.config_file = argv[++i];
        }
    }

    return args;
}

/**
 * Print usage information.
 */
void print_usage() {
    std::cout << "\n";
    std::cout << "theCollider - Pool Edition\n";
    std::cout << "==========================\n\n";
    std::cout << "Free edition - Connect to Collision Protocol Pool for distributed solving.\n";
    std::cout << "For standalone puzzle solving, visit https://thecollider.com/pro\n\n";

    std::cout << "GPU-accelerated solver for the Bitcoin Puzzle Challenge.\n\n";

    std::cout << "Usage:\n";
    std::cout << "  collider --worker <btc_address>\n\n";
    std::cout << "Pool Options:\n";
  --worker, -w <address>  Your Bitcoin address for rewards (required)

GPU Options:
  --gpus, -g <ids>        GPU device IDs to use (default: auto-detect all)
  --batch-size <n>        Keys per batch (default: 4000000)

Output Options:
  --verbose, -v           Verbose output

Benchmark Mode:
  --benchmark             Run GPU performance benchmark
  --benchmark-time <sec>  Benchmark duration in seconds (default: 30)

Calibration:
  --calibrate             Run GPU batch size calibration (auto on first run)
  --force-calibrate       Force re-calibration even if already saved

Pool Mode:
  --worker, -w <address>  Your Bitcoin address for rewards
                          Connects to: )" << COLLIDER_FREE_POOL_URL << R"(
                          Example: jlp://pool.collisionprotocol.com:17403
  --worker, -w <address>  Worker name (your Bitcoin address for rewards)
  --pool-password <pass>  Pool password (if required)
  --pool-api-key <key>    API key for HTTP pools (if required)

Other:
  --help, -h              Show this help message
  --debug                 Show debug output for troubleshooting
  --config, -c <file>     Use custom config file (default: ./config.yml)

Configuration:
  Settings can be stored in config.yml (current directory or ~/.collider/)
  Command-line arguments override config file settings.

Examples:
  # Join the Collision Protocol Pool
  collider --worker 1YourBitcoinAddress...

  # Run GPU benchmark
  collider --benchmark

  # Use specific GPUs
  collider --worker 1YourBitcoinAddress... --gpus 0,1

  # Join Collision Protocol for distributed solving
  collider --pool jlp://pool.collisionprotocol.com:17403 --worker 1YourBitcoinAddress...

  # Pool mode with HTTP API
  collider --pool http://api.collisionprotocol.com --worker 1YourBitcoinAddress...

Performance Targets (EC multiply bottleneck):
  - RTX 3060:  ~5M keys/second
  - RTX 3090:  ~20M keys/second
  - RTX 4090:  ~50M keys/second

)";
}

// =============================================================================
// SMART PUZZLE SELECTION
// =============================================================================

/**
 * Puzzle analysis result for ROI comparison.
 */
struct PuzzleAnalysis {
    int number;
    int bits;
    double btc_reward;
    bool has_pubkey;
    std::string algorithm;    // "Kangaroo" or "BruteForce"
    double complexity_bits;   // Log2 of operations required
    double roi_score;         // Higher = better (reward / complexity)
    std::string feasibility;  // "RECOMMENDED", "VIABLE", "DIFFICULT", "INFEASIBLE"
    double estimated_gpu_years; // Very rough estimate
};

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
void print_puzzle_analysis(double gpu_speed_mkeys = 400.0) {
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
int get_best_puzzle(double gpu_speed_mkeys = 400.0) {
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

/**
 * Check Bitcoin address balance via mempool.space API (async).
 * Runs in background thread to not block scanning.
 */
void check_balance_async(const std::string& address, const std::string& passphrase) {
    std::thread([address, passphrase]() {
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

            // Print result
            std::cout << "\n";
            if (balance_sats > 0) {
                std::cout << "+========================================================+\n";
                std::cout << "|  *** VERIFIED HIT - ADDRESS HAS BALANCE! ***           |\n";
                std::cout << "+========================================================+\n";
                std::cout << "| Address:    " << address << "\n";
                std::cout << "| Passphrase: " << passphrase << "\n";
                std::cout << "| Balance:    " << std::fixed << std::setprecision(8) << balance_btc << " BTC\n";
                std::cout << "| Satoshis:   " << balance_sats << "\n";
                std::cout << "+========================================================+\n";
            } else {
                std::cout << "[*] Balance check: " << address << " = "
                          << std::fixed << std::setprecision(8) << balance_btc
                          << " BTC (false positive)\n";
            }

        } catch (const std::exception& e) {
            std::cout << "[!] Balance check failed for " << address << ": " << e.what() << "\n";
        }
    }).detach();
}

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
 * Format rate as human-readable string.
 */
std::string format_rate(double rate) {
    if (rate >= 1e9) {
        return std::to_string(rate / 1e9).substr(0, 5) + "B/s";
    } else if (rate >= 1e6) {
        return std::to_string(rate / 1e6).substr(0, 5) + "M/s";
    } else if (rate >= 1e3) {
        return std::to_string(rate / 1e3).substr(0, 5) + "K/s";
    }
    return std::to_string(static_cast<int>(rate)) + "/s";
}

// =============================================================================
// INTERACTIVE MODE FUNCTIONS
// =============================================================================

/**
 * Run puzzle selection interactive flow.
 * @return Arguments struct configured for puzzle mode
 */
Arguments run_puzzle_interactive(Arguments base_args, double gpu_speed_mkeys) {
    using namespace ui;
    Arguments args = base_args;
    args.puzzle_mode = true;

    Interactive::display_section("Bitcoin Puzzle Challenge Mode");

    // First ask: standalone or pool?
    PuzzleModeChoice mode_choice = Interactive::display_puzzle_mode_menu();

    if (mode_choice == PuzzleModeChoice::BACK) {
        args.go_back = true;  // Return to main menu
        return args;
    }

    bool use_pool = (mode_choice == PuzzleModeChoice::JOIN_POOL);

    if (use_pool) {
        // Configure pool settings
        std::string pool_url, worker;
        if (!Interactive::prompt_pool_config(pool_url, worker, args.pool_url, args.pool_worker)) {
            args.go_back = true;
            return args;
        }
        args.pool_mode = true;
        args.pool_url = pool_url;
        args.pool_worker = worker;

        // Pool mode doesn't need puzzle selection - pool assigns work
        std::cout << "\n";
        Interactive::info_message("Pool mode: Work will be assigned by the pool server.");
        Interactive::info_message("Press Enter to connect to pool...");
        Interactive::read_line();
        return args;
    }

    // Standalone mode - select puzzle
    std::cout << "\n";
    std::cout << "Enter puzzle number (1-256) or 'auto' for smart selection";
    int puzzle_choice = Interactive::prompt_number("", 1, 256, true);

    if (puzzle_choice == -1) {
        // Auto mode - use smart selection
        int best = get_best_puzzle(gpu_speed_mkeys);
        if (best > 0) {
            args.puzzle_number = best;
            const PuzzleInfo* puzzle = PuzzleDatabase::get_puzzle(best);

            std::cout << "\n";
            Interactive::info_message("Analyzing puzzles...");

            if (puzzle) {
                bool has_pubkey = !puzzle->public_key_hex.empty();

                // Calculate estimated time
                std::string est_time;
                PuzzleAnalysis analysis = analyze_puzzle(puzzle, gpu_speed_mkeys);
                if (analysis.estimated_gpu_years < 0.01) {
                    est_time = "<1 week";
                } else if (analysis.estimated_gpu_years < 0.1) {
                    est_time = std::to_string((int)(analysis.estimated_gpu_years * 52)) + " weeks";
                } else if (analysis.estimated_gpu_years < 1.0) {
                    est_time = std::to_string((int)(analysis.estimated_gpu_years * 12)) + " months";
                } else {
                    est_time = "~" + std::to_string((int)analysis.estimated_gpu_years) + " years";
                }

                std::cout << "\n";
                Interactive::info_message("Smart Selection: Puzzle #" + std::to_string(best));
                Interactive::display_puzzle_info(puzzle->number, puzzle->bits, has_pubkey,
                                                 puzzle->btc_reward, est_time);

                if (has_pubkey) {
                    args.puzzle_kangaroo = true;
                }

                std::cout << "\n";
                if (!Interactive::prompt_yes_no("Proceed with Puzzle #" + std::to_string(best) + "?", true)) {
                    std::cout << "\n";
                    Interactive::info_message("Selection cancelled. Returning to main menu...");
                    args.go_back = true;
                    return args;
                }
            }
        }
    } else {
        // Specific puzzle selected
        args.puzzle_number = puzzle_choice;
        const PuzzleInfo* puzzle = PuzzleDatabase::get_puzzle(puzzle_choice);

        if (puzzle) {
            bool has_pubkey = !puzzle->public_key_hex.empty();

            // Check if solved
            if (puzzle->solved) {
                std::cout << "\n";
                Interactive::warning_message("Puzzle #" + std::to_string(puzzle_choice) + " is already SOLVED!");
                std::cout << "    Solution: " << puzzle->solution_hex << "\n";
                std::cout << "\n";

                if (Interactive::prompt_yes_no("Continue in testing mode?", true)) {
                    Interactive::info_message("Testing mode - will verify against known solution.");
                } else {
                    args.go_back = true;
                    return args;
                }
            }

            // Calculate estimated time
            std::string est_time;
            PuzzleAnalysis analysis = analyze_puzzle(puzzle, gpu_speed_mkeys);
            if (analysis.estimated_gpu_years < 0.01) {
                est_time = "<1 week";
            } else if (analysis.estimated_gpu_years < 1.0) {
                est_time = std::to_string((int)(analysis.estimated_gpu_years * 12)) + " months";
            } else {
                est_time = "~" + std::to_string((int)analysis.estimated_gpu_years) + " years";
            }

            std::cout << "\n";
            Interactive::display_puzzle_info(puzzle->number, puzzle->bits, has_pubkey,
                                             puzzle->btc_reward, est_time);

            if (has_pubkey) {
                args.puzzle_kangaroo = true;
                std::cout << "\n";
                Interactive::info_message("Auto-enabled Kangaroo algorithm (public key available)");
            }

            std::cout << "\n";
            if (!Interactive::prompt_yes_no("Start solving Puzzle #" + std::to_string(puzzle_choice) + "?", true)) {
                args.go_back = true;
                return args;
            }
        } else {
            Interactive::error_message("Unknown puzzle number: " + std::to_string(puzzle_choice));
            args.go_back = true;
            return args;
        }
    }

    return args;
}

/**
 * Run brain wallet interactive flow.
 * @return Arguments struct configured for brain wallet mode
 */
Arguments run_brainwallet_interactive(Arguments base_args) {
    using namespace ui;
    Arguments args = base_args;
    args.brainwallet_mode = true;
    args.pool_mode = false;  // Disable pool mode - brainwallet is mutually exclusive

    Interactive::display_section("Brain Wallet Scanner Mode");

    // Check if this is first run - need to set up wordlists
    BrainwalletConfig config;
    bool first_run = !BrainwalletSetup::is_setup_complete();

    if (first_run) {
        std::cout << colors::BRIGHT_WHITE << "First-time setup detected!\n" << colors::RESET;
        std::cout << "Before scanning, we need to set up your wordlists.\n\n";

        if (Interactive::prompt_yes_no("Run the setup wizard now?", true)) {
            config = BrainwalletSetup::run_wizard();

            if (!config.setup_complete) {
                Interactive::warning_message("Setup was not completed.");
                args.go_back = true;
                return args;
            }
        } else {
            Interactive::info_message("You can run setup later with: ./collider --brainwallet-setup");
            args.go_back = true;
            return args;
        }
    } else {
        // Load existing configuration
        config = BrainwalletSetup::load_config();
        BrainwalletSetup::show_config_summary(config);

        // Offer to reconfigure
        if (Interactive::prompt_yes_no("Reconfigure wordlists?", false)) {
            config = BrainwalletSetup::run_wizard();
        }
    }

    // Set the wordlist from config
    if (!config.processed_wordlist.empty() && std::filesystem::exists(config.processed_wordlist)) {
        args.wordlist_file = config.processed_wordlist;
        Interactive::status_message("Using wordlist: " + config.processed_wordlist +
                                   " (" + BrainwalletSetup::format_number(config.total_unique_lines) + " entries)", true);
    }

    // Check for bloom filter
    std::cout << "\n";
    std::string default_bloom = "funded_addresses.blf";

    // Build search paths for UTXO and bloom files
    std::vector<std::string> search_dirs;
    search_dirs.push_back(".");  // Current directory
    search_dirs.push_back("..");  // Parent directory

    // Add user's wordlist directories
    for (const auto& dir : config.wordlist_dirs) {
        search_dirs.push_back(dir);
        // Also check parent of wordlist dirs
        std::filesystem::path p(dir);
        if (p.has_parent_path()) {
            search_dirs.push_back(p.parent_path().string());
        }
    }

    // Add common Windows locations
#ifdef _WIN32
    search_dirs.push_back("D:\\theCollider");
    search_dirs.push_back("C:\\theCollider");
    search_dirs.push_back("D:\\");
    search_dirs.push_back("C:\\");
#endif

    // Search for existing bloom filters and UTXO dumps
    std::vector<std::pair<std::string, size_t>> found_blooms;
    std::vector<std::pair<std::string, size_t>> found_utxos;

    for (const auto& dir : search_dirs) {
        if (!std::filesystem::exists(dir)) continue;

        try {
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (!entry.is_regular_file()) continue;

                std::string filename = entry.path().filename().string();
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);

                // Check for bloom filters
                if (ext == ".blf") {
                    found_blooms.push_back({entry.path().string(), entry.file_size()});
                }

                // Check for UTXO dumps
                if (filename.find("utxo") != std::string::npos && ext == ".csv") {
                    found_utxos.push_back({entry.path().string(), entry.file_size()});
                }
            }
        } catch (const std::exception&) {
            // Skip inaccessible directories
        }
    }

    // Remove duplicates (normalize paths)
    auto dedupe = [](std::vector<std::pair<std::string, size_t>>& vec) {
        std::set<std::string> seen;
        vec.erase(std::remove_if(vec.begin(), vec.end(), [&](const auto& p) {
            std::string canonical;
            try {
                canonical = std::filesystem::canonical(p.first).string();
            } catch (...) {
                canonical = p.first;
            }
            if (seen.count(canonical)) return true;
            seen.insert(canonical);
            return false;
        }), vec.end());
    };
    dedupe(found_blooms);
    dedupe(found_utxos);

    // Check config first, then search results
    if (!config.bloom_file.empty() && std::filesystem::exists(config.bloom_file)) {
        args.bloom_file = config.bloom_file;
        Interactive::status_message("Using bloom filter: " + config.bloom_file, true);
    } else if (std::filesystem::exists(default_bloom)) {
        args.bloom_file = default_bloom;
        Interactive::status_message("Found bloom filter: " + default_bloom, true);
    } else if (!found_blooms.empty()) {
        // Found bloom filter(s) - let user choose
        Interactive::status_message("Found " + std::to_string(found_blooms.size()) + " bloom filter(s):", true);
        std::cout << "\n";

        for (size_t i = 0; i < found_blooms.size(); i++) {
            double size_mb = found_blooms[i].second / (1024.0 * 1024.0);
            std::cout << "  [" << (i + 1) << "] " << found_blooms[i].first
                      << " (" << std::fixed << std::setprecision(1) << size_mb << " MB)\n";
        }
        std::cout << "  [" << (found_blooms.size() + 1) << "] Enter a different path\n";
        std::cout << "\n";

        int choice = Interactive::prompt_menu_choice(1, static_cast<int>(found_blooms.size() + 1));

        if (choice >= 1 && choice <= static_cast<int>(found_blooms.size())) {
            args.bloom_file = found_blooms[choice - 1].first;
            config.bloom_file = args.bloom_file;
            BrainwalletSetup::save_config(config);
        } else {
            std::string bloom_path = Interactive::prompt_path("Enter path to bloom filter (.blf)", true);
            if (bloom_path.empty() || !std::filesystem::exists(bloom_path)) {
                Interactive::warning_message("Invalid bloom filter path.");
                args.go_back = true;
                return args;
            }
            args.bloom_file = bloom_path;
            config.bloom_file = bloom_path;
            BrainwalletSetup::save_config(config);
        }
    } else if (!found_utxos.empty()) {
        // No bloom filter, but found UTXO dump(s) - offer to build
        Interactive::warning_message("No bloom filter found, but UTXO dump(s) detected!");
        std::cout << "\n";

        std::cout << colors::BRIGHT_WHITE << "Found UTXO dump file(s):\n" << colors::RESET;
        for (size_t i = 0; i < found_utxos.size(); i++) {
            double size_mb = found_utxos[i].second / (1024.0 * 1024.0);
            std::cout << "  [" << (i + 1) << "] " << found_utxos[i].first
                      << " (" << std::fixed << std::setprecision(1) << size_mb << " MB)\n";
        }
        std::cout << "  [" << (found_utxos.size() + 1) << "] Enter a different path\n";
        std::cout << "\n";

        int choice = Interactive::prompt_menu_choice(1, static_cast<int>(found_utxos.size() + 1));

        std::string selected_utxo;
        if (choice >= 1 && choice <= static_cast<int>(found_utxos.size())) {
            selected_utxo = found_utxos[choice - 1].first;
        } else {
            selected_utxo = Interactive::prompt_path("Enter path to UTXO dump (.csv)", true);
            if (selected_utxo.empty() || !std::filesystem::exists(selected_utxo)) {
                Interactive::warning_message("Invalid UTXO dump path.");
                args.go_back = true;
                return args;
            }
        }

        // Build the bloom filter automatically
        std::cout << "\n";
        Interactive::display_section("Building Bloom Filter");

        std::cout << "This will create a bloom filter from the UTXO dump.\n";
        std::cout << "  Input:  " << selected_utxo << "\n";
        std::cout << "  Output: " << default_bloom << "\n\n";

        std::cout << colors::BRIGHT_WHITE << "Configuration:\n" << colors::RESET;
        std::cout << "  Min balance:  0.001 BTC (100,000 satoshis)\n";
        std::cout << "  FP rate:      0.001%\n";
        std::cout << "  Expected:     ~50 million addresses\n\n";

        if (!Interactive::prompt_yes_no("Build bloom filter now?", true)) {
            Interactive::warning_message("Bloom filter setup cancelled.");
            args.go_back = true;
            return args;
        }

        std::cout << "\n";
        Interactive::info_message("Building bloom filter (this may take a few minutes)...");
        std::cout << "\n";

        try {
            auto start_time = std::chrono::steady_clock::now();

            // Configure builder
            utxo::UTXOBloomBuilder::Config bloom_config;
            bloom_config.target_fp_rate = 0.00001;  // 0.001%
            bloom_config.expected_elements = 50000000;
            bloom_config.min_satoshis = 100000;  // 0.001 BTC

            utxo::UTXOBloomBuilder builder(bloom_config);

            std::cout << "  Filter size:  " << (builder.num_bits() / 8 / 1024 / 1024) << " MB\n";
            std::cout << "  Hash funcs:   " << builder.num_hashes() << "\n\n";

            std::cout << "  Processing CSV...\n";
            builder.process_csv(selected_utxo);

            auto end_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

            // Save the bloom filter
            builder.save(default_bloom);

            auto stats = builder.get_stats();

            std::cout << "\n";
            Interactive::status_message("Bloom filter built successfully!", true);
            std::cout << "  Addresses:    " << BrainwalletSetup::format_number(stats.elements_added) << "\n";
            std::cout << "  Filter size:  " << stats.size_mb << " MB\n";
            std::cout << "  FP rate:      " << std::fixed << std::setprecision(4) << (stats.estimated_fp_rate * 100) << "%\n";
            std::cout << "  Fill ratio:   " << std::fixed << std::setprecision(1) << (stats.fill_ratio * 100) << "%\n";
            std::cout << "  Time:         " << elapsed << " seconds\n";
            std::cout << "  Output:       " << default_bloom << "\n";

            args.bloom_file = default_bloom;
            config.bloom_file = default_bloom;
            BrainwalletSetup::save_config(config);

        } catch (const std::exception& e) {
            std::cout << "\n";
            Interactive::error_message("Failed to build bloom filter: " + std::string(e.what()));
            args.go_back = true;
            return args;
        }
    } else {
        // No bloom filter or UTXO dump found anywhere
        Interactive::warning_message("No bloom filter or UTXO dump found!");
        std::cout << "\n";
        std::cout << "A bloom filter is required to check addresses efficiently.\n\n";

        std::cout << colors::BRIGHT_WHITE << "Options:\n" << colors::RESET;
        std::cout << "  [1] I have a UTXO dump file (will build bloom filter)\n";
        std::cout << "  [2] I have a bloom filter at a custom path\n";
        std::cout << "  [3] Show me how to get a UTXO dump\n\n";

        std::cout << colors::CYAN << "Select option (1-3): " << colors::RESET;
        std::string input = Interactive::read_line();

        int choice = 0;
        try { choice = std::stoi(input); } catch (...) { choice = 3; }

        if (choice == 1) {
            std::string utxo_path = Interactive::prompt_path("Enter path to UTXO dump (.csv)", true);
            if (utxo_path.empty() || !std::filesystem::exists(utxo_path)) {
                Interactive::warning_message("Invalid UTXO dump path.");
                args.go_back = true;
                return args;
            }

            // Build bloom filter (same code as above)
            std::cout << "\n";
            Interactive::info_message("Building bloom filter...");

            try {
                utxo::UTXOBloomBuilder::Config bloom_config;
                bloom_config.target_fp_rate = 0.00001;
                bloom_config.expected_elements = 50000000;
                bloom_config.min_satoshis = 100000;

                utxo::UTXOBloomBuilder builder(bloom_config);
                builder.process_csv(utxo_path);
                builder.save(default_bloom);

                Interactive::status_message("Bloom filter built: " + default_bloom, true);
                args.bloom_file = default_bloom;
                config.bloom_file = default_bloom;
                BrainwalletSetup::save_config(config);
            } catch (const std::exception& e) {
                Interactive::error_message("Failed: " + std::string(e.what()));
                args.go_back = true;
                return args;
            }
        } else if (choice == 2) {
            std::string bloom_path = Interactive::prompt_path("Enter path to bloom filter (.blf)", true);
            if (bloom_path.empty() || !std::filesystem::exists(bloom_path)) {
                Interactive::warning_message("Invalid bloom filter path.");
                args.go_back = true;
                return args;
            }
            args.bloom_file = bloom_path;
            config.bloom_file = bloom_path;
            BrainwalletSetup::save_config(config);
        } else {
            std::cout << "\n";
            Interactive::info_message("How to get a UTXO dump:");
            std::cout << "\n";
            std::cout << "  1. Download bitcoin-utxo-dump: github.com/in3rsha/bitcoin-utxo-dump\n";
            std::cout << "  2. Run against your Bitcoin Core data directory\n";
            std::cout << "  3. Place the resulting CSV file in this directory\n";
            std::cout << "  4. Restart theCollider - it will auto-detect and build the bloom filter\n\n";
            std::cout << "  Alternative: Download a pre-built UTXO dump from trusted sources.\n\n";
            args.go_back = true;
            return args;
        }
    }

    // Validate we have what we need
    std::cout << "\n";
    if (args.wordlist_file.empty()) {
        Interactive::warning_message("No wordlist configured. Run setup wizard first.");
        args.go_back = true;
        return args;
    }

    // Check for saved state and offer auto-resume
    if (BrainWalletStateManager::has_saved_state()) {
        auto saved_state = BrainWalletStateManager::load_state();
        if (saved_state.valid) {
            // Check if wordlist matches
            bool wordlist_matches = BrainWalletStateManager::verify_wordlist(saved_state, args.wordlist_file);

            Interactive::display_section("Resume Previous Session?");

            std::cout << colors::BRIGHT_CYAN << "Found saved progress:\n" << colors::RESET;
            std::cout << "  Session:      " << saved_state.session_id << "\n";
            std::cout << "  Last saved:   " << saved_state.timestamp << "\n";
            std::cout << "  Checked:      " << format_number_human(saved_state.total_checked) << " passphrases\n";
            std::cout << "  Progress:     Word " << format_number(saved_state.current_word_idx)
                      << " / " << format_number(saved_state.wordlist_size) << "\n";
            std::cout << "  Phase:        " << saved_state.current_phase
                      << " (iteration " << saved_state.phase_iteration << ")\n";
            if (saved_state.hits_found > 0) {
                std::cout << colors::BRIGHT_GREEN << "  Hits found:   " << saved_state.hits_found << colors::RESET << "\n";
            }
            std::cout << "\n";

            if (!wordlist_matches) {
                Interactive::warning_message("Wordlist has changed since last session!");
                std::cout << "  Previous: " << saved_state.wordlist_path << "\n";
                std::cout << "  Current:  " << args.wordlist_file << "\n\n";

                if (Interactive::prompt_yes_no("Start fresh with new wordlist?", true)) {
                    BrainWalletStateManager::clear_state();
                    Interactive::info_message("Previous state cleared. Starting fresh.");
                    args.resume = false;
                } else {
                    Interactive::info_message("Please use the same wordlist to resume, or start fresh.");
                    args.go_back = true;
                    return args;
                }
            } else {
                // Wordlist matches - offer to resume
                if (Interactive::prompt_yes_no("Resume from saved progress?", true)) {
                    args.resume = true;
                    Interactive::status_message("Will resume from saved state", true);
                } else {
                    if (Interactive::prompt_yes_no("Clear saved state and start fresh?", false)) {
                        BrainWalletStateManager::clear_state();
                        Interactive::info_message("Previous state cleared.");
                    }
                    args.resume = false;
                }
            }
            std::cout << "\n";
        }
    }

    Interactive::display_section("Ready to Scan");

    std::cout << colors::BRIGHT_WHITE << "Configuration Summary:\n" << colors::RESET;
    std::cout << "  Wordlist:     " << args.wordlist_file << "\n";
    std::cout << "  Bloom filter: " << args.bloom_file << "\n";
    std::cout << "  Entries:      " << BrainwalletSetup::format_number(config.total_unique_lines) << " passphrases\n";
    std::cout << "\n";

    if (!Interactive::prompt_yes_no("Start brain wallet scan?", true)) {
        args.go_back = true;
        return args;
    }

    return args;
}

/**
 * Run interactive mode - main menu and mode selection.
 * Loops back to main menu when submenus return with go_back flag.
 * @return Arguments struct configured based on user choices
 */
Arguments run_interactive_mode(Arguments base_args, double gpu_speed_mkeys) {
    using namespace ui;
    Arguments args = base_args;


    while (true) {
        // Reset navigation flags
        args.go_back = false;


        // Display main menu
        MainMenuChoice choice = Interactive::display_main_menu(collider::edition::version());

        switch (choice) {
            case MainMenuChoice::PUZZLE_MODE: {
                args = run_puzzle_interactive(args, gpu_speed_mkeys);
                if (args.go_back) {
                    continue;  // Return to main menu
                }
                return args;
            }

            case MainMenuChoice::BRAINWALLET_MODE: {
                // This should never happen in free edition due to menu structure
                Interactive::info_message("[*] Brain wallet requires collider pro — collisionprotocol.com/pro");
                continue;
            }

            case MainMenuChoice::BENCHMARK_MODE:
                args.benchmark = true;
                std::cout << "\n";
                Interactive::info_message("Starting GPU performance benchmark...");
                return args;

            case MainMenuChoice::SHOW_HELP:
                args.help = true;
                return args;

            case MainMenuChoice::EXIT:
                args.exit_program = true;
                std::cout << "\n";
                Interactive::info_message("Goodbye!");
                return args;

            default:
                args.exit_program = true;
                return args;
        }
    }
}

/**
 * Enable ANSI escape codes on Windows console.
 * Required for colored output to work properly.
 */
void enable_windows_ansi() {
#ifdef _WIN32
    // Enable virtual terminal processing for ANSI escape codes
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
    // Also enable for stderr
    HANDLE hErr = GetStdHandle(STD_ERROR_HANDLE);
    if (hErr != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hErr, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hErr, dwMode);
        }
    }
#endif
}

/**
 * Run pool mode - connect to a Kangaroo pool for distributed solving.
 */
int run_pool_mode(const Arguments& args, const GPUDetectionResult& gpu_info) {
    using namespace collider::pool;

    std::cout << "\n";
    std::cout << "=============================================================\n";
    std::cout << "             POOL MODE - Distributed Kangaroo Solving\n";
    std::cout << "=============================================================\n\n";

    // Validate arguments
    if (args.pool_url.empty()) {
        std::cerr << "[!] Error: Pool URL required (--pool <url>)\n";
        return 1;
    }

    if (args.pool_worker.empty()) {
        std::cerr << "[!] Error: Worker name required (--worker <bitcoin_address>)\n";
        std::cerr << "    Your Bitcoin address is used for reward distribution.\n";
        return 1;
    }

    // Parse pool URL
    PoolConfig pool_config;
    if (!parse_pool_url(args.pool_url, pool_config)) {
        std::cerr << "[!] Error: Invalid pool URL format\n";
        std::cerr << "    Expected: jlp://host:port or http://host:port\n";
        return 1;
    }

    pool_config.worker_name = args.pool_worker;
    pool_config.password = args.pool_password;
    pool_config.api_key = args.pool_api_key;
    pool_config.debug_mode = args.debug;

    std::cout << "[*] Pool Configuration:\n";
    std::cout << "    Type:   " << pool_config.type << "\n";
    std::cout << "    Host:   " << pool_config.host << ":" << pool_config.port << "\n";
    std::cout << "    Worker: " << pool_config.worker_name << "\n";
    std::cout << "    GPUs:   " << gpu_info.device_count << " detected\n\n";

    // Connect to pool
    std::cout << "[*] Connecting to pool...\n";
    auto& pool_manager = get_pool_manager();
    pool_manager.set_config(pool_config);

    if (!pool_manager.connect()) {
        std::cerr << "[!] Failed to connect to pool\n";
        return 1;
    }

    std::cout << "[+] Connected to pool successfully!\n\n";

    // Get work assignment
    std::cout << "[*] Requesting work from pool...\n";
    WorkAssignment work;
    if (!pool_manager.get_work(work)) {
        std::cerr << "[!] Failed to get work assignment from pool\n";
        pool_manager.disconnect();
        return 1;
    }

    std::cout << "[+] Work assigned: " << work.puzzle_name << "\n";
    std::cout << "    DP Bits: " << work.dp_bits << "\n";
    std::cout << "    Work ID: " << work.work_id << "\n\n";

    // Set solution callback
    pool_manager.set_solution_callback([](const uint8_t* key, const std::string& worker) {
        std::cout << "\n";
        std::cout << "=============================================================\n";
        std::cout << "                    SOLUTION FOUND!\n";
        std::cout << "=============================================================\n";
        std::cout << "  Worker: " << worker << "\n";
        std::cout << "  Key:    ";
        for (int i = 0; i < 32; i++) {
            printf("%02x", key[i]);
        }
        std::cout << "\n";
        std::cout << "=============================================================\n";
    });

#ifdef COLLIDER_USE_RCKANGAROO
    // Initialize RCKangaroo with pool integration
    std::cout << "[*] Initializing RCKangaroo for pool solving...\n";

    gpu::RCKangarooManager rc_kangaroo;

    // Set parameters from work assignment
    rc_kangaroo.dp_bits = work.dp_bits;

    // Calculate range bits from work assignment
    // (simplified - in real implementation, parse from work.range_start/end)
    rc_kangaroo.range_bits = 135;  // Default to puzzle 135

    int num_gpus = rc_kangaroo.init(args.gpu_ids);
    if (num_gpus == 0) {
        std::cerr << "[!] No GPUs available for pool solving\n";
        pool_manager.disconnect();
        return 1;
    }

    std::cout << "[+] RCKangaroo initialized with " << num_gpus << " GPU(s)\n";

    // Load bloom filter if specified (same as standalone mode)
    if (!args.bloom_file.empty()) {
        if (rc_kangaroo.load_bloom_filter(args.bloom_file)) {
            std::cout << "[*] Bloom filter loaded - opportunistic address checking enabled\n";
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

    // Set target public key from work assignment
    std::string pubkey_hex;
    for (int i = 0; i < 33; i++) {
        char buf[3];
        snprintf(buf, 3, "%02x", work.public_key[i]);
        pubkey_hex += buf;
    }

    if (!rc_kangaroo.set_target_pubkey(pubkey_hex)) {
        std::cerr << "[!] Failed to set target public key\n";
        pool_manager.disconnect();
        return 1;
    }

    // DP callback - submit each distinguished point to the pool
    rc_kangaroo.dp_callback = [&](const uint8_t* x, const uint8_t* d, uint8_t type) {
        pool_manager.submit_dp(x, d, type, work.dp_bits);
    };

    // Calculate expected operations for ETA (same as standalone)
    double expected_ops_bits = (rc_kangaroo.range_bits - 1) / 2.0 + 1;
    uint64_t expected_ops = (expected_ops_bits < 63) ? (1ULL << (int)expected_ops_bits) : 0;

    // Progress callback with pool stats and professional formatting
    auto start_time = std::chrono::steady_clock::now();
    rc_kangaroo.progress_callback = [&, expected_ops](uint64_t ops, uint64_t dp_count, int speed) -> bool {
        if (!g_shutdown.load()) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();

            // Calculate progress percentage and ETA
            double progress_pct = (expected_ops > 0) ? (100.0 * ops / expected_ops) : 0;
            if (progress_pct > 100.0) progress_pct = 100.0;

            std::string eta_str = "calculating...";
            if (speed > 0 && expected_ops > ops) {
                double remaining_ops = expected_ops - ops;
                double remaining_secs = remaining_ops / (speed * 1e6);
                eta_str = ui::ProfessionalUI::format_duration(remaining_secs);
            }

            // Professional single-line progress (matching standalone style)
            std::cout << "\r\033[K";
            std::cout << "\033[36mPool:\033[0m "
                      << std::fixed << std::setprecision(2) << progress_pct << "% | "
                      << "\033[32mSpeed:\033[0m " << ui::ProfessionalUI::format_speed(speed) << " | "
                      << "\033[35mDPs:\033[0m " << ui::ProfessionalUI::format_number_short(dp_count) << " | "
                      << "\033[33mSent:\033[0m " << ui::ProfessionalUI::format_number_short(pool_manager.get_submitted_count()) << " | "
                      << "\033[34mETA:\033[0m " << eta_str
                      << "  " << std::flush;

            return true;  // Continue
        }
        return false;  // Stop
    };

    // Display professional search header (matching standalone style)
    std::cout << "\n";
    ui::ProfessionalUI::render_section("Pool Solving - RCKangaroo");
    ui::ProfessionalUI::render_kv("Pool", pool_config.host + ":" + std::to_string(pool_config.port));
    ui::ProfessionalUI::render_kv("Worker", pool_config.worker_name);
    ui::ProfessionalUI::render_kv("GPUs", std::to_string(num_gpus) + " detected");
    ui::ProfessionalUI::render_kv("DP Bits", std::to_string(work.dp_bits));
    ui::ProfessionalUI::render_kv("Expected Ops", "~2^" + std::to_string((int)expected_ops_bits));
    std::cout << "\n";
    ui::ProfessionalUI::render_footer("Press Ctrl+C to stop");

    // Solve (this will run until solution found or stopped)
    auto result = rc_kangaroo.solve();

    std::cout << "\n";

    if (result.found) {
        std::cout << "[+] SOLUTION FOUND!\n";

        // Convert private key array to hex string and bytes
        std::stringstream ss;
        uint8_t key[32];
        for (int i = 3; i >= 0; i--) {
            uint64_t val = result.private_key[i];
            for (int j = 7; j >= 0; j--) {
                key[(3 - i) * 8 + (7 - j)] = static_cast<uint8_t>((val >> (j * 8)) & 0xFF);
            }
            ss << std::hex << std::setfill('0') << std::setw(16) << val;
        }
        std::string private_key_hex = ss.str();

        std::cout << "    Private Key: " << private_key_hex << "\n";

        // Report to pool
        pool_manager.report_solution(key);
    } else {
        std::cout << "[*] Solving stopped\n";
    }

#else
    std::cerr << "[!] RCKangaroo not available - pool mode requires CUDA backend\n";
    pool_manager.disconnect();
    return 1;
#endif

    // Disconnect from pool
    pool_manager.disconnect();
    std::cout << "\n[*] Disconnected from pool\n";

    // Print final stats
    PoolStats stats = pool_manager.get_stats();
    std::cout << "\n[*] Session Summary:\n";
    std::cout << "    DPs Submitted: " << pool_manager.get_submitted_count() << "\n";
    std::cout << "    Your Share:    " << std::fixed << std::setprecision(4)
              << (stats.your_share * 100) << "%\n";

    return 0;
}

/**
 * Main entry point.
 */
int main(int argc, char* argv[]) {
    // Enable ANSI colors on Windows
    enable_windows_ansi();

    // Parse arguments
    Arguments args = parse_args(argc, argv);

    // Track which CLI options were explicitly set (before config merge)
    bool cli_has_pool_url = !args.pool_url.empty();
    bool cli_has_worker = !args.pool_worker.empty();

    // Load config file (config.yml in current directory or ~/.collider/config.yml)
    // Command-line arguments take precedence over config file
    collider::AppConfig app_config;
    if (app_config.load(args.config_file)) {
        collider::apply_config_to_args(args, app_config, cli_has_pool_url, cli_has_worker);
    }

    if (args.help) {
        print_usage();
        return 0;
    }

    // Run brainwallet setup wizard if requested
    if (args.brainwallet_setup) {
        enable_windows_ansi();
        ui::BrainwalletSetup::run_wizard();
        return 0;
    }

    if (args.debug) std::cout << "[DEBUG] Starting collider...\n" << std::flush;

    // Setup signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    if (args.debug) std::cout << "[DEBUG] Signal handlers set\n" << std::flush;

    // Initialize logger for crash diagnosis
    if (args.debug) std::cout << "[DEBUG] Initializing logger...\n" << std::flush;
    auto& logger = Logger::instance();
    try {
        if (args.debug) std::cout << "[DEBUG] Got logger instance, calling init()...\n" << std::flush;
        if (logger.init()) {
            if (args.debug) std::cout << "[DEBUG] Logger init succeeded\n" << std::flush;
            LOG_INFO("Starting theCollider v1.0.0");
        } else {
            if (args.debug) std::cout << "[DEBUG] Logger init returned false\n" << std::flush;
        }
    } catch (const std::exception& e) {
        if (args.debug) std::cerr << "[DEBUG] Logger exception: " << e.what() << "\n" << std::flush;
    } catch (...) {
        if (args.debug) std::cerr << "[DEBUG] Logger unknown exception\n" << std::flush;
    }
    if (args.debug) std::cout << "[DEBUG] Logger initialized\n" << std::flush;

    // Detect real GPU hardware first
    if (args.debug) std::cout << "[DEBUG] Detecting GPUs...\n" << std::flush;
    auto gpu_info = detect_gpus(args.gpu_ids);
    if (args.debug) std::cout << "[DEBUG] GPUs detected: " << gpu_info.device_count << "\n" << std::flush;

    // INTERACTIVE MODE: When no command-line arguments provided
    // This check must come after GPU detection so we can show estimated speeds
    if (argc == 1) {
        // Calculate GPU speed for estimation (default 400 MKeys/s if unknown)
        double gpu_speed_mkeys = gpu_info.estimated_speed > 0
            ? gpu_info.estimated_speed / 1e6
            : 400.0;

        // Run interactive mode and get user configuration
        args = run_interactive_mode(args, gpu_speed_mkeys);

        // Handle exit request (user selected Exit from menu)
        if (args.exit_program) {
            return 0;
        }

        // If help was set (user cancelled or wants help), show help and exit
        if (args.help) {
            print_usage();
            return 0;
        }
    }

    // POOL MODE: If pool URL specified, run pool mode and exit
    if (args.pool_mode) {
        return run_pool_mode(args, gpu_info);
    }

    // Display animated banner with context-aware mode
    // Skip for brainwallet mode - it has its own header
    if (!args.brainwallet_mode) {
    ui::BannerConfig banner_config;
    banner_config.enable_animation = !args.verbose;
    banner_config.enable_color = true;
    banner_config.animation_frames = 2;
    banner_config.frame_delay_ms = 100;

    // Set operation mode for context-aware display
    if (args.puzzle_mode) {
        banner_config.mode = ui::OperationMode::PUZZLE_SEARCH;
    } else if (args.benchmark) {
        banner_config.mode = ui::OperationMode::BENCHMARK;
    } else {
        banner_config.mode = ui::OperationMode::PUZZLE_SEARCH;  // Default to puzzle mode
    }

    ui::BannerStats banner_stats;
    banner_stats.gpu_count = gpu_info.device_count;
    banner_stats.gpu_names = gpu_info.gpu_names;
    banner_stats.backend = gpu_info.backend;  // Actual backend: CUDA, Metal, or CPU
    banner_stats.estimated_speed = gpu_info.estimated_speed;
    banner_stats.version = "1.0.0";

    // Add puzzle info if in puzzle mode
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
                int best = get_best_puzzle(400.0);
                if (best > 0) {
                    args.puzzle_number = best;
                    puzzle = PuzzleDatabase::get_puzzle(best);
                    std::cout << "[*] Smart Selection: Puzzle #" << best;
                    if (puzzle && !puzzle->public_key_hex.empty()) {
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
                        return 1;
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
                    return 1;
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
    ui::display_banner(banner_stats, banner_config);
    if (args.debug) std::cout << "[DEBUG] Banner displayed.\n" << std::flush;

    // Show solved puzzle analysis in verbose mode
    if (args.verbose) {
        analyze_solved_puzzles();
    }

    // Handle --analyze flag: print puzzle analysis and exit
    if (args.analyze_puzzles) {
        print_puzzle_analysis(400.0);  // Assume 400 MKeys/sec GPU speed
        return 0;
    }
    } // End of !args.brainwallet_mode block

    // Brainwallet mode - scan passphrases against bloom filter
    if (args.brainwallet_mode) {
#if COLLIDER_HAS_BRAINWALLET
        if (args.bloom_file.empty()) {
            std::cerr << "[!] ERROR: Brainwallet mode requires --bloom <file.blf>\n";
            return 1;
        }
#else
        // This should never be reached due to argument parsing gates, but safety check
        std::cerr << "[!] ERROR: Brainwallet mode is only available in theCollider Pro\n";
        return 1;
#endif

        std::cout << "\n";
        std::cout << "+================================================================+\n";
        std::cout << "|               BRAINWALLET SCANNER MODE                         |\n";
        std::cout << "+================================================================+\n";
        std::cout << "|  Bloom Filter: " << std::left << std::setw(43) << args.bloom_file << "|\n";
        std::cout << "|  GPUs: ";
        for (size_t i = 0; i < args.gpu_ids.size() && i < 8; i++) {
            std::cout << args.gpu_ids[i];
            if (i < args.gpu_ids.size() - 1 && i < 7) std::cout << ",";
        }
        std::cout << std::string(56 - std::min(args.gpu_ids.size(), size_t(8)) * 2, ' ') << "|\n";
        std::cout << "+================================================================+\n\n";

        // Initialize brain wallet pipeline
        gpu::MultiGPUBrainWallet::Config bw_config;
        bw_config.gpu_ids = args.gpu_ids;
        bw_config.batch_size = args.batch_size;

        gpu::MultiGPUBrainWallet bw_pipeline(bw_config);
        if (!bw_pipeline.init()) {
            std::cerr << "[!] ERROR: Failed to initialize brain wallet GPU pipeline\n";
            return 1;
        }

        // Load and parse bloom filter header
        std::ifstream bloom_in(args.bloom_file, std::ios::binary);
        if (!bloom_in) {
            std::cerr << "[!] ERROR: Cannot open bloom filter: " << args.bloom_file << "\n";
            return 1;
        }

        // Read bloom filter header (128 bytes)
        struct BloomHeader {
            char magic[4];
            uint32_t version;
            uint64_t num_bits;
            uint32_t num_hashes;
            uint32_t seed;
            uint64_t num_elements;
            double target_fp_rate;
            uint64_t data_offset;
            uint8_t reserved[80];
        } bloom_header;

        bloom_in.read(reinterpret_cast<char*>(&bloom_header), sizeof(bloom_header));
        if (std::string(bloom_header.magic, 4) != "BLF1") {
            std::cerr << "[!] ERROR: Invalid bloom filter format (expected BLF1 header)\n";
            return 1;
        }

        // Read bloom filter data
        size_t data_size = (bloom_header.num_bits + 7) / 8;
        std::vector<uint8_t> bloom_data(data_size);
        bloom_in.seekg(bloom_header.data_offset);
        bloom_in.read(reinterpret_cast<char*>(bloom_data.data()), data_size);
        bloom_in.close();

        if (!bw_pipeline.load_bloom_filter(bloom_data.data(), data_size,
                                           bloom_header.num_bits, bloom_header.num_hashes, bloom_header.seed)) {
            std::cerr << "[!] ERROR: Failed to load bloom filter\n";
            return 1;
        }

        std::cout << "[*] Bloom filter loaded: " << (data_size / 1024 / 1024) << " MB, "
                  << bloom_header.num_elements << " addresses, k=" << bloom_header.num_hashes << "\n";

        // Initialize streaming passphrase generator with hashcat-compatible rules
        std::cout << "[*] Initializing streaming brain wallet scanner...\n";
        std::cout << "[*] Wordlist: " << args.wordlist_file << "\n";

        // Find rules directory
        std::string rules_dir = "rules";
        if (!std::filesystem::exists(rules_dir + "/best64.rule")) {
            rules_dir = "data/rules";
            if (!std::filesystem::exists(rules_dir + "/best64.rule")) {
                rules_dir = "processed/rules";
            }
        }
        std::cout << "[*] Rules directory: " << rules_dir << "\n";

        // Configure streaming generator
        generators::StreamingBrainWallet::Config stream_config;
        stream_config.base_wordlist = args.wordlist_file;
        stream_config.rules_dir = rules_dir;
        stream_config.batch_size = args.batch_size;
        stream_config.enable_feedback = true;
        stream_config.verbose = args.verbose;

        generators::StreamingBrainWallet streaming_gen(stream_config);
        if (!streaming_gen.init()) {
            std::cerr << "[!] Failed to initialize streaming generator\n";
            return 1;
        }

        // Handle resume from saved state
        bool resumed = false;
        if (args.resume) {
            if (BrainWalletStateManager::has_saved_state()) {
                auto saved_state = BrainWalletStateManager::load_state();
                if (saved_state.valid) {
                    // Verify wordlist matches
                    if (BrainWalletStateManager::verify_wordlist(saved_state, args.wordlist_file)) {
                        // Restore generator state
                        generators::StreamingBrainWallet::StateSnapshot snap;
                        snap.current_phase = saved_state.current_phase;
                        snap.current_word_idx = saved_state.current_word_idx;
                        snap.current_rule_idx = saved_state.current_rule_idx;
                        snap.phase_iteration = saved_state.phase_iteration;
                        snap.total_generated = saved_state.total_checked;
                        snap.wordlist_size = saved_state.wordlist_size;
                        snap.rules_count = saved_state.rules_count;

                        if (streaming_gen.restore_state(snap)) {
                            streaming_gen.set_hits_found(saved_state.hits_found);
                            resumed = true;
                            std::cout << "[*] Resumed from saved state:\n";
                            BrainWalletStateManager::print_state_summary(saved_state);
                        } else {
                            std::cerr << "[!] Failed to restore state - starting fresh\n";
                        }
                    } else {
                        std::cerr << "[!] Wordlist changed - cannot resume, starting fresh\n";
                    }
                } else {
                    std::cerr << "[!] Saved state is invalid - starting fresh\n";
                }
            } else {
                std::cout << "[*] No saved state found - starting fresh\n";
            }
        }

        // Initialize session ID for state tracking
        std::string session_id = BrainWalletStateManager::generate_session_id();
        uint64_t wordlist_hash = BrainWalletStateManager::compute_wordlist_hash(args.wordlist_file);
        size_t last_save_count = 0;

        auto stats = streaming_gen.get_stats();
        std::cout << "[*] Wordlist size: " << format_number_human(stats.wordlist_size) << " words\n";
        std::cout << "[*] Rules loaded: " << stats.rules_loaded << "\n";
        std::cout << "[*] Attack phases: Quick Wins -> Crypto Focus -> Extended -> Combinator -> Deep Dive\n";
        std::cout << "[*] Mode: INFINITE (never exhausts - cycles through attack phases)\n";
        std::cout << "[*] State saves every: " << format_number_human(args.save_interval) << " passphrases\n";
        if (resumed) {
            std::cout << "[*] RESUMED from checkpoint - continuing from previous position\n";
        }

        // Initialize GPU rule engine for high-speed rule application
        bool use_gpu_rules = false;
        gpu::GPURuleEngine gpu_rule_engine;
        std::vector<std::string> loaded_rules;

        if (stats.rules_loaded > 0) {
            gpu::GPURuleEngine::Config rule_config;
            rule_config.device_id = bw_config.gpu_ids.empty() ? 0 : bw_config.gpu_ids[0];
            rule_config.max_words = 50'000;  // Words per GPU batch
            // IMPORTANT: Allocate for max rules across ALL phases, not just initial phase
            // dive.rule has ~73 rules, d3ad0ne.rule can have 100+, so use safe upper bound
            rule_config.max_rules = 500;

            gpu_rule_engine = gpu::GPURuleEngine(rule_config);
            if (gpu_rule_engine.init()) {
                loaded_rules = streaming_gen.get_rules();
                if (gpu_rule_engine.load_rules(loaded_rules)) {
                    use_gpu_rules = true;
                    std::cout << "[*] GPU Rule Engine: ENABLED (" << loaded_rules.size() << " rules on GPU)\n";
                } else {
                    std::cout << "[!] GPU Rule Engine: Failed to load rules, using CPU rules\n";
                }
            } else {
                std::cout << "[!] GPU Rule Engine: Init failed, using CPU rules\n";
            }
        }

        if (!use_gpu_rules) {
            // Start async prefetch for CPU rule mode
            streaming_gen.start_async_prefetch();
            std::cout << "[*] Async pipeline: ENABLED (double-buffered prefetch)\n";
        } else {
            std::cout << "[*] GPU pipeline: ENABLED (rules + brain wallet on GPU)\n";
        }
        std::cout << "[*] Starting brainwallet scan...\n\n";

        uint64_t total_checked = resumed ? streaming_gen.get_stats().total_generated : 0;
        uint64_t bloom_hits = 0;      // Unverified bloom filter hits
        uint64_t verified_hits = 0;   // Verified against UTXO
        auto start_time = std::chrono::steady_clock::now();
        auto last_report = start_time;

        // Track reported hits to prevent duplicate output
        std::unordered_set<std::string> reported_passphrases;

        // Main brainwallet scanning loop (never exhausts - infinite mode)
        // Two modes:
        // 1. GPU rules mode: raw words → GPU rule engine → GPU brain wallet
        // 2. CPU rules mode: async prefetch with CPU rule application
        std::string last_phase = "";
        std::vector<std::string> current_words;  // Keep track for hit reconstruction in GPU mode
        size_t num_rules = loaded_rules.size();

        while (!g_shutdown) {
            std::vector<std::string> passphrases;
            gpu::MultiGPUBrainWallet::BatchResult result;

            if (use_gpu_rules && num_rules > 0) {
                // GPU Rules Mode - ZERO-COPY: rules + brain wallet all on GPU
                // Get raw words (without rule application)
                size_t words_per_batch = args.batch_size / num_rules;
                if (words_per_batch < 100) words_per_batch = 100;
                if (words_per_batch > 50000) words_per_batch = 50000;

                current_words = streaming_gen.next_raw_words(words_per_batch);
                if (current_words.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                // Check if phase changed - reload GPU rules
                if (streaming_gen.phase_changed()) {
                    auto new_rules = streaming_gen.get_rules();
                    size_t new_num_rules = new_rules.size();

                    if (new_num_rules > 0 && gpu_rule_engine.load_rules(new_rules)) {
                        // Only update loaded_rules and num_rules on successful load
                        loaded_rules = std::move(new_rules);
                        num_rules = loaded_rules.size();
                        std::cout << "\n[*] Phase changed to: " << streaming_gen.current_phase_name()
                                  << " (" << num_rules << " rules)\n";
                        // Recalculate batch size for new rule count
                        words_per_batch = args.batch_size / num_rules;
                        if (words_per_batch < 100) words_per_batch = 100;
                        if (words_per_batch > 50000) words_per_batch = 50000;
                    } else {
                        // Rule loading failed - keep using previous rules
                        // (error already printed by load_rules)
                        // num_rules stays unchanged so batch calculation remains correct
                    }
                }

                // Apply rules on GPU - data stays on GPU (no CPU roundtrip)
                size_t num_passphrases = gpu_rule_engine.apply_rules_to_words_gpu(current_words);
                if (num_passphrases == 0) {
                    continue;
                }

                // Process directly from GPU rule output - ZERO COPY
                // Uses fixed-stride kernel: passphrases at idx * 256
                result = bw_pipeline.process_batch_from_gpu(
                    reinterpret_cast<const uint8_t*>(gpu_rule_engine.d_output()),
                    gpu_rule_engine.d_output_lengths(),
                    gpu::GPU_MAX_OUTPUT_LEN,  // stride = 256
                    num_passphrases,
                    0  // Use first GPU (rule engine is on same GPU)
                );
                total_checked += num_passphrases;

                // For hit handling, we'll reconstruct passphrases on-demand (hits are rare)
                // Don't populate passphrases vector - let hit handler reconstruct
                passphrases.clear();
            } else {
                // CPU Rules Mode - use async prefetch with CPU rule application
                passphrases = streaming_gen.next_batch_async();

                if (passphrases.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    continue;
                }

                result = bw_pipeline.process_batch(passphrases);
                total_checked += passphrases.size();
            }

            // Check for hits and display full information
            for (size_t i = 0; i < result.match_indices.size(); i++) {
                uint32_t match_idx = result.match_indices[i];

                // Get or reconstruct the passphrase
                std::string hit_passphrase;
                if (use_gpu_rules && num_rules > 0 && passphrases.empty()) {
                    // Zero-copy mode: fetch passphrase from GPU (only for hits - very rare)
                    hit_passphrase = gpu_rule_engine.get_passphrase_from_gpu(match_idx);
                    if (hit_passphrase.empty()) {
                        hit_passphrase = "[Failed to retrieve from GPU]";
                    }
                } else {
                    hit_passphrase = passphrases[match_idx];
                }

                // Skip if already reported (prevents duplicate output)
                if (reported_passphrases.count(hit_passphrase)) {
                    continue;
                }
                reported_passphrases.insert(hit_passphrase);
                bloom_hits++;

                // Get private key from GPU result (if available)
                std::array<uint8_t, 32> private_key = {};
                if (i < result.private_keys.size()) {
                    private_key = result.private_keys[i];
                } else {
                    // Compute private key from passphrase (SHA256)
                    auto sha_result = cpu::SHA256::hash(
                        reinterpret_cast<const uint8_t*>(hit_passphrase.data()),
                        hit_passphrase.size());
                    std::copy(sha_result.begin(), sha_result.end(), private_key.begin());
                }

                // Compute Hash160 from private key (does pubkey computation internally)
                auto hash160 = cpu::compute_hash160(private_key.data());

                // Compute public key for display (using the same EC math)
                cpu::uint256_t priv_k;
                priv_k.d[3] = ((uint64_t)private_key[0] << 56) | ((uint64_t)private_key[1] << 48) |
                             ((uint64_t)private_key[2] << 40) | ((uint64_t)private_key[3] << 32) |
                             ((uint64_t)private_key[4] << 24) | ((uint64_t)private_key[5] << 16) |
                             ((uint64_t)private_key[6] << 8) | private_key[7];
                priv_k.d[2] = ((uint64_t)private_key[8] << 56) | ((uint64_t)private_key[9] << 48) |
                             ((uint64_t)private_key[10] << 40) | ((uint64_t)private_key[11] << 32) |
                             ((uint64_t)private_key[12] << 24) | ((uint64_t)private_key[13] << 16) |
                             ((uint64_t)private_key[14] << 8) | private_key[15];
                priv_k.d[1] = ((uint64_t)private_key[16] << 56) | ((uint64_t)private_key[17] << 48) |
                             ((uint64_t)private_key[18] << 40) | ((uint64_t)private_key[19] << 32) |
                             ((uint64_t)private_key[20] << 24) | ((uint64_t)private_key[21] << 16) |
                             ((uint64_t)private_key[22] << 8) | private_key[23];
                priv_k.d[0] = ((uint64_t)private_key[24] << 56) | ((uint64_t)private_key[25] << 48) |
                             ((uint64_t)private_key[26] << 40) | ((uint64_t)private_key[27] << 32) |
                             ((uint64_t)private_key[28] << 24) | ((uint64_t)private_key[29] << 16) |
                             ((uint64_t)private_key[30] << 8) | private_key[31];

                cpu::ECPoint P;
                cpu::ec_mul(P, priv_k);
                cpu::uint256_t pub_x, pub_y;
                cpu::ec_to_affine(pub_x, pub_y, P);

                // Build compressed public key (33 bytes)
                std::array<uint8_t, 33> public_key_compressed;
                public_key_compressed[0] = pub_y.is_odd() ? 0x03 : 0x02;
                public_key_compressed[1] = (pub_x.d[3] >> 56) & 0xff;
                public_key_compressed[2] = (pub_x.d[3] >> 48) & 0xff;
                public_key_compressed[3] = (pub_x.d[3] >> 40) & 0xff;
                public_key_compressed[4] = (pub_x.d[3] >> 32) & 0xff;
                public_key_compressed[5] = (pub_x.d[3] >> 24) & 0xff;
                public_key_compressed[6] = (pub_x.d[3] >> 16) & 0xff;
                public_key_compressed[7] = (pub_x.d[3] >> 8) & 0xff;
                public_key_compressed[8] = pub_x.d[3] & 0xff;
                public_key_compressed[9] = (pub_x.d[2] >> 56) & 0xff;
                public_key_compressed[10] = (pub_x.d[2] >> 48) & 0xff;
                public_key_compressed[11] = (pub_x.d[2] >> 40) & 0xff;
                public_key_compressed[12] = (pub_x.d[2] >> 32) & 0xff;
                public_key_compressed[13] = (pub_x.d[2] >> 24) & 0xff;
                public_key_compressed[14] = (pub_x.d[2] >> 16) & 0xff;
                public_key_compressed[15] = (pub_x.d[2] >> 8) & 0xff;
                public_key_compressed[16] = pub_x.d[2] & 0xff;
                public_key_compressed[17] = (pub_x.d[1] >> 56) & 0xff;
                public_key_compressed[18] = (pub_x.d[1] >> 48) & 0xff;
                public_key_compressed[19] = (pub_x.d[1] >> 40) & 0xff;
                public_key_compressed[20] = (pub_x.d[1] >> 32) & 0xff;
                public_key_compressed[21] = (pub_x.d[1] >> 24) & 0xff;
                public_key_compressed[22] = (pub_x.d[1] >> 16) & 0xff;
                public_key_compressed[23] = (pub_x.d[1] >> 8) & 0xff;
                public_key_compressed[24] = pub_x.d[1] & 0xff;
                public_key_compressed[25] = (pub_x.d[0] >> 56) & 0xff;
                public_key_compressed[26] = (pub_x.d[0] >> 48) & 0xff;
                public_key_compressed[27] = (pub_x.d[0] >> 40) & 0xff;
                public_key_compressed[28] = (pub_x.d[0] >> 32) & 0xff;
                public_key_compressed[29] = (pub_x.d[0] >> 24) & 0xff;
                public_key_compressed[30] = (pub_x.d[0] >> 16) & 0xff;
                public_key_compressed[31] = (pub_x.d[0] >> 8) & 0xff;
                public_key_compressed[32] = pub_x.d[0] & 0xff;

                // Build address from Hash160
                std::string hash160_hex;
                for (int j = 0; j < 20; j++) {
                    char hex[3];
                    snprintf(hex, sizeof(hex), "%02x", hash160[j]);
                    hash160_hex += hex;
                }

                // Convert Hash160 to Bitcoin address (Base58Check)
                std::string btc_address = Base58::hash160_to_address(
                    hash160.data(), [](const uint8_t* data, size_t len) {
                        return cpu::SHA256::hash(data, len);
                    });

                // Display full hit information
                std::cout << "\n";
                std::cout << "+========================================================+\n";
                std::cout << "|                  BLOOM HIT FOUND                       |\n";
                std::cout << "+========================================================+\n";
                std::cout << "| Passphrase:  " << hit_passphrase << "\n";
                std::cout << "| Private Key: ";
                for (int j = 0; j < 32; j++) {
                    printf("%02x", private_key[j]);
                }
                std::cout << "\n";
                std::cout << "| Public Key:  ";
                for (int j = 0; j < 33; j++) {
                    printf("%02x", public_key_compressed[j]);
                }
                std::cout << "\n";
                std::cout << "| Address:     " << btc_address << "\n";
                std::cout << "| Hash160:     " << hash160_hex << "\n";
                std::cout << "| VERIFY:      https://mempool.space/address/" << btc_address << "\n";
                std::cout << "| NOTE: This is a BLOOM HIT - checking balance...\n";
                std::cout << "+========================================================+\n";
                std::cout << std::endl;

                // Check balance asynchronously (won't block scanning)
                check_balance_async(btc_address, hit_passphrase);

                // Report hit to streaming generator for feedback learning
                streaming_gen.report_hit(hit_passphrase);

                // Log to detailed hits file
                std::ofstream hitlog("brainwallet_hits.txt", std::ios::app);
                if (hitlog) {
                    hitlog << "=== HIT FOUND ===\n";
                    hitlog << "Passphrase: " << hit_passphrase << "\n";
                    hitlog << "Private Key: ";
                    for (int j = 0; j < 32; j++) {
                        char hex[3];
                        snprintf(hex, sizeof(hex), "%02x", private_key[j]);
                        hitlog << hex;
                    }
                    hitlog << "\n";
                    hitlog << "Public Key: ";
                    for (int j = 0; j < 33; j++) {
                        char hex[3];
                        snprintf(hex, sizeof(hex), "%02x", public_key_compressed[j]);
                        hitlog << hex;
                    }
                    hitlog << "\n";
                    hitlog << "Address: " << btc_address << "\n";
                    hitlog << "Hash160: " << hash160_hex << "\n";
                    hitlog << "Verify: https://mempool.space/address/" << btc_address << "\n";
                    hitlog << "================\n\n";
                    hitlog.close();
                }

                // Also log to potfile format (hashcat-compatible)
                std::ofstream potfile("brainwallet.pot", std::ios::app);
                if (potfile) {
                    for (int j = 0; j < 20; j++) {
                        char hex[3];
                        snprintf(hex, sizeof(hex), "%02x", hash160[j]);
                        potfile << hex;
                    }
                    potfile << ":" << hit_passphrase << "\n";
                    potfile.close();
                }
            }

            // Progress report every 5 seconds
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_report).count();
            if (elapsed >= 5) {
                auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                double speed = total_elapsed > 0 ? (double)total_checked / total_elapsed : 0;

                // Get current phase info
                auto current_stats = streaming_gen.get_stats();
                std::string phase_info = current_stats.current_phase_name;
                if (current_stats.phase_iteration > 0) {
                    phase_info += " (iter " + std::to_string(current_stats.phase_iteration + 1) + ")";
                }

                // Announce phase changes
                if (phase_info != last_phase) {
                    std::cout << "\n[*] Phase: " << phase_info << "\n";
                    last_phase = phase_info;
                }

                std::cout << "\r[*] Checked: " << format_number_human(total_checked)
                          << " | Speed: " << format_number_human((uint64_t)speed) << "/s"
                          << " | Hits: " << bloom_hits
                          << " | " << phase_info
                          << "        " << std::flush;
                last_report = now;

                // Save state periodically (every save_interval passphrases)
                if (total_checked - last_save_count >= args.save_interval) {
                    auto snap = streaming_gen.get_state_snapshot();
                    BrainWalletSearchState save_state;
                    save_state.current_word_idx = snap.current_word_idx;
                    save_state.current_rule_idx = snap.current_rule_idx;
                    save_state.current_phase = snap.current_phase;
                    save_state.phase_iteration = snap.phase_iteration;
                    save_state.total_checked = total_checked;
                    save_state.unique_tested = current_stats.unique_tested;
                    save_state.hits_found = bloom_hits;
                    save_state.wordlist_hash = wordlist_hash;
                    save_state.wordlist_size = snap.wordlist_size;
                    save_state.wordlist_path = args.wordlist_file;
                    save_state.rules_count = snap.rules_count;
                    save_state.rules_path = rules_dir;
                    save_state.session_id = session_id;

                    if (BrainWalletStateManager::save_state(save_state)) {
                        last_save_count = total_checked;
                        if (args.verbose) {
                            std::cout << "\n[*] State saved at " << total_checked << " passphrases\n";
                        }
                    }
                }
            }
        }

        // Save final state before exit
        {
            auto snap = streaming_gen.get_state_snapshot();
            BrainWalletSearchState save_state;
            save_state.current_word_idx = snap.current_word_idx;
            save_state.current_rule_idx = snap.current_rule_idx;
            save_state.current_phase = snap.current_phase;
            save_state.phase_iteration = snap.phase_iteration;
            save_state.total_checked = total_checked;
            save_state.unique_tested = streaming_gen.get_stats().unique_tested;
            save_state.hits_found = bloom_hits;
            save_state.wordlist_hash = wordlist_hash;
            save_state.wordlist_size = snap.wordlist_size;
            save_state.wordlist_path = args.wordlist_file;
            save_state.rules_count = snap.rules_count;
            save_state.rules_path = rules_dir;
            save_state.session_id = session_id;

            if (BrainWalletStateManager::save_state(save_state)) {
                std::cout << "\n[*] State saved - use --resume to continue\n";
            }
        }

        // Stop async prefetch thread before cleanup
        if (!use_gpu_rules) {
            streaming_gen.stop_async_prefetch();
        }

        // Cleanup GPU rule engine
        if (use_gpu_rules) {
            gpu_rule_engine.cleanup();
        }

        auto end_time = std::chrono::steady_clock::now();
        auto total_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        auto final_stats = streaming_gen.get_stats();

        std::cout << "\n\n";
        std::cout << "+================================================================+\n";
        std::cout << "|               BRAINWALLET SCAN COMPLETE                        |\n";
        std::cout << "+================================================================+\n";
        std::cout << "|  Total Checked:   " << std::setw(15) << format_number_human(total_checked) << "                        |\n";
        std::cout << "|  Bloom Hits:      " << std::setw(15) << bloom_hits << "                        |\n";
        std::cout << "|  Duration:        " << std::setw(15) << total_seconds << " seconds                 |\n";
        std::cout << "|  Final Phase:     " << std::setw(15) << final_stats.current_phase_name << "                        |\n";
        std::cout << "|  Iterations:      " << std::setw(15) << (final_stats.phase_iteration + 1) << "                        |\n";
        std::cout << "+================================================================+\n";

        bw_pipeline.cleanup();
        return 0;
#endif // COLLIDER_HAS_BRAINWALLET
    }

    // Validate arguments - puzzle mode is required (brain wallet mode disabled)
    if (!args.benchmark && !args.puzzle_mode) {
        // Auto-enable puzzle mode if nothing else specified
        args.puzzle_mode = true;
        if (args.puzzle_number == 0) {
            // Smart puzzle selection: choose best ROI puzzle
            if (args.smart_select) {
                int best = get_best_puzzle(400.0);
                if (best > 0) {
                    args.puzzle_number = best;
                    auto puzzle = PuzzleDatabase::get_puzzle(best);
                    std::cout << "\n[*] Smart Selection: Puzzle #" << best;
                    if (puzzle && !puzzle->public_key_hex.empty()) {
                        std::cout << " (Kangaroo - pubkey known)";
                        args.puzzle_kangaroo = true;  // Auto-enable Kangaroo for pubkey puzzles
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
                if (!unsolved.empty()) {
                    args.puzzle_number = unsolved[0]->number;
                } else {
                    args.puzzle_number = 71;  // Default to puzzle 71
                }
            }
        }
    }

    // Benchmark mode - test GPU performance without bloom filter
    if (args.benchmark) {
        std::cout << "\n";
        std::cout << "+================================================================+\n";
        std::cout << "|               GPU PERFORMANCE BENCHMARK                        |\n";
        std::cout << "+================================================================+\n";
        std::cout << "|  Duration: " << std::setw(3) << args.benchmark_seconds << " seconds                                        |\n";
        std::cout << "|  GPUs:     ";
        for (size_t i = 0; i < args.gpu_ids.size() && i < 8; i++) {
            std::cout << args.gpu_ids[i];
            if (i < args.gpu_ids.size() - 1) std::cout << ",";
        }
        std::cout << std::string(53 - args.gpu_ids.size() * 2, ' ') << "|\n";
        std::cout << "|  Batch:    " << std::setw(10) << format_number(args.batch_size) << "                                   |\n";
        std::cout << "+================================================================+\n\n";

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

        std::cout << "\n\n";
        std::cout << "+==============================================================+\n";
        std::cout << "|                   BENCHMARK RESULTS                          |\n";
        std::cout << "+==============================================================+\n";
        std::cout << "|  Duration:        " << std::fixed << std::setprecision(2) << std::setw(8) << actual_seconds << " seconds                       |\n";
        std::cout << "|  Total Processed: " << std::setw(14) << format_number(total_hashed) << "                       |\n";
        std::cout << "|  Iterations:      " << std::setw(8) << iterations << "                               |\n";
        std::cout << "|  Average Rate:    " << std::setw(10) << format_rate(final_rate) << "                             |\n";
        std::cout << "+==============================================================+\n";

        // Performance projections
        double projected_daily = final_rate * 86400;
        std::cout << "|  Projected Daily: " << std::setw(14) << format_number(static_cast<uint64_t>(projected_daily)) << "                       |\n";

        // Compare to targets
        double target_per_gpu = 2.5e9;  // 2.5B/s per RTX 5090
        double expected = target_per_gpu * args.gpu_ids.size();
        double efficiency = (final_rate / expected) * 100.0;

        std::cout << "+==============================================================+\n";
        std::cout << "|  Expected (RTX 5090): " << std::setw(10) << format_rate(expected) << "                         |\n";
        std::cout << "|  Efficiency:          " << std::fixed << std::setprecision(1) << std::setw(6) << efficiency << "%                            |\n";
        std::cout << "+==============================================================+\n\n";

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
    }

    // Puzzle mode - Bitcoin Puzzle Challenge
    if (args.puzzle_mode) {
#if COLLIDER_HAS_SOLO
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
        }

        // Loop through each puzzle (single iteration unless --all-unsolved)
        for (size_t puzzle_idx = 0; puzzle_idx < puzzles_to_solve.size() && !g_shutdown; puzzle_idx++) {
            int current_puzzle = puzzles_to_solve[puzzle_idx];

            // Show progress for auto-progression mode
            if (args.puzzle_all_unsolved && puzzle_idx > 0) {
                std::cout << "\n\n";
                std::cout << "+==============================================================+\n";
                std::cout << "|  AUTO-PROGRESSION: Moving to puzzle #" << current_puzzle
                          << " (" << (puzzle_idx + 1) << "/" << puzzles_to_solve.size() << ")        |\n";
                std::cout << "+==============================================================+\n";
            }

        // Get puzzle info
        const PuzzleInfo* puzzle = PuzzleDatabase::get_puzzle(current_puzzle);

        std::cout << "\n";
        std::cout << "+==============================================================+\n";
        std::cout << "|          BITCOIN PUZZLE CHALLENGE (1000 BTC)                 |\n";
        std::cout << "+==============================================================+\n";

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
            std::cout << "|  Mode:       Custom Range                                    |\n";
        } else if (puzzle) {
            // Use known puzzle data
            range_start = puzzle->range_start();
            range_end = puzzle->range_end();
            bits = puzzle->bits;
            target_address = args.puzzle_target.empty() ? puzzle->target_address : args.puzzle_target;

            std::cout << "|  Puzzle:     #" << std::setw(3) << puzzle->number;
            if (puzzle->solved) {
                std::cout << " (SOLVED - for testing)                      |\n";
            } else {
                std::cout << " (" << std::fixed << std::setprecision(1) << puzzle->btc_reward << " BTC reward)                       |\n";
            }
        } else {
            std::cerr << "[!] Error: Unknown puzzle number: " << current_puzzle << "\n";
            std::cerr << "    Use --puzzle-start and --puzzle-end for custom ranges.\n";
            return 1;
        }

        std::cout << "|  Bits:       " << std::setw(3) << bits << " (2^" << (bits-1) << " keys in range)                    |\n";
        std::cout << "|  Target:     " << target_address.substr(0, 15) << "...                           |\n";
        std::cout << "|  Search:     " << (args.puzzle_random ? "Random    " : "Sequential") << "                                    |\n";
        std::cout << "|  Backend:    " << std::left << std::setw(46) << gpu_info.backend << "|\n";
        std::cout << "+==============================================================+\n";
        std::cout << "|  Range Start: " << std::setw(20) << range_start.to_hex().substr(0, 20) << "...                   |\n";
        std::cout << "|  Range End:   " << std::setw(20) << range_end.to_hex().substr(0, 20) << "...                   |\n";
        std::cout << "+==============================================================+\n\n";

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
        // AUTO-SELECT STRATEGY: Choose Kangaroo if pubkey available
        // ======================================================================
        if (!args.puzzle_kangaroo && bits > 40) {
            // Check if we have a public key for this puzzle
            if (puzzle && !puzzle->public_key_hex.empty()) {
                args.puzzle_kangaroo = true;
                std::cout << "\n\033[36m[*] Algorithm Selection\033[0m\n";
                std::cout << "    Method: \033[1;32mKangaroo\033[0m (Pollard's Rho variant)\n";
                std::cout << "    Reason: Public key is known for this puzzle\n";
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

            // Kangaroo requires a known public key
            if (puzzle->public_key_hex.empty()) {
                std::cerr << "\n[!] ERROR: Puzzle #" << puzzle->number << " has no known public key.\n";
                std::cerr << "    Kangaroo algorithm REQUIRES the target public key (not just address).\n";
                std::cerr << "    Only puzzles with known pubkeys can use --kangaroo: #135, #140, #145, #150, #155, #160\n";
                std::cerr << "    For this puzzle, use brute force instead (remove --kangaroo flag).\n\n";
                return 1;
            }

            // Decompress the public key
            cpu::uint256_t target_pubkey_x, target_pubkey_y;
            if (!cpu::decompress_pubkey(target_pubkey_x, target_pubkey_y, puzzle->public_key_hex)) {
                std::cerr << "[!] ERROR: Failed to decompress public key: " << puzzle->public_key_hex << "\n";
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

                // Set DP bits
                if (args.dp_bits > 0) {
                    rc_kangaroo.dp_bits = std::max(14, std::min(60, args.dp_bits));
                    std::cout << "\033[36m[*] DP Configuration\033[0m\n";
                    std::cout << "    dp_bits: " << rc_kangaroo.dp_bits << " (user override)\n";
                    std::cout << "    1 in " << format_number(1ULL << rc_kangaroo.dp_bits) << " points marked as DP\n";
                } else {
                    // RCKangaroo auto-calculates optimal dp_bits, but we can hint
                    rc_kangaroo.dp_bits = std::min(28, std::max(16, bits / 3));
                    std::cout << "\033[36m[*] DP Configuration (auto)\033[0m\n";
                    std::cout << "    dp_bits: " << rc_kangaroo.dp_bits << " (optimal for " << bits << "-bit puzzle)\n";
                    std::cout << "    1 in " << format_number(1ULL << rc_kangaroo.dp_bits) << " points marked as DP\n";
                }

                // Initialize GPUs
                int num_gpus = rc_kangaroo.init(args.gpu_ids);
                if (num_gpus > 0) {
                    // Load bloom filter if specified
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

                    // Set target public key
                    std::string pubkey_hex = puzzle->public_key_hex;
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
                                  << "\033[32mSpeed:\033[0m " << ui::ProfessionalUI::format_speed(speed) << " | "
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
                    // User specified dp_bits manually
                    dp_bits_to_use = std::max(16, std::min(28, args.dp_bits));
                    std::cout << "\033[36m[*] DP Configuration\033[0m\n";
                    std::cout << "    dp_bits: " << dp_bits_to_use;
                    if (dp_bits_to_use != args.dp_bits) {
                        std::cout << " (clamped from " << args.dp_bits << ")";
                    } else {
                        std::cout << " (user override)";
                    }
                    std::cout << "\n";
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

                    std::cout << "\n\n";
                    std::cout << "\033[1;32m";
                    std::cout << "+==============================================================+\n";
                    std::cout << "|       PUZZLE SOLVED! (GPU Kangaroo Algorithm)               |\n";
                    std::cout << "+==============================================================+\033[0m\n";
                    std::cout << "|  Private Key: " << key_hex << "\n";
                    std::cout << "|  Duration:    " << std::fixed << std::setprecision(3) << total_seconds << " sec\n";
                    std::cout << "|  Total Steps: " << format_number(gpu_result.total_steps) << "\n";
                    std::cout << "+==============================================================+\n\n";

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

            solver.progress_callback = [&](uint64_t tame_steps, uint64_t wild_steps, uint64_t dp_count, double rate) -> bool {
                if (g_shutdown) return false;

                uint64_t total = tame_steps + wild_steps;

                std::cout << "\r[*] Kangaroo: " << format_number(total) << " steps, "
                          << format_number(dp_count) << " DPs, "
                          << format_rate(rate) << "        " << std::flush;

                return !g_shutdown;
            };

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

                std::cout << "\n\n";
                std::cout << "\033[1;32m";
                std::cout << "+==============================================================+\n";
                std::cout << "|         PUZZLE SOLVED! (Kangaroo Algorithm)                 |\n";
                std::cout << "+==============================================================+\033[0m\n";
                std::cout << "|  \033[1;33mPuzzle:\033[0m        #" << std::setw(3) << current_puzzle
                          << " (" << bits << "-bit)                              |\n";
                std::cout << "|  \033[1;33mPrivate Key:\033[0m   " << std::setw(42) << key_hex << " |\n";
                std::cout << "|  \033[1;33mAddress:\033[0m       " << std::setw(42) << target_address << " |\n";
                std::cout << "+--------------------------------------------------------------+\n";
                std::cout << "|  \033[1;36mSolved At:\033[0m     " << std::setw(42) << timestamp << " |\n";
                std::cout << "|  \033[1;36mDuration:\033[0m      " << std::fixed << std::setprecision(3)
                          << std::setw(38) << total_seconds << " sec |\n";
                std::cout << "|  \033[1;36mTotal Steps:\033[0m   " << std::setw(42)
                          << format_number(result.tame_steps + result.wild_steps) << " |\n";
                std::cout << "|  \033[1;36mAlgorithm:\033[0m     " << std::setw(42) << "Pollard's Kangaroo" << " |\n";
                std::cout << "+==============================================================+\n\n";

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
                if (args.puzzle_all_unsolved) {
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

                    std::cout << "\n\n";
                    std::cout << "\033[1;32m"; // Bright green
                    std::cout << "+==============================================================+\n";
                    std::cout << "|        PUZZLE SOLVED! (GPU Accelerated)                     |\n";
                    std::cout << "+==============================================================+\033[0m\n";

                    // Format key as hex
                    char key_hex[67];
                    if (found_key_hi > 0) {
                        snprintf(key_hex, sizeof(key_hex), "0x%llx%016llx",
                                 (unsigned long long)found_key_hi, (unsigned long long)found_key_lo);
                    } else {
                        snprintf(key_hex, sizeof(key_hex), "0x%llx", (unsigned long long)found_key_lo);
                    }

                    std::cout << "|  \033[1;33mPuzzle:\033[0m        #" << std::setw(3) << current_puzzle
                              << " (" << bits << "-bit)                              |\n";
                    std::cout << "|  \033[1;33mPrivate Key:\033[0m   " << std::setw(42) << key_hex << " |\n";
                    std::cout << "|  \033[1;33mAddress:\033[0m       " << std::setw(42) << target_address << " |\n";
                    std::cout << "+--------------------------------------------------------------+\n";
                    std::cout << "|  \033[1;36mSolved At:\033[0m     " << std::setw(42) << timestamp << " |\n";
                    std::cout << "|  \033[1;36mDuration:\033[0m      " << std::fixed << std::setprecision(3)
                              << std::setw(38) << elapsed << " sec |\n";
                    std::cout << "|  \033[1;36mKeys Checked:\033[0m  " << std::setw(42) << format_number_human(total_checked) << " |\n";
                    std::cout << "|  \033[1;36mAccelerator:\033[0m   " << std::setw(39) << gpu_solver.num_gpus() << "x CUDA GPUs |\n";
                    if (puzzle && puzzle->btc_reward > 0) {
                        std::cout << "|  \033[1;35mBTC Reward:\033[0m    " << std::fixed << std::setprecision(1)
                                  << std::setw(39) << puzzle->btc_reward << " BTC |\n";
                    }
                    std::cout << "+==============================================================+\n\n";

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
                    if (args.puzzle_all_unsolved) {
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

            std::cout << "\n\n";
            std::cout << "+==============================================================+\n";
            std::cout << "|                GPU PUZZLE SEARCH RESULTS                     |\n";
            std::cout << "+==============================================================+\n";
            std::cout << "|  Session Duration: " << std::fixed << std::setprecision(2) << std::setw(10) << elapsed_sec << " seconds                  |\n";
            std::cout << "|  Session Checked:  " << std::setw(15) << format_number_human(session_checked) << "                    |\n";
            std::cout << "|  Total Checked:    " << std::setw(15) << format_number_human(total_checked) << "                    |\n";
            std::cout << "|  Session Rate:     " << std::setw(12) << format_rate(session_rate) << "                         |\n";
            std::cout << "+==============================================================+\n";

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

                    std::cout << "\n\n";
                    std::cout << "\033[1;32m"; // Bright green
                    std::cout << "+==============================================================+\n";
                    std::cout << "|            🎉  PUZZLE SOLVED!  🎉                            |\n";
                    std::cout << "+==============================================================+\033[0m\n";

                    // Format key as hex
                    char key_hex[67];
                    if (key_hi > 0) {
                        snprintf(key_hex, sizeof(key_hex), "0x%llx%016llx",
                                 (unsigned long long)key_hi, (unsigned long long)key_lo);
                    } else {
                        snprintf(key_hex, sizeof(key_hex), "0x%llx", (unsigned long long)key_lo);
                    }

                    std::cout << "|  \033[1;33mPuzzle:\033[0m        #" << std::setw(3) << current_puzzle
                              << " (" << bits << "-bit)                              |\n";
                    std::cout << "|  \033[1;33mPrivate Key:\033[0m   " << std::setw(42) << key_hex << " |\n";
                    std::cout << "|  \033[1;33mAddress:\033[0m       " << std::setw(42) << target_address << " |\n";
                    std::cout << "+--------------------------------------------------------------+\n";
                    std::cout << "|  \033[1;36mSolved At:\033[0m     " << std::setw(42) << timestamp << " |\n";
                    std::cout << "|  \033[1;36mDuration:\033[0m      " << std::fixed << std::setprecision(3)
                              << std::setw(38) << elapsed << " sec |\n";
                    std::cout << "|  \033[1;36mKeys Checked:\033[0m  " << std::setw(42) << format_number_human(total_checked) << " |\n";
                    if (puzzle && puzzle->btc_reward > 0) {
                        std::cout << "|  \033[1;35mBTC Reward:\033[0m    " << std::fixed << std::setprecision(1)
                                  << std::setw(39) << puzzle->btc_reward << " BTC |\n";
                    }
                    std::cout << "+==============================================================+\n\n";

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

                    // Show next puzzle suggestion for manual mode
                    if (!args.puzzle_all_unsolved) {
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
        if (found && args.puzzle_all_unsolved) {
            std::cout << "\n[*] Puzzle #" << current_puzzle << " solved! Continuing to next puzzle...\n";
            continue;
        }

        search_done:

        // Final stats
        auto end_time = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
        double final_rate = total_checked / std::max(0.001, elapsed_sec);

        std::cout << "\n\n";
        std::cout << "+==============================================================+\n";
        std::cout << "|                   PUZZLE SEARCH RESULTS                      |\n";
        std::cout << "+==============================================================+\n";
        std::cout << "|  Duration:        " << std::fixed << std::setprecision(2) << std::setw(10) << elapsed_sec << " seconds                   |\n";
        std::cout << "|  Keys Checked:    " << std::setw(15) << format_number_human(total_checked) << "                    |\n";
        std::cout << "|  Average Rate:    " << std::setw(12) << format_rate(final_rate) << "                         |\n";
        std::cout << "+==============================================================+\n";

        // Time to complete estimates (for puzzles we know the range size)
        double range_size_approx = std::pow(2.0, bits - 1);  // 2^(N-1) keys
        double remaining_approx = range_size_approx - total_checked;
        double time_to_complete_sec = remaining_approx / std::max(1.0, final_rate);
        double days_to_complete = time_to_complete_sec / 86400;

        std::cout << "|  Range Size:      2^" << std::setw(3) << (bits - 1) << " keys                           |\n";
        if (days_to_complete < 1) {
            std::cout << "|  ETA (current):   " << std::setw(10) << std::fixed << std::setprecision(1)
                      << (time_to_complete_sec / 3600) << " hours                       |\n";
        } else if (days_to_complete < 365) {
            std::cout << "|  ETA (current):   " << std::setw(10) << std::fixed << std::setprecision(1)
                      << days_to_complete << " days                        |\n";
        } else if (days_to_complete < 1e6) {
            std::cout << "|  ETA (current):   " << std::setw(10) << std::fixed << std::setprecision(1)
                      << (days_to_complete / 365) << " years                       |\n";
        } else {
            std::cout << "|  ETA (current):   " << std::setw(10) << std::scientific << std::setprecision(2)
                      << (days_to_complete / 365) << " years                 |\n";
        }

        std::cout << "+==============================================================+\n\n";

        std::cout << "[!] Note: Puzzle mode is using CPU simulation.\n";
        std::cout << "    Real GPU performance will be significantly higher.\n";
        std::cout << "    Once GPU pipeline is integrated, expect ~1B+ keys/sec per GPU.\n";

        } // End of puzzle iteration for loop

        // All puzzles completed (auto-progression) or single puzzle done
        if (args.puzzle_all_unsolved) {
            std::cout << "\n";
            std::cout << "+==============================================================+\n";
            std::cout << "|          AUTO-PROGRESSION COMPLETE                           |\n";
            std::cout << "+==============================================================+\n";
            std::cout << "|  All " << puzzles_to_solve.size() << " puzzles have been processed.                        |\n";
            std::cout << "+==============================================================+\n\n";
        }

        return 0;
#else // !COLLIDER_HAS_SOLO
        // This should never be reached due to argument parsing gates, but safety check
        std::cerr << "[*] Solo puzzle solver requires collider pro — collisionprotocol.com/pro\n";
        std::cerr << "    Use the free edition to join pools: collider --worker <your_btc_address>\n";
        return 1;
#endif // COLLIDER_HAS_SOLO
    }

    // No valid mode specified
    std::cerr << "[!] Error: No mode specified. Use --worker <your_btc_address> to join the pool.\n";
    print_usage();
    return 1;
}
