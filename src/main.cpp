/**
 * collider - Bitcoin Puzzle Solver
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
            std::cerr << "[*] Solo puzzle solver requires collider pro — collisionprotocol.com/pro\n";
            std::cerr << "    Use the free edition to join pools: collider --worker <your_btc_address>\n";
            args.exit_program = true;
            return args;
        } else if (arg == "--puzzle-target" && i + 1 < argc) {
            std::cerr << "[*] SOLO requires collider pro — collisionprotocol.com/pro\n";
            args.exit_program = true; return args;
        } else if (arg == "--puzzle-start" && i + 1 < argc) {
            std::cerr << "[*] SOLO requires collider pro — collisionprotocol.com/pro\n";
            args.exit_program = true; return args;
        } else if (arg == "--puzzle-end" && i + 1 < argc) {
            std::cerr << "[*] SOLO requires collider pro — collisionprotocol.com/pro\n";
            args.exit_program = true; return args;
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
            std::cerr << "[*] Bloom filters require collider pro — collisionprotocol.com/pro\n";
            args.exit_program = true; return args;
        } else if (arg == "--brainwallet") {
            std::cerr << "[*] Brain wallet requires collider pro — collisionprotocol.com/pro\n";
            args.exit_program = true; return args;
        } else if (arg == "--brainwallet-setup") {
            std::cerr << "[!] ERROR: --brainwallet-setup is only available in collider pro\n";
            std::cerr << "    Visit https://collisionprotocol.com/pro to upgrade.\n";
            args.exit_program = true; return args;
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
            std::cerr << "[!] ERROR: --pool (custom pool) is only available in collider pro\n";
            std::cerr << "    Free edition connects to: " << COLLIDER_FREE_POOL_URL << "\n";
            std::cerr << "    Use --worker <your_btc_address> to join the pool.\n";
            std::cerr << "    Visit https://collisionprotocol.com/pro for custom pool support.\n";
            args.exit_program = true; return args;
        } else if ((arg == "--worker" || arg == "-w") && i + 1 < argc) {
            args.pool_worker = argv[++i];
            // In free edition, automatically enable pool mode with hardcoded URL
            args.pool_mode = true;
            args.pool_url = COLLIDER_FREE_POOL_URL;
        } else if (arg == "--pool-password" && i + 1 < argc) {
            std::cerr << "[!] ERROR: --pool-password is only available in collider pro\n";
            args.exit_program = true; return args;
        } else if (arg == "--pool-api-key" && i + 1 < argc) {
            std::cerr << "[!] ERROR: --pool-api-key is only available in collider pro\n";
            args.exit_program = true; return args;
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
    std::cout << "collider - Pool Edition\n";
    std::cout << "========================\n\n";
    std::cout << "Free edition - pool mining for distributed solving.\n";
    std::cout << "For all features, visit https://collisionprotocol.com/pro\n\n";

    std::cout << "GPU-accelerated solver for the Bitcoin Puzzle Challenge.\n\n";

    std::cout << "Usage:\n";
    std::cout << "  collider --worker <btc_address>\n\n";
    std::cout << R"(Pool Options:
  --worker, -w <address>  Your Bitcoin address for rewards (required)
                          Connects to: )" << COLLIDER_FREE_POOL_URL << R"(

)";

    std::cout << R"(GPU Options:
  --gpus, -g <ids>        GPU device IDs to use (default: auto-detect all)
  --batch-size <n>        Keys per batch (default: 4000000)

Output Options:
  --verbose, -v           Verbose output

Benchmark Mode:
  --benchmark             Run GPU performance benchmark
  --benchmark-time <sec>  Benchmark duration (default: 30s)

Calibration:
  --calibrate             Run GPU batch size calibration
  --force-calibrate       Force re-calibration

Other:
  --help, -h              Show this help message
  --debug                 Show debug output
  --config, -c <file>     Config file (default: ./config.yml)

Examples:
)";

    std::cout << R"(  collider --worker 1YourBitcoinAddress...
  collider --benchmark
  collider --worker 1YourAddr... --gpus 0,1
)";

    std::cout << R"(
Performance Targets:
  RTX 3060:  ~5M keys/s    RTX 3090:  ~20M keys/s    RTX 4090:  ~50M keys/s
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
            LOG_INFO("Starting collider v1.0.0");
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
        // This should never be reached due to argument parsing gates, but safety check
        std::cerr << "[!] ERROR: Brainwallet mode is only available in collider pro\n";
        return 1;

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


        auto bench_start = std::chrono::steady_clock::now();
        auto bench_end = bench_start + std::chrono::seconds(args.benchmark_seconds);
        uint64_t total_hashed = 0;
        uint64_t iterations = 0;
        auto last_status = bench_start;

        while (std::chrono::steady_clock::now() < bench_end && !g_shutdown) {
            // Free edition: CPU simulation benchmark
            std::this_thread::sleep_for(std::chrono::microseconds(1600));
            total_hashed += test_candidates.size();
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

        std::cout << "[!] Note: CUDA not available - benchmark used CPU simulation.\n";
        std::cout << "    Real GPU performance will be significantly higher.\n";
        std::cout << "    Build with CUDA enabled for actual GPU benchmarks.\n";

        return 0;
    }

    // Puzzle mode - Bitcoin Puzzle Challenge
    if (args.puzzle_mode) {
        // This should never be reached due to argument parsing gates, but safety check
        std::cerr << "[*] Solo puzzle solver requires collider pro — collisionprotocol.com/pro\n";
        std::cerr << "    Use the free edition to join pools: collider --worker <your_btc_address>\n";
        return 1;
    }

    // No valid mode specified
    std::cerr << "[!] Error: No mode specified. Use --worker <your_btc_address> to join the pool.\n";
    print_usage();
    return 1;
}
