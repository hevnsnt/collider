/**
 * Collider Core Types
 *
 * Common type definitions for the intelligence layer and GPU pipeline.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <optional>
#include <functional>
#include "edition.hpp"

namespace collider {

// -----------------------------------------------------------------------------
// Candidate Types
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// Crypto Types
// -----------------------------------------------------------------------------

/**
 * 256-bit unsigned integer for private keys.
 */
struct uint256_t {
    uint64_t limbs[4];  // Little-endian: limbs[0] is least significant

    bool operator==(const uint256_t& other) const {
        return limbs[0] == other.limbs[0] &&
               limbs[1] == other.limbs[1] &&
               limbs[2] == other.limbs[2] &&
               limbs[3] == other.limbs[3];
    }
};

/**
 * Secp256k1 elliptic curve point (affine coordinates).
 */
struct ECPoint {
    uint256_t x;
    uint256_t y;
    bool is_infinity = false;
};

/**
 * Bitcoin address (20-byte RIPEMD160 hash).
 */
struct BitcoinAddress {
    uint8_t hash160[20];

    bool operator==(const BitcoinAddress& other) const {
        return std::memcmp(hash160, other.hash160, 20) == 0;
    }
};


// -----------------------------------------------------------------------------
// Rule Types
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// Statistics Types
// -----------------------------------------------------------------------------

/**
 * Real-time performance statistics.
 */
struct PerformanceStats {
    uint64_t candidates_tested;
    uint64_t candidates_remaining;
    double keys_per_second;
    double elapsed_seconds;
    uint32_t cracks_found;

    // Per-GPU stats
    struct GPUStats {
        uint32_t device_id;
        double utilization;      // 0.0 - 1.0
        double temperature_c;
        double power_watts;
        double keys_per_second;
    };
    std::vector<GPUStats> gpu_stats;
};


// -----------------------------------------------------------------------------
// Callback Types
// -----------------------------------------------------------------------------


/**
 * Callback for progress updates.
 */
using ProgressCallback = std::function<void(const PerformanceStats&)>;


// -----------------------------------------------------------------------------
// Configuration Types
// -----------------------------------------------------------------------------

/**
 * Collider configuration.
 */
struct Config {
    // GPU settings
    std::vector<uint32_t> gpu_device_ids = {0, 1, 2, 3};  // 4x RTX 5090
    size_t batch_size = 4'000'000;

    // Wordlist settings
    std::vector<std::string> wordlist_paths;
    std::vector<std::string> rule_paths;


    // Output settings
    std::string output_path;
    std::string potfile_path;
    bool verbose = false;

};

}  // namespace collider
