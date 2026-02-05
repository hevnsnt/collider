/**
 * RCKangaroo Wrapper for collider
 *
 * Integrates RetiredCoder's RCKangaroo (GPLv3) as the Kangaroo solver backend.
 * RCKangaroo achieves ~8 GKeys/s on RTX 4090, ~4 GKeys/s on RTX 3090.
 *
 * Optional bloom filter integration: checks each DP against a bloom filter
 * of funded Bitcoin addresses (opportunistic collision detection).
 *
 * Original software: (c) 2024, RetiredCoder (RC)
 * https://github.com/RetiredC/RCKangaroo
 */

#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <vector>
#include <functional>
#include <atomic>

namespace collider {
namespace gpu {

/**
 * Bloom filter hit - a potential match against the address database
 */
struct BloomHit {
    std::array<uint64_t, 4> private_key;  // 256-bit private key
    std::array<uint8_t, 20> hash160;      // RIPEMD160(SHA256(pubkey))
    std::string address;                   // Bitcoin address (if computed)
    uint64_t ops_at_hit;                   // Operations count when found
};

/**
 * Result from RCKangaroo solve
 */
struct RCKangarooResult {
    bool found;
    std::array<uint64_t, 4> private_key;  // 256-bit private key
    uint64_t total_ops;
    uint64_t dp_count;
    double elapsed_seconds;
    double k_value;  // K coefficient (target is 1.15)
    uint32_t error_count;

    // Bloom filter results
    uint64_t bloom_checks;                // Total bloom filter checks performed
    std::vector<BloomHit> bloom_hits;     // Potential matches (verify externally)
};

/**
 * RCKangaroo Solver Manager
 *
 * High-performance GPU Kangaroo solver using RetiredCoder's implementation.
 * Supports ranges up to 170 bits, achieves K=1.15 (optimal).
 */
class RCKangarooManager {
public:
    // Configuration
    int dp_bits = 20;           // Distinguished point bits (14-60)
    int range_bits = 135;       // Search range in bits (32-170)
    bool benchmark_mode = false;
    std::atomic<bool> stop_flag{false};

    // Bloom filter configuration
    bool bloom_enabled = false;         // Enable bloom filter checking
    std::string bloom_file;             // Path to .blf bloom filter file

    // Progress callback: (ops, dp_count, speed_mkeys) -> continue?
    std::function<bool(uint64_t, uint64_t, int)> progress_callback;

    // Bloom hit callback: (hit) -> called when bloom filter match found
    std::function<void(const BloomHit&)> bloom_hit_callback;

    // DP callback for pool mode: (x[32], d[32], type) -> called for each new DP
    // Used to submit DPs to pool server
    std::function<void(const uint8_t*, const uint8_t*, uint8_t)> dp_callback;

    RCKangarooManager();
    ~RCKangarooManager();

    /**
     * Initialize with specific GPU IDs, or empty vector for auto-detect
     * Returns number of GPUs initialized
     */
    int init(const std::vector<int>& gpu_ids = {});

    /**
     * Get number of initialized GPUs
     */
    int num_gpus() const;

    /**
     * Set the target public key to solve
     * @param compressed_hex Compressed public key (33 bytes hex, e.g., "02abc...")
     * @return true if valid public key
     */
    bool set_target_pubkey(const std::string& compressed_hex);

    /**
     * Set the target public key from X,Y coordinates
     */
    bool set_target_pubkey(const std::array<uint64_t, 4>& x, const std::array<uint64_t, 4>& y);

    /**
     * Set the search range offset (starting point)
     * @param start_hex Hexadecimal offset (e.g., "40000000000000000000000000000000000")
     */
    void set_start_offset(const std::string& start_hex);

    /**
     * Load precomputed tame kangaroos from file (speeds up solving)
     * @param filename Path to tames file
     * @return true if loaded successfully
     */
    bool load_tames(const std::string& filename);

    /**
     * Generate and save tame kangaroos to file
     * @param filename Path to save tames
     * @param max_ops Maximum operations multiplier (e.g., 0.5 for half of expected ops)
     * @return true if generated successfully
     */
    bool generate_tames(const std::string& filename, double max_ops = 0.5);

    /**
     * Solve for the private key
     * @return Result containing private key if found
     */
    RCKangarooResult solve();

    /**
     * Run benchmark mode (solve random keys to measure K value)
     * @param num_points Number of random points to solve
     * @return Average K value achieved
     */
    double benchmark(int num_points = 10);

    /**
     * Get current speed in MKeys/s
     */
    int get_speed() const;

    /**
     * Load bloom filter from .blf file for opportunistic address checking
     * @param filename Path to .blf bloom filter file
     * @return true if loaded successfully
     */
    bool load_bloom_filter(const std::string& filename);

    /**
     * Get bloom filter statistics
     * @return Number of bloom filter checks performed so far
     */
    uint64_t get_bloom_checks() const;

private:
    struct Impl;
    Impl* impl_;
};

/**
 * Convert private key array to hex string
 */
std::string private_key_to_hex(const std::array<uint64_t, 4>& key);

/**
 * Convert hex string to private key array
 */
bool hex_to_private_key(const std::string& hex, std::array<uint64_t, 4>& key);

}  // namespace gpu
}  // namespace collider
