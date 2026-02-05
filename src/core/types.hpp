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

namespace collider {

// -----------------------------------------------------------------------------
// Candidate Types
// -----------------------------------------------------------------------------

/**
 * Source identifier for tracking where candidates originate.
 */
enum class CandidateSource : uint8_t {
    KNOWN_BRAIN_WALLET = 0,  // Previously compromised brain wallets (highest priority)
    PASSWORD_TOP10K = 1,      // Top passwords by frequency
    PASSWORD_COMMON = 2,      // Common password lists
    LYRICS = 3,               // Song lyrics
    QUOTES = 4,               // Famous quotes
    WIKIPEDIA = 5,            // Wikipedia titles/phrases
    CRYPTO_FORUM = 6,         // BitcoinTalk, Reddit crypto
    PCFG_GENERATED = 7,       // Probabilistically generated
    COMBINATOR = 8,           // Word combinations
    MARKOV = 9,               // Markov chain generated
    USER_WORDLIST = 10,       // User-provided wordlist
    FEEDBACK = 11,            // Generated from cracked passwords
};

/**
 * A candidate passphrase with metadata.
 */
struct Candidate {
    std::string phrase;
    float priority;           // Higher = more likely (0.0 - 1.0)
    CandidateSource source;
    std::string rule_applied; // Hashcat rule that generated this variant

    bool operator<(const Candidate& other) const {
        return priority < other.priority;  // Min-heap by default
    }

    bool operator>(const Candidate& other) const {
        return priority > other.priority;
    }
};

/**
 * Batch of candidates for GPU processing.
 */
struct CandidateBatch {
    static constexpr size_t DEFAULT_BATCH_SIZE = 4'000'000;  // 4M candidates

    std::vector<std::string> phrases;
    std::vector<float> priorities;
    std::vector<CandidateSource> sources;

    size_t size() const { return phrases.size(); }
    bool empty() const { return phrases.empty(); }

    void reserve(size_t n) {
        phrases.reserve(n);
        priorities.reserve(n);
        sources.reserve(n);
    }

    void push_back(Candidate&& c) {
        phrases.push_back(std::move(c.phrase));
        priorities.push_back(c.priority);
        sources.push_back(c.source);
    }

    void clear() {
        phrases.clear();
        priorities.clear();
        sources.clear();
    }
};

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

/**
 * Result of a successful crack.
 */
struct CrackResult {
    std::string passphrase;
    uint256_t private_key;
    BitcoinAddress address;
    CandidateSource source;
    std::string rule_applied;
    uint64_t timestamp;
};

// -----------------------------------------------------------------------------
// Rule Types
// -----------------------------------------------------------------------------

/**
 * A hashcat-compatible rule.
 */
struct Rule {
    std::string definition;   // Raw rule string (e.g., "c$1$2$3")
    float efficiency;         // Cracks per rule application (empirical)
    std::string description;  // Human-readable description
};

/**
 * Collection of rules for a specific purpose.
 */
struct RuleSet {
    std::string name;
    std::vector<Rule> rules;
    float total_efficiency;

    size_t size() const { return rules.size(); }
};

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

/**
 * Source effectiveness tracking.
 */
struct SourceStats {
    CandidateSource source;
    uint64_t candidates_tested;
    uint32_t cracks_found;
    double crack_rate;  // cracks / candidates_tested
};

// -----------------------------------------------------------------------------
// Callback Types
// -----------------------------------------------------------------------------

/**
 * Callback when a crack is found.
 */
using CrackCallback = std::function<void(const CrackResult&)>;

/**
 * Callback for progress updates.
 */
using ProgressCallback = std::function<void(const PerformanceStats&)>;

/**
 * Callback for candidate generation (streaming).
 */
using CandidateCallback = std::function<void(Candidate&&)>;

// -----------------------------------------------------------------------------
// Configuration Types
// -----------------------------------------------------------------------------

/**
 * Collider configuration.
 */
struct Config {
    // GPU settings
    std::vector<uint32_t> gpu_device_ids = {0, 1, 2, 3};  // 4x RTX 5090
    size_t batch_size = CandidateBatch::DEFAULT_BATCH_SIZE;

    // Wordlist settings
    std::vector<std::string> wordlist_paths;
    std::vector<std::string> rule_paths;
    bool use_pcfg = true;
    bool use_combinator = true;

    // Bloom filter settings
    std::string bloom_filter_path;  // Path to address bloom filter
    double bloom_error_rate = 0.0000001;  // 1 in 10 million FP rate

    // Output settings
    std::string output_path;
    std::string potfile_path;
    bool verbose = false;

    // Performance tuning
    size_t priority_queue_size = 100'000'000;  // 100M candidates in queue
    size_t dedup_bloom_size = 1'000'000'000;   // 1B for deduplication
};

}  // namespace collider
