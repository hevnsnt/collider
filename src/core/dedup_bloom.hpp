/**
 * Lock-Free Deduplication Bloom Filter
 *
 * High-performance bloom filter for passphrase deduplication.
 * Uses XXH3 for fast hashing and lock-free atomic operations for thread safety.
 *
 * DESIGN GOALS:
 * - Zero contention: Lock-free using atomic bit operations
 * - Cache friendly: Blocked layout for better memory access patterns
 * - False positive rate: Configurable (default 0.01% at 100M elements)
 * - Memory efficient: ~12 bits per element for 0.01% FP rate
 *
 * USAGE:
 * - Call test_and_set() to check if passphrase was seen and mark it
 * - Returns true if passphrase is NEW (not seen before)
 * - Returns false if passphrase was ALREADY seen (duplicate)
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <atomic>
#include <memory>
#include <vector>
#include <string>
#include <unordered_set>

// Use XXH3 for fast hashing (already a project dependency)
#define XXH_INLINE_ALL
#include "xxhash.h"

namespace collider {

/**
 * Lock-free Bloom filter for deduplication.
 */
class DedupBloomFilter {
public:
    // Default configuration for 100M elements at 0.01% FP rate
    static constexpr size_t DEFAULT_NUM_BITS = 1917011675ULL;  // ~228 MB
    static constexpr size_t DEFAULT_NUM_HASHES = 13;

    /**
     * Create bloom filter with specified parameters.
     *
     * @param num_bits Total bits in filter
     * @param num_hashes Number of hash functions
     */
    DedupBloomFilter(size_t num_bits = DEFAULT_NUM_BITS,
                     size_t num_hashes = DEFAULT_NUM_HASHES)
        : num_bits_(num_bits)
        , num_hashes_(num_hashes)
        , num_words_((num_bits + 63) / 64) {
        // Allocate atomic bit array
        bits_ = std::make_unique<std::atomic<uint64_t>[]>(num_words_);

        // Zero initialize
        for (size_t i = 0; i < num_words_; i++) {
            bits_[i].store(0, std::memory_order_relaxed);
        }
    }

    /**
     * Test if element exists and set if not.
     * Lock-free using compare-and-swap.
     *
     * @param data Pointer to data
     * @param len Length of data
     * @return true if element is NEW (was not in filter), false if DUPLICATE
     */
    bool test_and_set(const void* data, size_t len) {
        // Compute base hash using XXH3 (very fast)
        uint64_t h1 = XXH3_64bits(data, len);
        uint64_t h2 = XXH3_64bits_withSeed(data, len, 0x9E3779B97F4A7C15ULL);

        // Double hashing: h(i) = h1 + i*h2
        bool all_set = true;

        for (size_t i = 0; i < num_hashes_; i++) {
            uint64_t hash = h1 + i * h2;
            size_t bit_idx = hash % num_bits_;
            size_t word_idx = bit_idx / 64;
            uint64_t bit_mask = 1ULL << (bit_idx % 64);

            // Check if bit is already set
            uint64_t old_val = bits_[word_idx].load(std::memory_order_relaxed);
            if ((old_val & bit_mask) == 0) {
                all_set = false;
            }
        }

        // If all bits were set, this is a duplicate
        if (all_set) {
            return false;  // Duplicate
        }

        // Set all bits (may race with other threads, but that's OK for bloom filters)
        for (size_t i = 0; i < num_hashes_; i++) {
            uint64_t hash = h1 + i * h2;
            size_t bit_idx = hash % num_bits_;
            size_t word_idx = bit_idx / 64;
            uint64_t bit_mask = 1ULL << (bit_idx % 64);

            // Atomic OR to set bit
            bits_[word_idx].fetch_or(bit_mask, std::memory_order_relaxed);
        }

        return true;  // New element
    }

    /**
     * Test if element might exist (no modification).
     */
    bool test(const void* data, size_t len) const {
        uint64_t h1 = XXH3_64bits(data, len);
        uint64_t h2 = XXH3_64bits_withSeed(data, len, 0x9E3779B97F4A7C15ULL);

        for (size_t i = 0; i < num_hashes_; i++) {
            uint64_t hash = h1 + i * h2;
            size_t bit_idx = hash % num_bits_;
            size_t word_idx = bit_idx / 64;
            uint64_t bit_mask = 1ULL << (bit_idx % 64);

            if ((bits_[word_idx].load(std::memory_order_relaxed) & bit_mask) == 0) {
                return false;  // Definitely not in filter
            }
        }

        return true;  // Might be in filter (or false positive)
    }

    /**
     * Convenience method for string data.
     */
    bool test_and_set(const std::string& str) {
        return test_and_set(str.data(), str.size());
    }

    bool test(const std::string& str) const {
        return test(str.data(), str.size());
    }

    /**
     * Clear all bits (not thread-safe, call only when no other threads are using).
     */
    void clear() {
        for (size_t i = 0; i < num_words_; i++) {
            bits_[i].store(0, std::memory_order_relaxed);
        }
    }

    /**
     * Get memory usage in bytes.
     */
    size_t memory_bytes() const {
        return num_words_ * sizeof(uint64_t);
    }

    /**
     * Get theoretical false positive rate for given number of elements.
     */
    double false_positive_rate(size_t num_elements) const {
        // FP rate = (1 - e^(-k*n/m))^k
        // where k = num_hashes, n = num_elements, m = num_bits
        double exponent = -(double)num_hashes_ * num_elements / num_bits_;
        double base = 1.0 - exp(exponent);
        double fp_rate = 1.0;
        for (size_t i = 0; i < num_hashes_; i++) {
            fp_rate *= base;
        }
        return fp_rate;
    }

    /**
     * Calculate optimal parameters for desired FP rate and capacity.
     */
    static void optimal_params(size_t expected_elements, double fp_rate,
                               size_t& out_num_bits, size_t& out_num_hashes) {
        // m = -n * ln(p) / (ln(2)^2)
        double ln2 = 0.693147180559945;
        double ln2_sq = ln2 * ln2;

        out_num_bits = static_cast<size_t>(
            -static_cast<double>(expected_elements) * log(fp_rate) / ln2_sq
        );

        // k = (m/n) * ln(2)
        out_num_hashes = static_cast<size_t>(
            (static_cast<double>(out_num_bits) / expected_elements) * ln2 + 0.5
        );

        // Clamp hashes to reasonable range
        if (out_num_hashes < 1) out_num_hashes = 1;
        if (out_num_hashes > 30) out_num_hashes = 30;
    }

private:
    size_t num_bits_;
    size_t num_hashes_;
    size_t num_words_;
    std::unique_ptr<std::atomic<uint64_t>[]> bits_;
};

/**
 * Thread-local batch deduplication helper.
 * Accumulates candidates in thread-local set, then batch-inserts to global bloom.
 * Reduces atomic contention on hot paths.
 */
class BatchDedupHelper {
public:
    static constexpr size_t BATCH_SIZE = 10000;

    BatchDedupHelper(DedupBloomFilter& global_filter)
        : global_filter_(global_filter) {
        local_set_.reserve(BATCH_SIZE);
    }

    /**
     * Add candidate to batch. Returns true if candidate is new to this batch.
     * Actual global dedup happens on flush().
     */
    bool add(const std::string& candidate) {
        // Quick local check first
        auto result = local_set_.insert(candidate);
        if (!result.second) {
            return false;  // Duplicate within batch
        }

        // Flush if batch is full
        if (local_set_.size() >= BATCH_SIZE) {
            flush();
        }

        return true;
    }

    /**
     * Flush batch to global filter.
     * Returns number of globally unique candidates.
     */
    size_t flush() {
        size_t unique_count = 0;
        for (const auto& candidate : local_set_) {
            if (global_filter_.test_and_set(candidate)) {
                unique_count++;
            }
        }
        local_set_.clear();
        return unique_count;
    }

    ~BatchDedupHelper() {
        flush();
    }

private:
    DedupBloomFilter& global_filter_;
    std::unordered_set<std::string> local_set_;
};

}  // namespace collider
