/**
 * GPU Rule Application Engine
 *
 * High-performance GPU implementation of hashcat-style rule processing.
 * Processes millions of word+rule combinations in parallel.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

#ifdef COLLIDER_USE_CUDA
#include <cuda_runtime.h>
#else
typedef int cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
#endif

namespace collider {
namespace gpu {

// Maximum output length after rule application
constexpr int GPU_MAX_OUTPUT_LEN = 256;

// Forward declare CUDA functions
extern "C" {

/**
 * Apply rules to words (cross product: each word x each rule).
 *
 * Generates num_words * num_rules output candidates.
 * Output index = word_idx * num_rules + rule_idx
 *
 * @param d_words          Device: packed word bytes
 * @param d_word_offsets   Device: offset for each word
 * @param d_word_lengths   Device: length of each word
 * @param num_words        Number of words
 * @param d_rules          Device: packed rule bytes
 * @param d_rule_offsets   Device: offset for each rule
 * @param d_rule_lengths   Device: length of each rule
 * @param num_rules        Number of rules
 * @param d_output         Device: output buffer (num_words * num_rules * GPU_MAX_OUTPUT_LEN)
 * @param d_output_lengths Device: length of each output (num_words * num_rules)
 * @param stream           CUDA stream
 */
cudaError_t gpu_apply_rules_cross_product(
    const char* d_words,
    const uint32_t* d_word_offsets,
    const uint32_t* d_word_lengths,
    uint32_t num_words,
    const char* d_rules,
    const uint32_t* d_rule_offsets,
    const uint32_t* d_rule_lengths,
    uint32_t num_rules,
    char* d_output,
    uint32_t* d_output_lengths,
    cudaStream_t stream
);

/**
 * Apply single rule to all words.
 *
 * @param d_words          Device: packed word bytes
 * @param d_word_offsets   Device: offset for each word
 * @param d_word_lengths   Device: length of each word
 * @param d_rule           Device: single rule string
 * @param rule_len         Length of rule
 * @param d_output         Device: output buffer (count * GPU_MAX_OUTPUT_LEN)
 * @param d_output_lengths Device: length of each output
 * @param count            Number of words
 * @param stream           CUDA stream
 */
cudaError_t gpu_apply_single_rule(
    const char* d_words,
    const uint32_t* d_word_offsets,
    const uint32_t* d_word_lengths,
    const char* d_rule,
    int rule_len,
    char* d_output,
    uint32_t* d_output_lengths,
    size_t count,
    cudaStream_t stream
);

/**
 * Get max output length constant for memory allocation.
 */
int gpu_get_max_output_len();

}  // extern "C"

/**
 * GPU Rule Engine - Host-side manager for GPU rule processing.
 */
class GPURuleEngine {
public:
    struct Config {
        int device_id = 0;
        size_t max_words = 100'000;       // Max words per batch
        size_t max_rules = 1000;          // Max rules
        size_t max_word_bytes = 8 * 1024 * 1024;  // 8 MB for packed words
        size_t max_rule_bytes = 256 * 1024;       // 256 KB for packed rules
    };

    GPURuleEngine() = default;
    explicit GPURuleEngine(const Config& config) : config_(config) {}

    // Initialize GPU buffers
    bool init();

    // Load rules to GPU
    bool load_rules(const std::vector<std::string>& rules);

    // Load words to GPU and apply all rules
    // Returns passphrases ready for the brain wallet pipeline
    struct RuleResult {
        std::vector<std::string> passphrases;
        size_t total_generated = 0;
    };

    RuleResult apply_rules_to_words(const std::vector<std::string>& words);

    // Cleanup
    void cleanup();

    // Get number of loaded rules
    size_t num_rules() const { return rules_count_; }

    // Get device ID
    int device_id() const { return config_.device_id; }

    // Direct GPU buffer access (for zero-copy brain wallet integration)
    // After calling apply_rules_to_words_gpu(), these contain the results on GPU
    char* d_output() const { return d_output_; }
    uint32_t* d_output_lengths() const { return d_output_lengths_; }

    /**
     * Apply rules on GPU and return count (keeps data on GPU).
     * Use d_output() and d_output_lengths() to access results directly.
     *
     * @param words Input words
     * @return Number of passphrases generated (words.size() * num_rules)
     */
    size_t apply_rules_to_words_gpu(const std::vector<std::string>& words);

    /**
     * Get a single passphrase from GPU output by index.
     * Used for hit reconstruction (only called on rare bloom hits).
     *
     * @param index Index into the rule output (0 to num_passphrases-1)
     * @return The passphrase string, or empty if invalid
     */
    std::string get_passphrase_from_gpu(size_t index);

private:
    Config config_;
    bool initialized_ = false;

    // Device memory
    char* d_words_ = nullptr;
    uint32_t* d_word_offsets_ = nullptr;
    uint32_t* d_word_lengths_ = nullptr;
    char* d_rules_ = nullptr;
    uint32_t* d_rule_offsets_ = nullptr;
    uint32_t* d_rule_lengths_ = nullptr;
    char* d_output_ = nullptr;
    uint32_t* d_output_lengths_ = nullptr;

    // Host pinned memory for fast transfers
    char* h_words_ = nullptr;
    uint32_t* h_word_offsets_ = nullptr;
    uint32_t* h_word_lengths_ = nullptr;
    char* h_output_ = nullptr;
    uint32_t* h_output_lengths_ = nullptr;

    // State
    size_t rules_count_ = 0;
    cudaStream_t stream_ = 0;
};

}  // namespace gpu
}  // namespace collider
