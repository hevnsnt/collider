/**
 * GPU Rule Engine Kernel
 *
 * Executes compiled rule bytecode on GPU to generate candidates.
 * Each thread processes one (word_idx, rule_idx) pair to generate one candidate.
 *
 * ARCHITECTURE:
 * - Wordlist stored in GPU global memory (uploaded once)
 * - Rules stored as packed bytecode in constant/global memory
 * - Each thread: reads word, applies rule bytecode, outputs candidate
 * - Output is variable-length, stored with length prefix
 *
 * MEMORY LAYOUT:
 * - d_words: Packed word data (length-prefixed strings)
 * - d_word_offsets: Offset into d_words for each word index
 * - d_rules: Packed rule bytecode
 * - d_rule_offsets: Offset into d_rules for each rule index
 * - d_output: Output candidates (MAX_PASSWORD_LEN per candidate)
 * - d_output_lens: Length of each output candidate
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>

namespace collider {
namespace gpu {

// Maximum password length
constexpr int MAX_PASSWORD_LEN = 64;

// Rule operation codes (must match gpu_rules.hpp)
enum RuleOp : uint8_t {
    NOP = 0x00,
    LOWERCASE = 0x01,
    UPPERCASE = 0x02,
    CAPITALIZE = 0x03,
    INVCAP = 0x04,
    TOGGLE_ALL = 0x05,
    TOGGLE_AT = 0x06,
    APPEND = 0x10,
    PREPEND = 0x11,
    DELETE_FIRST = 0x20,
    DELETE_LAST = 0x21,
    DELETE_AT = 0x22,
    DELETE_RANGE = 0x23,
    INSERT_AT = 0x24,
    REPLACE = 0x30,
    PURGE = 0x31,
    OVERWRITE_AT = 0x32,
    DUPLICATE = 0x40,
    DUPLICATE_FIRST = 0x41,
    DUPLICATE_LAST = 0x42,
    DUPLICATE_ALL = 0x43,
    DUPLICATE_N = 0x44,
    REVERSE = 0x50,
    ROTATE_LEFT = 0x51,
    ROTATE_RIGHT = 0x52,
    MEMORIZE = 0x60,
    APPEND_MEMORY = 0x61,
    TRUNCATE_AT = 0x70,
    EXTRACT = 0x71,
    SWAP_FIRST_LAST = 0x80,
    SWAP_LAST_PAIR = 0x81,
    SWAP_AT = 0x82,
    REJECT_LESS = 0x90,
    REJECT_GREATER = 0x91,
    REJECT_CONTAIN = 0x92,
    REJECT_NOT_CONTAIN = 0x93,
    REJECT_EQUALS = 0x94,
    BITWISE_LEFT = 0xA0,
    BITWISE_RIGHT = 0xA1,
    PASSTHROUGH = 0xFF,
};

/**
 * GPU Rule Interpreter - applies bytecode to word in registers.
 *
 * @param word Input word (register array)
 * @param len  Input word length (modified in place)
 * @param rule Rule bytecode pointer
 * @param rule_len Length of rule bytecode
 * @param mem Memory buffer for M/Q operations
 * @param mem_len Length of memorized word
 * @return true if rule succeeded, false if rejected
 */
__device__ bool apply_rule_bytecode(
    char* word,
    int& len,
    const uint8_t* rule,
    int rule_len,
    char* mem,
    int& mem_len
) {
    int pos = 0;

    while (pos < rule_len) {
        uint8_t op = rule[pos++];
        if (op == NOP) break;

        switch (op) {
            case PASSTHROUGH:
                // Do nothing
                break;

            case LOWERCASE:
                for (int i = 0; i < len; i++) {
                    if (word[i] >= 'A' && word[i] <= 'Z') {
                        word[i] += 32;
                    }
                }
                break;

            case UPPERCASE:
                for (int i = 0; i < len; i++) {
                    if (word[i] >= 'a' && word[i] <= 'z') {
                        word[i] -= 32;
                    }
                }
                break;

            case CAPITALIZE:
                // Capitalize first, lowercase rest
                for (int i = 0; i < len; i++) {
                    if (i == 0) {
                        if (word[i] >= 'a' && word[i] <= 'z') word[i] -= 32;
                    } else {
                        if (word[i] >= 'A' && word[i] <= 'Z') word[i] += 32;
                    }
                }
                break;

            case INVCAP:
                // Lowercase first, uppercase rest
                for (int i = 0; i < len; i++) {
                    if (i == 0) {
                        if (word[i] >= 'A' && word[i] <= 'Z') word[i] += 32;
                    } else {
                        if (word[i] >= 'a' && word[i] <= 'z') word[i] -= 32;
                    }
                }
                break;

            case TOGGLE_ALL:
                for (int i = 0; i < len; i++) {
                    if (word[i] >= 'a' && word[i] <= 'z') word[i] -= 32;
                    else if (word[i] >= 'A' && word[i] <= 'Z') word[i] += 32;
                }
                break;

            case TOGGLE_AT: {
                uint8_t n = rule[pos++];
                if (n < len) {
                    if (word[n] >= 'a' && word[n] <= 'z') word[n] -= 32;
                    else if (word[n] >= 'A' && word[n] <= 'Z') word[n] += 32;
                }
                break;
            }

            case APPEND: {
                uint8_t c = rule[pos++];
                if (len < MAX_PASSWORD_LEN - 1) {
                    word[len++] = c;
                }
                break;
            }

            case PREPEND: {
                uint8_t c = rule[pos++];
                if (len < MAX_PASSWORD_LEN - 1) {
                    // Shift right
                    for (int i = len; i > 0; i--) {
                        word[i] = word[i-1];
                    }
                    word[0] = c;
                    len++;
                }
                break;
            }

            case DELETE_FIRST:
                if (len > 0) {
                    for (int i = 0; i < len - 1; i++) {
                        word[i] = word[i+1];
                    }
                    len--;
                }
                break;

            case DELETE_LAST:
                if (len > 0) len--;
                break;

            case DELETE_AT: {
                uint8_t n = rule[pos++];
                if (n < len) {
                    for (int i = n; i < len - 1; i++) {
                        word[i] = word[i+1];
                    }
                    len--;
                }
                break;
            }

            case DELETE_RANGE: {
                uint8_t start = rule[pos++];
                uint8_t count = rule[pos++];
                if (start < len) {
                    int end = (start + count < len) ? start + count : len;
                    int shift = end - start;
                    for (int i = start; i < len - shift; i++) {
                        word[i] = word[i + shift];
                    }
                    len -= shift;
                }
                break;
            }

            case INSERT_AT: {
                uint8_t n = rule[pos++];
                uint8_t c = rule[pos++];
                if (n <= len && len < MAX_PASSWORD_LEN - 1) {
                    for (int i = len; i > (int)n; i--) {
                        word[i] = word[i-1];
                    }
                    word[n] = c;
                    len++;
                }
                break;
            }

            case REPLACE: {
                uint8_t x = rule[pos++];
                uint8_t y = rule[pos++];
                for (int i = 0; i < len; i++) {
                    if ((uint8_t)word[i] == x) word[i] = y;
                }
                break;
            }

            case PURGE: {
                uint8_t x = rule[pos++];
                int j = 0;
                for (int i = 0; i < len; i++) {
                    if ((uint8_t)word[i] != x) {
                        word[j++] = word[i];
                    }
                }
                len = j;
                break;
            }

            case OVERWRITE_AT: {
                uint8_t n = rule[pos++];
                uint8_t c = rule[pos++];
                if (n < len) {
                    word[n] = c;
                }
                break;
            }

            case DUPLICATE:
                if (len * 2 <= MAX_PASSWORD_LEN) {
                    for (int i = 0; i < len; i++) {
                        word[len + i] = word[i];
                    }
                    len *= 2;
                }
                break;

            case DUPLICATE_FIRST: {
                uint8_t n = rule[pos++];
                if (len > 0 && len + n <= MAX_PASSWORD_LEN) {
                    char c = word[0];
                    for (int i = len + n - 1; i >= n; i--) {
                        word[i] = word[i - n];
                    }
                    for (int i = 0; i < n; i++) {
                        word[i] = c;
                    }
                    len += n;
                }
                break;
            }

            case DUPLICATE_LAST: {
                uint8_t n = rule[pos++];
                if (len > 0 && len + n <= MAX_PASSWORD_LEN) {
                    char c = word[len - 1];
                    for (int i = 0; i < n; i++) {
                        word[len + i] = c;
                    }
                    len += n;
                }
                break;
            }

            case DUPLICATE_ALL:
                if (len * 2 <= MAX_PASSWORD_LEN) {
                    // Double each character: "abc" -> "aabbcc"
                    for (int i = len - 1; i >= 0; i--) {
                        word[i * 2] = word[i];
                        word[i * 2 + 1] = word[i];
                    }
                    len *= 2;
                }
                break;

            case DUPLICATE_N: {
                uint8_t n = rule[pos++];
                if (len > 0 && len * (n + 1) <= MAX_PASSWORD_LEN) {
                    for (int r = 1; r <= n; r++) {
                        for (int i = 0; i < len; i++) {
                            word[len * r + i] = word[i];
                        }
                    }
                    len *= (n + 1);
                }
                break;
            }

            case REVERSE: {
                for (int i = 0; i < len / 2; i++) {
                    char tmp = word[i];
                    word[i] = word[len - 1 - i];
                    word[len - 1 - i] = tmp;
                }
                break;
            }

            case ROTATE_LEFT:
                if (len > 1) {
                    char first = word[0];
                    for (int i = 0; i < len - 1; i++) {
                        word[i] = word[i + 1];
                    }
                    word[len - 1] = first;
                }
                break;

            case ROTATE_RIGHT:
                if (len > 1) {
                    char last = word[len - 1];
                    for (int i = len - 1; i > 0; i--) {
                        word[i] = word[i - 1];
                    }
                    word[0] = last;
                }
                break;

            case MEMORIZE:
                for (int i = 0; i < len; i++) {
                    mem[i] = word[i];
                }
                mem_len = len;
                break;

            case APPEND_MEMORY:
                if (len + mem_len <= MAX_PASSWORD_LEN) {
                    for (int i = 0; i < mem_len; i++) {
                        word[len + i] = mem[i];
                    }
                    len += mem_len;
                }
                break;

            case TRUNCATE_AT: {
                uint8_t n = rule[pos++];
                if (n < len) len = n;
                break;
            }

            case EXTRACT: {
                uint8_t start = rule[pos++];
                uint8_t count = rule[pos++];
                if (start < len) {
                    int new_len = (start + count <= len) ? count : (len - start);
                    for (int i = 0; i < new_len; i++) {
                        word[i] = word[start + i];
                    }
                    len = new_len;
                }
                break;
            }

            case SWAP_FIRST_LAST:
                if (len >= 2) {
                    char tmp = word[0];
                    word[0] = word[1];
                    word[1] = tmp;
                }
                break;

            case SWAP_LAST_PAIR:
                if (len >= 2) {
                    char tmp = word[len - 2];
                    word[len - 2] = word[len - 1];
                    word[len - 1] = tmp;
                }
                break;

            case SWAP_AT: {
                uint8_t n = rule[pos++];
                uint8_t m = rule[pos++];
                if (n < len && m < len) {
                    char tmp = word[n];
                    word[n] = word[m];
                    word[m] = tmp;
                }
                break;
            }

            // Rejection rules - return false if condition not met
            case REJECT_LESS: {
                uint8_t n = rule[pos++];
                if (len < n) return false;
                break;
            }

            case REJECT_GREATER: {
                uint8_t n = rule[pos++];
                if (len > n) return false;
                break;
            }

            case REJECT_CONTAIN: {
                uint8_t c = rule[pos++];
                for (int i = 0; i < len; i++) {
                    if ((uint8_t)word[i] == c) return false;
                }
                break;
            }

            case REJECT_NOT_CONTAIN: {
                uint8_t c = rule[pos++];
                bool found = false;
                for (int i = 0; i < len; i++) {
                    if ((uint8_t)word[i] == c) { found = true; break; }
                }
                if (!found) return false;
                break;
            }

            case REJECT_EQUALS: {
                uint8_t n = rule[pos++];
                if (len != n) return false;
                break;
            }

            case BITWISE_LEFT: {
                uint8_t n = rule[pos++];
                if (n < len) {
                    word[n] = (word[n] << 1) & 0xFF;
                }
                break;
            }

            case BITWISE_RIGHT: {
                uint8_t n = rule[pos++];
                if (n < len) {
                    word[n] = ((uint8_t)word[n]) >> 1;
                }
                break;
            }

            default:
                // Unknown op - skip
                break;
        }
    }

    return true;
}

/**
 * GPU kernel for applying rules to generate candidates.
 *
 * Each thread processes one (word_idx, rule_idx) pair.
 *
 * @param d_words Packed word data (length-prefixed: uint8_t len, then chars)
 * @param d_word_offsets Offset into d_words for each word
 * @param num_words Total number of words
 * @param d_rules Packed rule bytecode
 * @param d_rule_offsets Offset into d_rules for each rule
 * @param d_rule_lengths Length of each rule's bytecode
 * @param num_rules Total number of rules
 * @param d_output Output buffer (MAX_PASSWORD_LEN per candidate)
 * @param d_output_lens Output length for each candidate
 * @param d_valid Output validity flag (true if rule succeeded)
 * @param total_candidates Total candidates to generate
 */
__global__ void apply_rules_kernel(
    const uint8_t* __restrict__ d_words,
    const uint32_t* __restrict__ d_word_offsets,
    uint32_t num_words,
    const uint8_t* __restrict__ d_rules,
    const uint32_t* __restrict__ d_rule_offsets,
    const uint16_t* __restrict__ d_rule_lengths,
    uint32_t num_rules,
    char* __restrict__ d_output,
    uint8_t* __restrict__ d_output_lens,
    bool* __restrict__ d_valid,
    uint64_t total_candidates
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_candidates) return;

    // Decode (word_idx, rule_idx) from linear index
    uint32_t word_idx = idx / num_rules;
    uint32_t rule_idx = idx % num_rules;

    if (word_idx >= num_words) {
        d_valid[idx] = false;
        d_output_lens[idx] = 0;
        return;
    }

    // Load word from global memory
    uint32_t word_offset = d_word_offsets[word_idx];
    uint8_t word_len = d_words[word_offset];
    const uint8_t* word_data = &d_words[word_offset + 1];

    // Copy word to register array
    char word[MAX_PASSWORD_LEN];
    int len = word_len;
    for (int i = 0; i < len && i < MAX_PASSWORD_LEN; i++) {
        word[i] = word_data[i];
    }

    // Load rule
    uint32_t rule_offset = d_rule_offsets[rule_idx];
    uint16_t rule_len = d_rule_lengths[rule_idx];
    const uint8_t* rule = &d_rules[rule_offset];

    // Memory buffer for M/Q operations
    char mem[MAX_PASSWORD_LEN];
    int mem_len = 0;

    // Apply rule
    bool valid = apply_rule_bytecode(word, len, rule, rule_len, mem, mem_len);

    // Write output
    char* output = &d_output[idx * MAX_PASSWORD_LEN];
    if (valid && len > 0) {
        for (int i = 0; i < len; i++) {
            output[i] = word[i];
        }
        d_output_lens[idx] = len;
        d_valid[idx] = true;
    } else {
        d_output_lens[idx] = 0;
        d_valid[idx] = false;
    }
}

/**
 * Host wrapper for GPU rule application.
 */
struct GPURuleEngine {
    // Device buffers
    uint8_t* d_words = nullptr;
    uint32_t* d_word_offsets = nullptr;
    uint8_t* d_rules = nullptr;
    uint32_t* d_rule_offsets = nullptr;
    uint16_t* d_rule_lengths = nullptr;
    char* d_output = nullptr;
    uint8_t* d_output_lens = nullptr;
    bool* d_valid = nullptr;

    uint32_t num_words = 0;
    uint32_t num_rules = 0;
    size_t max_candidates = 0;

    bool initialized = false;

    /**
     * Initialize GPU rule engine with wordlist and rules.
     */
    bool init(
        const std::vector<std::string>& words,
        const uint8_t* packed_rules,
        size_t rules_size,
        const uint32_t* rule_offsets,
        const uint16_t* rule_lengths,
        uint32_t rules_count,
        size_t batch_size
    ) {
        num_words = words.size();
        num_rules = rules_count;
        max_candidates = batch_size;

        // Pack words with length prefix
        std::vector<uint8_t> packed_words;
        std::vector<uint32_t> word_offsets;
        word_offsets.reserve(num_words);

        for (const auto& word : words) {
            word_offsets.push_back(packed_words.size());
            packed_words.push_back(static_cast<uint8_t>(word.length()));
            for (char c : word) {
                packed_words.push_back(static_cast<uint8_t>(c));
            }
        }

        // Allocate device memory
        cudaError_t err;

        err = cudaMalloc(&d_words, packed_words.size());
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_word_offsets, num_words * sizeof(uint32_t));
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_rules, rules_size);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_rule_offsets, num_rules * sizeof(uint32_t));
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_rule_lengths, num_rules * sizeof(uint16_t));
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_output, max_candidates * MAX_PASSWORD_LEN);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_output_lens, max_candidates);
        if (err != cudaSuccess) return false;

        err = cudaMalloc(&d_valid, max_candidates * sizeof(bool));
        if (err != cudaSuccess) return false;

        // Copy data to device
        cudaMemcpy(d_words, packed_words.data(), packed_words.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_word_offsets, word_offsets.data(), num_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rules, packed_rules, rules_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_rule_offsets, rule_offsets, num_rules * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rule_lengths, rule_lengths, num_rules * sizeof(uint16_t), cudaMemcpyHostToDevice);

        initialized = true;
        return true;
    }

    /**
     * Generate candidates for a range of (word_idx, rule_idx) pairs.
     */
    bool generate_batch(
        uint64_t start_word_idx,
        uint64_t start_rule_idx,
        size_t count,
        std::vector<std::string>& output,
        cudaStream_t stream = 0
    ) {
        if (!initialized || count == 0) return false;

        // Ensure count doesn't exceed max
        if (count > max_candidates) count = max_candidates;

        // Calculate total candidates
        uint64_t total = count;

        // Launch kernel
        int threads_per_block = 256;
        int num_blocks = (total + threads_per_block - 1) / threads_per_block;

        apply_rules_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            d_words, d_word_offsets, num_words,
            d_rules, d_rule_offsets, d_rule_lengths, num_rules,
            d_output, d_output_lens, d_valid,
            total
        );

        // Copy results back
        // Note: std::vector<bool> is a specialization that lacks .data() method
        // Use std::vector<char> instead for CUDA memory transfers
        std::vector<char> h_output(count * MAX_PASSWORD_LEN);
        std::vector<uint8_t> h_lens(count);
        std::vector<char> h_valid(count);  // char instead of bool for .data() compatibility

        cudaMemcpyAsync(h_output.data(), d_output, count * MAX_PASSWORD_LEN, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_lens.data(), d_output_lens, count, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_valid.data(), d_valid, count * sizeof(char), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Build output strings
        output.clear();
        output.reserve(count);
        for (size_t i = 0; i < count; i++) {
            if (h_valid[i] && h_lens[i] > 0) {
                output.emplace_back(&h_output[i * MAX_PASSWORD_LEN], h_lens[i]);
            }
        }

        return true;
    }

    /**
     * Cleanup GPU resources.
     */
    void cleanup() {
        if (d_words) { cudaFree(d_words); d_words = nullptr; }
        if (d_word_offsets) { cudaFree(d_word_offsets); d_word_offsets = nullptr; }
        if (d_rules) { cudaFree(d_rules); d_rules = nullptr; }
        if (d_rule_offsets) { cudaFree(d_rule_offsets); d_rule_offsets = nullptr; }
        if (d_rule_lengths) { cudaFree(d_rule_lengths); d_rule_lengths = nullptr; }
        if (d_output) { cudaFree(d_output); d_output = nullptr; }
        if (d_output_lens) { cudaFree(d_output_lens); d_output_lens = nullptr; }
        if (d_valid) { cudaFree(d_valid); d_valid = nullptr; }
        initialized = false;
    }

    ~GPURuleEngine() {
        cleanup();
    }
};

}  // namespace gpu
}  // namespace collider
