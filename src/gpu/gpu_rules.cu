/**
 * GPU Rule Application Engine
 *
 * High-performance GPU implementation of hashcat-style rule processing.
 * Processes millions of word+rule combinations in parallel.
 *
 * Supported rules:
 *   : - No change (passthrough)
 *   l - Lowercase all
 *   u - Uppercase all
 *   c - Capitalize first
 *   C - Lowercase first, uppercase rest
 *   t - Toggle case
 *   r - Reverse
 *   d - Duplicate
 *   f - Reflect (append reversed)
 *   $ - Append character
 *   ^ - Prepend character
 *   s - Substitute char
 *   @ - Purge char
 *   [ - Delete first
 *   ] - Delete last
 *   { - Rotate left
 *   } - Rotate right
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace collider {
namespace gpu {

// Maximum output length after rules like 'd' (duplicate)
constexpr int MAX_OUTPUT_LEN = 256;

// =============================================================================
// DEVICE HELPER FUNCTIONS
// =============================================================================

__device__ __forceinline__ char gpu_tolower(char c) {
    return (c >= 'A' && c <= 'Z') ? (c + 32) : c;
}

__device__ __forceinline__ char gpu_toupper(char c) {
    return (c >= 'a' && c <= 'z') ? (c - 32) : c;
}

__device__ __forceinline__ bool gpu_isupper(char c) {
    return (c >= 'A' && c <= 'Z');
}

__device__ __forceinline__ bool gpu_islower(char c) {
    return (c >= 'a' && c <= 'z');
}

/**
 * Apply a single rule to a word in-place.
 *
 * @param word     The word buffer (modified in place)
 * @param len      Current length of word (updated)
 * @param rule     Rule string
 * @param rule_len Length of rule string
 * @return New length of word after rule application
 */
__device__ int apply_rule_gpu(char* word, int len, const char* rule, int rule_len) {
    if (len <= 0 || len >= MAX_OUTPUT_LEN) return len;

    for (int i = 0; i < rule_len && len > 0 && len < MAX_OUTPUT_LEN; i++) {
        char cmd = rule[i];

        switch (cmd) {
            case ':':  // No-op
                break;

            case 'l':  // Lowercase all
                for (int j = 0; j < len; j++) {
                    word[j] = gpu_tolower(word[j]);
                }
                break;

            case 'u':  // Uppercase all
                for (int j = 0; j < len; j++) {
                    word[j] = gpu_toupper(word[j]);
                }
                break;

            case 'c':  // Capitalize first, lowercase rest
                if (len > 0) {
                    word[0] = gpu_toupper(word[0]);
                    for (int j = 1; j < len; j++) {
                        word[j] = gpu_tolower(word[j]);
                    }
                }
                break;

            case 'C':  // Lowercase first, uppercase rest
                if (len > 0) {
                    word[0] = gpu_tolower(word[0]);
                    for (int j = 1; j < len; j++) {
                        word[j] = gpu_toupper(word[j]);
                    }
                }
                break;

            case 't':  // Toggle case
                for (int j = 0; j < len; j++) {
                    if (gpu_isupper(word[j])) {
                        word[j] = gpu_tolower(word[j]);
                    } else if (gpu_islower(word[j])) {
                        word[j] = gpu_toupper(word[j]);
                    }
                }
                break;

            case 'r':  // Reverse
                for (int j = 0; j < len / 2; j++) {
                    char tmp = word[j];
                    word[j] = word[len - 1 - j];
                    word[len - 1 - j] = tmp;
                }
                break;

            case 'd':  // Duplicate entire word
                if (len * 2 < MAX_OUTPUT_LEN) {
                    for (int j = 0; j < len; j++) {
                        word[len + j] = word[j];
                    }
                    len *= 2;
                }
                break;

            case 'f':  // Reflect (append reversed)
                if (len * 2 < MAX_OUTPUT_LEN) {
                    for (int j = 0; j < len; j++) {
                        word[len + j] = word[len - 1 - j];
                    }
                    len *= 2;
                }
                break;

            case '$':  // Append character
                if (i + 1 < rule_len && len + 1 < MAX_OUTPUT_LEN) {
                    word[len++] = rule[++i];
                }
                break;

            case '^':  // Prepend character
                if (i + 1 < rule_len && len + 1 < MAX_OUTPUT_LEN) {
                    // Shift right
                    for (int j = len; j > 0; j--) {
                        word[j] = word[j-1];
                    }
                    word[0] = rule[++i];
                    len++;
                }
                break;

            case 's':  // Substitute char X with char Y
                if (i + 2 < rule_len) {
                    char from = rule[++i];
                    char to = rule[++i];
                    for (int j = 0; j < len; j++) {
                        if (word[j] == from) word[j] = to;
                    }
                }
                break;

            case '@':  // Purge all instances of char
                if (i + 1 < rule_len) {
                    char purge = rule[++i];
                    int new_len = 0;
                    for (int j = 0; j < len; j++) {
                        if (word[j] != purge) {
                            word[new_len++] = word[j];
                        }
                    }
                    len = new_len;
                }
                break;

            case '[':  // Delete first character
                if (len > 0) {
                    for (int j = 0; j < len - 1; j++) {
                        word[j] = word[j + 1];
                    }
                    len--;
                }
                break;

            case ']':  // Delete last character
                if (len > 0) {
                    len--;
                }
                break;

            case '{':  // Rotate left
                if (len > 0) {
                    char first = word[0];
                    for (int j = 0; j < len - 1; j++) {
                        word[j] = word[j + 1];
                    }
                    word[len - 1] = first;
                }
                break;

            case '}':  // Rotate right
                if (len > 0) {
                    char last = word[len - 1];
                    for (int j = len - 1; j > 0; j--) {
                        word[j] = word[j - 1];
                    }
                    word[0] = last;
                }
                break;

            case 'T':  // Toggle case at position N
                if (i + 1 < rule_len) {
                    int pos = rule[++i] - '0';
                    if (pos >= 0 && pos < len) {
                        if (gpu_isupper(word[pos])) {
                            word[pos] = gpu_tolower(word[pos]);
                        } else if (gpu_islower(word[pos])) {
                            word[pos] = gpu_toupper(word[pos]);
                        }
                    }
                }
                break;

            case 'p':  // Duplicate N times
                if (i + 1 < rule_len) {
                    int times = rule[++i] - '0';
                    if (times > 0 && times < 10 && len * (times + 1) < MAX_OUTPUT_LEN) {
                        int orig_len = len;
                        for (int t = 0; t < times; t++) {
                            for (int j = 0; j < orig_len; j++) {
                                word[len++] = word[j];
                            }
                        }
                    }
                }
                break;

            case 'q':  // Duplicate all characters
                if (len * 2 < MAX_OUTPUT_LEN) {
                    // Work backwards to avoid overwriting
                    for (int j = len - 1; j >= 0; j--) {
                        word[j * 2] = word[j];
                        word[j * 2 + 1] = word[j];
                    }
                    len *= 2;
                }
                break;

            case 'k':  // Swap first two characters
                if (len >= 2) {
                    char tmp = word[0];
                    word[0] = word[1];
                    word[1] = tmp;
                }
                break;

            case 'K':  // Swap last two characters
                if (len >= 2) {
                    char tmp = word[len - 2];
                    word[len - 2] = word[len - 1];
                    word[len - 1] = tmp;
                }
                break;

            case '+':  // Increment character at position N
                if (i + 1 < rule_len) {
                    int pos = rule[++i] - '0';
                    if (pos >= 0 && pos < len) {
                        word[pos]++;
                    }
                }
                break;

            case '-':  // Decrement character at position N
                if (i + 1 < rule_len) {
                    int pos = rule[++i] - '0';
                    if (pos >= 0 && pos < len) {
                        word[pos]--;
                    }
                }
                break;

            case ' ':  // Skip spaces
                break;

            default:
                // Unknown rule character, skip
                break;
        }
    }

    return len;
}

// =============================================================================
// GPU KERNELS
// =============================================================================

/**
 * Apply rules to a batch of words.
 * Each thread handles one (word, rule) pair.
 *
 * @param d_words       Input words (packed, variable length)
 * @param d_word_offsets Offset into d_words for each word
 * @param d_word_lengths Length of each word
 * @param d_rules       Rules (packed, variable length)
 * @param d_rule_offsets Offset into d_rules for each rule
 * @param d_rule_lengths Length of each rule
 * @param d_word_indices Which word to use for each output
 * @param d_rule_indices Which rule to use for each output
 * @param d_output      Output buffer (packed)
 * @param d_output_offsets Pre-computed output offsets (MAX_OUTPUT_LEN per candidate)
 * @param d_output_lengths Output: actual length of each result
 * @param count         Number of (word, rule) pairs to process
 */
__global__ void apply_rules_kernel(
    const char* __restrict__ d_words,
    const uint32_t* __restrict__ d_word_offsets,
    const uint32_t* __restrict__ d_word_lengths,
    const char* __restrict__ d_rules,
    const uint32_t* __restrict__ d_rule_offsets,
    const uint32_t* __restrict__ d_rule_lengths,
    const uint32_t* __restrict__ d_word_indices,
    const uint32_t* __restrict__ d_rule_indices,
    char* __restrict__ d_output,
    const uint32_t* __restrict__ d_output_offsets,
    uint32_t* __restrict__ d_output_lengths,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Get word and rule indices
    uint32_t word_idx = d_word_indices[idx];
    uint32_t rule_idx = d_rule_indices[idx];

    // Get word
    const char* word_ptr = d_words + d_word_offsets[word_idx];
    int word_len = d_word_lengths[word_idx];

    // Get rule
    const char* rule_ptr = d_rules + d_rule_offsets[rule_idx];
    int rule_len = d_rule_lengths[rule_idx];

    // Copy word to output buffer
    char* output = d_output + d_output_offsets[idx];

    #pragma unroll 4
    for (int i = 0; i < word_len && i < MAX_OUTPUT_LEN; i++) {
        output[i] = word_ptr[i];
    }

    // Apply rule
    int result_len = apply_rule_gpu(output, word_len, rule_ptr, rule_len);

    // Store result length
    d_output_lengths[idx] = result_len;
}

/**
 * Simplified kernel for wordlist + single rule.
 * More efficient when applying the same rule to many words.
 */
__global__ void apply_single_rule_kernel(
    const char* __restrict__ d_words,
    const uint32_t* __restrict__ d_word_offsets,
    const uint32_t* __restrict__ d_word_lengths,
    const char* __restrict__ d_rule,
    int rule_len,
    char* __restrict__ d_output,
    uint32_t* __restrict__ d_output_lengths,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Get word
    const char* word_ptr = d_words + d_word_offsets[idx];
    int word_len = d_word_lengths[idx];

    // Output location (fixed stride for simplicity)
    char* output = d_output + idx * MAX_OUTPUT_LEN;

    // Copy word
    #pragma unroll 4
    for (int i = 0; i < word_len && i < MAX_OUTPUT_LEN; i++) {
        output[i] = word_ptr[i];
    }

    // Apply rule
    int result_len = apply_rule_gpu(output, word_len, d_rule, rule_len);
    d_output_lengths[idx] = result_len;
}

/**
 * Optimized kernel: wordlist x rules (cross product).
 * Thread (i,j) processes word[i] with rule[j].
 * Output index = word_idx * num_rules + rule_idx
 */
__global__ void apply_rules_cross_product_kernel(
    const char* __restrict__ d_words,
    const uint32_t* __restrict__ d_word_offsets,
    const uint32_t* __restrict__ d_word_lengths,
    uint32_t num_words,
    const char* __restrict__ d_rules,
    const uint32_t* __restrict__ d_rule_offsets,
    const uint32_t* __restrict__ d_rule_lengths,
    uint32_t num_rules,
    char* __restrict__ d_output,
    uint32_t* __restrict__ d_output_lengths,
    size_t total_count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_count) return;

    // Decompose linear index into (word_idx, rule_idx)
    uint32_t word_idx = idx / num_rules;
    uint32_t rule_idx = idx % num_rules;

    if (word_idx >= num_words) return;

    // Get word
    const char* word_ptr = d_words + d_word_offsets[word_idx];
    int word_len = d_word_lengths[word_idx];

    // Get rule
    const char* rule_ptr = d_rules + d_rule_offsets[rule_idx];
    int rule_len = d_rule_lengths[rule_idx];

    // Output location
    char* output = d_output + idx * MAX_OUTPUT_LEN;

    // Copy word
    for (int i = 0; i < word_len && i < MAX_OUTPUT_LEN; i++) {
        output[i] = word_ptr[i];
    }

    // Apply rule
    int result_len = apply_rule_gpu(output, word_len, rule_ptr, rule_len);
    d_output_lengths[idx] = result_len;
}

// =============================================================================
// HOST API
// =============================================================================

extern "C" {

/**
 * Apply rules to words (cross product: each word x each rule).
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
) {
    size_t total = (size_t)num_words * num_rules;

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    apply_rules_cross_product_kernel<<<blocks, threads, 0, stream>>>(
        d_words,
        d_word_offsets,
        d_word_lengths,
        num_words,
        d_rules,
        d_rule_offsets,
        d_rule_lengths,
        num_rules,
        d_output,
        d_output_lengths,
        total
    );

    return cudaGetLastError();
}

/**
 * Apply single rule to all words.
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
) {
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    apply_single_rule_kernel<<<blocks, threads, 0, stream>>>(
        d_words,
        d_word_offsets,
        d_word_lengths,
        d_rule,
        rule_len,
        d_output,
        d_output_lengths,
        count
    );

    return cudaGetLastError();
}

/**
 * Get max output length constant for memory allocation.
 */
int gpu_get_max_output_len() {
    return MAX_OUTPUT_LEN;
}

}  // extern "C"

}  // namespace gpu
}  // namespace collider
