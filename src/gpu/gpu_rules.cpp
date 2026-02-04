/**
 * GPU Rule Engine - Host-side implementation
 */

#include "gpu_rules.hpp"
#include <iostream>
#include <cstring>

#ifdef COLLIDER_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace collider {
namespace gpu {

bool GPURuleEngine::init() {
#ifdef COLLIDER_USE_CUDA
    cudaError_t err;

    // Set device
    err = cudaSetDevice(config_.device_id);
    if (err != cudaSuccess) {
        std::cerr << "[!] Failed to set CUDA device: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Create stream
    err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        std::cerr << "[!] Failed to create CUDA stream: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Allocate device memory for words
    err = cudaMalloc(&d_words_, config_.max_word_bytes);
    if (err != cudaSuccess) {
        std::cerr << "[!] Failed to allocate d_words_: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    err = cudaMalloc(&d_word_offsets_, config_.max_words * sizeof(uint32_t));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(&d_word_lengths_, config_.max_words * sizeof(uint32_t));
    if (err != cudaSuccess) return false;

    // Allocate device memory for rules
    err = cudaMalloc(&d_rules_, config_.max_rule_bytes);
    if (err != cudaSuccess) return false;

    err = cudaMalloc(&d_rule_offsets_, config_.max_rules * sizeof(uint32_t));
    if (err != cudaSuccess) return false;

    err = cudaMalloc(&d_rule_lengths_, config_.max_rules * sizeof(uint32_t));
    if (err != cudaSuccess) return false;

    // Allocate device memory for output (words x rules x max_output_len)
    size_t output_size = config_.max_words * config_.max_rules * GPU_MAX_OUTPUT_LEN;
    err = cudaMalloc(&d_output_, output_size);
    if (err != cudaSuccess) return false;

    err = cudaMalloc(&d_output_lengths_, config_.max_words * config_.max_rules * sizeof(uint32_t));
    if (err != cudaSuccess) return false;

    // Allocate pinned host memory for fast transfers
    err = cudaMallocHost(&h_words_, config_.max_word_bytes);
    if (err != cudaSuccess) return false;

    err = cudaMallocHost(&h_word_offsets_, config_.max_words * sizeof(uint32_t));
    if (err != cudaSuccess) return false;

    err = cudaMallocHost(&h_word_lengths_, config_.max_words * sizeof(uint32_t));
    if (err != cudaSuccess) return false;

    size_t h_output_size = config_.max_words * config_.max_rules * GPU_MAX_OUTPUT_LEN;
    err = cudaMallocHost(&h_output_, h_output_size);
    if (err != cudaSuccess) return false;

    err = cudaMallocHost(&h_output_lengths_, config_.max_words * config_.max_rules * sizeof(uint32_t));
    if (err != cudaSuccess) return false;

    initialized_ = true;
    return true;
#else
    std::cerr << "[!] CUDA not available\n";
    return false;
#endif
}

bool GPURuleEngine::load_rules(const std::vector<std::string>& rules) {
#ifdef COLLIDER_USE_CUDA
    if (!initialized_) return false;
    if (rules.empty()) return false;
    if (rules.size() > config_.max_rules) {
        std::cerr << "[!] Too many rules: " << rules.size() << " > " << config_.max_rules << "\n";
        return false;
    }

    // Pack rules into contiguous buffer
    std::vector<char> packed_rules;
    std::vector<uint32_t> offsets(rules.size());
    std::vector<uint32_t> lengths(rules.size());

    size_t total_bytes = 0;
    for (size_t i = 0; i < rules.size(); i++) {
        offsets[i] = total_bytes;
        lengths[i] = rules[i].size();
        total_bytes += rules[i].size();
    }

    if (total_bytes > config_.max_rule_bytes) {
        std::cerr << "[!] Rules too large: " << total_bytes << " > " << config_.max_rule_bytes << "\n";
        return false;
    }

    packed_rules.resize(total_bytes);
    for (size_t i = 0; i < rules.size(); i++) {
        std::memcpy(packed_rules.data() + offsets[i], rules[i].data(), rules[i].size());
    }

    // Copy to device
    cudaMemcpyAsync(d_rules_, packed_rules.data(), total_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_rule_offsets_, offsets.data(), rules.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_rule_lengths_, lengths.data(), rules.size() * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);

    rules_count_ = rules.size();
    return true;
#else
    return false;
#endif
}

GPURuleEngine::RuleResult GPURuleEngine::apply_rules_to_words(const std::vector<std::string>& words) {
    RuleResult result;

#ifdef COLLIDER_USE_CUDA
    if (!initialized_ || rules_count_ == 0) return result;
    if (words.empty()) return result;

    // CRITICAL: Set device before any CUDA operations
    // In multi-GPU systems, another GPU may have been selected
    cudaError_t err = cudaSetDevice(config_.device_id);
    if (err != cudaSuccess) {
        std::cerr << "[!] Failed to set CUDA device for rules: " << cudaGetErrorString(err) << "\n";
        return result;
    }

    size_t num_words = std::min(words.size(), config_.max_words);

    // Pack words into pinned buffer
    size_t total_word_bytes = 0;
    for (size_t i = 0; i < num_words; i++) {
        h_word_offsets_[i] = total_word_bytes;
        h_word_lengths_[i] = words[i].size();
        std::memcpy(h_words_ + total_word_bytes, words[i].data(), words[i].size());
        total_word_bytes += words[i].size();
    }

    // Copy words to device
    cudaMemcpyAsync(d_words_, h_words_, total_word_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_word_offsets_, h_word_offsets_, num_words * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_word_lengths_, h_word_lengths_, num_words * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_);

    // Apply rules
    err = gpu_apply_rules_cross_product(
        d_words_, d_word_offsets_, d_word_lengths_, num_words,
        d_rules_, d_rule_offsets_, d_rule_lengths_, rules_count_,
        d_output_, d_output_lengths_,
        stream_
    );

    if (err != cudaSuccess) {
        std::cerr << "[!] GPU rule application failed: " << cudaGetErrorString(err) << "\n";
        return result;
    }

    // Copy results back
    size_t total_outputs = num_words * rules_count_;
    size_t output_bytes = total_outputs * GPU_MAX_OUTPUT_LEN;

    cudaMemcpyAsync(h_output_, d_output_, output_bytes, cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(h_output_lengths_, d_output_lengths_, total_outputs * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // Extract passphrases
    result.passphrases.reserve(total_outputs);
    for (size_t i = 0; i < total_outputs; i++) {
        uint32_t len = h_output_lengths_[i];
        if (len > 0 && len < GPU_MAX_OUTPUT_LEN) {
            const char* ptr = h_output_ + i * GPU_MAX_OUTPUT_LEN;
            result.passphrases.emplace_back(ptr, len);
        }
    }

    result.total_generated = result.passphrases.size();
#endif

    return result;
}

size_t GPURuleEngine::apply_rules_to_words_gpu(const std::vector<std::string>& words) {
#ifdef COLLIDER_USE_CUDA
    if (!initialized_ || rules_count_ == 0) return 0;
    if (words.empty()) return 0;

    // CRITICAL: Set device before any CUDA operations
    cudaError_t err = cudaSetDevice(config_.device_id);
    if (err != cudaSuccess) {
        std::cerr << "[!] Failed to set CUDA device for rules: " << cudaGetErrorString(err) << "\n";
        return 0;
    }

    size_t num_words = std::min(words.size(), config_.max_words);

    // Pack words into pinned buffer
    size_t total_word_bytes = 0;
    for (size_t i = 0; i < num_words; i++) {
        h_word_offsets_[i] = total_word_bytes;
        h_word_lengths_[i] = words[i].size();
        std::memcpy(h_words_ + total_word_bytes, words[i].data(), words[i].size());
        total_word_bytes += words[i].size();
    }

    // Copy words to device
    cudaMemcpyAsync(d_words_, h_words_, total_word_bytes, cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_word_offsets_, h_word_offsets_, num_words * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(d_word_lengths_, h_word_lengths_, num_words * sizeof(uint32_t), cudaMemcpyHostToDevice, stream_);

    // Apply rules - results stay on GPU in d_output_ and d_output_lengths_
    err = gpu_apply_rules_cross_product(
        d_words_, d_word_offsets_, d_word_lengths_, num_words,
        d_rules_, d_rule_offsets_, d_rule_lengths_, rules_count_,
        d_output_, d_output_lengths_,
        stream_
    );

    if (err != cudaSuccess) {
        std::cerr << "[!] GPU rule application failed: " << cudaGetErrorString(err) << "\n";
        return 0;
    }

    // Sync to ensure rules are applied before brain wallet kernel uses them
    cudaStreamSynchronize(stream_);

    return num_words * rules_count_;
#else
    return 0;
#endif
}

std::string GPURuleEngine::get_passphrase_from_gpu(size_t index) {
#ifdef COLLIDER_USE_CUDA
    if (!initialized_) return "";

    cudaSetDevice(config_.device_id);

    // Get the length first
    uint32_t len = 0;
    cudaMemcpy(&len, d_output_lengths_ + index, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (len == 0 || len >= GPU_MAX_OUTPUT_LEN) return "";

    // Copy the passphrase
    char buffer[GPU_MAX_OUTPUT_LEN];
    cudaMemcpy(buffer, d_output_ + index * GPU_MAX_OUTPUT_LEN, len, cudaMemcpyDeviceToHost);

    return std::string(buffer, len);
#else
    return "";
#endif
}

void GPURuleEngine::cleanup() {
#ifdef COLLIDER_USE_CUDA
    if (d_words_) cudaFree(d_words_);
    if (d_word_offsets_) cudaFree(d_word_offsets_);
    if (d_word_lengths_) cudaFree(d_word_lengths_);
    if (d_rules_) cudaFree(d_rules_);
    if (d_rule_offsets_) cudaFree(d_rule_offsets_);
    if (d_rule_lengths_) cudaFree(d_rule_lengths_);
    if (d_output_) cudaFree(d_output_);
    if (d_output_lengths_) cudaFree(d_output_lengths_);

    if (h_words_) cudaFreeHost(h_words_);
    if (h_word_offsets_) cudaFreeHost(h_word_offsets_);
    if (h_word_lengths_) cudaFreeHost(h_word_lengths_);
    if (h_output_) cudaFreeHost(h_output_);
    if (h_output_lengths_) cudaFreeHost(h_output_lengths_);

    if (stream_) cudaStreamDestroy(stream_);

    d_words_ = nullptr;
    d_word_offsets_ = nullptr;
    d_word_lengths_ = nullptr;
    d_rules_ = nullptr;
    d_rule_offsets_ = nullptr;
    d_rule_lengths_ = nullptr;
    d_output_ = nullptr;
    d_output_lengths_ = nullptr;
    h_words_ = nullptr;
    h_word_offsets_ = nullptr;
    h_word_lengths_ = nullptr;
    h_output_ = nullptr;
    h_output_lengths_ = nullptr;
    stream_ = 0;

    initialized_ = false;
#endif
}

}  // namespace gpu
}  // namespace collider
