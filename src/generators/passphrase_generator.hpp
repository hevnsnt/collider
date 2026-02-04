/**
 * Collider Passphrase Generator
 *
 * Multi-source passphrase generation with rule application and priority ordering.
 * Implements the intelligence layer described in WORDLIST-ARCHITECTURE.md.
 */

#pragma once

#include "../core/types.hpp"
#include "../core/rule_engine.hpp"
#include "priority_queue.hpp"

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <memory>
#include <thread>
#include <atomic>
#include <filesystem>

// Platform-specific includes for memory mapping
#ifdef _WIN32
#define NOMINMAX  // Prevent Windows.h from defining min/max macros
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace collider {

/**
 * Base class for passphrase sources.
 */
class PassphraseSource {
public:
    virtual ~PassphraseSource() = default;

    /**
     * Generate candidates from this source.
     * @param callback Called for each generated candidate
     */
    virtual void generate(CandidateCallback callback) = 0;

    /**
     * Get the source type.
     */
    virtual CandidateSource type() const = 0;

    /**
     * Get estimated total candidates.
     */
    virtual size_t estimated_size() const = 0;
};

/**
 * Wordlist file source.
 */
class WordlistSource : public PassphraseSource {
public:
    WordlistSource(
        const std::string& path,
        CandidateSource source_type,
        float base_priority = 1.0f
    ) : path_(path),
        source_type_(source_type),
        base_priority_(base_priority) {

        // Count lines for estimation
        std::ifstream file(path);
        estimated_size_ = std::count(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>(),
            '\n'
        );
    }

    void generate(CandidateCallback callback) override {
        std::ifstream file(path_);
        std::string line;

        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;

            // Normalize
            line = normalize(line);
            if (line.empty()) continue;

            callback(Candidate{
                .phrase = std::move(line),
                .priority = base_priority_,
                .source = source_type_,
                .rule_applied = ":"
            });
        }
    }

    CandidateSource type() const override { return source_type_; }
    size_t estimated_size() const override { return estimated_size_; }

private:
    std::string path_;
    CandidateSource source_type_;
    float base_priority_;
    size_t estimated_size_ = 0;

    std::string normalize(std::string_view input) {
        std::string result;
        result.reserve(input.size());

        // Trim and collapse whitespace
        bool last_was_space = true;
        for (char c : input) {
            if (std::isspace(c)) {
                if (!last_was_space) {
                    result += ' ';
                    last_was_space = true;
                }
            } else {
                result += c;
                last_was_space = false;
            }
        }

        // Trim trailing space
        if (!result.empty() && result.back() == ' ') {
            result.pop_back();
        }

        return result;
    }
};

/**
 * Frequency-weighted wordlist (like RockYou with counts).
 */
class FrequencyWordlistSource : public PassphraseSource {
public:
    FrequencyWordlistSource(
        const std::string& path,
        CandidateSource source_type = CandidateSource::PASSWORD_COMMON
    ) : path_(path), source_type_(source_type) {}

    void generate(CandidateCallback callback) override {
        std::ifstream file(path_);
        std::string line;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            // Format: count<tab>password or just password
            std::string password;
            float priority = 0.01f;

            size_t tab = line.find('\t');
            if (tab != std::string::npos) {
                uint64_t count = std::stoull(line.substr(0, tab));
                password = line.substr(tab + 1);
                // Log scale for frequency
                priority = std::min(1.0f, std::log10(1.0f + count) / 6.0f);
            } else {
                password = line;
            }

            callback(Candidate{
                .phrase = std::move(password),
                .priority = priority,
                .source = source_type_,
                .rule_applied = ":"
            });
        }
    }

    CandidateSource type() const override { return source_type_; }
    size_t estimated_size() const override { return 0; }  // Unknown without reading

private:
    std::string path_;
    CandidateSource source_type_;
};

/**
 * Memory-mapped wordlist source for large files.
 *
 * OPTIMIZATION: Uses mmap for zero-copy file reading, dramatically reducing
 * memory overhead and I/O latency for multi-GB wordlists. The kernel handles
 * page-in/page-out efficiently, and we avoid double-buffering data.
 *
 * Performance characteristics:
 * - Streaming access pattern (sequential read)
 * - Near-zero startup time regardless of file size
 * - Memory pressure handled by kernel page cache
 * - Works well with files larger than RAM
 */
class MmapWordlistSource : public PassphraseSource {
public:
    MmapWordlistSource(
        const std::string& path,
        CandidateSource source_type,
        float base_priority = 1.0f
    ) : path_(path),
        source_type_(source_type),
        base_priority_(base_priority),
        mapped_data_(nullptr),
        mapped_size_(0)
#ifdef _WIN32
        , file_handle_(INVALID_HANDLE_VALUE),
        mapping_handle_(nullptr)
#else
        , fd_(-1)
#endif
    {
        // Get file size for estimation
        struct stat st;
        if (stat(path.c_str(), &st) == 0) {
            mapped_size_ = st.st_size;
            // Estimate ~20 bytes per line average for wordlists
            estimated_size_ = mapped_size_ / 20;
        }
    }

    ~MmapWordlistSource() {
        unmap();
    }

    void generate(CandidateCallback callback) override {
        if (!map()) {
            // Fallback to standard file reading if mmap fails
            fallback_generate(callback);
            return;
        }

        const char* data = static_cast<const char*>(mapped_data_);
        const char* end = data + mapped_size_;
        const char* line_start = data;

        while (line_start < end) {
            // Find end of line
            const char* line_end = line_start;
            while (line_end < end && *line_end != '\n' && *line_end != '\r') {
                ++line_end;
            }

            // Extract line (skip empty and comments)
            size_t line_len = line_end - line_start;
            if (line_len > 0 && *line_start != '#') {
                std::string line(line_start, line_len);
                line = normalize(line);

                if (!line.empty()) {
                    callback(Candidate{
                        .phrase = std::move(line),
                        .priority = base_priority_,
                        .source = source_type_,
                        .rule_applied = ":"
                    });
                }
            }

            // Skip newline characters
            while (line_end < end && (*line_end == '\n' || *line_end == '\r')) {
                ++line_end;
            }
            line_start = line_end;
        }

        unmap();
    }

    CandidateSource type() const override { return source_type_; }
    size_t estimated_size() const override { return estimated_size_; }

private:
    std::string path_;
    CandidateSource source_type_;
    float base_priority_;
    size_t estimated_size_ = 0;

    void* mapped_data_;
    size_t mapped_size_;

#ifdef _WIN32
    HANDLE file_handle_;
    HANDLE mapping_handle_;

    bool map() {
        file_handle_ = CreateFileA(
            path_.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
            nullptr
        );

        if (file_handle_ == INVALID_HANDLE_VALUE) return false;

        mapping_handle_ = CreateFileMappingA(
            file_handle_,
            nullptr,
            PAGE_READONLY,
            0, 0,
            nullptr
        );

        if (!mapping_handle_) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }

        mapped_data_ = MapViewOfFile(
            mapping_handle_,
            FILE_MAP_READ,
            0, 0,
            0
        );

        if (!mapped_data_) {
            CloseHandle(mapping_handle_);
            CloseHandle(file_handle_);
            mapping_handle_ = nullptr;
            file_handle_ = INVALID_HANDLE_VALUE;
            return false;
        }

        return true;
    }

    void unmap() {
        if (mapped_data_) {
            UnmapViewOfFile(mapped_data_);
            mapped_data_ = nullptr;
        }
        if (mapping_handle_) {
            CloseHandle(mapping_handle_);
            mapping_handle_ = nullptr;
        }
        if (file_handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle_);
            file_handle_ = INVALID_HANDLE_VALUE;
        }
    }

#else  // POSIX (Linux, macOS)
    int fd_;

    bool map() {
        fd_ = open(path_.c_str(), O_RDONLY);
        if (fd_ < 0) return false;

        // Advise kernel about sequential access pattern
        #ifdef POSIX_FADV_SEQUENTIAL
        posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
        #endif

        mapped_data_ = mmap(
            nullptr,
            mapped_size_,
            PROT_READ,
            MAP_PRIVATE | MAP_NORESERVE,  // MAP_NORESERVE for better overcommit behavior
            fd_,
            0
        );

        if (mapped_data_ == MAP_FAILED) {
            mapped_data_ = nullptr;
            close(fd_);
            fd_ = -1;
            return false;
        }

        // Advise kernel about sequential access
        madvise(mapped_data_, mapped_size_, MADV_SEQUENTIAL);

        return true;
    }

    void unmap() {
        if (mapped_data_) {
            munmap(mapped_data_, mapped_size_);
            mapped_data_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }
#endif

    void fallback_generate(CandidateCallback callback) {
        // Standard ifstream fallback if mmap fails
        std::ifstream file(path_);
        std::string line;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            line = normalize(line);
            if (line.empty()) continue;

            callback(Candidate{
                .phrase = std::move(line),
                .priority = base_priority_,
                .source = source_type_,
                .rule_applied = ":"
            });
        }
    }

    std::string normalize(std::string_view input) {
        std::string result;
        result.reserve(input.size());

        bool last_was_space = true;
        for (char c : input) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!last_was_space) {
                    result += ' ';
                    last_was_space = true;
                }
            } else {
                result += c;
                last_was_space = false;
            }
        }

        if (!result.empty() && result.back() == ' ') {
            result.pop_back();
        }

        return result;
    }
};

/**
 * Combinator source - combines words from multiple lists.
 */
class CombinatorSource : public PassphraseSource {
public:
    CombinatorSource(
        const std::vector<std::string>& wordlist_paths,
        size_t min_words = 2,
        size_t max_words = 4,
        const std::vector<std::string>& separators = {"", " ", "_", "-"}
    ) : min_words_(min_words),
        max_words_(max_words),
        separators_(separators) {

        // Load all wordlists
        for (const auto& path : wordlist_paths) {
            std::ifstream file(path);
            std::string line;
            std::vector<std::string> words;

            while (std::getline(file, line)) {
                if (!line.empty() && line[0] != '#') {
                    words.push_back(line);
                }
            }

            if (!words.empty()) {
                wordlists_.push_back(std::move(words));
            }
        }
    }

    void generate(CandidateCallback callback) override {
        if (wordlists_.empty()) return;

        // Generate combinations
        for (size_t num_words = min_words_; num_words <= max_words_; ++num_words) {
            generate_combinations(num_words, callback);
        }
    }

    CandidateSource type() const override { return CandidateSource::COMBINATOR; }

    size_t estimated_size() const override {
        if (wordlists_.empty()) return 0;

        size_t total = 0;
        size_t list_size = wordlists_[0].size();

        for (size_t n = min_words_; n <= max_words_; ++n) {
            size_t combinations = 1;
            for (size_t i = 0; i < n; ++i) {
                combinations *= list_size;
            }
            total += combinations * separators_.size();
        }

        return total;
    }

private:
    std::vector<std::vector<std::string>> wordlists_;
    size_t min_words_;
    size_t max_words_;
    std::vector<std::string> separators_;

    void generate_combinations(size_t num_words, CandidateCallback callback) {
        std::vector<size_t> indices(num_words, 0);
        const auto& words = wordlists_[0];  // Use first list for now
        size_t list_size = words.size();

        while (true) {
            // Generate phrase for current indices
            for (const auto& sep : separators_) {
                std::string phrase;
                for (size_t i = 0; i < num_words; ++i) {
                    if (i > 0) phrase += sep;
                    phrase += words[indices[i]];
                }

                // Priority decreases with length
                float priority = 0.1f / static_cast<float>(num_words);

                callback(Candidate{
                    .phrase = std::move(phrase),
                    .priority = priority,
                    .source = CandidateSource::COMBINATOR,
                    .rule_applied = ":"
                });
            }

            // Increment indices
            size_t pos = num_words - 1;
            while (pos < num_words) {
                indices[pos]++;
                if (indices[pos] < list_size) break;
                indices[pos] = 0;
                if (pos == 0) return;  // All combinations exhausted
                pos--;
            }
        }
    }
};

/**
 * Rule-expanding source - applies rules to base candidates.
 */
class RuleExpandingSource : public PassphraseSource {
public:
    RuleExpandingSource(
        std::unique_ptr<PassphraseSource> base_source,
        const std::vector<std::string>& rules
    ) : base_source_(std::move(base_source)),
        rules_(rules) {}

    void generate(CandidateCallback callback) override {
        base_source_->generate([this, &callback](Candidate&& base) {
            // Apply each rule
            for (const auto& rule : rules_) {
                try {
                    std::string mutated = rule_engine_.apply(base.phrase, rule);

                    // Skip if unchanged (passthrough) or empty
                    if (mutated.empty()) continue;
                    if (rule == ":" && mutated == base.phrase) {
                        // Still emit passthrough once
                        callback(Candidate{
                            .phrase = std::move(mutated),
                            .priority = base.priority,
                            .source = base.source,
                            .rule_applied = rule
                        });
                        continue;
                    }

                    // Slightly lower priority for rule-generated variants
                    callback(Candidate{
                        .phrase = std::move(mutated),
                        .priority = base.priority * 0.9f,
                        .source = base.source,
                        .rule_applied = rule
                    });
                } catch (...) {
                    // Skip invalid rules
                }
            }
        });
    }

    CandidateSource type() const override { return base_source_->type(); }

    size_t estimated_size() const override {
        return base_source_->estimated_size() * rules_.size();
    }

private:
    std::unique_ptr<PassphraseSource> base_source_;
    std::vector<std::string> rules_;
    RuleEngine rule_engine_;
};

/**
 * Main passphrase generator coordinating all sources.
 */
class PassphraseGenerator {
public:
    explicit PassphraseGenerator(
        std::shared_ptr<CandidatePriorityQueue> queue,
        std::shared_ptr<WeightedSourceManager> source_manager = nullptr
    ) : queue_(queue),
        source_manager_(source_manager ? source_manager
                                       : std::make_shared<WeightedSourceManager>()) {}

    /**
     * Add a source to the generator.
     */
    void add_source(std::unique_ptr<PassphraseSource> source) {
        sources_.push_back(std::move(source));
    }

    /**
     * Add a wordlist file with automatic source detection.
     */
    void add_wordlist(const std::string& path, CandidateSource type) {
        sources_.push_back(std::make_unique<WordlistSource>(path, type));
    }

    /**
     * Add a wordlist with rule expansion.
     */
    void add_wordlist_with_rules(
        const std::string& path,
        CandidateSource type,
        const std::vector<std::string>& rules
    ) {
        auto base = std::make_unique<WordlistSource>(path, type);
        sources_.push_back(
            std::make_unique<RuleExpandingSource>(std::move(base), rules)
        );
    }

    /**
     * Start generating candidates in background thread.
     */
    void start() {
        if (running_.exchange(true)) return;

        generator_thread_ = std::thread([this]() {
            run_generation();
        });
    }

    /**
     * Stop generation.
     */
    void stop() {
        running_ = false;
        if (generator_thread_.joinable()) {
            generator_thread_.join();
        }
    }

    /**
     * Check if generation is complete.
     */
    bool is_complete() const { return complete_; }

    /**
     * Get progress (0.0 - 1.0).
     */
    float progress() const {
        if (total_estimated_ == 0) return 0.0f;
        return static_cast<float>(candidates_generated_) /
               static_cast<float>(total_estimated_);
    }

    /**
     * Get total candidates generated.
     */
    uint64_t candidates_generated() const { return candidates_generated_; }

    ~PassphraseGenerator() {
        stop();
    }

private:
    std::shared_ptr<CandidatePriorityQueue> queue_;
    std::shared_ptr<WeightedSourceManager> source_manager_;
    std::vector<std::unique_ptr<PassphraseSource>> sources_;

    std::thread generator_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> complete_{false};
    std::atomic<uint64_t> candidates_generated_{0};
    size_t total_estimated_ = 0;

    void run_generation() {
        // Calculate total estimate
        total_estimated_ = 0;
        for (const auto& source : sources_) {
            total_estimated_ += source->estimated_size();
        }

        // Process each source
        for (const auto& source : sources_) {
            if (!running_) break;

            source->generate([this, &source](Candidate&& candidate) {
                if (!running_) return;

                // Recalculate priority with source manager
                candidate.priority = source_manager_->calculate_priority(
                    candidate.phrase,
                    candidate.source,
                    candidate.priority  // Use as frequency hint
                );

                // Push to queue
                queue_->push(std::move(candidate));
                candidates_generated_++;
            });
        }

        complete_ = true;
        running_ = false;
    }
};

/**
 * Factory for creating standard source configurations.
 */
class SourceFactory {
public:
    /**
     * Create known brain wallet source (highest priority).
     */
    static std::unique_ptr<PassphraseSource> create_brain_wallet_source(
        const std::string& path
    ) {
        return std::make_unique<WordlistSource>(
            path,
            CandidateSource::KNOWN_BRAIN_WALLET,
            1.0f  // Maximum priority
        );
    }

    /**
     * Create password list with frequency weighting.
     */
    static std::unique_ptr<PassphraseSource> create_password_source(
        const std::string& path,
        bool has_frequencies = false
    ) {
        if (has_frequencies) {
            return std::make_unique<FrequencyWordlistSource>(path);
        }
        return std::make_unique<WordlistSource>(
            path,
            CandidateSource::PASSWORD_COMMON,
            0.5f
        );
    }

    /**
     * Create lyrics/quotes source with crypto rules.
     */
    static std::unique_ptr<PassphraseSource> create_lyrics_source(
        const std::string& path
    ) {
        auto base = std::make_unique<WordlistSource>(
            path,
            CandidateSource::LYRICS,
            0.3f
        );

        // Apply crypto-specific rules
        return std::make_unique<RuleExpandingSource>(
            std::move(base),
            builtin_rules::CRYPTO_RULES
        );
    }

    /**
     * Create combinator source for multi-word passphrases.
     */
    static std::unique_ptr<PassphraseSource> create_combinator_source(
        const std::vector<std::string>& wordlist_paths,
        size_t min_words = 2,
        size_t max_words = 4
    ) {
        return std::make_unique<CombinatorSource>(
            wordlist_paths,
            min_words,
            max_words
        );
    }
};

}  // namespace collider
