/**
 * Streaming Brain Wallet Generator
 *
 * Implements DEFCON Crack Me If You Can strategies for comprehensive
 * brain wallet passphrase generation. Unlike the batch generator,
 * this streams candidates infinitely in priority order:
 *
 * Phase 1: Quick Wins
 *   - Known brain wallet passwords
 *   - Top passwords + best64 rules
 *   - Crypto-specific wordlist + best64
 *
 * Phase 2: Medium Effort
 *   - Full wordlist + OneRuleToRuleThemStill
 *   - Lyrics/quotes + light rules
 *
 * Phase 3: Deep Search
 *   - PCFG-generated candidates
 *   - Combinator attacks (word + word)
 *   - Full mutation chains
 *
 * Phase 4: Infinite Exploration
 *   - Markov-based generation
 *   - Hybrid attacks
 *   - Loop back with new patterns from feedback
 */

#pragma once

#include <string>
#include <vector>
#include <queue>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <memory>
#include <functional>
#include <algorithm>
#include <random>
#include <chrono>
#include <atomic>
#include <mutex>
#include <iostream>
#include <thread>
#include <condition_variable>
#include <future>

namespace collider {
namespace generators {

/**
 * Hashcat-compatible Rule Engine
 *
 * Supports the most common hashcat rule functions:
 * : - No change (passthrough)
 * l - Lowercase all
 * u - Uppercase all
 * c - Capitalize first, lowercase rest
 * C - Lowercase first, uppercase rest
 * t - Toggle case of all characters
 * r - Reverse string
 * d - Duplicate string
 * f - Reflect (append reversed)
 * $ - Append character
 * ^ - Prepend character
 * s - Substitute character
 * @ - Purge character
 * [ - Delete first character
 * ] - Delete last character
 * { - Rotate left
 * } - Rotate right
 * And many more...
 */
class HashcatRuleEngine {
public:
    struct Rule {
        std::string raw;      // Original rule string
        std::string name;     // Optional name/comment
    };

    /**
     * Load rules from a hashcat-format rule file.
     */
    bool load_rules(const std::string& path) {
        std::ifstream file(path);
        if (!file) return false;

        rules_.clear();
        std::string line;
        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty()) continue;
            if (line[0] == '#') continue;

            // Trim whitespace
            size_t start = line.find_first_not_of(" \t");
            size_t end = line.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) continue;

            std::string rule = line.substr(start, end - start + 1);
            if (!rule.empty()) {
                rules_.push_back({rule, ""});
            }
        }

        return !rules_.empty();
    }

    /**
     * Add a rule directly.
     */
    void add_rule(const std::string& rule, const std::string& name = "") {
        rules_.push_back({rule, name});
    }

    /**
     * Apply a rule to a word.
     */
    std::string apply(const std::string& word, const Rule& rule) const {
        return apply_rule_string(word, rule.raw);
    }

    /**
     * Apply a rule string to a word.
     */
    std::string apply_rule_string(const std::string& word, const std::string& rule) const {
        std::string result = word;

        for (size_t i = 0; i < rule.size(); i++) {
            char cmd = rule[i];

            switch (cmd) {
                case ':': // No-op
                    break;

                case 'l': // Lowercase all
                    for (char& c : result) c = std::tolower(c);
                    break;

                case 'u': // Uppercase all
                    for (char& c : result) c = std::toupper(c);
                    break;

                case 'c': // Capitalize first
                    for (size_t j = 0; j < result.size(); j++) {
                        result[j] = (j == 0) ? std::toupper(result[j]) : std::tolower(result[j]);
                    }
                    break;

                case 'C': // Lowercase first, uppercase rest
                    for (size_t j = 0; j < result.size(); j++) {
                        result[j] = (j == 0) ? std::tolower(result[j]) : std::toupper(result[j]);
                    }
                    break;

                case 't': // Toggle case
                    for (char& c : result) {
                        c = std::isupper(c) ? std::tolower(c) : std::toupper(c);
                    }
                    break;

                case 'r': // Reverse
                    std::reverse(result.begin(), result.end());
                    break;

                case 'd': // Duplicate
                    result += result;
                    break;

                case 'f': // Reflect (append reversed)
                    {
                        std::string rev = result;
                        std::reverse(rev.begin(), rev.end());
                        result += rev;
                    }
                    break;

                case '$': // Append character
                    if (i + 1 < rule.size()) {
                        result += rule[++i];
                    }
                    break;

                case '^': // Prepend character
                    if (i + 1 < rule.size()) {
                        result = rule[++i] + result;
                    }
                    break;

                case 's': // Substitute char X with char Y
                    if (i + 2 < rule.size()) {
                        char from = rule[++i];
                        char to = rule[++i];
                        for (char& c : result) {
                            if (c == from) c = to;
                        }
                    }
                    break;

                case '@': // Purge all instances of char
                    if (i + 1 < rule.size()) {
                        char purge = rule[++i];
                        result.erase(std::remove(result.begin(), result.end(), purge), result.end());
                    }
                    break;

                case '[': // Delete first character
                    if (!result.empty()) result = result.substr(1);
                    break;

                case ']': // Delete last character
                    if (!result.empty()) result.pop_back();
                    break;

                case '{': // Rotate left
                    if (!result.empty()) {
                        char first = result[0];
                        result = result.substr(1) + first;
                    }
                    break;

                case '}': // Rotate right
                    if (!result.empty()) {
                        char last = result.back();
                        result = last + result.substr(0, result.size() - 1);
                    }
                    break;

                case 'T': // Toggle case at position N
                    if (i + 1 < rule.size()) {
                        size_t pos = rule[++i] - '0';
                        if (pos < result.size()) {
                            result[pos] = std::isupper(result[pos]) ?
                                std::tolower(result[pos]) : std::toupper(result[pos]);
                        }
                    }
                    break;

                case 'p': // Duplicate N times
                    if (i + 1 < rule.size()) {
                        int times = rule[++i] - '0';
                        std::string original = result;
                        for (int t = 0; t < times; t++) {
                            result += original;
                        }
                    }
                    break;

                case 'q': // Duplicate all characters
                    {
                        std::string doubled;
                        for (char c : result) {
                            doubled += c;
                            doubled += c;
                        }
                        result = doubled;
                    }
                    break;

                case 'k': // Swap first two characters
                    if (result.size() >= 2) {
                        std::swap(result[0], result[1]);
                    }
                    break;

                case 'K': // Swap last two characters
                    if (result.size() >= 2) {
                        std::swap(result[result.size()-2], result[result.size()-1]);
                    }
                    break;

                case '+': // Increment character at position N
                    if (i + 1 < rule.size()) {
                        size_t pos = rule[++i] - '0';
                        if (pos < result.size()) {
                            result[pos]++;
                        }
                    }
                    break;

                case '-': // Decrement character at position N
                    if (i + 1 < rule.size()) {
                        size_t pos = rule[++i] - '0';
                        if (pos < result.size()) {
                            result[pos]--;
                        }
                    }
                    break;

                case ' ': // Space - skip
                    break;

                default:
                    // Unknown rule character, skip
                    break;
            }
        }

        return result;
    }

    /**
     * Get all loaded rules.
     */
    const std::vector<Rule>& rules() const { return rules_; }

    /**
     * Get rule count.
     */
    size_t size() const { return rules_.size(); }

private:
    std::vector<Rule> rules_;
};

/**
 * Attack Phase Configuration
 */
struct AttackPhase {
    std::string name;
    std::string description;
    std::vector<std::string> wordlist_paths;
    std::vector<std::string> rule_paths;
    bool use_combinator = false;
    size_t combinator_min_words = 2;
    size_t combinator_max_words = 2;
    bool use_pcfg = false;
    int priority = 0;  // Lower = earlier
};

/**
 * Streaming Brain Wallet Generator
 *
 * Generates passphrases infinitely in priority order.
 * Never exhausts - cycles through phases with increasing complexity.
 */
class StreamingBrainWallet {
public:
    struct Config {
        std::string base_wordlist;          // Primary wordlist path
        std::string rules_dir;              // Directory containing rule files
        std::string pcfg_model_path;        // PCFG model file (optional)
        size_t batch_size = 1'000'000;      // Candidates per batch (1M for GPU saturation)
        bool enable_feedback = true;        // Learn from discovered patterns
        bool enable_dedup = false;          // Global dedup - DISABLED by default (major bottleneck)
        bool verbose = false;               // Debug output
    };

    explicit StreamingBrainWallet(const Config& config)
        : config_(config)
        , rng_(std::chrono::steady_clock::now().time_since_epoch().count())
    {
        // Initialize rule engine
        rule_engine_ = std::make_unique<HashcatRuleEngine>();
    }

    /**
     * Initialize the generator.
     * Must be called before generating candidates.
     */
    bool init() {
        // Load primary wordlist
        if (!load_wordlist(config_.base_wordlist)) {
            std::cerr << "[!] Failed to load wordlist: " << config_.base_wordlist << "\n";
            return false;
        }

        // Set up attack phases based on DEFCON strategies
        setup_attack_phases();

        // Load rules for first phase
        if (!phases_.empty() && !phases_[0].rule_paths.empty()) {
            for (const auto& rule_path : phases_[0].rule_paths) {
                if (rule_engine_->load_rules(rule_path)) {
                    if (config_.verbose) {
                        std::cout << "[*] Loaded " << rule_engine_->size()
                                  << " rules from " << rule_path << "\n";
                    }
                }
            }
        }

        // If no rules loaded, add default rules
        if (rule_engine_->size() == 0) {
            add_default_rules();
        }

        current_phase_ = 0;
        current_word_idx_ = 0;
        current_rule_idx_ = 0;
        phase_iteration_ = 0;
        total_generated_ = 0;

        initialized_ = true;
        return true;
    }

    /**
     * Generate next batch of candidate passphrases.
     * Returns empty vector only if completely exhausted (should not happen).
     */
    std::vector<std::string> next_batch() {
        if (!initialized_) return {};

        std::vector<std::string> batch;
        batch.reserve(config_.batch_size);

        size_t skipped_duplicates = 0;
        size_t max_attempts = config_.batch_size * 10;  // Prevent infinite loop
        size_t attempts = 0;

        while (batch.size() < config_.batch_size && attempts < max_attempts) {
            attempts++;
            std::string candidate = next_candidate();
            if (candidate.empty()) {
                // Advance to next phase or restart
                if (!advance_phase()) {
                    // All phases complete, restart from beginning with higher iteration
                    restart_with_next_iteration();
                }
                continue;
            }

            // Skip invalid UTF-8 candidates (encoding corruption)
            if (!is_valid_utf8(candidate)) {
                continue;
            }

            // Skip duplicates GLOBALLY (if enabled)
            // Note: Disabled by default because global dedup with 100M+ entries is a major bottleneck
            if (config_.enable_dedup) {
                std::lock_guard<std::mutex> lock(seen_mutex_);
                if (!seen_global_.insert(candidate).second) {
                    skipped_duplicates++;
                    continue;  // Already tested this passphrase
                }
            }

            batch.push_back(candidate);
        }

        total_generated_ += batch.size();
        return batch;
    }

    /**
     * Check if string is valid UTF-8.
     * Filters out encoding-corrupted passphrases.
     */
    static bool is_valid_utf8(const std::string& s) {
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(s.data());
        size_t len = s.size();
        size_t i = 0;

        while (i < len) {
            if (bytes[i] <= 0x7F) {
                // ASCII - valid
                i++;
            } else if ((bytes[i] & 0xE0) == 0xC0) {
                // 2-byte UTF-8
                if (i + 1 >= len || (bytes[i+1] & 0xC0) != 0x80) return false;
                i += 2;
            } else if ((bytes[i] & 0xF0) == 0xE0) {
                // 3-byte UTF-8
                if (i + 2 >= len || (bytes[i+1] & 0xC0) != 0x80 || (bytes[i+2] & 0xC0) != 0x80) return false;
                i += 3;
            } else if ((bytes[i] & 0xF8) == 0xF0) {
                // 4-byte UTF-8
                if (i + 3 >= len || (bytes[i+1] & 0xC0) != 0x80 || (bytes[i+2] & 0xC0) != 0x80 || (bytes[i+3] & 0xC0) != 0x80) return false;
                i += 4;
            } else {
                // Invalid UTF-8 start byte
                return false;
            }
        }
        return true;
    }

    /**
     * Get global dedup stats.
     */
    size_t unique_passphrases_tested() const {
        std::lock_guard<std::mutex> lock(seen_mutex_);
        return seen_global_.size();
    }

    /**
     * Clear global dedup cache (use sparingly - only for true restart).
     */
    void clear_dedup_cache() {
        std::lock_guard<std::mutex> lock(seen_mutex_);
        seen_global_.clear();
    }

    /**
     * Report a successful hit for feedback learning.
     */
    void report_hit(const std::string& passphrase) {
        if (!config_.enable_feedback) return;

        // Extract patterns from successful hit
        std::lock_guard<std::mutex> lock(feedback_mutex_);

        // Add base word variations to high-priority queue
        std::string lower = passphrase;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        // Remove common suffixes to find base word
        for (const auto& suffix : {"123", "1234", "12345", "!", "!!", "1", "2", "01", "2024", "2023"}) {
            if (lower.size() > strlen(suffix) &&
                lower.substr(lower.size() - strlen(suffix)) == suffix) {
                std::string base = lower.substr(0, lower.size() - strlen(suffix));
                high_priority_words_.push_back(base);
            }
        }

        // Add the passphrase itself for variation generation
        high_priority_words_.push_back(lower);

        hits_found_++;
    }

    /**
     * Get current statistics.
     */
    struct Stats {
        uint64_t total_generated;
        uint64_t unique_tested;  // Unique passphrases (after dedup)
        uint64_t hits_found;
        int current_phase;
        std::string current_phase_name;
        size_t wordlist_size;
        size_t rules_loaded;
        size_t phase_iteration;
    };

    Stats get_stats() const {
        Stats s;
        s.total_generated = total_generated_;
        s.unique_tested = unique_passphrases_tested();
        s.hits_found = hits_found_;
        s.current_phase = current_phase_;
        s.current_phase_name = (current_phase_ < phases_.size()) ?
            phases_[current_phase_].name : "Unknown";
        s.wordlist_size = wordlist_.size();
        s.rules_loaded = rule_engine_->size();
        s.phase_iteration = phase_iteration_;
        return s;
    }

    // =========================================================================
    // STATE PERSISTENCE (for checkpoint/resume)
    // =========================================================================

    /**
     * Snapshot of generator state for save/restore.
     */
    struct StateSnapshot {
        size_t current_phase = 0;
        size_t current_word_idx = 0;
        size_t current_rule_idx = 0;
        size_t phase_iteration = 0;
        uint64_t total_generated = 0;
        size_t wordlist_size = 0;
        size_t rules_count = 0;
    };

    /**
     * Get current state snapshot for persistence.
     */
    StateSnapshot get_state_snapshot() const {
        StateSnapshot snap;
        snap.current_phase = current_phase_;
        snap.current_word_idx = current_word_idx_;
        snap.current_rule_idx = current_rule_idx_;
        snap.phase_iteration = phase_iteration_;
        snap.total_generated = total_generated_;
        snap.wordlist_size = wordlist_.size();
        snap.rules_count = rule_engine_ ? rule_engine_->size() : 0;
        return snap;
    }

    /**
     * Restore state from snapshot after init().
     * Must be called AFTER init() to have wordlist and rules loaded.
     *
     * @param snap The state snapshot to restore
     * @return true if restore successful, false if state invalid
     */
    bool restore_state(const StateSnapshot& snap) {
        if (!initialized_) {
            std::cerr << "[!] Cannot restore state before init()\n";
            return false;
        }

        // Validate wordlist size matches
        if (snap.wordlist_size > 0 && snap.wordlist_size != wordlist_.size()) {
            std::cerr << "[!] Wordlist size mismatch: saved " << snap.wordlist_size
                      << ", current " << wordlist_.size() << "\n";
            return false;
        }

        // Validate phase
        if (snap.current_phase >= phases_.size()) {
            std::cerr << "[!] Invalid phase index: " << snap.current_phase
                      << " (max " << (phases_.size() - 1) << ")\n";
            return false;
        }

        // Validate word index
        if (snap.current_word_idx > wordlist_.size()) {
            std::cerr << "[!] Invalid word index: " << snap.current_word_idx
                      << " (max " << wordlist_.size() << ")\n";
            return false;
        }

        // Restore state
        current_phase_ = snap.current_phase;
        current_word_idx_ = snap.current_word_idx;
        current_rule_idx_ = snap.current_rule_idx;
        phase_iteration_ = snap.phase_iteration;
        total_generated_ = snap.total_generated;

        // Load rules for the current phase
        if (current_phase_ < phases_.size()) {
            const auto& phase = phases_[current_phase_];
            if (!phase.rule_paths.empty()) {
                rule_engine_ = std::make_unique<HashcatRuleEngine>();
                for (const auto& path : phase.rule_paths) {
                    rule_engine_->load_rules(path);
                }
                if (rule_engine_->size() == 0) {
                    add_default_rules();
                }
            }
        }

        // Validate rule index after loading rules for phase
        if (rule_engine_ && snap.current_rule_idx > rule_engine_->size()) {
            std::cerr << "[!] Warning: Rule index adjusted from " << snap.current_rule_idx
                      << " to 0 (rules changed)\n";
            current_rule_idx_ = 0;
        }

        if (config_.verbose) {
            std::cout << "[*] State restored:\n";
            std::cout << "    Phase: " << current_phase_ << " (" << phases_[current_phase_].name << ")\n";
            std::cout << "    Word index: " << current_word_idx_ << " / " << wordlist_.size() << "\n";
            std::cout << "    Rule index: " << current_rule_idx_ << " / "
                      << (rule_engine_ ? rule_engine_->size() : 0) << "\n";
            std::cout << "    Total generated: " << total_generated_ << "\n";
        }

        return true;
    }

    /**
     * Get wordlist path (for state persistence).
     */
    const std::string& get_wordlist_path() const {
        return config_.base_wordlist;
    }

    /**
     * Get rules directory path (for state persistence).
     */
    const std::string& get_rules_dir() const {
        return config_.rules_dir;
    }

    /**
     * Set total generated count (for resume from checkpoint).
     */
    void set_total_generated(uint64_t count) {
        total_generated_ = count;
    }

    /**
     * Set hits found count (for resume from checkpoint).
     */
    void set_hits_found(uint64_t count) {
        hits_found_ = count;
    }

private:
    Config config_;
    std::unique_ptr<HashcatRuleEngine> rule_engine_;
    std::vector<std::string> wordlist_;
    std::vector<AttackPhase> phases_;

    // Generator state
    bool initialized_ = false;
    std::atomic<size_t> current_phase_{0};
    size_t current_word_idx_ = 0;
    size_t current_rule_idx_ = 0;
    std::atomic<size_t> phase_iteration_{0};
    std::atomic<uint64_t> total_generated_{0};

    // Deduplication - GLOBAL to prevent testing same passphrase multiple times
    std::unordered_set<std::string> seen_global_;
    mutable std::mutex seen_mutex_;

    // Deduplication within batch (cleared each batch for speed)
    std::unordered_set<std::string> seen_in_batch_;

    // Feedback system
    std::mutex feedback_mutex_;
    std::vector<std::string> high_priority_words_;
    std::atomic<uint64_t> hits_found_{0};

    // For Markov/random generation
    std::mt19937 rng_;

    // ==========================================================================
    // ASYNC PIPELINE - Double-buffered batch generation
    // ==========================================================================
    std::vector<std::string> prefetch_buffer_;
    std::mutex prefetch_mutex_;
    std::condition_variable prefetch_cv_;
    std::thread prefetch_thread_;
    std::atomic<bool> prefetch_running_{false};
    std::atomic<bool> prefetch_ready_{false};
    std::atomic<bool> shutdown_prefetch_{false};

public:
    /**
     * Start async prefetch thread.
     * Call after init() to enable double-buffered batch generation.
     */
    void start_async_prefetch() {
        if (prefetch_running_) return;

        shutdown_prefetch_ = false;
        prefetch_running_ = true;
        prefetch_thread_ = std::thread([this]() {
            while (!shutdown_prefetch_) {
                // Generate next batch into prefetch buffer
                auto batch = generate_batch_internal();

                {
                    std::unique_lock<std::mutex> lock(prefetch_mutex_);
                    prefetch_buffer_ = std::move(batch);
                    prefetch_ready_ = true;
                }
                prefetch_cv_.notify_one();

                // Wait until the batch is consumed
                {
                    std::unique_lock<std::mutex> lock(prefetch_mutex_);
                    prefetch_cv_.wait(lock, [this]() {
                        return !prefetch_ready_ || shutdown_prefetch_;
                    });
                }
            }
        });
    }

    /**
     * Stop async prefetch thread.
     */
    void stop_async_prefetch() {
        if (!prefetch_running_) return;

        shutdown_prefetch_ = true;
        prefetch_cv_.notify_all();

        if (prefetch_thread_.joinable()) {
            prefetch_thread_.join();
        }
        prefetch_running_ = false;
    }

    /**
     * Get next batch - uses prefetched batch if available.
     * Returns immediately with prefetched data while triggering next prefetch.
     */
    std::vector<std::string> next_batch_async() {
        if (!prefetch_running_) {
            // Fall back to synchronous generation
            return next_batch();
        }

        std::vector<std::string> result;

        {
            std::unique_lock<std::mutex> lock(prefetch_mutex_);
            // Wait for prefetch to be ready
            prefetch_cv_.wait(lock, [this]() {
                return prefetch_ready_ || shutdown_prefetch_;
            });

            if (shutdown_prefetch_) return {};

            // Swap buffers (O(1) operation)
            result = std::move(prefetch_buffer_);
            prefetch_ready_ = false;
        }

        // Signal prefetch thread to generate next batch
        prefetch_cv_.notify_one();

        return result;
    }

    ~StreamingBrainWallet() {
        stop_async_prefetch();
    }

    // ==========================================================================
    // RAW WORD MODE - For use with GPU rule engine
    // ==========================================================================

    /**
     * Get next batch of raw words (without any rule application).
     * Used with GPU rule engine which applies rules on GPU.
     *
     * @param batch_size Number of words to return
     * @return Vector of raw words from wordlist
     */
    std::vector<std::string> next_raw_words(size_t batch_size) {
        std::vector<std::string> words;
        words.reserve(batch_size);

        // Handle high-priority feedback words first
        if (config_.enable_feedback && !high_priority_words_.empty()) {
            std::lock_guard<std::mutex> lock(feedback_mutex_);
            while (!high_priority_words_.empty() && words.size() < batch_size) {
                words.push_back(high_priority_words_.back());
                high_priority_words_.pop_back();
            }
        }

        // Fill rest from wordlist
        while (words.size() < batch_size && raw_word_idx_ < wordlist_.size()) {
            const std::string& word = wordlist_[raw_word_idx_++];
            if (is_valid_utf8(word)) {
                words.push_back(word);
            }
        }

        // Check if we've exhausted the wordlist - advance phase
        if (raw_word_idx_ >= wordlist_.size()) {
            raw_word_idx_ = 0;

            // Advance to next phase (or restart cycle)
            if (!advance_phase()) {
                restart_with_next_iteration();
            }
            phase_changed_ = true;  // Signal to reload GPU rules
        }

        return words;
    }

    /**
     * Check if phase changed since last call (for GPU rule reload).
     * Resets the flag after reading.
     */
    bool phase_changed() {
        bool changed = phase_changed_;
        phase_changed_ = false;
        return changed;
    }

    /**
     * Get current phase name.
     */
    std::string current_phase_name() const {
        if (current_phase_ < phases_.size()) {
            return phases_[current_phase_].name;
        }
        return "Unknown";
    }

    /**
     * Reset raw word position to beginning.
     */
    void reset_raw_words() {
        raw_word_idx_ = 0;
        raw_word_iteration_ = 0;
    }

    /**
     * Get number of words in wordlist (for calculating batch sizes).
     */
    size_t wordlist_size() const { return wordlist_.size(); }

    /**
     * Get number of rules loaded.
     */
    size_t num_rules() const { return rule_engine_ ? rule_engine_->size() : 0; }

    /**
     * Get all loaded rules (for GPU rule engine).
     */
    std::vector<std::string> get_rules() const {
        std::vector<std::string> rules;
        if (rule_engine_) {
            for (const auto& rule : rule_engine_->rules()) {
                rules.push_back(rule.raw);
            }
        }
        return rules;
    }

private:
    // Raw word generator state (for GPU rules mode)
    size_t raw_word_idx_ = 0;
    size_t raw_word_iteration_ = 0;
    bool phase_changed_ = false;  // Signals GPU rules need reload


    /**
     * Internal batch generation (used by both sync and async paths).
     */
    std::vector<std::string> generate_batch_internal() {
        std::vector<std::string> batch;
        batch.reserve(config_.batch_size);

        size_t max_attempts = config_.batch_size * 10;
        size_t attempts = 0;

        while (batch.size() < config_.batch_size && attempts < max_attempts) {
            attempts++;
            std::string candidate = next_candidate();
            if (candidate.empty()) {
                if (!advance_phase()) {
                    restart_with_next_iteration();
                }
                continue;
            }

            if (!is_valid_utf8(candidate)) {
                continue;
            }

            // Skip duplicates (if enabled)
            // Note: Disabled by default because global dedup with 100M+ entries is a major bottleneck
            if (config_.enable_dedup) {
                std::lock_guard<std::mutex> lock(seen_mutex_);
                if (!seen_global_.insert(candidate).second) {
                    continue;
                }
            }

            batch.push_back(std::move(candidate));
        }

        total_generated_ += batch.size();
        return batch;
    }

    /**
     * Load a wordlist from file.
     */
    bool load_wordlist(const std::string& path) {
        std::ifstream file(path);
        if (!file) return false;

        wordlist_.clear();
        std::string line;
        while (std::getline(file, line)) {
            // Trim
            size_t start = line.find_first_not_of(" \t");
            size_t end = line.find_last_not_of(" \t\r\n");
            if (start == std::string::npos) continue;

            line = line.substr(start, end - start + 1);
            if (!line.empty() && line[0] != '#') {
                wordlist_.push_back(line);
            }
        }

        if (config_.verbose) {
            std::cout << "[*] Loaded " << wordlist_.size() << " words\n";
        }

        return !wordlist_.empty();
    }

    /**
     * Set up attack phases based on DEFCON strategies.
     */
    void setup_attack_phases() {
        phases_.clear();

        // Phase 1: Quick Wins - best64 rules only
        AttackPhase phase1;
        phase1.name = "Quick Wins";
        phase1.description = "Top words + best64 rules";
        phase1.rule_paths = {config_.rules_dir + "/best64.rule"};
        phase1.priority = 1;
        phases_.push_back(phase1);

        // Phase 2: Crypto-specific rules
        AttackPhase phase2;
        phase2.name = "Crypto Focus";
        phase2.description = "Wordlist + crypto-specific rules";
        phase2.rule_paths = {config_.rules_dir + "/crypto.rule"};
        phase2.priority = 2;
        phases_.push_back(phase2);

        // Phase 3: Extended rules (d3ad0ne)
        AttackPhase phase3;
        phase3.name = "Extended Rules";
        phase3.description = "Wordlist + extended rule set";
        phase3.rule_paths = {config_.rules_dir + "/d3ad0ne.rule"};
        phase3.priority = 3;
        phases_.push_back(phase3);

        // Phase 4: Combinator (word + word)
        AttackPhase phase4;
        phase4.name = "Combinator";
        phase4.description = "Two-word combinations";
        phase4.use_combinator = true;
        phase4.combinator_min_words = 2;
        phase4.combinator_max_words = 2;
        phase4.priority = 4;
        phases_.push_back(phase4);

        // Phase 5: Deep dive rules
        AttackPhase phase5;
        phase5.name = "Deep Dive";
        phase5.description = "Full dive.rule set";
        phase5.rule_paths = {config_.rules_dir + "/dive.rule"};
        phase5.priority = 5;
        phases_.push_back(phase5);
    }

    /**
     * Add default rules if no rule files found.
     */
    void add_default_rules() {
        // Best64 equivalent
        rule_engine_->add_rule(":", "passthrough");
        rule_engine_->add_rule("l", "lowercase");
        rule_engine_->add_rule("u", "uppercase");
        rule_engine_->add_rule("c", "capitalize");
        rule_engine_->add_rule("t", "toggle");
        rule_engine_->add_rule("r", "reverse");
        rule_engine_->add_rule("d", "duplicate");
        rule_engine_->add_rule("$1", "append 1");
        rule_engine_->add_rule("$!", "append !");
        rule_engine_->add_rule("$1$2$3", "append 123");
        rule_engine_->add_rule("c$1", "cap + 1");
        rule_engine_->add_rule("c$!", "cap + !");
        rule_engine_->add_rule("c$1$2$3", "cap + 123");
        rule_engine_->add_rule("sa4", "a->4");
        rule_engine_->add_rule("se3", "e->3");
        rule_engine_->add_rule("si1", "i->1");
        rule_engine_->add_rule("so0", "o->0");
        rule_engine_->add_rule("ss$", "s->$");

        // Crypto-specific
        rule_engine_->add_rule("$b$t$c", "append btc");
        rule_engine_->add_rule("$B$T$C", "append BTC");
        rule_engine_->add_rule("$2$0$0$9", "append 2009");
        rule_engine_->add_rule("$2$0$1$3", "append 2013");
        rule_engine_->add_rule("$2$0$1$7", "append 2017");
        rule_engine_->add_rule("$2$0$2$1", "append 2021");
        rule_engine_->add_rule("c$2$0$0$9", "cap + 2009");
        rule_engine_->add_rule("c$2$0$1$7", "cap + 2017");

        if (config_.verbose) {
            std::cout << "[*] Added " << rule_engine_->size() << " default rules\n";
        }
    }

    /**
     * Generate the next candidate passphrase.
     */
    std::string next_candidate() {
        if (current_phase_ >= phases_.size()) return "";

        const auto& phase = phases_[current_phase_];

        // Check for high-priority feedback words first
        if (!high_priority_words_.empty() && (total_generated_ % 100 == 0)) {
            std::lock_guard<std::mutex> lock(feedback_mutex_);
            if (!high_priority_words_.empty()) {
                std::string word = high_priority_words_.back();
                high_priority_words_.pop_back();

                // Apply current rule
                if (current_rule_idx_ < rule_engine_->size()) {
                    return rule_engine_->apply(word, rule_engine_->rules()[current_rule_idx_]);
                }
                return word;
            }
        }

        // Combinator phase
        if (phase.use_combinator) {
            return generate_combination();
        }

        // Standard rule-based generation
        if (current_word_idx_ >= wordlist_.size()) {
            return "";  // Phase exhausted
        }

        const std::string& word = wordlist_[current_word_idx_];
        std::string candidate;

        if (rule_engine_->size() > 0 && current_rule_idx_ < rule_engine_->size()) {
            candidate = rule_engine_->apply(word, rule_engine_->rules()[current_rule_idx_]);

            // Advance rule index
            current_rule_idx_++;
            if (current_rule_idx_ >= rule_engine_->size()) {
                current_rule_idx_ = 0;
                current_word_idx_++;
            }
        } else {
            // No rules, just use word directly
            candidate = word;
            current_word_idx_++;
        }

        return candidate;
    }

    /**
     * Generate a word combination (combinator attack).
     */
    std::string generate_combination() {
        // Simple two-word combinator
        // Uses current indices to iterate through word pairs
        size_t word1_idx = current_word_idx_ / wordlist_.size();
        size_t word2_idx = current_word_idx_ % wordlist_.size();

        if (word1_idx >= wordlist_.size()) {
            return "";  // Exhausted all combinations
        }

        current_word_idx_++;

        const std::string& word1 = wordlist_[word1_idx % wordlist_.size()];
        const std::string& word2 = wordlist_[word2_idx];

        // Combine with different separators based on rule index
        switch (current_rule_idx_ % 8) {
            case 0: return word1 + word2;
            case 1: return word1 + " " + word2;
            case 2: return word1 + "_" + word2;
            case 3: return word1 + "-" + word2;
            case 4: return word1 + "1" + word2;
            case 5: return word1 + "123" + word2;
            case 6: {
                std::string cap1 = word1, cap2 = word2;
                if (!cap1.empty()) cap1[0] = std::toupper(cap1[0]);
                if (!cap2.empty()) cap2[0] = std::toupper(cap2[0]);
                return cap1 + cap2;
            }
            default: return word1 + word2 + "!";
        }
    }

    /**
     * Advance to the next attack phase.
     */
    bool advance_phase() {
        current_phase_++;
        current_word_idx_ = 0;
        current_rule_idx_ = 0;

        if (current_phase_ >= phases_.size()) {
            return false;  // All phases complete
        }

        // Load rules for new phase
        const auto& phase = phases_[current_phase_];
        if (!phase.rule_paths.empty()) {
            rule_engine_ = std::make_unique<HashcatRuleEngine>();
            for (const auto& path : phase.rule_paths) {
                rule_engine_->load_rules(path);
            }

            if (rule_engine_->size() == 0) {
                add_default_rules();
            }
        }

        if (config_.verbose) {
            std::cout << "\n[*] Entering Phase " << (current_phase_ + 1)
                      << ": " << phase.name << "\n";
            std::cout << "    " << phase.description << "\n";
            std::cout << "    Rules loaded: " << rule_engine_->size() << "\n";
        }

        return true;
    }

    /**
     * Restart from phase 1 with next iteration level.
     */
    void restart_with_next_iteration() {
        phase_iteration_++;
        current_phase_ = 0;
        current_word_idx_ = 0;
        current_rule_idx_ = 0;

        // For subsequent iterations, we could:
        // - Use more complex rule combinations
        // - Apply multi-level mutations
        // - Generate Markov-based candidates

        if (config_.verbose) {
            std::cout << "\n[*] Starting iteration " << (phase_iteration_ + 1) << "\n";
        }

        // Reload rules for first phase
        if (!phases_.empty() && !phases_[0].rule_paths.empty()) {
            rule_engine_ = std::make_unique<HashcatRuleEngine>();
            for (const auto& path : phases_[0].rule_paths) {
                rule_engine_->load_rules(path);
            }
        }
    }
};

}  // namespace generators
}  // namespace collider
