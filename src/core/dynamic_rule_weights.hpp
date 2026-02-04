/**
 * Collider Dynamic Rule Weighting
 *
 * Adaptive rule weighting system that learns from crack success rates.
 * Rules that produce more cracks get higher priority, enabling the system
 * to focus compute on the most effective transformations.
 */

#pragma once

#include "types.hpp"
#include "rule_engine.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>

namespace collider {

/**
 * Statistics for a single rule.
 */
struct RuleStats {
    std::string rule;
    uint64_t attempts;        // Number of times rule was applied
    uint64_t cracks;          // Number of successful cracks
    double weight;            // Current priority weight (0.0 - 1.0)
    double ema_crack_rate;    // Exponential moving average of crack rate
    std::chrono::steady_clock::time_point last_crack;  // Time of last crack

    RuleStats(const std::string& r = "")
        : rule(r), attempts(0), cracks(0), weight(0.5),
          ema_crack_rate(0.0), last_crack(std::chrono::steady_clock::now()) {}

    double crack_rate() const {
        return attempts > 0 ? static_cast<double>(cracks) / attempts : 0.0;
    }
};

/**
 * Dynamic Rule Weight Manager
 *
 * Tracks rule effectiveness and dynamically adjusts priorities.
 * Uses exponential moving average with time decay for adaptive learning.
 */
class DynamicRuleWeights {
public:
    struct Config {
        double alpha = 0.01;                    // EMA smoothing factor (lower = slower adaptation)
        double decay_rate = 0.001;              // Weight decay per hour without cracks
        double min_weight = 0.01;               // Minimum weight (prevent starvation)
        double max_weight = 1.0;                // Maximum weight
        size_t min_attempts_for_adaptation = 1000;  // Attempts before weights adapt
        double exploration_rate = 0.1;          // Fraction of capacity for low-weight rules
        bool enable_rule_combinations = true;   // Track rule pair effectiveness
    };

    explicit DynamicRuleWeights(const Config& config = {}) : config_(config) {}

    /**
     * Initialize with a set of rules.
     */
    void initialize(const std::vector<std::string>& rules) {
        std::lock_guard<std::mutex> lock(mutex_);

        for (const auto& rule : rules) {
            if (rule_stats_.find(rule) == rule_stats_.end()) {
                rule_stats_[rule] = RuleStats(rule);
            }
        }

        recalculate_weights();
    }

    /**
     * Record that a rule was attempted.
     */
    void record_attempt(const std::string& rule, size_t count = 1) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rule_stats_.find(rule);
        if (it != rule_stats_.end()) {
            it->second.attempts += count;
            total_attempts_ += count;
        }
    }

    /**
     * Record a successful crack using a rule.
     */
    void record_crack(const std::string& rule) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rule_stats_.find(rule);
        if (it != rule_stats_.end()) {
            it->second.cracks++;
            it->second.last_crack = std::chrono::steady_clock::now();
            total_cracks_++;

            // Update exponential moving average
            double instant_rate = it->second.attempts > 0 ?
                1.0 / it->second.attempts : 1.0;
            it->second.ema_crack_rate =
                config_.alpha * instant_rate +
                (1.0 - config_.alpha) * it->second.ema_crack_rate;

            // Trigger weight recalculation periodically
            if (total_cracks_ % 10 == 0) {
                recalculate_weights();
            }
        }
    }

    /**
     * Record a crack from a rule combination (base rule + mutator).
     */
    void record_combination_crack(
        const std::string& base_rule,
        const std::string& mutator_rule
    ) {
        if (!config_.enable_rule_combinations) return;

        std::lock_guard<std::mutex> lock(mutex_);

        std::string combo = base_rule + "+" + mutator_rule;
        combo_cracks_[combo]++;
    }

    /**
     * Get the current weight for a rule.
     */
    double get_weight(const std::string& rule) const {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = rule_stats_.find(rule);
        if (it != rule_stats_.end()) {
            return it->second.weight;
        }
        return config_.min_weight;
    }

    /**
     * Get rules sorted by current weight (highest first).
     */
    std::vector<std::pair<std::string, double>> get_weighted_rules() const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<std::pair<std::string, double>> result;
        result.reserve(rule_stats_.size());

        for (const auto& [rule, stats] : rule_stats_) {
            result.emplace_back(rule, stats.weight);
        }

        std::sort(result.begin(), result.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        return result;
    }

    /**
     * Get the top N rules by weight.
     */
    std::vector<std::string> get_top_rules(size_t n) const {
        auto weighted = get_weighted_rules();

        std::vector<std::string> result;
        result.reserve(std::min(n, weighted.size()));

        for (size_t i = 0; i < n && i < weighted.size(); ++i) {
            result.push_back(weighted[i].first);
        }

        return result;
    }

    /**
     * Sample a rule based on weights (weighted random selection).
     */
    std::string sample_rule() const {
        std::lock_guard<std::mutex> lock(mutex_);

        if (rule_stats_.empty()) return ":";

        // Build cumulative distribution
        std::vector<std::pair<std::string, double>> cdf;
        double sum = 0.0;

        for (const auto& [rule, stats] : rule_stats_) {
            sum += stats.weight;
            cdf.emplace_back(rule, sum);
        }

        // Sample
        double r = static_cast<double>(rand()) / RAND_MAX * sum;
        for (const auto& [rule, cum] : cdf) {
            if (r <= cum) return rule;
        }

        return cdf.back().first;
    }

    /**
     * Get exploration rules (low-weight rules that need more data).
     */
    std::vector<std::string> get_exploration_rules(size_t n) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<std::pair<std::string, uint64_t>> candidates;

        for (const auto& [rule, stats] : rule_stats_) {
            if (stats.attempts < config_.min_attempts_for_adaptation) {
                candidates.emplace_back(rule, stats.attempts);
            }
        }

        // Sort by fewest attempts
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        std::vector<std::string> result;
        for (size_t i = 0; i < n && i < candidates.size(); ++i) {
            result.push_back(candidates[i].first);
        }

        return result;
    }

    /**
     * Generate a batch of rules mixing exploitation and exploration.
     */
    std::vector<std::string> generate_rule_batch(size_t batch_size) const {
        std::vector<std::string> result;
        result.reserve(batch_size);

        size_t explore_count = static_cast<size_t>(batch_size * config_.exploration_rate);
        size_t exploit_count = batch_size - explore_count;

        // Exploitation: top weighted rules
        auto top = get_top_rules(exploit_count);
        result.insert(result.end(), top.begin(), top.end());

        // Exploration: under-tested rules
        auto explore = get_exploration_rules(explore_count);
        result.insert(result.end(), explore.begin(), explore.end());

        return result;
    }

    /**
     * Apply time decay to weights.
     * Call periodically (e.g., every minute) to decay unused rules.
     */
    void apply_decay() {
        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::steady_clock::now();

        for (auto& [rule, stats] : rule_stats_) {
            auto hours_since_crack = std::chrono::duration_cast<std::chrono::hours>(
                now - stats.last_crack).count();

            if (hours_since_crack > 0) {
                double decay = std::exp(-config_.decay_rate * hours_since_crack);
                stats.weight = std::max(config_.min_weight,
                                        stats.weight * decay);
            }
        }
    }

    /**
     * Get detailed statistics.
     */
    struct Statistics {
        uint64_t total_attempts;
        uint64_t total_cracks;
        double overall_crack_rate;
        std::vector<RuleStats> rule_stats;
        std::vector<std::pair<std::string, uint32_t>> top_combinations;
    };

    Statistics get_statistics() const {
        std::lock_guard<std::mutex> lock(mutex_);

        Statistics stats;
        stats.total_attempts = total_attempts_;
        stats.total_cracks = total_cracks_;
        stats.overall_crack_rate = total_attempts_ > 0 ?
            static_cast<double>(total_cracks_) / total_attempts_ : 0.0;

        for (const auto& [rule, rs] : rule_stats_) {
            stats.rule_stats.push_back(rs);
        }

        std::sort(stats.rule_stats.begin(), stats.rule_stats.end(),
            [](const auto& a, const auto& b) { return a.cracks > b.cracks; });

        for (const auto& [combo, count] : combo_cracks_) {
            stats.top_combinations.emplace_back(combo, count);
        }

        std::sort(stats.top_combinations.begin(), stats.top_combinations.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        if (stats.top_combinations.size() > 20) {
            stats.top_combinations.resize(20);
        }

        return stats;
    }

    /**
     * Save learned weights to file.
     */
    void save(const std::string& path) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ofstream f(path, std::ios::binary);
        if (!f) return;

        uint64_t count = rule_stats_.size();
        f.write(reinterpret_cast<const char*>(&count), sizeof(count));

        for (const auto& [rule, stats] : rule_stats_) {
            uint32_t len = rule.size();
            f.write(reinterpret_cast<const char*>(&len), sizeof(len));
            f.write(rule.data(), len);
            f.write(reinterpret_cast<const char*>(&stats.attempts), sizeof(stats.attempts));
            f.write(reinterpret_cast<const char*>(&stats.cracks), sizeof(stats.cracks));
            f.write(reinterpret_cast<const char*>(&stats.weight), sizeof(stats.weight));
            f.write(reinterpret_cast<const char*>(&stats.ema_crack_rate), sizeof(stats.ema_crack_rate));
        }
    }

    /**
     * Load learned weights from file.
     */
    void load(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ifstream f(path, std::ios::binary);
        if (!f) return;

        uint64_t count;
        f.read(reinterpret_cast<char*>(&count), sizeof(count));

        for (uint64_t i = 0; i < count && f.good(); ++i) {
            uint32_t len;
            f.read(reinterpret_cast<char*>(&len), sizeof(len));

            std::string rule(len, '\0');
            f.read(rule.data(), len);

            RuleStats stats(rule);
            f.read(reinterpret_cast<char*>(&stats.attempts), sizeof(stats.attempts));
            f.read(reinterpret_cast<char*>(&stats.cracks), sizeof(stats.cracks));
            f.read(reinterpret_cast<char*>(&stats.weight), sizeof(stats.weight));
            f.read(reinterpret_cast<char*>(&stats.ema_crack_rate), sizeof(stats.ema_crack_rate));

            if (f.good()) {
                rule_stats_[rule] = stats;
                total_attempts_ += stats.attempts;
                total_cracks_ += stats.cracks;
            }
        }
    }

private:
    Config config_;
    std::unordered_map<std::string, RuleStats> rule_stats_;
    std::unordered_map<std::string, uint32_t> combo_cracks_;  // Rule combinations
    uint64_t total_attempts_ = 0;
    uint64_t total_cracks_ = 0;
    mutable std::mutex mutex_;

    void recalculate_weights() {
        // Calculate weights based on EMA crack rate with Bayesian smoothing

        // Find max EMA for normalization
        double max_ema = 0.0;
        for (const auto& [rule, stats] : rule_stats_) {
            max_ema = std::max(max_ema, stats.ema_crack_rate);
        }

        if (max_ema == 0.0) {
            // No cracks yet, use uniform weights
            double uniform = 1.0 / rule_stats_.size();
            for (auto& [rule, stats] : rule_stats_) {
                stats.weight = uniform;
            }
            return;
        }

        // Calculate weights with Bayesian smoothing
        // Prior: assume 1 crack per 100000 attempts
        const double prior_cracks = 1.0;
        const double prior_attempts = 100000.0;

        double total_weight = 0.0;
        for (auto& [rule, stats] : rule_stats_) {
            // Bayesian posterior estimate
            double posterior_rate =
                (stats.cracks + prior_cracks) /
                (stats.attempts + prior_attempts);

            // Blend with EMA
            stats.weight = 0.7 * (stats.ema_crack_rate / max_ema) +
                          0.3 * (posterior_rate * 10000.0);

            stats.weight = std::clamp(stats.weight,
                                      config_.min_weight,
                                      config_.max_weight);
            total_weight += stats.weight;
        }

        // Normalize to sum to 1
        if (total_weight > 0) {
            for (auto& [rule, stats] : rule_stats_) {
                stats.weight /= total_weight;
            }
        }
    }
};

}  // namespace collider
