/**
 * Collider Markov Chain Generator
 *
 * Character-level Markov chain for probability-ordered password generation.
 * Learns character transition probabilities from training corpus.
 */

#pragma once

#include "../core/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <array>
#include <functional>

namespace collider {
namespace markov {

// Character classes for efficient indexing (ASCII 0-127 + special)
constexpr size_t CHAR_CLASSES = 128;
constexpr char START_CHAR = '\x01';  // Start of string marker
constexpr char END_CHAR = '\x02';    // End of string marker

/**
 * Markov chain transition matrix.
 * Stores probability of each character following a given n-gram.
 */
class TransitionMatrix {
public:
    explicit TransitionMatrix(size_t order = 2) : order_(order) {}

    /**
     * Get probability of character following given context.
     */
    double get_probability(const std::string& context, char next) const {
        auto it = transitions_.find(context);
        if (it == transitions_.end()) return 0.0;

        size_t idx = static_cast<size_t>(static_cast<unsigned char>(next));
        if (idx >= CHAR_CLASSES) return 0.0;

        return it->second[idx];
    }

    /**
     * Set probability for a transition.
     */
    void set_probability(const std::string& context, char next, double prob) {
        size_t idx = static_cast<size_t>(static_cast<unsigned char>(next));
        if (idx >= CHAR_CLASSES) return;

        transitions_[context][idx] = prob;
    }

    /**
     * Get all transitions from a context, sorted by probability.
     */
    std::vector<std::pair<char, double>> get_sorted_transitions(const std::string& context) const {
        std::vector<std::pair<char, double>> result;

        auto it = transitions_.find(context);
        if (it == transitions_.end()) return result;

        for (size_t i = 0; i < CHAR_CLASSES; ++i) {
            if (it->second[i] > 0.0) {
                result.emplace_back(static_cast<char>(i), it->second[i]);
            }
        }

        std::sort(result.begin(), result.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        return result;
    }

    /**
     * Check if context exists in matrix.
     */
    bool has_context(const std::string& context) const {
        return transitions_.find(context) != transitions_.end();
    }

    size_t order() const { return order_; }
    size_t num_contexts() const { return transitions_.size(); }

    /**
     * Save matrix to file.
     */
    void save(const std::string& path) const {
        std::ofstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot create file: " + path);

        // Header
        file.write("MKVC", 4);  // Magic
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        uint32_t order = static_cast<uint32_t>(order_);
        file.write(reinterpret_cast<const char*>(&order), sizeof(order));
        uint32_t num_ctx = static_cast<uint32_t>(transitions_.size());
        file.write(reinterpret_cast<const char*>(&num_ctx), sizeof(num_ctx));

        // Transitions
        for (const auto& [context, probs] : transitions_) {
            uint32_t ctx_len = static_cast<uint32_t>(context.size());
            file.write(reinterpret_cast<const char*>(&ctx_len), sizeof(ctx_len));
            file.write(context.data(), ctx_len);

            // Count non-zero probabilities
            uint32_t num_probs = 0;
            for (size_t i = 0; i < CHAR_CLASSES; ++i) {
                if (probs[i] > 0.0) num_probs++;
            }
            file.write(reinterpret_cast<const char*>(&num_probs), sizeof(num_probs));

            // Write non-zero probabilities
            for (size_t i = 0; i < CHAR_CLASSES; ++i) {
                if (probs[i] > 0.0) {
                    uint8_t ch = static_cast<uint8_t>(i);
                    file.write(reinterpret_cast<const char*>(&ch), sizeof(ch));
                    float prob = static_cast<float>(probs[i]);
                    file.write(reinterpret_cast<const char*>(&prob), sizeof(prob));
                }
            }
        }
    }

    /**
     * Load matrix from file.
     */
    static TransitionMatrix load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file: " + path);

        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) != "MKVC") {
            throw std::runtime_error("Invalid Markov file format");
        }

        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));

        uint32_t order;
        file.read(reinterpret_cast<char*>(&order), sizeof(order));

        TransitionMatrix matrix(order);

        uint32_t num_ctx;
        file.read(reinterpret_cast<char*>(&num_ctx), sizeof(num_ctx));

        for (uint32_t i = 0; i < num_ctx; ++i) {
            uint32_t ctx_len;
            file.read(reinterpret_cast<char*>(&ctx_len), sizeof(ctx_len));
            std::string context(ctx_len, '\0');
            file.read(context.data(), ctx_len);

            uint32_t num_probs;
            file.read(reinterpret_cast<char*>(&num_probs), sizeof(num_probs));

            for (uint32_t j = 0; j < num_probs; ++j) {
                uint8_t ch;
                file.read(reinterpret_cast<char*>(&ch), sizeof(ch));
                float prob;
                file.read(reinterpret_cast<char*>(&prob), sizeof(prob));
                matrix.set_probability(context, static_cast<char>(ch), prob);
            }
        }

        return matrix;
    }

private:
    size_t order_;
    std::unordered_map<std::string, std::array<double, CHAR_CLASSES>> transitions_;
};

/**
 * Markov chain trainer.
 * Learns character transitions from password corpus.
 */
class Trainer {
public:
    struct Config {
        size_t order = 2;           // Context length (n-gram order)
        size_t min_length = 4;      // Minimum password length
        size_t max_length = 64;     // Maximum password length
        double min_probability = 1e-9;  // Minimum probability to keep
        bool use_smoothing = true;  // Laplace smoothing
        double smoothing_factor = 0.01;
    };

    explicit Trainer(const Config& config = {}) : config_(config) {}

    /**
     * Train on a password file.
     */
    void train(const std::string& password_file) {
        std::ifstream file(password_file);
        std::string line;

        while (std::getline(file, line)) {
            if (line.empty()) continue;
            if (line.size() < config_.min_length) continue;
            if (line.size() > config_.max_length) continue;

            train_password(line);
        }

        total_passwords_++;
    }

    /**
     * Train on multiple password files.
     */
    void train_multiple(const std::vector<std::string>& paths) {
        for (const auto& path : paths) {
            train(path);
        }
    }

    /**
     * Build the transition matrix.
     */
    TransitionMatrix build_matrix() const {
        TransitionMatrix matrix(config_.order);

        for (const auto& [context, counts] : transition_counts_) {
            // Calculate total count for this context
            uint64_t total = 0;
            for (size_t i = 0; i < CHAR_CLASSES; ++i) {
                total += counts[i];
            }

            if (total == 0) continue;

            // Apply smoothing if enabled
            double denominator = static_cast<double>(total);
            if (config_.use_smoothing) {
                denominator += config_.smoothing_factor * CHAR_CLASSES;
            }

            // Calculate probabilities
            for (size_t i = 0; i < CHAR_CLASSES; ++i) {
                double count = static_cast<double>(counts[i]);
                if (config_.use_smoothing) {
                    count += config_.smoothing_factor;
                }

                double prob = count / denominator;
                if (prob >= config_.min_probability) {
                    matrix.set_probability(context, static_cast<char>(i), prob);
                }
            }
        }

        return matrix;
    }

    /**
     * Get training statistics.
     */
    struct Stats {
        uint64_t total_passwords;
        size_t unique_contexts;
        uint64_t total_transitions;
    };

    Stats get_stats() const {
        Stats s;
        s.total_passwords = total_passwords_;
        s.unique_contexts = transition_counts_.size();
        s.total_transitions = 0;
        for (const auto& [ctx, counts] : transition_counts_) {
            for (size_t i = 0; i < CHAR_CLASSES; ++i) {
                if (counts[i] > 0) s.total_transitions++;
            }
        }
        return s;
    }

private:
    Config config_;
    std::unordered_map<std::string, std::array<uint64_t, CHAR_CLASSES>> transition_counts_;
    uint64_t total_passwords_ = 0;

    void train_password(const std::string& password) {
        // Add start markers
        std::string padded;
        padded.reserve(password.size() + config_.order + 1);
        for (size_t i = 0; i < config_.order; ++i) {
            padded += START_CHAR;
        }
        padded += password;
        padded += END_CHAR;

        // Record transitions
        for (size_t i = config_.order; i < padded.size(); ++i) {
            std::string context = padded.substr(i - config_.order, config_.order);
            char next = padded[i];

            size_t idx = static_cast<size_t>(static_cast<unsigned char>(next));
            if (idx < CHAR_CLASSES) {
                transition_counts_[context][idx]++;
            }
        }

        total_passwords_++;
    }
};

/**
 * Generator state for probability-ordered enumeration.
 */
struct GeneratorState {
    std::string prefix;      // Current password prefix
    double probability;      // Cumulative probability
    size_t depth;            // Current position

    bool operator<(const GeneratorState& other) const {
        return probability < other.probability;  // Max-heap
    }
};

/**
 * Markov chain generator.
 * Produces candidates in (approximate) probability order.
 */
class Generator {
public:
    explicit Generator(const TransitionMatrix& matrix,
                       size_t min_length = 4,
                       size_t max_length = 32)
        : matrix_(matrix),
          min_length_(min_length),
          max_length_(max_length) {}

    /**
     * Generate next candidate.
     */
    std::optional<Candidate> next() {
        if (!initialized_) {
            initialize();
        }

        while (!state_queue_.empty()) {
            GeneratorState state = state_queue_.top();
            state_queue_.pop();

            // Get context for next character
            std::string context;
            if (state.prefix.size() < matrix_.order()) {
                // Pad with start markers
                context = std::string(matrix_.order() - state.prefix.size(), START_CHAR);
                context += state.prefix;
            } else {
                context = state.prefix.substr(state.prefix.size() - matrix_.order());
            }

            // Get possible next characters
            auto transitions = matrix_.get_sorted_transitions(context);

            for (const auto& [ch, prob] : transitions) {
                double new_prob = state.probability * prob;

                // Prune low probability branches
                if (new_prob < min_probability_) continue;

                if (ch == END_CHAR) {
                    // Complete password
                    if (state.prefix.size() >= min_length_) {
                        return Candidate{
                            .phrase = state.prefix,
                            .priority = static_cast<float>(new_prob),
                            .source = CandidateSource::MARKOV,
                            .rule_applied = ":"
                        };
                    }
                } else if (state.prefix.size() < max_length_) {
                    // Continue building
                    GeneratorState new_state{
                        state.prefix + ch,
                        new_prob,
                        state.depth + 1
                    };
                    state_queue_.push(new_state);
                }
            }
        }

        return std::nullopt;
    }

    /**
     * Reset generator to start.
     */
    void reset() {
        initialized_ = false;
        state_queue_ = {};
    }

    /**
     * Set minimum probability threshold.
     */
    void set_min_probability(double min_prob) {
        min_probability_ = min_prob;
    }

private:
    const TransitionMatrix& matrix_;
    size_t min_length_;
    size_t max_length_;
    double min_probability_ = 1e-20;

    std::priority_queue<GeneratorState> state_queue_;
    bool initialized_ = false;

    void initialize() {
        // Start with empty prefix
        state_queue_.push(GeneratorState{"", 1.0, 0});
        initialized_ = true;
    }
};

/**
 * PassphraseSource implementation for Markov generator.
 */
class MarkovSource : public PassphraseSource {
public:
    explicit MarkovSource(const TransitionMatrix& matrix,
                          size_t min_length = 4,
                          size_t max_length = 32,
                          size_t max_candidates = 1000000)
        : matrix_(matrix),
          generator_(matrix_, min_length, max_length),
          max_candidates_(max_candidates) {}

    void generate(CandidateCallback callback) override {
        size_t generated = 0;
        while (generated < max_candidates_) {
            auto candidate = generator_.next();
            if (!candidate) break;

            callback(std::move(*candidate));
            generated++;
        }
    }

    CandidateSource type() const override {
        return CandidateSource::MARKOV;
    }

    size_t estimated_size() const override {
        return max_candidates_;
    }

private:
    const TransitionMatrix& matrix_;
    Generator generator_;
    size_t max_candidates_;
};

/**
 * Utility functions
 */

/**
 * Train a Markov model from password files.
 */
inline TransitionMatrix train_from_files(
    const std::vector<std::string>& paths,
    size_t order = 2
) {
    Trainer::Config config;
    config.order = order;

    Trainer trainer(config);
    trainer.train_multiple(paths);

    return trainer.build_matrix();
}

}  // namespace markov
}  // namespace collider
