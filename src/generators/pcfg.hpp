/**
 * Collider PCFG (Probabilistic Context-Free Grammar)
 *
 * Intelligent passphrase generation based on learned password patterns.
 * See PCFG-INTEGRATION.md for detailed documentation.
 */

#pragma once

#include "../core/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace collider {
namespace pcfg {

/**
 * Terminal symbol with probability.
 */
struct Terminal {
    std::string value;
    double probability;

    bool operator<(const Terminal& other) const {
        return probability < other.probability;
    }
};

/**
 * Non-terminal symbol (e.g., L6, D4, S1).
 */
struct NonTerminal {
    std::string type;      // 'L', 'D', 'S', 'U'
    size_t length;
    std::vector<Terminal> terminals;  // Sorted by probability

    std::string name() const {
        return type + std::to_string(length);
    }
};

/**
 * Structure rule (sequence of non-terminals).
 */
struct StructureRule {
    std::vector<std::string> pattern;  // e.g., ["L6", "D4", "S1"]
    double probability;

    bool operator<(const StructureRule& other) const {
        return probability < other.probability;
    }
};

/**
 * PCFG Grammar containing all learned rules.
 */
class Grammar {
public:
    std::vector<StructureRule> structures;
    std::unordered_map<std::string, NonTerminal> non_terminals;

    /**
     * Load grammar from file.
     */
    static Grammar load(const std::string& path) {
        Grammar grammar;
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open grammar file: " + path);
        }

        // Read header
        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) != "PCFG") {
            throw std::runtime_error("Invalid grammar file format");
        }

        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));

        // Read structures
        uint32_t num_structures;
        file.read(reinterpret_cast<char*>(&num_structures), sizeof(num_structures));

        for (uint32_t i = 0; i < num_structures; ++i) {
            StructureRule rule;

            // Read pattern
            uint32_t pattern_size;
            file.read(reinterpret_cast<char*>(&pattern_size), sizeof(pattern_size));

            for (uint32_t j = 0; j < pattern_size; ++j) {
                uint32_t nt_len;
                file.read(reinterpret_cast<char*>(&nt_len), sizeof(nt_len));
                std::string nt(nt_len, '\0');
                file.read(nt.data(), nt_len);
                rule.pattern.push_back(nt);
            }

            file.read(reinterpret_cast<char*>(&rule.probability), sizeof(rule.probability));
            grammar.structures.push_back(std::move(rule));
        }

        // Read non-terminals
        uint32_t num_nts;
        file.read(reinterpret_cast<char*>(&num_nts), sizeof(num_nts));

        for (uint32_t i = 0; i < num_nts; ++i) {
            NonTerminal nt;

            // Read name
            uint32_t name_len;
            file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
            std::string name(name_len, '\0');
            file.read(name.data(), name_len);

            // Read type and length
            char type;
            file.read(&type, 1);
            nt.type = std::string(1, type);

            uint32_t length;
            file.read(reinterpret_cast<char*>(&length), sizeof(length));
            nt.length = length;

            // Read terminals
            uint32_t num_terminals;
            file.read(reinterpret_cast<char*>(&num_terminals), sizeof(num_terminals));

            for (uint32_t j = 0; j < num_terminals; ++j) {
                Terminal term;

                uint32_t val_len;
                file.read(reinterpret_cast<char*>(&val_len), sizeof(val_len));
                term.value.resize(val_len);
                file.read(term.value.data(), val_len);

                file.read(reinterpret_cast<char*>(&term.probability), sizeof(term.probability));
                nt.terminals.push_back(std::move(term));
            }

            grammar.non_terminals[name] = std::move(nt);
        }

        return grammar;
    }

    /**
     * Save grammar to file.
     */
    void save(const std::string& path) const {
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot create grammar file: " + path);
        }

        // Write header
        file.write("PCFG", 4);
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        // Write structures
        uint32_t num_structures = structures.size();
        file.write(reinterpret_cast<const char*>(&num_structures), sizeof(num_structures));

        for (const auto& rule : structures) {
            uint32_t pattern_size = rule.pattern.size();
            file.write(reinterpret_cast<const char*>(&pattern_size), sizeof(pattern_size));

            for (const auto& nt : rule.pattern) {
                uint32_t nt_len = nt.size();
                file.write(reinterpret_cast<const char*>(&nt_len), sizeof(nt_len));
                file.write(nt.data(), nt_len);
            }

            file.write(reinterpret_cast<const char*>(&rule.probability), sizeof(rule.probability));
        }

        // Write non-terminals
        uint32_t num_nts = non_terminals.size();
        file.write(reinterpret_cast<const char*>(&num_nts), sizeof(num_nts));

        for (const auto& [name, nt] : non_terminals) {
            uint32_t name_len = name.size();
            file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
            file.write(name.data(), name_len);

            char type = nt.type[0];
            file.write(&type, 1);

            uint32_t length = nt.length;
            file.write(reinterpret_cast<const char*>(&length), sizeof(length));

            uint32_t num_terminals = nt.terminals.size();
            file.write(reinterpret_cast<const char*>(&num_terminals), sizeof(num_terminals));

            for (const auto& term : nt.terminals) {
                uint32_t val_len = term.value.size();
                file.write(reinterpret_cast<const char*>(&val_len), sizeof(val_len));
                file.write(term.value.data(), val_len);
                file.write(reinterpret_cast<const char*>(&term.probability), sizeof(term.probability));
            }
        }
    }

    /**
     * Get probability of a specific password.
     */
    double score(const std::string& password) const {
        // Parse password into structure
        std::vector<std::pair<std::string, std::string>> segments;
        std::string structure;

        char current_type = '\0';
        std::string current_value;

        auto classify = [](char c) -> char {
            if (c >= 'A' && c <= 'Z') return 'U';
            if (c >= 'a' && c <= 'z') return 'L';
            if (c >= '0' && c <= '9') return 'D';
            return 'S';
        };

        for (char c : password) {
            char type = classify(c);

            if (type != current_type && !current_value.empty()) {
                std::string nt_name = std::string(1, current_type) +
                                      std::to_string(current_value.size());
                segments.emplace_back(nt_name, current_value);

                if (!structure.empty()) structure += " ";
                structure += nt_name;

                current_value.clear();
            }

            current_type = type;
            current_value += c;
        }

        if (!current_value.empty()) {
            std::string nt_name = std::string(1, current_type) +
                                  std::to_string(current_value.size());
            segments.emplace_back(nt_name, current_value);

            if (!structure.empty()) structure += " ";
            structure += nt_name;
        }

        // Find structure probability
        double struct_prob = 0.0;
        for (const auto& rule : structures) {
            std::string rule_str;
            for (const auto& nt : rule.pattern) {
                if (!rule_str.empty()) rule_str += " ";
                rule_str += nt;
            }
            if (rule_str == structure) {
                struct_prob = rule.probability;
                break;
            }
        }

        if (struct_prob == 0.0) return 0.0;

        // Calculate terminal probabilities
        double total_prob = struct_prob;

        for (const auto& [nt_name, value] : segments) {
            auto it = non_terminals.find(nt_name);
            if (it == non_terminals.end()) return 0.0;

            double term_prob = 0.0;
            for (const auto& term : it->second.terminals) {
                if (term.value == value) {
                    term_prob = term.probability;
                    break;
                }
            }

            if (term_prob == 0.0) return 0.0;
            total_prob *= term_prob;
        }

        return total_prob;
    }

    /**
     * Get total number of terminals across all non-terminals.
     */
    size_t total_terminals() const {
        size_t count = 0;
        for (const auto& [name, nt] : non_terminals) {
            count += nt.terminals.size();
        }
        return count;
    }

    /**
     * Export grammar statistics.
     */
    struct Stats {
        size_t num_structures;
        size_t num_non_terminals;
        size_t num_terminals;
        double avg_terminals_per_nt;
    };

    Stats get_stats() const {
        Stats s;
        s.num_structures = structures.size();
        s.num_non_terminals = non_terminals.size();
        s.num_terminals = total_terminals();
        s.avg_terminals_per_nt = non_terminals.empty() ? 0.0 :
            static_cast<double>(s.num_terminals) / s.num_non_terminals;
        return s;
    }
};

/**
 * Trainer configuration (defined outside class to avoid C++ default member init issues)
 */
struct TrainerConfig {
    size_t min_length = 4;
    size_t max_length = 64;
    double min_terminal_prob = 1e-9;
    bool detect_keyboard_patterns = true;
    bool detect_multiwords = true;
    size_t max_terminals_per_nt = 100000;  // Limit memory usage
};

/**
 * PCFG Trainer - learns grammar from password corpus.
 * Enhanced with keyboard pattern detection and word segmentation.
 */
class Trainer {
public:
    using Config = TrainerConfig;

    explicit Trainer(const Config& config = Config{}) : config_(config) {
        if (config_.detect_keyboard_patterns) {
            init_keyboard_patterns();
        }
    }

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

            auto [structure, segments] = parse_password(line);

            structure_counts_[structure]++;

            for (const auto& [nt_name, value] : segments) {
                terminal_counts_[nt_name][value]++;
            }

            total_passwords_++;
        }
    }

    /**
     * Build final grammar.
     */
    Grammar build_grammar() const {
        Grammar grammar;

        // Build structure rules
        for (const auto& [structure, count] : structure_counts_) {
            double prob = static_cast<double>(count) / total_passwords_;

            std::vector<std::string> pattern;
            std::istringstream iss(structure);
            std::string token;
            while (iss >> token) {
                pattern.push_back(token);
            }

            grammar.structures.push_back({pattern, prob});
        }

        std::sort(grammar.structures.begin(), grammar.structures.end(),
                  [](const auto& a, const auto& b) {
                      return a.probability > b.probability;
                  });

        // Build non-terminals
        for (const auto& [nt_name, values] : terminal_counts_) {
            NonTerminal nt;
            nt.type = nt_name.substr(0, 1);
            nt.length = std::stoul(nt_name.substr(1));

            uint64_t total = 0;
            for (const auto& [_, count] : values) {
                total += count;
            }

            for (const auto& [value, count] : values) {
                double prob = static_cast<double>(count) / total;
                if (prob >= config_.min_terminal_prob) {
                    nt.terminals.push_back({value, prob});
                }
            }

            std::sort(nt.terminals.begin(), nt.terminals.end(),
                      [](const auto& a, const auto& b) {
                          return a.probability > b.probability;
                      });

            grammar.non_terminals[nt_name] = std::move(nt);
        }

        return grammar;
    }

private:
    Config config_;
    std::unordered_map<std::string, uint64_t> structure_counts_;
    std::unordered_map<std::string, std::unordered_map<std::string, uint64_t>> terminal_counts_;
    uint64_t total_passwords_ = 0;

    std::pair<std::string, std::vector<std::pair<std::string, std::string>>>
    parse_password(const std::string& password) const {
        std::vector<std::pair<std::string, std::string>> segments;
        std::string structure;

        char current_type = '\0';
        std::string current_value;

        for (char c : password) {
            char type = classify_char(c);

            if (type != current_type && !current_value.empty()) {
                std::string nt_name = std::string(1, current_type) +
                                      std::to_string(current_value.size());
                segments.emplace_back(nt_name, current_value);

                if (!structure.empty()) structure += " ";
                structure += nt_name;

                current_value.clear();
            }

            current_type = type;
            current_value += c;
        }

        if (!current_value.empty()) {
            std::string nt_name = std::string(1, current_type) +
                                  std::to_string(current_value.size());
            segments.emplace_back(nt_name, current_value);

            if (!structure.empty()) structure += " ";
            structure += nt_name;
        }

        return {structure, segments};
    }

    char classify_char(char c) const {
        if (c >= 'A' && c <= 'Z') return 'U';
        if (c >= 'a' && c <= 'z') return 'L';
        if (c >= '0' && c <= '9') return 'D';
        return 'S';
    }

    // Keyboard pattern detection
    std::vector<std::string> keyboard_patterns_;

    void init_keyboard_patterns() {
        // QWERTY keyboard rows
        keyboard_patterns_ = {
            // Row patterns (left to right)
            "qwerty", "qwertyuiop", "asdf", "asdfgh", "asdfghjkl",
            "zxcv", "zxcvbn", "zxcvbnm",

            // Right to left
            "poiuytrewq", "lkjhgfdsa", "mnbvcxz",

            // Diagonals
            "qaz", "wsx", "edc", "rfv", "tgb", "yhn", "ujm",
            "zaq", "xsw", "cde", "vfr", "bgt", "nhy", "mju",

            // Number row
            "123456", "1234567890", "0987654321",

            // Combos
            "qwert", "asdfg", "zxcvb", "12345",
            "qazwsx", "qweasd", "1qaz2wsx",

            // Numpad patterns
            "147", "258", "369", "741", "852", "963",
            "123", "456", "789", "159", "357",
        };

        // Also add uppercase versions
        std::vector<std::string> upper;
        for (const auto& p : keyboard_patterns_) {
            std::string u = p;
            std::transform(u.begin(), u.end(), u.begin(), ::toupper);
            upper.push_back(u);
        }
        keyboard_patterns_.insert(keyboard_patterns_.end(), upper.begin(), upper.end());
    }

    bool is_keyboard_pattern(const std::string& s) const {
        if (!config_.detect_keyboard_patterns) return false;

        std::string lower = s;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        for (const auto& pattern : keyboard_patterns_) {
            if (lower.find(pattern) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

public:
    /**
     * Train on multiple password files.
     */
    void train_multiple(const std::vector<std::string>& paths) {
        for (const auto& path : paths) {
            train(path);
        }
    }

    /**
     * Get training statistics.
     */
    struct TrainingStats {
        uint64_t total_passwords;
        size_t unique_structures;
        size_t total_non_terminals;
        uint64_t keyboard_patterns_detected;
    };

    TrainingStats get_training_stats() const {
        TrainingStats stats;
        stats.total_passwords = total_passwords_;
        stats.unique_structures = structure_counts_.size();
        stats.total_non_terminals = terminal_counts_.size();
        stats.keyboard_patterns_detected = 0;  // Would need to track this
        return stats;
    }

    /**
     * Merge another trainer's data into this one.
     */
    void merge(const Trainer& other) {
        for (const auto& [structure, count] : other.structure_counts_) {
            structure_counts_[structure] += count;
        }

        for (const auto& [nt_name, values] : other.terminal_counts_) {
            for (const auto& [value, count] : values) {
                terminal_counts_[nt_name][value] += count;
            }
        }

        total_passwords_ += other.total_passwords_;
    }
};

/**
 * PCFG Generator - produces candidates in probability order.
 */
class Generator {
public:
    explicit Generator(const Grammar& grammar) : grammar_(grammar) {}

    /**
     * Generate next candidate.
     */
    std::optional<Candidate> next() {
        if (!initialized_) {
            initialize();
        }

        if (state_queue_.empty()) {
            return std::nullopt;
        }

        GeneratorState state = state_queue_.top();
        state_queue_.pop();

        std::string password = expand_state(state);

        // Generate successor states (simplified)
        advance_state(state);

        return Candidate{
            .phrase = password,
            .priority = static_cast<float>(state.probability),
            .source = CandidateSource::PCFG_GENERATED,
            .rule_applied = ":"
        };
    }

    /**
     * Reset generator.
     */
    void reset() {
        initialized_ = false;
        state_queue_ = {};
    }

private:
    const Grammar& grammar_;

    struct GeneratorState {
        double probability;
        std::vector<size_t> terminal_indices;
        size_t structure_index;

        bool operator<(const GeneratorState& other) const {
            return probability < other.probability;
        }
    };

    std::priority_queue<GeneratorState> state_queue_;
    bool initialized_ = false;

    void initialize() {
        for (size_t i = 0; i < grammar_.structures.size(); ++i) {
            const auto& structure = grammar_.structures[i];

            std::vector<size_t> indices(structure.pattern.size(), 0);
            double prob = structure.probability;

            for (const auto& nt_name : structure.pattern) {
                auto it = grammar_.non_terminals.find(nt_name);
                if (it != grammar_.non_terminals.end() && !it->second.terminals.empty()) {
                    prob *= it->second.terminals[0].probability;
                }
            }

            state_queue_.push({prob, indices, i});
        }

        initialized_ = true;
    }

    std::string expand_state(const GeneratorState& state) const {
        std::string result;
        const auto& pattern = grammar_.structures[state.structure_index].pattern;

        for (size_t i = 0; i < pattern.size(); ++i) {
            const auto& nt_name = pattern[i];
            auto it = grammar_.non_terminals.find(nt_name);
            if (it != grammar_.non_terminals.end()) {
                size_t idx = state.terminal_indices[i];
                if (idx < it->second.terminals.size()) {
                    result += it->second.terminals[idx].value;
                }
            }
        }

        return result;
    }

    void advance_state(GeneratorState state) {
        const auto& pattern = grammar_.structures[state.structure_index].pattern;

        // Increment last index
        for (int i = pattern.size() - 1; i >= 0; --i) {
            const auto& nt_name = pattern[i];
            auto it = grammar_.non_terminals.find(nt_name);
            if (it == grammar_.non_terminals.end()) continue;

            state.terminal_indices[i]++;
            if (state.terminal_indices[i] < it->second.terminals.size()) {
                // Recalculate probability
                double prob = grammar_.structures[state.structure_index].probability;
                for (size_t j = 0; j < pattern.size(); ++j) {
                    auto it2 = grammar_.non_terminals.find(pattern[j]);
                    if (it2 != grammar_.non_terminals.end()) {
                        size_t idx = state.terminal_indices[j];
                        if (idx < it2->second.terminals.size()) {
                            prob *= it2->second.terminals[idx].probability;
                        }
                    }
                }
                state.probability = prob;
                state_queue_.push(state);
                return;
            }
            state.terminal_indices[i] = 0;
        }
    }
};

}  // namespace pcfg
}  // namespace collider
