/**
 * Brain Wallet Passphrase Generation Engine
 *
 * A comprehensive, research-backed passphrase generator optimized for
 * brain wallet recovery. Based on:
 *
 * Academic Research:
 *   - Matt Weir's PCFG (28-129% improvement over JtR)
 *   - OMEN Markov model (probability-ordered enumeration)
 *   - CMU Neural Network password research
 *
 * Competition Insights:
 *   - DEFCON Crack Me If You Can strategies
 *   - Team Hashcat techniques
 *   - OneRuleToRuleThemAll (68.36% crack rate)
 *
 * Brain Wallet Specifics:
 *   - Song lyrics, quotes, book passages
 *   - Cryptocurrency terminology
 *   - Common memorable phrase patterns
 *
 * Architecture:
 *   Corpus → PCFG Training → Markov Enhancement → Mutation Engine →
 *   Combinator → Brain Wallet Patterns → Priority Queue → GPU Pipeline
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <functional>
#include <memory>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <array>
#include <regex>

namespace collider {
namespace generators {

// Forward declarations
class PCFGEngine;
class MarkovEngine;
class MutationEngine;
class CombinatorEngine;
class BrainWalletPatterns;

/**
 * Keyboard layout definitions for walk generation.
 */
struct KeyboardLayout {
    std::string name;
    std::vector<std::string> rows;
    std::unordered_map<char, std::pair<int, int>> key_positions;

    void build_positions() {
        for (size_t row = 0; row < rows.size(); row++) {
            for (size_t col = 0; col < rows[row].size(); col++) {
                key_positions[rows[row][col]] = {static_cast<int>(row), static_cast<int>(col)};
                key_positions[std::toupper(rows[row][col])] = {static_cast<int>(row), static_cast<int>(col)};
            }
        }
    }
};

// Standard keyboard layouts
inline KeyboardLayout QWERTY_LAYOUT = {
    "QWERTY",
    {"1234567890-=", "qwertyuiop[]", "asdfghjkl;'", "zxcvbnm,./"},
    {}
};

inline KeyboardLayout QWERTZ_LAYOUT = {
    "QWERTZ",
    {"1234567890-=", "qwertzuiop[]", "asdfghjkl;'", "yxcvbnm,./"},
    {}
};

inline KeyboardLayout AZERTY_LAYOUT = {
    "AZERTY",
    {"1234567890-=", "azertyuiop[]", "qsdfghjklm;'", "wxcvbn,./"},
    {}
};

/**
 * Leet speak substitution tables.
 * Multiple levels of substitution for thoroughness.
 */
struct LeetTable {
    // Level 1: Most common substitutions
    static inline std::unordered_map<char, std::vector<std::string>> level1 = {
        {'a', {"4", "@"}},
        {'e', {"3"}},
        {'i', {"1", "!"}},
        {'o', {"0"}},
        {'s', {"5", "$"}},
        {'t', {"7"}},
    };

    // Level 2: Extended substitutions
    static inline std::unordered_map<char, std::vector<std::string>> level2 = {
        {'a', {"4", "@", "/\\", "^"}},
        {'b', {"8", "6", "|3"}},
        {'c', {"(", "<", "{"}},
        {'e', {"3", "&"}},
        {'g', {"9", "6"}},
        {'h', {"#", "|-|"}},
        {'i', {"1", "!", "|"}},
        {'l', {"1", "|", "7"}},
        {'o', {"0", "()", "[]"}},
        {'s', {"5", "$", "z"}},
        {'t', {"7", "+"}},
        {'z', {"2"}},
    };

    // Common full-word substitutions
    static inline std::unordered_map<std::string, std::vector<std::string>> words = {
        {"and", {"&", "n", "nd"}},
        {"for", {"4"}},
        {"to", {"2"}},
        {"you", {"u"}},
        {"your", {"ur", "yr"}},
        {"are", {"r"}},
        {"be", {"b"}},
        {"see", {"c"}},
        {"why", {"y"}},
        {"okay", {"ok", "k"}},
        {"love", {"luv", "<3"}},
        {"the", {"da", "teh"}},
    };
};

/**
 * Common password suffixes and prefixes.
 * Derived from breach analysis.
 */
struct CommonAffixes {
    // Most common suffixes (from breach data)
    static inline std::vector<std::string> suffixes = {
        // Digits
        "1", "2", "3", "12", "13", "21", "22", "23",
        "123", "1234", "12345", "123456",
        "01", "07", "11", "69", "77", "88", "99",
        "007", "111", "123", "321", "420", "666", "777", "911",

        // Years
        "2020", "2021", "2022", "2023", "2024", "2025",
        "19", "20", "21", "22", "23", "24", "25",
        "90", "91", "92", "93", "94", "95", "96", "97", "98", "99",
        "00", "01", "02", "03", "04", "05",

        // Symbols
        "!", "!!", "!!!", "!@", "!@#", "!@#$",
        ".", "..", "...",
        "?", "??",
        "@", "#", "$", "*",

        // Common combos
        "1!", "12!", "123!", "1234!",
        "!", "!1", "!12", "!123",
        "@1", "@123",
    };

    // Common prefixes
    static inline std::vector<std::string> prefixes = {
        "the", "my", "a", "i", "1", "123",
        "!", "@", "#", "$",
    };

    // Separators for multi-word passphrases
    static inline std::vector<std::string> separators = {
        "", " ", "_", "-", ".", ",",
        "1", "2", "12", "123",
        "!", "@", "#",
    };
};

/**
 * Cryptocurrency-specific terms and patterns.
 */
struct CryptoTerms {
    static inline std::vector<std::string> terms = {
        // Bitcoin terminology
        "bitcoin", "btc", "satoshi", "nakamoto", "hodl", "hodler",
        "moon", "lambo", "whale", "dip", "pump", "dump",
        "blockchain", "block", "chain", "hash", "mining", "miner",
        "wallet", "cold", "hot", "seed", "private", "public", "key",

        // Trading
        "bull", "bear", "long", "short", "fomo", "fud",
        "ath", "atl", "dyor", "wagmi", "ngmi", "gm", "gn",

        // Crypto culture
        "crypto", "defi", "nft", "web3", "dao", "dex", "cex",
        "eth", "ethereum", "vitalik", "solana", "sol",

        // Numbers significant to crypto
        "21", "21million", "21m", "2100000", "100000000",
        "genesis", "genesis block",
    };

    // BIP39 word list (first 100 most common)
    static inline std::vector<std::string> bip39_common = {
        "abandon", "ability", "able", "about", "above", "absent",
        "absorb", "abstract", "absurd", "abuse", "access", "accident",
        "account", "accuse", "achieve", "acid", "acoustic", "acquire",
        "action", "actor", "actress", "actual", "adapt", "add",
        "addict", "address", "adjust", "admit", "adult", "advance",
        "advice", "aerobic", "affair", "afford", "afraid", "again",
        "age", "agent", "agree", "ahead", "aim", "air",
        "airport", "aisle", "alarm", "album", "alcohol", "alert",
        "alien", "all", "alley", "allow", "almost", "alone",
        "alpha", "already", "also", "alter", "always", "amateur",
        // ... abbreviated for space, full list loaded from file
    };
};

/**
 * Brain Wallet Pattern Templates
 *
 * Common patterns used by humans when creating "memorable" passphrases.
 */
struct BrainWalletTemplates {
    // "I {verb} {noun}" patterns
    static inline std::vector<std::string> i_patterns = {
        "i love {}", "i hate {}", "i want {}", "i need {}",
        "i am {}", "i like {}", "i miss {}", "i have {}",
        "i love {} forever", "i love {} 4ever",
        "i <3 {}", "{} is my love", "my {} is the best",
    };

    // Possessive patterns
    static inline std::vector<std::string> my_patterns = {
        "my {}", "my {} 123", "my {} forever",
        "my favorite {}", "my secret {}",
        "my dog {}", "my cat {}", "my name is {}",
        "{} is mine", "{} belongs to me",
    };

    // Question patterns
    static inline std::vector<std::string> question_patterns = {
        "what is {}", "who is {}", "where is {}",
        "why {}", "how {}",
    };

    // Famous opening lines
    static inline std::vector<std::string> famous_openings = {
        "to be or not to be",
        "it was the best of times",
        "call me ishmael",
        "in the beginning",
        "once upon a time",
        "a long time ago",
        "in a galaxy far far away",
        "the quick brown fox",
        "lorem ipsum dolor",
    };

    // Common religious/spiritual phrases
    static inline std::vector<std::string> religious = {
        "in god we trust",
        "god is love",
        "jesus saves",
        "praise the lord",
        "amen",
        "hallelujah",
        "the lord is my shepherd",
        "our father who art in heaven",
    };
};

/**
 * PCFG (Probabilistic Context-Free Grammar) Engine
 *
 * Based on Matt Weir's research. Learns password structure from training data
 * and generates candidates in probability order.
 */
class PCFGEngine {
public:
    // Token types
    enum class TokenType { Letter, Digit, Symbol, Keyboard };

    struct Token {
        TokenType type;
        size_t length;
        std::string value;  // Actual value (for terminals)
    };

    // Base structure (e.g., "L8D3S1" = 8 letters + 3 digits + 1 symbol)
    struct BaseStructure {
        std::vector<std::pair<TokenType, size_t>> pattern;
        double probability;

        std::string to_string() const {
            std::string s;
            for (const auto& [type, len] : pattern) {
                switch (type) {
                    case TokenType::Letter: s += "L" + std::to_string(len); break;
                    case TokenType::Digit: s += "D" + std::to_string(len); break;
                    case TokenType::Symbol: s += "S" + std::to_string(len); break;
                    case TokenType::Keyboard: s += "K" + std::to_string(len); break;
                }
            }
            return s;
        }
    };

    /**
     * Train PCFG on password corpus.
     */
    void train(const std::vector<std::string>& passwords) {
        structure_counts_.clear();
        letter_counts_.clear();
        digit_counts_.clear();
        symbol_counts_.clear();
        total_passwords_ = passwords.size();

        for (const auto& password : passwords) {
            auto structure = parse_structure(password);
            structure_counts_[structure.to_string()]++;

            // Extract terminals
            extract_terminals(password, structure);
        }

        // Calculate probabilities
        calculate_probabilities();
    }

    /**
     * Generate candidates in probability order.
     */
    class Generator {
    public:
        Generator(const PCFGEngine& engine, size_t max_candidates = 1000000)
            : engine_(engine), max_candidates_(max_candidates) {
            initialize();
        }

        bool has_next() const {
            return !queue_.empty() && generated_ < max_candidates_;
        }

        std::string next() {
            if (!has_next()) return "";

            auto top = queue_.top();
            queue_.pop();

            std::string candidate = expand(top);
            generated_++;

            // Add next variations
            expand_queue(top);

            return candidate;
        }

    private:
        struct QueueItem {
            BaseStructure structure;
            std::vector<size_t> indices;  // Index into terminal lists
            double probability;

            bool operator<(const QueueItem& other) const {
                return probability < other.probability;  // Max-heap
            }
        };

        const PCFGEngine& engine_;
        size_t max_candidates_;
        size_t generated_ = 0;
        std::priority_queue<QueueItem> queue_;

        void initialize();
        std::string expand(const QueueItem& item);
        void expand_queue(const QueueItem& item);
    };

    Generator generator(size_t max = 1000000) const {
        return Generator(*this, max);
    }

    /**
     * Score a password (log probability).
     */
    double score(const std::string& password) const {
        auto structure = parse_structure(password);
        auto it = structure_probs_.find(structure.to_string());
        if (it == structure_probs_.end()) return -100.0;

        double log_prob = std::log(it->second);
        // Add terminal probabilities...
        return log_prob;
    }

    /**
     * Save/load trained model.
     */
    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    size_t total_passwords_ = 0;

    std::unordered_map<std::string, size_t> structure_counts_;
    std::unordered_map<std::string, double> structure_probs_;

    // Terminal counts by type and length
    std::unordered_map<size_t, std::unordered_map<std::string, size_t>> letter_counts_;
    std::unordered_map<size_t, std::unordered_map<std::string, size_t>> digit_counts_;
    std::unordered_map<size_t, std::unordered_map<std::string, size_t>> symbol_counts_;

    // Sorted terminals by probability
    std::unordered_map<size_t, std::vector<std::pair<std::string, double>>> letter_probs_;
    std::unordered_map<size_t, std::vector<std::pair<std::string, double>>> digit_probs_;
    std::unordered_map<size_t, std::vector<std::pair<std::string, double>>> symbol_probs_;

    BaseStructure parse_structure(const std::string& password) const {
        BaseStructure structure;
        TokenType current_type = TokenType::Letter;
        size_t current_len = 0;

        for (char c : password) {
            TokenType type;
            if (std::isalpha(c)) type = TokenType::Letter;
            else if (std::isdigit(c)) type = TokenType::Digit;
            else type = TokenType::Symbol;

            if (type == current_type) {
                current_len++;
            } else {
                if (current_len > 0) {
                    structure.pattern.push_back({current_type, current_len});
                }
                current_type = type;
                current_len = 1;
            }
        }

        if (current_len > 0) {
            structure.pattern.push_back({current_type, current_len});
        }

        return structure;
    }

    void extract_terminals(const std::string& password, const BaseStructure& structure) {
        size_t pos = 0;
        for (const auto& [type, len] : structure.pattern) {
            std::string terminal = password.substr(pos, len);
            pos += len;

            // Convert letters to lowercase for counting
            if (type == TokenType::Letter) {
                std::string lower = terminal;
                std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
                letter_counts_[len][lower]++;
            } else if (type == TokenType::Digit) {
                digit_counts_[len][terminal]++;
            } else {
                symbol_counts_[len][terminal]++;
            }
        }
    }

    void calculate_probabilities() {
        // Structure probabilities
        for (const auto& [structure, count] : structure_counts_) {
            structure_probs_[structure] = static_cast<double>(count) / total_passwords_;
        }

        // Terminal probabilities
        auto calc_terminal_probs = [](
            const std::unordered_map<size_t, std::unordered_map<std::string, size_t>>& counts,
            std::unordered_map<size_t, std::vector<std::pair<std::string, double>>>& probs
        ) {
            for (const auto& [len, term_counts] : counts) {
                size_t total = 0;
                for (const auto& [term, count] : term_counts) {
                    total += count;
                }

                std::vector<std::pair<std::string, double>> sorted;
                for (const auto& [term, count] : term_counts) {
                    sorted.push_back({term, static_cast<double>(count) / total});
                }

                // Sort by probability descending
                std::sort(sorted.begin(), sorted.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });

                probs[len] = std::move(sorted);
            }
        };

        calc_terminal_probs(letter_counts_, letter_probs_);
        calc_terminal_probs(digit_counts_, digit_probs_);
        calc_terminal_probs(symbol_counts_, symbol_probs_);
    }
};

/**
 * Markov Chain Engine (OMEN-style)
 *
 * Learns character transition probabilities for generating
 * password candidates in probability order.
 */
class MarkovEngine {
public:
    static constexpr size_t ORDER = 4;  // n-gram order

    /**
     * Train on password corpus.
     */
    void train(const std::vector<std::string>& passwords) {
        ngram_counts_.clear();
        start_counts_.clear();
        total_starts_ = 0;

        for (const auto& password : passwords) {
            if (password.size() < ORDER) continue;

            // Start n-gram
            std::string start = password.substr(0, ORDER);
            start_counts_[start]++;
            total_starts_++;

            // Transitions
            for (size_t i = 0; i + ORDER < password.size(); i++) {
                std::string ngram = password.substr(i, ORDER);
                char next = password[i + ORDER];
                ngram_counts_[ngram][next]++;
            }
        }

        calculate_probabilities();
    }

    /**
     * Generate password with given length.
     */
    std::string generate(size_t length, std::mt19937& rng) const {
        if (start_probs_.empty()) return "";

        // Choose starting n-gram
        std::string password = sample_start(rng);

        // Extend to desired length
        while (password.size() < length) {
            std::string ngram = password.substr(password.size() - ORDER, ORDER);
            char next = sample_next(ngram, rng);
            if (next == '\0') break;
            password += next;
        }

        return password;
    }

    /**
     * Score password (log probability).
     */
    double score(const std::string& password) const {
        if (password.size() < ORDER) return -100.0;

        double log_prob = 0.0;

        // Start probability
        std::string start = password.substr(0, ORDER);
        auto it = start_probs_.find(start);
        if (it != start_probs_.end()) {
            log_prob += std::log(it->second);
        } else {
            return -100.0;
        }

        // Transition probabilities
        for (size_t i = 0; i + ORDER < password.size(); i++) {
            std::string ngram = password.substr(i, ORDER);
            char next = password[i + ORDER];

            auto ngram_it = ngram_probs_.find(ngram);
            if (ngram_it != ngram_probs_.end()) {
                auto next_it = ngram_it->second.find(next);
                if (next_it != ngram_it->second.end()) {
                    log_prob += std::log(next_it->second);
                } else {
                    log_prob += std::log(0.001);  // Smoothing
                }
            } else {
                log_prob += std::log(0.001);
            }
        }

        return log_prob;
    }

    void save(const std::string& path) const;
    void load(const std::string& path);

private:
    std::unordered_map<std::string, std::unordered_map<char, size_t>> ngram_counts_;
    std::unordered_map<std::string, size_t> start_counts_;
    size_t total_starts_ = 0;

    std::unordered_map<std::string, std::unordered_map<char, double>> ngram_probs_;
    std::unordered_map<std::string, double> start_probs_;

    void calculate_probabilities() {
        // Start probabilities
        for (const auto& [start, count] : start_counts_) {
            start_probs_[start] = static_cast<double>(count) / total_starts_;
        }

        // Transition probabilities
        for (const auto& [ngram, next_counts] : ngram_counts_) {
            size_t total = 0;
            for (const auto& [next, count] : next_counts) {
                total += count;
            }

            for (const auto& [next, count] : next_counts) {
                ngram_probs_[ngram][next] = static_cast<double>(count) / total;
            }
        }
    }

    std::string sample_start(std::mt19937& rng) const {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        double cumulative = 0.0;

        for (const auto& [start, prob] : start_probs_) {
            cumulative += prob;
            if (r <= cumulative) return start;
        }

        return start_probs_.begin()->first;
    }

    char sample_next(const std::string& ngram, std::mt19937& rng) const {
        auto it = ngram_probs_.find(ngram);
        if (it == ngram_probs_.end()) return '\0';

        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng);
        double cumulative = 0.0;

        for (const auto& [next, prob] : it->second) {
            cumulative += prob;
            if (r <= cumulative) return next;
        }

        return it->second.begin()->first;
    }
};

/**
 * Keyboard Walk Generator
 *
 * Generates all keyboard walk patterns for multiple layouts.
 */
class KeyboardWalkGenerator {
public:
    KeyboardWalkGenerator() {
        // Initialize layouts
        QWERTY_LAYOUT.build_positions();
        QWERTZ_LAYOUT.build_positions();
        AZERTY_LAYOUT.build_positions();

        layouts_ = {QWERTY_LAYOUT, QWERTZ_LAYOUT, AZERTY_LAYOUT};
    }

    /**
     * Generate all keyboard walks up to given length.
     */
    std::vector<std::string> generate_all(size_t min_len = 4, size_t max_len = 12) const {
        std::vector<std::string> walks;

        for (const auto& layout : layouts_) {
            // Horizontal walks
            for (const auto& row : layout.rows) {
                for (size_t start = 0; start < row.size(); start++) {
                    for (size_t len = min_len; len <= std::min(max_len, row.size() - start); len++) {
                        walks.push_back(row.substr(start, len));
                        // Reverse
                        std::string rev = row.substr(start, len);
                        std::reverse(rev.begin(), rev.end());
                        walks.push_back(rev);
                    }
                }
            }

            // Diagonal walks
            generate_diagonals(layout, min_len, max_len, walks);

            // Shape patterns (Z, W, M, etc.)
            generate_shapes(layout, walks);
        }

        // Common known patterns
        std::vector<std::string> common = {
            "qwerty", "qwertyuiop", "asdf", "asdfgh", "asdfghjkl",
            "zxcv", "zxcvbn", "zxcvbnm", "qazwsx", "1qaz2wsx",
            "qaz", "wsx", "edc", "rfv", "tgb", "yhn", "ujm",
            "1234", "12345", "123456", "1234567", "12345678",
            "0987", "09876", "098765", "0987654", "09876543",
            "1q2w3e", "1q2w3e4r", "1qaz", "2wsx", "3edc",
            "!qaz", "@wsx", "#edc", "$rfv",
            "qweasd", "qweasdzxc", "asdzxc",
            "1234qwer", "qwer1234",
            "poiuytrewq", "lkjhgfdsa", "mnbvcxz",
        };

        walks.insert(walks.end(), common.begin(), common.end());

        // Deduplicate
        std::sort(walks.begin(), walks.end());
        walks.erase(std::unique(walks.begin(), walks.end()), walks.end());

        return walks;
    }

    /**
     * Check if string is a keyboard walk.
     */
    bool is_keyboard_walk(const std::string& s) const {
        if (s.size() < 3) return false;

        for (const auto& layout : layouts_) {
            bool is_walk = true;
            for (size_t i = 1; i < s.size() && is_walk; i++) {
                auto it1 = layout.key_positions.find(std::tolower(s[i-1]));
                auto it2 = layout.key_positions.find(std::tolower(s[i]));

                if (it1 == layout.key_positions.end() || it2 == layout.key_positions.end()) {
                    is_walk = false;
                    break;
                }

                int row_diff = std::abs(it1->second.first - it2->second.first);
                int col_diff = std::abs(it1->second.second - it2->second.second);

                // Adjacent keys: row diff <= 1 and col diff <= 1
                if (row_diff > 1 || col_diff > 1) {
                    is_walk = false;
                }
            }

            if (is_walk) return true;
        }

        return false;
    }

private:
    std::vector<KeyboardLayout> layouts_;

    void generate_diagonals(const KeyboardLayout& layout, size_t min_len, size_t max_len,
                            std::vector<std::string>& walks) const {
        // Generate diagonal patterns
        size_t num_rows = layout.rows.size();

        for (size_t start_row = 0; start_row < num_rows; start_row++) {
            for (size_t start_col = 0; start_col < layout.rows[start_row].size(); start_col++) {
                // Down-right diagonal
                std::string diag;
                for (size_t r = start_row, c = start_col;
                     r < num_rows && c < layout.rows[r].size() && diag.size() < max_len;
                     r++, c++) {
                    diag += layout.rows[r][c];
                    if (diag.size() >= min_len) {
                        walks.push_back(diag);
                    }
                }

                // Down-left diagonal
                diag.clear();
                for (size_t r = start_row; r < num_rows && diag.size() < max_len; r++) {
                    int c = static_cast<int>(start_col) - (r - start_row);
                    if (c >= 0 && c < static_cast<int>(layout.rows[r].size())) {
                        diag += layout.rows[r][c];
                        if (diag.size() >= min_len) {
                            walks.push_back(diag);
                        }
                    } else {
                        break;
                    }
                }
            }
        }
    }

    void generate_shapes(const KeyboardLayout& layout, std::vector<std::string>& walks) const {
        // Z-shape, W-shape, etc.
        // These are layout-specific patterns
        if (layout.name == "QWERTY") {
            walks.push_back("qazxsw");   // Z shape
            walks.push_back("wsxzaq");   // Reverse Z
            walks.push_back("qazwsxedc");
            walks.push_back("zaqwsxcde");
        }
    }
};

/**
 * Mutation Engine
 *
 * Applies hashcat-style mutations to base words.
 * Optimized based on OneRuleToRuleThemAll research.
 */
class MutationEngine {
public:
    enum class MutationType {
        None,
        Capitalize,     // First letter uppercase
        Uppercase,      // All uppercase
        Lowercase,      // All lowercase
        ToggleCase,     // Toggle all cases
        Leet,           // Leet speak substitution
        Reverse,        // Reverse string
        Duplicate,      // Duplicate string
        Reflect,        // Append reverse
        AppendDigit,    // Append digits
        PrependDigit,   // Prepend digits
        AppendSymbol,   // Append symbols
        AppendYear,     // Append year
    };

    /**
     * Generate all mutations of a word.
     * Returns in probability order (most likely first).
     */
    std::vector<std::string> mutate(const std::string& word, int max_mutations = 100) const {
        std::vector<std::string> results;
        results.reserve(max_mutations);

        // Level 0: Original
        results.push_back(word);

        // Level 1: Case variations (highest hit rate)
        add_case_mutations(word, results);

        // Level 2: Common suffixes
        add_suffix_mutations(word, results);

        // Level 3: Leet speak
        add_leet_mutations(word, results);

        // Level 4: Combinations
        add_combined_mutations(word, results, max_mutations);

        // Deduplicate while preserving order
        std::unordered_set<std::string> seen;
        std::vector<std::string> unique;
        for (const auto& s : results) {
            if (seen.insert(s).second) {
                unique.push_back(s);
                if (unique.size() >= static_cast<size_t>(max_mutations)) break;
            }
        }

        return unique;
    }

    /**
     * Apply single hashcat rule.
     */
    std::string apply_rule(const std::string& word, const std::string& rule) const {
        std::string result = word;

        for (size_t i = 0; i < rule.size(); i++) {
            char cmd = rule[i];

            switch (cmd) {
                case ':': break;  // No-op
                case 'l': std::transform(result.begin(), result.end(), result.begin(), ::tolower); break;
                case 'u': std::transform(result.begin(), result.end(), result.begin(), ::toupper); break;
                case 'c': if (!result.empty()) { result[0] = std::toupper(result[0]); } break;
                case 'C': if (!result.empty()) { result[0] = std::tolower(result[0]); } break;
                case 't': for (char& c : result) { c = std::isupper(c) ? std::tolower(c) : std::toupper(c); } break;
                case 'r': std::reverse(result.begin(), result.end()); break;
                case 'd': result += result; break;
                case 'f': { std::string rev = result; std::reverse(rev.begin(), rev.end()); result += rev; } break;
                case '{': if (!result.empty()) { char c = result[0]; result = result.substr(1) + c; } break;
                case '}': if (!result.empty()) { char c = result.back(); result = c + result.substr(0, result.size()-1); } break;
                case '[': if (!result.empty()) result = result.substr(1); break;
                case ']': if (!result.empty()) result.pop_back(); break;
                case '$': if (i + 1 < rule.size()) { result += rule[++i]; } break;
                case '^': if (i + 1 < rule.size()) { result = rule[++i] + result; } break;
                // ... more rules can be added
            }
        }

        return result;
    }

private:
    void add_case_mutations(const std::string& word, std::vector<std::string>& results) const {
        // First letter capitalized
        if (!word.empty()) {
            std::string cap = word;
            cap[0] = std::toupper(cap[0]);
            results.push_back(cap);
        }

        // All uppercase
        std::string upper = word;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
        results.push_back(upper);

        // All lowercase
        std::string lower = word;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        results.push_back(lower);

        // Toggle case (sWAP)
        std::string toggle = word;
        for (char& c : toggle) {
            c = std::isupper(c) ? std::tolower(c) : std::toupper(c);
        }
        results.push_back(toggle);
    }

    void add_suffix_mutations(const std::string& word, std::vector<std::string>& results) const {
        // Most common suffixes (from breach data)
        static const std::vector<std::string> top_suffixes = {
            "1", "!", "123", "12", "2", "3", "1!", "!!", ".",
            "2023", "2024", "2025", "123!", "12!", "1234",
            "01", "11", "13", "21", "22", "23", "69", "77", "99",
            "007", "123456", "@", "#", "$", "1@", "12@",
        };

        for (const auto& suffix : top_suffixes) {
            results.push_back(word + suffix);
        }

        // Also with capitalization
        if (!word.empty()) {
            std::string cap = word;
            cap[0] = std::toupper(cap[0]);
            for (const auto& suffix : top_suffixes) {
                results.push_back(cap + suffix);
            }
        }
    }

    void add_leet_mutations(const std::string& word, std::vector<std::string>& results) const {
        // Simple leet (most common substitutions only)
        std::string leet = word;
        bool changed = false;

        for (char& c : leet) {
            char lower = std::tolower(c);
            switch (lower) {
                case 'a': c = '4'; changed = true; break;
                case 'e': c = '3'; changed = true; break;
                case 'i': c = '1'; changed = true; break;
                case 'o': c = '0'; changed = true; break;
                case 's': c = '$'; changed = true; break;
                case 't': c = '7'; changed = true; break;
            }
        }

        if (changed) {
            results.push_back(leet);

            // Leet + suffix
            results.push_back(leet + "!");
            results.push_back(leet + "1");
            results.push_back(leet + "123");
        }
    }

    void add_combined_mutations(const std::string& word, std::vector<std::string>& results,
                                 int max_mutations) const {
        if (results.size() >= static_cast<size_t>(max_mutations)) return;

        // Reverse
        std::string rev = word;
        std::reverse(rev.begin(), rev.end());
        results.push_back(rev);

        // Duplicate
        results.push_back(word + word);

        // Reflect
        results.push_back(word + rev);

        // Double with separator
        results.push_back(word + "1" + word);
        results.push_back(word + "_" + word);

        // Prefix + word
        results.push_back("the" + word);
        results.push_back("my" + word);
        results.push_back("i" + word);
    }
};

/**
 * Combinator Engine (PRINCE-style)
 *
 * Combines words from wordlists in probability order.
 */
class CombinatorEngine {
public:
    /**
     * Set wordlists for combination.
     */
    void add_wordlist(const std::vector<std::string>& words) {
        wordlists_.push_back(words);
    }

    /**
     * Generate word combinations.
     */
    class Generator {
    public:
        Generator(const CombinatorEngine& engine, size_t min_words = 2, size_t max_words = 3)
            : engine_(engine), min_words_(min_words), max_words_(max_words) {
            initialize();
        }

        bool has_next() const { return current_ < combinations_.size(); }

        std::string next() {
            if (!has_next()) return "";
            return combinations_[current_++];
        }

    private:
        const CombinatorEngine& engine_;
        size_t min_words_;
        size_t max_words_;
        size_t current_ = 0;
        std::vector<std::string> combinations_;

        void initialize() {
            // Generate combinations
            if (engine_.wordlists_.empty()) return;

            const auto& words = engine_.wordlists_[0];

            // 2-word combinations
            if (min_words_ <= 2) {
                for (size_t i = 0; i < std::min(words.size(), size_t(1000)); i++) {
                    for (size_t j = 0; j < std::min(words.size(), size_t(1000)); j++) {
                        if (i != j) {
                            // Direct concatenation
                            combinations_.push_back(words[i] + words[j]);

                            // With separators
                            for (const auto& sep : CommonAffixes::separators) {
                                if (!sep.empty()) {
                                    combinations_.push_back(words[i] + sep + words[j]);
                                }
                            }
                        }
                    }
                }
            }

            // 3-word combinations (limited due to explosion)
            if (max_words_ >= 3 && words.size() >= 3) {
                for (size_t i = 0; i < std::min(words.size(), size_t(100)); i++) {
                    for (size_t j = 0; j < std::min(words.size(), size_t(100)); j++) {
                        for (size_t k = 0; k < std::min(words.size(), size_t(100)); k++) {
                            if (i != j && j != k && i != k) {
                                combinations_.push_back(words[i] + words[j] + words[k]);
                            }
                        }
                    }
                }
            }
        }
    };

    Generator generator(size_t min_words = 2, size_t max_words = 3) const {
        return Generator(*this, min_words, max_words);
    }

private:
    std::vector<std::vector<std::string>> wordlists_;
};

/**
 * Brain Wallet Passphrase Engine
 *
 * The main engine that combines all components.
 */
class BrainWalletEngine {
public:
    struct Config {
        // PCFG
        bool use_pcfg = true;
        size_t pcfg_max_candidates = 1000000;

        // Markov
        bool use_markov = true;
        size_t markov_min_length = 6;
        size_t markov_max_length = 32;

        // Mutations
        int mutations_per_word = 50;

        // Combinations
        size_t max_combined_words = 3;

        // Keyboard walks
        bool include_keyboard_walks = true;

        // Brain wallet specific
        bool use_brain_wallet_patterns = true;
        bool use_crypto_terms = true;

        Config()
            : use_pcfg(true)
            , pcfg_max_candidates(1000000)
            , use_markov(true)
            , markov_min_length(6)
            , markov_max_length(32)
            , mutations_per_word(50)
            , max_combined_words(3)
            , include_keyboard_walks(true)
            , use_brain_wallet_patterns(true)
            , use_crypto_terms(true)
        {}
    };

    explicit BrainWalletEngine(const Config& config = Config()) : config_(config) {
        keyboard_gen_ = std::make_unique<KeyboardWalkGenerator>();
        mutation_engine_ = std::make_unique<MutationEngine>();
        combinator_ = std::make_unique<CombinatorEngine>();
        pcfg_ = std::make_unique<PCFGEngine>();
        markov_ = std::make_unique<MarkovEngine>();
    }

    /**
     * Load training data from various sources.
     */
    void load_password_corpus(const std::string& path) {
        std::ifstream file(path);
        if (!file) return;

        std::vector<std::string> passwords;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line[0] != '#') {
                // Remove potential hash prefix (for potfiles)
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    line = line.substr(colon + 1);
                }
                passwords.push_back(line);
            }
        }

        password_corpus_ = std::move(passwords);

        // Train PCFG and Markov
        if (config_.use_pcfg && !password_corpus_.empty()) {
            pcfg_->train(password_corpus_);
        }
        if (config_.use_markov && !password_corpus_.empty()) {
            markov_->train(password_corpus_);
        }
    }

    /**
     * Load phrase corpus (lyrics, quotes, etc.)
     */
    void load_phrase_corpus(const std::string& path) {
        std::ifstream file(path);
        if (!file) return;

        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line[0] != '#') {
                phrase_corpus_.push_back(line);
            }
        }
    }

    /**
     * Load wordlist.
     */
    void load_wordlist(const std::string& path) {
        std::ifstream file(path);
        if (!file) return;

        std::vector<std::string> words;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line[0] != '#') {
                words.push_back(line);
            }
        }

        wordlists_.push_back(std::move(words));
        combinator_->add_wordlist(wordlists_.back());
    }

    /**
     * Generate passphrase candidates.
     * Main entry point for candidate generation.
     */
    class CandidateGenerator {
    public:
        CandidateGenerator(const BrainWalletEngine& engine)
            : engine_(engine) {
            initialize();
        }

        bool has_next() const { return current_idx_ < candidates_.size(); }

        std::string next() {
            if (!has_next()) return "";
            return candidates_[current_idx_++];
        }

        size_t remaining() const {
            return candidates_.size() - current_idx_;
        }

    private:
        const BrainWalletEngine& engine_;
        size_t current_idx_ = 0;
        std::vector<std::string> candidates_;

        void initialize() {
            // Phase 1: Direct wordlist entries with mutations
            for (const auto& wordlist : engine_.wordlists_) {
                for (const auto& word : wordlist) {
                    auto mutations = engine_.mutation_engine_->mutate(
                        word, engine_.config_.mutations_per_word);
                    candidates_.insert(candidates_.end(), mutations.begin(), mutations.end());
                }
            }

            // Phase 2: Phrase corpus with mutations
            for (const auto& phrase : engine_.phrase_corpus_) {
                candidates_.push_back(phrase);

                // Normalized versions
                std::string normalized = phrase;
                // Remove spaces
                normalized.erase(std::remove(normalized.begin(), normalized.end(), ' '),
                                 normalized.end());
                candidates_.push_back(normalized);

                // Replace spaces with common separators
                for (const auto& sep : CommonAffixes::separators) {
                    if (sep != " " && !sep.empty()) {
                        std::string with_sep = phrase;
                        size_t pos = 0;
                        while ((pos = with_sep.find(' ', pos)) != std::string::npos) {
                            with_sep.replace(pos, 1, sep);
                            pos += sep.length();
                        }
                        candidates_.push_back(with_sep);
                    }
                }

                // Apply mutations to normalized
                auto mutations = engine_.mutation_engine_->mutate(
                    normalized, engine_.config_.mutations_per_word / 2);
                candidates_.insert(candidates_.end(), mutations.begin(), mutations.end());
            }

            // Phase 3: Keyboard walks
            if (engine_.config_.include_keyboard_walks) {
                auto walks = engine_.keyboard_gen_->generate_all();
                for (const auto& walk : walks) {
                    candidates_.push_back(walk);
                    // Mutations of walks
                    candidates_.push_back(walk + "!");
                    candidates_.push_back(walk + "1");
                    candidates_.push_back(walk + "123");

                    std::string upper = walk;
                    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
                    candidates_.push_back(upper);
                }
            }

            // Phase 4: Crypto terms
            if (engine_.config_.use_crypto_terms) {
                for (const auto& term : CryptoTerms::terms) {
                    auto mutations = engine_.mutation_engine_->mutate(
                        term, engine_.config_.mutations_per_word);
                    candidates_.insert(candidates_.end(), mutations.begin(), mutations.end());
                }
            }

            // Phase 5: Brain wallet patterns
            if (engine_.config_.use_brain_wallet_patterns) {
                add_brain_wallet_patterns();
            }

            // Phase 6: Word combinations
            auto combo_gen = engine_.combinator_->generator(2, engine_.config_.max_combined_words);
            while (combo_gen.has_next()) {
                std::string combo = combo_gen.next();
                candidates_.push_back(combo);

                // Limited mutations for combinations
                auto mutations = engine_.mutation_engine_->mutate(combo, 10);
                candidates_.insert(candidates_.end(), mutations.begin(), mutations.end());
            }

            // Deduplicate
            std::sort(candidates_.begin(), candidates_.end());
            candidates_.erase(std::unique(candidates_.begin(), candidates_.end()),
                             candidates_.end());
        }

        void add_brain_wallet_patterns() {
            // Extract top words from wordlists
            std::vector<std::string> top_words;
            for (const auto& wordlist : engine_.wordlists_) {
                for (size_t i = 0; i < std::min(wordlist.size(), size_t(100)); i++) {
                    top_words.push_back(wordlist[i]);
                }
            }

            // "I love X" patterns
            for (const auto& pattern : BrainWalletTemplates::i_patterns) {
                for (const auto& word : top_words) {
                    size_t pos = pattern.find("{}");
                    if (pos != std::string::npos) {
                        std::string filled = pattern;
                        filled.replace(pos, 2, word);
                        candidates_.push_back(filled);

                        // No spaces version
                        std::string no_space = filled;
                        no_space.erase(std::remove(no_space.begin(), no_space.end(), ' '),
                                      no_space.end());
                        candidates_.push_back(no_space);
                    }
                }
            }

            // Famous openings
            for (const auto& opening : BrainWalletTemplates::famous_openings) {
                candidates_.push_back(opening);

                // Without spaces
                std::string no_space = opening;
                no_space.erase(std::remove(no_space.begin(), no_space.end(), ' '),
                              no_space.end());
                candidates_.push_back(no_space);

                // Capitalized words
                std::string cap = opening;
                bool next_cap = true;
                for (char& c : cap) {
                    if (c == ' ') {
                        next_cap = true;
                    } else if (next_cap) {
                        c = std::toupper(c);
                        next_cap = false;
                    }
                }
                candidates_.push_back(cap);
            }

            // Religious phrases
            for (const auto& phrase : BrainWalletTemplates::religious) {
                candidates_.push_back(phrase);
                std::string no_space = phrase;
                no_space.erase(std::remove(no_space.begin(), no_space.end(), ' '),
                              no_space.end());
                candidates_.push_back(no_space);
            }
        }
    };

    CandidateGenerator generator() const {
        return CandidateGenerator(*this);
    }

    /**
     * Score a passphrase (for priority ordering).
     * Higher score = more likely to be a real password.
     */
    double score_passphrase(const std::string& passphrase) const {
        double score = 0.0;

        // PCFG score
        if (config_.use_pcfg) {
            score += pcfg_->score(passphrase) * 0.4;
        }

        // Markov score
        if (config_.use_markov) {
            score += markov_->score(passphrase) * 0.4;
        }

        // Length penalty (brain wallets tend to be longer)
        if (passphrase.size() >= 8 && passphrase.size() <= 32) {
            score += 0.1;
        }

        // Keyboard walk detection
        if (keyboard_gen_->is_keyboard_walk(passphrase)) {
            score += 0.5;  // Keyboard walks are common
        }

        // Contains crypto terms
        for (const auto& term : CryptoTerms::terms) {
            if (passphrase.find(term) != std::string::npos) {
                score += 0.3;
                break;
            }
        }

        return score;
    }

    /**
     * Save trained models.
     */
    void save_models(const std::string& base_path) const {
        if (config_.use_pcfg) {
            pcfg_->save(base_path + ".pcfg");
        }
        if (config_.use_markov) {
            markov_->save(base_path + ".markov");
        }
    }

    /**
     * Load trained models.
     */
    void load_models(const std::string& base_path) {
        if (config_.use_pcfg) {
            pcfg_->load(base_path + ".pcfg");
        }
        if (config_.use_markov) {
            markov_->load(base_path + ".markov");
        }
    }

private:
    Config config_;

    std::vector<std::string> password_corpus_;
    std::vector<std::string> phrase_corpus_;
    std::vector<std::vector<std::string>> wordlists_;

    std::unique_ptr<KeyboardWalkGenerator> keyboard_gen_;
    std::unique_ptr<MutationEngine> mutation_engine_;
    std::unique_ptr<CombinatorEngine> combinator_;
    std::unique_ptr<PCFGEngine> pcfg_;
    std::unique_ptr<MarkovEngine> markov_;
};

}  // namespace generators
}  // namespace collider
