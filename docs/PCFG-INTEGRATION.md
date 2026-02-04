# PCFG Integration: Probabilistic Context-Free Grammar for Superflayer

## Overview

PCFG (Probabilistic Context-Free Grammar) enables **intelligent passphrase generation** by learning patterns from known passwords. Instead of blindly iterating through wordlists, PCFG generates candidates in **probability order**—most likely passphrases first.

---

## How PCFG Works

### Password Structure Analysis

PCFG breaks passwords into structural components:

```
Password: "Bitcoin2013!"
Structure: U1 L6 D4 S1
           │  │   │  └── 1 Symbol (!)
           │  │   └───── 4 Digits (2013)
           │  └───────── 6 Lowercase (itcoin)
           └──────────── 1 Uppercase (B)
```

### Grammar Rules

From training data, PCFG learns production rules with probabilities:

```
S → L D         (0.35)   # Letters + Digits
S → L D S       (0.22)   # Letters + Digits + Symbol
S → U L D       (0.18)   # Cap + Letters + Digits
S → L           (0.12)   # Letters only
S → D L         (0.08)   # Digits + Letters
S → ...         (0.05)   # Other patterns

L6 → "bitcoin" (0.0001)
L6 → "crypto"  (0.00008)
L6 → "wallet"  (0.00006)
...

D4 → "2013"    (0.002)
D4 → "2017"    (0.0018)
D4 → "2021"    (0.0015)
D4 → "1234"    (0.0012)
...

S1 → "!"       (0.45)
S1 → "@"       (0.20)
S1 → "#"       (0.15)
...
```

### Probability-Ordered Generation

PCFG generates candidates by combining rules in probability order:

```
Priority 1: "password123" (P = 0.0001)
Priority 2: "bitcoin2013!" (P = 0.00008)
Priority 3: "crypto2017@" (P = 0.00006)
...
Priority N: "xK3m9$zQ" (P = 1e-15)
```

---

## Training Pipeline

### Input Requirements

| Requirement | Recommendation | Why |
|-------------|----------------|-----|
| Size | 100K - 50M passwords | Statistical significance |
| Format | Plaintext, one per line | Direct parsing |
| Duplicates | Include duplicates | Frequency matters |
| Source | Brain wallet specific | Domain relevance |

### Training Sources for Brain Wallets

1. **Known compromised brain wallets** (17,956 passwords) - HIGHEST VALUE
2. **Crypto exchange breaches** (if available)
3. **BitcoinTalk password leaks** (forum users likely brain wallet users)
4. **General password corpuses** with crypto keyword filtering

### Training Process

```python
# Pseudocode for PCFG training
class PCFGTrainer:
    def train(self, password_file: str):
        structures = defaultdict(int)
        terminals = defaultdict(lambda: defaultdict(int))

        for password in read_passwords(password_file):
            # Parse structure
            structure, segments = self.parse(password)
            structures[structure] += 1

            # Record terminal values
            for segment_type, value in segments:
                terminals[segment_type][value] += 1

        # Convert to probabilities
        self.structure_probs = self.normalize(structures)
        self.terminal_probs = {
            k: self.normalize(v) for k, v in terminals.items()
        }

    def parse(self, password: str) -> Tuple[str, List]:
        """Parse password into structure and segments."""
        segments = []
        current_type = None
        current_value = ""

        for char in password:
            char_type = self.classify(char)

            if char_type != current_type:
                if current_value:
                    segments.append((
                        f"{current_type}{len(current_value)}",
                        current_value
                    ))
                current_type = char_type
                current_value = char
            else:
                current_value += char

        if current_value:
            segments.append((
                f"{current_type}{len(current_value)}",
                current_value
            ))

        structure = " ".join(s[0] for s in segments)
        return structure, segments

    def classify(self, char: str) -> str:
        if char.isupper(): return 'U'
        if char.islower(): return 'L'
        if char.isdigit(): return 'D'
        return 'S'
```

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PCFG SUBSYSTEM                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        TRAINING PHASE                                 │   │
│  │                                                                       │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │   │
│  │  │  Password   │ →  │  Structure  │ →  │  Grammar    │              │   │
│  │  │   Corpus    │    │   Parser    │    │  Builder    │              │   │
│  │  └─────────────┘    └─────────────┘    └──────┬──────┘              │   │
│  │                                               │                       │   │
│  │                                               ▼                       │   │
│  │                                        ┌─────────────┐               │   │
│  │                                        │  Grammar    │               │   │
│  │                                        │   Rules     │               │   │
│  │                                        │   (.pcfg)   │               │   │
│  │                                        └──────┬──────┘               │   │
│  └───────────────────────────────────────────────┼──────────────────────┘   │
│                                                  │                          │
│  ┌───────────────────────────────────────────────┼──────────────────────┐   │
│  │                      GENERATION PHASE         │                       │   │
│  │                                               ▼                       │   │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │   │
│  │  │  Priority   │ ←  │  Candidate  │ ←  │  Grammar    │              │   │
│  │  │   Queue     │    │  Generator  │    │   Rules     │              │   │
│  │  └──────┬──────┘    └─────────────┘    └─────────────┘              │   │
│  │         │                                                             │   │
│  └─────────┼─────────────────────────────────────────────────────────────┘   │
│            │                                                                 │
│            ▼                                                                 │
│    [GPU CRACKING PIPELINE]                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## C++ Implementation

### Grammar Rules Data Structure

```cpp
// src/generators/pcfg.hpp

#pragma once

#include "../core/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>

namespace superflayer {
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
    static Grammar load(const std::string& path);

    /**
     * Save grammar to file.
     */
    void save(const std::string& path) const;

    /**
     * Get probability of a specific password.
     */
    double score(const std::string& password) const;
};

/**
 * PCFG Trainer - learns grammar from password corpus.
 */
class Trainer {
public:
    struct Config {
        size_t min_length = 4;
        size_t max_length = 64;
        double min_terminal_prob = 1e-9;  // Prune rare terminals
        bool detect_keyboard_patterns = true;
        bool detect_multiwords = true;
    };

    explicit Trainer(const Config& config = {}) : config_(config) {}

    /**
     * Train on a password file.
     */
    void train(const std::string& password_file);

    /**
     * Add passwords from memory.
     */
    void add_passwords(const std::vector<std::string>& passwords);

    /**
     * Build final grammar.
     */
    Grammar build_grammar() const;

private:
    Config config_;
    std::unordered_map<std::string, uint64_t> structure_counts_;
    std::unordered_map<std::string, std::unordered_map<std::string, uint64_t>> terminal_counts_;
    uint64_t total_passwords_ = 0;

    std::pair<std::string, std::vector<std::pair<std::string, std::string>>>
    parse_password(const std::string& password) const;

    char classify_char(char c) const;
};

/**
 * PCFG Generator - produces candidates in probability order.
 */
class Generator {
public:
    explicit Generator(const Grammar& grammar) : grammar_(grammar) {}

    /**
     * Generate next candidate.
     * Returns nullopt when exhausted (never for practical purposes).
     */
    std::optional<Candidate> next();

    /**
     * Generate batch of candidates.
     */
    std::vector<Candidate> next_batch(size_t count);

    /**
     * Reset generator to start.
     */
    void reset();

    /**
     * Get current probability threshold.
     */
    double current_probability() const;

private:
    const Grammar& grammar_;

    // Priority queue for probability-ordered generation
    struct GeneratorState {
        double probability;
        std::vector<size_t> terminal_indices;  // Current index for each non-terminal
        size_t structure_index;

        bool operator<(const GeneratorState& other) const {
            return probability < other.probability;  // Max-heap
        }
    };

    std::priority_queue<GeneratorState> state_queue_;
    bool initialized_ = false;

    void initialize();
    std::string expand_state(const GeneratorState& state) const;
};

/**
 * Keyboard pattern detector for enhanced PCFG.
 */
class KeyboardPatternDetector {
public:
    /**
     * Detect keyboard walks (e.g., "qwerty", "asdfgh").
     */
    static bool is_keyboard_walk(const std::string& s);

    /**
     * Get keyboard pattern type.
     */
    static std::string classify_pattern(const std::string& s);

private:
    static const std::vector<std::string> KEYBOARD_ROWS;
    static const std::vector<std::string> KEYBOARD_DIAGONALS;
};

/**
 * Multi-word detector for phrases.
 */
class MultiwordDetector {
public:
    explicit MultiwordDetector(const std::string& dictionary_path);

    /**
     * Split string into words if possible.
     */
    std::vector<std::string> detect_words(const std::string& s) const;

    /**
     * Check if string is a multi-word phrase.
     */
    bool is_multiword(const std::string& s) const;

private:
    std::unordered_set<std::string> dictionary_;
};

}  // namespace pcfg
}  // namespace superflayer
```

### Generator Implementation

```cpp
// src/generators/pcfg.cpp

#include "pcfg.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace superflayer {
namespace pcfg {

void Trainer::train(const std::string& password_file) {
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

std::pair<std::string, std::vector<std::pair<std::string, std::string>>>
Trainer::parse_password(const std::string& password) const {
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

char Trainer::classify_char(char c) const {
    if (c >= 'A' && c <= 'Z') return 'U';
    if (c >= 'a' && c <= 'z') return 'L';
    if (c >= '0' && c <= '9') return 'D';
    return 'S';
}

Grammar Trainer::build_grammar() const {
    Grammar grammar;

    // Build structure rules
    for (const auto& [structure, count] : structure_counts_) {
        double prob = static_cast<double>(count) / total_passwords_;

        // Parse structure into pattern
        std::vector<std::string> pattern;
        std::istringstream iss(structure);
        std::string token;
        while (iss >> token) {
            pattern.push_back(token);
        }

        grammar.structures.push_back({pattern, prob});
    }

    // Sort by probability
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

        // Sort by probability
        std::sort(nt.terminals.begin(), nt.terminals.end(),
                  [](const auto& a, const auto& b) {
                      return a.probability > b.probability;
                  });

        grammar.non_terminals[nt_name] = std::move(nt);
    }

    return grammar;
}

std::optional<Candidate> Generator::next() {
    if (!initialized_) {
        initialize();
    }

    if (state_queue_.empty()) {
        return std::nullopt;
    }

    GeneratorState state = state_queue_.top();
    state_queue_.pop();

    // Expand current state to password
    std::string password = expand_state(state);

    // Generate successor states
    // ... (increment terminal indices, push new states)

    return Candidate{
        .phrase = password,
        .priority = static_cast<float>(state.probability),
        .source = CandidateSource::PCFG_GENERATED,
        .rule_applied = ":"
    };
}

void Generator::initialize() {
    // Initialize with highest probability state for each structure
    for (size_t i = 0; i < grammar_.structures.size(); ++i) {
        const auto& structure = grammar_.structures[i];

        std::vector<size_t> indices(structure.pattern.size(), 0);
        double prob = structure.probability;

        for (const auto& nt_name : structure.pattern) {
            if (grammar_.non_terminals.count(nt_name)) {
                const auto& nt = grammar_.non_terminals.at(nt_name);
                if (!nt.terminals.empty()) {
                    prob *= nt.terminals[0].probability;
                }
            }
        }

        state_queue_.push({prob, indices, i});
    }

    initialized_ = true;
}

std::string Generator::expand_state(const GeneratorState& state) const {
    std::string result;
    const auto& pattern = grammar_.structures[state.structure_index].pattern;

    for (size_t i = 0; i < pattern.size(); ++i) {
        const auto& nt_name = pattern[i];
        if (grammar_.non_terminals.count(nt_name)) {
            const auto& nt = grammar_.non_terminals.at(nt_name);
            size_t idx = state.terminal_indices[i];
            if (idx < nt.terminals.size()) {
                result += nt.terminals[idx].value;
            }
        }
    }

    return result;
}

}  // namespace pcfg
}  // namespace superflayer
```

---

## Brain Wallet Specific Training

### Recommended Training Data

```
Priority 1: Known compromised brain wallets
            - 17,956 passwords from prior research
            - Direct evidence of vulnerable patterns

Priority 2: Crypto-filtered password leaks
            - RockYou/LinkedIn filtered for crypto keywords
            - "bitcoin", "crypto", "wallet", "satoshi", etc.

Priority 3: Forum-specific data
            - BitcoinTalk password patterns
            - Early crypto adopter culture

Priority 4: General password corpuses
            - Broad coverage
            - Lower weight in final grammar
```

### Training Command

```bash
# Train on brain wallet corpus
./superflayer --train \
    --input known_brain_wallets.txt \
    --input crypto_filtered_passwords.txt \
    --output brain_wallet.pcfg \
    --min-length 6 \
    --max-length 64 \
    --detect-multiwords \
    --detect-keyboard
```

---

## Performance Characteristics

### Generation Speed

| Mode | Speed | Use Case |
|------|-------|----------|
| Memory-cached | 50-100M/sec | Small grammars (<1GB RAM) |
| Streaming | 10-50M/sec | Large grammars |
| GPU-assisted | 500M+/sec | Structure expansion on GPU |

### Probability Coverage

| Candidates Generated | Probability Coverage |
|---------------------|---------------------|
| 10K | Top 0.1% of probability mass |
| 1M | Top 5% of probability mass |
| 100M | Top 30% of probability mass |
| 10B | Top 80% of probability mass |

### Memory Requirements

| Grammar Size | RAM Usage |
|--------------|-----------|
| 100K passwords trained | ~50 MB |
| 1M passwords trained | ~200 MB |
| 10M passwords trained | ~1 GB |
| 100M passwords trained | ~5 GB |

---

## Integration with Superflayer Pipeline

```cpp
// Example usage in main pipeline

#include "generators/pcfg.hpp"
#include "generators/passphrase_generator.hpp"

// 1. Train grammar (offline)
pcfg::Trainer trainer;
trainer.train("known_brain_wallets.txt");
trainer.train("crypto_passwords.txt");
auto grammar = trainer.build_grammar();
grammar.save("brain_wallet.pcfg");

// 2. Load and use in generation
auto grammar = pcfg::Grammar::load("brain_wallet.pcfg");
auto pcfg_generator = std::make_unique<pcfg::Generator>(grammar);

// 3. Wrap as PassphraseSource
class PCFGSource : public PassphraseSource {
    pcfg::Generator generator_;
public:
    PCFGSource(const pcfg::Grammar& g) : generator_(g) {}

    void generate(CandidateCallback callback) override {
        while (auto candidate = generator_.next()) {
            callback(std::move(*candidate));
        }
    }

    CandidateSource type() const override {
        return CandidateSource::PCFG_GENERATED;
    }

    size_t estimated_size() const override {
        return std::numeric_limits<size_t>::max();  // Infinite
    }
};

// 4. Add to main generator
PassphraseGenerator main_generator(priority_queue);
main_generator.add_source(std::make_unique<PCFGSource>(grammar));
```

---

## References

- [PCFG Cracker (lakiw)](https://github.com/lakiw/pcfg_cracker)
- [Password Cracking Using PCFGs (Weir et al.)](https://ieeexplore.ieee.org/document/5207658)
- [Next Gen PCFG Password Cracking](https://www.usenix.org/conference/usenixsecurity15/technical-sessions/presentation/weir)
