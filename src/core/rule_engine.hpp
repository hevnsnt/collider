/**
 * Collider Rule Engine
 *
 * Hashcat-compatible rule engine for passphrase mutation.
 * Implements the full hashcat rule language for maximum compatibility.
 */

#pragma once

#include "types.hpp"
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <cctype>
#include <algorithm>

namespace collider {

class RuleParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * Hashcat-compatible rule engine.
 *
 * Supports the core rule operations used in password cracking:
 * - Case modification (l, u, c, C, t, T)
 * - Character operations ($, ^, [, ], D, x, O, i, o, ', s, @)
 * - Duplication (d, p, r, f, {, })
 * - Memory operations (M, Q, X)
 * - Numeric operations (+, -, ., ,)
 * - Position operations (k, K, *, L, R, E)
 */
class RuleEngine {
public:
    RuleEngine() = default;

    /**
     * Apply a single rule to a word.
     *
     * @param word The base word to mutate
     * @param rule The hashcat rule string
     * @return The mutated word
     * @throws RuleParseError if the rule is invalid
     */
    std::string apply(std::string_view word, std::string_view rule) const {
        std::string result(word);
        std::string memory;  // For M/Q/X operations

        size_t i = 0;
        while (i < rule.size()) {
            char op = rule[i];

            switch (op) {
                // No-op / comment
                case ':':
                case ' ':
                    break;

                // Lowercase all
                case 'l':
                    for (char& c : result) c = std::tolower(c);
                    break;

                // Uppercase all
                case 'u':
                    for (char& c : result) c = std::toupper(c);
                    break;

                // Capitalize first, lowercase rest
                case 'c':
                    for (size_t j = 0; j < result.size(); ++j) {
                        result[j] = (j == 0) ? std::toupper(result[j])
                                             : std::tolower(result[j]);
                    }
                    break;

                // Lowercase first, uppercase rest
                case 'C':
                    for (size_t j = 0; j < result.size(); ++j) {
                        result[j] = (j == 0) ? std::tolower(result[j])
                                             : std::toupper(result[j]);
                    }
                    break;

                // Toggle case of all
                case 't':
                    for (char& c : result) {
                        if (std::islower(c)) c = std::toupper(c);
                        else if (std::isupper(c)) c = std::tolower(c);
                    }
                    break;

                // Toggle case at position N
                case 'T': {
                    size_t pos = parse_position(rule, ++i);
                    if (pos < result.size()) {
                        if (std::islower(result[pos]))
                            result[pos] = std::toupper(result[pos]);
                        else if (std::isupper(result[pos]))
                            result[pos] = std::tolower(result[pos]);
                    }
                    break;
                }

                // Reverse
                case 'r':
                    std::reverse(result.begin(), result.end());
                    break;

                // Duplicate entire word
                case 'd':
                    result += result;
                    break;

                // Duplicate word N times
                case 'p': {
                    size_t n = parse_position(rule, ++i);
                    std::string original = result;
                    for (size_t j = 0; j < n; ++j) {
                        result += original;
                    }
                    break;
                }

                // Reflect (append reversed)
                case 'f':
                    result += std::string(result.rbegin(), result.rend());
                    break;

                // Rotate left
                case '{':
                    if (!result.empty()) {
                        char first = result.front();
                        result.erase(0, 1);
                        result += first;
                    }
                    break;

                // Rotate right
                case '}':
                    if (!result.empty()) {
                        char last = result.back();
                        result.pop_back();
                        result.insert(result.begin(), last);
                    }
                    break;

                // Append character
                case '$':
                    if (++i < rule.size()) {
                        result += rule[i];
                    }
                    break;

                // Prepend character
                case '^':
                    if (++i < rule.size()) {
                        result.insert(result.begin(), rule[i]);
                    }
                    break;

                // Delete first character
                case '[':
                    if (!result.empty()) {
                        result.erase(0, 1);
                    }
                    break;

                // Delete last character
                case ']':
                    if (!result.empty()) {
                        result.pop_back();
                    }
                    break;

                // Delete character at position N
                case 'D': {
                    size_t pos = parse_position(rule, ++i);
                    if (pos < result.size()) {
                        result.erase(pos, 1);
                    }
                    break;
                }

                // Extract substring [N:N+M]
                case 'x': {
                    size_t start = parse_position(rule, ++i);
                    size_t len = parse_position(rule, ++i);
                    if (start < result.size()) {
                        result = result.substr(start, len);
                    } else {
                        result.clear();
                    }
                    break;
                }

                // Delete M characters starting at position N
                case 'O': {
                    size_t start = parse_position(rule, ++i);
                    size_t len = parse_position(rule, ++i);
                    if (start < result.size()) {
                        result.erase(start, len);
                    }
                    break;
                }

                // Insert character X at position N
                case 'i': {
                    size_t pos = parse_position(rule, ++i);
                    if (++i < rule.size() && pos <= result.size()) {
                        result.insert(pos, 1, rule[i]);
                    }
                    break;
                }

                // Overwrite character at position N with X
                case 'o': {
                    size_t pos = parse_position(rule, ++i);
                    if (++i < rule.size() && pos < result.size()) {
                        result[pos] = rule[i];
                    }
                    break;
                }

                // Truncate at position N
                case '\'': {
                    size_t pos = parse_position(rule, ++i);
                    if (pos < result.size()) {
                        result.resize(pos);
                    }
                    break;
                }

                // Replace all X with Y
                case 's': {
                    if (i + 2 < rule.size()) {
                        char from = rule[++i];
                        char to = rule[++i];
                        for (char& c : result) {
                            if (c == from) c = to;
                        }
                    }
                    break;
                }

                // Purge all instances of X
                case '@': {
                    if (++i < rule.size()) {
                        char purge = rule[i];
                        result.erase(
                            std::remove(result.begin(), result.end(), purge),
                            result.end()
                        );
                    }
                    break;
                }

                // Duplicate first character
                case 'z': {
                    size_t n = parse_position(rule, ++i);
                    if (!result.empty()) {
                        result.insert(0, n, result.front());
                    }
                    break;
                }

                // Duplicate last character
                case 'Z': {
                    size_t n = parse_position(rule, ++i);
                    if (!result.empty()) {
                        result.append(n, result.back());
                    }
                    break;
                }

                // Increment ASCII at position N
                case '+': {
                    size_t pos = parse_position(rule, ++i);
                    if (pos < result.size()) {
                        result[pos]++;
                    }
                    break;
                }

                // Decrement ASCII at position N
                case '-': {
                    size_t pos = parse_position(rule, ++i);
                    if (pos < result.size()) {
                        result[pos]--;
                    }
                    break;
                }

                // Swap first two characters
                case 'k':
                    if (result.size() >= 2) {
                        std::swap(result[0], result[1]);
                    }
                    break;

                // Swap last two characters
                case 'K':
                    if (result.size() >= 2) {
                        std::swap(result[result.size()-2], result[result.size()-1]);
                    }
                    break;

                // Swap characters at positions N and M
                case '*': {
                    size_t pos1 = parse_position(rule, ++i);
                    size_t pos2 = parse_position(rule, ++i);
                    if (pos1 < result.size() && pos2 < result.size()) {
                        std::swap(result[pos1], result[pos2]);
                    }
                    break;
                }

                // Memory: save current word
                case 'M':
                    memory = result;
                    break;

                // Memory: append saved word
                case 'Q':
                    result += memory;
                    break;

                // Memory: insert saved word at position N
                case 'X': {
                    size_t pos = parse_position(rule, ++i);
                    size_t start = parse_position(rule, ++i);
                    size_t len = parse_position(rule, ++i);
                    if (pos <= result.size() && start < memory.size()) {
                        result.insert(pos, memory.substr(start, len));
                    }
                    break;
                }

                // Reject unless length equals N
                case '<':
                case '>':
                case '_':
                case '!':
                case '/':
                case '=':
                case '(':
                case ')':
                case '%':
                    // Skip rejection rules (we generate all candidates)
                    if (op == '/' || op == '=' || op == '(' || op == ')' || op == '%') {
                        ++i;  // Skip the operand
                    }
                    ++i;
                    break;

                default:
                    // Unknown rule - skip
                    break;
            }

            ++i;
        }

        return result;
    }

    /**
     * Apply multiple rules to a word.
     *
     * @param word The base word
     * @param rules Vector of rule strings
     * @return Vector of mutated words (may contain duplicates)
     */
    std::vector<std::string> apply_all(
        std::string_view word,
        const std::vector<std::string>& rules
    ) const {
        std::vector<std::string> results;
        results.reserve(rules.size());

        for (const auto& rule : rules) {
            try {
                results.push_back(apply(word, rule));
            } catch (const RuleParseError&) {
                // Skip invalid rules
            }
        }

        return results;
    }

    /**
     * Load rules from a hashcat rule file.
     *
     * @param path Path to .rule file
     * @return RuleSet containing parsed rules
     */
    static RuleSet load_ruleset(const std::string& path);

private:
    /**
     * Parse a position argument (0-9 or A-Z for 10-35).
     */
    size_t parse_position(std::string_view rule, size_t i) const {
        if (i >= rule.size()) return 0;

        char c = rule[i];
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'Z') return 10 + (c - 'A');
        if (c >= 'a' && c <= 'z') return 10 + (c - 'a');

        return 0;
    }
};

// -----------------------------------------------------------------------------
// Built-in Rule Sets
// -----------------------------------------------------------------------------

namespace builtin_rules {

/**
 * best64.rule - The most efficient 64 rules.
 */
inline const std::vector<std::string> BEST64 = {
    ":",           // Passthrough
    "l",           // Lowercase
    "u",           // Uppercase
    "c",           // Capitalize
    "t",           // Toggle case
    "r",           // Reverse
    "d",           // Duplicate
    "$1",          // Append 1
    "$2",          // Append 2
    "$3",          // Append 3
    "$!",          // Append !
    "$@",          // Append @
    "^1",          // Prepend 1
    "c$1",         // Capitalize + append 1
    "c$!",         // Capitalize + append !
    "c$1$2$3",     // Capitalize + append 123
    "l$1",         // Lowercase + append 1
    "l$1$2$3",     // Lowercase + append 123
    "sa4",         // a -> 4
    "se3",         // e -> 3
    "si1",         // i -> 1
    "so0",         // o -> 0
    "ss$",         // s -> $
    "sa4se3si1so0", // Leet speak
    "$1$2$3$4",    // Append 1234
    "$2$0$0$0",    // Append 2000
    "$2$0$1$0",    // Append 2010
    "$2$0$2$0",    // Append 2020
    "c$2$0$1$0",   // Capitalize + 2010
    "c$2$0$2$0",   // Capitalize + 2020
    "l$!$!",       // Lowercase + !!
    "u$1",         // Uppercase + 1
    "[",           // Delete first
    "]",           // Delete last
    "d$1",         // Duplicate + 1
    "r$1",         // Reverse + 1
    "c$1$2$3$!",   // Capitalize + 123!
    "$1$!",        // Append 1!
    "^1^2",        // Prepend 21
    "c$_",         // Capitalize + _
    "$_$1",        // Append _1
    "ss5",         // s -> 5
    "sa@",         // a -> @
    "T0",          // Toggle position 0
    "T1",          // Toggle position 1
    "T2",          // Toggle position 2
    "T3",          // Toggle position 3
    "k",           // Swap first two
    "K",           // Swap last two
    "*01",         // Swap positions 0 and 1
    "*12",         // Swap positions 1 and 2
    "f",           // Reflect
    "{",           // Rotate left
    "}",           // Rotate right
    "c$1$!",       // Capitalize + 1!
    "$!$1",        // Append !1
    "$.$1",        // Append .1
    "$2$0$0$1",    // Append 2001
    "$2$0$1$5",    // Append 2015
    "$2$0$1$8",    // Append 2018
    "$2$0$1$9",    // Append 2019
    "$2$0$2$1",    // Append 2021
    "$2$0$2$2",    // Append 2022
    "$2$0$2$3",    // Append 2023
    "$2$0$2$4",    // Append 2024
};

/**
 * Crypto-specific rules for brain wallet attacks.
 */
inline const std::vector<std::string> CRYPTO_RULES = {
    // Bitcoin-related years
    "$2$0$0$9",    // 2009 - Bitcoin genesis
    "$2$0$1$0",    // 2010
    "$2$0$1$1",    // 2011
    "$2$0$1$3",    // 2013 - First bubble
    "$2$0$1$7",    // 2017 - ATH
    "$2$0$2$1",    // 2021 - Recent ATH

    // Crypto keywords
    "$b$t$c",
    "$B$T$C",
    "$e$t$h",
    "$E$T$H",
    "$c$o$i$n",
    "$C$O$I$N",
    "$s$a$t$o$s$h$i",

    // Symbol substitutions
    "sB8",         // B -> 8 (Bitcoin)
    "so0sO0",      // o,O -> 0
    "ss$sS$",      // s,S -> $

    // Common crypto patterns
    "c$b$t$c",
    "c$B$T$C",
    "l$b$t$c",
    "u$B$T$C",
    "c$2$0$0$9",
    "c$2$0$1$3",
    "c$2$0$1$7",

    // Numbers
    "$4$2",        // 42
    "$6$9",        // 69
    "$1$2$3$4$5$6", // 123456

    // Symbols
    "$!$!$!",
    "$@$#$!",
    "$$$$$",

    // HODL meme
    "$h$o$d$l",
    "$H$O$D$L",
    "c$H$O$D$L",
};

}  // namespace builtin_rules

}  // namespace collider
