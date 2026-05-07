/**
 * GPU Rule Engine - Hashcat-Compatible Rule Bytecode
 *
 * Compiles hashcat text rules into GPU-executable bytecode.
 * Enables rule application directly on GPU, avoiding CPU round-trips.
 *
 * ARCHITECTURE:
 * - Rules are compiled to compact bytecode (max 256 bytes per rule)
 * - Wordlist uploaded ONCE to GPU VRAM
 * - GPU applies rules in parallel: (word_idx, rule_idx) -> candidate
 * - Each thread generates one candidate from one word+rule pair
 *
 * SUPPORTED OPERATIONS (subset matching hashcat):
 * - Case: l (lowercase), u (uppercase), c (capitalize), C (inverted cap), t (toggle)
 * - Append/Prepend: $X (append), ^X (prepend)
 * - Delete/Insert: D (delete last), [ (delete first), iNX (insert at N)
 * - Replace: sXY (replace X with Y), @X (purge X)
 * - Duplicate: d (duplicate), p (duplicate with append)
 * - Reverse: r (reverse)
 * - Rotate: { (rotate left), } (rotate right)
 * - Memory: M (memorize), Q (append memorized)
 * - Substring: 'N (truncate at N)
 * - Position ops: DN (delete at N), xNM (extract N chars from M)
 *
 * REFERENCE: hashcat rule_functions.c
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>

namespace collider {
namespace rules {

// Maximum length for password candidates
constexpr size_t MAX_PASSWORD_LEN = 64;
// Maximum bytecode size per rule
constexpr size_t MAX_RULE_BYTECODE = 256;
// Maximum rules in a rule set
constexpr size_t MAX_RULES = 65536;
// Maximum words in GPU wordlist
constexpr size_t MAX_WORDS_GPU = 16777216;  // 16M words

/**
 * GPU Rule Operation Codes
 *
 * Packed format: [opcode:8][arg1:8][arg2:8][arg3:8]
 * Most ops use 1-2 bytes, complex ops use up to 4.
 */
enum class RuleOp : uint8_t {
    // Terminator
    NOP = 0x00,       // No operation / end of rule

    // Case operations (no args)
    LOWERCASE = 0x01,   // l - lowercase entire string
    UPPERCASE = 0x02,   // u - uppercase entire string
    CAPITALIZE = 0x03,  // c - capitalize first, lowercase rest
    INVCAP = 0x04,      // C - lowercase first, uppercase rest
    TOGGLE_ALL = 0x05,  // t - toggle case of all chars
    TOGGLE_AT = 0x06,   // TN - toggle case at position N [arg1=N]

    // Append/Prepend (1 byte arg)
    APPEND = 0x10,      // $X - append char X [arg1=char]
    PREPEND = 0x11,     // ^X - prepend char X [arg1=char]

    // Delete operations
    DELETE_FIRST = 0x20,  // [ - delete first char
    DELETE_LAST = 0x21,   // ] - delete last char
    DELETE_AT = 0x22,     // DN - delete char at position N [arg1=N]
    DELETE_RANGE = 0x23,  // xNM - delete from N, length M [arg1=N, arg2=M]

    // Insert
    INSERT_AT = 0x24,     // iNX - insert X at position N [arg1=N, arg2=char]

    // Replace/Substitute
    REPLACE = 0x30,       // sXY - replace all X with Y [arg1=X, arg2=Y]
    PURGE = 0x31,         // @X - purge/delete all X [arg1=X]
    OVERWRITE_AT = 0x32,  // oNX - overwrite at N with X [arg1=N, arg2=char]

    // Duplication
    DUPLICATE = 0x40,     // d - duplicate entire word
    DUPLICATE_FIRST = 0x41, // z - duplicate first char
    DUPLICATE_LAST = 0x42,  // Z - duplicate last char
    DUPLICATE_ALL = 0x43,   // q - duplicate every char
    DUPLICATE_N = 0x44,     // pN - append duplicated N times [arg1=N]

    // Reverse/Rotate
    REVERSE = 0x50,       // r - reverse string
    ROTATE_LEFT = 0x51,   // { - rotate left (first char to end)
    ROTATE_RIGHT = 0x52,  // } - rotate right (last char to front)

    // Memory operations (for rule chaining)
    MEMORIZE = 0x60,      // M - save current state
    APPEND_MEMORY = 0x61, // Q - append saved state

    // Truncate/Extract
    TRUNCATE_AT = 0x70,   // 'N - truncate at position N [arg1=N]
    EXTRACT = 0x71,       // xNM - extract M chars starting at N [arg1=N, arg2=M]

    // Positional character operations
    SWAP_FIRST_LAST = 0x80, // k - swap first two chars
    SWAP_LAST_PAIR = 0x81,  // K - swap last two chars
    SWAP_AT = 0x82,         // *NM - swap chars at N and M [arg1=N, arg2=M]

    // Conditional (skip rule if condition not met)
    REJECT_LESS = 0x90,   // <N - reject if length < N [arg1=N]
    REJECT_GREATER = 0x91, // >N - reject if length > N [arg1=N]
    REJECT_CONTAIN = 0x92, // !X - reject if contains X [arg1=X]
    REJECT_NOT_CONTAIN = 0x93, // /X - reject if not contains X [arg1=X]
    REJECT_EQUALS = 0x94,  // (N - reject if length != N [arg1=N]

    // Extended ops
    BITWISE_LEFT = 0xA0,  // L - bitwise shift left
    BITWISE_RIGHT = 0xA1, // R - bitwise shift right

    // Passthrough
    PASSTHROUGH = 0xFF,   // : - do nothing (identity rule)
};

/**
 * Compiled rule bytecode.
 */
struct CompiledRule {
    uint8_t bytecode[MAX_RULE_BYTECODE];
    size_t length = 0;           // Bytecode length in bytes
    std::string original;        // Original rule text (for debugging)
    bool valid = false;          // Was compilation successful?
    std::string error;           // Error message if invalid

    CompiledRule() {
        std::memset(bytecode, 0, sizeof(bytecode));
    }
};

/**
 * GPU-uploadable rule set.
 */
struct GPURuleSet {
    std::vector<CompiledRule> rules;
    std::vector<uint8_t> packed_bytecode;   // All rules packed contiguously
    std::vector<uint32_t> rule_offsets;     // Offset into packed_bytecode for each rule
    std::vector<uint16_t> rule_lengths;     // Length of each rule's bytecode

    size_t total_bytecode_size = 0;

    /**
     * Pack rules for GPU upload.
     */
    void pack() {
        packed_bytecode.clear();
        rule_offsets.clear();
        rule_lengths.clear();

        for (const auto& rule : rules) {
            if (!rule.valid) continue;

            rule_offsets.push_back(static_cast<uint32_t>(packed_bytecode.size()));
            rule_lengths.push_back(static_cast<uint16_t>(rule.length));

            for (size_t i = 0; i < rule.length; i++) {
                packed_bytecode.push_back(rule.bytecode[i]);
            }
        }

        total_bytecode_size = packed_bytecode.size();
    }
};

/**
 * Helper: Parse position character (0-9, A-Z)
 */
inline int parse_position(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return 10 + (c - 'A');
    if (c >= 'a' && c <= 'z') return 10 + (c - 'a');
    return -1;
}

/**
 * Rule Compiler - Converts hashcat text rules to GPU bytecode.
 */
class RuleCompiler {
public:
    /**
     * Compile a single hashcat rule to bytecode.
     */
    static CompiledRule compile(const std::string& rule_text) {
        CompiledRule result;
        result.original = rule_text;

        size_t pos = 0;
        size_t bytecode_pos = 0;

        auto emit = [&](uint8_t byte) {
            if (bytecode_pos >= MAX_RULE_BYTECODE - 1) {
                result.error = "Rule too long (max " + std::to_string(MAX_RULE_BYTECODE) + " bytes)";
                return false;
            }
            result.bytecode[bytecode_pos++] = byte;
            return true;
        };

        auto emit_op = [&](RuleOp op) {
            return emit(static_cast<uint8_t>(op));
        };

        while (pos < rule_text.length()) {
            char c = rule_text[pos++];

            // Skip whitespace
            if (c == ' ' || c == '\t') continue;

            // Skip comments
            if (c == '#') break;

            switch (c) {
                // Passthrough
                case ':':
                    emit_op(RuleOp::PASSTHROUGH);
                    break;

                // Case operations
                case 'l':
                    emit_op(RuleOp::LOWERCASE);
                    break;
                case 'u':
                    emit_op(RuleOp::UPPERCASE);
                    break;
                case 'c':
                    emit_op(RuleOp::CAPITALIZE);
                    break;
                case 'C':
                    emit_op(RuleOp::INVCAP);
                    break;
                case 't':
                    emit_op(RuleOp::TOGGLE_ALL);
                    break;
                case 'T':
                    if (pos < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        if (n >= 0) {
                            emit_op(RuleOp::TOGGLE_AT);
                            emit(static_cast<uint8_t>(n));
                        }
                    }
                    break;

                // Append/Prepend
                case '$':
                    if (pos < rule_text.length()) {
                        emit_op(RuleOp::APPEND);
                        emit(static_cast<uint8_t>(rule_text[pos++]));
                    }
                    break;
                case '^':
                    if (pos < rule_text.length()) {
                        emit_op(RuleOp::PREPEND);
                        emit(static_cast<uint8_t>(rule_text[pos++]));
                    }
                    break;

                // Delete
                case '[':
                    emit_op(RuleOp::DELETE_FIRST);
                    break;
                case ']':
                    emit_op(RuleOp::DELETE_LAST);
                    break;
                case 'D':
                    if (pos < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        if (n >= 0) {
                            emit_op(RuleOp::DELETE_AT);
                            emit(static_cast<uint8_t>(n));
                        }
                    }
                    break;

                // Insert
                case 'i':
                    if (pos + 1 < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        char x = rule_text[pos++];
                        if (n >= 0) {
                            emit_op(RuleOp::INSERT_AT);
                            emit(static_cast<uint8_t>(n));
                            emit(static_cast<uint8_t>(x));
                        }
                    }
                    break;

                // Overwrite
                case 'o':
                    if (pos + 1 < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        char x = rule_text[pos++];
                        if (n >= 0) {
                            emit_op(RuleOp::OVERWRITE_AT);
                            emit(static_cast<uint8_t>(n));
                            emit(static_cast<uint8_t>(x));
                        }
                    }
                    break;

                // Replace
                case 's':
                    if (pos + 1 < rule_text.length()) {
                        char x = rule_text[pos++];
                        char y = rule_text[pos++];
                        emit_op(RuleOp::REPLACE);
                        emit(static_cast<uint8_t>(x));
                        emit(static_cast<uint8_t>(y));
                    }
                    break;

                // Purge
                case '@':
                    if (pos < rule_text.length()) {
                        emit_op(RuleOp::PURGE);
                        emit(static_cast<uint8_t>(rule_text[pos++]));
                    }
                    break;

                // Duplicate
                case 'd':
                    emit_op(RuleOp::DUPLICATE);
                    break;
                case 'q':
                    emit_op(RuleOp::DUPLICATE_ALL);
                    break;
                case 'z':
                    if (pos < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        if (n >= 0) {
                            emit_op(RuleOp::DUPLICATE_FIRST);
                            emit(static_cast<uint8_t>(n));
                        }
                    }
                    break;
                case 'Z':
                    if (pos < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        if (n >= 0) {
                            emit_op(RuleOp::DUPLICATE_LAST);
                            emit(static_cast<uint8_t>(n));
                        }
                    }
                    break;
                case 'p':
                    if (pos < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        if (n >= 0) {
                            emit_op(RuleOp::DUPLICATE_N);
                            emit(static_cast<uint8_t>(n));
                        }
                    }
                    break;

                // Reverse/Rotate
                case 'r':
                    emit_op(RuleOp::REVERSE);
                    break;
                case '{':
                    emit_op(RuleOp::ROTATE_LEFT);
                    break;
                case '}':
                    emit_op(RuleOp::ROTATE_RIGHT);
                    break;

                // Memory
                case 'M':
                    emit_op(RuleOp::MEMORIZE);
                    break;
                case 'Q':
                    emit_op(RuleOp::APPEND_MEMORY);
                    break;

                // Truncate/Extract
                case '\'':
                    if (pos < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        if (n >= 0) {
                            emit_op(RuleOp::TRUNCATE_AT);
                            emit(static_cast<uint8_t>(n));
                        }
                    }
                    break;
                case 'x':
                    if (pos + 1 < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        int m = parse_position(rule_text[pos++]);
                        if (n >= 0 && m >= 0) {
                            emit_op(RuleOp::EXTRACT);
                            emit(static_cast<uint8_t>(n));
                            emit(static_cast<uint8_t>(m));
                        }
                    }
                    break;

                // Swap
                case 'k':
                    emit_op(RuleOp::SWAP_FIRST_LAST);
                    break;
                case 'K':
                    emit_op(RuleOp::SWAP_LAST_PAIR);
                    break;
                case '*':
                    if (pos + 1 < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        int m = parse_position(rule_text[pos++]);
                        if (n >= 0 && m >= 0) {
                            emit_op(RuleOp::SWAP_AT);
                            emit(static_cast<uint8_t>(n));
                            emit(static_cast<uint8_t>(m));
                        }
                    }
                    break;

                // Conditionals
                case '<':
                    if (pos < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        if (n >= 0) {
                            emit_op(RuleOp::REJECT_LESS);
                            emit(static_cast<uint8_t>(n));
                        }
                    }
                    break;
                case '>':
                    if (pos < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        if (n >= 0) {
                            emit_op(RuleOp::REJECT_GREATER);
                            emit(static_cast<uint8_t>(n));
                        }
                    }
                    break;
                case '!':
                    if (pos < rule_text.length()) {
                        emit_op(RuleOp::REJECT_CONTAIN);
                        emit(static_cast<uint8_t>(rule_text[pos++]));
                    }
                    break;
                case '/':
                    if (pos < rule_text.length()) {
                        emit_op(RuleOp::REJECT_NOT_CONTAIN);
                        emit(static_cast<uint8_t>(rule_text[pos++]));
                    }
                    break;
                case '(':
                    if (pos < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        if (n >= 0) {
                            emit_op(RuleOp::REJECT_EQUALS);
                            emit(static_cast<uint8_t>(n));
                        }
                    }
                    break;

                // Bitwise
                case 'L':
                    if (pos < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        if (n >= 0) {
                            emit_op(RuleOp::BITWISE_LEFT);
                            emit(static_cast<uint8_t>(n));
                        }
                    }
                    break;
                case 'R':
                    if (pos < rule_text.length()) {
                        int n = parse_position(rule_text[pos++]);
                        if (n >= 0) {
                            emit_op(RuleOp::BITWISE_RIGHT);
                            emit(static_cast<uint8_t>(n));
                        }
                    }
                    break;

                default:
                    // Unknown operation - skip (hashcat compatibility)
                    break;
            }
        }

        // Terminate with NOP
        emit_op(RuleOp::NOP);

        result.length = bytecode_pos;
        result.valid = result.error.empty();

        return result;
    }

    /**
     * Compile multiple rules from text lines.
     */
    static GPURuleSet compile_rules(const std::vector<std::string>& rule_texts) {
        GPURuleSet set;

        for (const auto& text : rule_texts) {
            // Skip empty lines and comments
            if (text.empty() || text[0] == '#') continue;

            auto compiled = compile(text);
            if (compiled.valid) {
                set.rules.push_back(std::move(compiled));
            }
        }

        set.pack();
        return set;
    }

    /**
     * Load and compile rules from file.
     */
    static GPURuleSet load_from_file(const std::string& path) {
        std::vector<std::string> lines;
        std::ifstream file(path);

        if (!file.is_open()) {
            std::cerr << "[!] Cannot open rule file: " << path << "\n";
            return GPURuleSet();
        }

        std::string line;
        while (std::getline(file, line)) {
            // Trim trailing whitespace
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
                                      line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            if (!line.empty()) {
                lines.push_back(line);
            }
        }

        return compile_rules(lines);
    }
};

}  // namespace rules
}  // namespace collider
