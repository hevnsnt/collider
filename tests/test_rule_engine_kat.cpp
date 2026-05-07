/**
 * test_rule_engine_kat -- Hashcat rule engine known-answer test.
 *
 * Distinct from tests/test_rule_engine.cpp (which exercises a broad menu of
 * operations as the engine evolves), this test pins the engine to a fixed
 * external reference: each entry is an (input word, hashcat rule, expected
 * output) triple drawn from the hashcat rule documentation. If the engine
 * starts disagreeing with hashcat's behaviour for any of these canonical
 * cases, the project's "compatible with hashcat .rule files" claim is broken
 * and this test fails loudly.
 *
 * Sources:
 *   - hashcat rule reference: https://hashcat.net/wiki/doku.php?id=rule_based_attack
 *   - These specific (input, rule, output) triples are documented examples or
 *     trivial extrapolations of the rule semantics on hashcat's own page.
 *
 * Returns 0 on pass, 1 on fail.
 */

#include "../src/core/rule_engine.hpp"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

struct RuleKAT {
    const char* description;
    const char* input;
    const char* rule;
    const char* expected;
};

// Each row is a hashcat-spec known-answer.
static const RuleKAT KATS[] = {
    // ----------------------------------------------------------------------
    // Identity / case
    // ----------------------------------------------------------------------
    {"noop ':' is identity",                   "password",      ":",      "password"},
    {"'l' lowercases all",                     "PASSWORD",      "l",      "password"},
    {"'l' on already-lowercase is no-op",      "password",      "l",      "password"},
    {"'u' uppercases all",                     "password",      "u",      "PASSWORD"},
    {"'u' on mixed input",                     "PassWord",      "u",      "PASSWORD"},
    {"'c' capitalizes first, lowercases rest", "password",      "c",      "Password"},
    {"'c' on all-caps input",                  "PASSWORD",      "c",      "Password"},
    {"'C' lowercases first, uppercases rest",  "password",      "C",      "pASSWORD"},
    {"'t' toggles case of every char",         "PaSsWoRd",      "t",      "pAsSwOrD"},

    // ----------------------------------------------------------------------
    // Reversal / duplication
    // ----------------------------------------------------------------------
    {"'r' reverses the word",                  "password",      "r",      "drowssap"},
    {"'r' on palindrome is identity",          "level",         "r",      "level"},
    {"'d' duplicates the word",                "pass",          "d",      "passpass"},
    {"'d' on empty stays empty",               "",              "d",      ""},
    {"'f' reflects (append reversed)",         "ab",            "f",      "abba"},

    // ----------------------------------------------------------------------
    // Append / prepend
    // ----------------------------------------------------------------------
    {"'$1' appends '1'",                       "password",      "$1",     "password1"},
    {"'$!' appends '!'",                       "password",      "$!",     "password!"},
    {"'$1$2$3' appends '123'",                 "password",      "$1$2$3", "password123"},
    {"'^1' prepends '1'",                      "password",      "^1",     "1password"},
    {"'^!' prepends '!'",                      "password",      "^!",     "!password"},
    // hashcat applies '^X' left-to-right, each one prepending to the current
    // string. So '^1^2' on "ab" yields prepend '1' -> "1ab" then prepend '2'
    // -> "21ab".
    {"'^1^2' prepends in left-to-right order", "ab",            "^1^2",   "21ab"},

    // ----------------------------------------------------------------------
    // Deletion
    // ----------------------------------------------------------------------
    {"'[' deletes first char",                 "password",      "[",      "assword"},
    {"']' deletes last char",                  "password",      "]",      "passwor"},
    {"'D0' deletes char at position 0",        "password",      "D0",     "assword"},
    {"'D2' deletes char at position 2",        "password",      "D2",     "pasword"},

    // ----------------------------------------------------------------------
    // Rotation
    // ----------------------------------------------------------------------
    {"'{' rotates left",                       "abcd",          "{",      "bcda"},
    {"'}' rotates right",                      "abcd",          "}",      "dabc"},

    // ----------------------------------------------------------------------
    // Substitution / purge
    // ----------------------------------------------------------------------
    {"'sa@' replaces 'a' with '@'",            "password",      "sa@",    "p@ssword"},
    {"'so0' replaces 'o' with '0'",            "boomstick",     "so0",    "b00mstick"},
    {"'@s' purges all 's'",                    "password",      "@s",     "paword"},

    // ----------------------------------------------------------------------
    // Position-indexed
    // ----------------------------------------------------------------------
    {"'T0' toggles case at position 0",        "password",      "T0",     "Password"},
    {"'T3' toggles case at position 3",        "password",      "T3",     "pasSword"},

    // ----------------------------------------------------------------------
    // Combinations (catches stateful interaction bugs)
    // ----------------------------------------------------------------------
    {"'c$1' capitalize then append '1'",       "password",      "c$1",    "Password1"},
    {"'l$!' lowercase then append '!'",        "PASSWORD",      "l$!",    "password!"},
    {"'u$!$!' uppercase then append '!!'",     "password",      "u$!$!",  "PASSWORD!!"},
};
static constexpr size_t NUM_KATS = sizeof(KATS) / sizeof(KATS[0]);

}  // namespace

int main() {
    collider::RuleEngine engine;

    int passed = 0;
    int failed = 0;

    printf("=== hashcat rule engine known-answer test ===\n");

    for (size_t i = 0; i < NUM_KATS; i++) {
        const RuleKAT& kat = KATS[i];
        std::string got;
        try {
            got = engine.apply(kat.input, kat.rule);
        } catch (const std::exception& e) {
            printf("  FAIL  [%s] threw exception: %s\n",
                   kat.description, e.what());
            failed++;
            continue;
        }

        const bool match = (got == kat.expected);
        if (match) {
            passed++;
        } else {
            failed++;
            printf("  FAIL  %s\n", kat.description);
            printf("        rule:     '%s'\n", kat.rule);
            printf("        input:    '%s'\n", kat.input);
            printf("        expected: '%s'\n", kat.expected);
            printf("        got:      '%s'\n", got.c_str());
        }
    }

    printf("Tested:  %zu rule vectors\n", NUM_KATS);
    printf("Correct: %d\n", passed);
    printf("Wrong:   %d\n", failed);

    if (failed == 0) {
        printf("PASS: rule engine matches hashcat reference for all KAT vectors.\n");
        return 0;
    } else {
        printf("FAIL: %d of %d rule applications disagree with hashcat reference.\n",
               failed, (int)NUM_KATS);
        return 1;
    }
}
