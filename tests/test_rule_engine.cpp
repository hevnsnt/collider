/**
 * Rule Engine Tests
 *
 * Tests for the hashcat-compatible rule engine.
 */

#include "../src/core/rule_engine.hpp"
#include <iostream>
#include <cassert>

using namespace collider;

void test_basic_operations() {
    RuleEngine engine;

    // Passthrough
    assert(engine.apply("password", ":") == "password");

    // Case modifications
    assert(engine.apply("Password", "l") == "password");
    assert(engine.apply("Password", "u") == "PASSWORD");
    assert(engine.apply("PASSWORD", "c") == "Password");
    assert(engine.apply("password", "C") == "pASSWORD");
    assert(engine.apply("Password", "t") == "pASSWORD");

    // Reverse
    assert(engine.apply("password", "r") == "drowssap");

    // Duplicate
    assert(engine.apply("pass", "d") == "passpass");

    std::cout << "[PASS] Basic operations\n";
}

void test_append_prepend() {
    RuleEngine engine;

    // Append
    assert(engine.apply("password", "$1") == "password1");
    assert(engine.apply("password", "$!") == "password!");
    assert(engine.apply("password", "$1$2$3") == "password123");

    // Prepend
    assert(engine.apply("password", "^1") == "1password");
    assert(engine.apply("password", "^!") == "!password");
    assert(engine.apply("password", "^3^2^1") == "123password");

    std::cout << "[PASS] Append/Prepend operations\n";
}

void test_delete_operations() {
    RuleEngine engine;

    // Delete first
    assert(engine.apply("password", "[") == "assword");

    // Delete last
    assert(engine.apply("password", "]") == "passwor");

    // Delete at position
    assert(engine.apply("password", "D0") == "assword");
    assert(engine.apply("password", "D4") == "passwrd");

    std::cout << "[PASS] Delete operations\n";
}

void test_substitution() {
    RuleEngine engine;

    // Replace all
    assert(engine.apply("password", "sa4") == "p4ssword");
    assert(engine.apply("aardvark", "sa4") == "44rdv4rk");
    assert(engine.apply("password", "se3") == "password");  // No 'e' in password
    assert(engine.apply("hello", "se3") == "h3llo");

    // Leet speak chain
    assert(engine.apply("password", "sa4so0") == "p4ssw0rd");

    std::cout << "[PASS] Substitution operations\n";
}

void test_toggle_at_position() {
    RuleEngine engine;

    assert(engine.apply("password", "T0") == "Password");
    assert(engine.apply("password", "T4") == "passWord");
    assert(engine.apply("PASSWORD", "T0") == "pASSWORD");

    std::cout << "[PASS] Toggle at position\n";
}

void test_combined_rules() {
    RuleEngine engine;

    // Capitalize and append
    assert(engine.apply("password", "c$1") == "Password1");
    assert(engine.apply("password", "c$1$2$3") == "Password123");
    assert(engine.apply("password", "c$!") == "Password!");

    // Lowercase and append
    assert(engine.apply("PASSWORD", "l$1$2$3") == "password123");

    // Complex combinations
    assert(engine.apply("password", "c$2$0$0$9") == "Password2009");
    assert(engine.apply("bitcoin", "c$2$0$1$3") == "Bitcoin2013");

    std::cout << "[PASS] Combined rules\n";
}

void test_crypto_rules() {
    RuleEngine engine;

    // Crypto-specific rules
    assert(engine.apply("bitcoin", "c$b$t$c") == "Bitcoinbtc");
    assert(engine.apply("bitcoin", "$B$T$C") == "bitcoinBTC");
    assert(engine.apply("satoshi", "c$2$0$0$9") == "Satoshi2009");

    // HODL
    assert(engine.apply("crypto", "$h$o$d$l") == "cryptohodl");

    std::cout << "[PASS] Crypto-specific rules\n";
}

void test_rotate_operations() {
    RuleEngine engine;

    // Rotate left
    assert(engine.apply("password", "{") == "asswordp");

    // Rotate right
    assert(engine.apply("password", "}") == "dpasswor");

    std::cout << "[PASS] Rotate operations\n";
}

void test_swap_operations() {
    RuleEngine engine;

    // Swap first two
    assert(engine.apply("password", "k") == "apssword");

    // Swap last two
    assert(engine.apply("password", "K") == "passwodr");

    // Swap at positions
    assert(engine.apply("password", "*01") == "apssword");
    assert(engine.apply("password", "*07") == "dassworP");

    std::cout << "[PASS] Swap operations\n";
}

void test_reflect() {
    RuleEngine engine;

    // Reflect (append reversed)
    assert(engine.apply("pass", "f") == "passssap");
    assert(engine.apply("abc", "f") == "abccba");

    std::cout << "[PASS] Reflect operation\n";
}

void test_apply_all() {
    RuleEngine engine;

    std::vector<std::string> rules = {":", "l", "u", "c", "$1"};
    auto results = engine.apply_all("Password", rules);

    assert(results.size() == 5);
    assert(results[0] == "Password");
    assert(results[1] == "password");
    assert(results[2] == "PASSWORD");
    assert(results[3] == "Password");
    assert(results[4] == "Password1");

    std::cout << "[PASS] Apply all rules\n";
}

void test_best64_rules() {
    RuleEngine engine;

    // Test some rules from best64
    for (const auto& rule : builtin_rules::BEST64) {
        // Should not throw
        std::string result = engine.apply("testword", rule);
        assert(!result.empty() || rule == "@t");  // @t purges 't', might empty result
    }

    std::cout << "[PASS] Best64 rules\n";
}

void test_crypto_builtin_rules() {
    RuleEngine engine;

    // Test crypto rules
    for (const auto& rule : builtin_rules::CRYPTO_RULES) {
        std::string result = engine.apply("bitcoin", rule);
        // Should complete without error
    }

    std::cout << "[PASS] Crypto builtin rules\n";
}

int main() {
    std::cout << "=== Rule Engine Tests ===\n\n";

    test_basic_operations();
    test_append_prepend();
    test_delete_operations();
    test_substitution();
    test_toggle_at_position();
    test_combined_rules();
    test_crypto_rules();
    test_rotate_operations();
    test_swap_operations();
    test_reflect();
    test_apply_all();
    test_best64_rules();
    test_crypto_builtin_rules();

    std::cout << "\n=== All Tests Passed ===\n";
    return 0;
}
