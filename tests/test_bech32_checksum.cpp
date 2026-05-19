/**
 * test_bech32_checksum.cpp -- v1.4.2 A.7 regression test.
 *
 * Pre-fix the brain-wallet UVRF builder and the runtime HitVerifier each
 * had their own bech32 decoder that SKIPPED the checksum bytes. A typo'd
 * bc1q address parsed to a deterministic but WRONG H160 and was silently
 * added to the UVRF / bloom.
 *
 * Vectors: a known-valid bc1q address must decode to a fixed 20-byte
 * program; the same address with a single-bit corruption in the checksum
 * (or any character) must decode to std::nullopt.
 *
 * Test vectors derived from BIP-173 examples + bitcoin-core fixtures.
 */

#include "../src/core/bech32.hpp"

#include <iostream>
#include <string>

using collider::bech32::decode_p2wpkh;

namespace {

int g_pass = 0;
int g_fail = 0;

void expect_decode_succeeds(const std::string& addr,
                            const std::array<uint8_t, 20>& expected,
                            const std::string& label) {
    auto r = decode_p2wpkh(addr, "bc");
    if (!r.has_value()) {
        ++g_fail;
        std::cerr << "[FAIL] " << label << " -- decode returned nullopt for valid addr\n";
        return;
    }
    if (*r != expected) {
        ++g_fail;
        std::cerr << "[FAIL] " << label << " -- decoded H160 mismatch\n";
        return;
    }
    ++g_pass;
    std::cout << "[ok  ] " << label << "\n";
}

void expect_decode_fails(const std::string& addr, const std::string& label) {
    auto r = decode_p2wpkh(addr, "bc");
    if (r.has_value()) {
        ++g_fail;
        std::cerr << "[FAIL] " << label << " -- decode succeeded on invalid input\n";
        std::cerr << "       decoded H160 = ";
        for (uint8_t b : *r) {
            char hex[3];
            snprintf(hex, sizeof(hex), "%02x", b);
            std::cerr << hex;
        }
        std::cerr << "\n";
        return;
    }
    ++g_pass;
    std::cout << "[ok  ] " << label << " (correctly rejected)\n";
}

}  // namespace

int main() {
    std::cout << "test_bech32_checksum (v1.4.2 A.7 regression suite)\n";

    // ---- Valid addresses (BIP-173 reference vectors) ----

    // BIP-173 example: bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4
    //   HRP = bc, witness version 0, program = 751e76e8199196d454941c45d1b3a323f1433bd6
    expect_decode_succeeds(
        "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
        {{0x75, 0x1E, 0x76, 0xE8, 0x19, 0x91, 0x96, 0xD4, 0x54, 0x94,
          0x1C, 0x45, 0xD1, 0xB3, 0xA3, 0x23, 0xF1, 0x43, 0x3B, 0xD6}},
        "BIP-173 vector 1 (lowercase)");

    // Same address all-uppercase (BIP-173 allows this).
    expect_decode_succeeds(
        "BC1QW508D6QEJXTDG4Y5R3ZARVARY0C5XW7KV8F3T4",
        {{0x75, 0x1E, 0x76, 0xE8, 0x19, 0x91, 0x96, 0xD4, 0x54, 0x94,
          0x1C, 0x45, 0xD1, 0xB3, 0xA3, 0x23, 0xF1, 0x43, 0x3B, 0xD6}},
        "BIP-173 vector 1 (all uppercase)");

    // ---- Invalid: checksum corruption ----

    // Original: bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4
    // Flip one char in checksum (last 6 chars): 't4' -> 't5'
    expect_decode_fails(
        "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t5",
        "checksum bit flip (last char)");

    // Flip one char in data portion that changes meaning -- the recomputed
    // checksum would differ, so decode must fail.
    expect_decode_fails(
        "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3z4",
        "data bit flip");

    // ---- Invalid: mixed case ----
    expect_decode_fails(
        "Bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
        "mixed case (BIP-173 forbids)");

    // ---- Invalid: bad HRP ----
    expect_decode_fails(
        "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
        "wrong HRP (testnet 'tb' against mainnet 'bc')");

    // ---- Invalid: bad characters ----
    // 'b' is not in the bech32 alphabet
    expect_decode_fails(
        "bc1bw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
        "invalid character ('b' not in alphabet)");

    // ---- Invalid: too short ----
    expect_decode_fails("bc1q", "too short");
    expect_decode_fails("bc1", "minimal HRP-only");
    expect_decode_fails("", "empty");

    // ---- Invalid: too long (BIP-173 90-char max) ----
    {
        std::string too_long = "bc1q";
        for (int i = 0; i < 90; ++i) too_long += "q";  // 4 + 90 = 94 chars
        expect_decode_fails(too_long, "exceeds 90-char BIP-173 limit");
    }

    // ---- Invalid: no separator ----
    expect_decode_fails(
        "bcqw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
        "missing '1' separator");

    // ---- Invalid: non-zero witness version (would need bech32m) ----
    // bc1p... addresses are P2TR (BIP-350), use witness version 1.
    // Even if checksum *somehow* matched with bech32, we should reject because
    // witness version != 0.
    expect_decode_fails(
        "bc1p0xlxvlhemja6c4dqv22uapctqupfhlxm9h8z3k2e72q4k9hcz7vqzk5jj0",
        "witness version 1 (would be P2TR/bech32m, not handled here)");

    std::cout << "Summary: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
