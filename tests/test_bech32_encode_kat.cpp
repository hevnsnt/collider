/**
 * test_bech32_encode_kat.cpp -- B1 / wire-v4 (2026-05-23).
 *
 * Pins the bech32 P2WPKH ENCODER against the same BIP-173 reference
 * vectors used by the collision-protocol server's Python encoder
 * (auth_v4.py / tests/test_auth_v4_kat.py). Wire-v4 requires the C++
 * client's bech32 string to match the server's bech32 string for the
 * same pubkey, byte-for-byte, otherwise the AUTH frame's worker_name
 * field fails the identity-binding check.
 *
 * If either side drifts (e.g. a polymod constant bug, a wrong
 * convertbits padding decision, a HRP off-by-one), this test catches
 * it on the C++ side before wire-v4 ever leaves the lab.
 *
 * Vectors:
 *   - BIP-173 reference: h160 = 751e76e8199196d454941c45d1b3a323f1433bd6
 *       mainnet "bc" -> bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4
 *       testnet "tb" -> tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx
 *   - Round-trip: encode(decode(x)) == x for the above addresses.
 *   - Edge: zero-vector h160 encodes to a stable bc1q... string and
 *     round-trips.
 */

#include "../src/core/bech32.hpp"

#include <array>
#include <iostream>
#include <string>

using collider::bech32::decode_p2wpkh;
using collider::bech32::encode_p2wpkh;

namespace {

int g_pass = 0;
int g_fail = 0;

std::array<uint8_t, 20> hex20(const char* hex) {
    std::array<uint8_t, 20> out{};
    for (int i = 0; i < 20; ++i) {
        unsigned hi = 0, lo = 0;
        char c = hex[i * 2];
        hi = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
        c = hex[i * 2 + 1];
        lo = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return out;
}

void expect_encode(const std::array<uint8_t, 20>& h160,
                   const std::string& hrp,
                   const std::string& expected,
                   const std::string& label) {
    auto r = encode_p2wpkh(h160, hrp);
    if (!r.has_value()) {
        ++g_fail;
        std::cerr << "[FAIL] " << label
                  << " -- encode returned nullopt\n";
        return;
    }
    if (*r != expected) {
        ++g_fail;
        std::cerr << "[FAIL] " << label
                  << " -- got=" << *r << "\n            want=" << expected << "\n";
        return;
    }
    ++g_pass;
    std::cout << "[PASS] " << label << " (" << *r << ")\n";
}

void expect_roundtrip(const std::string& addr, const std::string& hrp,
                      const std::string& label) {
    auto h = decode_p2wpkh(addr, hrp);
    if (!h.has_value()) {
        ++g_fail;
        std::cerr << "[FAIL] " << label << " -- decode failed\n";
        return;
    }
    auto re = encode_p2wpkh(*h, hrp);
    if (!re.has_value()) {
        ++g_fail;
        std::cerr << "[FAIL] " << label << " -- re-encode failed\n";
        return;
    }
    if (*re != addr) {
        ++g_fail;
        std::cerr << "[FAIL] " << label
                  << " -- round-trip mismatch: got=" << *re
                  << ", want=" << addr << "\n";
        return;
    }
    ++g_pass;
    std::cout << "[PASS] " << label << " (round-trip ok)\n";
}

}  // namespace

int main() {
    std::cout << "=== test_bech32_encode_kat (B1 wire-v4) ===\n";

    // BIP-173 reference vector, mainnet.
    auto h = hex20("751e76e8199196d454941c45d1b3a323f1433bd6");
    expect_encode(h, "bc",
                  "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4",
                  "BIP-173 mainnet vector");

    // BIP-173 reference vector, testnet.
    expect_encode(h, "tb",
                  "tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx",
                  "BIP-173 testnet vector");

    // Round-trip both.
    expect_roundtrip("bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4", "bc",
                     "mainnet round-trip");
    expect_roundtrip("tb1qw508d6qejxtdg4y5r3zarvary0c5xw7kxpjzsx", "tb",
                     "testnet round-trip");

    // Zero-hash160 edge: must encode + round-trip without crashing.
    std::array<uint8_t, 20> zeros{};
    auto rz = encode_p2wpkh(zeros, "bc");
    if (!rz.has_value() || rz->substr(0, 4) != "bc1q") {
        ++g_fail;
        std::cerr << "[FAIL] zero hash160 -- did not produce bc1q...\n";
    } else {
        ++g_pass;
        std::cout << "[PASS] zero hash160 (" << *rz << ")\n";
        expect_roundtrip(*rz, "bc", "zero hash160 round-trip");
    }

    std::cout << "\n=== Result: " << g_pass << " pass, " << g_fail
              << " fail ===\n";
    return g_fail == 0 ? 0 : 1;
}
