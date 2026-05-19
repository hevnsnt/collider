/**
 * SHA256, RIPEMD160, and Hash160 NIST/Bitcoin test vectors against the
 * production CPU crypto implementations in src/core/crypto_cpu.hpp.
 *
 * Why this exists: pre-fix the file shipped its own in-test reference
 * SHA256 and RIPEMD160 inside a `cpu_ref` namespace and ran the KATs
 * against those copies. Any regression in the production hashes would
 * have produced a passing test (theater). The fix is to call the same
 * `collider::cpu::SHA256` / `collider::cpu::RIPEMD160` /
 * `collider::cpu::compute_hash160` that the brain-wallet and kangaroo
 * paths use, so a real production regression turns this red.
 *
 * GPU-side hashes are exercised separately by test_gpu_hash160.cu and
 * test_brain_wallet_v2.cpp.
 */

#include "core/crypto_cpu.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

// =============================================================================
// Test Vectors
// =============================================================================

struct TestVector {
    const char* name;
    const char* input_hex;
    const char* expected_hex;
};

// SHA256 Test Vectors (NIST)
static const TestVector SHA256_TESTS[] = {
    { "Empty string", "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
    { "abc", "616263", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
    { "448 bits", "6162636462636465636465666465666765666768666768696768696a68696a6b696a6b6c6a6b6c6d6b6c6d6e6c6d6e6f6d6e6f706e6f7071",
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
    { "satoshi", "7361746f736869", "da2876b3eb31edb4436fa4650673fc6f01f90de2f1793c4ec332b2387b09726f" },
};

// RIPEMD160 Test Vectors
static const TestVector RIPEMD160_TESTS[] = {
    { "Empty string", "", "9c1185a5c5e9fc54612808977ee8f548b2258d31" },
    { "a", "61", "0bdc9d2d256b3ee9daae347be6f4dc835a467ffe" },
    { "abc", "616263", "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc" },
    { "message digest", "6d65737361676520646967657374", "5d0689ef49d2fae572b881b123a85ffa21595f36" },
};

// Hash160 Test Vectors (Bitcoin pubkey -> address). These are the
// canonical puzzle-1 / puzzle-2 hash160s computed by an independent
// reference (openssl + python hashlib, verified 2026-05-04). The
// production pipeline is privkey -> compressed pubkey -> SHA256 ->
// RIPEMD160; here we run the same RIPEMD160(SHA256(...)) over the
// already-encoded compressed pubkey bytes to keep this test focused
// on the hash chain rather than the EC math (which has its own KATs
// in test_ec_mul_known_answers.cu / test_kangaroo_small_puzzle.cu).
static const TestVector HASH160_TESTS[] = {
    { "Puzzle 1 pubkey", "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
      "751e76e8199196d454941c45d1b3a323f1433bd6" },
    { "Puzzle 2 pubkey", "02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
      "06afd46bcdfd22ef94ac122aa11f241244a37ecc" },
};

// =============================================================================
// Test Helpers
// =============================================================================

bool hex_to_bytes(const char* hex, uint8_t* bytes, size_t* len) {
    *len = strlen(hex) / 2;
    for (size_t i = 0; i < *len; i++) {
        unsigned int val;
        if (sscanf(hex + i*2, "%2x", &val) != 1) return false;
        bytes[i] = (uint8_t)val;
    }
    return true;
}

void bytes_to_hex(const uint8_t* bytes, size_t len, char* hex) {
    for (size_t i = 0; i < len; i++) {
        std::snprintf(hex + i*2, 3, "%02x", bytes[i]);
    }
    hex[len*2] = '\0';
}

bool compare_hex(const uint8_t* result, size_t len, const char* expected_hex) {
    char result_hex[128];
    bytes_to_hex(result, len, result_hex);
    return strcmp(result_hex, expected_hex) == 0;
}

// Run production SHA256 over a contiguous byte buffer.
static collider::cpu::SHA256::Hash prod_sha256(const uint8_t* msg, size_t len) {
    return collider::cpu::SHA256::hash(msg, len);
}

// Run production RIPEMD160 over a contiguous byte buffer.
static collider::cpu::RIPEMD160::Hash prod_ripemd160(const uint8_t* msg, size_t len) {
    return collider::cpu::RIPEMD160::hash(msg, len);
}

// Production hash160 over an arbitrary buffer: RIPEMD160(SHA256(msg)).
// (collider::cpu::compute_hash160 takes a 32-byte privkey and runs the
// full privkey -> pubkey -> hash160 pipeline; for the hash-chain KATs
// we only want the hashing tail, so we drive the two primitives directly.)
static collider::cpu::RIPEMD160::Hash prod_hash160_chain(const uint8_t* msg, size_t len) {
    auto sha = collider::cpu::SHA256::hash(msg, len);
    return collider::cpu::RIPEMD160::hash(sha.data(), sha.size());
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    int passed = 0, failed = 0;
    uint8_t input[256];
    size_t input_len;

    std::cout << "=== SHA256 Test Vectors (production collider::cpu::SHA256) ===\n";
    for (const auto& test : SHA256_TESTS) {
        hex_to_bytes(test.input_hex, input, &input_len);
        auto digest = prod_sha256(input, input_len);

        if (compare_hex(digest.data(), digest.size(), test.expected_hex)) {
            std::cout << "  PASS: " << test.name << "\n";
            passed++;
        } else {
            char got[65];
            bytes_to_hex(digest.data(), digest.size(), got);
            std::cout << "  FAIL: " << test.name << "\n";
            std::cout << "    Expected: " << test.expected_hex << "\n";
            std::cout << "    Got:      " << got << "\n";
            failed++;
        }
    }

    std::cout << "\n=== RIPEMD160 Test Vectors (production collider::cpu::RIPEMD160) ===\n";
    for (const auto& test : RIPEMD160_TESTS) {
        hex_to_bytes(test.input_hex, input, &input_len);
        auto digest = prod_ripemd160(input, input_len);

        if (compare_hex(digest.data(), digest.size(), test.expected_hex)) {
            std::cout << "  PASS: " << test.name << "\n";
            passed++;
        } else {
            char got[41];
            bytes_to_hex(digest.data(), digest.size(), got);
            std::cout << "  FAIL: " << test.name << "\n";
            std::cout << "    Expected: " << test.expected_hex << "\n";
            std::cout << "    Got:      " << got << "\n";
            failed++;
        }
    }

    std::cout << "\n=== Hash160 (Bitcoin) Test Vectors (production SHA256 + RIPEMD160) ===\n";
    for (const auto& test : HASH160_TESTS) {
        hex_to_bytes(test.input_hex, input, &input_len);
        auto digest = prod_hash160_chain(input, input_len);

        if (compare_hex(digest.data(), digest.size(), test.expected_hex)) {
            std::cout << "  PASS: " << test.name << "\n";
            passed++;
        } else {
            char got[41];
            bytes_to_hex(digest.data(), digest.size(), got);
            std::cout << "  FAIL: " << test.name << "\n";
            std::cout << "    Expected: " << test.expected_hex << "\n";
            std::cout << "    Got:      " << got << "\n";
            failed++;
        }
    }

    // ===== Full-pipeline Hash160 (privkey -> hash160) =====================
    // Exercises collider::cpu::compute_hash160, which is the entry point
    // used by the kangaroo H160 worker (src/core/kangaroo.hpp) and by
    // every CPU brain-wallet validation path. Pins privkey=1 against the
    // known Bitcoin compressed-pubkey hash160 (same vector as the KAT
    // above but driven through the EC pipeline this time).
    std::cout << "\n=== Hash160 Full Pipeline (privkey -> hash160) ===\n";
    {
        uint8_t privkey[32];
        std::memset(privkey, 0, sizeof(privkey));
        privkey[31] = 1;  // privkey = 1 (big-endian)
        auto h160 = collider::cpu::compute_hash160(privkey);
        const char* expected = "751e76e8199196d454941c45d1b3a323f1433bd6";
        if (compare_hex(h160.data(), h160.size(), expected)) {
            std::cout << "  PASS: privkey=1 -> compressed pubkey hash160\n";
            passed++;
        } else {
            char got[41];
            bytes_to_hex(h160.data(), h160.size(), got);
            std::cout << "  FAIL: privkey=1 full pipeline\n";
            std::cout << "    Expected: " << expected << "\n";
            std::cout << "    Got:      " << got << "\n";
            failed++;
        }
    }

    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";

    return failed > 0 ? 1 : 0;
}
