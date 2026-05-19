/**
 * Keccak-256 CPU reference KATs -- task F.
 *
 * Reference vectors from the original Keccak Team's test suite + ethers.js
 * + ETH yellow paper examples. These pin down Ethereum-flavor Keccak-256
 * vs FIPS-202 SHA-3-256 (different padding byte: 0x01 vs 0x06).
 */

#include "../../src/gpu/v2/keccak256_cpu.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using collider::gpu::v2::keccak256::keccak256;

static int failures = 0;

static std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s; s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(d[p[i] >> 4]);
        s.push_back(d[p[i] & 0xF]);
    }
    return s;
}

static void check_vector(const char* label, const uint8_t* in, size_t len,
                         const char* expected_hex) {
    uint8_t out[32];
    keccak256(in, len, out);
    std::string got = hex(out, 32);
    if (got != expected_hex) {
        std::fprintf(stderr,
            "FAIL: %s\n  got:      %s\n  expected: %s\n",
            label, got.c_str(), expected_hex);
        ++failures;
    }
}

static void check_str(const char* label, const char* s, const char* expected_hex) {
    check_vector(label, reinterpret_cast<const uint8_t*>(s),
                 std::strlen(s), expected_hex);
}

int main() {
    // KAT 1: empty string -- the classic Keccak-256("") vector. Distinct from
    // FIPS-202 SHA-3-256("") which is a6e0...c7d40 (different padding byte).
    check_str("keccak256(\"\")",
              "",
              "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470");

    // KAT 2: short string from the Ethereum yellow paper / ethers.js.
    check_str("keccak256(\"abc\")",
              "abc",
              "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45");

    // KAT 3: common test vector used by Solidity tooling.
    check_str("keccak256(\"hello\")",
              "hello",
              "1c8aff950685c2ed4bc3174f3472287b56d9517b9c948127319a09a7a36deac8");

    // KAT 4: Ethereum address derivation for privkey = 1. With privkey=1
    // the public key is the generator point G itself, whose (X, Y) bytes
    // are constants. keccak256 of those 64 bytes, last 20 bytes, equals
    // the well-known "privkey=1" Ethereum address:
    //   0x7E5F4552091A69125d5DfCb7b8C2659029395Bdf
    // This vector is independently citable -- it appears in every basic
    // Ethereum tutorial and on countless block explorers (the address has
    // received many dust deposits over the years from people trying out
    // the trivial privkey).
    {
        const uint8_t pubkey_xy[64] = {
            // G_x = 0x79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798
            0x79,0xbe,0x66,0x7e,0xf9,0xdc,0xbb,0xac,
            0x55,0xa0,0x62,0x95,0xce,0x87,0x0b,0x07,
            0x02,0x9b,0xfc,0xdb,0x2d,0xce,0x28,0xd9,
            0x59,0xf2,0x81,0x5b,0x16,0xf8,0x17,0x98,
            // G_y = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8
            0x48,0x3a,0xda,0x77,0x26,0xa3,0xc4,0x65,
            0x5d,0xa4,0xfb,0xfc,0x0e,0x11,0x08,0xa8,
            0xfd,0x17,0xb4,0x48,0xa6,0x85,0x54,0x19,
            0x9c,0x47,0xd0,0x8f,0xfb,0x10,0xd4,0xb8,
        };
        uint8_t hash[32];
        keccak256(pubkey_xy, sizeof(pubkey_xy), hash);
        std::string addr_hex = hex(hash + 12, 20);
        const char* expected = "7e5f4552091a69125d5dfcb7b8c2659029395bdf";
        if (addr_hex != expected) {
            std::fprintf(stderr,
                "FAIL: ETH addr for privkey=1\n  got:      %s\n  expected: %s\n",
                addr_hex.c_str(), expected);
            ++failures;
        }
    }

    // (Block-boundary KATs for input lengths 135/136/137 deferred until an
    // independent reference -- PyCryptodome keccak or ethers.js -- is wired
    // into the test harness. The trust-anchor vectors above already
    // exercise the padding code path for non-block-multiple lengths.)

    if (failures != 0) {
        std::fprintf(stderr, "test_keccak256_cpu: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_keccak256_cpu: PASS\n");
    return 0;
}
