/**
 * Known-answer tests for src/core/byte_codec.hpp.
 *
 * Pins the four host-side helpers (be32_to_limbs_le, limbs_le_to_be32,
 * hex_encode_lower, range_bits_from_be) to expected outputs so a
 * regression in any of them fails loudly. range_bits_from_be has a
 * dedicated KAT (test_range_bits.cpp); this file covers the other
 * three plus their roundtrip invariant.
 *
 * Header-only host code, no GPU dependency.
 */

#include "../src/core/byte_codec.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

int g_failures = 0;

void check_str(const char* name, const std::string& expected, const std::string& got) {
    if (expected == got) {
        std::printf("[ok  ] %s\n", name);
    } else {
        std::printf("[FAIL] %s: expected %s got %s\n",
                    name, expected.c_str(), got.c_str());
        ++g_failures;
    }
}

void check_u64(const char* name, uint64_t expected, uint64_t got) {
    if (expected == got) {
        std::printf("[ok  ] %s\n", name);
    } else {
        std::printf("[FAIL] %s: expected %016llx got %016llx\n",
                    name,
                    (unsigned long long)expected,
                    (unsigned long long)got);
        ++g_failures;
    }
}

}  // namespace

int main() {
    // ----- hex_encode_lower -----
    {
        // All zeros.
        const uint8_t in[16] = {0};
        char out[33] = {};
        ::collider::hex_encode_lower(in, 16, out);
        check_str("hex_encode_lower(zeros 16)",
                  "00000000000000000000000000000000", out);
    }
    {
        // 0x00 .. 0x0F sequence.
        uint8_t in[16];
        for (int i = 0; i < 16; ++i) in[i] = static_cast<uint8_t>(i);
        char out[33] = {};
        ::collider::hex_encode_lower(in, 16, out);
        check_str("hex_encode_lower(0..15)",
                  "000102030405060708090a0b0c0d0e0f", out);
    }
    {
        // High-byte values exercise the >>4 nibble path.
        const uint8_t in[4] = {0xDE, 0xAD, 0xBE, 0xEF};
        char out[9] = {};
        ::collider::hex_encode_lower(in, 4, out);
        check_str("hex_encode_lower(deadbeef)", "deadbeef", out);
    }
    {
        // 32-byte private key shape, the common production size.
        uint8_t in[32];
        for (int i = 0; i < 32; ++i) in[i] = static_cast<uint8_t>(0xA0 + i);
        char out[65] = {};
        ::collider::hex_encode_lower(in, 32, out);
        // Manually computed expected: a0 a1 a2 ... bf
        const std::string expected =
            "a0a1a2a3a4a5a6a7a8a9aaabacadaeafb0b1b2b3b4b5b6b7b8b9babbbcbdbebf";
        check_str("hex_encode_lower(32 bytes 0xA0..0xBF)", expected, out);
    }

    // ----- be32_to_limbs_le roundtrip with limbs_le_to_be32 -----
    {
        // Canonical secp256k1 generator x-coordinate (BE).
        const uint8_t Gx_be[32] = {
            0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
            0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
            0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
            0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98
        };
        uint64_t limbs[4];
        ::collider::be32_to_limbs_le(Gx_be, limbs);
        // limb[0] = low 8 bytes of BE input -> 0x59F2815B16F81798
        check_u64("be32_to_limbs_le(Gx)[0]", 0x59F2815B16F81798ULL, limbs[0]);
        check_u64("be32_to_limbs_le(Gx)[1]", 0x029BFCDB2DCE28D9ULL, limbs[1]);
        check_u64("be32_to_limbs_le(Gx)[2]", 0x55A06295CE870B07ULL, limbs[2]);
        check_u64("be32_to_limbs_le(Gx)[3]", 0x79BE667EF9DCBBACULL, limbs[3]);

        // Roundtrip: limbs back to BE bytes.
        uint8_t roundtrip[32];
        ::collider::limbs_le_to_be32(limbs, roundtrip);
        if (std::memcmp(Gx_be, roundtrip, 32) != 0) {
            std::printf("[FAIL] be<->limbs roundtrip\n");
            ++g_failures;
        } else {
            std::printf("[ok  ] be<->limbs roundtrip(Gx)\n");
        }
    }

    // limbs_le_to_be32 directly with simple values.
    {
        const uint64_t limbs[4] = {
            0x0102030405060708ULL,  // -> bytes 24..31 (low BE)
            0x1112131415161718ULL,  // -> bytes 16..23
            0x2122232425262728ULL,  // -> bytes  8..15
            0x3132333435363738ULL,  // -> bytes  0.. 7 (high BE)
        };
        uint8_t out[32];
        ::collider::limbs_le_to_be32(limbs, out);
        const uint8_t expected[32] = {
            0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38,
            0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08
        };
        if (std::memcmp(expected, out, 32) != 0) {
            std::printf("[FAIL] limbs_le_to_be32(crafted)\n");
            ++g_failures;
        } else {
            std::printf("[ok  ] limbs_le_to_be32(crafted)\n");
        }
    }

    // ----- hex_decode -----
    {
        // Roundtrip: encode then decode a 32-byte vector.
        uint8_t in[32];
        for (int i = 0; i < 32; ++i) in[i] = static_cast<uint8_t>(i * 7 + 1);
        char encoded[65];
        ::collider::hex_encode_lower(in, 32, encoded);

        uint8_t roundtrip[32];
        if (!::collider::hex_decode(encoded, 64, roundtrip, 32)) {
            std::printf("[FAIL] hex_decode(roundtrip): returned false\n");
            ++g_failures;
        } else if (std::memcmp(in, roundtrip, 32) != 0) {
            std::printf("[FAIL] hex_decode(roundtrip): byte mismatch\n");
            ++g_failures;
        } else {
            std::printf("[ok  ] hex_decode(roundtrip 32 bytes)\n");
        }
    }
    {
        // 0x prefix accepted.
        const char* hex = "0xdeadbeef";
        uint8_t out[4];
        if (!::collider::hex_decode(hex, 10, out, 4)) {
            std::printf("[FAIL] hex_decode(0xdeadbeef): returned false\n");
            ++g_failures;
        } else if (out[0] != 0xDE || out[1] != 0xAD ||
                   out[2] != 0xBE || out[3] != 0xEF) {
            std::printf("[FAIL] hex_decode(0xdeadbeef): wrong bytes %02x%02x%02x%02x\n",
                        out[0], out[1], out[2], out[3]);
            ++g_failures;
        } else {
            std::printf("[ok  ] hex_decode(0xdeadbeef with prefix)\n");
        }
    }
    {
        // Mixed case input.
        const char* hex = "AbCdEf";
        uint8_t out[3];
        if (!::collider::hex_decode(hex, 6, out, 3)) {
            std::printf("[FAIL] hex_decode(mixed case): returned false\n");
            ++g_failures;
        } else if (out[0] != 0xAB || out[1] != 0xCD || out[2] != 0xEF) {
            std::printf("[FAIL] hex_decode(mixed case): wrong bytes\n");
            ++g_failures;
        } else {
            std::printf("[ok  ] hex_decode(mixed case)\n");
        }
    }
    {
        // Reject odd length.
        uint8_t out[1];
        if (::collider::hex_decode("abc", 3, out, 1)) {
            std::printf("[FAIL] hex_decode: should reject odd-length input\n");
            ++g_failures;
        } else {
            std::printf("[ok  ] hex_decode rejects odd length\n");
        }
    }
    {
        // Reject invalid char.
        uint8_t out[2];
        if (::collider::hex_decode("ab!d", 4, out, 2)) {
            std::printf("[FAIL] hex_decode: should reject non-hex char\n");
            ++g_failures;
        } else {
            std::printf("[ok  ] hex_decode rejects non-hex char\n");
        }
    }
    {
        // Reject length mismatch.
        uint8_t out[5];
        if (::collider::hex_decode("abcdef", 6, out, 5)) {
            std::printf("[FAIL] hex_decode: should reject length mismatch\n");
            ++g_failures;
        } else {
            std::printf("[ok  ] hex_decode rejects length mismatch\n");
        }
    }

    if (g_failures > 0) {
        std::printf("FAIL: %d byte_codec cases failed\n", g_failures);
        return 1;
    }
    std::printf("test_byte_codec: 12/12 PASS\n");
    return 0;
}
