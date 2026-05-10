/**
 * Known-answer test for the canonical puzzle data in
 * src/core/puzzle_config.hpp.
 *
 * For every puzzle marked solved=true with a non-empty solution_hex:
 *   1. Parse the hex private key.
 *   2. Compute pubkey + Hash160 via cpu::compute_hash160.
 *   3. Verify the result matches the recorded target_h160_hex.
 *
 * This catches the class of corruption fixed in v1.4.0 phase 2.1
 * (the pre-1.4 puzzle table had wrong addresses + fabricated
 * solution_hex strings; a wrong hash160/private-key pairing would
 * have made every solver hit "wrong" against the recorded value).
 *
 * Pure host test, no GPU dependency.
 */

#include "../src/core/crypto_cpu.hpp"
#include "../src/core/puzzle_config.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

// Hex -> bytes, little-endian-aware. Input hex must be even-length and
// contain only [0-9a-fA-F]; leading 0x is optional.
bool hex_to_bytes(const std::string& hex_in, uint8_t out[32]) {
    std::string hex = hex_in;
    if (hex.size() >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) {
        hex = hex.substr(2);
    }
    if (hex.size() > 64) return false;
    // Right-align: last char of hex is byte 31 LSB.
    std::memset(out, 0, 32);
    int dst_byte = 31;
    for (int i = static_cast<int>(hex.size()) - 1; i >= 0 && dst_byte >= 0; ) {
        const char lo_c = hex[i];
        const char hi_c = (i > 0) ? hex[i - 1] : '0';
        auto hex_to_nibble = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return -1;
        };
        int lo = hex_to_nibble(lo_c);
        int hi = hex_to_nibble(hi_c);
        if (lo < 0 || hi < 0) return false;
        out[dst_byte] = static_cast<uint8_t>((hi << 4) | lo);
        --dst_byte;
        i -= 2;
    }
    return true;
}

bool hash160_to_hex(const std::array<uint8_t, 20>& h, std::string& out) {
    char buf[41];
    static const char kHex[] = "0123456789abcdef";
    for (int i = 0; i < 20; ++i) {
        buf[i * 2]     = kHex[(h[i] >> 4) & 0xFu];
        buf[i * 2 + 1] = kHex[h[i] & 0xFu];
    }
    buf[40] = '\0';
    out = buf;
    return true;
}

int g_failures = 0;
int g_checked  = 0;

}  // namespace

int main() {
    using ::collider::cpu::compute_hash160;
    const auto& puzzles = ::collider::PuzzleDatabase::get_all();
    for (const auto& p : puzzles) {
        if (!p.solved || p.solution_hex.empty()) continue;
        if (p.target_h160_hex == "unknown" || p.target_h160_hex.size() != 40) {
            std::printf("[skip] puzzle %d: target_h160_hex is %s\n",
                        p.number, p.target_h160_hex.c_str());
            continue;
        }

        uint8_t priv[32];
        if (!hex_to_bytes(p.solution_hex, priv)) {
            std::printf("[FAIL] puzzle %d: cannot parse solution_hex %s\n",
                        p.number, p.solution_hex.c_str());
            ++g_failures;
            continue;
        }

        std::array<uint8_t, 20> derived = compute_hash160(priv);
        std::string derived_hex;
        hash160_to_hex(derived, derived_hex);

        ++g_checked;
        if (derived_hex == p.target_h160_hex) {
            std::printf("[ok  ] puzzle %d: %s\n", p.number, derived_hex.c_str());
        } else {
            std::printf("[FAIL] puzzle %d: solution_hex=%s\n", p.number,
                        p.solution_hex.c_str());
            std::printf("       expected h160 = %s\n", p.target_h160_hex.c_str());
            std::printf("       derived  h160 = %s\n", derived_hex.c_str());
            ++g_failures;
        }
    }

    if (g_failures > 0) {
        std::printf("FAIL: %d of %d puzzle entries had wrong derived hash160\n",
                    g_failures, g_checked);
        return 1;
    }
    std::printf("test_puzzle_table_kat: %d/%d PASS\n", g_checked, g_checked);
    return 0;
}
