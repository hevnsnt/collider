/**
 * bech32.hpp -- BIP-173 Bech32 address decoder with checksum verification.
 *
 * pre-fix the brain-wallet UVRF builder and the runtime
 * HitVerifier each had their own bech32 decoder that SKIPPED the checksum
 * bytes entirely. A typo'd or corrupted bc1q... address parsed to a
 * deterministic but WRONG H160 and was silently added to the UVRF / bloom,
 * producing phantom verified entries.
 *
 * This module provides a single, BIP-173-compliant decoder that:
 *   - validates the HRP matches expected ("bc" for mainnet, "tb" for testnet)
 *   - rejects mixed-case input (BIP-173 invariant)
 *   - verifies the 6-character checksum via polymod
 *   - returns std::nullopt on any failure (no exceptions, no phantoms)
 *
 * Scope: BIP-173 (witness version 0 = P2WPKH / P2WSH). P2TR (BIP-350, bech32m)
 * uses a different polymod constant and is intentionally NOT handled here;
 * a separate decoder will land with v1.5 P2TR support.
 *
 * Reference: https://github.com/bitcoin/bips/blob/master/bip-0173.mediawiki
 * KAT vectors: tests/test_bech32_checksum.cpp.
 */

#pragma once

#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace collider {
namespace bech32 {

// BIP-173 alphabet (32 chars, no '1' / 'b' / 'i' / 'o' to avoid confusion).
constexpr const char* CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

// 5 generator polynomials for the BCH code (BIP-173 Section "Checksum").
constexpr uint32_t GEN[5] = {
    0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3
};

// BIP-173 constant: encoded polymod is XORed by this value (1 for bech32).
constexpr uint32_t BECH32_CONST = 1;

// BIP-350 constant for bech32m (P2TR). Not exposed yet; documented for
// the v1.5 P2TR work.
constexpr uint32_t BECH32M_CONST = 0x2BC830A3;

// Reverse-alphabet lookup: char -> 5-bit value, or 0xFF if invalid.
inline int charset_lookup(char c) {
    static const auto build_table = []() {
        std::array<int, 256> tbl{};
        for (auto& v : tbl) v = -1;
        for (int i = 0; i < 32; ++i) {
            tbl[static_cast<unsigned char>(CHARSET[i])] = i;
        }
        return tbl;
    }();
    return build_table[static_cast<unsigned char>(c)];
}

// BCH-code polymod over the 5-bit values.
inline uint32_t polymod(const std::vector<uint8_t>& values) {
    uint32_t chk = 1;
    for (uint8_t v : values) {
        uint8_t top = chk >> 25;
        chk = ((chk & 0x1FFFFFF) << 5) ^ v;
        for (int i = 0; i < 5; ++i) {
            if ((top >> i) & 1) chk ^= GEN[i];
        }
    }
    return chk;
}

// HRP expansion: each char split into high (>>5) and low (&31) halves with
// a zero separator between. See BIP-173 "Bech32".
inline std::vector<uint8_t> hrp_expand(const std::string& hrp) {
    std::vector<uint8_t> out;
    out.reserve(hrp.size() * 2 + 1);
    for (char c : hrp) out.push_back(static_cast<uint8_t>(c) >> 5);
    out.push_back(0);
    for (char c : hrp) out.push_back(static_cast<uint8_t>(c) & 31);
    return out;
}

// Verify that polymod(hrp_expand(hrp) || data) == constant.
inline bool verify_checksum(const std::string& hrp,
                            const std::vector<uint8_t>& data,
                            uint32_t expected_constant = BECH32_CONST) {
    std::vector<uint8_t> v = hrp_expand(hrp);
    v.insert(v.end(), data.begin(), data.end());
    return polymod(v) == expected_constant;
}

// Convert a sequence of `from_bits`-bit values to `to_bits`-bit values.
// Returns false on overflow or invalid padding (with pad=false).
inline bool convert_bits(std::vector<uint8_t>& out,
                         const std::vector<uint8_t>& in,
                         int from_bits, int to_bits, bool pad) {
    uint32_t acc = 0;
    int bits = 0;
    const uint32_t maxv = (1u << to_bits) - 1;
    const uint32_t max_acc = (1u << (from_bits + to_bits - 1)) - 1;
    for (uint8_t value : in) {
        if (value >> from_bits) return false;
        acc = ((acc << from_bits) | value) & max_acc;
        bits += from_bits;
        while (bits >= to_bits) {
            bits -= to_bits;
            out.push_back((acc >> bits) & maxv);
        }
    }
    if (pad) {
        if (bits) out.push_back((acc << (to_bits - bits)) & maxv);
    } else if (bits >= from_bits || ((acc << (to_bits - bits)) & maxv)) {
        return false;
    }
    return true;
}

/**
 * Decode a Bech32 P2WPKH address (witness version 0, 20-byte program).
 *
 * Returns std::nullopt if:
 *   - address is mixed-case (BIP-173 invariant)
 *   - length is out of [8, 90]
 *   - HRP doesn't match expected_hrp
 *   - any character is outside the bech32 alphabet
 *   - polymod checksum fails
 *   - witness version != 0
 *   - program length != 20 bytes
 *
 * On success returns the 20-byte witness program (= H160 of the public key).
 */
inline std::optional<std::array<uint8_t, 20>> decode_p2wpkh(
    const std::string& address,
    const std::string& expected_hrp = "bc")
{
    if (address.size() < 8 || address.size() > 90) return std::nullopt;

    // BIP-173: mixed case is invalid. Detect and reject.
    bool has_lower = false, has_upper = false;
    for (char c : address) {
        if (c >= 'a' && c <= 'z') has_lower = true;
        if (c >= 'A' && c <= 'Z') has_upper = true;
    }
    if (has_lower && has_upper) return std::nullopt;

    // Normalize to lowercase for processing.
    std::string a;
    a.reserve(address.size());
    for (char c : address) {
        if (c >= 'A' && c <= 'Z') a.push_back(c + ('a' - 'A'));
        else a.push_back(c);
    }

    // Find the LAST '1' (separator). HRP is everything before it.
    size_t sep = a.rfind('1');
    if (sep == std::string::npos) return std::nullopt;
    if (sep == 0) return std::nullopt;                  // empty HRP
    if (a.size() - sep < 7) return std::nullopt;        // need ver + 6 checksum at least

    std::string hrp = a.substr(0, sep);
    if (hrp != expected_hrp) return std::nullopt;

    // Decode 5-bit values from data part.
    std::vector<uint8_t> data;
    data.reserve(a.size() - sep - 1);
    for (size_t i = sep + 1; i < a.size(); ++i) {
        int v = charset_lookup(a[i]);
        if (v < 0) return std::nullopt;
        data.push_back(static_cast<uint8_t>(v));
    }

    // Verify checksum.
    if (!verify_checksum(hrp, data, BECH32_CONST)) return std::nullopt;

    // Strip checksum (last 6 5-bit values) and split version (first).
    if (data.size() < 7) return std::nullopt;
    uint8_t witness_version = data[0];
    std::vector<uint8_t> program_5bit(data.begin() + 1, data.end() - 6);

    if (witness_version != 0) return std::nullopt;    // Only v0 here (P2WPKH/WSH)

    std::vector<uint8_t> program_8bit;
    if (!convert_bits(program_8bit, program_5bit, 5, 8, false)) return std::nullopt;

    if (program_8bit.size() != 20) return std::nullopt;  // P2WPKH only

    std::array<uint8_t, 20> out;
    std::memcpy(out.data(), program_8bit.data(), 20);
    return out;
}

/**
 * Encode a 20-byte hash160 as a witness-version-0 P2WPKH bech32 address
 * (BIP-173). Mirrors the Python `encode_p2wpkh_address` used by the
 * collision-protocol wire-v4 server-side verifier so the C++ client and
 * Python server agree on the exact bech32 string for the same pubkey.
 *
 * Default HRP "bc" (mainnet). Testnet callers pass hrp="tb".
 *
 * Returns std::nullopt only if convert_bits fails, which is unreachable
 * for a 20-byte input but kept for symmetry with the decoder.
 */
inline std::optional<std::string> encode_p2wpkh(
    const std::array<uint8_t, 20>& h160,
    const std::string& hrp = "bc")
{
    // 1. Convert 20 8-bit bytes to 5-bit groups (32 groups, padded).
    std::vector<uint8_t> in(h160.begin(), h160.end());
    std::vector<uint8_t> data5;
    if (!convert_bits(data5, in, 8, 5, true)) return std::nullopt;

    // 2. Prepend witness version 0.
    std::vector<uint8_t> payload;
    payload.reserve(1 + data5.size() + 6);
    payload.push_back(0);
    payload.insert(payload.end(), data5.begin(), data5.end());

    // 3. Compute the 6-symbol checksum over hrp_expand(hrp) || payload || 6 zeros.
    std::vector<uint8_t> values = hrp_expand(hrp);
    values.insert(values.end(), payload.begin(), payload.end());
    values.insert(values.end(), 6, 0);
    uint32_t polymod_val = polymod(values) ^ BECH32_CONST;
    for (int i = 0; i < 6; ++i) {
        payload.push_back(static_cast<uint8_t>((polymod_val >> (5 * (5 - i))) & 31));
    }

    // 4. Assemble: hrp || '1' || charset-mapped payload.
    std::string out;
    out.reserve(hrp.size() + 1 + payload.size());
    out.append(hrp);
    out.push_back('1');
    for (uint8_t v : payload) {
        out.push_back(CHARSET[v]);
    }
    return out;
}

}  // namespace bech32
}  // namespace collider
