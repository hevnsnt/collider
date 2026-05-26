/**
 * wif.hpp -- B1 / wire-v4 (2026-05-23). WIF (Wallet Import Format)
 * encode + decode for the worker identity key.
 *
 * Format (compressed pubkey, mainnet):
 *   payload = 0x80 || privkey32 || 0x01           (34 bytes)
 *   wif     = Base58Check(payload)                 (~52 chars, starts with 'K' or 'L')
 *
 * Format (uncompressed pubkey, mainnet, legacy):
 *   payload = 0x80 || privkey32                    (33 bytes)
 *   wif     = Base58Check(payload)                 (~51 chars, starts with '5')
 *
 * Base58Check adds a 4-byte sha256(sha256(payload))[:4] checksum
 * before encoding; decode rejects any wif whose recomputed checksum
 * doesn't match.
 *
 * Used by collider-pro's wire-v4 client to load a worker's BTC
 * private key from a 0600-permissioned file and derive the matching
 * compressed pubkey + bech32 P2WPKH worker_name. See
 * src/core/worker_identity.hpp.
 *
 * Reference: https://en.bitcoin.it/wiki/Wallet_import_format
 */

#pragma once

#include "crypto_cpu.hpp"
#include "puzzle_config.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace collider {
namespace wif {

constexpr uint8_t MAINNET_PRIVKEY_PREFIX = 0x80;
constexpr uint8_t TESTNET_PRIVKEY_PREFIX = 0xEF;
constexpr uint8_t COMPRESSED_SUFFIX = 0x01;

struct DecodedKey {
    std::array<uint8_t, 32> privkey;
    bool compressed;        // true if the 0x01 compressed-pubkey suffix was present
    uint8_t network_prefix; // 0x80 mainnet, 0xEF testnet
};

namespace detail {

// SHA256d = SHA256(SHA256(x)). The canonical Base58Check checksum.
inline std::array<uint8_t, 32> sha256d(const uint8_t* data, size_t len) {
    auto first = collider::cpu::SHA256::hash(data, len);
    return collider::cpu::SHA256::hash(first.data(), first.size());
}

// Base58 encode of an arbitrary byte string. Leading zero bytes map
// to leading '1' characters.
inline std::string base58_encode(const std::vector<uint8_t>& input) {
    // Count leading zeros.
    size_t leading_zeros = 0;
    while (leading_zeros < input.size() && input[leading_zeros] == 0) {
        ++leading_zeros;
    }
    // Allocate enough space: ceil(log_58(256^N)) = ceil(N * 138 / 100).
    std::vector<uint8_t> b58((input.size() - leading_zeros) * 138 / 100 + 1, 0);
    for (size_t i = leading_zeros; i < input.size(); ++i) {
        int carry = input[i];
        for (auto it = b58.rbegin(); it != b58.rend(); ++it) {
            carry += 256 * (*it);
            *it = static_cast<uint8_t>(carry % 58);
            carry /= 58;
        }
        // carry should be 0 here; if not, our buffer was undersized.
    }
    // Skip leading zeros in b58 result.
    size_t first_nonzero = 0;
    while (first_nonzero < b58.size() && b58[first_nonzero] == 0) {
        ++first_nonzero;
    }
    static constexpr const char* ALPHABET =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::string out;
    out.reserve(leading_zeros + (b58.size() - first_nonzero));
    out.append(leading_zeros, '1');
    for (size_t i = first_nonzero; i < b58.size(); ++i) {
        out.push_back(ALPHABET[b58[i]]);
    }
    return out;
}

}  // namespace detail

/**
 * Decode a WIF private key. Returns std::nullopt if:
 *   - Base58 decode fails
 *   - decoded length is not 33 (uncompressed) or 34 (compressed)
 *   - 4-byte sha256d checksum at the end does not match
 *   - compressed-form trailing byte is not 0x01
 *
 * On success returns the 32-byte private key plus the compressed flag
 * and the network prefix byte for the caller to interpret.
 */
inline std::optional<DecodedKey> decode(std::string_view wif) {
    if (wif.empty()) return std::nullopt;
    std::vector<uint8_t> raw;
    try {
        raw = collider::Base58::decode(std::string(wif));
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    }
    // Valid lengths: 37 (uncompressed: prefix + priv32 + checksum4)
    //            or  38 (compressed:   prefix + priv32 + 0x01 + checksum4).
    if (raw.size() != 37 && raw.size() != 38) {
        return std::nullopt;
    }
    // Verify the 4-byte sha256d checksum at the tail.
    auto check = detail::sha256d(raw.data(), raw.size() - 4);
    for (int i = 0; i < 4; ++i) {
        if (check[i] != raw[raw.size() - 4 + i]) return std::nullopt;
    }
    DecodedKey out;
    out.network_prefix = raw[0];
    if (raw.size() == 38) {
        // raw[33] is the byte right after the 32-byte privkey; for a
        // compressed wif it must be 0x01.
        if (raw[33] != COMPRESSED_SUFFIX) return std::nullopt;
        out.compressed = true;
    } else {
        out.compressed = false;
    }
    std::copy(raw.begin() + 1, raw.begin() + 1 + 32, out.privkey.begin());
    return out;
}

/**
 * Encode a 32-byte private key as a WIF string.
 *
 * Default: compressed pubkey form, mainnet prefix. Produces a wif that
 * starts with 'K' or 'L' (~52 chars). The compressed flag controls
 * whether the trailing 0x01 byte is included; this MUST match the
 * pubkey form (compressed vs uncompressed) the wallet will publish,
 * otherwise the derived address won't match.
 */
inline std::string encode(const std::array<uint8_t, 32>& privkey,
                          bool compressed = true,
                          uint8_t network_prefix = MAINNET_PRIVKEY_PREFIX) {
    std::vector<uint8_t> payload;
    payload.reserve(compressed ? 38 : 37);
    payload.push_back(network_prefix);
    payload.insert(payload.end(), privkey.begin(), privkey.end());
    if (compressed) payload.push_back(COMPRESSED_SUFFIX);
    auto check = detail::sha256d(payload.data(), payload.size());
    payload.insert(payload.end(), check.begin(), check.begin() + 4);
    return detail::base58_encode(payload);
}

}  // namespace wif
}  // namespace collider
