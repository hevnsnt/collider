/**
 * Electrum v1 + v2 mnemonic seed derivation -- CPU reference (Phase 8, v1.4.0).
 *
 * Two distinct schemes, each historically used to crack mistyped or
 * leaked Electrum seeds:
 *
 * Electrum v1 (legacy hex-string seed):
 *   The user is given 12 English words. Each word maps to a 4-character
 *   slice of a 48-character hex string (the "raw seed"). To derive the
 *   master priv key, Electrum stretches the raw hex seed via SHA-256 chained
 *   100,000 times, prepending the seed each round (Electrum's
 *   `mnemonic.MnemonicCode.mnemonic_to_seed`):
 *
 *     m_0 = seed_hex_bytes
 *     m_{n+1} = SHA-256(m_n || seed_hex_bytes)
 *     priv = m_100000
 *
 * Electrum v2 (mnemonic-seed-version):
 *   Modern Electrum seeds carry a version byte derived from
 *   HMAC-SHA512(key="Seed version", msg=mnemonic_bytes); the first byte
 *   of the MAC must match the version prefix:
 *     standard         : 0x01
 *     segwit (P2WPKH)  : 0x10  (version "100")
 *     2FA              : 0x10  (version "101")
 *   If the version byte does not start with the expected nibble, the
 *   mnemonic is rejected. Valid mnemonics derive a seed via
 *   PBKDF2-HMAC-SHA512(2048 iterations, salt="electrum"+passphrase, dkLen=64).
 *
 * Both functions are header-only and depend only on the SHA-256 +
 * HMAC-SHA512 + PBKDF2-HMAC-SHA512 references in this directory.
 */

#pragma once

#include "address_derive_cpu.hpp"  // sha256
#include "sha512_cpu.hpp"          // hmac_sha512, pbkdf2_hmac_sha512

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace collider {
namespace gpu {
namespace v2 {
namespace electrum {

// ---------------------------------------------------------------------------
// Electrum v1: 100,000-round stretch of a 16-byte hex seed.
//
// The seed bytes input here is the RAW hex seed (16 bytes of randomness,
// not the user's 12 word display). Encoded as a hex string of 32 chars in
// Electrum's serialized form -- so the input we hash is the ASCII hex
// representation, not the raw bytes.
// ---------------------------------------------------------------------------

inline void v1_stretch_hex_seed(
    std::string_view hex_seed_str,
    uint8_t out_priv[32])
{
    // Electrum's CPython impl: stretched_seed = sha256(seed)
    // for _ in range(100000): stretched_seed = sha256(stretched_seed + seed)
    const auto* seed_p = reinterpret_cast<const uint8_t*>(hex_seed_str.data());
    size_t seed_n = hex_seed_str.size();

    uint8_t state[32];
    internal::sha256(seed_p, seed_n, state);

    std::vector<uint8_t> buf(32 + seed_n);
    for (uint32_t i = 0; i < 100000; ++i) {
        std::memcpy(buf.data(), state, 32);
        std::memcpy(buf.data() + 32, seed_p, seed_n);
        internal::sha256(buf.data(), buf.size(), state);
    }
    std::memcpy(out_priv, state, 32);
}

// ---------------------------------------------------------------------------
// Electrum v2: version-byte verification.
//
// Returns true if the mnemonic's HMAC-SHA512 first byte matches
// `expected_version_prefix`, false otherwise.
//
// Standard versions:
//   STANDARD (0x01): 'standard' Electrum
//   SEGWIT   (0x10): native segwit
//   TWO_FA   (0x10): 2FA legacy (matches segwit prefix; differentiate via 2nd nibble)
//
// In practice callers check first nibble only:
//   standard -> first nibble == 0x01
//   segwit / 2FA -> first byte == 0x10 + second nibble for distinction
// ---------------------------------------------------------------------------

inline uint8_t v2_version_byte(std::string_view mnemonic) {
    static const char* kKey = "Seed version";
    uint8_t mac[64];
    internal::hmac_sha512(
        reinterpret_cast<const uint8_t*>(kKey), 12,
        reinterpret_cast<const uint8_t*>(mnemonic.data()), mnemonic.size(),
        mac);
    return mac[0];
}

// Convenience: standard version requires first byte == 0x01.
inline bool v2_is_standard(std::string_view mnemonic) {
    return v2_version_byte(mnemonic) == 0x01;
}

// Convenience: segwit requires first byte == 0x10 AND first nibble of 2nd
// byte != 0x01. 2FA is first byte 0x10 with first nibble of 2nd == 0x01.
// Most callers just want "is this any v2 mnemonic" -- expose that too.
inline bool v2_is_segwit(std::string_view mnemonic) {
    uint8_t mac[64];
    static const char* kKey = "Seed version";
    internal::hmac_sha512(
        reinterpret_cast<const uint8_t*>(kKey), 12,
        reinterpret_cast<const uint8_t*>(mnemonic.data()), mnemonic.size(),
        mac);
    return mac[0] == 0x10 && (mac[1] >> 4) != 0x01;
}

// ---------------------------------------------------------------------------
// Electrum v2: seed derivation.
// PBKDF2-HMAC-SHA512(mnemonic, salt="electrum" + passphrase, 2048 iters,
//                    dkLen=64).
// ---------------------------------------------------------------------------

inline void v2_seed(
    std::string_view mnemonic,
    std::string_view passphrase,
    uint8_t out_seed[64])
{
    std::string salt = "electrum";
    salt.append(passphrase);
    internal::pbkdf2_hmac_sha512(
        reinterpret_cast<const uint8_t*>(mnemonic.data()), mnemonic.size(),
        reinterpret_cast<const uint8_t*>(salt.data()), salt.size(),
        2048,
        out_seed, 64);
}

}  // namespace electrum
}  // namespace v2
}  // namespace gpu
}  // namespace collider
