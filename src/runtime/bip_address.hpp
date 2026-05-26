/**
 * bip_address.hpp -- BIP-44 / BIP-49 / BIP-84 address hash helpers.
 *
 * Pinned here (instead of inline in bip_scanner_runner.cpp) so the
 * KAT test can verify the byte-level math against spec reference
 * vectors. The BIP scanner runtime + tests pull from this single
 * source.
 */
#pragma once

#include "core/crypto_cpu.hpp"

#include <array>
#include <cstdint>
#include <cstring>

namespace collider::bip_address {

// Compute hash160(compressed_pubkey) for a 33-byte compressed pubkey.
// Used directly for P2PKH (BIP-44) and P2WPKH (BIP-84) -- both index
// the scriptPubKey hash160 by hash160(pubkey).
inline std::array<uint8_t, 20> hash160_pubkey(const uint8_t* pub33) {
    auto sha = collider::cpu::SHA256::hash(pub33, 33);
    return collider::cpu::RIPEMD160::hash(sha.data(), 32);
}

// Compute hash160(redeem_script) where redeem_script is the BIP-49
// P2SH-P2WPKH redeem script. Per BIP-49: redeem_script is 22 bytes,
//   0x00 (segwit version 0) || 0x14 (push 20) || hash160(pubkey)
// The bloom indexes scriptPubKey hash160s for P2SH outputs, which
// equals hash160(redeem_script).
inline std::array<uint8_t, 20> hash160_p2sh_p2wpkh(const uint8_t* pub33) {
    auto h_pub = hash160_pubkey(pub33);
    std::array<uint8_t, 22> redeem{};
    redeem[0] = 0x00;
    redeem[1] = 0x14;
    std::memcpy(redeem.data() + 2, h_pub.data(), 20);
    auto sha = collider::cpu::SHA256::hash(redeem.data(), 22);
    return collider::cpu::RIPEMD160::hash(sha.data(), 32);
}

}  // namespace collider::bip_address
