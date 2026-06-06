// signing_keys.hpp - compiled-in public keys for verifying server-signed
// artifacts.
//
// These are PUBLIC keys only. They are safe to ship in the binary; the
// matching PRIVATE keys live exclusively on the release/build infrastructure
// (in a secret manager) and are NEVER committed or embedded here.
//
// kReleaseSigningPubKey is the Ed25519 public key (raw 32 bytes) that signs
// the AUTH_OK auto-update manifest. The client verifies the AUTH_OK
// manifest_sig against this key BEFORE fetching any update binary, so a
// malicious or compromised pool cannot advertise an arbitrary binary for the
// fleet to install (review finding C3). Rotating the release key requires
// shipping a client build with the new value here.

#pragma once

#include <array>
#include <cstdint>

namespace collider {
namespace keys {

// Ed25519 release-signing PUBLIC key (raw 32 bytes). The private counterpart
// signs the canonical update manifest server-side
// (collision-protocol/src/release_signing.py).
inline constexpr std::array<uint8_t, 32> kReleaseSigningPubKey = {
    0xbd, 0x02, 0x1a, 0xb4, 0xb0, 0x89, 0xbf, 0xe7,
    0x26, 0x80, 0x6b, 0x84, 0xd3, 0xa9, 0xf8, 0x21,
    0xf6, 0x2c, 0x0c, 0x51, 0x96, 0x25, 0xe3, 0xe6,
    0x8c, 0x01, 0xf1, 0x99, 0xce, 0xa7, 0xe6, 0x82,
};

}  // namespace keys
}  // namespace collider
