/**
 * bip32.hpp -- BIP-32 hierarchical deterministic key derivation.
 *
 * Sits on top of the existing crypto_cpu.hpp secp256k1 primitives.
 * Used by the v1.5.x BIP scanner runner (src/runtime/bip_scanner_runner.cpp)
 * to derive every standardized (and most historical pre-BIP-44) child
 * key from a single BIP-39 seed and probe each derived address against
 * the bloom filter.
 *
 * Spec: https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki
 *
 * Notation conventions match the BIP-32 spec:
 *   - master node:   m
 *   - hardened idx:  idx | 0x80000000 (written as N')
 *   - path example:  m/44'/0'/0'/0/i
 *
 * Implementation notes:
 *   - We implement CKDpriv (private parent -> private child) only. CKDpub
 *     is unnecessary for the scanner because we always have the master
 *     private key from the seed.
 *   - secp256k1 scalar arithmetic uses crypto_cpu's ec_mul + mod_add_n
 *     (private-key scalar modulo n = secp256k1 group order). For BIP-32
 *     we need (parent + I_L) mod n; crypto_cpu already exposes this via
 *     the scalar add helpers used inside ec_mul.
 *   - HMAC-SHA512 is provided by OpenSSL (linked for TLS already).
 */

#pragma once

#include "core/crypto_cpu.hpp"

#ifdef COLLIDER_HAS_OPENSSL
#  include <openssl/hmac.h>
#  include <openssl/sha.h>
#endif

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace collider {
namespace bip32 {

// 32-byte master/child private key + 32-byte chain code. Both fields
// are required to derive further children; storing them together
// avoids the two-value-per-node bookkeeping the spec encodes as the
// "extended key."
struct ExtKey {
    std::array<uint8_t, 32> key{};      // private key (scalar mod n)
    std::array<uint8_t, 32> chain{};    // chain code
    uint32_t child_number = 0;          // for status / debug
    uint8_t  depth = 0;                  // 0 = master
    std::array<uint8_t, 4> parent_fingerprint{};
};

// secp256k1 group order n in big-endian. Used to check derived
// scalars are in [1, n-1]; the BIP-32 spec mandates this check
// because a vanishingly-small fraction of HMAC outputs fall outside
// the valid range and must trigger derivation skip.
inline constexpr std::array<uint8_t, 32> kCurveOrderN = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41,
};

namespace detail {

// Strong equality + comparison helpers on 32-byte big-endian scalars.
inline bool is_zero_32be(const uint8_t* p) {
    uint8_t acc = 0;
    for (int i = 0; i < 32; ++i) acc |= p[i];
    return acc == 0;
}

inline int cmp_32be(const uint8_t* a, const uint8_t* b) {
    for (int i = 0; i < 32; ++i) {
        if (a[i] != b[i]) return (a[i] < b[i]) ? -1 : 1;
    }
    return 0;
}

// (a + b) mod n with both operands as 32-byte big-endian. Used by
// CKDpriv: child = (parent + I_L) mod n. We reduce by subtracting n
// once if the sum is >= n; the carry-from-add cannot exceed 1 because
// both a and b are < n.
inline void add_mod_n(const uint8_t* a, const uint8_t* b, uint8_t* out) {
    uint8_t sum[33]{};
    uint32_t carry = 0;
    for (int i = 31; i >= 0; --i) {
        uint32_t v = static_cast<uint32_t>(a[i]) + b[i] + carry;
        sum[i + 1] = static_cast<uint8_t>(v & 0xFF);
        carry = v >> 8;
    }
    sum[0] = static_cast<uint8_t>(carry);
    // If sum >= n (33-byte sum or 32-byte sum >= n), subtract n.
    bool ge_n = false;
    if (sum[0] != 0) {
        ge_n = true;
    } else {
        // Compare lower 32 bytes against n.
        ge_n = (cmp_32be(sum + 1, kCurveOrderN.data()) >= 0);
    }
    if (ge_n) {
        // Subtract n from sum (33 bytes).
        uint8_t n_padded[33]{};
        for (int i = 0; i < 32; ++i) n_padded[i + 1] = kCurveOrderN[i];
        int16_t borrow = 0;
        for (int i = 32; i >= 0; --i) {
            int32_t v = static_cast<int32_t>(sum[i]) - n_padded[i] - borrow;
            if (v < 0) { v += 256; borrow = 1; } else { borrow = 0; }
            sum[i] = static_cast<uint8_t>(v);
        }
    }
    std::memcpy(out, sum + 1, 32);
}

// secp256k1 scalar multiplication: pubkey = priv * G. Returns the
// 33-byte compressed public key (0x02/0x03 prefix + X).
inline std::array<uint8_t, 33> priv_to_pub(const uint8_t* priv32) {
    cpu::uint256_t k;
    // priv32 is big-endian; uint256_t's d[3] is the most-significant 64-bit limb.
    for (int i = 0; i < 4; ++i) {
        uint64_t limb = 0;
        for (int b = 0; b < 8; ++b) {
            limb = (limb << 8) | priv32[i * 8 + b];
        }
        k.d[3 - i] = limb;
    }
    cpu::ECPoint P;
    cpu::ec_mul(P, k);
    cpu::uint256_t Px, Py;
    cpu::ec_to_affine(Px, Py, P);
    std::array<uint8_t, 33> out{};
    out[0] = (Py.d[0] & 1ULL) ? 0x03 : 0x02;
    for (int i = 0; i < 4; ++i) {
        const uint64_t v = Px.d[3 - i];
        for (int b = 0; b < 8; ++b) {
            out[1 + i * 8 + b] = static_cast<uint8_t>((v >> ((7 - b) * 8)) & 0xFFu);
        }
    }
    return out;
}

#ifdef COLLIDER_HAS_OPENSSL
// OpenSSL HMAC-SHA512 wrapper. Returns 64 bytes. The OpenSSL HMAC API
// requires a non-zero-length key string for some versions; HMAC() handles
// zero-key gracefully on 1.1.x+ which is what every supported toolchain
// ships at this point.
inline std::array<uint8_t, 64> hmac_sha512(const uint8_t* key, size_t key_len,
                                           const uint8_t* data, size_t data_len) {
    std::array<uint8_t, 64> out{};
    unsigned int outlen = 0;
    HMAC(EVP_sha512(),
         key, static_cast<int>(key_len),
         data, data_len,
         out.data(), &outlen);
    if (outlen != 64) {
        throw std::runtime_error("HMAC-SHA512 unexpected output length");
    }
    return out;
}

// PBKDF2-HMAC-SHA512: the seed derivation used by BIP-39
// mnemonic-to-seed (2048 iterations, salt = "mnemonic" + passphrase).
inline std::array<uint8_t, 64> pbkdf2_hmac_sha512_64(
    const uint8_t* password, size_t password_len,
    const uint8_t* salt, size_t salt_len,
    int iterations) {
    std::array<uint8_t, 64> out{};
    int rc = PKCS5_PBKDF2_HMAC(
        reinterpret_cast<const char*>(password),
        static_cast<int>(password_len),
        salt, static_cast<int>(salt_len),
        iterations,
        EVP_sha512(),
        64, out.data());
    if (rc != 1) {
        throw std::runtime_error("PBKDF2-HMAC-SHA512 failed");
    }
    return out;
}
#endif  // COLLIDER_HAS_OPENSSL

}  // namespace detail

// Compute the BIP-39 seed from a (validated) mnemonic + optional
// passphrase. Per spec: PBKDF2(password=mnemonic, salt="mnemonic"+passphrase,
// c=2048, dkLen=64, HMAC-SHA512).
// Returns 64 bytes; throws on OpenSSL error or if !COLLIDER_HAS_OPENSSL.
inline std::array<uint8_t, 64> mnemonic_to_seed(
    const std::string& mnemonic,
    const std::string& passphrase = std::string{}) {
#ifdef COLLIDER_HAS_OPENSSL
    std::string salt = "mnemonic" + passphrase;
    return detail::pbkdf2_hmac_sha512_64(
        reinterpret_cast<const uint8_t*>(mnemonic.data()), mnemonic.size(),
        reinterpret_cast<const uint8_t*>(salt.data()), salt.size(),
        2048);
#else
    (void)mnemonic; (void)passphrase;
    throw std::runtime_error("bip39 seed needs OPENSSL (PBKDF2-HMAC-SHA512)");
#endif
}

// Build the master extended key from a BIP-32 seed (typically the
// 64-byte output of mnemonic_to_seed). Per spec:
//   I = HMAC-SHA512(key="Bitcoin seed", data=seed)
//   IL = I[0..31] -> master private key
//   IR = I[32..63] -> master chain code
// IL must be in [1, n-1]; if not, the seed is invalid (vanishingly
// rare). The throw on invalid IL matches the spec's "consider the
// generated seed invalid" prescription.
inline ExtKey master_from_seed(const uint8_t* seed, size_t seed_len) {
#ifdef COLLIDER_HAS_OPENSSL
    static const char* kHmacKey = "Bitcoin seed";
    auto I = detail::hmac_sha512(reinterpret_cast<const uint8_t*>(kHmacKey),
                                 12, seed, seed_len);
    ExtKey out;
    std::memcpy(out.key.data(),   I.data(),      32);
    std::memcpy(out.chain.data(), I.data() + 32, 32);
    if (detail::is_zero_32be(out.key.data()) ||
        detail::cmp_32be(out.key.data(), kCurveOrderN.data()) >= 0) {
        throw std::runtime_error("BIP-32: invalid IL from seed");
    }
    return out;
#else
    (void)seed; (void)seed_len;
    throw std::runtime_error("bip32 master needs OPENSSL");
#endif
}

// CKDpriv: derive child private key at the given index from a parent
// extended private key. Hardened index = parent_index | 0x80000000.
//   data = (hardened ? 0x00 || parent.key : parent.pubkey) || ser32(index)
//   I = HMAC-SHA512(key=parent.chain, data=data)
//   child.key   = (IL + parent.key) mod n
//   child.chain = IR
// Throws on invalid IL (extremely rare).
inline ExtKey ckd_priv(const ExtKey& parent, uint32_t index) {
#ifdef COLLIDER_HAS_OPENSSL
    const bool hardened = (index & 0x80000000u) != 0;
    std::vector<uint8_t> data;
    data.reserve(37);
    if (hardened) {
        data.push_back(0x00);
        data.insert(data.end(), parent.key.begin(), parent.key.end());
    } else {
        auto pub = detail::priv_to_pub(parent.key.data());
        data.insert(data.end(), pub.begin(), pub.end());
    }
    // ser32 = big-endian uint32
    data.push_back(static_cast<uint8_t>((index >> 24) & 0xFFu));
    data.push_back(static_cast<uint8_t>((index >> 16) & 0xFFu));
    data.push_back(static_cast<uint8_t>((index >>  8) & 0xFFu));
    data.push_back(static_cast<uint8_t>((index >>  0) & 0xFFu));

    auto I = detail::hmac_sha512(parent.chain.data(), parent.chain.size(),
                                 data.data(), data.size());

    if (detail::cmp_32be(I.data(), kCurveOrderN.data()) >= 0) {
        throw std::runtime_error("BIP-32: IL >= n; skip this index");
    }

    ExtKey out;
    detail::add_mod_n(parent.key.data(), I.data(), out.key.data());
    if (detail::is_zero_32be(out.key.data())) {
        throw std::runtime_error("BIP-32: derived key = 0; skip this index");
    }
    std::memcpy(out.chain.data(), I.data() + 32, 32);
    out.child_number = index;
    out.depth = static_cast<uint8_t>(parent.depth + 1);
    // parent fingerprint: first 4 bytes of HASH160(parent.pubkey).
    // Derive the pubkey ONCE via priv_to_pub, then hash it directly.
    // The historical compute_hash160(parent.key.data()) path re-derived
    // the pubkey internally (a second EC scalar multiplication per
    // CKDpriv call). Pinning hash160 to the already-derived pubkey
    // bytes halves the EC mul count on the BIP-32 derivation hot path,
    // which is the dominant cost for the BIP scanner runner.
    auto parent_pub = detail::priv_to_pub(parent.key.data());
    auto sha = cpu::SHA256::hash(parent_pub.data(), parent_pub.size());
    auto pkh = cpu::RIPEMD160::hash(sha.data(), sha.size());
    std::memcpy(out.parent_fingerprint.data(), pkh.data(), 4);
    return out;
#else
    (void)parent; (void)index;
    throw std::runtime_error("bip32 derive needs OPENSSL");
#endif
}

// Parse a derivation path string ("m/44'/0'/0'/0/i") into the vector
// of (possibly hardened) child indices. "m/" prefix is required.
// Apostrophe (') or "h" suffix marks hardened (sets the 0x80000000 bit).
// Throws std::invalid_argument on malformed input.
inline std::vector<uint32_t> parse_path(const std::string& path) {
    if (path.size() < 2 || (path[0] != 'm' && path[0] != 'M') ||
        path[1] != '/') {
        throw std::invalid_argument("BIP-32 path must start with m/");
    }
    std::vector<uint32_t> out;
    size_t i = 2;
    while (i < path.size()) {
        // Read a numeric token.
        size_t start = i;
        while (i < path.size() && path[i] >= '0' && path[i] <= '9') ++i;
        if (start == i) {
            throw std::invalid_argument("BIP-32 path: empty index segment");
        }
        uint64_t val = 0;
        for (size_t k = start; k < i; ++k) {
            val = val * 10 + static_cast<uint64_t>(path[k] - '0');
            if (val >= (1ULL << 31)) {
                throw std::invalid_argument("BIP-32 path: index overflow");
            }
        }
        uint32_t idx = static_cast<uint32_t>(val);
        if (i < path.size() && (path[i] == '\'' || path[i] == 'h' || path[i] == 'H')) {
            idx |= 0x80000000u;
            ++i;
        }
        out.push_back(idx);
        if (i < path.size()) {
            if (path[i] != '/') {
                throw std::invalid_argument("BIP-32 path: expected '/' after segment");
            }
            ++i;
        }
    }
    return out;
}

// Walk the path from `root` and return the leaf extended key.
inline ExtKey derive_path(const ExtKey& root, const std::vector<uint32_t>& path) {
    ExtKey cur = root;
    for (uint32_t idx : path) {
        cur = ckd_priv(cur, idx);
    }
    return cur;
}

}  // namespace bip32
}  // namespace collider
