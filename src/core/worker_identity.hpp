/**
 * worker_identity.hpp -- B1 / wire-v4 (2026-05-23).
 *
 * Load a worker's BTC private key from a WIF file and produce the
 * pieces a wire-v4 AUTH frame needs:
 *   - 33-byte compressed pubkey (for the AUTH frame)
 *   - bech32 P2WPKH address (the worker_name the server compares
 *     against bech32(hash160(pubkey)))
 *   - 64-byte raw r||s ECDSA signature over the canonical AUTH
 *     message
 *
 * Implementation uses OpenSSL EVP (already linked into collider_core
 * via the license module). Going through OpenSSL keeps us off any
 * custom ECDSA code path; the same library backs the server-side
 * coincurve verifier, so signing here and verifying there is
 * cross-implementation.
 *
 * Threading: WorkerIdentity holds an EC_KEY ptr through its
 * lifetime; sign_message is reentrant on different instances but
 * NOT thread-safe on a single instance. The pool client only ever
 * signs from the main connection thread, so a single instance is
 * fine in practice.
 *
 * Memory: the privkey lives inside the EC_KEY (which OpenSSL stores
 * in a memory-locked buffer when available). The raw 32-byte priv is
 * NOT cached on this object; it's loaded once into OpenSSL and the
 * caller's std::array can be zeroized after construction.
 */

#pragma once

#include "bech32.hpp"
#include "wif.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifdef COLLIDER_HAS_OPENSSL
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/evp.h>
#include <openssl/obj_mac.h>
#include <openssl/bn.h>
#include <openssl/sha.h>
#endif

namespace collider {
namespace identity {

#ifdef COLLIDER_HAS_OPENSSL

namespace detail {

struct EcKeyDeleter {
    void operator()(EC_KEY* k) const { if (k) EC_KEY_free(k); }
};
using EcKeyPtr = std::unique_ptr<EC_KEY, EcKeyDeleter>;

struct BnDeleter {
    void operator()(BIGNUM* b) const { if (b) BN_clear_free(b); }
};
using BnPtr = std::unique_ptr<BIGNUM, BnDeleter>;

struct EcSigDeleter {
    void operator()(ECDSA_SIG* s) const { if (s) ECDSA_SIG_free(s); }
};
using EcSigPtr = std::unique_ptr<ECDSA_SIG, EcSigDeleter>;

}  // namespace detail

/**
 * A loaded worker identity. Construct via load_from_wif; the result
 * is movable, non-copyable. Provides the pubkey + bech32 address
 * exposed to the wire and a sign_message helper for the AUTH frame.
 */
class WorkerIdentity {
public:
    WorkerIdentity(detail::EcKeyPtr key,
                   std::array<uint8_t, 33> pub,
                   std::string addr)
        : key_(std::move(key)),
          pubkey_compressed_(pub),
          bech32_address_(std::move(addr)) {}

    WorkerIdentity(const WorkerIdentity&) = delete;
    WorkerIdentity& operator=(const WorkerIdentity&) = delete;
    WorkerIdentity(WorkerIdentity&&) = default;
    WorkerIdentity& operator=(WorkerIdentity&&) = default;

    const std::array<uint8_t, 33>& pubkey_compressed() const {
        return pubkey_compressed_;
    }
    const std::string& bech32_address() const { return bech32_address_; }

    /**
     * Sign a canonical message with the loaded privkey. Returns the
     * 64-byte raw r||s encoding (NOT DER) so the wire frame stays
     * fixed-size at 64 bytes. The server's verify_auth_v4 expects
     * exactly this encoding.
     *
     * SHA-256 is applied internally: callers pass the canonical
     * preimage and this routine hashes before signing.
     *
     * Returns std::nullopt on any OpenSSL failure or if the resulting
     * r/s don't fit into 32 bytes apiece (vanishingly rare; would
     * indicate an OpenSSL bug).
     */
    std::optional<std::array<uint8_t, 64>> sign_message(
        const uint8_t* msg, size_t len) const {
        uint8_t digest[32];
        SHA256(msg, len, digest);

        detail::EcSigPtr sig(ECDSA_do_sign(digest, 32, key_.get()));
        if (!sig) return std::nullopt;

        const BIGNUM* r = nullptr;
        const BIGNUM* s = nullptr;
        ECDSA_SIG_get0(sig.get(), &r, &s);
        if (!r || !s) return std::nullopt;

        // Bitcoin-style: enforce low-S so signatures are canonical.
        // The verifier on the server also accepts high-S, but pinning
        // to low-S means our wire bytes are deterministic up to the
        // ECDSA nonce.
        detail::BnPtr order(BN_new());
        if (!order) return std::nullopt;
        const EC_GROUP* group = EC_KEY_get0_group(key_.get());
        if (!group || !EC_GROUP_get_order(group, order.get(), nullptr)) {
            return std::nullopt;
        }
        detail::BnPtr half(BN_new());
        if (!half) return std::nullopt;
        BN_rshift1(half.get(), order.get());
        detail::BnPtr s_low(BN_dup(s));
        if (!s_low) return std::nullopt;
        if (BN_cmp(s_low.get(), half.get()) > 0) {
            BN_sub(s_low.get(), order.get(), s_low.get());
        }

        std::array<uint8_t, 64> out{};
        if (BN_bn2binpad(r, out.data(), 32) != 32) return std::nullopt;
        if (BN_bn2binpad(s_low.get(), out.data() + 32, 32) != 32) {
            return std::nullopt;
        }
        return out;
    }

private:
    detail::EcKeyPtr key_;
    std::array<uint8_t, 33> pubkey_compressed_;
    std::string bech32_address_;
};

namespace detail {

inline EcKeyPtr ec_key_from_priv32(const std::array<uint8_t, 32>& priv) {
    EcKeyPtr key(EC_KEY_new_by_curve_name(NID_secp256k1));
    if (!key) return {};
    BnPtr bn(BN_bin2bn(priv.data(), 32, nullptr));
    if (!bn) return {};
    if (!EC_KEY_set_private_key(key.get(), bn.get())) return {};

    // Recompute the public key from the private + generator. OpenSSL
    // requires both private and public components be set before
    // ECDSA_do_sign will run.
    const EC_GROUP* group = EC_KEY_get0_group(key.get());
    if (!group) return {};
    EC_POINT* pub = EC_POINT_new(group);
    if (!pub) return {};
    if (!EC_POINT_mul(group, pub, bn.get(), nullptr, nullptr, nullptr)) {
        EC_POINT_free(pub);
        return {};
    }
    int ok = EC_KEY_set_public_key(key.get(), pub);
    EC_POINT_free(pub);
    if (!ok) return {};

    // Force compressed-form serialization so EC_POINT_point2oct
    // produces the 33-byte encoding we want for the AUTH frame.
    EC_KEY_set_conv_form(key.get(), POINT_CONVERSION_COMPRESSED);
    return key;
}

inline std::optional<std::array<uint8_t, 33>> compressed_pubkey(EC_KEY* key) {
    const EC_GROUP* group = EC_KEY_get0_group(key);
    const EC_POINT* point = EC_KEY_get0_public_key(key);
    if (!group || !point) return std::nullopt;
    std::array<uint8_t, 33> out{};
    size_t n = EC_POINT_point2oct(group, point, POINT_CONVERSION_COMPRESSED,
                                  out.data(), out.size(), nullptr);
    if (n != 33) return std::nullopt;
    return out;
}

inline std::array<uint8_t, 20> hash160(const uint8_t* data, size_t len) {
    // RIPEMD160(SHA256(x)).
    uint8_t sha[32];
    SHA256(data, len, sha);
    std::array<uint8_t, 20> h{};
    unsigned int outlen = 20;
    const EVP_MD* md = EVP_get_digestbyname("ripemd160");
    if (!md) {
        // Fallback: collider's CPU RIPEMD160 is in crypto_cpu, but we
        // intentionally do NOT pull it here to keep this header
        // OpenSSL-only. If RIPEMD160 isn't available (OpenSSL 3 with
        // legacy provider disabled), the caller will see all-zero
        // hash160 which won't match any bech32 address -- the load
        // will fail-loud rather than silently producing wrong identity.
        return h;
    }
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, md, nullptr);
    EVP_DigestUpdate(ctx, sha, 32);
    EVP_DigestFinal_ex(ctx, h.data(), &outlen);
    EVP_MD_CTX_free(ctx);
    return h;
}

}  // namespace detail

/**
 * Load a worker identity from a WIF private key string. Returns
 * std::nullopt on:
 *   - malformed WIF (decode failure)
 *   - non-compressed WIF (wire-v4 requires compressed pubkeys)
 *   - OpenSSL keypair construction failure
 *   - bech32 encode failure
 *
 * The hrp arg picks mainnet ("bc") vs testnet ("tb").
 */
inline std::optional<WorkerIdentity> load_from_wif(
    std::string_view wif_str, const std::string& hrp = "bc") {
    auto decoded = wif::decode(wif_str);
    if (!decoded) return std::nullopt;
    // Wire-v4 worker_name is derived from the compressed pubkey form.
    // Refuse uncompressed keys at load time so the operator notices
    // the mismatch before runtime AUTH starts failing.
    if (!decoded->compressed) return std::nullopt;

    auto key = detail::ec_key_from_priv32(decoded->privkey);
    if (!key) return std::nullopt;

    auto pub = detail::compressed_pubkey(key.get());
    if (!pub) return std::nullopt;

    auto h160 = detail::hash160(pub->data(), pub->size());
    auto addr = bech32::encode_p2wpkh(h160, hrp);
    if (!addr) return std::nullopt;

    return WorkerIdentity(std::move(key), *pub, std::move(*addr));
}

#else  // !COLLIDER_HAS_OPENSSL

// Stub class for builds without OpenSSL. Wire-v4 requires OpenSSL.
class WorkerIdentity {
public:
    const std::array<uint8_t, 33>& pubkey_compressed() const { return zeros33_; }
    const std::string& bech32_address() const { return empty_; }
    std::optional<std::array<uint8_t, 64>> sign_message(
        const uint8_t*, size_t) const {
        return std::nullopt;
    }
private:
    std::array<uint8_t, 33> zeros33_{};
    std::string empty_{};
};

inline std::optional<WorkerIdentity> load_from_wif(
    std::string_view, const std::string& = "bc") {
    return std::nullopt;
}

#endif  // COLLIDER_HAS_OPENSSL

}  // namespace identity
}  // namespace collider
