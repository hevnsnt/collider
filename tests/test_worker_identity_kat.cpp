/**
 * test_worker_identity_kat.cpp -- B1 / wire-v4 (2026-05-23).
 *
 * Pins the OpenSSL secp256k1 signer used to produce wire-v4 AUTH
 * frames. Three things have to hold for the server-side verifier
 * (collision-protocol auth_v4.py) to accept the client's AUTH:
 *
 *   1. The deterministic privkey c0c0...01 produces a known
 *      compressed pubkey and a known bech32 P2WPKH address. Both
 *      sides must agree byte-for-byte.
 *   2. sign_message produces a valid 64-byte raw r||s signature over
 *      the SHA-256 of the canonical AUTH message.
 *   3. Round-trip: a signature produced here verifies against the
 *      pubkey via OpenSSL ECDSA_do_verify. (Server-side cross-impl
 *      verification with coincurve is covered by an integration
 *      test once the wire framing lands.)
 *
 * The bech32 address for this privkey was derived from the Python
 * test_auth_v4_kat.py setup and pinned here. If either side's
 * encoding drifts, the KAT fails and prevents a wire mismatch from
 * shipping.
 */

#ifndef COLLIDER_HAS_OPENSSL
#include <iostream>
int main() {
    std::cout << "[SKIP] OpenSSL not available; wire-v4 needs OpenSSL\n";
    return 77;  // ctest SKIP
}
#else

#include "../src/core/worker_identity.hpp"
#include "../src/core/wif.hpp"

#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/sha.h>

#include <array>
#include <cstring>
#include <iostream>
#include <string>

namespace {

int g_pass = 0;
int g_fail = 0;

std::array<uint8_t, 32> hex32(const char* hex) {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 32; ++i) {
        unsigned hi = 0, lo = 0;
        char c = hex[i * 2];
        hi = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
        c = hex[i * 2 + 1];
        lo = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return out;
}

std::string to_hex(const uint8_t* p, size_t n) {
    static const char* H = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(H[p[i] >> 4]);
        s.push_back(H[p[i] & 0xf]);
    }
    return s;
}

}  // namespace

int main() {
    std::cout << "=== test_worker_identity_kat (B1 wire-v4) ===\n";

    // Deterministic privkey shared with Python test_auth_v4_kat.py.
    auto priv = hex32("c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0c0"
                      "c0c0c0c0c0c0c0c0c0c0c0c0c0c0c001");
    std::string wif = collider::wif::encode(priv, /*compressed=*/true);

    auto id = collider::identity::load_from_wif(wif, "bc");
    if (!id) {
        std::cerr << "[FAIL] load_from_wif returned nullopt\n";
        return 1;
    }
    ++g_pass;
    std::cout << "[PASS] load_from_wif (wif=" << wif << ")\n";
    std::cout << "       pubkey   = " << to_hex(id->pubkey_compressed().data(), 33) << "\n";
    std::cout << "       bech32   = " << id->bech32_address() << "\n";

    // Cross-implementation pin: the Python server-side test_auth_v4_kat
    // computes the same pubkey + bech32 for this privkey using
    // coincurve + the auth_v4 encoder. The strings below were
    // captured from `python -c "..."` against
    // collision-protocol/src/auth_v4.py. Any drift on either side
    // (different bech32 polymod constant, different compression
    // form, different hash160 impl, etc.) breaks this assertion
    // BEFORE the wire ever cuts over.
    {
        const std::string expected_pub =
            "033215195a6bf506beccd50c9e4f5dcd58c7b61fd2412f2a9e3d93d65be6934e46";
        const std::string expected_addr =
            "bc1q2hs02hlnt27w33a9r78yx435m2gagv76mxqag9";
        std::string got_pub = to_hex(id->pubkey_compressed().data(), 33);
        if (got_pub != expected_pub) {
            ++g_fail;
            std::cerr << "[FAIL] pubkey mismatch:\n  got  " << got_pub
                      << "\n  want " << expected_pub << "\n";
        } else {
            ++g_pass;
            std::cout << "[PASS] pubkey matches Python coincurve\n";
        }
        if (id->bech32_address() != expected_addr) {
            ++g_fail;
            std::cerr << "[FAIL] bech32 mismatch:\n  got  "
                      << id->bech32_address()
                      << "\n  want " << expected_addr << "\n";
        } else {
            ++g_pass;
            std::cout << "[PASS] bech32 matches Python auth_v4 encoder\n";
        }
    }

    // Sign a canonical AUTH-like message and verify with OpenSSL.
    const std::string canonical =
        "COLLIDER-WORKER-AUTH-v1\n"
        "\x04"  // proto_ver byte
        "12345678"      // ts_ms placeholder
        "0123456789ABCDEF"  // nonce16 placeholder
        "\x2a"          // name_len = 42
        + id->bech32_address();

    auto sig = id->sign_message(
        reinterpret_cast<const uint8_t*>(canonical.data()),
        canonical.size());
    if (!sig) {
        ++g_fail;
        std::cerr << "[FAIL] sign_message returned nullopt\n";
        return g_fail;
    }
    ++g_pass;
    std::cout << "[PASS] sign_message produced 64-byte sig\n";

    // Verify with OpenSSL ECDSA_do_verify using the recovered EC_KEY.
    {
        uint8_t digest[32];
        SHA256(reinterpret_cast<const uint8_t*>(canonical.data()),
               canonical.size(), digest);

        EC_KEY* key = EC_KEY_new_by_curve_name(NID_secp256k1);
        const EC_GROUP* group = EC_KEY_get0_group(key);
        EC_POINT* pub = EC_POINT_new(group);
        if (!EC_POINT_oct2point(group, pub, id->pubkey_compressed().data(),
                                33, nullptr)) {
            ++g_fail;
            std::cerr << "[FAIL] EC_POINT_oct2point\n";
        } else if (!EC_KEY_set_public_key(key, pub)) {
            ++g_fail;
            std::cerr << "[FAIL] EC_KEY_set_public_key\n";
        } else {
            ECDSA_SIG* sig_obj = ECDSA_SIG_new();
            BIGNUM* r = BN_bin2bn(sig->data(), 32, nullptr);
            BIGNUM* s = BN_bin2bn(sig->data() + 32, 32, nullptr);
            ECDSA_SIG_set0(sig_obj, r, s);  // takes ownership
            int rv = ECDSA_do_verify(digest, 32, sig_obj, key);
            ECDSA_SIG_free(sig_obj);
            if (rv != 1) {
                ++g_fail;
                std::cerr << "[FAIL] ECDSA_do_verify returned " << rv << "\n";
            } else {
                ++g_pass;
                std::cout << "[PASS] OpenSSL verify (round-trip ok)\n";
            }
        }
        EC_POINT_free(pub);
        EC_KEY_free(key);
    }

    // Tampered signature: verify must reject.
    {
        auto bad = *sig;
        bad[0] ^= 0x01;
        uint8_t digest[32];
        SHA256(reinterpret_cast<const uint8_t*>(canonical.data()),
               canonical.size(), digest);

        EC_KEY* key = EC_KEY_new_by_curve_name(NID_secp256k1);
        const EC_GROUP* group = EC_KEY_get0_group(key);
        EC_POINT* pub = EC_POINT_new(group);
        EC_POINT_oct2point(group, pub, id->pubkey_compressed().data(), 33, nullptr);
        EC_KEY_set_public_key(key, pub);
        ECDSA_SIG* sig_obj = ECDSA_SIG_new();
        BIGNUM* r = BN_bin2bn(bad.data(), 32, nullptr);
        BIGNUM* s = BN_bin2bn(bad.data() + 32, 32, nullptr);
        ECDSA_SIG_set0(sig_obj, r, s);
        int rv = ECDSA_do_verify(digest, 32, sig_obj, key);
        ECDSA_SIG_free(sig_obj);
        EC_POINT_free(pub);
        EC_KEY_free(key);
        if (rv == 1) {
            ++g_fail;
            std::cerr << "[FAIL] tampered sig was accepted\n";
        } else {
            ++g_pass;
            std::cout << "[PASS] tampered sig rejected\n";
        }
    }

    // Uncompressed WIF must be refused (wire-v4 needs compressed).
    {
        std::string ucwif = collider::wif::encode(priv, /*compressed=*/false);
        auto bad = collider::identity::load_from_wif(ucwif, "bc");
        if (bad) {
            ++g_fail;
            std::cerr << "[FAIL] uncompressed WIF accepted\n";
        } else {
            ++g_pass;
            std::cout << "[PASS] uncompressed WIF rejected\n";
        }
    }

    std::cout << "\n=== Result: " << g_pass << " pass, " << g_fail
              << " fail ===\n";
    return g_fail == 0 ? 0 : 1;
}

#endif  // COLLIDER_HAS_OPENSSL
