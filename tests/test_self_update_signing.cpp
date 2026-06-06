// test_self_update_signing.cpp
//
// Tests the C3 self-update signature gate:
//   * build_update_manifest produces the documented canonical byte string.
//   * verify_update_manifest_with_key REJECTS a zeroed signature (the
//     "unsigned" sentinel) and a forged signature, with NO key able to make
//     a zero signature pass.
//   * Round-trip: signing the canonical manifest with a TEST Ed25519 key and
//     verifying with that test pubkey via the real verify path ACCEPTS it,
//     and any tampering with a manifest field makes it FAIL.
//
// A fresh TEST keypair is generated here via OpenSSL; the production release
// key is never used. collider_core links OpenSSL and defines
// COLLIDER_HAS_OPENSSL transitively, so EVP is available to the test.

#include "runtime/self_update.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <openssl/evp.h>

using collider::update::build_update_manifest;
using collider::update::verify_update_manifest_with_key;

namespace {

int g_failures = 0;

void check(bool cond, const char* expr) {
    if (!cond) {
        std::fprintf(stderr, "[FAIL] %s\n", expr);
        ++g_failures;
    }
}

#define CHECK(expr) check((expr), #expr)

// Generate a fresh Ed25519 keypair. Fills `pub` (raw 32 bytes) and returns
// the EVP_PKEY (caller frees). Returns nullptr on failure.
EVP_PKEY* gen_test_keypair(std::array<uint8_t, 32>& pub) {
    EVP_PKEY* pkey = nullptr;
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
    if (!ctx) return nullptr;
    if (EVP_PKEY_keygen_init(ctx) != 1 || EVP_PKEY_keygen(ctx, &pkey) != 1) {
        EVP_PKEY_CTX_free(ctx);
        return nullptr;
    }
    EVP_PKEY_CTX_free(ctx);
    size_t publen = pub.size();
    if (EVP_PKEY_get_raw_public_key(pkey, pub.data(), &publen) != 1 ||
        publen != pub.size()) {
        EVP_PKEY_free(pkey);
        return nullptr;
    }
    return pkey;
}

// Sign `msg` with the Ed25519 private key. Returns a 64-byte signature.
std::array<uint8_t, 64> sign_msg(EVP_PKEY* pkey,
                                 const std::vector<uint8_t>& msg) {
    std::array<uint8_t, 64> sig{};
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) return sig;
    size_t siglen = sig.size();
    if (EVP_DigestSignInit(ctx, nullptr, nullptr, nullptr, pkey) == 1) {
        EVP_DigestSign(ctx, sig.data(), &siglen, msg.data(), msg.size());
    }
    EVP_MD_CTX_free(ctx);
    return sig;
}

}  // namespace

int main() {
    std::printf("=== self_update signature gate tests ===\n");

    const std::string latest = "1.6.0";
    const std::string min_ver = "1.5.0";
    const std::string url = "https://dl.example.com/collider-1.6.0.exe";
    std::array<uint8_t, 32> sha{};
    for (size_t i = 0; i < sha.size(); ++i) sha[i] = static_cast<uint8_t>(i + 1);

    // --- Canonical manifest format ----------------------------------------
    // DOMAIN(19) || u16le(5)||"1.6.0" || u16le(5)||"1.5.0"
    //   || u16le(len(url))||url || sha256(32)
    const std::vector<uint8_t> manifest =
        build_update_manifest(latest, min_ver, url, sha);
    const std::string domain = "COLLIDER-UPDATE-v1\n";  // 19 bytes
    size_t expected_len = domain.size() + 2 + latest.size() + 2 +
                          min_ver.size() + 2 + url.size() + 32;
    CHECK(manifest.size() == expected_len);
    // Domain prefix is present and exact.
    CHECK(std::string(manifest.begin(),
                      manifest.begin() + static_cast<long>(domain.size())) ==
          domain);
    // First length prefix is little-endian 5 (len("1.6.0")).
    CHECK(manifest[domain.size()] == 0x05 && manifest[domain.size() + 1] == 0x00);

    // Generate a TEST keypair (NOT the production release key).
    std::array<uint8_t, 32> test_pub{};
    EVP_PKEY* test_key = gen_test_keypair(test_pub);
    CHECK(test_key != nullptr);
    if (!test_key) {
        std::printf("could not generate test keypair; aborting\n");
        return 1;
    }

    // --- Round-trip: valid signature with the test key is ACCEPTED --------
    const std::array<uint8_t, 64> good_sig = sign_msg(test_key, manifest);
    CHECK(verify_update_manifest_with_key(test_pub, latest, min_ver, url, sha,
                                          good_sig));

    // --- Zeroed signature is REJECTED (the "unsigned" sentinel) -----------
    const std::array<uint8_t, 64> zero_sig{};
    CHECK(!verify_update_manifest_with_key(test_pub, latest, min_ver, url, sha,
                                           zero_sig));

    // --- Forged signature (flip one byte of a good sig) is REJECTED -------
    std::array<uint8_t, 64> forged = good_sig;
    forged[0] ^= 0x01;
    CHECK(!verify_update_manifest_with_key(test_pub, latest, min_ver, url, sha,
                                           forged));

    // --- Tampered manifest field is REJECTED with the same good sig -------
    // An attacker who swaps the download_url (or version, or hash) but keeps
    // the original signature must fail: the signature is over the ORIGINAL
    // bytes, so any field change breaks it.
    CHECK(!verify_update_manifest_with_key(
        test_pub, latest, min_ver,
        "https://evil.example.com/payload.exe", sha, good_sig));
    CHECK(!verify_update_manifest_with_key(test_pub, "9.9.9", min_ver, url, sha,
                                           good_sig));
    std::array<uint8_t, 32> sha2 = sha;
    sha2[31] ^= 0xFF;
    CHECK(!verify_update_manifest_with_key(test_pub, latest, min_ver, url, sha2,
                                           good_sig));

    // --- A DIFFERENT key (wrong signer) is REJECTED -----------------------
    // A signature from one key must not verify under another pubkey: this is
    // exactly the malicious-pool case (their key is not the release key).
    std::array<uint8_t, 32> other_pub{};
    EVP_PKEY* other_key = gen_test_keypair(other_pub);
    CHECK(other_key != nullptr);
    if (other_key) {
        const std::array<uint8_t, 64> other_sig = sign_msg(other_key, manifest);
        // other_sig is valid under other_pub but NOT under test_pub.
        CHECK(verify_update_manifest_with_key(other_pub, latest, min_ver, url,
                                              sha, other_sig));
        CHECK(!verify_update_manifest_with_key(test_pub, latest, min_ver, url,
                                               sha, other_sig));
        EVP_PKEY_free(other_key);
    }

    EVP_PKEY_free(test_key);

    if (g_failures == 0) {
        std::printf("all self_update signature tests passed\n");
    } else {
        std::printf("%d self_update signature test(s) FAILED\n", g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}
