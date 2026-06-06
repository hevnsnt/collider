/**
 * Signed-license-assertion + hit-verifier key-confirmation tests
 * (adversarial review v1.5.5, finding C4).
 *
 * The production license gate is an Ed25519 detached signature over a
 * canonical byte string, verified against a public key compiled into the
 * binary. These tests use a FRESH TEST KEYPAIR generated at runtime (never
 * the production key) so a forged-cache rejection can be demonstrated
 * without access to the issuer's private key.
 *
 * Coverage:
 *   (a) a hand-written / forged cache (no valid signature) is rejected
 *   (b) a properly test-key-signed assertion is accepted
 *   (c) a regressed monotonic counter is rejected (anti-rollback)
 *   (d) hit_verifier drops a hit whose private key does NOT derive to the
 *       matched hash160, and accepts one that does
 *
 * Network / filesystem paths are NOT exercised here; the algorithm under
 * test is verify_assertion + canonical_assertion_bytes + ed25519_verify and
 * HitVerifier::confirm_key.
 */

#ifndef COLLIDER_PRO
int main() { return 77; }   // skip on Free builds
#else

#include "../src/license/license_check.hpp"
#include "../src/core/hit_verifier.hpp"
#include "../src/core/crypto_cpu.hpp"

#include <openssl/evp.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace collider::license;

static int failures = 0;
#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        std::fprintf(stderr, "FAIL: %s   (%s:%d)\n",                \
                     msg, __FILE__, __LINE__);                      \
        ++failures;                                                 \
    }                                                               \
} while (0)

static long long now_epoch() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// ---------------------------------------------------------------------------
// Test-only Ed25519 keypair (NOT the production key).
// ---------------------------------------------------------------------------
// CHECK_OK: like CHECK but for the test's own crypto setup. CRITICAL: this
// must NOT use assert(), because the test builds with CMAKE_BUILD_TYPE=Release
// (NDEBUG defined) where assert() compiles to a no-op. The original version
// wrapped the load-bearing EVP_DigestSign calls in assert(), so under Release
// the signing was compiled OUT entirely and every signature came back empty,
// failing verify with size != 64. Evaluate the expression unconditionally and
// record a failure if it does not hold.
#define CHECK_OK(cond, msg) do {                                    \
    if (!(cond)) {                                                  \
        std::fprintf(stderr, "FAIL(setup): %s   (%s:%d)\n",         \
                     msg, __FILE__, __LINE__);                      \
        ++failures;                                                 \
    }                                                               \
} while (0)

struct TestKeypair {
    std::array<uint8_t, 32> pub{};
    EVP_PKEY* pkey = nullptr;

    TestKeypair() {
        EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_ED25519, nullptr);
        CHECK_OK(ctx != nullptr, "EVP_PKEY_CTX_new_id");
        CHECK_OK(EVP_PKEY_keygen_init(ctx) == 1, "EVP_PKEY_keygen_init");
        CHECK_OK(EVP_PKEY_keygen(ctx, &pkey) == 1, "EVP_PKEY_keygen");
        EVP_PKEY_CTX_free(ctx);
        size_t publen = pub.size();
        CHECK_OK(EVP_PKEY_get_raw_public_key(pkey, pub.data(), &publen) == 1,
                 "EVP_PKEY_get_raw_public_key");
        CHECK_OK(publen == 32, "raw pubkey is 32 bytes");
    }
    ~TestKeypair() { if (pkey) EVP_PKEY_free(pkey); }

    // Detached Ed25519 signature over msg (one-shot). Uses unconditional
    // return-value checks (NOT assert) so it works in Release builds.
    std::vector<uint8_t> sign(const std::vector<uint8_t>& msg) const {
        std::vector<uint8_t> sig;
        EVP_MD_CTX* md = EVP_MD_CTX_new();
        CHECK_OK(md != nullptr, "EVP_MD_CTX_new");
        if (!md) return sig;
        if (EVP_DigestSignInit(md, nullptr, nullptr, nullptr, pkey) == 1) {
            size_t siglen = 0;
            if (EVP_DigestSign(md, nullptr, &siglen,
                               msg.data(), msg.size()) == 1) {
                sig.resize(siglen);
                if (EVP_DigestSign(md, sig.data(), &siglen,
                                   msg.data(), msg.size()) == 1) {
                    sig.resize(siglen);
                } else {
                    sig.clear();
                    CHECK_OK(false, "EVP_DigestSign(final)");
                }
            } else {
                CHECK_OK(false, "EVP_DigestSign(size)");
            }
        } else {
            CHECK_OK(false, "EVP_DigestSignInit");
        }
        EVP_MD_CTX_free(md);
        return sig;
    }
};

static LicenseAssertion make_assertion(const std::string& key,
                                       const std::string& mid,
                                       long long expiry,
                                       long long counter,
                                       const TestKeypair& kp) {
    LicenseAssertion a;
    a.tier        = "pro";
    a.license_key = key;
    a.machine_id  = mid;
    a.expiry_unix = expiry;
    a.issued_unix = now_epoch();
    a.counter     = counter;
    a.signature   = kp.sign(canonical_assertion_bytes(a));
    return a;
}

static AssertionVerifyInput make_input(const LicenseAssertion& a,
                                       const TestKeypair& kp,
                                       const std::string& expected_mid,
                                       long long now,
                                       long long counter_floor) {
    AssertionVerifyInput in;
    in.assertion           = a;
    in.pubkey32            = kp.pub.data();
    in.expected_machine_id = expected_mid;
    in.now_epoch           = now;
    in.counter_floor       = counter_floor;
    return in;
}

// (b) A properly test-key-signed assertion is accepted.
static void test_accepts_signed_assertion() {
    TestKeypair kp;
    long long now = now_epoch();
    auto a = make_assertion("CLLDR-AAAA-BBBB-CCCC-DDDD", "machine-xyz",
                            now + 86400, 5, kp);
    auto in = make_input(a, kp, "machine-xyz", now, /*floor=*/0);
    CHECK(verify_assertion(in), "properly signed assertion accepted");
}

// (a) A hand-written / forged cache (no valid signature) is rejected.
static void test_rejects_forged_cache() {
    TestKeypair kp;
    long long now = now_epoch();

    // Forged: attacker hand-writes "valid forever" fields but cannot sign.
    LicenseAssertion forged;
    forged.tier        = "pro";
    forged.license_key = "CLLDR-AAAA-BBBB-CCCC-DDDD";
    forged.machine_id  = "machine-xyz";
    forged.expiry_unix = 4102444800;  // year 2100
    forged.issued_unix = now;
    forged.counter     = 999999;
    forged.signature.assign(64, 0x00);  // all-zero junk signature

    auto in = make_input(forged, kp, "machine-xyz", now, 0);
    CHECK(!verify_assertion(in), "forged cache with junk signature rejected");

    // Also: a genuine signature over DIFFERENT fields, then a field is
    // tampered post-sign (attacker pushes expiry far out).
    auto a = make_assertion("CLLDR-AAAA-BBBB-CCCC-DDDD", "machine-xyz",
                            now + 60, 5, kp);
    a.expiry_unix = 4102444800;  // tamper without re-signing
    auto in2 = make_input(a, kp, "machine-xyz", now, 0);
    CHECK(!verify_assertion(in2),
          "post-sign expiry tamper breaks the signature");

    // Also: assertion minted for a different host copied onto this one.
    auto a3 = make_assertion("CLLDR-AAAA-BBBB-CCCC-DDDD", "other-machine",
                             now + 86400, 5, kp);
    auto in3 = make_input(a3, kp, "machine-xyz", now, 0);
    CHECK(!verify_assertion(in3),
          "assertion bound to another machine rejected on this host");

    // Also: the WRONG public key (e.g. production key vs this test key)
    // cannot verify a test-signed assertion.
    TestKeypair other;
    auto in4 = make_input(a, other, "machine-xyz", now, 0);
    CHECK(!verify_assertion(in4),
          "signature does not verify under a different public key");
}

// (c) A regressed monotonic counter is rejected.
static void test_rejects_regressed_counter() {
    TestKeypair kp;
    long long now = now_epoch();

    // Server has already issued counter=10; the client floor is 10.
    auto current = make_assertion("CLLDR-K", "m", now + 86400, 10, kp);
    auto in_ok = make_input(current, kp, "m", now, /*floor=*/10);
    CHECK(verify_assertion(in_ok), "counter == floor accepted");

    // Attacker replays an OLDER assertion (counter=3) after the floor moved.
    auto old = make_assertion("CLLDR-K", "m", 4102444800, 3, kp);
    auto in_replay = make_input(old, kp, "m", now, /*floor=*/10);
    CHECK(!verify_assertion(in_replay),
          "older assertion (counter < floor) rejected (anti-rollback)");

    // A newer assertion (counter=11) still passes against floor=10.
    auto newer = make_assertion("CLLDR-K", "m", now + 86400, 11, kp);
    auto in_new = make_input(newer, kp, "m", now, /*floor=*/10);
    CHECK(verify_assertion(in_new), "newer counter accepted");
}

// Expired assertion (signature valid) is still rejected on the clock check.
static void test_rejects_expired() {
    TestKeypair kp;
    long long now = now_epoch();
    auto a = make_assertion("CLLDR-K", "m", now - 1, 5, kp);  // expired
    auto in = make_input(a, kp, "m", now, 0);
    CHECK(!verify_assertion(in), "expired (signed) assertion rejected");
}

// (d) hit_verifier key confirmation.
static void test_hit_verifier_confirm_key() {
    using ::collider::HitVerifier;

    // A fixed test private key (1) -> derive its real compressed-P2PKH h160.
    std::array<uint8_t, 32> priv{};
    priv[31] = 0x01;  // private key = 1 (valid secp256k1 scalar)

    auto real_h160 = collider::cpu::compute_hash160(priv.data());
    collider::utxo::H160 h160_match;
    std::memcpy(h160_match.data, real_h160.data(), 20);

    // Accept: the private key derives to the claimed compressed-P2PKH h160.
    CHECK(HitVerifier::confirm_key(priv.data(), /*addr_type=*/1, h160_match),
          "confirm_key accepts a key that derives to the matched h160 (P2PKH)");

    // Drop: a bogus h160 the key does not derive to.
    collider::utxo::H160 bogus;
    std::memset(bogus.data, 0xAB, 20);
    CHECK(!HitVerifier::confirm_key(priv.data(), 1, bogus),
          "confirm_key drops a hit whose key does not derive to the h160");

    // Drop: right h160 but WRONG address-type interpretation (uncompressed
    // P2PKH h160 differs from the compressed one for the same key).
    CHECK(!HitVerifier::confirm_key(priv.data(), /*addr_type=*/0, h160_match),
          "confirm_key drops on address-type mismatch (uncompressed vs compressed)");

    // Accept: uncompressed P2PKH h160 for the same key, addr_type=0.
    auto unc_h160 = collider::cpu::compute_hash160_uncompressed(priv.data());
    collider::utxo::H160 unc_match;
    std::memcpy(unc_match.data, unc_h160.data(), 20);
    CHECK(HitVerifier::confirm_key(priv.data(), /*addr_type=*/0, unc_match),
          "confirm_key accepts uncompressed P2PKH for addr_type=0");

    // Accept: P2SH-P2WPKH h160 for the same key, addr_type=2.
    auto p2sh_h160 = collider::cpu::compute_hash160_p2sh_p2wpkh(priv.data());
    collider::utxo::H160 p2sh_match;
    std::memcpy(p2sh_match.data, p2sh_h160.data(), 20);
    CHECK(HitVerifier::confirm_key(priv.data(), /*addr_type=*/2, p2sh_match),
          "confirm_key accepts P2SH-P2WPKH for addr_type=2");
}

int main() {
    test_accepts_signed_assertion();
    test_rejects_forged_cache();
    test_rejects_regressed_counter();
    test_rejects_expired();
    test_hit_verifier_confirm_key();

    if (failures != 0) {
        std::fprintf(stderr, "test_license_assertion: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_license_assertion: PASS\n");
    return 0;
}

#endif  // COLLIDER_PRO
