/**
 * License cache lifecycle tests (Phase 10, v1.4.0).
 *
 * Covers the HMAC-bound cache that gates Pro features after a successful
 * remote license validation. Without these tests, a regression that weakens
 * the HMAC could let a forged cache file unlock Pro indefinitely.
 *
 * Coverage:
 *   * verify_cache_entry accepts a properly-signed, unexpired entry whose
 *     embedded key matches the license key
 *   * tampered valid bit -> rejected
 *   * tampered email     -> rejected
 *   * tampered expiry    -> rejected
 *   * mismatched key     -> rejected
 *   * expired (now > expiry) -> rejected
 *   * HMAC for the same payload differs across distinct license keys
 *
 * Network / file-system paths are NOT exercised here -- those need a
 * real TLS endpoint and live in tests/test_license_e2e.cpp (follow-up).
 */

#ifndef COLLIDER_PRO
int main() { return 77; }   // skip on Free builds
#else

#include "../src/license/license_check.hpp"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <string>

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

// Produce a fully-signed, valid cache entry for input fixtures.
static CacheVerifyInput build_valid(const std::string& license_key,
                                    bool valid,
                                    const std::string& email,
                                    long long expiry_epoch,
                                    long long now) {
    CacheVerifyInput in;
    in.license_key   = license_key;
    in.embedded_key  = license_key;
    in.valid         = valid;
    in.email         = email;
    in.expiry_epoch  = expiry_epoch;
    in.now_epoch     = now;
    std::string payload = make_cache_payload(in.embedded_key, in.valid,
                                              in.email, in.expiry_epoch);
    in.claimed_hmac = compute_cache_hmac(in.license_key, payload);
    return in;
}

static void test_accepts_well_formed_entry() {
    long long now = now_epoch();
    auto in = build_valid("LIC-AAAA-BBBB-CCCC", true,
                          "user@example.com", now + 86400, now);
    CHECK(verify_cache_entry(in),
          "well-formed unexpired cache accepted");
}

static void test_rejects_tampered_valid_bit() {
    long long now = now_epoch();
    auto in = build_valid("LIC-K", false, "u@e.com", now + 86400, now);
    in.valid = true;  // flip without re-signing
    CHECK(!verify_cache_entry(in),
          "tampered valid bit rejected");
}

static void test_rejects_tampered_email() {
    long long now = now_epoch();
    auto in = build_valid("LIC-K", true, "real@e.com", now + 86400, now);
    in.email = "attacker@e.com";
    CHECK(!verify_cache_entry(in), "tampered email rejected");
}

static void test_rejects_tampered_expiry() {
    long long now = now_epoch();
    auto in = build_valid("LIC-K", true, "u@e.com", now + 1, now);  // expires in 1 sec
    // Push expiry forward without re-signing.
    in.expiry_epoch = now + 1000000;
    CHECK(!verify_cache_entry(in), "tampered expiry rejected");
}

static void test_rejects_mismatched_key() {
    long long now = now_epoch();
    auto in = build_valid("LIC-A", true, "u@e.com", now + 86400, now);
    in.license_key = "LIC-B";  // claim a different key
    // embedded_key still says LIC-A, so the equality check trips.
    CHECK(!verify_cache_entry(in),
          "license_key != embedded_key rejected");
}

static void test_rejects_expired_entry() {
    long long now = now_epoch();
    auto in = build_valid("LIC-K", true, "u@e.com",
                          now - 1,  // expiry in the past
                          now);
    CHECK(!verify_cache_entry(in), "expired entry rejected");
}

static void test_hmac_differs_across_license_keys() {
    long long now = now_epoch();
    std::string payload = make_cache_payload(
        "LIC-A", true, "u@e.com", now + 86400);
    std::string mac_a = compute_cache_hmac("LIC-A", payload);
    std::string mac_b = compute_cache_hmac("LIC-B", payload);
    CHECK(!mac_a.empty(), "HMAC produces non-empty result");
    CHECK(mac_a != mac_b,
          "same payload + different license keys -> different HMACs");
}

static void test_hmac_is_64_hex_chars() {
    long long now = now_epoch();
    std::string payload = make_cache_payload(
        "LIC-K", true, "u@e.com", now + 86400);
    std::string mac = compute_cache_hmac("LIC-K", payload);
    CHECK(mac.size() == 64, "HMAC-SHA256 hex output is 64 chars");
    for (char c : mac) {
        bool is_hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        CHECK(is_hex, "HMAC hex output is lowercase hex");
    }
}

int main() {
    test_accepts_well_formed_entry();
    test_rejects_tampered_valid_bit();
    test_rejects_tampered_email();
    test_rejects_tampered_expiry();
    test_rejects_mismatched_key();
    test_rejects_expired_entry();
    test_hmac_differs_across_license_keys();
    test_hmac_is_64_hex_chars();

    if (failures != 0) {
        std::fprintf(stderr, "test_license_cache: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_license_cache: PASS\n");
    return 0;
}

#endif  // COLLIDER_PRO
