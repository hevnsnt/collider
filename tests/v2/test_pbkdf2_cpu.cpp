/**
 * SHA-512 / HMAC / PBKDF2 CPU reference KATs (Phase 7/8, v1.4.0).
 *
 * Vectors:
 *   * SHA-512 "abc" : NIST CAVS published.
 *   * HMAC-SHA512   : RFC 4231 test case 1.
 *   * HMAC-SHA256   : RFC 4231 test case 1.
 *   * PBKDF2-HMAC-SHA512 : RFC 6070-style + BIP-39 reference.
 *   * PBKDF2-HMAC-SHA256 : RFC 6070 test case 1.
 *
 * Why this matters: BIP-39 mnemonic seed derivation is
 * PBKDF2-HMAC-SHA512(2048 iters), which is the bottleneck of every
 * BIP-39 brute force. Validating the reference here pins that path.
 */

#include "../../src/gpu/v2/sha512_cpu.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace collider::gpu::v2::internal;

static int failures = 0;
#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        std::fprintf(stderr, "FAIL: %s   (%s:%d)\n",                \
                     msg, __FILE__, __LINE__);                      \
        ++failures;                                                 \
    }                                                               \
} while (0)

static std::string hex_encode(const uint8_t* b, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s; s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(d[(b[i] >> 4) & 0xF]);
        s.push_back(d[b[i] & 0xF]);
    }
    return s;
}

// ---------------------------------------------------------------------------
// SHA-512 "abc" vector
// expected: ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a
//           2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f
// ---------------------------------------------------------------------------
static void test_sha512_abc() {
    uint8_t out[64];
    sha512(reinterpret_cast<const uint8_t*>("abc"), 3, out);
    CHECK(hex_encode(out, 64) ==
          "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
          "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
          "SHA-512 'abc' matches NIST vector");
}

static void test_sha512_empty() {
    uint8_t out[64];
    sha512(nullptr, 0, out);
    CHECK(hex_encode(out, 64) ==
          "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
          "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
          "SHA-512 empty matches NIST vector");
}

// ---------------------------------------------------------------------------
// HMAC-SHA512: RFC 4231 test case 1.
// key = 0x0b * 20, data = "Hi There"
// expected = 87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cde
//            daa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854
// ---------------------------------------------------------------------------
static void test_hmac_sha512_rfc4231_tc1() {
    uint8_t key[20];
    for (int i = 0; i < 20; ++i) key[i] = 0x0b;
    const char* msg = "Hi There";
    uint8_t mac[64];
    hmac_sha512(key, 20, reinterpret_cast<const uint8_t*>(msg), 8, mac);
    CHECK(hex_encode(mac, 64) ==
          "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cde"
          "daa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854",
          "HMAC-SHA512 RFC 4231 TC1");
}

// ---------------------------------------------------------------------------
// HMAC-SHA256: RFC 4231 test case 1.
// key = 0x0b * 20, data = "Hi There"
// expected = b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7
// ---------------------------------------------------------------------------
static void test_hmac_sha256_rfc4231_tc1() {
    uint8_t key[20];
    for (int i = 0; i < 20; ++i) key[i] = 0x0b;
    const char* msg = "Hi There";
    uint8_t mac[32];
    hmac_sha256(key, 20, reinterpret_cast<const uint8_t*>(msg), 8, mac);
    CHECK(hex_encode(mac, 32) ==
          "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7",
          "HMAC-SHA256 RFC 4231 TC1");
}

// ---------------------------------------------------------------------------
// PBKDF2-HMAC-SHA256: RFC 6070 test case 1.
// pw="password", salt="salt", iters=1, dkLen=20
// expected = 120fb6cffcf8b32c43e7225256c4f837a86548c92ccc35480805987cb70be17b
//            (first 20 bytes for the test)
//   first 20 = 120fb6cffcf8b32c43e7225256c4f837a86548c9
// ---------------------------------------------------------------------------
static void test_pbkdf2_hmac_sha256_rfc6070_tc1() {
    const char* pw = "password";
    const char* salt = "salt";
    uint8_t out[20];
    pbkdf2_hmac_sha256(reinterpret_cast<const uint8_t*>(pw), 8,
                       reinterpret_cast<const uint8_t*>(salt), 4,
                       1, out, 20);
    CHECK(hex_encode(out, 20) ==
          "120fb6cffcf8b32c43e7225256c4f837a86548c9",
          "PBKDF2-HMAC-SHA256 RFC 6070 TC1");
}

// ---------------------------------------------------------------------------
// PBKDF2-HMAC-SHA256: RFC 6070 test case 2.
// pw="password", salt="salt", iters=2, dkLen=20
// expected = ae4d0c95af6b46d32d0adff928f06dd02a303f8e
// ---------------------------------------------------------------------------
static void test_pbkdf2_hmac_sha256_rfc6070_tc2() {
    const char* pw = "password";
    const char* salt = "salt";
    uint8_t out[20];
    pbkdf2_hmac_sha256(reinterpret_cast<const uint8_t*>(pw), 8,
                       reinterpret_cast<const uint8_t*>(salt), 4,
                       2, out, 20);
    CHECK(hex_encode(out, 20) ==
          "ae4d0c95af6b46d32d0adff928f06dd02a303f8e",
          "PBKDF2-HMAC-SHA256 RFC 6070 TC2");
}

// ---------------------------------------------------------------------------
// PBKDF2-HMAC-SHA512 BIP-39 reference vector.
// mnemonic = "abandon abandon abandon abandon abandon abandon abandon
//             abandon abandon abandon abandon about"
// passphrase = "TREZOR"
// salt = "mnemonic" + passphrase
// iterations = 2048, output = 64 bytes
// expected seed (first 64 bytes from BIP-39 test vectors):
//   c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e53495531f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04
// ---------------------------------------------------------------------------
static void test_pbkdf2_bip39_abandon() {
    const std::string mnemonic =
        "abandon abandon abandon abandon abandon abandon "
        "abandon abandon abandon abandon abandon about";
    const std::string salt = "mnemonic" + std::string("TREZOR");
    uint8_t seed[64];
    pbkdf2_hmac_sha512(
        reinterpret_cast<const uint8_t*>(mnemonic.data()), mnemonic.size(),
        reinterpret_cast<const uint8_t*>(salt.data()), salt.size(),
        2048, seed, 64);
    CHECK(hex_encode(seed, 64) ==
          "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e5349553"
          "1f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04",
          "PBKDF2-HMAC-SHA512 BIP-39 abandon TREZOR");
}

int main() {
    test_sha512_abc();
    test_sha512_empty();
    test_hmac_sha512_rfc4231_tc1();
    test_hmac_sha256_rfc4231_tc1();
    test_pbkdf2_hmac_sha256_rfc6070_tc1();
    test_pbkdf2_hmac_sha256_rfc6070_tc2();
    test_pbkdf2_bip39_abandon();

    if (failures != 0) {
        std::fprintf(stderr, "test_pbkdf2_cpu: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_pbkdf2_cpu: PASS\n");
    return 0;
}
