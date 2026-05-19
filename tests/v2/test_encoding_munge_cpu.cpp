/**
 * Encoding-munger CPU reference KATs (Phase 6, v1.4.0).
 *
 * Tests each Encoding value against a known input/output. Inputs are
 * given as UTF-8 (the canonical wire form); outputs are the bytes a
 * generator would feed into SHA-256 if it used the named encoding.
 */

#include "../../src/gpu/v2/encoding_munge_cpu.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace collider::gpu::v2::encmunge;

static int failures = 0;
#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        std::fprintf(stderr, "FAIL: %s   (%s:%d)\n",                \
                     msg, __FILE__, __LINE__);                      \
        ++failures;                                                 \
    }                                                               \
} while (0)

static std::string hex(const std::vector<uint8_t>& v) {
    static const char* d = "0123456789abcdef";
    std::string s; s.reserve(v.size() * 2);
    for (auto b : v) {
        s.push_back(d[b >> 4]); s.push_back(d[b & 0xF]);
    }
    return s;
}

static std::vector<uint8_t> from_str(const char* s) {
    return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(s),
                                reinterpret_cast<const uint8_t*>(s) + std::strlen(s));
}

static void test_utf8_identity() {
    std::vector<uint8_t> out;
    auto in = from_str("abc");
    CHECK(munge(Encoding::UTF8, in.data(), in.size(), out), "utf8 ok");
    CHECK(hex(out) == "616263", "utf8 'abc' -> 616263");
}

static void test_utf16_le_ascii() {
    std::vector<uint8_t> out;
    auto in = from_str("abc");
    CHECK(munge(Encoding::UTF16_LE, in.data(), in.size(), out), "utf16_le ok");
    // 'a' = 61 00, 'b' = 62 00, 'c' = 63 00
    CHECK(hex(out) == "610062006300", "utf16_le 'abc' -> 610062006300");
}

static void test_utf16_be_ascii() {
    std::vector<uint8_t> out;
    auto in = from_str("abc");
    CHECK(munge(Encoding::UTF16_BE, in.data(), in.size(), out), "utf16_be ok");
    CHECK(hex(out) == "006100620063", "utf16_be 'abc' -> 006100620063");
}

static void test_utf32_le_ascii() {
    std::vector<uint8_t> out;
    auto in = from_str("abc");
    CHECK(munge(Encoding::UTF32_LE, in.data(), in.size(), out), "utf32_le ok");
    // 'a' = 61 00 00 00 ... etc
    CHECK(hex(out) == "610000006200000063000000",
          "utf32_le 'abc' -> 610000006200000063000000");
}

static void test_latin1_basic() {
    std::vector<uint8_t> out;
    // U+00E9 (é) is 0xE9 in latin1; UTF-8 it's 0xC3 0xA9.
    auto in = from_str("\xC3\xA9");
    CHECK(munge(Encoding::LATIN1, in.data(), in.size(), out),
          "latin1 accepts U+00E9");
    CHECK(hex(out) == "e9", "latin1 -> 0xe9");
}

static void test_latin1_rejects_emoji() {
    std::vector<uint8_t> out;
    // U+1F600 (😀) UTF-8: F0 9F 98 80
    auto in = from_str("\xF0\x9F\x98\x80");
    CHECK(!munge(Encoding::LATIN1, in.data(), in.size(), out),
          "latin1 rejects U+1F600");
}

static void test_strip_non_ascii() {
    std::vector<uint8_t> out;
    // "ab\xc3\xa9c" -- 'a' 'b' 0xC3 0xA9 'c'
    std::vector<uint8_t> in = {'a', 'b', 0xC3, 0xA9, 'c'};
    CHECK(munge(Encoding::STRIP_NON_ASCII, in.data(), in.size(), out),
          "strip_non_ascii ok");
    CHECK(hex(out) == "616263", "strip removes high bytes");
}

static void test_upper_lower_ascii() {
    std::vector<uint8_t> u, l;
    auto in = from_str("aBc1");
    CHECK(munge(Encoding::UPPER_ASCII, in.data(), in.size(), u), "upper ok");
    CHECK(hex(u) == "41424331", "UPPER 'aBc1' -> ABC1");
    CHECK(munge(Encoding::LOWER_ASCII, in.data(), in.size(), l), "lower ok");
    CHECK(hex(l) == "61626331", "LOWER 'aBc1' -> abc1");
}

static void test_utf16_surrogate_pair() {
    std::vector<uint8_t> out;
    // U+1F600 (😀) UTF-8: F0 9F 98 80
    // UTF-16-LE: high surrogate D83D, low surrogate DE00 -> 3DD8 00DE
    auto in = from_str("\xF0\x9F\x98\x80");
    CHECK(munge(Encoding::UTF16_LE, in.data(), in.size(), out),
          "utf16_le surrogate pair ok");
    CHECK(hex(out) == "3dd800de", "U+1F600 -> 3dd800de");
}

// task B: NULL_TERMINATED + NFC/NFD coverage.

static void test_null_terminated() {
    std::vector<uint8_t> out;
    auto in = from_str("abc");
    CHECK(munge(Encoding::NULL_TERMINATED, in.data(), in.size(), out),
          "null_terminated ok");
    CHECK(hex(out) == "61626300", "NULL_TERMINATED 'abc' -> 61626300");
}

static void test_nfd_cafe() {
    // "café" in NFC: 'c' 'a' 'f' U+00E9 = 63 61 66 C3A9
    // After NFD: 'c' 'a' 'f' 'e' U+0301 = 63 61 66 65 CC81
    std::vector<uint8_t> out;
    auto in = from_str("caf\xC3\xA9");
    CHECK(munge(Encoding::NFD, in.data(), in.size(), out), "nfd 'café' ok");
    CHECK(hex(out) == "6361666565cc81" || hex(out) == "636166" "65" "cc81",
          "NFD 'café' -> 'cafe' + U+0301");
    // Simpler form: 63 61 66 65 cc 81
    CHECK(hex(out) == "63616665cc81", "NFD 'café' bytes");
}

static void test_nfc_cafe() {
    // Input in NFD form: 63 61 66 65 cc 81 ('cafe' + combining acute)
    std::vector<uint8_t> in = {0x63, 0x61, 0x66, 0x65, 0xCC, 0x81};
    std::vector<uint8_t> out;
    CHECK(munge(Encoding::NFC, in.data(), in.size(), out), "nfc 'café' ok");
    // NFC: 63 61 66 C3 A9
    CHECK(hex(out) == "636166c3a9", "NFC NFD-cafe -> precomposed");
}

static void test_nfd_naive() {
    // "naïve" UTF-8 NFC: 6E 61 C3AF 76 65
    std::vector<uint8_t> out;
    auto in = from_str("na\xC3\xAFve");
    CHECK(munge(Encoding::NFD, in.data(), in.size(), out), "nfd 'naïve' ok");
    // NFD: 6E 61 69 cc 88 76 65 ('i' + U+0308 combining diaeresis)
    CHECK(hex(out) == "6e6169cc887665", "NFD 'naïve' -> n a i U+0308 v e");
}

static void test_nfc_idempotent() {
    // NFC applied to already-precomposed input is the identity.
    std::vector<uint8_t> out;
    auto in = from_str("caf\xC3\xA9");
    CHECK(munge(Encoding::NFC, in.data(), in.size(), out), "nfc on NFC ok");
    CHECK(hex(out) == "636166c3a9", "NFC idempotent");
}

static void test_nfd_ascii_passthrough() {
    // NFD on pure ASCII is the identity.
    std::vector<uint8_t> out;
    auto in = from_str("hello");
    CHECK(munge(Encoding::NFD, in.data(), in.size(), out), "nfd ascii ok");
    CHECK(hex(out) == "68656c6c6f", "NFD on ASCII -> identity");
}

int main() {
    test_utf8_identity();
    test_utf16_le_ascii();
    test_utf16_be_ascii();
    test_utf32_le_ascii();
    test_latin1_basic();
    test_latin1_rejects_emoji();
    test_strip_non_ascii();
    test_upper_lower_ascii();
    test_utf16_surrogate_pair();
    test_null_terminated();
    test_nfd_cafe();
    test_nfc_cafe();
    test_nfd_naive();
    test_nfc_idempotent();
    test_nfd_ascii_passthrough();

    if (failures != 0) {
        std::fprintf(stderr, "test_encoding_munge_cpu: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_encoding_munge_cpu: PASS\n");
    return 0;
}
