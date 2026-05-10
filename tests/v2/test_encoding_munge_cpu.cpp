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

    if (failures != 0) {
        std::fprintf(stderr, "test_encoding_munge_cpu: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_encoding_munge_cpu: PASS\n");
    return 0;
}
