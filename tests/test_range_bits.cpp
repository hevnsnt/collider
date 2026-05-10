/**
 * Known-answer test for range_bits_from_be in src/core/byte_codec.hpp.
 *
 * The helper is used by CudaRCKangarooBackend::initialize() to derive
 * range_bits from a pool work assignment's [range_start, range_end)
 * span. A wrong return value silently mis-budgets the kangaroo solver
 * (jump-table magnitudes, K-factor estimates), so this KAT pins down
 * the canonical Bitcoin-puzzle ranges plus a few edge cases.
 *
 * No GPU dependency; runs as a pure host test.
 */

#include "../src/core/byte_codec.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

// Convenience: BE-encode a uint128 (high 16 bytes zero) into a 32-byte BE buffer.
void encode_be(uint64_t hi, uint64_t lo, uint8_t out[32]) {
    std::memset(out, 0, 32);
    for (int i = 0; i < 8; ++i) {
        out[16 + i] = static_cast<uint8_t>(hi >> (56 - i * 8));
        out[24 + i] = static_cast<uint8_t>(lo >> (56 - i * 8));
    }
}

// BE-encode a 256-bit value where the top bit position is at `bit`
// (so end - start = 2^bit). For a Bitcoin puzzle of N bits the range is
// [2^(N-1), 2^N - 1] which has size 2^(N-1).
void encode_pow2_minus_1(int bits, uint8_t out[32]) {
    // out = 2^bits - 1 in BE
    std::memset(out, 0, 32);
    for (int i = 0; i < bits; ++i) {
        const int byte_pos = 31 - (i / 8);
        out[byte_pos] |= static_cast<uint8_t>(1u << (i % 8));
    }
}

void encode_pow2(int bits, uint8_t out[32]) {
    std::memset(out, 0, 32);
    if (bits >= 256) return;
    const int byte_pos = 31 - (bits / 8);
    out[byte_pos] = static_cast<uint8_t>(1u << (bits % 8));
}

int g_failures = 0;

void check(const char* name, int expected, int actual) {
    if (expected == actual) {
        std::printf("[ok  ] %s: %d\n", name, actual);
    } else {
        std::printf("[FAIL] %s: expected %d got %d\n", name, expected, actual);
        ++g_failures;
    }
}

}  // namespace

int main() {
    // Canonical Bitcoin puzzle ranges: [2^(N-1), 2^N).
    // range_size = 2^N - 2^(N-1) = 2^(N-1). bit_length = N-1, ceil(log2)
    // returns N-1 + 1 = N? No, bit_length(2^k) = k+1, so range_bits_from_be
    // on 2^N - 1 - (2^(N-1)) = 2^(N-1) returns N (since the helper
    // computes ceil(log2(diff)) and diff = 2^(N-1) has bit_length N).
    //
    // Wait: ceil(log2(2^(N-1))) = N-1, since log2(2^(N-1)) = N-1 exactly.
    // The helper returns "highest set bit + 1" of diff.
    // For diff = 2^(N-1), the highest set bit is at position N-1, so
    // helper returns N-1 + 1 = N. For Puzzle 135, that returns 135.
    //
    // So: puzzle N has range_bits = N when the range is the canonical
    // [2^(N-1), 2^N - 1].
    {
        uint8_t start[32], end[32];
        encode_pow2(74, start);          // 2^74
        encode_pow2_minus_1(75, end);    // 2^75 - 1
        check("puzzle 75", 75, ::collider::range_bits_from_be(start, end));
    }
    {
        uint8_t start[32], end[32];
        encode_pow2(134, start);
        encode_pow2_minus_1(135, end);
        check("puzzle 135", 135, ::collider::range_bits_from_be(start, end));
    }
    {
        uint8_t start[32], end[32];
        encode_pow2(159, start);
        encode_pow2_minus_1(160, end);
        check("puzzle 160", 160, ::collider::range_bits_from_be(start, end));
    }

    // Edge: end == start -> empty range -> 0.
    {
        uint8_t start[32], end[32];
        encode_be(0x0000000000000001ull, 0x0000000000000000ull, start);
        std::memcpy(end, start, 32);
        check("empty range start==end", 0,
              ::collider::range_bits_from_be(start, end));
    }

    // Edge: end < start -> inverted -> 0.
    {
        uint8_t start[32], end[32];
        encode_be(0, 100, start);
        encode_be(0, 50, end);
        check("inverted range end<start", 0,
              ::collider::range_bits_from_be(start, end));
    }

    // size = end - start + 1 = 2 -> bit_length(2) = 2.
    {
        uint8_t start[32], end[32];
        encode_be(0, 100, start);
        encode_be(0, 101, end);
        check("size = 2 (end=start+1)", 2,
              ::collider::range_bits_from_be(start, end));
    }

    // size = 0x10000 (17 bits to represent 2^16) -> bit_length is 17.
    {
        uint8_t start[32], end[32];
        encode_be(0, 0, start);
        encode_be(0, 0xFFFF, end);
        check("size = 2^16 (end=0xFFFF)", 17,
              ::collider::range_bits_from_be(start, end));
    }

    // diff = 2^65 -> size = 2^65 + 1, bit_length is 66.
    {
        uint8_t start[32], end[32];
        encode_be(0, 0, start);
        encode_be(2u, 0, end);  // bits 0..63 are zero, bit 65 set
        check("size = 2^65 + 1", 66,
              ::collider::range_bits_from_be(start, end));
    }

    if (g_failures > 0) {
        std::printf("FAIL: %d range_bits_from_be cases failed\n", g_failures);
        return 1;
    }
    std::printf("test_range_bits: %d/%d PASS\n", 7, 7);
    return 0;
}
