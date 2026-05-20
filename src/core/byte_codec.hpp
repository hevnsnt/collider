/**
 * Big-endian 32-byte <-> 4-limb little-endian-by-limb uint64 codec.
 *
 * The repeated pattern across the host code: pubkeys, scalar ranges, DP
 * coordinates, and kangaroo state all use 32-byte big-endian on the wire
 * and 4xu64 little-endian-by-limb in arithmetic. Six near-identical copies
 * of these conversion loops grew up across kangaroo_metal.mm,
 * test_metal_secp256k1.mm, and main.cpp before this header existed.
 *
 * Both `cpu::uint256_t::d` and `puzzle::UInt256::parts` are `uint64_t[4]`
 * with limb[0] holding bits 0..63, so callers can pass `obj.d` or
 * `obj.parts` directly via array-to-pointer decay. The MSL kernel side
 * keeps its own copy because Metal can't include host headers.
 */

#pragma once

#include <cstddef>  // size_t (required on Apple clang; MSVC pulls it in via cstdint)
#include <cstdint>

namespace collider {

// Convert 32-byte big-endian byte array to 4-limb little-endian-by-limb
// uint64 array. out[0] holds bits 0..63 (i.e., the low 8 bytes of the
// big-endian input); out[3] holds bits 192..255 (the high 8 bytes).
inline void be32_to_limbs_le(const uint8_t in[32], uint64_t out[4]) {
    for (int i = 0; i < 4; ++i) {
        uint64_t v = 0;
        for (int j = 0; j < 8; ++j) {
            v = (v << 8) | static_cast<uint64_t>(in[(3 - i) * 8 + j]);
        }
        out[i] = v;
    }
}

// Inverse of be32_to_limbs_le: 4xu64 LE-by-limb -> 32 bytes BE.
inline void limbs_le_to_be32(const uint64_t in[4], uint8_t out[32]) {
    for (int i = 0; i < 4; ++i) {
        const uint64_t v = in[3 - i];
        for (int j = 0; j < 8; ++j) {
            out[i * 8 + j] = static_cast<uint8_t>(v >> (56 - 8 * j));
        }
    }
}

// Encode `len` bytes as a lowercase ASCII hex string into `out`. The
// caller is responsible for sizing out as 2*len + 1 bytes (NUL-
// terminated). Used everywhere the legacy `snprintf(buf+i*2,3,"%02x",
// data[i])` pattern shows up (~13 call sites pre-1.4); centralized
// here so the snprintf overhead can be removed in one place if the
// future profiler flags it.
inline void hex_encode_lower(const uint8_t* in, size_t len, char* out) {
    static constexpr char kHex[] = "0123456789abcdef";
    for (size_t i = 0; i < len; ++i) {
        out[i * 2]     = kHex[(in[i] >> 4) & 0xFu];
        out[i * 2 + 1] = kHex[in[i]        & 0xFu];
    }
    out[len * 2] = '\0';
}

// Decode an ASCII hex string into `out_len` bytes. Returns true on
// success, false if `hex_len` is odd, exceeds `out_len * 2`, or
// contains a non-[0-9a-fA-F] character. Strips an optional leading
// "0x" or "0X". Bytes are written in stream order: the first hex pair
// becomes out[0], the second pair becomes out[1], etc. Used to undup
// the various sscanf("%02x") loops scattered through the codebase.
inline bool hex_decode(const char* hex, size_t hex_len,
                       uint8_t* out, size_t out_len) {
    // Strip leading "0x" / "0X" if present.
    if (hex_len >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) {
        hex += 2;
        hex_len -= 2;
    }
    if ((hex_len & 1u) != 0u) return false;
    if (hex_len / 2u != out_len) return false;

    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < out_len; ++i) {
        int hi = nibble(hex[i * 2]);
        int lo = nibble(hex[i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return true;
}

// Compute range_bits = ceil(log2(end_be - start_be + 1)) for a 32-byte
// BE scalar range. The +1 reflects the JLP wire convention that
// `range_end` is INCLUSIVE -- the canonical Bitcoin puzzle N has
// [start=2^(N-1), end=2^N - 1], so size = end - start + 1 = 2^(N-1)
// and bit_length = N. Without the +1 the function returned N-1 for
// every puzzle, mis-sizing the kangaroo solver's K-factor budget.
// Returns 0 if end < start (inverted range) or end == start (a
// single-key range; the helper rejects that since a 1-key range
// can't be solved by Kangaroo). Used by CudaRCKangarooBackend::
// initialize() in place of the pre-1.4 hardcoded 135.
// KAT coverage: tests/test_range_bits.cpp pins puzzles 75/135/160.
inline int range_bits_from_be(const uint8_t start_be[32],
                              const uint8_t end_be[32]) {
    uint64_t s[4], e[4];
    be32_to_limbs_le(start_be, s);
    be32_to_limbs_le(end_be, e);

    // diff = e - s (256-bit, with borrow propagation). This is the
    // INCLUSIVE-end size minus 1; we add 1 below to get the actual
    // range size.
    uint64_t diff[4];
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        const uint64_t a = e[i];
        const uint64_t b = s[i];
        diff[i] = a - b - borrow;
        borrow  = (borrow == 0) ? ((a < b) ? 1ull : 0ull)
                                : ((a <= b) ? 1ull : 0ull);
    }

    if (borrow != 0) {
        return 0;  // end < start
    }

    // diff == 0 -> end == start, single-key range. Reject.
    bool diff_is_zero = (diff[0] | diff[1] | diff[2] | diff[3]) == 0;
    if (diff_is_zero) return 0;

    // Add 1 to convert inclusive-end "diff" into actual range size.
    // The carry can only propagate through if diff was 0xFF...FF in
    // the lower limbs.
    uint64_t size[4] = { diff[0], diff[1], diff[2], diff[3] };
    {
        uint64_t carry = 1;
        for (int i = 0; i < 4 && carry != 0; ++i) {
            const uint64_t old = size[i];
            size[i] = old + carry;
            carry = (size[i] < old) ? 1ull : 0ull;
        }
    }

    // bit_length(size): return position-of-MSB + 1.
    for (int i = 3; i >= 0; --i) {
        if (size[i] != 0) {
            uint64_t v = size[i];
            int msb = 0;
            while (v >>= 1) ++msb;
            return i * 64 + msb + 1;
        }
    }
    // Unreachable: we returned 0 above when diff == 0, and size = diff + 1
    // can only be all-zero if size overflowed past 2^256, which requires
    // the input to be (0, max_u256), impossible after the diff_is_zero
    // check.
    return 0;
}

// Compute bit_length(v) for a 32-byte big-endian value v.
// Returns 0 if v is zero, otherwise the index of the MSB + 1
// (e.g., bit_length_be32(2^134) = 135, bit_length_be32(1) = 1).
//
// This is distinct from range_bits_from_be(), which computes
// bit_length(end - start + 1) (the chunk WIDTH). bit_length_be32()
// computes bit_length(start) itself, which equals the PUZZLE's range_bits
// for any chunk: for puzzle N, range_start = 2^(N-1), so
// bit_length_be32(range_start) = N regardless of chunk width.
inline int bit_length_be32(const uint8_t v_be[32]) {
    uint64_t limbs[4];
    be32_to_limbs_le(v_be, limbs);
    for (int i = 3; i >= 0; --i) {
        if (limbs[i] != 0) {
            uint64_t v = limbs[i];
            int msb = 0;
            while (v >>= 1) ++msb;
            return i * 64 + msb + 1;
        }
    }
    return 0;
}

}  // namespace collider
