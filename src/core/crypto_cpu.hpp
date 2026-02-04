/**
 * CPU Reference Crypto Implementation
 *
 * Portable SHA256, RIPEMD160, and secp256k1 for:
 * - Testing on non-GPU systems (macOS Metal backend not implemented)
 * - Verification of GPU results
 * - Small puzzle testing
 *
 * Note: This is NOT optimized for speed. Use GPU pipeline for production.
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <array>
#include <string>

namespace collider {
namespace cpu {

// ============================================================================
// SHA256 Implementation
// ============================================================================

class SHA256 {
public:
    static constexpr size_t HASH_SIZE = 32;
    using Hash = std::array<uint8_t, HASH_SIZE>;

    static Hash hash(const uint8_t* data, size_t len) {
        SHA256 ctx;
        ctx.update(data, len);
        return ctx.finalize();
    }

private:
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t buffer[64];
    uint32_t buflen;

    static constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    static uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
    static uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    static uint32_t sig0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    static uint32_t sig1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
    static uint32_t ep0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    static uint32_t ep1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

public:
    SHA256() {
        state[0] = 0x6a09e667; state[1] = 0xbb67ae85;
        state[2] = 0x3c6ef372; state[3] = 0xa54ff53a;
        state[4] = 0x510e527f; state[5] = 0x9b05688c;
        state[6] = 0x1f83d9ab; state[7] = 0x5be0cd19;
        bitlen = 0;
        buflen = 0;
    }

    void update(const uint8_t* data, size_t len) {
        for (size_t i = 0; i < len; i++) {
            buffer[buflen++] = data[i];
            if (buflen == 64) {
                transform();
                bitlen += 512;
                buflen = 0;
            }
        }
    }

    Hash finalize() {
        uint32_t i = buflen;

        // Pad
        buffer[i++] = 0x80;
        if (buflen < 56) {
            while (i < 56) buffer[i++] = 0;
        } else {
            while (i < 64) buffer[i++] = 0;
            transform();
            memset(buffer, 0, 56);
        }

        // Append length
        bitlen += buflen * 8;
        buffer[63] = bitlen;
        buffer[62] = bitlen >> 8;
        buffer[61] = bitlen >> 16;
        buffer[60] = bitlen >> 24;
        buffer[59] = bitlen >> 32;
        buffer[58] = bitlen >> 40;
        buffer[57] = bitlen >> 48;
        buffer[56] = bitlen >> 56;
        transform();

        // Output (big-endian)
        Hash hash;
        for (int j = 0; j < 8; j++) {
            hash[j * 4 + 0] = (state[j] >> 24) & 0xff;
            hash[j * 4 + 1] = (state[j] >> 16) & 0xff;
            hash[j * 4 + 2] = (state[j] >> 8) & 0xff;
            hash[j * 4 + 3] = state[j] & 0xff;
        }
        return hash;
    }

private:
    void transform() {
        uint32_t w[64];

        // Load words (big-endian)
        for (int i = 0; i < 16; i++) {
            w[i] = (buffer[i * 4] << 24) | (buffer[i * 4 + 1] << 16) |
                   (buffer[i * 4 + 2] << 8) | buffer[i * 4 + 3];
        }

        // Extend
        for (int i = 16; i < 64; i++) {
            w[i] = ep1(w[i-2]) + w[i-7] + ep0(w[i-15]) + w[i-16];
        }

        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h + sig1(e) + ch(e, f, g) + K[i] + w[i];
            uint32_t t2 = sig0(a) + maj(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }
};

constexpr uint32_t SHA256::K[64];

// ============================================================================
// RIPEMD160 Implementation
// ============================================================================

class RIPEMD160 {
public:
    static constexpr size_t HASH_SIZE = 20;
    using Hash = std::array<uint8_t, HASH_SIZE>;

    static Hash hash(const uint8_t* data, size_t len) {
        RIPEMD160 ctx;
        ctx.update(data, len);
        return ctx.finalize();
    }

private:
    uint32_t state[5];
    uint64_t count;
    uint8_t buffer[64];

    static uint32_t rotl(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
    static uint32_t f0(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
    static uint32_t f1(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
    static uint32_t f2(uint32_t x, uint32_t y, uint32_t z) { return (x | ~y) ^ z; }
    static uint32_t f3(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
    static uint32_t f4(uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | ~z); }

public:
    RIPEMD160() {
        state[0] = 0x67452301; state[1] = 0xEFCDAB89;
        state[2] = 0x98BADCFE; state[3] = 0x10325476;
        state[4] = 0xC3D2E1F0;
        count = 0;
    }

    void update(const uint8_t* data, size_t len) {
        size_t idx = (count / 8) % 64;
        count += len * 8;

        size_t partLen = 64 - idx;
        size_t i = 0;

        if (len >= partLen) {
            memcpy(buffer + idx, data, partLen);
            transform(buffer);

            for (i = partLen; i + 63 < len; i += 64) {
                transform(data + i);
            }
            idx = 0;
        }

        memcpy(buffer + idx, data + i, len - i);
    }

    Hash finalize() {
        uint8_t padding[64] = {0x80};
        uint8_t bits[8];

        for (int i = 0; i < 8; i++) {
            bits[i] = (count >> (i * 8)) & 0xff;
        }

        size_t idx = (count / 8) % 64;
        size_t padLen = (idx < 56) ? (56 - idx) : (120 - idx);

        update(padding, padLen);
        update(bits, 8);

        Hash hash;
        for (int i = 0; i < 5; i++) {
            hash[i * 4 + 0] = state[i] & 0xff;
            hash[i * 4 + 1] = (state[i] >> 8) & 0xff;
            hash[i * 4 + 2] = (state[i] >> 16) & 0xff;
            hash[i * 4 + 3] = (state[i] >> 24) & 0xff;
        }
        return hash;
    }

private:
    void transform(const uint8_t* block) {
        static const int rl[80] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
            3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
            1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
            4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
        };
        static const int rr[80] = {
            5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
            6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
            15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
            8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
            12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
        };
        static const int sl[80] = {
            11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
            7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
            11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
            11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
            9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
        };
        static const int sr[80] = {
            8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
            9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
            9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
            15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
            8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
        };
        static const uint32_t kl[5] = {0, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
        static const uint32_t kr[5] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0};

        uint32_t x[16];
        for (int i = 0; i < 16; i++) {
            x[i] = block[i * 4] | (block[i * 4 + 1] << 8) |
                   (block[i * 4 + 2] << 16) | (block[i * 4 + 3] << 24);
        }

        uint32_t al = state[0], bl = state[1], cl = state[2], dl = state[3], el = state[4];
        uint32_t ar = state[0], br = state[1], cr = state[2], dr = state[3], er = state[4];

        for (int j = 0; j < 80; j++) {
            int round = j / 16;
            uint32_t fl, fr, t;

            switch (round) {
                case 0: fl = f0(bl, cl, dl); fr = f4(br, cr, dr); break;
                case 1: fl = f1(bl, cl, dl); fr = f3(br, cr, dr); break;
                case 2: fl = f2(bl, cl, dl); fr = f2(br, cr, dr); break;
                case 3: fl = f3(bl, cl, dl); fr = f1(br, cr, dr); break;
                default: fl = f4(bl, cl, dl); fr = f0(br, cr, dr); break;
            }

            t = rotl(al + fl + x[rl[j]] + kl[round], sl[j]) + el;
            al = el; el = dl; dl = rotl(cl, 10); cl = bl; bl = t;

            t = rotl(ar + fr + x[rr[j]] + kr[round], sr[j]) + er;
            ar = er; er = dr; dr = rotl(cr, 10); cr = br; br = t;
        }

        uint32_t t = state[1] + cl + dr;
        state[1] = state[2] + dl + er;
        state[2] = state[3] + el + ar;
        state[3] = state[4] + al + br;
        state[4] = state[0] + bl + cr;
        state[0] = t;
    }
};

// ============================================================================
// secp256k1 Mini Implementation (Simplified for small key testing)
// ============================================================================

/**
 * Simple 256-bit integer for modular arithmetic
 */
struct uint256_t {
    uint64_t d[4];  // Little-endian

    uint256_t() : d{0, 0, 0, 0} {}

    explicit uint256_t(uint64_t lo) : d{lo, 0, 0, 0} {}

    uint256_t(uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3)
        : d{d0, d1, d2, d3} {}

    bool is_zero() const {
        return d[0] == 0 && d[1] == 0 && d[2] == 0 && d[3] == 0;
    }

    bool is_odd() const { return d[0] & 1; }

    // Comparison
    bool operator<(const uint256_t& o) const {
        for (int i = 3; i >= 0; i--) {
            if (d[i] < o.d[i]) return true;
            if (d[i] > o.d[i]) return false;
        }
        return false;
    }

    bool operator>=(const uint256_t& o) const { return !(*this < o); }

    bool operator==(const uint256_t& o) const {
        return d[0] == o.d[0] && d[1] == o.d[1] && d[2] == o.d[2] && d[3] == o.d[3];
    }
};

// secp256k1 field prime p = 2^256 - 2^32 - 977
static const uint256_t SECP256K1_P(
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
);

// secp256k1 generator point G
static const uint256_t SECP256K1_GX(
    0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
);

static const uint256_t SECP256K1_GY(
    0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
    0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL
);

// secp256k1 curve order n
static const uint256_t SECP256K1_N(
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
);

// ============================================================================
// Cross-platform 128-bit arithmetic helpers
// ============================================================================

#ifdef _MSC_VER
#include <intrin.h>

// MSVC: Use intrinsics for 64-bit add with carry
inline uint64_t add256(uint256_t& r, const uint256_t& a, const uint256_t& b) {
    unsigned char carry = 0;
    carry = _addcarry_u64(carry, a.d[0], b.d[0], &r.d[0]);
    carry = _addcarry_u64(carry, a.d[1], b.d[1], &r.d[1]);
    carry = _addcarry_u64(carry, a.d[2], b.d[2], &r.d[2]);
    carry = _addcarry_u64(carry, a.d[3], b.d[3], &r.d[3]);
    return carry;
}

// MSVC: Use intrinsics for 64-bit subtract with borrow
inline uint64_t sub256(uint256_t& r, const uint256_t& a, const uint256_t& b) {
    unsigned char borrow = 0;
    borrow = _subborrow_u64(borrow, a.d[0], b.d[0], &r.d[0]);
    borrow = _subborrow_u64(borrow, a.d[1], b.d[1], &r.d[1]);
    borrow = _subborrow_u64(borrow, a.d[2], b.d[2], &r.d[2]);
    borrow = _subborrow_u64(borrow, a.d[3], b.d[3], &r.d[3]);
    return borrow;
}

// Helper: 64x64 -> 128 bit multiply for MSVC
inline void mul64_128(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    lo = _umul128(a, b, &hi);
}

#else
// GCC/Clang: Use native 128-bit types

// Add two 256-bit numbers, return carry
inline uint64_t add256(uint256_t& r, const uint256_t& a, const uint256_t& b) {
    __uint128_t carry = 0;
    for (int i = 0; i < 4; i++) {
        __uint128_t sum = (__uint128_t)a.d[i] + b.d[i] + carry;
        r.d[i] = (uint64_t)sum;
        carry = sum >> 64;
    }
    return (uint64_t)carry;
}

// Subtract: r = a - b, return borrow
inline uint64_t sub256(uint256_t& r, const uint256_t& a, const uint256_t& b) {
    __int128_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        __int128_t diff = (__int128_t)a.d[i] - b.d[i] - borrow;
        r.d[i] = (uint64_t)diff;
        borrow = (diff < 0) ? 1 : 0;
    }
    return (uint64_t)borrow;
}

// Helper: 64x64 -> 128 bit multiply for GCC/Clang
inline void mul64_128(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
    __uint128_t prod = (__uint128_t)a * b;
    lo = (uint64_t)prod;
    hi = (uint64_t)(prod >> 64);
}

#endif

// Modular reduction mod p
inline void mod_p(uint256_t& a) {
    while (a >= SECP256K1_P) {
        sub256(a, a, SECP256K1_P);
    }
}

// Modular addition: r = (a + b) mod p
inline void mod_add(uint256_t& r, const uint256_t& a, const uint256_t& b) {
    if (add256(r, a, b) || r >= SECP256K1_P) {
        sub256(r, r, SECP256K1_P);
    }
}

// Modular subtraction: r = (a - b) mod p
inline void mod_sub(uint256_t& r, const uint256_t& a, const uint256_t& b) {
    if (sub256(r, a, b)) {
        add256(r, r, SECP256K1_P);
    }
}

// Modular multiplication (slow but correct schoolbook method)
// Cross-platform: uses mul64_128 helper for 64x64->128 multiply
inline void mod_mul(uint256_t& r, const uint256_t& a, const uint256_t& b) {
    // 512-bit product stored as 8 x 64-bit limbs
    uint64_t prod[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    // 256x256 -> 512 bit multiplication (schoolbook)
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t mul_lo, mul_hi;
            mul64_128(a.d[i], b.d[j], mul_lo, mul_hi);

            uint64_t sum_lo = prod[i+j] + mul_lo;
            uint64_t carry1 = (sum_lo < prod[i+j]) ? 1 : 0;

            sum_lo += carry;
            carry1 += (sum_lo < carry) ? 1 : 0;

            prod[i+j] = sum_lo;
            carry = mul_hi + carry1;
        }
        // Propagate final carry
        for (int k = i + 4; k < 8 && carry; k++) {
            uint64_t old = prod[k];
            prod[k] += carry;
            carry = (prod[k] < old) ? 1 : 0;
        }
    }

    // Reduce mod p where p = 2^256 - 0x1000003D1
    // Strategy: repeatedly fold high bits until result < 2^256
    // Then do final mod p subtraction

    // c = 0x1000003D1 = 2^32 + 977
    const uint64_t C_LO = 0x1000003D1ULL;

    // We'll reduce in place: prod = low + high * C
    // Repeat until high part is zero

    for (int round = 0; round < 3; round++) {
        // Check if high is zero
        if (prod[4] == 0 && prod[5] == 0 && prod[6] == 0 && prod[7] == 0) break;

        // Compute high[4..7] * C and add to low[0..3]
        // high * C where C = 0x1000003D1 (33 bits)
        // Result can be up to 256 + 33 = 289 bits

        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            // prod[i] += prod[i+4] * C + carry
            uint64_t mul_lo, mul_hi;
            mul64_128(prod[i + 4], C_LO, mul_lo, mul_hi);

            // Add mul_lo to prod[i]
            uint64_t s1 = prod[i] + mul_lo;
            uint64_t c1 = (s1 < prod[i]) ? 1 : 0;

            // Add carry
            uint64_t s2 = s1 + carry;
            uint64_t c2 = (s2 < s1) ? 1 : 0;

            prod[i] = s2;

            // New carry = mul_hi + c1 + c2
            carry = mul_hi + c1 + c2;

            // Clear the high limb we just processed
            prod[i + 4] = 0;
        }

        // The carry from this round becomes the new high part
        prod[4] = carry;
    }

    // Copy result
    for (int i = 0; i < 4; i++) {
        r.d[i] = prod[i];
    }

    // Final reduction: while r >= p, subtract p
    while (r >= SECP256K1_P) {
        sub256(r, r, SECP256K1_P);
    }
}

// Modular inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
inline void mod_inv(uint256_t& r, const uint256_t& a) {
    // Use binary exponentiation with p-2
    // This is slow but correct for small-scale testing
    uint256_t base = a;
    uint256_t exp = SECP256K1_P;
    sub256(exp, exp, uint256_t(2));  // exp = p - 2

    r = uint256_t(1);

    for (int i = 0; i < 256; i++) {
        int limb = i / 64;
        int bit = i % 64;

        if ((exp.d[limb] >> bit) & 1) {
            mod_mul(r, r, base);
        }
        mod_mul(base, base, base);
    }
}

// Modular exponentiation: r = base^exp mod p
inline void mod_pow(uint256_t& r, const uint256_t& base_in, const uint256_t& exp) {
    uint256_t base = base_in;
    r = uint256_t(1);

    for (int i = 0; i < 256; i++) {
        int limb = i / 64;
        int bit = i % 64;

        if ((exp.d[limb] >> bit) & 1) {
            mod_mul(r, r, base);
        }
        mod_mul(base, base, base);
    }
}

// Parse hex string to uint256_t (big-endian hex to little-endian limbs)
inline bool hex_to_uint256(uint256_t& out, const std::string& hex) {
    out = uint256_t(0);
    size_t start = 0;
    if (hex.size() >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) {
        start = 2;
    }

    std::string hex_str = hex.substr(start);

    // Pad to 64 chars (256 bits)
    while (hex_str.size() < 64) {
        hex_str = "0" + hex_str;
    }
    if (hex_str.size() > 64) return false;

    // Parse 16 hex chars per limb, from most significant to least
    // hex_str[0..15] -> d[3], hex_str[16..31] -> d[2], etc.
    for (int limb = 3; limb >= 0; limb--) {
        uint64_t val = 0;
        for (int i = 0; i < 16; i++) {
            char c = hex_str[(3 - limb) * 16 + i];
            uint8_t nibble;
            if (c >= '0' && c <= '9') nibble = c - '0';
            else if (c >= 'a' && c <= 'f') nibble = 10 + c - 'a';
            else if (c >= 'A' && c <= 'F') nibble = 10 + c - 'A';
            else return false;
            val = (val << 4) | nibble;
        }
        out.d[limb] = val;
    }
    return true;
}

// Decompress a compressed public key (02/03 + 32 bytes X)
// Returns true on success, outputs X and Y coordinates
inline bool decompress_pubkey(uint256_t& out_x, uint256_t& out_y, const std::string& compressed_hex) {
    if (compressed_hex.size() != 66) return false;  // 02/03 + 64 hex chars

    uint8_t prefix = 0;
    char p0 = compressed_hex[0], p1 = compressed_hex[1];
    if (p0 == '0' && p1 == '2') prefix = 0x02;
    else if (p0 == '0' && p1 == '3') prefix = 0x03;
    else return false;

    // Parse X coordinate
    std::string x_hex = compressed_hex.substr(2);
    if (!hex_to_uint256(out_x, x_hex)) return false;

    // Compute y² = x³ + 7 (mod p)
    uint256_t x2, x3, y2;
    mod_mul(x2, out_x, out_x);       // x²
    mod_mul(x3, x2, out_x);          // x³
    mod_add(y2, x3, uint256_t(7));   // x³ + 7

    // Compute y = y²^((p+1)/4) mod p
    // For secp256k1, (p+1)/4 = 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c
    uint256_t exp_sqrt(
        0xFFFFFFFFBFFFFF0CULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x3FFFFFFFFFFFFFFFULL
    );
    mod_pow(out_y, y2, exp_sqrt);

    // Verify: y² == y2 (sanity check)
    uint256_t check;
    mod_mul(check, out_y, out_y);
    if (!(check == y2)) return false;

    // Adjust parity: prefix 02 = even Y, 03 = odd Y
    bool y_is_odd = out_y.is_odd();
    bool need_odd = (prefix == 0x03);

    if (y_is_odd != need_odd) {
        // y = p - y
        sub256(out_y, SECP256K1_P, out_y);
    }

    return true;
}

// Point in Jacobian coordinates
struct ECPoint {
    uint256_t X, Y, Z;

    bool is_infinity() const { return Z.is_zero(); }

    void set_infinity() {
        X = uint256_t(1);
        Y = uint256_t(1);
        Z = uint256_t(0);
    }
};

// Point doubling in Jacobian coordinates
inline void ec_double(ECPoint& R, const ECPoint& P) {
    if (P.is_infinity()) {
        R.set_infinity();
        return;
    }

    uint256_t S, M, T, Y2;

    // Y^2
    mod_mul(Y2, P.Y, P.Y);

    // S = 4 * X * Y^2
    mod_mul(S, P.X, Y2);
    mod_add(S, S, S);
    mod_add(S, S, S);

    // M = 3 * X^2 (a=0 for secp256k1)
    mod_mul(M, P.X, P.X);
    mod_add(T, M, M);
    mod_add(M, T, M);

    // X' = M^2 - 2*S
    mod_mul(R.X, M, M);
    mod_sub(R.X, R.X, S);
    mod_sub(R.X, R.X, S);

    // Y' = M * (S - X') - 8 * Y^4
    mod_sub(T, S, R.X);
    mod_mul(T, M, T);
    mod_mul(Y2, Y2, Y2);  // Y^4
    mod_add(Y2, Y2, Y2);  // 2*Y^4
    mod_add(Y2, Y2, Y2);  // 4*Y^4
    mod_add(Y2, Y2, Y2);  // 8*Y^4
    mod_sub(R.Y, T, Y2);

    // Z' = 2 * Y * Z
    mod_mul(R.Z, P.Y, P.Z);
    mod_add(R.Z, R.Z, R.Z);
}

// Point addition (mixed: P Jacobian, Q affine)
inline void ec_add(ECPoint& R, const ECPoint& P, const uint256_t& Qx, const uint256_t& Qy) {
    if (P.is_infinity()) {
        R.X = Qx;
        R.Y = Qy;
        R.Z = uint256_t(1);
        return;
    }

    uint256_t Z1Z1, U2, S2, H, HH, I, J, r, V;

    mod_mul(Z1Z1, P.Z, P.Z);
    mod_mul(U2, Qx, Z1Z1);
    mod_mul(S2, Qy, P.Z);
    mod_mul(S2, S2, Z1Z1);
    mod_sub(H, U2, P.X);
    mod_sub(r, S2, P.Y);
    mod_add(r, r, r);

    if (H.is_zero()) {
        if (r.is_zero()) {
            // P == Q, need to double
            ec_double(R, P);
            return;
        }
        // P == -Q, result is infinity
        R.set_infinity();
        return;
    }

    mod_mul(HH, H, H);
    mod_add(I, HH, HH);
    mod_add(I, I, I);
    mod_mul(J, H, I);
    mod_mul(V, P.X, I);

    // X3 = r^2 - J - 2*V
    mod_mul(R.X, r, r);
    mod_sub(R.X, R.X, J);
    mod_sub(R.X, R.X, V);
    mod_sub(R.X, R.X, V);

    // Y3 = r * (V - X3) - 2 * Y1 * J
    mod_sub(V, V, R.X);
    mod_mul(R.Y, r, V);
    mod_mul(J, P.Y, J);
    mod_add(J, J, J);
    mod_sub(R.Y, R.Y, J);

    // Z3 = 2 * Z1 * H
    mod_mul(R.Z, P.Z, H);
    mod_add(R.Z, R.Z, R.Z);
}

// Scalar multiplication: R = k * G
inline void ec_mul(ECPoint& R, const uint256_t& k) {
    R.set_infinity();

    // Simple double-and-add (slow but correct)
    for (int i = 255; i >= 0; i--) {
        ECPoint T;
        ec_double(T, R);
        R = T;

        int limb = i / 64;
        int bit = i % 64;

        if ((k.d[limb] >> bit) & 1) {
            ec_add(T, R, SECP256K1_GX, SECP256K1_GY);
            R = T;
        }
    }
}

// Convert Jacobian to affine coordinates
inline void ec_to_affine(uint256_t& x, uint256_t& y, const ECPoint& P) {
    if (P.is_infinity()) {
        x = uint256_t(0);
        y = uint256_t(0);
        return;
    }

    uint256_t z_inv, z_inv2, z_inv3;
    mod_inv(z_inv, P.Z);
    mod_mul(z_inv2, z_inv, z_inv);
    mod_mul(z_inv3, z_inv2, z_inv);
    mod_mul(x, P.X, z_inv2);
    mod_mul(y, P.Y, z_inv3);
}

/**
 * Compute Hash160 from a private key
 * Pipeline: privkey -> pubkey -> SHA256 -> RIPEMD160
 *
 * @param privkey_bytes 32-byte big-endian private key
 * @return 20-byte Hash160
 */
inline std::array<uint8_t, 20> compute_hash160(const uint8_t* privkey_bytes) {
    // Convert private key bytes to uint256_t (big-endian input)
    uint256_t k;
    k.d[3] = ((uint64_t)privkey_bytes[0] << 56) | ((uint64_t)privkey_bytes[1] << 48) |
             ((uint64_t)privkey_bytes[2] << 40) | ((uint64_t)privkey_bytes[3] << 32) |
             ((uint64_t)privkey_bytes[4] << 24) | ((uint64_t)privkey_bytes[5] << 16) |
             ((uint64_t)privkey_bytes[6] << 8) | privkey_bytes[7];
    k.d[2] = ((uint64_t)privkey_bytes[8] << 56) | ((uint64_t)privkey_bytes[9] << 48) |
             ((uint64_t)privkey_bytes[10] << 40) | ((uint64_t)privkey_bytes[11] << 32) |
             ((uint64_t)privkey_bytes[12] << 24) | ((uint64_t)privkey_bytes[13] << 16) |
             ((uint64_t)privkey_bytes[14] << 8) | privkey_bytes[15];
    k.d[1] = ((uint64_t)privkey_bytes[16] << 56) | ((uint64_t)privkey_bytes[17] << 48) |
             ((uint64_t)privkey_bytes[18] << 40) | ((uint64_t)privkey_bytes[19] << 32) |
             ((uint64_t)privkey_bytes[20] << 24) | ((uint64_t)privkey_bytes[21] << 16) |
             ((uint64_t)privkey_bytes[22] << 8) | privkey_bytes[23];
    k.d[0] = ((uint64_t)privkey_bytes[24] << 56) | ((uint64_t)privkey_bytes[25] << 48) |
             ((uint64_t)privkey_bytes[26] << 40) | ((uint64_t)privkey_bytes[27] << 32) |
             ((uint64_t)privkey_bytes[28] << 24) | ((uint64_t)privkey_bytes[29] << 16) |
             ((uint64_t)privkey_bytes[30] << 8) | privkey_bytes[31];

    // Compute public key: P = k * G
    ECPoint P;
    ec_mul(P, k);

    // Convert to affine
    uint256_t pub_x, pub_y;
    ec_to_affine(pub_x, pub_y, P);

    // Encode compressed public key (33 bytes: prefix || x)
    uint8_t pubkey[33];
    pubkey[0] = pub_y.is_odd() ? 0x03 : 0x02;

    // X coordinate big-endian
    pubkey[1] = (pub_x.d[3] >> 56) & 0xff;
    pubkey[2] = (pub_x.d[3] >> 48) & 0xff;
    pubkey[3] = (pub_x.d[3] >> 40) & 0xff;
    pubkey[4] = (pub_x.d[3] >> 32) & 0xff;
    pubkey[5] = (pub_x.d[3] >> 24) & 0xff;
    pubkey[6] = (pub_x.d[3] >> 16) & 0xff;
    pubkey[7] = (pub_x.d[3] >> 8) & 0xff;
    pubkey[8] = pub_x.d[3] & 0xff;
    pubkey[9] = (pub_x.d[2] >> 56) & 0xff;
    pubkey[10] = (pub_x.d[2] >> 48) & 0xff;
    pubkey[11] = (pub_x.d[2] >> 40) & 0xff;
    pubkey[12] = (pub_x.d[2] >> 32) & 0xff;
    pubkey[13] = (pub_x.d[2] >> 24) & 0xff;
    pubkey[14] = (pub_x.d[2] >> 16) & 0xff;
    pubkey[15] = (pub_x.d[2] >> 8) & 0xff;
    pubkey[16] = pub_x.d[2] & 0xff;
    pubkey[17] = (pub_x.d[1] >> 56) & 0xff;
    pubkey[18] = (pub_x.d[1] >> 48) & 0xff;
    pubkey[19] = (pub_x.d[1] >> 40) & 0xff;
    pubkey[20] = (pub_x.d[1] >> 32) & 0xff;
    pubkey[21] = (pub_x.d[1] >> 24) & 0xff;
    pubkey[22] = (pub_x.d[1] >> 16) & 0xff;
    pubkey[23] = (pub_x.d[1] >> 8) & 0xff;
    pubkey[24] = pub_x.d[1] & 0xff;
    pubkey[25] = (pub_x.d[0] >> 56) & 0xff;
    pubkey[26] = (pub_x.d[0] >> 48) & 0xff;
    pubkey[27] = (pub_x.d[0] >> 40) & 0xff;
    pubkey[28] = (pub_x.d[0] >> 32) & 0xff;
    pubkey[29] = (pub_x.d[0] >> 24) & 0xff;
    pubkey[30] = (pub_x.d[0] >> 16) & 0xff;
    pubkey[31] = (pub_x.d[0] >> 8) & 0xff;
    pubkey[32] = pub_x.d[0] & 0xff;

    // SHA256(pubkey)
    auto sha_hash = SHA256::hash(pubkey, 33);

    // RIPEMD160(SHA256(pubkey))
    auto ripe_hash = RIPEMD160::hash(sha_hash.data(), 32);

    return ripe_hash;
}

/**
 * Convert hex string to byte array
 */
inline std::array<uint8_t, 20> hex_to_hash160(const std::string& hex) {
    std::array<uint8_t, 20> result;
    for (int i = 0; i < 20; i++) {
        char hi = hex[i * 2];
        char lo = hex[i * 2 + 1];

        auto hex_val = [](char c) -> uint8_t {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            return 0;
        };

        result[i] = (hex_val(hi) << 4) | hex_val(lo);
    }
    return result;
}

/**
 * Convert 64-bit key value to 32-byte big-endian private key
 */
inline void key_to_bytes(uint8_t* out, uint64_t lo, uint64_t hi = 0) {
    memset(out, 0, 32);
    // Big-endian: high bytes first
    out[16] = (hi >> 56) & 0xff;
    out[17] = (hi >> 48) & 0xff;
    out[18] = (hi >> 40) & 0xff;
    out[19] = (hi >> 32) & 0xff;
    out[20] = (hi >> 24) & 0xff;
    out[21] = (hi >> 16) & 0xff;
    out[22] = (hi >> 8) & 0xff;
    out[23] = hi & 0xff;
    out[24] = (lo >> 56) & 0xff;
    out[25] = (lo >> 48) & 0xff;
    out[26] = (lo >> 40) & 0xff;
    out[27] = (lo >> 32) & 0xff;
    out[28] = (lo >> 24) & 0xff;
    out[29] = (lo >> 16) & 0xff;
    out[30] = (lo >> 8) & 0xff;
    out[31] = lo & 0xff;
}

}  // namespace cpu
}  // namespace collider
