/**
 * Header-only SHA-512 + HMAC + PBKDF2 reference (Phase 7/8, v1.4.0).
 *
 * Used as the CPU reference for HMAC-SHA512 (DerivationScheme::HMAC_SHA512_PW),
 * SHA512_PW_HALF, BIP-39 PBKDF2, and the modular legacy KDF framework.
 *
 * Implementation follows FIPS 180-4 (SHA-512) and RFC 2104 (HMAC),
 * RFC 8018 (PBKDF2). Header-only; no external deps. Validated by
 * tests/v2/test_pbkdf2_cpu.cpp against RFC 4231 and BIP-39 vectors.
 */

#pragma once

#include "address_derive_cpu.hpp"   // pulls in internal::sha256

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace collider {
namespace gpu {
namespace v2 {
namespace internal {

// ---------------------------------------------------------------------------
// SHA-512
// ---------------------------------------------------------------------------

inline uint64_t rotr64(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

inline void sha512_compress(uint64_t state[8], const uint8_t block[128]) {
    static const uint64_t K[80] = {
        0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL,
        0xe9b5dba58189dbbcULL, 0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL,
        0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL, 0xd807aa98a3030242ULL,
        0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
        0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL,
        0xc19bf174cf692694ULL, 0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL,
        0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL, 0x2de92c6f592b0275ULL,
        0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
        0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL,
        0xbf597fc7beef0ee4ULL, 0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL,
        0x06ca6351e003826fULL, 0x142929670a0e6e70ULL, 0x27b70a8546d22ffcULL,
        0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
        0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL,
        0x92722c851482353bULL, 0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL,
        0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL, 0xd192e819d6ef5218ULL,
        0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
        0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL,
        0x34b0bcb5e19b48a8ULL, 0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL,
        0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL, 0x748f82ee5defb2fcULL,
        0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
        0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL,
        0xc67178f2e372532bULL, 0xca273eceea26619cULL, 0xd186b8c721c0c207ULL,
        0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL, 0x06f067aa72176fbaULL,
        0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
        0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL,
        0x431d67c49c100d4cULL, 0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL,
        0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL,
    };
    uint64_t W[80];
    for (int i = 0; i < 16; ++i) {
        W[i] = 0;
        for (int j = 0; j < 8; ++j) {
            W[i] |= static_cast<uint64_t>(block[i * 8 + j]) << (56 - 8 * j);
        }
    }
    for (int i = 16; i < 80; ++i) {
        uint64_t s0 = rotr64(W[i - 15], 1) ^ rotr64(W[i - 15], 8) ^ (W[i - 15] >> 7);
        uint64_t s1 = rotr64(W[i - 2], 19) ^ rotr64(W[i - 2], 61) ^ (W[i - 2] >> 6);
        W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }
    uint64_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint64_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 80; ++i) {
        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        uint64_t ch = (e & f) ^ ((~e) & g);
        uint64_t T1 = h + S1 + ch + K[i] + W[i];
        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        uint64_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t T2 = S0 + mj;
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

inline void sha512(const uint8_t* data, size_t len, uint8_t out[64]) {
    uint64_t state[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL, 0x3c6ef372fe94f82bULL,
        0xa54ff53a5f1d36f1ULL, 0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL,
    };
    uint64_t bit_len_lo = static_cast<uint64_t>(len) * 8;
    uint64_t bit_len_hi = 0;  // we don't support len > 2^61 bytes
    uint8_t block[128];
    size_t off = 0;
    while (off + 128 <= len) {
        sha512_compress(state, data + off);
        off += 128;
    }
    size_t rem = len - off;
    std::memcpy(block, data + off, rem);
    block[rem] = 0x80;
    if (rem + 1 + 16 > 128) {
        std::memset(block + rem + 1, 0, 128 - rem - 1);
        sha512_compress(state, block);
        std::memset(block, 0, 112);
    } else {
        std::memset(block + rem + 1, 0, 112 - rem - 1);
    }
    // 128-bit big-endian length: high 64 = 0; low 64 = bit_len_lo
    for (int i = 0; i < 8; ++i) block[112 + i] = static_cast<uint8_t>(bit_len_hi >> (56 - 8 * i));
    for (int i = 0; i < 8; ++i) block[120 + i] = static_cast<uint8_t>(bit_len_lo >> (56 - 8 * i));
    sha512_compress(state, block);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            out[i * 8 + j] = static_cast<uint8_t>(state[i] >> (56 - 8 * j));
        }
    }
}

// ---------------------------------------------------------------------------
// HMAC-SHA512
// ---------------------------------------------------------------------------

inline void hmac_sha512(const uint8_t* key, size_t key_len,
                        const uint8_t* msg, size_t msg_len,
                        uint8_t out[64]) {
    uint8_t k0[128];
    if (key_len > 128) {
        sha512(key, key_len, k0);
        std::memset(k0 + 64, 0, 64);
    } else {
        std::memcpy(k0, key, key_len);
        std::memset(k0 + key_len, 0, 128 - key_len);
    }
    uint8_t i_pad[128], o_pad[128];
    for (int i = 0; i < 128; ++i) {
        i_pad[i] = k0[i] ^ 0x36;
        o_pad[i] = k0[i] ^ 0x5c;
    }
    // inner = SHA512(i_pad || msg)
    std::vector<uint8_t> inner;
    inner.reserve(128 + msg_len);
    inner.insert(inner.end(), i_pad, i_pad + 128);
    inner.insert(inner.end(), msg, msg + msg_len);
    uint8_t inner_hash[64];
    sha512(inner.data(), inner.size(), inner_hash);
    // outer = SHA512(o_pad || inner_hash)
    uint8_t outer[128 + 64];
    std::memcpy(outer, o_pad, 128);
    std::memcpy(outer + 128, inner_hash, 64);
    sha512(outer, 128 + 64, out);
}

// ---------------------------------------------------------------------------
// HMAC-SHA256
// ---------------------------------------------------------------------------

inline void hmac_sha256(const uint8_t* key, size_t key_len,
                        const uint8_t* msg, size_t msg_len,
                        uint8_t out[32]) {
    uint8_t k0[64];
    if (key_len > 64) {
        sha256(key, key_len, k0);
        std::memset(k0 + 32, 0, 32);
    } else {
        std::memcpy(k0, key, key_len);
        std::memset(k0 + key_len, 0, 64 - key_len);
    }
    uint8_t i_pad[64], o_pad[64];
    for (int i = 0; i < 64; ++i) {
        i_pad[i] = k0[i] ^ 0x36;
        o_pad[i] = k0[i] ^ 0x5c;
    }
    std::vector<uint8_t> inner;
    inner.reserve(64 + msg_len);
    inner.insert(inner.end(), i_pad, i_pad + 64);
    inner.insert(inner.end(), msg, msg + msg_len);
    uint8_t inner_hash[32];
    sha256(inner.data(), inner.size(), inner_hash);
    uint8_t outer[64 + 32];
    std::memcpy(outer, o_pad, 64);
    std::memcpy(outer + 64, inner_hash, 32);
    sha256(outer, 64 + 32, out);
}

// ---------------------------------------------------------------------------
// PBKDF2-HMAC-SHA512 (RFC 8018)
// ---------------------------------------------------------------------------

inline void pbkdf2_hmac_sha512(
    const uint8_t* password, size_t password_len,
    const uint8_t* salt,     size_t salt_len,
    uint32_t iterations,
    uint8_t* out, size_t out_len)
{
    constexpr int H = 64;
    uint32_t blocks = static_cast<uint32_t>((out_len + H - 1) / H);
    std::vector<uint8_t> salt_extended(salt_len + 4);
    std::memcpy(salt_extended.data(), salt, salt_len);
    for (uint32_t i = 1; i <= blocks; ++i) {
        salt_extended[salt_len + 0] = static_cast<uint8_t>(i >> 24);
        salt_extended[salt_len + 1] = static_cast<uint8_t>(i >> 16);
        salt_extended[salt_len + 2] = static_cast<uint8_t>(i >> 8);
        salt_extended[salt_len + 3] = static_cast<uint8_t>(i);
        uint8_t U[H], T[H];
        hmac_sha512(password, password_len,
                    salt_extended.data(), salt_extended.size(), U);
        std::memcpy(T, U, H);
        for (uint32_t j = 1; j < iterations; ++j) {
            uint8_t U_next[H];
            hmac_sha512(password, password_len, U, H, U_next);
            std::memcpy(U, U_next, H);
            for (int k = 0; k < H; ++k) T[k] ^= U[k];
        }
        size_t copy_off = static_cast<size_t>(i - 1) * H;
        size_t copy_len = (out_len - copy_off < H) ? (out_len - copy_off) : H;
        std::memcpy(out + copy_off, T, copy_len);
    }
}

// PBKDF2-HMAC-SHA256 -- same skeleton, 32-byte block.
inline void pbkdf2_hmac_sha256(
    const uint8_t* password, size_t password_len,
    const uint8_t* salt,     size_t salt_len,
    uint32_t iterations,
    uint8_t* out, size_t out_len)
{
    constexpr int H = 32;
    uint32_t blocks = static_cast<uint32_t>((out_len + H - 1) / H);
    std::vector<uint8_t> salt_extended(salt_len + 4);
    std::memcpy(salt_extended.data(), salt, salt_len);
    for (uint32_t i = 1; i <= blocks; ++i) {
        salt_extended[salt_len + 0] = static_cast<uint8_t>(i >> 24);
        salt_extended[salt_len + 1] = static_cast<uint8_t>(i >> 16);
        salt_extended[salt_len + 2] = static_cast<uint8_t>(i >> 8);
        salt_extended[salt_len + 3] = static_cast<uint8_t>(i);
        uint8_t U[H], T[H];
        hmac_sha256(password, password_len,
                    salt_extended.data(), salt_extended.size(), U);
        std::memcpy(T, U, H);
        for (uint32_t j = 1; j < iterations; ++j) {
            uint8_t U_next[H];
            hmac_sha256(password, password_len, U, H, U_next);
            std::memcpy(U, U_next, H);
            for (int k = 0; k < H; ++k) T[k] ^= U[k];
        }
        size_t copy_off = static_cast<size_t>(i - 1) * H;
        size_t copy_len = (out_len - copy_off < H) ? (out_len - copy_off) : H;
        std::memcpy(out + copy_off, T, copy_len);
    }
}

}  // namespace internal
}  // namespace v2
}  // namespace gpu
}  // namespace collider
