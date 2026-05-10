/**
 * Address-derivation CPU reference (Phase 4, restructure plan v1.4.0).
 *
 * Five derivations, one per `AddressType` enum value:
 *
 *   P2PKH_UNCOMPRESSED -> h160(SHA256(0x04 || X || Y))
 *   P2PKH_COMPRESSED   -> h160(SHA256(02_or_03 || X))
 *   P2SH_P2WPKH        -> h160(0x00 0x14 || compressed_h160)
 *   P2WPKH_V0          -> compressed_h160 (witness program)
 *   P2TR_BIP86         -> tweaked x-only output key under BIP-86
 *
 * "h160" = RIPEMD-160(SHA-256(data)).
 *
 * This file is the AUTHORITATIVE CPU reference. The eventual GPU kernel
 * (`v2_addr_kernel_<TYPE>`) MUST byte-match these outputs on every test
 * vector. KAT tests live in tests/v2/test_address_derive_cpu.cpp; vectors
 * are sourced from BIP-49 / BIP-84 / BIP-86 specs.
 *
 * Implementation notes:
 *   * No external SHA / RIPEMD library dependency. The implementation is
 *     header-only: all hash primitives are inline functions defined here
 *     so the reference is portable to any host (Windows / Linux / macOS,
 *     CI agents without OpenSSL, etc.).
 *   * BIP-86 tweak uses the secp256k1 generator's x-only public key for
 *     `H_taggedhash`. We accept the tweaked key as a 32-byte x-only value;
 *     callers must lift it externally if a full point is needed.
 *   * EC operations (priv -> pub) are NOT in this file. The caller passes
 *     in (X, Y) of the public point in big-endian 32 bytes each. This
 *     matches how the GPU kernel will consume EC outputs.
 */

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace collider {
namespace gpu {
namespace v2 {

// ---------------------------------------------------------------------------
// SHA-256 (header-only, FIPS 180-4 reference)
// ---------------------------------------------------------------------------

namespace internal {

inline uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline void sha256_compress(uint32_t state[8], const uint8_t block[64]) {
    static const uint32_t K[64] = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
    };
    uint32_t W[64];
    for (int i = 0; i < 16; ++i) {
        W[i] =  (uint32_t)block[i * 4    ] << 24
             |  (uint32_t)block[i * 4 + 1] << 16
             |  (uint32_t)block[i * 4 + 2] <<  8
             |  (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(W[i - 15], 7) ^ rotr32(W[i - 15], 18) ^ (W[i - 15] >> 3);
        uint32_t s1 = rotr32(W[i - 2], 17) ^ rotr32(W[i - 2], 19) ^ (W[i - 2] >> 10);
        W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t T1 = h + S1 + ch + K[i] + W[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t T2 = S0 + mj;
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

inline void sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    uint32_t state[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
    };
    uint64_t bit_len = static_cast<uint64_t>(len) * 8;
    uint8_t block[64];
    size_t off = 0;
    while (off + 64 <= len) {
        sha256_compress(state, data + off);
        off += 64;
    }
    size_t rem = len - off;
    std::memcpy(block, data + off, rem);
    block[rem] = 0x80;
    if (rem + 1 + 8 > 64) {
        std::memset(block + rem + 1, 0, 64 - rem - 1);
        sha256_compress(state, block);
        std::memset(block, 0, 56);
    } else {
        std::memset(block + rem + 1, 0, 56 - rem - 1);
    }
    for (int i = 0; i < 8; ++i) {
        block[56 + i] = static_cast<uint8_t>(bit_len >> (56 - 8 * i));
    }
    sha256_compress(state, block);
    for (int i = 0; i < 8; ++i) {
        out[i * 4    ] = static_cast<uint8_t>(state[i] >> 24);
        out[i * 4 + 1] = static_cast<uint8_t>(state[i] >> 16);
        out[i * 4 + 2] = static_cast<uint8_t>(state[i] >>  8);
        out[i * 4 + 3] = static_cast<uint8_t>(state[i]      );
    }
}

// ---------------------------------------------------------------------------
// RIPEMD-160 (header-only)
// ---------------------------------------------------------------------------

inline uint32_t rol32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }
inline uint32_t r_f1(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
inline uint32_t r_f2(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | ((~x) & z); }
inline uint32_t r_f3(uint32_t x, uint32_t y, uint32_t z) { return (x | (~y)) ^ z; }
inline uint32_t r_f4(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & (~z)); }
inline uint32_t r_f5(uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | (~z)); }

inline void ripemd160_compress(uint32_t state[5], const uint8_t block[64]) {
    static const int rL[80] = {
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
         7, 4,13, 1,10, 6,15, 3,12, 0, 9, 5, 2,14,11, 8,
         3,10,14, 4, 9,15, 8, 1, 2, 7, 0, 6,13,11, 5,12,
         1, 9,11,10, 0, 8,12, 4,13, 3, 7,15,14, 5, 6, 2,
         4, 0, 5, 9, 7,12, 2,10,14, 1, 3, 8,11, 6,15,13
    };
    static const int rR[80] = {
         5,14, 7, 0, 9, 2,11, 4,13, 6,15, 8, 1,10, 3,12,
         6,11, 3, 7, 0,13, 5,10,14,15, 8,12, 4, 9, 1, 2,
        15, 5, 1, 3, 7,14, 6, 9,11, 8,12, 2,10, 0, 4,13,
         8, 6, 4, 1, 3,11,15, 0, 5,12, 2,13, 9, 7,10,14,
        12,15,10, 4, 1, 5, 8, 7, 6, 2,13,14, 0, 3, 9,11
    };
    static const int sL[80] = {
        11,14,15,12, 5, 8, 7, 9,11,13,14,15, 6, 7, 9, 8,
         7, 6, 8,13,11, 9, 7,15, 7,12,15, 9,11, 7,13,12,
        11,13, 6, 7,14, 9,13,15,14, 8,13, 6, 5,12, 7, 5,
        11,12,14,15,14,15, 9, 8, 9,14, 5, 6, 8, 6, 5,12,
         9,15, 5,11, 6, 8,13,12, 5,12,13,14,11, 8, 5, 6
    };
    static const int sR[80] = {
         8, 9, 9,11,13,15,15, 5, 7, 7, 8,11,14,14,12, 6,
         9,13,15, 7,12, 8, 9,11, 7, 7,12, 7, 6,15,13,11,
         9, 7,15,11, 8, 6, 6,14,12,13, 5,14,13,13, 7, 5,
        15, 5, 8,11,14,14, 6,14, 6, 9,12, 9,12, 5,15, 8,
         8, 5,12, 9,12, 5,14, 6, 8,13, 6, 5,15,13,11,11
    };
    static const uint32_t KL[5] = {0, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e};
    static const uint32_t KR[5] = {0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0};

    uint32_t X[16];
    for (int i = 0; i < 16; ++i) {
        X[i] =  (uint32_t)block[i * 4    ]
             | ((uint32_t)block[i * 4 + 1] << 8)
             | ((uint32_t)block[i * 4 + 2] << 16)
             | ((uint32_t)block[i * 4 + 3] << 24);
    }
    uint32_t A = state[0], B = state[1], C = state[2], D = state[3], E = state[4];
    uint32_t Ar = A, Br = B, Cr = C, Dr = D, Er = E;
    for (int j = 0; j < 80; ++j) {
        const int round = j / 16;
        uint32_t fL, fR;
        switch (round) {
            case 0: fL = r_f1(B,  C,  D);  fR = r_f5(Br, Cr, Dr); break;
            case 1: fL = r_f2(B,  C,  D);  fR = r_f4(Br, Cr, Dr); break;
            case 2: fL = r_f3(B,  C,  D);  fR = r_f3(Br, Cr, Dr); break;
            case 3: fL = r_f4(B,  C,  D);  fR = r_f2(Br, Cr, Dr); break;
            default:fL = r_f5(B,  C,  D);  fR = r_f1(Br, Cr, Dr); break;
        }
        uint32_t T = rol32(A + fL + X[rL[j]] + KL[round], sL[j]) + E;
        A = E; E = D; D = rol32(C, 10); C = B; B = T;
        T  = rol32(Ar + fR + X[rR[j]] + KR[round], sR[j]) + Er;
        Ar = Er; Er = Dr; Dr = rol32(Cr, 10); Cr = Br; Br = T;
    }
    uint32_t T = state[1] + C + Dr;
    state[1] = state[2] + D + Er;
    state[2] = state[3] + E + Ar;
    state[3] = state[4] + A + Br;
    state[4] = state[0] + B + Cr;
    state[0] = T;
}

inline void ripemd160(const uint8_t* data, size_t len, uint8_t out[20]) {
    uint32_t state[5] = {
        0x67452301U, 0xefcdab89U, 0x98badcfeU, 0x10325476U, 0xc3d2e1f0U
    };
    uint64_t bit_len = static_cast<uint64_t>(len) * 8;
    uint8_t block[64];
    size_t off = 0;
    while (off + 64 <= len) {
        ripemd160_compress(state, data + off);
        off += 64;
    }
    size_t rem = len - off;
    std::memcpy(block, data + off, rem);
    block[rem] = 0x80;
    if (rem + 1 + 8 > 64) {
        std::memset(block + rem + 1, 0, 64 - rem - 1);
        ripemd160_compress(state, block);
        std::memset(block, 0, 56);
    } else {
        std::memset(block + rem + 1, 0, 56 - rem - 1);
    }
    for (int i = 0; i < 8; ++i) {
        block[56 + i] = static_cast<uint8_t>(bit_len >> (i * 8));
    }
    ripemd160_compress(state, block);
    for (int i = 0; i < 5; ++i) {
        out[i * 4    ] = static_cast<uint8_t>(state[i]      );
        out[i * 4 + 1] = static_cast<uint8_t>(state[i] >>  8);
        out[i * 4 + 2] = static_cast<uint8_t>(state[i] >> 16);
        out[i * 4 + 3] = static_cast<uint8_t>(state[i] >> 24);
    }
}

inline void hash160(const uint8_t* data, size_t len, uint8_t out[20]) {
    uint8_t s[32];
    sha256(data, len, s);
    ripemd160(s, 32, out);
}

}  // namespace internal

// ---------------------------------------------------------------------------
// Public address-derivation reference (CPU)
//
// Inputs are the X/Y coordinates of the public point in BIG-ENDIAN 32-byte
// form. Outputs are 20-byte h160 (or, for P2TR_BIP86, the 32-byte tweaked
// x-only output key).
// ---------------------------------------------------------------------------

inline void cpu_derive_p2pkh_uncompressed(
    const uint8_t pub_x[32], const uint8_t pub_y[32], uint8_t h160_out[20])
{
    uint8_t buf[65];
    buf[0] = 0x04;
    std::memcpy(buf +  1, pub_x, 32);
    std::memcpy(buf + 33, pub_y, 32);
    internal::hash160(buf, 65, h160_out);
}

inline void cpu_compressed_pubkey(
    const uint8_t pub_x[32], const uint8_t pub_y[32], uint8_t out[33])
{
    out[0] = (pub_y[31] & 1) ? 0x03 : 0x02;
    std::memcpy(out + 1, pub_x, 32);
}

inline void cpu_derive_p2pkh_compressed(
    const uint8_t pub_x[32], const uint8_t pub_y[32], uint8_t h160_out[20])
{
    uint8_t comp[33];
    cpu_compressed_pubkey(pub_x, pub_y, comp);
    internal::hash160(comp, 33, h160_out);
}

inline void cpu_derive_p2wpkh_v0(
    const uint8_t pub_x[32], const uint8_t pub_y[32], uint8_t h160_out[20])
{
    // Witness program for P2WPKH is just the compressed-pubkey h160.
    cpu_derive_p2pkh_compressed(pub_x, pub_y, h160_out);
}

inline void cpu_derive_p2sh_p2wpkh(
    const uint8_t pub_x[32], const uint8_t pub_y[32], uint8_t h160_out[20])
{
    // redeem_script = 0x00 0x14 || h160(compressed_pubkey); address = h160(redeem_script)
    uint8_t inner_h160[20];
    cpu_derive_p2pkh_compressed(pub_x, pub_y, inner_h160);
    uint8_t redeem[22];
    redeem[0] = 0x00;
    redeem[1] = 0x14;
    std::memcpy(redeem + 2, inner_h160, 20);
    internal::hash160(redeem, 22, h160_out);
}

// ---------------------------------------------------------------------------
// BIP-86 tweak for P2TR.
//
// Returns the x-only output key:
//   t = TaggedHash("TapTweak", x_only_pubkey)
//   Q = P + t * G   (the y-coordinate is implicit / parity-fixed)
//   output = Q.x
//
// This function ONLY computes `t` and accepts the caller's add of t*G.
// Pure scalar arithmetic on device requires the EC implementation; the host
// caller (during testing) can do the EC add via a host secp256k1 library.
//
// For Phase 4 we expose the tweak scalar so KAT tests can validate it
// against the BIP-86 spec vectors. Callers that have an EC primitive can
// then verify the full tweaked output key.
// ---------------------------------------------------------------------------

inline void cpu_bip86_tap_tweak(
    const uint8_t x_only_pub[32], uint8_t tweak_out[32])
{
    // tag_hash = SHA256("TapTweak")
    uint8_t tag_hash[32];
    static const char kTag[] = "TapTweak";
    internal::sha256(reinterpret_cast<const uint8_t*>(kTag),
                     sizeof(kTag) - 1, tag_hash);
    // payload = tag_hash || tag_hash || x_only_pub
    uint8_t buf[32 + 32 + 32];
    std::memcpy(buf,      tag_hash,    32);
    std::memcpy(buf + 32, tag_hash,    32);
    std::memcpy(buf + 64, x_only_pub,  32);
    internal::sha256(buf, sizeof(buf), tweak_out);
}

// Convenience: produce an "address fingerprint" 20 bytes for any address
// type given a pubkey. P2TR_BIP86 returns the x-only tweaked input bytes
// (NOT the actual output key, which needs EC; tests assert on the tweak
// value separately).
enum class AddrType : uint8_t {
    P2PKH_UNCOMPRESSED = 0,
    P2PKH_COMPRESSED   = 1,
    P2SH_P2WPKH        = 2,
    P2WPKH_V0          = 3,
    P2TR_BIP86_TWEAK   = 4,  // returns the 32-byte tweak in out[..32]
};

inline void cpu_derive_address_fingerprint(
    AddrType which,
    const uint8_t pub_x[32], const uint8_t pub_y[32],
    uint8_t out[32], size_t& out_len)
{
    switch (which) {
        case AddrType::P2PKH_UNCOMPRESSED:
            cpu_derive_p2pkh_uncompressed(pub_x, pub_y, out);
            out_len = 20; break;
        case AddrType::P2PKH_COMPRESSED:
            cpu_derive_p2pkh_compressed(pub_x, pub_y, out);
            out_len = 20; break;
        case AddrType::P2SH_P2WPKH:
            cpu_derive_p2sh_p2wpkh(pub_x, pub_y, out);
            out_len = 20; break;
        case AddrType::P2WPKH_V0:
            cpu_derive_p2wpkh_v0(pub_x, pub_y, out);
            out_len = 20; break;
        case AddrType::P2TR_BIP86_TWEAK:
            cpu_bip86_tap_tweak(pub_x, out);
            out_len = 32; break;
    }
}

}  // namespace v2
}  // namespace gpu
}  // namespace collider
