/**
 * sha512_device.cuh -- SHA-512 transform on CUDA device.
 *
 * Implements the FIPS 180-4 SHA-512 transform as a __device__ inline
 * function. Per-block, in-place, single-thread (one block at a time
 * per device thread; PBKDF2 inner loop runs MANY blocks per thread
 * across the 2048 iterations).
 *
 * Memory: this header alone uses ~640 bytes of register/local memory
 * per thread (8 H[] words = 64 B + 16-word message schedule rolled
 * into the round loop). The 80-round constant table is in __constant__
 * memory, 640 B total, broadcast-friendly.
 *
 * Not __host__-callable; the CPU side uses OpenSSL's PKCS5_PBKDF2_HMAC.
 * The KAT in tests/test_sha512_gpu_kat.cu cross-checks the device
 * transform against std::vector<uint8_t> known-answer vectors via a
 * one-block kernel.
 *
 * Reference: NIST FIPS 180-4 (2015 March), section 4.2.3 + 5.3.5.
 */

#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(COLLIDER_USE_CUDA)

namespace collider::gpu::sha512 {

// Per-round constants K[0..79]. FIPS 180-4 section 4.2.3.
__device__ __constant__ static const uint64_t kK[80] = {
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

// Initial hash values H[0..7]. FIPS 180-4 section 5.3.5.
__device__ __forceinline__ void init_h(uint64_t h[8]) {
    h[0] = 0x6a09e667f3bcc908ULL;
    h[1] = 0xbb67ae8584caa73bULL;
    h[2] = 0x3c6ef372fe94f82bULL;
    h[3] = 0xa54ff53a5f1d36f1ULL;
    h[4] = 0x510e527fade682d1ULL;
    h[5] = 0x9b05688c2b3e6c1fULL;
    h[6] = 0x1f83d9abfb41bd6bULL;
    h[7] = 0x5be0cd19137e2179ULL;
}

__device__ __forceinline__ uint64_t rotr(uint64_t x, int n) {
    return (x >> n) | (x << (64 - n));
}

__device__ __forceinline__ uint64_t Ch(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (~x & z);
}
__device__ __forceinline__ uint64_t Maj(uint64_t x, uint64_t y, uint64_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}
__device__ __forceinline__ uint64_t Sig0(uint64_t x) {
    return rotr(x, 28) ^ rotr(x, 34) ^ rotr(x, 39);
}
__device__ __forceinline__ uint64_t Sig1(uint64_t x) {
    return rotr(x, 14) ^ rotr(x, 18) ^ rotr(x, 41);
}
__device__ __forceinline__ uint64_t sig0(uint64_t x) {
    return rotr(x, 1) ^ rotr(x, 8) ^ (x >> 7);
}
__device__ __forceinline__ uint64_t sig1(uint64_t x) {
    return rotr(x, 19) ^ rotr(x, 61) ^ (x >> 6);
}

// Big-endian 64-bit load from byte pointer. SHA-512 input blocks are
// 1024 bits = 128 bytes = 16 64-bit big-endian words.
__device__ __forceinline__ uint64_t be64_load(const uint8_t* p) {
    return (uint64_t(p[0]) << 56) | (uint64_t(p[1]) << 48) |
           (uint64_t(p[2]) << 40) | (uint64_t(p[3]) << 32) |
           (uint64_t(p[4]) << 24) | (uint64_t(p[5]) << 16) |
           (uint64_t(p[6]) <<  8) |  uint64_t(p[7]);
}

// Process ONE 128-byte block into h[0..7]. Caller is responsible for
// padding / length encoding (final block per FIPS 180-4 sec 5.1.2).
__device__ __forceinline__ void transform_block(uint64_t h[8],
                                                const uint8_t* block) {
    uint64_t w[80];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        w[i] = be64_load(block + i * 8);
    }
    #pragma unroll
    for (int i = 16; i < 80; ++i) {
        w[i] = sig1(w[i - 2]) + w[i - 7] + sig0(w[i - 15]) + w[i - 16];
    }

    uint64_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint64_t e = h[4], f = h[5], g = h[6], hh = h[7];

    #pragma unroll
    for (int i = 0; i < 80; ++i) {
        uint64_t T1 = hh + Sig1(e) + Ch(e, f, g) + kK[i] + w[i];
        uint64_t T2 = Sig0(a) + Maj(a, b, c);
        hh = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

// Compute SHA-512 over a contiguous message of arbitrary length.
// Handles N full 128-byte blocks + one final padded block (or two
// final blocks when the message length lands in 112..127 mod 128).
// Required for HMAC-SHA512 key-shrinking when the key is longer
// than 128 bytes (24-word BIP-39 mnemonics are ~167 bytes UTF-8).
//
// Output: 64-byte big-endian digest.
__device__ __forceinline__ void hash_short(const uint8_t* msg,
                                           uint32_t len,
                                           uint8_t out[64]) {
    uint64_t h[8];
    init_h(h);

    // Process all full 128-byte blocks first.
    uint32_t off = 0;
    while (off + 128 <= len) {
        transform_block(h, msg + off);
        off += 128;
    }

    // Final block(s): tail bytes + 0x80 + zero-pad + 128-bit length.
    // If the remaining tail + 17 bytes (0x80 + 16-byte length) fits
    // in one block (tail_len <= 111), use one final block; else two.
    uint8_t block[128] = {0};
    const uint32_t tail_len = len - off;
    for (uint32_t i = 0; i < tail_len; ++i) {
        block[i] = msg[off + i];
    }
    block[tail_len] = 0x80;

    const uint64_t bit_len = uint64_t(len) * 8ULL;
    if (tail_len <= 111) {
        // Single final block.
        block[120] = uint8_t((bit_len >> 56) & 0xFF);
        block[121] = uint8_t((bit_len >> 48) & 0xFF);
        block[122] = uint8_t((bit_len >> 40) & 0xFF);
        block[123] = uint8_t((bit_len >> 32) & 0xFF);
        block[124] = uint8_t((bit_len >> 24) & 0xFF);
        block[125] = uint8_t((bit_len >> 16) & 0xFF);
        block[126] = uint8_t((bit_len >>  8) & 0xFF);
        block[127] = uint8_t( bit_len        & 0xFF);
        transform_block(h, block);
    } else {
        // Two final blocks: tail+0x80+zeros in block 1, length in block 2.
        transform_block(h, block);
        uint8_t block2[128] = {0};
        block2[120] = uint8_t((bit_len >> 56) & 0xFF);
        block2[121] = uint8_t((bit_len >> 48) & 0xFF);
        block2[122] = uint8_t((bit_len >> 40) & 0xFF);
        block2[123] = uint8_t((bit_len >> 32) & 0xFF);
        block2[124] = uint8_t((bit_len >> 24) & 0xFF);
        block2[125] = uint8_t((bit_len >> 16) & 0xFF);
        block2[126] = uint8_t((bit_len >>  8) & 0xFF);
        block2[127] = uint8_t( bit_len        & 0xFF);
        transform_block(h, block2);
    }

    // Emit big-endian.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint64_t v = h[i];
        out[i * 8 + 0] = uint8_t((v >> 56) & 0xFF);
        out[i * 8 + 1] = uint8_t((v >> 48) & 0xFF);
        out[i * 8 + 2] = uint8_t((v >> 40) & 0xFF);
        out[i * 8 + 3] = uint8_t((v >> 32) & 0xFF);
        out[i * 8 + 4] = uint8_t((v >> 24) & 0xFF);
        out[i * 8 + 5] = uint8_t((v >> 16) & 0xFF);
        out[i * 8 + 6] = uint8_t((v >>  8) & 0xFF);
        out[i * 8 + 7] = uint8_t( v        & 0xFF);
    }
}

}  // namespace collider::gpu::sha512

#endif  // __CUDACC__ || COLLIDER_USE_CUDA
