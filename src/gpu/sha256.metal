/*
 * SHA-256 compute kernel for Apple Metal.
 *
 * Free-licensed. Provides a batch SHA-256 throughput kernel that
 * mirrors the CUDA `sha256_batch` entry point in src/gpu/sha256.cu.
 * Used by the v1.4.0 Free benchmark to measure GPU SHA-256 H/s on
 * Apple Silicon.
 *
 * One thread per input. Each input is 64 bytes (a single SHA-256
 * block; no padding handling, the kernel is a pure throughput test
 * not a general-purpose hasher).
 */

#include <metal_stdlib>
using namespace metal;

constant uint K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

inline uint rotr32(uint x, int n) { return (x >> n) | (x << (32 - n)); }

kernel void sha256_bench(
    device const uchar*  inputs   [[buffer(0)]],   // count * 64 bytes
    device       uchar*  outputs  [[buffer(1)]],   // count * 32 bytes
    constant     uint&   count    [[buffer(2)]],
    uint                 gid      [[thread_position_in_grid]])
{
    if (gid >= count) return;

    // Load 64-byte input as 16 big-endian uint32 words.
    uint W[64];
    device const uchar* src = inputs + gid * 64;
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint)src[i*4    ] << 24)
             | ((uint)src[i*4 + 1] << 16)
             | ((uint)src[i*4 + 2] <<  8)
             | ((uint)src[i*4 + 3]      );
    }
    for (int i = 16; i < 64; ++i) {
        uint s0 = rotr32(W[i-15], 7) ^ rotr32(W[i-15], 18) ^ (W[i-15] >> 3);
        uint s1 = rotr32(W[i-2], 17) ^ rotr32(W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }

    uint a = 0x6a09e667u, b = 0xbb67ae85u, c = 0x3c6ef372u, d = 0xa54ff53au;
    uint e = 0x510e527fu, f = 0x9b05688cu, g = 0x1f83d9abu, h = 0x5be0cd19u;

    for (int i = 0; i < 64; ++i) {
        uint S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint ch = (e & f) ^ ((~e) & g);
        uint T1 = h + S1 + ch + K256[i] + W[i];
        uint S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint mj = (a & b) ^ (a & c) ^ (b & c);
        uint T2 = S0 + mj;
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }
    a += 0x6a09e667u; b += 0xbb67ae85u; c += 0x3c6ef372u; d += 0xa54ff53au;
    e += 0x510e527fu; f += 0x9b05688cu; g += 0x1f83d9abu; h += 0x5be0cd19u;

    device uchar* out = outputs + gid * 32;
    for (int i = 0; i < 4; ++i) {
        uint v = (i == 0 ? a : i == 1 ? b : i == 2 ? c : d);
        out[i*4    ] = (uchar)(v >> 24);
        out[i*4 + 1] = (uchar)(v >> 16);
        out[i*4 + 2] = (uchar)(v >>  8);
        out[i*4 + 3] = (uchar)(v      );
    }
    for (int i = 0; i < 4; ++i) {
        uint v = (i == 0 ? e : i == 1 ? f : i == 2 ? g : h);
        out[16 + i*4    ] = (uchar)(v >> 24);
        out[16 + i*4 + 1] = (uchar)(v >> 16);
        out[16 + i*4 + 2] = (uchar)(v >>  8);
        out[16 + i*4 + 3] = (uchar)(v      );
    }
}
