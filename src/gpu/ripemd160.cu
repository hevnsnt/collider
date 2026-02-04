/**
 * Collider GPU RIPEMD160 Implementation
 *
 * Used for Bitcoin address generation: RIPEMD160(SHA256(pubkey))
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace collider {
namespace gpu {

// RIPEMD160 constants
static __constant__ uint32_t KL[5] = {
    0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E
};

static __constant__ uint32_t KR[5] = {
    0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000
};

// Message schedule for left rounds
static __constant__ int RL[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};

// Message schedule for right rounds
static __constant__ int RR[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};

// Rotation amounts for left rounds
static __constant__ int SL[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};

// Rotation amounts for right rounds
static __constant__ int SR[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

// Initial hash values
static __constant__ uint32_t RIPEMD160_H0[5] = {
    0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0
};

// Bit rotation
__device__ __forceinline__ uint32_t rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// Boolean functions
__device__ __forceinline__ uint32_t f0(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ y ^ z;
}

__device__ __forceinline__ uint32_t f1(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) | (~x & z);
}

__device__ __forceinline__ uint32_t f2(uint32_t x, uint32_t y, uint32_t z) {
    return (x | ~y) ^ z;
}

__device__ __forceinline__ uint32_t f3(uint32_t x, uint32_t y, uint32_t z) {
    return (x & z) | (y & ~z);
}

__device__ __forceinline__ uint32_t f4(uint32_t x, uint32_t y, uint32_t z) {
    return x ^ (y | ~z);
}

/**
 * RIPEMD160 hash of a 32-byte input (SHA256 output).
 */
__device__ void ripemd160_hash(
    const uint8_t* message,  // 32 bytes (SHA256 output)
    uint8_t* hash            // 20 bytes output
) {
    uint32_t H[5];

    // Initialize hash values
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        H[i] = RIPEMD160_H0[i];
    }

    // Prepare message block (32 bytes + padding)
    uint8_t block[64] = {0};

    // Copy message
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        block[i] = message[i];
    }

    // Padding
    block[32] = 0x80;

    // Length in bits (256 = 0x100)
    block[56] = 0x00;
    block[57] = 0x01;

    // Parse block into 16 32-bit words (little-endian)
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        X[i] = block[i*4] |
               (block[i*4 + 1] << 8) |
               (block[i*4 + 2] << 16) |
               (block[i*4 + 3] << 24);
    }

    // Initialize working variables
    uint32_t AL = H[0], BL = H[1], CL = H[2], DL = H[3], EL = H[4];
    uint32_t AR = H[0], BR = H[1], CR = H[2], DR = H[3], ER = H[4];

    // OPTIMIZED: Explicit round unrolling to avoid switch statement divergence
    // Each round uses a different boolean function - unroll for uniform execution

    // Round 0 (j=0-15): fL=f0, fR=f4
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        uint32_t fL = f0(BL, CL, DL);
        uint32_t fR = f4(BR, CR, DR);
        uint32_t tL = rotl(AL + fL + X[RL[j]] + KL[0], SL[j]) + EL;
        AL = EL; EL = DL; DL = rotl(CL, 10); CL = BL; BL = tL;
        uint32_t tR = rotl(AR + fR + X[RR[j]] + KR[0], SR[j]) + ER;
        AR = ER; ER = DR; DR = rotl(CR, 10); CR = BR; BR = tR;
    }

    // Round 1 (j=16-31): fL=f1, fR=f3
    #pragma unroll
    for (int j = 16; j < 32; j++) {
        uint32_t fL = f1(BL, CL, DL);
        uint32_t fR = f3(BR, CR, DR);
        uint32_t tL = rotl(AL + fL + X[RL[j]] + KL[1], SL[j]) + EL;
        AL = EL; EL = DL; DL = rotl(CL, 10); CL = BL; BL = tL;
        uint32_t tR = rotl(AR + fR + X[RR[j]] + KR[1], SR[j]) + ER;
        AR = ER; ER = DR; DR = rotl(CR, 10); CR = BR; BR = tR;
    }

    // Round 2 (j=32-47): fL=f2, fR=f2
    #pragma unroll
    for (int j = 32; j < 48; j++) {
        uint32_t fL = f2(BL, CL, DL);
        uint32_t fR = f2(BR, CR, DR);
        uint32_t tL = rotl(AL + fL + X[RL[j]] + KL[2], SL[j]) + EL;
        AL = EL; EL = DL; DL = rotl(CL, 10); CL = BL; BL = tL;
        uint32_t tR = rotl(AR + fR + X[RR[j]] + KR[2], SR[j]) + ER;
        AR = ER; ER = DR; DR = rotl(CR, 10); CR = BR; BR = tR;
    }

    // Round 3 (j=48-63): fL=f3, fR=f1
    #pragma unroll
    for (int j = 48; j < 64; j++) {
        uint32_t fL = f3(BL, CL, DL);
        uint32_t fR = f1(BR, CR, DR);
        uint32_t tL = rotl(AL + fL + X[RL[j]] + KL[3], SL[j]) + EL;
        AL = EL; EL = DL; DL = rotl(CL, 10); CL = BL; BL = tL;
        uint32_t tR = rotl(AR + fR + X[RR[j]] + KR[3], SR[j]) + ER;
        AR = ER; ER = DR; DR = rotl(CR, 10); CR = BR; BR = tR;
    }

    // Round 4 (j=64-79): fL=f4, fR=f0
    #pragma unroll
    for (int j = 64; j < 80; j++) {
        uint32_t fL = f4(BL, CL, DL);
        uint32_t fR = f0(BR, CR, DR);
        uint32_t tL = rotl(AL + fL + X[RL[j]] + KL[4], SL[j]) + EL;
        AL = EL; EL = DL; DL = rotl(CL, 10); CL = BL; BL = tL;
        uint32_t tR = rotl(AR + fR + X[RR[j]] + KR[4], SR[j]) + ER;
        AR = ER; ER = DR; DR = rotl(CR, 10); CR = BR; BR = tR;
    }

    // Final addition
    uint32_t t = H[1] + CL + DR;
    H[1] = H[2] + DL + ER;
    H[2] = H[3] + EL + AR;
    H[3] = H[4] + AL + BR;
    H[4] = H[0] + BL + CR;
    H[0] = t;

    // Output hash (little-endian)
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        hash[i*4]     = H[i] & 0xff;
        hash[i*4 + 1] = (H[i] >> 8) & 0xff;
        hash[i*4 + 2] = (H[i] >> 16) & 0xff;
        hash[i*4 + 3] = (H[i] >> 24) & 0xff;
    }
}

/**
 * Batch RIPEMD160 kernel.
 */
__global__ void ripemd160_batch_kernel(
    const uint8_t* __restrict__ inputs,   // 32 bytes each
    uint8_t* __restrict__ outputs,        // 20 bytes each
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* in = inputs + idx * 32;
    uint8_t* out = outputs + idx * 20;

    ripemd160_hash(in, out);
}

/**
 * Combined SHA256 + RIPEMD160 for public key to address.
 * Input: 65-byte uncompressed public key (04 || x || y)
 * Output: 20-byte address hash (hash160)
 */
__device__ void pubkey_to_hash160(
    const uint8_t* pubkey,  // 65 bytes
    uint8_t* hash160        // 20 bytes
) {
    uint8_t sha256_out[32];

    // First SHA256 of pubkey
    // (Would call sha256_hash here - simplified for this file)

    // Then RIPEMD160
    ripemd160_hash(sha256_out, hash160);
}

// Host wrapper
extern "C" {

cudaError_t ripemd160_batch(
    const uint8_t* d_inputs,
    uint8_t* d_outputs,
    size_t count,
    cudaStream_t stream
) {
    const int threads_per_block = 256;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    ripemd160_batch_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_inputs,
        d_outputs,
        count
    );

    return cudaGetLastError();
}

}  // extern "C"

}  // namespace gpu
}  // namespace collider
