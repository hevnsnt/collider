/**
 * Collider GPU SHA256 Implementation
 *
 * Optimized for RTX 5090 (Blackwell) architecture.
 * Processes batches of passphrases to generate private keys.
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace collider {
namespace gpu {

// SHA256 constants
static __constant__ uint32_t K[64] = {
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

// Initial hash values
static __constant__ uint32_t SHA256_H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

// Bitwise operations
__device__ __forceinline__ uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t gamma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t gamma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

/**
 * Compress a single 64-byte block into the SHA256 state.
 *
 * @param H   Running hash state (8 x uint32_t), updated in-place
 * @param block  64-byte input block
 */
__device__ void sha256_compress_block(uint32_t H[8], const uint8_t block[64]) {
    uint32_t W[64];

    // Parse block into 16 32-bit words (big-endian)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i*4] << 24) |
               ((uint32_t)block[i*4 + 1] << 16) |
               ((uint32_t)block[i*4 + 2] << 8) |
               (uint32_t)block[i*4 + 3];
    }

    // Extend to 64 words
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    // Initialize working variables
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

    // Main loop - OPTIMIZED: Partial unroll (8 iterations) for reduced register pressure
    // Full unroll of 64 iterations causes register spilling; 8 is optimal for Blackwell
    #pragma unroll 8
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);

        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Add compressed chunk to running hash
    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

/**
 * SHA256 hash of a single message (arbitrary length).
 *
 * Handles multi-block processing for messages of any size, including
 * messages > 55 bytes that require two (or more) padding blocks.
 *
 * @param message Input message bytes
 * @param len Message length in bytes (no limit)
 * @param hash Output 32-byte hash
 */
__device__ void sha256_hash(
    const uint8_t* message,
    size_t len,
    uint8_t* hash
) {
    uint32_t H[8];

    // Initialize hash values
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        H[i] = SHA256_H0[i];
    }

    // Process full 64-byte blocks from the message
    size_t offset = 0;
    size_t remaining = len;

    while (remaining >= 64) {
        uint8_t block[64];
        for (int i = 0; i < 64; i++) {
            block[i] = message[offset + i];
        }
        sha256_compress_block(H, block);
        offset += 64;
        remaining -= 64;
    }

    // Final block(s) with padding
    uint8_t block[64];
    for (int i = 0; i < 64; i++) {
        block[i] = 0;
    }

    // Copy remaining message bytes into final block
    for (size_t i = 0; i < remaining; i++) {
        block[i] = message[offset + i];
    }

    // Append the 0x80 padding marker
    block[remaining] = 0x80;

    if (remaining >= 56) {
        // Not enough room for the 8-byte length field in this block.
        // Compress this block and start a fresh one for the length.
        sha256_compress_block(H, block);
        for (int i = 0; i < 64; i++) {
            block[i] = 0;
        }
    }

    // Append total message length in bits as big-endian 64-bit integer
    uint64_t bit_len = (uint64_t)len * 8;
    block[56] = (uint8_t)(bit_len >> 56);
    block[57] = (uint8_t)(bit_len >> 48);
    block[58] = (uint8_t)(bit_len >> 40);
    block[59] = (uint8_t)(bit_len >> 32);
    block[60] = (uint8_t)(bit_len >> 24);
    block[61] = (uint8_t)(bit_len >> 16);
    block[62] = (uint8_t)(bit_len >> 8);
    block[63] = (uint8_t)(bit_len);

    sha256_compress_block(H, block);

    // Output hash (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i*4]     = (H[i] >> 24) & 0xff;
        hash[i*4 + 1] = (H[i] >> 16) & 0xff;
        hash[i*4 + 2] = (H[i] >> 8) & 0xff;
        hash[i*4 + 3] = H[i] & 0xff;
    }
}

/**
 * Batch SHA256 kernel.
 *
 * Each thread processes one passphrase.
 *
 * @param passphrases Concatenated passphrases
 * @param offsets Start offset of each passphrase
 * @param lengths Length of each passphrase
 * @param hashes Output hashes (32 bytes each)
 * @param count Number of passphrases
 */
__global__ void sha256_batch_kernel(
    const uint8_t* __restrict__ passphrases,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    uint8_t* __restrict__ hashes,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= count) return;

    const uint8_t* msg = passphrases + offsets[idx];
    size_t len = lengths[idx];
    uint8_t* out = hashes + idx * 32;

    sha256_hash(msg, len, out);
}

// sha256_hash_long removed: sha256_hash now handles messages of any length.

/**
 * SHA256 kernel for fixed-size 33-byte inputs (compressed public keys).
 * Optimized single-block processing.
 */
__global__ void sha256_pubkey33_kernel(
    const uint8_t* __restrict__ pubkeys,
    uint8_t* __restrict__ hashes,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* pubkey = pubkeys + idx * 33;
    uint8_t* out = hashes + idx * 32;

    uint32_t W[64];
    uint32_t H[8];

    // Initialize hash values
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        H[i] = SHA256_H0[i];
    }

    // Build message block: 33 bytes data + 0x80 + zeros + 8-byte length
    // Total: 33 + 1 + 22 + 8 = 64 bytes (single block)

    // First 8 words from pubkey (32 bytes) - big-endian conversion
    // Note: Cannot use vectorized uint32_t loads due to idx*33 misalignment
    // Cast to uint32_t before shifting to prevent sign extension for bytes >= 0x80
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        W[i] = ((uint32_t)pubkey[i*4] << 24) | ((uint32_t)pubkey[i*4 + 1] << 16) |
               ((uint32_t)pubkey[i*4 + 2] << 8) | (uint32_t)pubkey[i*4 + 3];
    }

    // Word 8: last byte of pubkey + 0x80 + zeros
    W[8] = ((uint32_t)pubkey[32] << 24) | (0x80u << 16);

    // Words 9-13: zeros
    W[9] = 0; W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0;

    // Words 14-15: length in bits (33 * 8 = 264 = 0x108)
    W[14] = 0;
    W[15] = 264;

    // Extend to 64 words
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    // Compression - OPTIMIZED: Partial unroll for reduced register pressure
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

    #pragma unroll 8
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;

    // Output (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i*4]     = (H[i] >> 24) & 0xff;
        out[i*4 + 1] = (H[i] >> 16) & 0xff;
        out[i*4 + 2] = (H[i] >> 8) & 0xff;
        out[i*4 + 3] = H[i] & 0xff;
    }
}

/**
 * SHA256 kernel for fixed-size 65-byte inputs (uncompressed public keys).
 * Specialized two-block processing -- avoids the generic byte-by-byte loop.
 *
 * Block 1: bytes 0-63 (16 words from pubkey)
 * Block 2: byte 64 + 0x80 padding + zeros + length (65*8=520 bits)
 */
__global__ void sha256_pubkey65_kernel(
    const uint8_t* __restrict__ pubkeys,
    uint8_t* __restrict__ hashes,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* pubkey = pubkeys + idx * 65;
    uint8_t* out = hashes + idx * 32;

    uint32_t H[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) H[i] = SHA256_H0[i];

    uint32_t W[64];

    // Block 1: first 64 bytes (words 0-15)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)pubkey[i*4] << 24) | ((uint32_t)pubkey[i*4+1] << 16) |
               ((uint32_t)pubkey[i*4+2] << 8) | (uint32_t)pubkey[i*4+3];
    }

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

    #pragma unroll 8
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;

    // Block 2: byte 64 + 0x80 padding + zeros + length
    W[0] = ((uint32_t)pubkey[64] << 24) | (0x80u << 16);
    #pragma unroll
    for (int i = 1; i < 14; i++) W[i] = 0;
    W[14] = 0;
    W[15] = 520;  // 65 * 8 bits

    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
    }

    a = H[0]; b = H[1]; c = H[2]; d = H[3];
    e = H[4]; f = H[5]; g = H[6]; h = H[7];

    #pragma unroll 8
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint32_t t2 = sigma0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[i*4]     = (H[i] >> 24) & 0xff;
        out[i*4 + 1] = (H[i] >> 16) & 0xff;
        out[i*4 + 2] = (H[i] >> 8) & 0xff;
        out[i*4 + 3] = H[i] & 0xff;
    }
}

// Host wrapper functions
extern "C" {

/**
 * Launch batch SHA256 computation for variable-length passphrases.
 */
cudaError_t sha256_batch(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    uint8_t* d_hashes,
    size_t count,
    cudaStream_t stream
) {
    const int threads_per_block = 256;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    sha256_batch_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_passphrases,
        d_offsets,
        d_lengths,
        d_hashes,
        count
    );

    return cudaGetLastError();
}

/**
 * Launch batch SHA256 for 33-byte compressed public keys.
 */
cudaError_t sha256_pubkey33_batch(
    const uint8_t* d_pubkeys,
    uint8_t* d_hashes,
    size_t count,
    cudaStream_t stream
) {
    const int threads_per_block = 256;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    sha256_pubkey33_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_pubkeys,
        d_hashes,
        count
    );

    return cudaGetLastError();
}

/**
 * Launch batch SHA256 for 65-byte uncompressed public keys.
 */
cudaError_t sha256_pubkey65_batch(
    const uint8_t* d_pubkeys,
    uint8_t* d_hashes,
    size_t count,
    cudaStream_t stream
) {
    const int threads_per_block = 256;
    const int blocks = (count + threads_per_block - 1) / threads_per_block;

    sha256_pubkey65_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_pubkeys,
        d_hashes,
        count
    );

    return cudaGetLastError();
}

}  // extern "C"

}  // namespace gpu
}  // namespace collider
