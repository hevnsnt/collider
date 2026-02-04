/**
 * Collider Fused GPU Pipeline
 *
 * Fused kernels that combine multiple stages for maximum throughput:
 * - SHA256(passphrase) → private_key
 * - EC_MUL(private_key) → public_key
 * - SHA256(public_key) → intermediate_hash
 * - RIPEMD160(intermediate_hash) → hash160
 * - Bloom filter check
 *
 * Fusion reduces memory bandwidth and kernel launch overhead.
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace collider {
namespace gpu {

// Forward declarations from other kernel files
extern "C" {
    cudaError_t secp256k1_init_table(cudaStream_t stream);
}

// =============================================================================
// EMBEDDED SHA256 (for fusion)
// =============================================================================

static __device__ __constant__ uint32_t SHA256_K[64] = {
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

__device__ __forceinline__ uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ uint32_t sig0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ uint32_t sig1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ uint32_t gam0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ uint32_t gam1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

/**
 * SHA256 hash for short messages (< 56 bytes, single block after padding).
 * Optimized for passphrase hashing.
 */
__device__ void sha256_short(const uint8_t* msg, size_t len, uint8_t* hash) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint32_t W[64];

    // Prepare message block with padding
    uint8_t block[64] = {0};
    for (size_t i = 0; i < len && i < 55; i++) {
        block[i] = msg[i];
    }
    block[len] = 0x80;

    // Length in bits (big-endian)
    uint64_t bit_len = len * 8;
    block[63] = bit_len & 0xff;
    block[62] = (bit_len >> 8) & 0xff;
    block[61] = (bit_len >> 16) & 0xff;
    block[60] = (bit_len >> 24) & 0xff;
    block[59] = (bit_len >> 32) & 0xff;
    block[58] = (bit_len >> 40) & 0xff;
    block[57] = (bit_len >> 48) & 0xff;
    block[56] = (bit_len >> 56) & 0xff;

    // Parse block into 16 words (big-endian)
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i*4] << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) |
               ((uint32_t)block[i*4+3]);
    }

    // Extend to 64 words
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gam1(W[i-2]) + W[i-7] + gam0(W[i-15]) + W[i-16];
    }

    // Compression
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sig1(e) + ch(e, f, g) + SHA256_K[i] + W[i];
        uint32_t t2 = sig0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;

    // Output (big-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i*4]   = (state[i] >> 24) & 0xff;
        hash[i*4+1] = (state[i] >> 16) & 0xff;
        hash[i*4+2] = (state[i] >> 8) & 0xff;
        hash[i*4+3] = state[i] & 0xff;
    }
}

/**
 * SHA256 for 33-byte compressed public key (special case).
 */
__device__ void sha256_33bytes(const uint8_t* pubkey, uint8_t* hash) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint32_t W[64];
    uint8_t block[64] = {0};

    // Copy 33 bytes of pubkey
    #pragma unroll
    for (int i = 0; i < 33; i++) {
        block[i] = pubkey[i];
    }
    block[33] = 0x80;  // Padding

    // Length = 33 * 8 = 264 bits = 0x108
    block[62] = 0x01;
    block[63] = 0x08;

    // Parse block
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i*4] << 24) |
               ((uint32_t)block[i*4+1] << 16) |
               ((uint32_t)block[i*4+2] << 8) |
               ((uint32_t)block[i*4+3]);
    }

    // Extend
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        W[i] = gam1(W[i-2]) + W[i-7] + gam0(W[i-15]) + W[i-16];
    }

    // Compression
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = h + sig1(e) + ch(e, f, g) + SHA256_K[i] + W[i];
        uint32_t t2 = sig0(a) + maj(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;

    // Output
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        hash[i*4]   = (state[i] >> 24) & 0xff;
        hash[i*4+1] = (state[i] >> 16) & 0xff;
        hash[i*4+2] = (state[i] >> 8) & 0xff;
        hash[i*4+3] = state[i] & 0xff;
    }
}

// =============================================================================
// EMBEDDED RIPEMD160 (for fusion)
// =============================================================================

static __device__ __constant__ uint32_t RIPEMD_K_LEFT[5] = {
    0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E
};

static __device__ __constant__ uint32_t RIPEMD_K_RIGHT[5] = {
    0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000
};

static __device__ __constant__ int RIPEMD_R_LEFT[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};

static __device__ __constant__ int RIPEMD_R_RIGHT[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};

static __device__ __constant__ int RIPEMD_S_LEFT[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};

static __device__ __constant__ int RIPEMD_S_RIGHT[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

__device__ __forceinline__ uint32_t rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

__device__ __forceinline__ uint32_t ripemd_f(int j, uint32_t x, uint32_t y, uint32_t z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

/**
 * RIPEMD160 hash for 32-byte input (SHA256 output).
 */
__device__ void ripemd160_32bytes(const uint8_t* msg, uint8_t* hash) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476, h4 = 0xC3D2E1F0;

    // Prepare message block with padding
    uint8_t block[64] = {0};
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        block[i] = msg[i];
    }
    block[32] = 0x80;  // Padding bit

    // Length = 32 * 8 = 256 bits = 0x100 (little-endian)
    block[56] = 0x00;
    block[57] = 0x01;

    // Parse block into 16 words (little-endian)
    uint32_t X[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        X[i] = ((uint32_t)block[i*4]) |
               ((uint32_t)block[i*4+1] << 8) |
               ((uint32_t)block[i*4+2] << 16) |
               ((uint32_t)block[i*4+3] << 24);
    }

    // Left path
    uint32_t al = h0, bl = h1, cl = h2, dl = h3, el = h4;
    // Right path
    uint32_t ar = h0, br = h1, cr = h2, dr = h3, er = h4;

    for (int j = 0; j < 80; j++) {
        int round = j / 16;

        // Left path
        uint32_t tl = al + ripemd_f(j, bl, cl, dl) + X[RIPEMD_R_LEFT[j]] + RIPEMD_K_LEFT[round];
        tl = rotl(tl, RIPEMD_S_LEFT[j]) + el;
        al = el; el = dl; dl = rotl(cl, 10); cl = bl; bl = tl;

        // Right path
        uint32_t tr = ar + ripemd_f(79 - j, br, cr, dr) + X[RIPEMD_R_RIGHT[j]] + RIPEMD_K_RIGHT[round];
        tr = rotl(tr, RIPEMD_S_RIGHT[j]) + er;
        ar = er; er = dr; dr = rotl(cr, 10); cr = br; br = tr;
    }

    // Final addition
    uint32_t t = h1 + cl + dr;
    h1 = h2 + dl + er;
    h2 = h3 + el + ar;
    h3 = h4 + al + br;
    h4 = h0 + bl + cr;
    h0 = t;

    // Output (little-endian)
    hash[0]  = h0 & 0xff; hash[1]  = (h0 >> 8) & 0xff;
    hash[2]  = (h0 >> 16) & 0xff; hash[3]  = (h0 >> 24) & 0xff;
    hash[4]  = h1 & 0xff; hash[5]  = (h1 >> 8) & 0xff;
    hash[6]  = (h1 >> 16) & 0xff; hash[7]  = (h1 >> 24) & 0xff;
    hash[8]  = h2 & 0xff; hash[9]  = (h2 >> 8) & 0xff;
    hash[10] = (h2 >> 16) & 0xff; hash[11] = (h2 >> 24) & 0xff;
    hash[12] = h3 & 0xff; hash[13] = (h3 >> 8) & 0xff;
    hash[14] = (h3 >> 16) & 0xff; hash[15] = (h3 >> 24) & 0xff;
    hash[16] = h4 & 0xff; hash[17] = (h4 >> 8) & 0xff;
    hash[18] = (h4 >> 16) & 0xff; hash[19] = (h4 >> 24) & 0xff;
}

// =============================================================================
// BLOOM FILTER CHECK (embedded for fusion)
// =============================================================================

__device__ __forceinline__ bool bloom_check_inline(
    const uint8_t* hash160,
    const uint8_t* bits,
    size_t num_bits,
    int num_hashes
) {
    // Double hashing: h(i) = h1 + i*h2 mod m
    uint64_t h1 = 0, h2 = 0;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        h1 |= (uint64_t)hash160[i] << (i * 8);
        h2 |= (uint64_t)hash160[8 + i] << (i * 8);
    }
    h2 |= 1;  // Ensure odd

    for (int i = 0; i < num_hashes; i++) {
        uint64_t idx = (h1 + (uint64_t)i * h2) % num_bits;
        uint64_t byte_idx = idx / 8;
        int bit_idx = idx % 8;

        if (!(bits[byte_idx] & (1 << bit_idx))) {
            return false;
        }
    }

    return true;
}

// =============================================================================
// SECP256K1 SCALAR MULTIPLICATION (simplified inline version)
// =============================================================================

// Import structures from secp256k1.cu
struct uint256 {
    uint32_t limbs[8];
};

struct ECPointAffine {
    uint256 x;
    uint256 y;
};

// Extern reference to precomputed table
extern __device__ void ec_mul_simple(ECPointAffine& result, const uint256& scalar);

// =============================================================================
// FUSED PIPELINE KERNEL
// =============================================================================

/**
 * Fully fused brain wallet pipeline kernel.
 *
 * For each passphrase:
 * 1. SHA256(passphrase) → private_key (32 bytes)
 * 2. EC_MUL(private_key, G) → public_key (33 bytes compressed)
 * 3. SHA256(public_key) → intermediate (32 bytes)
 * 4. RIPEMD160(intermediate) → hash160 (20 bytes)
 * 5. Bloom filter check → match flag
 *
 * If match, store index for host to verify.
 */
__global__ void brain_wallet_fused_kernel(
    const uint8_t* __restrict__ passphrases,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    const uint8_t* __restrict__ bloom_filter,
    uint64_t bloom_bits,
    int bloom_hashes,
    uint32_t* __restrict__ match_indices,
    uint32_t* __restrict__ match_count,
    uint8_t* __restrict__ private_keys_out,  // Optional: store for verification
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Step 1: SHA256(passphrase) → private_key
    const uint8_t* passphrase = passphrases + offsets[idx];
    uint32_t len = lengths[idx];

    uint8_t private_key[32];
    sha256_short(passphrase, len, private_key);

    // Optionally store private key for later verification
    if (private_keys_out != nullptr) {
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            private_keys_out[idx * 32 + i] = private_key[i];
        }
    }

    // Step 2: EC multiply to get public key
    // Convert private key bytes to uint256
    uint256 scalar;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        scalar.limbs[i] = ((uint32_t)private_key[i*4+3]) |
                          ((uint32_t)private_key[i*4+2] << 8) |
                          ((uint32_t)private_key[i*4+1] << 16) |
                          ((uint32_t)private_key[i*4] << 24);
    }

    ECPointAffine pubkey;
    ec_mul_simple(pubkey, scalar);

    // Compress public key (02/03 prefix + x coordinate)
    uint8_t compressed_pubkey[33];
    // Determine prefix based on y coordinate parity
    compressed_pubkey[0] = (pubkey.y.limbs[0] & 1) ? 0x03 : 0x02;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        compressed_pubkey[1 + i*4]     = (pubkey.x.limbs[7-i] >> 24) & 0xff;
        compressed_pubkey[1 + i*4 + 1] = (pubkey.x.limbs[7-i] >> 16) & 0xff;
        compressed_pubkey[1 + i*4 + 2] = (pubkey.x.limbs[7-i] >> 8) & 0xff;
        compressed_pubkey[1 + i*4 + 3] = pubkey.x.limbs[7-i] & 0xff;
    }

    // Step 3: SHA256(public_key) → intermediate hash
    uint8_t sha256_pubkey[32];
    sha256_33bytes(compressed_pubkey, sha256_pubkey);

    // Step 4: RIPEMD160(sha256) → hash160
    uint8_t hash160[20];
    ripemd160_32bytes(sha256_pubkey, hash160);

    // Step 5: Bloom filter check
    bool match = bloom_check_inline(hash160, bloom_filter, bloom_bits, bloom_hashes);

    if (match) {
        // Atomic increment and store index
        uint32_t slot = atomicAdd(match_count, 1);
        if (slot < 1024) {  // Limit to prevent overflow
            match_indices[slot] = idx;
        }
    }
}

/**
 * Fixed-stride variant of the fused kernel.
 * Used with GPU rule engine output where each passphrase is at idx * stride.
 * This avoids GPU→CPU→GPU roundtrip when using GPU rule application.
 */
__global__ void brain_wallet_fused_kernel_fixed_stride(
    const uint8_t* __restrict__ passphrases,  // Fixed stride: idx * stride
    const uint32_t* __restrict__ lengths,      // Length of each passphrase
    uint32_t stride,                           // Bytes between consecutive passphrases
    const uint8_t* __restrict__ bloom_filter,
    uint64_t bloom_bits,
    int bloom_hashes,
    uint32_t* __restrict__ match_indices,
    uint32_t* __restrict__ match_count,
    uint8_t* __restrict__ private_keys_out,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Step 1: SHA256(passphrase) → private_key
    // Fixed stride access: passphrase at idx * stride
    const uint8_t* passphrase = passphrases + idx * stride;
    uint32_t len = lengths[idx];

    // Skip empty passphrases
    if (len == 0 || len > stride) return;

    uint8_t private_key[32];
    sha256_short(passphrase, len, private_key);

    // Optionally store private key for later verification
    if (private_keys_out != nullptr) {
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            private_keys_out[idx * 32 + i] = private_key[i];
        }
    }

    // Step 2: EC multiply to get public key
    uint256 scalar;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        scalar.limbs[i] = ((uint32_t)private_key[i*4+3]) |
                          ((uint32_t)private_key[i*4+2] << 8) |
                          ((uint32_t)private_key[i*4+1] << 16) |
                          ((uint32_t)private_key[i*4] << 24);
    }

    ECPointAffine pubkey;
    ec_mul_simple(pubkey, scalar);

    // Compress public key
    uint8_t compressed_pubkey[33];
    compressed_pubkey[0] = (pubkey.y.limbs[0] & 1) ? 0x03 : 0x02;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        compressed_pubkey[1 + i*4]     = (pubkey.x.limbs[7-i] >> 24) & 0xff;
        compressed_pubkey[1 + i*4 + 1] = (pubkey.x.limbs[7-i] >> 16) & 0xff;
        compressed_pubkey[1 + i*4 + 2] = (pubkey.x.limbs[7-i] >> 8) & 0xff;
        compressed_pubkey[1 + i*4 + 3] = pubkey.x.limbs[7-i] & 0xff;
    }

    // Step 3-4: SHA256 → RIPEMD160 → hash160
    uint8_t sha256_pubkey[32];
    sha256_33bytes(compressed_pubkey, sha256_pubkey);

    uint8_t hash160[20];
    ripemd160_32bytes(sha256_pubkey, hash160);

    // Step 5: Bloom filter check
    bool match = bloom_check_inline(hash160, bloom_filter, bloom_bits, bloom_hashes);

    if (match) {
        uint32_t slot = atomicAdd(match_count, 1);
        if (slot < 1024) {
            match_indices[slot] = idx;
        }
    }
}

/**
 * Simplified fused kernel without EC multiply (for testing bloom filter path).
 * Uses pre-computed public keys.
 */
__global__ void address_hash_fused_kernel(
    const uint8_t* __restrict__ pubkeys,  // 33 bytes each (compressed)
    const uint8_t* __restrict__ bloom_filter,
    uint64_t bloom_bits,
    int bloom_hashes,
    uint8_t* __restrict__ match_flags,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* pubkey = pubkeys + idx * 33;

    // SHA256(pubkey)
    uint8_t sha256_out[32];
    sha256_33bytes(pubkey, sha256_out);

    // RIPEMD160(sha256)
    uint8_t hash160[20];
    ripemd160_32bytes(sha256_out, hash160);

    // Bloom check
    match_flags[idx] = bloom_check_inline(hash160, bloom_filter, bloom_bits, bloom_hashes) ? 1 : 0;
}

// =============================================================================
// HOST API
// =============================================================================

extern "C" {

/**
 * Initialize the fused pipeline (precompute EC table).
 */
cudaError_t fused_pipeline_init(cudaStream_t stream) {
    return secp256k1_init_table(stream);
}

/**
 * Run the full fused brain wallet pipeline.
 * OPTIMIZED: Using 256 threads per block for better occupancy on Blackwell/Ada.
 */
cudaError_t fused_brain_wallet_batch(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    const uint8_t* d_bloom_filter,
    uint64_t bloom_bits,
    int bloom_hashes,
    uint32_t* d_match_indices,
    uint32_t* d_match_count,
    uint8_t* d_private_keys,  // Can be nullptr if not needed
    size_t count,
    cudaStream_t stream
) {
    // Reset match count
    cudaMemsetAsync(d_match_count, 0, sizeof(uint32_t), stream);

    // OPTIMIZED: 256 threads per block for better SM occupancy
    // RTX 5090 (Blackwell SM 10.0): 2048 threads/SM, 256 threads = 8 blocks/SM
    // RTX 4090 (Ada SM 8.9): 1536 threads/SM, 256 threads = 6 blocks/SM
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    brain_wallet_fused_kernel<<<blocks, threads, 0, stream>>>(
        d_passphrases,
        d_offsets,
        d_lengths,
        d_bloom_filter,
        bloom_bits,
        bloom_hashes,
        d_match_indices,
        d_match_count,
        d_private_keys,
        count
    );

    return cudaGetLastError();
}

/**
 * Run the fused brain wallet pipeline with fixed-stride input.
 * Used with GPU rule engine output where each passphrase is at idx * stride.
 * This keeps data on GPU (no roundtrip to CPU).
 *
 * @param d_passphrases   Device: passphrase buffer (fixed stride)
 * @param d_lengths       Device: length of each passphrase
 * @param stride          Bytes between consecutive passphrases (e.g., 256)
 * @param d_bloom_filter  Device: bloom filter
 * @param bloom_bits      Number of bits in bloom filter
 * @param bloom_hashes    Number of hash functions
 * @param d_match_indices Device: output match indices
 * @param d_match_count   Device: output match count
 * @param d_private_keys  Device: optional private key output
 * @param count           Number of passphrases
 * @param stream          CUDA stream
 */
cudaError_t fused_brain_wallet_batch_fixed_stride(
    const uint8_t* d_passphrases,
    const uint32_t* d_lengths,
    uint32_t stride,
    const uint8_t* d_bloom_filter,
    uint64_t bloom_bits,
    int bloom_hashes,
    uint32_t* d_match_indices,
    uint32_t* d_match_count,
    uint8_t* d_private_keys,
    size_t count,
    cudaStream_t stream
) {
    // Reset match count
    cudaMemsetAsync(d_match_count, 0, sizeof(uint32_t), stream);

    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    brain_wallet_fused_kernel_fixed_stride<<<blocks, threads, 0, stream>>>(
        d_passphrases,
        d_lengths,
        stride,
        d_bloom_filter,
        bloom_bits,
        bloom_hashes,
        d_match_indices,
        d_match_count,
        d_private_keys,
        count
    );

    return cudaGetLastError();
}

/**
 * Run just the address hashing and bloom check (for testing).
 */
cudaError_t fused_address_hash_batch(
    const uint8_t* d_pubkeys,
    const uint8_t* d_bloom_filter,
    uint64_t bloom_bits,
    int bloom_hashes,
    uint8_t* d_match_flags,
    size_t count,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    address_hash_fused_kernel<<<blocks, threads, 0, stream>>>(
        d_pubkeys,
        d_bloom_filter,
        bloom_bits,
        bloom_hashes,
        d_match_flags,
        count
    );

    return cudaGetLastError();
}

}  // extern "C"

}  // namespace gpu
}  // namespace collider
