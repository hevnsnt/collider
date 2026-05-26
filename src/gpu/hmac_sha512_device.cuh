/**
 * hmac_sha512_device.cuh -- HMAC-SHA512 on CUDA device.
 *
 * Implements RFC 2104 HMAC over SHA-512 as __device__ inline. Used
 * exclusively by the BIP-39 PBKDF2 kernel; the public API takes a
 * key + msg + length and writes a 64-byte MAC.
 *
 * Optimization for PBKDF2: when the same key is used across many
 * HMAC calls (the PBKDF2 inner loop iterates 2048 times with the
 * same password = HMAC key), the caller can pre-compute the
 * ipad/opad HASH STATES (one SHA-512 transform each) and reuse
 * them across the inner loop. hmac_sha512_with_preimage_states
 * exposes that fast path.
 *
 * SHA-512 block size = 128 bytes. HMAC pads keys shorter than 128
 * bytes with zeros; for keys >= 128 bytes the key is first hashed
 * down to a 64-byte digest, then zero-padded. The BIP-39 mnemonic
 * (which is the HMAC key here) can reach 24 words * up to 8 chars
 * + 23 separator spaces ~= 215 bytes for English, so the hash-down
 * path is reachable in normal operation. derive_padded_keys()
 * below implements both branches.
 *
 * Reference: RFC 2104.
 */

#pragma once

#include "gpu/sha512_device.cuh"

#include <cstdint>

#if defined(__CUDACC__) || defined(COLLIDER_USE_CUDA)

namespace collider::gpu::hmac_sha512 {

constexpr int SHA512_BLOCK_BYTES = 128;
constexpr int SHA512_DIGEST_BYTES = 64;

// Build the ipad / opad padded key buffers from a raw key. If
// key_len > 128 the key is first hashed down to 64 bytes (per RFC
// 2104); then padded with zeros to 128 bytes; then XORed with 0x36
// (ipad) / 0x5c (opad).
//
// Outputs ipad_block[128] and opad_block[128].
__device__ __forceinline__ void derive_padded_keys(
    const uint8_t* key, uint32_t key_len,
    uint8_t ipad_block[SHA512_BLOCK_BYTES],
    uint8_t opad_block[SHA512_BLOCK_BYTES]) {
    uint8_t k0[SHA512_BLOCK_BYTES] = {0};
    if (key_len > SHA512_BLOCK_BYTES) {
        // Hash the key down to 64 bytes. (Caller never hits this for
        // typical BIP-39 mnemonics but the spec requires it.)
        uint8_t digest[SHA512_DIGEST_BYTES];
        ::collider::gpu::sha512::hash_short(key, key_len, digest);
        for (int i = 0; i < SHA512_DIGEST_BYTES; ++i) k0[i] = digest[i];
    } else {
        for (uint32_t i = 0; i < key_len; ++i) k0[i] = key[i];
    }
    #pragma unroll
    for (int i = 0; i < SHA512_BLOCK_BYTES; ++i) {
        ipad_block[i] = k0[i] ^ 0x36;
        opad_block[i] = k0[i] ^ 0x5c;
    }
}

// Pre-process the ipad / opad blocks through ONE SHA-512 transform
// each, leaving the hash state ready to absorb the message (ipad
// path) or the inner digest (opad path) on top. This is the PBKDF2
// fast-path: precompute these states ONCE per password and reuse
// across 2048 HMAC iterations.
__device__ __forceinline__ void precompute_states(
    const uint8_t ipad_block[SHA512_BLOCK_BYTES],
    const uint8_t opad_block[SHA512_BLOCK_BYTES],
    uint64_t ipad_state[8],
    uint64_t opad_state[8]) {
    ::collider::gpu::sha512::init_h(ipad_state);
    ::collider::gpu::sha512::transform_block(ipad_state, ipad_block);
    ::collider::gpu::sha512::init_h(opad_state);
    ::collider::gpu::sha512::transform_block(opad_state, opad_block);
}

// One HMAC-SHA512 invocation with pre-computed ipad/opad states.
// msg is the input bytes to HMAC; msg_len must be <= 119 (so the
// final inner-hash block fits with padding + length). For PBKDF2
// inner-loop msg is always exactly 64 bytes (prior U value); the
// FIRST iteration's msg is salt + INT32_BE(block_index) which for
// BIP-39 ("mnemonic" + passphrase) is at most ~50 bytes -- well
// under the 119-byte ceiling.
//
// Output: 64-byte HMAC in `mac`.
__device__ __forceinline__ void hmac_with_states(
    const uint64_t ipad_state[8],
    const uint64_t opad_state[8],
    const uint8_t* msg, uint32_t msg_len,
    uint8_t mac[SHA512_DIGEST_BYTES]) {
    // Inner hash: ipad_state (already absorbed ipad_block) || msg
    uint64_t h_inner[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) h_inner[i] = ipad_state[i];

    // The total message length absorbed by h_inner is BLOCK + msg_len
    // bytes; pad + encode that.
    const uint32_t total_inner_len = SHA512_BLOCK_BYTES + msg_len;
    uint8_t blk[SHA512_BLOCK_BYTES] = {0};
    for (uint32_t i = 0; i < msg_len; ++i) blk[i] = msg[i];
    blk[msg_len] = 0x80;
    const uint64_t bit_len_inner = uint64_t(total_inner_len) * 8ULL;
    blk[120] = uint8_t((bit_len_inner >> 56) & 0xFF);
    blk[121] = uint8_t((bit_len_inner >> 48) & 0xFF);
    blk[122] = uint8_t((bit_len_inner >> 40) & 0xFF);
    blk[123] = uint8_t((bit_len_inner >> 32) & 0xFF);
    blk[124] = uint8_t((bit_len_inner >> 24) & 0xFF);
    blk[125] = uint8_t((bit_len_inner >> 16) & 0xFF);
    blk[126] = uint8_t((bit_len_inner >>  8) & 0xFF);
    blk[127] = uint8_t( bit_len_inner        & 0xFF);
    ::collider::gpu::sha512::transform_block(h_inner, blk);

    // Inner digest into a 64-byte buffer.
    uint8_t inner_digest[SHA512_DIGEST_BYTES];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint64_t v = h_inner[i];
        inner_digest[i * 8 + 0] = uint8_t((v >> 56) & 0xFF);
        inner_digest[i * 8 + 1] = uint8_t((v >> 48) & 0xFF);
        inner_digest[i * 8 + 2] = uint8_t((v >> 40) & 0xFF);
        inner_digest[i * 8 + 3] = uint8_t((v >> 32) & 0xFF);
        inner_digest[i * 8 + 4] = uint8_t((v >> 24) & 0xFF);
        inner_digest[i * 8 + 5] = uint8_t((v >> 16) & 0xFF);
        inner_digest[i * 8 + 6] = uint8_t((v >>  8) & 0xFF);
        inner_digest[i * 8 + 7] = uint8_t( v        & 0xFF);
    }

    // Outer hash: opad_state (absorbed opad_block) || inner_digest
    uint64_t h_outer[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) h_outer[i] = opad_state[i];
    const uint32_t total_outer_len = SHA512_BLOCK_BYTES + SHA512_DIGEST_BYTES;
    uint8_t outer_blk[SHA512_BLOCK_BYTES] = {0};
    #pragma unroll
    for (int i = 0; i < SHA512_DIGEST_BYTES; ++i) {
        outer_blk[i] = inner_digest[i];
    }
    outer_blk[SHA512_DIGEST_BYTES] = 0x80;
    const uint64_t bit_len_outer = uint64_t(total_outer_len) * 8ULL;
    outer_blk[120] = uint8_t((bit_len_outer >> 56) & 0xFF);
    outer_blk[121] = uint8_t((bit_len_outer >> 48) & 0xFF);
    outer_blk[122] = uint8_t((bit_len_outer >> 40) & 0xFF);
    outer_blk[123] = uint8_t((bit_len_outer >> 32) & 0xFF);
    outer_blk[124] = uint8_t((bit_len_outer >> 24) & 0xFF);
    outer_blk[125] = uint8_t((bit_len_outer >> 16) & 0xFF);
    outer_blk[126] = uint8_t((bit_len_outer >>  8) & 0xFF);
    outer_blk[127] = uint8_t( bit_len_outer        & 0xFF);
    ::collider::gpu::sha512::transform_block(h_outer, outer_blk);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const uint64_t v = h_outer[i];
        mac[i * 8 + 0] = uint8_t((v >> 56) & 0xFF);
        mac[i * 8 + 1] = uint8_t((v >> 48) & 0xFF);
        mac[i * 8 + 2] = uint8_t((v >> 40) & 0xFF);
        mac[i * 8 + 3] = uint8_t((v >> 32) & 0xFF);
        mac[i * 8 + 4] = uint8_t((v >> 24) & 0xFF);
        mac[i * 8 + 5] = uint8_t((v >> 16) & 0xFF);
        mac[i * 8 + 6] = uint8_t((v >>  8) & 0xFF);
        mac[i * 8 + 7] = uint8_t( v        & 0xFF);
    }
}

}  // namespace collider::gpu::hmac_sha512

#endif  // __CUDACC__ || COLLIDER_USE_CUDA
