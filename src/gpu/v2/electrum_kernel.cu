/**
 * Electrum v1 + v2 GPU kernels (Phase 8, v1.4.0).
 *
 * v1: priv = SHA-256^100000(seed || ... ).
 * v2: seed = PBKDF2-HMAC-SHA512(mnemonic, "electrum"+passphrase, 2048),
 *     priv = top 32 bytes of seed (BIP-32 master entropy / 2).
 *
 * Mirrors src/gpu/v2/electrum_cpu.hpp byte-for-byte.
 */

#include "brain_wallet_v2.hpp"
#include "device_hashes.cuh"
#include "puzzle_check.cuh"   // static __device__ helper + extern __constant__

#include <cuda_runtime.h>

namespace collider {
namespace gpu {
namespace v2 {

// ---------------------------------------------------------------------------
// Electrum v1: x_0 = SHA-256(seed); x_{n+1} = SHA-256(x_n || seed); priv = x_100000
// ---------------------------------------------------------------------------

__global__ void v2_kernel_electrum_v1(
    const uint8_t* __restrict__ passphrases,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    size_t count,
    V2MatchRecord* __restrict__ matches,
    uint32_t* __restrict__ match_count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const uint8_t* seed = passphrases + offsets[idx];
    uint32_t       slen = lengths[idx];
    if (slen > 64) return;  // bound stack buffer

    uint8_t state[32];
    device::sha256(seed, slen, state);

    uint8_t buf[32 + 64];
    for (uint32_t i = 0; i < 32; ++i) buf[i] = state[i];
    for (uint32_t i = 0; i < slen; ++i) buf[32 + i] = seed[i];

    for (uint32_t i = 0; i < 100000; ++i) {
        for (uint32_t j = 0; j < 32; ++j) buf[j] = state[j];
        device::sha256(buf, 32 + slen, state);
    }

    v2_check_priv_against_puzzles(
        (uint32_t)idx, 0,
        0xE1,  // marker for Electrum v1
        state, matches, match_count);
}

cudaError_t v2_electrum_v1_brute(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    size_t count,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream)
{
    if (count == 0) return cudaErrorInvalidValue;
    constexpr int BLOCK = 64;   // small block; per-thread cost is high
    int blocks = (int)((count + BLOCK - 1) / BLOCK);
    v2_kernel_electrum_v1<<<blocks, BLOCK, 0, stream>>>(
        d_passphrases, d_offsets, d_lengths, count,
        d_matches, d_match_count);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Electrum v2 / BIP-39 PBKDF2-HMAC-SHA512(2048) on device.
// ---------------------------------------------------------------------------

namespace {

// PBKDF2-HMAC-SHA512: produces 64-byte output for one block (i=1).
// Sufficient for v2 seed (we only need 64 bytes) and for the BIP-39 seed.
__device__ static void pbkdf2_hmac_sha512_64(
    const uint8_t* pw, uint32_t pw_len,
    const uint8_t* salt_prefix, uint32_t salt_prefix_len,
    const uint8_t* salt_suffix, uint32_t salt_suffix_len,
    uint32_t iterations,
    uint8_t out[64])
{
    // Build salt || INT(1)
    uint8_t salt_full[256];
    uint32_t salt_len = 0;
    if (salt_prefix_len + salt_suffix_len + 4 > sizeof(salt_full)) return;
    for (uint32_t i = 0; i < salt_prefix_len; ++i) salt_full[salt_len++] = salt_prefix[i];
    for (uint32_t i = 0; i < salt_suffix_len; ++i) salt_full[salt_len++] = salt_suffix[i];
    salt_full[salt_len++] = 0;
    salt_full[salt_len++] = 0;
    salt_full[salt_len++] = 0;
    salt_full[salt_len++] = 1;

    uint8_t U[64], T[64];
    device::hmac_sha512(pw, pw_len, salt_full, salt_len, U);
    for (int k = 0; k < 64; ++k) T[k] = U[k];
    for (uint32_t j = 1; j < iterations; ++j) {
        uint8_t Un[64];
        device::hmac_sha512(pw, pw_len, U, 64, Un);
        for (int k = 0; k < 64; ++k) {
            U[k] = Un[k];
            T[k] ^= Un[k];
        }
    }
    for (int k = 0; k < 64; ++k) out[k] = T[k];
}

}  // namespace

__global__ void v2_kernel_electrum_v2(
    const uint8_t* __restrict__ passphrases,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    size_t count,
    const uint8_t* __restrict__ salt_suffix,
    uint32_t                    salt_suffix_len,
    V2MatchRecord* __restrict__ matches,
    uint32_t* __restrict__ match_count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const uint8_t* mn = passphrases + offsets[idx];
    uint32_t       mlen = lengths[idx];
    if (mlen > 256) return;

    static const uint8_t kSalt[8] = {'e','l','e','c','t','r','u','m'};

    uint8_t seed[64];
    pbkdf2_hmac_sha512_64(
        mn, mlen,
        kSalt, 8,
        salt_suffix, salt_suffix_len,
        2048, seed);

    // Top 32 bytes of seed -> priv (BIP-32 master split convention).
    v2_check_priv_against_puzzles(
        (uint32_t)idx, 0,
        0xE2,  // marker for Electrum v2
        seed, matches, match_count);
}

cudaError_t v2_electrum_v2_brute(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    size_t count,
    const std::string& bip39_passphrase,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream)
{
    if (count == 0) return cudaErrorInvalidValue;
    if (bip39_passphrase.size() > 200) return cudaErrorInvalidValue;
    // Push the salt suffix to device
    uint8_t* d_suffix = nullptr;
    cudaError_t rc = cudaSuccess;
    if (!bip39_passphrase.empty()) {
        rc = cudaMalloc((void**)&d_suffix, bip39_passphrase.size());
        if (rc != cudaSuccess) return rc;
        rc = cudaMemcpyAsync(d_suffix, bip39_passphrase.data(),
                              bip39_passphrase.size(),
                              cudaMemcpyHostToDevice, stream);
        if (rc != cudaSuccess) { cudaFree(d_suffix); return rc; }
    }
    constexpr int BLOCK = 32;
    int blocks = (int)((count + BLOCK - 1) / BLOCK);
    v2_kernel_electrum_v2<<<blocks, BLOCK, 0, stream>>>(
        d_passphrases, d_offsets, d_lengths, count,
        d_suffix, (uint32_t)bip39_passphrase.size(),
        d_matches, d_match_count);
    rc = cudaGetLastError();
    // Stream-ordered free; destruction blocks until kernel completes.
    if (d_suffix) cudaFreeAsync(d_suffix, stream);
    return rc;
}

}  // namespace v2
}  // namespace gpu
}  // namespace collider
