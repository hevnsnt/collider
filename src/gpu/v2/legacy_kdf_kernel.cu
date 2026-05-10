/**
 * Legacy-KDF GPU kernel framework (Phase 7, v1.4.0).
 *
 * Many defunct early web-wallet generators combined MD5 / SHA-1 / SHA-256
 * iterations with static salts in non-standard ways. The dispatch table
 * below lets new KDFs be added by appending one function and one enum
 * value -- no kernel rewrite. Each KDF takes the passphrase bytes and
 * writes 32 priv bytes.
 *
 * The example KDFs shipped here cover the most-cited historical patterns;
 * additional ones (MultiBit, Mycelium, BitcoinJ early test) can be added
 * trivially by extending kKdfTable. Each sticks to MD5 / SHA-1 / SHA-256
 * to avoid bloating register usage.
 */

#include "brain_wallet_v2.hpp"
#include "device_hashes.cuh"
#include "puzzle_check.cuh"   // static __device__ helper + extern __constant__

#include <cuda_runtime.h>

namespace collider {
namespace gpu {
namespace v2 {

// ---------------------------------------------------------------------------
// Per-KDF derivers. Each must produce a 32-byte priv from a passphrase.
// ---------------------------------------------------------------------------

// KDF 0: SHA-256(passphrase)              -- "stock"; baseline
__device__ static void kdf_sha256(
    const uint8_t* pw, uint32_t len, uint8_t out[32])
{
    device::sha256(pw, len, out);
}

// KDF 1: MD5(passphrase) || MD5(MD5(passphrase) || passphrase)
//        truncated to 32 bytes.
//   Pattern seen in early PHP web-wallet generators (e.g. brainwallet.io
//   2013 era). Deterministic, no salt, MD5 base.
__device__ static void kdf_md5_concat(
    const uint8_t* pw, uint32_t len, uint8_t out[32])
{
    if (len > 200) len = 200;
    uint8_t a[16], b[16];
    device::md5(pw, len, a);
    uint8_t buf[16 + 200];
    for (int i = 0; i < 16; ++i) buf[i] = a[i];
    for (uint32_t i = 0; i < len; ++i) buf[16 + i] = pw[i];
    device::md5(buf, 16 + len, b);
    for (int i = 0; i < 16; ++i) {
        out[i]      = a[i];
        out[16 + i] = b[i];
    }
}

// KDF 2: SHA-1(passphrase) || SHA-1(SHA-1(passphrase)) truncated/padded to 32.
//   Pattern seen in some early MultiBit Classic alpha builds (pre-PBKDF2).
__device__ static void kdf_sha1_chain(
    const uint8_t* pw, uint32_t len, uint8_t out[32])
{
    uint8_t a[20], b[20];
    device::sha1(pw, len, a);
    device::sha1(a, 20, b);
    for (int i = 0; i < 20; ++i) out[i] = a[i];
    for (int i = 0; i < 12; ++i) out[20 + i] = b[i];
}

// KDF 3: 1024 iterations of SHA-256(state || passphrase || "BTC-SALT")
//   Models early web-wallet "stretching" loops with a static suffix.
__device__ static void kdf_sha256_iter_salt(
    const uint8_t* pw, uint32_t len, uint8_t out[32])
{
    if (len > 100) len = 100;
    static const uint8_t SALT[] = "BTC-SALT";
    constexpr uint32_t SALT_LEN = sizeof(SALT) - 1;

    uint8_t state[32];
    device::sha256(pw, len, state);
    // Buffer layout written ONCE before the loop:
    //   [0..32)        state (overwritten each iter)
    //   [32..32+len)   pw    (constant)
    //   [32+len..32+len+SALT_LEN)  SALT  (constant)
    uint8_t buf[32 + 100 + 8];
    for (uint32_t i = 0; i < len; ++i)      buf[32 + i]           = pw[i];
    for (uint32_t i = 0; i < SALT_LEN; ++i) buf[32 + len + i]     = SALT[i];
    for (uint32_t k = 0; k < 1024; ++k) {
        for (int i = 0; i < 32; ++i) buf[i] = state[i];
        device::sha256(buf, 32 + len + SALT_LEN, state);
    }
    for (int i = 0; i < 32; ++i) out[i] = state[i];
}

// KDF 4: MD5(passphrase) -> SHA-256 -> RIPEMD-160 (16/32/20). Out:
//   sha256_h || ripemd_h || md5_h truncated to 32 bytes.
//   Models tools that confused chained hashing for "secure stretching".
__device__ static void kdf_mixed_md5_sha_ripemd(
    const uint8_t* pw, uint32_t len, uint8_t out[32])
{
    uint8_t mh[16], sh[32], rh[20];
    device::md5(pw, len, mh);
    device::sha256(mh, 16, sh);
    device::ripemd160(sh, 32, rh);
    for (int i = 0; i < 20; ++i) out[i]      = rh[i];
    for (int i = 0; i < 12; ++i) out[20 + i] = sh[i];
}

// ---------------------------------------------------------------------------
// Dispatch table
// ---------------------------------------------------------------------------

constexpr uint8_t kKdfCount = 5;

__device__ static void kdf_dispatch(
    uint8_t id, const uint8_t* pw, uint32_t len, uint8_t out[32])
{
    switch (id) {
        case 0: kdf_sha256(pw, len, out);                  break;
        case 1: kdf_md5_concat(pw, len, out);              break;
        case 2: kdf_sha1_chain(pw, len, out);              break;
        case 3: kdf_sha256_iter_salt(pw, len, out);        break;
        case 4: kdf_mixed_md5_sha_ripemd(pw, len, out);    break;
        default: kdf_sha256(pw, len, out);                 break;
    }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

__global__ void v2_kernel_legacy_kdf(
    const uint8_t* __restrict__ passphrases,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    size_t count,
    uint8_t kdf_id,
    V2MatchRecord* __restrict__ matches,
    uint32_t* __restrict__ match_count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    const uint8_t* pw = passphrases + offsets[idx];
    uint32_t       len = lengths[idx];

    uint8_t priv[32];
    kdf_dispatch(kdf_id, pw, len, priv);

    v2_check_priv_against_puzzles(
        (uint32_t)idx, (uint64_t)kdf_id,
        kdf_id, priv, matches, match_count);
}

cudaError_t v2_legacy_kdf_brute(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    size_t count,
    uint8_t kdf_id,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream)
{
    if (count == 0) return cudaErrorInvalidValue;
    if (kdf_id >= kKdfCount) return cudaErrorInvalidValue;
    constexpr int BLOCK = 256;
    int blocks = (int)((count + BLOCK - 1) / BLOCK);
    v2_kernel_legacy_kdf<<<blocks, BLOCK, 0, stream>>>(
        d_passphrases, d_offsets, d_lengths, count, kdf_id,
        d_matches, d_match_count);
    return cudaGetLastError();
}

}  // namespace v2
}  // namespace gpu
}  // namespace collider
