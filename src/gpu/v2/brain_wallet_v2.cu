/**
 * Brain Wallet v2 GPU Pipeline — Item 6 implementation
 *
 * This file ships the FIRST real kernel: puzzle-mode.  It evaluates a
 * passphrase against a fixed list of "puzzle target" private keys, with
 * each target carrying a (mask, value) pair.  The kernel can short-circuit
 * before EC_MUL when puzzle-mode is the only enabled mode, which makes it
 * an order of magnitude faster than the standard brain-wallet pipeline.
 *
 * Items 1, 2, 4, 5 still return cudaErrorNotSupported -- their kernels
 * land in follow-up commits per docs/BRAINWALLET-V2-SPEC.md.
 *
 * The SHA-256 helpers are intentionally duplicated here (rather than
 * refactored out of fused_pipeline.cu) so this kernel is self-contained:
 *   - no risk of breaking the production fused pipeline
 *   - the v2 file can be excluded/included in the build independently
 *   - simpler diff to review
 */

#include "brain_wallet_v2.hpp"
#include "device_hashes.cuh"   // device::sha512 + hmac_sha512 for S7/S8
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>
#include <type_traits>

namespace collider {
namespace gpu {
namespace v2 {

// ===========================================================================
// SHA-256 is provided by device_hashes.cuh (device::sha256). The legacy
// in-file v2_sha256 + SHA256_K_V2 were removed in PR #15 review (Gemini):
// duplication wasted constant memory in the unity build and risked drift
// vs the shared header.
// ===========================================================================

// Convenience alias so the existing call sites stay readable.
static __device__ __forceinline__ void v2_sha256(
    const uint8_t* msg, size_t len, uint8_t out[32])
{
    device::sha256(msg, (uint32_t)len, out);
}

// ===========================================================================
// Constant-memory storage for puzzle targets (item 6)
// ===========================================================================

__device__ __constant__ PuzzleTarget c_puzzle_targets[PUZZLE_TARGET_MAX];
__device__ __constant__ int          c_puzzle_target_count;

// ===========================================================================
// Helpers for derivation schemes used by the puzzle-mode kernel.
//
// Each helper takes:
//   - the passphrase bytes
//   - puzzle_n N (so the function can encode N into the hash input)
//   - 32-byte output buffer (big-endian SHA-256 result)
//
// Scheme parity must match the host-side reference in
// tools/brainwallet/puzzle_mode.py (functions _s_*).
// ===========================================================================

// ---------------------------------------------------------------------------
// DerivationScheme enum implementations (Phase 3, v1.4.0).
// Each function takes the passphrase and writes a 32-byte priv into `out`.
// SHA-256 schemes use the in-file v2_sha256; SHA-512 schemes use the
// shared device::sha512 / device::hmac_sha512 from device_hashes.cuh.
// ---------------------------------------------------------------------------

// DerivationScheme::SHA256_PW                  priv = SHA256(pw)
static __device__ void v2_d_sha256_pw(
    const uint8_t* pw, size_t pw_len, uint8_t out[32])
{
    v2_sha256(pw, pw_len, out);
}

// DerivationScheme::SHA256_SHA256_PW          priv = SHA256(SHA256(pw))
static __device__ void v2_d_sha256_sha256_pw(
    const uint8_t* pw, size_t pw_len, uint8_t out[32])
{
    uint8_t inner[32];
    v2_sha256(pw, pw_len, inner);
    v2_sha256(inner, 32, out);
}

// DerivationScheme::SHA256_PW_NEWLINE         priv = SHA256(pw || 0x0a)
static __device__ void v2_d_sha256_pw_newline(
    const uint8_t* pw, size_t pw_len, uint8_t out[32])
{
    if (pw_len > 255) pw_len = 255;
    uint8_t buf[256];
    for (size_t i = 0; i < pw_len; ++i) buf[i] = pw[i];
    buf[pw_len] = 0x0a;
    v2_sha256(buf, pw_len + 1, out);
}

// DerivationScheme::SHA256_PW_PW              priv = SHA256(pw || pw)
static __device__ void v2_d_sha256_pw_pw(
    const uint8_t* pw, size_t pw_len, uint8_t out[32])
{
    if (pw_len > 128) pw_len = 128;
    uint8_t buf[256];
    for (size_t i = 0; i < pw_len; ++i) buf[i] = pw[i];
    for (size_t i = 0; i < pw_len; ++i) buf[pw_len + i] = pw[i];
    v2_sha256(buf, pw_len * 2, out);
}

// DerivationScheme::SHA256_SHA256_PW_PW       priv = SHA256(SHA256(pw) || pw)
static __device__ void v2_d_sha256_sha256_pw_pw(
    const uint8_t* pw, size_t pw_len, uint8_t out[32])
{
    // Cap at 128 to match the Metal port's buffer budget (combo[32+128]).
    if (pw_len > 128) pw_len = 128;
    uint8_t buf[256];
    uint8_t inner[32];
    v2_sha256(pw, pw_len, inner);
    for (int i = 0; i < 32; ++i) buf[i] = inner[i];
    for (size_t i = 0; i < pw_len; ++i) buf[32 + i] = pw[i];
    v2_sha256(buf, 32 + pw_len, out);
}

// DerivationScheme::SHA256_ITER_16            priv = SHA256^16(pw)
static __device__ void v2_d_sha256_iter_16(
    const uint8_t* pw, size_t pw_len, uint8_t out[32])
{
    uint8_t a[32];
    v2_sha256(pw, pw_len, a);
    #pragma unroll
    for (int i = 0; i < 15; ++i) {
        v2_sha256(a, 32, a);
    }
    for (int i = 0; i < 32; ++i) out[i] = a[i];
}

// DerivationScheme::HMAC_SHA512_PW            priv = HMAC-SHA512(pw, "")[:32]
// (Empty key, passphrase as message; matches the legacy "HMAC stretch" pattern
// in some early Java/Android brain-wallet generators.)
static __device__ void v2_d_hmac_sha512_pw(
    const uint8_t* pw, size_t pw_len, uint8_t out[32])
{
    uint8_t mac[64];
    static const uint8_t empty_key[1] = {0};
    device::hmac_sha512(empty_key, 0, pw, (uint32_t)pw_len, mac);
    for (int i = 0; i < 32; ++i) out[i] = mac[i];
}

// DerivationScheme::SHA512_PW_HALF            priv = SHA512(pw)[:32]
static __device__ void v2_d_sha512_pw_half(
    const uint8_t* pw, size_t pw_len, uint8_t out[32])
{
    uint8_t full[64];
    device::sha512(pw, (uint32_t)pw_len, full);
    for (int i = 0; i < 32; ++i) out[i] = full[i];
}

// (The legacy per-puzzle-N salted scheme `v2_scheme_S2` and its kernel
// `v2_puzzle_only_kernel_S2` were removed in PR #15 review (Gemini): the
// kernel was unreachable from the public API after the Phase 3 multi-scheme
// refactor and reported its hits as SHA256_PW, which would have misled the
// host. The CPU prototype in tools/brainwallet/puzzle_mode.py retains the
// algorithm if anyone ever wants to reintroduce it.)

// ===========================================================================
// Pack a 32-byte big-endian hash into 4 little-endian-by-limb uint64s.
//   limb[0] = bits 0..63   (= hash_be[24..31])
//   limb[1] = bits 64..127  (= hash_be[16..23])
//   limb[2] = bits 128..191 (= hash_be[ 8..15])
//   limb[3] = bits 192..255 (= hash_be[ 0.. 7])
// ===========================================================================
static __device__ __forceinline__ void v2_hash_to_limbs(
    const uint8_t hash_be[32], uint64_t out_limbs[4])
{
    auto pack_be64 = [&](const uint8_t* p) -> uint64_t {
        uint64_t v = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) v = (v << 8) | (uint64_t)p[i];
        return v;
    };
    out_limbs[0] = pack_be64(hash_be + 24);
    out_limbs[1] = pack_be64(hash_be + 16);
    out_limbs[2] = pack_be64(hash_be +  8);
    out_limbs[3] = pack_be64(hash_be +  0);
}

// ---------------------------------------------------------------------------
// Multi-scheme dispatch kernel (Phase 3, v1.4.0).
//
// Templated on a single DerivationScheme so the compiler unrolls the inner
// hash composition. The host calls this kernel once per scheme bit set in
// scheme_mask; each call exercises ALL puzzle targets against THAT scheme's
// derivation. Dynamic dispatch on the device is avoided -- separate kernel
// launches are cheap and keep register pressure down.
// ---------------------------------------------------------------------------

template <DerivationScheme S>
static __device__ __forceinline__ void v2_derive(
    const uint8_t* pw, size_t pw_len, uint8_t out[32])
{
    if      constexpr (S == DerivationScheme::SHA256_PW)
        v2_d_sha256_pw(pw, pw_len, out);
    else if constexpr (S == DerivationScheme::SHA256_SHA256_PW)
        v2_d_sha256_sha256_pw(pw, pw_len, out);
    else if constexpr (S == DerivationScheme::SHA256_PW_NEWLINE)
        v2_d_sha256_pw_newline(pw, pw_len, out);
    else if constexpr (S == DerivationScheme::SHA256_PW_PW)
        v2_d_sha256_pw_pw(pw, pw_len, out);
    else if constexpr (S == DerivationScheme::SHA256_SHA256_PW_PW)
        v2_d_sha256_sha256_pw_pw(pw, pw_len, out);
    else if constexpr (S == DerivationScheme::SHA256_ITER_16)
        v2_d_sha256_iter_16(pw, pw_len, out);
    else if constexpr (S == DerivationScheme::HMAC_SHA512_PW)
        v2_d_hmac_sha512_pw(pw, pw_len, out);
    else if constexpr (S == DerivationScheme::SHA512_PW_HALF)
        v2_d_sha512_pw_half(pw, pw_len, out);
    else
        v2_d_sha256_pw(pw, pw_len, out);  // fallback for unknown values
}

// __launch_bounds__(256, 4): hint the compiler that we launch with up
// to 256 threads/block and want at least 4 resident blocks/SM. Without
// this hint nvcc over-allocates registers for the S7/S8 (SHA-512 +
// HMAC) instantiations and occupancy collapses on Ampere+. (Gemini PR
// #15 MED finding.)
template <DerivationScheme S>
__global__ __launch_bounds__(256, 4) void v2_puzzle_only_kernel_scheme(
    const uint8_t* __restrict__ passphrases,
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ lengths,
    size_t count,
    V2MatchRecord* __restrict__ matches,
    uint32_t* __restrict__ match_count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* pw = passphrases + offsets[idx];
    uint32_t pw_len = lengths[idx];

    uint8_t hash_out[32];
    v2_derive<S>(pw, pw_len, hash_out);

    uint64_t hash_limbs[4];
    v2_hash_to_limbs(hash_out, hash_limbs);

    const int n_targets = c_puzzle_target_count;
    for (int ti = 0; ti < n_targets; ++ti) {
        const PuzzleTarget& t = c_puzzle_targets[ti];
        bool match = true;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if ((hash_limbs[j] & t.low_mask[j]) != t.low_value[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            uint32_t slot = atomicAdd(match_count, 1u);
            if (slot < V2_MAX_MATCHES_PER_BATCH) {
                V2MatchRecord rec{};
                rec.pp_idx    = (uint32_t)idx;
                rec.puzzle_n  = t.puzzle_n;
                rec.scheme_id = (uint8_t)S;
                rec.kind      = (uint8_t)V2MatchRecord::Kind::PUZZLE_KEY_HIT;
                matches[slot] = rec;
            }
        }
    }
}

// ===========================================================================
// Public API
// ===========================================================================

// v2_init / v2_shutdown / v2_set_puzzle_targets / v2_brain_wallet_batch
// have C++ linkage to match the namespaced declarations in
// brain_wallet_v2.hpp. (An earlier extern "C" wrapper here caused a header /
// definition linkage mismatch -- caught by code review and removed.)

cudaError_t v2_init(cudaStream_t /*stream*/) {
    // No persistent state needed for puzzle-only mode (constant memory is
    // initialized per-call by v2_set_puzzle_targets). Once we add full brain-
    // wallet mode, this is where we'll preallocate the EC table and persistent
    // device buffers.
    return cudaSuccess;
}

void v2_shutdown() {
    // Nothing to free yet.
}

cudaError_t v2_set_puzzle_targets(const std::vector<PuzzleTarget>& targets) {
    if (targets.size() > PUZZLE_TARGET_MAX) return cudaErrorInvalidValue;
    int n = static_cast<int>(targets.size());
    if (n > 0) {
        cudaError_t err = cudaMemcpyToSymbol(
            c_puzzle_targets, targets.data(),
            sizeof(PuzzleTarget) * n, 0, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) return err;
    }
    cudaError_t err = cudaMemcpyToSymbol(c_puzzle_target_count, &n, sizeof(int),
                                         0, cudaMemcpyHostToDevice);
    return err;
}

cudaError_t v2_brain_wallet_batch(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    size_t count,
    uint32_t scheme_mask,
    uint32_t addr_mask,
    const uint8_t* /*d_bloom*/,
    uint64_t /*bloom_bits*/,
    int /*bloom_hashes*/,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream)
{
    // Phase 3 (v1.4.0): puzzle-only path supports all 8 DerivationScheme
    // values. S1..S6 are SHA-256 compositions; S7 (HMAC-SHA512) and S8
    // (truncated SHA-512) use device::hmac_sha512 / device::sha512 from
    // device_hashes.cuh. Multi-address (addr_mask != 0) routes through
    // the legacy brain-wallet pipeline; that's Phase 4.
    if (addr_mask != 0) {
        return cudaErrorNotSupported;
    }
    if (scheme_mask == 0) {
        return cudaErrorInvalidValue;
    }

    constexpr int BLOCK = 256;
    const int blocks = static_cast<int>((count + BLOCK - 1) / BLOCK);

    // Reset the match counter at the start of every batch. Without this
    // the kernel's atomicAdd into d_match_count keeps growing across
    // batches; the host then reads stale match records, and once the
    // counter exceeds d_matches' capacity the kernel walks off the end.
    // (Gemini PR #15 HIGH finding.)
    if (count > 0) {
        cudaError_t rc = cudaMemsetAsync(d_match_count, 0,
                                         sizeof(uint32_t), stream);
        if (rc != cudaSuccess) return rc;
    }

    // Per-scheme dispatch: one kernel launch per enabled scheme bit.
    // Stops at the first launch error to avoid burying it under later
    // launches' errors.
    //
    // The kernel template is instantiated per DerivationScheme value, so
    // the dispatch table is built statically as a fold over the enum. New
    // schemes plug in by adding one entry to kSchemes; the if-chain that
    // used to live here mostly auto-grew on each phase landing and was
    // cumulative copy-paste.
    using SchemeLauncher = cudaError_t (*)(
        const uint8_t*, const uint32_t*, const uint32_t*, size_t,
        V2MatchRecord*, uint32_t*, int, int, cudaStream_t);

    auto make_launcher = []<DerivationScheme S>() -> SchemeLauncher {
        return [](const uint8_t* d_passphrases,
                  const uint32_t* d_offsets,
                  const uint32_t* d_lengths,
                  size_t count,
                  V2MatchRecord* d_matches,
                  uint32_t* d_match_count,
                  int blocks, int block_size,
                  cudaStream_t stream) -> cudaError_t {
            v2_puzzle_only_kernel_scheme<S>
                <<<blocks, block_size, 0, stream>>>(
                    d_passphrases, d_offsets, d_lengths, count,
                    d_matches, d_match_count);
            return cudaGetLastError();
        };
    };

    struct SchemeEntry {
        DerivationScheme scheme;
        SchemeLauncher   launcher;
    };
    const SchemeEntry kSchemes[] = {
        { DerivationScheme::SHA256_PW,
          make_launcher.template operator()<DerivationScheme::SHA256_PW>() },
        { DerivationScheme::SHA256_SHA256_PW,
          make_launcher.template operator()<DerivationScheme::SHA256_SHA256_PW>() },
        { DerivationScheme::SHA256_PW_NEWLINE,
          make_launcher.template operator()<DerivationScheme::SHA256_PW_NEWLINE>() },
        { DerivationScheme::SHA256_PW_PW,
          make_launcher.template operator()<DerivationScheme::SHA256_PW_PW>() },
        { DerivationScheme::SHA256_SHA256_PW_PW,
          make_launcher.template operator()<DerivationScheme::SHA256_SHA256_PW_PW>() },
        { DerivationScheme::SHA256_ITER_16,
          make_launcher.template operator()<DerivationScheme::SHA256_ITER_16>() },
        // SHA-512-based schemes (Phase 3 finish, v1.4.0): device::sha512
        // and device::hmac_sha512 live in device_hashes.cuh.
        { DerivationScheme::HMAC_SHA512_PW,
          make_launcher.template operator()<DerivationScheme::HMAC_SHA512_PW>() },
        { DerivationScheme::SHA512_PW_HALF,
          make_launcher.template operator()<DerivationScheme::SHA512_PW_HALF>() },
    };

    for (const auto& entry : kSchemes) {
        if ((scheme_mask & scheme_bit(entry.scheme)) == 0) continue;
        const cudaError_t rc = entry.launcher(
            d_passphrases, d_offsets, d_lengths, count,
            d_matches, d_match_count, blocks, BLOCK, stream);
        if (rc != cudaSuccess) return rc;
    }
    return cudaSuccess;
}

// v2_weak_prng_brute lives in src/gpu/v2/weak_prng_kernel.cu (Phase 5).

cudaError_t v2_bip39_brute(
    const uint8_t* /*d_mnemonics*/,
    const uint32_t* /*d_offsets*/,
    const uint32_t* /*d_lengths*/,
    size_t /*count*/,
    const std::string& /*bip39_passphrase*/,
    const std::vector<std::vector<uint32_t>>& /*parsed_paths*/,
    uint32_t /*addr_mask*/,
    const uint8_t* /*d_bloom*/,
    uint64_t /*bloom_bits*/,
    int /*bloom_hashes*/,
    V2MatchRecord* /*d_matches*/,
    uint32_t* /*d_match_count*/,
    cudaStream_t /*stream*/)
{
    return cudaErrorNotSupported;
}

}  // namespace v2
}  // namespace gpu
}  // namespace collider
