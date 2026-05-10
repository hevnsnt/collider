/**
 * Multi-address derivation + bloom probe GPU kernel (Phase 4, v1.4.0).
 *
 * Takes a batch of public keys (X, Y in big-endian 32-byte form, one
 * point per thread) and runs the five address derivations enabled in
 * `addr_mask`, probing each h160 / x-only output against the supplied
 * bloom filter.
 *
 * Pipeline split: EC multiply (priv -> pub) lives in src/gpu/secp256k1.cu's
 * ec_mul_batch kernels. The orchestrator calls them first, hands the
 * (X, Y) buffers to this kernel, and reads back V2MatchRecord entries.
 *
 * Address algorithms mirror src/gpu/v2/address_derive_cpu.hpp byte-for-byte.
 */

#include "brain_wallet_v2.hpp"
#include "device_hashes.cuh"

#include <cuda_runtime.h>

// Forward declaration of the production probe defined in
// src/gpu/h160_bloom_filter.cu. It lives in `collider::gpu`, so the
// declaration must match that namespace exactly (mangled symbol).
namespace collider {
namespace gpu {
extern __device__ bool bloom_check_h160(
    const uint8_t* bloom_data,
    uint64_t num_bits,
    uint32_t num_hashes,
    uint32_t seed,
    const uint8_t* h160);
}  // namespace gpu
}  // namespace collider

namespace collider {
namespace gpu {
namespace v2 {

namespace {

// ---------------------------------------------------------------------------
// Bloom probe -- delegates to the production MurmurHash3 implementation
// in src/gpu/h160_bloom_filter.cu so v2 reads the SAME bloom files the
// existing brain_wallet pipeline writes. Cross-TU __device__ calls work
// under separable compilation.
//
// REQUIRES: input is exactly 20 bytes (h160). The production hash hardcodes
// length=20. P2TR-BIP86's 32-byte tweaked output key cannot be probed by
// this path (the tweak alone is not a valid h160; producing the address
// requires an EC tweak-add the kernel doesn't currently do).
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool bloom_probe_h160(
    const uint8_t h160[20],
    const uint8_t* bloom, uint64_t bits, int hashes, uint32_t seed)
{
    if (!bloom || bits == 0) return false;
    return ::collider::gpu::bloom_check_h160(
        bloom, bits, (uint32_t)hashes, seed, h160);
}

// ---------------------------------------------------------------------------
// Address derivations.
// pub_x and pub_y are 32-byte big-endian.
// ---------------------------------------------------------------------------

__device__ static void derive_p2pkh_uncompressed(
    const uint8_t pub_x[32], const uint8_t pub_y[32], uint8_t h160_out[20])
{
    uint8_t buf[65];
    buf[0] = 0x04;
    for (int i = 0; i < 32; ++i) buf[1 + i]      = pub_x[i];
    for (int i = 0; i < 32; ++i) buf[33 + i]     = pub_y[i];
    device::hash160(buf, 65, h160_out);
}

__device__ static void compressed_pubkey(
    const uint8_t pub_x[32], const uint8_t pub_y[32], uint8_t out[33])
{
    out[0] = (pub_y[31] & 1u) ? 0x03 : 0x02;
    for (int i = 0; i < 32; ++i) out[1 + i] = pub_x[i];
}

__device__ static void derive_p2pkh_compressed(
    const uint8_t pub_x[32], const uint8_t pub_y[32], uint8_t h160_out[20])
{
    uint8_t comp[33];
    compressed_pubkey(pub_x, pub_y, comp);
    device::hash160(comp, 33, h160_out);
}

__device__ static void derive_p2wpkh_v0(
    const uint8_t pub_x[32], const uint8_t pub_y[32], uint8_t h160_out[20])
{
    derive_p2pkh_compressed(pub_x, pub_y, h160_out);
}

__device__ static void derive_p2sh_p2wpkh(
    const uint8_t pub_x[32], const uint8_t pub_y[32], uint8_t h160_out[20])
{
    uint8_t inner[20];
    derive_p2pkh_compressed(pub_x, pub_y, inner);
    uint8_t redeem[22];
    redeem[0] = 0x00;
    redeem[1] = 0x14;
    for (int i = 0; i < 20; ++i) redeem[2 + i] = inner[i];
    device::hash160(redeem, 22, h160_out);
}

// derive_p2tr_bip86_tweak removed: BIP-86 bloom probe is deferred until
// the kernel grows EC tweak-add support. The host-side reference at
// src/gpu/v2/address_derive_cpu.hpp::cpu_bip86_tap_tweak retains the
// algorithm for KAT testing.

}  // namespace

// ---------------------------------------------------------------------------
// Kernel: per-thread, derive every enabled address type, probe bloom.
// ---------------------------------------------------------------------------

// pub_xy is a packed array of (X, Y) per point, 64 bytes per point,
// matching the `ECPointAffine` layout produced by secp256k1_batch_mul.
//   pub_xy[idx*64 .. idx*64+32)  = X (big-endian)
//   pub_xy[idx*64+32 .. idx*64+64) = Y (big-endian)
__global__ void v2_kernel_multi_address(
    const uint8_t* __restrict__ pub_xy,
    size_t count,
    uint32_t addr_mask,
    const uint8_t* __restrict__ bloom,
    uint64_t bloom_bits,
    int      bloom_hashes,
    uint32_t bloom_seed,
    V2MatchRecord* __restrict__ matches,
    uint32_t* __restrict__ match_count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* pub_x = pub_xy + (idx * 64);
    const uint8_t* pub_y = pub_xy + (idx * 64) + 32;

    auto emit_h160 = [&](uint8_t addr_type, const uint8_t h[20]) {
        if (!bloom_probe_h160(h, bloom, bloom_bits, bloom_hashes, bloom_seed))
            return;
        uint32_t slot = atomicAdd(match_count, 1u);
        if (slot < V2_MAX_MATCHES_PER_BATCH) {
            V2MatchRecord rec{};
            rec.pp_idx    = (uint32_t)idx;
            rec.addr_type = addr_type;
            rec.kind      = (uint8_t)V2MatchRecord::Kind::ADDR_BLOOM_HIT;
            matches[slot] = rec;
        }
    };

    if (addr_mask & addr_bit(AddressType::P2PKH_UNCOMPRESSED)) {
        uint8_t h[20];
        derive_p2pkh_uncompressed(pub_x, pub_y, h);
        emit_h160((uint8_t)AddressType::P2PKH_UNCOMPRESSED, h);
    }
    if (addr_mask & addr_bit(AddressType::P2PKH_COMPRESSED)) {
        uint8_t h[20];
        derive_p2pkh_compressed(pub_x, pub_y, h);
        emit_h160((uint8_t)AddressType::P2PKH_COMPRESSED, h);
    }
    if (addr_mask & addr_bit(AddressType::P2SH_P2WPKH)) {
        uint8_t h[20];
        derive_p2sh_p2wpkh(pub_x, pub_y, h);
        emit_h160((uint8_t)AddressType::P2SH_P2WPKH, h);
    }
    if (addr_mask & addr_bit(AddressType::P2WPKH_V0)) {
        uint8_t h[20];
        derive_p2wpkh_v0(pub_x, pub_y, h);
        emit_h160((uint8_t)AddressType::P2WPKH_V0, h);
    }
    // P2TR-BIP86: the production bloom is keyed on 20-byte h160 only.
    // Computing the BIP-86 tweaked output key (32-byte x-only) and from
    // there a 20-byte address fingerprint requires an EC tweak-add that
    // this kernel does not currently do. P2TR is intentionally NOT probed
    // against the bloom here; it remains a host-side follow-up.
}

cudaError_t v2_multi_address_check(
    const uint8_t* d_pub_xy, size_t count, uint32_t addr_mask,
    const uint8_t* d_bloom, uint64_t bloom_bits, int bloom_hashes,
    uint32_t bloom_seed,
    V2MatchRecord* d_matches, uint32_t* d_match_count,
    cudaStream_t stream)
{
    if (count == 0 || addr_mask == 0) return cudaErrorInvalidValue;
    constexpr int BLOCK = 256;
    int blocks = (int)((count + BLOCK - 1) / BLOCK);
    v2_kernel_multi_address<<<blocks, BLOCK, 0, stream>>>(
        d_pub_xy, count, addr_mask,
        d_bloom, bloom_bits, bloom_hashes, bloom_seed,
        d_matches, d_match_count);
    return cudaGetLastError();
}

}  // namespace v2
}  // namespace gpu
}  // namespace collider
