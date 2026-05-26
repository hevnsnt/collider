/**
 * bip39_pbkdf2.cuh -- host-side API for BIP-39 PBKDF2-HMAC-SHA512
 * batched on CUDA.
 *
 * BIP-39 spec: seed = PBKDF2(password=mnemonic_utf8,
 *                            salt="mnemonic" + passphrase_utf8,
 *                            iterations=2048,
 *                            dkLen=64,
 *                            prf=HMAC-SHA512)
 *
 * The output is exactly 64 bytes (= one HMAC-SHA512 digest), so the
 * PBKDF2 outer loop runs ONCE: only the inner 2048-iteration HMAC
 * chain matters for throughput. Each device thread does one entire
 * PBKDF2 derivation (one mnemonic) so a batch of B mnemonics
 * dispatches B threads.
 *
 * Per-thread inputs (fixed layout for coalesced loads):
 *   - 256-byte password slot (mnemonic UTF-8, NUL-padded). 256 is
 *     comfortably above the worst-case 24-word English mnemonic
 *     (108 chars + 23 spaces = 131 bytes); padded so the device
 *     code doesn't need a per-thread length lookup in shared mem.
 *   - one shared salt (256 bytes max, "mnemonic" + passphrase).
 *   - one shared iteration count (2048).
 *
 * Per-thread output:
 *   - 64-byte seed (big-endian).
 */

#pragma once

#include <cstddef>
#include <cstdint>

#if defined(COLLIDER_USE_CUDA)
#  include <cuda_runtime.h>
#endif

namespace collider::gpu::bip39 {

constexpr int kMaxMnemonicBytes = 256;  // per-thread password slot
// Salt length is bounded by SHA-512 single-block packing inside
// hmac_sha512_device.cuh's HMAC inner-hash path: the kernel packs
// (salt || 4-byte counter || 0x80 padding || 64-bit length) into a
// single 128-byte block. With 4 bytes of counter + 9 bytes of
// padding/length, the salt itself is bounded by 115 bytes. Going
// past this used to overflow first_msg on stack AND violate the
// HMAC contract (silent wrong-seed output). The host-side
// run_pbkdf2_batch rejects len > kMaxSaltBytes; tightened to 115
// to match the kernel's true capacity. BIP-39 salts are
// "mnemonic" + passphrase, typically <50 bytes; 115 leaves plenty
// of room for a long human-chosen passphrase but stops well short
// of the kernel's stack-overflow cliff.
constexpr int kMaxSaltBytes     = 115;
constexpr int kPbkdf2Iterations = 2048;  // BIP-39 fixed
constexpr int kSeedBytes        = 64;

// Host-side request descriptor. The host copies a packed
// kMaxMnemonicBytes-per-thread password buffer + a kMaxSaltBytes
// salt + a per-thread password_len array into device memory, dispatches
// the kernel, and copies back count * kSeedBytes seeds.
struct Pbkdf2Batch {
    const uint8_t* mnemonic_bytes;   // host pointer; size = count * kMaxMnemonicBytes
    const uint32_t* mnemonic_lens;   // host pointer; size = count
    size_t          count;
    const uint8_t*  salt_bytes;      // host pointer; size = salt_len
    uint32_t        salt_len;
    uint8_t*        out_seeds;       // host pointer; size = count * kSeedBytes
};

#if defined(COLLIDER_USE_CUDA)

// Run one batch through the PBKDF2 kernel synchronously. Returns 0
// on success, non-zero CUDA error code otherwise. The kernel is
// launched on the supplied stream; the function cudaStreamSynchronizes
// before returning so the caller can read out_seeds immediately.
cudaError_t run_pbkdf2_batch(const Pbkdf2Batch& batch, cudaStream_t stream);

#else
// Non-CUDA build: not implemented. Caller must fall back to CPU
// PBKDF2 (bip32::mnemonic_to_seed via OpenSSL).
inline int run_pbkdf2_batch(const Pbkdf2Batch&, void*) { return -1; }
#endif

}  // namespace collider::gpu::bip39
