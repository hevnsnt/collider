/**
 * bip39_pbkdf2.cu -- BIP-39 PBKDF2-HMAC-SHA512 batched kernel.
 *
 * Per-thread algorithm (one BIP-39 mnemonic per thread):
 *
 *   1. Derive ipad_block / opad_block from password (HMAC RFC 2104).
 *      If password > 128 bytes, hash it down to 64 bytes first.
 *   2. Pre-compute ipad_state + opad_state (one SHA-512 transform
 *      of each padded key block). Reused across all 2048 iters.
 *   3. U_1 = HMAC(password, salt || INT32_BE(1)) using pre-computed
 *      states; accumulator F = U_1.
 *   4. For i = 2..2048: U_i = HMAC(password, U_{i-1}); F ^= U_i.
 *   5. Write F to out_seeds[tid * 64..tid * 64 + 64].
 *
 * The PBKDF2 outer loop (over block index) is fixed at j=1 because
 * dkLen=64 = one HMAC-SHA512 output. So the kernel runs exactly
 * 2048 HMAC-SHA512 invocations per thread. Each HMAC = 2 SHA-512
 * transforms (inner + outer). With pre-computed pad states, the
 * inner-loop cost per HMAC is 2 transforms.
 */

#include "gpu/bip39_pbkdf2.cuh"

#if defined(COLLIDER_USE_CUDA)

#include "gpu/sha512_device.cuh"
#include "gpu/hmac_sha512_device.cuh"

#include <cuda_runtime.h>

namespace collider::gpu::bip39 {

__global__ void pbkdf2_kernel(
    const uint8_t* __restrict__ mnemonic_bytes,  // count * kMaxMnemonicBytes
    const uint32_t* __restrict__ mnemonic_lens,  // count
    const uint8_t* __restrict__ salt_bytes,
    uint32_t salt_len,
    uint8_t* __restrict__ out_seeds,             // count * 64
    uint32_t count) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    const uint8_t* pwd = mnemonic_bytes + tid * kMaxMnemonicBytes;
    const uint32_t pwd_len = mnemonic_lens[tid];

    // RFC 2104 derive ipad/opad blocks.
    uint8_t ipad_block[hmac_sha512::SHA512_BLOCK_BYTES];
    uint8_t opad_block[hmac_sha512::SHA512_BLOCK_BYTES];
    hmac_sha512::derive_padded_keys(pwd, pwd_len, ipad_block, opad_block);

    uint64_t ipad_state[8];
    uint64_t opad_state[8];
    hmac_sha512::precompute_states(ipad_block, opad_block,
                                   ipad_state, opad_state);

    // First iteration message: salt || INT32_BE(1) where 1 is the
    // PBKDF2 block index (we only need one block for dkLen=64).
    // Bounded to kMaxSaltBytes = 115 by the host-side guard in
    // run_pbkdf2_batch (see bip39_pbkdf2.cuh for the derivation of
    // that number). 115 + 4 counter bytes + 9 padding bytes = 128
    // = SHA512_BLOCK_BYTES, exactly fitting the single-block inner
    // hash. Pre-fix kMaxSaltBytes was 256, which overflowed
    // first_msg AND violated the HMAC contract.
    uint8_t first_msg[hmac_sha512::SHA512_BLOCK_BYTES] = {0};
    for (uint32_t i = 0; i < salt_len && i < kMaxSaltBytes; ++i) {
        first_msg[i] = salt_bytes[i];
    }
    first_msg[salt_len + 0] = 0x00;
    first_msg[salt_len + 1] = 0x00;
    first_msg[salt_len + 2] = 0x00;
    first_msg[salt_len + 3] = 0x01;  // block index = 1

    uint8_t U[hmac_sha512::SHA512_DIGEST_BYTES];
    hmac_sha512::hmac_with_states(ipad_state, opad_state,
                                  first_msg, salt_len + 4, U);

    // F = U_1
    uint8_t F[hmac_sha512::SHA512_DIGEST_BYTES];
    #pragma unroll
    for (int i = 0; i < hmac_sha512::SHA512_DIGEST_BYTES; ++i) {
        F[i] = U[i];
    }

    // U_{i+1} = HMAC(password, U_i); F ^= U_{i+1}; for i = 1..2047
    for (int iter = 1; iter < kPbkdf2Iterations; ++iter) {
        uint8_t U_next[hmac_sha512::SHA512_DIGEST_BYTES];
        hmac_sha512::hmac_with_states(ipad_state, opad_state,
                                      U, hmac_sha512::SHA512_DIGEST_BYTES,
                                      U_next);
        #pragma unroll
        for (int i = 0; i < hmac_sha512::SHA512_DIGEST_BYTES; ++i) {
            U[i] = U_next[i];
            F[i] ^= U_next[i];
        }
    }

    uint8_t* out = out_seeds + tid * kSeedBytes;
    #pragma unroll
    for (int i = 0; i < kSeedBytes; ++i) {
        out[i] = F[i];
    }
}

cudaError_t run_pbkdf2_batch(const Pbkdf2Batch& batch, cudaStream_t stream) {
    if (batch.count == 0) return cudaSuccess;
    if (batch.salt_len > kMaxSaltBytes) return cudaErrorInvalidValue;

    uint8_t* d_mnemonics = nullptr;
    uint32_t* d_lens = nullptr;
    uint8_t* d_salt = nullptr;
    uint8_t* d_seeds = nullptr;
    cudaError_t rc = cudaSuccess;

    const size_t mnemonics_bytes = batch.count * kMaxMnemonicBytes;
    const size_t lens_bytes      = batch.count * sizeof(uint32_t);
    const size_t seeds_bytes     = batch.count * kSeedBytes;

    rc = cudaMallocAsync(&d_mnemonics, mnemonics_bytes, stream);
    if (rc) goto done;
    rc = cudaMallocAsync(&d_lens, lens_bytes, stream);
    if (rc) goto done;
    rc = cudaMallocAsync(&d_salt, batch.salt_len, stream);
    if (rc) goto done;
    rc = cudaMallocAsync(&d_seeds, seeds_bytes, stream);
    if (rc) goto done;

    rc = cudaMemcpyAsync(d_mnemonics, batch.mnemonic_bytes,
                         mnemonics_bytes, cudaMemcpyHostToDevice, stream);
    if (rc) goto done;
    rc = cudaMemcpyAsync(d_lens, batch.mnemonic_lens, lens_bytes,
                         cudaMemcpyHostToDevice, stream);
    if (rc) goto done;
    rc = cudaMemcpyAsync(d_salt, batch.salt_bytes, batch.salt_len,
                         cudaMemcpyHostToDevice, stream);
    if (rc) goto done;

    {
        // Each thread runs 2048 HMAC-SHA512 invocations which is
        // sequential within the thread; high thread occupancy is the
        // only knob. Pick threads-per-block = 64 which is the lowest
        // value that still saturates a 32-warp SM without spilling
        // local memory.
        const int tpb = 64;
        const int blocks = static_cast<int>(
            (batch.count + tpb - 1) / tpb);
        pbkdf2_kernel<<<blocks, tpb, 0, stream>>>(
            d_mnemonics, d_lens, d_salt, batch.salt_len,
            d_seeds, static_cast<uint32_t>(batch.count));
        rc = cudaGetLastError();
        if (rc) goto done;
    }

    rc = cudaMemcpyAsync(batch.out_seeds, d_seeds, seeds_bytes,
                         cudaMemcpyDeviceToHost, stream);
    if (rc) goto done;
    rc = cudaStreamSynchronize(stream);

done:
    if (d_mnemonics) cudaFreeAsync(d_mnemonics, stream);
    if (d_lens)      cudaFreeAsync(d_lens, stream);
    if (d_salt)      cudaFreeAsync(d_salt, stream);
    if (d_seeds)     cudaFreeAsync(d_seeds, stream);
    return rc;
}

}  // namespace collider::gpu::bip39

#endif  // COLLIDER_USE_CUDA
