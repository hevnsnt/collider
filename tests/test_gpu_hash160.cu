/**
 * test_gpu_hash160 — Bloom oracle test for the GPU brain-wallet pipeline.
 *
 * Wave 0 of the 2026-05-04 review. This is the HEADLINE test that proves
 * (or disproves) C-CRIT-1 (SHA256 -> scalar byte-swap) end-to-end through
 * the production API.
 *
 * STRATEGY (no production code modification needed):
 *   1. Generate N test passphrases on host.
 *   2. For each passphrase, compute the EXPECTED hash160 via crypto_cpu.hpp
 *      (the canonical CPU reference: SHA256 -> ec_mul on G -> SHA256 -> RIPEMD160).
 *   3. Build a tiny bloom filter populated with all N expected hash160s, using
 *      the SAME bit-slicing scheme as fused_pipeline.cu's bloom_check_inline.
 *   4. Call fused_brain_wallet_batch_fixed_stride with the N passphrases and
 *      the oracle bloom.
 *   5. PASS iff match_count == N (every passphrase finds its expected hash160).
 *
 * EXPECTED FAILURE on current code: match_count is ~0 because the GPU
 * computes hash160 for the SHA256-of-passphrase with its 32-bit words reversed
 * (C-CRIT-1). After Wave 1 fix #1.1, all N passphrases should match.
 */

#include <cuda_runtime.h>
#include "../src/core/crypto_cpu.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <array>
#include <string>

// Public host API from src/gpu/brain_wallet_gpu.hpp / fused_pipeline.cu
extern "C" {
    cudaError_t fused_pipeline_init(cudaStream_t stream);
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
    );
}

// =============================================================================
// Host bloom helpers — MUST match bloom_check_inline in src/gpu/fused_pipeline.cu
// (h1 = first 8 bytes LE, h2 = next 8 bytes LE, double-hash mod num_bits)
// =============================================================================

static void bloom_h1_h2(const uint8_t* h160, uint64_t& h1, uint64_t& h2) {
    h1 = 0; h2 = 0;
    for (int i = 0; i < 8; i++) {
        h1 |= (uint64_t)h160[i]     << (i * 8);
        h2 |= (uint64_t)h160[8 + i] << (i * 8);
    }
}

static void bloom_insert(uint8_t* bloom, uint64_t bloom_bits, int num_hashes,
                         const uint8_t* h160) {
    uint64_t h1, h2;
    bloom_h1_h2(h160, h1, h2);
    for (int i = 0; i < num_hashes; i++) {
        uint64_t idx = (h1 + (uint64_t)i * h2) % bloom_bits;
        bloom[idx / 8] |= (uint8_t)(1u << (idx % 8));
    }
}

static bool bloom_probe_host(const uint8_t* bloom, uint64_t bloom_bits, int num_hashes,
                             const uint8_t* h160) {
    uint64_t h1, h2;
    bloom_h1_h2(h160, h1, h2);
    for (int i = 0; i < num_hashes; i++) {
        uint64_t idx = (h1 + (uint64_t)i * h2) % bloom_bits;
        if (!(bloom[idx / 8] & (1u << (idx % 8)))) return false;
    }
    return true;
}

// =============================================================================
// Test passphrases
// =============================================================================

static const std::vector<std::string> TEST_PASSPHRASES = {
    "abc",
    "satoshi",
    "password",
    "123456",
    "Bitcoin",
    "puzzle",
    "satoshi nakamoto",
    "correct horse battery staple",       // 28 bytes
    "the quick brown fox jumps over the lazy dog",  // 43 bytes
    "all your bitcoin are belong to us",  // 33 bytes
    std::string(55, 'a'),                 // 55 bytes (single SHA256 block boundary)
    std::string(56, 'b'),                 // 56 bytes (forces multi-block; A-CRIT-1)
    std::string(64, 'c'),                 // 64 bytes (one full block)
    std::string(89, 'd'),                 // 89 bytes (multi-block)
    "1",                                  // 1 byte
    "",                                   // 0 bytes (edge case)
};

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA devices available: %s\n", cudaGetErrorString(err));
        return 77;  // CTest skip
    }
    cudaSetDevice(0);

    err = fused_pipeline_init(/*stream*/0);
    if (err != cudaSuccess) {
        fprintf(stderr, "fused_pipeline_init failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    const size_t N = TEST_PASSPHRASES.size();
    const uint32_t STRIDE = 128;  // bytes per slot, > longest passphrase

    // Step 1+2: build expected hash160 for each passphrase via the CPU reference.
    std::vector<std::array<uint8_t, 20>> expected_h160(N);
    for (size_t i = 0; i < N; i++) {
        const std::string& pp = TEST_PASSPHRASES[i];

        // Step A: SHA256(passphrase) -> 32-byte private key
        auto privkey_arr = collider::cpu::SHA256::hash(
            reinterpret_cast<const uint8_t*>(pp.data()), pp.size());

        // Step B: compute_hash160(privkey) does the rest of the chain on CPU
        expected_h160[i] = collider::cpu::compute_hash160(privkey_arr.data());
    }

    // Step 3: build a bloom filter populated with all expected hash160s.
    // 8192 bits = 1 KB; FP rate negligible at N=16.
    constexpr uint64_t BLOOM_BITS = 8192;
    constexpr int BLOOM_HASHES = 8;
    std::vector<uint8_t> bloom(BLOOM_BITS / 8, 0);
    for (size_t i = 0; i < N; i++) {
        bloom_insert(bloom.data(), BLOOM_BITS, BLOOM_HASHES, expected_h160[i].data());
    }

    // Sanity: every expected h160 should now probe true on host
    for (size_t i = 0; i < N; i++) {
        if (!bloom_probe_host(bloom.data(), BLOOM_BITS, BLOOM_HASHES, expected_h160[i].data())) {
            fprintf(stderr, "INTERNAL ERROR: host bloom probe failed for entry %zu\n", i);
            return 2;
        }
    }

    // Pack passphrases into a fixed-stride buffer
    std::vector<uint8_t> packed(N * STRIDE, 0);
    std::vector<uint32_t> lengths(N, 0);
    for (size_t i = 0; i < N; i++) {
        const std::string& pp = TEST_PASSPHRASES[i];
        if (pp.size() > STRIDE) {
            fprintf(stderr, "Passphrase %zu exceeds stride\n", i);
            return 2;
        }
        memcpy(&packed[i * STRIDE], pp.data(), pp.size());
        lengths[i] = (uint32_t)pp.size();
    }

    // Allocate device buffers
    uint8_t*  d_passphrases = nullptr;
    uint32_t* d_lengths     = nullptr;
    uint8_t*  d_bloom       = nullptr;
    uint32_t* d_match_idx   = nullptr;
    uint32_t* d_match_cnt   = nullptr;

    cudaMalloc(&d_passphrases, packed.size());
    cudaMalloc(&d_lengths,     N * sizeof(uint32_t));
    cudaMalloc(&d_bloom,       bloom.size());
    cudaMalloc(&d_match_idx,   1024 * sizeof(uint32_t));
    cudaMalloc(&d_match_cnt,   sizeof(uint32_t));

    cudaMemcpy(d_passphrases, packed.data(),  packed.size(),         cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths,     lengths.data(), N * sizeof(uint32_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_bloom,       bloom.data(),   bloom.size(),          cudaMemcpyHostToDevice);
    cudaMemset(d_match_cnt, 0, sizeof(uint32_t));

    err = fused_brain_wallet_batch_fixed_stride(
        d_passphrases, d_lengths, STRIDE,
        d_bloom, BLOOM_BITS, BLOOM_HASHES,
        d_match_idx, d_match_cnt,
        /*d_private_keys=*/nullptr,
        N,
        /*stream=*/0
    );
    if (err != cudaSuccess) {
        fprintf(stderr, "fused_brain_wallet_batch_fixed_stride failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    uint32_t match_count = 0;
    std::vector<uint32_t> match_indices(1024, 0);
    cudaMemcpy(&match_count, d_match_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(match_indices.data(), d_match_idx, 1024 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_passphrases);
    cudaFree(d_lengths);
    cudaFree(d_bloom);
    cudaFree(d_match_idx);
    cudaFree(d_match_cnt);

    // Report
    printf("=== GPU brain-wallet hash160 oracle test ===\n");
    printf("Passphrases: %zu\n", N);
    printf("Matches:     %u (expected %zu)\n", match_count, N);

    if ((size_t)match_count == N) {
        printf("PASS: every passphrase produced the expected hash160 on the GPU.\n");
        return 0;
    } else {
        printf("FAIL: GPU produced wrong hash160 for %zu passphrases.\n",
               N - match_count);
        printf("This is expected on current code per Track C C-CRIT-1\n");
        printf("(SHA256 -> scalar byte-swap in fused_pipeline.cu around lines 1090, 1175,\n");
        printf(" and analogous sites in mega_fused_kernel.cu).\n");
        printf("After Wave 1 fix #1.1, all %zu should match.\n", N);

        // Show which matched and which didn't (best-effort: indices that landed)
        std::vector<bool> matched(N, false);
        for (uint32_t i = 0; i < match_count && i < 1024; i++) {
            if (match_indices[i] < N) matched[match_indices[i]] = true;
        }
        for (size_t i = 0; i < N; i++) {
            const std::string& pp = TEST_PASSPHRASES[i];
            std::string preview = pp.size() > 40 ? pp.substr(0, 37) + "..." : pp;
            printf("  [%zu] %s  \"%s\" (%zu bytes)\n",
                   i, matched[i] ? "MATCH" : "MISS ", preview.c_str(), pp.size());
        }
        return 1;
    }
}
