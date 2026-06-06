// SPDX-License-Identifier: BUSL-1.1
//
// R1 regression KAT: prove the OFFSETS-BASED wordlist / CPU-rules path reports
// a funded UNCOMPRESSED-only h160 instead of silently dropping it.
//
// Background (adversarial review C2, finished by R1):
//   The initial C2 fix added the multi-address probe + per-hit address-type
//   plumbing ONLY to the fixed-stride GPU-rules entry point
//   (fused_brain_wallet_batch_fixed_stride_multiaddr). The wordlist /
//   CPU-rules path runs through a DIFFERENT kernel: the offsets-based
//   brain_wallet_fused_kernel behind fused_brain_wallet_batch, which probed
//   COMPRESSED P2PKH only and never wrote match_addr_types. So a funded
//   uncompressed-only or P2SH-P2WPKH wallet discovered through an ordinary
//   wordlist scan was re-derived host-side as compressed, missed the UVRF
//   lookup, and was counted as a bloom collision and dropped.
//
//   R1 gives the offsets-based kernel + fused_brain_wallet_batch the SAME
//   multi_addr + match_addr_types plumbing via the new
//   fused_brain_wallet_batch_multiaddr entry point. This test pins that
//   contract on the offsets path that the existing C2 KAT
//   (test_brainwallet_multiaddr_report_kat.cu) did NOT cover: it only ran the
//   fixed-stride entry.
//
//   1. Insert ONLY the uncompressed-P2PKH h160 of a known passphrase into the
//      bloom (compressed + P2SH deliberately absent).
//   2. Run the OFFSETS-BASED multi_addr kernel via the production
//      fused_brain_wallet_batch_multiaddr entry point (the wordlist path).
//   3. Assert the hit is reported AND tagged addr_type==0 (uncompressed).
//   4. Separately assert P2SH-P2WPKH (addr_type==2) and the compressed
//      control (addr_type==1).
//
// Exit codes: 0 = pass, 1 = fail, 77 = skipped (no CUDA device).

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../src/core/crypto_cpu.hpp"
#include "../src/tools/utxo_bloom_builder.hpp"  // murmurhash3_128

extern "C" {
    cudaError_t fused_pipeline_init(cudaStream_t stream);
    cudaError_t fused_brain_wallet_batch_multiaddr(
        const uint8_t* d_passphrases,
        const uint32_t* d_offsets,
        const uint32_t* d_lengths,
        const uint8_t* d_bloom_filter,
        uint64_t bloom_bits,
        uint64_t bloom_mask,
        int bloom_hashes,
        uint32_t bloom_seed,
        uint32_t* d_match_indices,
        uint8_t* d_match_addr_types,
        uint32_t* d_match_count,
        uint8_t* d_private_keys,
        size_t count,
        cudaStream_t stream,
        uint8_t multi_addr);
}

namespace {

int g_failures = 0;

#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::fprintf(stderr, "FAIL: %s\n", (msg));                     \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

void bloom_insert(uint8_t* bloom, uint64_t bloom_bits, int num_hashes,
                  uint32_t seed, const uint8_t* h160) {
    auto [h1, h2] = ::collider::utxo::murmurhash3_128(h160, 20, seed);
    for (int i = 0; i < num_hashes; i++) {
        uint64_t idx = (h1 + (uint64_t)i * h2) % bloom_bits;
        bloom[idx / 8] |= (uint8_t)(1u << (idx % 8));
    }
}

// One GPU multi_addr dispatch over a single passphrase through the
// OFFSETS-BASED entry (the wordlist path) whose ONLY bloom entry is the given
// h160. Returns the reported addr_type, or -1 if the hit was NOT reported
// (the R1 bug we are guarding against).
int dispatch_single_offsets(const std::string& passphrase,
                            const std::array<uint8_t, 20>& only_h160,
                            uint32_t bloom_seed, int bloom_hashes) {
    const uint64_t bloom_bits = (1ull << 16);  // 64 Kbit, power of two
    const size_t bloom_bytes = (bloom_bits + 7) / 8;
    std::vector<uint8_t> bloom(bloom_bytes, 0);
    bloom_insert(bloom.data(), bloom_bits, bloom_hashes, bloom_seed,
                 only_h160.data());

    // Offsets-based packing: contiguous passphrase bytes + per-candidate
    // offset + length (this is exactly what process_batch_single_gpu builds).
    std::vector<uint8_t> packed(passphrase.begin(), passphrase.end());
    if (packed.empty()) packed.push_back(0);  // never pass a null device ptr
    uint32_t offset = 0;
    uint32_t length = static_cast<uint32_t>(passphrase.size());

    uint8_t*  d_passphrases = nullptr;
    uint32_t* d_offsets     = nullptr;
    uint32_t* d_lengths     = nullptr;
    uint8_t*  d_bloom       = nullptr;
    uint32_t* d_match_idx   = nullptr;
    uint8_t*  d_match_types  = nullptr;
    uint32_t* d_match_cnt   = nullptr;

    cudaMalloc(&d_passphrases, packed.size());
    cudaMalloc(&d_offsets,     sizeof(uint32_t));
    cudaMalloc(&d_lengths,     sizeof(uint32_t));
    cudaMalloc(&d_bloom,       bloom.size());
    cudaMalloc(&d_match_idx,   sizeof(uint32_t));
    cudaMalloc(&d_match_types, sizeof(uint8_t));
    cudaMalloc(&d_match_cnt,   sizeof(uint32_t));

    cudaMemcpy(d_passphrases, packed.data(), packed.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, &offset, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, &length, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bloom, bloom.data(), bloom.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_match_cnt, 0, sizeof(uint32_t));
    cudaMemset(d_match_types, 0xff, sizeof(uint8_t));

    const uint64_t bloom_mask = bloom_bits - 1;  // power-of-two fast path

    cudaError_t err = fused_brain_wallet_batch_multiaddr(
        d_passphrases, d_offsets, d_lengths,
        d_bloom, bloom_bits, bloom_mask, bloom_hashes, bloom_seed,
        d_match_idx, d_match_types, d_match_cnt,
        /*d_private_keys=*/nullptr,
        /*count=*/1, /*stream=*/0, /*multi_addr=*/uint8_t{1});

    int result = -1;
    if (err == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess &&
        cudaGetLastError() == cudaSuccess) {
        uint32_t match_count = 0;
        cudaMemcpy(&match_count, d_match_cnt, sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        if (match_count == 1) {
            uint8_t addr_type = 0xff;
            cudaMemcpy(&addr_type, d_match_types, sizeof(uint8_t),
                       cudaMemcpyDeviceToHost);
            result = static_cast<int>(addr_type);
        }
    }

    cudaFree(d_passphrases); cudaFree(d_offsets); cudaFree(d_lengths);
    cudaFree(d_bloom);       cudaFree(d_match_idx);
    cudaFree(d_match_types); cudaFree(d_match_cnt);
    return result;
}

}  // namespace

int main() {
    namespace cpu = ::collider::cpu;

    int device_count = 0;
    cudaError_t derr = cudaGetDeviceCount(&device_count);
    if (derr != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "No CUDA device; skipping R1 wordlist KAT.\n");
        return 77;  // ctest treats 77 as "skipped"
    }
    cudaSetDevice(0);
    if (fused_pipeline_init(/*stream=*/0) != cudaSuccess) {
        std::fprintf(stderr, "fused_pipeline_init failed\n");
        return 2;
    }
    cudaDeviceSynchronize();

    constexpr uint32_t BLOOM_SEED   = 0xCAFED00Du;
    constexpr int      BLOOM_HASHES = 10;

    // A fixed known passphrase. privkey = SHA256(passphrase).
    const std::string passphrase = "r1-wordlist-uncompressed-only-regression";
    auto sha = cpu::SHA256::hash(
        reinterpret_cast<const uint8_t*>(passphrase.data()), passphrase.size());

    const auto h_comp   = cpu::compute_hash160(sha.data());
    const auto h_uncomp = cpu::compute_hash160_uncompressed(sha.data());
    const auto h_p2sh   = cpu::compute_hash160_p2sh_p2wpkh(sha.data());

    CHECK(h_comp != h_uncomp,
          "compressed and uncompressed h160 unexpectedly equal");
    CHECK(h_comp != h_p2sh,
          "compressed and P2SH h160 unexpectedly equal");

    // (1) UNCOMPRESSED-ONLY through the OFFSETS path: this is the path the
    //     existing C2 KAT did NOT cover. Pre-R1 the offsets kernel never
    //     wrote match_addr_types and probed compressed only, so this hit was
    //     dropped. Post-R1 it is reported with addr_type==0.
    {
        int addr_type = dispatch_single_offsets(passphrase, h_uncomp,
                                                BLOOM_SEED, BLOOM_HASHES);
        CHECK(addr_type == 0,
              "uncompressed-only funded h160 was NOT reported as addr_type 0 "
              "via the offsets / wordlist path (R1 regression: hit dropped)");
    }

    // (2) P2SH-P2WPKH-ONLY through the offsets path: expect addr_type==2.
    {
        int addr_type = dispatch_single_offsets(passphrase, h_p2sh,
                                                BLOOM_SEED, BLOOM_HASHES);
        CHECK(addr_type == 2,
              "P2SH-P2WPKH-only funded h160 was NOT reported as addr_type 2 "
              "via the offsets / wordlist path");
    }

    // (3) COMPRESSED control: insert only h_comp; expect addr_type==1 (the
    //     path that always worked).
    {
        int addr_type = dispatch_single_offsets(passphrase, h_comp,
                                                BLOOM_SEED, BLOOM_HASHES);
        CHECK(addr_type == 1,
              "compressed funded h160 was NOT reported as addr_type 1 via the "
              "offsets / wordlist path");
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", g_failures);
        return 1;
    }
    std::printf("All R1 wordlist-path multi-addr reporting KATs passed.\n");
    return 0;
}
