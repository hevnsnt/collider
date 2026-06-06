// SPDX-License-Identifier: BUSL-1.1
//
// C2 regression KAT: prove a funded UNCOMPRESSED-only h160 is REPORTED,
// not silently dropped, and that the $HEX[] passphrase form is decoded to
// raw bytes instead of being rejected by the UTF-8 gate.
//
// Background (adversarial review C2):
//   The fused brain-wallet kernel's multi_addr path probes the bloom under
//   three derivations (compressed P2PKH, uncompressed P2PKH, P2SH-P2WPKH)
//   and reports a match index when ANY of them hits. But the host hit
//   handler always re-derived the COMPRESSED-P2PKH hash160 (cpu::
//   compute_hash160) before verifying the hit against the UVRF. For an
//   uncompressed-only or P2SH-only target the host therefore verified the
//   WRONG hash160, the UVRF lookup missed, and the real hit was counted as
//   a bloom collision and dropped.
//
//   The fix carries a per-hit address-type byte alongside each match index
//   (kernel -> fused_brain_wallet_batch_fixed_stride_multiaddr ->
//   BatchResult.match_addr_types -> handle_bloom_hits) so the host
//   re-derives the EXACT hash160 the kernel matched on. This test pins that
//   contract end to end:
//
//   1. Insert ONLY the uncompressed-P2PKH h160 of a known passphrase into
//      the bloom (compressed + P2SH are deliberately absent).
//   2. Run the GPU multi_addr kernel via the production
//      fused_brain_wallet_batch_fixed_stride_multiaddr entry point.
//   3. Assert the hit is reported AND the kernel tags it addr_type==0
//      (uncompressed). A compressed-only re-derivation would NOT reproduce
//      the inserted h160; the test confirms the host helper
//      cpu::compute_hash160_uncompressed does.
//   4. Separately assert P2SH-P2WPKH (addr_type==2) the same way.
//   5. $HEX[] decode cases (host-only, no GPU needed).
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
#include "../src/generators/streaming_brain_wallet.hpp"

extern "C" {
    cudaError_t fused_pipeline_init(cudaStream_t stream);
    cudaError_t fused_brain_wallet_batch_fixed_stride_multiaddr(
        const uint8_t* d_passphrases,
        const uint32_t* d_lengths,
        uint32_t stride,
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

// One GPU multi_addr dispatch over a single passphrase whose ONLY bloom
// entry is the given h160. Returns the reported addr_type, or -1 if the
// hit was NOT reported (which is the C2 bug we are guarding against).
int dispatch_single(const std::string& passphrase,
                    const std::array<uint8_t, 20>& only_h160,
                    uint32_t bloom_seed, int bloom_hashes) {
    const uint32_t STRIDE = 64;
    const uint64_t bloom_bits = (1ull << 16);  // 64 Kbit, power of two
    const size_t bloom_bytes = (bloom_bits + 7) / 8;
    std::vector<uint8_t> bloom(bloom_bytes, 0);
    bloom_insert(bloom.data(), bloom_bits, bloom_hashes, bloom_seed,
                 only_h160.data());

    std::vector<uint8_t> packed(STRIDE, 0);
    std::memcpy(packed.data(), passphrase.data(), passphrase.size());
    uint32_t length = static_cast<uint32_t>(passphrase.size());

    uint8_t*  d_passphrases = nullptr;
    uint32_t* d_lengths     = nullptr;
    uint8_t*  d_bloom       = nullptr;
    uint32_t* d_match_idx   = nullptr;
    uint8_t*  d_match_types  = nullptr;
    uint32_t* d_match_cnt   = nullptr;

    cudaMalloc(&d_passphrases, packed.size());
    cudaMalloc(&d_lengths,     sizeof(uint32_t));
    cudaMalloc(&d_bloom,       bloom.size());
    cudaMalloc(&d_match_idx,   sizeof(uint32_t));
    cudaMalloc(&d_match_types, sizeof(uint8_t));
    cudaMalloc(&d_match_cnt,   sizeof(uint32_t));

    cudaMemcpy(d_passphrases, packed.data(), packed.size(),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, &length, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bloom, bloom.data(), bloom.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_match_cnt, 0, sizeof(uint32_t));
    cudaMemset(d_match_types, 0xff, sizeof(uint8_t));

    const uint64_t bloom_mask = bloom_bits - 1;  // power-of-two fast path

    cudaError_t err = fused_brain_wallet_batch_fixed_stride_multiaddr(
        d_passphrases, d_lengths, STRIDE,
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

    cudaFree(d_passphrases); cudaFree(d_lengths); cudaFree(d_bloom);
    cudaFree(d_match_idx);   cudaFree(d_match_types); cudaFree(d_match_cnt);
    return result;
}

void test_hex_decode() {
    // $HEX[] decode must produce raw bytes and bypass the UTF-8 gate.
    {
        std::string s = "$HEX[68656c6c6f]";  // "hello"
        bool ok = ::collider::generators::StreamingBrainWallet::decode_hex_token(s);
        CHECK(ok, "decode_hex_token rejected a well-formed $HEX[hello]");
        CHECK(s == "hello", "$HEX[68656c6c6f] did not decode to 'hello'");
    }
    {
        // Non-UTF-8 raw bytes (0xFF 0x00 0x80) must survive accept_candidate.
        std::string s = "$HEX[ff0080]";
        bool ok = ::collider::generators::StreamingBrainWallet::accept_candidate(s);
        CHECK(ok, "accept_candidate dropped a $HEX[] raw-byte passphrase");
        CHECK(s.size() == 3 &&
              (uint8_t)s[0] == 0xff && (uint8_t)s[1] == 0x00 &&
              (uint8_t)s[2] == 0x80,
              "$HEX[ff0080] did not decode to the 3 raw bytes");
    }
    {
        // Empty $HEX[] is a valid empty passphrase (decodes to "").
        std::string s = "$HEX[]";
        bool ok = ::collider::generators::StreamingBrainWallet::decode_hex_token(s);
        CHECK(ok, "decode_hex_token rejected empty $HEX[]");
        CHECK(s.empty(), "$HEX[] did not decode to empty string");
    }
    {
        // Malformed (odd length) is NOT a token: left intact, falls back to
        // the UTF-8 gate as an ordinary candidate.
        std::string s = "$HEX[abc]";
        bool ok = ::collider::generators::StreamingBrainWallet::decode_hex_token(s);
        CHECK(!ok, "odd-length $HEX[abc] should not decode as a token");
        CHECK(s == "$HEX[abc]", "malformed $HEX[abc] was mutated");
    }
    {
        // A bare raw-byte string that is invalid UTF-8 and NOT $HEX-wrapped
        // is still dropped (the gate is only bypassed for decoded tokens).
        std::string s;
        s.push_back(static_cast<char>(0xff));
        bool ok = ::collider::generators::StreamingBrainWallet::accept_candidate(s);
        CHECK(!ok, "lone 0xff (non-$HEX) should be rejected by the UTF-8 gate");
    }
}

}  // namespace

int main() {
    namespace cpu = ::collider::cpu;

    // ---- Host-only $HEX[] tests (always run) ----
    test_hex_decode();

    // ---- GPU end-to-end C2 reporting tests ----
    int device_count = 0;
    cudaError_t derr = cudaGetDeviceCount(&device_count);
    if (derr != cudaSuccess || device_count == 0) {
        std::fprintf(stderr,
            "No CUDA device; ran $HEX[] host tests only (%d failure(s)).\n",
            g_failures);
        if (g_failures != 0) return 1;
        return 77;  // ctest treats 77 as "skipped"
    }
    cudaSetDevice(0);
    if (fused_pipeline_init(/*stream=*/0) != cudaSuccess) {
        std::fprintf(stderr, "fused_pipeline_init failed\n");
        return 2;
    }
    cudaDeviceSynchronize();

    constexpr uint32_t BLOOM_SEED  = 0xCAFED00Du;
    constexpr int      BLOOM_HASHES = 10;

    // A fixed known passphrase. privkey = SHA256(passphrase).
    const std::string passphrase = "c2-uncompressed-only-regression";
    auto sha = cpu::SHA256::hash(
        reinterpret_cast<const uint8_t*>(passphrase.data()), passphrase.size());

    // The three host derivations (these are the helpers the runner now uses).
    const auto h_comp   = cpu::compute_hash160(sha.data());
    const auto h_uncomp = cpu::compute_hash160_uncompressed(sha.data());
    const auto h_p2sh   = cpu::compute_hash160_p2sh_p2wpkh(sha.data());

    // Sanity: the three derivations must differ (otherwise the test proves
    // nothing). Compressed vs uncompressed always differ for a real key.
    CHECK(h_comp != h_uncomp,
          "compressed and uncompressed h160 unexpectedly equal");
    CHECK(h_comp != h_p2sh,
          "compressed and P2SH h160 unexpectedly equal");

    // (1) UNCOMPRESSED-ONLY: insert only h_uncomp. Pre-fix the host dropped
    //     this hit (it re-derived h_comp, which is absent from the bloom).
    //     Post-fix the kernel reports it with addr_type==0 and the host's
    //     compute_hash160_uncompressed reproduces exactly h_uncomp.
    {
        int addr_type = dispatch_single(passphrase, h_uncomp,
                                        BLOOM_SEED, BLOOM_HASHES);
        CHECK(addr_type == 0,
              "uncompressed-only funded h160 was NOT reported as addr_type 0 "
              "(C2 regression: the hit is being dropped)");
        // The host re-derivation for addr_type 0 must reproduce the bloom entry.
        auto host_rederived = cpu::compute_hash160_uncompressed(sha.data());
        CHECK(host_rederived == h_uncomp,
              "host compute_hash160_uncompressed does not match the GPU-matched "
              "uncompressed h160");
    }

    // (2) P2SH-P2WPKH-ONLY: insert only h_p2sh; expect addr_type==2.
    {
        int addr_type = dispatch_single(passphrase, h_p2sh,
                                        BLOOM_SEED, BLOOM_HASHES);
        CHECK(addr_type == 2,
              "P2SH-P2WPKH-only funded h160 was NOT reported as addr_type 2");
        auto host_rederived = cpu::compute_hash160_p2sh_p2wpkh(sha.data());
        CHECK(host_rederived == h_p2sh,
              "host compute_hash160_p2sh_p2wpkh does not match the GPU-matched "
              "P2SH h160");
    }

    // (3) COMPRESSED control: insert only h_comp; expect addr_type==1 (the
    //     default path, which always worked).
    {
        int addr_type = dispatch_single(passphrase, h_comp,
                                        BLOOM_SEED, BLOOM_HASHES);
        CHECK(addr_type == 1,
              "compressed funded h160 was NOT reported as addr_type 1");
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed.\n", g_failures);
        return 1;
    }
    std::printf("All C2 multi-addr reporting + $HEX[] decode KATs passed.\n");
    return 0;
}
