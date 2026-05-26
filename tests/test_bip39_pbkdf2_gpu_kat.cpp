/**
 * test_bip39_pbkdf2_gpu_kat -- pin GPU PBKDF2 output against CPU.
 *
 * For every BIP-39 mnemonic in the trezor vector set, run the GPU
 * batched PBKDF2 kernel and assert the 64-byte seed matches the
 * canonical CPU output (OpenSSL PKCS5_PBKDF2_HMAC via
 * bip32::mnemonic_to_seed).
 *
 * KAT failure here means the GPU kernel diverged from the spec, and
 * the BIP scan GPU path will silently produce wrong seeds (and miss
 * every real hit). The check is byte-for-byte identical, no slack.
 *
 * Skipped (ctest exit 77) when CUDA is unavailable. Built only when
 * COLLIDER_USE_CUDA + COLLIDER_PRO are both on.
 */

#include "../src/core/bip32.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef COLLIDER_USE_CUDA
int main() {
    std::printf("[SKIP] PBKDF2 GPU KAT needs CUDA\n");
    return 77;
}
#else

#include "../src/gpu/bip39_pbkdf2.cuh"
#include <cuda_runtime.h>

namespace {

int g_pass = 0;
int g_fail = 0;

std::string hex_lower(const uint8_t* p, size_t n) {
    static const char* h = "0123456789abcdef";
    std::string out;
    out.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(h[(p[i] >> 4) & 0xF]);
        out.push_back(h[(p[i] >> 0) & 0xF]);
    }
    return out;
}

struct Vec { const char* mnemonic; const char* passphrase; };

// A reduced trezor vector set (mnemonic + passphrase) sufficient to
// catch every kernel-correctness regression we care about: PBKDF2
// round count, ipad/opad derivation, salt prefix, multi-block
// password (24-word mnemonic exceeds 128 bytes). The CPU reference
// (bip32::mnemonic_to_seed) computes the expected seed at runtime
// so this test does not need static expected hashes baked in -- the
// CPU KAT already pins those.
const Vec kVecs[] = {
    {"abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about", "TREZOR"},
    {"legal winner thank year wave sausage worth useful legal winner thank yellow", "TREZOR"},
    {"letter advice cage absurd amount doctor acoustic avoid letter advice cage above", "TREZOR"},
    {"zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong", "TREZOR"},
    {"hamster diagram private dinosaur mention crumble shadow turtle dragon enrich enhance pause", ""},
    {"scheme spot photo card baby mountain device kick cradle pact join borrow", ""},
    {"horn tenant knee talent sponsor spell gate clip pulse soap slush warm silver nephew swap uncle crack brave", "TREZOR"},
    {"void come effort suffer camp survey warrant move gas wait pen kid silver disagree purpose enroll question stove pulse blade thunder bind chronic mistake", "TREZOR"},
};

}  // namespace

int main() {
    std::printf("=== test_bip39_pbkdf2_gpu_kat ===\n");

    constexpr size_t kBatch = sizeof(kVecs) / sizeof(kVecs[0]);
    using namespace ::collider::gpu::bip39;

    std::vector<uint8_t>  mnemonics(kBatch * kMaxMnemonicBytes, 0);
    std::vector<uint32_t> lens(kBatch);

    // Loop over every CUDA device. Pre-fix this test ran on device 0
    // only; a 2-GPU box with a SM-specific kernel bug on GPU 1 (e.g.
    // register spill divergence on a different arch) would pass the
    // test but silently produce wrong seeds on every worker pinned to
    // GPU 1 in the BIP scan worker pool. Running each vector on every
    // device closes that gap.
    int device_count = 0;
    cudaError_t cnt_rc = cudaGetDeviceCount(&device_count);
    if (cnt_rc != cudaSuccess || device_count == 0) {
        std::printf("[SKIP] no CUDA device visible (rc=%d count=%d)\n",
                    static_cast<int>(cnt_rc), device_count);
        return 77;
    }

    for (int dev = 0; dev < device_count; ++dev) {
        std::printf("--- device %d/%d ---\n", dev, device_count);
        if (cudaSetDevice(dev) != cudaSuccess) {
            std::fprintf(stderr, "[!] cudaSetDevice(%d) failed\n", dev);
            ++g_fail;
            continue;
        }

    // All vectors share the same passphrase set; build the salt
    // per-vector and dispatch one-at-a-time to keep the test simple
    // (the production code batches; this test pins correctness, not
    // throughput).
    cudaStream_t stream = nullptr;
    cudaError_t rc = cudaStreamCreate(&stream);
    if (rc) {
        std::fprintf(stderr, "[!] cudaStreamCreate failed on dev %d: %d\n",
                     dev, rc);
        ++g_fail;
        continue;
    }

    for (size_t i = 0; i < kBatch; ++i) {
        const Vec& v = kVecs[i];
        std::string mn = v.mnemonic;
        std::string pp = v.passphrase;
        std::string salt = std::string("mnemonic") + pp;

        std::vector<uint8_t> mn_buf(kMaxMnemonicBytes, 0);
        std::memcpy(mn_buf.data(), mn.data(), mn.size());
        std::vector<uint32_t> ln_buf{static_cast<uint32_t>(mn.size())};

        std::vector<uint8_t> seed_gpu(kSeedBytes);

        Pbkdf2Batch batch{};
        batch.mnemonic_bytes = mn_buf.data();
        batch.mnemonic_lens  = ln_buf.data();
        batch.count          = 1;
        batch.salt_bytes     = reinterpret_cast<const uint8_t*>(salt.data());
        batch.salt_len       = static_cast<uint32_t>(salt.size());
        batch.out_seeds      = seed_gpu.data();

        rc = run_pbkdf2_batch(batch, stream);
        if (rc) {
            std::fprintf(stderr, "[!] vec %zu: kernel rc=%d (%s)\n",
                         i, rc, cudaGetErrorString(rc));
            ++g_fail;
            continue;
        }

        auto seed_cpu = ::collider::bip32::mnemonic_to_seed(mn, pp);
        const bool match = (std::memcmp(seed_gpu.data(), seed_cpu.data(),
                                        kSeedBytes) == 0);
        if (match) {
            ++g_pass;
            std::printf("[PASS] %zu words: %s...\n",
                        std::count(mn.begin(), mn.end(), ' ') + 1,
                        mn.substr(0, 32).c_str());
        } else {
            ++g_fail;
            std::fprintf(stderr, "[FAIL] vec %zu mismatch\n", i);
            std::fprintf(stderr, "       CPU: %s\n",
                         hex_lower(seed_cpu.data(), kSeedBytes).c_str());
            std::fprintf(stderr, "       GPU: %s\n",
                         hex_lower(seed_gpu.data(), kSeedBytes).c_str());
        }
    }

    cudaStreamDestroy(stream);
    }  // end per-device loop

    std::printf("=== Result: %d pass, %d fail ===\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#endif  // COLLIDER_USE_CUDA
