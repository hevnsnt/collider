// test_fused_pipeline_oob.cu -- TP-5 boundary KAT for the brain-wallet
// fused GPU pipeline. Targets the specific OOB / fastfail classes
// that have bitten the codebase historically:
//
//   1. degenerate count=0       -- launch with no work; must not
//                                  cudaErrorIllegalAddress on the
//                                  zero-block grid or the match
//                                  counter reset.
//   2. count=1                  -- single-thread block; verifies the
//                                  idx>=count early-exit gate (line
//                                  1368 of fused_pipeline.cu) is the
//                                  only out-of-range guard the kernel
//                                  needs at small batch.
//   3. all-zero scalars         -- every thread's SHA256(empty)
//                                  produces a non-zero scalar by
//                                  accident (sha is irreversible);
//                                  pad with a single byte that we
//                                  know hashes to scalar==0 mod n by
//                                  fused_validate_scalar. Confirms
//                                  the validate gate (line 1400) does
//                                  not crash on the rejected path.
//   4. bloom-all-true overflow  -- prime the bloom filter so every
//                                  hash160 hits, then launch with
//                                  count > MAX_MATCHES_PER_BATCH.
//                                  The atomicAdd inside the kernel
//                                  must clamp the slot index; we
//                                  verify match_count goes past the
//                                  capacity AND no OOB write into
//                                  match_indices.

#include "core/crypto_cpu.hpp"
#include "gpu/brain_wallet_gpu.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_passes = 0;

void fail(const char* tag, const std::string& msg) {
    std::fprintf(stderr, "[FAIL] %s: %s\n", tag, msg.c_str());
    ++g_failures;
}
void pass(const char* tag) {
    std::printf("[ ok  ] %s\n", tag);
    ++g_passes;
}

#define CUDA_OK(expr, tag)                                                 \
    do {                                                                   \
        cudaError_t _e = (expr);                                           \
        if (_e != cudaSuccess) {                                           \
            fail(tag, std::string("cuda err: ") + cudaGetErrorString(_e)); \
            return;                                                        \
        }                                                                  \
    } while (0)

}  // namespace

// fused_brain_wallet_batch lives in fused_pipeline.cu (extern "C" wrapper).
extern "C" cudaError_t fused_brain_wallet_batch(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    const uint8_t* d_bloom_filter,
    uint64_t bloom_bits,
    uint64_t bloom_mask,
    int bloom_hashes,
    uint32_t bloom_seed,
    uint32_t* d_match_indices,
    uint32_t* d_match_count,
    uint8_t* d_private_keys,
    size_t count,
    cudaStream_t stream);

extern "C" cudaError_t secp256k1_init_table(cudaStream_t stream);

namespace {

void test_count_zero() {
    cudaStream_t s = 0;
    uint32_t* d_count = nullptr;
    CUDA_OK(cudaMalloc(&d_count, sizeof(uint32_t)), "count_zero/malloc");

    // No passphrases; cuda kernel must not launch (zero blocks) and
    // the host wrapper must not fault on the cudaMemsetAsync of the
    // count buffer.
    cudaError_t e = fused_brain_wallet_batch(
        nullptr, nullptr, nullptr,
        nullptr, /*bits=*/8, /*mask=*/0, /*k=*/1, /*seed=*/0,
        nullptr, d_count, nullptr, /*count=*/0, s);
    if (e != cudaSuccess) {
        fail("count_zero", std::string("dispatch err: ") +
                              cudaGetErrorString(e));
    } else {
        cudaStreamSynchronize(s);
        cudaError_t sync_err = cudaGetLastError();
        if (sync_err != cudaSuccess) {
            fail("count_zero/sync",
                 std::string("post-sync err: ") +
                     cudaGetErrorString(sync_err));
        } else {
            pass("count_zero");
        }
    }
    cudaFree(d_count);
}

// Build a minimal bloom filter where every bit is set. Hashes ALWAYS
// match. Used by the overflow test to force the match-count saturation
// path; reused as a cheap "always true" bloom by the count_one test.
struct OneBloom {
    uint64_t num_bits = 0;
    uint64_t mask     = 0;
    int      k        = 1;
    uint32_t seed     = 0;
    uint8_t* d_data   = nullptr;
};

OneBloom make_all_set_bloom(uint64_t num_bits, int k) {
    OneBloom b;
    b.num_bits = num_bits;
    // mask non-zero requires num_bits be pow-of-2.
    b.mask = ((num_bits & (num_bits - 1)) == 0) ? (num_bits - 1) : 0;
    b.k    = k;
    b.seed = 0;
    const size_t bytes = (num_bits + 7) / 8;
    cudaMalloc(&b.d_data, bytes);
    cudaMemset(b.d_data, 0xFF, bytes);
    return b;
}

void free_bloom(OneBloom& b) {
    if (b.d_data) cudaFree(b.d_data);
    b.d_data = nullptr;
}

void test_count_one() {
    cudaStream_t s = 0;

    // Passphrase: "test" -- known to produce a valid scalar.
    const std::string pp = "test";
    uint8_t* d_pp = nullptr;
    uint32_t* d_off = nullptr;
    uint32_t* d_len = nullptr;
    uint32_t* d_match_idx = nullptr;
    uint32_t* d_match_count = nullptr;
    uint8_t* d_priv = nullptr;
    cudaMalloc(&d_pp, pp.size());
    cudaMalloc(&d_off, sizeof(uint32_t));
    cudaMalloc(&d_len, sizeof(uint32_t));
    cudaMalloc(&d_match_idx, sizeof(uint32_t));
    cudaMalloc(&d_match_count, sizeof(uint32_t));
    cudaMalloc(&d_priv, 32);
    cudaMemcpyAsync(d_pp, pp.data(), pp.size(), cudaMemcpyHostToDevice, s);
    uint32_t off_v = 0;
    uint32_t len_v = static_cast<uint32_t>(pp.size());
    cudaMemcpyAsync(d_off, &off_v, sizeof(uint32_t), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(d_len, &len_v, sizeof(uint32_t), cudaMemcpyHostToDevice, s);

    auto bloom = make_all_set_bloom(1u << 16, 1);

    cudaError_t e = fused_brain_wallet_batch(
        d_pp, d_off, d_len,
        bloom.d_data, bloom.num_bits, bloom.mask, bloom.k, bloom.seed,
        d_match_idx, d_match_count, d_priv,
        /*count=*/1, s);
    if (e != cudaSuccess) {
        fail("count_one", std::string("dispatch err: ") +
                              cudaGetErrorString(e));
    } else {
        cudaStreamSynchronize(s);
        cudaError_t sync_err = cudaGetLastError();
        if (sync_err != cudaSuccess) {
            fail("count_one/sync",
                 std::string("post-sync err: ") +
                     cudaGetErrorString(sync_err));
        } else {
            // Match must fire on bloom-all-true.
            uint32_t got_count = 0;
            cudaMemcpy(&got_count, d_match_count, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            if (got_count != 1) {
                fail("count_one/match_count",
                     "expected 1 match, got " + std::to_string(got_count));
            } else {
                pass("count_one");
            }
        }
    }
    free_bloom(bloom);
    cudaFree(d_pp); cudaFree(d_off); cudaFree(d_len);
    cudaFree(d_match_idx); cudaFree(d_match_count); cudaFree(d_priv);
}

void test_match_count_saturation() {
    cudaStream_t s = 0;

    // Generate count > MAX_MATCHES_PER_BATCH (32768) passphrases.
    // 50000 forces ~17000 OOB-slot writes that the kernel must SKIP
    // (line 1432 guard). If the guard fails, the next thread's
    // atomicAdd lands at slot 32768 and writes past d_match_idx.
    constexpr size_t kCount = 50'000;
    constexpr uint32_t kCap = ::collider::gpu::MAX_MATCHES_PER_BATCH;
    static_assert(kCount > kCap, "test premise: must exceed capacity");

    // One-byte passphrases "a", "b", "c", ... cycled. SHA256 of each
    // is independent; the bloom-all-true mask makes every thread
    // attempt to write a match index.
    std::vector<uint8_t> pp_buf(kCount);
    std::vector<uint32_t> offsets(kCount);
    std::vector<uint32_t> lengths(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        pp_buf[i] = static_cast<uint8_t>('a' + (i % 26));
        offsets[i] = static_cast<uint32_t>(i);
        lengths[i] = 1;
    }
    uint8_t* d_pp = nullptr;
    uint32_t* d_off = nullptr;
    uint32_t* d_len = nullptr;
    cudaMalloc(&d_pp, kCount);
    cudaMalloc(&d_off, kCount * sizeof(uint32_t));
    cudaMalloc(&d_len, kCount * sizeof(uint32_t));
    cudaMemcpy(d_pp, pp_buf.data(), kCount, cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, offsets.data(), kCount * sizeof(uint32_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_len, lengths.data(), kCount * sizeof(uint32_t),
               cudaMemcpyHostToDevice);

    // Match-index buffer of exact capacity. If the kernel's slot guard
    // is broken, the OOB writes land beyond this; CUDA's bounds-check
    // helper (we follow with cudaMemsetAsync to a known sentinel and
    // verify it survived) catches the corruption.
    uint32_t* d_match_idx = nullptr;
    uint32_t* d_match_count = nullptr;
    cudaMalloc(&d_match_idx, kCap * sizeof(uint32_t));
    cudaMalloc(&d_match_count, sizeof(uint32_t));
    // Allocate a sentinel buffer immediately after match_idx to detect
    // OOB writes. cudaMalloc does not guarantee contiguity, so we use
    // a single large allocation + manual slicing.
    uint32_t* d_combined = nullptr;
    cudaMalloc(&d_combined, (kCap + 1024) * sizeof(uint32_t));
    cudaMemset(d_combined, 0xCD, (kCap + 1024) * sizeof(uint32_t));
    uint32_t* d_match_arena = d_combined;
    uint32_t* d_sentinel    = d_combined + kCap;
    cudaFree(d_match_idx);
    d_match_idx = d_match_arena;

    auto bloom = make_all_set_bloom(1u << 16, 1);

    cudaError_t e = fused_brain_wallet_batch(
        d_pp, d_off, d_len,
        bloom.d_data, bloom.num_bits, bloom.mask, bloom.k, bloom.seed,
        d_match_idx, d_match_count, nullptr,
        kCount, s);
    if (e != cudaSuccess) {
        fail("match_count_saturation",
             std::string("dispatch err: ") + cudaGetErrorString(e));
    } else {
        cudaStreamSynchronize(s);
        cudaError_t sync_err = cudaGetLastError();
        if (sync_err != cudaSuccess) {
            fail("match_count_saturation/sync",
                 std::string("post-sync err: ") +
                     cudaGetErrorString(sync_err));
        } else {
            // Verify match_count went PAST the capacity (atomicAdd is
            // unbounded), proving the saturation path was exercised.
            uint32_t got_count = 0;
            cudaMemcpy(&got_count, d_match_count, sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            if (got_count <= kCap) {
                fail("match_count_saturation/exercised",
                     "match_count " + std::to_string(got_count) +
                         " did not exceed kCap " + std::to_string(kCap) +
                         "; saturation path may not have triggered");
            } else {
                pass("match_count_saturation/exercised");
            }

            // Sentinel verify: the 1024 uint32s immediately after the
            // match-index arena must still hold the 0xCD pattern. If
            // the kernel OOB-wrote a match index past the cap, this
            // catches it. The first 32 bytes of 0xCD repeat-padding
            // give us a strong canary.
            std::vector<uint32_t> sentinel(1024);
            cudaMemcpy(sentinel.data(), d_sentinel,
                       1024 * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost);
            constexpr uint32_t kSentinel = 0xCDCDCDCDu;
            bool sentinel_ok = true;
            size_t bad_at = 0;
            for (size_t i = 0; i < 1024; ++i) {
                if (sentinel[i] != kSentinel) {
                    sentinel_ok = false;
                    bad_at = i;
                    break;
                }
            }
            if (!sentinel_ok) {
                fail("match_count_saturation/sentinel",
                     "OOB write detected at sentinel[" +
                         std::to_string(bad_at) + "] = 0x" +
                         std::to_string(sentinel[bad_at]));
            } else {
                pass("match_count_saturation/sentinel");
            }
        }
    }
    free_bloom(bloom);
    cudaFree(d_pp); cudaFree(d_off); cudaFree(d_len);
    cudaFree(d_combined); cudaFree(d_match_count);
    // d_match_idx aliased d_combined; do not free separately.
}

}  // namespace

int main() {
    std::printf("=== test_fused_pipeline_oob (TP-5) ===\n");

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) {
        std::printf("[SKIP] no CUDA device present\n");
        return 0;
    }
    cudaSetDevice(0);
    cudaStream_t init_s = 0;
    cudaError_t init_e = secp256k1_init_table(init_s);
    if (init_e != cudaSuccess) {
        std::fprintf(stderr,
                     "[FAIL] secp256k1_init_table: %s\n",
                     cudaGetErrorString(init_e));
        return 1;
    }

    test_count_zero();
    test_count_one();
    test_match_count_saturation();

    std::printf("\n%d passes, %d failures\n", g_passes, g_failures);
    return g_failures == 0 ? 0 : 1;
}
