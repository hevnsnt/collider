/**
 * test_multi_gpu_brain_wallet -- v1.4.2 regression test for the
 * per-device-context EC table assumption underlying BW-B3.
 *
 * Pre-BW-B3 the mega kernel selected its precomputed EC table via a
 * cross-device shared global; the fix relies on the fact that
 * __device__ globals (g_ec_precomputed_table, g_mega_ec_table) are
 * actually per-device-context, so a write on GPU 0 doesn't disturb
 * GPU 1's copy. The whole design hinges on that property holding under
 * concurrent dispatch.
 *
 * This test exercises that property end-to-end by dispatching the same
 * passphrase set through fused_brain_wallet_batch (the production brain-
 * wallet kernel; same code path the C-1 KAT validates single-GPU) on
 * GPU 0 and GPU 1 IN PARALLEL THREADS. Both must produce match_count==N
 * and the same match index set. A regression that re-introduces cross-
 * device state in either fused_pipeline.cu or its EC table init will
 * manifest as a mismatch here.
 *
 * SKIP CODE: 77 when fewer than 2 CUDA devices are available.
 */

#include <cuda_runtime.h>
#include "../src/core/crypto_cpu.hpp"
#include "../src/tools/utxo_bloom_builder.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <thread>
#include <vector>

// Forward declarations from the production codebase.
extern "C" {
    cudaError_t fused_pipeline_init(cudaStream_t stream);
    cudaError_t fused_brain_wallet_batch(
        const uint8_t* d_passphrases,
        const uint32_t* d_offsets,
        const uint32_t* d_lengths,
        const uint8_t* d_bloom_filter,
        uint64_t bloom_bits,
        uint64_t bloom_mask,        // Tier 1 perf F-mask
        int bloom_hashes,
        uint32_t bloom_seed,
        uint32_t* d_match_indices,
        uint32_t* d_match_count,
        uint8_t* d_private_keys,
        size_t count,
        cudaStream_t stream
    );
}

namespace {

// Bloom configuration -- mirrors test_gpu_hash160's KAT setup.
constexpr uint64_t BLOOM_BITS    = 8192;
constexpr int      BLOOM_HASHES  = 8;
constexpr uint32_t BLOOM_SEED    = 0xDEADBEEFu;

// 16 test passphrases (small enough to keep the test sub-second; large
// enough to make a wrong-table regression statistically obvious).
const std::vector<std::string> TEST_PASSPHRASES = {
    "abc", "satoshi", "password", "123456", "Bitcoin", "puzzle",
    "satoshi nakamoto", "correct horse battery staple",
    "the quick brown fox jumps over the lazy dog",
    "all your bitcoin are belong to us",
    std::string(55, 'a'), std::string(56, 'b'),
    std::string(64, 'c'), std::string(89, 'd'),
    "1", "treasure",
};

// Per-GPU run state -- packaged so a worker thread can own one.
struct GpuRun {
    int      device_id     = -1;
    uint32_t match_count   = 0;
    std::vector<uint32_t> match_indices;
    bool     ok            = false;
    std::string err;
};

void run_on_device(GpuRun& run,
                   const std::vector<uint8_t>& bloom_bytes,
                   const std::vector<uint8_t>& packed_passphrases,
                   const std::vector<uint32_t>& offsets,
                   const std::vector<uint32_t>& lengths) {
    cudaError_t err = cudaSetDevice(run.device_id);
    if (err != cudaSuccess) { run.err = cudaGetErrorString(err); return; }
    // fused_pipeline_init already ran sequentially on this device in main();
    // here we just confirm we're on the right device.

    // Allocate device buffers.
    uint8_t*  d_passphrases = nullptr;
    uint32_t* d_offsets     = nullptr;
    uint32_t* d_lengths     = nullptr;
    uint8_t*  d_bloom       = nullptr;
    uint32_t* d_match_idx   = nullptr;
    uint32_t* d_match_cnt   = nullptr;

    cudaMalloc(&d_passphrases, packed_passphrases.size());
    cudaMalloc(&d_offsets,     offsets.size() * sizeof(uint32_t));
    cudaMalloc(&d_lengths,     lengths.size() * sizeof(uint32_t));
    cudaMalloc(&d_bloom,       bloom_bytes.size());
    cudaMalloc(&d_match_idx,   1024 * sizeof(uint32_t));
    cudaMalloc(&d_match_cnt,   sizeof(uint32_t));

    cudaMemcpy(d_passphrases, packed_passphrases.data(), packed_passphrases.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets,     offsets.data(),  offsets.size() * sizeof(uint32_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths,     lengths.data(),  lengths.size() * sizeof(uint32_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_bloom,       bloom_bytes.data(), bloom_bytes.size(),              cudaMemcpyHostToDevice);
    cudaMemset(d_match_cnt, 0, sizeof(uint32_t));

    // Tier 1 perf F-mask: BLOOM_BITS == 8192 is a power of two, so the
    // kernel uses the bitwise-AND fast path.
    static_assert((BLOOM_BITS & (BLOOM_BITS - 1)) == 0,
                  "Test bloom size must remain a power of two.");
    constexpr uint64_t BLOOM_MASK = BLOOM_BITS - 1;
    err = fused_brain_wallet_batch(
        d_passphrases, d_offsets, d_lengths,
        d_bloom, BLOOM_BITS, BLOOM_MASK, BLOOM_HASHES, BLOOM_SEED,
        d_match_idx, d_match_cnt, /*d_private_keys=*/nullptr,
        offsets.size(), /*stream=*/0
    );
    if (err != cudaSuccess) {
        run.err = std::string("fused_brain_wallet_batch: ") + cudaGetErrorString(err);
        return;
    }

    cudaDeviceSynchronize();
    cudaError_t last_err = cudaGetLastError();
    if (last_err != cudaSuccess) {
        run.err = std::string("kernel runtime error: ") + cudaGetErrorString(last_err);
        return;
    }

    uint32_t cnt = 0;
    cudaMemcpy(&cnt, d_match_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    run.match_count = cnt;
    run.match_indices.assign(std::min<uint32_t>(cnt, 1024), 0);
    if (cnt > 0) {
        cudaMemcpy(run.match_indices.data(), d_match_idx,
                   std::min<uint32_t>(cnt, 1024) * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
    }

    cudaFree(d_passphrases); cudaFree(d_offsets); cudaFree(d_lengths);
    cudaFree(d_bloom);       cudaFree(d_match_idx); cudaFree(d_match_cnt);

    run.ok = run.err.empty();
}

}  // namespace

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count < 2) {
        std::fprintf(stderr, "Need >=2 CUDA devices, found %d (%s); skipping.\n",
                     device_count, cudaGetErrorString(err));
        return 77;
    }

    const size_t N = TEST_PASSPHRASES.size();
    std::printf("=== Multi-GPU concurrent brainwallet test (BW-B3 regression) ===\n");
    std::printf("Devices: %d  Passphrases: %zu  Bloom seed: 0x%08x\n",
                device_count, N, BLOOM_SEED);

    // Initialize EC tables SEQUENTIALLY on each GPU (avoids any startup
    // races between fused_pipeline_init calls on different contexts).
    // The actual concurrency test is the kernel dispatch below.
    for (int dev = 0; dev < 2; dev++) {
        cudaSetDevice(dev);
        cudaError_t init_err = fused_pipeline_init(/*stream*/0);
        if (init_err != cudaSuccess) {
            std::fprintf(stderr, "fused_pipeline_init failed on device %d: %s\n",
                         dev, cudaGetErrorString(init_err));
            return 1;
        }
        cudaDeviceSynchronize();
    }

    // Expected H160s via the CPU reference.
    std::vector<std::array<uint8_t, 20>> expected_h160(N);
    for (size_t i = 0; i < N; i++) {
        const std::string& pp = TEST_PASSPHRASES[i];
        auto pk = collider::cpu::SHA256::hash(
            reinterpret_cast<const uint8_t*>(pp.data()), pp.size());
        expected_h160[i] = collider::cpu::compute_hash160(pk.data());
    }

    // Build a bloom filter containing every expected H160.
    std::vector<uint8_t> bloom(BLOOM_BITS / 8, 0);
    for (size_t i = 0; i < N; i++) {
        auto [h1, h2] = ::collider::utxo::murmurhash3_128(
            expected_h160[i].data(), 20, BLOOM_SEED);
        for (int k = 0; k < BLOOM_HASHES; k++) {
            uint64_t idx = (h1 + (uint64_t)k * h2) % BLOOM_BITS;
            bloom[idx / 8] |= (uint8_t)(1u << (idx % 8));
        }
    }

    // Pack passphrases densely with offsets+lengths arrays (the format
    // fused_brain_wallet_batch expects).
    std::vector<uint8_t>  packed_passphrases;
    std::vector<uint32_t> offsets(N);
    std::vector<uint32_t> lengths(N);
    for (size_t i = 0; i < N; i++) {
        offsets[i] = static_cast<uint32_t>(packed_passphrases.size());
        lengths[i] = static_cast<uint32_t>(TEST_PASSPHRASES[i].size());
        packed_passphrases.insert(
            packed_passphrases.end(),
            reinterpret_cast<const uint8_t*>(TEST_PASSPHRASES[i].data()),
            reinterpret_cast<const uint8_t*>(TEST_PASSPHRASES[i].data()) + TEST_PASSPHRASES[i].size());
    }

    // Launch GPU 0 and GPU 1 in parallel threads. This is the actual
    // race condition we're regression-testing: if BW-B3 regresses, the
    // two launches will race on a shared device-side global.
    GpuRun run0{0, 0, {}, false, ""};
    GpuRun run1{1, 0, {}, false, ""};
    std::thread t0(run_on_device, std::ref(run0),
                   std::cref(bloom), std::cref(packed_passphrases),
                   std::cref(offsets), std::cref(lengths));
    std::thread t1(run_on_device, std::ref(run1),
                   std::cref(bloom), std::cref(packed_passphrases),
                   std::cref(offsets), std::cref(lengths));
    t0.join();
    t1.join();

    if (!run0.ok || !run1.ok) {
        std::fprintf(stderr, "GPU 0 ok=%d err=%s\n", run0.ok, run0.err.c_str());
        std::fprintf(stderr, "GPU 1 ok=%d err=%s\n", run1.ok, run1.err.c_str());
        return 1;
    }

    std::printf("GPU 0: match_count = %u\n", run0.match_count);
    std::printf("GPU 1: match_count = %u\n", run1.match_count);

    if (run0.match_count != N || run1.match_count != N) {
        std::fprintf(stderr,
                     "FAIL: expected %zu matches per GPU; got GPU0=%u GPU1=%u.\n"
                     "      A regression of BW-B3 would manifest this way --\n"
                     "      one GPU reads the wrong EC table and produces wrong H160s.\n",
                     N, run0.match_count, run1.match_count);
        return 2;
    }

    // Both GPUs must have hit the SAME set of indices. Order may differ
    // (atomicAdd into d_match_count is not order-preserving), so compare
    // as sets.
    std::set<uint32_t> set0(run0.match_indices.begin(), run0.match_indices.end());
    std::set<uint32_t> set1(run1.match_indices.begin(), run1.match_indices.end());
    if (set0 != set1) {
        std::fprintf(stderr,
                     "FAIL: GPU 0 and GPU 1 produced different match-index sets.\n"
                     "      This is the smoking gun for BW-B3 regression.\n");
        return 3;
    }

    std::printf("PASS: GPU 0 and GPU 1 produced identical match sets "
                "of size %zu under concurrent dispatch.\n", set0.size());
    return 0;
}
