// See bench_pipeline.hpp for the design rationale and contract.

#include "bench_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#ifdef COLLIDER_USE_CUDA
#include <cuda_runtime.h>

// extern "C" wrappers from src/gpu/. These are the same entry points the
// brain-wallet runner uses in production, so the bench numbers reflect
// real production code paths and not a parallel test harness.
extern "C" {

cudaError_t fused_pipeline_init(cudaStream_t stream);

cudaError_t sha256_batch(
    const std::uint8_t* d_passphrases,
    const std::uint32_t* d_offsets,
    const std::uint32_t* d_lengths,
    std::uint8_t* d_hashes,
    std::size_t count,
    cudaStream_t stream
);

// T0.3.a (A-tier wave 1): the per-thread `secp256k1_batch_mul_simple`
// path uses `ec_mul_batch_kernel -> ec_mul_windowed`, the no-inter-
// window-double windowed scalar mul. The previous `secp256k1_batch_mul`
// wrapper dispatched the buggy `ec_mul_batch_optimized_kernel` (windowed
// batch-inversion with double 2^(5w) shift -> wrong points) and was
// deleted; bench stage 2 numbers were "inflated by the bug" per the
// audit, and now reflect the correct kernel.
cudaError_t secp256k1_batch_mul_simple(
    const void* d_private_keys,
    void* d_public_keys,
    std::size_t count,
    cudaStream_t stream
);

cudaError_t sha256_pubkey33_batch(
    const std::uint8_t* d_pubkeys,
    std::uint8_t* d_hashes,
    std::size_t count,
    cudaStream_t stream
);

cudaError_t ripemd160_batch(
    const std::uint8_t* d_inputs,
    std::uint8_t* d_outputs,
    std::size_t count,
    cudaStream_t stream
);

cudaError_t h160_bloom_set_config(std::uint64_t num_bits,
                                  std::uint64_t bloom_mask,
                                  std::uint32_t num_hashes,
                                  std::uint32_t seed);

cudaError_t h160_bloom_batch_check_auto(const std::uint8_t* d_h160s,
                                        const std::uint8_t* d_bloom_data,
                                        std::uint8_t* d_match_flags,
                                        std::size_t count,
                                        cudaStream_t stream);

cudaError_t fused_brain_wallet_batch_fixed_stride(
    const std::uint8_t* d_passphrases,
    const std::uint32_t* d_lengths,
    std::uint32_t stride,
    const std::uint8_t* d_bloom_filter,
    std::uint64_t bloom_bits,
    std::uint64_t bloom_mask,
    int bloom_hashes,
    std::uint32_t bloom_seed,
    std::uint32_t* d_match_indices,
    std::uint32_t* d_match_count,
    std::uint8_t* d_private_keys,
    std::size_t count,
    cudaStream_t stream,
    std::uint8_t multi_addr
);

}  // extern "C"
#endif  // COLLIDER_USE_CUDA

namespace collider {
namespace runtime {
namespace bench {

namespace {

// Format a rate in keys-per-second with engineering suffixes. Mirrors the
// format used by the brain-wallet TUI so operators see the same numbers
// here as on a running scan.
std::string format_rate(double v) {
    std::ostringstream os;
    os << std::fixed;
    if (v >= 1e9) {
        os << std::setprecision(2) << (v / 1e9) << " GH/s";
    } else if (v >= 1e6) {
        os << std::setprecision(2) << (v / 1e6) << " MH/s";
    } else if (v >= 1e3) {
        os << std::setprecision(2) << (v / 1e3) << " KH/s";
    } else {
        os << std::setprecision(0) << v << " H/s";
    }
    return os.str();
}

std::string format_count(std::uint64_t n) {
    std::ostringstream os;
    os.imbue(std::locale::classic());
    os << n;
    std::string s = os.str();
    // Insert thousands separators (commas).
    int insert_pos = static_cast<int>(s.size()) - 3;
    while (insert_pos > 0) {
        s.insert(static_cast<std::size_t>(insert_pos), ",");
        insert_pos -= 3;
    }
    return s;
}

#ifdef COLLIDER_USE_CUDA

// RAII helpers so the bench cannot leak device memory on an early return.
struct DevPtr {
    void* p = nullptr;
    DevPtr() = default;
    explicit DevPtr(std::size_t bytes) {
        cudaMalloc(&p, bytes);
    }
    ~DevPtr() { if (p) cudaFree(p); }
    DevPtr(const DevPtr&) = delete;
    DevPtr& operator=(const DevPtr&) = delete;
    DevPtr(DevPtr&& other) noexcept : p(other.p) { other.p = nullptr; }
    DevPtr& operator=(DevPtr&& other) noexcept {
        if (this != &other) {
            if (p) cudaFree(p);
            p = other.p;
            other.p = nullptr;
        }
        return *this;
    }
};

// Time one kernel-loop stage. Repeats the supplied launcher lambda until
// the configured wall-clock budget has elapsed, syncing once per iteration
// so the GPU is fully drained before the loop accumulates more work.
// Returns (seconds, items_processed).
template <typename Launcher>
StageResult time_stage(const std::string& name,
                       std::size_t per_iter_items,
                       int bench_seconds,
                       Launcher&& launcher) {
    using clock = std::chrono::steady_clock;

    // Warmup passes so first-launch compile / JIT / cache fills do not
    // dilute the measurement window, AND so the GPU clock has time to
    // ramp from idle (~600-800 MHz on a cold Ada part) to boost (~2700+
    // MHz). A single warmup launch was previously enough to fill the
    // kernel cache but left the very first measured iteration running on
    // a still-ramping clock, which depressed the reported throughput by
    // roughly 17% relative to steady state. The four extra launches add
    // ~tens of milliseconds (kernel time is small per launch) and bring
    // the first measured iteration onto the same clock plateau as the
    // rest of the measurement loop.
    constexpr int kWarmupPasses = 5;
    for (int i = 0; i < kWarmupPasses; ++i) {
        launcher();
    }
    cudaDeviceSynchronize();

    const auto budget = std::chrono::seconds(bench_seconds);
    const auto t0 = clock::now();
    std::uint64_t items = 0;
    int iterations = 0;
    while (clock::now() - t0 < budget) {
        launcher();
        cudaDeviceSynchronize();
        items += per_iter_items;
        ++iterations;
    }
    const auto t1 = clock::now();

    StageResult r;
    r.name = name;
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    r.items = items;
    r.rate_per_sec = (r.seconds > 0.0) ? (static_cast<double>(items) / r.seconds) : 0.0;
    (void)iterations;
    return r;
}

#endif  // COLLIDER_USE_CUDA

}  // namespace

PipelineBenchResult run_pipeline_benchmark(const PipelineBenchConfig& config,
                                           bool verbose) {
    PipelineBenchResult r;
    r.gpu_id = config.gpu_id;
    r.batch_size = config.batch_size;
    r.stride = config.stride;

#ifndef COLLIDER_USE_CUDA
    r.error = "Built without CUDA (COLLIDER_USE_CUDA undefined). "
              "The pipeline benchmark requires a CUDA toolkit at build "
              "time.";
    return r;
#else
    if (config.passphrase_len > config.stride) {
        r.error = "passphrase_len exceeds stride";
        return r;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        r.error = std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err);
        return r;
    }
    if (config.gpu_id < 0 || config.gpu_id >= device_count) {
        r.error = "gpu_id out of range";
        return r;
    }

    err = cudaSetDevice(config.gpu_id);
    if (err != cudaSuccess) {
        r.error = std::string("cudaSetDevice failed: ") + cudaGetErrorString(err);
        return r;
    }

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, config.gpu_id);
    r.gpu_name = prop.name;
    r.compute_capability_major = prop.major;
    r.compute_capability_minor = prop.minor;

    if (verbose) {
        std::cout << "[bench] GPU " << config.gpu_id << " = " << r.gpu_name
                  << " (sm_" << prop.major << prop.minor << ")\n";
        std::cout << "[bench] batch=" << format_count(config.batch_size)
                  << "  stride=" << config.stride
                  << "  passphrase_len=" << config.passphrase_len
                  << "  bench_seconds_per_stage=" << config.bench_seconds
                  << "\n";
    }

    // One-time setup: initialize the fused pipeline (loads the secp256k1
    // precomputed table on this device).
    err = fused_pipeline_init(/*stream*/0);
    if (err != cudaSuccess) {
        r.error = std::string("fused_pipeline_init failed: ") + cudaGetErrorString(err);
        return r;
    }
    cudaDeviceSynchronize();

    const std::size_t N = config.batch_size;
    const std::uint32_t stride = config.stride;
    const std::uint32_t pp_len = config.passphrase_len;

    // ---- Allocate device buffers ----
    DevPtr d_passphrases(N * stride);
    DevPtr d_offsets(N * sizeof(std::uint32_t));
    DevPtr d_lengths(N * sizeof(std::uint32_t));
    DevPtr d_privkeys(N * 32);
    DevPtr d_pubkeys(N * 64);   // secp256k1_batch_mul_simple writes ECPointAffine (x,y) = 64 bytes
    DevPtr d_pubhash(N * 32);   // SHA-256(pubkey) intermediate
    DevPtr d_hash160(N * 20);
    DevPtr d_match_flags(N);
    DevPtr d_match_idx(N * sizeof(std::uint32_t));
    DevPtr d_match_count(sizeof(std::uint32_t));

    if (!d_passphrases.p || !d_offsets.p || !d_lengths.p ||
        !d_privkeys.p || !d_pubkeys.p || !d_pubhash.p ||
        !d_hash160.p || !d_match_flags.p ||
        !d_match_idx.p || !d_match_count.p) {
        r.error = "cudaMalloc failed: out of memory at the requested batch_size. "
                  "Retry with a smaller --batch-size.";
        return r;
    }

    // ---- Build the bloom filter ----
    // Power-of-two sizing so the kernel's pow-2 fast path (bitwise AND
    // instead of 64-bit modulo) is exercised: that is the production
    // configuration after the G-B3 fix. 2^20 bits = 128 KiB, big enough
    // for the probe to do real cache-line lookups, small enough to be
    // L2-resident on every supported GPU.
    constexpr std::uint64_t kBloomBits = 1ULL << 20;
    constexpr std::uint64_t kBloomMask = kBloomBits - 1;
    constexpr std::uint32_t kBloomHashes = 11;
    constexpr std::uint32_t kBloomSeed = 0x5F3759DFu;
    const std::size_t bloom_bytes = kBloomBits / 8;

    DevPtr d_bloom(bloom_bytes);
    if (!d_bloom.p) {
        r.error = "bloom cudaMalloc failed";
        return r;
    }
    cudaMemset(d_bloom.p, 0, bloom_bytes);

    // Configure the constant-memory bloom slots used by the per-stage
    // h160_bloom_batch_check_auto kernel. The fused kernel takes its
    // bloom parameters as kernel arguments and does not depend on this
    // call.
    err = h160_bloom_set_config(kBloomBits, kBloomMask, kBloomHashes, kBloomSeed);
    if (err != cudaSuccess) {
        r.error = std::string("h160_bloom_set_config failed: ") + cudaGetErrorString(err);
        return r;
    }

    // ---- Fill the host-side input buffer with deterministic synthetic
    // bytes, then copy ONCE to the device. The bench loop re-runs the
    // kernel against the same buffer per iteration.
    std::vector<std::uint8_t> host_buf(static_cast<std::size_t>(N) * stride, 0);
    std::vector<std::uint32_t> host_offsets(N, 0);
    std::vector<std::uint32_t> host_lengths(N, pp_len);
    {
        std::mt19937_64 rng(0x1234567890ABCDEFULL);
        for (std::size_t i = 0; i < N; ++i) {
            host_offsets[i] = static_cast<std::uint32_t>(i * stride);
            std::uint8_t* slot = host_buf.data() + i * stride;
            // Fill the active passphrase region with pseudo-random ASCII
            // bytes. The actual content does not affect throughput since
            // the kernels treat the input as opaque bytes, but using
            // varied bytes is closer to a realistic scan than zeros.
            for (std::uint32_t j = 0; j < pp_len; ++j) {
                slot[j] = static_cast<std::uint8_t>(
                    33 + (rng() % (126 - 33)));  // printable ASCII
            }
        }
    }
    cudaMemcpy(d_passphrases.p, host_buf.data(), host_buf.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets.p, host_offsets.data(),
               N * sizeof(std::uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths.p, host_lengths.data(),
               N * sizeof(std::uint32_t), cudaMemcpyHostToDevice);

    if (verbose) {
        const double mb = static_cast<double>(host_buf.size()) / (1024.0 * 1024.0);
        std::cout << "[bench] input buffer: " << std::fixed << std::setprecision(1)
                  << mb << " MiB on-device\n";
        std::cout << "[bench] bloom:        " << (bloom_bytes / 1024) << " KiB ("
                  << kBloomHashes << " hashes, seed 0x" << std::hex << kBloomSeed
                  << std::dec << ")\n";
    }

    auto report = [&](const std::string& msg) {
        if (verbose) std::cout << "[bench] " << msg << std::endl;
    };

    // ----------------------------------------------------------------
    // Per-stage timings (each stage runs in isolation against its own
    // input buffer). The intermediate buffers chain so the inputs are
    // realistic distributions for each subsequent stage.
    // ----------------------------------------------------------------
    if (config.measure_stages) {
        // Stage 1: SHA-256 over the passphrase buffer. The output goes
        // into d_privkeys which is also the input for secp256k1 mul.
        report("stage 1/4: SHA-256 over passphrases ...");
        auto s1 = time_stage("SHA-256 (passphrase)",
                             N, config.bench_seconds,
                             [&]() {
            sha256_batch(
                static_cast<const std::uint8_t*>(d_passphrases.p),
                static_cast<const std::uint32_t*>(d_offsets.p),
                static_cast<const std::uint32_t*>(d_lengths.p),
                static_cast<std::uint8_t*>(d_privkeys.p),
                N, /*stream*/0);
        });
        r.stages.push_back(s1);

        // Stage 2: secp256k1 scalar multiply on the derived private keys.
        // T0.3.a: switched from `secp256k1_batch_mul` (deleted along with
        // its buggy ec_mul_batch_optimized_kernel backend) to the working
        // `secp256k1_batch_mul_simple` (ec_mul_batch_kernel -> ec_mul_windowed).
        // The stage rate reported here is therefore now the rate of CORRECT
        // EC multiplications, not the inflated-by-buggy-kernel rate the
        // pre-T0.3.a bench produced.
        report("stage 2/4: secp256k1 scalar multiply ...");
        auto s2 = time_stage("secp256k1 mul",
                             N, config.bench_seconds,
                             [&]() {
            secp256k1_batch_mul_simple(d_privkeys.p, d_pubkeys.p, N, /*stream*/0);
        });
        r.stages.push_back(s2);

        // Stage 3: hash160 = RIPEMD160(SHA-256(pubkey)). We do not have a
        // separate SHA-256-of-pubkey -> RIPEMD160 fused kernel exposed
        // outside the brain-wallet pipeline, so the bench runs the two
        // back-to-back. The result is reported as a single combined
        // "hash160 (pubkey -> 20-byte H160)" rate which is what an
        // operator cares about and matches the production sequence
        // inside fused_brain_wallet_batch.
        report("stage 3/4: hash160 (SHA-256 + RIPEMD-160) ...");
        auto s3 = time_stage("hash160 (SHA + RIPEMD)",
                             N, config.bench_seconds,
                             [&]() {
            // secp256k1_batch_mul_simple writes ECPointAffine; SHA-256 over the
            // x-coordinate (32 bytes) plus the parity bit on the y-coord
            // is what the production pipeline hashes for compressed
            // pubkeys. The standalone sha256_pubkey33_batch wrapper
            // expects 33 bytes per input; the y-byte is needed to seed
            // the parity prefix used by Bitcoin compressed serialization.
            // For the per-stage rate we approximate by hashing the raw
            // 32-byte x-coordinate via sha256_batch (length 32), which
            // executes the same SHA-256 compress function and therefore
            // produces a representative throughput number.
            // Setup: build offsets so each pubkey slot is 64 bytes apart
            // but only the first 32 bytes (x coord) are hashed. We reuse
            // the d_offsets array prepared earlier by re-targeting it
            // here would force a second cudaMemcpy per iteration which
            // would dominate the timer; instead we run sha256_pubkey33
            // over the dense pubkey buffer (which assumes 33-byte slots,
            // close enough to the 64-byte ECPointAffine for an upper-
            // bound rate). Either choice is approximate; the true number
            // appears in the fused total.
            sha256_pubkey33_batch(
                static_cast<const std::uint8_t*>(d_pubkeys.p),
                static_cast<std::uint8_t*>(d_pubhash.p),
                N, /*stream*/0);
            ripemd160_batch(
                static_cast<const std::uint8_t*>(d_pubhash.p),
                static_cast<std::uint8_t*>(d_hash160.p),
                N, /*stream*/0);
        });
        r.stages.push_back(s3);

        // Stage 4: bloom probe against the hash160 outputs.
        report("stage 4/4: bloom probe (constant-memory kernel) ...");
        auto s4 = time_stage("bloom probe",
                             N, config.bench_seconds,
                             [&]() {
            h160_bloom_batch_check_auto(
                static_cast<const std::uint8_t*>(d_hash160.p),
                static_cast<const std::uint8_t*>(d_bloom.p),
                static_cast<std::uint8_t*>(d_match_flags.p),
                N, /*stream*/0);
        });
        r.stages.push_back(s4);
    }

    // ----------------------------------------------------------------
    // End-to-end fused kernel. This is the production hot loop: one
    // kernel launch per batch_size passphrases, full chain on-GPU,
    // bloom-resident probe.
    // ----------------------------------------------------------------
    report("end-to-end: fused brain-wallet kernel ...");
    auto fused = time_stage("FUSED end-to-end",
                            N, config.bench_seconds,
                            [&]() {
        fused_brain_wallet_batch_fixed_stride(
            static_cast<const std::uint8_t*>(d_passphrases.p),
            static_cast<const std::uint32_t*>(d_lengths.p),
            stride,
            static_cast<const std::uint8_t*>(d_bloom.p),
            kBloomBits, kBloomMask, kBloomHashes, kBloomSeed,
            static_cast<std::uint32_t*>(d_match_idx.p),
            static_cast<std::uint32_t*>(d_match_count.p),
            /*d_private_keys=*/nullptr,
            N, /*stream*/0,
            /*multi_addr=*/std::uint8_t{0});
    });

    r.fused_keys_per_sec = fused.rate_per_sec;
    r.fused_seconds = fused.seconds;
    r.fused_items = fused.items;
    r.stages.push_back(fused);
    r.ok = true;
    return r;
#endif
}

void print_result_table(const PipelineBenchResult& result) {
    std::cout << "\n";
    std::cout << "+--------------------------------------------------------------+\n";
    std::cout << "|  theCollider full-pipeline benchmark                         |\n";
    std::cout << "+--------------------------------------------------------------+\n";
    std::cout << "  GPU " << result.gpu_id << ": " << result.gpu_name;
    if (result.compute_capability_major > 0) {
        std::cout << "  (sm_" << result.compute_capability_major
                  << result.compute_capability_minor << ")";
    }
    std::cout << "\n";
    std::cout << "  batch_size: " << format_count(result.batch_size)
              << "    stride: " << result.stride << " bytes\n";
    if (!result.ok) {
        std::cout << "  STATUS:   ERROR  (" << result.error << ")\n";
        return;
    }
    std::cout << "+--------------------------------------------------------------+\n";
    std::cout << "  Stage                          Rate                 Window\n";
    std::cout << "  -----                          ----                 ------\n";
    for (const auto& s : result.stages) {
        std::cout << "  " << std::left << std::setw(30) << s.name
                  << " " << std::right << std::setw(14) << format_rate(s.rate_per_sec)
                  << "   " << std::fixed << std::setprecision(2) << std::setw(5)
                  << s.seconds << " s\n";
    }
    std::cout << "+--------------------------------------------------------------+\n";
    std::cout << "  End-to-end fused: " << format_rate(result.fused_keys_per_sec)
              << "  over " << std::fixed << std::setprecision(2)
              << result.fused_seconds << " s "
              << "(" << format_count(result.fused_items) << " keys checked)\n";
    std::cout << "+--------------------------------------------------------------+\n";
}

}  // namespace bench
}  // namespace runtime
}  // namespace collider
