/**
 * puzzle_solver_benchmark.cpp - --benchmark Free SHA-256 path.
 *
 * Extracted from src/runtime/puzzle_solver.cpp during the v1.4.2
 * structural decomposition. Hosts run_sha256_only_benchmark(), which
 * mirrors the original inline #ifndef COLLIDER_PRO branch of
 * run_benchmark verbatim.
 *
 * The Pro --benchmark path stays in puzzle_solver.cpp as a thin
 * delegation to runtime/bench_pipeline.hpp (run_pipeline_benchmark +
 * print_result_table) per the v1.4.2 audit requirement that the
 * benchmark numbers reported in docs/PRO.md and README.md come from
 * the same kernel the brain-wallet runner uses in production.
 *
 * This translation unit is only meaningful in Free builds. The TU
 * itself compiles in both editions (the helper just becomes a no-op
 * decl when COLLIDER_PRO is defined) so we do not need a per-edition
 * source-list branch in CMakeLists.txt.
 */
#include "runtime/puzzle_solver_helpers.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifndef COLLIDER_PRO

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef COLLIDER_HAS_OPENSSL
#include <openssl/sha.h>
#endif

#ifdef __APPLE__
// CommonCrypto uses the ARMv8 SHA crypto extensions on Apple Silicon,
// roughly 100-300x faster than OpenSSL's software fallback when OpenSSL
// is built without crypto-extension support (Homebrew default).
#include <CommonCrypto/CommonDigest.h>
#endif

#if defined(__APPLE__) && defined(COLLIDER_USE_METAL)
#include "gpu/sha256_metal_bench.hpp"
#endif

#ifdef COLLIDER_USE_CUDA
#include <cuda_runtime.h>
// Forward decl from src/gpu/sha256.cu (no header for that one yet).
extern "C" cudaError_t sha256_batch(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    uint8_t* d_hashes,
    size_t count,
    cudaStream_t stream);
#endif

#include "ui/box_render.hpp"

#endif  // !COLLIDER_PRO

// Forward decl: defined at namespace-global scope in puzzle_solver.cpp.
std::string format_number(uint64_t n);

namespace collider::runtime::detail {

#ifndef COLLIDER_PRO

int run_sha256_only_benchmark(const Arguments& args,
                              const GPUDetectionResult& gpu_info) {
    // Free benchmark: SHA-256 throughput on CPU + GPU/backend info.
    // Gives users a real number to validate their hardware without
    // requiring the brain-wallet pipeline (Pro-only).
    {
        namespace boxui = ::collider::ui::box;
        std::cout << "\n";
        boxui::top(std::cout);
        boxui::centered(std::cout, "COLLIDER FREE BENCHMARK");
        boxui::top(std::cout);
        boxui::kv(std::cout, "Hardware",
                  gpu_info.gpu_names.empty()
                      ? std::string("(no GPU detected)")
                      : gpu_info.gpu_names);
        boxui::kv(std::cout, "Backend", gpu_info.backend);
        boxui::bottom(std::cout);
    }
    std::cout << "  Measuring CPU SHA-256 throughput over "
              << args.benchmark_seconds << "s...\n\n";

    // Streaming CPU SHA-256 over a 1 MB buffer. The 1 MB stream
    // saturates the ARMv8 SHA crypto unit on Apple Silicon and the
    // SHA-NI extension on modern Intel/AMD; the per-call init/final
    // overhead vanishes against the per-block hash time, so the
    // reported rate reflects actual hardware throughput.
    constexpr size_t kBufSize = 1 << 20;   // 1 MiB
    std::vector<uint8_t> buf(kBufSize);
    for (size_t i = 0; i < buf.size(); ++i) buf[i] = (uint8_t)(i & 0xff);
    uint8_t digest[32] = {0};
    const auto bench_dur =
        std::chrono::seconds(args.benchmark_seconds > 0
                             ? args.benchmark_seconds : 5);
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t bytes_hashed = 0;
    uint64_t streams_done = 0;
    const char* sha_backend = nullptr;
#if defined(__APPLE__)
    sha_backend = "CommonCrypto (ARMv8 SHA crypto extensions)";
    while (std::chrono::steady_clock::now() - t0 < bench_dur) {
        CC_SHA256_CTX ctx;
        CC_SHA256_Init(&ctx);
        CC_SHA256_Update(&ctx, buf.data(), (CC_LONG)buf.size());
        CC_SHA256_Final(digest, &ctx);
        bytes_hashed += buf.size();
        streams_done += 1;
    }
#elif defined(COLLIDER_HAS_OPENSSL)
    sha_backend = "OpenSSL EVP";
    while (std::chrono::steady_clock::now() - t0 < bench_dur) {
        SHA256_CTX ctx;
        SHA256_Init(&ctx);
        SHA256_Update(&ctx, buf.data(), buf.size());
        SHA256_Final(digest, &ctx);
        bytes_hashed += buf.size();
        streams_done += 1;
    }
#endif
    std::cout << "[CPU]\n";
    if (sha_backend != nullptr) {
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        const double sec = (double)elapsed_ms / 1000.0;
        const double mbps = (double)bytes_hashed / (sec * 1024.0 * 1024.0);
        // Report the equivalent rate of 64-byte SHA-256 blocks per second
        // for direct comparison with the GPU number below.
        const double blocks_per_sec = (double)bytes_hashed / 64.0 / sec;
        std::cout << "  Backend:    " << sha_backend << "\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
                  << mbps << " MB/s   ("
                  << format_number((uint64_t)blocks_per_sec)
                  << " 64-byte blocks/s)\n";
        std::cout << "  Method:     streaming SHA-256 over 1 MiB buffer, "
                  << streams_done << " streams in " << std::fixed
                  << std::setprecision(1) << sec << "s\n";
    } else {
        std::cout << "  SHA-256 unavailable (built without OpenSSL).\n";
    }
    (void)digest;

    // -------------------------------------------------------------
    // GPU SHA-256 benchmark
    // -------------------------------------------------------------
    std::cout << "\n[GPU]\n";
#if defined(__APPLE__) && defined(COLLIDER_USE_METAL)
    {
        std::cout << "  Running Metal sha256_bench kernel ("
                  << args.benchmark_seconds << "s)...\n";
        auto gpu = collider::gpu::run_sha256_metal_benchmark(
            args.benchmark_seconds);
        if (gpu.ok) {
            std::cout << "  Backend:    Metal (" << gpu.device_name << ")\n";
            std::cout << "  SHA-256:    "
                      << format_number((uint64_t)gpu.hashes_per_second)
                      << " H/s (batched, 64-byte input)\n";
        } else {
            std::cout << "  Metal benchmark failed: " << gpu.error << "\n";
        }
    }
#elif defined(COLLIDER_USE_CUDA)
    {
        // Allocate a 64-byte * batch input buffer and run sha256_batch
        // for the configured number of seconds.
        constexpr uint32_t kBatchSize = 1u << 18;   // 262144 inputs / dispatch
        uint8_t* d_in = nullptr;
        uint32_t* d_offsets = nullptr;
        uint32_t* d_lengths = nullptr;
        uint8_t* d_out = nullptr;
        cudaError_t cerr = cudaMalloc(&d_in, kBatchSize * 64);
        if (cerr == cudaSuccess) cerr = cudaMalloc(&d_offsets, kBatchSize * sizeof(uint32_t));
        if (cerr == cudaSuccess) cerr = cudaMalloc(&d_lengths, kBatchSize * sizeof(uint32_t));
        if (cerr == cudaSuccess) cerr = cudaMalloc(&d_out, kBatchSize * 32);
        if (cerr == cudaSuccess) {
            std::vector<uint32_t> h_offsets(kBatchSize), h_lengths(kBatchSize);
            for (uint32_t i = 0; i < kBatchSize; ++i) {
                h_offsets[i] = i * 64;
                h_lengths[i] = 64;
            }
            cudaMemcpy(d_offsets, h_offsets.data(),
                       kBatchSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_lengths, h_lengths.data(),
                       kBatchSize * sizeof(uint32_t), cudaMemcpyHostToDevice);
            cudaMemset(d_in, 0xAB, kBatchSize * 64);
            // Warmup
            sha256_batch(d_in, d_offsets, d_lengths, d_out, kBatchSize, 0);
            cudaDeviceSynchronize();

            std::cout << "  Running CUDA sha256_batch kernel ("
                      << args.benchmark_seconds << "s)...\n";
            const auto gt0 = std::chrono::steady_clock::now();
            const auto gdur = std::chrono::seconds(
                args.benchmark_seconds > 0 ? args.benchmark_seconds : 5);
            uint64_t ghashes = 0;
            while (std::chrono::steady_clock::now() - gt0 < gdur) {
                sha256_batch(d_in, d_offsets, d_lengths, d_out, kBatchSize, 0);
                cudaDeviceSynchronize();
                ghashes += kBatchSize;
            }
            const auto gms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - gt0).count();
            double grate = (double)ghashes * 1000.0 / (double)gms;
            std::cout << "  Backend:    CUDA ("
                      << (gpu_info.gpu_names.empty() ? std::string("GPU")
                                                      : gpu_info.gpu_names) << ")\n";
            std::cout << "  SHA-256:    " << format_number((uint64_t)grate)
                      << " H/s (batched, 64-byte input)\n";
        } else {
            std::cout << "  GPU buffer alloc failed: "
                      << cudaGetErrorString(cerr) << "\n";
        }
        if (d_in)      cudaFree(d_in);
        if (d_offsets) cudaFree(d_offsets);
        if (d_lengths) cudaFree(d_lengths);
        if (d_out)     cudaFree(d_out);
    }
#else
    std::cout << "  No GPU backend compiled in (CPU-only build).\n";
#endif

    std::cout << "\nFor sustained-rate Kangaroo throughput on this hardware,\n";
    std::cout << "connect to the live pool:\n";
    std::cout << "  ./collider --pool jlps://collisionprotocol.com:17403 --worker bc1q...\n";
    std::cout << "\nFor the full brain-wallet pipeline benchmark, theCollider Pro\n";
    std::cout << "exercises SHA256 -> EC -> hash160 -> bloom across all GPUs.\n";
    std::cout << "https://collisionprotocol.com/pro\n";
    return 0;
}

#else  // COLLIDER_PRO

// In Pro builds the Free SHA-256 benchmark is not callable: the call
// site in run_benchmark is itself #ifndef COLLIDER_PRO. The decl in
// puzzle_solver_helpers.hpp stays visible so the header doesn't need
// edition-aware gating; this stub satisfies the linker only if some
// future Pro-side caller appears (today none does).
int run_sha256_only_benchmark(const Arguments& /*args*/,
                              const GPUDetectionResult& /*gpu_info*/) {
    return 0;
}

#endif  // !COLLIDER_PRO

}  // namespace collider::runtime::detail
