// Full-pipeline brain-wallet benchmark shared between --benchmark and the
// standalone benchmarks/bench_gpu_pipeline.cu driver. Exposes one entry
// point that times each kernel stage (SHA-256, secp256k1 scalar multiply,
// hash160, bloom probe, fused end-to-end) on a single GPU using a
// pre-loaded fixed-stride passphrase buffer.
// The benchmark intentionally:
//   - Pre-fills the input buffer ONCE with deterministic synthetic bytes.
//     Each iteration re-runs the kernel against the same on-device buffer
//     so host-to-device transfer cost is excluded from the steady-state
//     stage rates.
//   - Uses an empty bloom (no insertions). The bloom probe still executes
//     the full per-passphrase hash chain (MurmurHash3-128 x num_hashes
//     lookups) so the probe rate is real; the only thing the empty bloom
//     skips is the rare match-recording write.
//   - Runs each stage in a separate timed loop so the per-stage number is
//     not contaminated by the others, and additionally runs the fused
//     kernel for the end-to-end "keys/s" figure that operators see in
//     production scans.
// This is the source of truth for the throughput numbers used in
// docs/PRO.md and README.md. The standalone driver and the --benchmark
// CLI flag both call run_pipeline_benchmark so the output table cannot
// drift between the two surfaces.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace collider {
namespace runtime {
namespace bench {

struct PipelineBenchConfig {
    // Which GPU to bench (single-GPU only; aggregate numbers come from
    // running the bench per GPU and summing).
    int gpu_id = 0;

    // Wall-clock seconds per stage. Each stage gets its own bench_seconds
    // window plus a brief warmup; total runtime is roughly
    // bench_seconds * (num_stages + 1).
    int bench_seconds = 30;

    // Per-dispatch batch size. The default matches the production
    // brain-wallet runner default (4M passphrases per kernel call).
    std::size_t batch_size = 4'000'000;

    // Bytes between consecutive passphrases in the fixed-stride buffer.
    // 64 bytes covers realistic passphrase lengths and keeps VRAM use
    // modest (batch_size * stride = 256 MiB at the defaults).
    std::uint32_t stride = 64;

    // Synthetic passphrase length used for every slot. 16 bytes is
    // representative of brain-wallet candidate lengths; the SHA-256 cost
    // is dominated by the per-block schedule once length exceeds zero,
    // so the absolute value matters less than keeping it under one block.
    std::uint32_t passphrase_len = 16;

    // If true, also bench the per-stage kernels in isolation. If false,
    // only the fused end-to-end number is produced (cheaper, ~bench_seconds
    // total). Default true matches the marketing-claim requirement.
    bool measure_stages = true;
};

struct StageResult {
    std::string name;            // e.g. "SHA-256", "secp256k1 mul"
    double seconds = 0.0;        // wall-clock elapsed inside the bench loop
    std::uint64_t items = 0;     // total passphrases / pubkeys processed
    double rate_per_sec = 0.0;   // items / seconds
};

struct PipelineBenchResult {
    int gpu_id = 0;
    std::string gpu_name;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    std::size_t batch_size = 0;
    std::uint32_t stride = 0;

    // Per-stage figures (empty when measure_stages == false except for
    // the end-to-end fused entry which is always populated).
    std::vector<StageResult> stages;

    // Fused end-to-end rate (passphrase keys per second).
    double fused_keys_per_sec = 0.0;
    double fused_seconds = 0.0;
    std::uint64_t fused_items = 0;

    bool ok = false;
    std::string error;
};

// Run the benchmark on one GPU. Prints progress to stdout if `verbose`
// is true; the structured result is always returned. Allocates device
// buffers internally and frees them before returning. Initializes the
// fused pipeline (loads the secp256k1 precomputed table) as a one-time
// setup cost; that cost is excluded from the timed loops.
PipelineBenchResult run_pipeline_benchmark(const PipelineBenchConfig& config,
                                           bool verbose);

// Render a single result block as a fixed-width text table. Used by both
// the --benchmark CLI path and the standalone driver so the output is
// byte-identical.
void print_result_table(const PipelineBenchResult& result);

}  // namespace bench
}  // namespace runtime
}  // namespace collider
