/**
 * Free-licensed Metal SHA-256 throughput benchmark.
 *
 * Returns the sustained H/s measured against the Apple GPU over
 * `seconds` wall-clock seconds with `batch_size` 64-byte inputs per
 * dispatch. The kernel is `sha256_bench` in src/gpu/sha256.metal.
 *
 * Implementation lives in src/gpu/sha256_metal_bench.mm; this header
 * is the only Free/Pro-shared declaration. Compiled only on macOS
 * (APPLE && COLLIDER_USE_METAL).
 */

#pragma once

#include <cstdint>
#include <string>

namespace collider {
namespace gpu {

struct Sha256MetalBenchResult {
    bool        ok = false;
    double      hashes_per_second = 0.0;
    uint64_t    total_hashes = 0;
    double      elapsed_seconds = 0.0;
    std::string device_name;
    std::string error;
};

Sha256MetalBenchResult run_sha256_metal_benchmark(
    int seconds = 5,
    uint32_t batch_size = 1u << 18);   // 262144 inputs/dispatch by default

}  // namespace gpu
}  // namespace collider
