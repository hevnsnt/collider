// Standalone brain-wallet GPU pipeline benchmark.
//
// Drives src/runtime/bench_pipeline.{hpp,cpp} so the numbers reported here
// are byte-identical to what `collider_pro --benchmark` prints. The driver
// is a thin CLI wrapper: parse `--time`, `--gpu`, `--batch`; call
// run_pipeline_benchmark; print the result table.
//
// Usage:
//   bench_gpu_pipeline                # 30s per stage, GPU 0, 4M batch
//   bench_gpu_pipeline --time 5       # 5s per stage (smoke)
//   bench_gpu_pipeline --gpu 1        # bench a specific GPU
//   bench_gpu_pipeline --batch 2000000
//   bench_gpu_pipeline --fused-only   # skip per-stage timings
//
// The .cu extension keeps the toolchain selection consistent with the rest
// of the benchmark targets and means the standalone binary picks up the
// CUDA runtime library automatically.

#include "../src/runtime/bench_pipeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

void print_usage() {
    std::cout <<
        "bench_gpu_pipeline - theCollider full-pipeline benchmark\n"
        "\n"
        "Options:\n"
        "  --time SECONDS    Wall-clock seconds per stage (default 30)\n"
        "  --gpu N           GPU index to bench (default 0)\n"
        "  --batch N         Passphrases per kernel dispatch (default 4M)\n"
        "  --stride N        Bytes per slot in the fixed-stride buffer (default 64)\n"
        "  --pp-len N        Synthetic passphrase length (default 16)\n"
        "  --fused-only      Skip per-stage timings, run only the end-to-end\n"
        "                    fused kernel\n"
        "  -h, --help        Print this help and exit\n";
}

bool parse_int(const char* s, int& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
}

bool parse_size(const char* s, std::size_t& out) {
    if (s == nullptr || *s == '\0') return false;
    char* end = nullptr;
    long long v = std::strtoll(s, &end, 10);
    if (end == s || *end != '\0' || v < 0) return false;
    out = static_cast<std::size_t>(v);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    collider::runtime::bench::PipelineBenchConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << name << " requires a value\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            print_usage();
            return 0;
        } else if (a == "--time") {
            if (!parse_int(need_value("--time"), cfg.bench_seconds) ||
                cfg.bench_seconds <= 0) {
                std::cerr << "--time must be a positive integer\n";
                return 2;
            }
        } else if (a == "--gpu") {
            if (!parse_int(need_value("--gpu"), cfg.gpu_id) || cfg.gpu_id < 0) {
                std::cerr << "--gpu must be a non-negative integer\n";
                return 2;
            }
        } else if (a == "--batch") {
            std::size_t b = 0;
            if (!parse_size(need_value("--batch"), b) || b == 0) {
                std::cerr << "--batch must be a positive integer\n";
                return 2;
            }
            cfg.batch_size = b;
        } else if (a == "--stride") {
            int v = 0;
            if (!parse_int(need_value("--stride"), v) || v <= 0 || v > 4096) {
                std::cerr << "--stride must be in (0, 4096]\n";
                return 2;
            }
            cfg.stride = static_cast<std::uint32_t>(v);
        } else if (a == "--pp-len") {
            int v = 0;
            if (!parse_int(need_value("--pp-len"), v) || v < 0 || v > 4096) {
                std::cerr << "--pp-len must be in [0, 4096]\n";
                return 2;
            }
            cfg.passphrase_len = static_cast<std::uint32_t>(v);
        } else if (a == "--fused-only") {
            cfg.measure_stages = false;
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage();
            return 2;
        }
    }

    if (cfg.passphrase_len > cfg.stride) {
        std::cerr << "--pp-len (" << cfg.passphrase_len
                  << ") must be <= --stride (" << cfg.stride << ")\n";
        return 2;
    }

    auto result = collider::runtime::bench::run_pipeline_benchmark(cfg,
                                                                   /*verbose=*/true);
    collider::runtime::bench::print_result_table(result);
    return result.ok ? 0 : 1;
}
