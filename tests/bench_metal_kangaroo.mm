/**
 * Metal Kangaroo throughput benchmark.
 *
 * Drives the production-shaped config (1024 kangaroos x 1024 steps,
 * 32 rounds = ~33M ops) so dispatch overhead is amortized and the
 * reported Mops/s reflects the actual kernel throughput. Used as the
 * baseline for Phase 5 Montgomery batch inversion: pre-change run
 * captures the affine 1-inversion-per-step throughput; post-change
 * re-run measures the speedup.
 *
 * Mac-only. ctest target: MetalKangarooBench (mark as long-running).
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/gpu/kangaroo_metal.hpp"
#include "../src/core/crypto_cpu.hpp"
#include "../src/core/byte_codec.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

std::array<collider::gpu::KangarooSeed, collider::gpu::kJumpTableSize>
make_bench_jumps()
{
    std::array<collider::gpu::KangarooSeed, collider::gpu::kJumpTableSize> jumps{};
    for (size_t i = 0; i < collider::gpu::kJumpTableSize; ++i) {
        ::collider::cpu::uint256_t scalar;
        scalar.d[0] = static_cast<uint64_t>(i + 1);
        scalar.d[1] = scalar.d[2] = scalar.d[3] = 0;

        ::collider::cpu::ECPoint p;
        ::collider::cpu::ec_mul(p, scalar);
        ::collider::cpu::uint256_t px, py;
        ::collider::cpu::ec_to_affine(px, py, p);

        ::collider::limbs_le_to_be32(px.d,    jumps[i].x.data());
        ::collider::limbs_le_to_be32(py.d,    jumps[i].y.data());
        ::collider::limbs_le_to_be32(scalar.d, jumps[i].d.data());
        jumps[i].type = 0;
    }
    return jumps;
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::fprintf(stderr, "no Metal device, skipping\n");
            return 77;
        }

        // Production-shaped config. dp_bits=20 keeps DP rate sane on
        // this scale (~1/1M chance per step).
        collider::gpu::KangarooMetalConfig cfg;
        cfg.num_kangaroos    = 1024;
        cfg.steps_per_round  = 1024;
        cfg.dp_bits          = 20;
        cfg.work_id          = 0xbeefcafe12345678ULL;
        cfg.dp_max_per_round = 4096;

        // Allow CLI override for quick tuning experiments:
        //   bench_metal_kangaroo [num_kangaroos] [steps] [rounds]
        const uint32_t rounds = (argc >= 4) ? (uint32_t)std::atoi(argv[3]) : 32;
        if (argc >= 2) cfg.num_kangaroos   = (uint32_t)std::atoi(argv[1]);
        if (argc >= 3) cfg.steps_per_round = (uint32_t)std::atoi(argv[2]);

        collider::gpu::KangarooMetalSolver solver;
        if (!solver.init(cfg)) {
            std::fprintf(stderr, "init failed: %s\n", solver.error().c_str());
            return 1;
        }

        auto jumps = make_bench_jumps();
        if (!solver.set_jump_table(jumps)) {
            std::fprintf(stderr, "set_jump_table failed: %s\n",
                         solver.error().c_str());
            return 1;
        }

        std::vector<collider::gpu::KangarooSeed> seeds;
        seeds.reserve(cfg.num_kangaroos);
        for (uint32_t i = 0; i < cfg.num_kangaroos; ++i) {
            ::collider::cpu::uint256_t scalar;
            scalar.d[0] = static_cast<uint64_t>(1000000ULL + i);
            scalar.d[1] = scalar.d[2] = scalar.d[3] = 0;
            ::collider::cpu::ECPoint p;
            ::collider::cpu::ec_mul(p, scalar);
            ::collider::cpu::uint256_t px, py;
            ::collider::cpu::ec_to_affine(px, py, p);

            collider::gpu::KangarooSeed s{};
            ::collider::limbs_le_to_be32(px.d,    s.x.data());
            ::collider::limbs_le_to_be32(py.d,    s.y.data());
            ::collider::limbs_le_to_be32(scalar.d, s.d.data());
            s.type = static_cast<uint8_t>(i & 1);
            seeds.push_back(s);
        }
        if (!solver.seed_kangaroos(seeds)) {
            std::fprintf(stderr, "seed_kangaroos failed: %s\n",
                         solver.error().c_str());
            return 1;
        }

        // Warm-up round (not counted): kernel JIT, buffer allocation.
        (void)solver.step_round();

        // Timed rounds.
        uint64_t total_dp = 0;
        const auto wall_start = std::chrono::steady_clock::now();
        for (uint32_t r = 0; r < rounds; ++r) {
            auto dps = solver.step_round();
            total_dp += dps.size();
        }
        const auto wall_end = std::chrono::steady_clock::now();

        const double wall_secs =
            std::chrono::duration<double>(wall_end - wall_start).count();
        const uint64_t total_ops =
            static_cast<uint64_t>(cfg.num_kangaroos) *
            cfg.steps_per_round * rounds;
        const double mops = (wall_secs > 0.0)
            ? (static_cast<double>(total_ops) / 1e6 / wall_secs) : 0.0;

        std::printf("MetalKangarooBench: kangaroos=%u steps=%u rounds=%u "
                    "ops=%llu DPs=%llu wall=%.3fs throughput=%.2f Mops/s\n",
                    cfg.num_kangaroos, cfg.steps_per_round, rounds,
                    (unsigned long long)total_ops,
                    (unsigned long long)total_dp,
                    wall_secs, mops);
        return 0;
    }
}
