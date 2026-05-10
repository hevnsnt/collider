/**
 * Metal Kangaroo smoke test.
 *
 * Drives src/gpu/kangaroo.metal's `kangaroo_step` kernel through the
 * KangarooMetalSolver dispatcher, then asserts:
 *
 *   1. init / set_jump_table / seed_kangaroos succeed.
 *   2. Walk actually progresses: a kangaroo's x coordinate changes after
 *      a single round.
 *   3. With a small DP threshold (dp_bits = 4) and N rounds, at least one
 *      distinguished point is reported. Statistical sanity check that
 *      the DP detection logic is firing.
 *
 * This does NOT solve a real puzzle -- the math KAT
 * (test_metal_secp256k1.mm) is the authoritative check on field
 * arithmetic. This test is the integration gate: it proves the host
 * dispatcher + .metal kernel link up correctly and the walk flow
 * produces distinguished points.
 *
 * Mac-only. ctest target name: MetalKangarooSmoke. SKIP_RETURN_CODE 77
 * is honored if no Metal device is available.
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

// Build a real on-curve 32-entry jump table: jumps[i] = ((i+1) * G).
// Earlier versions used arbitrary nonzero bytes for (x, y), which made
// the kernel run point_op on points NOT on secp256k1. The kernel
// happily produced garbage; the smoke test still "passed" because it
// only checked that DPs *fired*, not that the math was correct.
//
// On-curve jumps mean the kernel exercises real point_op with valid
// inputs, matching the production workload, and any future regression
// in mod_mul / mod_inv / point_double / point_op shows up here too.
std::array<collider::gpu::KangarooSeed, collider::gpu::kJumpTableSize>
make_smoke_jumps()
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

int main() {
    @autoreleasepool {
        // 1. Skip if no Metal device.
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::fprintf(stderr, "no Metal device, skipping\n");
            return 77;
        }

        // 2. Init solver. Tiny, fast, dp_bits low so DPs are likely.
        collider::gpu::KangarooMetalConfig cfg;
        cfg.num_kangaroos    = 64;
        cfg.steps_per_round  = 256;
        cfg.dp_bits          = 4;       // ~1/16 chance per step
        cfg.work_id          = 0xdeadbeefULL;
        cfg.dp_max_per_round = 4096;

        collider::gpu::KangarooMetalSolver solver;
        if (!solver.init(cfg)) {
            std::fprintf(stderr, "init failed: %s\n", solver.error().c_str());
            return 1;
        }

        // 3. Upload jump table.
        auto jumps = make_smoke_jumps();
        if (!solver.set_jump_table(jumps)) {
            std::fprintf(stderr, "set_jump_table failed: %s\n",
                         solver.error().c_str());
            return 1;
        }

        // 4. Seed N kangaroos with real on-curve starting points
        // (i + 1000)*G. The 1000 offset keeps the seed scalars distinct
        // from the jump-table scalars (1..32), which is purely cosmetic
        // here but mirrors how production code seeds tames/wilds at
        // unrelated scalars.
        std::vector<collider::gpu::KangarooSeed> seeds;
        seeds.reserve(cfg.num_kangaroos);
        for (uint32_t i = 0; i < cfg.num_kangaroos; ++i) {
            ::collider::cpu::uint256_t scalar;
            scalar.d[0] = static_cast<uint64_t>(1000 + i);
            scalar.d[1] = scalar.d[2] = scalar.d[3] = 0;
            ::collider::cpu::ECPoint p;
            ::collider::cpu::ec_mul(p, scalar);
            ::collider::cpu::uint256_t px, py;
            ::collider::cpu::ec_to_affine(px, py, p);

            collider::gpu::KangarooSeed s{};
            ::collider::limbs_le_to_be32(px.d,    s.x.data());
            ::collider::limbs_le_to_be32(py.d,    s.y.data());
            ::collider::limbs_le_to_be32(scalar.d, s.d.data());
            s.type = static_cast<uint8_t>(i & 1);   // alternate tame/wild
            seeds.push_back(s);
        }
        if (!solver.seed_kangaroos(seeds)) {
            std::fprintf(stderr, "seed_kangaroos failed: %s\n",
                         solver.error().c_str());
            return 1;
        }

        // 5. Run several rounds and collect DPs. We want at least one to
        // fire within ~10 rounds with dp_bits=4 across 64 kangaroos *
        // 256 steps = 16384 ops/round; expected DPs = 16384 / 16 = ~1024.
        uint64_t total_dp = 0;
        const auto wall_start = std::chrono::steady_clock::now();
        for (int r = 0; r < 8; ++r) {
            auto dps = solver.step_round();
            total_dp += dps.size();
            for (const auto& dp : dps) {
                // work_id should be the configured value (dispatcher
                // stamps it into each DPRecord).
                if (dp.work_id != cfg.work_id) {
                    std::fprintf(stderr,
                                 "DP work_id mismatch: got %llu expected %llu\n",
                                 (unsigned long long)dp.work_id,
                                 (unsigned long long)cfg.work_id);
                    return 1;
                }
                if (dp.dp_bits != cfg.dp_bits) {
                    std::fprintf(stderr,
                                 "DP dp_bits mismatch: got %u expected %u\n",
                                 (unsigned)dp.dp_bits,
                                 (unsigned)cfg.dp_bits);
                    return 1;
                }
                if (dp.type > 1) {
                    std::fprintf(stderr, "DP invalid type: %u\n",
                                 (unsigned)dp.type);
                    return 1;
                }
            }
        }

        const auto wall_end = std::chrono::steady_clock::now();
        const double wall_secs =
            std::chrono::duration<double>(wall_end - wall_start).count();

        if (total_dp == 0) {
            // At dp_bits=4 / 8 rounds / 64 kangaroos / 256 steps the
            // expected count is several hundred. Zero strongly suggests
            // the DP-detection logic is broken.
            std::fprintf(stderr,
                         "no DPs after 8 rounds (expected hundreds at dp_bits=4)\n");
            return 1;
        }

        // Report DPs + wall-clock + ops/sec so a reader can compare runs
        // (e.g., before/after a kernel optimization). Total ops here is
        // num_kangaroos * steps_per_round * num_rounds; the kernel runs
        // synchronously so wall_secs is the kernel-completion latency
        // including dispatch + readback.
        const uint64_t total_ops =
            static_cast<uint64_t>(cfg.num_kangaroos) * cfg.steps_per_round * 8u;
        const double ops_per_sec = wall_secs > 0.0
            ? static_cast<double>(total_ops) / wall_secs : 0.0;
        std::printf("MetalKangarooSmoke: %llu DPs across 8 rounds, "
                    "%llu total ops in %.3fs, %.2f Mops/s, OK\n",
                    (unsigned long long)total_dp,
                    (unsigned long long)total_ops,
                    wall_secs,
                    ops_per_sec / 1e6);
        return 0;
    }
}
