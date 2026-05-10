/**
 * Metal Kangaroo smoke test.
 *
 * Drives src/gpu/kangaroo.metal's `kangaroo_step` kernel through the
 * KangarooMetalSolver dispatcher, then asserts:
 *
 *   1. init / set_jump_table / seed_kangaroos succeed.
 *   2. With a small DP threshold (dp_bits = 4) and N rounds, at least one
 *      distinguished point is reported. Statistical sanity check that
 *      the DP detection logic is firing.
 *   3. v1.4.1 JACOBIAN PATH VERIFICATION: every reported DP's affine X
 *      satisfies the secp256k1 curve equation y^2 = x^3 + 7 (mod p)
 *      AND has the dp_bits-leading-zero property. If the Jacobian
 *      kernel is buggy (wrong point arithmetic, batch inversion off
 *      by one, sentinel state leaking into output), one of these
 *      checks catches it. The previous affine kernel passed both
 *      trivially; with Jacobian + batch inversion this is the
 *      integration gate.
 *
 * This does NOT solve a real puzzle -- the math KAT
 * (test_metal_secp256k1.mm) is the authoritative check on field
 * arithmetic. This test is the integration gate: it proves the host
 * dispatcher + .metal kernel link up correctly and the walk flow
 * produces distinguished points whose affine X coordinates are
 * recovered correctly from Jacobian via batch inversion.
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
        uint64_t dp_oncurve_checks = 0;
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

                // v1.4.1 Jacobian path verification: dp.x_be is the
                // affine X coordinate recovered via threadgroup batch
                // inversion. Two checks must hold:
                //   (a) The dp_bits MSBs of x_be are zero (DP property).
                //   (b) There exists a y on secp256k1 such that
                //       y^2 = x^3 + 7 (mod p), i.e. x is on the curve.
                // (a) catches a misaligned is_distinguished() check;
                // (b) catches Jacobian arithmetic bugs (wrong batch
                // inversion would produce off-curve x).
                {
                    // Check (a): dp_bits leading zero bits in big-endian X.
                    const uint8_t* xb = dp.x_be;
                    uint32_t bits_to_check = dp.dp_bits;
                    uint32_t byte_off = 0;
                    bool dp_property_ok = true;
                    while (bits_to_check >= 8) {
                        if (xb[byte_off] != 0) { dp_property_ok = false; break; }
                        ++byte_off; bits_to_check -= 8;
                    }
                    if (dp_property_ok && bits_to_check > 0) {
                        const uint8_t mask =
                            static_cast<uint8_t>(0xFFu << (8u - bits_to_check));
                        if ((xb[byte_off] & mask) != 0) dp_property_ok = false;
                    }
                    if (!dp_property_ok) {
                        std::fprintf(stderr,
                            "DP property failed: dp_bits=%u but X has nonzero "
                            "leading bits (Jacobian->affine conversion likely "
                            "buggy)\n", (unsigned)dp.dp_bits);
                        return 1;
                    }

                    // Check (b): X must satisfy y^2 = x^3 + 7 mod p for some y.
                    // Equivalently, x^3 + 7 must be a quadratic residue mod p.
                    // We compute it and check via Euler's criterion: a is QR
                    // iff a^((p-1)/2) == 1 (mod p). If x came out of a real
                    // walk, x^3 + 7 is automatically a QR. If the Jacobian
                    // arithmetic produced garbage, x^3+7 would be a QR with
                    // probability ~1/2; over many DPs the test catches the
                    // bug statistically. We also accept the special case
                    // x^3 + 7 == 0 (would be a 2-torsion point; not on
                    // secp256k1 since p is odd, so this should never fire).
                    ::collider::cpu::uint256_t x_le;
                    ::collider::be32_to_limbs_le(xb, x_le.d);

                    ::collider::cpu::uint256_t x2_le, x3_le, rhs_le;
                    ::collider::cpu::mod_mul(x2_le, x_le, x_le);
                    ::collider::cpu::mod_mul(x3_le, x2_le, x_le);
                    ::collider::cpu::mod_add(rhs_le, x3_le,
                                             ::collider::cpu::uint256_t(7));

                    // Euler's criterion exponent (p-1)/2 for secp256k1:
                    // 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF7FFFFE17
                    const ::collider::cpu::uint256_t euler_exp(
                        0xFFFFFFFF7FFFFE17ULL,
                        0xFFFFFFFFFFFFFFFFULL,
                        0xFFFFFFFFFFFFFFFFULL,
                        0x7FFFFFFFFFFFFFFFULL);
                    ::collider::cpu::uint256_t legendre;
                    ::collider::cpu::mod_pow(legendre, rhs_le, euler_exp);

                    // QR iff legendre == 1. (legendre == p-1 means non-residue.)
                    const bool is_qr =
                        (legendre.d[0] == 1 && legendre.d[1] == 0 &&
                         legendre.d[2] == 0 && legendre.d[3] == 0);
                    if (!is_qr) {
                        std::fprintf(stderr,
                            "DP off-curve: x^3+7 is NOT a quadratic residue "
                            "mod p. Jacobian->affine batch inversion is "
                            "producing garbage X coordinates. Round %d.\n", r);
                        return 1;
                    }
                    ++dp_oncurve_checks;
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
        std::printf("MetalKangarooSmoke: %llu DPs across 8 rounds "
                    "(%llu Jacobian-path verified on-curve), "
                    "%llu total ops in %.3fs, %.2f Mops/s, OK\n",
                    (unsigned long long)total_dp,
                    (unsigned long long)dp_oncurve_checks,
                    (unsigned long long)total_ops,
                    wall_secs,
                    ops_per_sec / 1e6);
        return 0;
    }
}
