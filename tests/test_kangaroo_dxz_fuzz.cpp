/**
 * test_kangaroo_dxz_fuzz -- pins the algebraic correctness of the P-B1
 * dx==0 SIMT guard from src/gpu/kangaroo_kernel.cu (kangaroo_step_kernel_jlp
 * and kangaroo_step_kernel_sota, ~lines 1125-1180 and 1350-1450).
 *
 * Background
 * ----------
 * The JLP / SOTA kangaroo kernels run GPU_GRP_SIZE kangaroos per thread.
 * On every step, each lane g computes dx[g] = jump_x - px[g], then a
 * single batch Montgomery inversion is performed across the entire group
 * (one mod_inv + ~3*GPU_GRP_SIZE mod_mul, vs GPU_GRP_SIZE independent
 * mod_invs). The batch inversion's product chain is
 *
 *     prod[i] = dx[0] * dx[1] * ... * dx[i]
 *
 * If any dx[g] == 0 the product chain zeros out at lane g and every
 * downstream inversion becomes 0 -- every kangaroo in the SIMT group
 * walks pure garbage on the next step. P-B1 added the guard:
 *
 *     bool is_zero = (dx[g] == 0);
 *     dx_was_zero[g] = is_zero;
 *     if (is_zero) dx[g] = {1, 0, 0, 0};
 *     // ... batch_mod_inv proceeds ...
 *     // post-inversion: lanes with dx_was_zero[g] skip the EC add and
 *     // call rerandomize_kangaroo() instead. Distance is preserved.
 *
 * Why this test exists
 * --------------------
 * test_kangaroo_small_puzzle.cu admits at lines 48-50:
 *
 *     "The P-B1 functional guard is verified by inspection of the
 *      kernel diff and by a follow-up integration test
 *      (test_kangaroo_dxz_fuzz) being added separately by the
 *      runner team."
 *
 * That follow-up test was never added. The guard is the load-bearing
 * piece preventing single-lane dx=0 collisions from poisoning entire
 * SIMT groups; if anyone refactors the kernel and forgets the guard,
 * the puzzle solver still runs but produces ~0 valid DPs.
 *
 * Test design
 * -----------
 * The dx==0 guard is an algebraic property of the batch-inversion
 * pre-processing step. It can be exercised end-to-end without a GPU by
 * mirroring the kernel's product-chain math against the production CPU
 * crypto primitives (collider::cpu::mod_mul, mod_inv) in crypto_cpu.hpp.
 * The same primitives are what the GPU kernel's mod_mul / mod_inv are
 * mathematically equivalent to (the GPU implementation differs only in
 * the carry/shift instructions used; both produce results in [0, p)).
 *
 * For each fuzz iteration we:
 *
 *   1. Generate GPU_GRP_SIZE random non-zero dx values (uniform in [1, p)).
 *   2. Inject ONE dx == 0 at a random lane k (the failure case).
 *   3. Run the guard's substitute-with-1 pre-processing, then run the
 *      Montgomery batch inversion the same way the kernel does.
 *   4. Assert: every lane's inverse satisfies inv * original_dx == 1 mod p
 *      for the non-zero lanes (positive control: the inversion is correct).
 *   5. Assert: dx_was_zero[k] is set; the kernel's downstream code will
 *      see this flag and call rerandomize_kangaroo() instead of doing
 *      the (bogus) EC add. We mirror the rerandomize-vs-add decision and
 *      verify the bogus lane is detected.
 *
 * Without the guard, step 4 would fail every time -- the entire product
 * chain zeroes out at lane k, mod_inv of zero is mathematically undefined
 * (we'd hit Fermat's a^(p-2) with a=0 producing 0), and the back-prop
 * would propagate zero to every lane. The "every non-zero lane should
 * invert correctly" property is a strict witness for guard correctness.
 *
 * Coverage relationship (T3.3 closed the kernel-driver gap, 2026-05-17):
 * --------------------------------------------------------------------
 * This test pins the ALGEBRA of the guard against the CPU primitives.
 * tests/test_kangaroo_dxz_kernel.cu closes the kernel-driver gap: it
 * launches the actual CUDA kernel (via the test_only_kangaroo_dxz_*
 * extern "C" wrapper in kangaroo_kernel.cu) with crafted dx==0 inputs
 * and verifies the same property holds. Both tests should pass; if a
 * refactor breaks only the kernel path while leaving the CPU mirror
 * intact, the kernel-driver test catches it.
 */

#include "core/crypto_cpu.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

// Mirror the production kernel default. The test exercises group sizes
// from 32 to 128 to cover the realistic GPU_GRP_SIZE range (the
// production build picks 64 by default but can be rebuilt at 32 / 128
// per the Tuning Guide at the top of kangaroo_kernel.cu).
constexpr int kFuzzIterations = 64;

// Run the host-side analogue of batch_mod_inv from kangaroo_kernel.cu
// (Montgomery's trick). After this returns, dx[g] holds the modular
// inverse of the value originally in dx[g] (after any guard
// substitutions). Mirrors the kernel implementation step-for-step.
void cpu_batch_mod_inv(std::vector<collider::cpu::uint256_t>& dx) {
    using collider::cpu::uint256_t;
    using collider::cpu::mod_mul;
    using collider::cpu::mod_inv;

    const size_t n = dx.size();
    std::vector<uint256_t> products(n);

    // Forward accumulation: products[i] = dx[0] * ... * dx[i].
    products[0] = dx[0];
    for (size_t g = 1; g < n; g++) {
        mod_mul(products[g], products[g-1], dx[g]);
    }

    // Single inversion of the final product.
    uint256_t inv_all;
    mod_inv(inv_all, products[n-1]);

    // Backward propagation. Mirror the kernel's update order so a
    // subtle off-by-one in the back-prop loop is reflected here.
    for (size_t g = n - 1; g > 0; g--) {
        uint256_t new_inv;
        mod_mul(new_inv, inv_all, products[g-1]);

        uint256_t old_dx = dx[g];
        mod_mul(inv_all, inv_all, old_dx);

        dx[g] = new_inv;
    }
    dx[0] = inv_all;
}

// True if v is the multiplicative identity in F_p (canonical form: 1).
bool is_one(const collider::cpu::uint256_t& v) {
    return v.d[0] == 1 && v.d[1] == 0 && v.d[2] == 0 && v.d[3] == 0;
}

// Sample a uniformly random non-zero field element. We don't need
// rigorous uniformity (the test pins a property, not a distribution);
// rejection sampling against 0 is enough.
collider::cpu::uint256_t random_field_element(std::mt19937_64& rng) {
    collider::cpu::uint256_t v;
    do {
        for (int i = 0; i < 4; i++) {
            v.d[i] = rng();
        }
        // Stay strictly below p so the product chain remains canonical;
        // the field prime's top limb is FFFFFFFFFFFFFFFF so masking the
        // high limb doesn't help. Easiest: reject when v >= p.
    } while (v.is_zero() || !(v < collider::cpu::SECP256K1_P));
    return v;
}

struct FuzzResult {
    int passed;
    int failed;
};

// Run one fuzz iteration with the given group size and zero-injection
// position. Returns true on pass.
bool run_one(int group_size, int zero_lane, std::mt19937_64& rng,
             std::string& fail_reason) {
    using collider::cpu::uint256_t;
    using collider::cpu::mod_mul;

    std::vector<uint256_t> dx(group_size);
    std::vector<uint256_t> original_dx(group_size);
    std::vector<bool> dx_was_zero(group_size, false);

    // Build the batch: random non-zero dx everywhere, then inject zero
    // at zero_lane.
    for (int g = 0; g < group_size; g++) {
        dx[g] = random_field_element(rng);
    }
    if (zero_lane >= 0 && zero_lane < group_size) {
        dx[zero_lane] = uint256_t(0);
    }

    // Snapshot the originals BEFORE the guard pre-process so we can
    // verify inv(dx[g]) * original_dx[g] == 1 for every non-zero lane.
    for (int g = 0; g < group_size; g++) {
        original_dx[g] = dx[g];
    }

    // The P-B1 guard pre-process: detect dx == 0 lanes, substitute
    // dx = 1 so the product chain stays invertible, and flag the lane
    // for downstream rerandomization. This is the EXACT logic from
    // kangaroo_kernel.cu lines 1131-1161.
    for (int g = 0; g < group_size; g++) {
        bool is_zero = dx[g].is_zero();
        dx_was_zero[g] = is_zero;
        if (is_zero) {
            dx[g] = uint256_t(1);
        }
    }

    // Guard property A: every dx_was_zero[g] flag must agree with the
    // pre-substitution input. This is a trivial mirror but pins the
    // guard's flag-setting -- if a future refactor stops setting the
    // flag, the kernel will substitute dx=1 but never trigger the
    // rerandomize fall-out path, leading to silently wrong EC adds.
    for (int g = 0; g < group_size; g++) {
        bool input_was_zero = original_dx[g].is_zero();
        if (dx_was_zero[g] != input_was_zero) {
            fail_reason = "dx_was_zero[" + std::to_string(g) +
                          "] flag mismatch";
            return false;
        }
    }

    // Run Montgomery batch inversion. Without the guard's substitute-1
    // step, this would produce all-zero results (the product chain
    // zeros out at the injected lane).
    cpu_batch_mod_inv(dx);

    // Guard property B: for every NON-zero original lane g, the result
    // must be the multiplicative inverse:
    //     dx[g] * original_dx[g] == 1 mod p
    // If the guard misbehaved (e.g. substituted with 0 instead of 1)
    // the product chain would zero out and every lane's inverse would
    // be 0, failing this check on every non-zero lane.
    for (int g = 0; g < group_size; g++) {
        if (dx_was_zero[g]) continue;  // bogus lane; downstream rerandomizes
        uint256_t check;
        mod_mul(check, dx[g], original_dx[g]);
        if (!is_one(check)) {
            fail_reason = "inv * original_dx != 1 at lane " +
                          std::to_string(g) + " (group_size=" +
                          std::to_string(group_size) + ", zero_lane=" +
                          std::to_string(zero_lane) + ")";
            return false;
        }
    }

    // Guard property C: for the substituted lane(s), dx now holds the
    // inverse of 1, which is 1. (The kernel discards this value via
    // the rerandomize fall-out path, but we still pin that the
    // substitution chain produces the expected mathematical witness.)
    for (int g = 0; g < group_size; g++) {
        if (!dx_was_zero[g]) continue;
        if (!is_one(dx[g])) {
            fail_reason = "lane " + std::to_string(g) +
                          " substituted as dx=1 did not invert to 1";
            return false;
        }
    }

    return true;
}

}  // namespace

int main() {
    std::mt19937_64 rng(0xBEEFFACEDEADC0DEULL);
    FuzzResult result{0, 0};

    // Sweep across the GPU_GRP_SIZE range. The production default is 64;
    // the Tuning Guide notes 32 (Pascal), 64 (Turing/Ampere) and 128
    // (Ada/Blackwell) builds. Cover the boundary cases plus the actual
    // production default.
    const int group_sizes[] = {32, 64, 128};

    std::printf("=== Kangaroo dx==0 SIMT guard fuzz test ===\n");

    for (int gs : group_sizes) {
        std::printf("\n[ Group size = %d ]\n", gs);

        // Sub-case 1: no zero injection (positive control). All inverses
        // must be correct. If this fails, the underlying batch_mod_inv
        // mirror is broken -- not a P-B1 guard issue.
        {
            std::string why;
            if (run_one(gs, /*zero_lane*/-1, rng, why)) {
                std::printf("  PASS  no-zero positive control\n");
                result.passed++;
            } else {
                std::printf("  FAIL  no-zero positive control: %s\n",
                            why.c_str());
                result.failed++;
            }
        }

        // Sub-case 2: zero at boundary lanes (0 and gs-1) -- these stress
        // the product-chain head and tail.
        for (int lane : {0, gs - 1}) {
            std::string why;
            if (run_one(gs, lane, rng, why)) {
                std::printf("  PASS  zero injected at lane %d (boundary)\n",
                            lane);
                result.passed++;
            } else {
                std::printf("  FAIL  zero injected at lane %d: %s\n",
                            lane, why.c_str());
                result.failed++;
            }
        }

        // Sub-case 3: random zero-lane positions across many iterations.
        // Each iteration draws a fresh GROUP_SIZE-element dx batch and
        // injects one zero at a random position.
        std::uniform_int_distribution<int> lane_dist(0, gs - 1);
        int sub_pass = 0, sub_fail = 0;
        std::string first_fail;
        for (int it = 0; it < kFuzzIterations; ++it) {
            int lane = lane_dist(rng);
            std::string why;
            if (run_one(gs, lane, rng, why)) {
                ++sub_pass;
            } else {
                ++sub_fail;
                if (first_fail.empty()) first_fail = why;
            }
        }
        if (sub_fail == 0) {
            std::printf("  PASS  %d random-lane fuzz iterations\n", sub_pass);
            result.passed++;
        } else {
            std::printf("  FAIL  %d/%d random-lane fuzz iterations (first: %s)\n",
                        sub_fail, sub_pass + sub_fail, first_fail.c_str());
            result.failed++;
        }
    }

    std::printf("\n=== Results: %d passed, %d failed ===\n",
                result.passed, result.failed);
    return result.failed == 0 ? 0 : 1;
}
