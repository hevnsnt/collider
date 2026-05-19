/**
 * test_kangaroo_dxz_kernel: T3.3 (2026-05-17). Kernel-driver complement
 * to test_kangaroo_dxz_fuzz.cpp.
 *
 * The CPU-mirror test (tests/test_kangaroo_dxz_fuzz.cpp) pins the
 * algebraic correctness of the dx==0 SIMT guard by replaying the
 * batch-inversion product chain on the host with the production CPU
 * mod_mul / mod_inv primitives. That covers the math but does NOT
 * verify the guard is actually present in the kernel binary. A
 * refactor that drops the guard from kangaroo_step_kernel_jlp /
 * kangaroo_step_kernel_sota while leaving the CPU mirror intact
 * would still pass the existing test.
 *
 * This test closes the gap by launching the test-only kernel-driver
 * entry point `test_only_kangaroo_dxz_guard_launch` (defined in
 * src/gpu/kangaroo_kernel.cu) with crafted dx==0 inputs and
 * verifying the SIMT guard fires: lanes with dx==0 are flagged via
 * the dx_was_zero[] mask and their substituted-to-1 entries invert
 * back to 1, while every other lane's inverse satisfies
 * dx_inv * dx_original == 1 (mod p). Without the guard the entire
 * batch's back-propagated inverses would be zero, a strict witness
 * the test fails on.
 *
 * Coverage relationship:
 *   - test_kangaroo_dxz_fuzz.cpp: HOST mirror; pins algebra.
 *   - test_kangaroo_dxz_kernel.cu (this file): KERNEL driver; pins
 *     that the algebra runs on the actual CUDA path.
 *
 * The kernel launches one thread per dispatch because the guard runs
 * within one SIMT group. The host test invokes the kernel many times
 * with different (input, zero-lane) combinations to cover boundary
 * and random injection positions.
 */

#include "core/crypto_cpu.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

// Defined in src/gpu/kangaroo_kernel.cu (extern "C" wrapper).
extern "C" cudaError_t test_only_kangaroo_dxz_guard_launch(
    const void* d_input_dx,
    void*       d_output_dx,
    uint8_t*    d_dx_was_zero,
    int*        out_gpu_grp_size,
    cudaStream_t stream);

namespace {

constexpr int kFuzzIterations = 32;

bool is_one(const collider::cpu::uint256_t& v) {
    return v.d[0] == 1 && v.d[1] == 0 && v.d[2] == 0 && v.d[3] == 0;
}

collider::cpu::uint256_t random_field_element(std::mt19937_64& rng) {
    collider::cpu::uint256_t v;
    do {
        for (int i = 0; i < 4; ++i) v.d[i] = rng();
    } while (v.is_zero() || !(v < collider::cpu::SECP256K1_P));
    return v;
}

// Run one kernel launch with the given input dx batch (host-side).
// Returns true on pass. zero_lanes lists indices where dx was zeroed
// before the kernel sees them (the guard must flag and substitute).
bool drive_kernel_once(int grp_size,
                       std::vector<collider::cpu::uint256_t>& dx_host,
                       const std::vector<int>& zero_lanes,
                       std::string& fail_reason) {
    using collider::cpu::uint256_t;
    using collider::cpu::mod_mul;

    // Snapshot pre-launch dx values (post zero-injection but
    // before the guard substitutes 1) so we can verify per-lane.
    std::vector<uint256_t> original_dx = dx_host;

    // Pack into device-friendly limb layout. The kernel expects
    // GPU_GRP_SIZE entries; we always supply the full batch (the
    // caller validates that dx_host.size() matches the kernel's
    // compiled GPU_GRP_SIZE).
    std::vector<uint64_t> h_in(grp_size * 4, 0);
    for (int g = 0; g < grp_size; ++g) {
        for (int i = 0; i < 4; ++i) {
            h_in[g * 4 + i] = dx_host[g].d[i];
        }
    }

    uint64_t* d_in = nullptr;
    uint64_t* d_out = nullptr;
    uint8_t*  d_zero = nullptr;
    cudaError_t err;

    err = cudaMalloc(&d_in, grp_size * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) {
        fail_reason = std::string("cudaMalloc d_in failed: ") +
                      cudaGetErrorString(err);
        return false;
    }
    err = cudaMalloc(&d_out, grp_size * 4 * sizeof(uint64_t));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        fail_reason = std::string("cudaMalloc d_out failed: ") +
                      cudaGetErrorString(err);
        return false;
    }
    err = cudaMalloc(&d_zero, grp_size * sizeof(uint8_t));
    if (err != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out);
        fail_reason = std::string("cudaMalloc d_zero failed: ") +
                      cudaGetErrorString(err);
        return false;
    }

    err = cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(uint64_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_zero);
        fail_reason = std::string("cudaMemcpy H2D failed: ") +
                      cudaGetErrorString(err);
        return false;
    }

    int reported_grp = 0;
    err = test_only_kangaroo_dxz_guard_launch(
        d_in, d_out, d_zero, &reported_grp, 0);
    if (err != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_zero);
        fail_reason = std::string("kernel launch failed: ") +
                      cudaGetErrorString(err);
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_zero);
        fail_reason = std::string("device sync failed (kernel exec error): ") +
                      cudaGetErrorString(err);
        return false;
    }

    if (reported_grp != grp_size) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_zero);
        fail_reason = "kernel reported GPU_GRP_SIZE=" +
                      std::to_string(reported_grp) +
                      " but test was built for " +
                      std::to_string(grp_size);
        return false;
    }

    std::vector<uint64_t> h_out(grp_size * 4);
    std::vector<uint8_t>  h_zero(grp_size);
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_zero.data(), d_zero, h_zero.size() * sizeof(uint8_t),
               cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_zero);

    // Verify the dx_was_zero mask matches the zero-lane set.
    std::vector<bool> expected_zero(grp_size, false);
    for (int lane : zero_lanes) {
        if (lane < 0 || lane >= grp_size) continue;
        expected_zero[lane] = true;
    }
    for (int g = 0; g < grp_size; ++g) {
        bool kernel_flagged = (h_zero[g] != 0);
        if (kernel_flagged != expected_zero[g]) {
            fail_reason = "dx_was_zero[" + std::to_string(g) +
                          "] mismatch: kernel reported " +
                          std::to_string((int)kernel_flagged) +
                          " expected " + std::to_string((int)expected_zero[g]);
            return false;
        }
    }

    // Verify every NON-zero lane's inverse satisfies inv * original == 1.
    // For zero lanes the kernel substituted dx=1, so the returned value
    // should be 1 (the multiplicative inverse of 1).
    for (int g = 0; g < grp_size; ++g) {
        uint256_t got;
        for (int i = 0; i < 4; ++i) got.d[i] = h_out[g * 4 + i];

        if (expected_zero[g]) {
            if (!is_one(got)) {
                fail_reason = "substituted lane " + std::to_string(g) +
                              " did not invert to 1 (got [" +
                              std::to_string(got.d[3]) + " " +
                              std::to_string(got.d[2]) + " " +
                              std::to_string(got.d[1]) + " " +
                              std::to_string(got.d[0]) + "])";
                return false;
            }
        } else {
            uint256_t check;
            mod_mul(check, got, original_dx[g]);
            if (!is_one(check)) {
                fail_reason = "inv * original_dx != 1 at lane " +
                              std::to_string(g) +
                              " (guard likely dropped from kernel)";
                return false;
            }
        }
    }

    return true;
}

}  // namespace

int main() {
    // Skip with SKIP_RETURN_CODE 77 if no CUDA device is available so
    // CI machines without GPUs don't fail this test.
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count <= 0) {
        std::fprintf(stderr,
            "[skip] no CUDA device available "
            "(cudaGetDeviceCount: %s, count=%d)\n",
            cudaGetErrorString(err), device_count);
        return 77;
    }

    // Discover the kernel's compiled GPU_GRP_SIZE by invoking the launch
    // with a tiny no-op probe. The kernel writes its compile-time value
    // into out_gpu_grp_size, but the launch requires correctly-sized
    // buffers. To resolve, do a two-step: first allocate generously
    // (max plausible 256), confirm the kernel's GPU_GRP_SIZE matches
    // our allocation, then use that as the canonical group size.
    constexpr int kMaxPlausibleGrpSize = 256;
    std::vector<collider::cpu::uint256_t> probe_dx(kMaxPlausibleGrpSize);
    for (auto& v : probe_dx) {
        v = collider::cpu::uint256_t(1);  // dx=1 is always invertible.
    }
    std::string probe_reason;
    // Issue: drive_kernel_once requires grp_size to match the kernel's
    // compiled size. We can't know it without launching. The kernel
    // reports it via out_gpu_grp_size on first call. Detect by running
    // a minimal allocation and reading the value back.
    int kernel_grp = 0;
    {
        uint64_t* d_dummy_in = nullptr;
        uint64_t* d_dummy_out = nullptr;
        uint8_t*  d_dummy_zero = nullptr;
        // Allocate big enough for any plausible GPU_GRP_SIZE.
        cudaMalloc(&d_dummy_in, kMaxPlausibleGrpSize * 4 * sizeof(uint64_t));
        cudaMalloc(&d_dummy_out, kMaxPlausibleGrpSize * 4 * sizeof(uint64_t));
        cudaMalloc(&d_dummy_zero, kMaxPlausibleGrpSize * sizeof(uint8_t));
        // Pre-fill dx with all-1 so the launch is well-defined whatever
        // the compiled GPU_GRP_SIZE turns out to be.
        std::vector<uint64_t> ones(kMaxPlausibleGrpSize * 4, 0);
        for (int g = 0; g < kMaxPlausibleGrpSize; ++g) ones[g * 4] = 1;
        cudaMemcpy(d_dummy_in, ones.data(), ones.size() * sizeof(uint64_t),
                   cudaMemcpyHostToDevice);
        cudaError_t le = test_only_kangaroo_dxz_guard_launch(
            d_dummy_in, d_dummy_out, d_dummy_zero, &kernel_grp, 0);
        cudaDeviceSynchronize();
        cudaFree(d_dummy_in); cudaFree(d_dummy_out); cudaFree(d_dummy_zero);
        if (le != cudaSuccess) {
            std::fprintf(stderr,
                "[!] probe launch failed: %s\n", cudaGetErrorString(le));
            return 1;
        }
    }
    if (kernel_grp <= 0 || kernel_grp > kMaxPlausibleGrpSize) {
        std::fprintf(stderr,
            "[!] kernel reported invalid GPU_GRP_SIZE=%d\n", kernel_grp);
        return 1;
    }

    std::printf("=== Kangaroo dx==0 kernel-driver test ===\n");
    std::printf("kernel GPU_GRP_SIZE = %d\n", kernel_grp);

    std::mt19937_64 rng(0xC0FFEEC0DEBADBEEULL);
    int pass = 0, fail = 0;

    // Sub-case 1: no-zero positive control. Every lane has a non-zero
    // dx; the kernel returns true modular inverses for all lanes.
    {
        std::vector<collider::cpu::uint256_t> dx(kernel_grp);
        for (auto& v : dx) v = random_field_element(rng);
        std::string why;
        if (drive_kernel_once(kernel_grp, dx, /*zero_lanes*/{}, why)) {
            std::printf("  PASS  no-zero positive control\n");
            ++pass;
        } else {
            std::printf("  FAIL  no-zero positive control: %s\n", why.c_str());
            ++fail;
        }
    }

    // Sub-case 2: zero at boundary lanes (0 and last).
    for (int lane : {0, kernel_grp - 1}) {
        std::vector<collider::cpu::uint256_t> dx(kernel_grp);
        for (auto& v : dx) v = random_field_element(rng);
        dx[lane] = collider::cpu::uint256_t(0);
        std::string why;
        if (drive_kernel_once(kernel_grp, dx, {lane}, why)) {
            std::printf("  PASS  zero injected at boundary lane %d\n", lane);
            ++pass;
        } else {
            std::printf("  FAIL  zero injected at boundary lane %d: %s\n",
                        lane, why.c_str());
            ++fail;
        }
    }

    // Sub-case 3: random zero-lane positions across many iterations.
    std::uniform_int_distribution<int> lane_dist(0, kernel_grp - 1);
    int sub_pass = 0, sub_fail = 0;
    std::string first_fail;
    for (int it = 0; it < kFuzzIterations; ++it) {
        std::vector<collider::cpu::uint256_t> dx(kernel_grp);
        for (auto& v : dx) v = random_field_element(rng);
        int lane = lane_dist(rng);
        dx[lane] = collider::cpu::uint256_t(0);
        std::string why;
        if (drive_kernel_once(kernel_grp, dx, {lane}, why)) {
            ++sub_pass;
        } else {
            ++sub_fail;
            if (first_fail.empty()) first_fail = why;
        }
    }
    if (sub_fail == 0) {
        std::printf("  PASS  %d random-lane fuzz iterations\n", sub_pass);
        ++pass;
    } else {
        std::printf("  FAIL  %d/%d random-lane fuzz iterations (first: %s)\n",
                    sub_fail, sub_pass + sub_fail, first_fail.c_str());
        ++fail;
    }

    // Sub-case 4: multiple simultaneous zero lanes. The guard must
    // flag and substitute each independently; the back-propagation
    // must produce correct inverses for every non-zeroed lane.
    {
        std::vector<collider::cpu::uint256_t> dx(kernel_grp);
        for (auto& v : dx) v = random_field_element(rng);
        std::vector<int> zeros;
        // Inject 3 zeros if grp_size allows, else 2.
        const int num_zeros = (kernel_grp >= 4) ? 3 : 2;
        for (int k = 0; k < num_zeros; ++k) {
            int lane = (kernel_grp * (k + 1)) / (num_zeros + 1);
            zeros.push_back(lane);
            dx[lane] = collider::cpu::uint256_t(0);
        }
        std::string why;
        if (drive_kernel_once(kernel_grp, dx, zeros, why)) {
            std::printf("  PASS  %d simultaneous zero lanes\n", num_zeros);
            ++pass;
        } else {
            std::printf("  FAIL  %d simultaneous zero lanes: %s\n",
                        num_zeros, why.c_str());
            ++fail;
        }
    }

    std::printf("\n=== Results: %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
