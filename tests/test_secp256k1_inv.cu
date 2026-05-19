/**
 * test_secp256k1_inv — Known-Answer Test for secp256k1 modular inverse.
 *
 * Wave 0 of the 2026-05-04 review. Verifies mod_inv(a) is the modular inverse
 * of a (mod p) by computing a * mod_inv(a) on the GPU and checking it reduces
 * to 1. Tests N random scalars plus a few edge cases.
 *
 * EXPECTED FAILURE on current code: secp256k1.cu's mod_inv addition chain
 * produces a wrong exponent (Track C C-CRIT-2). This test should report many
 * inversions wrong. After Wave 1 fix #1.2, this test should report 0 wrong.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// Test entry point declared in src/gpu/secp256k1.cu (wave-0 test infrastructure)
extern "C" cudaError_t secp256k1_test_inverse_correctness(
    const void* d_scalars,
    uint8_t* d_results,
    size_t count,
    cudaStream_t stream
);

// secp256k1 prime p = 2^256 - 2^32 - 977 (big-endian bytes for reference; not
// used by host code, only documents the modulus the GPU is testing against).

static void seed_with(uint32_t* limbs, uint64_t hi, uint64_t lo) {
    // Place a 128-bit value into the high half (limbs[4..7]); leave low half random
    limbs[4] = (uint32_t)(lo);       limbs[5] = (uint32_t)(lo >> 32);
    limbs[6] = (uint32_t)(hi);       limbs[7] = (uint32_t)(hi >> 32);
}

int main() {
    // Use the first available CUDA device
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA devices available: %s\n", cudaGetErrorString(err));
        return 77;  // CTest "skip" code
    }
    cudaSetDevice(0);

    constexpr size_t N = 64;
    std::vector<uint32_t> scalars(N * 8, 0);

    // Deterministic PRNG so failures are reproducible
    std::mt19937_64 rng(0xC011DEC011DEull);

    // Random non-zero scalars in [1, p-1]. We don't enforce < p exactly here;
    // mod_inv should still produce a correct inverse mod p for any non-zero input
    // (it operates in the field; the input is reduced internally).
    for (size_t i = 0; i < N; i++) {
        for (int j = 0; j < 8; j++) {
            scalars[i * 8 + j] = (uint32_t)rng();
        }
        // Force MSB clear so value is < 2^255 (well below p ~ 2^256 - small)
        scalars[i * 8 + 7] &= 0x7FFFFFFFu;
        // Force at least one bit set (avoid zero)
        scalars[i * 8 + 0] |= 1u;
    }

    // Edge cases at the start
    // a = 1   -> inv(1) = 1
    memset(&scalars[0], 0, 32);  scalars[0] = 1;
    // a = 2   -> inv(2) = (p+1)/2 (a known-good value the kernel can compute)
    memset(&scalars[8], 0, 32);  scalars[8] = 2;
    // a = small composite to exercise reduction
    memset(&scalars[16], 0, 32); scalars[16] = 12345;
    // a = a high-bits value
    memset(&scalars[24], 0, 32); seed_with(&scalars[24], 0x0123456789ABCDEFull, 0xFEDCBA9876543210ull); scalars[24] = 1;

    // v1.4.2 A.6: canonical edges that broke addition-chain mod_inv in v1.4.0
    // and v1.4.1 - particularly the x223 addition-chain step. These ensure
    // the inverse holds for values near p, for all-ones values that require
    // internal reduction, and for boundary bit patterns.
    //
    // a = p - 1  (the involution: (p-1)^2 == 1 mod p, so inv(p-1) == p-1)
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    // p - 1 = 0xFFFFFFFF...FEFFFFFC2E
    memset(&scalars[32], 0, 32);
    scalars[32] = 0xFFFFFC2Eu; scalars[33] = 0xFFFFFFFEu;
    scalars[34] = 0xFFFFFFFFu; scalars[35] = 0xFFFFFFFFu;
    scalars[36] = 0xFFFFFFFFu; scalars[37] = 0xFFFFFFFFu;
    scalars[38] = 0xFFFFFFFFu; scalars[39] = 0xFFFFFFFFu;

    // a = 2^256 - 1  (the absolute maximum 256-bit value; will be reduced
    // internally to 2^32 + 976 = 0x1000003D0 mod p, then inverted)
    memset(&scalars[40], 0, 32);
    for (int j = 0; j < 8; j++) scalars[40 + j] = 0xFFFFFFFFu;

    // a = 2^223 - 1  (exercises the x223 step in the addition chain that
    // was historically miscoded as x224 / x222 in different forks)
    memset(&scalars[48], 0, 32);
    for (int j = 0; j < 6; j++) scalars[48 + j] = 0xFFFFFFFFu;
    scalars[54] = 0x7FFFFFFFu;  // bits 192..222 set; bit 223 clear
    scalars[55] = 0u;

    // a = n - 1, the group order minus one. Working in the FIELD (mod p),
    // this is just another bit pattern; the test verifies a * inv(a) == 1
    // for that pattern.
    // n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    memset(&scalars[56], 0, 32);
    scalars[56] = 0xD0364140u; scalars[57] = 0xBFD25E8Cu;
    scalars[58] = 0xAF48A03Bu; scalars[59] = 0xBAAEDCE6u;
    scalars[60] = 0xFFFFFFFEu; scalars[61] = 0xFFFFFFFFu;
    scalars[62] = 0xFFFFFFFFu; scalars[63] = 0xFFFFFFFFu;

    // Allocate device buffers
    uint32_t* d_scalars = nullptr;
    uint8_t*  d_results = nullptr;
    cudaMalloc(&d_scalars, N * 32);
    cudaMalloc(&d_results, N);

    cudaMemcpy(d_scalars, scalars.data(), N * 32, cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, N);

    err = secp256k1_test_inverse_correctness(d_scalars, d_results, N, /*stream*/0);
    if (err != cudaSuccess) {
        fprintf(stderr, "secp256k1_test_inverse_correctness launch failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_scalars); cudaFree(d_results);
        return 1;
    }
    cudaDeviceSynchronize();

    std::vector<uint8_t> results(N, 0);
    cudaMemcpy(results.data(), d_results, N, cudaMemcpyDeviceToHost);

    cudaFree(d_scalars);
    cudaFree(d_results);

    size_t passed = 0, failed = 0;
    for (size_t i = 0; i < N; i++) {
        if (results[i] == 1) passed++;
        else                 failed++;
    }

    printf("=== secp256k1 mod_inv KAT ===\n");
    printf("Tested:  %zu scalars\n", N);
    printf("Correct: %zu\n", passed);
    printf("Wrong:   %zu\n", failed);

    if (failed == 0) {
        printf("PASS: mod_inv produces correct inverses for all tested scalars.\n");
        return 0;
    } else {
        printf("FAIL: %zu of %zu inversions are wrong (a * mod_inv(a) != 1 mod p).\n",
               failed, N);
        printf("This is expected on current code per Track C C-CRIT-2.\n");
        printf("After Wave 1 fix #1.2 (replace secp256k1.cu mod_inv chain), this should be 0.\n");
        return 1;
    }
}
