/**
 * test_puzzle_optimized_inv -- Known-Answer Test for puzzle_optimized.cu's
 * mod_inv.
 *
 * Wave 1 / Track C C-CRIT-2 follow-up (2026-05-04). The puzzle_optimized.cu
 * translation unit ships its own copy of mod_inv (separate from secp256k1.cu
 * because U256 here is 4 x uint64 little-endian instead of 8 x uint32). The
 * original implementation was a hand-coded addition chain that produced the
 * wrong exponent (approximately a^(2^255 - 2^31 - 493) instead of a^(p-2)),
 * so a * mod_inv(a) != 1 for nearly any a.
 *
 * EXPECTED FAILURE on the original code: this test reports many wrong
 * inversions. After the binary-exponentiation fix, it should report 0.
 *
 * The test calls into puzzle_optimized_test_inverse_correctness, which is a
 * permanent host-callable test entry point declared at the bottom of
 * src/gpu/puzzle_optimized.cu. That function owns the full life cycle:
 * deterministic scalar generation, device alloc/copy, kernel launch, result
 * reduction. The CTest harness here just checks the wrong_count output.
 */

#include <cuda_runtime.h>
#include <cstdio>

extern "C" int puzzle_optimized_test_inverse_correctness(int* wrong_count_out);

int main() {
    // Fail-fast device check so the test reports SKIP cleanly when no CUDA
    // device is present (e.g. macOS Metal builds, Linux without an NVIDIA
    // card). The test entry point itself also detects this and returns -2,
    // but we check up front to keep the SKIP semantics symmetric with the
    // other CUDA tests in this directory.
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA devices available: %s\n",
                cudaGetErrorString(cuda_err));
        return 77;  // CTest "skip" code
    }

    int wrong_count = -1;
    int rc = puzzle_optimized_test_inverse_correctness(&wrong_count);

    if (rc == -2) {
        // Defensive: should already have been caught above.
        fprintf(stderr, "puzzle_optimized_test_inverse_correctness: no CUDA device\n");
        return 77;
    }
    if (rc != 0) {
        fprintf(stderr,
                "puzzle_optimized_test_inverse_correctness internal error rc=%d\n",
                rc);
        return 1;
    }

    constexpr int N = 64;  // matches the value inside puzzle_optimized.cu

    printf("=== puzzle_optimized.cu mod_inv KAT ===\n");
    printf("Tested:  %d scalars\n", N);
    printf("Correct: %d\n", N - wrong_count);
    printf("Wrong:   %d\n", wrong_count);

    if (wrong_count == 0) {
        printf("PASS: mod_inv produces correct inverses for all tested scalars.\n");
        return 0;
    } else {
        printf("FAIL: %d of %d inversions are wrong (a * mod_inv(a) != 1 mod p).\n",
               wrong_count, N);
        printf("Original puzzle_optimized.cu addition chain computed the wrong exponent\n");
        printf("(approximately a^(2^255 - 2^31 - 493) instead of a^(p-2)).\n");
        printf("After the binary-exponentiation fix, this should report 0 wrong.\n");
        return 1;
    }
}
