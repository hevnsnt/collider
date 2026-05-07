/**
 * test_ec_table_consistency — verifies the per-GPU EC precomputed table is on the curve.
 *
 * Wave 0 of the 2026-05-04 review. The table is generated on each device via
 * secp256k1_init_table(), which calls jacobian_to_affine internally. If
 * secp256k1.cu's mod_inv is wrong (Track C C-CRIT-2), the affine conversion
 * produces points NOT on y^2 = x^3 + 7 (mod p). This test counts off-curve
 * entries.
 *
 * EXPECTED FAILURE on current code: most table entries off-curve.
 * After Wave 1 fix #1.2: 0 off-curve entries.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

extern "C" cudaError_t secp256k1_init_table(cudaStream_t stream);
extern "C" cudaError_t secp256k1_cleanup();
extern "C" cudaError_t secp256k1_test_table_on_curve(
    uint32_t* d_off_curve_count,
    cudaStream_t stream
);

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA devices available: %s\n", cudaGetErrorString(err));
        return 77;  // CTest skip
    }
    cudaSetDevice(0);

    err = secp256k1_init_table(/*stream*/0);
    if (err != cudaSuccess) {
        fprintf(stderr, "secp256k1_init_table failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    uint32_t* d_count = nullptr;
    cudaMalloc(&d_count, sizeof(uint32_t));
    cudaMemset(d_count, 0, sizeof(uint32_t));

    err = secp256k1_test_table_on_curve(d_count, /*stream*/0);
    if (err != cudaSuccess) {
        fprintf(stderr, "secp256k1_test_table_on_curve launch failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_count);
        secp256k1_cleanup();
        return 1;
    }
    cudaDeviceSynchronize();

    uint32_t off_curve = 0;
    cudaMemcpy(&off_curve, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_count);

    // EC_NUM_WINDOWS=52, EC_TABLE_SIZE=32 -> 1664 entries total
    constexpr int TOTAL_ENTRIES = 52 * 32;

    secp256k1_cleanup();

    printf("=== secp256k1 precomputed table consistency ===\n");
    printf("Total entries: %d\n", TOTAL_ENTRIES);
    printf("Off-curve:     %u\n", off_curve);

    if (off_curve == 0) {
        printf("PASS: every table entry satisfies y^2 = x^3 + 7 (mod p).\n");
        return 0;
    } else {
        printf("FAIL: %u of %d entries are NOT on the secp256k1 curve.\n",
               off_curve, TOTAL_ENTRIES);
        printf("This is expected on current code per Track C C-CRIT-2 (broken mod_inv\n"
               "produces wrong jacobian_to_affine outputs during table generation).\n");
        printf("After Wave 1 fix #1.2, this should be 0.\n");
        return 1;
    }
}
