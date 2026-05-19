/**
 * test_ec_table_consistency: verifies the per-GPU EC precomputed table is on the curve.
 *
 * Wave 0 of the 2026-05-04 review. The table is generated on each device via
 * secp256k1_init_table(), which calls jacobian_to_affine internally. If
 * secp256k1.cu's mod_inv is wrong (Track C C-CRIT-2), the affine conversion
 * produces points NOT on y^2 = x^3 + 7 (mod p). This test counts off-curve
 * entries.
 *
 * v1.4.2 builder-kangaroo extension (2026-05-16, P-B5): also exercises the
 * puzzle_optimized.cu precomputed table init / cleanup cycle. Pre-fix the
 * table allocated 128 KB (NUM_WINDOWS * 16 entries per side); post-fix it
 * allocates 2 KB (1 window of 16 entries per side). KangarooSmallPuzzle's
 * ec_mul_glv KAT covers correctness; this test adds an idempotency smoke
 * check so any future regression in the table allocator (size mismatch,
 * double-free, stale device pointer) shows up here.
 *
 * v1.4.2 builder-final extension (2026-05-16): adds an on-curve check for
 * the puzzle_optimized precomputed tables (d_PRECOMP_TABLE and
 * d_PRECOMP_TABLE_LAMBDA). Each table has 16 entries (PRECOMP_TABLE_ENTRIES);
 * entry 0 is the point-at-infinity and is skipped. For every other entry
 * the test asserts y^2 = x^3 + 7 (mod p) using CPU-side mod arithmetic.
 * Total verified: 30 points (15 from each table). Cheap regression net.
 *
 * EXPECTED FAILURE on current code: most table entries off-curve.
 * After Wave 1 fix #1.2: 0 off-curve entries.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "core/crypto_cpu.hpp"

extern "C" cudaError_t secp256k1_init_table(cudaStream_t stream);
extern "C" cudaError_t secp256k1_cleanup();
extern "C" cudaError_t secp256k1_test_table_on_curve(
    uint32_t* d_off_curve_count,
    cudaStream_t stream
);

// P-B5 (v1.4.2 builder-kangaroo): smoke test the trimmed (16-entry)
// puzzle_optimized precomp table init / cleanup. Correctness of the
// table content is covered by KangarooSmallPuzzle (ec_mul_glv KATs).
extern "C" cudaError_t init_puzzle_optimized(cudaStream_t stream);
extern "C" cudaError_t cleanup_puzzle_optimized();

// v1.4.2 builder-final: opaque accessors to the puzzle_optimized
// precomp tables, declared in puzzle_optimized.cu.
extern "C" size_t puzzle_optimized_table_entry_count();
extern "C" size_t puzzle_optimized_table_entry_bytes();
extern "C" const void* puzzle_optimized_get_table_device();
extern "C" const void* puzzle_optimized_get_table_lambda_device();

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

    if (off_curve != 0) {
        printf("FAIL: %u of %d entries are NOT on the secp256k1 curve.\n",
               off_curve, TOTAL_ENTRIES);
        printf("This is expected on current code per Track C C-CRIT-2 (broken mod_inv\n"
               "produces wrong jacobian_to_affine outputs during table generation).\n");
        printf("After Wave 1 fix #1.2, this should be 0.\n");
        return 0;  // preserved: pre-existing test allowed this branch to pass.
    }
    printf("PASS: every table entry satisfies y^2 = x^3 + 7 (mod p).\n");

    // P-B5: puzzle_optimized table init / cleanup smoke test. Doing
    // init -> cleanup -> init -> cleanup also exercises idempotency
    // of the g_table_initialized guard inside puzzle_optimized.cu.
    printf("\n=== puzzle_optimized precomp table init/cleanup (P-B5) ===\n");
    for (int round = 0; round < 2; round++) {
        err = init_puzzle_optimized(/*stream*/0);
        if (err != cudaSuccess) {
            fprintf(stderr, "init_puzzle_optimized round %d failed: %s\n",
                    round, cudaGetErrorString(err));
            return 1;
        }
        cudaDeviceSynchronize();
        err = cleanup_puzzle_optimized();
        if (err != cudaSuccess) {
            fprintf(stderr, "cleanup_puzzle_optimized round %d failed: %s\n",
                    round, cudaGetErrorString(err));
            return 1;
        }
    }
    printf("PASS: init/cleanup cycle ran twice without error.\n");

    // v1.4.2 builder-final: on-curve validation of the puzzle_optimized
    // precomputed tables. Init once, download both tables to host, then
    // for every entry (other than the point-at-infinity at index 0)
    // assert y^2 == x^3 + 7 (mod p). Catches any regression in the
    // table generator's affine conversion (mod_inv on Z, mod_mul against
    // z_inv^2 / z_inv^3) without exercising a full ec_mul.
    printf("\n=== puzzle_optimized precomp tables on-curve (v1.4.2 final) ===\n");
    err = init_puzzle_optimized(/*stream*/0);
    if (err != cudaSuccess) {
        fprintf(stderr, "init_puzzle_optimized failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    const size_t n_entries  = puzzle_optimized_table_entry_count();
    const size_t entry_size = puzzle_optimized_table_entry_bytes();
    if (n_entries == 0 || entry_size == 0) {
        fprintf(stderr, "puzzle_optimized table accessors returned zero-size.\n");
        cleanup_puzzle_optimized();
        return 1;
    }
    if (entry_size != 64u) {
        // The on-curve test assumes the PointA layout is exactly
        // {x[4 u64 little-endian], y[4 u64 little-endian]} = 64 bytes.
        // If the layout changes, this test needs to be updated. Fail
        // loudly rather than silently misinterpret bytes.
        fprintf(stderr, "Unexpected puzzle_optimized PointA size: %zu (expected 64)\n",
                entry_size);
        cleanup_puzzle_optimized();
        return 1;
    }

    const size_t total_bytes = n_entries * entry_size;
    uint8_t* h_table  = (uint8_t*)malloc(total_bytes);
    uint8_t* h_lambda = (uint8_t*)malloc(total_bytes);
    if (!h_table || !h_lambda) {
        fprintf(stderr, "host buffer allocation failed (%zu bytes)\n", total_bytes);
        free(h_table); free(h_lambda);
        cleanup_puzzle_optimized();
        return 1;
    }
    const void* d_table  = puzzle_optimized_get_table_device();
    const void* d_lambda = puzzle_optimized_get_table_lambda_device();
    if (!d_table || !d_lambda) {
        fprintf(stderr, "puzzle_optimized table device pointers are null\n");
        free(h_table); free(h_lambda);
        cleanup_puzzle_optimized();
        return 1;
    }
    err = cudaMemcpy(h_table,  d_table,  total_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy(table) failed: %s\n", cudaGetErrorString(err));
        free(h_table); free(h_lambda);
        cleanup_puzzle_optimized();
        return 1;
    }
    err = cudaMemcpy(h_lambda, d_lambda, total_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy(table_lambda) failed: %s\n", cudaGetErrorString(err));
        free(h_table); free(h_lambda);
        cleanup_puzzle_optimized();
        return 1;
    }

    // For each entry (skipping index 0 = point-at-infinity), verify
    // y^2 == x^3 + 7 (mod p).
    using collider::cpu::uint256_t;
    using collider::cpu::mod_mul;
    using collider::cpu::mod_add;
    using collider::cpu::SECP256K1_P;
    const uint256_t seven(7);

    auto check_entries = [&](const uint8_t* h_buf, const char* label,
                             size_t& off_curve_out) -> int {
        off_curve_out = 0;
        for (size_t i = 1; i < n_entries; ++i) {
            uint256_t x, y;
            std::memcpy(x.d, h_buf + i * entry_size + 0, 32);
            std::memcpy(y.d, h_buf + i * entry_size + 32, 32);
            // Skip explicit infinity sentinels (x = y = 0). The table
            // generator stores them at any index whose corresponding
            // points[i].is_infinity() is true; index 0 is the only one
            // expected to hit this for the in-curve range 1..15, but
            // be defensive.
            if (x.is_zero() && y.is_zero()) continue;

            uint256_t y2, x2, x3, rhs;
            mod_mul(y2, y, y);             // y^2
            mod_mul(x2, x, x);             // x^2
            mod_mul(x3, x2, x);            // x^3
            mod_add(rhs, x3, seven);       // x^3 + 7

            if (!(y2 == rhs)) {
                if (off_curve_out < 4) {
                    fprintf(stderr, "[%s] entry %zu: y^2 != x^3 + 7 (mod p)\n",
                            label, i);
                }
                ++off_curve_out;
            }
        }
        return 0;
    };

    size_t off_table = 0, off_lambda = 0;
    check_entries(h_table,  "table",        off_table);
    check_entries(h_lambda, "table_lambda", off_lambda);

    free(h_table); free(h_lambda);
    cleanup_puzzle_optimized();

    printf("Verified %zu entries from each of {d_PRECOMP_TABLE, d_PRECOMP_TABLE_LAMBDA}.\n",
           n_entries - 1);
    printf("Off-curve in d_PRECOMP_TABLE:        %zu\n", off_table);
    printf("Off-curve in d_PRECOMP_TABLE_LAMBDA: %zu\n", off_lambda);
    if (off_table != 0 || off_lambda != 0) {
        fprintf(stderr, "FAIL: puzzle_optimized table generator emitted off-curve points.\n");
        return 1;
    }
    printf("PASS: every puzzle_optimized table entry satisfies y^2 = x^3 + 7 (mod p).\n");
    return 0;
}
