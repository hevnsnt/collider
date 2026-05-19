/**
 * test_secp256k1_batch_mul_kat: regression KAT and symbol-reintroduction
 * trap for the deleted `secp256k1_batch_mul` host wrapper (and its
 * `ec_mul_batch_optimized_kernel` device backend). T0.3.a + T3.1
 * (A-tier wave 1, 2026-05-17).
 *
 * What was deleted and why
 * ========================
 * `ec_mul_batch_optimized_kernel` performed a windowed scalar mul that
 * BOTH doubled R between iterations (5 doublings per window) AND
 * accumulated entries from a precomputed table whose w-th window
 * already encoded the 2^(5*w) factor for each entry. The net effect:
 *   R = 2^(5*(NW-1)) * sum_w table[w][win_w]
 *     = 2^(5*(NW-1)) * sum_w win_w * 2^(5*w) * G
 *     = 2^(5*(NW-1)) * k * G            (not k*G)
 * For NW = 52 windows the prefactor 2^255 mangles every multi-window
 * scalar; pubkeys for any non-trivial private key were silently wrong.
 *
 * The host wrapper that dispatched the buggy kernel
 * (`secp256k1_batch_mul`) was deleted along with the kernel itself.
 * The correct sibling `secp256k1_batch_mul_simple`, which dispatches
 * `ec_mul_batch_kernel` and through that `ec_mul_windowed` (no
 * inter-window doubling), is the single source of truth and is
 * exercised by `test_ec_mul_known_answers.cu`.
 *
 * How this test acts as a trap
 * ============================
 * Today (post-T0.3.a) the symbol is gone. This test compiles and
 * passes by documenting the deletion via the comment block above and
 * by exercising the SAME KAT vectors against the live correct sibling
 * `secp256k1_batch_mul_simple`. The vectors below are bit-identical
 * to what `test_ec_mul_known_answers.cu` already pins, restated here
 * so anyone resurrecting the deleted symbol has a self-contained
 * reference KAT they can re-wire the test body against.
 *
 * If a future contributor re-introduces `secp256k1_batch_mul`, they
 * MUST EITHER:
 *
 *   (a) Delete this test (and its CMakeLists.txt registration) and
 *       trust `test_ec_mul_known_answers` to cover the new wrapper's
 *       backend, OR
 *
 *   (b) Replace the call to `secp256k1_batch_mul_simple` below with
 *       `secp256k1_batch_mul` so this KAT actually exercises the
 *       resurrected symbol. The vectors below MUST stay bit-identical
 *       to libsecp256k1; any "optimization" that changes pubkey output
 *       for any scalar is the bug returning.
 *
 * The reintroduction trap is enforced by the documentation contract,
 * not by a link-time mechanism, because Windows static-lib weak
 * linkage is not portable across the MSVC + nvcc toolchain in use.
 *
 * Returns 77 (CTest skip) if no CUDA device, 0 on pass, 1 on fail.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Live correct sibling: the SAME shape as the deleted wrapper but
// dispatching the no-inter-window-double kernel. If you resurrect
// `secp256k1_batch_mul` see the comment block above for what to do.
extern "C" {
    cudaError_t secp256k1_init_table(cudaStream_t stream);
    cudaError_t secp256k1_cleanup();
    cudaError_t secp256k1_batch_mul_simple(
        const void* d_private_keys,   // count * 32 bytes (8 x uint32 LE limbs)
        void* d_public_keys,          // count * 64 bytes (ECPointAffine: x then y)
        size_t count,
        cudaStream_t stream
    );
}

// =============================================================================
// KAT vectors. Bit-identical to test_ec_mul_known_answers.cu so that
// any correct re-implementation of `secp256k1_batch_mul` produces the
// same compressed pubkeys.
// =============================================================================
struct BatchMulKAT {
    const char* label;
    uint8_t privkey[32];                // big-endian
    uint8_t expected_pubkey[33];        // compressed: prefix || x_be
};

static const BatchMulKAT TEST_VECTORS[] = {
    {
        "k=1 (G)",
        {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1},
        {0x02,
         0x79,0xBE,0x66,0x7E, 0xF9,0xDC,0xBB,0xAC, 0x55,0xA0,0x62,0x95,
         0xCE,0x87,0x0B,0x07, 0x02,0x9B,0xFC,0xDB, 0x2D,0xCE,0x28,0xD9,
         0x59,0xF2,0x81,0x5B, 0x16,0xF8,0x17,0x98}
    },
    {
        "k=2 (2G)",
        {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,2},
        {0x02,
         0xC6,0x04,0x7F,0x94, 0x41,0xED,0x7D,0x6D, 0x30,0x45,0x40,0x6E,
         0x95,0xC0,0x7C,0xD8, 0x5C,0x77,0x8E,0x4B, 0x8C,0xEF,0x3C,0xA7,
         0xAB,0xAC,0x09,0xB9, 0x5C,0x70,0x9E,0xE5}
    },
    {
        "k=7 (7G, last single-window scalar before the bug masks)",
        {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,7},
        {0x02,
         0x5C,0xBD,0xF0,0x64, 0x6E,0x5D,0xB4,0xEA, 0xA3,0x98,0xF3,0x65,
         0xF2,0xEA,0x7A,0x0E, 0x3D,0x41,0x9B,0x7E, 0x03,0x30,0xE3,0x9C,
         0xE9,0x2B,0xDD,0xED, 0xCA,0xC4,0xF9,0xBC}
    },
    {
        // 0xDEADBEEFCAFEBABE spans windows 0..1; the simplest multi-window
        // scalar. libsecp256k1 reference via coincurve. Same vector lives in
        // test_ec_mul_known_answers.cu; this restatement keeps the trap KAT
        // self-contained.
        "k=0xDEADBEEFCAFEBABE (2-window scalar, libsecp256k1 reference)",
        {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
         0xDE,0xAD,0xBE,0xEF, 0xCA,0xFE,0xBA,0xBE},
        {0x03,
         0x7B,0x51,0x6C,0x10, 0xE8,0x92,0x83,0x70, 0x32,0xB7,0x0E,0x61,
         0x85,0x65,0xA6,0xBC, 0x51,0x0B,0xDB,0x48, 0xAF,0x93,0x82,0xDB,
         0x97,0xDA,0x87,0x69, 0x79,0xD5,0x1B,0x5C}
    },
    {
        // 2^130 + 1: bit 0 and bit 130 lit, so windows 0 and 26 are
        // non-zero and EVERY window in between adds R doublings. This is
        // the scalar shape the deleted kernel mangled most visibly.
        "k=2^130+1 (far-apart windows, libsecp256k1 reference)",
        {0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,
         0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x04,
         0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00,
         0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x01},
        {0x02,
         0x2F,0x8A,0xAE,0x50, 0xDB,0x03,0x59,0xAD, 0x6B,0xE9,0xC8,0x36,
         0xB1,0x98,0x85,0x3E, 0xC5,0xB1,0x65,0xB0, 0xDA,0xB4,0x63,0xAC,
         0x97,0xC2,0xD4,0x62, 0x3B,0x33,0x39,0x14}
    },
    {
        // n minus 1: every window non-zero. Pubkey is the negation of G
        // (compressed prefix 03, x = Gx). Reference: libsecp256k1.
        "k=n-1 (every window non-zero, equals -G)",
        {0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
         0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
         0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
         0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x40},
        {0x03,
         0x79,0xBE,0x66,0x7E, 0xF9,0xDC,0xBB,0xAC, 0x55,0xA0,0x62,0x95,
         0xCE,0x87,0x0B,0x07, 0x02,0x9B,0xFC,0xDB, 0x2D,0xCE,0x28,0xD9,
         0x59,0xF2,0x81,0x5B, 0x16,0xF8,0x17,0x98}
    },
};

// big-endian privkey bytes to 8 x uint32_t little-endian limbs (limb[0] = LSB)
static void privkey_be_to_limbs(const uint8_t in_be[32], uint32_t out_limbs[8]) {
    for (int i = 0; i < 8; ++i) {
        const int byte_off = (7 - i) * 4;
        out_limbs[i] = (uint32_t)in_be[byte_off + 0] << 24
                     | (uint32_t)in_be[byte_off + 1] << 16
                     | (uint32_t)in_be[byte_off + 2] <<  8
                     | (uint32_t)in_be[byte_off + 3];
    }
}

// 8 x uint32_t little-endian limbs (x coord) to 32 big-endian bytes
static void affine_x_to_be(const uint32_t limbs[8], uint8_t out_be[32]) {
    for (int i = 0; i < 8; ++i) {
        const uint32_t v = limbs[7 - i];
        out_be[i*4 + 0] = (uint8_t)(v >> 24);
        out_be[i*4 + 1] = (uint8_t)(v >> 16);
        out_be[i*4 + 2] = (uint8_t)(v >>  8);
        out_be[i*4 + 3] = (uint8_t)v;
    }
}

int main() {
    int dev_count = 0;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "[skip] no CUDA device available\n");
        return 77;  // CTest SKIP
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    constexpr size_t NUM_VECTORS = sizeof(TEST_VECTORS) / sizeof(TEST_VECTORS[0]);

    std::vector<uint32_t> h_privlimbs(NUM_VECTORS * 8);
    for (size_t i = 0; i < NUM_VECTORS; ++i) {
        privkey_be_to_limbs(TEST_VECTORS[i].privkey, &h_privlimbs[i * 8]);
    }

    void* d_privkeys = nullptr;
    void* d_pubkeys  = nullptr;
    err = cudaMalloc(&d_privkeys, NUM_VECTORS * 32);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(privkeys) failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    err = cudaMalloc(&d_pubkeys, NUM_VECTORS * 64);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(pubkeys) failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_privkeys);
        return 1;
    }

    err = cudaMemcpy(d_privkeys, h_privlimbs.data(), NUM_VECTORS * 32,
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy(privkeys H to D) failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_privkeys); cudaFree(d_pubkeys);
        return 1;
    }

    err = secp256k1_init_table(/*stream*/0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "secp256k1_init_table failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_privkeys); cudaFree(d_pubkeys);
        return 1;
    }
    cudaDeviceSynchronize();

    // T0.3.a trap: this call used to target the deleted `secp256k1_batch_mul`.
    // The KAT semantics are unchanged. Any correct batch-mul implementation
    // produces the same pubkeys, so a future reintroducer can swap the
    // symbol here with no other change. See the file-level docblock for the
    // contract reintroducers must follow.
    err = secp256k1_batch_mul_simple(d_privkeys, d_pubkeys, NUM_VECTORS, /*stream*/0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "secp256k1_batch_mul_simple launch failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_privkeys); cudaFree(d_pubkeys); secp256k1_cleanup();
        return 1;
    }
    cudaDeviceSynchronize();

    // ECPointAffine layout: 8 x uint32 x limbs, then 8 x uint32 y limbs
    std::vector<uint32_t> h_pub(NUM_VECTORS * 16);
    err = cudaMemcpy(h_pub.data(), d_pubkeys, NUM_VECTORS * 64,
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy(pubkeys D to H) failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_privkeys); cudaFree(d_pubkeys); secp256k1_cleanup();
        return 1;
    }

    int failures = 0;
    for (size_t i = 0; i < NUM_VECTORS; ++i) {
        const uint32_t* x_limbs = &h_pub[i * 16];
        const uint32_t* y_limbs = &h_pub[i * 16 + 8];

        uint8_t actual_pubkey[33];
        actual_pubkey[0] = (y_limbs[0] & 1u) ? 0x03 : 0x02;
        affine_x_to_be(x_limbs, &actual_pubkey[1]);

        if (std::memcmp(actual_pubkey, TEST_VECTORS[i].expected_pubkey, 33) != 0) {
            std::fprintf(stderr, "FAIL [%s]\n  expected: ", TEST_VECTORS[i].label);
            for (int b = 0; b < 33; ++b)
                std::fprintf(stderr, "%02x", TEST_VECTORS[i].expected_pubkey[b]);
            std::fprintf(stderr, "\n  actual:   ");
            for (int b = 0; b < 33; ++b)
                std::fprintf(stderr, "%02x", actual_pubkey[b]);
            std::fprintf(stderr, "\n");
            ++failures;
        } else {
            std::fprintf(stdout, "PASS [%s]\n", TEST_VECTORS[i].label);
        }
    }

    cudaFree(d_privkeys);
    cudaFree(d_pubkeys);
    secp256k1_cleanup();

    if (failures != 0) {
        std::fprintf(stderr, "\n%d/%zu vectors failed\n", failures, NUM_VECTORS);
        return 1;
    }
    std::fprintf(stdout, "\nAll %zu vectors passed\n", NUM_VECTORS);
    return 0;
}
