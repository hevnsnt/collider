/**
 * test_ec_mul_known_answers -- Known-Answer Test for GPU secp256k1 EC scalar
 * multiplication.
 *
 * Verifies the GPU computes the correct compressed public key for a small set
 * of well-known Bitcoin private keys. This catches regressions in the entire
 * EC mul pipeline: precomputed table generation, ec_double_jacobian,
 * ec_add_mixed, the modular arithmetic primitives, and Jacobian to affine
 * conversion (which depends on mod_inv).
 *
 * Strategy:
 *   1. Build a small array of (privkey, expected_compressed_pubkey) test
 *      vectors. Privkeys are encoded as 32-byte big-endian (Bitcoin standard).
 *   2. Convert privkeys to the GPU's internal format: 8 x uint32 little-endian
 *      limbs, where limb[0] holds the LEAST significant 32 bits.
 *   3. Initialize the precomputed table on device 0 (calls
 *      secp256k1_init_table, which generates the table by repeated point
 *      additions starting from G).
 *   4. Call the production batch EC mul API
 *      (secp256k1_batch_mul_simple) to produce ECPointAffine outputs (x and y
 *      each as 8 x uint32 little-endian limbs).
 *   5. Re-encode each output as a 33-byte compressed public key (prefix 0x02
 *      if y is even, 0x03 if y is odd, then 32 bytes of x in big-endian).
 *   6. Compare byte-for-byte against the expected pubkey.
 *
 * The known-answer vectors come from the public secp256k1 generator G.
 * Single-window scalars (only window 0 non-zero):
 *   k = 1: 02 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
 *   k = 2: 02 C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
 *   k = 3: 02 F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9
 *   k = 7: 02 5CBDF0646E5DB4EAA398F365F2EA7A0E3D419B7E0330E39CE92BDDEDCAC4F9BC
 *
 * Multi-window scalars (added 2026-05-05 to catch the C-CRIT-1 over-doubling
 * bug in ec_mul_windowed: any scalar that lights up bits in more than one
 * 5-bit window will produce a wrong pubkey if the windowed mul accidentally
 * doubles R between windows AND uses a precomputed table whose entries
 * already carry the per-window 2^(5*w) factor). Pubkeys verified against
 * libsecp256k1 (via the Python coincurve binding):
 *   k = 0x000000000000000000000000000000000000000000000000DEADBEEFCAFEBABE
 *       -> 03 7B516C10E892837032B70E618565A6BC510BDB48AF9382DB97DA876979D51B5C
 *   k = 0x00000000000000000000000000000000000000000000000000000000075BCD15  (decimal 123456789)
 *       -> 02 08F4F37E2D8F74E18C1B8FDE2374D5F28402FB8AB7FD1CC5B786AA40851A70CB
 *   k = 0x0000000000000000000000000000000400000000000000000000000000000001  (2^130 + 1, far-apart windows)
 *       -> 02 2F8AAE50DB0359AD6BE9C836B198853EC5B165B0DAB463AC97C2D4623B333914
 *   k = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140  (n - 1, every window non-zero)
 *       -> 03 79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
 *
 * Returns 77 (CTest skip) if no CUDA device, 0 on pass, 1 on fail.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Production API from src/gpu/secp256k1.cu
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
// Test vectors
// =============================================================================
struct EcMulKAT {
    const char* label;
    uint8_t privkey[32];                // big-endian
    uint8_t expected_pubkey[33];        // compressed: prefix || x_be
};

// Known generator multiples on secp256k1.
static const EcMulKAT TEST_VECTORS[] = {
    {
        "k=1",
        // privkey = 0x...01 (big-endian 32 bytes)
        {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,1},
        // 02 || Gx
        {0x02,
         0x79,0xBE,0x66,0x7E, 0xF9,0xDC,0xBB,0xAC, 0x55,0xA0,0x62,0x95,
         0xCE,0x87,0x0B,0x07, 0x02,0x9B,0xFC,0xDB, 0x2D,0xCE,0x28,0xD9,
         0x59,0xF2,0x81,0x5B, 0x16,0xF8,0x17,0x98}
    },
    {
        "k=2",
        {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,2},
        // 2G compressed (33 bytes): prefix 02, then x = c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5
        {0x02,
         0xC6,0x04,0x7F,0x94, 0x41,0xED,0x7D,0x6D, 0x30,0x45,0x40,0x6E,
         0x95,0xC0,0x7C,0xD8, 0x5C,0x77,0x8E,0x4B, 0x8C,0xEF,0x3C,0xA7,
         0xAB,0xAC,0x09,0xB9, 0x5C,0x70,0x9E,0xE5}
    },
    {
        "k=3",
        {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,3},
        {0x02,
         0xF9,0x30,0x8A,0x01, 0x92,0x58,0xC3,0x10, 0x49,0x34,0x4F,0x85,
         0xF8,0x9D,0x52,0x29, 0xB5,0x31,0xC8,0x45, 0x83,0x6F,0x99,0xB0,
         0x86,0x01,0xF1,0x13, 0xBC,0xE0,0x36,0xF9}
    },
    {
        "k=7",
        {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,7},
        {0x02,
         0x5C,0xBD,0xF0,0x64, 0x6E,0x5D,0xB4,0xEA, 0xA3,0x98,0xF3,0x65,
         0xF2,0xEA,0x7A,0x0E, 0x3D,0x41,0x9B,0x7E, 0x03,0x30,0xE3,0x9C,
         0xE9,0x2B,0xDD,0xED, 0xCA,0xC4,0xF9,0xBC}
    },

    // ---- Multi-window vectors (Wave 1 / C-CRIT-1 regression coverage) ----
    // These scalars all have bits set in more than one 5-bit window, so they
    // exercise the windowed mul's inter-window add path. The previous
    // ec_mul_windowed double-applied the per-window 2^(5*w) factor and would
    // fail every one of these. Single-window scalars above (k <= 7) cannot
    // detect the bug because the only non-zero window is window 0.
    {
        "k=0xDEADBEEFCAFEBABE",
        // 24 zero bytes then 0xDEADBEEFCAFEBABE big-endian
        {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
         0xDE,0xAD,0xBE,0xEF, 0xCA,0xFE,0xBA,0xBE},
        // 03 || x: pubkey from libsecp256k1 (coincurve verification 2026-05-05)
        {0x03,
         0x7B,0x51,0x6C,0x10, 0xE8,0x92,0x83,0x70, 0x32,0xB7,0x0E,0x61,
         0x85,0x65,0xA6,0xBC, 0x51,0x0B,0xDB,0x48, 0xAF,0x93,0x82,0xDB,
         0x97,0xDA,0x87,0x69, 0x79,0xD5,0x1B,0x5C}
    },
    {
        "k=123456789",
        // 0x075BCD15 big-endian: 28 zero bytes then 07 5B CD 15.
        {0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,
         0x00,0x00,0x00,0x00, 0x07,0x5B,0xCD,0x15},
        // 02 || x (verified vs libsecp256k1)
        {0x02,
         0x08,0xF4,0xF3,0x7E, 0x2D,0x8F,0x74,0xE1, 0x8C,0x1B,0x8F,0xDE,
         0x23,0x74,0xD5,0xF2, 0x84,0x02,0xFB,0x8A, 0xB7,0xFD,0x1C,0xC5,
         0xB7,0x86,0xAA,0x40, 0x85,0x1A,0x70,0xCB}
    },
    {
        "k=2^130+1",
        // 256-bit big-endian layout: byte[i] holds bits [(31-i)*8 .. (31-i)*8+7].
        // Bit 130 -> i=15, bit-in-byte=2 -> byte[15] = 0x04.
        // Bit 0   -> i=31              -> byte[31] = 0x01. All other bytes 0.
        // This stresses far-apart windows: window 0 (bit 0) and window 26
        // (bit 130 sits at offset 130 - 26*5 = 0 inside window 26).
        {0,0,0,0,0,0,0,0,    0,0,0,0,0,0,0,0x04,
         0,0,0,0,0,0,0,0,    0,0,0,0,0,0,0,0x01},
        // 02 || x (verified vs libsecp256k1)
        {0x02,
         0x2F,0x8A,0xAE,0x50, 0xDB,0x03,0x59,0xAD, 0x6B,0xE9,0xC8,0x36,
         0xB1,0x98,0x85,0x3E, 0xC5,0xB1,0x65,0xB0, 0xDA,0xB4,0x63,0xAC,
         0x97,0xC2,0xD4,0x62, 0x3B,0x33,0x39,0x14}
    },
    {
        "k=n-1",
        // n - 1 has bits set across every 5-bit window of the 256-bit space.
        // (n-1)*G = -G, so the expected pubkey is the negation of G:
        // x is unchanged (Gx), y becomes p - Gy. Gy is even, so (p - Gy)
        // is odd and the compressed prefix is 0x03.
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
static constexpr size_t NUM_VECTORS = sizeof(TEST_VECTORS) / sizeof(TEST_VECTORS[0]);

// =============================================================================
// Encoding helpers
// =============================================================================

// Convert 32-byte big-endian privkey to 8 x uint32 little-endian limbs.
// limb[0] = least significant 32 bits.
static void privkey_be_to_limbs_le(const uint8_t* be32, uint32_t out_limbs[8]) {
    for (int i = 0; i < 8; i++) {
        // limb[i] holds bits [32*i, 32*i + 32) of the integer.
        // In big-endian bytes, those bits live at offset 32 - 4*(i+1) = 28 - 4*i.
        const uint8_t* p = be32 + (28 - 4 * i);
        out_limbs[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                       ((uint32_t)p[2] << 8)  | ((uint32_t)p[3]);
    }
}

// Encode (x_limbs_LE, y_limbs_LE) as 33-byte compressed pubkey: prefix || x_be.
// y parity determines prefix: 0x02 if y is even, 0x03 if odd.
static void affine_to_compressed_pubkey(
    const uint32_t x_limbs[8],
    const uint32_t y_limbs[8],
    uint8_t out_pubkey[33]
) {
    // Parity: low bit of y. y_limbs[0] is least significant 32 bits, bit 0 of
    // that limb is bit 0 of the entire integer.
    out_pubkey[0] = (y_limbs[0] & 1u) ? 0x03 : 0x02;

    // x in big-endian: limb[7] is most significant, limb[0] is least.
    for (int i = 0; i < 8; i++) {
        uint32_t limb = x_limbs[7 - i];
        out_pubkey[1 + 4*i + 0] = (uint8_t)(limb >> 24);
        out_pubkey[1 + 4*i + 1] = (uint8_t)(limb >> 16);
        out_pubkey[1 + 4*i + 2] = (uint8_t)(limb >> 8);
        out_pubkey[1 + 4*i + 3] = (uint8_t)(limb);
    }
}

static void hex_dump(const uint8_t* bytes, size_t n) {
    for (size_t i = 0; i < n; i++) printf("%02x", bytes[i]);
}

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA devices available: %s\n", cudaGetErrorString(err));
        return 77;  // CTest skip
    }
    cudaSetDevice(0);

    // Build packed privkey buffer in GPU's expected layout (8 x uint32 LE limbs each).
    std::vector<uint32_t> privkey_buffer(NUM_VECTORS * 8, 0);
    for (size_t i = 0; i < NUM_VECTORS; i++) {
        privkey_be_to_limbs_le(TEST_VECTORS[i].privkey, &privkey_buffer[i * 8]);
    }

    // Initialize precomputed EC table on device 0. This is what
    // secp256k1_batch_mul_simple uses internally. If the table generation
    // itself is broken (e.g. broken mod_inv corrupts jacobian_to_affine), we
    // will see wrong results downstream which is exactly what this test is for.
    err = secp256k1_init_table(/*stream*/0);
    if (err != cudaSuccess) {
        fprintf(stderr, "secp256k1_init_table failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // Allocate device buffers
    uint32_t* d_privkeys = nullptr;
    uint32_t* d_pubkeys  = nullptr;  // 8 + 8 = 16 limbs per output (x then y)
    err = cudaMalloc(&d_privkeys, NUM_VECTORS * 8 * sizeof(uint32_t));
    if (err != cudaSuccess) { secp256k1_cleanup(); return 1; }
    err = cudaMalloc(&d_pubkeys, NUM_VECTORS * 16 * sizeof(uint32_t));
    if (err != cudaSuccess) { cudaFree(d_privkeys); secp256k1_cleanup(); return 1; }

    err = cudaMemcpy(d_privkeys, privkey_buffer.data(),
                     NUM_VECTORS * 8 * sizeof(uint32_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_privkeys); cudaFree(d_pubkeys); secp256k1_cleanup();
        return 1;
    }

    err = secp256k1_batch_mul_simple(d_privkeys, d_pubkeys, NUM_VECTORS, /*stream*/0);
    if (err != cudaSuccess) {
        fprintf(stderr, "secp256k1_batch_mul_simple launch failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_privkeys); cudaFree(d_pubkeys); secp256k1_cleanup();
        return 1;
    }
    cudaDeviceSynchronize();

    std::vector<uint32_t> pubkey_buffer(NUM_VECTORS * 16, 0);
    cudaMemcpy(pubkey_buffer.data(), d_pubkeys,
               NUM_VECTORS * 16 * sizeof(uint32_t),
               cudaMemcpyDeviceToHost);

    cudaFree(d_privkeys);
    cudaFree(d_pubkeys);
    secp256k1_cleanup();

    // Compare each output against the expected compressed pubkey.
    int wrong = 0;
    printf("=== secp256k1 EC mul known-answer test ===\n");
    for (size_t i = 0; i < NUM_VECTORS; i++) {
        const uint32_t* x_limbs = &pubkey_buffer[i * 16 + 0];
        const uint32_t* y_limbs = &pubkey_buffer[i * 16 + 8];

        uint8_t got_pubkey[33];
        affine_to_compressed_pubkey(x_limbs, y_limbs, got_pubkey);

        bool match = (memcmp(got_pubkey, TEST_VECTORS[i].expected_pubkey, 33) == 0);
        printf("  %s  %s\n", match ? "PASS" : "FAIL", TEST_VECTORS[i].label);
        if (!match) {
            wrong++;
            printf("    expected: "); hex_dump(TEST_VECTORS[i].expected_pubkey, 33); printf("\n");
            printf("    got:      "); hex_dump(got_pubkey, 33); printf("\n");
        }
    }

    printf("Tested:  %zu vectors\n", NUM_VECTORS);
    printf("Correct: %zu\n", NUM_VECTORS - (size_t)wrong);
    printf("Wrong:   %d\n", wrong);

    if (wrong == 0) {
        printf("PASS: all known-answer vectors match.\n");
        return 0;
    } else {
        printf("FAIL: %d of %zu EC multiplications produced wrong pubkeys.\n",
               wrong, NUM_VECTORS);
        return 1;
    }
}
