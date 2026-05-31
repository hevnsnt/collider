/**
 * test_glv_decompose -- KAT for src/gpu/glv_decompose.cuh.
 *
 * Phase C.1 of v1.4.2 A-tier lift. The decomposition was previously
 * inline in puzzle_optimized.cu; this test exercises the extracted
 * shared header directly.
 *
 * For each test scalar k:
 *   1. Run glv::decompose(k) -> (k1, k2, k1_neg, k2_neg) on the GPU.
 *   2. Verify the Babai bound: k1[3] == 0 && k2[3] == 0.
 *   3. Verify the algebraic identity: signed_k1 + signed_k2 * lambda
 *      equals k modulo n.
 *
 * All multi-precision math on the CPU side is done with naive
 * 256-bit / 512-bit helpers (no GMP / OpenSSL dependency).
 */

#include "../src/gpu/glv_decompose.cuh"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ==========================================================================
// Device kernel: applies decompose to each input scalar.
// ==========================================================================

struct GlvOut {
    uint64_t k1[4];
    uint64_t k2[4];
    uint8_t  k1_neg;
    uint8_t  k2_neg;
    uint8_t  _pad[6];
};

__global__ void glv_decompose_kernel(const uint64_t* in_k,
                                      GlvOut* out,
                                      int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint64_t* k = in_k + idx * 4;
    GlvOut& o = out[idx];

    bool n1 = false, n2 = false;
    collider::gpu::glv::decompose(k, o.k1, o.k2, n1, n2);
    o.k1_neg = n1 ? 1 : 0;
    o.k2_neg = n2 ? 1 : 0;
}

// ==========================================================================
// Host-side multi-precision arithmetic (mod n).
// ==========================================================================
//
// secp256k1 group order n (little-endian limbs).
static const uint64_t N_HOST[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// secp256k1 endomorphism eigenvalue lambda (LE limbs).
static const uint64_t LAMBDA_HOST[4] = {
    0xDF02967C1B23BD72ULL, 0x122E22EA20816678ULL,
    0xA5261C028812645AULL, 0x5363AD4CC05C30E0ULL
};

static bool host_sub256(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t ai = a[i];
        uint64_t bi = b[i];
        uint64_t diff = ai - bi - borrow;
        uint64_t new_borrow = (ai < bi) ? 1ULL : ((ai == bi) ? borrow : 0ULL);
        r[i] = diff;
        borrow = new_borrow;
    }
    return borrow != 0;
}

static uint64_t host_add256(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t ai = a[i];
        uint64_t bi = b[i];
        uint64_t s = ai + bi;
        uint64_t c1 = (s < ai) ? 1ULL : 0ULL;
        s += carry;
        uint64_t c2 = (s < carry) ? 1ULL : 0ULL;
        r[i] = s;
        carry = c1 + c2;
    }
    return carry;
}

// Compare 256-bit LE arrays. Returns -1, 0, 1.
static int host_cmp256(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// r = (a + b) mod n, assuming a, b < n.
static void host_mod_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t carry = host_add256(r, a, b);
    if (carry || host_cmp256(r, N_HOST) >= 0) {
        host_sub256(r, r, N_HOST);
    }
}

// 256 x 256 -> 512-bit schoolbook multiply (LE limbs).
static void host_mul_256x256(uint64_t prod[8], const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 0; i < 8; i++) prod[i] = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            // 64x64 -> 128
            #if defined(_MSC_VER)
                uint64_t hi;
                uint64_t lo = _umul128(a[i], b[j], &hi);
            #else
                __uint128_t prod128 = (__uint128_t)a[i] * b[j];
                uint64_t lo = (uint64_t)prod128;
                uint64_t hi = (uint64_t)(prod128 >> 64);
            #endif
            // sum_lo = prod[i+j] + lo
            uint64_t s1 = prod[i + j] + lo;
            uint64_t c1 = (s1 < prod[i + j]) ? 1ULL : 0ULL;
            // sum_lo += carry
            uint64_t s2 = s1 + carry;
            uint64_t c2 = (s2 < carry) ? 1ULL : 0ULL;
            prod[i + j] = s2;
            // new carry = hi + c1 + c2
            carry = hi + c1 + c2;
        }
        // Propagate remaining carry.
        for (int k = i + 4; k < 8 && carry; k++) {
            uint64_t old = prod[k];
            prod[k] += carry;
            carry = (prod[k] < old) ? 1 : 0;
        }
    }
}

// Reduce a 512-bit value mod n using shift-and-subtract (slow but correct).
static void host_reduce_mod_n_512(uint64_t r[4], const uint64_t prod[8]) {
    // 9-limb working buffer so we can hold n << 256 (which needs limb index 8).
    uint64_t p[9];
    for (int i = 0; i < 8; i++) p[i] = prod[i];
    p[8] = 0;

    // For each bit shift k from 256 down to 0, subtract (n << k) if p >= (n << k).
    for (int shift = 256; shift >= 0; shift--) {
        // Build n_shifted in a 9-limb buffer.
        uint64_t ns[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        int word = shift / 64;
        int sh = shift % 64;
        for (int i = 0; i < 4; i++) {
            uint64_t lo = N_HOST[i] << sh;
            uint64_t hi = (sh == 0) ? 0ULL : (N_HOST[i] >> (64 - sh));
            if (word + i < 9) ns[word + i] |= lo;
            if (word + i + 1 < 9) ns[word + i + 1] |= hi;
        }
        // Compare p vs ns.
        int cmp = 0;
        for (int i = 8; i >= 0; i--) {
            if (p[i] > ns[i]) { cmp = 1; break; }
            if (p[i] < ns[i]) { cmp = -1; break; }
        }
        if (cmp >= 0) {
            // p -= ns
            uint64_t borrow = 0;
            for (int i = 0; i < 9; i++) {
                uint64_t ai = p[i];
                uint64_t bi = ns[i];
                uint64_t diff = ai - bi - borrow;
                uint64_t nb = (ai < bi) ? 1ULL : ((ai == bi) ? borrow : 0ULL);
                p[i] = diff;
                borrow = nb;
            }
        }
    }
    for (int i = 0; i < 4; i++) r[i] = p[i];
}

// r = (a * b) mod n.
static void host_mod_mul_n(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t prod[8];
    host_mul_256x256(prod, a, b);
    host_reduce_mod_n_512(r, prod);
}

// Convert a (magnitude, sign) into a value in [0, n). Magnitude must satisfy
// magnitude < n already; otherwise it must be reduced first.
static void host_signed_to_mod_n(uint64_t out[4],
                                  const uint64_t magnitude[4],
                                  bool neg) {
    // Reduce magnitude mod n in case k1[2] = 1 (which means magnitude
    // could be up to ~1.4 * 2^128, still well under n).
    uint64_t tmp[4] = { magnitude[0], magnitude[1], magnitude[2], magnitude[3] };
    while (host_cmp256(tmp, N_HOST) >= 0) {
        host_sub256(tmp, tmp, N_HOST);
    }
    if (!neg) {
        for (int i = 0; i < 4; i++) out[i] = tmp[i];
        return;
    }
    // Negate: out = n - tmp, unless tmp == 0.
    bool is_zero = (tmp[0] | tmp[1] | tmp[2] | tmp[3]) == 0;
    if (is_zero) {
        for (int i = 0; i < 4; i++) out[i] = 0;
    } else {
        host_sub256(out, N_HOST, tmp);
    }
}

// ==========================================================================
// Test runner.
// ==========================================================================

struct TestCase {
    const char* name;
    uint64_t k[4];
};

static void print_hex_le(const char* label, const uint64_t v[4]) {
    fprintf(stderr, "  %s = 0x%016llx%016llx%016llx%016llx\n",
            label,
            (unsigned long long)v[3], (unsigned long long)v[2],
            (unsigned long long)v[1], (unsigned long long)v[0]);
}

int main() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        fprintf(stderr, "No CUDA devices available: %s\n",
                cudaGetErrorString(err));
        return 77;  // CTest skip code
    }
    cudaSetDevice(0);

    // Build test cases.
    std::vector<TestCase> cases;

    cases.push_back({"k=1", {1, 0, 0, 0}});
    cases.push_back({"k=2", {2, 0, 0, 0}});
    cases.push_back({"k=3", {3, 0, 0, 0}});
    cases.push_back({"k=lambda",
        {LAMBDA_HOST[0], LAMBDA_HOST[1], LAMBDA_HOST[2], LAMBDA_HOST[3]}});

    // k = n - 1.
    {
        TestCase c;
        c.name = "k=n-1";
        uint64_t one[4] = {1, 0, 0, 0};
        host_sub256(c.k, N_HOST, one);
        cases.push_back(c);
    }
    // k = n - 2.
    {
        TestCase c;
        c.name = "k=n-2";
        uint64_t two[4] = {2, 0, 0, 0};
        host_sub256(c.k, N_HOST, two);
        cases.push_back(c);
    }
    // k = n - 7.
    {
        TestCase c;
        c.name = "k=n-7";
        uint64_t seven[4] = {7, 0, 0, 0};
        host_sub256(c.k, N_HOST, seven);
        cases.push_back(c);
    }
    // k = n / 2 + 1 (midpoint).
    {
        TestCase c;
        c.name = "k=n/2+1";
        c.k[0] = (N_HOST[0] >> 1) | (N_HOST[1] << 63);
        c.k[1] = (N_HOST[1] >> 1) | (N_HOST[2] << 63);
        c.k[2] = (N_HOST[2] >> 1) | (N_HOST[3] << 63);
        c.k[3] = N_HOST[3] >> 1;
        c.k[0] += 1;  // safe: low bit of n/2 may be 0 but adding 1 doesn't overflow
        cases.push_back(c);
    }

    // Deterministic PRNG seeded for reproducibility.
    std::mt19937_64 rng(0x6C7FA53E4AB9C8D1ULL);

    // 4 random 128-bit scalars.
    for (int t = 0; t < 4; t++) {
        TestCase c;
        c.name = "k=rand128";
        c.k[0] = rng();
        c.k[1] = rng();
        c.k[2] = 0;
        c.k[3] = 0;
        cases.push_back(c);
    }
    // 4 random 256-bit scalars (forced below n by clearing top bit set).
    for (int t = 0; t < 4; t++) {
        TestCase c;
        c.name = "k=rand256";
        c.k[0] = rng();
        c.k[1] = rng();
        c.k[2] = rng();
        c.k[3] = rng() & 0x7FFFFFFFFFFFFFFFULL;
        // Force k < n.
        while (host_cmp256(c.k, N_HOST) >= 0) {
            host_sub256(c.k, c.k, N_HOST);
        }
        cases.push_back(c);
    }

    // Near-n scalars: k = n - rand128(). These land in the top 2^128 of the
    // range and stress the Babai-rounding carry edge where the decompose
    // magnitude can reach ~2^128.5 -- the window-coverage assumption
    // ec_mul_glv depends on (audit solver M1). The prior 8-vector set
    // under-sampled this; sample it densely.
    for (int t = 0; t < 128; t++) {
        TestCase c;
        c.name = "k=n-rand128";
        uint64_t r[4] = { rng(), rng(), 0, 0 };
        host_sub256(c.k, N_HOST, r);   // n - r  in (n - 2^128, n)
        cases.push_back(c);
    }
    // 64 more full-width random scalars for breadth across the range.
    for (int t = 0; t < 64; t++) {
        TestCase c;
        c.name = "k=rand256b";
        c.k[0] = rng();
        c.k[1] = rng();
        c.k[2] = rng();
        c.k[3] = rng() & 0x7FFFFFFFFFFFFFFFULL;
        while (host_cmp256(c.k, N_HOST) >= 0) {
            host_sub256(c.k, c.k, N_HOST);
        }
        cases.push_back(c);
    }

    const int N = (int)cases.size();

    // Pack k values for GPU.
    std::vector<uint64_t> h_in(N * 4);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 4; j++) h_in[i * 4 + j] = cases[i].k[j];
    }

    uint64_t* d_in = nullptr;
    GlvOut* d_out = nullptr;
    cudaMalloc(&d_in, sizeof(uint64_t) * 4 * N);
    cudaMalloc(&d_out, sizeof(GlvOut) * N);
    cudaMemcpy(d_in, h_in.data(), sizeof(uint64_t) * 4 * N, cudaMemcpyHostToDevice);

    glv_decompose_kernel<<<(N + 31) / 32, 32>>>(d_in, d_out, N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_in);
        cudaFree(d_out);
        return 1;
    }

    std::vector<GlvOut> h_out(N);
    cudaMemcpy(h_out.data(), d_out, sizeof(GlvOut) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    // Verify each case.
    int failures = 0;
    for (int i = 0; i < N; i++) {
        const TestCase& c = cases[i];
        const GlvOut& o = h_out[i];

        // (1) Babai bound: k1[3] == 0 && k2[3] == 0.
        if (o.k1[3] != 0) {
            fprintf(stderr, "[FAIL] %s: k1[3] != 0 (Babai bound violated)\n", c.name);
            print_hex_le("k", c.k);
            print_hex_le("k1", o.k1);
            print_hex_le("k2", o.k2);
            failures++;
            continue;
        }
        if (o.k2[3] != 0) {
            fprintf(stderr, "[FAIL] %s: k2[3] != 0 (Babai bound violated)\n", c.name);
            print_hex_le("k", c.k);
            print_hex_le("k1", o.k1);
            print_hex_le("k2", o.k2);
            failures++;
            continue;
        }

        // (2) Algebraic identity: (signed_k1 + signed_k2 * lambda) mod n == k.
        uint64_t k1_eff[4], k2_eff[4];
        host_signed_to_mod_n(k1_eff, o.k1, o.k1_neg != 0);
        host_signed_to_mod_n(k2_eff, o.k2, o.k2_neg != 0);

        uint64_t k2_lambda[4];
        host_mod_mul_n(k2_lambda, k2_eff, LAMBDA_HOST);

        uint64_t reconstructed[4];
        host_mod_add(reconstructed, k1_eff, k2_lambda);

        if (host_cmp256(reconstructed, c.k) != 0) {
            fprintf(stderr, "[FAIL] %s: reconstruction mismatch\n", c.name);
            print_hex_le("k        ", c.k);
            print_hex_le("k1 (|.|) ", o.k1);
            fprintf(stderr, "  k1_neg = %u\n", (unsigned)o.k1_neg);
            print_hex_le("k2 (|.|) ", o.k2);
            fprintf(stderr, "  k2_neg = %u\n", (unsigned)o.k2_neg);
            print_hex_le("reconstr ", reconstructed);
            failures++;
            continue;
        }

        printf("[PASS] %s\n", c.name);
    }

    printf("\nSummary: %d/%d passed\n", N - failures, N);
    return failures == 0 ? 0 : 1;
}
