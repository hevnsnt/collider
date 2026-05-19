/**
 * test_random_below.cpp -- v1.4.2 A.3 regression test.
 *
 * The kangaroo Tame init pre-fix used per-limb `% range_size` which is
 * mathematically nonsensical (per-limb mod doesn't bound the full value)
 * AND had no branch for 3+-limb ranges, so on puzzles past 128 bits the
 * Tame kangaroos started anywhere on the curve.
 *
 * cpu::random_below now does bit-masked rejection sampling, uniformly in
 * [0, modulus). These cases verify all sample sizes (1-limb, 2-limb,
 * 3-limb, 4-limb) plus the SECP256K1_N edge.
 */

#include "../src/core/crypto_cpu.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>

using collider::cpu::uint256_t;
using collider::cpu::SECP256K1_N;

namespace {

int g_pass = 0;
int g_fail = 0;

#define EXPECT_LESS(a, b, label)                                            \
    do {                                                                    \
        if ((a) < (b)) {                                                    \
            ++g_pass;                                                       \
        } else {                                                            \
            ++g_fail;                                                       \
            std::cerr << "[FAIL] " << (label) << "\n"                       \
                      << "  expected sample < modulus\n";                   \
        }                                                                   \
    } while (0)

#define EXPECT_TRUE(cond, label)                                            \
    do {                                                                    \
        if (cond) {                                                         \
            ++g_pass;                                                       \
        } else {                                                            \
            ++g_fail;                                                       \
            std::cerr << "[FAIL] " << (label) << "\n";                      \
        }                                                                   \
    } while (0)

// Run N samples and check every one is in [0, modulus).
void test_bounded(const char* label, const uint256_t& modulus, int n_samples = 10000) {
    std::mt19937_64 rng(0xDEADBEEF12345678ULL);
    auto rng_callable = [&]() { return rng(); };

    int bound_violations = 0;
    for (int i = 0; i < n_samples; ++i) {
        uint256_t sample;
        collider::cpu::random_below(sample, rng_callable, modulus);
        if (!(sample < modulus)) {
            ++bound_violations;
        }
    }
    if (bound_violations == 0) {
        ++g_pass;
        std::cout << "[ok  ] " << label << ": " << n_samples
                  << " samples all in [0, modulus)\n";
    } else {
        ++g_fail;
        std::cerr << "[FAIL] " << label << ": " << bound_violations
                  << " out of " << n_samples << " samples violated bound\n";
    }
}

void test_zero_modulus() {
    std::mt19937_64 rng(1);
    uint256_t out;
    collider::cpu::random_below(out, [&]() { return rng(); }, uint256_t(0));
    EXPECT_TRUE(out.is_zero(), "random_below(rng, 0) returns 0");
}

void test_one_modulus() {
    std::mt19937_64 rng(2);
    auto rng_callable = [&]() { return rng(); };
    bool all_zero = true;
    for (int i = 0; i < 100; ++i) {
        uint256_t out;
        collider::cpu::random_below(out, rng_callable, uint256_t(1));
        if (!out.is_zero()) { all_zero = false; break; }
    }
    EXPECT_TRUE(all_zero, "random_below(rng, 1) is always 0");
}

void test_distribution_small() {
    // For modulus=10, all values 0..9 should appear with roughly equal
    // frequency over 100k samples. Chi-squared lite: max bucket should
    // not exceed expected mean by more than ~50%.
    std::mt19937_64 rng(0xBAADF00DDEADBEEFULL);
    auto rng_callable = [&]() { return rng(); };
    const uint256_t modulus(10);
    const int n = 100000;
    int counts[10] = {0};
    for (int i = 0; i < n; ++i) {
        uint256_t out;
        collider::cpu::random_below(out, rng_callable, modulus);
        // Out's d[0] should be the value (all higher limbs zero).
        counts[(int)out.d[0]]++;
    }
    int min_c = counts[0], max_c = counts[0];
    for (int i = 0; i < 10; ++i) {
        if (counts[i] < min_c) min_c = counts[i];
        if (counts[i] > max_c) max_c = counts[i];
    }
    // Each bucket should be ~10000. Allow 30% deviation for n=100k.
    const int expected = n / 10;
    bool ok = (min_c > expected * 7 / 10) && (max_c < expected * 13 / 10);
    if (ok) {
        ++g_pass;
        std::cout << "[ok  ] distribution mod=10: bucket counts in ["
                  << min_c << ", " << max_c << "] (expected ~"
                  << expected << ")\n";
    } else {
        ++g_fail;
        std::cerr << "[FAIL] distribution mod=10: buckets unbalanced ["
                  << min_c << ", " << max_c << "]\n";
    }
}

}  // namespace

int main() {
    std::cout << "test_random_below (v1.4.2 A.3 regression suite)\n";

    test_zero_modulus();
    test_one_modulus();
    test_distribution_small();

    // 1-limb modulus
    test_bounded("1-limb (2^40)",
                 uint256_t(uint64_t(1) << 40, 0, 0, 0));
    // 2-limb modulus (the case the pre-fix code half-handled)
    test_bounded("2-limb (2^100)",
                 uint256_t(0, uint64_t(1) << 36, 0, 0));
    // 3-limb modulus (the case the pre-fix code SILENTLY DROPPED -
    // this is the puzzle-130 / puzzle-200 territory)
    test_bounded("3-limb (2^160)",
                 uint256_t(0, 0, uint64_t(1) << 32, 0));
    // 4-limb modulus
    test_bounded("4-limb (2^200)",
                 uint256_t(0, 0, 0, uint64_t(1) << 8));
    // SECP256K1_N
    test_bounded("SECP256K1_N (group order)", SECP256K1_N);
    // n - 1 (the most pathological modulus, one less than the high mask)
    {
        uint256_t n_minus_1;
        collider::cpu::sub256(n_minus_1, SECP256K1_N, uint256_t(1));
        test_bounded("SECP256K1_N - 1", n_minus_1);
    }

    std::cout << "Summary: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
