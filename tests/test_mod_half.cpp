/**
 * test_mod_half.cpp -- v1.4.2 A.1 regression test.
 *
 * The kangaroo Wild1-Wild2 collision recovery formula
 *   k = (dist_w2 - dist_w1) / 2 (mod n)
 * requires modular halving over the group order n. Pre-fix, this was
 * implemented as a naive `a >> 1` (right-shift), which is correct only
 * when a is even. For odd a, the recovered key was wrong by ~(n-1)/2 -
 * a silent wrong-key bug on ~50% of Wild1-Wild2 collisions.
 *
 * These cases exercise both parities and the canonical edges.
 */

#include "../src/core/crypto_cpu.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

using collider::cpu::uint256_t;
using collider::cpu::SECP256K1_N;

namespace {

int g_pass = 0;
int g_fail = 0;

#define EXPECT_EQ_UINT256(actual, expected, label)                          \
    do {                                                                    \
        if ((actual) == (expected)) {                                       \
            ++g_pass;                                                       \
        } else {                                                            \
            ++g_fail;                                                       \
            std::cerr << "[FAIL] " << (label) << "\n"                       \
                      << "  expected: " << std::hex                         \
                      << (expected).d[3] << " " << (expected).d[2] << " "   \
                      << (expected).d[1] << " " << (expected).d[0] << "\n"  \
                      << "  actual:   "                                     \
                      << (actual).d[3] << " " << (actual).d[2] << " "       \
                      << (actual).d[1] << " " << (actual).d[0] << std::dec  \
                      << "\n";                                              \
        }                                                                   \
    } while (0)

// Compute 2*k mod n using the existing modular adder. Returns r and asserts
// that the canonical representative is in [0, n).
void double_mod_n(uint256_t& r, const uint256_t& k) {
    uint64_t carry = collider::cpu::add256(r, k, k);
    // If the doubled value overflowed 256 bits OR exceeds n, subtract n.
    // Since k < n < 2^256, the maximum carry is 1.
    if (carry || r >= SECP256K1_N) {
        collider::cpu::sub256(r, r, SECP256K1_N);
    }
}

// Modular subtraction over n (mirrors the kangaroo collision-recovery
// branch that computes `diff = (wild2_d - wild1_d) mod n`).
void sub_mod_n(uint256_t& r, const uint256_t& a, const uint256_t& b) {
    if (collider::cpu::sub256(r, a, b)) {
        // Borrowed: a < b. Result is (a - b + n) = n - (b - a).
        collider::cpu::add256(r, r, SECP256K1_N);
    }
}

void test_zero() {
    uint256_t r;
    collider::cpu::mod_half(r, uint256_t(0), SECP256K1_N);
    EXPECT_EQ_UINT256(r, uint256_t(0), "mod_half(0, n) == 0");
}

void test_two() {
    // 2 / 2 mod n = 1. Tests the even-fast-path.
    uint256_t r;
    collider::cpu::mod_half(r, uint256_t(2), SECP256K1_N);
    EXPECT_EQ_UINT256(r, uint256_t(1), "mod_half(2, n) == 1");
}

void test_one_odd_path() {
    // 1 / 2 mod n = (n+1)/2.
    // n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    // n+1 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364142
    // (n+1)/2 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A1
    uint256_t expected(
        0xDFE92F46681B20A1ULL,
        0x5D576E7357A4501DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x7FFFFFFFFFFFFFFFULL);
    uint256_t r;
    collider::cpu::mod_half(r, uint256_t(1), SECP256K1_N);
    EXPECT_EQ_UINT256(r, expected, "mod_half(1, n) == (n+1)/2");
}

void test_three_odd_path() {
    // 3 / 2 mod n = (n+3)/2 = (n+1)/2 + 1.
    uint256_t expected(
        0xDFE92F46681B20A2ULL,
        0x5D576E7357A4501DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x7FFFFFFFFFFFFFFFULL);
    uint256_t r;
    collider::cpu::mod_half(r, uint256_t(3), SECP256K1_N);
    EXPECT_EQ_UINT256(r, expected, "mod_half(3, n) == (n+3)/2");
}

void test_n_minus_1() {
    // (n-1) is even, so result is (n-1)/2. Tests even-fast-path at boundary.
    uint256_t n_minus_1;
    collider::cpu::sub256(n_minus_1, SECP256K1_N, uint256_t(1));
    uint256_t expected(
        0xDFE92F46681B20A0ULL,
        0x5D576E7357A4501DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x7FFFFFFFFFFFFFFFULL);
    uint256_t r;
    collider::cpu::mod_half(r, n_minus_1, SECP256K1_N);
    EXPECT_EQ_UINT256(r, expected, "mod_half(n-1, n) == (n-1)/2");
}

void test_n_minus_2_odd_path_with_carry() {
    // (n-2) is odd (n is odd, so n-2 is odd). Adding n: (n-2 + n) = 2n - 2,
    // which is *just* below 2^257. Exercises the 257-bit carry path.
    uint256_t n_minus_2;
    collider::cpu::sub256(n_minus_2, SECP256K1_N, uint256_t(2));
    // (n-2)/2 mod n: 2 * ((n-2)/2) mod n = n - 2 = -2 mod n, so result is
    // the value `r` such that 2r mod n = n - 2.
    // 2*((n-1)/2 - something)... easier: compute the answer directly.
    // (n + (n-2))/2 = (2n-2)/2 = n - 1.
    uint256_t expected;
    collider::cpu::sub256(expected, SECP256K1_N, uint256_t(1));
    uint256_t r;
    collider::cpu::mod_half(r, n_minus_2, SECP256K1_N);
    EXPECT_EQ_UINT256(r, expected, "mod_half(n-2, n) == n-1");
}

void test_roundtrip_random_even() {
    // For known random k with 2k < n (so diff = 2k is even), mod_half(diff, n)
    // recovers k. This is the case the pre-fix code happened to handle.
    uint256_t k(0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL,
                0x1111222233334444ULL, 0x0000000055556666ULL);  // < n/2
    uint256_t two_k;
    double_mod_n(two_k, k);
    // Sanity: 2k must be even.
    assert(!two_k.is_odd());
    uint256_t recovered;
    collider::cpu::mod_half(recovered, two_k, SECP256K1_N);
    EXPECT_EQ_UINT256(recovered, k, "roundtrip even: half(2k) == k");
}

void test_roundtrip_random_odd() {
    // For k > n/2, 2k mod n = 2k - n. Since n is odd, this is ODD.
    // This is the case the pre-fix code FAILED on - the W1-W2 silent
    // wrong-key bug. With the fix, we recover the original k.
    uint256_t k;
    // Pick k = n - 12345 (well above n/2).
    collider::cpu::sub256(k, SECP256K1_N, uint256_t(12345));
    uint256_t two_k;
    double_mod_n(two_k, k);
    // Sanity: 2k mod n must be odd.
    assert(two_k.is_odd());
    uint256_t recovered;
    collider::cpu::mod_half(recovered, two_k, SECP256K1_N);
    EXPECT_EQ_UINT256(recovered, k, "roundtrip odd: half(2k) == k for k > n/2");
}

void test_wild1_wild2_scenario() {
    // Simulates the exact kangaroo W1-W2 collision recovery:
    //   Wild1 position = (k + d_w1) * G
    //   Wild2 position = (-k + d_w2) * G
    //   Collision: (k + d_w1) == (-k + d_w2) mod n
    //   => 2k == d_w2 - d_w1 mod n
    //   => k == (d_w2 - d_w1) / 2 mod n
    //
    // For the buggy pre-fix code, we'd see this fail whenever d_w2-d_w1
    // is odd mod n - which happens on ~50% of valid collisions.
    uint256_t k(0xDEADBEEFCAFEBABEULL, 0x1122334455667788ULL,
                0x99AABBCCDDEEFF00ULL, 0x0000000000007777ULL);

    // Pick arbitrary d_w1, derive d_w2 such that d_w2 - d_w1 == 2k mod n.
    uint256_t d_w1(0x1234567890ABCDEFULL, 0xFEDCBA9876543210ULL,
                   0x0101010101010101ULL, 0x0000000000000003ULL);
    uint256_t two_k;
    double_mod_n(two_k, k);
    uint256_t d_w2;
    // d_w2 = (d_w1 + 2k) mod n
    if (collider::cpu::add256(d_w2, d_w1, two_k) || d_w2 >= SECP256K1_N) {
        collider::cpu::sub256(d_w2, d_w2, SECP256K1_N);
    }

    // The collision-recovery code computes diff = (d_w2 - d_w1) mod n.
    uint256_t diff;
    sub_mod_n(diff, d_w2, d_w1);

    // diff must equal 2k mod n by construction.
    EXPECT_EQ_UINT256(diff, two_k, "W1-W2 diff == 2k mod n");

    // The fixed code computes k = mod_half(diff, n).
    uint256_t recovered;
    collider::cpu::mod_half(recovered, diff, SECP256K1_N);
    EXPECT_EQ_UINT256(recovered, k, "W1-W2 recovered k matches original");
}

}  // namespace

int main() {
    std::cout << "test_mod_half (v1.4.2 A.1 regression suite)\n";

    test_zero();
    test_two();
    test_one_odd_path();
    test_three_odd_path();
    test_n_minus_1();
    test_n_minus_2_odd_path_with_carry();
    test_roundtrip_random_even();
    test_roundtrip_random_odd();
    test_wild1_wild2_scenario();

    std::cout << "Summary: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
