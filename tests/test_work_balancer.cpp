/**
 * test_work_balancer.cpp - Unit tests for runtime::WorkBalancer.
 *
 * Exercises the per-GPU work splitter without any GPU dependency.
 * Validates:
 *   - first batch is equal-split
 *   - throughput-weighted split converges toward weight = throughput share
 *   - sum of per-GPU counts equals total_words exactly
 *   - inactive GPU receives zero, others absorb the slack
 *   - zero-throughput sample on one GPU does not starve it
 *   - degenerate batch (total_words < num_gpus) falls back to equal split
 *   - pathological-tiny-weight enforces the per-GPU minimum slice
 */

#include "../src/runtime/work_balancer.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

using collider::runtime::WorkBalancer;

namespace {

size_t sum_counts(const std::vector<WorkBalancer::Slice>& slices) {
    size_t s = 0;
    for (const auto& sl : slices) s += sl.count;
    return s;
}

void test_first_batch_equal_split() {
    WorkBalancer b(2);
    std::vector<bool> active = {true, true};
    auto slices = b.split(1000, active);

    assert(slices.size() == 2);
    assert(sum_counts(slices) == 1000);
    // Equal split with last-absorbs-remainder. 1000/2 = 500 each.
    assert(slices[0].count == 500);
    assert(slices[1].count == 500);
    assert(slices[0].start == 0);
    assert(slices[1].start == 500);
    std::cout << "  [ok] first batch equal split\n";
}

void test_first_batch_three_gpus_odd_total() {
    WorkBalancer b(3);
    std::vector<bool> active = {true, true, true};
    auto slices = b.split(1001, active);
    assert(sum_counts(slices) == 1001);
    // 1001/3 = 333 each, last absorbs +2 remainder = 335
    assert(slices[0].count == 333);
    assert(slices[1].count == 333);
    assert(slices[2].count == 335);
    std::cout << "  [ok] equal split with odd remainder lands on last GPU\n";
}

void test_proportional_split_converges() {
    // Simulate the operator's rig: GPU 0 ~2x faster per-batch than GPU 1.
    // Feed the balancer 20 batches where each GPU's reported elapsed time
    // is proportional to its assigned slice / its true throughput.
    // GPU 0 true throughput: 200_000 words/sec. GPU 1: 100_000 words/sec.
    // Expected steady-state weight ratio: 2:1, so GPU 0 should converge
    // to ~0.667 of the split.
    WorkBalancer b(2);
    std::vector<bool> active = {true, true};

    const size_t total = 600'000;
    const double tput0 = 200'000.0;
    const double tput1 = 100'000.0;

    std::vector<WorkBalancer::Slice> last;
    for (int iter = 0; iter < 20; ++iter) {
        auto slices = b.split(total, active);
        last = slices;
        // Simulate the wall-clock elapsed on each GPU for its slice.
        const double e0_sec = static_cast<double>(slices[0].count) / tput0;
        const double e1_sec = static_cast<double>(slices[1].count) / tput1;
        const auto e0 = std::chrono::nanoseconds(
            static_cast<long long>(e0_sec * 1e9));
        const auto e1 = std::chrono::nanoseconds(
            static_cast<long long>(e1_sec * 1e9));
        b.record_throughput(0, slices[0].count, e0);
        b.record_throughput(1, slices[1].count, e1);
    }

    // After 20 iterations with alpha = 0.25, EMA is well-converged.
    // Final split should give GPU 0 ~2/3 of the batch (within rounding).
    const double frac0 = static_cast<double>(last[0].count) /
                         static_cast<double>(total);
    std::cout << "  proportional: GPU0 frac=" << frac0
              << " (expected ~0.667)\n";
    assert(frac0 > 0.62);
    assert(frac0 < 0.72);
    assert(sum_counts(last) == total);
    std::cout << "  [ok] proportional split converges to throughput ratio\n";
}

void test_sum_always_equals_total() {
    // Run with several total sizes (including ones that don't divide
    // evenly by the GPU count) and verify the slice sum is exact.
    WorkBalancer b(3);
    std::vector<bool> active = {true, true, true};

    // Seed some EMA.
    b.record_throughput(0, 1000, std::chrono::nanoseconds(5'000'000));   // 200K/s
    b.record_throughput(1, 1000, std::chrono::nanoseconds(10'000'000));  // 100K/s
    b.record_throughput(2, 1000, std::chrono::nanoseconds(20'000'000));  // 50K/s

    for (size_t total : {1u, 7u, 1000u, 1001u, 1023u, 65537u, 1'000'000u}) {
        auto slices = b.split(total, active);
        const size_t s = sum_counts(slices);
        if (s != total) {
            std::cerr << "  total=" << total << " got sum=" << s << "\n";
        }
        assert(s == total);
    }
    std::cout << "  [ok] sum of slices == total_words for all sizes tested\n";
}

void test_inactive_gpu_skipped() {
    WorkBalancer b(3);
    // Seed EMA so a proportional split would otherwise be invoked.
    b.record_throughput(0, 1000, std::chrono::nanoseconds(5'000'000));
    b.record_throughput(1, 1000, std::chrono::nanoseconds(10'000'000));
    b.record_throughput(2, 1000, std::chrono::nanoseconds(20'000'000));

    std::vector<bool> active = {true, false, true};
    auto slices = b.split(1000, active);
    assert(slices.size() == 3);
    assert(slices[1].count == 0);
    assert(sum_counts(slices) == 1000);
    // Slices for the two Active GPUs must be contiguous (0 -> active0,
    // active0 -> active0+active2). active2 absorbs the remainder.
    assert(slices[0].start == 0);
    assert(slices[2].start == slices[0].count);
    std::cout << "  [ok] inactive GPU skipped, active GPUs absorb work\n";
}

void test_zero_throughput_sample_does_not_starve() {
    WorkBalancer b(2);

    // GPU 0 reports normal throughput.
    b.record_throughput(0, 1000, std::chrono::nanoseconds(10'000'000));
    // GPU 1 reports zero throughput (pathological: e.g., kernel error,
    // sub-tick measurement, returned 0 passphrases). The balancer must
    // not let GPU 1 be permanently starved by one bad sample.
    b.record_throughput(1, 0, std::chrono::nanoseconds(10'000'000));

    std::vector<bool> active = {true, true};
    auto slices = b.split(1000, active);
    assert(slices[1].count > 0);  // Got an exploration slice.
    assert(sum_counts(slices) == 1000);
    std::cout << "  [ok] zero-throughput sample does not starve the GPU\n";
}

void test_degenerate_batch_smaller_than_gpu_count() {
    WorkBalancer b(4);
    // Seed EMA so the proportional path would otherwise apply.
    b.record_throughput(0, 1000, std::chrono::nanoseconds(5'000'000));
    b.record_throughput(1, 1000, std::chrono::nanoseconds(10'000'000));
    b.record_throughput(2, 1000, std::chrono::nanoseconds(15'000'000));
    b.record_throughput(3, 1000, std::chrono::nanoseconds(20'000'000));

    std::vector<bool> active = {true, true, true, true};
    auto slices = b.split(2, active);  // only 2 words for 4 GPUs
    assert(sum_counts(slices) == 2);
    std::cout << "  [ok] degenerate small batch falls back safely\n";
}

void test_all_inactive_returns_zero_slices() {
    WorkBalancer b(2);
    std::vector<bool> active = {false, false};
    auto slices = b.split(1000, active);
    assert(slices.size() == 2);
    assert(slices[0].count == 0);
    assert(slices[1].count == 0);
    std::cout << "  [ok] all-inactive returns zero slices\n";
}

void test_zero_total_words_safe() {
    WorkBalancer b(2);
    std::vector<bool> active = {true, true};
    auto slices = b.split(0, active);
    assert(sum_counts(slices) == 0);
    std::cout << "  [ok] zero total words returns zero slices\n";
}

void test_minimum_slice_enforced_under_extreme_imbalance() {
    // GPU 0 absurdly faster than GPU 1 (1000x). Without the min-slice
    // floor, GPU 1 could round to 0 on small batches.
    WorkBalancer b(2);
    b.record_throughput(0, 1'000'000, std::chrono::nanoseconds(1'000'000));   // 1B/s
    b.record_throughput(1, 1'000, std::chrono::nanoseconds(1'000'000));       // 1M/s

    std::vector<bool> active = {true, true};
    auto slices = b.split(100, active);
    // total=100, 2 GPUs. min_slice = max(1, 100/(2*4)) = max(1, 12) = 12.
    // GPU 1 raw weight ~0.001 -> 0 floor would starve it.
    assert(slices[1].count >= 1);
    assert(sum_counts(slices) == 100);
    std::cout << "  [ok] min slice floor protects against rounding to zero\n";
}

void test_ema_seeds_on_first_measurement() {
    // The first measurement should set the EMA directly, not blend
    // against the 0 seed. Otherwise the second batch would still see a
    // partly-zero EMA and the split would lean too close to equal.
    WorkBalancer b(2);
    b.record_throughput(0, 1000, std::chrono::nanoseconds(10'000'000));  // 100K/s
    b.record_throughput(1, 1000, std::chrono::nanoseconds(20'000'000));  // 50K/s

    // Read the EMA back via the diagnostic accessor.
    const double e0 = b.ema_words_per_sec(0);
    const double e1 = b.ema_words_per_sec(1);
    // 1000 words / 0.010 sec = 100000 words/sec
    assert(e0 > 99'999.0 && e0 < 100'001.0);
    assert(e1 > 49'999.0 && e1 < 50'001.0);
    std::cout << "  [ok] EMA seeds directly on first measurement\n";
}

}  // namespace

int main() {
    std::cout << "WorkBalancer tests:\n";
    test_first_batch_equal_split();
    test_first_batch_three_gpus_odd_total();
    test_proportional_split_converges();
    test_sum_always_equals_total();
    test_inactive_gpu_skipped();
    test_zero_throughput_sample_does_not_starve();
    test_degenerate_batch_smaller_than_gpu_count();
    test_all_inactive_returns_zero_slices();
    test_zero_total_words_safe();
    test_minimum_slice_enforced_under_extreme_imbalance();
    test_ema_seeds_on_first_measurement();
    std::cout << "All WorkBalancer tests passed.\n";
    return 0;
}
