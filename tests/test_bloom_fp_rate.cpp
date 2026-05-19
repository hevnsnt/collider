// test_bloom_fp_rate — empirical bloom false-positive-rate KAT.
//
// Pins the actual FP rate of UTXOBloomBuilder at two production targets:
//
//   - Loose (1e-5): used by the GPU first-pass probe inside the fused
//     brain-wallet kernel. A false positive here costs one CPU-side
//     definitive verification per spurious hit, so the rate matters for
//     the throughput-loss tail.
//   - Tight (1e-7): used by the CPU-side verification bloom for the
//     loaded address set. A false positive here costs one full address
//     set lookup per hit; not catastrophic but visible.
//
// The bloom math (m, k from n and target FP) lives in
// UTXOBloomBuilder::calculate_parameters. The kernel-side probe uses the
// SAME MurmurHash3-128 + double-hashing scheme by contract. If anyone
// drifts the math (e.g. drops the pow-of-2 m rounding from G-B3, or
// changes the hash function), this test fails on two axes:
//
//   1. THEORETICAL ASSERTION: estimated_fp_rate() (computed from k, m, n)
//      must be at-or-below the target FP. The builder calculates m from
//      the target then rounds up to the next power of 2 (G-B3 fix for
//      the kernel's bitwise-AND fast path), so estimated_fp is always
//      <= target. A regression in calculate_parameters that under-sizes
//      m would push estimated_fp above target and fail here.
//
//   2. EMPIRICAL ASSERTION: the measured FP rate across 10M synthetic
//      non-member probes must match the theoretical rate within a 2x
//      band. A regression in the MurmurHash3-128 implementation, the
//      double-hash index derivation, or the bit-array indexing would
//      drift the empirical away from theoretical and fail here.
//
// The "within 2x of target" requirement from Tier 4 T7 is captured by
// the conjunction of both assertions: estimated <= target (so empirical
// must be <= roughly 2*target by the empirical-vs-theoretical band), and
// the implementation tracks theory tightly.
//
// Two-axis structure rationale: at n=1M the pow-2-rounding for the loose
// (1e-5) and tight (1e-7) targets produces the SAME m (the smaller m
// rounds up to meet the larger m's pow-2 boundary), so the two trials
// would otherwise be indistinguishable. By asserting against each trial's
// own theoretical figure we test that the implementation is internally
// consistent regardless of whether the rounding boundary collapses the
// two cases together at this n.
//
// Determinism: the seed split between member and non-member streams is
// constant, so this test is byte-deterministic across runs.

#include "../src/tools/utxo_bloom_builder.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

namespace {

using collider::utxo::H160;
using collider::utxo::UTXOBloomBuilder;

// Generate a deterministic stream of synthetic H160s. Caller-supplied
// seed lets the test produce a member stream and a disjoint non-member
// stream (different seeds, both ranges are sparse in 2^160 so collisions
// are negligible).
class H160Stream {
public:
    explicit H160Stream(std::uint64_t seed) : rng_(seed) {}

    H160 next() {
        H160 h{};
        // Fill 20 bytes with PRNG output. 3 uint64 draws cover 24 bytes;
        // we keep the first 20.
        std::uint64_t a = rng_();
        std::uint64_t b = rng_();
        std::uint64_t c = rng_();
        std::memcpy(h.data + 0, &a, 8);
        std::memcpy(h.data + 8, &b, 8);
        std::memcpy(h.data + 16, &c, 4);
        return h;
    }

private:
    std::mt19937_64 rng_;
};

// Trial parameters. Each trial asserts:
//   1. estimated_fp_rate() <= target_fp (the builder met its contract)
//   2. estimated_fp_rate() <= target_fp * 2 (within 2x band, redundant
//      with #1 but spelled out for the "2x" language from the spec)
//   3. expected_hits = empirical_target * n_queries >= 1 OR allow zero
//      (the 1e-7 trial at 10M queries has Poisson(1) expectation and
//      legitimately produces zero hits ~37% of the time)
//   4. when expected_hits >= 5, the empirical hit count is within a 2x
//      band of the theoretical expectation. Below 5 expected hits the
//      Poisson noise floor dominates and any literal "2x" check would
//      produce false failures.

struct BloomTrial {
    const char* label;
    double target_fp;
};

bool run_trial(const BloomTrial& trial,
               std::uint64_t n_inserts,
               std::uint64_t n_queries) {
    UTXOBloomBuilder::Config cfg;
    cfg.target_fp_rate = trial.target_fp;
    cfg.expected_elements = n_inserts;
    cfg.seed = 0x5F3759DFu;

    UTXOBloomBuilder bloom(cfg);

    auto stats0 = bloom.get_stats();
    std::cout << "  [" << trial.label << "] target=" << trial.target_fp
              << " bits=" << stats0.num_bits
              << " (" << (stats0.num_bits / 8 / 1024 / 1024) << " MiB)"
              << " hashes=" << stats0.num_hashes << "\n";

    // ---- Insert 1M synthetic H160s. Deterministic stream.
    H160Stream member_stream(0x1111111111111111ULL);
    std::vector<H160> members;
    members.reserve(static_cast<std::size_t>(n_inserts));
    for (std::uint64_t i = 0; i < n_inserts; ++i) {
        H160 h = member_stream.next();
        bloom.add_h160(h);
        members.push_back(h);
    }

    // estimated_fp_rate uses elements_added_ from the BUILDER, so it
    // returns the realistic FP rate only AFTER the inserts. Calling it
    // before the inserts returns pow(0, k) == 0 which is not useful for
    // the assertions below.
    const double theoretical = bloom.estimated_fp_rate();
    std::cout << "    theoretical_after_inserts=" << theoretical << "\n";

    // ---- Assertion 1+2: estimated FP must be at-or-below target.
    // This is the builder's contract: pick m, k so the theoretical FP
    // does not exceed the requested target. The pow-of-2 m rounding
    // tightens it further but the floor is target_fp.
    if (theoretical > trial.target_fp) {
        std::cerr << "    FAIL: theoretical " << theoretical
                  << " exceeds target " << trial.target_fp
                  << " (UTXOBloomBuilder::calculate_parameters regression)\n";
        return false;
    }
    if (theoretical > trial.target_fp * 2.0) {
        std::cerr << "    FAIL: theoretical " << theoretical
                  << " above 2x target " << trial.target_fp << "\n";
        return false;
    }

    // Self-check: every inserted member must probe true (no false negatives
    // by construction in a correctly-implemented bloom).
    for (std::uint64_t i = 0; i < n_inserts; ++i) {
        if (!bloom.probably_contains(members[i])) {
            std::cerr << "FAIL: false negative on inserted member " << i
                      << " (impossible for a correct bloom — check the impl)\n";
            return false;
        }
    }

    // ---- Query 10M synthetic non-member H160s. Different seed so the
    // stream does not overlap the member stream.
    H160Stream nonmember_stream(0x2222222222222222ULL);
    std::uint64_t hits = 0;
    for (std::uint64_t i = 0; i < n_queries; ++i) {
        H160 h = nonmember_stream.next();
        if (bloom.probably_contains(h)) {
            ++hits;
        }
    }

    const double empirical = static_cast<double>(hits) / static_cast<double>(n_queries);
    const double expected_hits = theoretical * static_cast<double>(n_queries);

    std::cout << "    inserts=" << n_inserts
              << " queries=" << n_queries
              << " hits=" << hits
              << " empirical=" << empirical
              << " expected_hits=" << expected_hits
              << "\n";

    // ---- Assertion 3+4: empirical within 2x band of theoretical IF the
    // expected hit count is large enough to escape Poisson noise. Below
    // ~5 expected hits the noise floor dominates: at expected=1.0, P(k>=3)
    // ≈ 8% so a literal 2x upper band would fail 8% of the time on a
    // correct implementation. We restrict the band check to when
    // expected_hits >= 5 and otherwise only assert empirical <= 4x target
    // (a softer envelope that still catches a broken implementation
    // emitting orders-of-magnitude more hits).
    bool ok = true;
    if (expected_hits >= 5.0) {
        const double lower = theoretical * 0.5;
        const double upper = theoretical * 2.0;
        if (empirical < lower || empirical > upper) {
            std::cerr << "    FAIL: empirical " << empirical
                      << " outside [" << lower << ", " << upper
                      << "] band around theoretical " << theoretical
                      << "\n";
            ok = false;
        }
    } else {
        // Soft envelope: a working impl should not produce >4x the
        // theoretical expectation at any sample size; well above the
        // Poisson tail at expected_hits ~ 1.
        const double upper = std::max(theoretical * 4.0, 5.0 / n_queries);
        if (empirical > upper) {
            std::cerr << "    FAIL: empirical " << empirical
                      << " exceeds soft upper bound " << upper
                      << " (expected_hits=" << expected_hits
                      << " is below the Poisson band threshold)\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "    PASS\n";
    }
    return ok;
}

}  // namespace

int main() {
    // 1M inserts, 10M queries per trial. Total runtime is dominated by
    // the 10M MurmurHash3-128 evaluations in the query loop (~3-5 seconds
    // per trial on a modern desktop CPU).
    constexpr std::uint64_t kInserts = 1'000'000ULL;
    constexpr std::uint64_t kQueries = 10'000'000ULL;

    std::cout << "=== bloom false-positive-rate KAT ===\n";

    bool all_ok = true;

    // Loose bloom: GPU first-pass probe target. 1e-5 over 10M queries
    // would produce ~100 expected hits if the implementation hit the
    // target exactly; after pow-2 rounding the theoretical drops below
    // target by however much rounding tightens m.
    BloomTrial loose{
        /*label=*/"loose GPU bloom",
        /*target_fp=*/1e-5,
    };
    all_ok &= run_trial(loose, kInserts, kQueries);

    // Tight bloom: CPU-side verification target. 1e-7 over 10M queries
    // is at the Poisson-noise threshold (expected ~1 hit) so the empirical
    // band check falls into the soft-envelope branch.
    BloomTrial tight{
        /*label=*/"tight CPU bloom",
        /*target_fp=*/1e-7,
    };
    all_ok &= run_trial(tight, kInserts, kQueries);

    if (!all_ok) {
        std::cerr << "test_bloom_fp_rate: FAIL\n";
        return 1;
    }
    std::cout << "test_bloom_fp_rate: PASS\n";
    return 0;
}
