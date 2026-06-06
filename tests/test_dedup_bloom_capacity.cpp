// test_dedup_bloom_capacity — GEN-2 over-capacity guard for DedupBloomFilter.
//
// DedupBloomFilter (src/core/dedup_bloom.hpp) gives a bounded false-positive
// guarantee — and therefore bounded silent keyspace misses — ONLY up to the
// element count its (num_bits, num_hashes) pair was sized for. Past that
// design capacity the FP rate climbs and test_and_set() begins returning
// "duplicate" for genuinely-new candidates, silently dropping unique work.
// Unlike CandidatePriorityQueue there is no exact backstop, so the contract
// is: count the inserts, expose capacity, and emit a ONE-TIME alarm when the
// load crosses capacity.
//
// This KAT pins that contract:
//
//   1. count() tracks distinct inserts and ignores duplicates.
//   2. over_capacity() / over_capacity_warning_emitted() flip true exactly
//      when count() crosses design_capacity(), and the warning is emitted at
//      most once (idempotent on further inserts).
//   3. clear() resets the counter AND re-arms the one-time warning.
//   4. BatchDedupHelper.flush() returns the globally-unique count and that
//      count is reflected in the global filter's count().
//
// Deterministic and host-only: no GPU, no network, fixed string streams.

#include "../src/core/dedup_bloom.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>

using collider::BatchDedupHelper;
using collider::DedupBloomFilter;

namespace {

// A small filter so design capacity is reachable in a few hundred inserts.
// num_bits=4000, num_hashes=5 -> capacity = (4000/5)*ln2 ~= 554.
constexpr std::size_t kTestBits = 4000;
constexpr std::size_t kTestHashes = 5;

bool test_count_tracks_distinct_inserts() {
    DedupBloomFilter f(kTestBits, kTestHashes);

    if (f.count() != 0) {
        std::cerr << "FAIL: fresh filter count() != 0 (" << f.count() << ")\n";
        return false;
    }

    // Insert 100 distinct keys.
    for (int i = 0; i < 100; ++i) {
        bool is_new = f.test_and_set("key-" + std::to_string(i));
        if (!is_new) {
            // A false positive this early is astronomically unlikely with
            // capacity ~554; treat it as a real failure of the impl.
            std::cerr << "FAIL: distinct key " << i
                      << " reported as duplicate (count=" << f.count() << ")\n";
            return false;
        }
    }
    if (f.count() != 100) {
        std::cerr << "FAIL: count() after 100 distinct inserts = " << f.count()
                  << ", expected 100\n";
        return false;
    }

    // Re-insert the same 100 keys: all duplicates, count must not move.
    for (int i = 0; i < 100; ++i) {
        bool is_new = f.test_and_set("key-" + std::to_string(i));
        if (is_new) {
            std::cerr << "FAIL: repeated key " << i
                      << " reported as new (false negative on dedup)\n";
            return false;
        }
    }
    if (f.count() != 100) {
        std::cerr << "FAIL: count() changed on duplicate inserts = "
                  << f.count() << ", expected 100\n";
        return false;
    }

    std::cout << "  [count] distinct-insert accounting: PASS (capacity="
              << f.design_capacity() << ")\n";
    return true;
}

bool test_over_capacity_warning_path() {
    DedupBloomFilter f(kTestBits, kTestHashes);
    const std::uint64_t capacity = f.design_capacity();

    if (capacity == 0) {
        std::cerr << "FAIL: design_capacity() == 0 for a non-empty filter\n";
        return false;
    }
    if (f.over_capacity()) {
        std::cerr << "FAIL: fresh filter already reports over_capacity()\n";
        return false;
    }
    if (f.over_capacity_warning_emitted()) {
        std::cerr << "FAIL: fresh filter already emitted over-capacity warning\n";
        return false;
    }

    // Insert distinct keys up to and just past capacity. The over-capacity
    // predicate must be false at/below capacity and true once we cross it.
    // (The header writes a one-time WARNING line to stderr when crossing —
    // expected and intentional; the assertions below read the testable
    // predicate rather than scraping stderr.)
    const std::uint64_t overshoot = capacity + 50;
    for (std::uint64_t i = 0; i < overshoot; ++i) {
        f.test_and_set("cap-" + std::to_string(i));

        const bool predicate = f.over_capacity();
        const bool expected = f.count() > capacity;
        if (predicate != expected) {
            std::cerr << "FAIL: over_capacity() (" << predicate
                      << ") disagrees with count>" << capacity << " ("
                      << expected << ") at count=" << f.count() << "\n";
            return false;
        }
    }

    if (!f.over_capacity()) {
        std::cerr << "FAIL: over_capacity() false after inserting past capacity"
                  << " (count=" << f.count() << ", capacity=" << capacity
                  << ")\n";
        return false;
    }
    if (!f.over_capacity_warning_emitted()) {
        std::cerr << "FAIL: over-capacity warning flag not set after crossing"
                  << " capacity (count=" << f.count() << ")\n";
        return false;
    }

    std::cout << "  [warn] over-capacity path reached at count=" << f.count()
              << " (capacity=" << capacity << "): PASS\n";
    return true;
}

bool test_clear_resets_counter_and_rearms_warning() {
    DedupBloomFilter f(kTestBits, kTestHashes);
    const std::uint64_t capacity = f.design_capacity();

    // Drive it over capacity to set the counter and the warning flag.
    for (std::uint64_t i = 0; i < capacity + 25; ++i) {
        f.test_and_set("c1-" + std::to_string(i));
    }
    assert(f.over_capacity());
    assert(f.over_capacity_warning_emitted());

    f.clear();

    if (f.count() != 0) {
        std::cerr << "FAIL: count() after clear() = " << f.count()
                  << ", expected 0\n";
        return false;
    }
    if (f.over_capacity()) {
        std::cerr << "FAIL: over_capacity() still true after clear()\n";
        return false;
    }
    if (f.over_capacity_warning_emitted()) {
        std::cerr << "FAIL: warning flag not re-armed after clear()\n";
        return false;
    }

    // A key inserted after clear() must read as NEW again (bits were wiped).
    if (!f.test_and_set("c1-0")) {
        std::cerr << "FAIL: key not new after clear() (bits not reset)\n";
        return false;
    }
    if (f.count() != 1) {
        std::cerr << "FAIL: count() = " << f.count()
                  << " after one insert post-clear(), expected 1\n";
        return false;
    }

    std::cout << "  [clear] counter reset + warning re-armed: PASS\n";
    return true;
}

bool test_batch_dedup_helper_flush_count() {
    DedupBloomFilter f(kTestBits, kTestHashes);

    std::size_t reported_unique = 0;
    {
        BatchDedupHelper helper(f);
        // 200 distinct + 200 in-batch duplicates. The helper dedups within
        // the batch via its local set, so flush() should push exactly 200
        // unique keys to the global filter.
        for (int i = 0; i < 200; ++i) {
            helper.add("batch-" + std::to_string(i));
        }
        for (int i = 0; i < 200; ++i) {
            helper.add("batch-" + std::to_string(i));  // duplicates
        }
        reported_unique = helper.flush();
    }

    if (reported_unique != 200) {
        std::cerr << "FAIL: BatchDedupHelper.flush() returned "
                  << reported_unique << " globally-unique, expected 200\n";
        return false;
    }
    if (f.count() != 200) {
        std::cerr << "FAIL: global filter count() after flush = " << f.count()
                  << ", expected 200\n";
        return false;
    }

    std::cout << "  [batch] flush() unique-count + global count(): PASS\n";
    return true;
}

}  // namespace

int main() {
    std::cout << "=== dedup_bloom over-capacity guard (GEN-2) ===\n";

    bool ok = true;
    ok &= test_count_tracks_distinct_inserts();
    ok &= test_over_capacity_warning_path();
    ok &= test_clear_resets_counter_and_rearms_warning();
    ok &= test_batch_dedup_helper_flush_count();

    if (!ok) {
        std::cerr << "test_dedup_bloom_capacity: FAIL\n";
        return 1;
    }
    std::cout << "test_dedup_bloom_capacity: PASS\n";
    return 0;
}
