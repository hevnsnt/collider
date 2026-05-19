/**
 * test_bruteforce_generator -- coverage for the alphanumeric
 * incremental bruteforce generator.
 *
 * Pins:
 *   1. Length-1 covers the full 62-char alphanumeric set in ASCII lex order.
 *   2. Length-2 produces 62*62 = 3844 unique strings starting at "00" and
 *      ending at "zz".
 *   3. Multi-length runs iterate shortest-first.
 *   4. Resume from a mid-point snapshot continues from the exact next
 *      candidate.
 *   5. Batch boundaries don't drop or duplicate candidates.
 *   6. done() flips true only after the last candidate is emitted.
 */

#include "src/generators/bruteforce_generator.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace bf = ::collider::generators;

namespace {

int g_fails = 0;

void expect_eq_str(const std::string& got, const std::string& want, const char* label) {
    if (got != want) {
        std::fprintf(stderr, "[FAIL] %s: got '%s', want '%s'\n",
                     label, got.c_str(), want.c_str());
        ++g_fails;
    }
}

void expect_eq_u64(uint64_t got, uint64_t want, const char* label) {
    if (got != want) {
        std::fprintf(stderr, "[FAIL] %s: got %llu, want %llu\n",
                     label,
                     static_cast<unsigned long long>(got),
                     static_cast<unsigned long long>(want));
        ++g_fails;
    }
}

void expect_true(bool cond, const char* label) {
    if (!cond) {
        std::fprintf(stderr, "[FAIL] %s\n", label);
        ++g_fails;
    }
}

// Drain the generator into a flat vector, in 13-string batches so we
// exercise the batch-boundary path.
std::vector<std::string> drain(bf::BruteforceGenerator& g, size_t batch = 13) {
    std::vector<std::string> all;
    while (!g.done()) {
        auto chunk = g.next_batch(batch);
        if (chunk.empty()) break;
        for (auto& s : chunk) all.push_back(std::move(s));
    }
    return all;
}

}  // namespace

int main() {
    // ---- Test 1: length-1 covers full 62-char charset in lex order. ----
    {
        bf::BruteforceGenerator g({1});
        auto all = drain(g, 7);
        expect_eq_u64(all.size(), 62, "len1: count");
        expect_eq_str(all.front(), "0", "len1: first");
        expect_eq_str(all.back(),  "z", "len1: last");
        // Spot-check ASCII collation boundaries.
        expect_eq_str(all[9],  "9", "len1: digit boundary");
        expect_eq_str(all[10], "A", "len1: digit->upper");
        expect_eq_str(all[35], "Z", "len1: upper boundary");
        expect_eq_str(all[36], "a", "len1: upper->lower");
        expect_true(g.done(), "len1: done after drain");
        expect_eq_u64(g.total_emitted(), 62, "len1: total_emitted");
    }

    // ---- Test 2: length-2 produces 62*62 unique strings. ----
    {
        bf::BruteforceGenerator g({2});
        auto all = drain(g, 113);  // odd batch size; forces mid-batch boundaries
        expect_eq_u64(all.size(), 62ULL * 62ULL, "len2: count");
        expect_eq_str(all.front(), "00", "len2: first");
        expect_eq_str(all.back(),  "zz", "len2: last");
        // Carry boundary: after "0z" should come "10" (alphabet[1] + alphabet[0]).
        expect_eq_str(all[61], "0z", "len2: end of '0_' row");
        expect_eq_str(all[62], "10", "len2: carry from '0z' to '10'");
        // No duplicates.
        std::set<std::string> uniq(all.begin(), all.end());
        expect_eq_u64(uniq.size(), all.size(), "len2: all unique");
    }

    // ---- Test 3: multi-length [1,2,3] iterates shortest-first. ----
    {
        bf::BruteforceGenerator g({1, 2, 3});
        auto all = drain(g, 256);
        const uint64_t want = 62 + 62*62 + 62ULL*62*62;
        expect_eq_u64(all.size(), want, "multi: count");
        // First 62 are length 1, next 3844 are length 2, last 238328 are length 3.
        expect_true(all[0].size()   == 1, "multi: first is len1");
        expect_true(all[61].size()  == 1, "multi: last len1 at 61");
        expect_true(all[62].size()  == 2, "multi: first len2 at 62");
        expect_eq_str(all[62],   "00",  "multi: first len2 string");
        expect_true(all[62 + 62*62 - 1].size() == 2, "multi: last len2");
        expect_eq_str(all[62 + 62*62 - 1], "zz", "multi: last len2 string");
        expect_true(all[62 + 62*62].size() == 3, "multi: first len3");
        expect_eq_str(all[62 + 62*62], "000", "multi: first len3 string");
        expect_eq_str(all.back(),    "zzz", "multi: last len3 string");
    }

    // ---- Test 4: resume from snapshot. ----
    {
        bf::BruteforceGenerator full({2});
        auto baseline = drain(full, 64);
        expect_eq_u64(baseline.size(), 62ULL * 62ULL, "resume: baseline count");

        // Emit half from one instance, snapshot, then continue from a second.
        bf::BruteforceGenerator a({2});
        auto first_half = a.next_batch(1500);
        expect_eq_u64(first_half.size(), 1500, "resume: first batch size");
        auto snap = a.snapshot();

        bf::BruteforceGenerator b({2});
        b.restore(snap);
        auto second_half = drain(b, 117);

        // first_half + second_half must equal the baseline drain.
        std::vector<std::string> rejoined = first_half;
        rejoined.insert(rejoined.end(), second_half.begin(), second_half.end());
        expect_eq_u64(rejoined.size(), baseline.size(), "resume: rejoined size");
        bool match = true;
        for (size_t i = 0; i < rejoined.size(); ++i) {
            if (rejoined[i] != baseline[i]) {
                match = false;
                std::fprintf(stderr, "[FAIL] resume: mismatch at %zu: got '%s' want '%s'\n",
                             i, rejoined[i].c_str(), baseline[i].c_str());
                ++g_fails;
                break;
            }
        }
        expect_true(match, "resume: byte-equal to one-shot drain");
    }

    // ---- Test 5: batch sizes never drop or duplicate. ----
    {
        // Various odd batch sizes traversing the same keyspace must produce
        // identical sequences.
        const std::vector<size_t> batch_sizes = {1, 7, 62, 63, 100, 4096};
        std::vector<std::string> reference;
        {
            bf::BruteforceGenerator g({2});
            reference = drain(g, 4096);
        }
        for (size_t bs : batch_sizes) {
            bf::BruteforceGenerator g({2});
            auto got = drain(g, bs);
            char label[64];
            std::snprintf(label, sizeof(label), "batch=%zu: equal-to-reference", bs);
            expect_eq_u64(got.size(), reference.size(), label);
            bool ok = true;
            for (size_t i = 0; i < got.size() && i < reference.size(); ++i) {
                if (got[i] != reference[i]) { ok = false; break; }
            }
            expect_true(ok, label);
        }
    }

    // ---- Test 6: done() flips only after final emission. ----
    {
        bf::BruteforceGenerator g({1});
        expect_true(!g.done(), "done: false at start");
        auto first = g.next_batch(61);
        expect_eq_u64(first.size(), 61, "done: first batch");
        expect_true(!g.done(), "done: still false with 1 left");
        auto last = g.next_batch(10);
        expect_eq_u64(last.size(), 1, "done: final batch returns 1");
        expect_eq_str(last[0], "z", "done: final string");
        // Next call should yield empty and flip done.
        auto empty = g.next_batch(10);
        expect_eq_u64(empty.size(), 0, "done: empty after exhaust");
        expect_true(g.done(), "done: true after final next_batch");
    }

    // ---- Test 7: keyspace + progress sanity. ----
    {
        bf::BruteforceGenerator g({3});
        expect_eq_u64(g.total_keyspace(), 62ULL*62*62, "keyspace: len3");
        expect_eq_u64(g.current_length_keyspace(), 62ULL*62*62, "keyspace: current");
        // Halfway-ish progress check.
        g.next_batch(62*62*62 / 2);
        double p = g.current_progress();
        expect_true(p > 0.49 && p < 0.51, "progress: ~0.5 at halfway");
    }

    if (g_fails == 0) {
        std::printf("[OK] test_bruteforce_generator: all assertions passed\n");
        return 0;
    } else {
        std::fprintf(stderr, "[FAIL] test_bruteforce_generator: %d assertion(s) failed\n", g_fails);
        return 1;
    }
}
