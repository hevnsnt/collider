/**
 * Priority Queue Tests
 *
 * Tests for the probability-ordered candidate queue with bloom filter deduplication.
 */

#include "../src/generators/priority_queue.hpp"
#include <iostream>
#include <cassert>
#include <chrono>
#include <random>
#include <set>
#include <string>

using namespace collider;

void test_basic_push_pop() {
    CandidatePriorityQueue queue(1000, 10000);

    // Push some candidates
    queue.push(Candidate{
        .phrase = "password123",
        .priority = 0.5f,
        .source = CandidateSource::PASSWORD_COMMON,
        .rule_applied = ":"
    });

    queue.push(Candidate{
        .phrase = "bitcoin2009",
        .priority = 0.8f,  // Higher priority
        .source = CandidateSource::KNOWN_BRAIN_WALLET,
        .rule_applied = ":"
    });

    queue.push(Candidate{
        .phrase = "hello",
        .priority = 0.3f,  // Lower priority
        .source = CandidateSource::LYRICS,
        .rule_applied = ":"
    });

    assert(queue.size() == 3);

    // Pop should return highest priority first
    auto c1 = queue.pop();
    assert(c1.has_value());
    assert(c1->phrase == "bitcoin2009");
    assert(c1->priority == 0.8f);

    auto c2 = queue.pop();
    assert(c2.has_value());
    assert(c2->phrase == "password123");

    auto c3 = queue.pop();
    assert(c3.has_value());
    assert(c3->phrase == "hello");

    assert(queue.empty());

    std::cout << "[PASS] Basic push/pop\n";
}

void test_deduplication() {
    CandidatePriorityQueue queue(1000, 10000);

    // Push same phrase multiple times
    for (int i = 0; i < 10; i++) {
        queue.push(Candidate{
            .phrase = "duplicate",
            .priority = 0.5f + i * 0.01f,
            .source = CandidateSource::PASSWORD_COMMON,
            .rule_applied = ":"
        });
    }

    // Should only have 1 (first one)
    assert(queue.size() == 1);
    assert(queue.stats().duplicates_filtered == 9);

    std::cout << "[PASS] Deduplication\n";
}

void test_max_size() {
    CandidatePriorityQueue queue(5, 1000);  // Max 5 items

    // Push 10 items
    for (int i = 0; i < 10; i++) {
        queue.push(Candidate{
            .phrase = "item" + std::to_string(i),
            .priority = 0.1f * i,
            .source = CandidateSource::PASSWORD_COMMON,
            .rule_applied = ":"
        });
    }

    // Should keep only top 5
    assert(queue.size() == 5);

    // Pop should return in priority order (highest first)
    auto c = queue.pop();
    assert(c.has_value());
    assert(c->phrase == "item9");  // Highest priority

    std::cout << "[PASS] Max size limit\n";
}

void test_batch_pop() {
    CandidatePriorityQueue queue(1000, 10000);

    // Push 100 items
    for (int i = 0; i < 100; i++) {
        queue.push(Candidate{
            .phrase = "batch" + std::to_string(i),
            .priority = 0.01f * i,
            .source = CandidateSource::PASSWORD_COMMON,
            .rule_applied = ":"
        });
    }

    // Pop batch of 30
    auto batch = queue.pop_batch(30);

    assert(batch.size() == 30);
    assert(queue.size() == 70);

    // Should be in priority order (highest first)
    assert(batch.phrases[0] == "batch99");
    assert(batch.phrases[1] == "batch98");

    std::cout << "[PASS] Batch pop\n";
}

void test_source_manager() {
    WeightedSourceManager manager;

    // Known brain wallets should have highest weight
    float bw_priority = manager.calculate_priority(
        "test", CandidateSource::KNOWN_BRAIN_WALLET
    );

    float common_priority = manager.calculate_priority(
        "test", CandidateSource::PASSWORD_COMMON
    );

    float pcfg_priority = manager.calculate_priority(
        "test", CandidateSource::PCFG_GENERATED
    );

    assert(bw_priority > common_priority);
    assert(common_priority > pcfg_priority);

    std::cout << "[PASS] Source manager weights\n";
}

void test_source_learning() {
    WeightedSourceManager manager;

    // Record some cracks
    for (int i = 0; i < 10; i++) {
        manager.record_crack(CandidateSource::LYRICS);
    }

    for (int i = 0; i < 1000; i++) {
        manager.record_attempt(CandidateSource::LYRICS);
    }

    // Check stats
    auto stats = manager.get_stats();
    bool found = false;
    for (const auto& s : stats) {
        if (s.source == CandidateSource::LYRICS) {
            assert(s.candidates_tested == 1000);
            assert(s.cracks_found == 10);
            assert(s.crack_rate > 0.009 && s.crack_rate < 0.011);  // ~1%
            found = true;
        }
    }
    assert(found);

    std::cout << "[PASS] Source learning\n";
}

void test_bloom_filter() {
    BloomFilter filter(10000, 0.001);  // 10K capacity, 0.1% FP

    // Add some items
    for (int i = 0; i < 1000; i++) {
        filter.add("item" + std::to_string(i));
    }

    // Check membership
    int false_positives = 0;
    for (int i = 0; i < 1000; i++) {
        // Should definitely contain these
        assert(filter.probably_contains("item" + std::to_string(i)));
    }

    // Check non-members
    for (int i = 1000; i < 2000; i++) {
        if (filter.probably_contains("item" + std::to_string(i))) {
            false_positives++;
        }
    }

    // FP rate should be around 0.1%
    double fp_rate = static_cast<double>(false_positives) / 1000.0;
    assert(fp_rate < 0.02);  // Allow some margin

    std::cout << "[PASS] Bloom filter (FP rate: " << fp_rate * 100 << "%)\n";
}

void test_performance() {
    CandidatePriorityQueue queue(1'000'000, 10'000'000);

    auto start = std::chrono::high_resolution_clock::now();

    // Insert 100K items
    for (int i = 0; i < 100'000; i++) {
        queue.push(Candidate{
            .phrase = "perf" + std::to_string(i),
            .priority = static_cast<float>(rand()) / RAND_MAX,
            .source = CandidateSource::PASSWORD_COMMON,
            .rule_applied = ":"
        });
    }

    auto insert_end = std::chrono::high_resolution_clock::now();

    // Pop all as batches
    while (!queue.empty()) {
        queue.pop_batch(10000);
    }

    auto pop_end = std::chrono::high_resolution_clock::now();

    auto insert_ms = std::chrono::duration_cast<std::chrono::milliseconds>(insert_end - start).count();
    auto pop_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pop_end - insert_end).count();

    std::cout << "[PASS] Performance: "
              << "insert 100K in " << insert_ms << "ms, "
              << "pop all in " << pop_ms << "ms\n";
}

// v1.4.1 BW-B4: generate_variations now routes through the hashcat-compatible
// RuleEngine instead of baked-in case/digit/swap loops. These tests pin the
// behavior the brain wallet feedback loop depends on:
//   1. enough distinct outputs for the feedback queue to be worth populating
//   2. every output is a non-empty ASCII-printable string (no junk)
//   3. the original passphrase is included (the ":" identity rule)
//   4. leetspeak coverage exists (the missing pre-1.4.1 case)
void test_feedback_variations_size_and_quality() {
    auto variations = CandidatePriorityQueue::generate_variations("password");

    // Quality: every output is non-empty and ASCII-printable.
    for (const auto& v : variations) {
        assert(!v.empty());
        for (unsigned char c : v) {
            // 0x20..0x7E covers space + visible ASCII.
            assert(c >= 0x20 && c <= 0x7E);
        }
    }

    // Distinctness: at least 5 distinct outputs (we emit ~22 rules, but
    // dedup may collapse some; 5 is a generous floor that still catches
    // a regression where the rule list is empty).
    std::set<std::string> distinct(variations.begin(), variations.end());
    assert(distinct.size() >= 5);

    // Identity: the cracked passphrase itself MUST be in the output.
    // The feedback queue depends on the original being re-tried after a
    // hint surfaces (e.g. workers who started after the crack).
    assert(distinct.count("password") == 1);

    std::cout << "[PASS] Feedback variations size+quality (" << distinct.size()
              << " distinct from 'password')\n";
}

void test_feedback_variations_leetspeak_and_digits() {
    // Verify the rule engine actually wired through: leetspeak swaps and
    // digit-append behavior the pre-1.4.1 inline loop did not produce.
    auto variations = CandidatePriorityQueue::generate_variations("password");
    std::set<std::string> distinct(variations.begin(), variations.end());

    // sa@ should swap a -> @
    assert(distinct.count("p@ssword") == 1);
    // so0 should swap o -> 0
    assert(distinct.count("passw0rd") == 1);
    // ss5 should swap s -> 5
    assert(distinct.count("pa55word") == 1);
    // $1 should append "1"
    assert(distinct.count("password1") == 1);

    std::cout << "[PASS] Feedback variations leetspeak + digit-append\n";
}

int main() {
    std::cout << "=== Priority Queue Tests ===\n\n";

    test_basic_push_pop();
    test_deduplication();
    test_max_size();
    test_batch_pop();
    test_source_manager();
    test_source_learning();
    test_bloom_filter();
    test_performance();
    test_feedback_variations_size_and_quality();
    test_feedback_variations_leetspeak_and_digits();

    std::cout << "\n=== All Tests Passed ===\n";
    return 0;
}
