/**
 * Passphrase Generator Tests (GEN-1)
 *
 * Host-only (no CUDA) coverage for src/generators/passphrase_generator.hpp.
 * Targets the three boundary bugs repaired in that header plus a baseline of
 * functional coverage that previously did not exist:
 *
 *   1. FrequencyWordlistSource must not throw when the frequency field is
 *      malformed or out of uint64 range (std::stoull guard).
 *   2. WordlistSource::normalize must not invoke std::isspace with a raw
 *      (possibly negative) char on non-ASCII bytes such as 0x80 (UB guard).
 *   3. CombinatorSource::estimated_size must saturate to SIZE_MAX instead of
 *      wrapping when list_size^n overflows size_t.
 *
 * Plus: WordlistSource basic enumeration and a minimal threaded
 * PassphraseGenerator run_generation() smoke test.
 *
 * Pure CPU/std test: no GPU, no network. Links collider_core + Threads.
 */

#include "../src/generators/passphrase_generator.hpp"
#include "../src/generators/priority_queue.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

using namespace collider;

namespace {

// Per-process entropy for temp filenames. getpid is spelled differently on
// Windows vs POSIX; this test does not need a real pid, only some per-process
// uniqueness, so the (header-only, portable) thread id hash is used instead.
uint64_t process_salt() {
    return static_cast<uint64_t>(
        std::hash<std::thread::id>{}(std::this_thread::get_id()));
}

// Write `contents` to a uniquely named temp file and return its path. The
// caller is responsible for removal (done via TempFile RAII below).
std::filesystem::path write_temp_file(const std::string& stem,
                                      const std::string& contents) {
    static std::atomic<uint64_t> counter{0};
    auto dir = std::filesystem::temp_directory_path();
    auto path = dir / (stem + "_" + std::to_string(counter.fetch_add(1)) +
                       "_" + std::to_string(process_salt()) + ".txt");
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    out.close();
    return path;
}

// RAII wrapper that removes a temp file on scope exit.
struct TempFile {
    std::filesystem::path path;
    explicit TempFile(std::filesystem::path p) : path(std::move(p)) {}
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    std::string str() const { return path.string(); }
};

// Drain every candidate a source emits into a vector (preserves order).
std::vector<Candidate> collect(PassphraseSource& source) {
    std::vector<Candidate> out;
    source.generate([&out](Candidate&& c) { out.push_back(std::move(c)); });
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Test 1: WordlistSource basic enumeration
// ---------------------------------------------------------------------------
void test_wordlist_basic_enumeration() {
    // Includes an empty line and a comment that must be skipped, plus a line
    // with surrounding whitespace that normalize() should trim.
    TempFile tf(write_temp_file(
        "wordlist_basic",
        "correct horse battery staple\n"
        "\n"
        "# a comment line\n"
        "  spaced  out  \n"
        "satoshi\n"));

    WordlistSource source(tf.str(), CandidateSource::USER_WORDLIST);
    auto candidates = collect(source);

    std::set<std::string> phrases;
    for (const auto& c : candidates) {
        phrases.insert(c.phrase);
        assert(c.source == CandidateSource::USER_WORDLIST);
    }

    // Empty + comment line dropped; whitespace collapsed/trimmed.
    assert(candidates.size() == 3);
    assert(phrases.count("correct horse battery staple") == 1);
    assert(phrases.count("spaced out") == 1);  // collapsed + trimmed
    assert(phrases.count("satoshi") == 1);
    assert(phrases.count("") == 0);

    std::cout << "[PASS] WordlistSource basic enumeration\n";
}

// ---------------------------------------------------------------------------
// Test 2: FrequencyWordlistSource with a malformed count line must not throw
// ---------------------------------------------------------------------------
void test_frequency_malformed_count_no_throw() {
    // Line A: valid count.
    // Line B: non-numeric count ("abc")  -> std::stoull throws invalid_argument
    // Line C: out-of-range count (> uint64 max) -> throws out_of_range
    // Line D: no tab at all (plain password).
    // The generator MUST NOT propagate any exception; every password payload
    // should still be emitted, with the malformed-count lines treated as
    // count=0 (lowest, non-zero-clamped priority).
    TempFile tf(write_temp_file(
        "freq_malformed",
        "100\tvalidpass\n"
        "abc\tmalformedpass\n"
        "99999999999999999999999999999\toverflowpass\n"
        "plainpassword\n"));

    FrequencyWordlistSource source(tf.str());

    std::vector<Candidate> candidates;
    bool threw = false;
    try {
        source.generate(
            [&candidates](Candidate&& c) { candidates.push_back(std::move(c)); });
    } catch (...) {
        threw = true;
    }

    assert(!threw && "FrequencyWordlistSource must swallow std::stoull errors");

    std::set<std::string> phrases;
    for (const auto& c : candidates) phrases.insert(c.phrase);

    // Every line's password survives; malformed/overflow counts are skipped
    // (treated as count=0) rather than throwing out of the callback.
    assert(candidates.size() == 4);
    assert(phrases.count("validpass") == 1);
    assert(phrases.count("malformedpass") == 1);
    assert(phrases.count("overflowpass") == 1);
    assert(phrases.count("plainpassword") == 1);

    std::cout << "[PASS] FrequencyWordlistSource malformed count (no throw)\n";
}

// ---------------------------------------------------------------------------
// Test 3: normalize() on a 0x80 byte must not be UB / must not crash
// ---------------------------------------------------------------------------
void test_normalize_high_byte_no_ub() {
    // 0x80 is a negative value when stored in a signed char. Passing it
    // straight to std::isspace(int) is undefined behavior; the fix casts
    // through unsigned char. We route a line containing 0x80 through the
    // public generate() path and assert the byte survives unchanged (0x80 is
    // not whitespace, so normalize keeps it).
    std::string contents;
    contents += "ab";
    contents += static_cast<char>(0x80);  // high, non-whitespace byte
    contents += "cd";
    contents += '\n';

    TempFile tf(write_temp_file("normalize_highbyte", contents));

    WordlistSource source(tf.str(), CandidateSource::USER_WORDLIST);
    auto candidates = collect(source);

    assert(candidates.size() == 1);
    const std::string& phrase = candidates[0].phrase;
    // 5 bytes: 'a','b',0x80,'c','d' (high byte preserved, not classified as space).
    assert(phrase.size() == 5);
    assert(static_cast<unsigned char>(phrase[2]) == 0x80);

    std::cout << "[PASS] normalize() high byte (0x80) no UB\n";
}

// ---------------------------------------------------------------------------
// Test 4: CombinatorSource::estimated_size overflow saturation
// ---------------------------------------------------------------------------
void test_combinator_estimated_size_saturates() {
    // Build a wordlist large enough that list_size^max_words overflows size_t.
    // 100k words ^ 4 ~= 1e20, far beyond 2^64 (~1.8e19). Without the saturation
    // guard this wrapped to a small value; with it, the estimate clamps to
    // SIZE_MAX.
    constexpr size_t kWords = 100'000;
    std::string contents;
    contents.reserve(kWords * 7);
    for (size_t i = 0; i < kWords; ++i) {
        contents += "w";
        contents += std::to_string(i);
        contents += '\n';
    }
    TempFile tf(write_temp_file("combinator_overflow", contents));

    // min_words..max_words = 2..4 over a 100k list overflows on the n=4 term.
    CombinatorSource source({tf.str()}, /*min_words=*/2, /*max_words=*/4);

    size_t est = source.estimated_size();
    assert(est == std::numeric_limits<size_t>::max() &&
           "estimated_size must saturate to SIZE_MAX on overflow, not wrap");

    // Sanity: a tiny list must NOT saturate (guard does not over-trigger).
    TempFile small(write_temp_file("combinator_small", "a\nb\nc\n"));
    CombinatorSource small_source({small.str()}, /*min_words=*/2, /*max_words=*/2);
    size_t small_est = small_source.estimated_size();
    // 3^2 = 9 combinations * 4 default separators = 36, well below SIZE_MAX.
    assert(small_est < std::numeric_limits<size_t>::max());
    assert(small_est == 9 * 4);

    std::cout << "[PASS] CombinatorSource::estimated_size overflow saturation\n";
}

// ---------------------------------------------------------------------------
// Test 5: threaded PassphraseGenerator run_generation() smoke
// ---------------------------------------------------------------------------
void test_generator_threaded_smoke() {
    // Phrases are chosen >= 8 chars so the WeightedSourceManager length penalty
    // does not zero them out, and all distinct so the queue's dedup does not
    // drop any. USER_WORDLIST has a non-trivial source weight.
    TempFile tf(write_temp_file(
        "generator_smoke",
        "alphapassphrase\n"
        "bravopassphrase\n"
        "charliepassphrase\n"
        "deltapassphrase\n"));

    auto queue = std::make_shared<CandidatePriorityQueue>(1000, 100000);

    PassphraseGenerator generator(queue);
    generator.add_wordlist(tf.str(), CandidateSource::USER_WORDLIST);

    generator.start();

    // Condition-based wait (no fixed sleep): poll is_complete() with a hard
    // timeout so a hang fails the test loudly instead of blocking ctest.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!generator.is_complete()) {
        if (std::chrono::steady_clock::now() > deadline) {
            assert(false && "generator did not complete within timeout");
        }
        std::this_thread::yield();
    }
    generator.stop();

    assert(generator.is_complete());
    assert(generator.candidates_generated() == 4);

    // Drain the queue and confirm the four distinct candidates landed.
    std::set<std::string> popped;
    while (auto c = queue->pop()) {
        popped.insert(c->phrase);
    }
    assert(popped.size() == 4);
    assert(popped.count("alphapassphrase") == 1);
    assert(popped.count("bravopassphrase") == 1);
    assert(popped.count("charliepassphrase") == 1);
    assert(popped.count("deltapassphrase") == 1);

    std::cout << "[PASS] PassphraseGenerator threaded run_generation smoke\n";
}

int main() {
    std::cout << "=== Passphrase Generator Tests (GEN-1) ===\n";

    test_wordlist_basic_enumeration();
    test_frequency_malformed_count_no_throw();
    test_normalize_high_byte_no_ub();
    test_combinator_estimated_size_saturates();
    test_generator_threaded_smoke();

    std::cout << "=== All passphrase generator tests passed ===\n";
    return 0;
}
