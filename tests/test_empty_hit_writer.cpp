/**
 * test_empty_hit_writer - Tier 1 perf F-log validation.
 *
 * Three scenarios:
 *   1. Steady state: enqueue 10000 records from one thread, stop(),
 *      assert every record was written. (Capacity-edge, no drop.)
 *   2. Overflow: enqueue 20000 records in a tight loop, assert
 *      written_count + dropped_count == 20000. (drop-oldest policy.)
 *   3. Concurrent enqueue + stop race: spawn a producer that keeps
 *      enqueuing while the main thread calls stop(). The writer
 *      must drain whatever it had observed before the stop signal
 *      and join cleanly. Test passes if no deadlock + no crash;
 *      we do NOT assert which records survived the race because
 *      that's inherently nondeterministic.
 */

#include "runtime/empty_hit_writer.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using collider::runtime::EmptyHitRecord;
using collider::runtime::EmptyHitWriter;

// Build a deterministic record from a sequence number so the line
// content is predictable; the test asserts every expected line landed.
EmptyHitRecord make_record(uint32_t seq) {
    EmptyHitRecord rec;
    std::snprintf(rec.ts_iso.data(), rec.ts_iso.size(),
                  "2026-05-15T12:00:%02u.000Z", seq % 60);
    // Privkey and h160 deterministically seeded from seq.
    for (int i = 0; i < 32; i++) rec.privkey[i] = static_cast<uint8_t>(seq + i);
    for (int i = 0; i < 20; i++) rec.h160[i]    = static_cast<uint8_t>(seq ^ i);
    rec.passphrase = "test#" + std::to_string(seq);
    return rec;
}

std::string make_tmp_path(const char* suffix) {
    // Use the temp directory + a deterministic-ish name; we delete on
    // entry to start from a clean file every run.
    auto tmp = std::filesystem::temp_directory_path()
             / (std::string("empty_hit_writer_") + suffix + ".txt");
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    return tmp.string();
}

size_t count_lines(const std::string& path) {
    std::ifstream f(path);
    if (!f) return 0;
    size_t n = 0;
    std::string line;
    while (std::getline(f, line)) ++n;
    return n;
}

int test_steady_state() {
    std::printf("[1/3] steady state, 10000 records, no drop...\n");
    auto path = make_tmp_path("steady");
    constexpr uint32_t N = 10000;
    {
        EmptyHitWriter writer(path);
        for (uint32_t i = 0; i < N; i++) {
            writer.enqueue(make_record(i));
        }
        // stop() in destructor; explicit call here is harmless.
        writer.stop();
        if (writer.dropped_count() != 0) {
            std::fprintf(stderr,
                "FAIL: dropped %llu records under capacity\n",
                static_cast<unsigned long long>(writer.dropped_count()));
            return 1;
        }
        if (writer.written_count() != N) {
            std::fprintf(stderr,
                "FAIL: wrote %llu of %u records\n",
                static_cast<unsigned long long>(writer.written_count()), N);
            return 1;
        }
    }
    size_t lines = count_lines(path);
    if (lines != N) {
        std::fprintf(stderr, "FAIL: file has %zu lines, expected %u\n", lines, N);
        return 1;
    }
    std::printf("       PASS: %u records written, file has %zu lines.\n", N, lines);
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return 0;
}

int test_overflow() {
    // Push 20000 records as fast as possible from the main thread. The
    // writer thread may keep up or lag depending on disk speed; the
    // invariant is that every record is accounted for (written or
    // dropped) by the time stop() returns.
    std::printf("[2/3] overflow, 20000 records, drop-oldest...\n");
    auto path = make_tmp_path("overflow");
    constexpr uint32_t N = 20000;
    EmptyHitWriter writer(path);
    for (uint32_t i = 0; i < N; i++) {
        writer.enqueue(make_record(i));
    }
    writer.stop();
    uint64_t wrote   = writer.written_count();
    uint64_t dropped = writer.dropped_count();
    std::printf("       written=%llu dropped=%llu total=%llu\n",
                static_cast<unsigned long long>(wrote),
                static_cast<unsigned long long>(dropped),
                static_cast<unsigned long long>(wrote + dropped));
    if (wrote + dropped != N) {
        std::fprintf(stderr, "FAIL: accounting mismatch %llu != %u\n",
                     static_cast<unsigned long long>(wrote + dropped), N);
        return 1;
    }
    // The bounded queue is 10000; if the producer outran the writer,
    // dropped > 0 is expected. If the disk is fast enough that the
    // writer kept up, dropped may be 0 and that's still correct.
    // Either way the file MUST contain `wrote` lines exactly.
    size_t lines = count_lines(path);
    if (lines != wrote) {
        std::fprintf(stderr, "FAIL: file has %zu lines, expected %llu\n",
                     lines, static_cast<unsigned long long>(wrote));
        return 1;
    }
    std::printf("       PASS: accounting is self-consistent.\n");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return 0;
}

int test_concurrent_stop() {
    std::printf("[3/3] concurrent stop + enqueue race...\n");
    auto path = make_tmp_path("race");
    EmptyHitWriter writer(path);

    std::atomic<bool> stop_producer{false};
    std::atomic<uint64_t> produced{0};

    std::thread producer([&]() {
        uint32_t seq = 0;
        while (!stop_producer.load(std::memory_order_acquire)) {
            writer.enqueue(make_record(seq++));
            produced.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // Let producer run a brief moment so the queue is non-empty when
    // stop is signalled, exercising the "drain whatever was swapped
    // out" branch of writer_loop.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    // stop() must return promptly even though the producer is still
    // enqueueing. Producer may continue past stop(); those enqueues
    // either land before writer observes stop_requested_ (and are
    // flushed) or after (and are lost). Either way no crash.
    writer.stop();
    stop_producer.store(true, std::memory_order_release);
    producer.join();

    uint64_t wrote   = writer.written_count();
    uint64_t dropped = writer.dropped_count();
    uint64_t total   = produced.load(std::memory_order_relaxed);
    std::printf("       produced=%llu written=%llu dropped=%llu\n",
                static_cast<unsigned long long>(total),
                static_cast<unsigned long long>(wrote),
                static_cast<unsigned long long>(dropped));
    // Invariants (the real test is "does stop() return without
    // deadlock under concurrent enqueue"):
    //
    //   1. wrote <= produced. The writer cannot have flushed records
    //      that were never enqueued.
    //   2. wrote + dropped <= produced. Records that landed in the
    //      queue after the writer observed stop_requested are silently
    //      lost (documented contract in empty_hit_writer.hpp); they are
    //      not counted in either tally. So the writer's bookkeeping is
    //      a lower bound on the producer's total.
    if (wrote > total) {
        std::fprintf(stderr, "FAIL: wrote (%llu) > produced (%llu)\n",
                     static_cast<unsigned long long>(wrote),
                     static_cast<unsigned long long>(total));
        return 1;
    }
    if (wrote + dropped > total) {
        std::fprintf(stderr,
            "FAIL: wrote + dropped (%llu) > produced (%llu)\n",
            static_cast<unsigned long long>(wrote + dropped),
            static_cast<unsigned long long>(total));
        return 1;
    }
    std::printf("       PASS: stop() joined cleanly with concurrent producer.\n");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return 0;
}

}  // namespace

// R-B4: writer must not silently drop records after the underlying stream
// goes into fail state. We construct a writer pointing at a path that
// CANNOT be opened. The constructor's open will fail, which surfaces a
// one-time warning to stderr and every subsequent enqueue is silently
// accounted as dropped via the existing not-open path. The invariant we
// check is: accounting still totals N (every enqueued record landed in
// `dropped_count`), stop() returns cleanly without deadlock, and no
// records were silently lost from accounting.
//
// Generating a deterministically-unopenable path across Windows + POSIX is
// the tricky part. Earlier revisions of this test tried two approaches that
// both proved fragile:
//
//   1. "Missing parent directory." EmptyHitWriter now opens its output via
//      collider::secure_open_ofstream which auto-creates the parent
//      directory under FailHard, so a missing parent is no longer a
//      reliable open-failure trigger.
//   2. "Pass the temp directory itself as the target path." On POSIX
//      open(O_CREAT | O_WRONLY) on a directory always fails with EISDIR.
//      On Windows however CreateFileA(GENERIC_WRITE, OPEN_ALWAYS) against
//      a directory path is rejected with ERROR_ACCESS_DENIED on most
//      Win11 builds but observably SUCCEEDS on others (machine-specific
//      driver / antivirus / filesystem-redirector behaviour). When it
//      succeeds the subsequent std::ofstream open against the same path
//      also succeeded, every record was actually written, and the test
//      asserted FAIL on wrote != 0.
//
// The structural fix used below is portable across Windows, Linux, and
// macOS: create a regular file in the temp directory, then point the
// writer at a path that uses that regular file as its parent. Both
// std::filesystem::create_directories (called inside secure_open_ofstream)
// and the subsequent CreateFileA / ::open refuse to descend into a path
// component that exists as a non-directory. POSIX returns ENOTDIR;
// Windows returns ERROR_DIRECTORY (267). The writer's out_ stays
// unopened on every supported platform, the writer-loop's
// !out_.is_open() branch fires for every enqueued record, and the
// dropped-count accounting must total N.
//
// We cannot reliably make an ALREADY-OPEN std::ofstream go bad
// cross-platform (Windows holds an exclusive handle by default and most
// fs-level failbit triggers are platform-specific). The accounting
// invariant is what the R-B4 fix really protects: the prior code's bug
// was that records after the FIRST failed write were accounted as
// dropped but the writer ALSO stopped trying, with no warning. The new
// code retries + warns once, but the accounting is the externally
// visible invariant the test asserts.
int test_unopenable_path() {
    std::printf("[4/4] unopenable path, accounting consistent...\n");
    // Stage a regular file inside the temp directory; that file becomes
    // the "parent" of our target path so any open beneath it fails with
    // ENOTDIR / ERROR_DIRECTORY on every supported platform.
    auto parent_file = std::filesystem::temp_directory_path()
                     / "empty_hit_writer_blocker.tmp";
    {
        std::error_code ec;
        std::filesystem::remove(parent_file, ec);
        std::ofstream blocker(parent_file);
        if (!blocker) {
            std::fprintf(stderr,
                "FAIL: could not stage blocker file %s for test setup\n",
                parent_file.string().c_str());
            return 1;
        }
        blocker << "blocker";
    }
    auto bad_path = parent_file / "child.txt";

    constexpr uint32_t N = 500;
    {
        EmptyHitWriter writer(bad_path.string());
        for (uint32_t i = 0; i < N; i++) {
            writer.enqueue(make_record(i));
        }
        writer.stop();
        uint64_t wrote   = writer.written_count();
        uint64_t dropped = writer.dropped_count();
        std::printf("       written=%llu dropped=%llu\n",
                    static_cast<unsigned long long>(wrote),
                    static_cast<unsigned long long>(dropped));
        if (wrote != 0) {
            std::fprintf(stderr,
                "FAIL: wrote %llu records to unopenable path\n",
                static_cast<unsigned long long>(wrote));
            std::error_code ec;
            std::filesystem::remove(parent_file, ec);
            return 1;
        }
        if (dropped != N) {
            std::fprintf(stderr,
                "FAIL: dropped %llu of %u records under unopenable path\n",
                static_cast<unsigned long long>(dropped), N);
            std::error_code ec;
            std::filesystem::remove(parent_file, ec);
            return 1;
        }
    }
    // Clean up the blocker file so repeated runs do not leave litter
    // behind in the temp directory.
    std::error_code ec;
    std::filesystem::remove(parent_file, ec);
    std::printf("       PASS: every record accounted as dropped, no silent loss.\n");
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_steady_state();
    rc |= test_overflow();
    rc |= test_concurrent_stop();
    rc |= test_unopenable_path();
    if (rc == 0) {
        std::printf("=== empty_hit_writer: all scenarios pass ===\n");
    } else {
        std::printf("=== empty_hit_writer: FAILURES ===\n");
    }
    return rc;
}
