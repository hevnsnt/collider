// phase 1 + 3 (builder-threading: scan-state-atomics + perf-accumulator).
//
// Concurrency stress test for collider::runtime::ScanState.
//
// One writer thread increments all global counters AND per-phase counters in
// a tight loop, calling commit_batch() every kBatchSize iterations. Four
// reader threads each take 10000 snapshots in a tight loop and validate every
// cross-counter invariant that the writer establishes at batch boundaries.
//
// Invariants asserted on every snapshot:
//   1. bloom_collisions_filtered >= tight_bloom_filtered (real_empty is non-negative).
//   2. bloom_hits <= total_checked (a bloom hit is by definition something we already counted).
//   3. seq is monotonic non-decreasing across snapshots from the same reader.
//   4. sum(phase_keys_processed) <= total_checked
//      (every per-phase key was also counted globally; the writer enforces
//       parity by incrementing both inside the same batch).
//   5. sum(empty_hits_by_phase) <= bloom_collisions_filtered - tight_bloom_filtered
//      i.e. sum_per_phase_empty_hits <= real_empty(). The writer only charges
//      a phase empty hit AFTER it has committed a (collision, NOT-tight) pair
//      in the same batch, so the sum across phases never exceeds real_empty().
//   6. current_phase is always in 0..kNumPhases-1.
//
// Wall-clock budget ~1 second; readers stop after they've done their quota,
// writer stops via an atomic shutdown flag.
//
// The test also verifies out-of-range writes are silently dropped: after the
// writer/reader threads join, the main thread calls inc_phase_keys(99),
// inc_empty_hits_by_phase(-1), and set_current_phase(7) and asserts none of
// them crash AND the snapshot's published phase index + counters are
// unchanged by those writes.

#include "runtime/scan_state.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <thread>
#include <vector>

namespace {

constexpr int kReaderCount = 4;
constexpr int kSnapshotsPerReader = 10000;
constexpr int kBatchSize = 100;
constexpr auto kStressDuration = std::chrono::seconds(1);

// Phase cycling cadence: the writer advances current_phase every
// kIterationsPerPhase commit_batch() calls, walking 0 -> 1 -> 2 -> 3 -> 4 -> 0.
// 100 commits per phase gives readers plenty of opportunity to see snapshots
// from every phase during the 1 s budget.
constexpr uint64_t kIterationsPerPhase = 100;

// Per-batch empty-hits charged to the current phase. Each batch the writer
// always commits 2 bloom_collisions + 1 tight_bloom (so real_empty grows by
// 1 per batch) and charges 1 empty-hit to the current phase. The invariant
// sum(empty_hits_by_phase) == real_empty() then holds exactly at every
// commit boundary.
constexpr uint64_t kEmptyHitPerBatch = 1;

struct ReaderResult {
    bool ok = true;
    uint64_t snapshots_taken = 0;
    uint64_t snapshots_consistent = 0;
    uint64_t snapshots_degraded = 0;
    uint64_t worst_seq_seen = 0;
    const char* failure_reason = nullptr;
    uint64_t failure_a = 0;
    uint64_t failure_b = 0;
};

void run_writer(collider::runtime::ScanState& state,
                std::atomic<bool>& shutdown,
                std::atomic<bool>& warmup_done) {
    uint64_t local_total = 0;
    uint64_t local_bloom_hits = 0;
    uint64_t local_collisions = 0;
    uint64_t local_tight = 0;
    uint64_t local_verified = 0;
    uint64_t dispatch_words = 1024;
    uint64_t batches_committed = 0;
    int current_phase = 0;
    uint64_t phase_iteration = 0;

    // Seed the published phase index before the first batch so readers that
    // race in immediately after warmup see a coherent phase value (rather
    // than the default 0 they would also see, but explicit is clearer).
    state.set_current_phase(current_phase, phase_iteration);

    // The writer establishes the invariant set at every commit_batch():
    //   total_checked grows by kBatchSize each batch,
    //   bloom_hits grows by 1 (always less than total),
    //   bloom_collisions_filtered grows by 2,
    //   tight_bloom_filtered grows by 1 (always <= collisions),
    //   verified_hits grows by 1,
    //   phase_keys_processed[current_phase] grows by kBatchSize (sum across phases stays == total_checked),
    //   empty_hits_by_phase[current_phase] grows by kEmptyHitPerBatch == real_empty grows.
    // Between commits, the relaxed increments may be visible to readers in
    // any order; the seq-lock guarantees readers only see a snapshot from a
    // committed batch boundary, where all invariants hold simultaneously.

    while (!shutdown.load(std::memory_order_relaxed)) {
        // True seq-lock: begin_batch() bumps seq to odd (writer-in-progress);
        // commit_batch() bumps seq to the next even value. Readers seeing an
        // odd seq retry, so cross-counter invariants below hold for any
        // snapshot that succeeds (the writer's relaxed stores are bracketed
        // by the seq publishes).
        state.begin_batch();
        // Mirror the production-code pattern: one bulk inc_total_checked per
        // batch (the GPU-rules path uses inc_total_checked(batch_total) and
        // the CPU-rules path uses inc_total_checked(passphrases.size())).
        state.inc_total_checked(kBatchSize);
        local_total += kBatchSize;
        state.inc_bloom_collisions_filtered(2);
        local_collisions += 2;
        state.inc_tight_bloom_filtered(1);
        local_tight += 1;
        state.inc_bloom_hits(1);
        local_bloom_hits += 1;
        state.inc_verified_hits(1);
        local_verified += 1;
        state.set_dispatch_words_per_gpu(dispatch_words);
        dispatch_words = (dispatch_words == 1024) ? 2048 : 1024;

        // Per-phase increments: charge the whole batch to current_phase.
        // The order matters: phase_keys is incremented AFTER total_checked,
        // so any reader seeing the phase_keys bump must also see (at least)
        // the same-batch total_checked bump on x86 TSO. The seq-lock contract
        // does not guarantee this for relaxed reads on weakly-ordered ISAs;
        // production runs on x86 / Apple Silicon (ARM with acquire-release on
        // atomic loads) where program order is observable, and the current
        // batch's writes are anyway bracketed by the commit-time seq publish.
        state.inc_phase_keys(current_phase, kBatchSize);
        state.inc_empty_hits_by_phase(current_phase, kEmptyHitPerBatch);

        // Publication point: readers from this commit onwards see a
        // consistent multi-counter snapshot.
        state.commit_batch();
        ++batches_committed;

        // Production-equivalent throttle. Real brain-wallet batches run at
        // ~10-100 Hz (60M keys/s / 1M batch size). A tight loop here would
        // bump seq at tens of MHz and starve reader threads of consistent
        // snapshots even on a quiet system. std::this_thread::yield() drops
        // the writer to the back of the scheduler queue between batches so
        // readers always get a quiescent window to land a consistent
        // snapshot, while still cycling thousands of batches per second.
        std::this_thread::yield();

        // Advance phase every kIterationsPerPhase batches; cycle 0..4.
        if (batches_committed % kIterationsPerPhase == 0) {
            current_phase = (current_phase + 1) % collider::runtime::kNumPhases;
            if (current_phase == 0) {
                ++phase_iteration;
            }
            state.set_current_phase(current_phase, phase_iteration);
        }

        if (local_total >= kBatchSize * 16) {
            // After a small warmup, let readers begin so the first snapshot
            // they take sees a non-trivial mix of all five counters in motion.
            warmup_done.store(true, std::memory_order_release);
        }
    }

    // Silence unused-local warnings (we only use these to drive batch counts).
    (void)local_bloom_hits;
    (void)local_verified;
}

void run_reader(const collider::runtime::ScanState& state,
                std::atomic<bool>& shutdown,
                const std::atomic<bool>& warmup_done,
                ReaderResult& out) {
    while (!warmup_done.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    uint64_t last_seq = 0;
    for (int i = 0; i < kSnapshotsPerReader && !shutdown.load(std::memory_order_relaxed); ++i) {
        auto snap = state.snapshot();
        out.snapshots_taken += 1;
        if (!snap.consistent) {
            // Retry budget exhausted: writer is publishing faster than this
            // reader can complete a snapshot. The seq-lock contract
            // explicitly tolerates this and surfaces it via the consistent
            // flag; cross-counter invariants are not guaranteed to hold for
            // degraded snapshots, so the reader skips strict checks (still
            // counts the snapshot toward the per-reader total + degraded
            // bucket for visibility).
            out.snapshots_degraded += 1;
            continue;
        }
        out.snapshots_consistent += 1;

        // Invariant 1: real_empty is non-negative.
        if (snap.bloom_collisions_filtered < snap.tight_bloom_filtered) {
            out.ok = false;
            out.failure_reason = "bloom_collisions_filtered < tight_bloom_filtered";
            out.failure_a = snap.bloom_collisions_filtered;
            out.failure_b = snap.tight_bloom_filtered;
            return;
        }

        // Invariant 2: bloom_hits <= total_checked.
        if (snap.bloom_hits > snap.total_checked) {
            out.ok = false;
            out.failure_reason = "bloom_hits > total_checked";
            out.failure_a = snap.bloom_hits;
            out.failure_b = snap.total_checked;
            return;
        }

        // Invariant 3: seq monotonic non-decreasing from this reader's view.
        if (snap.seq < last_seq) {
            out.ok = false;
            out.failure_reason = "snap.seq < last_seq (seq regressed)";
            out.failure_a = snap.seq;
            out.failure_b = last_seq;
            return;
        }
        last_seq = snap.seq;

        // Invariant 4: sum of per-phase keys <= total_checked.
        uint64_t sum_phase_keys = 0;
        for (int p = 0; p < collider::runtime::kNumPhases; ++p) {
            sum_phase_keys += snap.phase_keys_processed[p];
        }
        if (sum_phase_keys > snap.total_checked) {
            out.ok = false;
            out.failure_reason = "sum(phase_keys_processed) > total_checked";
            out.failure_a = sum_phase_keys;
            out.failure_b = snap.total_checked;
            return;
        }

        // Invariant 5: sum of per-phase empty-hits <= real_empty().
        uint64_t sum_phase_empty = 0;
        for (int p = 0; p < collider::runtime::kNumPhases; ++p) {
            sum_phase_empty += snap.empty_hits_by_phase[p];
        }
        if (sum_phase_empty > snap.real_empty()) {
            out.ok = false;
            out.failure_reason = "sum(empty_hits_by_phase) > real_empty()";
            out.failure_a = sum_phase_empty;
            out.failure_b = snap.real_empty();
            return;
        }

        // Invariant 6: current_phase is always in 0..kNumPhases-1.
        if (snap.current_phase < 0 ||
            snap.current_phase >= collider::runtime::kNumPhases) {
            out.ok = false;
            out.failure_reason = "current_phase out of range";
            out.failure_a = static_cast<uint64_t>(snap.current_phase);
            out.failure_b = static_cast<uint64_t>(collider::runtime::kNumPhases);
            return;
        }
    }
    out.worst_seq_seen = last_seq;
}

}  // namespace

int main() {
    using collider::runtime::ScanState;

    std::cout << "test_scan_state_atomics (phase 1 builder-threading)\n";

    ScanState state;

    // Seed via the pre-loop setter (resume baseline path). Must run BEFORE
    // any reader starts; the setter is relaxed and not snapshot-safe.
    state.set_total_checked(0);

    std::atomic<bool> shutdown{false};
    std::atomic<bool> warmup_done{false};

    std::thread writer(run_writer, std::ref(state), std::ref(shutdown), std::ref(warmup_done));

    std::vector<ReaderResult> results(kReaderCount);
    std::vector<std::thread> readers;
    readers.reserve(kReaderCount);
    for (int i = 0; i < kReaderCount; ++i) {
        readers.emplace_back(run_reader, std::cref(state), std::ref(shutdown),
                             std::cref(warmup_done), std::ref(results[i]));
    }

    // Stress for the wall-clock budget, then join.
    std::this_thread::sleep_for(kStressDuration);
    shutdown.store(true, std::memory_order_relaxed);

    for (auto& t : readers) t.join();
    writer.join();

    int failures = 0;
    uint64_t total_snapshots = 0;
    uint64_t total_consistent = 0;
    uint64_t total_degraded = 0;
    for (int i = 0; i < kReaderCount; ++i) {
        const auto& r = results[i];
        total_snapshots += r.snapshots_taken;
        total_consistent += r.snapshots_consistent;
        total_degraded += r.snapshots_degraded;
        if (!r.ok) {
            std::cerr << "[FAIL] reader " << i << ": "
                      << (r.failure_reason ? r.failure_reason : "<unknown>")
                      << " (a=" << r.failure_a << ", b=" << r.failure_b << ")\n";
            ++failures;
        }
    }
    // Quality gate: at least some snapshots must be consistent. If they're
    // all degraded the seq-lock isn't actually being exercised in its
    // happy-path mode and the per-counter invariant assertions above were
    // all skipped. The exact threshold is generous; on a quiet system we
    // expect thousands of consistent snapshots per reader.
    if (total_consistent < static_cast<uint64_t>(kReaderCount) * 10) {
        std::cerr << "[FAIL] only " << total_consistent
                  << " consistent snapshots across " << kReaderCount
                  << " readers (degraded=" << total_degraded
                  << "); seq-lock happy path is not being exercised\n";
        ++failures;
    }

    // Final-state sanity check (single-threaded reader after writer joined).
    auto final_snap = state.snapshot();
    if (final_snap.bloom_collisions_filtered < final_snap.tight_bloom_filtered) {
        std::cerr << "[FAIL] final snapshot: collisions < tight_filtered\n";
        ++failures;
    }
    if (final_snap.bloom_hits > final_snap.total_checked) {
        std::cerr << "[FAIL] final snapshot: bloom_hits > total_checked\n";
        ++failures;
    }
    if (final_snap.real_empty() !=
        (final_snap.bloom_collisions_filtered - final_snap.tight_bloom_filtered)) {
        std::cerr << "[FAIL] real_empty() helper does not match the manual subtraction\n";
        ++failures;
    }
    if (final_snap.seq == 0) {
        std::cerr << "[FAIL] final seq is zero (writer never committed)\n";
        ++failures;
    }

    // Final per-phase invariants. Because the writer always increments
    // phase_keys + global total in the same batch (and inc_phase_keys by
    // exactly kBatchSize), sum(phase_keys_processed) == total_checked
    // strictly at quiescence. Same for empty hits == real_empty().
    uint64_t final_sum_phase_keys = 0;
    uint64_t final_sum_phase_empty = 0;
    for (int p = 0; p < collider::runtime::kNumPhases; ++p) {
        final_sum_phase_keys += final_snap.phase_keys_processed[p];
        final_sum_phase_empty += final_snap.empty_hits_by_phase[p];
    }
    if (final_sum_phase_keys != final_snap.total_checked) {
        std::cerr << "[FAIL] final sum(phase_keys_processed) != total_checked ("
                  << final_sum_phase_keys << " vs " << final_snap.total_checked
                  << ")\n";
        ++failures;
    }
    if (final_sum_phase_empty != final_snap.real_empty()) {
        std::cerr << "[FAIL] final sum(empty_hits_by_phase) != real_empty() ("
                  << final_sum_phase_empty << " vs " << final_snap.real_empty()
                  << ")\n";
        ++failures;
    }
    if (final_snap.current_phase < 0 ||
        final_snap.current_phase >= collider::runtime::kNumPhases) {
        std::cerr << "[FAIL] final current_phase out of range: "
                  << final_snap.current_phase << "\n";
        ++failures;
    }

    // Out-of-range write contract: per the API contract, set_current_phase,
    // inc_phase_keys, inc_empty_hits_by_phase silently drop any index outside
    // 0..kNumPhases-1. Capture the snapshot, fire the bad writes, capture
    // again, and assert the per-phase counters + current_phase are
    // bit-identical. No crash, no exception.
    const auto before_bad_writes = state.snapshot();
    // Bracket the out-of-range writes in a proper seq-lock batch so the
    // post-call snapshot lands on an even seq. The functional check below is
    // unchanged: out-of-range writes must leave every per-phase counter and
    // current_phase / phase_iteration bit-identical.
    state.begin_batch();
    state.inc_phase_keys(99, 1000000);              // way too high
    state.inc_phase_keys(-7, 5);                    // negative
    state.inc_empty_hits_by_phase(collider::runtime::kNumPhases, 42);  // off-by-one
    state.inc_empty_hits_by_phase(-1, 99);
    state.set_current_phase(99, 12345);             // bad phase, bad iter
    state.set_current_phase(-3, 0);
    state.commit_batch();
    const auto after_bad_writes = state.snapshot();

    for (int p = 0; p < collider::runtime::kNumPhases; ++p) {
        if (before_bad_writes.phase_keys_processed[p] !=
            after_bad_writes.phase_keys_processed[p]) {
            std::cerr << "[FAIL] out-of-range inc_phase_keys leaked into phase "
                      << p << " (before=" << before_bad_writes.phase_keys_processed[p]
                      << " after=" << after_bad_writes.phase_keys_processed[p] << ")\n";
            ++failures;
        }
        if (before_bad_writes.empty_hits_by_phase[p] !=
            after_bad_writes.empty_hits_by_phase[p]) {
            std::cerr << "[FAIL] out-of-range inc_empty_hits_by_phase leaked into phase "
                      << p << "\n";
            ++failures;
        }
    }
    if (before_bad_writes.current_phase != after_bad_writes.current_phase) {
        std::cerr << "[FAIL] out-of-range set_current_phase mutated current_phase\n";
        ++failures;
    }
    if (before_bad_writes.phase_iteration != after_bad_writes.phase_iteration) {
        std::cerr << "[FAIL] out-of-range set_current_phase mutated phase_iteration\n";
        ++failures;
    }

    // PhaseRateTracker single-threaded smoke test. Drive a synthetic snapshot
    // sequence: at t0 phase 0 has 0 keys, at t0+1s phase 0 has 1_000_000 keys.
    // keys_per_sec(0) over a generous window should report close to 1_000_000.
    // Other phases stay flat and report 0. We do not depend on real wall clock;
    // PhaseRateTracker captures std::chrono::steady_clock::now() at sample
    // time, so a 1 ms sleep between samples is sufficient to produce a
    // measurable non-zero rate and exercise the slope math.
    {
        using namespace std::chrono_literals;
        collider::runtime::PhaseRateTracker tracker;
        collider::runtime::ScanSnapshot s0;
        for (int p = 0; p < collider::runtime::kNumPhases; ++p) {
            s0.phase_keys_processed[p] = 0;
        }
        tracker.sample(s0);
        std::this_thread::sleep_for(10ms);

        collider::runtime::ScanSnapshot s1 = s0;
        s1.phase_keys_processed[0] = 1'000'000;  // 1 M keys in ~10 ms => ~100 M/s
        tracker.sample(s1);

        const double rate0 = tracker.keys_per_sec(0, 5000);
        if (rate0 <= 0.0) {
            std::cerr << "[FAIL] PhaseRateTracker keys_per_sec(0) returned "
                      << rate0 << " for a strictly-increasing series\n";
            ++failures;
        }
        // Other phases never moved; their rate must be exactly 0.0.
        for (int p = 1; p < collider::runtime::kNumPhases; ++p) {
            const double rp = tracker.keys_per_sec(p, 5000);
            if (rp != 0.0) {
                std::cerr << "[FAIL] PhaseRateTracker keys_per_sec(" << p
                          << ") expected 0.0, got " << rp << "\n";
                ++failures;
            }
        }
        // Out-of-range and degenerate-window inputs must return 0.0 without
        // crashing, matching the documented contract.
        if (tracker.keys_per_sec(-1, 5000) != 0.0) {
            std::cerr << "[FAIL] PhaseRateTracker keys_per_sec(-1) did not return 0.0\n";
            ++failures;
        }
        if (tracker.keys_per_sec(collider::runtime::kNumPhases, 5000) != 0.0) {
            std::cerr << "[FAIL] PhaseRateTracker keys_per_sec(kNumPhases) did not return 0.0\n";
            ++failures;
        }
        if (tracker.keys_per_sec(0, 0) != 0.0) {
            std::cerr << "[FAIL] PhaseRateTracker keys_per_sec(0, window=0) did not return 0.0\n";
            ++failures;
        }
        if (tracker.keys_per_sec(0, -100) != 0.0) {
            std::cerr << "[FAIL] PhaseRateTracker keys_per_sec(0, window=-100) did not return 0.0\n";
            ++failures;
        }
    }

    std::cout << "  readers=" << kReaderCount
              << ", snapshots=" << total_snapshots
              << " (consistent=" << total_consistent
              << ", degraded=" << total_degraded << ")"
              << ", final_seq=" << final_snap.seq
              << ", final_total_checked=" << final_snap.total_checked
              << ", final_bloom_hits=" << final_snap.bloom_hits
              << ", final_sum_phase_keys=" << final_sum_phase_keys
              << ", final_sum_phase_empty=" << final_sum_phase_empty
              << ", final_current_phase=" << final_snap.current_phase
              << "\n";

    if (failures == 0) {
        std::cout << "PASS\n";
        return 0;
    }
    std::cout << "FAIL (" << failures << " failures)\n";
    return 1;
}
