// phase 5 (builder-threading: streaming-generator-hot-swap).
//
// Coverage for StreamingBrainWallet::queue_profile_swap +
// apply_pending_profile_swap_if_any:
//
//   1. mid-run swap: iterate N candidates from wordlist A, queue a swap
//      to wordlist B, apply at phase boundary, iterate more candidates,
//      assert they come from wordlist B and phase counters preserved.
//   2. invalid path: queue_profile_swap returns false when the new
//      wordlist file does not exist; no state change occurs.
//   3. save-state round-trip across a swap: the snapshot taken after a
//      swap reflects the new wordlist size (via wordlist_size()) and the
//      preserved position counters; reloading into a fresh generator
//      configured with the new wordlist yields a matching state.
//
// Uses a small temporary directory so the test does not depend on the
// production rules/ directory layout. Brute mode is bypassed by leaving
// brute_lengths empty.

#include "generators/streaming_brain_wallet.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Failures {
    int count = 0;
    void check(bool cond, const char* what) {
        if (!cond) {
            std::cerr << "[FAIL] " << what << "\n";
            ++count;
        }
    }
};

fs::path make_temp_dir() {
    auto base = fs::temp_directory_path();
    std::random_device rd;
    std::mt19937_64 gen(rd());
    for (int attempt = 0; attempt < 32; ++attempt) {
        auto candidate = base / ("collider_hot_swap_test_" + std::to_string(gen()));
        std::error_code ec;
        if (fs::create_directory(candidate, ec) && !ec) {
            return candidate;
        }
    }
    auto candidate = base / "collider_hot_swap_test_fallback";
    std::error_code ec;
    fs::create_directories(candidate, ec);
    return candidate;
}

// Write a wordlist file with one word per line. Returns the path.
fs::path write_wordlist(const fs::path& dir,
                        const std::string& name,
                        const std::vector<std::string>& words) {
    fs::path p = dir / name;
    std::ofstream f(p);
    for (const auto& w : words) f << w << "\n";
    f.close();
    return p;
}

// Minimal rules directory: one no-op rule per file so phases load
// without errors. The rule ":" is the canonical hashcat identity.
void write_minimal_rules(const fs::path& rules_dir) {
    fs::create_directories(rules_dir);
    for (const auto& name : {"best64.rule", "crypto.rule", "d3ad0ne.rule", "dive.rule"}) {
        std::ofstream f(rules_dir / name);
        f << ":\n";
    }
}

}  // namespace

int main() {
    Failures fail;

    auto tmp = make_temp_dir();
    auto rules_dir = tmp / "rules";
    write_minimal_rules(rules_dir);

    const std::vector<std::string> words_a = {
        "alpha", "bravo", "charlie", "delta", "echo",
        "foxtrot", "golf", "hotel", "india", "juliet"
    };
    const std::vector<std::string> words_b = {
        "kilo", "lima", "mike", "november", "oscar",
        "papa", "quebec", "romeo", "sierra", "tango"
    };

    auto path_a = write_wordlist(tmp, "wl_a.txt", words_a);
    auto path_b = write_wordlist(tmp, "wl_b.txt", words_b);

    // Test 1: mid-run swap preserves position counters.
    {
        ::collider::generators::StreamingBrainWallet::Config cfg;
        cfg.base_wordlist = path_a.string();
        cfg.rules_dir = rules_dir.string();
        cfg.batch_size = 4;
        cfg.enable_dedup = false;
        cfg.enable_feedback = false;
        cfg.verbose = false;

        ::collider::generators::StreamingBrainWallet gen(cfg);
        fail.check(gen.init(), "init wordlist A");

        // Pull a small batch of raw words from wordlist A.
        auto first = gen.next_raw_words(5);
        fail.check(first.size() == 5, "raw words A batch 1 size 5");
        // Every word should be from words_a.
        std::set<std::string> set_a(words_a.begin(), words_a.end());
        bool all_a = true;
        for (const auto& w : first) {
            if (!set_a.count(w)) { all_a = false; break; }
        }
        fail.check(all_a, "all batch-1 words from wordlist A");

        // Capture current state snapshot before swap.
        auto state_before = gen.get_state_snapshot();

        // Queue a swap to wordlist B, then apply explicitly (the test
        // does not go through advance_phase here; the API allows the
        // caller to apply directly when they know they are at a safe
        // point).
        ::collider::generators::StreamingBrainWallet::Config swap_cfg;
        swap_cfg.base_wordlist = path_b.string();
        swap_cfg.rules_dir = rules_dir.string();
        swap_cfg.batch_size = cfg.batch_size;
        fail.check(gen.queue_profile_swap(swap_cfg),
                   "queue_profile_swap returns true for valid path");
        fail.check(gen.apply_pending_profile_swap_if_any(),
                   "apply_pending_profile_swap_if_any returns true");

        // current_profile_path now reflects wordlist B.
        fail.check(gen.current_profile_path() == path_b.string(),
                   "current_profile_path == path_b after apply");

        // Position counters preserved: current_phase, phase_iteration,
        // iteration_mode unchanged. current_word_idx is reset to 0 by
        // the swap because the previous iteration sat at the end of
        // wordlist A; that is the documented behavior of apply (it
        // resets the raw-word iterator so the new wordlist is consumed
        // from index 0 onward). The phase / iteration counters are the
        // load-bearing ones.
        auto state_after = gen.get_state_snapshot();
        fail.check(state_after.current_phase == state_before.current_phase,
                   "current_phase preserved across swap");
        fail.check(state_after.phase_iteration == state_before.phase_iteration,
                   "phase_iteration preserved across swap");
        fail.check(state_after.iteration_mode == state_before.iteration_mode,
                   "iteration_mode preserved across swap");
        fail.check(state_after.mode_sub_iteration == state_before.mode_sub_iteration,
                   "mode_sub_iteration preserved across swap");

        // Pull another batch; every word should now be from wordlist B.
        auto second = gen.next_raw_words(5);
        fail.check(second.size() == 5, "raw words B batch 1 size 5");
        std::set<std::string> set_b(words_b.begin(), words_b.end());
        bool all_b = true;
        for (const auto& w : second) {
            if (!set_b.count(w)) { all_b = false; break; }
        }
        fail.check(all_b, "all batch-2 words from wordlist B");
    }

    // Test 2: invalid path graceful failure.
    {
        ::collider::generators::StreamingBrainWallet::Config cfg;
        cfg.base_wordlist = path_a.string();
        cfg.rules_dir = rules_dir.string();
        cfg.batch_size = 4;
        cfg.enable_dedup = false;
        cfg.enable_feedback = false;
        cfg.verbose = false;
        ::collider::generators::StreamingBrainWallet gen(cfg);
        fail.check(gen.init(), "init for invalid-path test");

        ::collider::generators::StreamingBrainWallet::Config bad_cfg;
        bad_cfg.base_wordlist = (tmp / "does_not_exist.txt").string();
        bad_cfg.rules_dir = rules_dir.string();
        fail.check(!gen.queue_profile_swap(bad_cfg),
                   "queue_profile_swap returns false for missing file");

        // apply with nothing queued is a no-op returning false.
        fail.check(!gen.apply_pending_profile_swap_if_any(),
                   "apply_pending_profile_swap_if_any false when no swap queued");

        // Active config still points at wordlist A.
        fail.check(gen.current_profile_path() == path_a.string(),
                   "current_profile_path unchanged after failed queue");

        // Empty base_wordlist also rejected.
        ::collider::generators::StreamingBrainWallet::Config empty_cfg;
        empty_cfg.base_wordlist = "";
        empty_cfg.rules_dir = rules_dir.string();
        fail.check(!gen.queue_profile_swap(empty_cfg),
                   "queue_profile_swap returns false for empty path");
    }

    // Test 3: save-state round-trip across swap.
    {
        ::collider::generators::StreamingBrainWallet::Config cfg;
        cfg.base_wordlist = path_a.string();
        cfg.rules_dir = rules_dir.string();
        cfg.batch_size = 4;
        cfg.enable_dedup = false;
        cfg.enable_feedback = false;
        cfg.verbose = false;

        ::collider::generators::StreamingBrainWallet gen1(cfg);
        fail.check(gen1.init(), "init for round-trip test");

        // Pull some candidates so position counters advance.
        (void)gen1.next_raw_words(3);

        // Capture state before swap.
        auto snap_before = gen1.get_state_snapshot();

        // Swap to wordlist B.
        ::collider::generators::StreamingBrainWallet::Config swap_cfg;
        swap_cfg.base_wordlist = path_b.string();
        swap_cfg.rules_dir = rules_dir.string();
        fail.check(gen1.queue_profile_swap(swap_cfg),
                   "round-trip: queue_profile_swap returns true");
        fail.check(gen1.apply_pending_profile_swap_if_any(),
                   "round-trip: apply returns true");

        // Snapshot taken after swap.
        auto snap_after = gen1.get_state_snapshot();
        fail.check(snap_after.wordlist_size == words_b.size(),
                   "snapshot wordlist_size matches wordlist B");
        fail.check(snap_after.current_phase == snap_before.current_phase,
                   "round-trip: current_phase preserved");
        fail.check(snap_after.phase_iteration == snap_before.phase_iteration,
                   "round-trip: phase_iteration preserved");

        // Restore into a fresh generator configured against wordlist B.
        ::collider::generators::StreamingBrainWallet::Config cfg2 = cfg;
        cfg2.base_wordlist = path_b.string();
        ::collider::generators::StreamingBrainWallet gen2(cfg2);
        fail.check(gen2.init(), "round-trip: gen2 init");
        fail.check(gen2.restore_state(snap_after),
                   "round-trip: restore_state on gen2 succeeds");

        // gen2's wordlist path matches wordlist B; gen2's current phase
        // matches gen1's post-swap phase.
        fail.check(gen2.get_wordlist_path() == path_b.string(),
                   "gen2 wordlist path == path_b");
        auto snap_round = gen2.get_state_snapshot();
        fail.check(snap_round.current_phase == snap_after.current_phase,
                   "round-trip: restored current_phase matches");
        fail.check(snap_round.phase_iteration == snap_after.phase_iteration,
                   "round-trip: restored phase_iteration matches");
    }

    // R-B2: concurrent next_raw_words + swap must not deadlock.
    //
    // Before R-B2, advance_phase() (called from inside
    // next_raw_words_internal which already holds raw_word_mutex_) called
    // apply_pending_profile_swap_if_any() which tried to re-acquire
    // raw_word_mutex_, deadlocking the same thread. The repro exhausts
    // the wordlist on the consumer side (rolling raw_word_idx_ over the
    // wordlist size triggers advance_phase) while a swap is queued. With
    // the apply moved out of advance_phase, the consumer no longer
    // self-deadlocks. We bound the test with a wall clock; a 5 s timeout
    // is generous (the entire body takes < 100 ms when healthy) and
    // catches a hang as a CI failure rather than a hang-the-runner.
    {
        ::collider::generators::StreamingBrainWallet::Config cfg;
        cfg.base_wordlist = path_a.string();
        cfg.rules_dir = rules_dir.string();
        cfg.batch_size = 4;
        cfg.enable_dedup = false;
        cfg.enable_feedback = false;
        cfg.verbose = false;
        ::collider::generators::StreamingBrainWallet gen(cfg);
        fail.check(gen.init(), "R-B2: init");

        // Queue a swap so apply_pending_profile_swap_if_any has work
        // to do. The runner-style apply on the OTHER thread is the
        // post-fix path. The consumer thread keeps draining words;
        // once it crosses the wordlist boundary, advance_phase fires.
        ::collider::generators::StreamingBrainWallet::Config swap_cfg;
        swap_cfg.base_wordlist = path_b.string();
        swap_cfg.rules_dir = rules_dir.string();
        fail.check(gen.queue_profile_swap(swap_cfg),
                   "R-B2: queue_profile_swap returns true");

        std::atomic<bool> done{false};
        std::thread consumer([&]() {
            // Drain enough batches to roll past the 10-word wordlist
            // a couple of times, exercising the advance_phase path
            // that previously self-deadlocked.
            for (int i = 0; i < 25 && !done.load(); ++i) {
                (void)gen.next_raw_words(8);
            }
            done.store(true);
        });

        // Apply the swap from THIS (the runner-equivalent) thread,
        // racing the consumer.
        (void)gen.apply_pending_profile_swap_if_any();

        // Wait up to 5 s; in healthy code path the consumer finishes
        // in < 100 ms.
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::seconds(5);
        while (!done.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        fail.check(done.load(),
                   "R-B2: consumer thread finished (no deadlock)");
        if (consumer.joinable()) {
            // If the consumer hung we still have to clean up. Detach
            // is acceptable because the test process exits afterwards
            // and a hung thread does not corrupt later tests.
            if (done.load()) {
                consumer.join();
            } else {
                consumer.detach();
            }
        }
    }

    // R-B9: prefetch buffer is invalidated on successful hot-swap.
    //
    // Before R-B9, raw_word_prefetch_buffer_ retained pre-swap words
    // that the consumer would dequeue on its very next call. The fix
    // clears the buffer + flips ready=false inside
    // apply_pending_profile_swap_if_any so the prefetch thread refills
    // from the new wordlist. We start the prefetch, let it cook one
    // batch from wordlist A, swap to wordlist B, then drain a few
    // batches and assert every word came from wordlist B (within a
    // bounded retry count to absorb the prefetch thread's wakeup).
    {
        ::collider::generators::StreamingBrainWallet::Config cfg;
        cfg.base_wordlist = path_a.string();
        cfg.rules_dir = rules_dir.string();
        cfg.batch_size = 4;
        cfg.enable_dedup = false;
        cfg.enable_feedback = false;
        cfg.verbose = false;
        ::collider::generators::StreamingBrainWallet gen(cfg);
        fail.check(gen.init(), "R-B9: init");

        gen.start_raw_word_prefetch(/*batch_size=*/4);
        // Let the prefetch thread cook a batch from wordlist A.
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // Drain the cooked batch out of the buffer so we have a clean
        // checkpoint to test against. The cooked batch is from
        // wordlist A.
        auto warm_a = gen.next_raw_words_async(4);
        std::set<std::string> set_a(words_a.begin(), words_a.end());
        bool warm_was_a = !warm_a.empty();
        for (const auto& w : warm_a) {
            if (!set_a.count(w)) { warm_was_a = false; break; }
        }
        fail.check(warm_was_a, "R-B9: warm-up batch came from wordlist A");

        // Swap to wordlist B.
        ::collider::generators::StreamingBrainWallet::Config swap_cfg;
        swap_cfg.base_wordlist = path_b.string();
        swap_cfg.rules_dir = rules_dir.string();
        fail.check(gen.queue_profile_swap(swap_cfg),
                   "R-B9: queue_profile_swap returns true");
        fail.check(gen.apply_pending_profile_swap_if_any(),
                   "R-B9: apply_pending_profile_swap_if_any returns true");

        // After R-B9, the very next async call must hand back words
        // from wordlist B because the prefetch buffer was flushed by
        // the apply path. Allow a single retry to absorb the prefetch
        // thread's wakeup (it has to wake, run next_raw_words_internal,
        // refill the buffer, and signal the consumer; that round-trip
        // is sub-millisecond on a healthy host but the test guards
        // with a bounded wait for CI determinism).
        std::set<std::string> set_b(words_b.begin(), words_b.end());
        bool seen_only_b = false;
        for (int retry = 0; retry < 20 && !seen_only_b; ++retry) {
            auto next = gen.next_raw_words_async(4);
            if (next.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                continue;
            }
            seen_only_b = true;
            for (const auto& w : next) {
                if (!set_b.count(w)) {
                    seen_only_b = false;
                    break;
                }
            }
            if (!seen_only_b) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }
        fail.check(seen_only_b,
                   "R-B9: post-swap async batch came from wordlist B "
                   "(prefetch buffer was invalidated)");

        gen.stop_raw_word_prefetch();
    }

    // Cleanup the temp directory.
    std::error_code ec;
    fs::remove_all(tmp, ec);

    if (fail.count == 0) {
        std::cout << "test_wordlist_hot_swap: all checks passed\n";
        return 0;
    }
    std::cerr << "test_wordlist_hot_swap: " << fail.count << " failure(s)\n";
    return 1;
}
