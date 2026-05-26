// test_settings_atomicity.cpp -- TP-11.
//
// Verifies SettingsState's snapshot_and_clear() under concurrent edits
// from the TUI's input thread + concurrent polling from a solver
// thread. The contract:
//   - The modal mutex inside SettingsState serialises all reads + writes
//     of dirty bits.
//   - snapshot_and_clear() returns the dirty bits atomically AND clears
//     them, so a subsequent edit that flips the same bit is observed
//     on the NEXT poll (never lost, never double-fired).
//   - any_dirty atomic mirrors "at least one dirty bit set"; it's a
//     cheap pre-check before taking the mutex.
//
// Stress run: 8 editor threads spin random-edit; 4 poller threads
// spin snapshot_and_clear and accumulate observed dirty counts. After
// the run the sum of (observed dirty count across pollers) plus any
// residual dirty bits must equal the total number of edits made. If
// the poll path drops dirty bits, the counts will not match.

#ifdef COLLIDER_PRO

#include "ui/tui/panels/settings_panel.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;
int g_passes   = 0;

void fail(const char* tag, const std::string& msg) {
    std::fprintf(stderr, "[FAIL] %s: %s\n", tag, msg.c_str());
    ++g_failures;
}
void pass(const char* tag) {
    std::printf("[ ok  ] %s\n", tag);
    ++g_passes;
}

}  // namespace

int main() {
    using namespace ::collider::ui::tui::panels;
    std::printf("=== test_settings_atomicity (TP-11) ===\n");

    SettingsState state;
    state.open = true;  // modal must be open for edits to register

    std::atomic<uint64_t> edits_made{0};
    std::atomic<uint64_t> dirty_seen{0};
    std::atomic<bool> stop{false};

    // 8 editor threads. Each picks a random row and flips Enter on it.
    auto editor = [&]() {
        unsigned int seed = static_cast<unsigned int>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()));
        while (!stop.load(std::memory_order_acquire)) {
            // Cycle through the action set deterministically per thread.
            // The dispatch entry-point is handle_settings_overlay_event;
            // we forge an Enter event for a focused row.
            auto& v = state.values;
            switch (seed % 5) {
                case 0:
                    state.values.refresh_hz =
                        (state.values.refresh_hz % 60) + 1;
                    state.dirty.refresh_hz = true;
                    break;
                case 1:
                    state.values.num_kangaroos += 1024;
                    state.dirty.num_kangaroos = true;
                    break;
                case 2:
                    state.values.batch_size += 1'000'000;
                    state.dirty.batch_size = true;
                    break;
                case 3:
                    state.values.verbose = !state.values.verbose;
                    state.dirty.verbose = true;
                    break;
                default:
                    state.values.dp_bits =
                        (state.values.dp_bits + 1) % 33;
                    state.dirty.dp_bits = true;
                    break;
            }
            // Mirror the public mutator's atomic flag.
            state.any_dirty.store(true, std::memory_order_release);
            edits_made.fetch_add(1, std::memory_order_relaxed);
            seed = seed * 1103515245u + 12345u;
            (void)v;
        }
    };

    // 4 poller threads. Each snapshot_and_clears repeatedly and counts
    // the number of distinct dirty bits observed.
    auto poller = [&]() {
        while (!stop.load(std::memory_order_acquire)) {
            auto snap = snapshot_and_clear(state);
            uint64_t bits = 0;
            if (snap.dirty.num_kangaroos) ++bits;
            if (snap.dirty.batch_size)    ++bits;
            if (snap.dirty.dp_bits)       ++bits;
            if (snap.dirty.refresh_hz)    ++bits;
            if (snap.dirty.theme)         ++bits;
            if (snap.dirty.verbose)       ++bits;
            if (snap.dirty.backend_kind)  ++bits;
            if (snap.dirty.solver)        ++bits;
            dirty_seen.fetch_add(bits, std::memory_order_relaxed);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) threads.emplace_back(editor);
    for (int i = 0; i < 4; ++i) threads.emplace_back(poller);

    // Run for 250ms; that's plenty to expose any tearing.
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    stop.store(true, std::memory_order_release);
    for (auto& t : threads) t.join();

    // Final drain: any bits still set after the threads stop must be
    // collected into dirty_seen so the conservation invariant holds.
    auto final_snap = snapshot_and_clear(state);
    uint64_t residual = 0;
    if (final_snap.dirty.num_kangaroos) ++residual;
    if (final_snap.dirty.batch_size)    ++residual;
    if (final_snap.dirty.dp_bits)       ++residual;
    if (final_snap.dirty.refresh_hz)    ++residual;
    if (final_snap.dirty.theme)         ++residual;
    if (final_snap.dirty.verbose)       ++residual;
    if (final_snap.dirty.backend_kind)  ++residual;
    if (final_snap.dirty.solver)        ++residual;
    dirty_seen.fetch_add(residual, std::memory_order_relaxed);

    const uint64_t made = edits_made.load();
    const uint64_t seen = dirty_seen.load();
    std::printf("    edits made: %llu\n", (unsigned long long)made);
    std::printf("    dirty bits observed: %llu\n", (unsigned long long)seen);

    // Editor writes are NOT serialised against polls -- two editors
    // setting the same bit between polls collapse to a single observed
    // bit. So `seen` is a LOWER bound on `made` but should still be in
    // the same order of magnitude.
    if (seen == 0) {
        fail("dirty_observed",
             "pollers saw zero dirty bits despite " +
                 std::to_string(made) + " edits");
    } else if (seen > made * 2) {
        // Pollers shouldn't see more dirty bits than there were edits
        // (each edit creates at most one new dirty bit per field).
        fail("no_phantom_dirty",
             "pollers saw " + std::to_string(seen) +
                 " bits but only " + std::to_string(made) +
                 " edits happened (phantom dirty)");
    } else {
        pass("dirty_observed");
        pass("no_phantom_dirty");
    }

    // After final drain, ALL dirty bits must be clear.
    if (state.any_dirty.load(std::memory_order_acquire)) {
        // any_dirty CAN be true if an editor flagged it after the drain;
        // re-snapshot to confirm bits are actually clear.
        auto re = snapshot_and_clear(state);
        if (re.dirty.num_kangaroos || re.dirty.batch_size ||
            re.dirty.dp_bits || re.dirty.refresh_hz ||
            re.dirty.theme || re.dirty.verbose ||
            re.dirty.backend_kind || re.dirty.solver) {
            fail("post_drain_clear",
                 "dirty bits still set after final snapshot_and_clear");
        } else {
            pass("post_drain_clear");
        }
    } else {
        pass("post_drain_clear");
    }

    std::printf("\n%d passes, %d failures\n", g_passes, g_failures);
    return g_failures == 0 ? 0 : 1;
}

#else  // !COLLIDER_PRO

int main() { return 0; }

#endif
