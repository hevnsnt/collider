// phase 4 (builder-panels: input handler).
//
// Exercises the keyboard dispatch surface. Each test feeds synthetic
// FTXUI events into InputHandler::on_event() and asserts that the
// corresponding RuntimeControlState atomics / mutex-guarded fields
// mutate as documented in the locked contract. State machine
// behavior (the scan loop's pause-drain, the GPU toggle's
// drain-and-free) lives in builder-threading's backend tests; this
// test only validates that the operator's key arrives at the right
// field with the right value.
//
// Strategy: each subtest resets the runtime state to a known baseline,
// dispatches one or more events, and asserts a single observable
// effect. The InputHandler is reset between subtests so the chord
// latch from one case cannot bleed into the next.

#include "runtime/runtime_control.hpp"
#include "ui/tui/input_handler.hpp"
#include "ui/tui/keybindings.hpp"
#include "ui/tui/panels/bloom_picker.hpp"
#include "ui/tui/panels/help_overlay.hpp"
#include "ui/tui/panels/recent_hits_panel.hpp"
#include "ui/tui/panels/wordlist_picker.hpp"

#include <ftxui/component/event.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

// Reset RuntimeControlState to the documented neutral baseline so each
// subtest starts from the same point.
void reset_runtime_state() {
    using namespace collider::runtime;
    auto& rc = global_runtime_control();
    rc.quit_requested.store(false, std::memory_order_release);
    rc.pause_requested.store(false, std::memory_order_release);
    rc.is_paused.store(false, std::memory_order_release);
    rc.save_requested.store(false, std::memory_order_release);
    // Restore all GPU bits.
    rc.gpu_enable_mask.store(0xFFu, std::memory_order_release);
    for (int i = 0; i < RuntimeControlState::kMaxGpus; ++i) {
        rc.gpu_phase[i].store(
            RuntimeControlState::GpuPhase::Active,
            std::memory_order_release);
    }
    rc.requested_batch_size.store(0, std::memory_order_release);
    rc.last_applied_batch_size.store(5'000'000ULL, std::memory_order_release);
    rc.requested_rule_chunk_size.store(0, std::memory_order_release);
    rc.last_applied_rule_chunk_size.store(500ULL, std::memory_order_release);
    rc.requested_theme_variant.store(-1, std::memory_order_release);
    // Tier 2 D1: reset focused-panel state so each subtest starts
    // from the default multi-panel layout.
    rc.requested_focused_panel.store(
        RuntimeControlState::kFocusNone, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(rc.bloom_mu);
        rc.requested_bloom_path.clear();
        rc.last_applied_bloom_path.clear();
    }
    {
        std::lock_guard<std::mutex> lk(rc.profile_mu);
        rc.requested_wordlist_profile.clear();
        rc.last_applied_wordlist_profile.clear();
    }
    {
        std::lock_guard<std::mutex> lk(rc.banner_mu);
        rc.banner_text.clear();
    }
}

}  // namespace

int main() {
    using namespace collider::ui::tui;
    using ftxui::Event;
    auto& rc = collider::runtime::global_runtime_control();

    // ================================================================
    // 1. Quit: 'q' sets quit_requested.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        const bool consumed = h.on_event(Event::Character('q'));
        if (!consumed) return fail("quit: 'q' not consumed");
        if (!rc.quit_requested.load(std::memory_order_acquire))
            return fail("quit: quit_requested not set");
    }

    // ================================================================
    // 2. Pause toggles. First 'p' sets pause_requested, second clears.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        h.on_event(Event::Character('p'));
        if (!rc.pause_requested.load(std::memory_order_acquire))
            return fail("pause: first 'p' did not set pause_requested");
        h.on_event(Event::Character('p'));
        if (rc.pause_requested.load(std::memory_order_acquire))
            return fail("pause: second 'p' did not clear pause_requested");
    }

    // ================================================================
    // 3. Batch size up: '+' produces a non-zero requested_batch_size
    //    above the current last_applied value.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        h.on_event(Event::Character('+'));
        const uint64_t req = rc.requested_batch_size.load(
            std::memory_order_acquire);
        if (req == 0) return fail("batch up: requested_batch_size not set");
        const uint64_t cur = rc.last_applied_batch_size.load(
            std::memory_order_acquire);
        if (req <= cur) return fail("batch up: requested not above current");
    }

    // ================================================================
    // 4. Batch size down: '-' produces a requested below the current.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        h.on_event(Event::Character('-'));
        const uint64_t req = rc.requested_batch_size.load(
            std::memory_order_acquire);
        if (req == 0) return fail("batch down: requested_batch_size not set");
        const uint64_t cur = rc.last_applied_batch_size.load(
            std::memory_order_acquire);
        if (req >= cur) return fail("batch down: requested not below current");
    }

    // ================================================================
    // 5. Rule chunk cycle: 'r' advances from 500 to 1000.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        h.on_event(Event::Character('r'));
        const uint64_t req = rc.requested_rule_chunk_size.load(
            std::memory_order_acquire);
        if (req != 1000ULL)
            return fail("rule chunk: expected 1000 after 'r' from 500");
    }

    // ================================================================
    // 6. Save now: 's' sets save_requested.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        h.on_event(Event::Character('s'));
        if (!rc.save_requested.load(std::memory_order_acquire))
            return fail("save: save_requested not set");
    }

    // ================================================================
    // 7. Theme cycle: 't' updates requested_theme_variant from -1 to 1.
    //    The cycle now spans four variants (Default / HighContrast /
    //    Monochrome / Light) so the modulus is 4. T-B2: verify Light
    //    (variant index 3) is reachable through repeated 't' presses
    //    starting from -1. The previous bug was kThemeVariantCount=3
    //    which made `next %= 3` wrap before ever reaching Light.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        h.on_event(Event::Character('t'));
        const int req = rc.requested_theme_variant.load(
            std::memory_order_acquire);
        // First press from -1 lands on 1 (HighContrast) per the
        // dispatcher's "prev == -1 -> next = 1" branch.
        if (req != 1)
            return fail("theme: first 't' from -1 should land on 1");

        // Walk the cycle until Light (3) is observed; confirms the
        // Light variant is reachable from the keyboard. The modulus
        // is 4 so at most 4 more presses are needed.
        bool saw_light = false;
        for (int i = 0; i < 5 && !saw_light; ++i) {
            h.on_event(Event::Character('t'));
            const int v = rc.requested_theme_variant.load(
                std::memory_order_acquire);
            if (v < 0 || v >= 4) {
                return fail("theme: variant out of range during cycle");
            }
            if (v == 3) saw_light = true;
        }
        if (!saw_light)
            return fail("theme: Light (variant 3) not reachable via 't' cycle");

        // Wrap: pressing 't' from 3 must land on 0 (Default), not on
        // an out-of-range value.
        rc.requested_theme_variant.store(3, std::memory_order_release);
        h.on_event(Event::Character('t'));
        const int after_wrap = rc.requested_theme_variant.load(
            std::memory_order_acquire);
        if (after_wrap != 0)
            return fail("theme: cycle should wrap from 3 to 0");
    }

    // ================================================================
    // 8. GPU chord: 'g' starts chord (awaiting_chord true), '0' toggles
    //    bit 0 in gpu_enable_mask (XOR), chord clears.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        const uint8_t before = rc.gpu_enable_mask.load(
            std::memory_order_acquire);
        h.on_event(Event::Character('g'));
        if (!h.awaiting_chord())
            return fail("chord: 'g' did not set awaiting_chord");
        h.on_event(Event::Character('0'));
        if (h.awaiting_chord())
            return fail("chord: '0' did not clear awaiting_chord");
        const uint8_t after = rc.gpu_enable_mask.load(
            std::memory_order_acquire);
        if (after == before)
            return fail("chord: gpu_enable_mask did not change after g+0");
        if ((before & 0x01u) == (after & 0x01u))
            return fail("chord: bit 0 was not toggled");
    }

    // ================================================================
    // 9. Chord reset on second 'g': pressing 'g' then 'g' cancels the
    //    chord WITHOUT dispatching a GPU toggle. The chord is
    //    one-deep per spec; the second 'g' is treated as "any other
    //    key" and clears awaiting_chord cleanly. A subsequent '0' is
    //    just a character (no chord active) so gpu_enable_mask must
    //    NOT change across the whole g+g+0 sequence.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        const uint8_t before = rc.gpu_enable_mask.load(
            std::memory_order_acquire);
        h.on_event(Event::Character('g'));
        if (!h.awaiting_chord())
            return fail("chord-reset: first 'g' did not latch");
        h.on_event(Event::Character('g'));
        // The second 'g' is one-deep cancel: awaiting_chord must
        // clear and no further dispatch happens.
        if (h.awaiting_chord())
            return fail("chord-reset: second 'g' did not cancel chord");
        // No GPU toggle should have fired across the cancel.
        const uint8_t mid = rc.gpu_enable_mask.load(
            std::memory_order_acquire);
        if (mid != before)
            return fail("chord-reset: gpu_enable_mask changed on 'g'+'g'");
        // The trailing '0' arrives with no chord active, so it should
        // be a plain unmapped character (lookup returns None) and the
        // mask must stay unchanged.
        h.on_event(Event::Character('0'));
        const uint8_t after = rc.gpu_enable_mask.load(
            std::memory_order_acquire);
        if (after != before)
            return fail("chord-reset: bit 0 toggled outside of chord");
    }

    // ================================================================
    // 10. Bloom picker open: 'b' sets BloomPickerState::open = true.
    //     selected_path stays empty (no selection yet).
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        panels::BloomPickerState picker;
        h.on_event(Event::Character('b'), &picker);
        if (!picker.open)
            return fail("bloom: 'b' did not open picker");
        if (!picker.selected_path.empty())
            return fail("bloom: selected_path should be empty before commit");
    }

    // ================================================================
    // 11. Bloom picker routing: while open, 'q' does NOT trigger quit.
    //     The modal swallows every key except its own (Esc / Enter /
    //     arrows). This protects against an accidental quit while the
    //     picker is up.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        panels::BloomPickerState picker;
        // Pre-populate so the picker can render meaningful state, but
        // also synthesize a candidate so Enter can commit a selection.
        picker.candidates = {"alpha.blf", "beta.blf"};
        picker.open = true;
        picker.selected_index = 1;

        // 'q' should be consumed by the modal, NOT trigger quit.
        h.on_event(Event::Character('q'), &picker);
        if (rc.quit_requested.load(std::memory_order_acquire))
            return fail("bloom-modal: 'q' while open should not trigger quit");
        if (!picker.open)
            return fail("bloom-modal: 'q' should not close modal");

        // Enter should commit + close + write to requested_bloom_path.
        h.on_event(Event::Return, &picker);
        if (picker.open)
            return fail("bloom-modal: Enter did not close modal");
        std::string committed;
        {
            std::lock_guard<std::mutex> lk(rc.bloom_mu);
            committed = rc.requested_bloom_path;
        }
        if (committed != "beta.blf")
            return fail("bloom-modal: requested_bloom_path != selected");
    }

    // ================================================================
    // 12. Bloom picker cancel: Esc closes without writing
    //     requested_bloom_path.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        panels::BloomPickerState picker;
        picker.candidates = {"alpha.blf"};
        picker.open = true;
        picker.selected_index = 0;
        h.on_event(Event::Escape, &picker);
        if (picker.open)
            return fail("bloom-cancel: Esc did not close modal");
        std::string committed;
        {
            std::lock_guard<std::mutex> lk(rc.bloom_mu);
            committed = rc.requested_bloom_path;
        }
        if (!committed.empty())
            return fail("bloom-cancel: Esc should not write bloom path");
    }

    // ================================================================
    // 13. Keybindings table integrity: lookup() returns the expected
    //     Action for every documented key. Reserved Phase-5 binds
    //     ('w', 'l') return Action::None per the contract.
    // ================================================================
    {
        if (lookup('q') != Action::Quit)
            return fail("table: 'q' should map to Quit");
        if (lookup('?') != Action::Help)
            return fail("table: '?' should map to Help");
        if (lookup('p') != Action::Pause)
            return fail("table: 'p' should map to Pause");
        if (lookup('+') != Action::BatchSizeUp)
            return fail("table: '+' should map to BatchSizeUp");
        if (lookup('-') != Action::BatchSizeDown)
            return fail("table: '-' should map to BatchSizeDown");
        if (lookup('r') != Action::RuleChunkCycle)
            return fail("table: 'r' should map to RuleChunkCycle");
        if (lookup('b') != Action::BloomPicker)
            return fail("table: 'b' should map to BloomPicker");
        if (lookup('s') != Action::SaveNow)
            return fail("table: 's' should map to SaveNow");
        if (lookup('t') != Action::ThemeCycle)
            return fail("table: 't' should map to ThemeCycle");
        if (lookup('g') != Action::GpuToggle)
            return fail("table: 'g' should map to GpuToggle");
        // Phase 5 wiring (Wave 9): w/l now dispatch live Actions.
        if (lookup('w') != Action::WordlistPicker)
            return fail("table: 'w' should map to WordlistPicker");
        if (lookup('l') != Action::RecentHits)
            return fail("table: 'l' should map to RecentHits");
        // Uppercase: lookup() lowercases first.
        if (lookup('Q') != Action::Quit)
            return fail("table: 'Q' should fold to Quit");
    }

    // ================================================================
    // 14. active_footer_bindings() omits the always-on quit entry but
    //     surfaces every live bind (including the Phase 5 wordlist +
    //     recent-hits binds wired in Wave 9).
    // ================================================================
    {
        auto live = active_footer_bindings();
        bool found_wordlist = false;
        bool found_recent = false;
        for (const auto& kb : live) {
            if (kb.action == Action::Quit)
                return fail("footer: Quit should be omitted from active list");
            if (kb.action == Action::WordlistPicker) found_wordlist = true;
            if (kb.action == Action::RecentHits) found_recent = true;
        }
        if (!found_wordlist)
            return fail("footer: WordlistPicker missing from active list");
        if (!found_recent)
            return fail("footer: RecentHits missing from active list");
    }

    // ================================================================
    // 15. format_cheatsheet() produces non-empty multi-line output
    //     containing each documented key. The Phase-6 help overlay
    //     splits on '\n' and renders each line, so a non-empty
    //     line-count is the only assertion that matters here.
    // ================================================================
    {
        const std::string sheet = format_cheatsheet();
        if (sheet.empty())
            return fail("cheatsheet: format_cheatsheet returned empty");
        // Count newlines: one per binding. The table has 12 entries,
        // so we expect at least 12 lines.
        int nl = 0;
        for (char c : sheet) if (c == '\n') ++nl;
        if (nl < 12)
            return fail("cheatsheet: expected at least 12 lines");
    }

    // ================================================================
    // 16. Wordlist picker open: 'w' sets WordlistPickerState::open.
    //     Phase 5 / Wave 9 wiring. The new host-context overload is
    //     used so the handler can dispatch the wordlist Action; the
    //     bloom-only overload (used by tests 10-12) keeps working.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        collider::ui::tui::panels::WordlistPickerState picker;
        collider::ui::tui::InputHandlerHostContext host;
        host.wordlist_picker = &picker;
        h.on_event(Event::Character('w'), host);
        if (!picker.open)
            return fail("wordlist: 'w' did not open picker");
        if (!picker.selected_path.empty())
            return fail("wordlist: selected_path should be empty before commit");
    }

    // ================================================================
    // 17. Wordlist picker routing: while open, 'q' does NOT trigger
    //     quit (the modal swallows every key except its own).
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        collider::ui::tui::panels::WordlistPickerState picker;
        collider::ui::tui::panels::WordlistProfile p;
        p.path = "/tmp/alpha.txt";
        p.display_name = "alpha.txt";
        picker.candidates.push_back(p);
        picker.candidates.push_back({"/tmp/beta.txt", "beta.txt", 0, false});
        picker.open = true;
        picker.selected_index = 1;

        collider::ui::tui::InputHandlerHostContext host;
        host.wordlist_picker = &picker;

        h.on_event(Event::Character('q'), host);
        if (rc.quit_requested.load(std::memory_order_acquire))
            return fail("wordlist-modal: 'q' while open triggered quit");
        if (!picker.open)
            return fail("wordlist-modal: 'q' should not close modal");

        // Enter commits + closes + writes to requested_wordlist_profile.
        h.on_event(Event::Return, host);
        if (picker.open)
            return fail("wordlist-modal: Enter did not close modal");
        std::string committed;
        {
            std::lock_guard<std::mutex> lk(rc.profile_mu);
            committed = rc.requested_wordlist_profile;
        }
        if (committed != "/tmp/beta.txt")
            return fail("wordlist-modal: requested_wordlist_profile mismatch");
    }

    // ================================================================
    // 18. Wordlist picker cancel: Esc closes without writing the
    //     requested profile.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        collider::ui::tui::panels::WordlistPickerState picker;
        picker.candidates.push_back({"/tmp/alpha.txt", "alpha.txt", 0, false});
        picker.open = true;
        picker.selected_index = 0;

        collider::ui::tui::InputHandlerHostContext host;
        host.wordlist_picker = &picker;
        h.on_event(Event::Escape, host);
        if (picker.open)
            return fail("wordlist-cancel: Esc did not close modal");
        std::string committed;
        {
            std::lock_guard<std::mutex> lk(rc.profile_mu);
            committed = rc.requested_wordlist_profile;
        }
        if (!committed.empty())
            return fail("wordlist-cancel: Esc should not write profile");
    }

    // ================================================================
    // 19. Recent hits open: 'l' sets RecentHitsState::open. The opener
    //     also tails the supplied found-empty path; an empty path
    //     gives an empty hits list but should not crash the opener.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        collider::ui::tui::panels::RecentHitsState hits;
        collider::ui::tui::InputHandlerHostContext host;
        host.recent_hits = &hits;
        host.found_empty_path = "";  // intentionally empty
        h.on_event(Event::Character('l'), host);
        if (!hits.open)
            return fail("recent-hits: 'l' did not open modal");
    }

    // ================================================================
    // 20. Recent hits modal swallows non-modal keys; Esc closes.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        collider::ui::tui::panels::RecentHitsState hits;
        hits.open = true;
        // Pre-populate so navigation does not surface an empty-state
        // edge case.
        collider::ui::tui::panels::RecentHit r;
        r.ts_iso = "2026-05-15T12:00:00Z";
        r.passphrase = "abc";
        hits.hits.push_back(r);

        collider::ui::tui::InputHandlerHostContext host;
        host.recent_hits = &hits;

        // 'q' should be swallowed; quit_requested must stay false.
        h.on_event(Event::Character('q'), host);
        if (rc.quit_requested.load(std::memory_order_acquire))
            return fail("recent-hits-modal: 'q' triggered quit");
        if (!hits.open)
            return fail("recent-hits-modal: 'q' closed modal unexpectedly");

        // Esc closes the modal cleanly.
        h.on_event(Event::Escape, host);
        if (hits.open)
            return fail("recent-hits-modal: Esc did not close modal");
    }

    // ================================================================
    // 21. Help overlay open: '?' with a host containing help_overlay
    //     opens the modal. The dispatch path no longer banners
    //     "arrives in Phase 6"; the modal itself IS the surface.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        collider::ui::tui::panels::HelpOverlayState help;
        collider::ui::tui::InputHandlerHostContext host;
        host.help_overlay = &help;
        const bool consumed = h.on_event(Event::Character('?'), host);
        if (!consumed)
            return fail("help: '?' not consumed");
        if (!help.open)
            return fail("help: '?' did not open overlay");
    }

    // ================================================================
    // 22. Help overlay routing: while open, 'q' must NOT trigger quit.
    //     The overlay swallows every key except '?' / Esc / 'q' (which
    //     close the overlay itself, not the app).
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        collider::ui::tui::panels::HelpOverlayState help;
        help.open = true;

        collider::ui::tui::InputHandlerHostContext host;
        host.help_overlay = &help;

        // 'q' while open should close the overlay, NOT trigger quit.
        h.on_event(Event::Character('q'), host);
        if (rc.quit_requested.load(std::memory_order_acquire))
            return fail("help-modal: 'q' while open triggered quit");
        if (help.open)
            return fail("help-modal: 'q' did not close overlay");
    }

    // ================================================================
    // 23. Help overlay close on '?' toggle: pressing '?' a second
    //     time closes the overlay.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        collider::ui::tui::panels::HelpOverlayState help;
        help.open = true;

        collider::ui::tui::InputHandlerHostContext host;
        host.help_overlay = &help;

        h.on_event(Event::Character('?'), host);
        if (help.open)
            return fail("help-toggle: second '?' did not close overlay");
    }

    // ================================================================
    // 24. Help overlay close on Esc.
    // ================================================================
    {
        reset_runtime_state();
        InputHandler h;
        collider::ui::tui::panels::HelpOverlayState help;
        help.open = true;

        collider::ui::tui::InputHandlerHostContext host;
        host.help_overlay = &help;
        h.on_event(Event::Escape, host);
        if (help.open)
            return fail("help-esc: Esc did not close overlay");
    }

    // ================================================================
    // builder-tests Q4 additions: clamp-edge coverage for the batch
    // size step and full coverage of the rule chunk cycle. The existing
    // subtests cover the happy-path adjustments; these guard the
    // boundary cases that the dispatcher special-cases (banner copy
    // changes and the wrap arithmetic in particular).
    // ================================================================

    // ================================================================
    // 25. Batch size '+' clamped at maximum: starting at kBatchSizeMax
    //     (30M), one '+' press must NOT overflow into a higher value;
    //     requested_batch_size lands on the same 30M and the banner
    //     surfaces the "already at maximum" copy. The dispatcher
    //     clamps cur*5/4 to kBatchSizeMax and then detects next == cur.
    // ================================================================
    {
        reset_runtime_state();
        rc.last_applied_batch_size.store(30'000'000ULL,
                                         std::memory_order_release);
        InputHandler h;
        h.on_event(Event::Character('+'));
        const uint64_t req = rc.requested_batch_size.load(
            std::memory_order_acquire);
        if (req != 30'000'000ULL)
            return fail("batch-clamp-up: requested != kBatchSizeMax");
        const std::string banner = rc.get_banner();
        if (banner.find("maximum") == std::string::npos)
            return fail("batch-clamp-up: banner missing 'maximum' copy");
    }

    // ================================================================
    // 26. Batch size '-' clamped at minimum: starting at kBatchSizeMin
    //     (1M), one '-' press must NOT underflow; the dispatcher clamps
    //     cur*3/4 (=750K) up to 1M and surfaces the "already at minimum"
    //     banner.
    // ================================================================
    {
        reset_runtime_state();
        rc.last_applied_batch_size.store(1'000'000ULL,
                                         std::memory_order_release);
        InputHandler h;
        h.on_event(Event::Character('-'));
        const uint64_t req = rc.requested_batch_size.load(
            std::memory_order_acquire);
        if (req != 1'000'000ULL)
            return fail("batch-clamp-down: requested != kBatchSizeMin");
        const std::string banner = rc.get_banner();
        if (banner.find("minimum") == std::string::npos)
            return fail("batch-clamp-down: banner missing 'minimum' copy");
    }

    // ================================================================
    // 27. Rule chunk cycle wrap 1000 -> 200. The Wave 9 test 5 covers
    //     500 -> 1000; this seals the wrap edge so a future refactor
    //     that breaks the back-edge trips here.
    // ================================================================
    {
        reset_runtime_state();
        rc.last_applied_rule_chunk_size.store(1000ULL,
                                              std::memory_order_release);
        InputHandler h;
        h.on_event(Event::Character('r'));
        const uint64_t req = rc.requested_rule_chunk_size.load(
            std::memory_order_acquire);
        if (req != 200ULL)
            return fail("rule-chunk-wrap: expected 200 after 'r' from 1000");
    }

    // ================================================================
    // 28. Rule chunk cycle forward 200 -> 500. Confirms the lowest
    //     canonical value advances into the middle value rather than
    //     skipping to 1000 or falling into the non-canonical branch.
    // ================================================================
    {
        reset_runtime_state();
        rc.last_applied_rule_chunk_size.store(200ULL,
                                              std::memory_order_release);
        InputHandler h;
        h.on_event(Event::Character('r'));
        const uint64_t req = rc.requested_rule_chunk_size.load(
            std::memory_order_acquire);
        if (req != 500ULL)
            return fail("rule-chunk-forward: expected 500 after 'r' from 200");
    }

    // ================================================================
    // 29. Rule chunk cycle non-canonical fallback. If the runtime
    //     state holds a value outside {200, 500, 1000} (e.g. an older
    //     runtime.yml from before the cycle was introduced, or a
    //     hand-edited config), pressing 'r' falls through the if/else
    //     ladder to the documented default (500). This protects the
    //     operator from being stuck on a bad value with no way to
    //     escape via 'r'.
    // ================================================================
    {
        reset_runtime_state();
        rc.last_applied_rule_chunk_size.store(800ULL,
                                              std::memory_order_release);
        InputHandler h;
        h.on_event(Event::Character('r'));
        const uint64_t req = rc.requested_rule_chunk_size.load(
            std::memory_order_acquire);
        if (req != 500ULL)
            return fail("rule-chunk-default: expected 500 after 'r' from 800");
    }

    // ================================================================
    // 25. Tier 2 D1: focused-panel mode dispatch.
    //
    // Plain digits 1..4 set requested_focused_panel to the matching
    // panel index (status=0, gpu=1, performance=2, plugins=3).
    // Digit 0 returns to kFocusNone (multi-panel layout). The chord
    // handler runs first, so g+digit still toggles GPUs unambiguously;
    // outside chord mode the digits are free.
    // ================================================================
    {
        using collider::runtime::RuntimeControlState;
        reset_runtime_state();
        InputHandler h;
        // Pre-condition: no focus requested.
        if (rc.requested_focused_panel.load(std::memory_order_acquire)
                != RuntimeControlState::kFocusNone) {
            return fail("focus: baseline not kFocusNone");
        }

        const bool c1 = h.on_event(Event::Character('1'));
        if (!c1) return fail("focus: '1' not consumed");
        if (rc.requested_focused_panel.load(std::memory_order_acquire)
                != RuntimeControlState::kFocusStatus) {
            return fail("focus: '1' did not set kFocusStatus");
        }

        const bool c2 = h.on_event(Event::Character('2'));
        if (!c2) return fail("focus: '2' not consumed");
        if (rc.requested_focused_panel.load(std::memory_order_acquire)
                != RuntimeControlState::kFocusGpu) {
            return fail("focus: '2' did not set kFocusGpu");
        }

        const bool c3 = h.on_event(Event::Character('3'));
        if (!c3) return fail("focus: '3' not consumed");
        if (rc.requested_focused_panel.load(std::memory_order_acquire)
                != RuntimeControlState::kFocusPerformance) {
            return fail("focus: '3' did not set kFocusPerformance");
        }

        const bool c4 = h.on_event(Event::Character('4'));
        if (!c4) return fail("focus: '4' not consumed");
        if (rc.requested_focused_panel.load(std::memory_order_acquire)
                != RuntimeControlState::kFocusPlugins) {
            return fail("focus: '4' did not set kFocusPlugins");
        }

        // Digit 0 clears focus.
        const bool c0 = h.on_event(Event::Character('0'));
        if (!c0) return fail("focus: '0' not consumed");
        if (rc.requested_focused_panel.load(std::memory_order_acquire)
                != RuntimeControlState::kFocusNone) {
            return fail("focus: '0' did not clear focus");
        }
    }

    // ================================================================
    // 26. Tier 2 D1: Esc clears focus when no modal/chord is active.
    //
    // Focus the perf panel, then press Esc; requested_focused_panel
    // must be kFocusNone after the Esc. The Esc dispatch must not
    // open / close any modal because none is passed in the host.
    // ================================================================
    {
        using collider::runtime::RuntimeControlState;
        reset_runtime_state();
        InputHandler h;
        h.on_event(Event::Character('3'));   // focus perf
        if (rc.requested_focused_panel.load(std::memory_order_acquire)
                != RuntimeControlState::kFocusPerformance) {
            return fail("focus-esc: pre-condition not set");
        }
        const bool consumed = h.on_event(Event::Escape);
        if (!consumed) return fail("focus-esc: Esc not consumed");
        if (rc.requested_focused_panel.load(std::memory_order_acquire)
                != RuntimeControlState::kFocusNone) {
            return fail("focus-esc: Esc did not clear focus");
        }
    }

    // ================================================================
    // 27. Tier 2 D1: g+digit chord still toggles GPUs (focus dispatch
    //     must NOT intercept the digit when the chord is awaiting).
    //
    // Verifies the chord handler precedence: 'g' arms the chord, the
    // next digit completes the chord (toggling that GPU bit), and
    // requested_focused_panel is left alone.
    // ================================================================
    {
        using collider::runtime::RuntimeControlState;
        reset_runtime_state();
        InputHandler h;
        const uint8_t before = rc.gpu_enable_mask.load(
            std::memory_order_acquire);
        h.on_event(Event::Character('g'));
        if (!h.awaiting_chord())
            return fail("focus-chord: 'g' did not arm chord");
        h.on_event(Event::Character('2'));   // chord, not focus
        if (h.awaiting_chord())
            return fail("focus-chord: '2' did not consume chord");
        const uint8_t after = rc.gpu_enable_mask.load(
            std::memory_order_acquire);
        if (after == before)
            return fail("focus-chord: gpu_enable_mask unchanged after g+2");
        // requested_focused_panel must stay None: the digit went to
        // chord dispatch, not focus dispatch.
        if (rc.requested_focused_panel.load(std::memory_order_acquire)
                != RuntimeControlState::kFocusNone) {
            return fail("focus-chord: chord-digit leaked into focus dispatch");
        }
    }

    std::printf("test_input_handler: OK\n");
    return 0;
}
