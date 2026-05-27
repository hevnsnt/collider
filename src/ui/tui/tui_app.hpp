// tui_app.hpp -- FREE STUB (installed by scripts/sync-to-free.sh)
//
// In Pro, TuiApp is the FTXUI dashboard: render thread, snapshot
// queues, panels, settings overlay, keybindings, modal stack. Free
// ships none of that, but pool_solver.cpp / puzzle_solver_kangaroo.cpp
// / puzzle_solver_bsgs.cpp / puzzle_solver_bruteforce.cpp /
// puzzle_solver.cpp call a small handful of TuiApp methods on
// pointers obtained from launch_session(). Pro callers see real
// updates land on the dashboard; free callers' updates are no-ops.
//
// Method set was harvested by:
//   grep -hoE -- "(pool_tui|tui_app)\s*->\s*[a-zA-Z_]+" \
//       src/runtime/pool_solver.cpp src/runtime/puzzle_solver*.cpp
// at sync-script extension time. Add a new method here whenever a
// new caller appears in a free-shipped source file.
#pragma once

#include <chrono>
#include <string>

#include "ui/tui/tui_mode.hpp"
#include "ui/tui/panels/settings_panel.hpp"

namespace collider::ui::tui {

class TuiApp {
public:
    TuiApp() = default;
    ~TuiApp() = default;

    TuiApp(const TuiApp&) = delete;
    TuiApp& operator=(const TuiApp&) = delete;

    // Hot-path methods polled by free runners.
    bool requested_quit() const noexcept { return false; }
    panels::SettingsState* settings_state() noexcept { return nullptr; }

    // Per-mode info setters. In Pro these drive the right-column
    // panels; in free they discard the argument silently.
    void set_pool_info(const PoolInfo& /*info*/) noexcept {}
    void set_challenge_info(const ChallengeInfo& /*info*/) noexcept {}
    void set_bip_scan_info(const BipScanInfo& /*info*/) noexcept {}

    // Status/progress setters.
    void set_current_phase_name(std::string /*name*/) noexcept {}
    void set_chunk_progress(int /*current*/, int /*total*/) noexcept {}
    void set_keys_per_sec_current(double /*v*/) noexcept {}
    void set_keys_per_sec_avg(double /*v*/) noexcept {}
    void set_refresh_hz(int /*hz*/) noexcept {}

    // Other common setters the Pro tree exposes; cheaper to stub
    // them now than re-extend when a new free caller appears.
    void set_mode_label(std::string /*label*/) noexcept {}
    void set_session_start(std::chrono::steady_clock::time_point) noexcept {}
    void set_session_baseline(uint64_t) noexcept {}
    void set_version(std::string) noexcept {}
    void set_wordlist_size(uint64_t) noexcept {}
    void set_warmup_status(std::string) noexcept {}
    void set_sample_passphrase(std::string) noexcept {}
    void set_outer_iteration(int) noexcept {}
    void set_current_wordlist_profile(std::string) noexcept {}
    void set_safety_reserve_bytes(uint64_t) noexcept {}
    void set_tui_mode(TuiMode) noexcept {}

    // Lifecycle.
    void start() noexcept {}
    void stop() noexcept {}
};

}  // namespace collider::ui::tui
