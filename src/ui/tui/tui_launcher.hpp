// tui_launcher.hpp -- FREE STUB (installed by scripts/sync-to-free.sh)
//
// In Pro, launch_session() boots the FTXUI render loop, builds the
// CookedModeGuard, and returns the active TuiApp handle. In free
// there is no render loop -- but pool_solver / puzzle_solver* still
// construct a LaunchConfig, call launch_session(cfg), and chain
// session.app->set_xxx(...) updates throughout the scan. We honor
// that contract here:
//   - LaunchConfig is a struct with every field the callers set.
//   - launch_session() returns a LaunchedSession whose `app` is a
//     valid-but-no-op TuiApp pointer (NOT null, so foo->bar() does
//     not segfault), and whose `cooked_guard` is a default-
//     constructed guard.
//   - The free build pays one TuiApp + CookedModeGuard heap alloc
//     per scan, then every method call against them is a no-op.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ui/tui/cooked_mode_guard.hpp"
#include "ui/tui/theme.hpp"
#include "ui/tui/tui_app.hpp"
#include "ui/tui/tui_mode.hpp"

namespace collider::ui::tui {

struct LaunchConfig {
    std::string                            mode_label;
    std::string                            version;
    std::chrono::steady_clock::time_point  session_start{};
    uint64_t                               session_baseline       = 0;
    uint64_t                               wordlist_size          = 0;
    uint64_t                               safety_reserve_bytes   = 0;
    std::string                            initial_phase_name;
    int                                    initial_current_chunk  = 0;
    int                                    initial_total_chunks   = 1;
    TuiMode                                tui_mode               = TuiMode::Brainwallet;
    std::vector<int>                       gpu_ids;
    RenderConfig                           render_cfg{};
    CookedModeGuard::Options               guard_opts{};
};

struct LaunchedSession {
    std::unique_ptr<TuiApp>          app;
    std::unique_ptr<CookedModeGuard> cooked_guard;
};

inline LaunchedSession launch_session(const LaunchConfig& /*cfg*/) {
    LaunchedSession s;
    s.app          = std::make_unique<TuiApp>();
    s.cooked_guard = std::make_unique<CookedModeGuard>();
    return s;
}

}  // namespace collider::ui::tui
