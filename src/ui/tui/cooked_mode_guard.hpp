// cooked_mode_guard.hpp -- FREE STUB (installed by scripts/sync-to-free.sh)
//
// The Pro CookedModeGuard sets the terminal into raw/cbreak +
// alt-screen + cursor-hide for FTXUI's render loop, then restores on
// destruction. In free there is no FTXUI loop and no alt-screen, so
// the guard is a no-op RAII that just holds its Options for ABI
// compatibility with launch_session's return value.
#pragma once

namespace collider::ui::tui {

class CookedModeGuard {
public:
    struct Options {
        bool alt_screen              = false;
        bool hide_cursor             = false;
        bool install_signal_handlers = false;
        bool raw_mode                = false;
    };

    CookedModeGuard() = default;
    explicit CookedModeGuard(const Options& /*opts*/) {}
    ~CookedModeGuard() = default;

    CookedModeGuard(const CookedModeGuard&) = delete;
    CookedModeGuard& operator=(const CookedModeGuard&) = delete;
};

}  // namespace collider::ui::tui
