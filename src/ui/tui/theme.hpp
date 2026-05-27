// theme.hpp -- FREE STUB (installed by scripts/sync-to-free.sh)
//
// Carries ThemeVariant + RenderConfig + detect_default_variant. The
// Pro version drives FTXUI colors; free uses none of it but pool /
// puzzle callers reference these types when filling out a LaunchConfig
// to pass to launch_session (which is itself a stub no-op in free).
#pragma once

#include <chrono>
#include <cstdint>

namespace collider::ui::tui {

enum class ThemeVariant : int {
    Default        = 0,
    Light          = 1,
    SolarizedDark  = 2,
    Monochrome     = 3,
};

inline ThemeVariant detect_default_variant() noexcept {
    return ThemeVariant::Default;
}

struct RenderConfig {
    ThemeVariant theme       = ThemeVariant::Default;
    int          refresh_hz  = 20;
    bool         alt_screen  = true;
};

}  // namespace collider::ui::tui
