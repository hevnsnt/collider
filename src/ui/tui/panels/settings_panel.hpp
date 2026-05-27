// settings_panel.hpp -- FREE STUB (installed by scripts/sync-to-free.sh)
//
// In Pro this declares the FTXUI settings overlay state + the
// SettingsValues schema persisted to ~/.collider/settings.json by
// settings_sidecar. In free no overlay exists, but pool_solver /
// puzzle_solver_kangaroo reference SettingsValues + SettingsSnapshot
// when polling pool_tui->settings_state(); the stubs let those polls
// compile and just always return "no settings state" (nullptr) so the
// poll branch is dead in free.
#pragma once

#include <cstdint>
#include <string>

namespace collider::ui::tui::panels {

struct SettingsValues {
    std::string backend_kind;      // "cuda" / "metal" / "cpu"
    std::string solver;            // "kangaroo" / "bsgs" / "brute"
    int         num_kangaroos    = 0;
    int         batch_size       = 0;
    int         dp_bits          = 0;
    int         refresh_hz       = 20;
    int         theme            = 0;
    bool        verbose          = false;
};

struct SettingsDirty {
    bool num_kangaroos = false;
    bool batch_size    = false;
    bool dp_bits       = false;
    bool refresh_hz    = false;
    bool theme         = false;
    bool verbose       = false;
    bool backend_kind  = false;
    bool solver        = false;
};

struct SettingsSnapshot {
    SettingsValues values;
    SettingsDirty  dirty;
    bool           restart_requested = false;
};

// In Pro this is the live settings state held by TuiApp. Free
// returns nullptr from settings_state() so callers branch out, and
// this opaque forward-decl is just so the type name resolves where
// it appears in declarations.
struct SettingsState;

inline SettingsSnapshot snapshot_and_clear(SettingsState& /*state*/) {
    return SettingsSnapshot{};
}

}  // namespace collider::ui::tui::panels
