// settings_sidecar.hpp -- FREE STUB (installed by scripts/sync-to-free.sh)
//
// In Pro this persists SettingsValues to ~/.collider/settings.json
// so the operator's last-session TUI tweaks (theme, refresh_hz, etc.)
// survive across launches. Free has no settings UI to persist FROM,
// but pool_solver / puzzle_solver_kangaroo call save() after polling
// snapshot_and_clear() -- which in free returns an empty snapshot
// with no dirty bits, so the call is dead in the free build but the
// symbol still needs to resolve.
#pragma once

#include "ui/tui/panels/settings_panel.hpp"

namespace collider::settings_sidecar {

inline bool save(const ::collider::ui::tui::panels::SettingsValues& /*v*/) {
    return true;  // no-op success
}

inline bool load(::collider::ui::tui::panels::SettingsValues& /*v*/) {
    return false;  // "no saved settings"; caller skips the YAML overlay
}

}  // namespace collider::settings_sidecar
