/**
 * interactive_ui.hpp - Interactive top-level menu / setup wizard glue
 * for theCollider.
 *
 * Extracted out of src/main.cpp during the v1.4.1 A.3 refactor. Hosts the
 * argv-less startup flow (banner + main menu + per-mode submenus) and the
 * Windows ANSI enabler. The actual mode runners (puzzle / pool / brain
 * wallet) live in src/runtime/.
 *
 * `run_interactive_mode` mutates a base Arguments and returns the user's
 * selection; main() then dispatches to the matching mode runner. The
 * brain wallet submenu is gated on COLLIDER_PRO inside the .cpp; this
 * header is identical in Free + Pro builds.
 */
#pragma once

#include "cli/cli_parser.hpp"  // Arguments

namespace collider::ui {

/**
 * Enable ANSI escape codes on Windows console.
 * Required for colored output to work properly. No-op on non-Windows.
 */
void enable_windows_ansi();

/**
 * Run puzzle-mode interactive submenu (standalone vs pool, puzzle picker).
 * Pure UI; returns an Arguments configured for the chosen flow.
 */
Arguments run_puzzle_interactive(Arguments base_args, double gpu_speed_mkeys);

#ifdef COLLIDER_PRO
/**
 * Run brain wallet interactive submenu (wordlist setup, bloom filter
 * detection / build, resume prompt). PRO-only.
 */
Arguments run_brainwallet_interactive(Arguments base_args);
#endif

/**
 * Run interactive mode - top-level main menu loop. Calls into the per-
 * mode submenus above. Returns when the user picks a mode (or sets the
 * exit_program / help flag in Arguments).
 */
Arguments run_interactive_mode(Arguments base_args, double gpu_speed_mkeys);

}  // namespace collider::ui
