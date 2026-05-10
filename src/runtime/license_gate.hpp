/**
 * license_gate.hpp - Pre-dispatch license activation + startup validation.
 *
 * Extracted out of src/main.cpp during the v1.4.1 A.3 (6/6) refactor.
 * The whole TU is COLLIDER_PRO-only; the free build never sees any
 * license-related symbols (the underlying license/license_check.hpp is
 * Pro-only too).
 *
 * Two entry points:
 *   - process_activate_flag(argc, argv, exit_code)
 *       Inspects argv for `--activate KEY`. If present, validates the
 *       key and persists it, then returns true with exit_code set so
 *       main() returns immediately. If --activate isn't there, returns
 *       false (no-op) and main() continues.
 *
 *   - validate_startup_license(exit_code)
 *       Reads the persisted license_key file and validates it. On
 *       failure sets exit_code=1 and returns false (main bails); on
 *       success returns true.
 */
#pragma once

#ifdef COLLIDER_PRO

namespace collider::runtime {

/**
 * Handle `--activate KEY` from argv. If matched, returns true with the
 * process exit code in `out_exit_code` so main() can return immediately.
 * If argv does not start with --activate, returns false; caller should
 * continue past the activation flow.
 */
bool process_activate_flag(int argc, char* argv[], int& out_exit_code);

/**
 * Read the persisted license key and validate it (cache-backed, 24h).
 * Returns true on success. On failure, prints an error to stderr and
 * returns false with `out_exit_code` set to 1.
 */
bool validate_startup_license(int& out_exit_code);

}  // namespace collider::runtime

#endif  // COLLIDER_PRO
