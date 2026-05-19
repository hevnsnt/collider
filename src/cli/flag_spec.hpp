/**
 * flag_spec.hpp - Flag registry primitives for the CLI parser.
 *
 * The parser is a flat table walk: for each argv token, look up the matching
 * FlagSpec by long_name or short_name, then call its apply() callback. This
 * replaces the 600-line if/else chain that lived inline in cli_parser.cpp.
 *
 * Why a callback per flag instead of a Kind enum + branchless dispatch:
 *   - A handful of flags have non-uniform semantics (variadic --brute, optional
 *     value --puzzle, file-reading --pool-password-file, side-effect
 *     --brainwallet that clears pool_mode + pool_url, Pro-gated --bloom and
 *     friends that emit a one-time hint in the Free build).
 *   - Pure Kind enums force every flag into one of a handful of shapes, and
 *     the special cases above each need an escape hatch. A function-pointer
 *     callback keeps each flag's behavior local to its registry entry without
 *     a tangle of fallback branches.
 *
 * The callback contract:
 *   - i is the current argv index (already pointing at the matched flag).
 *     apply() advances i to the LAST argv slot it consumed (so the parser's
 *     outer loop's ++i moves past it).
 *   - Return 0 on success, negative on hard error (with err_msg populated).
 *     Positive returns are reserved for Free-build early-exit codes (currently
 *     only --puzzle-only-v2 in the Free path uses exit code 2).
 */
#pragma once

#include <string>

#include "cli/cli_parser.hpp"     // ::Arguments (global)
#include "core/yaml_config.hpp"   // collider::CLIFlags

namespace collider::cli {

// Arguments lives at global scope (it predates the collider namespace);
// CLIFlags lives in namespace collider. The apply callback signature uses
// the fully-qualified names so the registry doesn't get tangled in any
// using-declarations.
using ApplyFn = int (*)(int& i, int argc, char* argv[],
                        ::Arguments& args, ::collider::CLIFlags& cli,
                        std::string& err_msg);

struct FlagSpec {
    const char* long_name;   // e.g. "--puzzle"; must start with "--"
    const char* short_name;  // e.g. "-P"; nullptr or "" when none
    ApplyFn apply;
    const char* help_group;  // used for grouping in --help; nullptr = hide
};

}  // namespace collider::cli
