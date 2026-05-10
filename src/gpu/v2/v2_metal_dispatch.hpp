/**
 * Brain Wallet v2 -- Metal dispatch entry point (host header).
 *
 * Implementation in v2_metal_dispatch.mm (Objective-C++). Compiled only
 * on macOS Pro builds; the orchestrator dispatches to this from its
 * `__APPLE__` branch.
 */

#pragma once

#include "brain_wallet_v2.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace collider {
namespace gpu {
namespace v2 {
namespace metal {

/**
 * Run the v2 puzzle-only multi-scheme kernel against a passphrase batch
 * and a list of puzzle targets.
 *
 * Inputs are host-side (Metal uses unified memory; we copy through
 * MTLBuffer for clarity rather than zero-copy maps).
 *
 * @param targets             puzzle target list (already built via
 *                            make_puzzle_target)
 * @param passphrases_packed  packed passphrase bytes
 * @param offsets             per-passphrase byte offset into packed
 * @param lengths             per-passphrase byte length
 * @param scheme_mask         bitmask of DerivationScheme values to dispatch
 * @param matches_out         appended with V2MatchRecord per hit (cleared first)
 * @param error_out           populated on failure with a human-readable msg
 * @return true on success, false on any error
 */
bool v2_metal_run_puzzle_only(
    const std::vector<PuzzleTarget>& targets,
    const std::vector<uint8_t>& passphrases_packed,
    const std::vector<uint32_t>& offsets,
    const std::vector<uint32_t>& lengths,
    uint32_t scheme_mask,
    std::vector<V2MatchRecord>& matches_out,
    std::string& error_out);

}  // namespace metal
}  // namespace v2
}  // namespace gpu
}  // namespace collider
