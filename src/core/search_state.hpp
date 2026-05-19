/**
 * Search State Persistence
 *
 * Saves and restores puzzle search progress to allow resuming after interruption.
 * State files are stored in ~/.collider/state/
 *
 * SAFETY FEATURES:
 * - Atomic saves: Write to temp file, then rename (survives Ctrl+C)
 * - Checksum validation: Detects file corruption
 * - Bounds checking: Validates loaded state against puzzle parameters
 */

#pragma once

#include "paths.hpp"

#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <random>

#ifdef _WIN32
#include <io.h>
#define fsync _commit
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace collider {

/**
 * On-disk format version for PuzzleSearchState. Bumped at every
 * structural change to the persisted layout. Loader migrates older
 * versions where possible and rejects newer versions (forward
 * compatibility is a v1.5 concern).
 *
 *   v1 = legacy, no checksum.
 *   v2 = added zone_idx / zone_checked for the Center-Heavy strategy.
 *   v3 = removed Center-Heavy; 128-bit position (position_lo/hi only).
 *   v4 = v1.4.2 R-B8: added position_full[4] so the position field can
 *         hold any uint256 value, removing the silent truncation that
 *         was masked at runtime by the v1.4.2 R-B7 brute-force rejection.
 *         v3 -> v4 migration copies lo/hi into position_full[0..1] and
 *         zeroes [2..3]. Validation now checks all 256 bits against the
 *         [2^(N-1), 2^N - 1] puzzle window.
 */
static constexpr uint32_t kSearchStateVersion = 4;

/**
 * Puzzle search state - for resuming Bitcoin puzzle search.
 *
 * removed the Center-Heavy zone fields (zone_idx, zone_checked)
 * along with the scanning strategy. The on-disk format moved from v2 to v3;
 * old save files fail checksum validation and are ignored (the parser still
 * silently skips zone_idx / zone_checked keys on read for graceful degrade).
 *
 * B8 (v4): added position_full[4] for full UInt256 representation.
 * position_lo / position_hi remain present for backwards compatibility but
 * are now derived from position_full[0] / position_full[1]; writers
 * populate both. Readers prefer position_full when present.
 */
struct PuzzleSearchState {
    int puzzle_number = 0;              // Which puzzle (1-160)
    uint64_t position_lo = 0;           // Current position (low 64 bits) - mirror of position_full[0]
    uint64_t position_hi = 0;           // Current position (high 64 bits) - mirror of position_full[1]
    // v4 (R-B8): full UInt256 position in LE-limb order (parts[0] = low
    // 64 bits, parts[3] = high 64 bits). For puzzles <= 128 bits, parts[2]
    // and parts[3] are always zero and the state is equivalent to v3.
    // For puzzles > 128 bits (future v1.5 multi-limb brute path) the
    // upper limbs hold real bits.
    uint64_t position_full[4] = {0, 0, 0, 0};
    uint64_t total_checked = 0;         // Total keys checked
    std::string timestamp;              // Last save timestamp
    uint32_t checksum = 0;              // FNV-1a-ish checksum for validation

    // v4 (R-B8): version of the file we LOADED from (or kSearchStateVersion
    // if this state was newly constructed). save_puzzle_state always
    // writes kSearchStateVersion regardless of this field's contents.
    // Lets callers detect "I just migrated from v3 to v4" without having
    // to re-read the file.
    uint32_t loaded_version = kSearchStateVersion;

    bool valid = false;                 // Was state loaded successfully?
};

/**
 * State manager for persistent puzzle search state.
 */
class SearchStateManager {
public:
    /**
     * Get state directory path.
     */
    static std::string get_state_dir() {
        return collider::paths::state_dir().string();
    }

    /**
     * Ensure state directory exists.
     */
    static bool ensure_state_dir() {
        std::string dir = get_state_dir();
        if (!std::filesystem::exists(dir)) {
            try {
                std::filesystem::create_directories(dir);
            } catch (const std::exception& e) {
                std::cerr << "[!] Failed to create state directory: " << e.what() << "\n";
                return false;
            }
        }
        return true;
    }

    /**
     * Get current timestamp string.
     */
    static std::string get_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        std::ostringstream ss;
        ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }

    /**
     * Get puzzle state file path.
     */
    static std::string get_puzzle_state_path(int puzzle_number) {
        return get_state_dir() + "/puzzle_" + std::to_string(puzzle_number) + ".state";
    }

    /**
     * Compute simple checksum for state validation.
     * Uses FNV-1a style hash for speed and simplicity.
     *
     * v4 (R-B8): checksum domain extended to cover all four position_full
     * limbs. position_lo / position_hi are no longer mixed separately;
     * they are required to equal position_full[0] / position_full[1] for
     * a state to validate (see validate_state), so adding them to the
     * hash would be redundant.
     */
    static uint32_t compute_checksum(const PuzzleSearchState& state) {
        uint32_t hash = 2166136261u;  // FNV offset basis
        auto mix = [&hash](uint64_t val) {
            for (int i = 0; i < 8; i++) {
                hash ^= static_cast<uint8_t>(val >> (i * 8));
                hash *= 16777619u;  // FNV prime
            }
        };
        mix(state.puzzle_number);
        mix(state.position_full[0]);
        mix(state.position_full[1]);
        mix(state.position_full[2]);
        mix(state.position_full[3]);
        mix(state.total_checked);
        return hash;
    }

    /**
     * v3 checksum domain. Pinned exactly to the pre-v4 implementation so
     * v3 files can be checksum-verified during a v3 -> v4 migration.
     * Used only by load_puzzle_state; not part of any save path.
     */
    static uint32_t compute_checksum_v3(const PuzzleSearchState& state) {
        uint32_t hash = 2166136261u;
        auto mix = [&hash](uint64_t val) {
            for (int i = 0; i < 8; i++) {
                hash ^= static_cast<uint8_t>(val >> (i * 8));
                hash *= 16777619u;
            }
        };
        mix(state.puzzle_number);
        mix(state.position_lo);
        mix(state.position_hi);
        mix(state.total_checked);
        return hash;
    }

    /**
     * Validate state bounds against puzzle parameters.
     * Returns error message or empty string if valid.
     *
     * v4 (R-B8): validation now covers all 256 bits of position_full.
     * For puzzle N, the private key lies in [2^(N-1), 2^N - 1], so any
     * limb whose index exceeds (N-1)/64 must be zero, and the limb
     * containing the high bit must not exceed (1 << ((N-1) % 64 + 1)) - 1
     * (equivalently, the position must fit in N bits). Pre-v4 we only
     * checked up to limb 1 (128 bits), so a future multi-limb brute
     * scan that overflowed into limb 2/3 would silently load without
     * complaint.
     *
     * The position_lo/hi mirror fields must agree with the canonical
     * position_full[0]/[1]. Disagreement indicates a malformed file or
     * an out-of-sync writer; treat as invalid.
     */
    static std::string validate_state(const PuzzleSearchState& state) {
        // Puzzle number bounds (1-160)
        if (state.puzzle_number < 1 || state.puzzle_number > 160) {
            return "Invalid puzzle number: " + std::to_string(state.puzzle_number);
        }

        // Mirror-field consistency. Writers always populate both
        // representations; readers reject mismatches rather than guess.
        if (state.position_lo != state.position_full[0] ||
            state.position_hi != state.position_full[1])
        {
            return "position_lo / position_hi do not mirror position_full";
        }

        // For puzzle N, private key is in range [2^(N-1), 2^N - 1]
        // Validate position_full[*] is within this range across all 256 bits.
        const int bit_length = state.puzzle_number;

        // Determine the highest limb index that may contain non-zero bits.
        // For bit_length B (1 .. 256), valid limbs are 0 .. floor((B-1)/64).
        // Limbs above that index must be exactly zero.
        const int high_limb = (bit_length - 1) / 64;   // 0..3
        const int high_bits = ((bit_length - 1) % 64) + 1;  // 1..64

        for (int i = 3; i > high_limb; --i) {
            if (state.position_full[i] != 0) {
                return "position_full[" + std::to_string(i)
                       + "] non-zero but puzzle is only "
                       + std::to_string(bit_length) + " bits";
            }
        }

        // The high limb must fit in high_bits bits.
        if (high_limb >= 0 && high_limb < 4) {
            if (high_bits < 64) {
                const uint64_t max_high = (1ULL << high_bits) - 1ULL;
                if (state.position_full[high_limb] > max_high) {
                    return "position_full[" + std::to_string(high_limb)
                           + "] exceeds puzzle range";
                }
            }
            // high_bits == 64 case: no upper bound within the limb (all
            // 64 bits are in-range), no extra check needed.
        }

        return "";  // Valid
    }

    /**
     * Save puzzle search state with atomic write.
     *
     * SAFETY: Writes to temp file first, flushes to disk, then atomic rename.
     * If process is killed mid-write, original state file remains intact.
     *
     * v4 (R-B8): writes position_full[0..3] in addition to the legacy
     * position_lo / position_hi mirror. To keep checksums stable across
     * the v3 -> v4 boundary for the same low 128 bits, the mirror is
     * always synced to position_full[0..1] before hashing.
     */
    static bool save_puzzle_state(const PuzzleSearchState& state_in) {
        if (!ensure_state_dir()) return false;

        // Local copy so we can reconcile the mirror fields with
        // position_full before checksumming. This is also what makes
        // save_puzzle_state safe to call with a state whose mirrors got
        // out of sync (e.g. caller updated position_full but forgot to
        // touch position_lo/hi, or wrote v3-style position_lo/hi without
        // touching position_full).
        // Resolution rule: if position_full is entirely zero and the
        // mirror fields are non-zero, the caller is using the v3-style
        // API; promote the mirror values into position_full[0..1] and
        // proceed. Otherwise position_full wins and the mirror fields
        // are derived from it. This makes the v3 puzzle_solver call
        // sites (which set lo/hi only) still work without modification,
        // while the new v4 callers (set position_full or both) get
        // consistent behavior.
        PuzzleSearchState state = state_in;
        const bool full_is_zero = (state.position_full[0] == 0 &&
                                   state.position_full[1] == 0 &&
                                   state.position_full[2] == 0 &&
                                   state.position_full[3] == 0);
        const bool mirror_nonzero = (state.position_lo != 0 ||
                                     state.position_hi != 0);
        if (full_is_zero && mirror_nonzero) {
            state.position_full[0] = state.position_lo;
            state.position_full[1] = state.position_hi;
            // [2..3] stay zero.
        } else {
            state.position_lo = state.position_full[0];
            state.position_hi = state.position_full[1];
        }

        std::string path = get_puzzle_state_path(state.puzzle_number);
        std::string temp_path = path + ".tmp";

        // Compute checksum AFTER mirror reconciliation so any caller-side
        // drift is invisible to load.
        uint32_t checksum = compute_checksum(state);

        // Write to temporary file first
        {
            std::ofstream file(temp_path, std::ios::out | std::ios::trunc);
            if (!file.is_open()) {
                std::cerr << "[!] Failed to create temp state file: " << temp_path << "\n";
                return false;
            }

            // Version banner is part of the contract. The loader uses
            // it to pick a parsing path. Bump kSearchStateVersion when
            // the on-disk schema changes (NOT for cosmetic edits).
            file << "# Collider Puzzle Search State v"
                 << kSearchStateVersion << "\n";
            file << "# Do not modify manually - checksum protected\n";
            file << "state_version=" << kSearchStateVersion << "\n\n";
            file << "puzzle_number=" << state.puzzle_number << "\n";
            // Mirror fields for v3 readers (we never ship a v3 reader
            // anymore, but the format is forward-compatible if one
            // appears in third-party tooling).
            file << "position_lo=" << state.position_lo << "\n";
            file << "position_hi=" << state.position_hi << "\n";
            // v4 canonical fields. Order matches the limb index.
            file << "position_full_0=" << state.position_full[0] << "\n";
            file << "position_full_1=" << state.position_full[1] << "\n";
            file << "position_full_2=" << state.position_full[2] << "\n";
            file << "position_full_3=" << state.position_full[3] << "\n";
            file << "total_checked=" << state.total_checked << "\n";
            file << "timestamp=" << get_timestamp() << "\n";
            file << "checksum=" << checksum << "\n";

            // Flush C++ buffers
            file.flush();

            // Force OS to write to disk (critical for crash safety)
            // Note: This closes the stream properly
        }

        // Sync to disk before rename (extra safety on some filesystems)
#ifndef _WIN32
        {
            FILE* f = fopen(temp_path.c_str(), "r");
            if (f) {
                fsync(fileno(f));
                fclose(f);
            }
        }
#endif

        // Atomic rename: temp -> final
        // On POSIX, rename() is atomic. On Windows, we try rename first.
        try {
            std::filesystem::rename(temp_path, path);
        } catch (const std::exception& e) {
            // On Windows, might fail if target exists - try remove then rename
            std::filesystem::remove(path);
            try {
                std::filesystem::rename(temp_path, path);
            } catch (const std::exception& e2) {
                std::cerr << "[!] Failed to save state file: " << e2.what() << "\n";
                return false;
            }
        }

        return true;
    }

    /**
     * Load puzzle search state with validation.
     *
     * Validates checksum and bounds before returning state.
     * Returns invalid state if file is corrupted or out of bounds.
     *
     * v4 (R-B8): files with state_version >= 4 carry the canonical
     * position_full[0..3] fields. Older files (v3, no state_version key)
     * carry only position_lo / position_hi; on read we copy those into
     * position_full[0..1] and zero [2..3], producing a state that is
     * structurally a valid v4 with 128-bit position semantics. The
     * loaded_version field carries the original on-disk version so the
     * caller can detect a fresh migration (e.g. to log it once).
     *
     * Forward compatibility: state_version > kSearchStateVersion is
     * rejected. A future v5 reader will decide whether v4 files can be
     * migrated forward; v4 cannot.
     */
    static PuzzleSearchState load_puzzle_state(int puzzle_number) {
        PuzzleSearchState state;
        state.puzzle_number = puzzle_number;

        std::string path = get_puzzle_state_path(puzzle_number);
        std::ifstream file(path);
        if (!file.is_open()) {
            return state;  // No saved state
        }

        uint32_t loaded_checksum = 0;
        bool has_checksum = false;
        // Track which on-disk fields actually appeared; needed for the
        // v3 -> v4 migration decision below.
        bool has_state_version  = false;
        bool has_position_full  = false;
        uint32_t parsed_version = 3;  // default: legacy v3 file

        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            auto pos = line.find('=');
            if (pos == std::string::npos) continue;

            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            try {
                if (key == "puzzle_number") {
                    state.puzzle_number = std::stoi(value);
                } else if (key == "state_version") {
                    parsed_version = static_cast<uint32_t>(std::stoul(value));
                    has_state_version = true;
                } else if (key == "zone_idx" || key == "zone_checked") {
                    // legacy v2 fields; silently ignored on read
                    // so old state files don't crash the parser. Such files
                    // still fail the checksum check below.
                    (void)value;
                } else if (key == "position_lo") {
                    state.position_lo = std::stoull(value);
                } else if (key == "position_hi") {
                    state.position_hi = std::stoull(value);
                } else if (key == "position_full_0") {
                    state.position_full[0] = std::stoull(value);
                    has_position_full = true;
                } else if (key == "position_full_1") {
                    state.position_full[1] = std::stoull(value);
                    has_position_full = true;
                } else if (key == "position_full_2") {
                    state.position_full[2] = std::stoull(value);
                    has_position_full = true;
                } else if (key == "position_full_3") {
                    state.position_full[3] = std::stoull(value);
                    has_position_full = true;
                } else if (key == "total_checked") {
                    state.total_checked = std::stoull(value);
                } else if (key == "timestamp") {
                    state.timestamp = value;
                } else if (key == "checksum") {
                    loaded_checksum = static_cast<uint32_t>(std::stoul(value));
                    has_checksum = true;
                }
            } catch (const std::exception& e) {
                std::cerr << "[!] State file parse error: " << e.what() << "\n";
                return state;  // Return invalid state
            }
        }

        // Reject files newer than what this binary understands. The
        // loaded state stays invalid and the caller will treat the file
        // as missing.
        if (parsed_version > kSearchStateVersion) {
            std::cerr << "[!] State file version " << parsed_version
                      << " is newer than this binary's " << kSearchStateVersion
                      << "; refusing to load. Upgrade the binary or move "
                         "the file aside.\n";
            return state;
        }

        // v3 -> v4 migration: file did not carry position_full keys, so
        // derive them from position_lo / position_hi and zero the upper
        // two limbs. Then re-checksum so the caller's "did we migrate?"
        // probe can compare loaded_checksum against the recomputed value.
        if (!has_position_full) {
            state.position_full[0] = state.position_lo;
            state.position_full[1] = state.position_hi;
            state.position_full[2] = 0;
            state.position_full[3] = 0;
        }

        // Reconcile the mirror fields with position_full so validation
        // and any downstream consumer see a self-consistent state. If
        // both kinds of fields were present and disagreed, position_full
        // wins (it is the v4 canonical form).
        if (has_position_full) {
            state.position_lo = state.position_full[0];
            state.position_hi = state.position_full[1];
        }

        state.loaded_version = parsed_version;

        // Verify checksum if present. v3 files used a smaller checksum
        // domain (position_lo / position_hi only); we recompute under
        // that domain when migrating so a clean v3 file is loadable. v4
        // files use the new domain (all four position_full limbs). If a
        // v3 file's checksum verifies and bounds pass, we re-hash under
        // the v4 domain and accept the state, effectively endorsing
        // the migration. Future v5 readers should do the same dance.
        if (has_checksum) {
            const uint32_t expected = (parsed_version >= kSearchStateVersion)
                ? compute_checksum(state)
                : compute_checksum_v3(state);
            if (expected != loaded_checksum) {
                std::cerr << "[!] State file checksum mismatch - file may "
                             "be corrupted\n";
                std::cerr << "    Expected: " << loaded_checksum
                          << ", Got: " << expected
                          << " (parsed version v" << parsed_version
                          << ", binary v" << kSearchStateVersion << ")\n";
                return state;  // Return invalid state
            }
        }
        // has_state_version was used above to pick the parse path; we
        // don't otherwise depend on it. void-cast to silence -Wunused
        // warnings on older toolchains.
        (void)has_state_version;

        // Validate bounds
        std::string error = validate_state(state);
        if (!error.empty()) {
            std::cerr << "[!] State validation failed: " << error << "\n";
            return state;  // Return invalid state
        }

        state.valid = true;
        return state;
    }

    /**
     * Clear puzzle state (after completion).
     */
    static void clear_puzzle_state(int puzzle_number) {
        std::string path = get_puzzle_state_path(puzzle_number);
        std::string temp_path = path + ".tmp";
        // Clean up both files
        std::filesystem::remove(path);
        std::filesystem::remove(temp_path);
    }

    /**
     * Check if state file exists for a puzzle.
     */
    static bool has_saved_state(int puzzle_number) {
        std::string path = get_puzzle_state_path(puzzle_number);
        return std::filesystem::exists(path);
    }
};

}  // namespace collider
