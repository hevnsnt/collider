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
 * Puzzle search state - for resuming Bitcoin puzzle search.
 */
struct PuzzleSearchState {
    int puzzle_number = 0;              // Which puzzle (1-160)
    size_t zone_idx = 0;                // Current zone index
    uint64_t position_lo = 0;           // Current position (low 64 bits)
    uint64_t position_hi = 0;           // Current position (high 64 bits)
    uint64_t total_checked = 0;         // Total keys checked
    uint64_t zone_checked = 0;          // Keys checked in current zone
    std::string timestamp;              // Last save timestamp
    uint32_t checksum = 0;              // CRC32-like checksum for validation

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
        std::string home;
#ifdef _WIN32
        const char* userprofile = std::getenv("USERPROFILE");
        home = userprofile ? userprofile : ".";
#else
        const char* home_env = std::getenv("HOME");
        home = home_env ? home_env : ".";
#endif
        return home + "/.collider/state";
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
        mix(state.zone_idx);
        mix(state.position_lo);
        mix(state.position_hi);
        mix(state.total_checked);
        mix(state.zone_checked);
        return hash;
    }

    /**
     * Validate state bounds against puzzle parameters.
     * Returns error message or empty string if valid.
     */
    static std::string validate_state(const PuzzleSearchState& state) {
        // Puzzle number bounds (1-160)
        if (state.puzzle_number < 1 || state.puzzle_number > 160) {
            return "Invalid puzzle number: " + std::to_string(state.puzzle_number);
        }

        // For puzzle N, private key is in range [2^(N-1), 2^N - 1]
        // Validate position is within this range
        int bit_length = state.puzzle_number;

        // Check position_hi isn't impossibly large
        // For puzzles <= 64, position_hi should be 0
        if (bit_length <= 64 && state.position_hi != 0) {
            return "Position overflow for puzzle " + std::to_string(bit_length);
        }

        // For puzzles 65-128, position_hi should fit in (bit_length - 64) bits
        if (bit_length > 64 && bit_length <= 128) {
            uint64_t max_hi = (1ULL << (bit_length - 64)) - 1;
            if (state.position_hi > max_hi) {
                return "Position_hi exceeds puzzle range";
            }
        }

        // Zone index sanity check (max ~100 zones typically)
        if (state.zone_idx > 1000) {
            return "Zone index suspiciously large: " + std::to_string(state.zone_idx);
        }

        return "";  // Valid
    }

    /**
     * Save puzzle search state with atomic write.
     *
     * SAFETY: Writes to temp file first, flushes to disk, then atomic rename.
     * If process is killed mid-write, original state file remains intact.
     */
    static bool save_puzzle_state(const PuzzleSearchState& state) {
        if (!ensure_state_dir()) return false;

        std::string path = get_puzzle_state_path(state.puzzle_number);
        std::string temp_path = path + ".tmp";

        // Compute checksum before save
        uint32_t checksum = compute_checksum(state);

        // Write to temporary file first
        {
            std::ofstream file(temp_path, std::ios::out | std::ios::trunc);
            if (!file.is_open()) {
                std::cerr << "[!] Failed to create temp state file: " << temp_path << "\n";
                return false;
            }

            file << "# Collider Puzzle Search State v2\n";
            file << "# Do not modify manually - checksum protected\n\n";
            file << "puzzle_number=" << state.puzzle_number << "\n";
            file << "zone_idx=" << state.zone_idx << "\n";
            file << "position_lo=" << state.position_lo << "\n";
            file << "position_hi=" << state.position_hi << "\n";
            file << "total_checked=" << state.total_checked << "\n";
            file << "zone_checked=" << state.zone_checked << "\n";
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
                } else if (key == "zone_idx") {
                    state.zone_idx = std::stoull(value);
                } else if (key == "position_lo") {
                    state.position_lo = std::stoull(value);
                } else if (key == "position_hi") {
                    state.position_hi = std::stoull(value);
                } else if (key == "total_checked") {
                    state.total_checked = std::stoull(value);
                } else if (key == "zone_checked") {
                    state.zone_checked = std::stoull(value);
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

        // Verify checksum if present (backwards compatible with v1 files)
        if (has_checksum) {
            uint32_t computed = compute_checksum(state);
            if (computed != loaded_checksum) {
                std::cerr << "[!] State file checksum mismatch - file may be corrupted\n";
                std::cerr << "    Expected: " << loaded_checksum << ", Got: " << computed << "\n";
                return state;  // Return invalid state
            }
        }

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
