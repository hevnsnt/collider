/**
 * Brainwallet Search State Persistence
 *
 * Saves and restores brain wallet search progress to allow resuming after interruption.
 * State files are stored in ~/.collider/state/
 *
 * SAFETY FEATURES:
 * - Atomic saves: Write to temp file, then rename (survives Ctrl+C)
 * - Checksum validation: Detects file corruption
 * - Wordlist hash: Invalidates state if wordlist changes
 * - Bounds checking: Validates loaded state parameters
 *
 * STATE TRACKING:
 * - Current word index in wordlist
 * - Current rule index in rule set
 * - Current phase in attack sequence
 * - Total passphrases checked
 * - Global dedup hash (optional checkpoint)
 */

#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdint>
#include <iomanip>
#include <functional>

#ifdef _WIN32
#include <io.h>
#define fsync _commit
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace collider {

/**
 * Brain wallet search state - for resuming passphrase scanning.
 */
struct BrainWalletSearchState {
    // Position tracking
    size_t current_word_idx = 0;          // Current word index in wordlist
    size_t current_rule_idx = 0;          // Current rule index in rule set
    size_t current_phase = 0;             // Current attack phase (0-based)
    size_t phase_iteration = 0;           // Iteration count for cyclic phases

    // Progress tracking
    uint64_t total_checked = 0;           // Total passphrases checked
    uint64_t unique_tested = 0;           // Unique passphrases after dedup
    uint64_t hits_found = 0;              // Matches found

    // Wordlist integrity
    uint64_t wordlist_hash = 0;           // Hash of wordlist for change detection
    size_t wordlist_size = 0;             // Number of words in wordlist
    std::string wordlist_path;            // Path to wordlist file

    // Rule set info
    size_t rules_count = 0;               // Number of rules loaded
    std::string rules_path;               // Path to rule file (if any)

    // Metadata
    std::string timestamp;                // Last save timestamp
    std::string session_id;               // Unique session identifier
    uint32_t checksum = 0;                // FNV-1a checksum for validation

    bool valid = false;                   // Was state loaded successfully?
};

/**
 * State manager for persistent brain wallet search state.
 */
class BrainWalletStateManager {
public:
    // Save interval configuration
    static constexpr size_t DEFAULT_SAVE_INTERVAL = 1000000;  // Save every 1M passphrases
    static constexpr size_t MIN_SAVE_INTERVAL = 10000;        // Minimum 10K
    static constexpr size_t MAX_SAVE_INTERVAL = 100000000;    // Maximum 100M

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
     * Generate session ID based on timestamp and random component.
     */
    static std::string generate_session_id() {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        std::ostringstream ss;
        ss << "bw_" << std::hex << ms;
        return ss.str();
    }

    /**
     * Get brain wallet state file path.
     * Uses a fixed name since we only track one brain wallet session at a time.
     */
    static std::string get_state_path() {
        return get_state_dir() + "/brainwallet.state";
    }

    /**
     * Compute FNV-1a hash of a string for wordlist change detection.
     */
    static uint64_t compute_wordlist_hash(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return 0;

        uint64_t hash = 14695981039346656037ULL;  // FNV offset basis
        char buffer[8192];

        while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
            for (std::streamsize i = 0; i < file.gcount(); i++) {
                hash ^= static_cast<uint8_t>(buffer[i]);
                hash *= 1099511628211ULL;  // FNV prime
            }
        }

        return hash;
    }

    /**
     * Compute checksum for state validation.
     * Uses FNV-1a style hash for speed and simplicity.
     */
    static uint32_t compute_checksum(const BrainWalletSearchState& state) {
        uint32_t hash = 2166136261u;  // FNV offset basis

        auto mix = [&hash](uint64_t val) {
            for (int i = 0; i < 8; i++) {
                hash ^= static_cast<uint8_t>(val >> (i * 8));
                hash *= 16777619u;  // FNV prime
            }
        };

        mix(state.current_word_idx);
        mix(state.current_rule_idx);
        mix(state.current_phase);
        mix(state.phase_iteration);
        mix(state.total_checked);
        mix(state.unique_tested);
        mix(state.hits_found);
        mix(state.wordlist_hash);
        mix(state.wordlist_size);
        mix(state.rules_count);

        return hash;
    }

    /**
     * Validate state bounds and integrity.
     * Returns error message or empty string if valid.
     */
    static std::string validate_state(const BrainWalletSearchState& state) {
        // Basic sanity checks
        if (state.current_word_idx > state.wordlist_size && state.wordlist_size > 0) {
            return "Word index exceeds wordlist size";
        }

        if (state.current_rule_idx > state.rules_count && state.rules_count > 0) {
            return "Rule index exceeds rules count";
        }

        if (state.current_phase > 100) {  // Sanity limit
            return "Phase number suspiciously large: " + std::to_string(state.current_phase);
        }

        if (state.phase_iteration > 10000) {  // Sanity limit
            return "Phase iteration suspiciously large: " + std::to_string(state.phase_iteration);
        }

        return "";  // Valid
    }

    /**
     * Save brain wallet search state with atomic write.
     *
     * SAFETY: Writes to temp file first, flushes to disk, then atomic rename.
     * If process is killed mid-write, original state file remains intact.
     */
    static bool save_state(const BrainWalletSearchState& state) {
        if (!ensure_state_dir()) return false;

        std::string path = get_state_path();
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

            file << "# Collider Brain Wallet Search State v1\n";
            file << "# Do not modify manually - checksum protected\n\n";

            // Position tracking
            file << "current_word_idx=" << state.current_word_idx << "\n";
            file << "current_rule_idx=" << state.current_rule_idx << "\n";
            file << "current_phase=" << state.current_phase << "\n";
            file << "phase_iteration=" << state.phase_iteration << "\n";

            // Progress tracking
            file << "total_checked=" << state.total_checked << "\n";
            file << "unique_tested=" << state.unique_tested << "\n";
            file << "hits_found=" << state.hits_found << "\n";

            // Wordlist info
            file << "wordlist_hash=" << state.wordlist_hash << "\n";
            file << "wordlist_size=" << state.wordlist_size << "\n";
            file << "wordlist_path=" << state.wordlist_path << "\n";

            // Rule set info
            file << "rules_count=" << state.rules_count << "\n";
            file << "rules_path=" << state.rules_path << "\n";

            // Metadata
            file << "session_id=" << state.session_id << "\n";
            file << "timestamp=" << get_timestamp() << "\n";
            file << "checksum=" << checksum << "\n";

            // Flush C++ buffers
            file.flush();
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
        // Windows often holds file handles briefly after close - retry with backoff
        int max_retries = 5;
        int retry_delay_ms = 10;  // Start with 10ms

        for (int attempt = 0; attempt < max_retries; attempt++) {
            try {
                // Try direct rename first
                std::filesystem::rename(temp_path, path);
                return true;  // Success!
            } catch (const std::exception& e) {
                // On Windows, might fail if target exists or handle still held
                try {
                    std::error_code ec;
                    std::filesystem::remove(path, ec);  // Ignore errors, may not exist

                    // Small delay to let Windows release handles (antivirus, indexer, etc.)
                    if (attempt < max_retries - 1) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
                        retry_delay_ms *= 2;  // Exponential backoff
                    }

                    std::filesystem::rename(temp_path, path);
                    return true;  // Success on retry!
                } catch (const std::exception& e2) {
                    if (attempt == max_retries - 1) {
                        std::cerr << "[!] Failed to save state file: " << e2.what() << "\n";
                        // Leave the .tmp file in place - can manually rename later
                        return false;
                    }
                    // Continue to next retry
                }
            }
        }

        return false;
    }

    /**
     * Load brain wallet search state with validation.
     *
     * Validates checksum and bounds before returning state.
     * Returns invalid state if file is corrupted or out of bounds.
     */
    static BrainWalletSearchState load_state() {
        BrainWalletSearchState state;

        std::string path = get_state_path();
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
                if (key == "current_word_idx") {
                    state.current_word_idx = std::stoull(value);
                } else if (key == "current_rule_idx") {
                    state.current_rule_idx = std::stoull(value);
                } else if (key == "current_phase") {
                    state.current_phase = std::stoull(value);
                } else if (key == "phase_iteration") {
                    state.phase_iteration = std::stoull(value);
                } else if (key == "total_checked") {
                    state.total_checked = std::stoull(value);
                } else if (key == "unique_tested") {
                    state.unique_tested = std::stoull(value);
                } else if (key == "hits_found") {
                    state.hits_found = std::stoull(value);
                } else if (key == "wordlist_hash") {
                    state.wordlist_hash = std::stoull(value);
                } else if (key == "wordlist_size") {
                    state.wordlist_size = std::stoull(value);
                } else if (key == "wordlist_path") {
                    state.wordlist_path = value;
                } else if (key == "rules_count") {
                    state.rules_count = std::stoull(value);
                } else if (key == "rules_path") {
                    state.rules_path = value;
                } else if (key == "session_id") {
                    state.session_id = value;
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

        // Verify checksum if present
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
     * Verify that current wordlist matches saved state.
     * Returns true if wordlist hash matches, false if changed.
     */
    static bool verify_wordlist(const BrainWalletSearchState& state, const std::string& wordlist_path) {
        if (state.wordlist_path != wordlist_path) {
            std::cerr << "[!] Wordlist path changed from: " << state.wordlist_path << "\n";
            std::cerr << "    To: " << wordlist_path << "\n";
            return false;
        }

        uint64_t current_hash = compute_wordlist_hash(wordlist_path);
        if (current_hash != state.wordlist_hash) {
            std::cerr << "[!] Wordlist content has changed since last save\n";
            std::cerr << "    Previous hash: " << std::hex << state.wordlist_hash << "\n";
            std::cerr << "    Current hash:  " << std::hex << current_hash << std::dec << "\n";
            return false;
        }

        return true;
    }

    /**
     * Clear brain wallet state (after completion or manual reset).
     */
    static void clear_state() {
        std::string path = get_state_path();
        std::string temp_path = path + ".tmp";
        // Clean up both files
        std::filesystem::remove(path);
        std::filesystem::remove(temp_path);
    }

    /**
     * Check if state file exists.
     */
    static bool has_saved_state() {
        std::string path = get_state_path();
        return std::filesystem::exists(path);
    }

    /**
     * Print state summary for user.
     */
    static void print_state_summary(const BrainWalletSearchState& state) {
        std::cout << "\n[*] Brain Wallet Search State:\n";
        std::cout << "    Session ID: " << state.session_id << "\n";
        std::cout << "    Last saved: " << state.timestamp << "\n";
        std::cout << "    Wordlist: " << state.wordlist_path << " (" << state.wordlist_size << " words)\n";
        std::cout << "    Progress:\n";
        std::cout << "      - Word index: " << state.current_word_idx << " / " << state.wordlist_size << "\n";
        std::cout << "      - Rule index: " << state.current_rule_idx << " / " << state.rules_count << "\n";
        std::cout << "      - Phase: " << state.current_phase << " (iteration " << state.phase_iteration << ")\n";
        std::cout << "      - Total checked: " << state.total_checked << "\n";
        std::cout << "      - Unique tested: " << state.unique_tested << "\n";
        std::cout << "      - Hits found: " << state.hits_found << "\n";
        std::cout << "\n";
    }
};

}  // namespace collider
