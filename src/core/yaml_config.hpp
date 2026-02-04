/**
 * yaml_config.hpp - Simple YAML configuration loader for theCollider
 *
 * Parses a subset of YAML (key: value pairs with sections) without external dependencies.
 * Command-line arguments override config file settings.
 */

#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <filesystem>
#include <iostream>
#include <algorithm>
#include "edition.hpp"

namespace collider {

/**
 * Application configuration loaded from config.yml
 */
struct AppConfig {
    // Pool configuration
    std::string pool_url;
    std::string pool_worker;
    std::string pool_password;
    std::string pool_api_key;

    // Puzzle configuration
    int puzzle_number = 0;
    bool smart_select = true;
    int min_bits = 0;
    int max_bits = 160;
    bool kangaroo = true;
    int dp_bits = -1;
    bool random_search = true;
    bool auto_next = false;
    std::string checkpoint;

    // Brainwallet configuration
    bool brainwallet_enabled = false;
    std::string wordlist;
    size_t save_interval = 1000000;
    bool resume = false;

    // Bloom filter
    std::string bloom_file;

    // GPU configuration
    std::vector<int> gpu_devices;
    size_t batch_size = 0;
    bool force_calibrate = false;

    // Settings
    bool verbose = false;
    bool debug = false;
    int benchmark_seconds = 30;

    // Paths
    std::string data_dir = "./processed";
    std::string checkpoint_dir = "./checkpoints";
    std::string log_dir = "./logs";

    /**
     * Get possible config file paths (in order of priority)
     */
    static std::vector<std::string> get_config_paths() {
        std::vector<std::string> paths;

        // 1. Current directory
        paths.push_back("./config.yml");
        paths.push_back("./config.yaml");

        // 2. User home directory
        std::string home;
#ifdef _WIN32
        const char* userprofile = std::getenv("USERPROFILE");
        home = userprofile ? userprofile : "";
#else
        const char* home_env = std::getenv("HOME");
        home = home_env ? home_env : "";
#endif
        if (!home.empty()) {
            paths.push_back(home + "/.collider/config.yml");
            paths.push_back(home + "/.collider/config.yaml");
        }

        return paths;
    }

    /**
     * Load configuration from YAML file.
     * Returns true if a config file was found and loaded.
     */
    bool load(const std::string& explicit_path = "") {
        std::string config_path;

        // Use explicit path if provided
        if (!explicit_path.empty()) {
            if (std::filesystem::exists(explicit_path)) {
                config_path = explicit_path;
            } else {
                std::cerr << "[!] Config file not found: " << explicit_path << "\n";
                return false;
            }
        } else {
            // Search default paths
            for (const auto& path : get_config_paths()) {
                if (std::filesystem::exists(path)) {
                    config_path = path;
                    break;
                }
            }
        }

        if (config_path.empty()) {
            return false;  // No config file found (this is OK)
        }

        std::ifstream file(config_path);
        if (!file.is_open()) {
            std::cerr << "[!] Failed to open config file: " << config_path << "\n";
            return false;
        }

        std::cout << "[*] Loading config from: " << config_path << "\n";

        std::string line;
        std::string current_section;
        int line_number = 0;

        while (std::getline(file, line)) {
            line_number++;

            // Trim leading whitespace and count indent
            size_t indent = 0;
            while (indent < line.length() && (line[indent] == ' ' || line[indent] == '\t')) {
                indent++;
            }
            std::string trimmed = line.substr(indent);

            // Skip empty lines and comments
            if (trimmed.empty() || trimmed[0] == '#' || trimmed.substr(0, 3) == "---") {
                continue;
            }

            // Remove trailing comments
            size_t comment_pos = trimmed.find('#');
            if (comment_pos != std::string::npos) {
                trimmed = trimmed.substr(0, comment_pos);
            }

            // Trim trailing whitespace
            while (!trimmed.empty() && (trimmed.back() == ' ' || trimmed.back() == '\t')) {
                trimmed.pop_back();
            }

            if (trimmed.empty()) continue;

            // Parse key: value
            size_t colon_pos = trimmed.find(':');
            if (colon_pos == std::string::npos) continue;

            std::string key = trimmed.substr(0, colon_pos);
            std::string value = (colon_pos + 1 < trimmed.length()) ? trimmed.substr(colon_pos + 1) : "";

            // Trim key and value
            while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
            while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);
            while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.pop_back();

            // Section header (no value, or just whitespace after colon)
            if (value.empty() && indent == 0) {
                current_section = key;
                continue;
            }

            // Remove quotes from string values
            if (value.length() >= 2 &&
                ((value.front() == '"' && value.back() == '"') ||
                 (value.front() == '\'' && value.back() == '\''))) {
                value = value.substr(1, value.length() - 2);
            }

            // Parse value based on section and key
            try {
                parse_value(current_section, key, value);
            } catch (const std::exception& e) {
                std::cerr << "[!] Config parse error at line " << line_number << ": " << e.what() << "\n";
            }
        }

        return true;
    }

private:
    void parse_value(const std::string& section, const std::string& key, const std::string& value) {
        if (section == "pool") {
#if COLLIDER_HAS_CUSTOM_POOL
            if (key == "url") pool_url = value;
            else if (key == "password") pool_password = value;
            else if (key == "api_key") pool_api_key = value;
#endif
            // Worker is available in both editions
            if (key == "worker") pool_worker = value;
#if !COLLIDER_HAS_CUSTOM_POOL
            // Free edition: override pool URL to hardcoded value
            else if (key == "url") {
                std::cout << "[*] Config: Ignoring pool.url (Free edition uses hardcoded pool)\n";
                pool_url = COLLIDER_FREE_POOL_URL;
            }
#endif
        }
        else if (section == "puzzle") {
#if COLLIDER_HAS_SOLO
            if (key == "number") puzzle_number = std::stoi(value);
            else if (key == "smart_select") smart_select = parse_bool(value);
            else if (key == "min_bits") min_bits = std::stoi(value);
            else if (key == "max_bits") max_bits = std::stoi(value);
            else if (key == "kangaroo") kangaroo = parse_bool(value);
            else if (key == "dp_bits") dp_bits = std::stoi(value);
            else if (key == "random_search") random_search = parse_bool(value);
            else if (key == "auto_next") auto_next = parse_bool(value);
            else if (key == "checkpoint") checkpoint = value;
#else
            // Free edition: ignore puzzle configuration
            std::cout << "[*] Solo puzzle solver requires collider pro — collisionprotocol.com/pro\n";
#endif
        }
        else if (section == "brainwallet") {
#if COLLIDER_HAS_BRAINWALLET
            if (key == "enabled") brainwallet_enabled = parse_bool(value);
            else if (key == "wordlist") wordlist = value;
            else if (key == "save_interval") save_interval = std::stoull(value);
            else if (key == "resume") resume = parse_bool(value);
#else
            // Free edition: ignore brainwallet configuration  
            std::cout << "[*] Brain wallet requires collider pro — collisionprotocol.com/pro\n";
#endif
        }
        else if (section == "bloom") {
#if COLLIDER_HAS_BLOOM
            if (key == "file") bloom_file = value;
#else
            // Free edition: ignore bloom configuration
            std::cout << "[*] Bloom filters require collider pro — collisionprotocol.com/pro\n";
#endif
        }
        else if (section == "gpu") {
            if (key == "devices") gpu_devices = parse_int_list(value);
            else if (key == "batch_size") batch_size = std::stoull(value);
            else if (key == "force_calibrate") force_calibrate = parse_bool(value);
        }
        else if (section == "settings") {
            if (key == "verbose") verbose = parse_bool(value);
            else if (key == "debug") debug = parse_bool(value);
            else if (key == "benchmark_seconds") benchmark_seconds = std::stoi(value);
        }
        else if (section == "paths") {
            if (key == "data_dir") data_dir = value;
            else if (key == "checkpoint_dir") checkpoint_dir = value;
            else if (key == "log_dir") log_dir = value;
        }
    }

    static bool parse_bool(const std::string& value) {
        std::string lower = value;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return (lower == "true" || lower == "yes" || lower == "1" || lower == "on");
    }

    static std::vector<int> parse_int_list(const std::string& value) {
        std::vector<int> result;

        // Handle empty or []
        if (value.empty() || value == "[]") return result;

        // Remove brackets
        std::string clean = value;
        if (clean.front() == '[') clean.erase(0, 1);
        if (!clean.empty() && clean.back() == ']') clean.pop_back();

        // Split by comma
        std::stringstream ss(clean);
        std::string item;
        while (std::getline(ss, item, ',')) {
            // Trim whitespace
            while (!item.empty() && (item.front() == ' ' || item.front() == '\t')) item.erase(0, 1);
            while (!item.empty() && (item.back() == ' ' || item.back() == '\t')) item.pop_back();

            if (!item.empty()) {
                result.push_back(std::stoi(item));
            }
        }
        return result;
    }
};

/**
 * Apply config file settings to Arguments struct.
 * Only applies settings where command-line wasn't explicitly set.
 */
template<typename Arguments>
void apply_config_to_args(Arguments& args, const AppConfig& config, bool cli_has_pool_url, bool cli_has_worker) {
    // Pool settings (only if not set via CLI)
    if (!cli_has_pool_url && !config.pool_url.empty()) {
        args.pool_url = config.pool_url;
        args.pool_mode = true;
    }
    if (!cli_has_worker && !config.pool_worker.empty()) {
        args.pool_worker = config.pool_worker;
    }
    if (args.pool_password.empty() && !config.pool_password.empty()) {
        args.pool_password = config.pool_password;
    }
    if (args.pool_api_key.empty() && !config.pool_api_key.empty()) {
        args.pool_api_key = config.pool_api_key;
    }

    // Puzzle settings (only non-default CLI values take precedence)
    if (args.puzzle_number == 0 && config.puzzle_number > 0) {
        args.puzzle_number = config.puzzle_number;
    }
    if (config.min_bits > 0) args.puzzle_min_bits = config.min_bits;
    if (config.max_bits < 160) args.puzzle_max_bits = config.max_bits;
    if (args.dp_bits < 0 && config.dp_bits >= 0) {
        args.dp_bits = config.dp_bits;
    }
    if (!config.checkpoint.empty() && args.puzzle_checkpoint.empty()) {
        args.puzzle_checkpoint = config.checkpoint;
    }

    // Apply boolean settings from config (CLI overrides by being explicit)
    // These are trickier since we can't tell if CLI explicitly set false vs default
    args.smart_select = config.smart_select;
    args.puzzle_kangaroo = config.kangaroo;
    args.puzzle_random = config.random_search;
    args.puzzle_auto_next = config.auto_next;

    // Brainwallet settings
    if (config.brainwallet_enabled) {
        args.brainwallet_mode = true;
    }
    if (args.wordlist_file.empty() && !config.wordlist.empty()) {
        args.wordlist_file = config.wordlist;
    }
    if (config.save_interval != 1000000) {
        args.save_interval = config.save_interval;
    }
    if (config.resume) args.resume = true;

    // Bloom filter
    if (args.bloom_file.empty() && !config.bloom_file.empty()) {
        args.bloom_file = config.bloom_file;
    }

    // GPU settings
    if (args.gpu_ids.empty() && !config.gpu_devices.empty()) {
        args.gpu_ids = config.gpu_devices;
    }
    if (args.batch_size == 4'000'000 && config.batch_size > 0) {  // 4M is default
        args.batch_size = config.batch_size;
    }
    if (config.force_calibrate) {
        args.calibrate = true;
        args.force_calibrate = true;
    }

    // Settings
    if (config.verbose) args.verbose = true;
    if (config.debug) args.debug = true;
    if (config.benchmark_seconds != 30) {
        args.benchmark_seconds = config.benchmark_seconds;
    }
}

}  // namespace collider
