/**
 * yaml_config.hpp - Simple YAML configuration loader for theCollider
 *
 * Parses a subset of YAML (key: value pairs with sections) without external dependencies.
 * Command-line arguments override config file settings.
 *
 * Override precedence (Wave 5 refactor, 2026-05-04):
 *   1. CLI flags explicitly supplied on argv (tracked via CLIFlags::*_set bits).
 *   2. config.yml fields whose value differs from the AppConfig default.
 *   3. Arguments default values.
 *
 * The CLIFlags struct is populated INSIDE parse_args at the moment argv supplies
 * a value -- not inferred post-parse from the resulting Arguments. This eliminates
 * the entire "value silently overridden" bug class documented in track-e
 * (e.g. --batch-size 4000000, --random against random_search:false, --puzzle 0).
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

    // v1.4.1: per-run target / range / pubkey override. Each field is
    // empty by default; non-empty values overlay onto Arguments only
    // when the matching CLIFlags bit is unset (CLI flags win).
    //
    //   target   - Bitcoin address to scan against (overrides the
    //              puzzle's bundled address from puzzle_history.json)
    //   start    - Range start in hex (with 0x prefix)
    //   end      - Range end in hex (with 0x prefix)
    //   pubkey   - 33-byte compressed public key in hex (02/03 + 32B).
    //              Required for kangaroo when scanning a target whose
    //              pubkey isn't already in puzzle_history.json (rare;
    //              only useful for newly-revealed pubkeys or research
    //              targets outside the canonical Bitcoin Puzzle set).
    std::string puzzle_target;
    std::string puzzle_start;
    std::string puzzle_end;
    std::string puzzle_pubkey;

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

    // Paths (NOTE: not yet consumed by apply_config_to_args; see track-e finding 7).
    // Documented as a known gap rather than silently dropped: leaving them parsed
    // so future code can opt in. Removing them would break user configs.
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
            if (key == "url") pool_url = value;
            else if (key == "worker") pool_worker = value;
            else if (key == "password") pool_password = value;
            else if (key == "api_key") pool_api_key = value;
        }
        else if (section == "puzzle") {
            if (key == "number") puzzle_number = std::stoi(value);
            else if (key == "smart_select") smart_select = parse_bool(value);
            else if (key == "min_bits") min_bits = std::stoi(value);
            else if (key == "max_bits") max_bits = std::stoi(value);
            else if (key == "kangaroo") kangaroo = parse_bool(value);
            else if (key == "dp_bits") dp_bits = std::stoi(value);
            else if (key == "random_search") random_search = parse_bool(value);
            else if (key == "auto_next") auto_next = parse_bool(value);
            else if (key == "checkpoint") checkpoint = value;
            // v1.4.1: target / range / pubkey overrides
            else if (key == "target") puzzle_target = value;
            else if (key == "start") puzzle_start = value;
            else if (key == "end") puzzle_end = value;
            else if (key == "pubkey") puzzle_pubkey = value;
        }
        else if (section == "brainwallet") {
            if (key == "enabled") brainwallet_enabled = parse_bool(value);
            else if (key == "wordlist") wordlist = value;
            else if (key == "save_interval") save_interval = std::stoull(value);
            else if (key == "resume") resume = parse_bool(value);
        }
        else if (section == "bloom") {
            if (key == "file") bloom_file = value;
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

public:
    // -----------------------------------------------------------------------
    // Pool worker (BTC payout address) persistence.
    //
    // Guided mode in interactive_ui calls save_pool_worker() after the user
    // enters their BTC address so the next launch defaults to it. We touch
    // ONLY ~/.collider/config.yml (or %USERPROFILE%/.collider/config.yml on
    // Windows) -- never the user's working-directory ./config.yml, which is
    // the operator-managed file. A power user with their own config remains
    // in control; the home-directory file is purely the "remember my BTC
    // address from guided mode" store.
    //
    // Returns the path written, or empty string on any failure (callers
    // emit a hint, never abort).
    // -----------------------------------------------------------------------
    static std::string save_pool_worker(const std::string& worker_addr,
                                         const std::string& pool_url = "") {
        if (worker_addr.empty()) return {};

        std::string home;
#ifdef _WIN32
        const char* userprofile = std::getenv("USERPROFILE");
        home = userprofile ? userprofile : "";
#else
        const char* home_env = std::getenv("HOME");
        home = home_env ? home_env : "";
#endif
        if (home.empty()) return {};

        std::filesystem::path dir = std::filesystem::path(home) / ".collider";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        // Even if create_directories fails (e.g. read-only home), keep
        // going -- the open-for-write below will be the authoritative
        // failure point.

        const std::filesystem::path file_path = dir / "config.yml";

        // If the file already exists and has a pool.worker line, do
        // nothing. Operator preference wins; never silently overwrite.
        if (std::filesystem::exists(file_path)) {
            std::ifstream existing(file_path);
            std::string line;
            bool in_pool = false;
            while (std::getline(existing, line)) {
                std::string trim = line;
                size_t indent = 0;
                while (indent < trim.size() && (trim[indent] == ' ' || trim[indent] == '\t')) ++indent;
                std::string key = trim.substr(indent);
                if (indent == 0 && key.rfind("pool:", 0) == 0) {
                    in_pool = true;
                } else if (indent == 0 && !key.empty() && key[0] != '#') {
                    in_pool = false;
                }
                if (in_pool && key.rfind("worker:", 0) == 0) {
                    return file_path.string();  // Already configured.
                }
            }
            // Existing file but no pool.worker -- don't risk corrupting
            // it with a hand-rolled YAML edit. Hint and return.
            return {};
        }

        // File does not exist -- write a minimal new one.
        std::ofstream out(file_path);
        if (!out) return {};
        out << "# theCollider config (auto-created from guided mode).\n";
        out << "# Edit freely -- the format is documented in example-config.yml.\n";
        out << "\n";
        out << "pool:\n";
        out << "  worker: \"" << worker_addr << "\"\n";
        if (!pool_url.empty()) {
            out << "  url: \"" << pool_url << "\"\n";
        }
        out << "\n";
        return file_path.string();
    }
};

/**
 * CLI flags tracking struct -- one bit per Arguments field.
 *
 * Each *_set bit is set INSIDE parse_args at the moment argv supplies the value.
 * apply_config_to_args() consults these bits before overriding -- if the bit is
 * true, config never overrides the CLI value, regardless of whether the CLI value
 * happens to equal the Arguments default.
 *
 * This eliminates the bug class where sentinel-based detection (e.g.
 * "args.batch_size == 4_000_000 means user did not set it") wrongly treated an
 * explicit CLI value equal to the default as "unset".
 */
struct CLIFlags {
    // Pool / mode
    bool pool_url_set = false;          // --pool / -p
    bool pool_worker_set = false;       // --worker / -w
    bool pool_password_set = false;     // --pool-password
    bool pool_api_key_set = false;      // --pool-api-key
    bool brainwallet_set = false;       // --brainwallet

    // Puzzle
    bool puzzle_number_set = false;     // --puzzle N (with value)
    bool puzzle_min_bits_set = false;   // --puzzle-min-bits
    bool puzzle_max_bits_set = false;   // --puzzle-max-bits
    bool dp_bits_set = false;           // --dp-bits
    bool puzzle_checkpoint_set = false; // --puzzle-checkpoint
    bool puzzle_random_set = false;     // --random or --sequential
    bool puzzle_auto_next_set = false;  // --auto-next
    bool puzzle_kangaroo_set = false;   // --kangaroo
    bool smart_select_set = false;      // --no-smart
    // v1.4.1: per-run override flags
    bool puzzle_target_set = false;     // --puzzle-target
    bool puzzle_start_set = false;      // --puzzle-start
    bool puzzle_end_set = false;        // --puzzle-end
    bool puzzle_pubkey_set = false;     // --pubkey

    // Brainwallet / search
    bool resume_set = false;            // --resume
    bool save_interval_set = false;     // --save-interval

    // Bloom filter / GPU
    bool bloom_file_set = false;        // --bloom
    bool gpu_ids_set = false;           // --gpus / -g
    bool batch_size_set = false;        // --batch-size
    bool force_calibrate_set = false;   // --force-calibrate

    // Settings
    bool verbose_set = false;           // --verbose / -v
    bool debug_set = false;             // --debug
    bool benchmark_seconds_set = false; // --benchmark-time
};

/**
 * Apply config file settings to Arguments struct.
 * For every field, config only overrides when the CLI did not explicitly set it
 * (per the matching CLIFlags::*_set bit). The previous "default-value-as-sentinel"
 * pattern is removed.
 */
template<typename Arguments>
void apply_config_to_args(Arguments& args, const AppConfig& config, const CLIFlags& cli) {
    // ------------------------------------------------------------------------
    // Pool / mode
    // ------------------------------------------------------------------------
    if (!cli.pool_url_set && !config.pool_url.empty()) {
        args.pool_url = config.pool_url;
        // Auto-enable pool mode when config supplies a URL, but only when the
        // user did not request brainwallet mode (CLI or config) and did not
        // already set pool_mode via CLI.
        if (!cli.brainwallet_set && !args.brainwallet_mode && !args.pool_mode) {
            args.pool_mode = true;
        }
    }
    if (!cli.pool_worker_set && !config.pool_worker.empty()) {
        args.pool_worker = config.pool_worker;
    }
    if (!cli.pool_password_set && !config.pool_password.empty()) {
        args.pool_password = config.pool_password;
    }
    if (!cli.pool_api_key_set && !config.pool_api_key.empty()) {
        args.pool_api_key = config.pool_api_key;
    }

    // ------------------------------------------------------------------------
    // Puzzle
    // ------------------------------------------------------------------------
    if (!cli.puzzle_number_set && config.puzzle_number > 0) {
        args.puzzle_number = config.puzzle_number;
    }
    if (!cli.puzzle_min_bits_set && config.min_bits > 0) {
        args.puzzle_min_bits = config.min_bits;
    }
    if (!cli.puzzle_max_bits_set && config.max_bits < 160) {
        args.puzzle_max_bits = config.max_bits;
    }
    if (!cli.dp_bits_set && config.dp_bits >= 0) {
        args.dp_bits = config.dp_bits;
    }
    if (!cli.puzzle_checkpoint_set && !config.checkpoint.empty()) {
        args.puzzle_checkpoint = config.checkpoint;
    }

    // v1.4.1: target / range / pubkey overlay. CLI wins when set.
    if (!cli.puzzle_target_set && !config.puzzle_target.empty()) {
        args.puzzle_target = config.puzzle_target;
    }
    if (!cli.puzzle_start_set && !config.puzzle_start.empty()) {
        args.puzzle_range_start = config.puzzle_start;
    }
    if (!cli.puzzle_end_set && !config.puzzle_end.empty()) {
        args.puzzle_range_end = config.puzzle_end;
    }
    if (!cli.puzzle_pubkey_set && !config.puzzle_pubkey.empty()) {
        args.puzzle_pubkey = config.puzzle_pubkey;
    }

    // Boolean settings: config wins only if CLI did not set a *_set bit.
    if (!cli.smart_select_set) {
        args.smart_select = config.smart_select;
    }
    if (!cli.puzzle_kangaroo_set) {
        args.puzzle_kangaroo = config.kangaroo;
    }
    if (!cli.puzzle_random_set) {
        args.puzzle_random = config.random_search;
    }
    if (!cli.puzzle_auto_next_set) {
        args.puzzle_auto_next = config.auto_next;
    }

    // ------------------------------------------------------------------------
    // Brainwallet / search state
    // ------------------------------------------------------------------------
    // Brainwallet enable: only honour config when CLI did not explicitly
    // request --brainwallet AND CLI did not set --pool / pool_mode. This
    // closes the "config silently turns brainwallet on under --pool" hole
    // documented as Critical in track-e (Scenario B).
    if (!cli.brainwallet_set && !cli.pool_url_set && !args.pool_mode &&
        config.brainwallet_enabled) {
        args.brainwallet_mode = true;
    }
    if (args.wordlist_file.empty() && !config.wordlist.empty()) {
        args.wordlist_file = config.wordlist;
    }
    if (!cli.save_interval_set && config.save_interval != 1000000) {
        args.save_interval = config.save_interval;
    }
    if (!cli.resume_set && config.resume) {
        args.resume = true;
    }

    // ------------------------------------------------------------------------
    // Bloom filter / GPU
    // ------------------------------------------------------------------------
    // Bloom-filter opportunistic address checking is a Pro feature
    // (collisionprotocol.com/pro). The free build silently ignores any
    // bloom file from config.yml and emits a one-time hint if one was
    // supplied, instead of loading 142 MB of address data into the GPU.
    if (!cli.bloom_file_set && !config.bloom_file.empty()) {
#ifdef COLLIDER_PRO
        args.bloom_file = config.bloom_file;
#else
        std::cerr << "[Pro] Ignoring bloom filter '" << config.bloom_file
                  << "' from config -- opportunistic address scanning is a "
                     "Pro feature. https://collisionprotocol.com/pro"
                  << std::endl;
#endif
    }
    if (!cli.gpu_ids_set && !config.gpu_devices.empty()) {
        args.gpu_ids = config.gpu_devices;
    }
    if (!cli.batch_size_set && config.batch_size > 0) {
        args.batch_size = config.batch_size;
    }
    if (!cli.force_calibrate_set && config.force_calibrate) {
        args.calibrate = true;
        args.force_calibrate = true;
    }

    // ------------------------------------------------------------------------
    // Settings
    // ------------------------------------------------------------------------
    if (!cli.verbose_set && config.verbose) {
        args.verbose = true;
    }
    if (!cli.debug_set && config.debug) {
        args.debug = true;
    }
    if (!cli.benchmark_seconds_set && config.benchmark_seconds != 30) {
        args.benchmark_seconds = config.benchmark_seconds;
    }
}

}  // namespace collider
