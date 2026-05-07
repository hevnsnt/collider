/**
 * Brainwallet Setup Wizard
 *
 * First-run setup for brainwallet mode:
 * - Wordlist discovery and validation
 * - Preprocessing (dedupe, normalize, clean)
 * - PCFG training
 * - Configuration persistence
 */

#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <sstream>
#include <iomanip>
#include <chrono>
#include "interactive.hpp"
#include "../generators/pcfg.hpp"

namespace collider {
namespace ui {

namespace fs = std::filesystem;

/**
 * Brainwallet configuration - persisted between runs
 */
struct BrainwalletConfig {
    std::string bloom_file;
    std::vector<std::string> wordlist_dirs;
    std::string processed_wordlist;  // Combined, deduplicated wordlist
    std::string pcfg_model;          // Trained PCFG model path
    bool setup_complete = false;

    // Stats from last processing
    size_t total_raw_lines = 0;
    size_t total_unique_lines = 0;
    size_t wordlist_count = 0;
};

/**
 * Wordlist file info
 */
struct WordlistInfo {
    std::string path;
    std::string filename;
    size_t size_bytes;
    size_t line_count;
    bool valid;
    std::string error;
};

/**
 * Brainwallet Setup Wizard
 */
class BrainwalletSetup {
public:
    /**
     * Get the config directory path
     */
    static std::string get_config_dir() {
        std::string home;
#ifdef _WIN32
        const char* userprofile = std::getenv("USERPROFILE");
        if (userprofile) home = userprofile;
#else
        const char* home_env = std::getenv("HOME");
        if (home_env) home = home_env;
#endif
        if (home.empty()) home = ".";
        return home + "/.thecollider";
    }

    /**
     * Get the config file path
     */
    static std::string get_config_path() {
        return get_config_dir() + "/brainwallet_config.txt";
    }

    /**
     * Get the processed wordlist directory
     */
    static std::string get_processed_dir() {
        return get_config_dir() + "/processed";
    }

    /**
     * Check if setup has been completed
     */
    static bool is_setup_complete() {
        return fs::exists(get_config_path());
    }

    /**
     * Load configuration from file
     */
    static BrainwalletConfig load_config() {
        BrainwalletConfig config;
        std::string path = get_config_path();

        if (!fs::exists(path)) {
            return config;
        }

        std::ifstream file(path);
        std::string line;

        while (std::getline(file, line)) {
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;

            std::string key = line.substr(0, eq);
            std::string value = line.substr(eq + 1);

            if (key == "bloom_file") config.bloom_file = value;
            else if (key == "processed_wordlist") config.processed_wordlist = value;
            else if (key == "pcfg_model") config.pcfg_model = value;
            else if (key == "setup_complete") config.setup_complete = (value == "true");
            else if (key == "total_unique_lines") config.total_unique_lines = std::stoull(value);
            else if (key == "wordlist_count") config.wordlist_count = std::stoull(value);
            else if (key == "wordlist_dir") config.wordlist_dirs.push_back(value);
        }

        return config;
    }

    /**
     * Save configuration to file
     */
    static void save_config(const BrainwalletConfig& config) {
        // Create config directory
        fs::create_directories(get_config_dir());

        std::ofstream file(get_config_path());
        file << "bloom_file=" << config.bloom_file << "\n";
        file << "processed_wordlist=" << config.processed_wordlist << "\n";
        file << "pcfg_model=" << config.pcfg_model << "\n";
        file << "setup_complete=" << (config.setup_complete ? "true" : "false") << "\n";
        file << "total_unique_lines=" << config.total_unique_lines << "\n";
        file << "wordlist_count=" << config.wordlist_count << "\n";

        for (const auto& dir : config.wordlist_dirs) {
            file << "wordlist_dir=" << dir << "\n";
        }
    }

    /**
     * Scan a directory for wordlist files
     */
    static std::vector<WordlistInfo> scan_directory(const std::string& dir_path, bool recursive = true) {
        std::vector<WordlistInfo> wordlists;

        if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
            return wordlists;
        }

        auto scan_file = [&](const fs::path& path) {
            if (!fs::is_regular_file(path)) return;

            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            // Accept common wordlist extensions
            if (ext != ".txt" && ext != ".lst" && ext != ".dic" &&
                ext != ".wordlist" && ext != ".words" && ext != "") {
                return;
            }

            // Skip very small files (< 100 bytes)
            auto size = fs::file_size(path);
            if (size < 100) return;

            WordlistInfo info;
            info.path = path.string();
            info.filename = path.filename().string();
            info.size_bytes = size;
            info.valid = true;
            info.line_count = 0;

            wordlists.push_back(info);
        };

        try {
            if (recursive) {
                for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
                    scan_file(entry.path());
                }
            } else {
                for (const auto& entry : fs::directory_iterator(dir_path)) {
                    scan_file(entry.path());
                }
            }
        } catch (const std::exception& e) {
            // Ignore access errors
        }

        return wordlists;
    }

    /**
     * Count lines in a file
     */
    static size_t count_lines(const std::string& path) {
        std::ifstream file(path);
        size_t count = 0;
        std::string line;
        while (std::getline(file, line)) {
            count++;
        }
        return count;
    }

    /**
     * Format file size for display
     */
    static std::string format_size(size_t bytes) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(1);

        if (bytes >= 1024ULL * 1024 * 1024) {
            oss << (static_cast<double>(bytes) / (1024.0 * 1024 * 1024)) << " GB";
        } else if (bytes >= 1024ULL * 1024) {
            oss << (static_cast<double>(bytes) / (1024.0 * 1024)) << " MB";
        } else if (bytes >= 1024) {
            oss << (static_cast<double>(bytes) / 1024.0) << " KB";
        } else {
            oss << bytes << " B";
        }
        return oss.str();
    }

    /**
     * Format number with commas
     */
    static std::string format_number(size_t num) {
        std::string s = std::to_string(num);
        int n = static_cast<int>(s.length()) - 3;
        while (n > 0) {
            s.insert(n, ",");
            n -= 3;
        }
        return s;
    }

    /**
     * Normalize a passphrase line
     * - Trim whitespace
     * - Remove control characters
     * - Skip empty lines
     * - Skip very long lines (> 256 chars - probably not passphrases)
     */
    static std::string normalize_line(const std::string& line) {
        std::string result;
        result.reserve(line.size());

        // Trim leading whitespace
        size_t start = 0;
        while (start < line.size() && std::isspace(static_cast<unsigned char>(line[start]))) {
            start++;
        }

        // Trim trailing whitespace
        size_t end = line.size();
        while (end > start && std::isspace(static_cast<unsigned char>(line[end - 1]))) {
            end--;
        }

        // Skip empty or too long
        if (start >= end || (end - start) > 256) {
            return "";
        }

        // Copy, removing control characters
        for (size_t i = start; i < end; i++) {
            unsigned char c = static_cast<unsigned char>(line[i]);
            if (c >= 32 && c != 127) {  // Printable ASCII
                result.push_back(line[i]);
            }
        }

        return result;
    }

    /**
     * Process wordlists: combine, deduplicate, normalize
     */
    static bool process_wordlists(
        const std::vector<std::string>& wordlist_paths,
        const std::string& output_path,
        size_t& total_lines,
        size_t& unique_lines,
        std::function<void(const std::string&)> progress_callback = nullptr
    ) {
        std::unordered_set<std::string> seen;
        std::ofstream out(output_path);

        if (!out.is_open()) {
            return false;
        }

        total_lines = 0;
        unique_lines = 0;

        for (const auto& path : wordlist_paths) {
            if (progress_callback) {
                progress_callback("Processing: " + fs::path(path).filename().string());
            }

            std::ifstream in(path);
            if (!in.is_open()) continue;

            std::string line;
            while (std::getline(in, line)) {
                total_lines++;

                std::string normalized = normalize_line(line);
                if (normalized.empty()) continue;

                // Skip if already seen
                if (seen.find(normalized) != seen.end()) continue;

                seen.insert(normalized);
                out << normalized << "\n";
                unique_lines++;

                // Progress update every 1M lines
                if (progress_callback && total_lines % 1000000 == 0) {
                    progress_callback("Processed " + format_number(total_lines) + " lines...");
                }
            }
        }

        return true;
    }

    /**
     * Run the interactive setup wizard
     * @return Configured BrainwalletConfig
     */
    static BrainwalletConfig run_wizard() {
        BrainwalletConfig config;

        Interactive::display_section("Brain Wallet Setup Wizard");

        std::cout << colors::BRIGHT_WHITE << "Welcome to the Brain Wallet Scanner setup!\n" << colors::RESET;
        std::cout << "\nThis wizard will help you:\n";
        std::cout << "  1. Locate and validate your wordlists\n";
        std::cout << "  2. Combine and deduplicate them for optimal scanning\n";
        std::cout << "  3. Optionally train a PCFG model for smart generation\n\n";

        // Step 1: Wordlist directories
        Interactive::display_section("Step 1: Wordlist Location");

        std::vector<std::string> dirs;
        std::cout << "Enter directories containing wordlists (one per line).\n";
        std::cout << "Type 'done' when finished, or 'default' for common locations.\n\n";

        while (true) {
            std::cout << colors::CYAN << "Directory " << (dirs.size() + 1) << ": " << colors::RESET;
            std::string input = Interactive::read_line();

            if (input.empty()) continue;

            std::string lower = input;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

            if (lower == "done" || lower == "d") {
                if (dirs.empty()) {
                    Interactive::warning_message("Please add at least one directory.");
                    continue;
                }
                break;
            }

            if (lower == "default") {
                // Add common default locations
#ifdef _WIN32
                dirs.push_back("C:\\wordlists");
                dirs.push_back("D:\\wordlists");
                dirs.push_back(".\\wordlists");
#else
                dirs.push_back("/usr/share/wordlists");
                dirs.push_back("~/wordlists");
                dirs.push_back("./wordlists");
                dirs.push_back("./data");
#endif
                Interactive::status_message("Added default wordlist locations", true);
                break;
            }

            // Expand home directory
            if (input[0] == '~') {
                std::string home;
#ifdef _WIN32
                home = std::getenv("USERPROFILE") ? std::getenv("USERPROFILE") : "";
#else
                home = std::getenv("HOME") ? std::getenv("HOME") : "";
#endif
                input = home + input.substr(1);
            }

            if (fs::exists(input) && fs::is_directory(input)) {
                dirs.push_back(input);
                Interactive::status_message("Added: " + input, true);
            } else {
                Interactive::warning_message("Directory not found: " + input);
            }
        }

        config.wordlist_dirs = dirs;

        // Step 2: Scan for wordlists
        Interactive::display_section("Step 2: Scanning for Wordlists");

        std::vector<WordlistInfo> all_wordlists;
        size_t total_size = 0;

        for (const auto& dir : dirs) {
            Interactive::info_message("Scanning: " + dir);
            auto found = scan_directory(dir);

            for (auto& wl : found) {
                total_size += wl.size_bytes;
            }

            all_wordlists.insert(all_wordlists.end(), found.begin(), found.end());
        }

        if (all_wordlists.empty()) {
            Interactive::error_message("No wordlist files found!");
            std::cout << "\nPlease add .txt wordlist files to one of the directories and try again.\n";
            return config;
        }

        std::cout << "\n";
        Interactive::status_message("Found " + format_number(all_wordlists.size()) +
                                   " wordlist files (" + format_size(total_size) + " total)", true);

        // Show summary
        std::cout << "\n" << colors::BRIGHT_WHITE << "Wordlists found:" << colors::RESET << "\n";

        size_t max_show = 10;
        for (size_t i = 0; i < std::min(all_wordlists.size(), max_show); i++) {
            std::cout << "  " << colors::DIM << (i + 1) << "." << colors::RESET << " "
                      << all_wordlists[i].filename << " ("
                      << format_size(all_wordlists[i].size_bytes) << ")\n";
        }
        if (all_wordlists.size() > max_show) {
            std::cout << "  " << colors::DIM << "... and "
                      << (all_wordlists.size() - max_show) << " more" << colors::RESET << "\n";
        }

        // Step 3: Process wordlists
        Interactive::display_section("Step 3: Processing Wordlists");

        std::cout << "This will:\n";
        std::cout << "  - Combine all wordlists into one file\n";
        std::cout << "  - Remove duplicate entries\n";
        std::cout << "  - Normalize formatting (trim, clean)\n";
        std::cout << "  - Remove invalid entries (empty, too long)\n\n";

        if (!Interactive::prompt_yes_no("Process wordlists now?", true)) {
            Interactive::warning_message("Skipping wordlist processing.");
            save_config(config);
            return config;
        }

        // Create output directory
        fs::create_directories(get_processed_dir());
        std::string output_path = get_processed_dir() + "/combined_wordlist.txt";

        std::cout << "\n";
        Interactive::info_message("Processing wordlists...");

        std::vector<std::string> paths;
        for (const auto& wl : all_wordlists) {
            paths.push_back(wl.path);
        }

        auto start_time = std::chrono::steady_clock::now();

        size_t total_lines = 0, unique_lines = 0;
        bool success = process_wordlists(
            paths, output_path, total_lines, unique_lines,
            [](const std::string& msg) {
                std::cout << "  " << colors::DIM << msg << colors::RESET << "\r" << std::flush;
            }
        );

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

        std::cout << std::string(60, ' ') << "\r";  // Clear progress line

        if (success) {
            std::cout << "\n";
            Interactive::status_message("Processing complete!", true);
            std::cout << "  " << colors::CYAN << "Total lines scanned:" << colors::RESET << " "
                      << format_number(total_lines) << "\n";
            std::cout << "  " << colors::CYAN << "Unique passphrases:" << colors::RESET << " "
                      << format_number(unique_lines) << "\n";
            std::cout << "  " << colors::CYAN << "Duplicates removed:" << colors::RESET << " "
                      << format_number(total_lines - unique_lines) << "\n";
            std::cout << "  " << colors::CYAN << "Processing time:" << colors::RESET << " "
                      << elapsed << " seconds\n";
            std::cout << "  " << colors::CYAN << "Output file:" << colors::RESET << " "
                      << output_path << "\n";

            config.processed_wordlist = output_path;
            config.total_raw_lines = total_lines;
            config.total_unique_lines = unique_lines;
            config.wordlist_count = all_wordlists.size();
        } else {
            Interactive::error_message("Failed to process wordlists.");
        }

        // Step 4: PCFG Training (optional)
        Interactive::display_section("Step 4: PCFG Training (Optional)");

        std::cout << colors::BRIGHT_WHITE << "What is PCFG?\n" << colors::RESET;
        std::cout << "PCFG (Probabilistic Context-Free Grammar) learns password patterns\n";
        std::cout << "and generates candidates in probability order - most likely first.\n\n";

        std::cout << colors::BRIGHT_WHITE << "Benefits:\n" << colors::RESET;
        std::cout << "  - Tests 'bitcoin123' before 'xq7$mZpK'\n";
        std::cout << "  - Generates passwords NOT in your wordlists\n";
        std::cout << "  - Learns patterns like [Name][Year][Symbol]\n";
        std::cout << "  - Much faster to find human-chosen passphrases\n\n";

        if (Interactive::prompt_yes_no("Train PCFG model from your wordlists?", true)) {
            std::string pcfg_path = get_processed_dir() + "/brainwallet.pcfg";

            std::cout << "\n";
            Interactive::info_message("Training PCFG model...");
            std::cout << "  This analyzes password structure patterns from your wordlists.\n";
            std::cout << "  " << colors::DIM << "(This may take a few minutes for large wordlists)" << colors::RESET << "\n\n";

            // Train PCFG model using the pcfg::Trainer
            try {
                pcfg::Trainer::Config trainer_config;
                trainer_config.min_length = 4;
                trainer_config.max_length = 64;
                trainer_config.detect_keyboard_patterns = true;
                trainer_config.detect_multiwords = true;
                trainer_config.max_terminals_per_nt = 100000;

                pcfg::Trainer trainer(trainer_config);

                // Train on the combined processed wordlist
                if (!config.processed_wordlist.empty() && fs::exists(config.processed_wordlist)) {
                    Interactive::info_message("Training on: " + fs::path(config.processed_wordlist).filename().string());

                    auto train_start = std::chrono::steady_clock::now();
                    trainer.train(config.processed_wordlist);
                    auto train_end = std::chrono::steady_clock::now();
                    auto train_elapsed = std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start).count();

                    // Get training statistics
                    auto train_stats = trainer.get_training_stats();
                    std::cout << "  " << colors::DIM << "Processed " << format_number(train_stats.total_passwords)
                              << " passwords in " << train_elapsed << "s" << colors::RESET << "\n";

                    // Build and save the grammar
                    Interactive::info_message("Building grammar...");
                    auto grammar = trainer.build_grammar();

                    // Validate grammar before saving
                    auto stats = grammar.get_stats();
                    if (stats.num_structures == 0) {
                        Interactive::warning_message("No patterns learned - wordlist may be too small or uniform.");
                        std::cout << "  The wordlist will be used directly for now.\n";
                    } else {
                        // Save the trained model
                        grammar.save(pcfg_path);
                        config.pcfg_model = pcfg_path;

                        std::cout << "\n";
                        Interactive::status_message("PCFG model trained successfully!", true);
                        std::cout << "  " << colors::CYAN << "Structures learned:" << colors::RESET << " "
                                  << format_number(stats.num_structures) << "\n";
                        std::cout << "  " << colors::CYAN << "Non-terminals:" << colors::RESET << " "
                                  << format_number(stats.num_non_terminals) << "\n";
                        std::cout << "  " << colors::CYAN << "Total terminals:" << colors::RESET << " "
                                  << format_number(stats.num_terminals) << "\n";
                        std::cout << "  " << colors::CYAN << "Avg terminals/NT:" << colors::RESET << " "
                                  << std::fixed << std::setprecision(1) << stats.avg_terminals_per_nt << "\n";
                        std::cout << "  " << colors::CYAN << "Model saved to:" << colors::RESET << " "
                                  << pcfg_path << "\n";
                    }
                } else {
                    Interactive::warning_message("No processed wordlist found to train on.");
                    std::cout << "  Please run wordlist processing first.\n";
                }
            } catch (const std::exception& e) {
                Interactive::error_message("PCFG training failed: " + std::string(e.what()));
                std::cout << "  The wordlist will be used directly for now.\n";
            }
        }

        // Save configuration
        config.setup_complete = true;
        save_config(config);

        Interactive::display_section("Setup Complete!");

        std::cout << colors::GREEN << "Brain wallet scanner is ready to use.\n" << colors::RESET;
        std::cout << "\nConfiguration saved to: " << get_config_path() << "\n";
        std::cout << "Processed wordlist: " << config.processed_wordlist << "\n";
        std::cout << "\nYou can now run brainwallet scans with:\n";
        std::cout << "  " << colors::CYAN << "./thepuzzler --brainwallet --bloom <your_bloom_filter.blf>"
                  << colors::RESET << "\n\n";

        return config;
    }

    /**
     * Show current configuration summary
     */
    static void show_config_summary(const BrainwalletConfig& config) {
        std::cout << colors::BRIGHT_WHITE << "Current Configuration:" << colors::RESET << "\n";
        std::cout << "  Wordlist: " << config.processed_wordlist << "\n";
        std::cout << "  Entries: " << format_number(config.total_unique_lines) << " unique passphrases\n";
        std::cout << "  Sources: " << config.wordlist_count << " wordlist files\n";
        if (!config.pcfg_model.empty()) {
            std::cout << "  PCFG: " << config.pcfg_model << "\n";
        }
        std::cout << "\n";
    }
};

} // namespace ui
} // namespace collider
