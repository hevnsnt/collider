/**
 * interactive_ui.cpp - Implementation of theCollider's interactive
 * startup flow.
 *
 * Extracted verbatim from src/main.cpp during the v1.4.1 A.3 refactor;
 * no behavior changes. The bodies still depend on a handful of free
 * functions that currently live in main.cpp (format_number*,
 * normalize_path, analyze_puzzle, get_best_puzzle); those are forward-
 * declared below and migrate to runtime/puzzle_solver.cpp in a follow-
 * up commit of the same refactor.
 */

// Prevent Windows min/max macros from breaking std::min/std::max.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "ui/interactive_ui.hpp"

#include "core/puzzle_analysis.hpp"
#include "core/puzzle_config.hpp"
#include "core/yaml_config.hpp"
#include "tools/utxo_bloom_builder.hpp"
#include "ui/banner.hpp"
#include "ui/interactive.hpp"

#ifdef COLLIDER_PRO
#include "core/brainwallet_state.hpp"
#include "ui/brainwallet_setup.hpp"
#endif

// Forward declarations for free functions that still live in src/main.cpp.
// These migrate into runtime/puzzle_solver.cpp (formatters) and
// runtime/brain_wallet_runner.cpp (normalize_path) in a follow-up commit
// of the v1.4.1 A.3 refactor.
std::string format_number(uint64_t n);
std::string format_number_human(uint64_t n);
std::string normalize_path(const std::string& path);

namespace collider::ui {

Arguments run_puzzle_interactive(Arguments base_args, double gpu_speed_mkeys) {
    using namespace ::collider::ui;
    Arguments args = base_args;
    args.puzzle_mode = true;

    Interactive::display_section("Bitcoin Puzzle Challenge Mode");

    // First ask: standalone or pool?
    PuzzleModeChoice mode_choice = Interactive::display_puzzle_mode_menu();

    if (mode_choice == PuzzleModeChoice::BACK) {
        args.go_back = true;  // Return to main menu
        return args;
    }

    bool use_pool = (mode_choice == PuzzleModeChoice::JOIN_POOL);

    if (use_pool) {
        // Configure pool settings
        std::string pool_url, worker;
        if (!Interactive::prompt_pool_config(pool_url, worker, args.pool_url, args.pool_worker)) {
            args.go_back = true;
            return args;
        }
        args.pool_mode = true;
        args.pool_url = pool_url;
        args.pool_worker = worker;

        // v1.4.1: persist the BTC payout address to ~/.collider/config.yml
        // so the next launch defaults to it (user can still override at the
        // prompt). Only writes when the file doesn't already exist; never
        // overwrites an existing operator-managed config.
        const std::string saved =
            collider::AppConfig::save_pool_worker(worker, pool_url);
        if (!saved.empty()) {
            Interactive::info_message("Saved pool worker to " + saved
                + " (will default on next launch)");
        }

        // Pool mode doesn't need puzzle selection -- pool assigns work.
        // Pre-1.4.1 the runner waited on Enter here, which forced an extra
        // keystroke after the user had ALREADY confirmed the pool/worker
        // pair in prompt_pool_config above. Just connect.
        std::cout << "\n";
        Interactive::info_message("Pool mode: Work will be assigned by the pool server. Connecting...");
        return args;
    }

    // Standalone mode - select puzzle
    std::cout << "\n";
    std::cout << "Enter puzzle number (1-256) or 'auto' for smart selection";
    int puzzle_choice = Interactive::prompt_number("", 1, 256, true);

    if (puzzle_choice == -1) {
        // Auto mode - use smart selection
        int best = ::get_best_puzzle(gpu_speed_mkeys);
        if (best > 0) {
            args.puzzle_number = best;
            const ::collider::PuzzleInfo* puzzle = ::collider::PuzzleDatabase::get_puzzle(best);

            std::cout << "\n";
            Interactive::info_message("Analyzing puzzles...");

            if (puzzle) {
                bool has_pubkey = !puzzle->public_key_hex.empty();

                // Calculate estimated time
                std::string est_time;
                ::PuzzleAnalysis analysis = ::analyze_puzzle(puzzle, gpu_speed_mkeys);
                if (analysis.estimated_gpu_years < 0.01) {
                    est_time = "<1 week";
                } else if (analysis.estimated_gpu_years < 0.1) {
                    est_time = std::to_string((int)(analysis.estimated_gpu_years * 52)) + " weeks";
                } else if (analysis.estimated_gpu_years < 1.0) {
                    est_time = std::to_string((int)(analysis.estimated_gpu_years * 12)) + " months";
                } else {
                    est_time = "~" + std::to_string((int)analysis.estimated_gpu_years) + " years";
                }

                std::cout << "\n";
                Interactive::info_message("Smart Selection: Puzzle #" + std::to_string(best));
                Interactive::display_puzzle_info(puzzle->number, puzzle->bits, has_pubkey,
                                                 puzzle->btc_reward, est_time);

                if (has_pubkey) {
                    args.puzzle_kangaroo = true;
                }

                std::cout << "\n";
                if (!Interactive::prompt_yes_no("Proceed with Puzzle #" + std::to_string(best) + "?", true)) {
                    std::cout << "\n";
                    Interactive::info_message("Selection cancelled. Returning to main menu...");
                    args.go_back = true;
                    return args;
                }
            }
        }
    } else {
        // Specific puzzle selected
        args.puzzle_number = puzzle_choice;
        const ::collider::PuzzleInfo* puzzle = ::collider::PuzzleDatabase::get_puzzle(puzzle_choice);

        if (puzzle) {
            bool has_pubkey = !puzzle->public_key_hex.empty();

            // Check if solved
            if (puzzle->solved) {
                std::cout << "\n";
                Interactive::warning_message("Puzzle #" + std::to_string(puzzle_choice) + " is already SOLVED!");
                std::cout << "    Solution: " << puzzle->solution_hex << "\n";
                std::cout << "\n";

                if (Interactive::prompt_yes_no("Continue in testing mode?", true)) {
                    Interactive::info_message("Testing mode - will verify against known solution.");
                } else {
                    args.go_back = true;
                    return args;
                }
            }

            // Calculate estimated time
            std::string est_time;
            ::PuzzleAnalysis analysis = ::analyze_puzzle(puzzle, gpu_speed_mkeys);
            if (analysis.estimated_gpu_years < 0.01) {
                est_time = "<1 week";
            } else if (analysis.estimated_gpu_years < 1.0) {
                est_time = std::to_string((int)(analysis.estimated_gpu_years * 12)) + " months";
            } else {
                est_time = "~" + std::to_string((int)analysis.estimated_gpu_years) + " years";
            }

            std::cout << "\n";
            Interactive::display_puzzle_info(puzzle->number, puzzle->bits, has_pubkey,
                                             puzzle->btc_reward, est_time);

            if (has_pubkey) {
                args.puzzle_kangaroo = true;
                std::cout << "\n";
                Interactive::info_message("Auto-enabled Kangaroo algorithm (public key available)");
            }

            std::cout << "\n";
            if (!Interactive::prompt_yes_no("Start solving Puzzle #" + std::to_string(puzzle_choice) + "?", true)) {
                args.go_back = true;
                return args;
            }
        } else {
            Interactive::error_message("Unknown puzzle number: " + std::to_string(puzzle_choice));
            args.go_back = true;
            return args;
        }
    }

    return args;
}

#ifdef COLLIDER_PRO
Arguments run_brainwallet_interactive(Arguments base_args) {
    using namespace ::collider::ui;
    Arguments args = base_args;
    args.brainwallet_mode = true;
    args.pool_mode = false;  // Disable pool mode - brainwallet is mutually exclusive

    Interactive::display_section("Brain Wallet Scanner Mode");

    // Check if this is first run - need to set up wordlists
    BrainwalletConfig config;
    bool first_run = !BrainwalletSetup::is_setup_complete();

    if (first_run) {
        std::cout << colors::BRIGHT_WHITE << "First-time setup detected!\n" << colors::RESET;
        std::cout << "Before scanning, we need to set up your wordlists.\n\n";

        if (Interactive::prompt_yes_no("Run the setup wizard now?", true)) {
            config = BrainwalletSetup::run_wizard();

            if (!config.setup_complete) {
                Interactive::warning_message("Setup was not completed.");
                args.go_back = true;
                return args;
            }
        } else {
            Interactive::info_message("You can run setup later with: ./collider --brainwallet-setup");
            args.go_back = true;
            return args;
        }
    } else {
        // Load existing configuration
        config = BrainwalletSetup::load_config();
        BrainwalletSetup::show_config_summary(config);

        // Offer to reconfigure
        if (Interactive::prompt_yes_no("Reconfigure wordlists?", false)) {
            config = BrainwalletSetup::run_wizard();
        }
    }

    // Set the wordlist from config
    if (!config.processed_wordlist.empty() && std::filesystem::exists(config.processed_wordlist)) {
        args.wordlist_file = config.processed_wordlist;
        Interactive::status_message("Using wordlist: " + config.processed_wordlist +
                                   " (" + BrainwalletSetup::format_number(config.total_unique_lines) + " entries)", true);
    }

    // Check for bloom filter
    std::cout << "\n";
    std::string default_bloom = "funded_addresses.blf";

    // Build search paths for UTXO and bloom files
    std::vector<std::string> search_dirs;
    search_dirs.push_back(".");  // Current directory
    search_dirs.push_back("..");  // Parent directory

    // Add user's wordlist directories
    for (const auto& dir : config.wordlist_dirs) {
        search_dirs.push_back(dir);
        // Also check parent of wordlist dirs
        std::filesystem::path p(dir);
        if (p.has_parent_path()) {
            search_dirs.push_back(p.parent_path().string());
        }
    }

    // Add common Windows locations
#ifdef _WIN32
    search_dirs.push_back("D:\\theCollider");
    search_dirs.push_back("C:\\theCollider");
    search_dirs.push_back("D:\\");
    search_dirs.push_back("C:\\");
#endif

    // Search for existing bloom filters and UTXO dumps
    std::vector<std::pair<std::string, size_t>> found_blooms;
    std::vector<std::pair<std::string, size_t>> found_utxos;

    for (const auto& dir : search_dirs) {
        if (!std::filesystem::exists(dir)) continue;

        try {
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (!entry.is_regular_file()) continue;

                std::string filename = entry.path().filename().string();
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);

                // Check for bloom filters
                if (ext == ".blf") {
                    found_blooms.push_back({entry.path().string(), entry.file_size()});
                }

                // Check for UTXO dumps
                if (filename.find("utxo") != std::string::npos && ext == ".csv") {
                    found_utxos.push_back({entry.path().string(), entry.file_size()});
                }
            }
        } catch (const std::exception&) {
            // Skip inaccessible directories
        }
    }

    // Remove duplicates (normalize paths)
    auto dedupe = [](std::vector<std::pair<std::string, size_t>>& vec) {
        std::set<std::string> seen;
        vec.erase(std::remove_if(vec.begin(), vec.end(), [&](const auto& p) {
            std::string canonical;
            try {
                canonical = std::filesystem::canonical(p.first).string();
            } catch (...) {
                canonical = p.first;
            }
            if (seen.count(canonical)) return true;
            seen.insert(canonical);
            return false;
        }), vec.end());
    };
    dedupe(found_blooms);
    dedupe(found_utxos);

    // Check config first, then search results
    if (!config.bloom_file.empty() && std::filesystem::exists(config.bloom_file)) {
        args.bloom_file = config.bloom_file;
        Interactive::status_message("Using bloom filter: " + config.bloom_file, true);
    } else if (std::filesystem::exists(default_bloom)) {
        args.bloom_file = default_bloom;
        Interactive::status_message("Found bloom filter: " + default_bloom, true);
    } else if (!found_blooms.empty()) {
        // Found bloom filter(s) - let user choose
        Interactive::status_message("Found " + std::to_string(found_blooms.size()) + " bloom filter(s):", true);
        std::cout << "\n";

        for (size_t i = 0; i < found_blooms.size(); i++) {
            double size_mb = found_blooms[i].second / (1024.0 * 1024.0);
            std::cout << "  [" << (i + 1) << "] " << found_blooms[i].first
                      << " (" << std::fixed << std::setprecision(1) << size_mb << " MB)\n";
        }
        std::cout << "  [" << (found_blooms.size() + 1) << "] Enter a different path\n";
        std::cout << "\n";

        int choice = Interactive::prompt_menu_choice(1, static_cast<int>(found_blooms.size() + 1));

        if (choice >= 1 && choice <= static_cast<int>(found_blooms.size())) {
            args.bloom_file = found_blooms[choice - 1].first;
            config.bloom_file = args.bloom_file;
            BrainwalletSetup::save_config(config);
        } else {
            std::string bloom_path = Interactive::prompt_path("Enter path to bloom filter (.blf)", true);
            if (bloom_path.empty() || !std::filesystem::exists(bloom_path)) {
                Interactive::warning_message("Invalid bloom filter path.");
                args.go_back = true;
                return args;
            }
            args.bloom_file = bloom_path;
            config.bloom_file = bloom_path;
            BrainwalletSetup::save_config(config);
        }
    } else if (!found_utxos.empty()) {
        // No bloom filter, but found UTXO dump(s) - offer to build
        Interactive::warning_message("No bloom filter found, but UTXO dump(s) detected!");
        std::cout << "\n";

        std::cout << colors::BRIGHT_WHITE << "Found UTXO dump file(s):\n" << colors::RESET;
        for (size_t i = 0; i < found_utxos.size(); i++) {
            double size_mb = found_utxos[i].second / (1024.0 * 1024.0);
            std::cout << "  [" << (i + 1) << "] " << found_utxos[i].first
                      << " (" << std::fixed << std::setprecision(1) << size_mb << " MB)\n";
        }
        std::cout << "  [" << (found_utxos.size() + 1) << "] Enter a different path\n";
        std::cout << "\n";

        int choice = Interactive::prompt_menu_choice(1, static_cast<int>(found_utxos.size() + 1));

        std::string selected_utxo;
        if (choice >= 1 && choice <= static_cast<int>(found_utxos.size())) {
            selected_utxo = found_utxos[choice - 1].first;
        } else {
            selected_utxo = Interactive::prompt_path("Enter path to UTXO dump (.csv)", true);
            if (selected_utxo.empty() || !std::filesystem::exists(selected_utxo)) {
                Interactive::warning_message("Invalid UTXO dump path.");
                args.go_back = true;
                return args;
            }
        }

        // Build the bloom filter automatically
        std::cout << "\n";
        Interactive::display_section("Building Bloom Filter");

        std::cout << "This will create a bloom filter from the UTXO dump.\n";
        std::cout << "  Input:  " << selected_utxo << "\n";
        std::cout << "  Output: " << default_bloom << "\n\n";

        std::cout << colors::BRIGHT_WHITE << "Configuration:\n" << colors::RESET;
        std::cout << "  Min balance:  0.001 BTC (100,000 satoshis)\n";
        std::cout << "  FP rate:      0.001%\n";
        std::cout << "  Expected:     ~50 million addresses\n\n";

        if (!Interactive::prompt_yes_no("Build bloom filter now?", true)) {
            Interactive::warning_message("Bloom filter setup cancelled.");
            args.go_back = true;
            return args;
        }

        std::cout << "\n";
        Interactive::info_message("Building bloom filter (this may take a few minutes)...");
        std::cout << "\n";

        try {
            auto start_time = std::chrono::steady_clock::now();

            // Configure builder
            ::collider::utxo::UTXOBloomBuilder::Config bloom_config;
            bloom_config.target_fp_rate = 0.00001;  // 0.001%
            bloom_config.expected_elements = 50000000;
            bloom_config.min_satoshis = 100000;  // 0.001 BTC

            ::collider::utxo::UTXOBloomBuilder builder(bloom_config);

            std::cout << "  Filter size:  " << (builder.num_bits() / 8 / 1024 / 1024) << " MB\n";
            std::cout << "  Hash funcs:   " << builder.num_hashes() << "\n\n";

            std::cout << "  Processing CSV...\n";
            builder.process_csv(selected_utxo);

            auto end_time = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

            // Save the bloom filter
            builder.save(default_bloom);

            auto stats = builder.get_stats();

            std::cout << "\n";
            Interactive::status_message("Bloom filter built successfully!", true);
            std::cout << "  Addresses:    " << BrainwalletSetup::format_number(stats.elements_added) << "\n";
            std::cout << "  Filter size:  " << stats.size_mb << " MB\n";
            std::cout << "  FP rate:      " << std::fixed << std::setprecision(4) << (stats.estimated_fp_rate * 100) << "%\n";
            std::cout << "  Fill ratio:   " << std::fixed << std::setprecision(1) << (stats.fill_ratio * 100) << "%\n";
            std::cout << "  Time:         " << elapsed << " seconds\n";
            std::cout << "  Output:       " << default_bloom << "\n";

            args.bloom_file = default_bloom;
            config.bloom_file = default_bloom;
            BrainwalletSetup::save_config(config);

        } catch (const std::exception& e) {
            std::cout << "\n";
            Interactive::error_message("Failed to build bloom filter: " + std::string(e.what()));
            args.go_back = true;
            return args;
        }
    } else {
        // No bloom filter or UTXO dump found anywhere
        Interactive::warning_message("No bloom filter or UTXO dump found!");
        std::cout << "\n";
        std::cout << "A bloom filter is required to check addresses efficiently.\n\n";

        std::cout << colors::BRIGHT_WHITE << "Options:\n" << colors::RESET;
        std::cout << "  [1] I have a UTXO dump file (will build bloom filter)\n";
        std::cout << "  [2] I have a bloom filter at a custom path\n";
        std::cout << "  [3] Show me how to get a UTXO dump\n\n";

        std::cout << colors::CYAN << "Select option (1-3): " << colors::RESET;
        std::string input = Interactive::read_line();

        int choice = 0;
        try { choice = std::stoi(input); } catch (...) { choice = 3; }

        if (choice == 1) {
            std::string utxo_path = Interactive::prompt_path("Enter path to UTXO dump (.csv)", true);
            if (utxo_path.empty() || !std::filesystem::exists(utxo_path)) {
                Interactive::warning_message("Invalid UTXO dump path.");
                args.go_back = true;
                return args;
            }

            // Build bloom filter (same code as above)
            std::cout << "\n";
            Interactive::info_message("Building bloom filter...");

            try {
                ::collider::utxo::UTXOBloomBuilder::Config bloom_config;
                bloom_config.target_fp_rate = 0.00001;
                bloom_config.expected_elements = 50000000;
                bloom_config.min_satoshis = 100000;

                ::collider::utxo::UTXOBloomBuilder builder(bloom_config);
                builder.process_csv(utxo_path);
                builder.save(default_bloom);

                Interactive::status_message("Bloom filter built: " + default_bloom, true);
                args.bloom_file = default_bloom;
                config.bloom_file = default_bloom;
                BrainwalletSetup::save_config(config);
            } catch (const std::exception& e) {
                Interactive::error_message("Failed: " + std::string(e.what()));
                args.go_back = true;
                return args;
            }
        } else if (choice == 2) {
            std::string bloom_path = Interactive::prompt_path("Enter path to bloom filter (.blf)", true);
            if (bloom_path.empty() || !std::filesystem::exists(bloom_path)) {
                Interactive::warning_message("Invalid bloom filter path.");
                args.go_back = true;
                return args;
            }
            args.bloom_file = bloom_path;
            config.bloom_file = bloom_path;
            BrainwalletSetup::save_config(config);
        } else {
            std::cout << "\n";
            Interactive::info_message("How to get a UTXO dump:");
            std::cout << "\n";
            std::cout << "  1. Download bitcoin-utxo-dump: github.com/in3rsha/bitcoin-utxo-dump\n";
            std::cout << "  2. Run against your Bitcoin Core data directory\n";
            std::cout << "  3. Place the resulting CSV file in this directory\n";
            std::cout << "  4. Restart theCollider - it will auto-detect and build the bloom filter\n\n";
            std::cout << "  Alternative: Download a pre-built UTXO dump from trusted sources.\n\n";
            args.go_back = true;
            return args;
        }
    }

    // Validate we have what we need
    std::cout << "\n";
    if (args.wordlist_file.empty()) {
        Interactive::warning_message("No wordlist configured. Run setup wizard first.");
        args.go_back = true;
        return args;
    }

    // Check for saved state and offer auto-resume
    if (BrainWalletStateManager::has_saved_state()) {
        auto saved_state = BrainWalletStateManager::load_state();
        if (saved_state.valid) {
            // Check if wordlist matches
            bool wordlist_matches = BrainWalletStateManager::verify_wordlist(saved_state, args.wordlist_file);

            Interactive::display_section("Resume Previous Session?");

            std::cout << colors::BRIGHT_CYAN << "Found saved progress:\n" << colors::RESET;
            std::cout << "  Session:      " << saved_state.session_id << "\n";
            std::cout << "  Last saved:   " << saved_state.timestamp << "\n";
            std::cout << "  Checked:      " << ::format_number_human(saved_state.total_checked) << " passphrases\n";
            std::cout << "  Progress:     Word " << ::format_number(saved_state.current_word_idx)
                      << " / " << ::format_number(saved_state.wordlist_size) << "\n";
            std::cout << "  Phase:        " << saved_state.current_phase
                      << " (iteration " << saved_state.phase_iteration << ")\n";
            if (saved_state.hits_found > 0) {
                std::cout << colors::BRIGHT_GREEN << "  Hits found:   " << saved_state.hits_found << colors::RESET << "\n";
            }
            std::cout << "\n";

            if (!wordlist_matches) {
                Interactive::warning_message("Wordlist has changed since last session!");
                std::cout << "  Previous: " << saved_state.wordlist_path << "\n";
                std::cout << "  Current:  " << args.wordlist_file << "\n\n";

                if (Interactive::prompt_yes_no("Start fresh with new wordlist?", true)) {
                    BrainWalletStateManager::clear_state();
                    Interactive::info_message("Previous state cleared. Starting fresh.");
                    args.resume = false;
                } else {
                    Interactive::info_message("Please use the same wordlist to resume, or start fresh.");
                    args.go_back = true;
                    return args;
                }
            } else {
                // Wordlist matches - offer to resume
                if (Interactive::prompt_yes_no("Resume from saved progress?", true)) {
                    args.resume = true;
                    Interactive::status_message("Will resume from saved state", true);
                } else {
                    if (Interactive::prompt_yes_no("Clear saved state and start fresh?", false)) {
                        BrainWalletStateManager::clear_state();
                        Interactive::info_message("Previous state cleared.");
                    }
                    args.resume = false;
                }
            }
            std::cout << "\n";
        }
    }

    Interactive::display_section("Ready to Scan");

    std::cout << colors::BRIGHT_WHITE << "Configuration Summary:\n" << colors::RESET;
    std::cout << "  Wordlist:     " << ::normalize_path(args.wordlist_file) << "\n";
    std::cout << "  Bloom filter: " << args.bloom_file << "\n";
    std::cout << "  Entries:      " << BrainwalletSetup::format_number(config.total_unique_lines) << " passphrases\n";
    std::cout << "\n";

    if (!Interactive::prompt_yes_no("Start brain wallet scan?", true)) {
        args.go_back = true;
        return args;
    }

    return args;
}
#endif  // COLLIDER_PRO

Arguments run_interactive_mode(Arguments base_args, double gpu_speed_mkeys) {
    using namespace ::collider::ui;
    Arguments args = base_args;

    while (true) {
        // Reset navigation flags
        args.go_back = false;

        // Display main menu
        MainMenuChoice choice = Interactive::display_main_menu("1.4.0");

        switch (choice) {
            case MainMenuChoice::PUZZLE_MODE: {
                args = run_puzzle_interactive(args, gpu_speed_mkeys);
                if (args.go_back) {
                    continue;  // Return to main menu
                }
                return args;
            }

            case MainMenuChoice::BRAINWALLET_MODE: {
#ifdef COLLIDER_PRO
                args = run_brainwallet_interactive(args);
                if (args.go_back) {
                    continue;  // Return to main menu
                }
                return args;
#else
                std::cout << "\n[PRO] Brain wallet scanning requires theCollider Pro.\n"
                          << "      Purchase at: https://collisionprotocol.com/pro\n\n";
                continue;  // Return to main menu
#endif
            }

            case MainMenuChoice::BENCHMARK_MODE:
                args.benchmark = true;
                std::cout << "\n";
                Interactive::info_message("Starting GPU performance benchmark...");
                return args;

            case MainMenuChoice::SHOW_HELP:
                args.help = true;
                return args;

            case MainMenuChoice::EXIT:
                args.exit_program = true;
                std::cout << "\n";
                Interactive::info_message("Goodbye!");
                return args;

            default:
                args.exit_program = true;
                return args;
        }
    }
}

void enable_windows_ansi() {
#ifdef _WIN32
    // Enable virtual terminal processing for ANSI escape codes
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hOut, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hOut, dwMode);
        }
    }
    // Also enable for stderr
    HANDLE hErr = GetStdHandle(STD_ERROR_HANDLE);
    if (hErr != INVALID_HANDLE_VALUE) {
        DWORD dwMode = 0;
        if (GetConsoleMode(hErr, &dwMode)) {
            dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
            SetConsoleMode(hErr, dwMode);
        }
    }
#endif
}

}  // namespace collider::ui
