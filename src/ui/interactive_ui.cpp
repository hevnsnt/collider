// interactive_ui.cpp: theCollider's interactive startup flow.
// Hosts the menu prompts dispatched from main(); the helpers it relies
// on (format_number_human, normalize_path, analyze_puzzle,
// get_best_puzzle) live in collider::runtime / core/puzzle_analysis.

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

#include "core/paths.hpp"
#include "core/puzzle_analysis.hpp"
#include "core/puzzle_config.hpp"
#include "core/version.hpp"
#include "core/yaml_config.hpp"
#include "tools/utxo_bloom_builder.hpp"
#include "ui/banner.hpp"
#include "ui/interactive.hpp"
#include "ui/tui/menu/main_menu.hpp"     // TR-1: TUI-native main menu
#include "ui/tui/menu/mode_config.hpp"   // TR-1-full: mode-config modals

#ifdef COLLIDER_PRO
#include "core/brainwallet_state.hpp"
#include "ui/brainwallet_setup.hpp"
#endif

#include "runtime/format.hpp"
#include "runtime/bip_scanner_runner.hpp"  // resolve_* auto-detect helpers

// format_number is still defined in runtime/puzzle_solver.cpp; forward
// declare it here so we keep linking against the existing TU.
std::string format_number(uint64_t n);

using collider::runtime::format_number_human;
using collider::runtime::normalize_path;

namespace collider::ui {

#ifdef COLLIDER_PRO
// Auto-detect a funded-address bloom filter for opportunistic scanning in
// pool / standalone-puzzle modes (the on-DP H160 probe described in PRO.md).
//
// Behavior diverges from the brain-wallet flow's bloom block in one key
// way: opportunistic scanning is optional. Brain-wallet mode REQUIRES a
// bloom (the entire pipeline is built around the probe); pool/puzzle
// modes work fine without one (you just don't get the side-channel scan).
// So when no .blf is found here we print a brief note and continue,
// rather than blocking the user with a UTXO-build wizard.
//
// Search order matches BrainWalletRunner::auto_detect_bloom_files: CWD,
// CWD parent, well-known Windows roots; then the canonical
// funded_addresses.blf name from PRO.md before the historical seen.blf.
static void maybe_pick_opportunistic_bloom(Arguments& args,
                                           const std::string& mode_label) {
    if (!args.bloom_file.empty()) {
        // Already set via CLI flag or ~/.collider/config.yml; respect
        // the operator's explicit choice without re-prompting.
        return;
    }

    std::vector<std::string> search_dirs;
    search_dirs.emplace_back(".");
    search_dirs.emplace_back("..");
#ifdef _WIN32
    search_dirs.emplace_back("D:\\theCollider");
    search_dirs.emplace_back("C:\\theCollider");
    search_dirs.emplace_back("D:\\");
    search_dirs.emplace_back("C:\\");
#endif

    // Probe the canonical names first; a found_blooms scan catches any
    // operator-renamed file in the same directories.
    static const char* const kCanonicalNames[] = {
        "funded_addresses.blf",
        "seen.blf",
        "seen_tight.blf",
    };
    for (const auto& dir : search_dirs) {
        std::error_code ec;
        if (!std::filesystem::exists(dir, ec) || ec) continue;
        for (const char* name : kCanonicalNames) {
            auto candidate = std::filesystem::path(dir) / name;
            if (std::filesystem::exists(candidate, ec) && !ec) {
                args.bloom_file = candidate.string();
                // Silent -- opportunistic-bloom info shows up in the
                // confirm/dashboard later; no cout drop-out here.
                return;
            }
        }
    }

    // Wider sweep: any *.blf in the search roots. Useful when the
    // operator built a custom bloom from their own UTXO dump.
    std::vector<std::pair<std::string, size_t>> found;
    for (const auto& dir : search_dirs) {
        std::error_code ec;
        if (!std::filesystem::exists(dir, ec) || ec) continue;
        try {
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (!entry.is_regular_file()) continue;
                auto ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(),
                               [](unsigned char c) {
                                   return static_cast<char>(std::tolower(c));
                               });
                if (ext == ".blf") {
                    found.emplace_back(entry.path().string(),
                                       entry.file_size());
                }
            }
        } catch (const std::exception&) {
            // Skip unreadable directories silently.
        }
    }
    // Deduplicate by canonical path.
    {
        std::set<std::string> seen;
        found.erase(std::remove_if(found.begin(), found.end(),
            [&](const auto& p) {
                std::string canon;
                try { canon = std::filesystem::canonical(p.first).string(); }
                catch (...) { canon = p.first; }
                return !seen.insert(canon).second;
            }), found.end());
    }

    if (found.empty()) {
        // Silent: opportunistic scanning is best-effort. Informing
        // the operator here would dump cout between TUI screens; the
        // dashboard PROFILE row makes the inactive state visible.
        return;
    }

    if (found.size() == 1) {
        args.bloom_file = found[0].first;
        return;  // silent
    }

    // Multiple custom blooms: T1-A polish: TUI picker_modal in place
    // of the cout/cin "Continue without opportunistic scanning" menu.
    std::vector<std::string> labels;
    labels.reserve(found.size() + 1);
    for (const auto& [path, size] : found) {
        double size_mb = size / (1024.0 * 1024.0);
        std::ostringstream lbl;
        lbl << path << "  (" << std::fixed << std::setprecision(1)
            << size_mb << " MB)";
        labels.push_back(lbl.str());
    }
    labels.push_back("Continue without opportunistic scanning");
    const int pick = ::collider::ui::tui::menu::picker_modal(
        std::string("OPPORTUNISTIC BLOOM (") + mode_label + ")",
        "Pro mode: derive each candidate pubkey against the chosen "
        "bloom filter to opportunistically catch funded-address hits. "
        "Pick a bloom file or continue without the side-channel scan.",
        labels);
    if (pick == 0 || pick == static_cast<int>(labels.size())) {
        return;  // silent -- dashboard PROFILE shows inactive
    }
    args.bloom_file = found[pick - 1].first;
    // Silent -- the path is in the upcoming confirm modal's kv-rows.
}
#endif  // COLLIDER_PRO

Arguments run_puzzle_interactive(Arguments base_args, double gpu_speed_mkeys) {
    using namespace ::collider::ui;
    Arguments args = base_args;
    args.puzzle_mode = true;

    // TR-1-full: TUI standalone/pool picker. Replaces the cout-prompt
    // "Choose mode: [1] Standalone [2] Join Pool [0] Back".
    const int mode_pick = ::collider::ui::tui::menu::puzzle_mode_picker_modal();
    if (mode_pick == 0) {
        args.go_back = true;  // Back to main menu
        return args;
    }
    const bool use_pool = (mode_pick == 2);

    if (use_pool) {
        // T1-B polish: TUI pool config modal in place of the cout
        // "Pool URL: ... Worker: ..." cin chain. Keeps the operator
        // inside FTXUI from main-menu to scan view.
        std::string pool_url, worker;
        if (!::collider::ui::tui::menu::pool_config_modal(
                pool_url, worker, args.pool_url, args.pool_worker)) {
            args.go_back = true;
            return args;
        }
        args.pool_mode = true;
        args.pool_url = pool_url;
        args.pool_worker = worker;

        // persist the BTC payout address to ~/.collider/config.yml
        // so the next launch defaults to it (user can still override at the
        // prompt). Only writes when the file doesn't already exist; never
        // overwrites an existing operator-managed config.
        const std::string saved =
            collider::AppConfig::save_pool_worker(worker, pool_url);
        if (!saved.empty()) {
            // Silent: cache write happened but we don't cout about it.
            (void)saved;
        }

        // Pool mode doesn't need puzzle selection -- pool assigns work.
        // The pool_config_modal above already captured the URL + worker
        // pair; connect immediately without an extra Enter keystroke.
#ifdef COLLIDER_PRO
        // Pro: offer the opportunistic-bloom side-channel before connecting.
        // Skipped silently for Free; --bloom is Pro-gated at the CLI parser.
        maybe_pick_opportunistic_bloom(args, "Pool mode");
#endif
        // Silent connect -- the pool runtime emits its own status via
        // session_log + stdio_capture into ~/.collider/logs.
        return args;
    }

    // T1-B polish: TUI puzzle number picker in place of the cout
    // "Enter puzzle number (1-256) or 'auto'" prompt.
    int puzzle_choice = ::collider::ui::tui::menu::puzzle_number_modal();
    if (puzzle_choice == 0) {
        args.go_back = true;
        return args;
    }

    if (puzzle_choice == -1) {
        // Auto mode - use smart selection
        int best = ::get_best_puzzle(gpu_speed_mkeys);
        if (best > 0) {
            args.puzzle_number = best;
            const ::collider::PuzzleInfo* puzzle = ::collider::PuzzleDatabase::get_puzzle(best);
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

                if (has_pubkey) {
                    args.puzzle_kangaroo = true;
                }

                // T1-B: TUI confirm in place of cout-based prompt.
                // The "Analyzing puzzles..." + display_puzzle_info
                // cout block was removed; the confirm modal renders
                // the same info in the kv-rows below.
                std::vector<std::pair<std::string, std::string>> auto_kv = {
                    {"Puzzle", "#" + std::to_string(best)
                                + " (smart-selected)"},
                    {"Bits", std::to_string(puzzle->bits)},
                    {"Reward", std::to_string(puzzle->btc_reward) + " BTC"},
                    {"Algorithm", has_pubkey ? "Kangaroo (pubkey known)"
                                             : "BSGS"},
                    {"Estimated time", est_time},
                };
                const auto auto_confirm =
                    ::collider::ui::tui::menu::confirm_config_modal(
                        "PUZZLE #" + std::to_string(best) + " -- CONFIRM",
                        "Smart-selected based on your GPU throughput. "
                        "Standalone solver runs locally against the "
                        "puzzle's full keyspace. Solution writes to "
                        "puzzle_<N>.txt on success.",
                        auto_kv,
                        "Start solver");
                if (auto_confirm ==
                    ::collider::ui::tui::menu::ConfirmResult::Back) {
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
                // T1-B polish: TUI yes/no for the testing-mode prompt.
                // The cout warning was removed; the modal title +
                // intro already surface the "is already SOLVED" fact.
                const int testing_pick =
                    ::collider::ui::tui::menu::yes_no_modal(
                        "PUZZLE #" + std::to_string(puzzle_choice)
                            + " IS ALREADY SOLVED",
                        "Solution: " + puzzle->solution_hex + "\n\n"
                        "Continue in testing mode? The solver will run "
                        "as if the puzzle were unsolved and verify "
                        "the recovered key matches the known solution. "
                        "Useful for benchmarking or smoke-testing.",
                        /*default_yes=*/true,
                        "Continue in testing mode",
                        "Back to main menu");
                if (testing_pick != 1) {
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

            if (has_pubkey) {
                args.puzzle_kangaroo = true;
            }

            // T1-B: TUI confirm in place of cout-based prompt. The
            // display_puzzle_info cout block + "Auto-enabled Kangaroo"
            // info_message were removed; the modal kv-rows surface
            // the same info inside the alt-screen.
            std::vector<std::pair<std::string, std::string>> std_kv = {
                {"Puzzle", "#" + std::to_string(puzzle_choice)},
                {"Bits", std::to_string(puzzle->bits)},
                {"Reward", std::to_string(puzzle->btc_reward) + " BTC"},
                {"Algorithm", has_pubkey ? "Kangaroo (pubkey known)"
                                         : "BSGS"},
                {"Estimated time", est_time},
            };
            if (puzzle->solved) {
                std_kv.push_back({"Status", "SOLVED (testing mode)"});
            }
            const auto std_confirm =
                ::collider::ui::tui::menu::confirm_config_modal(
                    "PUZZLE #" + std::to_string(puzzle_choice) + " -- CONFIRM",
                    "Standalone solver runs locally against the puzzle's "
                    "full keyspace. Solution writes to puzzle_<N>.txt on "
                    "success.",
                    std_kv,
                    "Start solver");
            if (std_confirm ==
                ::collider::ui::tui::menu::ConfirmResult::Back) {
                args.go_back = true;
                return args;
            }
        } else {
            // Unknown puzzle: bounce back to main menu silently. The
            // puzzle_number_modal validates 1..256 so this only fires
            // on a database row that's missing for that number.
            args.go_back = true;
            return args;
        }
    }

#ifdef COLLIDER_PRO
    // Pro: same opportunistic-bloom side-channel for standalone puzzle
    // solving as for pool mode. Only meaningful when the RCKangaroo
    // backend ends up handling the puzzle (it's the only backend that
    // currently wires the bloom probe; MultiGPU/CPU kangaroo and the
    // bruteforce path ignore the flag today). Offering the prompt
    // anyway lets a user who switches backends mid-development still
    // pick up the bloom, and makes the feature reachable through the
    // standard interactive entry path per docs/PRO.md.
    maybe_pick_opportunistic_bloom(args, "Standalone puzzle mode");
#endif

    return args;
}

#ifdef COLLIDER_PRO
Arguments run_brainwallet_interactive(Arguments base_args) {
    using namespace ::collider::ui;
    Arguments args = base_args;
    args.brainwallet_mode = true;
    args.pool_mode = false;  // Disable pool mode - brainwallet is mutually exclusive

    // No display_section here -- the section heading would dump cout
    // before the first TUI modal. Modal title already identifies the
    // mode for the operator.

    // Check if this is first run - need to set up wordlists
    BrainwalletConfig config;
    bool first_run = !BrainwalletSetup::is_setup_complete();

    if (first_run) {
        // T1-A polish: TUI yes/no in place of the cout first-run prompt.
        const int pick = ::collider::ui::tui::menu::yes_no_modal(
            "BRAIN WALLET -- FIRST-TIME SETUP",
            "No wordlist + bloom configuration was found in "
            "~/.thecollider/. Run the setup wizard to choose source "
            "directories, build a combined wordlist, and register a "
            "bloom filter. The wizard can also be re-run later with "
            "`./collider --brainwallet-setup`.",
            /*default_yes=*/true,
            "Run setup wizard",
            "Back to main menu");
        if (pick != 1) {
            args.go_back = true;
            return args;
        }
        config = BrainwalletSetup::run_wizard();
        if (!config.setup_complete) {
            Interactive::warning_message("Setup was not completed.");
            args.go_back = true;
            return args;
        }
    } else {
        // T1-A polish: TUI reconfigure picker in place of the cout
        // "Reconfigure wordlists? (y/N)" prompt. The picker shows the
        // current wordlist + entry count + bloom inline so the
        // operator's decision is informed at the same screen, and the
        // overall flow stays inside the TUI alt-screen without
        // flashing back to raw shell text.
        config = BrainwalletSetup::load_config();
        const std::string entries_summary =
            BrainwalletSetup::format_number(config.total_unique_lines) +
            " passphrases (" + std::to_string(config.wordlist_dirs.size()) +
            " sources)";
        const int reconfig_pick =
            ::collider::ui::tui::menu::brainwallet_reconfigure_modal(
                config.processed_wordlist,
                entries_summary,
                config.bloom_file);
        if (reconfig_pick == 0) {
            args.go_back = true;
            return args;
        }
        if (reconfig_pick == 2) {
            config = BrainwalletSetup::run_wizard();
        }
    }

    // Set the wordlist from config (silent -- the final confirm modal
    // shows the resolved path in its kv-rows; no cout here).
    if (!config.processed_wordlist.empty() &&
        std::filesystem::exists(config.processed_wordlist)) {
        args.wordlist_file = config.processed_wordlist;
    }

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

    // Check config first, then search results. Auto-detected paths
    // are silent -- the final confirm modal renders them in kv-rows.
    if (!config.bloom_file.empty() && std::filesystem::exists(config.bloom_file)) {
        args.bloom_file = config.bloom_file;
    } else if (std::filesystem::exists(default_bloom)) {
        args.bloom_file = default_bloom;
    } else if (!found_blooms.empty()) {
        // T1-A polish: TUI picker for bloom filter discovery.
        std::vector<std::string> labels;
        labels.reserve(found_blooms.size() + 1);
        for (const auto& [path, size] : found_blooms) {
            double size_mb = size / (1024.0 * 1024.0);
            std::ostringstream lbl;
            lbl << path << "  (" << std::fixed << std::setprecision(1)
                << size_mb << " MB)";
            labels.push_back(lbl.str());
        }
        labels.push_back("Enter a different path...");
        const int pick = ::collider::ui::tui::menu::picker_modal(
            "BLOOM FILTER -- AUTO-DETECTED",
            "Multiple .blf files were found on disk. Pick one to use, "
            "or supply a custom path on the next screen. The chosen "
            "path is persisted to your brainwallet config so the next "
            "launch starts on it automatically.",
            labels);
        if (pick == 0) {
            args.go_back = true;
            return args;
        }
        if (pick <= static_cast<int>(found_blooms.size())) {
            args.bloom_file = found_blooms[pick - 1].first;
            config.bloom_file = args.bloom_file;
            BrainwalletSetup::save_config(config);
        } else {
            auto entered = ::collider::ui::tui::menu::text_input_modal(
                "BLOOM FILTER -- CUSTOM PATH",
                "Enter the absolute path to a .blf bloom filter built "
                "by build_bloom.exe (or any compatible tool). Path "
                "must exist on disk; Esc to go back.",
                "Path",
                /*default_value=*/"",
                /*must_be_existing_path=*/true);
            if (!entered || entered->empty()) {
                args.go_back = true;
                return args;
            }
            args.bloom_file = *entered;
            config.bloom_file = *entered;
            BrainwalletSetup::save_config(config);
        }
    } else if (!found_utxos.empty()) {
        // T1-A polish: TUI picker for UTXO dump discovery + TUI
        // confirm before kicking off the (potentially multi-minute)
        // bloom build. Replaces the cout/cin menu chain.
        std::vector<std::string> utxo_labels;
        utxo_labels.reserve(found_utxos.size() + 1);
        for (const auto& [path, size] : found_utxos) {
            double size_mb = size / (1024.0 * 1024.0);
            std::ostringstream lbl;
            lbl << path << "  (" << std::fixed << std::setprecision(1)
                << size_mb << " MB)";
            utxo_labels.push_back(lbl.str());
        }
        utxo_labels.push_back("Enter a different path...");
        const int upick = ::collider::ui::tui::menu::picker_modal(
            "UTXO DUMP -- AUTO-DETECTED",
            "No bloom filter was found, but UTXO dump CSV file(s) "
            "were. Pick one to build a bloom filter from, or supply a "
            "custom path on the next screen. Building a bloom filter "
            "can take a few minutes for a full UTXO set.",
            utxo_labels);
        if (upick == 0) {
            args.go_back = true;
            return args;
        }
        std::string selected_utxo;
        if (upick <= static_cast<int>(found_utxos.size())) {
            selected_utxo = found_utxos[upick - 1].first;
        } else {
            auto entered = ::collider::ui::tui::menu::text_input_modal(
                "UTXO DUMP -- CUSTOM PATH",
                "Enter the absolute path to a UTXO dump .csv "
                "(bitcoin-utxo-dump format). Path must exist on disk; "
                "Esc to go back.",
                "Path", "", /*must_be_existing_path=*/true);
            if (!entered || entered->empty()) {
                args.go_back = true;
                return args;
            }
            selected_utxo = *entered;
        }

        // Confirm-build modal.
        std::vector<std::pair<std::string, std::string>> bcfg = {
            {"UTXO dump", selected_utxo},
            {"Output bloom", default_bloom},
            {"Min balance", "0.001 BTC (100,000 sat)"},
            {"FP rate", "0.001%"},
            {"Expected entries", "~50 million addresses"},
        };
        const auto bconf = ::collider::ui::tui::menu::confirm_config_modal(
            "BUILD BLOOM FILTER -- CONFIRM",
            "Streams the UTXO CSV and inserts every address whose "
            "balance is at least min-balance into the bloom. Build "
            "time scales with UTXO set size; expect a few minutes "
            "for a full mainnet dump.",
            bcfg,
            "Build");
        if (bconf == ::collider::ui::tui::menu::ConfirmResult::Back) {
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
        // T1-A polish: TUI 3-option picker for the no-bloom-found
        // branch. Replaces the cout "Options: [1] [2] [3]" menu +
        // read_line + nested prompts.
        const int npick = ::collider::ui::tui::menu::picker_modal(
            "BLOOM FILTER -- NOT FOUND",
            "A bloom filter is required to check addresses against "
            "the funded-set efficiently. Pick how to provide one. "
            "Esc to go back to the main menu.",
            {
                "I have a UTXO dump file (will build a bloom)",
                "I have a bloom filter at a custom path",
                "Show me how to get a UTXO dump",
            });
        if (npick == 0) {
            args.go_back = true;
            return args;
        }
        if (npick == 1) {
            auto utxo_entered =
                ::collider::ui::tui::menu::text_input_modal(
                    "UTXO DUMP -- PATH",
                    "Enter the absolute path to a UTXO dump .csv "
                    "(bitcoin-utxo-dump format).",
                    "Path", "", /*must_be_existing_path=*/true);
            if (!utxo_entered || utxo_entered->empty()) {
                args.go_back = true;
                return args;
            }

            // Build bloom filter (same code as above).
            try {
                ::collider::utxo::UTXOBloomBuilder::Config bloom_config;
                bloom_config.target_fp_rate = 0.00001;
                bloom_config.expected_elements = 50000000;
                bloom_config.min_satoshis = 100000;

                ::collider::utxo::UTXOBloomBuilder builder(bloom_config);
                builder.process_csv(*utxo_entered);
                builder.save(default_bloom);

                args.bloom_file = default_bloom;
                config.bloom_file = default_bloom;
                BrainwalletSetup::save_config(config);
            } catch (const std::exception& e) {
                // Fatal build failure: surface to operator.
                std::cerr << "[!] Bloom build failed: " << e.what() << "\n";
                args.go_back = true;
                return args;
            }
        } else if (npick == 2) {
            auto bloom_entered =
                ::collider::ui::tui::menu::text_input_modal(
                    "BLOOM FILTER -- CUSTOM PATH",
                    "Enter the absolute path to an existing .blf "
                    "bloom filter.",
                    "Path", "", /*must_be_existing_path=*/true);
            if (!bloom_entered || bloom_entered->empty()) {
                args.go_back = true;
                return args;
            }
            args.bloom_file = *bloom_entered;
            config.bloom_file = *bloom_entered;
            BrainwalletSetup::save_config(config);
        } else {
            // "Show me how to get a UTXO dump" -- informational
            // confirm_config_modal (no input needed; Back returns).
            std::vector<std::pair<std::string, std::string>> kv = {
                {"Step 1", "Download bitcoin-utxo-dump"},
                {"   URL", "github.com/in3rsha/bitcoin-utxo-dump"},
                {"Step 2", "Run against your Bitcoin Core data dir"},
                {"Step 3", "Place the .csv next to this binary"},
                {"Step 4", "Restart theCollider; auto-detect kicks in"},
                {"Alt",    "Download a pre-built dump from trusted sources"},
            };
            ::collider::ui::tui::menu::confirm_config_modal(
                "HOW TO GET A UTXO DUMP",
                "theCollider's bloom filter is built from a Bitcoin "
                "Core UTXO dump CSV. The steps below produce one. "
                "After you have the CSV in place, restart and the "
                "auto-detect path picks it up.",
                kv,
                "OK -- back to menu");
            args.go_back = true;
            return args;
        }
    }

    // Validate we have what we need. Empty wordlist is a degenerate
    // state (setup wizard never completed or its output was deleted);
    // surface via a TUI yes/no instead of a cout warning.
    if (args.wordlist_file.empty()) {
        ::collider::ui::tui::menu::yes_no_modal(
            "BRAIN WALLET -- NO WORDLIST",
            "No wordlist is configured. Run `./collider --brainwallet"
            "-setup` from the shell, or pick Reconfigure on the next "
            "Brain Wallet entry. Esc to return to the main menu.",
            /*default_yes=*/false,
            "OK -- back to menu",
            "OK -- back to menu");
        args.go_back = true;
        return args;
    }

    // T1-A: TUI resume modal in place of the cout-based three-prompt
    // chain. brainwallet_resume_modal() collapses the cut/cin chain
    // (Resume? / Start fresh? / Clear saved state?) into one TUI
    // screen with the saved-state summary visible at decision time.
    if (BrainWalletStateManager::has_saved_state()) {
        auto saved_state = BrainWalletStateManager::load_state();
        if (saved_state.valid) {
            const bool wordlist_matches =
                BrainWalletStateManager::verify_wordlist(
                    saved_state, args.wordlist_file);
            const std::string progress = "Word "
                + ::format_number(saved_state.current_word_idx)
                + " / "
                + ::format_number(saved_state.wordlist_size);
            const std::string phase = saved_state.current_phase
                + " (iteration "
                + std::to_string(saved_state.phase_iteration) + ")";
            const int resume_pick =
                ::collider::ui::tui::menu::brainwallet_resume_modal(
                    saved_state.session_id,
                    saved_state.timestamp,
                    ::format_number_human(saved_state.total_checked),
                    progress,
                    phase,
                    saved_state.hits_found,
                    wordlist_matches);
            if (resume_pick == 1) {  // Resume
                args.resume = true;
            } else if (resume_pick == 2) {  // Discard + fresh
                BrainWalletStateManager::clear_state();
                args.resume = false;
            } else {  // Back to main menu
                args.go_back = true;
                return args;
            }
        }
    }

    // T1-A polish: TUI yes/no for multi-address derivation in place of
    // the cout "Enable multi-address derivation? (Y/n)" prompt that
    // used to flash the operator back to raw shell text in the middle
    // of an otherwise-TUI flow.
    {
        const int pick = ::collider::ui::tui::menu::yes_no_modal(
            "BRAIN WALLET -- ADDRESS COVERAGE",
            "Pro probes each candidate pubkey against three Bitcoin "
            "H160 paths: compressed P2PKH / P2WPKH (BIP-84), "
            "uncompressed P2PKH, and P2SH-P2WPKH (BIP-49). "
            "Multi-address mode adds two extra per-pubkey hashes for "
            "the uncompressed-P2PKH and P2SH-P2WPKH paths inside the "
            "fused kernel. Recommended unless throughput is the only "
            "concern.",
            /*default_yes=*/true,
            "Enable multi-address (recommended)",
            "Single-address only (compressed P2PKH / P2WPKH)");
        if (pick == 0) {
            args.go_back = true;
            return args;
        }
        args.brainwallet_v2_mode = (pick == 1);
    }

    // T1-A: TUI confirm modal in place of cout-based summary + yes/no.
    // Same kv-rows pattern the BIP scan flow uses so the visual style is
    // consistent across modes. The cout::display_section line above
    // ("Address-Type Coverage") is the last raw shell prompt the
    // operator sees before the in-session TUI takes over; everything
    // post-confirmation drops straight into the FTXUI scan view.
    std::vector<std::pair<std::string, std::string>> bw_kv = {
        {"Wordlist", ::normalize_path(args.wordlist_file)},
        {"Bloom filter", args.bloom_file},
        {"Entries",
         BrainwalletSetup::format_number(config.total_unique_lines)
             + " passphrases"},
        {"Address mode",
         args.brainwallet_v2_mode ? "multi (3 H160 paths)"
                                  : "single (compressed P2PKH)"},
    };
    if (args.resume) {
        bw_kv.push_back({"Mode", "Resume from saved state"});
    }
    const auto bw_choice = ::collider::ui::tui::menu::confirm_config_modal(
        "BRAIN WALLET SCAN -- CONFIRM",
        "Iterates every passphrase in the wordlist, derives the brain "
        "wallet key, computes hash160 for each enabled address path, "
        "and probes the bloom filter for funded addresses. Hits land "
        "in hits/<timestamp>.json next to the binary.",
        bw_kv,
        "Start scan");
    if (bw_choice == ::collider::ui::tui::menu::ConfirmResult::Back) {
        args.go_back = true;
        return args;
    }
    return args;
}

// BIP-39 / BIP-32 mnemonic scanner interactive menu. Walks the
// operator through picking a candidate-phrase file + the UTXO bloom,
// then routes to run_bip_scan_mode. Files are validated for existence
// before we commit; the scanner re-validates per spec (BIP-39
// checksum) line by line so a typoed phrase doesn't block the run.
Arguments run_bip_scan_interactive(Arguments base_args) {
    using namespace ::collider::ui;
    Arguments args = base_args;
    args.bip_scan_mode    = true;
    args.brainwallet_mode = false;
    args.pool_mode        = false;
    args.puzzle_mode      = false;

    // User feedback 2026-05-24: BIP scan was wordlist-driven by default
    // which mismatches the "scan random mnemonics forever" mental model
    // most operators have. Surface the choice up-front via a TUI modal:
    // combinatorial (default) vs wordlist-driven. The combinatorial
    // backend (run_combinatorial_scan) iterates BIP-39 entropy space
    // exhaustively, generating valid mnemonics by construction (no
    // re-rolls), and is checkpoint-resumable across restarts.
    const int source_pick =
        ::collider::ui::tui::menu::bip_scan_source_picker_modal();
    if (source_pick == 0) {
        args.go_back = true;  // Back to main menu
        return args;
    }
    if (source_pick == 1) {
        // Combinatorial / random forever-loop mode. Skip the wordlist
        // prompt entirely; only need the bloom filter. Default to
        // 12 words (most common BIP-39 length, easiest to surface
        // progress on the dashboard); operator can pin a different
        // width via --bip-words on the CLI.
        args.bip_combinatorial = true;
    }

    // Source picker already showed the user the mode they picked;
    // the upcoming confirm modal renders the full config table.
    // No pre-confirm cout text -- it dumps to PowerShell and breaks
    // the all-TUI flow.

    // Auto-detect phrases corpus, bloom filter. The scanner runtime
    // does its own auto-discovery for the BIP-39 wordlist + bloom
    // when called from the CLI; here we mirror the same logic so the
    // interactive flow ASKS only when auto-discovery fails. The
    // wordlist auto-detect is skipped in combinatorial mode where no
    // file is needed.
    if (!args.bip_combinatorial && args.bip_scan_wordlist.empty()) {
        args.bip_scan_wordlist =
            ::collider::runtime::resolve_candidate_phrases();
    }
    if (args.bloom_file.empty()) {
        args.bloom_file = ::collider::runtime::resolve_bloom_filter();
    }
    if (args.bloom_tight_file.empty()) {
        // Mirror the brainwallet path: ~/.collider/seen_tight.blf if
        // present, otherwise skip silently (primary bloom alone).
        std::string default_tight =
            (collider::paths::collider_home() / "seen_tight.blf").string();
        if (std::filesystem::exists(default_tight)) {
            args.bloom_tight_file = default_tight;
        }
    }

    if (!args.bip_combinatorial) {
        if (args.bip_scan_wordlist.empty()) {
            // T1-A polish: TUI text-input modal in place of the cout
            // "Path to phrases file: " prompt.
            const std::string default_hint =
                (collider::paths::collider_home() / "wordlists" /
                 "bip_phrases.txt").string();
            auto entered = ::collider::ui::tui::menu::text_input_modal(
                "BIP-39 PHRASES -- PATH",
                "No candidate phrases file was auto-detected. Provide "
                "the absolute path to a text file with one BIP-39 "
                "mnemonic phrase per line (whitespace-separated "
                "words). A typical drop location is " + default_hint +
                ".",
                "Path", "", /*must_be_existing_path=*/true);
            if (!entered || entered->empty()) {
                args.go_back = true;
                return args;
            }
            args.bip_scan_wordlist = *entered;
        }
        // Auto-detected case: silent. Confirm modal kv-rows show the
        // resolved path so the operator sees what's being used.
    }

    if (args.bloom_file.empty()) {
        // T1-A polish: TUI text-input modal for the bloom-filter path.
        const std::string default_hint =
            (collider::paths::collider_home() / "funded_addresses.blf").string();
        auto entered = ::collider::ui::tui::menu::text_input_modal(
            "BIP-39 SCAN -- BLOOM FILTER PATH",
            "No bloom .blf was auto-detected. Provide the absolute "
            "path to an existing .blf built by build_bloom. A typical "
            "drop location is " + default_hint + ".",
            "Path", "", /*must_be_existing_path=*/true);
        if (!entered || entered->empty()) {
            args.go_back = true;
            return args;
        }
        args.bloom_file = *entered;
    }
    // Auto-detected bloom + tight bloom: silent. Confirm modal kv-rows
    // show the resolved paths so the operator sees what's being used
    // without a cout drop-out before the modal.

    // TR-1-full: replace cout summary + Y/N prompt with a TUI confirm
    // modal that renders the same key/value table inside an alt-screen
    // bordered panel + Start/Back picker. Keeps the operator inside a
    // TUI context all the way from the main menu through to the in-
    // session brainwallet TUI.
    std::vector<std::pair<std::string, std::string>> kv;
    if (args.bip_combinatorial) {
        kv.push_back({"Mode", "Combinatorial (random / forever)"});
        kv.push_back({"Word count",
                      std::to_string(args.bip_combinatorial_word_count)});
        kv.push_back({"Bloom filter", args.bloom_file});
    } else {
        kv.push_back({"Mode", "Wordlist-driven"});
        kv.push_back({"Phrases file", args.bip_scan_wordlist});
        kv.push_back({"Bloom filter", args.bloom_file});
    }
    if (!args.bloom_tight_file.empty()) {
        kv.push_back({"Tight bloom", args.bloom_tight_file});
    }
    const std::string confirm_intro = args.bip_combinatorial
        ? "Walks every BIP-39 entropy value of the chosen width. Each "
          "entropy maps to exactly one valid mnemonic by construction "
          "so there are no wasted re-rolls. The space is 2^128 (12 "
          "words) and is not physically exhaustible -- progress "
          "checkpoints to ~/.collider/bip_combinatorial.json. Stop with "
          "q at any time. Hits land in bip_hits.txt."
        : "Iterates BIP-39 mnemonic candidates from the phrases file, "
          "derives every historical and modern derivation path, and "
          "probes ~190 addresses per phrase against the bloom filter. "
          "Hits land in bip_hits.txt next to the binary.";
    const auto choice = ::collider::ui::tui::menu::confirm_config_modal(
        "BIP-39 / BIP-32 SCAN -- CONFIRM",
        confirm_intro,
        kv,
        "Start scan");
    if (choice == ::collider::ui::tui::menu::ConfirmResult::Back) {
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

        // TR-1: TUI-native main menu. Replaces the historical cout
        // banner-then-numbered-list with a fullscreen FTXUI picker so
        // the operator never sees the "type a number then Enter"
        // shell-prompt feel before the in-session TUI takes over.
        MainMenuChoice choice = ::collider::ui::tui::menu::run_main_menu(
            collider::kVersion);

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

            case MainMenuChoice::BIP_SCAN_MODE: {
#ifdef COLLIDER_PRO
                args = run_bip_scan_interactive(args);
                if (args.go_back) {
                    continue;  // Return to main menu
                }
                return args;
#else
                std::cout << "\n[PRO] BIP-39/32 mnemonic scanning requires theCollider Pro.\n"
                          << "      Purchase at: https://collisionprotocol.com/pro\n\n";
                continue;
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
    // Set console output code page to UTF-8 so Unicode glyphs (arrows,
    // box-drawing, emoji) render correctly. Without this, the default
    // CP-437 / CP-1252 console renders our UTF-8 bytes as mojibake
    // (e.g. "->" written as U+2192 prints as "GammaringAE" on CP-437).
    SetConsoleOutputCP(65001);

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
