/**
 * Interactive Mode - User-Friendly Menu System
 *
 * Provides interactive prompts and menus for collider when launched
 * without command-line arguments.
 */

#pragma once

#include <iostream>
#include <string>
#include <limits>
#include <algorithm>
#include <cctype>
#include <filesystem>
#include "../core/edition.hpp"

#ifdef _WIN32
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#endif

namespace collider {
namespace ui {

/**
 * Interactive mode menu choices.
 */
enum class MainMenuChoice {
    PUZZLE_MODE = 1,
    BRAINWALLET_MODE = 2,
    BENCHMARK_MODE = 3,
    SHOW_HELP = 4,
    EXIT = 0
};

/**
 * Puzzle solving mode choices.
 */
enum class PuzzleModeChoice {
    STANDALONE = 1,
    JOIN_POOL = 2,
    BACK = 0
};

/**
 * ANSI color helpers for interactive prompts.
 */
namespace colors {
    inline const char* RESET       = "\033[0m";
    inline const char* BOLD        = "\033[1m";
    inline const char* DIM         = "\033[2m";
    inline const char* CYAN        = "\033[36m";
    inline const char* GREEN       = "\033[32m";
    inline const char* YELLOW      = "\033[33m";
    inline const char* RED         = "\033[31m";
    inline const char* WHITE       = "\033[37m";
    inline const char* BRIGHT_CYAN = "\033[96m";
    inline const char* BRIGHT_GREEN = "\033[92m";
    inline const char* BRIGHT_WHITE = "\033[97m";
}

/**
 * Interactive mode utilities.
 */
class Interactive {
public:
    /**
     * Clear input buffer to prevent stale input.
     */
    static void clear_input() {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    /**
     * Read a single line of input, trimming whitespace.
     */
    static std::string read_line() {
        std::string input;
        std::getline(std::cin, input);

        // Trim leading and trailing whitespace
        size_t start = input.find_first_not_of(" \t\r\n");
        size_t end = input.find_last_not_of(" \t\r\n");

        if (start == std::string::npos) return "";
        return input.substr(start, end - start + 1);
    }

    /**
     * Prompt user for a yes/no answer.
     * @param prompt The question to ask
     * @param default_yes If true, pressing Enter means Yes
     * @return true for yes, false for no
     */
    static bool prompt_yes_no(const std::string& prompt, bool default_yes = true) {
        std::cout << prompt;
        if (default_yes) {
            std::cout << " (Y/n): ";
        } else {
            std::cout << " (y/N): ";
        }
        std::cout << std::flush;

        std::string input = read_line();

        if (input.empty()) {
            return default_yes;
        }

        char first = std::tolower(input[0]);
        return (first == 'y');
    }

    /**
     * Prompt user for a number within a range.
     * @param prompt The prompt message
     * @param min_val Minimum acceptable value
     * @param max_val Maximum acceptable value
     * @param allow_auto If true, "auto" is accepted (returns -1)
     * @return The number entered, or -1 for "auto"
     */
    static int prompt_number(const std::string& prompt, int min_val, int max_val, bool allow_auto = false) {
        while (true) {
            std::cout << prompt;
            if (allow_auto) {
                std::cout << " (or 'auto'): ";
            } else {
                std::cout << ": ";
            }
            std::cout << std::flush;

            std::string input = read_line();

            if (input.empty()) {
                std::cout << colors::YELLOW << "[!] Please enter a value." << colors::RESET << "\n";
                continue;
            }

            // Check for "auto"
            std::string lower_input = input;
            std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
            if (allow_auto && (lower_input == "auto" || lower_input == "a")) {
                return -1;  // Sentinel for auto
            }

            // Try to parse as number
            try {
                int value = std::stoi(input);
                if (value < min_val || value > max_val) {
                    std::cout << colors::YELLOW << "[!] Please enter a number between "
                              << min_val << " and " << max_val << "." << colors::RESET << "\n";
                    continue;
                }
                return value;
            } catch (...) {
                std::cout << colors::YELLOW << "[!] Invalid input. Please enter a number";
                if (allow_auto) std::cout << " or 'auto'";
                std::cout << "." << colors::RESET << "\n";
            }
        }
    }

    /**
     * Prompt user for a menu choice.
     * @param min_choice Minimum valid choice
     * @param max_choice Maximum valid choice
     * @return The selected choice number
     */
    static int prompt_menu_choice(int min_choice, int max_choice) {
        while (true) {
            std::cout << "\n" << colors::BRIGHT_CYAN << "Enter choice ("
                      << min_choice << "-" << max_choice << "): " << colors::RESET;
            std::cout << std::flush;

            std::string input = read_line();

            if (input.empty()) {
                continue;
            }

            try {
                int choice = std::stoi(input);
                if (choice >= min_choice && choice <= max_choice) {
                    return choice;
                }
                std::cout << colors::YELLOW << "[!] Please enter a number between "
                          << min_choice << " and " << max_choice << "." << colors::RESET << "\n";
            } catch (...) {
                std::cout << colors::YELLOW << "[!] Invalid input. Please enter a number."
                          << colors::RESET << "\n";
            }
        }
    }

    /**
     * Prompt user for a file path with validation.
     * @param prompt The prompt message
     * @param must_exist If true, file must exist
     * @return The path entered (may be empty if user cancels)
     */
    static std::string prompt_path(const std::string& prompt, bool must_exist = true) {
        while (true) {
            std::cout << prompt << " (or 'cancel' to go back): ";
            std::cout << std::flush;

            std::string input = read_line();

            if (input.empty()) {
                continue;
            }

            // Check for cancel
            std::string lower = input;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            if (lower == "cancel" || lower == "c" || lower == "back" || lower == "q") {
                return "";
            }

            if (must_exist) {
                if (std::filesystem::exists(input)) {
                    return input;
                }
                std::cout << colors::RED << "[!] File not found: " << input << colors::RESET << "\n";
                std::cout << "    Please check the path and try again.\n";
            } else {
                return input;
            }
        }
    }

    /**
     * Display a boxed header.
     */
    static void display_header(const std::string& title, const std::string& version = "") {
        const int width = 64;
        std::string full_title = title;
        if (!version.empty()) {
            full_title += " v" + version;
        }

        // Center the title
        int padding = (width - 2 - static_cast<int>(full_title.length())) / 2;
        std::string padded_title = std::string(padding, ' ') + full_title;
        padded_title += std::string(width - 2 - padded_title.length(), ' ');

        std::cout << "\n";
        std::cout << colors::CYAN << "+" << std::string(width - 2, '=') << "+" << colors::RESET << "\n";
        std::cout << colors::CYAN << "|" << colors::BRIGHT_WHITE << padded_title
                  << colors::CYAN << "|" << colors::RESET << "\n";
        std::cout << colors::CYAN << "+" << std::string(width - 2, '=') << "+" << colors::RESET << "\n";
    }

    /**
     * Display the main menu and get user choice.
     */
    static MainMenuChoice display_main_menu(const std::string& version) {
        // Free Edition Menu
        display_header("collider", version);
        
        std::cout << "\n";
        std::cout << colors::BRIGHT_WHITE << "Open Source GPU Solver" << colors::RESET << "\n\n";
        
        std::cout << colors::BRIGHT_WHITE << "What would you like to do?" << colors::RESET << "\n\n";

        std::cout << "  " << colors::BRIGHT_GREEN << "[1]" << colors::RESET
                  << " Pool Solver\n";
        std::cout << "  " << colors::BRIGHT_GREEN << "[2]" << colors::RESET
                  << " Benchmark\n";
        std::cout << "  " << colors::BRIGHT_GREEN << "[3]" << colors::RESET
                  << " Help\n";
        std::cout << "\n";
        std::cout << colors::YELLOW << "  Upgrade to Pro:" << colors::RESET << "\n";
        std::cout << colors::DIM << "  collisionprotocol.com/pro" << colors::RESET << "\n";
        std::cout << "\n";
        std::cout << "  " << colors::DIM << "[0]" << colors::RESET
                  << colors::DIM << " Exit" << colors::RESET << "\n";

        int choice = prompt_menu_choice(0, 3);
        // Map free edition choices to the enum
        switch (choice) {
            case 1: return MainMenuChoice::PUZZLE_MODE; // Pool mode only
            case 2: return MainMenuChoice::BENCHMARK_MODE;
            case 3: return MainMenuChoice::SHOW_HELP;
            case 0: return MainMenuChoice::EXIT;
            default: return MainMenuChoice::EXIT;
        }
    }

    /**
     * Display a section header (simpler than boxed header).
     */
    static void display_section(const std::string& title) {
        std::cout << "\n";
        std::cout << colors::BRIGHT_WHITE << title << colors::RESET << "\n";
        std::cout << colors::DIM << std::string(title.length(), '-') << colors::RESET << "\n\n";
    }

    /**
     * Display puzzle information in a formatted way.
     */
    static void display_puzzle_info(int number, int bits, bool has_pubkey,
                                     double reward, const std::string& estimated_time) {
        std::cout << colors::CYAN << "[*] " << colors::BRIGHT_WHITE
                  << "Puzzle #" << number << colors::RESET << "\n";
        std::cout << "    - Public key: ";
        if (has_pubkey) {
            std::cout << colors::GREEN << "KNOWN" << colors::RESET
                      << " (Kangaroo method available)\n";
        } else {
            std::cout << colors::YELLOW << "UNKNOWN" << colors::RESET
                      << " (Brute force only)\n";
        }
        std::cout << "    - Range: " << bits << " bits (2^" << bits << " keys)\n";
        std::cout << "    - Expected time: " << estimated_time << "\n";
        std::cout << "    - Reward: " << colors::BRIGHT_GREEN << reward << " BTC"
                  << colors::RESET << "\n";
    }

    /**
     * Display a progress message with status indicator.
     */
    static void status_message(const std::string& msg, bool success = true) {
        if (success) {
            std::cout << colors::GREEN << "[+] " << colors::RESET << msg << "\n";
        } else {
            std::cout << colors::RED << "[!] " << colors::RESET << msg << "\n";
        }
    }

    /**
     * Display an info message.
     */
    static void info_message(const std::string& msg) {
        std::cout << colors::CYAN << "[*] " << colors::RESET << msg << "\n";
    }

    /**
     * Display a warning message.
     */
    static void warning_message(const std::string& msg) {
        std::cout << colors::YELLOW << "[!] " << colors::RESET << msg << "\n";
    }

    /**
     * Display an error message.
     */
    static void error_message(const std::string& msg) {
        std::cout << colors::RED << "[ERROR] " << colors::RESET << msg << "\n";
    }

    /**
     * Pause and wait for user to press Enter.
     */
    static void press_enter_to_continue() {
        std::cout << "\n" << colors::DIM << "Press Enter to continue..." << colors::RESET;
        std::cout << std::flush;
        read_line();
    }

    /**
     * Display puzzle mode submenu (standalone vs pool).
     */
    static PuzzleModeChoice display_puzzle_mode_menu() {
        std::cout << "\n";
        std::cout << colors::BRIGHT_WHITE << "How would you like to solve?" << colors::RESET << "\n\n";

        std::cout << "  " << colors::BRIGHT_GREEN << "[1]" << colors::RESET
                  << " Standalone (solo mining)\n";
        std::cout << "  " << colors::BRIGHT_GREEN << "[2]" << colors::RESET
                  << " Join Pool (distributed solving)\n";
        std::cout << "  " << colors::DIM << "[0]" << colors::RESET
                  << " Back to main menu\n";

        int choice = prompt_menu_choice(0, 2);
        return static_cast<PuzzleModeChoice>(choice);
    }

    /**
     * Prompt for pool configuration.
     * @param pool_url Output: pool URL
     * @param worker Output: worker name (Bitcoin address)
     * @param default_url Pre-filled URL from config
     * @param default_worker Pre-filled worker from config
     * @return true if configured, false if cancelled
     */
    static bool prompt_pool_config(std::string& pool_url, std::string& worker,
                                   const std::string& default_url = "",
                                   const std::string& default_worker = "") {
        std::cout << "\n";
        display_section("Pool Configuration");

        // Pool URL
        if (!default_url.empty()) {
            std::cout << "Pool URL [" << colors::CYAN << default_url << colors::RESET << "]: ";
            std::string input = read_line();
            pool_url = input.empty() ? default_url : input;
        } else {
            std::cout << "Pool URL (e.g., jlp://pool.example.com:17403): ";
            pool_url = read_line();
            if (pool_url.empty()) {
                error_message("Pool URL is required");
                return false;
            }
        }

        // Worker name (Bitcoin address)
        if (!default_worker.empty()) {
            std::cout << "Worker address [" << colors::CYAN << default_worker << colors::RESET << "]: ";
            std::string input = read_line();
            worker = input.empty() ? default_worker : input;
        } else {
            std::cout << "Worker address (your Bitcoin address for rewards): ";
            worker = read_line();
            if (worker.empty()) {
                error_message("Worker address is required");
                return false;
            }
        }

        std::cout << "\n";
        info_message("Pool: " + pool_url);
        info_message("Worker: " + worker);

        return true;
    }
};

} // namespace ui
} // namespace collider
