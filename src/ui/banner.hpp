/**
 * TheCollider - Clean ANSI Banner with Shine Wipe Effect
 *
 * Simple, context-aware display with animated shine.
 */

#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <sstream>
#include <iomanip>

#include "box_render.hpp"

#ifdef _WIN32
#include <windows.h>
#endif

namespace collider {
namespace ui {

/**
 * Operation mode - determines what status text to show.
 */
enum class OperationMode {
    PUZZLE_SEARCH,      // Bitcoin puzzle challenge
    BRAIN_WALLET,       // Brain wallet recovery with bloom filter
    BENCHMARK,          // Performance benchmark
    UNKNOWN
};

/**
 * Banner configuration options.
 */
struct BannerConfig {
    bool enable_animation = true;
    bool enable_color = true;
    int animation_frames = 1;      // Number of full shine sweeps (one is plenty; two doubled the perceived startup latency)
    int frame_delay_ms = 40;       // Delay between shine positions
    bool show_stats = true;
    OperationMode mode = OperationMode::UNKNOWN;
};

/**
 * System stats to display in banner.
 */
struct BannerStats {
    int gpu_count = 0;
    std::string gpu_names = "";
    std::string backend = "CPU";           // "CUDA", "Metal", "CPU"
    uint64_t estimated_speed = 0;
    std::string bloom_file = "";
    uint64_t bloom_entries = 0;
    std::string version = "1.0.0";

    // Puzzle-specific
    int puzzle_number = 0;
    int puzzle_bits = 0;
    double puzzle_reward = 0;
};

/**
 * ANSI color codes.
 */
namespace ansi {
    inline const char* RESET       = "\033[0m";
    inline const char* BOLD        = "\033[1m";
    inline const char* DIM         = "\033[2m";

    inline const char* RED         = "\033[31m";
    inline const char* GREEN       = "\033[32m";
    inline const char* YELLOW      = "\033[33m";
    inline const char* BLUE        = "\033[34m";
    inline const char* MAGENTA     = "\033[35m";
    inline const char* CYAN        = "\033[36m";
    inline const char* WHITE       = "\033[37m";

    inline const char* BRIGHT_RED     = "\033[91m";
    inline const char* BRIGHT_GREEN   = "\033[92m";
    inline const char* BRIGHT_YELLOW  = "\033[93m";
    inline const char* BRIGHT_BLUE    = "\033[94m";
    inline const char* BRIGHT_MAGENTA = "\033[95m";
    inline const char* BRIGHT_CYAN    = "\033[96m";
    inline const char* BRIGHT_WHITE   = "\033[97m";

    // 256-color
    inline std::string color256(int code) {
        return "\033[38;5;" + std::to_string(code) + "m";
    }

    inline const char* CURSOR_HIDE  = "\033[?25l";
    inline const char* CURSOR_SHOW  = "\033[?25h";

    inline std::string cursor_up(int n) {
        return "\033[" + std::to_string(n) + "A";
    }
}

// ---------------------------------------------------------------------------
// Single source of truth for "X items per second" formatting.
//
// Auto-picks an SI prefix (K / M / G / T) based on magnitude and emits two
// decimals of precision (one for sub-MKey/s, zero for sub-KKey/s) so output
// width is bounded and predictable. Default unit is "Keys/s" since most
// callers are measuring kangaroo / brute-force key throughput; the SHA
// benchmark and brain-wallet check loops can override it.
//
// Replaces the three independent formatters that grew up alongside this
// header (`Banner::format_speed(uint64_t)`,
// `ProfessionalUI::format_speed(int mkeys_per_sec)`, and a substr-truncating
// free function in main.cpp). Callers that had a value already in MKeys/s
// should multiply by 1e6 before calling.
// ---------------------------------------------------------------------------
inline std::string format_rate(double rate, std::string_view unit = "Keys/s") {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    if (rate >= 1e12) {
        oss << (rate / 1e12) << " T" << unit;
    } else if (rate >= 1e9) {
        oss << (rate / 1e9)  << " G" << unit;
    } else if (rate >= 1e6) {
        oss << (rate / 1e6)  << " M" << unit;
    } else if (rate >= 1e3) {
        oss.precision(1);
        oss << (rate / 1e3)  << " K" << unit;
    } else {
        oss.precision(0);
        oss << rate << " " << unit;
    }
    return oss.str();
}

/**
 * Clean banner with shine wipe effect.
 */
class Banner {
public:
    Banner(const BannerConfig& config = BannerConfig{})
        : config_(config) {
        initialize_terminal();
    }

    ~Banner() {
        if (config_.enable_color) {
            std::cout << ansi::RESET << ansi::CURSOR_SHOW;
        }
    }

    void display(const BannerStats& stats = BannerStats{}) {
        if (config_.enable_animation && config_.enable_color) {
            display_animated(stats);
        } else {
            display_static(stats);
        }
    }

private:
    BannerConfig config_;

    void initialize_terminal() {
#ifdef _WIN32
        // Enable ANSI escape sequences on Windows
        HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hOut != INVALID_HANDLE_VALUE) {
            DWORD dwMode = 0;
            if (GetConsoleMode(hOut, &dwMode)) {
                dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
                SetConsoleMode(hOut, dwMode);
            }
        }
        // Set console to UTF-8 for proper Unicode character display
        SetConsoleOutputCP(65001);
#endif
        if (std::getenv("NO_COLOR") != nullptr) {
            config_.enable_color = false;
        }
    }

    /**
     * Decide whether to use ASCII or Unicode block art for the logo.
     *
     * Honors explicit env-var overrides first:
     *   COLLIDER_ASCII=1   -> force ASCII
     *   COLLIDER_UNICODE=1 -> force Unicode (used for testing on Windows)
     *
     * Otherwise:
     *   Windows: ASCII by default (cmd.exe + PowerShell default to legacy
     *            code pages that can't render U+2588 block chars cleanly).
     *   Linux/Mac: Unicode IF the active locale is UTF-8. Falls back to
     *              ASCII when LANG / LC_ALL / LC_CTYPE is unset, "C", or
     *              not a UTF-8 locale -- a Mac terminal launched without
     *              an inherited locale (some shells, ssh sessions, GUI
     *              app launches) won't render U+2588 and would show the
     *              "���" replacement-char triple per glyph instead.
     */
    bool use_ascii_fallback() const {
        if (std::getenv("COLLIDER_ASCII") != nullptr)   return true;
        if (std::getenv("COLLIDER_UNICODE") != nullptr) return false;
#ifdef _WIN32
        // Windows console default code pages don't render U+2588 cleanly.
        return true;
#elif defined(__APPLE__)
        // Even with LANG=en_US.UTF-8, many monospace fonts shipped with
        // Mac terminals (Menlo, SF Mono in some weights, third-party
        // fonts) lack U+2588 FULL BLOCK and render it as the
        // replacement-glyph triple. ASCII is the safe default; users
        // who want the block-letter version can opt in via
        // COLLIDER_UNICODE=1.
        return true;
#else
        // Linux: probe LC_ALL > LC_CTYPE > LANG.
        const char* loc = std::getenv("LC_ALL");
        if (!loc || !*loc) loc = std::getenv("LC_CTYPE");
        if (!loc || !*loc) loc = std::getenv("LANG");
        if (!loc || !*loc) return true;
        std::string s(loc);
        for (auto& c : s) c = (char)std::tolower((unsigned char)c);
        if (s.find("utf-8") != std::string::npos) return false;
        if (s.find("utf8")  != std::string::npos) return false;
        return true;
#endif
    }

    /**
     * Get the logo lines - clean block letters.
     * THE on top, COLLIDER below.
     * Uses ASCII # on Windows for compatibility.
     */
    std::vector<std::string> get_logo_lines() {
        if (use_ascii_fallback()) {
            // ASCII version for Windows compatibility
            return {
                R"(        #####  #   #  #####             )",
                R"(          #    #   #  #                 )",
                R"(          #    #####  ####              )",
                R"(          #    #   #  #                 )",
                R"(          #    #   #  #####             )",
                R"(                                        )",
                R"(    ####   ####  #     #     ##### ####  ##### ####  )",
                R"(   #      #   #  #     #       #   #   # #     #   # )",
                R"(   #      #   #  #     #       #   #   # ####  ####  )",
                R"(   #      #   #  #     #       #   #   # #     #  #  )",
                R"(    ####   ####  ##### ##### ##### ####  ##### #   # )",
            };
        } else {
            // Unicode version for Unix/Mac
            return {
                R"(        █████  █   █  █████             )",
                R"(          █    █   █  █                 )",
                R"(          █    █████  ████              )",
                R"(          █    █   █  █                 )",
                R"(          █    █   █  █████             )",
                R"(                                        )",
                R"(    ████   ████  █     █     █████ ████  █████ ████  )",
                R"(   █      █   █  █     █       █   █   █ █     █   █ )",
                R"(   █      █   █  █     █       █   █   █ ████  ████  )",
                R"(   █      █   █  █     █       █   █   █ █     █  █  )",
                R"(    ████   ████  █████ █████ █████ ████  █████ █   █ )",
            };
        }
    }

    /**
     * Apply shine wipe effect to a single line.
     * @param line The text line
     * @param shine_pos Position of shine center (0 to line.length())
     * @param base_color Base 256-color code (orange/red gradient)
     */
    std::string apply_shine(const std::string& line, int shine_pos, int base_color) {
        std::ostringstream out;

        // Shine width (characters affected by the gleam)
        const int shine_width = 5;

        for (size_t i = 0; i < line.size(); i++) {
            int dist = std::abs(static_cast<int>(i) - shine_pos);

            if (line[i] == ' ') {
                out << ' ';
            } else if (dist <= shine_width) {
                // In shine zone - gradient from white to base
                if (dist == 0) {
                    // Center of shine - bright white
                    out << ansi::BOLD << ansi::color256(231) << line[i] << ansi::RESET;
                } else if (dist <= 2) {
                    // Near center - bright yellow
                    out << ansi::BOLD << ansi::color256(226) << line[i] << ansi::RESET;
                } else {
                    // Edge of shine - gold
                    out << ansi::color256(220) << line[i] << ansi::RESET;
                }
            } else {
                // Outside shine - base color
                out << ansi::color256(base_color) << line[i] << ansi::RESET;
            }
        }

        return out.str();
    }

    /**
     * Render the full logo with shine at given position.
     */
    std::string render_logo_with_shine(int shine_pos) {
        std::ostringstream out;
        auto lines = get_logo_lines();

        // Color gradient for rows (top to bottom): orange -> red
        // 208=orange, 202=red-orange, 196=red
        std::vector<int> row_colors = {208, 208, 214, 202, 202, 240, 196, 196, 202, 208, 208};

        for (size_t i = 0; i < lines.size(); i++) {
            int base_color = (i < row_colors.size()) ? row_colors[i] : 202;
            out << apply_shine(lines[i], shine_pos, base_color) << "\n";
        }

        return out.str();
    }

    /**
     * Create context-aware status box. Uses ui::box for consistent
     * borders and alignment across long values (e.g. multi-GPU names).
     */
    std::string create_stats_box(const BannerStats& stats) {
        namespace boxui = ::collider::ui::box;
        std::ostringstream out;

        out << "\n";

        const std::string color_open = config_.enable_color
            ? std::string(boxui::ansi::CYAN) : std::string{};
        const std::string color_label_white = config_.enable_color
            ? std::string(boxui::ansi::BRIGHT_WHITE) : std::string{};
        const std::string color_label_green = config_.enable_color
            ? std::string(boxui::ansi::BRIGHT_GREEN) : std::string{};
        const std::string color_label_yellow = config_.enable_color
            ? std::string(boxui::ansi::BRIGHT_YELLOW) : std::string{};
        const std::string color_label_magenta = config_.enable_color
            ? std::string(boxui::ansi::BRIGHT_MAGENTA) : std::string{};

        if (!color_open.empty()) out << color_open;
        boxui::top(out, '-');

        // Mode-specific header (centered)
        std::string mode_text;
        switch (config_.mode) {
            case OperationMode::PUZZLE_SEARCH: mode_text = "BITCOIN PUZZLE CHALLENGE"; break;
            case OperationMode::BRAIN_WALLET:  mode_text = "BRAIN WALLET RECOVERY";    break;
            case OperationMode::BENCHMARK:     mode_text = "PERFORMANCE BENCHMARK";    break;
            default:                           mode_text = "SYSTEM STATUS";            break;
        }
        boxui::centered(out, mode_text, color_label_white);
        boxui::sep(out);

        // Hardware line. Long multi-GPU strings auto-truncate with "...".
        std::string hw_info;
        if      (stats.backend == "CUDA")  hw_info = "CUDA - "  + stats.gpu_names;
        else if (stats.backend == "Metal") hw_info = "Metal - " + stats.gpu_names;
        else                               hw_info = "CPU (reference mode)";
        boxui::kv(out, "Hardware", hw_info, color_label_green);

        // Speed estimate
        if (stats.estimated_speed > 0) {
            boxui::kv(out, "Speed",
                      format_rate(static_cast<double>(stats.estimated_speed)),
                      color_label_yellow);
        }

        // Mode-specific info
        if (config_.mode == OperationMode::PUZZLE_SEARCH && stats.puzzle_number > 0) {
            std::ostringstream puzzle_info;
            puzzle_info << "Puzzle #" << stats.puzzle_number << " (" << stats.puzzle_bits << "-bit";
            if (stats.puzzle_reward > 0) {
                puzzle_info << ", " << std::fixed << std::setprecision(1)
                            << stats.puzzle_reward << " BTC";
            }
            puzzle_info << ")";
            boxui::kv(out, "Target", puzzle_info.str(), color_label_magenta);
        } else if (config_.mode == OperationMode::BRAIN_WALLET && !stats.bloom_file.empty()) {
            std::string bloom_info = stats.bloom_file;
            if (stats.bloom_entries > 0) {
                bloom_info += " (" + std::to_string(stats.bloom_entries / 1'000'000) + "M)";
            }
            boxui::kv(out, "Bloom", bloom_info, color_label_magenta);
        }

        boxui::bottom(out, '-');

        if (config_.enable_color) out << boxui::ansi::RESET;
        return out.str();
    }

    void display_animated(const BannerStats& stats) {
        auto lines = get_logo_lines();
        int logo_height = static_cast<int>(lines.size());
        int logo_width = 0;
        for (const auto& line : lines) {
            if (static_cast<int>(line.size()) > logo_width) {
                logo_width = static_cast<int>(line.size());
            }
        }

        if (config_.enable_color) {
            std::cout << ansi::CURSOR_HIDE;
        }

        // Shine wipe animation - sweep left to right
        for (int cycle = 0; cycle < config_.animation_frames; cycle++) {
            // Shine sweeps from -shine_width to logo_width + shine_width
            for (int shine_pos = -5; shine_pos <= logo_width + 5; shine_pos += 2) {
                if (cycle > 0 || shine_pos > -5) {
                    std::cout << ansi::cursor_up(logo_height);
                }

                std::cout << render_logo_with_shine(shine_pos);
                std::cout.flush();

                std::this_thread::sleep_for(std::chrono::milliseconds(config_.frame_delay_ms));
            }
        }

        if (config_.show_stats) {
            std::cout << create_stats_box(stats);
        }

        if (config_.enable_color) {
            std::cout << ansi::CURSOR_SHOW;
        }
    }

    void display_static(const BannerStats& stats) {
        auto lines = get_logo_lines();

        if (config_.enable_color) {
            // Display with base colors (no shine)
            std::vector<int> row_colors = {208, 208, 214, 202, 202, 240, 196, 196, 202, 208, 208};
            for (size_t i = 0; i < lines.size(); i++) {
                int color = (i < row_colors.size()) ? row_colors[i] : 202;
                std::cout << ansi::color256(color) << lines[i] << ansi::RESET << "\n";
            }
        } else {
            for (const auto& line : lines) {
                std::cout << line << "\n";
            }
        }

        if (config_.show_stats) {
            std::cout << create_stats_box(stats);
        }
    }
};

/**
 * Convenience function to display the banner.
 */
inline void display_banner(const BannerStats& stats = BannerStats{},
                          const BannerConfig& config = BannerConfig{}) {
    Banner banner(config);
    banner.display(stats);
}

// =============================================================================
// PROFESSIONAL UI COMPONENTS
// =============================================================================

/**
 * Professional output formatting utilities.
 */
class ProfessionalUI {
public:
    static constexpr int DEFAULT_WIDTH = 66;

    /**
     * Render a box with title and optional subtitle.
     */
    static void render_box(const std::string& title, const std::string& subtitle = "",
                          int width = DEFAULT_WIDTH) {
        std::cout << "\n";
        std::cout << ansi::CYAN << "+" << std::string(width - 2, '=') << "+" << ansi::RESET << "\n";

        // Center title
        int title_padding = (width - 2 - static_cast<int>(title.length())) / 2;
        std::cout << ansi::CYAN << "|" << ansi::RESET;
        std::cout << std::string(title_padding, ' ');
        std::cout << ansi::BRIGHT_WHITE << title << ansi::RESET;
        std::cout << std::string(width - 2 - title_padding - static_cast<int>(title.length()), ' ');
        std::cout << ansi::CYAN << "|" << ansi::RESET << "\n";

        if (!subtitle.empty()) {
            int sub_padding = (width - 2 - static_cast<int>(subtitle.length())) / 2;
            std::cout << ansi::CYAN << "|" << ansi::RESET;
            std::cout << std::string(sub_padding, ' ');
            std::cout << ansi::DIM << subtitle << ansi::RESET;
            std::cout << std::string(width - 2 - sub_padding - static_cast<int>(subtitle.length()), ' ');
            std::cout << ansi::CYAN << "|" << ansi::RESET << "\n";
        }

        std::cout << ansi::CYAN << "+" << std::string(width - 2, '=') << "+" << ansi::RESET << "\n";
    }

    /**
     * Render a section header.
     */
    static void render_section(const std::string& title) {
        std::cout << "\n";
        std::cout << ansi::BRIGHT_WHITE << title << ansi::RESET << "\n";
        std::cout << ansi::DIM << std::string(title.length(), '-') << ansi::RESET << "\n";
    }

    /**
     * Render a key-value line with proper alignment.
     */
    static void render_kv(const std::string& key, const std::string& value,
                         int key_width = 16, const std::string& suffix = "") {
        std::cout << std::left << std::setw(key_width) << (key + ":") << " ";
        std::cout << ansi::BRIGHT_WHITE << value << ansi::RESET;
        if (!suffix.empty()) {
            std::cout << "  " << ansi::DIM << suffix << ansi::RESET;
        }
        std::cout << "\n";
    }

    /**
     * Render a progress bar.
     */
    static void render_progress_bar(double progress, int width = 40,
                                    const std::string& label = "") {
        int filled = static_cast<int>(progress * width);
        if (filled > width) filled = width;
        if (filled < 0) filled = 0;

        std::cout << "[";
        std::cout << ansi::BRIGHT_GREEN << std::string(filled, '#') << ansi::RESET;
        std::cout << std::string(width - filled, '-');
        std::cout << "] " << std::fixed << std::setprecision(2) << (progress * 100) << "%";
        if (!label.empty()) {
            std::cout << " " << label;
        }
        std::cout << "\n";
    }

    /**
     * Render a separator line.
     */
    static void render_separator(int width = DEFAULT_WIDTH, char ch = '-') {
        std::cout << ansi::DIM << std::string(width - 2, ch) << ansi::RESET << "\n";
    }

    /**
     * Format large numbers with appropriate suffix (K, M, G, T).
     */
    static std::string format_number_short(uint64_t n) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);
        if (n >= 1000000000000ULL) {
            oss << (static_cast<double>(n) / 1e12) << "T";
        } else if (n >= 1000000000ULL) {
            oss << (static_cast<double>(n) / 1e9) << "G";
        } else if (n >= 1000000ULL) {
            oss << (static_cast<double>(n) / 1e6) << "M";
        } else if (n >= 1000ULL) {
            oss << (static_cast<double>(n) / 1e3) << "K";
        } else {
            oss << n;
        }
        return oss.str();
    }

    // Speed/rate formatting now lives at namespace scope as
    // `collider::ui::format_rate(double, std::string_view)`. The previous
    // `format_speed(int mkeys_per_sec)` version mis-scaled M into T and
    // produced "1.21 TKeys/s" displays for ~1 MKey/s inputs; converting
    // its callers to multiply their value by 1e6 first is more honest.


    /**
     * Format duration in human-readable form.
     */
    static std::string format_duration(double seconds) {
        if (seconds < 60) {
            return std::to_string(static_cast<int>(seconds)) + "s";
        } else if (seconds < 3600) {
            int mins = static_cast<int>(seconds / 60);
            int secs = static_cast<int>(seconds) % 60;
            return std::to_string(mins) + "m " + std::to_string(secs) + "s";
        } else if (seconds < 86400) {
            int hours = static_cast<int>(seconds / 3600);
            int mins = (static_cast<int>(seconds) % 3600) / 60;
            return std::to_string(hours) + "h " + std::to_string(mins) + "m";
        } else {
            int days = static_cast<int>(seconds / 86400);
            int hours = (static_cast<int>(seconds) % 86400) / 3600;
            return std::to_string(days) + "d " + std::to_string(hours) + "h";
        }
    }

    /**
     * Render a status message with color coding.
     */
    static void status_ok(const std::string& msg) {
        std::cout << ansi::BRIGHT_GREEN << "[+] " << ansi::RESET << msg << "\n";
    }

    static void status_info(const std::string& msg) {
        std::cout << ansi::BRIGHT_CYAN << "[*] " << ansi::RESET << msg << "\n";
    }

    static void status_warn(const std::string& msg) {
        std::cout << ansi::BRIGHT_YELLOW << "[!] " << ansi::RESET << msg << "\n";
    }

    static void status_error(const std::string& msg) {
        std::cout << ansi::BRIGHT_RED << "[ERROR] " << ansi::RESET << msg << "\n";
    }

    /**
     * Render a "Found!" celebration banner.
     */
    static void render_found_banner(const std::string& what = "KEY FOUND") {
        namespace boxui = ::collider::ui::box;
        std::cout << "\n";
        boxui::top(std::cout);
        boxui::centered(std::cout, what, boxui::ansi::BRIGHT_GREEN);
        boxui::bottom(std::cout);
    }

    /**
     * Render footer with instructions.
     */
    static void render_footer(const std::string& msg = "Press Ctrl+C to stop and save checkpoint") {
        std::cout << "\n" << ansi::DIM << "[" << msg << "]" << ansi::RESET << "\n";
    }
};

// Convenience alias
using UI = ProfessionalUI;

// PoolProgressDisplay was previously defined here; it now lives in
// src/ui/pool_progress.hpp. Keeping the header focused on banner +
// general UI utilities; pool-specific widgets live alongside the pool
// client integration.

} // namespace ui
} // namespace collider
