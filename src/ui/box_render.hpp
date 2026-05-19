/**
 * Single-source-of-truth boxed UI rendering.
 *
 * Pre-1.4.1 the banners across banner.hpp, puzzle_solver.cpp, and the
 * various solve-success messages each used hardcoded `setw()` widths
 * paired with hardcoded space-padding -- the totals didn't match, so
 * the trailing `|` drifted left or right of the border depending on
 * the exact label/value lengths. Long values (e.g. "CUDA - NVIDIA
 * GeForce RTX 3060, NVIDIA GeForce RTX 2060 SUPER") overflowed the
 * box entirely.
 *
 * This header is the fix: every box uses the same INNER_WIDTH, every
 * line is built by a function that knows the visible-length math
 * (skipping ANSI escape codes) and pads or truncates as needed.
 *
 * Usage:
 *   collider::ui::box::top(std::cout);
 *   collider::ui::box::centered(std::cout, "PUZZLE SOLVED!", ANSI_GREEN);
 *   collider::ui::box::sep(std::cout);
 *   collider::ui::box::kv(std::cout, "Private Key", "0x2832...", ANSI_YELLOW);
 *   collider::ui::box::bottom(std::cout);
 */

#pragma once

#include <ostream>
#include <string>
#include <string_view>

namespace collider {
namespace ui {
namespace box {

// One inner width for every box in the application. 62 chars matches
// the most common existing border width and leaves room for label +
// value pairs at typical lengths. Increase only with a sweep across
// every existing border-string literal that hasn't been migrated.
inline constexpr std::size_t kInnerWidth = 62;

// ANSI escape code helpers. Empty string == no color.
namespace ansi {
inline constexpr std::string_view RESET         = "\033[0m";
inline constexpr std::string_view BRIGHT_GREEN  = "\033[1;32m";
inline constexpr std::string_view BRIGHT_YELLOW = "\033[1;33m";
inline constexpr std::string_view BRIGHT_CYAN   = "\033[1;36m";
inline constexpr std::string_view BRIGHT_MAGENTA= "\033[1;35m";
inline constexpr std::string_view BRIGHT_WHITE  = "\033[1;37m";
inline constexpr std::string_view CYAN          = "\033[36m";
inline constexpr std::string_view YELLOW        = "\033[33m";
inline constexpr std::string_view GREEN         = "\033[32m";
}

// Border line: `+<fill>{kInnerWidth}+\n`. Use '=' for outer, '-' for
// in-box separators.
inline void top(std::ostream& os, char fill = '=') {
    os << '+';
    for (std::size_t i = 0; i < kInnerWidth; ++i) os << fill;
    os << "+\n";
}
inline void bottom(std::ostream& os, char fill = '=') { top(os, fill); }
inline void sep(std::ostream& os) { top(os, '-'); }

// Truncate `s` to at most `max_len` visible chars; if truncated,
// replace the last 3 chars of the budget with "...". Used to keep
// long strings (long GPU names, long keys) from overflowing.
inline std::string truncate_visible(std::string_view s, std::size_t max_len) {
    if (s.size() <= max_len) return std::string(s);
    if (max_len <= 3) return std::string(max_len, '.');
    std::string out(s.substr(0, max_len - 3));
    out += "...";
    return out;
}

// Render `|  <label>: <value>[pad] |\n`.
// label_color and value_color are ANSI escape sequences inserted
// AROUND the visible label/value text -- they don't count toward
// visible width. Visible budget for value = kInnerWidth - 2 (leading
// "  ") - label_len - 2 (": ") - 1 (trailing " "). If value exceeds,
// it's truncated with "..."; if shorter, the gap is space-padded.
inline void kv(std::ostream& os,
               std::string_view label,
               std::string_view value,
               std::string_view label_color = {},
               std::string_view value_color = {}) {
    constexpr std::size_t kLeadingPad   = 2;   // "  " after the "|"
    constexpr std::size_t kSeparatorLen = 2;   // ": "
    constexpr std::size_t kTrailingPad  = 1;   // " " before the "|"
    const std::size_t budget =
        (kInnerWidth > kLeadingPad + label.size() + kSeparatorLen + kTrailingPad)
            ? kInnerWidth - kLeadingPad - label.size() - kSeparatorLen - kTrailingPad
            : 0;
    const std::string truncated = truncate_visible(value, budget);

    os << "|  ";
    if (!label_color.empty()) os << label_color;
    os << label;
    if (!label_color.empty()) os << ansi::RESET;
    os << ": ";
    if (!value_color.empty()) os << value_color;
    os << truncated;
    if (!value_color.empty()) os << ansi::RESET;
    for (std::size_t i = truncated.size(); i < budget; ++i) os << ' ';
    os << " |\n";
}

// Render `|<centered>|\n`. The visible text is centered within
// kInnerWidth; ANSI color (if any) wraps only the visible text.
inline void centered(std::ostream& os,
                     std::string_view text,
                     std::string_view color = {}) {
    const std::string truncated = truncate_visible(text, kInnerWidth);
    const std::size_t total_pad = kInnerWidth - truncated.size();
    const std::size_t left_pad  = total_pad / 2;
    const std::size_t right_pad = total_pad - left_pad;

    os << '|';
    for (std::size_t i = 0; i < left_pad; ++i) os << ' ';
    if (!color.empty()) os << color;
    os << truncated;
    if (!color.empty()) os << ansi::RESET;
    for (std::size_t i = 0; i < right_pad; ++i) os << ' ';
    os << "|\n";
}

// Render an arbitrary inner line: `|<text padded to inner>|\n`. The
// text is left-aligned at column 0 (no leading/trailing padding); use
// kv() for the standard 2-space-indented label-value form.
inline void line(std::ostream& os,
                 std::string_view text,
                 std::string_view color = {}) {
    const std::string truncated = truncate_visible(text, kInnerWidth);
    os << '|';
    if (!color.empty()) os << color;
    os << truncated;
    if (!color.empty()) os << ansi::RESET;
    for (std::size_t i = truncated.size(); i < kInnerWidth; ++i) os << ' ';
    os << "|\n";
}

}  // namespace box
}  // namespace ui
}  // namespace collider
