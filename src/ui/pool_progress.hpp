/**
 * Pool-mining progress display.
 *
 * Single source of truth for the per-tick status line that all three
 * pool backends (CUDA RCKangaroo, Apple Metal Kangaroo, CPU
 * KangarooSolver) print while running. Owns:
 *   - a baseline capture of pool-wide and worker-lifetime DP counts so
 *     the "Session share %" reflects this run rather than all-time
 *     totals;
 *   - rate formatting that auto-picks units (KKeys / MKeys / GKeys /
 *     TKeys) based on magnitude, via collider::ui::format_rate;
 *   - the colored single-line layout, identical across backends, with
 *     ANSI escapes suppressed when stdout is not a TTY or NO_COLOR is
 *     set.
 *
 * Caller passes primitive ints (not collider::pool::PoolStats) so this
 * UI header stays free of any pool-module dependency.
 */

#pragma once

#include "banner.hpp"  // ProfessionalUI::format_number_short, format_rate

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#ifdef _WIN32
#include <io.h>
#define COLLIDER_FILENO  _fileno
#define COLLIDER_ISATTY  _isatty
#else
#include <unistd.h>
#define COLLIDER_FILENO  fileno
#define COLLIDER_ISATTY  isatty
#endif

namespace collider {
namespace ui {

class PoolProgressDisplay {
public:
    PoolProgressDisplay() {
        // Decide once -- tick() runs at multi-Hz and shouldn't getenv on
        // every call. NO_COLOR (https://no-color.org) suppresses any
        // colored output regardless of TTY status; for piped output
        // (logs, CI), suppress the carriage-return repaint too so each
        // sample lands as its own line in the captured stream.
        const bool no_color = (std::getenv("NO_COLOR") != nullptr);
        const bool is_tty   = COLLIDER_ISATTY(COLLIDER_FILENO(stdout)) != 0;
        use_color_  = !no_color && is_tty;
        use_repaint_ = is_tty;
    }

    // ops_per_sec  : raw kangaroo step rate in ops/s. Sub-MKey/s renders
    //                as KKeys/s with one decimal; >= 1 MKey/s as MKeys/s;
    //                >= 1 GKey/s as GKeys/s with two decimals; etc.
    // local_dps    : DPs surfaced this session by this worker.
    // sent_dps     : DPs the pool client has acknowledged on the wire.
    // pool_total   : pool-wide lifetime DP count (PoolStats.total_dps).
    // your_total   : worker-lifetime DP count (PoolStats.your_dps).
    //
    // Repaint at most once every kMinRepaintMs to avoid flooding the
    // terminal from a tight kernel-dispatch loop.
    void tick(double   ops_per_sec,
              uint64_t local_dps,
              uint64_t sent_dps,
              uint64_t pool_total,
              uint64_t your_total)
    {
        const auto now = std::chrono::steady_clock::now();
        if (last_paint_.time_since_epoch().count() != 0) {
            const auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   now - last_paint_).count();
            if (delta < kMinRepaintMs) return;
        }
        last_paint_ = now;

        if (!baseline_captured_ && pool_total > 0) {
            baseline_pool_dps_ = pool_total;
            baseline_your_dps_ = your_total;
            baseline_captured_ = true;
        }
        const uint64_t pool_delta = (pool_total > baseline_pool_dps_)
            ? (pool_total - baseline_pool_dps_) : 0;
        const uint64_t your_delta = (your_total > baseline_your_dps_)
            ? (your_total - baseline_your_dps_) : 0;
        const double share_pct = (pool_delta > 0)
            ? (100.0 * static_cast<double>(your_delta)
                      / static_cast<double>(pool_delta))
            : 0.0;

        const char* CY    = use_color_ ? "\033[36m" : "";
        const char* GR    = use_color_ ? "\033[32m" : "";
        const char* MA    = use_color_ ? "\033[35m" : "";
        const char* YE    = use_color_ ? "\033[33m" : "";
        const char* BL    = use_color_ ? "\033[34m" : "";
        const char* RESET = use_color_ ? "\033[0m"  : "";

        // TTY: \r + clear-line so each tick repaints in place.
        // Non-TTY (logs/pipes): plain newline so each tick is a record.
        if (use_repaint_) {
            std::cout << "\r\033[K";
        }
        std::cout << GR << "Speed:"        << RESET << " " << format_rate(ops_per_sec) << " | "
                  << MA << "Local DPs:"    << RESET << " " << ProfessionalUI::format_number_short(local_dps) << " | "
                  << YE << "Sent:"         << RESET << " " << ProfessionalUI::format_number_short(sent_dps) << " | "
                  << CY << "Your total:"   << RESET << " " << ProfessionalUI::format_number_short(your_total) << " | "
                  << BL << "Pool total:"   << RESET << " " << ProfessionalUI::format_number_short(pool_total) << " | "
                  << MA << "Session share:"<< RESET << " " << std::fixed << std::setprecision(1) << share_pct << "%";
        if (use_repaint_) {
            std::cout << "  " << std::flush;
        } else {
            std::cout << "\n";
        }
    }

private:
    static constexpr int kMinRepaintMs = 100;  // ~10 Hz max repaint rate

    std::chrono::steady_clock::time_point last_paint_{};
    bool     baseline_captured_ = false;
    uint64_t baseline_pool_dps_ = 0;
    uint64_t baseline_your_dps_ = 0;
    bool     use_color_   = true;
    bool     use_repaint_ = true;
};

}  // namespace ui
}  // namespace collider

#undef COLLIDER_FILENO
#undef COLLIDER_ISATTY
