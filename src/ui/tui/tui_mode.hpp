// tui_mode.hpp -- FREE STUB (installed by scripts/sync-to-free.sh)
//
// The Pro tree's src/ui/tui/ directory is excluded en bloc from the
// free distribution because the FTXUI dashboard, panels, and stdio
// capture machinery are Pro features. However a handful of FREE
// source files (pool_solver.cpp, puzzle_solver_kangaroo.cpp,
// puzzle_solver.cpp) reference TYPES that live in this header --
// TuiMode discriminator + per-mode info structs that the TuiApp
// methods take as arguments. Those references compile against this
// stub: the types exist with the same names and field shapes the
// callers expect; in the free build any TuiApp method that takes a
// PoolInfo / ChallengeInfo / BipScanInfo just receives an opaque
// blob and discards it (the stub TuiApp methods are no-ops -- see
// the stub tui_app.hpp shipped alongside this file).
//
// Pro keeps the canonical src/ui/tui/tui_mode.hpp with the full
// per-mode info contract; the panels read every field.
#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace collider::ui::tui {

enum class TuiMode : int {
    Brainwallet = 0,
    Pool        = 1,
    Challenge   = 2,
    Benchmark   = 3,
    BipScan     = 4,
};

inline TuiMode mode_from_label(const std::string& label) {
    if (label == "Pool")        return TuiMode::Pool;
    if (label == "Challenge")   return TuiMode::Challenge;
    if (label == "Benchmark")   return TuiMode::Benchmark;
    if (label == "BIP-Scan")    return TuiMode::BipScan;
    return TuiMode::Brainwallet;
}

struct PoolInfo {
    uint64_t    work_id            = 0;
    int         dp_bits            = 0;
    std::string kangaroo_type;
    uint64_t    dps_submitted      = 0;
    uint64_t    pool_total_dps     = 0;
    double      your_share         = 0.0;
    std::string pool_endpoint;
};

struct ChallengeInfo {
    int         puzzle_number      = 0;
    int         puzzle_bits        = 0;
    uint64_t    ops_completed      = 0;
    uint64_t    expected_ops       = 0;
    uint64_t    dps_found          = 0;
    std::string backend_name;
    std::string range_label;
};

struct BipScanInfo {
    uint64_t    phrases_read       = 0;
    uint64_t    phrases_valid      = 0;
    uint64_t    addresses_probed   = 0;
    uint64_t    bloom_hits         = 0;
    std::string current_profile;
    std::string mode_label;
    int         word_count = 0;
    uint64_t    bloom_elements = 0;
    int         derivation_profiles = 0;
    int         addresses_per_phrase = 0;
    unsigned    worker_threads = 0;
    int         gpu_count = 0;
    double      phrases_per_sec = 0.0;
    double      addresses_per_sec = 0.0;
    struct GpuShare {
        int       device_id = 0;
        uint64_t  addresses_dispatched = 0;
        double    addresses_per_sec = 0.0;
    };
    std::vector<GpuShare> gpu_shares;
    struct FaultedDevice {
        int         device_id = 0;
        std::string error;
    };
    std::string gpu_init_message;
    std::vector<FaultedDevice> gpu_faulted_devices;
    int         gpu_count_requested = 0;
    bool        pbkdf_gpu_active = false;
    bool        gpu_disabled_by_flag = false;
};

}  // namespace collider::ui::tui
