// snapshot.hpp -- FREE STUB (installed by scripts/sync-to-free.sh)
//
// Snapshot types passed between the runner thread and the FTXUI
// render thread in Pro. In free no render thread exists, but the
// types are referenced as parameter types on TuiApp::post_*_snapshot
// methods and on GpuTelemetrySampler::Config::post_fn lambda, so
// they need to exist as POD structs.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace collider::ui::tui {

struct GpuTelemetrySnapshot {
    struct Device {
        int       device_id        = 0;
        int       sm_pct           = 0;
        int       mem_pct          = 0;
        int       temp_c           = 0;
        int       fan_pct          = 0;
        int       power_w          = 0;
        int       power_limit_w    = 0;
        int       sm_clock_mhz     = 0;
        int       mem_clock_mhz    = 0;
        uint64_t  vram_used_bytes  = 0;
        uint64_t  vram_total_bytes = 0;
        std::string name;
        std::string pcie_gen;
    };
    std::vector<Device> devices;
};

struct ScanSnapshot {
    uint64_t total_checked            = 0;
    uint64_t total_dispatches         = 0;
    uint64_t bloom_hits               = 0;
    uint64_t bloom_collisions_filtered = 0;
    uint64_t tight_bloom_filtered     = 0;
    uint64_t verified_hits            = 0;
    int      current_phase            = 0;
    int      phase_iteration          = 0;
    std::array<uint64_t, 5> empty_hits_by_phase{};
    uint64_t dispatch_words_per_gpu   = 0;
    double   keys_per_sec_current     = 0.0;
    double   keys_per_sec_avg         = 0.0;
};

}  // namespace collider::ui::tui
