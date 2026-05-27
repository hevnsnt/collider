// gpu_telemetry.hpp -- FREE STUB (installed by scripts/sync-to-free.sh)
//
// In Pro, GpuTelemetrySampler runs a background thread that polls
// NVML at 2 Hz and posts GpuTelemetrySnapshot updates to the TUI's
// GPU panel. Free has no GPU panel; the runner doesn't need the
// telemetry stream. The stub satisfies the symbol surface so the
// runner's optional sampler setup compiles cleanly.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>

#include "ui/tui/snapshot.hpp"

namespace collider::platform {

class GpuTelemetrySampler {
public:
    struct Config {
        std::vector<int>          cuda_device_ids;
        std::chrono::milliseconds interval{500};
        std::function<void(const ::collider::ui::tui::GpuTelemetrySnapshot&)> post_fn;
    };

    GpuTelemetrySampler() = default;
    explicit GpuTelemetrySampler(Config /*cfg*/) {}

    void start() noexcept {}
    void stop()  noexcept {}
};

}  // namespace collider::platform
