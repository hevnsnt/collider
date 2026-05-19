// instrumentation_hook.cpp: storage for the gpu-layer instrumentation hooks.
// See instrumentation_hook.hpp for the design contract.
//
// The callback slots live here so that translation units which only include
// the header (e.g. fused_pipeline.cu) do not each get a private definition.
// The default value of both slots is nullptr; the runtime layer overwrites
// them in its startup adapter. When no adapter is registered the launch
// path is single-load + branch and never indirects through a function
// pointer.

#include "instrumentation_hook.hpp"

namespace collider::gpu::instrumentation {

std::atomic<StartCallback> g_start_cb{nullptr};
std::atomic<StopCallback>  g_stop_cb{nullptr};

void register_callbacks(StartCallback start, StopCallback stop) noexcept {
    // Memory order: release on the writer + acquire on the reader is not
    // strictly required because the callbacks themselves carry no
    // user-visible state -- they take stage / stream / token and return
    // a token. relaxed is sufficient. The runtime adapter only registers
    // once, so contention on these slots is nil.
    g_start_cb.store(start, std::memory_order_relaxed);
    g_stop_cb.store(stop,   std::memory_order_relaxed);
}

}  // namespace collider::gpu::instrumentation
