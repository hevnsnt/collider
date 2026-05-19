// instrumentation_hook.hpp: minimal gpu-layer-owned instrumentation surface.
//
// The fused brain-wallet pipeline (fused_pipeline.cu) emits CUDA-event-based
// kernel-launch timing samples for the perf panel. Previously the .cu file
// reached UP into the runtime layer (#include "runtime/perf_instrumentation.hpp")
// and called the runtime PerfCollector singleton directly. That reversed the
// natural module layering: GPU is a leaf layer, runtime sits on top of it,
// and headers must not point the wrong way.
//
// This header inverts the dependency. The gpu layer defines an opaque
// callback registry; the runtime layer registers an adapter at startup that
// forwards into its PerfCollector. fused_pipeline.cu only sees this header
// and the small token type below; it has no compile-time dependency on
// any runtime/* header. If the runtime layer is unused (e.g. in tests or
// in a free build that excludes perf), the callbacks remain null and the
// instrument_* entry points become single-load + branch no-ops.
//
// Lifetime: callbacks are set once at runtime-init and read on every kernel
// launch. We use std::atomic on the function-pointer slots so the launch
// path performs a relaxed-load + predicted-not-taken branch when no adapter
// is registered.

#pragma once

#include <atomic>
#include <cstdint>

namespace collider::gpu::instrumentation {

// Kernel identifiers exposed by the gpu layer. The list is intentionally
// minimal -- only the stages the gpu layer itself attributes timing to. The
// runtime adapter maps these to its own KernelId enum.
enum class StageId : int {
    EcMul = 0,
    BloomProbe,
    Count_
};

// Slot token returned by start_cb. Opaque to callers; the runtime adapter
// packs its own KernelId + ring slot into the .opaque field. The launch site
// stores the token on its stack and threads it back into stop_cb so the
// runtime collector can pair start/stop events under multi-stream dispatch.
// A token with kInvalidOpaque == 0xFFFFFFFFFFFFFFFFull means the start
// callback either was not registered, was disabled, or refused the request;
// the matching stop_cb call must be a no-op in that case.
struct StageToken {
    std::uint64_t opaque{0xFFFFFFFFFFFFFFFFull};

    static constexpr std::uint64_t kInvalid = 0xFFFFFFFFFFFFFFFFull;

    bool valid() const noexcept { return opaque != kInvalid; }
};

// Callback signatures. `stream` is an opaque cudaStream_t (void* so this
// header has zero cuda_runtime dependency; the runtime adapter casts back).
using StartCallback = StageToken (*)(StageId stage, void* stream) noexcept;
using StopCallback  = void (*)(StageToken token, void* stream) noexcept;

// Adapter registration. Called once by the runtime layer at startup. Passing
// nullptr to either slot disables that direction (useful for tests).
void register_callbacks(StartCallback start, StopCallback stop) noexcept;

// Launch-site entry points. These are inline so the disabled path is a
// single relaxed load + branch; the compiler can hoist the test out of
// the kernel-launch hot region when both callbacks are null.

// Internal slot pointers. Defined in instrumentation_hook.cpp.
extern std::atomic<StartCallback> g_start_cb;
extern std::atomic<StopCallback>  g_stop_cb;

inline StageToken instrument_start(StageId stage, void* stream) noexcept {
    auto cb = g_start_cb.load(std::memory_order_relaxed);
    if (cb == nullptr) return StageToken{};
    return cb(stage, stream);
}

inline void instrument_stop(StageToken token, void* stream) noexcept {
    auto cb = g_stop_cb.load(std::memory_order_relaxed);
    if (cb == nullptr) return;
    if (!token.valid()) return;
    cb(token, stream);
}

}  // namespace collider::gpu::instrumentation
