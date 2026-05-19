// gpu_fault_observer.hpp -- gpu-layer-owned fault report surface.
//
// Q-T1.1 inversion (2026-05-17): the GPU translation units used to reach UP
// into the runtime layer via #include "../runtime/runtime_control.hpp" and
// call ::collider::runtime::global_runtime_control().mark_gpu_faulted(id)
// directly. That reversed the natural module layering: gpu is a leaf, the
// runtime layer is built on top of it, and a leaf header pointing at a
// caller's header (a) introduces a circular #include risk the moment any
// runtime/* header wants to pull in a gpu/* type, and (b) ties the leaf
// to a singleton it has no business knowing about (e.g. unit tests of
// gpu kernels had to either link the runtime layer or stub out
// RuntimeControlState just to build).
//
// This header inverts the dependency. The gpu layer declares an observer
// surface (`report_fault` + a callback slot); the runtime layer registers
// the callback at startup that does the actual mark_gpu_faulted
// bookkeeping. fused_pipeline.cu / brain_wallet_gpu.cpp / gpu_rules.cpp /
// puzzle_optimized.cu / rckangaroo_wrapper.cu only see this header.
//
// Lifetime: the callback is set once at runtime-init and read on every
// kernel fault. We use std::atomic on the function-pointer slot so the
// fault path performs a relaxed-load + predicted-not-taken branch when
// no callback is registered (e.g. in a unit test that doesn't link the
// runtime layer). When the callback is null, report_fault is a no-op.
//
// Storage is header-only via an inline variable (C++17) so this header
// can ship without an accompanying .cpp file. Every TU that includes
// this header sees the same atomic slot under the One Definition Rule
// guarantee for inline variables.
//
// This mirrors the instrumentation_hook.hpp inversion that the perf
// collector already uses; the design rationale + memory-order argument
// from that header applies verbatim here.

#pragma once

#include <atomic>
#include <string_view>

namespace collider::gpu {

// Adapter signature. `device_id` is the CUDA device index reported by
// cudaGetDevice (matches the gpu_id the runtime stores in its per-GPU
// phase array). `reason` is a free-form short label the adapter forwards
// to its log / banner ("fused-kernel-launch", "async-fault",
// "bloom-set-config", etc.); the gpu layer does not interpret it.
using FaultCallback = void (*)(int device_id, std::string_view reason) noexcept;

// Internal slot pointer. C++17 inline variable so the storage is shared
// across every TU that includes this header without an accompanying .cpp.
inline std::atomic<FaultCallback> g_fault_cb{nullptr};

// Adapter registration. Called once by the runtime layer at startup.
// Passing nullptr disables fault reporting (useful for tests).
//
// Memory order: the callback itself carries no user-visible state (it
// takes int + string_view and returns void). The runtime adapter
// registers exactly once at startup; the gpu-side readers only need to
// eventually see the non-null value. relaxed on both sides is sufficient,
// matching the analogous reasoning in instrumentation_hook.cpp.
inline void set_fault_callback(FaultCallback cb) noexcept {
    g_fault_cb.store(cb, std::memory_order_relaxed);
}

// Fault-site entry point. Inline so the disabled path is a single
// relaxed load + branch; the compiler hoists the test out of the cold
// fault-handling region. Safe to call from any thread.
inline void report_fault(int device_id, std::string_view reason) noexcept {
    auto cb = g_fault_cb.load(std::memory_order_relaxed);
    if (cb == nullptr) return;
    cb(device_id, reason);
}

}  // namespace collider::gpu
