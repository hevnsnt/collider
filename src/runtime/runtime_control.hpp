// RuntimeControlState: atomic snapshot of operator-requested tuning state
// crossing from the TUI input thread into the scan loop. Writers are the
// FTXUI keyboard handler and TUI render thread; the scan loop polls at
// batch (and a few phase) boundaries.
// See docs/internals/runtime-control.md for design rationale.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>

// Q-T1.1 inversion (2026-05-17): the dispatch cap is owned by the gpu layer
// (it bounds PTX-side per-device tables, RCKangaroo's MAX_GPU_CNT, the
// puzzle slot table, etc.). The runtime per-GPU phase array MUST match. We
// pull the gpu cap here and static_assert the equality below so a change
// in either direction fails at compile time.
#include "../gpu/gpu_caps.hpp"

namespace collider::runtime {

struct RuntimeControlState {
    // ===== Lifecycle ===================================================
    // quit_requested is set by the 'q' keypress (and the SIGINT handler
    // path inside CookedModeGuard). The scan loop polls this at every
    // batch boundary and flips g_shutdown when it sees true. The reader
    // does NOT clear this flag; it is a one-way latch from "running" to
    // "shutting down".
    std::atomic<bool> quit_requested{false};

    // ===== Pause / resume ==============================================
    // pause_requested is toggled by 'p'. While true, the scan loop drains
    // its in-flight GPU dispatches at the next batch boundary and then
    // sets is_paused=true. While paused, the scan loop spins reading
    // RuntimeControlState (so save_requested / quit_requested / GPU
    // toggles still work) until pause_requested goes false, at which
    // point it clears is_paused and resumes normal dispatch.
    std::atomic<bool> pause_requested{false};
    std::atomic<bool> is_paused{false};

    // ===== Save now ====================================================
    // Toggled by 's'. The scan loop bypasses its save_interval throttle
    // on the next batch boundary, persists the current streaming-gen
    // state, and clears the flag.
    std::atomic<bool> save_requested{false};

    // ===== GPU enable mask + per-GPU phase =============================
    // The TUI keybind 'g<N>' flips bit N of gpu_enable_mask. The scan
    // loop reads the mask and gpu_phase[i] at every batch boundary:
    //   - mask bit cleared + gpu_phase[i] == Active  ==>  transition to
    //     Draining, sync the GPU's stream, call drain_and_free() on the
    //     rule engine + brain-wallet context for that device, set
    //     gpu_phase[i] = Disabled.
    //   - mask bit set + gpu_phase[i] == Disabled    ==>  transition to
    //     Initializing, re-init the rule engine + brain-wallet context
    //     with the original config, set gpu_phase[i] = Active.
    // The dispatch path skips any GPU whose phase is not Active.
    // Up to kMaxGpus (8) devices supported; matches the existing GPU
    // detection ceiling in the rest of the codebase.
    static constexpr int kMaxGpus = ::collider::gpu::kMaxDispatchableGpus;
    static_assert(kMaxGpus == 8,
                  "kMaxGpus tracks gpu::kMaxDispatchableGpus; bump both at "
                  "once after auditing the rule-engine per-GPU work-balancer "
                  "stack arrays and third_party/RCKangaroo/defs.h MAX_GPU_CNT.");
    std::atomic<uint8_t> gpu_enable_mask{0xff};
    enum class GpuPhase : uint8_t {
        Active = 0,
        Draining = 1,
        Disabled = 2,
        Initializing = 3,
        // T1.1 (2026-05-17): GPU encountered an unrecoverable context fault
        // (illegal address, misaligned address, launch-out-of-resources,
        // bloom load OOM). The dispatch loop must treat this identically to
        // Disabled: skip every chunk dispatch on this GPU until process
        // restart. Set from gpu_rules.cpp::apply_rules_to_words_gpu and
        // brain_wallet_gpu.cpp::process_batch_from_gpu when their
        // post-launch error check returns a sticky context error. An
        // optional best-effort cudaDeviceReset + reinit may transition
        // Faulted back to Active; if reinit fails the GPU stays Faulted.
        Faulted = 4,
    };
    std::array<std::atomic<GpuPhase>, kMaxGpus> gpu_phase{};

    // ===== Batch size tuning ===========================================
    // The TUI '+' / '-' keys nudge requested_batch_size. The scan loop
    // reads it at every batch boundary; if it differs from the current
    // batch size it tries to reallocate the GPU buffers at the new size
    // and, on success, sets last_applied_batch_size + clears the request.
    // On cudaMalloc failure it logs a banner and clears the request
    // without touching last_applied (i.e. the previous size stays in
    // effect).
    std::atomic<uint64_t> requested_batch_size{0};
    std::atomic<uint64_t> last_applied_batch_size{0};

    // ===== Rule chunk size cycle =======================================
    // The 'r' key cycles through the chunk-size values {200, 500, 1000}.
    // The scan loop interprets a non-zero requested_rule_chunk_size as
    // "re-init the rule engines with this max_rules at the next batch
    // boundary"; on success it sets last_applied + clears the request.
    std::atomic<uint64_t> requested_rule_chunk_size{0};
    std::atomic<uint64_t> last_applied_rule_chunk_size{0};

    // ===== Bloom hot-swap (phase-boundary swap, not batch-boundary) ===
    // The bloom picker modal writes a target path into requested_bloom_path
    // under bloom_mu. The scan loop polls at every PHASE boundary (not
    // batch boundary; in-flight bloom probes must complete first) and
    // performs a full drain + per-GPU reload. On success it copies the
    // path into last_applied_bloom_path + clears requested_bloom_path;
    // on failure it logs a banner and clears the request.
    mutable std::mutex bloom_mu;
    std::string requested_bloom_path;
    std::string last_applied_bloom_path;

    // ===== Wordlist hot-swap (phase-boundary swap) =====================
    // The wordlist picker modal writes a target profile path into
    // requested_wordlist_profile under profile_mu. The scan loop polls
    // at every PHASE boundary (not batch boundary) and forwards the new
    // profile to StreamingBrainWallet::queue_profile_swap, which in turn
    // applies the swap at the next phase advance. On success the runner
    // copies the path into last_applied_wordlist_profile and clears
    // requested_wordlist_profile; on failure it banners the operator
    // and clears the request without touching last_applied.
    // Mirrors the bloom path pattern above so the input handler and the
    // scan loop can use the same one-shot writer / consumer convention.
    mutable std::mutex profile_mu;
    std::string requested_wordlist_profile;
    std::string last_applied_wordlist_profile;

    // ===== Theme cycle (consumed by TUI render thread, not scan loop) =
    // The 't' key cycles Default -> HighContrast -> Monochrome -> Light
    // -> Default. -1 means "no change"; 0..3 are the four variant indices
    // matching collider::ui::tui::ThemeVariant. The TUI render thread is
    // the only reader; the scan loop ignores this field.
    std::atomic<int> requested_theme_variant{-1};

    // ===== Focused-panel mode (Tier 2 D1; render thread only) ========
    // The Ctrl+1..4 keys focus a single panel; Ctrl+0 / Esc clears.
    // Encoding:
    //    -1  no focus (default 2-column layout)
    //     0  status panel
    //     1  GPU panel
    //     2  performance panel
    //     3  plugins panel
    // The TUI render thread is the only reader; the scan loop ignores
    // this field. The Ctrl+N input dispatch is the writer; the render
    // thread reads at every frame to decide layout. Cleared back to -1
    // when Esc / Ctrl+0 is pressed.
    static constexpr int kFocusNone = -1;
    static constexpr int kFocusStatus = 0;
    static constexpr int kFocusGpu = 1;
    static constexpr int kFocusPerformance = 2;
    static constexpr int kFocusPlugins = 3;
    std::atomic<int> requested_focused_panel{kFocusNone};

    // ===== Banner message ==============================================
    // The scan loop sets a short banner string after any user-action
    // change ("Batch size set to 6M", "GPU 0 disabled", "Bloom reloaded:
    // funded.blf"). The TUI status panel reads + dim-renders this for
    // ~5 seconds after banner_set_at, then clears.
    mutable std::mutex banner_mu;
    std::string banner_text;
    std::chrono::steady_clock::time_point banner_set_at;

    void set_banner(std::string msg) noexcept {
        std::lock_guard<std::mutex> lk(banner_mu);
        banner_text = std::move(msg);
        banner_set_at = std::chrono::steady_clock::now();
    }

    // Read a copy of the banner text + age. Returns empty string if the
    // banner has been cleared or is older than max_age_ms. Used by the
    // TUI status panel; lockable from any thread.
    std::string get_banner(int max_age_ms = 5000) const {
        std::lock_guard<std::mutex> lk(banner_mu);
        if (banner_text.empty()) return {};
        const auto age = std::chrono::steady_clock::now() - banner_set_at;
        const auto age_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(age).count();
        if (age_ms > max_age_ms) return {};
        return banner_text;
    }

    // Read the requested bloom path under bloom_mu. Empty string means
    // "no swap requested".
    std::string get_requested_bloom_path() const {
        std::lock_guard<std::mutex> lk(bloom_mu);
        return requested_bloom_path;
    }

    // Set the requested bloom path. Called by the bloom picker modal.
    void set_requested_bloom_path(std::string path) {
        std::lock_guard<std::mutex> lk(bloom_mu);
        requested_bloom_path = std::move(path);
    }

    // Atomically consume the requested bloom path (returns the value and
    // clears the field). Called by the scan loop at phase boundaries.
    std::string take_requested_bloom_path() {
        std::lock_guard<std::mutex> lk(bloom_mu);
        std::string out = std::move(requested_bloom_path);
        requested_bloom_path.clear();
        return out;
    }

    // Publish the applied bloom path under bloom_mu. Called by the scan
    // loop after a successful swap.
    void set_last_applied_bloom_path(std::string path) {
        std::lock_guard<std::mutex> lk(bloom_mu);
        last_applied_bloom_path = std::move(path);
    }

    std::string get_last_applied_bloom_path() const {
        std::lock_guard<std::mutex> lk(bloom_mu);
        return last_applied_bloom_path;
    }

    // Wordlist profile path mirror of the bloom path accessors above.
    // Empty string means "no swap requested".
    std::string get_requested_wordlist_profile() const {
        std::lock_guard<std::mutex> lk(profile_mu);
        return requested_wordlist_profile;
    }

    void set_requested_wordlist_profile(std::string path) {
        std::lock_guard<std::mutex> lk(profile_mu);
        requested_wordlist_profile = std::move(path);
    }

    // Atomically consume the requested wordlist profile path (returns
    // the value and clears the field). Called by the scan loop at phase
    // boundaries.
    std::string take_requested_wordlist_profile() {
        std::lock_guard<std::mutex> lk(profile_mu);
        std::string out = std::move(requested_wordlist_profile);
        requested_wordlist_profile.clear();
        return out;
    }

    void set_last_applied_wordlist_profile(std::string path) {
        std::lock_guard<std::mutex> lk(profile_mu);
        last_applied_wordlist_profile = std::move(path);
    }

    std::string get_last_applied_wordlist_profile() const {
        std::lock_guard<std::mutex> lk(profile_mu);
        return last_applied_wordlist_profile;
    }

    // T1.1 (2026-05-17): mark a GPU as Faulted. Called from the GPU launch
    // wrappers (gpu_rules.cpp, brain_wallet_gpu.cpp) when a sticky context
    // error survives the launch and the device is no longer safe to
    // dispatch onto. Idempotent: re-marking an already-Faulted device is a
    // no-op. The dispatch loop in brain_wallet_runner polls per-GPU phase
    // each chunk and skips Faulted devices the same way it skips Disabled
    // ones. gpu_index < 0 or gpu_index >= kMaxGpus is silently ignored
    // (the runner only seeds phases for the active GPU set).
    //
    // The `reason` argument is a short label (e.g. "fused-async-fault: an
    // illegal memory access was encountered") that the runtime preserves
    // so the gpu_faulted milestone can carry the actual CUDA error string
    // into the persistent log. Without this, a post-mortem reader can only
    // see "GPU N removed from enable mask" without knowing WHY, which was
    // the diagnostic gap that hid the post-hit-fault interaction.
    void mark_gpu_faulted(int gpu_index, std::string_view reason = {}) noexcept {
        if (gpu_index < 0 || gpu_index >= kMaxGpus) return;
        gpu_phase[gpu_index].store(GpuPhase::Faulted,
                                   std::memory_order_release);
        if (!reason.empty()) {
            std::lock_guard<std::mutex> lk(fault_reason_mu);
            gpu_fault_reason[gpu_index].assign(reason);
        }
    }

    // Read the reason captured by the most recent mark_gpu_faulted call
    // for `gpu_index`. Empty string when no fault has been recorded.
    // Caller holds the snapshot under the mutex; safe to format into
    // log payloads on any thread.
    std::string get_gpu_fault_reason(int gpu_index) const {
        if (gpu_index < 0 || gpu_index >= kMaxGpus) return {};
        std::lock_guard<std::mutex> lk(fault_reason_mu);
        return gpu_fault_reason[gpu_index];
    }

private:
    mutable std::mutex fault_reason_mu;
    std::array<std::string, kMaxGpus> gpu_fault_reason{};
};

// Q-T1.1 inversion (2026-05-17): the gpu fault path used to be wired
// directly via #include "../runtime/runtime_control.hpp" in every gpu/*
// fault site (brain_wallet_gpu.cpp, gpu_rules.cpp, puzzle_optimized.cu,
// rckangaroo_wrapper.cu). That reversed the natural module layering: gpu
// is a leaf of runtime. The inversion: gpu now owns a generic observer
// surface (gpu_fault_observer.hpp); the runtime side registers an adapter
// that forwards reports into RuntimeControlState::mark_gpu_faulted. The
// adapter lives inline below so we do not need an accompanying .cpp file.
//
// install_gpu_fault_callback() is the public entry point: it installs the
// bridge adapter on the gpu observer slot. The function is idempotent
// (the gpu slot is a single atomic; the adapter is the same function
// pointer each time). global_runtime_control() invokes it on first
// access so every PRO startup path (brain_wallet_runner, tui_app,
// input_handler, puzzle/pool solvers) wires the bridge automatically.
// Explicit startup callers may also invoke install_gpu_fault_callback()
// directly to self-document the wiring.
inline void install_gpu_fault_callback() noexcept;

// Singleton accessor. Exactly one RuntimeControlState per process; both
// the TUI input thread and the scan loop reference it. Inline because
// C++11 function-local statics are thread-safe-init and the indirection
// would otherwise force a stand-alone TU just for this 3-line body.
//
// First-access side effect: installs the gpu fault bridge so
// report_fault() forwards into mark_gpu_faulted as it always did. The
// install is performed exactly once via a function-local static; the
// _anchor variable is a [[maybe_unused]] side-effect handle.
inline RuntimeControlState& global_runtime_control() noexcept {
    static RuntimeControlState instance;
    static const auto _anchor = []() noexcept {
        install_gpu_fault_callback();
        return 0;
    }();
    (void)_anchor;
    return instance;
}

}  // namespace collider::runtime

// Bridge adapter. Defined after the namespace close so it can include the
// gpu observer header without forward-declaration noise inside the runtime
// namespace. Inline definition keeps everything header-only; the inline
// variable in gpu_fault_observer.hpp gives the adapter slot a single
// process-wide instance under ODR.
#include "../gpu/gpu_fault_observer.hpp"

namespace collider::runtime {

namespace detail {

// Forwards a gpu-side fault report into the runtime's per-GPU phase
// bookkeeping. The string_view `reason` is preserved in the runtime
// state so the gpu_faulted milestone payload can surface the actual
// CUDA error string into the persistent log.
inline void gpu_fault_bridge_adapter(int device_id,
                                     std::string_view reason) noexcept {
    global_runtime_control().mark_gpu_faulted(device_id, reason);
}

}  // namespace detail

inline void install_gpu_fault_callback() noexcept {
    ::collider::gpu::set_fault_callback(&detail::gpu_fault_bridge_adapter);
}

}  // namespace collider::runtime
