// session_log.hpp -- per-process session log + atomic session state +
// crash handler for theCollider.
//
// Why this exists, what it is NOT:
//   The existing collider::Logger (core/logger.hpp) is a single rotating
//   ~/.collider/collider.log singleton built for human-tail diagnosis. It
//   stays. What it does not do:
//     * Distinguish one process invocation from another. Two collider
//       runs in the same hour share a log file with interleaved lines.
//     * Capture a structured snapshot of the operational state that a
//       crash handler can dump.
//     * Survive an asynchronous SEH / SIGSEGV cleanly: the rotating log
//       loses the most recent pre-crash buffer because std::ofstream is
//       not signal-safe.
//
//   This header adds a parallel session log that:
//     * Opens one file per process invocation at
//       ~/.collider/logs/collider-<YYYY-MM-DDTHH-MM-SS>-<pid>.log so
//       every crash diagnoses against a single, complete file.
//     * Flushes after every milestone (NOT every line) so the file is
//       readable mid-run without paying flush cost on hot paths.
//     * Writes a side-car ~/.collider/session_state.json atomically
//       (tmp + rename) holding the structured operational snapshot
//       a recovery tool would want to read.
//     * Installs a Windows SetUnhandledExceptionFilter / POSIX signal
//       handler that writes ~/.collider/crash-<ts>.log with the
//       exception code, a best-effort stack walk, and the verbatim
//       contents of the last session_state.json.
//
// Naming: the public entry points sit in the existing
// `collider::log` namespace (the same namespace that hosts
// core/log.hpp's ScopedLine helper) so call sites use a single, short
// prefix.
//
// Lifecycle:
//   1. init_session_log()        -- once at process start, AFTER paths
//                                   are callable. Returns false on
//                                   failure; the rest of the API is
//                                   a no-op when not initialized.
//   2. write_startup_banner(...) -- once, after CLI parse completes.
//   3. install_crash_handler()   -- once, right after the banner.
//   4. write_hardware_enum(...)  -- once, after GPU detection.
//   5. milestone(event, detail)  -- called at every interesting
//                                   transition (auth, work, phase
//                                   change, save, hit, fault, etc.).
//                                   Each call flushes.
//   6. update_session_state(...) -- called on every batch; the
//                                   internal throttle ensures the
//                                   atomic on-disk write happens at
//                                   most once every ~5 seconds.
//   7. shutdown_session_log()    -- explicit teardown. init_session_log()
//                                   registers this as a std::atexit hook
//                                   so a clean process exit always
//                                   force-flushes the latest SessionState
//                                   snapshot to disk regardless of which
//                                   thread executed the final code path.
//                                   Safe to call additionally from tests
//                                   that need deterministic teardown.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "core/paths.hpp"

// Arguments lives at global scope (declared in cli/cli_parser.hpp,
// outside any namespace). Forward-declare it here at the same global
// scope so the public API can take it by const-ref without dragging
// the full cli_parser.hpp into every consumer.
struct Arguments;

namespace collider {

namespace platform {
struct DeviceInfo;
}  // namespace platform

namespace log {

// ---------------------------------------------------------------------------
// SessionState -- operational snapshot mirrored to ~/.collider/session_state.json
// ---------------------------------------------------------------------------
//
// Every field except the always-on bookkeeping (mode, pid, log_path,
// boot_ts, last_update_ts) is an optional so the JSON serializer can
// omit them rather than emit zero / empty defaults. A reader can tell
// "field unset" from "field zero" unambiguously.
//
// The struct is intentionally flat. Nested groups (per-GPU runtime,
// per-work counters) live in dedicated sub-structs.
struct SessionState {
    // Common (always written when the session is initialized)
    std::string mode;             // "brainwallet" / "puzzle" / "pool" / "brainwallet-v2" / "benchmark"
    int pid = 0;
    std::string log_path;
    std::chrono::system_clock::time_point boot_ts{};
    std::chrono::system_clock::time_point last_update_ts{};

    // ----- Brainwallet mode -----
    std::optional<uint64_t> total_checked;
    std::optional<uint64_t> bloom_hits;
    std::optional<std::string> wordlist_path;
    std::optional<uint64_t> wordlist_hash;
    std::optional<int> current_phase_idx;
    std::optional<std::string> current_phase_name;
    std::optional<uint64_t> last_save_count;

    // ----- Pool mode (the BTC challenge focus) -----
    std::optional<uint64_t> current_work_id;
    std::optional<std::string> work_range_start_hex;
    std::optional<std::string> work_range_end_hex;
    std::optional<int> work_dp_bits;
    std::optional<uint64_t> dp_count_submitted_this_work;  // total since current_work_id assigned
    std::optional<uint64_t> dp_count_submitted_total;       // process lifetime
    std::optional<uint32_t> dp_seq_last;                    // monotonic counter sent
    std::optional<std::chrono::system_clock::time_point> work_started_at;
    std::optional<std::chrono::system_clock::time_point> last_dp_submit_at;
    std::optional<bool> connected;
    std::optional<std::string> pool_endpoint;

    // ----- Puzzle mode -----
    std::optional<int> puzzle_number;
    std::optional<std::string> puzzle_algorithm;        // "RCKangaroo" / "MultiGPU" / "CPU" / "GPU brute" / "CPU brute"
    std::optional<uint64_t> total_steps;
    std::optional<std::string> position_full_hex;       // 256-bit current position (brute) or N/A (kangaroo)

    // ----- GPU runtime state (per device) -----
    struct GpuRuntime {
        int device_id = 0;
        std::string name;
        std::string phase;                  // "Active" / "Disabled" / "Faulted" / "Initializing" / "Draining"
        std::optional<int> util_pct;
        std::optional<int> power_w;
        std::optional<int> temp_c;
        std::optional<int> pcie_gen;
    };
    std::vector<GpuRuntime> gpus;
};

// ---------------------------------------------------------------------------
// Lifecycle / initialization
// ---------------------------------------------------------------------------

// Initialize the per-session log file at
//   ~/.collider/logs/collider-<YYYY-MM-DDTHH-MM-SS>-<pid>.log
// in append mode. Rotates older session log files in that directory so
// at most kSessionLogRetainCount survive (oldest deleted first).
//
// Returns true on success. False means the API is in no-op mode for
// the rest of this process (every other function returns without
// side effects). Failure paths: USERPROFILE / HOME unresolved AND the
// "." fallback unwritable, std::ofstream open failure (disk full,
// permissions), etc.
//
// Safe to call exactly once per process. Subsequent calls are no-ops
// (return true to preserve idempotency for early-init guards).
bool init_session_log();

// Write the startup banner block. Includes:
//   * theCollider version (collider::kVersion) + build flags
//     (Pro / Free, CUDA / Metal / CPU, NVML support, OpenSSL support).
//   * Build timestamp (__DATE__ / __TIME__).
//   * Git SHA when COLLIDER_GIT_SHA is wired through the build.
//   * Full argv as a single space-delimited line, with
//     --pool-password / --pool-password-file values redacted and
//     --activate license keys redacted.
//   * Resolved config file path (args.config_file or the default).
//   * Current working directory (std::filesystem::current_path).
//   * Process PID.
//   * Boot timestamp.
//
// Idempotent: calling twice writes two banners (useful for crash
// recovery diagnostics that re-banner after a reconnect storm).
void write_startup_banner(int argc, char** argv, const Arguments& args);

// Write the hardware enumeration block, one indented sub-block per
// device:
//   * device_id, name, vendor
//   * compute_major.minor (SM)
//   * total VRAM in MB, free VRAM in MB
//   * multiprocessor_count
//   * NVML symbol availability (resolved at runtime; logged regardless
//     of whether nvml linked in this build)
//   * supports_fp16 / supports_int8
//
// Each device that reports an SM NOT in the compile-time arch set
// (COLLIDER_CUDA_ARCH_LIST) also fires a milestone("sm_mismatch", ...)
// so the warning lands in both the session log AND collider.log via
// the existing logger.
void write_hardware_enum(const std::vector<platform::DeviceInfo>& devices);

// ---------------------------------------------------------------------------
// Activity / milestone events
// ---------------------------------------------------------------------------

// Append a milestone event to the session log. `event` is a short
// keyword (e.g. "bloom_loaded", "auth_ok", "work_received"); `detail`
// is free-form context (path, counts, identifiers). Always flushes
// the log file so the line survives a SIGKILL.
//
// Calls are serialized through an internal mutex. Safe from any
// thread. NOT safe from a signal handler (uses std::ofstream + std::
// stringstream + the mutex). The crash handler installed by
// install_crash_handler() is its own dedicated, signal-safe writer.
void milestone(const char* event, const std::string& detail);

// Convenience overload for the common "no detail" case.
inline void milestone(const char* event) { milestone(event, std::string{}); }

// ---------------------------------------------------------------------------
// Session state persistence
// ---------------------------------------------------------------------------

// Update the on-disk ~/.collider/session_state.json snapshot atomically
// (tmp + rename). Debounced internally: callers can fire this on every
// batch / every DP submission; the actual disk write happens at most
// once every kSessionStateMinIntervalMs (5 seconds by default). The
// most recent state is always serialized at process exit and at every
// milestone() call regardless of throttle (so the crash dump always
// sees the latest snapshot).
//
// The struct is copied internally; the caller may modify or destroy
// `state` immediately after the call returns.
void update_session_state(const SessionState& state);

// Force-flush the most recent SessionState to disk immediately, ignoring
// the debounce throttle. Called internally on every milestone() and on
// shutdown so the crash dump path has a fresh snapshot. Exposed for the
// rare external caller (test harnesses, the explicit "save now" UI
// action) that wants an immediate flush.
void flush_session_state();

// Explicit teardown. Force-flushes the SessionStateStore and writes a
// final "SHUTDOWN" sentinel line to the per-session log. init_session_log()
// installs this as a std::atexit hook so a clean process exit always
// lands a current state.json; tests may call it directly to drain state
// before a fixture tear-down. Idempotent: subsequent calls are no-ops.
void shutdown_session_log();

// ---------------------------------------------------------------------------
// Crash handler
// ---------------------------------------------------------------------------

// Install the OS-level crash handler:
//
//   Windows: SetUnhandledExceptionFilter is set to a filter that
//     writes ~/.collider/crash-<ts>.log with the exception code +
//     faulting address, a best-effort StackWalk64 + symbol lookup
//     (links dbghelp.lib), the verbatim contents of the last
//     session_state.json on disk, and the process uptime; then
//     calls TerminateProcess so we do not loop.
//
//   POSIX: signal handlers for SIGSEGV, SIGABRT, SIGFPE, SIGBUS are
//     installed via sigaction with SA_SIGINFO. The handler writes
//     the same crash file using async-signal-safe primitives only
//     (open / write / _exit; backtrace + backtrace_symbols_fd for
//     the stack). The verbatim session_state.json is read with a
//     plain ::read() loop, also async-signal-safe.
//
// Both paths pre-allocate the crash buffer at install time so the
// handler does not allocate in signal context.
//
// Safe to call exactly once per process. Subsequent calls are no-ops.
void install_crash_handler();

// ---------------------------------------------------------------------------
// Tunables (exposed for the test suite; not configurable from the CLI)
// ---------------------------------------------------------------------------

// Keep at most this many old session log files in ~/.collider/logs/.
// Older files are deleted on init_session_log(). Set conservatively:
// each file is a few KB to a few MB depending on run length; 20 covers
// "the last week or so of typical use" without crowding disk.
inline constexpr int kSessionLogRetainCount = 20;

// Minimum interval between two on-disk writes of session_state.json
// triggered by update_session_state(). milestone() and shutdown bypass
// this throttle. 5 seconds chosen so the JSON is fresh enough for a
// recovery tool to be useful, but the disk-write rate stays bounded
// even on a fully-saturated DP submission path (which can fire
// update_session_state() dozens of times per second).
inline constexpr int kSessionStateMinIntervalMs = 5000;

// ---------------------------------------------------------------------------
// Test-only hooks
// ---------------------------------------------------------------------------
//
// The helpers below live in collider::log::detail so the test suite
// (tests/test_session_log.cpp) can exercise them in isolation. They
// were anonymous-namespace TU locals in the original file and the
// 2026-05-17 review asked for direct test coverage of (a) argv
// redaction across both --flag value and --flag=value forms, (b) the
// JSON shape produced by serialize_state, and (c) the atomic write
// helper's tmp+rename + unwind contract. Exposing them through a
// detail namespace (rather than the public API surface) keeps the
// production call sites looking the same while giving the test code
// a non-friend handle.
namespace detail {

// Redact credentials in argv for the startup banner. Recognized flags
// (--pool-password, --pool-password-file, --pool-api-key, --activate)
// have their value replaced with "REDACTED" in BOTH the --flag value
// and --flag=value forms. All other tokens pass through unchanged.
// Returns a single space-delimited string identical to what the
// banner emits.
std::string redact_argv(int argc, char** argv);

// Serialize a SessionState snapshot to the same JSON byte sequence
// the on-disk session_state.json would receive. Used by the test
// suite to pin the schema and by the in-process writer below.
std::string serialize_state(const SessionState& s);

// Atomic, durable write of `content` to `path`. Mirrors the canonical
// fsync+unwind contract from src/gpu/rckangaroo_wrapper.cu:1495-1639
// (save_herd_state): owner-only DACL on the tmp via
// secure_open_ofstream, FILE*-based write + fflush + fsync/_commit,
// fclose, atomic rename. Any failure removes the partial tmp and
// returns false; the pre-existing target (if any) is unchanged.
bool atomic_write_state_file(const std::filesystem::path& path,
                             const std::string& content);

}  // namespace detail

}  // namespace log
}  // namespace collider
