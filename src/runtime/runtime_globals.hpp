/**
 * runtime_globals.hpp - Shared runtime state across the mode runners.
 *
 * Extracted out of src/main.cpp during the v1.4.1 A.3 refactor. Hosts the
 * extern declarations for the cross-runtime atomics (signal flag) and the
 * shutdown print/log emitter so the puzzle / pool / brain wallet drivers
 * can share them without each TU getting its own copy.
 *
 * The definitions still live in src/main.cpp.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

/**
 * Global shutdown flag. Set ONLY by the SIGINT/SIGTERM handler installed
 * in main(). All workers poll this lock-free atomic to drain cleanly.
 */
extern std::atomic<bool> g_shutdown;

/**
 * Captured signal number for delayed reporting from main thread. Set
 * inside the signal handler (atomic store is async-signal-safe). Read once
 * after the main loop exits so the LOG_INFO is emitted from non-signal
 * context.
 */
extern std::atomic<int> g_shutdown_signal;

/**
 * Idempotent latch so emit_shutdown_message_from_main only prints once.
 */
extern std::atomic<bool> g_shutdown_logged;

/**
 * Emit the deferred SIGINT/SIGTERM print + log from the main thread (or
 * any thread that is NOT inside a signal handler). Idempotent: no-op
 * after the first call, no-op when no signal arrived.
 */
void emit_shutdown_message_from_main();

/**
 * GPU detection result. Returned from detect_gpus() in main.cpp; consumed
 * by every mode runner.
 */
struct GPUDetectionResult {
    int device_count;
    std::string gpu_names;
    uint64_t estimated_speed;  // keys/second
    std::string backend;
};
