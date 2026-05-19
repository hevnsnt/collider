/**
 * gpu_detection.hpp - Pre-dispatch GPU hardware detection.
 *
 * Extracted out of src/main.cpp during the v1.4.1 A.3 (6/6) refactor.
 * detect_gpus() is called before any mode runner so the interactive flow,
 * pool-mode driver, puzzle-mode driver, and brain-wallet runner all see
 * the same view of the available accelerator hardware.
 *
 * The result type (GPUDetectionResult) lives in runtime/runtime_globals.hpp
 * so that runtime modules (pool_solver, puzzle_solver, brain_wallet_runner)
 * can take it by const ref without including this file.
 */
#pragma once

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "runtime/runtime_globals.hpp"  // GPUDetectionResult

/**
 * Detect GPU hardware and return formatted info for banners + dispatch.
 *
 * @param requested_ids  In-out: caller-supplied GPU IDs (e.g. from --gpus).
 *                       If empty, detect_gpus auto-fills with every visible
 *                       device's index. The mutated vector is what later
 *                       runtime drivers feed into their backend init.
 *
 * @return GPUDetectionResult populated with device count, formatted GPU
 *         names, estimated EC scalar-multiply throughput (keys/sec), and
 *         backend tag ("CUDA" / "Metal" / "CPU").
 */
GPUDetectionResult detect_gpus(std::vector<int>& requested_ids);

/**
 * Parse the compile-time CUDA arch list ("7.5,8.6,8.9,12.0") into a set of
 * (major, minor) pairs. Exposed so the session log's hardware enumeration
 * can fire a milestone("sm_mismatch", ...) using the SAME parser that
 * drives the startup stderr warning in detect_gpus(); duplicating the
 * parser would risk the two emitting different verdicts after a future
 * COLLIDER_CUDA_ARCH_LIST format change.
 *
 * Returns an empty set when COLLIDER_CUDA_ARCH_LIST is not defined (Metal /
 * CPU builds). Callers MUST treat an empty set as "no SM check applies"
 * and skip the membership test rather than treating every device as a
 * mismatch.
 */
std::set<std::pair<int, int>> compile_time_sm_set();

/**
 * Render a (major, minor) SM set as "{7.5, 8.6, 8.9, 12.0}" for logging.
 * Shared with detect_gpus() so the startup banner and the session-log
 * milestone format the same way.
 */
std::string sm_set_to_string(const std::set<std::pair<int, int>>& s);
