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
