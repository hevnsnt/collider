/**
 * pool_solver.hpp - Pool-mode runtime driver.
 *
 * Extracted from src/main.cpp during the v1.4.1 A.3 refactor. Hosts the
 * pool client mode: parse pool URL, connect, request work, dispatch to
 * the IKangarooBackend, plumb DP submissions back to the pool client.
 *
 * Returns the process exit code (0 on clean shutdown, non-zero on
 * configuration / connection / backend failures).
 */
#pragma once

#include "cli/cli_parser.hpp"        // Arguments
#include "runtime/runtime_globals.hpp"  // GPUDetectionResult

namespace collider::runtime {

/**
 * Run pool mode - connect to a Kangaroo pool for distributed solving.
 * The local PoolManager's destructor disconnects automatically, so error
 * paths can `return 1;` without manual cleanup.
 */
int run_pool_mode(const Arguments& args, const GPUDetectionResult& gpu_info);

}  // namespace collider::runtime
