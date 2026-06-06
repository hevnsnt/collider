// SPDX-License-Identifier: GPL-3.0-or-later
// Shared CUDA host-side helpers.
//
// T1.6 (2026-05-17). Centralizes the pre-launch context-state probe
// pattern that was previously inlined at exactly two sites
// (gpu_rules.cpp::apply_rules_to_words_gpu and
// brain_wallet_gpu.cpp::fused_brain_wallet_batch_fixed_stride). The
// remaining kernel launches in this codebase (kangaroo_kernel.cu,
// puzzle_optimized.cu, v2/*.cu, h160_bloom_filter.cu) had no
// pre-launch probe at all, so a sticky context error from an earlier
// kernel would surface as a confusing post-launch failure on an
// unrelated kernel with no way to tell which dispatch was the
// original culprit.
//
// Usage:
//
//   cudaError_t pre = collider::gpu::pre_launch_context_probe(
//       "kangaroo_step_kernel_sota");
//   if (pre != cudaSuccess) {
//       // Sticky error: device is poisoned, this launch is doomed.
//       // The helper has already drained the last-error slot and
//       // logged the offending device id + cudaGetErrorString to
//       // stderr. Callers decide whether to abort, mark the GPU as
//       // Faulted (via runtime_control), or attempt a recovery.
//       return pre;
//   }
//   // ... launch the kernel ...
//
// The helper is intentionally header-only so .cu / .cpp / .mm
// translation units can pull it in without changing the link list.
// It uses cudaPeekAtLastError to sample without clearing, then
// drains via cudaGetLastError so the post-launch return value of
// cudaGetLastError reflects ONLY the new launch's status (not the
// pre-existing sticky error). Note that sticky context errors
// (cudaErrorIllegalAddress, cudaErrorMisalignedAddress,
// cudaErrorLaunchOutOfResources) survive cudaGetLastError; the
// drain only resets the last-error slot, not the context itself,
// so the device remains unusable until cudaDeviceReset. The probe
// exists to attribute the failure to the correct dispatch, not to
// recover from it (recovery is T1.1 / W2's GpuPhase::Faulted work).

#pragma once

#include <cuda_runtime.h>

#include <climits>
#include <cstdio>

namespace collider {
namespace gpu {

// Safe blocks-from-items computation shared across .cu translation units.
//
// History: several kernel launch sites computed
// `const int blocks = (count + threads - 1) / threads;`. Once count
// exceeds INT_MAX this signed-int expression overflows (UB on MSVC) and
// produces a garbage or negative grid dimension; CUDA then either rejects
// the launch with a confusing error or, worse, launches an under-sized
// grid that silently skips the tail of the input. fused_pipeline.cu grew
// a TU-local guard (fused_compute_grid_blocks) for its own launches; the
// h160 bloom kernels (h160_bloom_filter.cu) never got it. This is the
// shared, header-only version so every TU computes the grid identically:
// 64-bit unsigned math plus an explicit INT_MAX refusal at the grid-X
// bound.
inline cudaError_t compute_grid_blocks(unsigned long long total_items,
                                       int threads,
                                       int* blocks_out) {
    if (threads <= 0 || blocks_out == nullptr) {
        std::fprintf(stderr,
            "[!] compute_grid_blocks: invalid threads=%d or null out\n",
            threads);
        return cudaErrorInvalidValue;
    }
    unsigned long long blocks =
        (total_items + (unsigned long long)threads - 1ull) /
        (unsigned long long)threads;
    if (blocks == 0ull) blocks = 1ull;
    if (blocks > (unsigned long long)INT_MAX) {
        std::fprintf(stderr,
            "[!] compute_grid_blocks: launch refused, blocks=%llu exceeds "
            "INT_MAX (grid-X bound). total_items=%llu threads=%d. "
            "Split host-side.\n",
            blocks, total_items, threads);
        return cudaErrorInvalidConfiguration;
    }
    *blocks_out = (int)blocks;
    return cudaSuccess;
}

// Pre-launch context-state probe. Returns cudaSuccess if the device's
// last-error slot is clean (safe to launch); otherwise returns the
// sticky error, logs a diagnostic naming the device + the kernel the
// caller was about to launch, and drains the last-error slot so the
// caller's post-launch cudaGetLastError reflects only the new launch.
//
// `kernel_name` is a compile-time string literal in production use;
// it appears in the diagnostic so the operator can trace the
// cascade back to the actual culprit kernel.
inline cudaError_t pre_launch_context_probe(const char* kernel_name) {
    cudaError_t pre = cudaPeekAtLastError();
    if (pre == cudaSuccess) {
        return cudaSuccess;
    }
    int dev = -1;
    // Best-effort device id lookup. If cudaGetDevice itself fails
    // (extreme degenerate case) we still log with dev=-1 so the
    // operator sees the kernel name and the original error string.
    (void)cudaGetDevice(&dev);
    std::fprintf(stderr,
        "[!] %s: context already in error state on device %d: %s. "
        "Sticky error from a prior kernel on this device; this "
        "launch will fail with the same error code. Original "
        "culprit was the kernel that first produced the sticky "
        "fault.\n",
        kernel_name ? kernel_name : "<kernel>",
        dev,
        cudaGetErrorString(pre));
    // Drain so the post-launch cudaGetLastError reflects THIS launch's
    // status, not the prior fault. The drain does not reset the
    // context itself; the device remains poisoned until
    // cudaDeviceReset (or T1.1's recovery path).
    (void)cudaGetLastError();
    return pre;
}

}  // namespace gpu
}  // namespace collider
