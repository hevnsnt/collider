/**
 * Single canonical definition of the kangaroo jump-table size.
 *
 * The kangaroo walk picks one of N pre-computed jumps per step by
 * masking the low log2(N) bits of the X coordinate; N must be a power
 * of two. Three places in the codebase need to agree on the value:
 *
 *   1. C++ host code dispatching CUDA kernels (kangaroo_kernel.cu uses
 *      `NUM_JUMPS`, defined locally; jump-table buffers and shared-mem
 *      sizing depend on it).
 *   2. C++ host code dispatching Metal kernels (kangaroo_metal.hpp
 *      uses `kJumpTableSize`).
 *   3. The Metal kernel itself (kangaroo.metal hard-codes
 *      KANGAROO_JUMP_TABLE_SIZE via #define).
 *
 * Pre-1.4 the value lived in three separate places with a comment
 * begging future readers to keep them in sync. This header makes (1)
 * and (2) a single C++ source: kangaroo_kernel.cu and
 * kangaroo_metal.hpp now both pull NUM_JUMPS from here. (3) is in
 * Metal Shading Language and can't include host headers, so it gets a
 * static_assert at the host/kernel boundary instead -- see the
 * KangarooMetalSolver constructor.
 */

#pragma once

#include <cstdint>

namespace collider {
namespace gpu {

// Number of distinct jumps in the kangaroo walk's jump table. Must be
// a power of two; the kernel masks (px[0] & (kKangarooJumpTableSize - 1))
// to pick an entry. v1.4.0 keeps the legacy 32; raising it would shrink
// the K factor at the cost of more constant-memory pressure on every
// step's jump fetch -- not a v1.4.0 change without per-kernel re-tuning.
inline constexpr uint32_t kKangarooJumpTableSize = 32;

static_assert((kKangarooJumpTableSize & (kKangarooJumpTableSize - 1)) == 0,
              "kKangarooJumpTableSize must be a power of two");

}  // namespace gpu
}  // namespace collider
