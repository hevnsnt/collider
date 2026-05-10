/**
 * Per-TU __device__ helper for checking a derived priv against the puzzle
 * target list (Phase 4/5/6/7/8 share this).
 *
 * Lives in a header so each .cu TU compiles its own static __device__ copy.
 * Avoids cross-TU __device__ references that bloat nvlink's symbol graph
 * (Windows nvlink has a 1MB default stack and overflows on big graphs).
 *
 * `c_puzzle_targets` and `c_puzzle_target_count` are still cross-TU
 * __constant__ symbols defined in brain_wallet_v2.cu. Each TU including
 * this header MUST also `extern __constant__` declare them.
 */

#pragma once

#include "brain_wallet_v2.hpp"
#include <cuda_runtime.h>

namespace collider {
namespace gpu {
namespace v2 {

extern __constant__ PuzzleTarget c_puzzle_targets[PUZZLE_TARGET_MAX];
extern __constant__ int          c_puzzle_target_count;

// Static so each TU gets its own inlined copy (no cross-TU edge for nvlink).
//
// `kind` selects the V2MatchRecord::Kind tag emitted on hit. Pass
// WEAK_PRNG_HIT from the weak-PRNG kernel; pass PUZZLE_KEY_HIT from the
// encoding-anomaly / electrum / legacy-KDF / multi-scheme kernels.
static __device__ __forceinline__ void v2_check_priv_against_puzzles(
    uint32_t pp_idx,
    uint64_t weak_seed,
    uint8_t  scheme_id,
    const uint8_t priv_be[32],
    V2MatchRecord* matches,
    uint32_t* match_count,
    V2MatchRecord::Kind kind = V2MatchRecord::Kind::PUZZLE_KEY_HIT)
{
    // Pack priv into 4 LE-by-limb uint64.
    uint64_t limbs[4];
    {
        // Inline byte-by-byte to avoid alignment/lambda issues across TUs.
        const uint8_t* p0 = priv_be + 24;
        const uint8_t* p1 = priv_be + 16;
        const uint8_t* p2 = priv_be +  8;
        const uint8_t* p3 = priv_be +  0;
        uint64_t v0 = 0, v1 = 0, v2_ = 0, v3 = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) v0 = (v0 << 8) | (uint64_t)p0[i];
        #pragma unroll
        for (int i = 0; i < 8; ++i) v1 = (v1 << 8) | (uint64_t)p1[i];
        #pragma unroll
        for (int i = 0; i < 8; ++i) v2_ = (v2_ << 8) | (uint64_t)p2[i];
        #pragma unroll
        for (int i = 0; i < 8; ++i) v3 = (v3 << 8) | (uint64_t)p3[i];
        limbs[0] = v0;
        limbs[1] = v1;
        limbs[2] = v2_;
        limbs[3] = v3;
    }

    int n = c_puzzle_target_count;
    for (int ti = 0; ti < n; ++ti) {
        const PuzzleTarget& t = c_puzzle_targets[ti];
        bool match = true;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if ((limbs[j] & t.low_mask[j]) != t.low_value[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            uint32_t slot = atomicAdd(match_count, 1u);
            if (slot < V2_MAX_MATCHES_PER_BATCH) {
                V2MatchRecord rec{};
                rec.pp_idx    = pp_idx;
                rec.weak_seed = weak_seed;
                rec.puzzle_n  = t.puzzle_n;
                rec.scheme_id = scheme_id;
                rec.kind      = (uint8_t)kind;
                matches[slot] = rec;
            }
        }
    }
}

}  // namespace v2
}  // namespace gpu
}  // namespace collider
