// bsgs_solver_gpu.hpp -- Phase F2: GPU Baby-Step Giant-Step solver.
//
// Pollard's Kangaroo is asymptotically equivalent to BSGS (both
// O(sqrt(n)) for the discrete-log problem on secp256k1) but BSGS is
// deterministic and parallelizes trivially: the baby table is a
// straight scalar multiplication batch, and each giant step is an
// independent lookup. Where Kangaroo wins (huge ranges where the baby
// table will not fit in VRAM) BSGS loses; where BSGS wins (small-ish
// ranges where the table fits and the determinism beats Kangaroo's
// variance) we hand the operator a single binary that does both.
//
// Scope of THIS implementation (v1.5.x first cut):
//   * Single-GPU. Multi-GPU work-splitting is a follow-up (each GPU
//     would get its own slice of the giant-step space).
//   * Range cap: bits <= kMaxBits (defined below). The baby table is
//     m = 2^(bits/2) entries; at 64 bytes each that pins
//     2^(bits/2 + 6) bytes of VRAM. The current cap targets puzzles
//     that fit in ~1 GB of VRAM headroom on a 24 GB card. Larger
//     ranges return BsgsResult::OutOfRange so the caller routes to
//     Kangaroo.
//   * Host-side sort + std::lower_bound for the baby-table lookup.
//     A GPU radix-sort + binary-search would be the natural
//     follow-up; this keeps the integration surface small.
//
// Algorithm (standard BSGS, b = range_start, N = range_end - b):
//   m = ceil(sqrt(N))
//   1. Baby table: { (j*G, j) : j in [0, m) }, sorted by X.
//   2. For each i in [0, ceil(N/m)):
//        P_i = H - (b + i*m)*G
//        if P_i.X is in the baby table at index j:
//          k = b + i*m + j  (verify k*G == H to rule out X-collision)
//   3. Return Found(k) on first verified hit.
//
// The "verify k*G == H" step exists because two distinct EC points
// can share an X coordinate (the point and its Y-negation). Without
// the verify a BSGS hit can produce a private key that differs from
// the true one by negation of the corresponding Y; the verify
// catches that and the loop continues searching.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace collider::gpu::bsgs {

// Output of a single bsgs_solve invocation. Found carries the
// recovered scalar k in 32-byte big-endian; the caller hex-encodes
// for display. OutOfRange means the input range exceeds the cap;
// the caller should fall back to a different algorithm.
enum class BsgsResultKind {
    Found,
    NotInRange,   // searched every giant step; H not in [b, b+N)
    OutOfRange,   // bits > kMaxBits; BSGS not attempted
    GpuError,     // CUDA error (description in error_message)
    Cancelled,    // should_continue() returned false
};

struct BsgsResult {
    BsgsResultKind kind = BsgsResultKind::NotInRange;
    uint8_t recovered_key_be[32] = {0};
    uint64_t baby_table_size = 0;
    uint64_t giant_steps_completed = 0;
    std::string error_message;
};

// Largest puzzle bit-width the host driver will attempt. Bits above
// this cap make the baby table too large to materialize. Update in
// lockstep with VRAM-headroom assumptions; see header rationale.
constexpr int kMaxBits = 48;

// Solve callback: invoked ~1 Hz with the count of giant steps the
// driver has completed so far. Return false to cancel cleanly.
using BsgsProgressFn = bool (*)(uint64_t giant_steps_done, void* user);

struct BsgsConfig {
    // 32 bytes BE; the public-key X+Y the operator is solving for.
    uint8_t target_pubkey_x_be[32] = {0};
    uint8_t target_pubkey_y_be[32] = {0};

    // 32 bytes BE for both. range_end is exclusive. range_end >
    // range_start; difference must fit in 2^kMaxBits.
    uint8_t range_start_be[32] = {0};
    uint8_t range_end_be[32]   = {0};

    // The bit-width passed in by the puzzle target. Used to dispatch
    // OutOfRange when bits > kMaxBits without touching the GPU.
    int bits = 0;

    // Optional progress callback. May be null.
    BsgsProgressFn progress = nullptr;
    void* progress_user = nullptr;

    // CUDA device the solve runs on. -1 = current device.
    int device_id = -1;
};

// Host driver entry. Runs the full baby/giant pipeline synchronously
// on the requested GPU. Returns when a key is found, the range is
// exhausted, the operator cancels via the progress callback, or a
// CUDA error fires. The returned struct's kind discriminates.
BsgsResult bsgs_solve(const BsgsConfig& cfg);

}  // namespace collider::gpu::bsgs
