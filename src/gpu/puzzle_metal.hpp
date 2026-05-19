/**
 * Apple Metal fused brute-force puzzle dispatcher (v1.4.1).
 *
 * Sits between the puzzle solver runtime (src/runtime/puzzle_solver.cpp via
 * MultiGPUPuzzleSolver) and the .metal kernel at src/gpu/puzzle.metal.
 * Mirrors the CUDA puzzle_search_optimized + puzzle_search_batch_optimized
 * dispatch (src/gpu/puzzle_optimized.cu and src/gpu/puzzle_gpu.cu) but in
 * C++/Obj-C++ for Mac builds.
 *
 * Compiled only on APPLE && COLLIDER_USE_METAL.
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace collider {
namespace gpu {

// Threadgroup width used when dispatching the puzzle_search kernel. Apple
// silicon SIMD-group width is 32 across M1/M2/M3, so 32 is the natural lower
// bound and gives the scheduler the most flexibility. The kernel itself has
// no threadgroup-cooperative state -- threads are independent -- so this is
// purely a perf hint, not a correctness constraint.
inline constexpr uint32_t kPuzzleMetalThreadgroupWidth = 32;

// Default batch size (keys per kernel launch). Sized to keep individual
// dispatches under ~5 seconds of wall time on M1, which lets the host
// react to Ctrl+C and update the on-screen rate without waiting an
// eternity. The host can override via PuzzleMetalSolver::set_batch_size.
// 4M matches the CUDA default in puzzle_gpu.hpp::Config.batch_size_per_gpu;
// on a healthy M-series setup we expect ~50 MKeys/s sustained, so a 4M
// batch is ~80 ms per dispatch.
inline constexpr uint64_t kPuzzleMetalDefaultBatchSize = 4'000'000ull;

// G-table layout: 32 windows of 256 entries, each entry is (X, Y) in the
// 4-limb LE-by-limb form the kernel expects. 32 * 256 * 8 ulong * 8 bytes
// = 524288 bytes = 512 KiB. Fits comfortably in shared system memory and
// the L2 cache on Apple silicon. The d=0 row of every window is the
// identity, encoded as (0, 0); the kernel skips zero windows.
inline constexpr size_t kPuzzleMetalGTableWindows  = 32;
inline constexpr size_t kPuzzleMetalGTableEntries  = 256;
inline constexpr size_t kPuzzleMetalGTableUlongs   =
    kPuzzleMetalGTableWindows * kPuzzleMetalGTableEntries * 8u;
inline constexpr size_t kPuzzleMetalGTableBytes    =
    kPuzzleMetalGTableUlongs * sizeof(uint64_t);

class PuzzleMetalSolver {
public:
    PuzzleMetalSolver();
    ~PuzzleMetalSolver();

    // Disable copy. Move is intentionally not defined; the runtime uses
    // pointer-to-impl wrappers (MultiGPUPuzzleSolver) for ownership and
    // never moves a PuzzleMetalSolver directly.
    PuzzleMetalSolver(const PuzzleMetalSolver&) = delete;
    PuzzleMetalSolver& operator=(const PuzzleMetalSolver&) = delete;

    // One-time setup. Creates the Metal device, loads the embedded MSL
    // source, builds the puzzle_search compute pipeline, allocates the
    // device-side buffers (target hash160, match-result trio, G-table),
    // and uploads the precomputed G-table. Returns false + populates
    // error() on any failure.
    bool init();

    // Set the 20-byte hash160 target the kernel compares against.
    bool set_target(const std::array<uint8_t, 20>& hash160);

    // Override the per-launch batch size. Must be > 0; rounded up to a
    // multiple of kPuzzleMetalThreadgroupWidth so the dispatch grid is
    // a clean fit. Returns the rounded-up value actually applied.
    uint64_t set_batch_size(uint64_t batch_size);
    uint64_t batch_size() const;

    // Run one batch. Returns true on a match (and populates found_lo /
    // found_hi with the recovered private key), false if the entire
    // [start_lo:start_lo+batch_size) range was checked with no match.
    // start_lo / start_hi: low / high 64 bits of the 256-bit base scalar.
    //   The kernel forms k = (start_lo, start_hi, 0, 0) + gid for each
    //   thread, so callers feed sequential 64-bit chunks of the puzzle
    //   range and rely on this stride.
    // The dispatcher waits synchronously for the GPU before reading
    // match flags, so the function returns after the entire batch has
    // been hashed.
    bool search_batch(uint64_t start_lo, uint64_t start_hi,
                      uint64_t batch_size,
                      uint64_t& found_lo, uint64_t& found_hi);

    std::string device_name() const;
    const std::string& error() const { return error_; }

    // KAT helper: hash one (privkey, target) pair through the kernel and
    // return the recovered private key on match. Used by the host-side
    // KAT runner to verify SHA-256 + RIPEMD-160 + EC mul agree with the
    // CPU reference. The privkey passed in is the *only* candidate the
    // kernel checks; total_keys is set to 1 so the compare is the last
    // thing that runs. Returns true on hash match (i.e., target_h160 is
    // indeed hash160(compress(privkey * G))).
    bool verify_one(uint64_t priv_lo, uint64_t priv_hi,
                    const std::array<uint8_t, 20>& target_h160);

private:
    struct Impl;
    Impl* impl_ = nullptr;
    std::string error_;
};

}  // namespace gpu
}  // namespace collider
