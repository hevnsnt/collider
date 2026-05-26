/**
 * CUDA RCKangaroo backend adapter -- implementation.
 */

#include "cuda_rckangaroo_backend.hpp"
#include "../core/byte_codec.hpp"
#include "../core/secure_write.hpp"  // secure_open_ofstream for bloom_hits.txt

// defs.h from the forked RCKangaroo carries KANG_MODE_BOTH / TAME_ONLY /
// WILD_ONLY. The wire-side u8 values in WorkAssignment.kangaroo_type are
// byte-stable with these constants (0/1/2). v1.5 pool mode MUST translate
// to TAME_ONLY or WILD_ONLY; BOTH is rejected as a server bug.
#include "../../third_party/RCKangaroo/defs.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <ostream>
#include <sstream>

namespace collider {
namespace kangaroo {

CudaRCKangarooBackend::CudaRCKangarooBackend(std::vector<int> gpu_ids)
    : gpu_ids_(std::move(gpu_ids))
{
}

bool CudaRCKangarooBackend::initialize(const collider::pool::WorkAssignment& work) {
    error_.clear();

    // v1.5: enforce the asymmetric tame/wild assignment from the pool
    // server. Without this, RCKangaroo would default to KANG_MODE_BOTH
    // and the worker would self-solve into the v1.4.x theft-vulnerable
    // code path (host-side tame x wild collision detection computes the
    // recovered private key into worker memory). The mapping below is
    // the SOLE bridge between the wire u8 and the kangaroo runtime mode.
    //
    // BOTH (kangaroo_type == 0) is REJECTED in pool mode: the v1.5 pool
    // server is contractually required to assign TAME_ONLY or WILD_ONLY
    // round-robin. A BOTH assignment hitting this path means either a
    // server bug, a downgrade attack from a malicious server, or a
    // protocol-version slip. Fail fast rather than silently configuring
    // a theft-vulnerable solver.
    switch (work.kangaroo_type) {
        case 1:  // TAME_ONLY
            rc_.mode = KANG_MODE_TAME_ONLY;
            break;
        case 2:  // WILD_ONLY
            rc_.mode = KANG_MODE_WILD_ONLY;
            break;
        case 0:  // BOTH (legacy / illegal in v1.5 pool mode)
        default: {
            std::ostringstream oss;
            oss << "pool work assignment carries illegal kangaroo_type="
                << static_cast<int>(work.kangaroo_type)
                << " (v1.5 pool servers must assign 1=TAME_ONLY or "
                   "2=WILD_ONLY; kangaroo_type=0/BOTH would re-enable the "
                   "v1.4.x theft-vulnerable local solve path). Refusing "
                   "to initialize. Check that the pool server is running "
                   "v1.5+ asymmetric protocol.";
            error_ = oss.str();
            return false;
        }
    }

    rc_.dp_bits    = work.dp_bits;
    // Pool range_bits = bit_length(range_start), NOT bit_length(chunk_size).
    //
    // The pool server fixes PntA = Q - range_start*G using the PUZZLE's
    // range_start (e.g., 2^134 for puzzle 135) -- not the chunk's start.
    // RCKangaroo computes PntA = PntToSolve - 2^(range_bits-1)*G, so we need
    // 2^(range_bits-1) = puzzle_range_start, i.e. range_bits = puzzle_bits.
    //
    // For any chunk of puzzle 135: chunk_start >= 2^134, so
    //   bit_length_be32(chunk_start) = 135  =>  HalfRange = 2^134  =>  PntA correct.
    //
    // Using range_bits_from_be(start, end) would give the CHUNK WIDTH instead:
    //   bit_length(2^40) = 41  =>  HalfRange = 2^40  =>  PntA 94 bits off.
    const int rb = ::collider::bit_length_be32(work.range_start);
    if (rb == 0) {
        error_ = "pool work assignment has zero range_start";
        return false;
    }
    if (rb < 32 || rb > 170) {
        // RCKangaroo accepts 32..170. Anything outside is either a buggy
        // pool, a probing/fuzz request, or a protocol mismatch.
        std::ostringstream oss;
        oss << "pool work range_bits=" << rb
            << " is outside RCKangaroo's 32..170 supported window";
        error_ = oss.str();
        return false;
    }
    rc_.range_bits = rb;

    num_gpus_ = rc_.init(gpu_ids_);
    if (num_gpus_ == 0) {
        error_ = "no GPUs available for pool solving";
        return false;
    }

    // Encode the 33-byte compressed pubkey for RCKangaroo's hex-string API.
    char hex[67];
    ::collider::hex_encode_lower(work.public_key, 33, hex);
    if (!rc_.set_target_pubkey(std::string(hex))) {
        error_ = std::string("failed to set target public key: ") + hex;
        return false;
    }

    // Do NOT call set_start_offset in pool mode. The server uses a FIXED
    // PntA = Q - puzzle_range_start*G (e.g. Q - 2^134*G for puzzle 135).
    // With range_bits=135 and no offset, PntToSolve=Q and
    // PntA = Q - 2^134*G -- an exact match. Calling set_start_offset
    // (chunk_start) would shift PntToSolve to Q - chunk_start*G, making
    // PntA = Q - (chunk_start + 2^134)*G, which the server rejects.

    initialized_ = true;
    return true;
}

bool CudaRCKangarooBackend::try_set_bloom_filter(const std::string& path) {
    if (!rc_.load_bloom_filter(path)) return false;
    // Match the standalone-kangaroo path's hit-log contract: every bloom
    // positive appends a line to bloom_hits.txt (owner-only DACL/0600).
    // Without this, pool-mode operators observed bloom_checks ticking on
    // the GPU counter while no file ever appeared, which made the
    // "opportunistic address checking" feature look broken even when it
    // was running. See puzzle_solver_kangaroo.cpp:maybe_load_rckangaroo_
    // bloom_filter for the original pattern; this is the pool-side mirror.
    rc_.bloom_hit_callback = [](const gpu::BloomHit& hit) {
        std::ofstream hitlog =
            ::collider::secure_open_ofstream("bloom_hits.txt", std::ios::app);
        if (hitlog) {
            char h160_hex[41];
            for (int i = 0; i < 20; i++) {
                std::snprintf(h160_hex + i*2, 3, "%02x", hit.hash160[i]);
            }
            hitlog << "H160: " << h160_hex
                   << " at ops " << hit.ops_at_hit << "\n";
        }
    };
    return true;
}

void CudaRCKangarooBackend::solve(BackendCallbacks cb) {
    if (!initialized_) {
        error_ = "CudaRCKangarooBackend::solve called before initialize";
        return;
    }

    const uint32_t dp_bits_u32 = rc_.dp_bits;  // already uint32_t in v1.4.0+

    // RCKangaroo's dp_callback signature is (x[32], d[32], type).
    rc_.dp_callback = [cb, dp_bits_u32](const uint8_t* x_be,
                                         const uint8_t* d_be,
                                         uint8_t type) {
        cb.on_dp(x_be, d_be, type, dp_bits_u32);
    };

    // RCKangaroo reports speed in MKeys/s (int); convert to ops/s.
    rc_.progress_callback = [cb](uint64_t /*ops*/, uint64_t dp_count,
                                   int speed_mkeys) -> bool {
        const double ops_per_sec = static_cast<double>(speed_mkeys) * 1e6;
        return cb.on_progress(ops_per_sec, dp_count);
    };

    gpu::RCKangarooResult result = rc_.solve();
    // v1.5: the result.found path that invoked cb.on_solution with the
    // recovered private key was DELETED. In pool mode rc_.mode is always
    // TAME_ONLY or WILD_ONLY (enforced in initialize() above), and the
    // forked RCKangaroo's contract guarantees no Mode != BOTH instance
    // can produce result.found=true (the per-instance host-side tame x
    // wild hashtable is bypassed; the solve loop exits only on the
    // external stop signal). The standalone solver runs through a
    // different backend dispatch and does not share this code path.
    // Reading result.private_key here would re-introduce the theft
    // surface v1.5 was designed to eliminate, so the read is gone too.
    (void)result;
}

std::string CudaRCKangarooBackend::device_summary() const {
    if (num_gpus_ == 0) return "no CUDA devices";
    std::ostringstream oss;
    oss << num_gpus_ << " CUDA GPU" << (num_gpus_ == 1 ? "" : "s");
    return oss.str();
}

}  // namespace kangaroo
}  // namespace collider
