/**
 * CUDA RCKangaroo backend adapter -- implementation.
 */

#include "cuda_rckangaroo_backend.hpp"
#include "../core/byte_codec.hpp"
#include "../core/secure_write.hpp"  // secure_open_ofstream for bloom_hits.txt

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
    if (result.found && cb.on_solution) {
        // RCKangaroo returns the private key as std::array<uint64_t, 4>
        // in LE-by-limb order; the pool wire wants 32 bytes BE. Reuse
        // the shared codec rather than open-coding the conversion.
        uint8_t key[32];
        ::collider::limbs_le_to_be32(result.private_key.data(), key);
        cb.on_solution(key);
    }
}

std::string CudaRCKangarooBackend::device_summary() const {
    if (num_gpus_ == 0) return "no CUDA devices";
    std::ostringstream oss;
    oss << num_gpus_ << " CUDA GPU" << (num_gpus_ == 1 ? "" : "s");
    return oss.str();
}

}  // namespace kangaroo
}  // namespace collider
