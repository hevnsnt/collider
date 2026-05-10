/**
 * CUDA RCKangaroo backend adapter -- implementation.
 */

#include "cuda_rckangaroo_backend.hpp"
#include "../core/byte_codec.hpp"

#include <cstdio>
#include <cstring>
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
    // Derive range_bits from the work assignment's [range_start, range_end)
    // span. A pool-issued chunk for puzzle 135 will give 135 here, but the
    // protocol does not pin us to that puzzle -- a chunk for puzzle 75 or
    // a custom range will yield the correct value. Pre-1.4 builds hard-
    // coded 135, which mis-budgeted DP rate and walk size on every other
    // puzzle. range_bits_from_be returns 0 on inverted/empty ranges.
    const int rb = ::collider::range_bits_from_be(work.range_start, work.range_end);
    if (rb <= 0) {
        error_ = "pool work assignment has empty or inverted range";
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
    initialized_ = true;
    return true;
}

bool CudaRCKangarooBackend::try_set_bloom_filter(const std::string& path) {
    return rc_.load_bloom_filter(path);
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
