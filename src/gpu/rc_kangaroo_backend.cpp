/**
 * RCKangarooBackend adapter -- implementation.
 *
 * See rc_kangaroo_backend.hpp for the relationship to the pool-side
 * CudaRCKangarooBackend. The two adapters share the same underlying
 * RCKangarooManager singleton (rckangaroo_wrapper.cu) but have
 * different initialize() shapes because their callers have different
 * input types.
 */

#include "rc_kangaroo_backend.hpp"
#include "../core/byte_codec.hpp"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <ostream>
#include <sstream>

namespace collider {
namespace kangaroo {

RCKangarooBackend::RCKangarooBackend(std::vector<int> gpu_ids)
    : gpu_ids_(std::move(gpu_ids))
{
}

bool RCKangarooBackend::initialize(const collider::pool::WorkAssignment& work) {
    // Synthesize a standalone-shaped init from a pool WorkAssignment so
    // a future caller that wants to use this adapter in pool mode can.
    // Standalone code calls initialize_standalone() directly for
    // legibility; this overload exists only to satisfy the
    // IKangarooBackend contract.
    char pubhex[67];
    ::collider::hex_encode_lower(work.public_key, 33, pubhex);
    char start_hex[67];
    ::collider::hex_encode_lower(work.range_start, 32, start_hex);

    const int rb = ::collider::range_bits_from_be(work.range_start, work.range_end);
    if (rb <= 0) {
        error_ = "pool work assignment has empty or inverted range";
        return false;
    }
    return initialize_standalone(std::string(pubhex),
                                 std::string(start_hex),
                                 rb,
                                 static_cast<int>(work.dp_bits));
}

bool RCKangarooBackend::initialize_standalone(const std::string& pubkey_hex_compressed,
                                              const std::string& range_start_hex,
                                              int range_bits,
                                              int dp_bits) {
    error_.clear();

    if (range_bits < gpu::RCKangarooManager::kMinRangeBits ||
        range_bits > gpu::RCKangarooManager::kMaxRangeBits) {
        std::ostringstream oss;
        oss << "range_bits=" << range_bits
            << " is outside RCKangaroo's supported window ["
            << gpu::RCKangarooManager::kMinRangeBits << ".."
            << gpu::RCKangarooManager::kMaxRangeBits << "]";
        error_ = oss.str();
        return false;
    }
    if (dp_bits < gpu::RCKangarooManager::kMinDpBits ||
        dp_bits > gpu::RCKangarooManager::kMaxDpBits) {
        std::ostringstream oss;
        oss << "dp_bits=" << dp_bits
            << " is outside RCKangaroo's supported window ["
            << gpu::RCKangarooManager::kMinDpBits << ".."
            << gpu::RCKangarooManager::kMaxDpBits << "]";
        error_ = oss.str();
        return false;
    }

    rc_.range_bits = range_bits;
    rc_.dp_bits    = static_cast<uint32_t>(dp_bits);

    num_gpus_ = rc_.init(gpu_ids_);
    if (num_gpus_ == 0) {
        error_ = "no GPUs available for standalone puzzle solving";
        return false;
    }

    if (!rc_.set_target_pubkey(pubkey_hex_compressed)) {
        error_ = std::string("failed to set target public key: ")
                 + pubkey_hex_compressed;
        return false;
    }

    rc_.set_start_offset(range_start_hex);

    initialized_ = true;
    return true;
}

bool RCKangarooBackend::try_set_bloom_filter(const std::string& path) {
    return rc_.load_bloom_filter(path);
}

void RCKangarooBackend::solve(BackendCallbacks cb) {
    if (!initialized_) {
        error_ = "RCKangarooBackend::solve called before initialize";
        return;
    }

    const uint32_t dp_bits_u32 = rc_.dp_bits;

    rc_.dp_callback = [cb, dp_bits_u32](const uint8_t* x_be,
                                         const uint8_t* d_be,
                                         uint8_t type) {
        if (cb.on_dp) {
            cb.on_dp(x_be, d_be, type, dp_bits_u32);
        }
    };

    rc_.progress_callback = [cb](uint64_t /*ops*/, uint64_t dp_count,
                                   int speed_mkeys) -> bool {
        const double ops_per_sec = static_cast<double>(speed_mkeys) * 1e6;
        if (cb.on_progress) {
            return cb.on_progress(ops_per_sec, dp_count);
        }
        return cb.should_continue ? cb.should_continue() : true;
    };

    gpu::RCKangarooResult result = rc_.solve();
    if (result.found && cb.on_solution) {
        uint8_t key[32];
        ::collider::limbs_le_to_be32(result.private_key.data(), key);
        cb.on_solution(key);
    }

    // Post-solve: if save was armed before solve(), flush the herd
    // buffers (populated by the patched RCGpuKang::Execute) to disk.
    if (!pending_save_path_.empty()) {
        if (rc_.save_herd_state(pending_save_path_)) {
            std::cerr << "[*] Saved kangaroo herd to "
                      << pending_save_path_ << "\n";
        } else {
            std::cerr << "[!] Kangaroo herd save to "
                      << pending_save_path_ << " failed.\n";
        }
        pending_save_path_.clear();
    }
}

std::string RCKangarooBackend::device_summary() const {
    if (num_gpus_ == 0) return "no CUDA devices";
    std::ostringstream oss;
    oss << num_gpus_ << " CUDA GPU" << (num_gpus_ == 1 ? "" : "s");
    return oss.str();
}

bool RCKangarooBackend::save_herd_state(const std::string& path) {
    // Two-phase: first call (pre-solve) arms the SaveKangsHost hooks;
    // the actual file write happens after solve() returns (see solve()
    // above). If solve has already finished and the save buffers are
    // populated, write immediately.
    if (!initialized_) return false;

    if (pending_save_path_.empty()) {
        // Pre-solve arming phase.
        if (!rc_.request_save_on_stop()) return false;
        pending_save_path_ = path;
        return true;
    }
    // Already armed; fall through to writing now (e.g. caller invoked
    // save_herd_state twice, or post-solve write was triggered out of
    // band). RCKangarooManager::save_herd_state handles the "no solve
    // ran" case by returning false.
    return rc_.save_herd_state(path);
}

bool RCKangarooBackend::load_herd_state(const std::string& path) {
    if (!initialized_) return false;
    return rc_.load_herd_state(path);
}

}  // namespace kangaroo
}  // namespace collider
