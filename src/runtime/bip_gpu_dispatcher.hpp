/**
 * bip_gpu_dispatcher.hpp -- multi-GPU dispatch path for BIP-39 scan.
 *
 * Per-phrase work in the BIP scanner:
 *   1. validate BIP-39 checksum                   (CPU, ~ns)
 *   2. PBKDF2-HMAC-SHA512 2048 rounds             (GPU, batch=256
 *                                                  per cudaStream;
 *                                                  see bip39_pbkdf2.cu)
 *   3. master = HMAC-SHA512(seed, "Bitcoin seed") (CPU, ~5 us)
 *   4. for each derivation profile:
 *      a. CKDpriv chain walk for the path          (CPU, ~30 us per step)
 *      b. priv -> pub (secp256k1 EC mul)           (GPU)
 *      c. hash160 of pub                           (GPU)
 *      d. bloom probe                              (GPU)
 *
 * Steps 4b/c/d run on GPU via the existing MultiAddressSession kernel
 * (src/gpu/v2/v2_orchestrator.hpp). This dispatcher owns one such
 * session per CUDA device. CPU walker threads in bip_scanner_runner
 * derive priv keys from the seeds, push them onto a shared queue, and
 * per-device worker threads drain the queue + batch into process_batch.
 * Match records flow back through the hit callback to the runner's
 * hits writer.
 *
 * PBKDF2 GPU port (commit 1da2a2e onward): PBKDF2 is no longer the
 * bottleneck. The CPU walker has a separate set of per-device
 * cudaStreams for PBKDF2 batching; this dispatcher only handles the
 * EC + hash160 + bloom downstream phase. They are decoupled by design
 * so a PBKDF2 stream-create failure doesn't take down the dispatcher
 * (and vice versa).
 *
 * Resilient multi-GPU init: init() commits any device that came up
 * and records faults for the rest (see faulted_devices()). A 2-GPU
 * box where device 1 is busy still runs the scan on device 0. Total
 * failure (every device faulted) returns 70 with last_error() set.
 */

#pragma once

#ifdef COLLIDER_PRO

#include "tools/utxo_bloom_builder.hpp"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace collider::runtime {

// One element on the dispatcher queue. Carries the derived priv key
// plus enough context for the hits writer to log the source phrase +
// derivation path on a bloom match. Hits are rare so the context
// copies are amortized; the alternative (look up phrase context by
// hash on hit) would require a separate dictionary keyed on priv key.
struct BipGpuWorkItem {
    std::array<uint8_t, 32> priv;
    std::string             mnemonic;      // source phrase
    std::string             derivation_path; // m/44'/0'/0'/0/N
    std::string             profile_label;
    int                     addr_mask;     // which address types to check
};

// Per-device statistics. Read by the TUI thread; written by the
// dispatcher worker thread. Atomic so the read is wait-free.
struct BipGpuDeviceStats {
    int                   device_id = 0;
    std::atomic<uint64_t> addresses_dispatched{0};
    std::atomic<uint64_t> matches_emitted{0};
    std::atomic<double>   addresses_per_sec{0.0};
    // Effective batch size after any halving cascade in init(). Equal
    // to the dispatcher Config::batch_size on the first attempt; lower
    // if init() had to halve to fit this device's transient VRAM.
    size_t                effective_batch_size = 0;
};

// Hit callback signature: invoked from the dispatcher's GPU worker
// thread on each bloom match. Caller is responsible for serialization
// (the existing hits_mu pattern in the scanner runner). Args:
//   mnemonic, path, priv (32B), addr_type_label, h160 (20B)
using BipGpuHitCallback = std::function<void(
    const std::string& mnemonic,
    const std::string& path,
    const std::string& profile_label,
    const uint8_t      priv[32],
    const char*        addr_type_label,
    const uint8_t      h160[20])>;

// BipGpuDispatcher: owns N per-device worker threads + the shared
// work queue. Producer threads (the BIP scan CPU pool) call
// enqueue(); consumer threads consume + dispatch to their device's
// MultiAddressSession. Hit records flow back through the hit
// callback. shutdown() drains the queue and joins workers.
//
// Per-device init failure record. Surfaced to the dashboard so the
// operator sees "1 of 2 GPU (GPU#1: out of memory)" instead of a
// silent "0 GPU" when only some devices failed.
struct FaultedDevice {
    int         device_id = 0;
    std::string error;     // cudaGetErrorString detail or session-init rc
};

// Not copyable; one instance per process.
class BipGpuDispatcher {
public:
    struct Config {
        // GPU device ids the runner asked for. The runner is responsible
        // for populating this from detect_gpus() before calling init();
        // we no longer silently default to {0} here because that hid the
        // "user has 2 GPUs but we only used 1" bug. Empty input is an
        // error.
        std::vector<int> gpu_ids;
        // Host pointer to the bloom bit array. MUST outlive the init()
        // call (the data is uploaded to GPU during init); does NOT need
        // to outlive the dispatcher (post-init the GPU copy is the only
        // one read). Caller in bip_scanner_runner.cpp passes
        // BloomLoadResult::data which lives the whole scan, so this
        // contract is trivially satisfied today.
        const uint8_t*   bloom_data = nullptr;
        uint64_t         bloom_bits = 0;
        int              bloom_hashes = 0;
        uint32_t         bloom_seed = 0;
        size_t           batch_size = 4096;   // priv keys per GPU dispatch
        size_t           queue_max = 65536;   // back-pressure cap
        BipGpuHitCallback on_hit;
    };

    BipGpuDispatcher();
    ~BipGpuDispatcher();

    BipGpuDispatcher(const BipGpuDispatcher&) = delete;
    BipGpuDispatcher& operator=(const BipGpuDispatcher&) = delete;

    // Initialize per-device sessions + spawn workers. Resilient to
    // per-device failure: any device that fails (cudaSetDevice, bloom
    // upload, secp table generation, OOM) is dropped from the working
    // set and recorded in faulted_devices(); init() returns 0 as long
    // as AT LEAST ONE device came up. Returns 70 only when every
    // requested device failed (or 64 if cfg is malformed).
    //
    // Callers should always inspect faulted_devices() after init even
    // when it returns 0 -- a partial init is an operator-visible
    // signal that needs surfacing.
    int init(const Config& cfg);

    // Producer entry. Blocks when the queue is full (back-pressure).
    // Returns false when the dispatcher is shutting down -- caller
    // should drop the item and exit its loop. The return value is
    // load-bearing: the runner's addrs_probed counter MUST NOT be
    // incremented when this returns false (the work was dropped).
    bool enqueue(BipGpuWorkItem&& item);

    // Drain queue, signal workers to exit, join. Idempotent.
    void shutdown();

    // Read-only per-device stats for the TUI dashboard. Indexed by
    // SUCCESSFUL device order (so faulted devices don't get an
    // empty row). unique_ptr because BipGpuDeviceStats holds
    // std::atomic members which are not movable.
    const std::vector<std::unique_ptr<BipGpuDeviceStats>>& device_stats() const {
        return stats_;
    }

    // Devices that failed init. Empty when every requested device
    // came up. Read by the TUI dashboard to render the per-device
    // failure detail next to the WORKERS row.
    const std::vector<FaultedDevice>& faulted_devices() const {
        return faulted_devices_;
    }

    // Number of devices the caller asked for at init time. Used by
    // the dashboard to render "M of N GPU active" without re-deriving
    // the count from sources outside the dispatcher.
    size_t requested_device_count() const { return requested_count_; }

    // Total addresses dispatched across all devices.
    uint64_t total_addresses_dispatched() const;

    // Human-readable summary of init outcome. Empty when init
    // returned 0 AND every device came up clean. Non-empty otherwise
    // (e.g. "1 of 2 GPU online; GPU#1 faulted"). For per-device
    // detail, read faulted_devices().
    const std::string& last_error() const { return last_error_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Stats vector exposed by reference; lives inside impl_'s lifetime.
    std::vector<std::unique_ptr<BipGpuDeviceStats>> stats_;
    std::vector<FaultedDevice> faulted_devices_;
    size_t      requested_count_ = 0;
    std::string last_error_;
};

}  // namespace collider::runtime

#endif  // COLLIDER_PRO
