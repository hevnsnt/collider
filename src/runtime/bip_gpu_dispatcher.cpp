/**
 * bip_gpu_dispatcher.cpp -- see bip_gpu_dispatcher.hpp for design.
 */

#include "runtime/bip_gpu_dispatcher.hpp"

#ifdef COLLIDER_PRO

#include "gpu/v2/v2_orchestrator.hpp"
#include "gpu/v2/brain_wallet_v2.hpp"
#include "runtime/bip_address.hpp"
#include "core/bip32.hpp"  // priv_to_pub
#include "core/crypto_cpu.hpp"

#include <cstring>
#include <iostream>

#if defined(COLLIDER_USE_CUDA)
#  include <cuda_runtime.h>
#endif

namespace collider::runtime {

// ---------------------------------------------------------------------------
// Per-device worker. Owns its MultiAddressSession; pulls batches from the
// shared queue; cudaSetDevice's its device on every entry to defend against
// other threads having stolen the current-device context. Drains the queue
// on shutdown.
// ---------------------------------------------------------------------------

struct BipGpuDispatcher::Impl {
    struct DeviceWorker {
        int device_id = 0;
        std::unique_ptr<::collider::gpu::v2::MultiAddressSession> session;
        std::thread thread;
    };

    Config cfg;
    std::vector<DeviceWorker> workers;

    std::deque<BipGpuWorkItem> queue;
    std::mutex                 queue_mu;
    std::condition_variable    queue_not_empty_cv;
    std::condition_variable    queue_not_full_cv;
    std::atomic<bool>          shutdown_requested{false};

    // Drain queue + signal workers; called once from shutdown().
    void signal_shutdown() {
        shutdown_requested.store(true, std::memory_order_release);
        queue_not_empty_cv.notify_all();
        queue_not_full_cv.notify_all();
    }

    // Pull up to batch_size items off the queue. Returns the number
    // pulled (may be 0 on shutdown drain). Blocks waiting for work
    // when the queue is empty AND shutdown not requested.
    size_t pull_batch(std::vector<BipGpuWorkItem>& out, size_t batch_size) {
        std::unique_lock<std::mutex> lk(queue_mu);
        queue_not_empty_cv.wait(lk, [&] {
            return !queue.empty() ||
                   shutdown_requested.load(std::memory_order_acquire);
        });
        out.clear();
        while (out.size() < batch_size && !queue.empty()) {
            out.push_back(std::move(queue.front()));
            queue.pop_front();
        }
        queue_not_full_cv.notify_all();
        return out.size();
    }
};

BipGpuDispatcher::BipGpuDispatcher() : impl_(std::make_unique<Impl>()) {}
BipGpuDispatcher::~BipGpuDispatcher() {
    if (impl_ && !impl_->shutdown_requested.load()) {
        shutdown();
    }
}

namespace {

// h160 helper for the hit-callback's address computation. Recomputed
// on the CPU because the GPU's V2MatchRecord does not carry the
// 20-byte digest (it only flags "this priv key hit"). Hits are rare
// enough that the per-hit CPU hash is negligible.
constexpr const char* addr_type_label(uint8_t at) {
    using ::collider::gpu::v2::AddressType;
    switch (static_cast<AddressType>(at)) {
        case AddressType::P2PKH_UNCOMPRESSED: return "P2PKH-uncompressed";
        case AddressType::P2PKH_COMPRESSED:   return "P2PKH-compressed";
        case AddressType::P2SH_P2WPKH:        return "P2SH-P2WPKH";
        case AddressType::P2WPKH_V0:          return "P2WPKH-bech32";
        case AddressType::P2TR_BIP86:         return "P2TR-bip86";
        case AddressType::ETHEREUM:           return "ETH";
        default:                               return "?";
    }
}

}  // namespace

int BipGpuDispatcher::init(const Config& cfg) {
    impl_->cfg = cfg;
    last_error_.clear();
    faulted_devices_.clear();
    requested_count_ = cfg.gpu_ids.size();
    if (!cfg.on_hit) {
        last_error_ = "on_hit callback required";
        std::cerr << "[!] BipGpuDispatcher::init: " << last_error_ << "\n";
        return 64;
    }
    if (cfg.gpu_ids.empty()) {
        // detect_gpus() in main.cpp auto-fills cfg.gpu_ids from
        // cudaGetDeviceCount before any runner is dispatched. An empty
        // list reaching us means either (a) the runner forgot to pass
        // args.gpu_ids through, or (b) the platform layer reported zero
        // devices. Either way, the dispatcher should not silently
        // default to {0} -- that hid the "user has 2 GPUs but only one
        // is used" bug for an entire sprint.
        last_error_ = "no GPU ids supplied (call detect_gpus first)";
        std::cerr << "[!] BipGpuDispatcher::init: " << last_error_ << "\n";
        return 64;
    }

#if defined(COLLIDER_USE_CUDA)
    // Validate at least one CUDA device is visible to the process.
    // RDP / Hyper-V / headless sessions sometimes hide GPUs from
    // user-mode processes; surface the cudaGetErrorString detail.
    int device_count = 0;
    cudaError_t cnt_rc = cudaGetDeviceCount(&device_count);
    if (cnt_rc != cudaSuccess) {
        last_error_ = std::string("cudaGetDeviceCount: ") +
                      cudaGetErrorString(cnt_rc);
        std::cerr << "[!] BipGpuDispatcher::init: " << last_error_ << "\n";
        return 70;
    }
    if (device_count == 0) {
        last_error_ = "no CUDA devices visible to process "
                      "(check driver / RDP / Hyper-V config)";
        std::cerr << "[!] BipGpuDispatcher::init: " << last_error_ << "\n";
        return 70;
    }

    // Per-device init loop. Failed devices are dropped (recorded in
    // faulted_devices_) instead of aborting the whole dispatcher --
    // a 2-GPU system where one card is busy / OOM should still run
    // the BIP scan on the working card.
    impl_->workers.reserve(cfg.gpu_ids.size());
    stats_.reserve(cfg.gpu_ids.size());
    auto fault = [&](int dev, std::string msg) {
        std::cerr << "[!] BipGpuDispatcher::init: GPU#" << dev
                  << " " << msg << " (skipped)\n";
        faulted_devices_.push_back({dev, std::move(msg)});
    };

    // Halving cascade: when a device's session init fails the
    // dispatcher retries on the same device with batch / 2, mirroring
    // the brain-wallet pipeline's TR-8 recovery pattern. This catches
    // (a) transient VRAM pressure on smaller cards in a heterogeneous
    // rig and (b) the first-launch-on-this-device CUDA module-load
    // window where a too-large initial allocation surfaces as
    // "invalid argument" rather than "out of memory". Per-device
    // floor and retry cap match the brain-wallet runner.
    constexpr int kMaxPerDeviceRetries = 4;
    constexpr size_t kMinPerDeviceBatch = 1024;

    for (int dev : cfg.gpu_ids) {
        if (dev >= device_count) {
            fault(dev, std::string("requested gpu_id but only ") +
                         std::to_string(device_count) + " device(s) present");
            continue;
        }
        cudaError_t set_rc = cudaSetDevice(dev);
        if (set_rc != cudaSuccess) {
            fault(dev, std::string("cudaSetDevice: ") +
                         cudaGetErrorString(set_rc));
            continue;
        }

        size_t per_device_batch = cfg.batch_size;
        std::unique_ptr<::collider::gpu::v2::MultiAddressSession> session;
        std::string last_detail;
        int          last_session_rc = 0;
        bool         device_ok = false;

        for (int attempt = 0; attempt <= kMaxPerDeviceRetries; ++attempt) {
            session = std::make_unique<
                ::collider::gpu::v2::MultiAddressSession>();
            last_session_rc = session->init(cfg.bloom_data, cfg.bloom_bits,
                                            cfg.bloom_hashes, cfg.bloom_seed,
                                            per_device_batch);
            if (last_session_rc == 0) {
                device_ok = true;
                if (attempt > 0) {
                    std::cerr << "[*] BipGpuDispatcher: GPU#" << dev
                              << " session init succeeded after " << attempt
                              << " halving(s) at batch " << per_device_batch
                              << "\n";
                }
                break;
            }
            last_detail = session->init_error_detail();
            session.reset();
            if (attempt == kMaxPerDeviceRetries) break;
            const size_t halved = per_device_batch / 2;
            if (halved < kMinPerDeviceBatch) break;
            per_device_batch = halved;
        }

        if (!device_ok) {
            std::string msg = std::string("MultiAddressSession::init rc=") +
                              std::to_string(last_session_rc);
            if (!last_detail.empty()) {
                msg += std::string(" (") + last_detail + ")";
            } else if (cudaError_t last_rc = cudaGetLastError();
                       last_rc != cudaSuccess) {
                // Fallback if init_error_detail() was empty (non-CUDA
                // build or pre-init bad-arg failure).
                msg += std::string(" (CUDA: ") +
                       cudaGetErrorString(last_rc) + ")";
            }
            fault(dev, std::move(msg));
            continue;
        }

        // Device came up. Commit a worker + a stats slot in lockstep
        // so device_stats()[i] always matches workers[i].
        Impl::DeviceWorker w;
        w.device_id = dev;
        w.session = std::move(session);
        impl_->workers.push_back(std::move(w));
        auto s = std::make_unique<BipGpuDeviceStats>();
        s->device_id = dev;
        s->effective_batch_size = per_device_batch;
        stats_.push_back(std::move(s));
    }

    if (impl_->workers.empty()) {
        // Every requested device failed. Surface the FIRST fault as
        // the headline; faulted_devices() has the rest.
        last_error_ = std::string("all ") +
                      std::to_string(cfg.gpu_ids.size()) +
                      " device(s) failed init";
        if (!faulted_devices_.empty()) {
            last_error_ += std::string(" (GPU#") +
                           std::to_string(faulted_devices_[0].device_id) +
                           ": " + faulted_devices_[0].error + ")";
        }
        std::cerr << "[!] BipGpuDispatcher::init: " << last_error_ << "\n";
        return 70;
    }
    if (!faulted_devices_.empty()) {
        // Partial init -- some devices came up, some didn't.
        last_error_ = std::to_string(impl_->workers.size()) +
                      " of " + std::to_string(cfg.gpu_ids.size()) +
                      " GPU online";
        // The detailed per-device messages live in faulted_devices().
    }

    // Spawn worker threads AFTER all sessions are initialized so a
    // partial-init failure cleans up via the destructor without a
    // running thread referencing a half-built session.
    for (size_t i = 0; i < impl_->workers.size(); ++i) {
        const size_t widx = i;
        auto& w = impl_->workers[widx];
        const int dev = w.device_id;
        // Resolve the per-device effective batch size set during the
        // init halving cascade. Each worker pulls + dispatches at most
        // this many items so a card that halved its session is never
        // asked to swallow more than its session's max_batch_count
        // (process_batch would otherwise reject with rc=64).
        const size_t per_device_batch = stats_[widx]->effective_batch_size > 0
            ? stats_[widx]->effective_batch_size
            : cfg.batch_size;
        w.thread = std::thread([this, widx, dev, per_device_batch]() {
            // Bind this thread to the device once at start; subsequent
            // process_batch + cudaMemcpy calls inherit the binding.
            cudaSetDevice(dev);

            const auto start = std::chrono::steady_clock::now();
            std::vector<BipGpuWorkItem> batch;
            std::vector<uint8_t> priv_buf;
            batch.reserve(per_device_batch);
            priv_buf.reserve(per_device_batch * 32);

            uint64_t local_dispatched = 0;
            uint64_t local_matches = 0;

            while (true) {
                size_t n = impl_->pull_batch(batch, per_device_batch);
                if (n == 0) {
                    if (impl_->shutdown_requested.load(
                            std::memory_order_acquire)) {
                        return;
                    }
                    continue;  // spurious wakeup
                }
                priv_buf.resize(n * 32);
                for (size_t j = 0; j < n; ++j) {
                    std::memcpy(priv_buf.data() + j * 32,
                                batch[j].priv.data(), 32);
                }
                // Build the batch request. addr_mask is the union of all
                // requested types across this batch -- the kernel checks
                // every requested type for every priv key, which is fine
                // for BIP scan where all profiles ultimately need
                // P2PKH_COMPRESSED + P2SH_P2WPKH coverage.
                uint32_t mask = 0;
                for (const auto& it : batch) {
                    mask |= static_cast<uint32_t>(it.addr_mask);
                }
                if (mask == 0) {
                    mask = ::collider::gpu::v2::addr_bit(
                               ::collider::gpu::v2::AddressType::
                                   P2PKH_COMPRESSED) |
                           ::collider::gpu::v2::addr_bit(
                               ::collider::gpu::v2::AddressType::
                                   P2SH_P2WPKH);
                }

                ::collider::gpu::v2::MultiAddressBatch req{};
                req.priv_batch  = priv_buf.data();
                req.count       = n;
                req.addr_mask   = mask;
                req.bloom       = impl_->cfg.bloom_data;
                req.bloom_bits  = impl_->cfg.bloom_bits;
                req.bloom_hashes = impl_->cfg.bloom_hashes;
                req.bloom_seed  = impl_->cfg.bloom_seed;

                auto& w_local = impl_->workers[widx];
                int rc = w_local.session->process_batch(req);
                if (rc != 0) {
                    std::cerr << "[!] GPU#" << dev
                              << " process_batch rc=" << rc << "\n";
                    // Keep the worker alive; transient CUDA errors
                    // typically recover next batch.
                    continue;
                }
                local_dispatched += n;

                // Retrieve matches via the structured API.
                auto recs = w_local.session->last_matches();
                for (const auto& r : recs) {
                    if (r.pp_idx >= batch.size()) continue;
                    const auto& item = batch[r.pp_idx];
                    // Recompute h160 for the hit log. We don't know
                    // which AddressType the GPU matched without the
                    // record's addr_type field; use it directly.
                    using ::collider::gpu::v2::AddressType;
                    std::array<uint8_t, 20> h160{};
                    // For P2SH_P2WPKH we need the redeem-script hash;
                    // for compressed P2PKH/P2WPKH we use hash160(pub).
                    // We don't have the pub here -- recompute from
                    // priv. EC mul on CPU is ~100us; hits are rare.
                    auto pub = ::collider::bip32::detail::priv_to_pub(
                        item.priv.data());
                    if (static_cast<AddressType>(r.addr_type) ==
                        AddressType::P2SH_P2WPKH) {
                        h160 = ::collider::bip_address::hash160_p2sh_p2wpkh(
                            pub.data());
                    } else {
                        h160 = ::collider::bip_address::hash160_pubkey(
                            pub.data());
                    }
                    impl_->cfg.on_hit(
                        item.mnemonic, item.derivation_path,
                        item.profile_label, item.priv.data(),
                        addr_type_label(r.addr_type), h160.data());
                    ++local_matches;
                }

                // Publish per-device counters.
                auto* stat = stats_[widx].get();
                stat->addresses_dispatched.store(
                    local_dispatched, std::memory_order_relaxed);
                stat->matches_emitted.store(
                    local_matches, std::memory_order_relaxed);
                const double secs = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start).count();
                stat->addresses_per_sec.store(
                    secs > 0 ? local_dispatched / secs : 0.0,
                    std::memory_order_relaxed);
            }
        });
    }
    return 0;
#else
    // Non-CUDA build: honest about it. Runner falls back to CPU based
    // on the non-zero return code rather than silently enqueueing into
    // a dispatcher that has no consumer threads.
    last_error_ = "binary built without COLLIDER_USE_CUDA";
    return 70;
#endif
}

bool BipGpuDispatcher::enqueue(BipGpuWorkItem&& item) {
#if defined(COLLIDER_USE_CUDA)
    std::unique_lock<std::mutex> lk(impl_->queue_mu);
    impl_->queue_not_full_cv.wait(lk, [&] {
        return impl_->queue.size() < impl_->cfg.queue_max ||
               impl_->shutdown_requested.load(std::memory_order_acquire);
    });
    if (impl_->shutdown_requested.load(std::memory_order_acquire)) {
        return false;
    }
    impl_->queue.push_back(std::move(item));
    impl_->queue_not_empty_cv.notify_one();
    return true;
#else
    (void)item;
    return false;  // non-CUDA: caller should use CPU path
#endif
}

void BipGpuDispatcher::shutdown() {
    impl_->signal_shutdown();
    for (auto& w : impl_->workers) {
        if (w.thread.joinable()) w.thread.join();
    }
    impl_->workers.clear();
}

uint64_t BipGpuDispatcher::total_addresses_dispatched() const {
    uint64_t total = 0;
    for (const auto& s : stats_) {
        total += s->addresses_dispatched.load(std::memory_order_relaxed);
    }
    return total;
}

}  // namespace collider::runtime

#endif  // COLLIDER_PRO
