// pool_manager.cpp - Pool manager implementation

#include "pool_manager.hpp"

#include "core/paths.hpp"
#include "core/secure_write.hpp"
#include "core/worker_identity.hpp"  // B1 wire-v4 signed AUTH

#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
    // _commit / _fileno: Windows equivalent of POSIX fsync. Used by
    // persist_dp_seq_map's fsync+unwind path so the OS write cache is
    // on stable storage before the tmp -> final rename. Mirrors the
    // canonical pattern in src/gpu/rckangaroo_wrapper.cu:1585+.
    #include <io.h>
#else
    #include <unistd.h>  // ::fsync, ::fileno
#endif

namespace collider {
namespace pool {

// Thread-safe singleton using Meyer's pattern (C++11+)
// Automatically cleaned up at program exit - no memory leak
PoolManager& get_pool_manager() {
    static PoolManager instance;
    return instance;
}

namespace {

// The user's collider state directory. paths::collider_home() returns
// ./.collider as last-resort fallback when neither USERPROFILE (Windows)
// nor HOME (POSIX) is set; previously this site (and the sibling in
// jlp_pool_client.cpp) used current_path()/.collider for that fallback,
// which resolves to the same on-disk location since both files are opened
// via std::ofstream against the process cwd. The order on Windows is now
// USERPROFILE-first (matches src/core/config.hpp, logger.hpp, and the
// other 14 sites); the pre-consolidation HOME-first order here was an
// MSYS-friendly outlier that nobody else followed.
std::filesystem::path dp_seq_persist_path() {
    return collider::paths::collider_home() / "pool_dp_seq.dat";
}

std::filesystem::path dp_queue_persist_path() {
    return collider::paths::collider_home() / "pool_dp_pending.dat";
}

}  // namespace

PoolManager::PoolManager()
    : connected_(false)
    , has_work_(false)
{
    start_time_ = std::chrono::steady_clock::now();
    // load the persisted DP sequence map at boot so a
    // restart after a crash doesn't reset all counters to 0 (which
    // would tip the server's replay-defence watermark and self-ban).
    load_dp_seq_map();
}

PoolManager::~PoolManager() {
    stop_supervisor();
    disconnect();
}

void PoolManager::set_config(const PoolConfig& config) {
    config_ = config;
}

// B1 helpers.

std::string PoolManager::dp_seq_key(uint64_t work_id) const {
    std::ostringstream key;
    key << config_.worker_name << "|" << work_id;
    return key.str();
}

uint32_t PoolManager::get_or_create_dp_seq(uint64_t work_id) {
    std::lock_guard<std::mutex> lock(dp_seq_mutex_);
    auto key = dp_seq_key(work_id);
    auto it = dp_seq_map_.find(key);
    if (it == dp_seq_map_.end()) {
        dp_seq_map_.emplace(key, 0);
        return 0;
    }
    return it->second;
}

void PoolManager::update_dp_seq(uint64_t work_id, uint32_t next_seq) {
    std::lock_guard<std::mutex> lock(dp_seq_mutex_);
    dp_seq_map_[dp_seq_key(work_id)] = next_seq;
}

void PoolManager::reset_dp_seq(uint64_t work_id) {
    std::lock_guard<std::mutex> lock(dp_seq_mutex_);
    dp_seq_map_[dp_seq_key(work_id)] = 0;
}

// Binary file format for the dp_seq_map_ persistence file:
//   [4 bytes magic 'DSQ1'][u32 count]
//   { [u32 key_len][key_len bytes key][u32 next_seq] } * count
// All multi-byte values little-endian. Keys are "worker_name|work_id".
void PoolManager::load_dp_seq_map() {
    namespace fs = std::filesystem;
    fs::path p = dp_seq_persist_path();
    std::error_code ec;
    if (!fs::exists(p, ec)) return;
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) return;
    auto read_u32 = [&](uint32_t& out) -> bool {
        unsigned char buf[4];
        if (!ifs.read(reinterpret_cast<char*>(buf), 4)) return false;
        out = static_cast<uint32_t>(buf[0])
            | (static_cast<uint32_t>(buf[1]) << 8)
            | (static_cast<uint32_t>(buf[2]) << 16)
            | (static_cast<uint32_t>(buf[3]) << 24);
        return true;
    };
    char magic[4] = {0};
    if (!ifs.read(magic, 4)) return;
    if (magic[0] != 'D' || magic[1] != 'S' || magic[2] != 'Q' || magic[3] != '1') {
        std::cerr << "[PoolManager] pool_dp_seq.dat: bad magic; ignoring."
                  << std::endl;
        return;
    }
    uint32_t count = 0;
    if (!read_u32(count)) return;
    // Sanity cap: refuse a file claiming more entries than would fit in
    // 4 MiB of map state (which is ~100k workers x 40-byte keys).
    if (count > 100000u) {
        std::cerr << "[PoolManager] pool_dp_seq.dat: count " << count
                  << " exceeds sanity cap; ignoring." << std::endl;
        return;
    }
    std::lock_guard<std::mutex> lock(dp_seq_mutex_);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t klen = 0;
        if (!read_u32(klen) || klen > 4096u) return;
        std::string key(klen, '\0');
        if (!ifs.read(key.data(), klen)) return;
        uint32_t next = 0;
        if (!read_u32(next)) return;
        dp_seq_map_[std::move(key)] = next;
    }
}

void PoolManager::persist_dp_seq_map() const {
    namespace fs = std::filesystem;
    fs::path p = dp_seq_persist_path();
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    auto tmp = p;
    tmp += ".tmp";

    // ~/.collider/pool_dp_seq.dat tracks per-worker DP sequence numbers
    // tied to this worker's identity. The 2026-05-09 review flagged
    // that this site's atomic-write contract was missing the
    // fsync+unwind durability guarantee that the canonical pattern in
    // src/gpu/rckangaroo_wrapper.cu:1495-1639 (save_herd_state) ships.
    // Port the full pattern here so a SIGINT or power loss between the
    // last fwrite and the rename does not leave a torn DSQ1 file on
    // disk for the next load_dp_seq_map() to reject (and silently lose
    // every worker's sequence baseline as a result, which on the next
    // run forces every submitted DP through the "AUTH not yet OK"
    // re-queue path with brand new sequence numbers, manifesting on
    // the server as a duplicate-DP-flood).
    //
    // Sequence (mirrors save_herd_state, FILE*-based throughout):
    //   1. secure_open_ofstream + close the tmp so the file is created
    //      with the owner-only DACL we have always promised (this file
    //      is not key material but lives next to wallet state; keep the
    //      access pattern consistent across pool persistence sites).
    //   2. Reopen via fopen("wb") so we have a real fd for fsync.
    //   3. fwrite header + payload, fflush libc->OS, fsync (POSIX) /
    //      _commit (Windows) the fd, fclose.
    //   4. Atomic rename tmp -> final. On any failure between (2) and
    //      (4), the unwind helper closes fp and removes the partial
    //      tmp. Pre-existing pool_dp_seq.dat (if any) is untouched.

    // Step 1: anchor permissions on the new tmp file.
    {
        std::ofstream ofs = ::collider::secure_open_ofstream(
            tmp, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            std::cerr << "[PoolManager] persist_dp_seq_map: cannot open "
                      << tmp.string() << std::endl;
            return;
        }
        // Close immediately; we only used the stream to apply the DACL.
        // The fopen below reopens in "wb" mode for the fsync ride.
    }

    // Step 2: reopen via FILE* so the fd reaches fsync / _commit.
    FILE* fp = std::fopen(tmp.string().c_str(), "wb");
    if (!fp) {
        std::cerr << "[PoolManager] persist_dp_seq_map: fopen failed for "
                  << tmp.string() << std::endl;
        fs::remove(tmp, ec);
        return;
    }

    auto unwind = [&fp, &tmp](const char* where) {
        if (fp) {
            std::fclose(fp);
            fp = nullptr;
        }
        std::error_code rm_ec;
        fs::remove(tmp, rm_ec);
        std::cerr << "[PoolManager] persist_dp_seq_map: " << where
                  << " failed during atomic save; partial "
                  << tmp.string() << " removed." << std::endl;
    };

    auto put_u32_fp = [&fp](uint32_t v) -> bool {
        unsigned char buf[4] = {
            static_cast<unsigned char>( v        & 0xFFu),
            static_cast<unsigned char>((v >>  8) & 0xFFu),
            static_cast<unsigned char>((v >> 16) & 0xFFu),
            static_cast<unsigned char>((v >> 24) & 0xFFu),
        };
        return std::fwrite(buf, 1, 4, fp) == 4;
    };

    // Step 3: write header + payload via FILE*. Order/format matches
    // the previous ofstream-based version byte-for-byte so existing
    // load_dp_seq_map() readers consume it unchanged.
    if (std::fwrite("DSQ1", 1, 4, fp) != 4) { unwind("magic write"); return; }

    {
        std::lock_guard<std::mutex> lock(dp_seq_mutex_);
        if (!put_u32_fp(static_cast<uint32_t>(dp_seq_map_.size()))) {
            unwind("count write");
            return;
        }
        for (const auto& [k, v] : dp_seq_map_) {
            if (!put_u32_fp(static_cast<uint32_t>(k.size()))) {
                unwind("key-length write");
                return;
            }
            if (std::fwrite(k.data(), 1, k.size(), fp) != k.size()) {
                unwind("key body write");
                return;
            }
            if (!put_u32_fp(v)) {
                unwind("seq value write");
                return;
            }
        }
    }

    // Flush libc -> OS, then sync OS -> disk before we close fp.
    if (std::fflush(fp) != 0) { unwind("fflush"); return; }
#ifdef _WIN32
    if (_commit(_fileno(fp)) != 0) { unwind("_commit"); return; }
#else
    if (::fsync(::fileno(fp)) != 0) { unwind("fsync"); return; }
#endif

    std::fclose(fp);
    fp = nullptr;

    // Step 4: atomic rename tmp -> final. Windows std::filesystem::
    // rename throws on a pre-existing target; POSIX rename(2) is atomic
    // against concurrent readers and replaces in place. The remove +
    // rename retry window on Windows is small but not zero; a power
    // loss inside it leaves the new payload at .tmp under its tmp name
    // and the pre-existing pool_dp_seq.dat removed -- strictly better
    // than the pre-fix outcome (a torn pool_dp_seq.dat the next run
    // would reject at the DSQ1 header check, then silently start every
    // worker at seq=0).
    fs::rename(tmp, p, ec);
    if (ec) {
        fs::remove(p, ec);
        ec.clear();
        fs::rename(tmp, p, ec);
        if (ec) {
            std::error_code rm_ec;
            fs::remove(tmp, rm_ec);
            std::cerr << "[PoolManager] persist_dp_seq_map: rename "
                      << tmp.string() << " -> " << p.string()
                      << " failed: " << ec.message() << std::endl;
        }
    }
}

// B1 + Pool-F6: wire up the seq-progress + work-id-assigned +
// solution + work callbacks on a JLP client. Used by both PoolManager::
// connect() (first time) and the supervisor's reconnect path so the
// behavior is identical in both. The supplied JLPPoolClient* must
// outlive this call.
void PoolManager::wire_client_callbacks(JLPPoolClient* jlp) {
    // v1.5: Solution-broadcast callback. Fires when the pool server's
    // SOLUTION frame arrives at the JLPPoolClient. We do NOT echo the
    // 32-byte payload here -- the host-level callback (run_pool_mode)
    // is the only place that decides what to do, and the v1.5 contract
    // is "treat as stop signal, never persist or display the bytes".
    jlp->set_solution_callback([this](const uint8_t* solution_payload) {
        std::cout << "\n[PoolManager] SOLUTION FOUND BY POOL "
                     "-- forwarding stop signal to host."
                  << std::endl;
        if (solution_callback_) {
            solution_callback_(solution_payload, config_.worker_name);
        }
    });

    // Work callback: dedup at the manager level (Pool-F6) so the host
    // sees a single notification per work_id even across reconnects.
    jlp->set_work_callback([this](const WorkAssignment& work) {
        const uint64_t prev = last_work_id_seen_.exchange(work.work_id);
        if (prev == work.work_id) {
            // Already surfaced to host on a previous WORK_ASN; suppress.
            return;
        }
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            current_work_ = work;
            has_work_ = true;
        }
        current_dp_bits_.store(work.dp_bits, std::memory_order_release);
        current_work_id_.store(work.work_id, std::memory_order_release);
        std::cout << "[PoolManager] Received new work: " << work.puzzle_name
                  << " (DP bits: " << work.dp_bits << ")" << std::endl;
    });

    // Work-id-assigned hook: reset the per-(worker, work_id) counter
    // when a genuinely new chunk arrives, then seed the client with
    // the persisted value for this key.
    jlp->set_work_id_assigned_callback([this, jlp](uint64_t work_id) {
        // Detect genuine reassignment via the manager-level dedup
        // sentinel. last_work_id_seen_ is updated by the work_callback
        // ABOVE; we read it via a relaxed load.
        const uint64_t seen = last_work_id_seen_.load(std::memory_order_relaxed);
        if (seen != work_id) {
            // First time seeing this work_id on the manager side:
            // reset the counter to 0 (server expects fresh chunk
            // sequence to start at 0).
            reset_dp_seq(work_id);
        }
        // Seed the client with the (possibly persisted) next-seq.
        uint32_t seed = get_or_create_dp_seq(work_id);
        jlp->seed_dp_sequence(work_id, seed);
    });

    // Kangaroo-type-changed hook (v1.5 type-mismatch epoch race): when the
    // server reassigns this worker from TAME_ONLY <-> WILD_ONLY, the JLP
    // client has already flushed its own dp_queue_. The reconnect-window
    // buffer (reconnect_buffer_) lives here in the manager and may still
    // hold DPs of the OLD type captured while a reconnect was in flight.
    // Drop them now so they can never be replayed and shipped under the
    // new work_id (the server drops + penalizes type mismatches).
    jlp->set_kangaroo_type_changed_callback(
        [this](uint8_t prev_type, uint8_t new_type) {
            size_t dropped = 0;
            {
                std::lock_guard<std::mutex> lock(reconnect_buf_mutex_);
                dropped = reconnect_buffer_.size();
                reconnect_buffer_.clear();
            }
            if (dropped != 0) {
                dropped_count_.fetch_add(dropped, std::memory_order_relaxed);
                std::cout << "[PoolManager] kangaroo_type changed "
                          << static_cast<int>(prev_type) << "->"
                          << static_cast<int>(new_type)
                          << "; flushed " << dropped
                          << " stale-type DP(s) from the reconnect-window "
                             "buffer." << std::endl;
            }
        });

    // Maintenance hook (v1.5.4): the server asked workers to back off.
    // Record the state so supervisor_loop uses the operator-suggested
    // retry window (not the exponential backoff) and does NOT treat the
    // lost session as a failure / churn. active=1 also requests a
    // graceful disconnect so the supervisor reconnect path takes over;
    // active=0 is an explicit all-clear that wakes the supervisor early.
    jlp->set_maintenance_callback(
        [this](bool active, uint32_t retry_after_secs, std::string message) {
            if (active) {
                maintenance_active_.store(true, std::memory_order_release);
                maintenance_retry_secs_.store(retry_after_secs,
                                              std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lock(maintenance_mutex_);
                    maintenance_message_ = std::move(message);
                }
                std::cout << "[PoolManager] Pool entered MAINTENANCE; "
                          << "backing off for ~" << retry_after_secs
                          << "s as requested by the operator." << std::endl;
                // Drop the host sentinel so is_connected() reports down and
                // the supervisor takes over the maintenance-aware reconnect
                // wait WITHOUT counting it as a fault. We must NOT call
                // client_->disconnect() here: this callback runs ON the
                // receiver thread (handle_maintenance), and disconnect()
                // joins that same thread, which would deadlock. The
                // supervisor thread owns teardown of the old client in its
                // reconnect-drive block; flipping connected_ is enough to
                // route control there.
                connected_.store(false, std::memory_order_release);
            } else {
                // All-clear: clear the flag and wake the supervisor so it
                // reconnects promptly instead of waiting out the window.
                maintenance_active_.store(false, std::memory_order_release);
                {
                    std::lock_guard<std::mutex> lock(maintenance_mutex_);
                    maintenance_message_.clear();
                }
                std::cout << "[PoolManager] Pool maintenance cleared "
                             "(all-clear); resuming." << std::endl;
            }
            {
                std::lock_guard<std::mutex> lock(supervisor_mutex_);
                supervisor_cv_.notify_all();
            }
        });

    // DP-seq-progress hook: mirror the client's bumped counter back
    // into the manager's map so a supervisor-driven reconnect that
    // recreates the client seeds it with the right next value, AND
    // count the wire send for Pool-F3 separated stats.
    jlp->set_dp_sequence_progress_callback(
        [this](uint64_t work_id, uint32_t next_seq) {
            // The delta between the new and stored value is the number
            // of DPs that actually went out on the wire in the batch
            // we just emitted. Use this to bump sent_count_.
            uint32_t prev = 0;
            {
                std::lock_guard<std::mutex> lock(dp_seq_mutex_);
                auto it = dp_seq_map_.find(dp_seq_key(work_id));
                if (it != dp_seq_map_.end()) {
                    prev = it->second;
                }
                dp_seq_map_[dp_seq_key(work_id)] = next_seq;
            }
            if (next_seq > prev) {
                sent_count_.fetch_add(next_seq - prev,
                                      std::memory_order_relaxed);
            }
        });
}

bool PoolManager::connect() {
    if (connected_) {
        disconnect();
    }

    // Create appropriate client.
    // NOTE: the HTTP pool path was removed. The HTTP client silently
    // downgraded https:// URLs to plaintext, leaking the worker's Bitcoin
    // address (= payout address) and API tokens in the clear. JLP+TLS
    // (jlps://) is the only supported pool protocol going forward.
    JLPPoolClient* jlp_raw = nullptr;
    if (config_.type == POOL_TYPE_JLP) {
        auto jlp_client = std::make_unique<JLPPoolClient>();
        jlp_client->set_timeout(config_.timeout_ms);
        jlp_client->set_reconnect(config_.auto_reconnect);
        jlp_client->set_debug_mode(config_.debug_mode);
        jlp_client->set_use_tls(config_.use_tls);
        jlp_client->set_verify_cert(config_.verify_cert);
        jlp_raw = jlp_client.get();
        client_ = std::move(jlp_client);
        // B1 wire-v4: if --worker-key was supplied, load the WIF and
        // attach the signed identity to this client (and every future
        // reconnect-created client) so AUTH frames are v4-signed.
        apply_worker_identity(jlp_raw);
    } else if (config_.type == POOL_TYPE_HTTP) {
        std::cerr << "[PoolManager] HTTP pool deprecated; use jlp:// or jlps://"
                  << std::endl;
        std::cerr << "[PoolManager]   The HTTP pool transport was removed in the "
                     "2026-05-04 security review (silent plaintext credential leak)."
                  << std::endl;
        return false;
    } else {
        std::cerr << "[PoolManager] Unknown pool type: '" << config_.type
                  << "'. Only 'jlp' is supported (use jlp:// or jlps:// URLs)."
                  << std::endl;
        return false;
    }

    // B1 + Pool-F6: wire up the seq + work-id + work + solution
    // callbacks via the shared helper so the supervisor reconnect path
    // gets identical behavior.
    wire_client_callbacks(jlp_raw);

    // replay any DPs persisted from a prior shutdown
    // BEFORE we authenticate. The first DP_BATCH_V2 to land post-AUTH
    // will include the backlog; under steady state the file is rewritten
    // on disconnect and cleared on successful send.
    {
        namespace fs = std::filesystem;
        fs::path qpath = dp_queue_persist_path();
        std::error_code ec;
        if (fs::exists(qpath, ec)) {
            std::ifstream ifs(qpath, std::ios::binary);
            if (ifs) {
                char magic[4] = {0};
                ifs.read(magic, 4);
                if (magic[0] == 'C' && magic[1] == 'D'
                    && magic[2] == 'P' && magic[3] == 'Q') {
                    auto read_u32 = [&](uint32_t& out) -> bool {
                        unsigned char buf[4];
                        if (!ifs.read(reinterpret_cast<char*>(buf), 4)) return false;
                        out = static_cast<uint32_t>(buf[0])
                            | (static_cast<uint32_t>(buf[1]) << 8)
                            | (static_cast<uint32_t>(buf[2]) << 16)
                            | (static_cast<uint32_t>(buf[3]) << 24);
                        return true;
                    };
                    auto read_u64 = [&](uint64_t& out) -> bool {
                        unsigned char buf[8];
                        if (!ifs.read(reinterpret_cast<char*>(buf), 8)) return false;
                        out = 0;
                        for (int i = 0; i < 8; ++i) {
                            out |= static_cast<uint64_t>(buf[i]) << (8 * i);
                        }
                        return true;
                    };
                    uint32_t count = 0;
                    if (read_u32(count) && count <= JLPPoolClient::MAX_DP_QUEUE_SIZE) {
                        std::vector<DistinguishedPoint> persisted;
                        persisted.reserve(count);
                        bool ok = true;
                        for (uint32_t i = 0; i < count && ok; ++i) {
                            DistinguishedPoint dp{};
                            if (!ifs.read(reinterpret_cast<char*>(dp.x), 32)) {
                                ok = false; break;
                            }
                            if (!ifs.read(reinterpret_cast<char*>(dp.d), 32)) {
                                ok = false; break;
                            }
                            char t = 0;
                            if (!ifs.read(&t, 1)) { ok = false; break; }
                            dp.type = static_cast<uint8_t>(t);
                            if (!read_u64(dp.dp_bits)) { ok = false; break; }
                            persisted.push_back(dp);
                        }
                        if (ok && !persisted.empty()) {
                            std::cout << "[PoolManager] Replaying "
                                      << persisted.size()
                                      << " persisted DP(s) from prior shutdown."
                                      << std::endl;
                            jlp_raw->preload_dp_queue(std::move(persisted));
                        }
                    }
                }
            }
            // Whether or not we successfully replayed, remove the file.
            // The in-memory queue is now the source of truth and we
            // don't want to re-replay on the next reconnect.
            fs::remove(qpath, ec);
        }
    }

    // Connect
    if (!client_->connect(config_.host, config_.port)) {
        return false;
    }

    // Authenticate
    if (!client_->authenticate(config_.worker_name, config_.password)) {
        std::cerr << "[PoolManager] Authentication failed" << std::endl;
        client_->disconnect();
        return false;
    }

    connected_ = true;
    start_time_ = std::chrono::steady_clock::now();
    // do NOT reset enqueued/sent counts on (re)connect;
    // they are cumulative across the process lifetime (the host panel
    // reads them via get_enqueued_count / get_sent_count and resetting
    // would visually rewind progress on every reconnect).

    std::cout << "[PoolManager] Connected to pool as " << config_.worker_name << std::endl;

    // Spawn the reconnect supervisor. The receiver loop in JLPPoolClient
    // exits cleanly on transient network errors and waits for an
    // "external supervisor" to call connect() / authenticate() -- this
    // is that supervisor.
    if (config_.auto_reconnect) {
        start_supervisor();
    }
    return true;
}

void PoolManager::disconnect() {
    stop_supervisor();
    if (client_) {
        // snapshot the JLP client's pending DP queue and
        // persist it to disk BEFORE we destroy the client. The next
        // PoolManager::connect() will preload the persisted backlog so
        // a clean shutdown (Ctrl+C, SIGTERM, host quit) doesn't lose
        // DPs that the network drain in JLPPoolClient::disconnect()
        // couldn't flush in time.
        if (auto* jlp = dynamic_cast<JLPPoolClient*>(client_.get())) {
            // Snapshot the in-flight sequence values one last time so
            // any uncomitted progress made between the last
            // dp_seq_progress callback and now is captured.
            uint64_t snap_work_id = 0;
            uint32_t snap_seq = 0;
            jlp->snapshot_dp_sequence(snap_work_id, snap_seq);
            if (snap_seq > 0) {
                update_dp_seq(snap_work_id, snap_seq);
            }
            // Persist the DP queue file. The JLP client's own
            // disconnect() drains as much as it can over the network
            // first; anything still queued after the drain is what we
            // serialize.
            std::string qpath = dp_queue_persist_path().string();
            if (!jlp->persist_dp_queue_to_disk(qpath)) {
                std::cerr << "[PoolManager] WARN failed to persist DP "
                             "backlog to " << qpath
                          << "; pending DPs will be lost." << std::endl;
            }
            // Roll any unsent DPs the JLP sender had to drop on
            // shutdown into our dropped_count_ so operators see the
            // total visible count.
            dropped_count_.fetch_add(jlp->get_shutdown_drop_count(),
                                     std::memory_order_relaxed);
        }
        client_->disconnect();
        client_.reset();
    }
    // persist the per-(worker, work_id) sequence map so
    // the next process start seeds the JLP client with the latest
    // counters instead of resetting to 0 and tripping the server's
    // replay watermark.
    persist_dp_seq_map();

    connected_ = false;
    current_dp_bits_.store(0, std::memory_order_release);
}

bool PoolManager::is_connected() const {
    return connected_ && client_ && client_->is_connected();
}

JLPPoolClient::UpdateAdvert PoolManager::get_update_advert() const {
    if (client_) {
        if (auto* jlp = dynamic_cast<JLPPoolClient*>(client_.get())) {
            return jlp->get_update_advert();
        }
    }
    return JLPPoolClient::UpdateAdvert{};
}

bool PoolManager::get_work(WorkAssignment& work) {
    if (!is_connected()) {
        return false;
    }

    // Fast path: cached work assignment from the work-callback.
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        if (has_work_) {
            work = current_work_;
            return true;
        }
    }

    // Slow path: explicit request to the pool. The previous version held
    // work_mutex_ across this blocking I/O call -- any concurrent
    // submit_dp from the kernel-callback path was forced to wait for the
    // entire request_work round-trip (multiple seconds at TLS handshake
    // + receive timeout). Now we make the call WITHOUT the lock; only
    // the (X, Y) state-update under the result is locked.
    if (client_->request_work(work)) {
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            current_work_ = work;
            has_work_ = true;
        }
        current_dp_bits_.store(work.dp_bits, std::memory_order_release);
        return true;
    }

    return false;
}

void PoolManager::submit_dp(const uint8_t* x, const uint8_t* d, uint8_t type, uint32_t dp_bits) {
    submit_dp(x, d, type, dp_bits, nullptr, nullptr);
}

void PoolManager::submit_dp(const uint8_t* x, const uint8_t* d, uint8_t type,
                            uint32_t dp_bits,
                            const std::vector<std::array<uint8_t, 32>>* ckpt_distances,
                            const std::vector<uint8_t>* ckpt_l1s2) {
    DistinguishedPoint dp;
    memcpy(dp.x, x, 32);
    memcpy(dp.d, d, 32);
    dp.type = type;
    dp.dp_bits = dp_bits;
    // v1.5.5 (task #9): copy the COMMITTABLE checkpoint chain onto the queued
    // DP so the JLP sender can emit a real DP_BATCH_V3 commitment for it. Both
    // pointers are null (or the chain has < 2 checkpoints) for the overwhelming
    // majority of DPs -- those whose producing kangaroo is outside the capture
    // window, hit a loop-escape, or whose backend never captures -- and such
    // DPs keep ckpt_distances empty and ship as V2. The pointers are only valid
    // for the duration of this call (the GPU readback buffers live on the
    // harvest stack), so we copy here rather than alias.
    if (ckpt_distances && ckpt_l1s2 && ckpt_distances->size() >= 2 &&
        ckpt_distances->size() == ckpt_l1s2->size()) {
        dp.ckpt_distances = *ckpt_distances;
        dp.ckpt_l1s2 = *ckpt_l1s2;
    }
    submit_dp(dp);
}

void PoolManager::submit_dp(const DistinguishedPoint& dp) {
    if (!is_connected()) {
        // Reconnect-window buffer (audit #21): hold the DP in a bounded
        // side queue until the supervisor brings a new client up + the
        // post-AUTH_OK WORK_REQ resolves. On successful reconnect the
        // drain step in supervisor_loop() replays these into the new
        // client's queue. If the buffer is also full, the DP is truly
        // dropped (operator-visible via dropped_count_); the cap is
        // generous enough that only multi-minute outages can hit it.
        //
        // We tag each DP with the work_id it was computed under. Stale
        // work_ids are NOT harmless: the server treats a DP whose work_id
        // does not match the worker's active assignment as a stale-work_id
        // strike, and 20 strikes force-close the connection. If the buffer
        // is replayed blindly after the server reassigned a different chunk
        // (e.g. after a server restart), every replayed DP strikes out and
        // the worker hot-loops reconnecting. The drain therefore replays
        // only DPs whose work_id still matches the active assignment.
        const uint64_t cap_work_id =
            current_work_id_.load(std::memory_order_acquire);
        std::lock_guard<std::mutex> lock(reconnect_buf_mutex_);
        if (reconnect_buffer_.size() < kReconnectBufferCap) {
            reconnect_buffer_.push_back(BufferedDP{dp, cap_work_id});
            reconnect_buffered_total_.fetch_add(1,
                                                std::memory_order_relaxed);
            return;
        }
        dropped_count_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    if (client_->submit_dp(dp)) {
        // this counter measures ENQUEUE success only.
        // Actual on-wire send is counted by sent_count_, bumped from
        // the dp_sequence_progress callback in wire_client_callbacks.
        enqueued_count_.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Queue full -- the JLP client's bounded MAX_DP_QUEUE_SIZE
        // backpressure rejected this. Pre-1.4 we silently incremented
        // submitted_count_, hiding the drop from operators. Now we
        // count + emit a rate-limited stderr warning.
        const uint64_t dropped =
            dropped_count_.fetch_add(1, std::memory_order_relaxed) + 1;
        // Warn every Nth drop to avoid log spam during sustained backpressure.
        constexpr uint64_t kWarnEvery = 1024;
        if ((dropped & (kWarnEvery - 1)) == 0) {
            std::fprintf(stderr,
                "[pool] WARN DP submission queue full -- dropped %llu DP(s) "
                "since startup. The pool sender thread is not draining fast "
                "enough; check network latency or DP rate.\n",
                static_cast<unsigned long long>(dropped));
        }
    }
}

PoolStatsLocal PoolManager::get_stats() const {
    if (!client_) {
        return PoolStatsLocal{};
    }
    return client_->get_stats();
}

double PoolManager::get_submission_rate() const {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed < 1.0) return 0.0;
    // rate is computed against the on-wire counter so a
    // stall in the sender thread visibly drops the rate to zero rather
    // than continuing to climb on enqueue alone.
    return static_cast<double>(sent_count_.load()) / elapsed;
}

// v1.5: PoolManager::report_solution() was DELETED together with the
// underlying PoolClient::report_solution and JLPPoolClient::report_solution
// surfaces. The pool server (collision-protocol) is the sole key-computer.
// See .claude/tasks/v1.5-asymmetric-kangaroo.md.

std::string PoolManager::get_status_string() const {
    std::stringstream ss;

    if (!is_connected()) {
        ss << "Disconnected";
        return ss.str();
    }

    PoolStatsLocal stats = get_stats();
    ss << "Connected | ";
    ss << "Your DPs: " << stats.your_dps << " | ";
    ss << "Pool DPs: " << stats.total_dps << " | ";
    ss << "Workers: " << stats.connected_workers << " | ";
    ss << "Rate: " << std::fixed << std::setprecision(1) << get_submission_rate() << " DP/s";

    if (stats.your_share > 0) {
        ss << " | Share: " << std::fixed << std::setprecision(2) << (stats.your_share * 100) << "%";
    }

    return ss.str();
}

// Static callback hook for RCKangaroo integration. Reads dp_bits from
// the atomic snapshot rather than locking work_mutex_ -- the kernel
// callback fires per-DP and contended locking would serialize an
// otherwise-parallel hot path.
void PoolManager::dp_callback_hook(void* user_data, const uint8_t* x, const uint8_t* d, uint8_t type) {
    PoolManager* manager = static_cast<PoolManager*>(user_data);
    if (manager && manager->is_connected()) {
        const uint32_t dp_bits =
            manager->current_dp_bits_.load(std::memory_order_acquire);
        manager->submit_dp(x, d, type, dp_bits);
    }
}

// ---------------------------------------------------------------------------
// Reconnect supervisor
// JLPPoolClient::receiver_loop() exits cleanly when its socket dies or
// the server tears down the session. The header at jlp_pool_client.cpp:897
// says "external supervisor must call connect() / authenticate() to
// recover" -- without an external supervisor, transient failures
// silently killed the worker.
// This thread polls is_connected() and, when it sees the receiver has
// exited, drives a full reconnect: disconnect() + connect() + authenticate()
// with jittered exponential backoff. After kMaxReconnectAttempts
// consecutive failures, supervisor_gave_up_ is set and the host loop
// can react. The receiver thread inside the client is a fresh thread
// per connect() because connect() calls replace_thread().
// ---------------------------------------------------------------------------

void PoolManager::start_supervisor() {
    // Idempotent: only one supervisor at a time.
    if (supervisor_thread_.joinable()) return;
    supervisor_stop_.store(false, std::memory_order_release);
    supervisor_gave_up_.store(false, std::memory_order_release);
    supervisor_thread_ = std::thread([this]() { this->supervisor_loop(); });
}

void PoolManager::stop_supervisor() {
    supervisor_stop_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(supervisor_mutex_);
        supervisor_cv_.notify_all();
    }
    if (supervisor_thread_.joinable()) {
        supervisor_thread_.join();
    }
}

void PoolManager::supervisor_loop() {
    // Backoff state. Reset to kInitialBackoffMs on every successful
    // reconnect; doubles up to MAX_RECONNECT_BACKOFF_MS on each
    // consecutive failure. Jitter avoids thundering-herd against the
    // pool when a fleet of workers reconnects together.
    uint32_t backoff_ms = kInitialBackoffMs;
    uint32_t consecutive_failures = 0;
    // the supervisor is now the only place that
    // tracks consecutive AUTH_FAILs (the dead in-receiver-thread block
    // was deleted). Cap at MAX_AUTH_FAIL_ATTEMPTS so a stale credential
    // doesn't hammer the pool forever.
    uint32_t consecutive_auth_failures = 0;

    // Reconnect-churn circuit-breaker state. last_session_start marks when
    // the most recent reconnect went fully live (post-WORK_ASN); a
    // default-constructed time_point means "no live session yet". When the
    // receiver later exits we measure how long that session lasted: a
    // sub-kSustainedSessionMs lifetime on the SAME work_id is churn, and
    // kMaxChurnReconnects of those in a row trips the breaker. Any sustained
    // session, or a session on a new work_id, resets churn_count.
    std::chrono::steady_clock::time_point last_session_start{};
    uint64_t   last_session_work_id = 0;
    uint32_t   churn_count = 0;
    std::random_device rd;
    std::mt19937 rng(rd());

    while (!supervisor_stop_.load(std::memory_order_acquire)) {
        // Sleep on the condition variable so disconnect() / shutdown
        // can wake us promptly. Wake either every 500ms (probe) or
        // when the stop flag flips.
        {
            std::unique_lock<std::mutex> lock(supervisor_mutex_);
            supervisor_cv_.wait_for(lock, std::chrono::milliseconds(500),
                [this]() { return supervisor_stop_.load(std::memory_order_acquire); });
        }
        if (supervisor_stop_.load(std::memory_order_acquire)) break;

        // Healthy: do nothing. is_connected() reads connected_ (the host
        // sentinel) AND the client's own connection flag, so a receiver
        // thread that exited will flip the second to false.
        if (is_connected()) {
            backoff_ms = kInitialBackoffMs;
            consecutive_failures = 0;
            continue;
        }

        // v1.5.4 maintenance: if the pool put us into a maintenance window,
        // the lost session is EXPECTED, not a fault. Skip churn evaluation
        // and the failure counters entirely, wait the operator-suggested
        // retry window (jittered), and loop back to drive a reconnect.
        // The maintenance flag is cleared by a subsequent successful AUTH
        // that does not re-assert maintenance (see below), or by an
        // explicit all-clear MAINTENANCE(active=0) frame.
        bool from_maintenance = false;
        if (maintenance_active_.load(std::memory_order_acquire)) {
            from_maintenance = true;
            // Consume the lost session marker so we don't double-evaluate
            // it as churn once maintenance clears.
            last_session_start = {};
            uint32_t retry = maintenance_retry_secs_.load(
                std::memory_order_acquire);
            // Clamp to [1s, 24h] before scaling to milliseconds. A buggy or
            // hostile server could advertise retry_after_secs near UINT32_MAX,
            // and `retry * 1000` (or the +50% jitter) would overflow uint32
            // and wrap to a tiny or zero wait, defeating the back-off. Compute
            // the window in uint64 and clamp the ceiling so it cannot wrap.
            if (retry < 1) retry = 1;
            constexpr uint32_t kMaxRetrySecs = 86400u;  // 24h
            if (retry > kMaxRetrySecs) retry = kMaxRetrySecs;
            const uint64_t retry_ms = static_cast<uint64_t>(retry) * 1000ull;
            // Jitter in [retry, 1.5*retry] so a fleet does not reconnect in
            // lockstep when the window ends.
            std::uniform_int_distribution<uint64_t> mjitter(
                retry_ms, retry_ms + retry_ms / 2ull);
            const uint64_t mdelay = mjitter(rng);
            std::cerr << "[PoolManager] Maintenance window active; waiting ~"
                      << (mdelay / 1000.0)
                      << "s before reconnecting (not counted as a failure)."
                      << std::endl;
            for (uint32_t slept = 0;
                 slept < mdelay &&
                 !supervisor_stop_.load(std::memory_order_acquire) &&
                 maintenance_active_.load(std::memory_order_acquire);
                 slept += 100) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (supervisor_stop_.load(std::memory_order_acquire)) break;
            // Reset the backoff so a post-maintenance failure starts fresh.
            // The normal jitter sleep below is skipped (from_maintenance).
            backoff_ms = kInitialBackoffMs;
        }
        // Receiver has exited. Before driving a reconnect, evaluate churn:
        // if the session we just lost came up post-WORK_ASN but died almost
        // immediately on the SAME work_id, the worker is hot-looping with no
        // useful uptime. Trip the breaker rather than spin. We evaluate this
        // exactly once per lost session by clearing last_session_start after
        // reading it.
        else if (last_session_start.time_since_epoch().count() != 0) {
            const auto session_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - last_session_start)
                    .count();
            uint64_t cur_work_id = 0;
            {
                std::lock_guard<std::mutex> lock(work_mutex_);
                cur_work_id = current_work_.work_id;
            }
            if (session_ms < kSustainedSessionMs &&
                cur_work_id == last_session_work_id) {
                ++churn_count;
            } else {
                // Either the session was sustained or it was on a different
                // work_id: this is normal reconnect territory.
                churn_count = 0;
            }
            last_session_start = {};  // consume; one evaluation per session

            if (churn_count >= kMaxChurnReconnects) {
                std::cerr << "[PoolManager] Reconnect supervisor: "
                          << churn_count
                          << " back-to-back short-lived sessions (< "
                          << (kSustainedSessionMs / 1000)
                          << "s each) on the same work_id="
                          << last_session_work_id
                          << "; the worker keeps getting dropped right after "
                             "AUTH. Backing off hard and giving up to avoid "
                             "hammering the pool. Restart, or check that this "
                             "work_id is not poisoned and the network path is "
                             "stable." << std::endl;
                supervisor_gave_up_.store(true, std::memory_order_release);
                return;
            }
        }

        // Drive a full reconnect.
        if (consecutive_failures >= kMaxReconnectAttempts) {
            std::cerr << "[PoolManager] Reconnect supervisor: "
                      << kMaxReconnectAttempts
                      << " consecutive attempts failed; giving up. "
                      << "The pool client will remain disconnected; "
                      << "host code can re-call connect() to retry."
                      << std::endl;
            supervisor_gave_up_.store(true, std::memory_order_release);
            return;
        }

        // Jitter: pick uniform delay in [backoff/2, backoff]. Skipped when
        // we already waited the maintenance retry window above (otherwise
        // we would double-sleep on top of the operator-suggested back-off).
        if (!from_maintenance) {
            std::uniform_int_distribution<uint32_t> jitter(backoff_ms / 2, backoff_ms);
            const uint32_t delay = jitter(rng);
            std::cerr << "[PoolManager] Reconnect attempt " << (consecutive_failures + 1)
                      << "/" << kMaxReconnectAttempts
                      << " in " << (delay / 1000.0) << "s..." << std::endl;
            // Sleep with periodic wake-on-stop so shutdown is responsive.
            for (uint32_t slept = 0; slept < delay && !supervisor_stop_.load(std::memory_order_acquire);
                 slept += 100)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (supervisor_stop_.load(std::memory_order_acquire)) break;
        }

        reconnect_attempts_.fetch_add(1, std::memory_order_relaxed);

        // Tear down the old client (if still around) and construct a
        // fresh one. Using the public connect() helper would race with
        // ourselves; we follow its body inline.
        // BEFORE destroying the old client, snapshot
        // its dp_sequence_next_ so we don't lose any in-flight counter
        // progress made between the last dp_seq_progress callback and
        // disconnect. Otherwise a flaky network would walk the
        // persisted counter backwards by up to one batch (100 DPs).
        if (client_) {
            if (auto* jlp = dynamic_cast<JLPPoolClient*>(client_.get())) {
                uint64_t snap_work_id = 0;
                uint32_t snap_seq = 0;
                jlp->snapshot_dp_sequence(snap_work_id, snap_seq);
                if (snap_seq > 0) {
                    update_dp_seq(snap_work_id, snap_seq);
                }
                // Roll any sender-shutdown drops into our visible
                // dropped_count_ so operators see a true total even
                // across reconnects.
                dropped_count_.fetch_add(jlp->get_shutdown_drop_count(),
                                         std::memory_order_relaxed);
            }
            client_->disconnect();
            client_.reset();
        }
        connected_.store(false);

        // Recreate the JLP client (same path as PoolManager::connect).
        if (config_.type != POOL_TYPE_JLP) {
            std::cerr << "[PoolManager] Supervisor refusing to reconnect non-JLP "
                      << "config; HTTP path is deleted." << std::endl;
            supervisor_gave_up_.store(true);
            return;
        }
        auto fresh = std::make_unique<JLPPoolClient>();
        fresh->set_timeout(config_.timeout_ms);
        fresh->set_reconnect(config_.auto_reconnect);
        fresh->set_debug_mode(config_.debug_mode);
        fresh->set_use_tls(config_.use_tls);
        fresh->set_verify_cert(config_.verify_cert);
        JLPPoolClient* jlp_raw = fresh.get();
        client_ = std::move(fresh);

        // B1 + Pool-F6: same callback wiring as the initial
        // connect() so seq seeding, WORK_ASN dedup, and on-wire sent
        // counts work identically on reconnect.
        wire_client_callbacks(jlp_raw);
        // B1 wire-v4: re-attach the cached identity on every fresh
        // reconnect-created client so v4 AUTH survives drops.
        apply_worker_identity(jlp_raw);

        // Clear maintenance optimistically before this attempt: if the
        // pool is still in maintenance, the fresh client's AUTH_OK is
        // followed by a MAINTENANCE(active=1) frame whose callback
        // re-asserts the flag (and disconnects us again). If the pool is
        // back up, the flag stays cleared and we resume normally. A
        // request_work() failure caused by that re-assertion must NOT
        // count as a fault, so the failure paths below check
        // maintenance_active_ before bumping consecutive_failures.
        maintenance_active_.store(false, std::memory_order_release);

        if (!client_->connect(config_.host, config_.port)) {
            if (!maintenance_active_.load(std::memory_order_acquire)) {
                ++consecutive_failures;
                backoff_ms = std::min(MAX_RECONNECT_BACKOFF_MS, backoff_ms * 2);
            }
            continue;
        }
        if (!client_->authenticate(config_.worker_name, config_.password)) {
            // Check for IP ban BEFORE disconnecting the client so the
            // ban flag (set by handle_auth_fail) is still readable.
            if (jlp_raw->is_ip_banned()) {
                std::cerr << "[PoolManager] Server has banned this IP. "
                          << "Reconnecting would only extend the ban.\n"
                          << "             " << jlp_raw->ban_reason() << "\n"
                          << "             Giving up. Restart when the ban expires."
                          << std::endl;
                client_->disconnect();
                supervisor_gave_up_.store(true, std::memory_order_release);
                return;
            }
            client_->disconnect();
            ++consecutive_failures;
            ++consecutive_auth_failures;
            // bound consecutive AUTH_FAILs at the
            // supervisor layer (the previous in-receiver-thread cap
            // was dead code and never executed).
            if (consecutive_auth_failures >= MAX_AUTH_FAIL_ATTEMPTS) {
                std::cerr << "[PoolManager] Reached "
                          << MAX_AUTH_FAIL_ATTEMPTS
                          << " consecutive AUTH_FAILs; giving up. "
                             "Check worker name / credentials and "
                             "reconnect manually." << std::endl;
                supervisor_gave_up_.store(true, std::memory_order_release);
                return;
            }
            backoff_ms = std::min(MAX_RECONNECT_BACKOFF_MS, backoff_ms * 2);
            continue;
        }

        // Send WORK_REQ and wait for WORK_ASN before marking the
        // connection live. Without this, the sender thread immediately
        // dequeues buffered DPs (tagged with the old work_id) before
        // the server-side WorkerConnection has current_work set. Every
        // such DP is logged as "submitted without an active work
        // assignment" and counts toward the 100-in-60m invalid-DP ban.
        {
            WorkAssignment fresh_work;
            if (!client_->request_work(fresh_work)) {
                client_->disconnect();
                // A MAINTENANCE(active=1) frame arriving right after
                // AUTH_OK disconnects us mid-WORK_REQ; that is expected,
                // not a fault, so do not bump the failure counters when
                // maintenance is active (the loop top will wait the window).
                if (!maintenance_active_.load(std::memory_order_acquire)) {
                    ++consecutive_failures;
                    backoff_ms =
                        std::min(MAX_RECONNECT_BACKOFF_MS, backoff_ms * 2);
                }
                continue;
            }
            {
                std::lock_guard<std::mutex> lock(work_mutex_);
                current_work_ = fresh_work;
                has_work_     = true;
            }
            current_dp_bits_.store(fresh_work.dp_bits,
                                   std::memory_order_release);
            current_work_id_.store(fresh_work.work_id,
                                   std::memory_order_release);
        }

        connected_.store(true);
        reconnect_successes_.fetch_add(1, std::memory_order_relaxed);
        backoff_ms = kInitialBackoffMs;
        consecutive_failures = 0;
        consecutive_auth_failures = 0;
        // Mark the session live for the churn circuit-breaker. The next
        // time the receiver exits we will measure how long this lasted.
        last_session_start   = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            last_session_work_id = current_work_.work_id;
        }
        std::cout << "[PoolManager] Reconnected to pool as "
                  << config_.worker_name << std::endl;

        // Audit #21 drain: replay DPs the producer buffered while we
        // were mid-reconnect. Lock + swap so the producer is unblocked
        // for the duration of the (potentially large) replay; new DPs
        // arriving during the drain go straight to client_->submit_dp
        // because connected_ is already true at this point.
        std::deque<BufferedDP> to_replay;
        {
            std::lock_guard<std::mutex> lock(reconnect_buf_mutex_);
            to_replay.swap(reconnect_buffer_);
        }
        // The active assignment after the fresh WORK_REQ above. Only DPs
        // computed under this exact work_id can be accepted; replaying a DP
        // tagged with a superseded work_id earns a stale-work_id strike and
        // (after 20) a force-close, so we purge those instead of resubmitting.
        const uint64_t active_work_id =
            current_work_id_.load(std::memory_order_acquire);
        uint64_t replayed_ok = 0;
        uint64_t replayed_drop = 0;
        uint64_t purged_stale = 0;
        for (const auto& entry : to_replay) {
            if (entry.work_id != active_work_id) {
                ++purged_stale;
                continue;
            }
            if (client_->submit_dp(entry.dp)) {
                ++replayed_ok;
            } else {
                ++replayed_drop;
            }
        }
        if (replayed_ok > 0) {
            reconnect_replayed_total_.fetch_add(replayed_ok,
                                                std::memory_order_relaxed);
            enqueued_count_.fetch_add(replayed_ok,
                                      std::memory_order_relaxed);
        }
        if (replayed_drop > 0) {
            dropped_count_.fetch_add(replayed_drop,
                                     std::memory_order_relaxed);
        }
        if (purged_stale > 0) {
            reconnect_purged_total_.fetch_add(purged_stale,
                                              std::memory_order_relaxed);
            dropped_count_.fetch_add(purged_stale,
                                     std::memory_order_relaxed);
        }
        if (!to_replay.empty()) {
            std::cout << "[PoolManager] Replayed "
                      << replayed_ok << " buffered DP(s) post-reconnect"
                      << (replayed_drop > 0
                              ? std::string(" (") +
                                    std::to_string(replayed_drop) +
                                    " could not be requeued)"
                              : std::string{})
                      << (purged_stale > 0
                              ? std::string(" (purged ") +
                                    std::to_string(purged_stale) +
                                    " DP(s) tied to a superseded work_id)"
                              : std::string{})
                      << std::endl;
        }
    }
}

// Parse pool URL
// Supported schemes:
//   jlp://host:port    - JLP without TLS (closed-test only; will warn)
//   jlps://host:port   - JLP with TLS (REQUIRED for any external/Internet pool)
//   host:port          - shorthand, defaults to jlp://
// REJECTED schemes:
//   http://, https://  - HTTP pool transport was removed (silent
//                        credential leak). Hard-fails with a clear
//                        migration message.
bool parse_pool_url(const std::string& url, PoolConfig& config) {
    std::regex url_regex(R"(^(?:([a-z]+)://)?([^:/]+)(?::(\d+))?(/.*)?$)");
    std::smatch match;

    if (!std::regex_match(url, match, url_regex)) {
        std::cerr << "[PoolManager] Malformed pool URL: '" << url << "'"
                  << std::endl;
        return false;
    }

    std::string scheme = match[1].str();
    config.host = match[2].str();
    std::string port_str = match[3].str();
    std::string path = match[4].str();

    // Determine type and TLS from scheme. Undocumented kangaroo:// /
    // kangaroos:// aliases that had no presence in docs, configs, or
    // runtime defaults are not accepted.
    config.use_tls = false;
    if (scheme.empty() || scheme == "jlp") {
        config.type = POOL_TYPE_JLP;
        config.use_tls = false;
        // plaintext warning is printed below once the
        // port has been validated, so the warning includes the actual
        // port the operator will hit. Doing it AFTER port parsing
        // keeps the std::stoi guard in one place.
    } else if (scheme == "jlps") {
        config.type = POOL_TYPE_JLP;
        config.use_tls = true;
    } else if (scheme == "http" || scheme == "https") {
        // HTTP pool path was removed during a security review. The
        // previous client silently downgraded https:// to plaintext TCP,
        // leaking the worker name (= Bitcoin payout address) and any
        // auth token.
        std::cerr << "[PoolManager] HTTP pool deprecated; use jlp:// or jlps:// "
                     "(input was: " << url << ")" << std::endl;
        std::cerr << "[PoolManager]   The HTTP transport was removed because the "
                     "previous implementation leaked credentials." << std::endl;
        std::cerr << "[PoolManager]   Migration: replace 'http://host:port' with "
                     "'jlp://host:17403' or, for TLS, 'jlps://host:17403'."
                  << std::endl;
        return false;
    } else {
        std::cerr << "[PoolManager] Unknown URL scheme: '" << scheme
                  << "' (supported: jlp://, jlps://)" << std::endl;
        return false;
    }

    // Set port with validation. std::stoi may throw on overflow; guard it.
    if (!port_str.empty()) {
        int port_value = 0;
        try {
            port_value = std::stoi(port_str);
        } catch (const std::exception& e) {
            std::cerr << "[PoolManager] Invalid port '" << port_str
                      << "': " << e.what() << std::endl;
            return false;
        }
        if (port_value < 1 || port_value > 65535) {
            std::cerr << "[PoolManager] Invalid port: " << port_value
                      << " (must be 1-65535)" << std::endl;
            return false;
        }
        config.port = static_cast<uint16_t>(port_value);
    } else {
        config.port = PoolConfig::default_port(config.type);
    }

    // Set defaults
    config.auto_reconnect = true;
    config.timeout_ms = 5000;
    config.verify_cert = true;  // Verify CA-signed certificates (Let's Encrypt, etc.)

    // emit the plaintext warning AFTER port validation
    // so the operator sees the real target host:port. This fires
    // before any TCP socket is opened or AUTH credentials hit the wire,
    // giving the operator a chance to abort the run. connect() also
    // calls warn_if_plaintext for callers that bypass parse_pool_url.
    if (!config.use_tls && config.type == POOL_TYPE_JLP) {
        JLPPoolClient::warn_if_plaintext(config.host, config.port);
    }

    return true;
}

void PoolManager::apply_worker_identity(JLPPoolClient* jlp) {
    if (!jlp) return;
    if (config_.worker_key_file.empty()) return;  // v3 path; nothing to do.

    // Load once and cache. Subsequent reconnects re-attach the same
    // shared_ptr so the EC_KEY isn't rebuilt every disconnect.
    if (!worker_identity_) {
        std::ifstream f(config_.worker_key_file);
        if (!f) {
            std::cerr << "[PoolManager] --worker-key: cannot open "
                      << config_.worker_key_file
                      << " (wire-v4 disabled; falling back to v3 AUTH)\n";
            return;
        }
        std::string wif;
        std::getline(f, wif);
        // Strip trailing whitespace / CR.
        while (!wif.empty() && (wif.back() == '\r' || wif.back() == '\n'
                                || wif.back() == ' ' || wif.back() == '\t')) {
            wif.pop_back();
        }
        if (wif.empty()) {
            std::cerr << "[PoolManager] --worker-key: file is empty "
                      << "(wire-v4 disabled)\n";
            return;
        }
        // Mainnet bech32 hrp ("bc"). If the operator is on testnet, a
        // future --worker-key-testnet flag will pass hrp="tb"; until
        // then mainnet is the only deployment scenario.
        auto id = collider::identity::load_from_wif(wif, "bc");
        // Wipe local string copy as soon as it's been consumed.
        std::fill(wif.begin(), wif.end(), '\0');
        if (!id) {
            std::cerr << "[PoolManager] --worker-key: WIF load failed "
                      << "(malformed or uncompressed key); wire-v4 disabled\n";
            return;
        }
        // Operator sanity check: bech32 derived from the WIF MUST match
        // the --worker name. If not, AUTH would fail on the server
        // anyway; surface the mismatch here so the operator notices
        // before connect attempts pile up.
        if (id->bech32_address() != config_.worker_name) {
            std::cerr << "[PoolManager] --worker-key: bech32 derived "
                      << "from key (" << id->bech32_address()
                      << ") does not match --worker (" << config_.worker_name
                      << "); wire-v4 disabled\n";
            return;
        }
        worker_identity_ =
            std::make_shared<collider::identity::WorkerIdentity>(std::move(*id));
        std::cerr << "[PoolManager] wire-v4 signed AUTH enabled "
                  << "(worker=" << config_.worker_name << ")\n";
    }
    jlp->set_worker_identity(worker_identity_);

    // v1.5.5 checkpoint-replay (task #9): enable DP_BATCH_V3 emit only when a
    // worker identity is loaded (= this connection negotiates protocol v4, the
    // same gate send_hello uses for the signed AUTH frame) AND the active GPU
    // backend compiled the per-kangaroo checkpoint capture. The client ANDs
    // this with its own per-DP "has a committable chain" test, so it never
    // sends a fabricated commitment. On a v3 session this method returns early
    // above and the freshly-created client keeps its default (capture
    // unavailable -> V2). Re-applied on every reconnect-created client.
    jlp->set_checkpoint_capture_available(checkpoint_capture_built_ &&
                                          worker_identity_ != nullptr);
}

} // namespace pool
} // namespace collider
