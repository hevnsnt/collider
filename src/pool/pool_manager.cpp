// pool_manager.cpp - Pool manager implementation

#include "pool_manager.hpp"

#include "core/paths.hpp"
#include "core/secure_write.hpp"

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
    // Solution callback: existing behavior (route to host solution_callback_).
    jlp->set_solution_callback([this](const uint8_t* key) {
        std::cout << "\n[PoolManager] SOLUTION FOUND BY POOL!" << std::endl;
        if (solution_callback_) {
            solution_callback_(key, config_.worker_name);
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
    DistinguishedPoint dp;
    memcpy(dp.x, x, 32);
    memcpy(dp.d, d, 32);
    dp.type = type;
    dp.dp_bits = dp_bits;
    submit_dp(dp);
}

void PoolManager::submit_dp(const DistinguishedPoint& dp) {
    if (!is_connected()) {
        // Queue is decoupled from connection state, but we should not
        // accept DPs while disconnected -- the supervisor may take a
        // while to come back, by which point the work_id might have
        // rolled and these DPs are stale.
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

void PoolManager::report_solution(const uint8_t* private_key) {
    if (!is_connected()) {
        return;
    }

    std::cout << "[PoolManager] Reporting solution to pool..." << std::endl;
    client_->report_solution(private_key);
}

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

        // Receiver has exited. Drive a full reconnect.
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

        // Jitter: pick uniform delay in [backoff/2, backoff].
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

        if (!client_->connect(config_.host, config_.port)) {
            ++consecutive_failures;
            backoff_ms = std::min(MAX_RECONNECT_BACKOFF_MS, backoff_ms * 2);
            continue;
        }
        if (!client_->authenticate(config_.worker_name, config_.password)) {
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

        connected_.store(true);
        reconnect_successes_.fetch_add(1, std::memory_order_relaxed);
        backoff_ms = kInitialBackoffMs;
        consecutive_failures = 0;
        consecutive_auth_failures = 0;
        std::cout << "[PoolManager] Reconnected to pool as "
                  << config_.worker_name << std::endl;
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

} // namespace pool
} // namespace collider
