// pool_manager.hpp - Manages pool connections and integrates with RCKangaroo
// Provides high-level API for pool-based solving

#pragma once

#include "pool_client.hpp"
#include "jlp_pool_client.hpp"
// http_pool_client.hpp removed: HTTP pool path was deleted in Wave 4 due to D-C1
// (silent https:// to plaintext downgrade leaking credentials). JLP+TLS (jlps://) is
// the only supported pool transport going forward.
#include <memory>
#include <atomic>
#include <thread>
#include <functional>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <string>
#include <limits>

namespace collider {
namespace pool {

// Pool connection configuration
// NOTE: HTTP pool support was REMOVED in Wave 4 (Track D security audit).
// `type` must be "jlp" only. http:// / https:// URLs are rejected at parse time.
struct PoolConfig {
    std::string type;        // "jlp" (only supported value; "http"/"websocket" deprecated and rejected)
    std::string host;
    uint16_t port;
    std::string worker_name; // Bitcoin address
    std::string password;    // Optional (currently unused by JLP protocol; reserved)
    std::string api_key;     // (DEPRECATED) was for HTTP pools; no longer used
    bool auto_reconnect;
    uint32_t timeout_ms;
    bool debug_mode = false; // Show debug output
    bool use_tls = false;    // Use TLS encryption (jlps://)
    bool verify_cert = true; // Verify TLS certificate (default true; opt-out only via explicit flag)

    // B1 wire-v4: when non-empty, PoolManager loads the WIF from this
    // file at first connect and the JLPPoolClient (every fresh instance
    // including reconnects) gets set_worker_identity() called with the
    // shared identity. Opt-in: empty path keeps the v3 wire path.
    std::string worker_key_file;

    // Default port by type. Only JLP is supported; HTTP path was deleted.
    static uint16_t default_port(const std::string& type) {
        if (type == POOL_TYPE_JLP) return 17403;
        return 17403;
    }
};

// v1.5: Callback fired on the SOLUTION server-to-client broadcast. The
// 32-byte solution_payload is the pool-server-computed recovered key
// (server publishes for transparency); the worker treats it as opaque
// stop-signal metadata. Caller is responsible for not persisting or
// echoing it -- see pool_solver.cpp run_pool_mode's set_solution_callback
// for the v1.5-compliant handler.
using SolutionFoundCallback = std::function<void(const uint8_t* solution_payload, const std::string& worker)>;

class PoolManager {
public:
    PoolManager();
    ~PoolManager();

    // Configuration
    void set_config(const PoolConfig& config);
    PoolConfig& config() { return config_; }

    // Connection
    bool connect();
    void disconnect();
    bool is_connected() const;

    // Work management
    bool get_work(WorkAssignment& work);

    // DP submission (called from Kangaroo solver)
    void submit_dp(const uint8_t* x, const uint8_t* d, uint8_t type, uint32_t dp_bits);
    // v1.5.5 (task #9): chain-carrying overload. ckpt_distances / ckpt_l1s2 are
    // the COMMITTABLE checkpoint chain for the kangaroo that produced this DP
    // (ordered 32-byte big-endian distances mod n + per-checkpoint L1S2 bits),
    // read back at GPU harvest. Both nullptr (or a chain shorter than 2
    // checkpoints) means "no commitment" -> the DP ships as DP_BATCH_V2. The
    // pointers are copied (not retained) onto the queued DistinguishedPoint.
    void submit_dp(const uint8_t* x, const uint8_t* d, uint8_t type,
                   uint32_t dp_bits,
                   const std::vector<std::array<uint8_t, 32>>* ckpt_distances,
                   const std::vector<uint8_t>* ckpt_l1s2);
    void submit_dp(const DistinguishedPoint& dp);

    // Statistics
    PoolStatsLocal get_stats() const;

    // pre-fix `submitted_count_` was bumped on enqueue,
    // not on actual wire send. DPs enqueued before AUTH_OK or refused
    // by the JLP client's queue-full backpressure were counted as
    // "submitted" even though they never hit the network. We now keep
    // two counters and expose both:
    //   enqueued_count_ = DPs accepted by submit_dp() and pushed into
    //                     the JLP client's bounded queue.
    //   sent_count_     = DPs that left the wire as part of a
    //                     DP_BATCH_V2 the server acknowledged with
    //                     a successful send.
    // Pre-v1.4.2-quality there was a third method, get_submitted_count(),
    // that aliased get_enqueued_count() for backward compatibility. It
    // was deleted in R-Q3c because the name kept misleading callers into
    // using it for "sent on wire" metrics; the two existing meaningful
    // counters (enqueued, sent) are the only API.
    uint64_t get_enqueued_count() const { return enqueued_count_.load(); }
    uint64_t get_sent_count() const { return sent_count_.load(); }

    // Returns the count of DPs the manager refused to forward to the
    // pool: queue-full backpressure from the JLP client or attempts to
    // submit while disconnected. A non-zero value with a steady rate
    // means MAX_DP_QUEUE_SIZE in jlp_pool_client.hpp is too small for
    // the workload, or the network sender is starved.
    uint64_t get_dropped_count() const { return dropped_count_.load(); }
    double get_submission_rate() const;

    // v1.5: client-to-server solution reporting was DELETED. The pool
    // server (collision-protocol) is the sole entity that computes the
    // recovered private key; the worker never holds it. The SOLUTION
    // wire message is now strictly server-to-client (broadcast on solve)
    // and the SolutionFoundCallback below fires only on the receive
    // path -- the caller treats the payload as a stop signal, not as a
    // key to store. See .claude/tasks/v1.5-asymmetric-kangaroo.md.

    // Callbacks
    void set_solution_callback(SolutionFoundCallback cb) { solution_callback_ = cb; }

    // Status
    std::string get_status_string() const;

    // For RCKangaroo integration - DP hook
    // Call this from the DP callback in RCKangaroo
    static void dp_callback_hook(void* user_data, const uint8_t* x, const uint8_t* d, uint8_t type);

    // Reconnection diagnostics. Counters are updated by the supervisor
    // thread; readers must tolerate concurrent writes (atomics).
    uint32_t reconnect_attempts() const { return reconnect_attempts_.load(); }
    uint32_t reconnect_successes() const { return reconnect_successes_.load(); }
    bool reconnect_supervisor_gave_up() const { return supervisor_gave_up_.load(); }

    // Reconnect-window DP buffer diagnostics. buffered = DPs held while
    // disconnected; replayed = buffered DPs resubmitted into the new client
    // post-reconnect (work_id still matched the active assignment); purged =
    // buffered DPs DROPPED at drain time because their work_id no longer
    // matched (resubmitting would earn server-side stale-work_id strikes).
    // All three are cumulative across the process lifetime.
    uint64_t reconnect_buffered_total() const {
        return reconnect_buffered_total_.load(std::memory_order_relaxed);
    }
    uint64_t reconnect_replayed_total() const {
        return reconnect_replayed_total_.load(std::memory_order_relaxed);
    }
    uint64_t reconnect_purged_total() const {
        return reconnect_purged_total_.load(std::memory_order_relaxed);
    }

    // v1.5.4 maintenance status (surfaced to the host so the TUI / logs
    // can tell the operator the pool asked everyone to back off). The
    // flag is set when a MAINTENANCE(active=1) frame arrives and cleared
    // on the next successful AUTH that does not re-assert maintenance.
    bool is_in_maintenance() const {
        return maintenance_active_.load(std::memory_order_acquire);
    }
    std::string maintenance_message() const {
        std::lock_guard<std::mutex> lock(maintenance_mutex_);
        return maintenance_message_;
    }

    // v1.5.5 checkpoint-replay (task #9). The runtime driver (pool_solver)
    // reports whether the active GPU backend compiled the per-kangaroo
    // checkpoint capture (RCKangarooManager::checkpoint_capture_built()).
    // PoolManager stores it and applies it to every client it creates (initial
    // connect + every supervisor reconnect) inside apply_worker_identity, ANDed
    // with whether a worker identity is loaded (= negotiated protocol v4), so a
    // reconnect-created client re-enables V3 emit consistently. Set BEFORE
    // connect(); a later change only affects subsequently-created clients.
    // Default false keeps the client on DP_BATCH_V2 (no commitment).
    void set_checkpoint_capture_built(bool built) {
        checkpoint_capture_built_ = built;
    }

    // v1.5.4: surface the update advert parsed from the current client's
    // AUTH_OK so the host (run_pool_mode) can decide whether to self-update
    // after the first successful authentication. Returns a not-present
    // advert when there is no JLP client or no advert was parsed.
    JLPPoolClient::UpdateAdvert get_update_advert() const;

private:
    PoolConfig config_;
    std::unique_ptr<PoolClient> client_;
    std::atomic<bool> connected_;
    // split counters.
    std::atomic<uint64_t> enqueued_count_{0};
    std::atomic<uint64_t> sent_count_{0};
    std::atomic<uint64_t> dropped_count_{0};

    // Reconnect-window DP buffer: holds DPs submitted while the
    // supervisor is mid-reconnect (between connected_.store(false) and
    // the new client's AUTH_OK + WORK_REQ success). Without this
    // buffer, every DP in that window was counted as dropped (audit
    // finding #21). On successful reconnect we replay the buffered DPs
    // into the new client's queue. Bounded to keep memory usage
    // predictable when an outage is long. Replays use atomic_swap
    // semantics so the producer side never blocks on the drain side.
    static constexpr size_t kReconnectBufferCap = 10000;
    std::mutex reconnect_buf_mutex_;
    // Each buffered DP is tagged with the work_id it was computed under.
    // On reconnect we replay only DPs whose work_id still matches the
    // active assignment and DROP the rest: a DP for a chunk we no longer
    // own can never be accepted, and resubmitting it only earns the
    // server's stale-work_id strikes (which escalate to a force-close +
    // reconnect loop -- the v1.5.4 post-restart stranding bug). The bare
    // DistinguishedPoint carries no work_id, so the tag lives alongside it
    // here rather than on the wire struct.
    struct BufferedDP {
        DistinguishedPoint dp;
        uint64_t work_id;
    };
    std::deque<BufferedDP> reconnect_buffer_;
    std::atomic<uint64_t> reconnect_buffered_total_{0};
    std::atomic<uint64_t> reconnect_replayed_total_{0};
    // Count of buffered DPs purged at drain time because their work_id no
    // longer matched the active assignment (operator-visible diagnostics).
    std::atomic<uint64_t> reconnect_purged_total_{0};
    // Lock-free snapshot of the currently-assigned work_id, updated every
    // time current_work_ changes (work_callback + reconnect WORK_ASN).
    // submit_dp() reads this on the producer hot path to tag buffered DPs
    // without contending for work_mutex_. 0 = no assignment yet.
    std::atomic<uint64_t> current_work_id_{0};
    std::chrono::steady_clock::time_point start_time_;
    SolutionFoundCallback solution_callback_;

    // PoolManager owns the per-(worker, work_id) DP
    // sequence counter so a supervisor-driven reconnect that recreates
    // the JLPPoolClient does NOT reset the counter to 0 and trip the
    // server's _dp_seq_high replay-defence watermark. Key format is
    // "worker_name|work_id" so a different worker (e.g. an operator
    // switches payout addresses mid-session) gets a fresh counter.
    // Persisted to ~/.collider/pool_dp_seq.dat on disconnect; loaded
    // on construction.
    mutable std::mutex dp_seq_mutex_;
    std::unordered_map<std::string, uint32_t> dp_seq_map_;
    void load_dp_seq_map();
    void persist_dp_seq_map() const;
    std::string dp_seq_key(uint64_t work_id) const;  // uses config_.worker_name
    uint32_t get_or_create_dp_seq(uint64_t work_id);
    void update_dp_seq(uint64_t work_id, uint32_t next_seq);
    void reset_dp_seq(uint64_t work_id);
    void wire_client_callbacks(JLPPoolClient* jlp);  // helper for connect / reconnect

    // B1 wire-v4: lazy-loaded shared worker identity. Loaded the first
    // time apply_worker_identity() is called (typically from connect()
    // before the first JLPPoolClient::set_worker_identity), reused for
    // every subsequent reconnect. nullptr means no --worker-key was
    // configured (legacy v3 wire path).
    std::shared_ptr<collider::identity::WorkerIdentity> worker_identity_;
    void apply_worker_identity(JLPPoolClient* jlp);  // load + attach

    // v1.5.5 (task #9): set by the runtime driver from the active backend's
    // RCKangarooManager::checkpoint_capture_built(). apply_worker_identity ANDs
    // it with "v4 identity loaded" to enable the client's DP_BATCH_V3 emit.
    bool checkpoint_capture_built_ = false;

    // WORK_ASN dedup state lives at the manager level
    // because it survives client recreation across reconnects. The
    // wire-level callback always fires; the manager filters duplicates
    // before surfacing to the host.
    std::atomic<uint64_t> last_work_id_seen_{std::numeric_limits<uint64_t>::max()};

    // Current work. dp_bits is also exposed via an atomic snapshot
    // (current_dp_bits_) so dp_callback_hook can read it without
    // contending for work_mutex_ on the kernel-callback hot path.
    WorkAssignment current_work_;
    std::mutex work_mutex_;
    bool has_work_;
    std::atomic<uint32_t> current_dp_bits_{0};

    // ---- Reconnect supervisor ----------------------------------------
    // Background thread that watches client_->is_connected() and drives
    // reconnect + reauth with bounded jittered backoff when the receiver
    // exits (transient network blip, server restart, etc.). The receiver
    // loop's previous "external supervisor must call connect()/
    // authenticate()" message had no actual supervisor; this is it.
    std::thread             supervisor_thread_;
    std::atomic<bool>       supervisor_stop_{false};
    std::mutex              supervisor_mutex_;
    std::condition_variable supervisor_cv_;
    std::atomic<uint32_t>   reconnect_attempts_{0};
    std::atomic<uint32_t>   reconnect_successes_{0};
    std::atomic<bool>       supervisor_gave_up_{false};

    // v1.5.4 maintenance back-off state. Set by the MaintenanceCallback
    // the manager registers on every fresh client (wire_client_callbacks).
    // When active, supervisor_loop uses maintenance_retry_secs_ (jittered)
    // as the reconnect wait instead of the exponential backoff, and does
    // NOT count the lost session as a failure or a churn event (a
    // maintenance window is expected, not a fault). maintenance_active_ is
    // atomic so is_in_maintenance() and the supervisor can read it without
    // the mutex; the message string is guarded by maintenance_mutex_.
    std::atomic<bool>       maintenance_active_{false};
    std::atomic<uint32_t>   maintenance_retry_secs_{0};
    mutable std::mutex      maintenance_mutex_;
    std::string             maintenance_message_;

    static constexpr uint32_t kMaxReconnectAttempts = 16;
    static constexpr uint32_t kInitialBackoffMs     = 1000;
    // The cap on reconnect backoff lives in pool_config.hpp as
    // MAX_RECONNECT_BACKOFF_MS, shared with JLPPoolClient so the two
    // reconnect paths can never disagree on the upper bound.

    // Reconnect-churn circuit-breaker. A session that passes AUTH +
    // WORK_ASN resets consecutive_failures to 0, so a worker that keeps
    // getting dropped IMMEDIATELY after auth (server flapping it, a poison
    // work_id, a one-way-reachability NAT issue) would otherwise hot-loop
    // forever: connect, auth, get dropped in under a second, repeat, with
    // the failure counter never climbing. We treat a successful session
    // that lasted less than kSustainedSessionMs as "not sustained" and, if
    // kMaxChurnReconnects of those happen back-to-back on the SAME work_id,
    // trip supervisor_gave_up_ instead of spinning. Genuine transient
    // drops (sessions that DID sustain, or a new work_id) reset the churn
    // counter so normal reconnect behavior is preserved.
    static constexpr uint32_t kMaxChurnReconnects = 5;
    static constexpr uint32_t kSustainedSessionMs = 30'000;

    void start_supervisor();
    void stop_supervisor();
    void supervisor_loop();
};

// Global pool manager instance for easy access from callbacks
PoolManager& get_pool_manager();

// Helper to parse pool URL
// Format: jlp://host:port or http://host:port/path
bool parse_pool_url(const std::string& url, PoolConfig& config);

} // namespace pool
} // namespace collider
