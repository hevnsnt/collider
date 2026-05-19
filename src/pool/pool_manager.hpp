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

    // Default port by type. Only JLP is supported; HTTP path was deleted.
    static uint16_t default_port(const std::string& type) {
        if (type == POOL_TYPE_JLP) return 17403;
        return 17403;
    }
};

// Callback when solution is found
using SolutionFoundCallback = std::function<void(const uint8_t* private_key, const std::string& worker)>;

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

    // Solution reporting
    void report_solution(const uint8_t* private_key);

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

private:
    PoolConfig config_;
    std::unique_ptr<PoolClient> client_;
    std::atomic<bool> connected_;
    // split counters.
    std::atomic<uint64_t> enqueued_count_{0};
    std::atomic<uint64_t> sent_count_{0};
    std::atomic<uint64_t> dropped_count_{0};
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

    static constexpr uint32_t kMaxReconnectAttempts = 16;
    static constexpr uint32_t kInitialBackoffMs     = 1000;
    // The cap on reconnect backoff lives in pool_config.hpp as
    // MAX_RECONNECT_BACKOFF_MS, shared with JLPPoolClient so the two
    // reconnect paths can never disagree on the upper bound.

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
