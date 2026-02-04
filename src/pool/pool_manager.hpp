// pool_manager.hpp - Manages pool connections and integrates with RCKangaroo
// Provides high-level API for pool-based solving

#pragma once

#include "pool_client.hpp"
#include "jlp_pool_client.hpp"
#include "http_pool_client.hpp"
#include <memory>
#include <atomic>
#include <thread>
#include <functional>

namespace collider {
namespace pool {

// Pool connection configuration
struct PoolConfig {
    std::string type;        // "jlp", "http", "websocket"
    std::string host;
    uint16_t port;
    std::string worker_name; // Bitcoin address
    std::string password;    // Optional
    std::string api_key;     // For HTTP pools
    bool auto_reconnect;
    uint32_t timeout_ms;
    bool debug_mode = false; // Show debug output
    bool use_tls = false;    // Use TLS encryption
    bool verify_cert = false; // Verify TLS certificate (false for self-signed)

    // Default port by type
    static uint16_t default_port(const std::string& type) {
        if (type == POOL_TYPE_JLP) return 17403;
        if (type == POOL_TYPE_HTTP) return 80;
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
    PoolStats get_stats() const;
    uint64_t get_submitted_count() const { return submitted_count_; }
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

private:
    PoolConfig config_;
    std::unique_ptr<PoolClient> client_;
    std::atomic<bool> connected_;
    std::atomic<uint64_t> submitted_count_;
    std::chrono::steady_clock::time_point start_time_;
    SolutionFoundCallback solution_callback_;

    // Current work
    WorkAssignment current_work_;
    std::mutex work_mutex_;
    bool has_work_;
};

// Global pool manager instance for easy access from callbacks
PoolManager& get_pool_manager();

// Helper to parse pool URL
// Format: jlp://host:port or http://host:port/path
bool parse_pool_url(const std::string& url, PoolConfig& config);

} // namespace pool
} // namespace collider
