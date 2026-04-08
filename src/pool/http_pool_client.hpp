// http_pool_client.hpp - HTTP/REST based pool client for modern pools
// Simpler API-based protocol for pools that use REST endpoints

#pragma once

#include "pool_client.hpp"
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    typedef SOCKET socket_t;
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    typedef int socket_t;
#endif

namespace collider {
namespace pool {

// HTTP Pool API endpoints (configurable)
struct HTTPPoolConfig {
    std::string base_url;           // e.g., "https://pool.example.com/api/v1"
    std::string auth_endpoint;      // e.g., "/auth"
    std::string work_endpoint;      // e.g., "/work"
    std::string submit_endpoint;    // e.g., "/submit"
    std::string stats_endpoint;     // e.g., "/stats"
    std::string solution_endpoint;  // e.g., "/solution"

    // Default configuration
    static HTTPPoolConfig defaults() {
        return {
            "",
            "/auth",
            "/work",
            "/submit",
            "/stats",
            "/solution"
        };
    }
};

class HTTPPoolClient : public PoolClient {
public:
    HTTPPoolClient();
    ~HTTPPoolClient() override;

    // Configuration
    void set_config(const HTTPPoolConfig& config) { config_ = config; }
    HTTPPoolConfig& config() { return config_; }

    // PoolClient interface
    bool connect(const std::string& host, uint16_t port) override;
    void disconnect() override;
    bool is_connected() const override;

    bool authenticate(const std::string& worker_name,
                     const std::string& password = "") override;

    bool request_work(WorkAssignment& work) override;
    bool submit_dp(const DistinguishedPoint& dp) override;
    bool submit_dps(const std::vector<DistinguishedPoint>& dps) override;

    PoolStats get_stats() override;
    bool report_solution(const uint8_t* private_key) override;

    void set_solution_callback(SolutionCallback cb) override;
    void set_work_callback(WorkCallback cb) override;

    std::string get_pool_type() const override { return POOL_TYPE_HTTP; }

    // HTTP-specific
    void set_api_key(const std::string& key) { api_key_ = key; }
    void set_use_ssl(bool use_ssl) { use_ssl_ = use_ssl; }

private:
    HTTPPoolConfig config_;
    std::string host_;
    uint16_t port_;
    bool use_ssl_;
    std::atomic<bool> connected_;
    std::atomic<bool> running_;

    // Authentication
    std::string worker_name_;
    std::string api_key_;
    std::string auth_token_;

    // Statistics
    PoolStats stats_;
    std::mutex stats_mutex_;

    // Callbacks
    SolutionCallback solution_callback_;
    WorkCallback work_callback_;

    // DP batching
    std::queue<DistinguishedPoint> dp_queue_;
    std::mutex dp_mutex_;
    std::condition_variable dp_cv_;
    std::thread sender_thread_;
    void sender_loop();

    // Polling thread
    std::thread poll_thread_;
    void poll_loop();

    // HTTP helpers
    std::string http_get(const std::string& endpoint);
    std::string http_post(const std::string& endpoint, const std::string& body);
    std::string make_request(const std::string& method, const std::string& endpoint,
                            const std::string& body = "");

    // JSON helpers (minimal implementation)
    std::string to_json_dp(const DistinguishedPoint& dp);
    std::string to_json_dps(const std::vector<DistinguishedPoint>& dps);
    bool parse_work_json(const std::string& json, WorkAssignment& work);
    bool parse_stats_json(const std::string& json, PoolStats& stats);

    // Hex encoding
    static std::string to_hex(const uint8_t* data, size_t len);
    static bool from_hex(const std::string& hex, uint8_t* data, size_t len);
};

} // namespace pool
} // namespace collider
