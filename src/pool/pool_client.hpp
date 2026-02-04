// pool_client.hpp - Abstract pool client interface for distributed Kangaroo solving
// theCollider - GPU-accelerated Bitcoin puzzle solver

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>
#include <atomic>

namespace collider {
namespace pool {

// Distinguished Point data structure
struct DistinguishedPoint {
    uint8_t x[32];           // X coordinate of the point (compressed)
    uint8_t d[32];           // Distance traveled (private key offset)
    uint8_t type;            // 0 = tame, 1 = wild
    uint64_t dp_bits;        // Number of DP bits used

    // Serialize for network transmission
    std::vector<uint8_t> serialize() const;
    static DistinguishedPoint deserialize(const uint8_t* data, size_t len);
};

// Work assignment from pool
struct WorkAssignment {
    uint8_t public_key[33];  // Compressed public key (target)
    uint8_t range_start[32]; // Start of search range
    uint8_t range_end[32];   // End of search range
    uint32_t dp_bits;        // Distinguished point bits
    uint64_t work_id;        // Unique work identifier
    std::string puzzle_name; // e.g., "Puzzle #135"
};

// Pool statistics
struct PoolStats {
    uint64_t total_dps;           // Total DPs in pool
    uint64_t your_dps;            // Your contributed DPs
    double your_share;            // Your percentage of work
    uint64_t connected_workers;   // Number of connected workers
    double pool_speed;            // Pool total speed (keys/s)
    std::string status;           // Pool status message
};

// Pool client interface
class PoolClient {
public:
    virtual ~PoolClient() = default;

    // Connection management
    virtual bool connect(const std::string& host, uint16_t port) = 0;
    virtual void disconnect() = 0;
    virtual bool is_connected() const = 0;

    // Authentication (if required)
    virtual bool authenticate(const std::string& worker_name,
                             const std::string& password = "") = 0;

    // Work management
    virtual bool request_work(WorkAssignment& work) = 0;
    virtual bool submit_dp(const DistinguishedPoint& dp) = 0;
    virtual bool submit_dps(const std::vector<DistinguishedPoint>& dps) = 0;

    // Statistics
    virtual PoolStats get_stats() = 0;

    // Solution notification (called when key is found)
    virtual bool report_solution(const uint8_t* private_key) = 0;

    // Callbacks
    using SolutionCallback = std::function<void(const uint8_t* private_key)>;
    using WorkCallback = std::function<void(const WorkAssignment& work)>;

    virtual void set_solution_callback(SolutionCallback cb) = 0;
    virtual void set_work_callback(WorkCallback cb) = 0;

    // Pool type identification
    virtual std::string get_pool_type() const = 0;
};

// Factory function to create pool clients
std::unique_ptr<PoolClient> create_pool_client(const std::string& type);

// Pool client types
constexpr const char* POOL_TYPE_JLP = "jlp";      // JeanLucPons compatible
constexpr const char* POOL_TYPE_HTTP = "http";    // REST API based
constexpr const char* POOL_TYPE_WS = "websocket"; // WebSocket based

} // namespace pool
} // namespace collider
