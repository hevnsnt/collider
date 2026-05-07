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

// Pool statistics. Wire format from server STATS_RSP (36 bytes, '<QIIffQI'):
//   total_dps:u64, total_workers:u32, active_workers:u32,
//   dps_per_second:f32, your_share:f32, your_dps:u64, uptime_seconds:u32
//
// `your_dps` is server-aggregated across ALL connections sharing the same
// worker name (= Bitcoin payout address) -- so a user running on Mac +
// Windows + Linux with the same address sees a unified per-worker total
// rather than per-machine subtotals.
struct PoolStats {
    uint64_t total_dps;           // Total DPs in pool (all workers, all time)
    uint32_t total_workers;       // Total registered workers (all-time)
    uint32_t active_workers;      // Currently connected workers
    float    dps_per_second;      // Aggregate pool DP rate
    float    your_share;          // Server-computed share fraction (0.0 - 1.0)
    uint64_t your_dps;            // YOUR aggregate DPs across all machines
    uint32_t uptime_seconds;      // Pool server uptime
    // Legacy aliases kept for code that still reads them; populated from
    // the wire fields above by handle_server_message.
    uint64_t connected_workers = 0;  // == active_workers
    uint64_t pool_speed = 0;         // keys/s, computed from dps_per_second * 2^dp_bits
    std::string status;
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
// Wave 4 security review: only POOL_TYPE_JLP is supported. POOL_TYPE_HTTP is kept
// solely so pool_manager can detect and reject legacy http:// configs with a
// migration message. POOL_TYPE_WS was never implemented.
constexpr const char* POOL_TYPE_JLP = "jlp";              // JeanLucPons compatible (only supported)
constexpr const char* POOL_TYPE_HTTP = "http";            // DEPRECATED: rejected at parse time (D-C1)
constexpr const char* POOL_TYPE_WS = "websocket";         // DEPRECATED: never implemented

} // namespace pool
} // namespace collider
