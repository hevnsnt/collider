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
//
// v1.5 (protocol_version=3): adds asymmetric tame/wild fields at the tail
// (kangaroo_type, start_offset_a, start_offset_b). Default kangaroo_type=0
// (= KANG_MODE_BOTH from third_party/RCKangaroo/defs.h) is REJECTED in pool
// mode by CudaRCKangarooBackend::initialize -- the server MUST assign 1
// (TAME_ONLY) or 2 (WILD_ONLY) so the worker never holds both halves of
// the Pollard rho trail data. The pool server is the sole entity that
// sees DPs of both types together; only the server can compute the
// recovered key. See .claude/tasks/v1.5-asymmetric-kangaroo.md.
struct WorkAssignment {
    uint8_t public_key[33];  // Compressed public key (target)
    uint8_t range_start[32]; // Start of search range
    uint8_t range_end[32];   // End of search range
    uint32_t dp_bits;        // Distinguished point bits
    uint64_t work_id;        // Unique work identifier
    std::string puzzle_name; // e.g., "Puzzle #135"
    // v1.5 asymmetric work-assignment fields (wire offsets 109..125).
    // kangaroo_type values: 0=BOTH (illegal in pool mode), 1=TAME_ONLY,
    // 2=WILD_ONLY. Mapped to KANG_MODE_* before being handed to the
    // backend. start_offset_a/_b bind this worker to a disjoint sub-range
    // [a, b) of the assigned chunk; reserved for future per-worker
    // chunk-slicing (currently informational, propagated end-to-end so
    // the server can revoke if the worker drifts outside).
    uint8_t  kangaroo_type   = 0;
    uint64_t start_offset_a  = 0;
    uint64_t start_offset_b  = 0;
};

// In-process pool statistics surfaced to the host (UI, progress display,
// session summary). Field layout matches the wire format from server
// STATS_RSP (36 bytes, '<QIIffQI':
//   total_dps:u64, total_workers:u32, active_workers:u32,
//   dps_per_second:f32, your_share:f32, your_dps:u64, uptime_seconds:u32)
// plus three host-only legacy alias fields that the wire struct never
// carries.
//
// IMPORTANT: this is NOT the wire type. The strictly-packed 36-byte wire
// struct is collider::pool::jlp_wire::PoolStats (jlp_wire_generated.hpp,
// auto-generated from protocol/jlp.yaml; the codegen owns its identifier).
// Two structs with the same simple name living in different namespaces was a
// known foot-gun (R-Q3b audit); this one is renamed to PoolStatsLocal to
// make the local-vs-wire distinction explicit in every reference, even when
// the surrounding code already had `using namespace collider::pool;`
// pulled in.
//
// `your_dps` is server-aggregated across ALL connections sharing the same
// worker name (= Bitcoin payout address) -- so a user running on Mac +
// Windows + Linux with the same address sees a unified per-worker total
// rather than per-machine subtotals.
struct PoolStatsLocal {
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
    virtual PoolStatsLocal get_stats() = 0;

    // v1.5: client-to-server report_solution() was DELETED.
    // No code path on the worker may possess the puzzle's private key;
    // the pool server is the sole key-computer (collision_detector in
    // collision-protocol). The SOLUTION wire message is now strictly
    // server-to-client, used by the server to broadcast "the pool solved
    // it; stop grinding" -- the broadcast payload is NOT stored on the
    // worker and never leaves the SolutionCallback invocation site.
    // See .claude/tasks/v1.5-asymmetric-kangaroo.md.

    // Callbacks
    // v1.5: SolutionCallback fires only on the server-to-client SOLUTION
    // BROADCAST. The 32-byte payload is the pool-server-computed recovered
    // key (server publishes it for transparency / audit); the worker
    // treats it as opaque stop-signal metadata and MUST NOT persist,
    // forward, or display it in copy-paste-friendly form. The parameter
    // is named `solution_payload` (not `private_key`) to keep that
    // contract in callers' line of sight.
    using SolutionCallback = std::function<void(const uint8_t* solution_payload)>;
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
