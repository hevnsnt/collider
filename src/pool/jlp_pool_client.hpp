// jlp_pool_client.hpp - JeanLucPons Kangaroo protocol compatible pool client
// Compatible with pools running JLP Kangaroo server (port 17403)

#pragma once

// Windows: Must define NOMINMAX before any includes to prevent min/max macro conflicts
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
#endif

#include "pool_client.hpp"
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>

// TLS support via OpenSSL
#ifdef COLLIDER_HAS_OPENSSL
    #include <openssl/ssl.h>
    #include <openssl/err.h>
#endif

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET socket_t;
    #define INVALID_SOCK INVALID_SOCKET
    #define SOCK_ERROR SOCKET_ERROR
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    typedef int socket_t;
    #define INVALID_SOCK -1
    #define SOCK_ERROR -1
    #define closesocket close
#endif

namespace collider {
namespace pool {

// JLP Protocol message types - must match collision-protocol server
enum class JLPMessageType : uint8_t {
    // Authentication
    AUTH      = 0x01,
    AUTH_OK   = 0x02,
    AUTH_FAIL = 0x03,

    // Work distribution
    WORK_REQ  = 0x10,
    WORK_ASN  = 0x11,

    // Distinguished points
    DP_SUBMIT = 0x20,
    DP_ACK    = 0x21,
    DP_BATCH  = 0x22,

    // Statistics
    STATS_REQ = 0x30,
    STATS_RSP = 0x31,

    // Solution
    SOLUTION  = 0x40,

    // Keepalive
    PING      = 0x50,
    PONG      = 0x51,

    // Error (named MSG_ERROR to avoid Windows ERROR macro conflict)
    MSG_ERROR = 0xFF
};

// JLP Protocol header - MUST match server format exactly:
// Python: struct.pack('<4sBBH', b'KANG', msg_type, flags, len(payload))
// [MAGIC:4][TYPE:1][FLAGS:1][LENGTH:2][PAYLOAD]
#pragma pack(push, 1)
struct JLPHeader {
    uint8_t magic[4];        // "KANG" (4 bytes)
    uint8_t type;            // Message type (1 byte)
    uint8_t flags;           // Flags, typically 0 (1 byte)
    uint16_t payload_size;   // Size of payload (2 bytes, little-endian)
};  // Total: 8 bytes

struct JLPClientHello {
    char worker_name[64];    // Worker identifier (Bitcoin address)
    uint32_t gpu_count;      // Number of GPUs
    uint64_t speed;          // Keys per second capability
};

// Work assignment structure - must match collision-protocol/src/jlp_protocol.py ServerConfig
// Python: struct.pack('<33s32s32sIQ', public_key, range_start, range_end, dp_bits, work_id)
struct JLPServerConfig {
    uint8_t public_key[33];  // 33 bytes - Compressed public key
    uint8_t range_start[32]; // 32 bytes - Range start (big-endian)
    uint8_t range_end[32];   // 32 bytes - Range end (big-endian)
    uint32_t dp_bits;        // 4 bytes - DP bits (little-endian)
    uint64_t work_id;        // 8 bytes - Work identifier (little-endian)
    // Total: 109 bytes
};

struct JLPDistinguishedPoint {
    uint8_t x[32];           // X coordinate
    uint8_t d[32];           // Distance
    uint8_t type;            // Tame (0) or Wild (1)
    uint8_t dp_bits;         // Number of leading-zero bits used (matches server)
};
// Wire format must be 66 bytes to match collision-protocol/src/jlp_protocol.py
// DistinguishedPoint.to_bytes() (struct.pack('<32s32sBB', x, d, type, dp_bits)).
// DP_BATCH payload is then [count:u32 LE][dp1:66][dp2:66]...
static_assert(sizeof(JLPDistinguishedPoint) == 66,
              "JLPDistinguishedPoint must be 66 bytes on the wire");
#pragma pack(pop)

// Connection / authentication state machine.
// Wave 4 (Track D D-H4): the receiver MUST gate work-affecting messages on
// AUTH_OK so a malicious or misbehaving server cannot inject WORK_ASN /
// SOLUTION / DP_ACK / STATS_RSP before the client has authenticated.
enum class AuthState : uint8_t {
    DISCONNECTED = 0,  // Not connected (initial / after disconnect)
    CONNECTING   = 1,  // TCP/TLS handshake in progress
    AUTH_SENT    = 2,  // AUTH message sent, waiting for AUTH_OK / AUTH_FAIL
    AUTH_OK      = 3,  // Authentication accepted; full message dispatch enabled
    AUTH_FAILED  = 4   // Authentication rejected; do not auto-retry indefinitely
};

class JLPPoolClient : public PoolClient {
public:
    // Queue limits to prevent unbounded memory growth
    static constexpr size_t MAX_DP_QUEUE_SIZE = 100000;  // ~6.5MB of DPs max
    static constexpr uint8_t SUPPORTED_PROTOCOL_VERSION = 1;

    // Wave 4 (Track D D-H5): bound the number of consecutive reconnect attempts
    // that hit AUTH_FAIL. Without this, one bad credential keeps hammering the
    // pool with the same worker name forever, looking like credential stuffing.
    static constexpr uint32_t MAX_AUTH_FAIL_ATTEMPTS = 3;

    // Wave 4 (Track D D-M5): bound the time we wait for an AUTH_OK / AUTH_FAIL
    // response after sending the AUTH message.
    static constexpr uint32_t AUTH_RESPONSE_TIMEOUT_MS = 10000;  // 10s

    JLPPoolClient();
    ~JLPPoolClient() override;

    // PoolClient interface
    bool connect(const std::string& host, uint16_t port) override;
    void disconnect() override;
    bool is_connected() const override;

    // Wave 4 D-M5: actually waits for AUTH_OK / AUTH_FAIL / MSG_ERROR or timeout.
    // Returns false on AUTH_FAIL, MSG_ERROR, or timeout. The `password` parameter
    // is currently ignored by the JLP wire protocol (no password field in the
    // ClientHello struct); it is accepted for interface compatibility with
    // PoolClient but a non-empty value will produce a one-time warning.
    bool authenticate(const std::string& worker_name,
                     const std::string& password = "") override;

    bool request_work(WorkAssignment& work) override;
    bool submit_dp(const DistinguishedPoint& dp) override;
    bool submit_dps(const std::vector<DistinguishedPoint>& dps) override;

    PoolStats get_stats() override;
    bool report_solution(const uint8_t* private_key) override;

    void set_solution_callback(SolutionCallback cb) override;
    void set_work_callback(WorkCallback cb) override;

    std::string get_pool_type() const override { return POOL_TYPE_JLP; }

    // JLP-specific settings
    void set_timeout(uint32_t timeout_ms) { timeout_ms_ = timeout_ms; }
    void set_reconnect(bool auto_reconnect) { auto_reconnect_ = auto_reconnect; }
    void set_debug_mode(bool debug) { debug_mode_ = debug; }
    void set_use_tls(bool use_tls) { use_tls_ = use_tls; }
    void set_verify_cert(bool verify) { verify_cert_ = verify; }

private:
    bool debug_mode_ = false;
    // Network
    socket_t socket_;
    std::string host_;
    uint16_t port_;
    uint32_t timeout_ms_;
    bool auto_reconnect_;
    std::atomic<bool> connected_;
    std::atomic<bool> running_;
    std::atomic<bool> last_receive_was_timeout_;  // Track if last recv was timeout vs disconnect

    // Wave 4 D-H4: connection-state machine. atomic so the receiver and main
    // threads can both read it without taking a lock on the hot dispatch path.
    std::atomic<AuthState> auth_state_{AuthState::DISCONNECTED};

    // Wave 4 D-M5: condition variable + mutex used by authenticate() to wait
    // for an AUTH_OK / AUTH_FAIL / MSG_ERROR transition.
    std::mutex auth_cv_mutex_;
    std::condition_variable auth_cv_;

    // Reconnection with exponential backoff
    static constexpr uint32_t RECONNECT_BASE_DELAY_MS = 1000;    // Start at 1 second
    static constexpr uint32_t RECONNECT_MAX_DELAY_MS = 60000;    // Cap at 60 seconds
    static constexpr double RECONNECT_BACKOFF_MULTIPLIER = 2.0;  // Double each time
    uint32_t reconnect_delay_ms_ = RECONNECT_BASE_DELAY_MS;
    uint32_t reconnect_attempts_ = 0;

    // Wave 4 D-H5: count *consecutive* AUTH_FAILs across reconnects so we can
    // give up instead of hammering the pool forever with bad creds.
    uint32_t consecutive_auth_failures_ = 0;

    // TLS support
    bool use_tls_ = false;
    bool verify_cert_ = true;  // Wave 4 D-H2: default to verify (was false / fail-open)
#ifdef COLLIDER_HAS_OPENSSL
    SSL_CTX* ssl_ctx_ = nullptr;
    SSL* ssl_ = nullptr;
    bool init_tls();
    void cleanup_tls();
    int ssl_send(const void* data, size_t size);
    int ssl_recv(void* data, size_t size);
#endif

    // Separate read and write mutexes for concurrent I/O on the SSL object.
    //
    // OpenSSL is documented to support one concurrent reader and one
    // concurrent writer on the same SSL session, provided each direction is
    // serialized internally. A SINGLE mutex covering both sides deadlocks:
    // the receiver thread blocks in SSL_read holding the mutex, and any
    // main-thread send_message() (e.g. WORK_REQ) cannot acquire the mutex
    // to send -- so the request never goes on the wire and the worker
    // appears idle to the pool until it times out.
    //
    // We therefore split into:
    //   ssl_write_mutex_  -- serializes sender_loop and main-thread sends
    //   ssl_read_mutex_   -- serializes only the receiver_loop (defensive;
    //                        only one thread reads in this design)
    // Renegotiation / shutdown / SSL_clear must be done with both threads
    // joined (see disconnect()) -- those are NOT covered by these mutexes.
    std::mutex ssl_write_mutex_;
    std::mutex ssl_read_mutex_;

    // Worker info
    std::string worker_name_;
    uint32_t gpu_count_;
    uint64_t speed_;

    // Current work
    WorkAssignment current_work_;
    bool work_received_ = false;  // true once WORK_ASN arrives (work_id==0 is a valid chunk)
    std::mutex work_mutex_;

    // Statistics
    PoolStats stats_;
    std::mutex stats_mutex_;

    // Callbacks
    SolutionCallback solution_callback_;
    WorkCallback work_callback_;

    // Receiver thread
    std::thread receiver_thread_;
    void receiver_loop();

    // DP queue for batched submission
    std::queue<DistinguishedPoint> dp_queue_;
    std::mutex dp_mutex_;
    std::condition_variable dp_cv_;
    std::thread sender_thread_;
    void sender_loop();

    // Protocol helpers
    bool send_message(JLPMessageType type, const void* data, size_t size);
    bool receive_message(JLPHeader& header, std::vector<uint8_t>& payload);
    bool send_hello();
    void handle_server_message(const JLPHeader& header, const std::vector<uint8_t>& payload);

    // Wave 4 B-LOW-6 helper: safely (re)assign a std::thread by joining the
    // existing one if joinable. Plain `t = std::thread(...)` on a joinable
    // thread invokes std::terminate per [thread.thread.assign]/p2.
    static void replace_thread(std::thread& t, std::thread new_thread);

    // Platform init
    static bool init_sockets();
    static void cleanup_sockets();
    static bool sockets_initialized_;
};

} // namespace pool
} // namespace collider
