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
#include "pool_config.hpp"
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <array>
#include <limits>

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

    // Distinguished points (v1 - no chunk attestation)
    DP_SUBMIT = 0x20,
    DP_ACK    = 0x21,
    DP_BATCH  = 0x22,

    // Distinguished points v2 - identical fields plus an 8-byte work_id
    // prefix attesting which assigned chunk the DP came from. The server
    // verifies the claimed work_id matches this worker's current
    // assignment and rejects the submission otherwise (catches the
    // wrong-chunk submission attack class). v2 wire layout matches
    // collision-protocol/src/jlp_protocol.py DistinguishedPointV2:
    //   struct.pack('<Q32s32sBB', work_id, x, d, type, dp_bits)
    DP_SUBMIT_V2 = 0x23,
    DP_BATCH_V2  = 0x24,

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

// v1.4.1 B.2: AUTH wire format (120 bytes) -- pre-1.4.1 the client
// sent JLPClientHello (76 bytes) but the Python server decoded
// 96 bytes (name + password) and silently mis-aligned gpu_count/speed
// onto the password slot. The new format is name + password + a
// timestamp_ms + a 16-byte nonce so the server can reject replays of
// captured AUTH packets and enforce client-clock drift bounds.
//
// Field order matches collision-protocol/src/jlp_protocol.py
// (struct.pack equivalent: '<64s32sQ16s').
struct JLPClientHelloV2 {
    char     worker_name[64];   // Worker identifier (Bitcoin address)
    char     password[32];      // Optional pool password
    uint64_t timestamp_ms;      // Wall-clock time in ms (LE)
    uint8_t  nonce[16];         // Per-AUTH random nonce
};
static_assert(sizeof(JLPClientHelloV2) == 120,
              "JLPClientHelloV2 must be 120 bytes on the wire");

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

// v2 distinguished point: 78-byte wire layout (was 74 pre-1.4.1). The
// 8-byte work_id prefix attests which assigned chunk the DP came from;
// the 4-byte sequence (v1.4.1 B.1) is a per-(worker, work_id) monotonic
// counter. The server tracks an expected window and rejects out-of-
// window sequences as replays. Matches the server's
// DistinguishedPointV2.to_bytes() (struct.pack('<QI32s32sBB',
// work_id, sequence, x, d, type, dp_bits)).
// DP_BATCH_V2 payload is [count:u32 LE][dp1:78][dp2:78]...
struct JLPDistinguishedPointV2 {
    uint64_t work_id;        // 8 bytes  - chunk id from WorkAssignment (LE)
    uint32_t sequence;       // 4 bytes  - v1.4.1 B.1 monotonic per-chunk counter (LE)
    uint8_t  x[32];          // 32 bytes - X coordinate
    uint8_t  d[32];          // 32 bytes - Distance
    uint8_t  type;           // 1 byte   - Tame (0) or Wild (1)
    // Wire format pins this to 1 byte (struct.pack 'B' on the server side
    // gives 78-byte total). The abstract DistinguishedPoint::dp_bits in
    // pool_client.hpp is uint64_t for API ergonomics; the sender narrows
    // to uint8_t here. Realistic values are 1..50, so truncation is safe.
    // (Gemini PR review pointed out the type difference; deliberate.)
    uint8_t  dp_bits;        // 1 byte   - Number of leading-zero bits used
};
static_assert(sizeof(JLPDistinguishedPointV2) == 78,
              "JLPDistinguishedPointV2 must be 78 bytes on the wire");
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

    // Reconnection with exponential backoff. The cap MAX_RECONNECT_BACKOFF_MS
    // is defined in pool_config.hpp and shared with PoolManager so the two
    // reconnect paths agree on the upper bound.
    static constexpr uint32_t RECONNECT_BASE_DELAY_MS = 1000;    // Start at 1 second
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
    std::string password_;       // v1.4.1 B.2: optional pool password
    uint32_t gpu_count_;
    uint64_t speed_;

    // Current work
    WorkAssignment current_work_;
    bool work_received_ = false;  // true once WORK_ASN arrives (work_id==0 is a valid chunk)
    // v1.4.1 B.1: per-(worker, work_id) monotonic DP counter. Reset to
    // 0 every time the server hands out a new chunk; incremented on
    // each DP submitted in DP_BATCH_V2. The server tracks the expected
    // window per (worker_name, work_id) and rejects out-of-window
    // sequences as replays. The counter lives next to current_work_
    // because both are reset together when WORK_ASN arrives.
    uint32_t dp_sequence_next_ = 0;
    std::mutex work_mutex_;

    // Statistics
    PoolStats stats_;
    std::mutex stats_mutex_;

    // Callbacks. v1.4.1 C.3: protected by callbacks_mutex_ so that
    // set_*_callback can race against an inbound message handler
    // without tearing the function pointer, and so the dedup state
    // (last_solution_pubkey_, last_work_id_fired_) is consistent
    // across threads. Pre-1.4.1 the SOLUTION handler read
    // solution_callback_ directly without a lock and called the
    // callback inside the read; if a SOLUTION message arrived twice
    // (server retry, bug, malicious replay) the callback fired twice,
    // which on a privacy-sensitive pool would expose the recovered key
    // to two writers (e.g., logged + filed twice).
    SolutionCallback solution_callback_;
    WorkCallback     work_callback_;
    mutable std::mutex callbacks_mutex_;
    std::array<uint8_t, 32> last_solution_pubkey_{};  // zero = none seen
    bool     solution_fired_ = false;
    uint64_t last_work_id_fired_ = std::numeric_limits<uint64_t>::max();
    void fire_solution_callback(const uint8_t* pubkey_bytes);
    void fire_work_callback(const WorkAssignment& work);

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
    //
    // v1.4.1 P-T2: noexcept. The body catches every exception path
    // (join's std::system_error, detach's std::system_error, anything
    // else from std::cerr) and std::thread::operator= is itself
    // noexcept per the standard, so the function never propagates an
    // exception. Marking it noexcept lets the compiler tighten code
    // generation in the reconnect-error path that called this --
    // pre-1.4.1 the implicit "potentially-throwing" classification on
    // a function used during error recovery is the kind of latent
    // hazard that can cause std::terminate during cleanup.
    static void replace_thread(std::thread& t, std::thread new_thread) noexcept;

    // Platform init
    static bool init_sockets();
    static void cleanup_sockets();
    static bool sockets_initialized_;
};

} // namespace pool
} // namespace collider
