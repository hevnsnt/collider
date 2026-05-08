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
#include <shared_mutex>
#include <queue>
#include <condition_variable>

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

// JLP Protocol message types (must match collision-protocol pool server)
// Server uses: [MAGIC:4][TYPE:1][FLAGS:1][LENGTH:2] = 8 byte header
enum class JLPMessageType : uint8_t {
    // Authentication
    AUTH = 0x01,           // Auth request (96 bytes: worker[64] + password[32])
    AUTH_OK = 0x02,        // Auth accepted (0 bytes)
    AUTH_FAIL = 0x03,      // Auth rejected (0 bytes)

    // Work management
    WORK_REQ = 0x10,       // Request work (0 bytes)
    WORK_ASN = 0x11,       // Work assignment (102 bytes)

    // Distinguished points (v1 - no chunk attestation)
    DP_SUBMIT = 0x20,      // Single DP (66 bytes)
    DP_ACK = 0x21,         // DP acknowledged (4 bytes: count)
    DP_BATCH = 0x22,       // Batch of DPs (4 + N*66 bytes)

    // Distinguished points v2 - identical fields plus an 8-byte work_id
    // prefix attesting which assigned chunk the DP came from. The server
    // verifies the claimed work_id matches the worker's current
    // assignment and rejects the submission otherwise (catches the
    // wrong-chunk submission attack class).
    DP_SUBMIT_V2 = 0x23,   // Single v2 DP (74 bytes)
    DP_BATCH_V2 = 0x24,    // Batch of v2 DPs (4 + N*74 bytes)

    // Statistics
    STATS_REQ = 0x30,      // Request stats (0 bytes)
    STATS_RSP = 0x31,      // Stats response (32 bytes)

    // Solution
    SOLUTION = 0x40,       // Solution found (32 bytes private key)

    // Keepalive
    PING = 0x50,           // Keepalive request
    PONG = 0x51,           // Keepalive response

    // Error
    MSG_ERROR = 0xFF       // Error message
};

// JLP Protocol header (8 bytes, matches collision-protocol server)
// Format: [MAGIC:4][TYPE:1][FLAGS:1][LENGTH:2]
#pragma pack(push, 1)
struct JLPHeader {
    uint8_t magic[4];        // "KANG"
    uint8_t type;            // Message type
    uint8_t flags;           // Flags (always 0)
    uint16_t payload_size;   // Size of payload following header (little-endian)
};

// AUTH payload - 96 bytes (must match protocol spec)
struct JLPAuthPayload {
    char worker_name[64];    // Worker name/Bitcoin address (null-padded)
    char password[32];       // Pool password (null-padded)
    // Total: 96 bytes
};

// Work assignment - 102 bytes (must match protocol spec)
struct JLPWorkAssignment {
    uint32_t puzzle_id;      // 4 bytes - Puzzle number
    uint8_t range_start[32]; // 32 bytes - Search range start (big-endian)
    uint8_t range_end[32];   // 32 bytes - Search range end (big-endian)
    uint8_t public_key[33];  // 33 bytes - Target public key (compressed)
    uint8_t dp_bits;         // 1 byte - Distinguished point bits
    // Total: 102 bytes
};

// Distinguished point - 66 bytes (must match protocol spec)
struct JLPDistinguishedPoint {
    uint8_t x[32];           // X coordinate (big-endian)
    uint8_t d[32];           // Distance traveled (big-endian)
    uint8_t type;            // 0 = tame, 1 = wild
    uint8_t dp_bits;         // Number of leading zero bits
    // Total: 66 bytes
};

// Distinguished point v2 - 74 bytes. Adds an 8-byte work_id prefix.
// Layout MUST match the Python server's struct '<Q32s32sBB':
//   work_id (uint64 LE) + x[32] + d[32] + type[1] + dp_bits[1]
struct JLPDistinguishedPointV2 {
    uint64_t work_id;        // 8 bytes - chunk id from WorkAssignment (low 64 bits)
    uint8_t x[32];           // 32 bytes - X coordinate (big-endian)
    uint8_t d[32];           // 32 bytes - Distance traveled (big-endian)
    uint8_t type;            // 1 byte  - 0 = tame, 1 = wild
    uint8_t dp_bits;         // 1 byte  - Number of leading zero bits
    // Total: 74 bytes
};

// DP batch header
struct JLPDPBatchHeader {
    uint32_t count;          // Number of DPs in batch
};
#pragma pack(pop)
static_assert(sizeof(JLPDistinguishedPoint) == 66,
              "JLPDistinguishedPoint must be 66 bytes");
static_assert(sizeof(JLPDistinguishedPointV2) == 74,
              "JLPDistinguishedPointV2 must be 74 bytes");

class JLPPoolClient : public PoolClient {
public:
    JLPPoolClient();
    ~JLPPoolClient() override;

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
    socket_t socket_ = INVALID_SOCK;
    std::string host_;
    uint16_t port_;
    uint32_t timeout_ms_;
    bool auto_reconnect_;
    std::atomic<bool> connected_;
    std::atomic<bool> running_;
    std::atomic<bool> last_receive_was_timeout_;  // Track if last recv was timeout vs disconnect

    // Reconnection with exponential backoff
    static constexpr uint32_t RECONNECT_BASE_DELAY_MS = 1000;    // Start at 1 second
    static constexpr uint32_t RECONNECT_MAX_DELAY_MS = 60000;    // Cap at 60 seconds
    static constexpr double RECONNECT_BACKOFF_MULTIPLIER = 2.0;  // Double each time
    std::atomic<uint32_t> reconnect_delay_ms_{RECONNECT_BASE_DELAY_MS};
    uint32_t reconnect_attempts_ = 0;

    // TLS support
    bool use_tls_ = false;
    bool verify_cert_ = false;  // Skip cert verification for self-signed certs
#ifdef COLLIDER_HAS_OPENSSL
    SSL_CTX* ssl_ctx_ = nullptr;
    SSL* ssl_ = nullptr;
    bool init_tls();
    void cleanup_tls();
    int ssl_send(const void* data, size_t size);
    int ssl_recv(void* data, size_t size);
#endif

    // Worker info
    std::string worker_name_;
    std::string password_;
    uint32_t gpu_count_;
    uint64_t speed_;

    // Current work
    WorkAssignment current_work_;
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

    // SSL I/O mutex - prevents concurrent SSL operations that corrupt TLS state
    // Receiver uses try_lock_for with 100ms timeout so sender isn't starved
    std::timed_mutex ssl_io_mutex_;

    // Protocol helpers
    bool send_message(JLPMessageType type, const void* data, size_t size);
    bool receive_message(JLPHeader& header, std::vector<uint8_t>& payload);
    bool send_hello();
    void handle_server_message(const JLPHeader& header, const std::vector<uint8_t>& payload);

    // Platform init
    static bool init_sockets();
    static void cleanup_sockets();
    static bool sockets_initialized_;
};

} // namespace pool
} // namespace collider
