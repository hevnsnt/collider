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

// JLP Protocol message types
enum class JLPMessageType : uint8_t {
    // Client -> Server
    CLIENT_HELLO = 0x00,
    CLIENT_DP = 0x01,
    CLIENT_SOLUTION = 0x02,
    CLIENT_STATUS = 0x03,
    CLIENT_GOODBYE = 0x04,

    // Server -> Client
    SERVER_CONFIG = 0x10,
    SERVER_WORK = 0x11,
    SERVER_STATUS = 0x12,
    SERVER_SOLUTION_FOUND = 0x13,
    SERVER_ERROR = 0x1F
};

// JLP Protocol header (all messages start with this)
#pragma pack(push, 1)
struct JLPHeader {
    uint8_t magic[4];        // "KANG"
    uint8_t version;         // Protocol version (1)
    JLPMessageType type;     // Message type
    uint32_t payload_size;   // Size of payload following header
};

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
};
#pragma pack(pop)

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
    socket_t socket_;
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
    uint32_t reconnect_delay_ms_ = RECONNECT_BASE_DELAY_MS;
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
