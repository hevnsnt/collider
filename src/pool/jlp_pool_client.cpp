// jlp_pool_client.cpp - JeanLucPons Kangaroo protocol implementation

#include "jlp_pool_client.hpp"
#include <cstring>
#include <iostream>
#include <chrono>
#include <algorithm>

namespace collider {
namespace pool {

bool JLPPoolClient::sockets_initialized_ = false;

#ifdef COLLIDER_HAS_OPENSSL
static bool ssl_initialized = false;

bool JLPPoolClient::init_tls() {
    if (!ssl_initialized) {
        SSL_library_init();
        SSL_load_error_strings();
        OpenSSL_add_all_algorithms();
        ssl_initialized = true;
    }

    // Create SSL context
    ssl_ctx_ = SSL_CTX_new(TLS_client_method());
    if (!ssl_ctx_) {
        std::cerr << "[Pool] Failed to create SSL context" << std::endl;
        return false;
    }

    // Set minimum TLS version to 1.2
    SSL_CTX_set_min_proto_version(ssl_ctx_, TLS1_2_VERSION);

    // Skip certificate verification for self-signed certs
    if (!verify_cert_) {
        SSL_CTX_set_verify(ssl_ctx_, SSL_VERIFY_NONE, nullptr);
    }

    // Create SSL connection
    ssl_ = SSL_new(ssl_ctx_);
    if (!ssl_) {
        std::cerr << "[Pool] Failed to create SSL connection" << std::endl;
        SSL_CTX_free(ssl_ctx_);
        ssl_ctx_ = nullptr;
        return false;
    }

    // Attach socket to SSL
    SSL_set_fd(ssl_, static_cast<int>(socket_));

    // Perform TLS handshake
    int ret = SSL_connect(ssl_);
    if (ret != 1) {
        int err = SSL_get_error(ssl_, ret);
        std::cerr << "[Pool] TLS handshake failed: " << err << std::endl;
        char buf[256];
        ERR_error_string_n(ERR_get_error(), buf, sizeof(buf));
        std::cerr << "[Pool] SSL error: " << buf << std::endl;
        SSL_free(ssl_);
        SSL_CTX_free(ssl_ctx_);
        ssl_ = nullptr;
        ssl_ctx_ = nullptr;
        return false;
    }

    std::cout << "[Pool] TLS connection established (" << SSL_get_version(ssl_) << ")" << std::endl;
    return true;
}

void JLPPoolClient::cleanup_tls() {
    if (ssl_) {
        SSL_shutdown(ssl_);
        SSL_free(ssl_);
        ssl_ = nullptr;
    }
    if (ssl_ctx_) {
        SSL_CTX_free(ssl_ctx_);
        ssl_ctx_ = nullptr;
    }
}

int JLPPoolClient::ssl_send(const void* data, size_t size) {
    if (!ssl_) return -1;
    return SSL_write(ssl_, data, static_cast<int>(size));
}

int JLPPoolClient::ssl_recv(void* data, size_t size) {
    if (!ssl_) return -1;
    return SSL_read(ssl_, data, static_cast<int>(size));
}
#endif

bool JLPPoolClient::init_sockets() {
#ifdef _WIN32
    if (!sockets_initialized_) {
        WSADATA wsa_data;
        if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
            return false;
        }
        sockets_initialized_ = true;
    }
#endif
    return true;
}

void JLPPoolClient::cleanup_sockets() {
#ifdef _WIN32
    if (sockets_initialized_) {
        WSACleanup();
        sockets_initialized_ = false;
    }
#endif
}

JLPPoolClient::JLPPoolClient()
    : socket_(INVALID_SOCK)
    , port_(17403)
    , timeout_ms_(30000)  // 30 second timeout (was 3 seconds)
    , auto_reconnect_(true)
    , connected_(false)
    , running_(false)
    , last_receive_was_timeout_(false)
    , gpu_count_(1)
    , speed_(0)
{
    init_sockets();
    memset(&stats_, 0, sizeof(stats_));
}

JLPPoolClient::~JLPPoolClient() {
    disconnect();
}

bool JLPPoolClient::connect(const std::string& host, uint16_t port) {
    if (connected_) {
        disconnect();
    }

    host_ = host;
    port_ = port;

    // Resolve hostname
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result) != 0) {
        std::cerr << "[Pool] Failed to resolve hostname: " << host << std::endl;
        return false;
    }

    // Create socket
    socket_ = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (socket_ == INVALID_SOCK) {
        freeaddrinfo(result);
        std::cerr << "[Pool] Failed to create socket" << std::endl;
        return false;
    }

    // Set timeout
#ifdef _WIN32
    DWORD timeout = timeout_ms_;
    setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
    setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
#else
    struct timeval tv;
    tv.tv_sec = timeout_ms_ / 1000;
    tv.tv_usec = (timeout_ms_ % 1000) * 1000;
    setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

    // Connect
    if (::connect(socket_, result->ai_addr, (int)result->ai_addrlen) == SOCK_ERROR) {
        freeaddrinfo(result);
        closesocket(socket_);
        socket_ = INVALID_SOCK;
        std::cerr << "[Pool] Failed to connect to " << host << ":" << port << std::endl;
        return false;
    }

    freeaddrinfo(result);

    // Initialize TLS if enabled
#ifdef COLLIDER_HAS_OPENSSL
    if (use_tls_) {
        if (!init_tls()) {
            closesocket(socket_);
            socket_ = INVALID_SOCK;
            return false;
        }
    }
#else
    if (use_tls_) {
        std::cerr << "[Pool] TLS requested but OpenSSL not available" << std::endl;
        closesocket(socket_);
        socket_ = INVALID_SOCK;
        return false;
    }
#endif

    connected_ = true;
    running_ = true;

    std::cout << "[Pool] Connected to " << host << ":" << port;
    if (use_tls_) std::cout << " (TLS)";
    std::cout << std::endl;

    // Start receiver thread
    receiver_thread_ = std::thread(&JLPPoolClient::receiver_loop, this);

    // Start sender thread for batched DP submission
    sender_thread_ = std::thread(&JLPPoolClient::sender_loop, this);

    return true;
}

void JLPPoolClient::disconnect() {
    running_ = false;
    connected_ = false;

    // Wake up sender thread
    dp_cv_.notify_all();

    if (socket_ != INVALID_SOCK) {
        // Send goodbye message
        send_message(JLPMessageType::CLIENT_GOODBYE, nullptr, 0);

#ifdef COLLIDER_HAS_OPENSSL
        if (use_tls_) {
            cleanup_tls();
        }
#endif

        closesocket(socket_);
        socket_ = INVALID_SOCK;
    }

    if (receiver_thread_.joinable()) {
        receiver_thread_.join();
    }
    if (sender_thread_.joinable()) {
        sender_thread_.join();
    }

    std::cout << "[Pool] Disconnected" << std::endl;
}

bool JLPPoolClient::is_connected() const {
    return connected_;
}

bool JLPPoolClient::authenticate(const std::string& worker_name, const std::string& password) {
    worker_name_ = worker_name;
    return send_hello();
}

bool JLPPoolClient::send_hello() {
    JLPClientHello hello;
    memset(&hello, 0, sizeof(hello));
    strncpy(hello.worker_name, worker_name_.c_str(), sizeof(hello.worker_name) - 1);
    hello.gpu_count = gpu_count_;
    hello.speed = speed_;

    return send_message(JLPMessageType::CLIENT_HELLO, &hello, sizeof(hello));
}

bool JLPPoolClient::request_work(WorkAssignment& work) {
    // Send status request (triggers work assignment)
    if (!send_message(JLPMessageType::CLIENT_STATUS, nullptr, 0)) {
        return false;
    }

    // Wait for work (with timeout)
    auto start = std::chrono::steady_clock::now();
    while (running_) {
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            if (current_work_.work_id != 0) {
                work = current_work_;
                return true;
            }
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > timeout_ms_) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return false;
}

bool JLPPoolClient::submit_dp(const DistinguishedPoint& dp) {
    std::lock_guard<std::mutex> lock(dp_mutex_);
    dp_queue_.push(dp);
    dp_cv_.notify_one();
    return true;
}

bool JLPPoolClient::submit_dps(const std::vector<DistinguishedPoint>& dps) {
    std::lock_guard<std::mutex> lock(dp_mutex_);
    for (const auto& dp : dps) {
        dp_queue_.push(dp);
    }
    dp_cv_.notify_one();
    return true;
}

PoolStats JLPPoolClient::get_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool JLPPoolClient::report_solution(const uint8_t* private_key) {
    return send_message(JLPMessageType::CLIENT_SOLUTION, private_key, 32);
}

void JLPPoolClient::set_solution_callback(SolutionCallback cb) {
    solution_callback_ = cb;
}

void JLPPoolClient::set_work_callback(WorkCallback cb) {
    work_callback_ = cb;
}

bool JLPPoolClient::send_message(JLPMessageType type, const void* data, size_t size) {
    if (!connected_ || socket_ == INVALID_SOCK) {
        return false;
    }

    JLPHeader header;
    header.magic[0] = 'K';
    header.magic[1] = 'A';
    header.magic[2] = 'N';
    header.magic[3] = 'G';
    header.version = 1;
    header.type = type;
    header.payload_size = static_cast<uint32_t>(size);

    // Send header
#ifdef COLLIDER_HAS_OPENSSL
    if (use_tls_ && ssl_) {
        if (ssl_send(&header, sizeof(header)) != sizeof(header)) {
            connected_ = false;
            return false;
        }
    } else
#endif
    {
        if (send(socket_, (const char*)&header, sizeof(header), 0) != sizeof(header)) {
            connected_ = false;
            return false;
        }
    }

    // Send payload
    if (size > 0 && data != nullptr) {
#ifdef COLLIDER_HAS_OPENSSL
        if (use_tls_ && ssl_) {
            if (ssl_send(data, size) != (int)size) {
                connected_ = false;
                return false;
            }
        } else
#endif
        {
            if (send(socket_, (const char*)data, (int)size, 0) != (int)size) {
                connected_ = false;
                return false;
            }
        }
    }

    return true;
}

bool JLPPoolClient::receive_message(JLPHeader& header, std::vector<uint8_t>& payload) {
    if (!connected_ || socket_ == INVALID_SOCK) {
        return false;
    }

    int received;

    // Receive header
#ifdef COLLIDER_HAS_OPENSSL
    if (use_tls_ && ssl_) {
        received = ssl_recv(&header, sizeof(header));
        if (received != sizeof(header)) {
            int ssl_err = SSL_get_error(ssl_, received);
            if (ssl_err == SSL_ERROR_WANT_READ || ssl_err == SSL_ERROR_WANT_WRITE) {
                last_receive_was_timeout_ = true;
                return false;
            }
            last_receive_was_timeout_ = false;
            return false;
        }
    } else
#endif
    {
        received = recv(socket_, (char*)&header, sizeof(header), MSG_WAITALL);
        if (received != sizeof(header)) {
            // Check if it's a timeout vs actual disconnect
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAETIMEDOUT || err == WSAEWOULDBLOCK) {
                last_receive_was_timeout_ = true;
                return false;
            }
#else
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ETIMEDOUT) {
                last_receive_was_timeout_ = true;
                return false;
            }
#endif
            last_receive_was_timeout_ = false;
            return false;
        }
    }
    last_receive_was_timeout_ = false;

    // Validate magic
    if (header.magic[0] != 'K' || header.magic[1] != 'A' ||
        header.magic[2] != 'N' || header.magic[3] != 'G') {
        std::cerr << "[Pool] Invalid message magic" << std::endl;
        return false;
    }

    // Validate payload size (max 10MB to prevent OOM)
    constexpr uint32_t MAX_PAYLOAD_SIZE = 10 * 1024 * 1024;
    if (header.payload_size > MAX_PAYLOAD_SIZE) {
        std::cerr << "[Pool] Payload size exceeds limit: " << header.payload_size << " bytes" << std::endl;
        return false;
    }

    // Receive payload
    if (header.payload_size > 0) {
        payload.resize(header.payload_size);
#ifdef COLLIDER_HAS_OPENSSL
        if (use_tls_ && ssl_) {
            received = ssl_recv(payload.data(), header.payload_size);
        } else
#endif
        {
            received = recv(socket_, (char*)payload.data(), header.payload_size, MSG_WAITALL);
        }
        if (received != (int)header.payload_size) {
            return false;
        }
    }

    return true;
}

void JLPPoolClient::receiver_loop() {
    while (running_ && connected_) {
        JLPHeader header;
        std::vector<uint8_t> payload;

        if (receive_message(header, payload)) {
            handle_server_message(header, payload);
        } else if (last_receive_was_timeout_) {
            // Just a timeout, not a real disconnect - continue waiting
            // This is normal when server has nothing to send
            continue;
        } else if (connected_ && auto_reconnect_) {
            // Actual connection loss - try to reconnect with exponential backoff
            closesocket(socket_);
            socket_ = INVALID_SOCK;
            connected_ = false;

            while (running_ && auto_reconnect_) {
                reconnect_attempts_++;
                std::cerr << "[Pool] Connection lost, reconnect attempt " << reconnect_attempts_
                          << " in " << (reconnect_delay_ms_ / 1000.0) << "s..." << std::endl;

                std::this_thread::sleep_for(std::chrono::milliseconds(reconnect_delay_ms_));

                if (!running_) break;

                if (connect(host_, port_) && authenticate(worker_name_)) {
                    // Success - reset backoff
                    std::cerr << "[Pool] Reconnected successfully after " << reconnect_attempts_
                              << " attempt(s)" << std::endl;
                    reconnect_delay_ms_ = RECONNECT_BASE_DELAY_MS;
                    reconnect_attempts_ = 0;
                    break;
                }

                // Failed - increase backoff (exponential with cap)
                reconnect_delay_ms_ = static_cast<uint32_t>(
                    std::min(static_cast<double>(RECONNECT_MAX_DELAY_MS),
                             reconnect_delay_ms_ * RECONNECT_BACKOFF_MULTIPLIER));
            }
        }
    }
}

void JLPPoolClient::sender_loop() {
    std::vector<JLPDistinguishedPoint> batch;
    batch.reserve(100);

    while (running_) {
        {
            std::unique_lock<std::mutex> lock(dp_mutex_);
            dp_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !dp_queue_.empty() || !running_;
            });

            // Collect batch
            while (!dp_queue_.empty() && batch.size() < 100) {
                const auto& dp = dp_queue_.front();
                JLPDistinguishedPoint jlp_dp;
                memcpy(jlp_dp.x, dp.x, 32);
                memcpy(jlp_dp.d, dp.d, 32);
                jlp_dp.type = dp.type;
                batch.push_back(jlp_dp);
                dp_queue_.pop();
            }
        }

        // Send batch
        if (!batch.empty() && connected_) {
            send_message(JLPMessageType::CLIENT_DP, batch.data(),
                        batch.size() * sizeof(JLPDistinguishedPoint));

            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.your_dps += batch.size();

            batch.clear();
        }
    }
}

void JLPPoolClient::handle_server_message(const JLPHeader& header,
                                          const std::vector<uint8_t>& payload) {
    switch (header.type) {
        case JLPMessageType::SERVER_CONFIG:
        case JLPMessageType::SERVER_WORK: {
            // Debug output only when enabled
            if (debug_mode_) {
                std::cerr << "[DEBUG] SERVER_WORK payload size: " << payload.size()
                          << " (expected: " << sizeof(JLPServerConfig) << ")" << std::endl;
                std::cerr << "[DEBUG] Raw payload (first 128 bytes): ";
                for (size_t i = 0; i < std::min(payload.size(), (size_t)128); i++) {
                    char buf[4];
                    snprintf(buf, sizeof(buf), "%02x", payload[i]);
                    std::cerr << buf;
                    if ((i + 1) % 33 == 0) std::cerr << " | ";
                }
                std::cerr << std::endl;
            }

            if (payload.size() >= sizeof(JLPServerConfig)) {
                const JLPServerConfig* config =
                    reinterpret_cast<const JLPServerConfig*>(payload.data());

                // Debug: show parsed public key (only when debug enabled)
                if (debug_mode_) {
                    std::cerr << "[DEBUG] Parsed pubkey: ";
                    for (int i = 0; i < 33; i++) {
                        char buf[4];
                        snprintf(buf, sizeof(buf), "%02x", config->public_key[i]);
                        std::cerr << buf;
                    }
                    std::cerr << std::endl;
                }

                std::lock_guard<std::mutex> lock(work_mutex_);
                memcpy(current_work_.public_key, config->public_key, 33);
                memcpy(current_work_.range_start, config->range_start, 32);
                memcpy(current_work_.range_end, config->range_end, 32);
                current_work_.dp_bits = config->dp_bits;
                current_work_.work_id = config->work_id;

                std::cout << "[Pool] Received work assignment (ID: " << config->work_id
                          << ", DP bits: " << config->dp_bits << ")" << std::endl;

                if (work_callback_) {
                    work_callback_(current_work_);
                }
            }
            break;
        }

        case JLPMessageType::SERVER_STATUS: {
            // Parse status update
            if (payload.size() >= 32) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                memcpy(&stats_.total_dps, payload.data(), 8);
                memcpy(&stats_.connected_workers, payload.data() + 8, 8);
                memcpy(&stats_.pool_speed, payload.data() + 16, 8);
            }
            break;
        }

        case JLPMessageType::SERVER_SOLUTION_FOUND: {
            std::cout << "[Pool] SOLUTION FOUND!" << std::endl;
            if (solution_callback_ && payload.size() >= 32) {
                solution_callback_(payload.data());
            }
            break;
        }

        case JLPMessageType::SERVER_ERROR: {
            std::string error(payload.begin(), payload.end());
            std::cerr << "[Pool] Server error: " << error << std::endl;
            break;
        }

        default:
            break;
    }
}

// DistinguishedPoint serialization
std::vector<uint8_t> DistinguishedPoint::serialize() const {
    std::vector<uint8_t> data(65 + 8);
    memcpy(data.data(), x, 32);
    memcpy(data.data() + 32, d, 32);
    data[64] = type;
    memcpy(data.data() + 65, &dp_bits, 8);
    return data;
}

DistinguishedPoint DistinguishedPoint::deserialize(const uint8_t* data, size_t len) {
    DistinguishedPoint dp;
    if (len >= 65) {
        memcpy(dp.x, data, 32);
        memcpy(dp.d, data + 32, 32);
        dp.type = data[64];
        if (len >= 73) {
            memcpy(&dp.dp_bits, data + 65, 8);
        }
    }
    return dp;
}

// Factory function
std::unique_ptr<PoolClient> create_pool_client(const std::string& type) {
    if (type == POOL_TYPE_JLP) {
        return std::make_unique<JLPPoolClient>();
    }
    // Add other pool types here
    return nullptr;
}

} // namespace pool
} // namespace collider
