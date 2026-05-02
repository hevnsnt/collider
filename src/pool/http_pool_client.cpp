// http_pool_client.cpp - HTTP/REST pool client implementation

#include "http_pool_client.hpp"
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

#ifdef _WIN32
    #pragma comment(lib, "ws2_32.lib")
    #define INVALID_SOCK INVALID_SOCKET
    #define SOCK_ERROR SOCKET_ERROR
    #define closesocket closesocket
#else
    #define INVALID_SOCK -1
    #define SOCK_ERROR -1
    #define closesocket close
#endif

namespace collider {
namespace pool {

// Escape special characters for JSON strings to prevent injection
static std::string json_escape(const std::string& str) {
    std::string result;
    result.reserve(str.size() * 2);  // Reserve extra space for escapes
    for (char c : str) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    // Escape control characters as \u00XX
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
        }
    }
    return result;
}

HTTPPoolClient::HTTPPoolClient()
    : port_(80)
    , use_ssl_(false)
    , connected_(false)
    , running_(false)
{
    config_ = HTTPPoolConfig::defaults();
    memset(&stats_, 0, sizeof(stats_));
}

HTTPPoolClient::~HTTPPoolClient() {
    disconnect();
}

bool HTTPPoolClient::connect(const std::string& host, uint16_t port) {
    host_ = host;
    port_ = port;

    // Test connection with a stats request
    std::string response = http_get(config_.stats_endpoint);
    if (response.empty()) {
        std::cerr << "[HTTPPool] Failed to connect to " << host << ":" << port << std::endl;
        return false;
    }

    connected_ = true;
    running_ = true;

    std::cout << "[HTTPPool] Connected to " << host << ":" << port << std::endl;

    // Start background threads
    sender_thread_ = std::thread(&HTTPPoolClient::sender_loop, this);
    poll_thread_ = std::thread(&HTTPPoolClient::poll_loop, this);

    return true;
}

void HTTPPoolClient::disconnect() {
    running_ = false;
    connected_ = false;
    dp_cv_.notify_all();

    if (sender_thread_.joinable()) {
        sender_thread_.join();
    }
    if (poll_thread_.joinable()) {
        poll_thread_.join();
    }

    close_persistent_socket();
}

bool HTTPPoolClient::is_connected() const {
    return connected_;
}

bool HTTPPoolClient::authenticate(const std::string& worker_name, const std::string& password) {
    worker_name_ = worker_name;

    std::stringstream body;
    body << "{\"worker\":\"" << json_escape(worker_name) << "\"";
    if (!password.empty()) {
        body << ",\"password\":\"" << json_escape(password) << "\"";
    }
    if (!api_key_.empty()) {
        body << ",\"api_key\":\"" << json_escape(api_key_) << "\"";
    }
    body << "}";

    std::string response = http_post(config_.auth_endpoint, body.str());
    if (response.empty()) {
        return false;
    }

    // Extract token from response (simple parsing)
    size_t token_pos = response.find("\"token\"");
    if (token_pos != std::string::npos) {
        size_t quote1 = response.find("\"", token_pos + 7);
        if (quote1 != std::string::npos) {
            size_t start = quote1 + 1;
            size_t end = response.find("\"", start);
            if (end != std::string::npos) {
                auth_token_ = response.substr(start, end - start);
            }
        }
    }

    if (auth_token_.empty()) {
        std::cerr << "[HTTPPool] Authentication failed: no token in response" << std::endl;
        return false;
    }

    std::cout << "[HTTPPool] Authenticated as " << worker_name << std::endl;
    return true;
}

bool HTTPPoolClient::request_work(WorkAssignment& work) {
    std::string response = http_get(config_.work_endpoint);
    if (response.empty()) {
        return false;
    }

    return parse_work_json(response, work);
}

bool HTTPPoolClient::submit_dp(const DistinguishedPoint& dp) {
    std::lock_guard<std::mutex> lock(dp_mutex_);
    dp_queue_.push(dp);
    dp_cv_.notify_one();
    return true;
}

bool HTTPPoolClient::submit_dps(const std::vector<DistinguishedPoint>& dps) {
    std::lock_guard<std::mutex> lock(dp_mutex_);
    for (const auto& dp : dps) {
        dp_queue_.push(dp);
    }
    dp_cv_.notify_one();
    return true;
}

PoolStats HTTPPoolClient::get_stats() {
    std::string response = http_get(config_.stats_endpoint);
    if (!response.empty()) {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        parse_stats_json(response, stats_);
    }
    return stats_;
}

bool HTTPPoolClient::report_solution(const uint8_t* private_key) {
    std::stringstream body;
    body << "{\"private_key\":\"" << to_hex(private_key, 32) << "\","
         << "\"worker\":\"" << json_escape(worker_name_) << "\"}";

    std::string response = http_post(config_.solution_endpoint, body.str());
    return !response.empty();
}

void HTTPPoolClient::set_solution_callback(SolutionCallback cb) {
    solution_callback_ = cb;
}

void HTTPPoolClient::set_work_callback(WorkCallback cb) {
    work_callback_ = cb;
}

void HTTPPoolClient::sender_loop() {
    std::vector<DistinguishedPoint> batch;
    batch.reserve(100);

    while (running_) {
        {
            std::unique_lock<std::mutex> lock(dp_mutex_);
            dp_cv_.wait_for(lock, std::chrono::seconds(1), [this] {
                return !dp_queue_.empty() || !running_;
            });

            while (!dp_queue_.empty() && batch.size() < 100) {
                batch.push_back(dp_queue_.front());
                dp_queue_.pop();
            }
        }

        if (!batch.empty() && connected_) {
            std::string body = to_json_dps(batch);
            std::string response = http_post(config_.submit_endpoint, body);

            if (!response.empty()) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                stats_.your_dps += batch.size();
            }

            batch.clear();
        }
    }
}

void HTTPPoolClient::poll_loop() {
    while (running_) {
        // Poll for status and new work every 30 seconds
        std::this_thread::sleep_for(std::chrono::seconds(30));

        if (!running_) break;

        // Update stats
        std::string stats_response = http_get(config_.stats_endpoint);
        if (!stats_response.empty()) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            parse_stats_json(stats_response, stats_);

            // Check for solution
            if (stats_response.find("\"solution_found\":true") != std::string::npos) {
                std::cout << "[HTTPPool] SOLUTION FOUND!" << std::endl;
                // Extract private key and call callback
                if (solution_callback_) {
                    // Parse private key from response
                    size_t key_pos = stats_response.find("\"private_key\"");
                    if (key_pos != std::string::npos) {
                        size_t start = stats_response.find("\"", key_pos + 13) + 1;
                        size_t end = stats_response.find("\"", start);
                        if (start != std::string::npos && end != std::string::npos) {
                            std::string hex_key = stats_response.substr(start, end - start);
                            uint8_t key[32];
                            if (from_hex(hex_key, key, 32)) {
                                solution_callback_(key);
                            }
                        }
                    }
                }
            }
        }
    }
}

std::string HTTPPoolClient::http_get(const std::string& endpoint) {
    return make_request("GET", endpoint);
}

std::string HTTPPoolClient::http_post(const std::string& endpoint, const std::string& body) {
    return make_request("POST", endpoint, body);
}

bool HTTPPoolClient::is_socket_alive(socket_t sock) {
    if (sock == INVALID_SOCK) return false;

    // Check if socket is still connected using a zero-byte peek
    char buf;
#ifdef _WIN32
    int result = recv(sock, &buf, 1, MSG_PEEK);
    if (result == SOCK_ERROR) {
        int err = WSAGetLastError();
        // WSAEWOULDBLOCK means socket is alive but no data ready
        return (err == WSAEWOULDBLOCK || err == WSAETIMEDOUT);
    }
#else
    int result = recv(sock, &buf, 1, MSG_PEEK | MSG_DONTWAIT);
    if (result == -1) {
        // EAGAIN/EWOULDBLOCK means socket is alive but no data ready
        return (errno == EAGAIN || errno == EWOULDBLOCK);
    }
#endif
    // result == 0 means peer closed connection
    return (result > 0);
}

socket_t HTTPPoolClient::connect_to_server() {
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port_);
    if (getaddrinfo(host_.c_str(), port_str.c_str(), &hints, &result) != 0) {
        return INVALID_SOCK;
    }

    socket_t sock = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (sock == INVALID_SOCK) {
        freeaddrinfo(result);
        return INVALID_SOCK;
    }

    // Set timeout
#ifdef _WIN32
    DWORD timeout = 10000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
#else
    struct timeval tv;
    tv.tv_sec = 10;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

    if (::connect(sock, result->ai_addr, (int)result->ai_addrlen) == SOCK_ERROR) {
        freeaddrinfo(result);
        closesocket(sock);
        return INVALID_SOCK;
    }
    freeaddrinfo(result);
    return sock;
}

void HTTPPoolClient::close_persistent_socket() {
    std::lock_guard<std::mutex> lock(socket_mutex_);
    if (persistent_socket_ != INVALID_SOCK) {
        closesocket(persistent_socket_);
        persistent_socket_ = INVALID_SOCK;
    }
}

std::string HTTPPoolClient::make_request(const std::string& method,
                                         const std::string& endpoint,
                                         const std::string& body) {
    std::lock_guard<std::mutex> lock(socket_mutex_);

    // Check if persistent connection is still alive and not timed out
    auto now = std::chrono::steady_clock::now();
    bool need_reconnect = (persistent_socket_ == INVALID_SOCK);
    if (!need_reconnect) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_request_time_).count();
        if (elapsed > KEEPALIVE_TIMEOUT_SEC || !is_socket_alive(persistent_socket_)) {
            closesocket(persistent_socket_);
            persistent_socket_ = INVALID_SOCK;
            need_reconnect = true;
        }
    }

    if (need_reconnect) {
        persistent_socket_ = connect_to_server();
        if (persistent_socket_ == INVALID_SOCK) {
            return "";
        }
    }

    // Build HTTP request with keep-alive
    std::stringstream request;
    request << method << " " << config_.base_url << endpoint << " HTTP/1.1\r\n";
    request << "Host: " << host_ << "\r\n";
    request << "User-Agent: collider/1.0\r\n";
    request << "Accept: application/json\r\n";
    request << "Connection: keep-alive\r\n";

    if (!auth_token_.empty()) {
        request << "Authorization: Bearer " << auth_token_ << "\r\n";
    }

    if (!body.empty()) {
        request << "Content-Type: application/json\r\n";
        request << "Content-Length: " << body.size() << "\r\n";
    }

    request << "\r\n";

    if (!body.empty()) {
        request << body;
    }

    std::string request_str = request.str();

    // Send request
    if (send(persistent_socket_, request_str.c_str(), (int)request_str.size(), 0) == SOCK_ERROR) {
        // Connection broken, close and retry once
        closesocket(persistent_socket_);
        persistent_socket_ = connect_to_server();
        if (persistent_socket_ == INVALID_SOCK) {
            return "";
        }
        if (send(persistent_socket_, request_str.c_str(), (int)request_str.size(), 0) == SOCK_ERROR) {
            closesocket(persistent_socket_);
            persistent_socket_ = INVALID_SOCK;
            return "";
        }
    }

    last_request_time_ = std::chrono::steady_clock::now();

    // Receive response headers first, then body based on Content-Length
    std::string response;
    char buffer[4096];
    int received;
    while ((received = recv(persistent_socket_, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[received] = '\0';
        response += buffer;

        // Check if we have received complete headers + body
        size_t header_end = response.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            // Check for Content-Length to know when body is complete
            size_t cl_pos = response.find("Content-Length:");
            if (cl_pos == std::string::npos) {
                cl_pos = response.find("content-length:");
            }
            if (cl_pos != std::string::npos) {
                size_t cl_start = cl_pos + 15;
                while (cl_start < response.size() && response[cl_start] == ' ') cl_start++;
                size_t cl_end = response.find("\r\n", cl_start);
                if (cl_end != std::string::npos) {
                    size_t content_length = std::stoull(response.substr(cl_start, cl_end - cl_start));
                    size_t body_received = response.size() - (header_end + 4);
                    if (body_received >= content_length) {
                        break;  // Complete response received
                    }
                }
            } else {
                // No Content-Length, check for Connection: close or chunked
                // For keep-alive without Content-Length, we cannot determine end
                // Fall back to checking for connection close
                std::string conn_header;
                size_t conn_pos = response.find("Connection:");
                if (conn_pos == std::string::npos) {
                    conn_pos = response.find("connection:");
                }
                if (conn_pos != std::string::npos) {
                    size_t conn_start = conn_pos + 11;
                    size_t conn_end = response.find("\r\n", conn_start);
                    if (conn_end != std::string::npos) {
                        conn_header = response.substr(conn_start, conn_end - conn_start);
                    }
                }
                if (conn_header.find("close") != std::string::npos) {
                    // Server will close connection, read until EOF
                    continue;
                }
                // No content-length and keep-alive: assume empty body
                break;
            }
        }
    }

    // If recv returned 0 or error, the connection was closed by server
    if (received <= 0) {
        closesocket(persistent_socket_);
        persistent_socket_ = INVALID_SOCK;
    }

    // Extract body from response
    size_t body_start = response.find("\r\n\r\n");
    if (body_start != std::string::npos) {
        return response.substr(body_start + 4);
    }

    return response;
}

std::string HTTPPoolClient::to_json_dp(const DistinguishedPoint& dp) {
    std::stringstream ss;
    ss << "{\"x\":\"" << to_hex(dp.x, 32) << "\","
       << "\"d\":\"" << to_hex(dp.d, 32) << "\","
       << "\"type\":" << (int)dp.type << ","
       << "\"dp_bits\":" << dp.dp_bits << "}";
    return ss.str();
}

std::string HTTPPoolClient::to_json_dps(const std::vector<DistinguishedPoint>& dps) {
    std::stringstream ss;
    ss << "{\"worker\":\"" << json_escape(worker_name_) << "\",\"dps\":[";
    for (size_t i = 0; i < dps.size(); i++) {
        if (i > 0) ss << ",";
        ss << to_json_dp(dps[i]);
    }
    ss << "]}";
    return ss.str();
}

bool HTTPPoolClient::parse_work_json(const std::string& json, WorkAssignment& work) {
    // Simple JSON parsing (not a full parser)
    auto extract_string = [&json](const std::string& key) -> std::string {
        size_t pos = json.find("\"" + key + "\"");
        if (pos == std::string::npos) return "";
        size_t start = json.find("\"", pos + key.size() + 2) + 1;
        size_t end = json.find("\"", start);
        if (start == std::string::npos || end == std::string::npos) return "";
        return json.substr(start, end - start);
    };

    auto extract_int = [&json](const std::string& key) -> uint64_t {
        try {
            size_t pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0;
            size_t start = json.find(":", pos) + 1;
            while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) start++;
            return std::stoull(json.substr(start));
        } catch (const std::exception&) {
            return 0;
        }
    };

    try {
        std::string pub_key = extract_string("public_key");
        std::string range_start = extract_string("range_start");
        std::string range_end = extract_string("range_end");

        if (pub_key.empty() || range_start.empty() || range_end.empty()) {
            return false;
        }

        from_hex(pub_key, work.public_key, 33);
        from_hex(range_start, work.range_start, 32);
        from_hex(range_end, work.range_end, 32);
        work.dp_bits = static_cast<uint32_t>(extract_int("dp_bits"));
        work.work_id = extract_int("work_id");
        work.puzzle_name = extract_string("puzzle_name");

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[HTTPPool] Failed to parse work JSON: " << e.what() << std::endl;
        return false;
    }
}

bool HTTPPoolClient::parse_stats_json(const std::string& json, PoolStats& stats) {
    auto extract_int = [&json](const std::string& key) -> uint64_t {
        try {
            size_t pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0;
            size_t start = json.find(":", pos) + 1;
            while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) start++;
            return std::stoull(json.substr(start));
        } catch (const std::exception&) {
            return 0;
        }
    };

    auto extract_double = [&json](const std::string& key) -> double {
        try {
            size_t pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0.0;
            size_t start = json.find(":", pos) + 1;
            while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) start++;
            return std::stod(json.substr(start));
        } catch (const std::exception&) {
            return 0.0;
        }
    };

    try {
        stats.total_dps = extract_int("total_dps");
        stats.your_dps = extract_int("your_dps");
        stats.your_share = extract_double("your_share");
        stats.connected_workers = extract_int("connected_workers");
        stats.pool_speed = extract_double("pool_speed");
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[HTTPPool] Failed to parse stats JSON: " << e.what() << std::endl;
        return false;
    }
}

std::string HTTPPoolClient::to_hex(const uint8_t* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; i++) {
        ss << std::setw(2) << (int)data[i];
    }
    return ss.str();
}

bool HTTPPoolClient::from_hex(const std::string& hex, uint8_t* data, size_t len) {
    if (hex.size() != len * 2) {
        return false;
    }
    for (size_t i = 0; i < len; i++) {
        std::string byte_str = hex.substr(i * 2, 2);
        data[i] = static_cast<uint8_t>(std::stoul(byte_str, nullptr, 16));
    }
    return true;
}

} // namespace pool
} // namespace collider
