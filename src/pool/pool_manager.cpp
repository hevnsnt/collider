// pool_manager.cpp - Pool manager implementation

#include "pool_manager.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <regex>

namespace collider {
namespace pool {

// Thread-safe singleton using Meyer's pattern (C++11+)
// Automatically cleaned up at program exit - no memory leak
PoolManager& get_pool_manager() {
    static PoolManager instance;
    return instance;
}

PoolManager::PoolManager()
    : connected_(false)
    , submitted_count_(0)
    , has_work_(false)
{
    start_time_ = std::chrono::steady_clock::now();
}

PoolManager::~PoolManager() {
    disconnect();
}

void PoolManager::set_config(const PoolConfig& config) {
    config_ = config;
}

bool PoolManager::connect() {
    if (connected_) {
        disconnect();
    }

    // Create appropriate client
    if (config_.type == POOL_TYPE_JLP) {
        auto jlp_client = std::make_unique<JLPPoolClient>();
        jlp_client->set_timeout(config_.timeout_ms);
        jlp_client->set_reconnect(config_.auto_reconnect);
        jlp_client->set_debug_mode(config_.debug_mode);
        jlp_client->set_use_tls(config_.use_tls);
        jlp_client->set_verify_cert(config_.verify_cert);
        client_ = std::move(jlp_client);
    } else if (config_.type == POOL_TYPE_HTTP) {
        auto http_client = std::make_unique<HTTPPoolClient>();
        if (!config_.api_key.empty()) {
            http_client->set_api_key(config_.api_key);
        }
        client_ = std::move(http_client);
    } else {
        std::cerr << "[PoolManager] Unknown pool type: " << config_.type << std::endl;
        return false;
    }

    // Set callbacks
    client_->set_solution_callback([this](const uint8_t* key) {
        std::cout << "\n[PoolManager] SOLUTION FOUND BY POOL!" << std::endl;
        if (solution_callback_) {
            solution_callback_(key, config_.worker_name);
        }
    });

    client_->set_work_callback([this](const WorkAssignment& work) {
        std::lock_guard<std::mutex> lock(work_mutex_);
        current_work_ = work;
        has_work_ = true;
        std::cout << "[PoolManager] Received new work: " << work.puzzle_name
                  << " (DP bits: " << work.dp_bits << ")" << std::endl;
    });

    // Connect
    if (!client_->connect(config_.host, config_.port)) {
        return false;
    }

    // Authenticate
    if (!client_->authenticate(config_.worker_name, config_.password)) {
        std::cerr << "[PoolManager] Authentication failed" << std::endl;
        client_->disconnect();
        return false;
    }

    connected_ = true;
    start_time_ = std::chrono::steady_clock::now();
    submitted_count_ = 0;

    std::cout << "[PoolManager] Connected to pool as " << config_.worker_name << std::endl;
    return true;
}

void PoolManager::disconnect() {
    if (client_) {
        client_->disconnect();
        client_.reset();
    }
    connected_ = false;
}

bool PoolManager::is_connected() const {
    return connected_ && client_ && client_->is_connected();
}

bool PoolManager::get_work(WorkAssignment& work) {
    if (!is_connected()) {
        return false;
    }

    // Check if we already have work
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        if (has_work_) {
            work = current_work_;
            return true;
        }
    }

    // Request work from pool
    if (client_->request_work(work)) {
        std::lock_guard<std::mutex> lock(work_mutex_);
        current_work_ = work;
        has_work_ = true;
        return true;
    }

    return false;
}

void PoolManager::submit_dp(const uint8_t* x, const uint8_t* d, uint8_t type, uint32_t dp_bits) {
    DistinguishedPoint dp;
    memcpy(dp.x, x, 32);
    memcpy(dp.d, d, 32);
    dp.type = type;
    dp.dp_bits = dp_bits;
    submit_dp(dp);
}

void PoolManager::submit_dp(const DistinguishedPoint& dp) {
    if (!is_connected()) {
        return;
    }

    client_->submit_dp(dp);
    submitted_count_++;
}

PoolStats PoolManager::get_stats() const {
    if (!client_) {
        return PoolStats{};
    }
    return client_->get_stats();
}

double PoolManager::get_submission_rate() const {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed < 1.0) return 0.0;
    return static_cast<double>(submitted_count_) / elapsed;
}

void PoolManager::report_solution(const uint8_t* private_key) {
    if (!is_connected()) {
        return;
    }

    std::cout << "[PoolManager] Reporting solution to pool..." << std::endl;
    client_->report_solution(private_key);
}

std::string PoolManager::get_status_string() const {
    std::stringstream ss;

    if (!is_connected()) {
        ss << "Disconnected";
        return ss.str();
    }

    PoolStats stats = get_stats();
    ss << "Connected | ";
    ss << "Your DPs: " << stats.your_dps << " | ";
    ss << "Pool DPs: " << stats.total_dps << " | ";
    ss << "Workers: " << stats.connected_workers << " | ";
    ss << "Rate: " << std::fixed << std::setprecision(1) << get_submission_rate() << " DP/s";

    if (stats.your_share > 0) {
        ss << " | Share: " << std::fixed << std::setprecision(2) << (stats.your_share * 100) << "%";
    }

    return ss.str();
}

// Static callback hook for RCKangaroo integration
void PoolManager::dp_callback_hook(void* user_data, const uint8_t* x, const uint8_t* d, uint8_t type) {
    PoolManager* manager = static_cast<PoolManager*>(user_data);
    if (manager && manager->is_connected()) {
        // Get DP bits from current work
        uint32_t dp_bits = 0;
        {
            std::lock_guard<std::mutex> lock(manager->work_mutex_);
            dp_bits = manager->current_work_.dp_bits;
        }
        manager->submit_dp(x, d, type, dp_bits);
    }
}

// Parse pool URL
bool parse_pool_url(const std::string& url, PoolConfig& config) {
    // Patterns:
    // jlp://host:port       - JLP without TLS
    // jlps://host:port      - JLP with TLS
    // http://host:port/path - HTTP without TLS
    // https://host:port/path - HTTP with TLS
    // host:port (defaults to JLP without TLS)

    std::regex url_regex(R"(^(?:([a-z]+)://)?([^:/]+)(?::(\d+))?(/.*)?$)");
    std::smatch match;

    if (!std::regex_match(url, match, url_regex)) {
        return false;
    }

    std::string scheme = match[1].str();
    config.host = match[2].str();
    std::string port_str = match[3].str();
    std::string path = match[4].str();

    // Determine type and TLS from scheme
    config.use_tls = false;
    if (scheme.empty() || scheme == "jlp" || scheme == "kangaroo") {
        config.type = POOL_TYPE_JLP;
        config.use_tls = false;
    } else if (scheme == "jlps" || scheme == "kangaroos") {
        config.type = POOL_TYPE_JLP;
        config.use_tls = true;
    } else if (scheme == "http") {
        config.type = POOL_TYPE_HTTP;
        config.use_tls = false;
    } else if (scheme == "https") {
        config.type = POOL_TYPE_HTTP;
        config.use_tls = true;
    } else {
        return false;
    }

    // Set port
    if (!port_str.empty()) {
        config.port = static_cast<uint16_t>(std::stoi(port_str));
    } else {
        config.port = PoolConfig::default_port(config.type);
    }

    // Set defaults
    config.auto_reconnect = true;
    config.timeout_ms = 5000;
    config.verify_cert = true;  // Verify CA-signed certificates (Let's Encrypt, etc.)

    return true;
}

} // namespace pool
} // namespace collider
