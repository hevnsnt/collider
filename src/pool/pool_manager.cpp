// pool_manager.cpp - Pool manager implementation

#include "pool_manager.hpp"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <regex>
#include <sstream>
#include <iomanip>

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
    stop_supervisor();
    disconnect();
}

void PoolManager::set_config(const PoolConfig& config) {
    config_ = config;
}

bool PoolManager::connect() {
    if (connected_) {
        disconnect();
    }

    // Create appropriate client.
    // NOTE: HTTP pool path was DELETED in Wave 4 (Track D security audit, finding D-C1).
    // The HTTP client silently downgraded https:// URLs to plaintext, leaking the
    // worker's Bitcoin address (= payout address) and API tokens in the clear. JLP+TLS
    // (jlps://) is the only supported pool protocol going forward.
    if (config_.type == POOL_TYPE_JLP) {
        auto jlp_client = std::make_unique<JLPPoolClient>();
        jlp_client->set_timeout(config_.timeout_ms);
        jlp_client->set_reconnect(config_.auto_reconnect);
        jlp_client->set_debug_mode(config_.debug_mode);
        jlp_client->set_use_tls(config_.use_tls);
        jlp_client->set_verify_cert(config_.verify_cert);
        client_ = std::move(jlp_client);
    } else if (config_.type == POOL_TYPE_HTTP) {
        std::cerr << "[PoolManager] HTTP pool deprecated; use jlp:// or jlps://"
                  << std::endl;
        std::cerr << "[PoolManager]   The HTTP pool transport was removed in the "
                     "2026-05-04 security review (silent plaintext credential leak)."
                  << std::endl;
        return false;
    } else {
        std::cerr << "[PoolManager] Unknown pool type: '" << config_.type
                  << "'. Only 'jlp' is supported (use jlp:// or jlps:// URLs)."
                  << std::endl;
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
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            current_work_ = work;
            has_work_ = true;
        }
        // Atomic snapshot of dp_bits so dp_callback_hook (kernel-callback
        // hot path) doesn't have to take work_mutex_ to read it. Pool
        // assignments don't change dp_bits mid-chunk; if a future
        // protocol does, the snapshot lags by at most one DP submit.
        current_dp_bits_.store(work.dp_bits, std::memory_order_release);
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

    // Spawn the reconnect supervisor. The receiver loop in JLPPoolClient
    // exits cleanly on transient network errors and waits for an
    // "external supervisor" to call connect() / authenticate() -- this
    // is that supervisor.
    if (config_.auto_reconnect) {
        start_supervisor();
    }
    return true;
}

void PoolManager::disconnect() {
    stop_supervisor();
    if (client_) {
        client_->disconnect();
        client_.reset();
    }
    connected_ = false;
    current_dp_bits_.store(0, std::memory_order_release);
}

bool PoolManager::is_connected() const {
    return connected_ && client_ && client_->is_connected();
}

bool PoolManager::get_work(WorkAssignment& work) {
    if (!is_connected()) {
        return false;
    }

    // Fast path: cached work assignment from the work-callback.
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        if (has_work_) {
            work = current_work_;
            return true;
        }
    }

    // Slow path: explicit request to the pool. The previous version held
    // work_mutex_ across this blocking I/O call -- any concurrent
    // submit_dp from the kernel-callback path was forced to wait for the
    // entire request_work round-trip (multiple seconds at TLS handshake
    // + receive timeout). Now we make the call WITHOUT the lock; only
    // the (X, Y) state-update under the result is locked.
    if (client_->request_work(work)) {
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            current_work_ = work;
            has_work_ = true;
        }
        current_dp_bits_.store(work.dp_bits, std::memory_order_release);
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
        // Queue is decoupled from connection state, but we should not
        // accept DPs while disconnected -- the supervisor may take a
        // while to come back, by which point the work_id might have
        // rolled and these DPs are stale.
        dropped_count_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    if (client_->submit_dp(dp)) {
        submitted_count_.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Queue full -- the JLP client's bounded MAX_DP_QUEUE_SIZE
        // backpressure rejected this. Pre-1.4 we silently incremented
        // submitted_count_, hiding the drop from operators. Now we
        // count + emit a rate-limited stderr warning.
        const uint64_t dropped =
            dropped_count_.fetch_add(1, std::memory_order_relaxed) + 1;
        // Warn every Nth drop to avoid log spam during sustained backpressure.
        constexpr uint64_t kWarnEvery = 1024;
        if ((dropped & (kWarnEvery - 1)) == 0) {
            std::fprintf(stderr,
                "[pool] WARN DP submission queue full -- dropped %llu DP(s) "
                "since startup. The pool sender thread is not draining fast "
                "enough; check network latency or DP rate.\n",
                static_cast<unsigned long long>(dropped));
        }
    }
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

// Static callback hook for RCKangaroo integration. Reads dp_bits from
// the atomic snapshot rather than locking work_mutex_ -- the kernel
// callback fires per-DP and contended locking would serialize an
// otherwise-parallel hot path.
void PoolManager::dp_callback_hook(void* user_data, const uint8_t* x, const uint8_t* d, uint8_t type) {
    PoolManager* manager = static_cast<PoolManager*>(user_data);
    if (manager && manager->is_connected()) {
        const uint32_t dp_bits =
            manager->current_dp_bits_.load(std::memory_order_acquire);
        manager->submit_dp(x, d, type, dp_bits);
    }
}

// ---------------------------------------------------------------------------
// Reconnect supervisor
//
// JLPPoolClient::receiver_loop() exits cleanly when its socket dies or
// the server tears down the session. The header at jlp_pool_client.cpp:897
// says "external supervisor must call connect() / authenticate() to
// recover" -- and prior to v1.4.0 there was no such supervisor, so
// transient failures silently killed the worker.
//
// This thread polls is_connected() and, when it sees the receiver has
// exited, drives a full reconnect: disconnect() + connect() + authenticate()
// with jittered exponential backoff. After kMaxReconnectAttempts
// consecutive failures, supervisor_gave_up_ is set and the host loop
// can react. The receiver thread inside the client is a fresh thread
// per connect() because connect() calls replace_thread().
// ---------------------------------------------------------------------------

void PoolManager::start_supervisor() {
    // Idempotent: only one supervisor at a time.
    if (supervisor_thread_.joinable()) return;
    supervisor_stop_.store(false, std::memory_order_release);
    supervisor_gave_up_.store(false, std::memory_order_release);
    supervisor_thread_ = std::thread([this]() { this->supervisor_loop(); });
}

void PoolManager::stop_supervisor() {
    supervisor_stop_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(supervisor_mutex_);
        supervisor_cv_.notify_all();
    }
    if (supervisor_thread_.joinable()) {
        supervisor_thread_.join();
    }
}

void PoolManager::supervisor_loop() {
    // Backoff state. Reset to kInitialBackoffMs on every successful
    // reconnect; doubles up to kMaxBackoffMs on each consecutive
    // failure. Jitter avoids thundering-herd against the pool when
    // a fleet of workers reconnects together.
    uint32_t backoff_ms = kInitialBackoffMs;
    uint32_t consecutive_failures = 0;
    std::random_device rd;
    std::mt19937 rng(rd());

    while (!supervisor_stop_.load(std::memory_order_acquire)) {
        // Sleep on the condition variable so disconnect() / shutdown
        // can wake us promptly. Wake either every 500ms (probe) or
        // when the stop flag flips.
        {
            std::unique_lock<std::mutex> lock(supervisor_mutex_);
            supervisor_cv_.wait_for(lock, std::chrono::milliseconds(500),
                [this]() { return supervisor_stop_.load(std::memory_order_acquire); });
        }
        if (supervisor_stop_.load(std::memory_order_acquire)) break;

        // Healthy: do nothing. is_connected() reads connected_ (the host
        // sentinel) AND the client's own connection flag, so a receiver
        // thread that exited will flip the second to false.
        if (is_connected()) {
            backoff_ms = kInitialBackoffMs;
            consecutive_failures = 0;
            continue;
        }

        // Receiver has exited. Drive a full reconnect.
        if (consecutive_failures >= kMaxReconnectAttempts) {
            std::cerr << "[PoolManager] Reconnect supervisor: "
                      << kMaxReconnectAttempts
                      << " consecutive attempts failed; giving up. "
                      << "The pool client will remain disconnected; "
                      << "host code can re-call connect() to retry."
                      << std::endl;
            supervisor_gave_up_.store(true, std::memory_order_release);
            return;
        }

        // Jitter: pick uniform delay in [backoff/2, backoff].
        std::uniform_int_distribution<uint32_t> jitter(backoff_ms / 2, backoff_ms);
        const uint32_t delay = jitter(rng);
        std::cerr << "[PoolManager] Reconnect attempt " << (consecutive_failures + 1)
                  << "/" << kMaxReconnectAttempts
                  << " in " << (delay / 1000.0) << "s..." << std::endl;
        // Sleep with periodic wake-on-stop so shutdown is responsive.
        for (uint32_t slept = 0; slept < delay && !supervisor_stop_.load(std::memory_order_acquire);
             slept += 100)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (supervisor_stop_.load(std::memory_order_acquire)) break;

        reconnect_attempts_.fetch_add(1, std::memory_order_relaxed);

        // Tear down the old client (if still around) and construct a
        // fresh one. Using the public connect() helper would race with
        // ourselves; we follow its body inline.
        if (client_) {
            client_->disconnect();
            client_.reset();
        }
        connected_.store(false);

        // Recreate the JLP client (same path as PoolManager::connect).
        if (config_.type != POOL_TYPE_JLP) {
            std::cerr << "[PoolManager] Supervisor refusing to reconnect non-JLP "
                      << "config; HTTP path is deleted." << std::endl;
            supervisor_gave_up_.store(true);
            return;
        }
        auto fresh = std::make_unique<JLPPoolClient>();
        fresh->set_timeout(config_.timeout_ms);
        fresh->set_reconnect(config_.auto_reconnect);
        fresh->set_debug_mode(config_.debug_mode);
        fresh->set_use_tls(config_.use_tls);
        fresh->set_verify_cert(config_.verify_cert);
        client_ = std::move(fresh);

        client_->set_solution_callback([this](const uint8_t* key) {
            std::cout << "\n[PoolManager] SOLUTION FOUND BY POOL!" << std::endl;
            if (solution_callback_) {
                solution_callback_(key, config_.worker_name);
            }
        });
        client_->set_work_callback([this](const WorkAssignment& work) {
            {
                std::lock_guard<std::mutex> lock(work_mutex_);
                current_work_ = work;
                has_work_ = true;
            }
            current_dp_bits_.store(work.dp_bits, std::memory_order_release);
        });

        if (!client_->connect(config_.host, config_.port)) {
            ++consecutive_failures;
            backoff_ms = std::min(kMaxBackoffMs, backoff_ms * 2);
            continue;
        }
        if (!client_->authenticate(config_.worker_name, config_.password)) {
            client_->disconnect();
            ++consecutive_failures;
            backoff_ms = std::min(kMaxBackoffMs, backoff_ms * 2);
            continue;
        }

        connected_.store(true);
        reconnect_successes_.fetch_add(1, std::memory_order_relaxed);
        backoff_ms = kInitialBackoffMs;
        consecutive_failures = 0;
        std::cout << "[PoolManager] Reconnected to pool as "
                  << config_.worker_name << std::endl;
    }
}

// Parse pool URL
//
// Supported schemes (Wave 4 security review):
//   jlp://host:port    - JLP without TLS (closed-test only; will warn)
//   jlps://host:port   - JLP with TLS (REQUIRED for any external/Internet pool)
//   host:port          - shorthand, defaults to jlp://
//
// REJECTED schemes:
//   http://, https://  - HTTP pool transport was deleted in Wave 4 (D-C1: silent
//                        credential leak). Hard-fails with a clear migration message.
bool parse_pool_url(const std::string& url, PoolConfig& config) {
    std::regex url_regex(R"(^(?:([a-z]+)://)?([^:/]+)(?::(\d+))?(/.*)?$)");
    std::smatch match;

    if (!std::regex_match(url, match, url_regex)) {
        std::cerr << "[PoolManager] Malformed pool URL: '" << url << "'"
                  << std::endl;
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
    } else if (scheme == "http" || scheme == "https") {
        // HTTP pool path was DELETED in the 2026-05-04 security review (D-C1).
        // The previous client silently downgraded https:// to plaintext TCP, leaking
        // the worker name (= Bitcoin payout address) and any auth token.
        std::cerr << "[PoolManager] HTTP pool deprecated; use jlp:// or jlps:// "
                     "(input was: " << url << ")" << std::endl;
        std::cerr << "[PoolManager]   The HTTP transport was removed because the "
                     "previous implementation leaked credentials." << std::endl;
        std::cerr << "[PoolManager]   Migration: replace 'http://host:port' with "
                     "'jlp://host:17403' or, for TLS, 'jlps://host:17403'."
                  << std::endl;
        return false;
    } else {
        std::cerr << "[PoolManager] Unknown URL scheme: '" << scheme
                  << "' (supported: jlp://, jlps://)" << std::endl;
        return false;
    }

    // Set port with validation. std::stoi may throw on overflow (D-L1); guard it.
    if (!port_str.empty()) {
        int port_value = 0;
        try {
            port_value = std::stoi(port_str);
        } catch (const std::exception& e) {
            std::cerr << "[PoolManager] Invalid port '" << port_str
                      << "': " << e.what() << std::endl;
            return false;
        }
        if (port_value < 1 || port_value > 65535) {
            std::cerr << "[PoolManager] Invalid port: " << port_value
                      << " (must be 1-65535)" << std::endl;
            return false;
        }
        config.port = static_cast<uint16_t>(port_value);
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
