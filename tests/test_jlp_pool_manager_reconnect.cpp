// test_jlp_pool_manager_reconnect.cpp
//
// PoolManager-level reconnect tests for the highest-risk supervisor logic in
// src/pool/pool_manager.cpp. These drive the REAL supervisor thread + a real
// JLPPoolClient against a mock TCP server, so a regression in the reconnect
// state machine fails the test instead of slipping through.
//
// Coverage (adversarial review H3, the A-blocker). These paths had ZERO
// tests:
//
//   1. stale_work_id_purge: buffer BufferedDP entries tagged work_id A while
//      disconnected, then reconnect to a mock that assigns a DIFFERENT
//      work_id B. The reconnect drain (pool_manager.cpp:1158-1218) must PURGE
//      all N buffered DPs (their A-target x/d can never be accepted under B)
//      and REPLAY zero. Asserts reconnect_purged_total() == N and
//      reconnect_replayed_total() == 0. FAILS if the purge guard regresses to
//      blindly replaying stale-work_id DPs (which earns server stale-work_id
//      strikes -> force-close -> hot reconnect loop, the v1.5.4 stranding bug).
//
//   2. churn_circuit_breaker: drive kMaxChurnReconnects back-to-back sub-30s
//      sessions on the SAME work_id. The supervisor (pool_manager.cpp:936-977)
//      must trip supervisor_gave_up_ rather than hot-loop forever. Asserts
//      reconnect_supervisor_gave_up() becomes true. FAILS if the churn counter
//      stops tripping (e.g. someone resets churn_count on a non-sustained
//      session or drops the same-work_id guard).
//
// The mock server here is frame-aware (it parses the KANG header type byte off
// the client stream) so it can complete the AUTH_OK + WORK_ASN handshake the
// supervisor expects, assign a per-connection work_id, and hold or close a
// session on command. This is a superset of the byte-counting mock in
// test_jlp_pool_reconnect.cpp; kept self-contained on purpose to avoid linkage
// gymnastics, matching the established convention in that file.

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    typedef SOCKET sock_t;
    static const sock_t INVALID_SOCK_T = INVALID_SOCKET;
    #define CLOSE_SOCK closesocket
    static int last_sock_err() { return WSAGetLastError(); }
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <errno.h>
    typedef int sock_t;
    static const sock_t INVALID_SOCK_T = -1;
    #define CLOSE_SOCK ::close
    static int last_sock_err() { return errno; }
#endif

#include "pool/pool_manager.hpp"
#include "pool/pool_client.hpp"
#include "pool/jlp_pool_client.hpp"          // JLPServerConfig wire layout
#include "pool/jlp_wire_generated.hpp"       // jlp_wire::PROTOCOL_VERSION, MessageType

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

using namespace collider::pool;
using namespace std::chrono_literals;

namespace {

#ifdef _WIN32
struct WSAGuard {
    WSAGuard()  { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); }
    ~WSAGuard() { WSACleanup(); }
};
#endif

// Wire frame type bytes (from jlp_wire_generated.hpp MessageType). Duplicated
// as named constants so the mock reads as protocol, not magic numbers.
constexpr uint8_t TYPE_AUTH     = 0x01;
constexpr uint8_t TYPE_AUTH_OK  = 0x02;
constexpr uint8_t TYPE_WORK_REQ = 0x10;
constexpr uint8_t TYPE_WORK_ASN = 0x11;

// Mock must stamp the same PROTOCOL_VERSION the client compiled against, or
// the client rejects the frame as a version mismatch. Track the constant
// rather than a literal so a future wire bump cannot strand this test.
constexpr uint8_t MOCK_PROTOCOL_VERSION =
    static_cast<uint8_t>(collider::pool::jlp_wire::PROTOCOL_VERSION);

std::vector<uint8_t> build_frame(uint8_t type, const void* payload, uint16_t len) {
    std::vector<uint8_t> out;
    out.reserve(8 + len);
    out.push_back('K'); out.push_back('A'); out.push_back('N'); out.push_back('G');
    out.push_back(type);
    out.push_back(MOCK_PROTOCOL_VERSION);
    out.push_back(static_cast<uint8_t>(len & 0xFF));
    out.push_back(static_cast<uint8_t>(len >> 8));
    if (payload && len > 0) {
        const uint8_t* p = static_cast<const uint8_t*>(payload);
        out.insert(out.end(), p, p + len);
    }
    return out;
}

// Build a valid 126-byte WORK_ASN payload (JLPServerConfig wire layout) for a
// given work_id. dp_bits is kept inside the client's accepted 8..32 window so
// handle_work_asn does not reject the assignment. kangaroo_type=1 (TAME_ONLY)
// is a legal pool assignment. range/pubkey bytes are arbitrary but non-zero.
std::vector<uint8_t> build_work_asn(uint64_t work_id) {
    JLPServerConfig cfg{};
    cfg.public_key[0] = 0x02;  // compressed pubkey prefix
    for (int i = 1; i < 33; ++i) cfg.public_key[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 32; ++i) {
        cfg.range_start[i] = 0x00;
        cfg.range_end[i]   = (i == 31) ? 0xFF : 0x00;  // range_end > range_start
    }
    cfg.dp_bits        = 24;          // inside 8..32
    cfg.work_id        = work_id;
    cfg.kangaroo_type  = 1;           // TAME_ONLY
    cfg.start_offset_a = 0;
    cfg.start_offset_b = 1;
    static_assert(sizeof(JLPServerConfig) == 126,
                  "WORK_ASN payload must be 126 bytes on the wire");
    std::vector<uint8_t> out(sizeof(JLPServerConfig));
    std::memcpy(out.data(), &cfg, sizeof(JLPServerConfig));
    return out;
}

// --- frame-aware mock server ------------------------------------------------
//
// Per accepted connection (indexed from 1), the run() thread:
//   - parses 8-byte headers + payloads off the client stream;
//   - on AUTH (0x01): replies AUTH_OK (zero payload), then (unless work is
//     gated) pushes a WORK_ASN for this connection's assigned work_id;
//   - on WORK_REQ (0x10): replies WORK_ASN for this connection's work_id (the
//     supervisor reconnect path sends an explicit WORK_REQ post-AUTH).
//
// close_current_client() ends the active session. Two knobs control behavior:
//   - work_id_for_connection(n): the work_id assigned on the n-th connection.
//   - gate_work_after_connection_: if >0, connections strictly after that
//     index withhold the post-AUTH WORK_ASN until release_gated_work() is
//     called, AND withhold the WORK_REQ reply too. This lets a test deposit
//     buffered DPs (tagged the prior work_id) before the new assignment lands.
class MockPoolServer {
public:
    MockPoolServer() {
        listen_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_ == INVALID_SOCK_T) std::abort();
#ifndef _WIN32
        int one = 1;
        setsockopt(listen_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port        = 0;
        if (::bind(listen_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::fprintf(stderr, "[mock] bind: err=%d\n", last_sock_err());
            std::abort();
        }
        sockaddr_in bound{};
        socklen_t blen = sizeof(bound);
        ::getsockname(listen_, reinterpret_cast<sockaddr*>(&bound), &blen);
        port_ = ntohs(bound.sin_port);
        ::listen(listen_, 8);
        thread_ = std::thread(&MockPoolServer::run, this);
    }

    ~MockPoolServer() { stop(); }

    uint16_t port() const { return port_; }

    // Assign work_id per connection index. Index 1 = first connect; each
    // supervisor reconnect bumps the index. base_work_id_ + (n-1) gives a
    // distinct id per connection unless same_work_id_ pins them all equal.
    void set_base_work_id(uint64_t id)        { base_work_id_.store(id); }
    void set_same_work_id(bool same)          { same_work_id_.store(same); }
    void set_gate_work_after(int connection)  { gate_work_after_.store(connection); }

    uint64_t work_id_for_connection(int n) const {
        if (same_work_id_.load()) return base_work_id_.load();
        return base_work_id_.load() + static_cast<uint64_t>(n - 1);
    }

    int clients_accepted() {
        std::lock_guard<std::mutex> lk(mu_);
        return clients_accepted_;
    }

    bool wait_for_nth_client(int n, std::chrono::milliseconds timeout = 5000ms) {
        std::unique_lock<std::mutex> lk(mu_);
        return cv_.wait_for(lk, timeout,
                            [&] { return clients_accepted_ >= n || stopped_; })
               && clients_accepted_ >= n;
    }

    // Release a gated WORK_ASN: the run() thread, parked after AUTH_OK on a
    // gated connection, sends the assignment for the current connection.
    void release_gated_work() {
        {
            std::lock_guard<std::mutex> lk(gate_mu_);
            gate_released_ = true;
        }
        gate_cv_.notify_all();
    }

    void close_current_client() {
        std::lock_guard<std::mutex> lk(client_mu_);
        if (current_client_ != INVALID_SOCK_T) {
#ifdef _WIN32
            ::shutdown(current_client_, SD_BOTH);
#else
            ::shutdown(current_client_, SHUT_RDWR);
#endif
            CLOSE_SOCK(current_client_);
            current_client_ = INVALID_SOCK_T;
        }
    }

    void stop() {
        if (stopped_.exchange(true)) return;
        if (listen_ != INVALID_SOCK_T) { CLOSE_SOCK(listen_); listen_ = INVALID_SOCK_T; }
        close_current_client();
        cv_.notify_all();
        gate_cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    bool send_all(sock_t c, const std::vector<uint8_t>& bytes) {
        size_t total = bytes.size(), sent = 0;
        const char* p = reinterpret_cast<const char*>(bytes.data());
        while (sent < total) {
            int n = ::send(c, p + sent, static_cast<int>(total - sent), 0);
            if (n <= 0) return false;
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    void run() {
        while (!stopped_) {
            sockaddr_in peer{};
            socklen_t plen = sizeof(peer);
            sock_t c = ::accept(listen_, reinterpret_cast<sockaddr*>(&peer), &plen);
            if (c == INVALID_SOCK_T) break;  // listen socket closed

            // Short recv timeout so the per-connection loop can poll stopped_
            // without depending on a teardown thread closing the socket out
            // from under a blocked recv (Windows closesocket-during-recv UB;
            // see test_jlp_pool_reconnect.cpp::MockJlpServer for the writeup).
#ifdef _WIN32
            DWORD rt = 50;
            setsockopt(c, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&rt), sizeof(rt));
#else
            struct timeval tv; tv.tv_sec = 0; tv.tv_usec = 50000;
            setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
            int conn_index;
            {
                std::lock_guard<std::mutex> lk(client_mu_);
                if (current_client_ != INVALID_SOCK_T) {
#ifdef _WIN32
                    ::shutdown(current_client_, SD_BOTH);
#else
                    ::shutdown(current_client_, SHUT_RDWR);
#endif
                    CLOSE_SOCK(current_client_);
                }
                current_client_ = c;
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                conn_index = ++clients_accepted_;
            }
            // Reset the per-connection gate latch.
            {
                std::lock_guard<std::mutex> lk(gate_mu_);
                gate_released_ = false;
            }
            cv_.notify_all();

            handle_connection(c, conn_index);

            {
                std::lock_guard<std::mutex> lk(client_mu_);
                if (current_client_ == c) {
#ifdef _WIN32
                    ::shutdown(c, SD_BOTH);
#else
                    ::shutdown(c, SHUT_RDWR);
#endif
                    CLOSE_SOCK(c);
                    current_client_ = INVALID_SOCK_T;
                }
            }
        }
    }

    // Parse frames off the client stream and respond. Returns when the peer
    // disconnects or we are stopped.
    void handle_connection(sock_t c, int conn_index) {
        std::vector<uint8_t> rx;
        std::vector<char> tmp(4096);
        const bool gated =
            (gate_work_after_.load() > 0 && conn_index > gate_work_after_.load());
        const uint64_t wid = work_id_for_connection(conn_index);

        while (!stopped_) {
            int n = ::recv(c, tmp.data(), static_cast<int>(tmp.size()), 0);
            if (n == 0) break;  // graceful FIN
            if (n < 0) {
                int err = last_sock_err();
#ifdef _WIN32
                if (err == WSAETIMEDOUT || err == WSAEWOULDBLOCK) continue;
#else
                if (err == EAGAIN || err == EWOULDBLOCK) continue;
#endif
                break;
            }
            rx.insert(rx.end(), tmp.data(), tmp.data() + n);

            // Drain whole frames out of rx.
            size_t off = 0;
            while (rx.size() - off >= 8) {
                // Header: magic[4] type flags payload_size(LE u16)
                const uint8_t type = rx[off + 4];
                const uint16_t plen =
                    static_cast<uint16_t>(rx[off + 6]) |
                    (static_cast<uint16_t>(rx[off + 7]) << 8);
                if (rx.size() - off < static_cast<size_t>(8) + plen) break;  // wait for body
                off += static_cast<size_t>(8) + plen;

                if (type == TYPE_AUTH) {
                    if (!send_all(c, build_frame(TYPE_AUTH_OK, nullptr, 0))) return;
                    if (gated) {
                        // Withhold the post-AUTH assignment until the test
                        // signals (so it can buffer DPs tagged the prior id).
                        std::unique_lock<std::mutex> lk(gate_mu_);
                        gate_cv_.wait_for(lk, 5000ms,
                            [&] { return gate_released_ || stopped_; });
                        if (stopped_) return;
                    }
                    auto asn = build_work_asn(wid);
                    if (!send_all(c, build_frame(TYPE_WORK_ASN, asn.data(),
                                                 static_cast<uint16_t>(asn.size()))))
                        return;
                } else if (type == TYPE_WORK_REQ) {
                    // Supervisor reconnect path sends an explicit WORK_REQ
                    // after AUTH_OK and BLOCKS in request_work() until the
                    // WORK_ASN lands. On a gated connection we wait for the
                    // release so the test can buffer first.
                    if (gated) {
                        std::unique_lock<std::mutex> lk(gate_mu_);
                        gate_cv_.wait_for(lk, 5000ms,
                            [&] { return gate_released_ || stopped_; });
                        if (stopped_) return;
                    }
                    auto asn = build_work_asn(wid);
                    if (!send_all(c, build_frame(TYPE_WORK_ASN, asn.data(),
                                                 static_cast<uint16_t>(asn.size()))))
                        return;
                }
                // Other frames (DP_BATCH_V2, PING, STATS_REQ) are ignored:
                // the tests only need the AUTH + WORK handshake to complete.
            }
            if (off > 0) rx.erase(rx.begin(), rx.begin() + off);
        }
    }

    sock_t   listen_         = INVALID_SOCK_T;
    sock_t   current_client_ = INVALID_SOCK_T;
    uint16_t port_           = 0;

    std::atomic<uint64_t> base_work_id_{100};
    std::atomic<bool>     same_work_id_{false};
    std::atomic<int>      gate_work_after_{0};

    std::thread       thread_;
    std::atomic<bool> stopped_{false};

    std::mutex              client_mu_;
    std::mutex              mu_;
    std::condition_variable cv_;
    int                     clients_accepted_ = 0;

    std::mutex              gate_mu_;
    std::condition_variable gate_cv_;
    bool                    gate_released_ = false;
};

// --- helpers ----------------------------------------------------------------

bool fail(const char* tname, const char* msg) {
    std::fprintf(stderr, "[FAIL] %s: %s\n", tname, msg);
    return false;
}

template <class Pred>
bool wait_for(Pred pred, std::chrono::milliseconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(20ms);
    }
    return pred();
}

PoolConfig make_config(uint16_t port) {
    PoolConfig cfg{};
    cfg.type           = POOL_TYPE_JLP;
    cfg.host           = "127.0.0.1";
    cfg.port           = port;
    cfg.worker_name    = "tb1qtestworkeraddressplaceholderxxxxxxxxxx";
    cfg.password       = "";
    cfg.auto_reconnect = true;     // REQUIRED: spawns the supervisor
    cfg.timeout_ms     = 5000;     // generous so request_work tolerates gating
    cfg.use_tls        = false;
    cfg.verify_cert    = false;
    return cfg;
}

// ---------------------------------------------------------------------------
// 1. stale_work_id_purge
//
//    - connect (conn#1) -> work_id A=100;
//    - close conn#1 server-side, wait until is_connected() == false;
//    - buffer N DPs via submit_dp() while disconnected (tagged A=100);
//    - the supervisor reconnects (conn#2); the mock assigns work_id B=101
//      (gated so the buffering above is guaranteed to land first);
//    - the reconnect drain must purge all N (B != A) and replay zero.
// ---------------------------------------------------------------------------
bool test_stale_work_id_purge() {
    constexpr int N = 8;
    MockPoolServer server;
    server.set_base_work_id(100);
    server.set_same_work_id(false);   // conn#1 -> 100, conn#2 -> 101, ...
    server.set_gate_work_after(1);    // conn#2+ withholds WORK_ASN until released

    PoolManager mgr;
    mgr.set_config(make_config(server.port()));

    if (!mgr.connect())
        return fail("stale_work_id_purge", "initial connect() failed");
    if (!server.wait_for_nth_client(1))
        return fail("stale_work_id_purge", "server never accepted conn#1");

    // Wait for work_id A=100 to be assigned (work_callback sets current_work_id_).
    WorkAssignment w{};
    if (!wait_for([&] { return mgr.get_work(w) && w.work_id == 100; }, 5000ms))
        return fail("stale_work_id_purge", "conn#1 work_id A=100 never assigned");

    // Drop the session. is_connected() should fall to false; the supervisor
    // will begin reconnecting on its next probe, but conn#2's WORK_ASN is
    // gated so current_work_id_ stays at A=100 until we release it below.
    server.close_current_client();
    if (!wait_for([&] { return !mgr.is_connected(); }, 5000ms))
        return fail("stale_work_id_purge", "is_connected() never fell after close");

    // Buffer N DPs while disconnected. Each is tagged with the active
    // work_id (still A=100) inside PoolManager::submit_dp.
    for (int i = 0; i < N; ++i) {
        DistinguishedPoint dp{};
        dp.x[0]   = static_cast<uint8_t>(i);
        dp.d[0]   = static_cast<uint8_t>(0x80 + i);
        dp.type   = 0;       // tame
        dp.dp_bits = 24;
        mgr.submit_dp(dp);
    }
    if (mgr.reconnect_buffered_total() < static_cast<uint64_t>(N))
        return fail("stale_work_id_purge",
                    "DPs were not buffered while disconnected");

    // Wait until the supervisor has driven conn#2 and is parked waiting for
    // the gated WORK_ASN, then release work_id B=101. The reconnect drain
    // runs immediately after request_work() returns.
    if (!server.wait_for_nth_client(2))
        return fail("stale_work_id_purge", "supervisor never reconnected (conn#2)");
    server.release_gated_work();

    // After reconnect completes the drain must have purged all N (A != B) and
    // replayed none.
    if (!wait_for([&] {
            return mgr.reconnect_purged_total() == static_cast<uint64_t>(N);
        }, 6000ms)) {
        std::fprintf(stderr,
            "[FAIL] stale_work_id_purge: purged=%llu (want %d), replayed=%llu\n",
            static_cast<unsigned long long>(mgr.reconnect_purged_total()), N,
            static_cast<unsigned long long>(mgr.reconnect_replayed_total()));
        mgr.disconnect();
        return false;
    }
    if (mgr.reconnect_replayed_total() != 0) {
        std::fprintf(stderr,
            "[FAIL] stale_work_id_purge: replayed=%llu (want 0); stale DPs "
            "were resubmitted under the new work_id\n",
            static_cast<unsigned long long>(mgr.reconnect_replayed_total()));
        mgr.disconnect();
        return false;
    }

    mgr.disconnect();
    return true;
}

// ---------------------------------------------------------------------------
// 2. churn_circuit_breaker
//
//    Drive kMaxChurnReconnects back-to-back short-lived sessions on the SAME
//    work_id by accepting + handshaking each connection, then closing it
//    immediately. The supervisor must trip supervisor_gave_up_ instead of
//    reconnecting forever. We do not close from the test side here: the mock
//    auto-handshakes and then we close each session as soon as it goes live.
//
//    kSustainedSessionMs is 30s; every session here lasts well under that, so
//    each counts as churn. After kMaxChurnReconnects (5) on the same work_id
//    the breaker trips.
// ---------------------------------------------------------------------------
bool test_churn_circuit_breaker() {
    MockPoolServer server;
    server.set_base_work_id(200);
    server.set_same_work_id(true);   // pin every session to work_id 200 -> churn
    server.set_gate_work_after(0);   // no gating; auto-handshake each session

    PoolManager mgr;
    mgr.set_config(make_config(server.port()));

    if (!mgr.connect())
        return fail("churn_circuit_breaker", "initial connect() failed");
    if (!server.wait_for_nth_client(1))
        return fail("churn_circuit_breaker", "server never accepted conn#1");

    // Each iteration: wait for the session to go live (is_connected()), then
    // kill it. The supervisor measures the (short) lifetime on the same
    // work_id and increments churn_count. We perform more than enough cycles
    // to exceed kMaxChurnReconnects; the breaker should trip mid-way.
    //
    // We bound the loop generously and rely on supervisor_gave_up_ to short-
    // circuit. Each cycle: handshake + a sub-second live window + supervisor
    // backoff (reset to ~[500ms,1000ms] because the session DID reach live).
    const auto deadline = std::chrono::steady_clock::now() + 60s;
    int killed = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        if (mgr.reconnect_supervisor_gave_up()) break;
        // Wait for the current session to come up.
        if (wait_for([&] {
                return mgr.is_connected() || mgr.reconnect_supervisor_gave_up();
            }, 5000ms)) {
            if (mgr.reconnect_supervisor_gave_up()) break;
            // Live: kill it immediately to register a sub-30s session.
            server.close_current_client();
            ++killed;
            // Wait for it to drop before looping so we count distinct sessions.
            wait_for([&] {
                return !mgr.is_connected() || mgr.reconnect_supervisor_gave_up();
            }, 5000ms);
        } else {
            // Could not bring a session up; let the supervisor keep trying.
            std::this_thread::sleep_for(100ms);
        }
    }

    const bool gave_up = mgr.reconnect_supervisor_gave_up();
    mgr.disconnect();

    if (!gave_up) {
        std::fprintf(stderr,
            "[FAIL] churn_circuit_breaker: supervisor did not give up after "
            "%d short-lived same-work_id sessions (kMaxChurnReconnects=5)\n",
            killed);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

struct TestCase {
    const char* name;
    bool (*fn)();
};

const TestCase TESTS[] = {
    {"stale_work_id_purge",     test_stale_work_id_purge},
    {"churn_circuit_breaker",   test_churn_circuit_breaker},
};

}  // namespace

int main() {
#ifdef _WIN32
    WSAGuard guard;
#endif
    std::printf("=== JLP pool manager reconnect tests ===\n");
    int failures = 0;
    for (const auto& t : TESTS) {
        std::printf("[ run ] %s\n", t.name);
        if (t.fn()) {
            std::printf("[ ok  ] %s\n", t.name);
        } else {
            std::printf("[FAIL ] %s\n", t.name);
            ++failures;
        }
    }
    std::printf("\n%zu tests, %d failures\n",
                sizeof(TESTS) / sizeof(TESTS[0]), failures);
    return failures == 0 ? 0 : 1;
}
