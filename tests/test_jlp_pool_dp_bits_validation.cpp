// test_jlp_pool_dp_bits_validation.cpp
//
// Regression test for the S1 security finding (v1.4.2 final-validation):
// a malicious or buggy pool server can send WORK_ASN with dp_bits outside
// the supportable window [8..32]. The client previously accepted the value
// verbatim and handed it to the kangaroo solver, which then burned GPU
// cycles forever without ever emitting a distinguished point (at 255
// leading-zero bits the per-step probability is 2^-255).
//
// What this test verifies:
//   1. A WORK_ASN with dp_bits=255 is rejected (work callback never fires).
//   2. The client disconnects from the misbehaving server after the
//      rejection (is_connected() transitions to false).
//   3. Valid dp_bits values inside [8..32] still flow through (sanity
//      check; the existing test_jlp_pool_protocol.cpp tests cover this
//      too but we re-exercise it here so a regression that flips the
//      bound the wrong way is caught by this file alone).
//
// Mock-server scaffolding mirrors the pattern used by
// tests/test_jlp_pool_protocol.cpp (same MockJlpServer shape, same
// drive_auth_ok preamble). Kept locally rather than factored into a
// shared header because the protocol-test mocks are deliberately
// minimal and the duplication is small (~150 lines).

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
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <cerrno>     // EAGAIN / EWOULDBLOCK for the SO_RCVTIMEO poll
    typedef int sock_t;
    static const sock_t INVALID_SOCK_T = -1;
    #define CLOSE_SOCK ::close
#endif

#include "pool/jlp_pool_client.hpp"
#include "pool/pool_client.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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

constexpr uint8_t TYPE_AUTH_OK   = 0x02;
constexpr uint8_t TYPE_WORK_ASN  = 0x11;
constexpr uint8_t MOCK_PROTOCOL_VERSION = 2;

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

// Lightweight mock server. Listens on 127.0.0.1:OS-assigned-port, accepts
// one client, lets the test drive the conversation byte-by-byte.
//
// Differs from the more elaborate MockJlpServer in test_jlp_pool_protocol.cpp
// only by being simpler: no PING kicker, no PONG filtering. The dp_bits
// validation test does not exercise post-AUTH long-running send/recv races
// (which is what the kicker exists to defeat), so the simpler shape suffices.
class MockServer {
public:
    MockServer() {
        listen_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_ == INVALID_SOCK_T) std::abort();
#ifndef _WIN32
        int one = 1;
        setsockopt(listen_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        if (::bind(listen_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) std::abort();
        sockaddr_in bound{};
        socklen_t blen = sizeof(bound);
        if (::getsockname(listen_, reinterpret_cast<sockaddr*>(&bound), &blen) < 0) std::abort();
        port_ = ntohs(bound.sin_port);
        if (::listen(listen_, 1) < 0) std::abort();
        thread_ = std::thread(&MockServer::run, this);
    }
    ~MockServer() { stop(); }

    uint16_t port() const { return port_; }

    std::vector<uint8_t> wait_recv(size_t n,
                                   std::chrono::milliseconds timeout = 5000ms) {
        std::unique_lock<std::mutex> lk(buf_mu_);
        buf_cv_.wait_for(lk, timeout, [&] {
            return rx_.size() >= n || stopped_;
        });
        if (rx_.size() < n) return {};
        std::vector<uint8_t> out(rx_.begin(), rx_.begin() + n);
        rx_.erase(rx_.begin(), rx_.begin() + n);
        return out;
    }

    bool wait_for_client(std::chrono::milliseconds timeout = 2000ms) {
        std::unique_lock<std::mutex> lk(client_mu_);
        return client_cv_.wait_for(lk, timeout,
                                   [&] { return client_ready_ || stopped_; })
               && client_ready_;
    }

    bool send_frame(uint8_t type, const void* payload, uint16_t len) {
        if (!wait_for_client()) return false;
        auto bytes = build_frame(type, payload, len);
        std::lock_guard<std::mutex> lk(client_mu_);
        if (client_ == INVALID_SOCK_T) return false;
        const char* p = reinterpret_cast<const char*>(bytes.data());
        size_t total = bytes.size();
        size_t sent = 0;
        while (sent < total) {
            int n = ::send(client_, p + sent, static_cast<int>(total - sent), 0);
            if (n <= 0) return false;
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    void stop() {
        if (stopped_.exchange(true)) return;
        if (listen_ != INVALID_SOCK_T) { CLOSE_SOCK(listen_); listen_ = INVALID_SOCK_T; }
        // shutdown(SD_BOTH) before closesocket: same Windows-UB fix
        // applied to MockJlpServer in test_jlp_pool_protocol.cpp. The
        // accept thread spawned by run() spends most of its life parked
        // in ::recv() on client_; a bare closesocket from this teardown
        // path while recv is in flight is documented unspecified
        // behaviour on Winsock and reliably reproduced process-exit
        // access violations. shutdown signals graceful close so recv
        // returns 0 before the handle is freed. POSIX shutdown(SHUT_RDWR)
        // has the same semantics, so the call works unmodified on
        // Linux + macOS.
        {
            std::lock_guard<std::mutex> lk(client_mu_);
            if (client_ != INVALID_SOCK_T) {
#ifdef _WIN32
                ::shutdown(client_, SD_BOTH);
#else
                ::shutdown(client_, SHUT_RDWR);
#endif
                CLOSE_SOCK(client_);
                client_ = INVALID_SOCK_T;
                client_ready_ = false;
            }
        }
        client_cv_.notify_all();
        buf_cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() {
        sockaddr_in peer{};
        socklen_t plen = sizeof(peer);
        sock_t c = ::accept(listen_, reinterpret_cast<sockaddr*>(&peer), &plen);
        if (c == INVALID_SOCK_T) return;

        // Apply a short recv timeout to the accepted socket so the recv
        // loop below can poll stopped_ on its own without depending on
        // stop() to unblock it. Mirrors the fix landed in
        // test_jlp_pool_protocol.cpp::MockJlpServer. A 50ms poll keeps
        // CPU cost negligible while letting run() exit on its own clock
        // rather than racing against the teardown thread closing the
        // socket (which on Windows is documented UB while recv is in
        // flight).
#ifdef _WIN32
        {
            DWORD recv_timeout_ms = 50;
            setsockopt(c, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&recv_timeout_ms),
                       sizeof(recv_timeout_ms));
        }
#else
        {
            struct timeval tv;
            tv.tv_sec  = 0;
            tv.tv_usec = 50000;  // 50ms
            setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        }
#endif

        {
            std::lock_guard<std::mutex> lk(client_mu_);
            client_ = c;
            client_ready_ = true;
        }
        client_cv_.notify_all();

        std::vector<char> tmp(4096);
        while (!stopped_) {
            int n = ::recv(c, tmp.data(), static_cast<int>(tmp.size()), 0);
            if (n == 0) {
                // Graceful FIN from peer.
                break;
            }
            if (n < 0) {
                // Benign recv timeout from SO_RCVTIMEO above means we
                // loop back to check stopped_. Real socket errors mean
                // the connection is gone and we exit.
#ifdef _WIN32
                int err = WSAGetLastError();
                if (err == WSAETIMEDOUT || err == WSAEWOULDBLOCK) continue;
#else
                int err = errno;
                if (err == EAGAIN || err == EWOULDBLOCK) continue;
#endif
                (void)err;
                break;
            }
            std::lock_guard<std::mutex> lk(buf_mu_);
            rx_.insert(rx_.end(), tmp.data(), tmp.data() + n);
            buf_cv_.notify_all();
        }
        // shutdown-then-close on exit mirrors stop(): if a parallel
        // teardown is racing here, the matching shutdown in stop() will
        // have flipped client_ to INVALID_SOCK_T already and we skip
        // the second close.
        std::lock_guard<std::mutex> lk(client_mu_);
        if (client_ != INVALID_SOCK_T) {
#ifdef _WIN32
            ::shutdown(client_, SD_BOTH);
#else
            ::shutdown(client_, SHUT_RDWR);
#endif
            CLOSE_SOCK(client_);
            client_ = INVALID_SOCK_T;
            client_ready_ = false;
        }
    }

    sock_t listen_ = INVALID_SOCK_T;
    sock_t client_ = INVALID_SOCK_T;
    uint16_t port_ = 0;
    std::thread thread_;
    std::atomic<bool> stopped_{false};
    std::mutex client_mu_;
    std::condition_variable client_cv_;
    bool client_ready_ = false;
    std::mutex buf_mu_;
    std::condition_variable buf_cv_;
    std::vector<uint8_t> rx_;
};

std::unique_ptr<JLPPoolClient> make_connected(uint16_t port) {
    auto c = std::make_unique<JLPPoolClient>();
    c->set_use_tls(false);
    c->set_reconnect(false);
    c->set_timeout(1);
    if (!c->connect("127.0.0.1", port)) return nullptr;
    return c;
}

template <class Pred>
bool wait_for(Pred pred, std::chrono::milliseconds timeout = 3000ms) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(10ms);
    }
    return pred();
}

bool drive_auth_ok(MockServer& server, JLPPoolClient& client) {
    std::atomic<bool> ok{false};
    std::thread t([&] { ok = client.authenticate("worker", ""); });
    // The AUTH frame is 8 header bytes + 120-byte v2 hello = 128 bytes.
    auto auth_bytes = server.wait_recv(128, 5000ms);
    if (auth_bytes.size() != 128) {
        if (t.joinable()) t.join();
        return false;
    }
    if (!server.send_frame(TYPE_AUTH_OK, nullptr, 0)) {
        if (t.joinable()) t.join();
        return false;
    }
    t.join();
    return ok.load();
}

bool fail(const char* tname, const char* why) {
    std::fprintf(stderr, "[FAIL] %s: %s\n", tname, why);
    return false;
}

// Build a 109-byte JLPServerConfig with the requested dp_bits value.
JLPServerConfig make_work_assignment(uint32_t dp_bits, uint64_t work_id) {
    JLPServerConfig cfg{};
    // Valid 33-byte compressed pubkey prefix (0x02 / 0x03 / 0x04 are
    // the only legal SEC1 prefixes; pick 0x02 so the client's pubkey
    // sanity check is happy).
    cfg.public_key[0] = 0x02;
    for (int i = 1; i < 33; ++i) cfg.public_key[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 32; ++i) cfg.range_start[i] = 0x10 + i;
    for (int i = 0; i < 32; ++i) cfg.range_end[i]   = 0x20 + i;
    cfg.dp_bits = dp_bits;
    cfg.work_id = work_id;
    return cfg;
}

// ---------------------------------------------------------------------------
// 1. Out-of-range dp_bits (255) is rejected: work callback NEVER fires.
// ---------------------------------------------------------------------------
bool test_dp_bits_255_rejected_callback_not_fired() {
    // Lambda captures (`callback_count`) MUST be declared before the
    // JLPPoolClient and the MockServer so they are destroyed LAST. The
    // JLPPoolClient destructor joins its receiver thread, but the receiver
    // can be mid fire_work_callback (lock-acquire-copy-release-invoke)
    // when the join arrives. The receiver completes the invocation before
    // the thread exits, so the captured `&callback_count` reference must
    // still point at live memory at that moment. Declaring `callback_count`
    // first guarantees it outlives the client and the server.
    std::atomic<int> callback_count{0};
    MockServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("dp_bits_255_rejected_callback_not_fired",
                             "connect failed");

    client->set_work_callback([&](const WorkAssignment&) { ++callback_count; });

    if (!drive_auth_ok(server, *client)) {
        return fail("dp_bits_255_rejected_callback_not_fired",
                    "auth handshake failed");
    }

    // The bad WORK_ASN. dp_bits=255 is well above the [8..32] window the
    // client supports.
    auto cfg = make_work_assignment(255, 0xDEADBEEFCAFE0001ULL);
    static_assert(sizeof(JLPServerConfig) == 109,
                  "JLPServerConfig must be 109 bytes on the wire");
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));

    // Give the receiver thread a generous window to ingest, reject, and
    // tear down the connection.
    std::this_thread::sleep_for(300ms);

    if (callback_count.load() != 0) {
        return fail("dp_bits_255_rejected_callback_not_fired",
                    "work callback was invoked for dp_bits=255 (must be "
                    "suppressed before WorkAssignment reaches the caller)");
    }
    // Explicit teardown before the function frame unwinds: clear the
    // callback so even a late receiver-thread dispatch is a no-op, then
    // reset the unique_ptr so all background threads are joined before
    // any captured locals begin scoped destruction.
    client->set_work_callback(nullptr);
    client.reset();
    return true;
}

// ---------------------------------------------------------------------------
// 2. Out-of-range dp_bits causes a hard disconnect.
// ---------------------------------------------------------------------------
bool test_dp_bits_255_triggers_disconnect() {
    MockServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("dp_bits_255_triggers_disconnect",
                             "connect failed");

    if (!drive_auth_ok(server, *client)) {
        return fail("dp_bits_255_triggers_disconnect", "auth handshake failed");
    }

    if (!client->is_connected()) {
        return fail("dp_bits_255_triggers_disconnect",
                    "client reports disconnected immediately after AUTH_OK");
    }

    auto cfg = make_work_assignment(255, 0xDEADBEEFCAFE0002ULL);
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));

    // The S1 fix flips connected_ to false from the WORK_ASN handler.
    // Wait up to 3 seconds for is_connected() to observe the change.
    if (!wait_for([&] { return !client->is_connected(); }, 3000ms)) {
        return fail("dp_bits_255_triggers_disconnect",
                    "client did NOT disconnect after bad dp_bits "
                    "(is_connected() still true)");
    }
    // Explicit teardown for symmetry with the other tests in this file:
    // no callback to clear, but resetting the client joins its receiver
    // and sender threads at a known point instead of inside the implicit
    // function-frame unwind.
    client.reset();
    return true;
}

// ---------------------------------------------------------------------------
// 3. dp_bits below the lower bound (7) is also rejected.
// ---------------------------------------------------------------------------
bool test_dp_bits_too_low_rejected() {
    // Capture (`callback_count`) declared first so it outlives the
    // client. See test_dp_bits_255_rejected_callback_not_fired for the
    // full lifetime rationale.
    std::atomic<int> callback_count{0};
    MockServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("dp_bits_too_low_rejected", "connect failed");

    client->set_work_callback([&](const WorkAssignment&) { ++callback_count; });

    if (!drive_auth_ok(server, *client)) {
        return fail("dp_bits_too_low_rejected", "auth handshake failed");
    }

    auto cfg = make_work_assignment(7, 0xDEADBEEFCAFE0003ULL);
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));
    std::this_thread::sleep_for(300ms);

    if (callback_count.load() != 0) {
        return fail("dp_bits_too_low_rejected",
                    "work callback was invoked for dp_bits=7 "
                    "(below the 8-bit lower bound)");
    }
    // Explicit teardown: see test_dp_bits_255_rejected_callback_not_fired.
    client->set_work_callback(nullptr);
    client.reset();
    return true;
}

// ---------------------------------------------------------------------------
// 4. Valid dp_bits (24, in the middle of [8..32]) still flows through.
//    A sanity check so a future fix that flips the bound the wrong way
//    is caught by this file alone.
// ---------------------------------------------------------------------------
bool test_dp_bits_valid_passes() {
    // Captures (`callback_count`, `captured`) declared first so they
    // outlive the client. See test_dp_bits_255_rejected_callback_not_fired
    // for the full lifetime rationale.
    std::atomic<int> callback_count{0};
    WorkAssignment captured{};
    MockServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("dp_bits_valid_passes", "connect failed");

    client->set_work_callback([&](const WorkAssignment& w) {
        captured = w;
        ++callback_count;
    });

    if (!drive_auth_ok(server, *client)) {
        return fail("dp_bits_valid_passes", "auth handshake failed");
    }

    auto cfg = make_work_assignment(24, 0xDEADBEEFCAFE0004ULL);
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));

    if (!wait_for([&] { return callback_count.load() >= 1; }, 3000ms)) {
        return fail("dp_bits_valid_passes",
                    "work callback NOT invoked for valid dp_bits=24");
    }
    if (captured.dp_bits != 24u) {
        return fail("dp_bits_valid_passes",
                    "captured WorkAssignment dp_bits != 24");
    }
    if (captured.work_id != cfg.work_id) {
        return fail("dp_bits_valid_passes",
                    "captured WorkAssignment work_id mismatch");
    }
    // Explicit teardown: see test_dp_bits_255_rejected_callback_not_fired.
    client->set_work_callback(nullptr);
    client.reset();
    return true;
}

// ---------------------------------------------------------------------------
// 5. Boundary values: dp_bits=8 and dp_bits=32 are accepted (inclusive
//    bounds). dp_bits=33 is rejected (exclusive above 32).
// ---------------------------------------------------------------------------
bool test_dp_bits_boundary_accepts_8_and_32() {
    for (uint32_t dp : { 8u, 32u }) {
        // Capture (`callback_count`) declared first so it outlives the
        // client. See test_dp_bits_255_rejected_callback_not_fired for
        // the full lifetime rationale.
        std::atomic<int> callback_count{0};
        MockServer server;
        auto client = make_connected(server.port());
        if (!client) return fail("dp_bits_boundary_accepts_8_and_32",
                                 "connect failed");
        client->set_work_callback([&](const WorkAssignment&) { ++callback_count; });
        if (!drive_auth_ok(server, *client)) {
            return fail("dp_bits_boundary_accepts_8_and_32",
                        "auth handshake failed");
        }
        auto cfg = make_work_assignment(dp, 0x4242);
        server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));
        if (!wait_for([&] { return callback_count.load() >= 1; }, 3000ms)) {
            std::fprintf(stderr,
                "[FAIL] dp_bits_boundary_accepts_8_and_32: dp_bits=%u "
                "should be accepted (at inclusive boundary)\n",
                static_cast<unsigned>(dp));
            return false;
        }
        // Explicit teardown inside each loop iteration: see
        // test_dp_bits_255_rejected_callback_not_fired for the rationale.
        client->set_work_callback(nullptr);
        client.reset();
    }
    return true;
}

bool test_dp_bits_boundary_rejects_33() {
    // Capture (`callback_count`) declared first so it outlives the
    // client. See test_dp_bits_255_rejected_callback_not_fired for the
    // full lifetime rationale.
    std::atomic<int> callback_count{0};
    MockServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("dp_bits_boundary_rejects_33", "connect failed");
    client->set_work_callback([&](const WorkAssignment&) { ++callback_count; });
    if (!drive_auth_ok(server, *client)) {
        return fail("dp_bits_boundary_rejects_33", "auth handshake failed");
    }
    auto cfg = make_work_assignment(33, 0x4243);
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));
    std::this_thread::sleep_for(300ms);
    if (callback_count.load() != 0) {
        return fail("dp_bits_boundary_rejects_33",
                    "work callback invoked for dp_bits=33 "
                    "(above inclusive upper bound)");
    }
    // Explicit teardown: see test_dp_bits_255_rejected_callback_not_fired.
    client->set_work_callback(nullptr);
    client.reset();
    return true;
}

struct TestCase {
    const char* name;
    bool (*fn)();
};

const TestCase TESTS[] = {
    {"dp_bits_255_rejected_callback_not_fired",
        test_dp_bits_255_rejected_callback_not_fired},
    {"dp_bits_255_triggers_disconnect",
        test_dp_bits_255_triggers_disconnect},
    {"dp_bits_too_low_rejected",
        test_dp_bits_too_low_rejected},
    {"dp_bits_valid_passes",
        test_dp_bits_valid_passes},
    {"dp_bits_boundary_accepts_8_and_32",
        test_dp_bits_boundary_accepts_8_and_32},
    {"dp_bits_boundary_rejects_33",
        test_dp_bits_boundary_rejects_33},
};

}  // namespace

int main() {
#ifdef _WIN32
    WSAGuard guard;
#endif
    std::printf("=== JLP pool dp_bits validation tests (S1) ===\n");
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
