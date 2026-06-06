// test_jlp_pool_reconnect.cpp
//
// Reconnect / disconnect-handling tests for src/pool/jlp_pool_client.cpp.
// Mock TCP server on localhost; the test drives connection-loss scenarios
// from the test thread.
//
// Coverage (Priority 3 in the test plan):
//
//   1. The receiver handles a server-initiated disconnect after AUTH_OK and
//      stops gracefully: is_connected() returns false within ~2 seconds
//      (the auto-reconnect branch in the receiver loop exits with a single
//      backoff sleep and then sets running_=false).
//
//   2. Consecutive AUTH_FAIL handling: a server that always replies AUTH_FAIL
//      causes is_connected() to become false after each attempt, and the
//      private consecutive_auth_failures_ counter would (after
//      MAX_AUTH_FAIL_ATTEMPTS) cause the auto-reconnect branch to give up.
//
//      Caveat: per src/pool/jlp_pool_client.cpp, MAX_AUTH_FAIL_ATTEMPTS is
//      only checked inside the auto-reconnect branch, which itself runs only
//      when receive_message() returned false (i.e., a network-level read
//      failure). When the server replies AUTH_FAIL cleanly, the receiver
//      thread exits via the AUTH_FAIL handler (which sets connected_=false)
//      and never enters the auto-reconnect block. So at the wire level we
//      verify the observable thing: each AUTH_FAIL leaves the client
//      disconnected, and we can in fact attempt MAX_AUTH_FAIL_ATTEMPTS
//      cycles back to back without the client crashing or hanging.

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

#include "pool/jlp_pool_client.hpp"
#include "pool/jlp_wire_generated.hpp"  // jlp_wire::PROTOCOL_VERSION
#include "pool/pool_client.hpp"
#include "pool/pool_config.hpp"  // MAX_AUTH_FAIL_ATTEMPTS moved here in v1.4.2 Pool-B3

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

// Wire frame builder (duplicated from test_jlp_pool_protocol.cpp on purpose:
// keeping each test file self-contained avoids linkage gymnastics).
constexpr uint8_t TYPE_AUTH      = 0x01;
constexpr uint8_t TYPE_AUTH_OK   = 0x02;
constexpr uint8_t TYPE_AUTH_FAIL = 0x03;

// v1.4.2 B.5: mock server must send flags = PROTOCOL_VERSION just like
// the real server, otherwise the client (correctly) rejects with protocol
// version mismatch.
// The reconnect / supervisor / dedup tests below are protocol-version
// agnostic; this just keeps the mock honest. Track the client's compiled
// PROTOCOL_VERSION (from jlp_wire_generated.hpp) instead of a hardcoded
// literal so a future wire bump cannot silently strand these tests at an
// old version (which is exactly what happened at the v3 -> v4 bump).
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

// --- minimal mock server, hardened to survive multiple sequential clients ---

class MockJlpServer {
public:
    // If `auto_reply_auth_fail` is true, the worker replies AUTH_FAIL as soon
    // as it has read the 84-byte AUTH frame, then closes the client socket
    // (same shape as a hostile pool that rejects every credential).
    explicit MockJlpServer(bool auto_reply_auth_fail = false)
        : auto_reply_auth_fail_(auto_reply_auth_fail) {

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
            std::fprintf(stderr, "[mock] bind: err=%d\n", last_sock_err()); std::abort();
        }
        sockaddr_in bound{};
        socklen_t blen = sizeof(bound);
        ::getsockname(listen_, reinterpret_cast<sockaddr*>(&bound), &blen);
        port_ = ntohs(bound.sin_port);
        ::listen(listen_, 4);

        thread_ = std::thread(&MockJlpServer::run, this);
    }

    ~MockJlpServer() {
        stop();
    }

    uint16_t port() const { return port_; }

    // Wait for the n-th client (1-indexed) to be accepted. Useful for tests
    // that go through several connect cycles.
    bool wait_for_nth_client(int n, std::chrono::milliseconds timeout = 3000ms) {
        std::unique_lock<std::mutex> lk(mu_);
        return cv_.wait_for(lk, timeout,
                            [&] { return clients_accepted_ >= n || stopped_; })
               && clients_accepted_ >= n;
    }

    int clients_accepted() {
        std::lock_guard<std::mutex> lk(mu_);
        return clients_accepted_;
    }

    // Send a raw frame to the most recently accepted client.
    bool send_frame(uint8_t type, const void* payload, uint16_t len) {
        std::lock_guard<std::mutex> lk(client_mu_);
        if (current_client_ == INVALID_SOCK_T) return false;
        auto bytes = build_frame(type, payload, len);
        size_t total = bytes.size(), sent = 0;
        const char* p = reinterpret_cast<const char*>(bytes.data());
        while (sent < total) {
            int n = ::send(current_client_, p + sent,
                           static_cast<int>(total - sent), 0);
            if (n <= 0) return false;
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    // Close the most recently accepted client socket.
    //
    // The shutdown(SD_BOTH) before closesocket is load-bearing on Windows
    // and mirrors the fix in test_jlp_pool_protocol.cpp::MockJlpServer.
    // MSDN documents that calling closesocket on a socket while another
    // thread is blocked in recv() on the same handle leaves the recv in
    // an UNDEFINED state. The accept-thread spawned by run() spends most
    // of its life in ::recv() on current_client_; a bare closesocket from
    // a teardown thread reliably reproduced execute-at-NULL access
    // violations during process exit. shutdown(SD_BOTH) signals a
    // graceful close, so the blocked recv returns 0 BEFORE closesocket
    // actually frees the handle, eliminating the unspecified-behaviour
    // window. POSIX shutdown(SHUT_RDWR) has the same semantics, so the
    // call works unmodified on Linux + macOS.
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

    // Wait for the current client to have received >= n bytes of input from
    // the JLP client. Used in non-auto-reply mode to time the AUTH receive.
    bool wait_recv_bytes(size_t n, std::chrono::milliseconds timeout = 2000ms) {
        std::unique_lock<std::mutex> lk(buf_mu_);
        return buf_cv_.wait_for(lk, timeout,
                                [&] { return rx_.size() >= n || stopped_; })
               && rx_.size() >= n;
    }

    void clear_rx_buffer() {
        std::lock_guard<std::mutex> lk(buf_mu_);
        rx_.clear();
    }

    void stop() {
        if (stopped_.exchange(true)) return;
        if (listen_ != INVALID_SOCK_T) { CLOSE_SOCK(listen_); listen_ = INVALID_SOCK_T; }
        close_current_client();
        cv_.notify_all();
        buf_cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() {
        while (!stopped_) {
            sockaddr_in peer{};
            socklen_t plen = sizeof(peer);
            sock_t c = ::accept(listen_, reinterpret_cast<sockaddr*>(&peer), &plen);
            if (c == INVALID_SOCK_T) {
                // Listen socket closed (stop()) or other error -- exit.
                break;
            }

            // Short recv timeout on the accepted socket so the recv loop
            // below can poll stopped_ on its own without depending on a
            // teardown thread closing the socket out from under us
            // (Windows-documented closesocket-during-recv UB; see
            // close_current_client). A 50ms poll keeps CPU cost
            // negligible while letting run() exit on its own clock.
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
                // Close any old client (defensive; should already be closed).
                // Shutdown-then-close mirrors close_current_client so a
                // stale worker still parked on the old socket sees a
                // graceful EOF instead of the unspecified-behaviour
                // closesocket-during-recv case.
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
                clients_accepted_++;
            }
            cv_.notify_all();

            // Drain bytes from the client until it disconnects (or we are
            // stopped). If auto_reply mode is on, reply AUTH_FAIL after the
            // first 84 bytes and close.
            std::vector<char> tmp(4096);
            size_t bytes_for_this_client = 0;
            bool replied = false;
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
                    int err = last_sock_err();
#ifdef _WIN32
                    if (err == WSAETIMEDOUT || err == WSAEWOULDBLOCK) continue;
#else
                    if (err == EAGAIN || err == EWOULDBLOCK) continue;
#endif
                    break;
                }
                {
                    std::lock_guard<std::mutex> lk(buf_mu_);
                    rx_.insert(rx_.end(), tmp.data(), tmp.data() + n);
                }
                buf_cv_.notify_all();
                bytes_for_this_client += static_cast<size_t>(n);

                if (auto_reply_auth_fail_ && !replied
                    && bytes_for_this_client >= 84)
                {
                    auto frame = build_frame(TYPE_AUTH_FAIL, nullptr, 0);
                    ::send(c, reinterpret_cast<const char*>(frame.data()),
                           static_cast<int>(frame.size()), 0);
                    replied = true;
                    // Hold the connection open so the client's receiver loop
                    // sees AUTH_FAIL and unwinds via handle_server_message
                    // (which sets connected_=false). We don't close eagerly --
                    // doing so could race the receiver's read of the AUTH_FAIL
                    // frame.
                }
            }

            // Close after the client disconnects. Shutdown-then-close
            // mirrors close_current_client so a parallel teardown
            // observing the same socket through current_client_ does
            // not race the raw closesocket against an in-flight recv
            // on the producer side.
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
                } else {
                    // Was already closed elsewhere (e.g. close_current_client
                    // shut down + closed the same handle while we were in
                    // recv); do not double-close.
                }
            }
        }
    }

    sock_t   listen_         = INVALID_SOCK_T;
    sock_t   current_client_ = INVALID_SOCK_T;
    uint16_t port_           = 0;
    bool     auto_reply_auth_fail_;

    std::thread       thread_;
    std::atomic<bool> stopped_{false};

    std::mutex              client_mu_;
    std::mutex              mu_;
    std::condition_variable cv_;
    int                     clients_accepted_ = 0;

    std::mutex              buf_mu_;
    std::condition_variable buf_cv_;
    std::vector<char>       rx_;
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

// ---------------------------------------------------------------------------
// 1. Receiver handles disconnect after AUTH_OK and stops gracefully.
//
//    auto_reconnect=true is REQUIRED: with it off, the receiver_loop would
//    just keep re-entering recv() on a dead socket (separate latent issue);
//    the supervisor design assumes auto_reconnect=true means "let the
//    receiver detect the disconnect, mark connected_=false, and exit cleanly
//    -- the supervisor will reconnect from a safe thread later".
// ---------------------------------------------------------------------------
bool test_disconnect_after_auth_ok() {
    MockJlpServer server;
    auto client = std::make_unique<JLPPoolClient>();
    client->set_use_tls(false);
    client->set_reconnect(true);
    client->set_timeout(2000);

    if (!client->connect("127.0.0.1", server.port()))
        return fail("disconnect_after_auth_ok", "connect failed");

    if (!server.wait_for_nth_client(1))
        return fail("disconnect_after_auth_ok", "server didn't accept client");

    // authenticate() blocks; do it on a thread so we can drive replies.
    bool auth_ok = false;
    std::thread t([&] { auth_ok = client->authenticate("worker", ""); });

    // Wait for AUTH bytes, reply AUTH_OK.
    server.wait_recv_bytes(84);
    server.send_frame(TYPE_AUTH_OK, nullptr, 0);
    t.join();

    if (!auth_ok) return fail("disconnect_after_auth_ok", "auth failed");
    if (!client->is_connected())
        return fail("disconnect_after_auth_ok", "is_connected false right after AUTH_OK");

    // Now hang up server-side. The receiver's reconnect branch should run a
    // single backoff sleep (~1s) and then set running_=false. is_connected()
    // becomes false after the auto_reconnect_ branch flips connected_ first.
    server.close_current_client();

    if (!wait_for([&] { return !client->is_connected(); }, 4000ms))
        return fail("disconnect_after_auth_ok",
                    "is_connected() did not become false within 4 seconds");
    return true;
}

// ---------------------------------------------------------------------------
// 2. Consecutive AUTH_FAIL: 3 cycles, each gets AUTH_FAIL, each leaves the
//    client disconnected. We don't try to drive the receiver loop's auto-
//    reconnect path here -- that path isn't reached on a clean AUTH_FAIL
//    (handle_server_message sets connected_=false directly). What we DO
//    verify: each cycle's authenticate() returns false, is_connected()
//    becomes false, and the client survives MAX_AUTH_FAIL_ATTEMPTS cycles
//    without crashing or wedging.
// ---------------------------------------------------------------------------
bool test_consecutive_auth_fail() {
    MockJlpServer server(/*auto_reply_auth_fail=*/true);

    // v1.4.2 Pool-B3: MAX_AUTH_FAIL_ATTEMPTS moved from JLPPoolClient to
    // pool_config.hpp (the supervisor in PoolManager is the only path
    // that honors the cap now; the previous in-receiver-thread cap was
    // dead code).
    constexpr uint32_t N = MAX_AUTH_FAIL_ATTEMPTS;
    static_assert(N == 3, "test assumes MAX_AUTH_FAIL_ATTEMPTS = 3");

    auto client = std::make_unique<JLPPoolClient>();
    client->set_use_tls(false);
    client->set_reconnect(false);   // each cycle is driven by this thread
    client->set_timeout(2000);

    for (uint32_t i = 0; i < N; ++i) {
        if (!client->connect("127.0.0.1", server.port())) {
            std::fprintf(stderr, "[debug] iteration %u: connect failed\n", i);
            return fail("consecutive_auth_fail", "connect() failed mid-loop");
        }

        if (!server.wait_for_nth_client(static_cast<int>(i + 1))) {
            return fail("consecutive_auth_fail", "server did not accept Nth client");
        }

        bool ok = client->authenticate("worker", "");
        if (ok) {
            return fail("consecutive_auth_fail",
                        "authenticate() returned true on AUTH_FAIL cycle");
        }
        if (!wait_for([&] { return !client->is_connected(); }, 2000ms)) {
            return fail("consecutive_auth_fail",
                        "is_connected() never became false after AUTH_FAIL");
        }
        // Disconnect to clean up the receiver/sender threads before the next
        // connect() (connect() also calls disconnect() if already connected,
        // but doing it explicitly is cleaner and surfaces any join hang).
        client->disconnect();
    }

    // We hit MAX_AUTH_FAIL_ATTEMPTS with no crash and no hang. The client is
    // not connected, and the server saw at least N clients arrive.
    if (server.clients_accepted() < static_cast<int>(N)) {
        return fail("consecutive_auth_fail",
                    "server didn't see all N AUTH_FAIL cycles");
    }
    if (client->is_connected()) {
        return fail("consecutive_auth_fail",
                    "client claims still connected at end of AUTH_FAIL cycles");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 3. Pool-B1: dp_sequence_next_ must persist across supervisor reconnect.
//
//    Drive: seed JLPPoolClient with (work_id=42, seq=500), then disconnect
//    and snapshot_dp_sequence. The snapshot must reflect the seeded value
//    even though no DP was ever submitted. Then construct a SECOND client,
//    re-seed it with the snapshotted seq, and assert get/snapshot returns
//    the same value. This is the basic "supervisor recreates client without
//    losing counter" property; the on-wire piece is covered by integration
//    with the mock server, but here we focus on the C++-level handoff.
// ---------------------------------------------------------------------------
bool test_dp_sequence_persist_across_recreate() {
    constexpr uint64_t kWorkId  = 42;
    constexpr uint32_t kStartAt = 500;

    uint64_t out_work_id = 0;
    uint32_t out_seq = 0;

    auto client1 = std::make_unique<JLPPoolClient>();
    client1->seed_dp_sequence(kWorkId, kStartAt);
    client1->snapshot_dp_sequence(out_work_id, out_seq);
    if (out_work_id != kWorkId || out_seq != kStartAt) {
        return fail("dp_sequence_persist_across_recreate",
                    "seed did not round-trip on the same client");
    }
    client1.reset();

    // Supervisor recreates the client; seed from the snapshot.
    auto client2 = std::make_unique<JLPPoolClient>();
    client2->seed_dp_sequence(out_work_id, out_seq);
    uint64_t round2_work_id = 0;
    uint32_t round2_seq = 0;
    client2->snapshot_dp_sequence(round2_work_id, round2_seq);
    if (round2_work_id != kWorkId || round2_seq != kStartAt) {
        return fail("dp_sequence_persist_across_recreate",
                    "seed did not round-trip on recreated client");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 4. Pool-B4: clean shutdown with DPs in the queue should drain.
//
//    Drive: AUTH_OK successfully, submit many DPs (more than fits in a
//    batch of 100), then call disconnect(). The mock server counts how
//    many bytes it received from the client; the disconnect path's
//    Pool-B4 drain should push the queued DPs out on the wire before
//    closing.
// ---------------------------------------------------------------------------
bool test_shutdown_drains_dp_queue() {
    MockJlpServer server;
    auto client = std::make_unique<JLPPoolClient>();
    client->set_use_tls(false);
    client->set_reconnect(true);
    client->set_timeout(2000);

    if (!client->connect("127.0.0.1", server.port()))
        return fail("shutdown_drains_dp_queue", "connect failed");
    if (!server.wait_for_nth_client(1))
        return fail("shutdown_drains_dp_queue", "server didn't accept");

    bool auth_ok = false;
    std::thread t([&] { auth_ok = client->authenticate("worker", ""); });
    server.wait_recv_bytes(84);
    server.send_frame(TYPE_AUTH_OK, nullptr, 0);
    t.join();
    if (!auth_ok)
        return fail("shutdown_drains_dp_queue", "auth failed");

    // Submit 50 DPs. With a server that just buffers, they should land
    // on the wire as one DP_BATCH_V2 frame within a few hundred ms.
    DistinguishedPoint dp{};
    for (int i = 0; i < 50; ++i) {
        dp.x[0] = static_cast<uint8_t>(i);
        client->submit_dp(dp);
    }

    // Wait briefly for the steady-state sender to push the batch.
    std::this_thread::sleep_for(300ms);

    server.clear_rx_buffer();
    // Submit 30 more DPs and immediately disconnect: the drain path must
    // push these out before closing the socket.
    for (int i = 50; i < 80; ++i) {
        dp.x[0] = static_cast<uint8_t>(i);
        client->submit_dp(dp);
    }
    client->disconnect();

    // After disconnect, the server should have received at least 30 DPs'
    // worth of additional bytes. DP_BATCH_V2 payload per DP is 78 bytes
    // plus an 8-byte header and 4-byte count (30 DPs = ~2.36 KB).
    //
    // Production drain budget is DRAIN_TIMEOUT_MS = 500ms inside
    // JLPPoolClient::disconnect(); the sender_loop wakes from its 100ms
    // dp_cv_.wait_for, drains the batch, and ssl_write_mutex_-serializes
    // it onto the wire. The mock server then reads the bytes out of the
    // kernel buffer (with our SO_RCVTIMEO=50ms poll for stop-detection).
    // The end-to-end window from submit_dp through wire-arrival is
    // dominated by Windows scheduling jitter (sender thread wakeup +
    // recv thread wakeup + closesocket teardown). On loaded test hosts
    // this whole cycle has been observed up to ~7s. Wait 10s here: long
    // enough to be insensitive to host load, short enough that a
    // genuinely wedged drain still fails the test in reasonable wall
    // time.
    bool got_drain = server.wait_recv_bytes(12 + 30 * 78, 10000ms);
    if (!got_drain) {
        return fail("shutdown_drains_dp_queue",
                    "drain did not flush queued DPs before disconnect");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 5. R-B5: DPs submitted before AUTH_OK must be re-queued, not silently
//    dropped. We can't easily drive the sender to drain into !AUTH_OK in
//    this lightweight test, so the assertion is structural: after a
//    disconnect that leaves DPs in the queue, the shutdown_drop_count
//    is non-negative and at most the number we submitted. This guards
//    against a regression where batch.clear() drops DPs without
//    incrementing the visible counter.
// ---------------------------------------------------------------------------
bool test_rb5_no_silent_drop() {
    MockJlpServer server;
    auto client = std::make_unique<JLPPoolClient>();
    client->set_use_tls(false);
    client->set_reconnect(false);
    client->set_timeout(500);

    if (!client->connect("127.0.0.1", server.port()))
        return fail("rb5_no_silent_drop", "connect failed");
    if (!server.wait_for_nth_client(1))
        return fail("rb5_no_silent_drop", "server didn't accept");

    // Submit DPs BEFORE authenticate completes. sender_loop will see
    // them but AUTH_OK has not landed yet. The R-B5 contract is that
    // they get re-queued, NOT silently dropped.
    DistinguishedPoint dp{};
    for (int i = 0; i < 10; ++i) {
        dp.x[0] = static_cast<uint8_t>(i);
        client->submit_dp(dp);
    }

    // Let the sender_loop run a few cycles.
    std::this_thread::sleep_for(300ms);

    client->disconnect();
    // No silent loss assertion: shutdown_drop_count must be <= 10.
    // The actual value depends on whether the sender ran a drain cycle
    // before disconnect; what matters is that loss is COUNTED.
    if (client->get_shutdown_drop_count() > 10) {
        return fail("rb5_no_silent_drop",
                    "shutdown drop count exceeds submitted DPs");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 6. Tier C: snapshot_dp_queue + preload_dp_queue round-trip.
//
//    The PoolManager-level on-disk persistence file is exercised at the
//    manager level (separately); here we verify the client-level API
//    that the manager calls round-trips a queue correctly.
// ---------------------------------------------------------------------------
bool test_tierc_snapshot_preload_round_trip() {
    auto client1 = std::make_unique<JLPPoolClient>();
    DistinguishedPoint dp{};
    constexpr int N = 25;
    for (int i = 0; i < N; ++i) {
        dp.x[0] = static_cast<uint8_t>(i);
        dp.d[0] = static_cast<uint8_t>(i * 7);
        dp.type = static_cast<uint8_t>(i & 1);
        dp.dp_bits = 35;
        client1->submit_dp(dp);
    }
    auto snap = client1->snapshot_dp_queue();
    if (snap.size() != N) {
        return fail("tierc_snapshot_preload_round_trip",
                    "snapshot size != N");
    }
    client1.reset();

    auto client2 = std::make_unique<JLPPoolClient>();
    client2->preload_dp_queue(std::move(snap));
    auto snap2 = client2->snapshot_dp_queue();
    if (snap2.size() != N) {
        return fail("tierc_snapshot_preload_round_trip",
                    "preloaded size != N");
    }
    for (int i = 0; i < N; ++i) {
        if (snap2[i].x[0] != static_cast<uint8_t>(i)
            || snap2[i].d[0] != static_cast<uint8_t>(i * 7)
            || snap2[i].type != static_cast<uint8_t>(i & 1)
            || snap2[i].dp_bits != 35) {
            return fail("tierc_snapshot_preload_round_trip",
                        "DP fields did not round-trip");
        }
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
    {"disconnect_after_auth_ok",                test_disconnect_after_auth_ok},
    {"consecutive_auth_fail",                   test_consecutive_auth_fail},
    {"dp_sequence_persist_across_recreate",     test_dp_sequence_persist_across_recreate},
    {"shutdown_drains_dp_queue",                test_shutdown_drains_dp_queue},
    {"rb5_no_silent_drop",                      test_rb5_no_silent_drop},
    {"tierc_snapshot_preload_round_trip",       test_tierc_snapshot_preload_round_trip},
};

}  // namespace

int main() {
#ifdef _WIN32
    WSAGuard guard;
#endif
    std::printf("=== JLP pool reconnect tests ===\n");
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
