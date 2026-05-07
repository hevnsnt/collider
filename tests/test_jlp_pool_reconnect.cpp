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
#include "pool/pool_client.hpp"

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

std::vector<uint8_t> build_frame(uint8_t type, const void* payload, uint16_t len) {
    std::vector<uint8_t> out;
    out.reserve(8 + len);
    out.push_back('K'); out.push_back('A'); out.push_back('N'); out.push_back('G');
    out.push_back(type);
    out.push_back(0);
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
    void close_current_client() {
        std::lock_guard<std::mutex> lk(client_mu_);
        if (current_client_ != INVALID_SOCK_T) {
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

            {
                std::lock_guard<std::mutex> lk(client_mu_);
                // Close any old client (defensive; should already be closed).
                if (current_client_ != INVALID_SOCK_T) {
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
                if (n <= 0) break;
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

            // Close after the client disconnects.
            {
                std::lock_guard<std::mutex> lk(client_mu_);
                if (current_client_ == c) {
                    CLOSE_SOCK(c);
                    current_client_ = INVALID_SOCK_T;
                } else {
                    // Was already closed elsewhere.
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

    constexpr uint32_t N = JLPPoolClient::MAX_AUTH_FAIL_ATTEMPTS;
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
// Driver
// ---------------------------------------------------------------------------

struct TestCase {
    const char* name;
    bool (*fn)();
};

const TestCase TESTS[] = {
    {"disconnect_after_auth_ok",  test_disconnect_after_auth_ok},
    {"consecutive_auth_fail",     test_consecutive_auth_fail},
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
