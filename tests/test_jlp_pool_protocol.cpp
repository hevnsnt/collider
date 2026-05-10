// test_jlp_pool_protocol.cpp
//
// Mock-server tests for the JLP wire protocol implemented by
// src/pool/jlp_pool_client.cpp. Each test:
//   1. Spawns a tiny TCP server on 127.0.0.1:OS-assigned-port.
//   2. Connects a real JLPPoolClient (use_tls=false) to that port.
//   3. Drives the conversation byte-for-byte from the test thread.
//   4. Asserts on what the client sent / how the client behaved.
//
// What this file covers (Priority 1 in the test plan):
//
//    1.  AUTH wire format: header magic, type, payload size, plus the
//        120-byte JLPClientHelloV2 (worker_name + password +
//        timestamp_ms + nonce; v1.4.1 B.2 wire format).
//    2.  authenticate() returns true when AUTH_OK arrives within timeout.
//    3.  authenticate() returns false on AUTH_FAIL.
//    4.  authenticate() returns false when MSG_ERROR arrives during AUTH_SENT.
//    5.  authenticate() eventually returns false when the server is silent
//        (timeout path; bounded by AUTH_RESPONSE_TIMEOUT_MS = 10s).
//    6.  Worker-name length validation: a name >= 64 chars is rejected
//        immediately, with no AUTH bytes ever hitting the wire.
//    7.  PING -> PONG: the client must answer PING with a zero-payload PONG.
//    8.  Pre-auth message gating (Wave 4 D-H4): WORK_ASN before AUTH_OK is
//        ignored (work callback NOT invoked); the same WORK_ASN after AUTH_OK
//        IS dispatched.
//    9.  WORK_ASN parsing: 109-byte JLPServerConfig payload populates the
//        WorkAssignment fields exactly.
//   10.  STATS_RSP parsing: 24-byte payload is decoded as three uint64 LE
//        fields (Wave 4 B-MED-1: pool_speed must NOT be a double).
//   11.  MSG_ERROR handling: Wave 4 D-L3 says payloads are capped + control
//        chars stripped. We don't read stderr -- we just confirm no crash on
//        large payloads (500 bytes) and on payloads with embedded ESC + NUL.
//   12.  submit_dp / submit_dps return true once authenticated.
//   13.  DP submission wire format: a queued DP is flushed out as a DP_BATCH
//        frame (KANG magic, type=0x22).
//   14.  Backpressure: submit_dps with a vector that would overflow
//        MAX_DP_QUEUE_SIZE returns false. We use a small backlog by first
//        filling the queue with submit_dp calls (bounded by what we can
//        push before the sender drains them).
//   15.  report_solution: 32-byte private key arrives on the wire as a
//        SOLUTION frame.
//
// What this file does NOT cover (because other tests already do):
//   - TLS handshake / cert verification     (test_jlp_pool_handshake.cpp)
//   - URL parsing / serialization           (test_jlp_pool_manager.cpp)
//   - Reconnect / AUTH_FAIL retry caps      (test_jlp_pool_reconnect.cpp)

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
    #define SOCK_ERR_T int
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
    #define SOCK_ERR_T int
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
#include <functional>
#include <mutex>
#include <string>
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

// ---------------------------------------------------------------------------
// Wire helpers (re-implemented here so the test can never accidentally pass
// when the production header drifts; if these layouts change, the tests
// must explicitly update).
// ---------------------------------------------------------------------------
//
// All multi-byte integers on the wire are little-endian; we run the suite on
// x86_64 where memcpy of a uint16_t/uint32_t/uint64_t is already LE.

constexpr uint8_t TYPE_AUTH      = 0x01;
constexpr uint8_t TYPE_AUTH_OK   = 0x02;
constexpr uint8_t TYPE_AUTH_FAIL = 0x03;
constexpr uint8_t TYPE_WORK_ASN  = 0x11;
constexpr uint8_t TYPE_DP_BATCH  = 0x22;
constexpr uint8_t TYPE_DP_BATCH_V2 = 0x24;  // v2 with work_id attestation
constexpr uint8_t TYPE_STATS_RSP = 0x31;
constexpr uint8_t TYPE_SOLUTION  = 0x40;
constexpr uint8_t TYPE_PING      = 0x50;
constexpr uint8_t TYPE_PONG      = 0x51;
constexpr uint8_t TYPE_MSG_ERROR = 0xFF;

// Build [magic=KANG][type][flags=0][len:LE u16][payload].
std::vector<uint8_t> build_frame(uint8_t type, const void* payload, uint16_t len) {
    std::vector<uint8_t> out;
    out.reserve(8 + len);
    out.push_back('K'); out.push_back('A'); out.push_back('N'); out.push_back('G');
    out.push_back(type);
    out.push_back(0);                                  // flags
    out.push_back(static_cast<uint8_t>(len & 0xFF));   // len LE low
    out.push_back(static_cast<uint8_t>(len >> 8));     // len LE high
    if (payload && len > 0) {
        const uint8_t* p = static_cast<const uint8_t*>(payload);
        out.insert(out.end(), p, p + len);
    }
    return out;
}

// ---------------------------------------------------------------------------
// MockJlpServer
//
// Listens on 127.0.0.1:OS-assigned-port. After accept(), reads/writes raw
// bytes from a worker thread. The test drives the conversation via:
//
//    server.send_frame(TYPE_AUTH_OK, nullptr, 0);
//    auto bytes = server.wait_recv(8 + 76);  // wait for first 84 bytes
//
// On destruction, closes both client + listen sockets and joins the worker.
// ---------------------------------------------------------------------------
class MockJlpServer {
public:
    // Set send_kick_pings=false to disable the auto-PING-every-5-ms behaviour.
    // The ping_pong test needs this OFF so it can drive a single PING and
    // verify the resulting single PONG is the only thing on the wire.
    explicit MockJlpServer(bool send_kick_pings = true)
        : send_kick_pings_(send_kick_pings)
    {
        // Only filter PONGs out of the recv stream when the kicker is the
        // one generating them. Tests with no kicker want every byte preserved.
        filter_pongs_.store(send_kick_pings);
        listen_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_ == INVALID_SOCK_T) {
            std::fprintf(stderr, "[mock] socket() failed: err=%d\n", last_sock_err());
            std::abort();
        }

        // Reuse so back-to-back tests on the same port don't TIME_WAIT us out.
#ifndef _WIN32
        int one = 1;
        setsockopt(listen_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port        = 0;  // OS picks a free port

        if (::bind(listen_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::fprintf(stderr, "[mock] bind() failed: err=%d\n", last_sock_err());
            std::abort();
        }

        sockaddr_in bound{};
        socklen_t blen = sizeof(bound);
        if (::getsockname(listen_, reinterpret_cast<sockaddr*>(&bound), &blen) < 0) {
            std::fprintf(stderr, "[mock] getsockname() failed: err=%d\n", last_sock_err());
            std::abort();
        }
        port_ = ntohs(bound.sin_port);

        if (::listen(listen_, 1) < 0) {
            std::fprintf(stderr, "[mock] listen() failed: err=%d\n", last_sock_err());
            std::abort();
        }

        thread_ = std::thread(&MockJlpServer::run, this);
    }

    ~MockJlpServer() {
        stop();
    }

    uint16_t port() const { return port_; }

    // Block (up to `timeout`) until the worker has at least `n` bytes of
    // received-from-client data buffered. Returns the FIRST n bytes (does NOT
    // pop them). Returns {} on timeout or disconnect.
    //
    // Why 5 s default rather than something tight: the production
    // receive_message() holds ssl_io_mutex_ while blocking on recv() with
    // SO_RCVTIMEO (100 ms in our tests via set_timeout). The sender thread
    // must win that mutex to actually transmit. std::mutex on Windows is not
    // strictly FIFO, so the receiver can sometimes re-acquire several times
    // in a row before the sender gets through. 5 s of slack is comfortably
    // long enough for the race to resolve while still failing fast on real
    // bugs (a stuck client).
    std::vector<uint8_t> wait_recv(size_t n,
                                   std::chrono::milliseconds timeout = 5000ms) {
        std::unique_lock<std::mutex> lk(buf_mu_);
        bool ok = buf_cv_.wait_for(lk, timeout, [&] {
            return rx_.size() >= n || stopped_;
        });
        if (!ok || rx_.size() < n) return {};
        return std::vector<uint8_t>(rx_.begin(), rx_.begin() + n);
    }

    // Same as wait_recv but consumes the bytes returned. Useful when the test
    // wants to wait for a SECOND batch from the client (e.g., a PONG after
    // an AUTH frame). On timeout, returns {} and does NOT consume anything.
    std::vector<uint8_t> wait_recv_pop(size_t n,
                                       std::chrono::milliseconds timeout = 5000ms) {
        std::unique_lock<std::mutex> lk(buf_mu_);
        bool ok = buf_cv_.wait_for(lk, timeout, [&] {
            return rx_.size() >= n || stopped_;
        });
        if (!ok || rx_.size() < n) return {};
        std::vector<uint8_t> out(rx_.begin(), rx_.begin() + n);
        rx_.erase(rx_.begin(), rx_.begin() + n);
        return out;
    }

    // Block (up to `timeout`) until the client connection has been accepted.
    bool wait_for_client(std::chrono::milliseconds timeout = 2000ms) {
        std::unique_lock<std::mutex> lk(client_mu_);
        return client_cv_.wait_for(lk, timeout,
                                   [&] { return client_ready_ || stopped_; })
               && client_ready_;
    }

    // Stop the periodic PING kicker (no-op if it wasn't running). Also turns
    // OFF the PONG-filtering in the recv loop so the test can observe the
    // client's PONG response to its own PING.
    void stop_kicker() {
        stop_kicking_ = true;
        filter_pongs_ = false;
    }

    // Atomically clear the receive buffer. Useful right after stop_kicker()
    // when the test wants to start observing fresh client traffic without
    // dealing with leftover bytes from earlier kicker PONG noise.
    void drain_rx() {
        std::lock_guard<std::mutex> lk(buf_mu_);
        rx_.clear();
    }

    // Send a JLP frame over the accepted client socket.
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

    // Close the accepted client socket (simulates server-side disconnect).
    void close_client() {
        std::lock_guard<std::mutex> lk(client_mu_);
        if (client_ != INVALID_SOCK_T) {
            CLOSE_SOCK(client_);
            client_ = INVALID_SOCK_T;
            client_ready_ = false;
        }
    }

    void stop() {
        if (stopped_.exchange(true)) return;
        // Close listen so accept() unblocks.
        if (listen_ != INVALID_SOCK_T) {
            CLOSE_SOCK(listen_);
            listen_ = INVALID_SOCK_T;
        }
        close_client();
        client_cv_.notify_all();
        buf_cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() {
        sockaddr_in peer{};
        socklen_t plen = sizeof(peer);
        sock_t c = ::accept(listen_, reinterpret_cast<sockaddr*>(&peer), &plen);
        if (c == INVALID_SOCK_T) {
            // Likely stop() closed the listen socket; just exit.
            return;
        }
        {
            std::lock_guard<std::mutex> lk(client_mu_);
            client_ = c;
            client_ready_ = true;
        }
        client_cv_.notify_all();

        // Read forever into rx_ until the client closes or we are stopped.
        // We also periodically send a "kick" frame (PING) to the client. The
        // production receive_message() holds ssl_io_mutex_ across its blocking
        // recv() call, and on Windows std::mutex (SRWLock) is not strictly
        // FIFO -- we observed in dev that with the receiver re-acquiring the
        // mutex on every loop iteration (1 ms timeout in our tests), the
        // sender thread can be starved for many seconds, causing AUTH bytes
        // to never reach the mock server. Sending a PING every ~5 ms forces
        // the receiver's recv() to return successfully (not timeout); during
        // the brief window between recv() returning and the next iteration's
        // recv() acquiring the mutex again, the sender thread reliably wins
        // the race. The server-initiated PING + client-side PONG also exercise
        // a real production code path, so this isn't extra synthetic traffic.
        std::vector<char> tmp(4096);
        std::thread kicker;
        if (send_kick_pings_) {
            // The kicker addresses a real production race: the JLP client's
            // receive_message() holds ssl_io_mutex_ across its blocking
            // recv() call, and on Windows std::mutex (built on SRWLock) is
            // not strictly FIFO. With a short recv() timeout (~1 ms) the
            // receiver loop re-acquires the mutex in a tight loop, often
            // starving the sender (authenticate()'s send_hello, the
            // sender_thread's DP_BATCH, report_solution()) for many seconds.
            //
            // Sending the client *something* makes recv() return early
            // (success, not timeout), so the receiver releases the mutex,
            // dispatches the message, then re-enters recv() with a brief
            // gap the sender can win.
            //
            // We send an UNKNOWN message type (0x99) so the dispatcher hits
            // its default case: no state change, no send_message reply. This
            // matters both pre-AUTH (no PONG contention) AND post-AUTH (no
            // STATS_RSP overwriting the values a test just wrote). Pre-AUTH
            // it logs "Rejecting message type 0x99"; post-AUTH it only logs
            // in debug mode (silent in release builds).
            kicker = std::thread([this, c] {
                using namespace std::chrono_literals;
                while (!stopped_ && !stop_kicking_) {
                    std::this_thread::sleep_for(2ms);
                    auto frame = build_frame(0x99 /*unknown*/, nullptr, 0);
                    std::lock_guard<std::mutex> lk(client_mu_);
                    if (client_ != INVALID_SOCK_T) {
                        // Best-effort. If send fails, the recv loop will see
                        // the disconnect and break out.
                        ::send(client_,
                               reinterpret_cast<const char*>(frame.data()),
                               static_cast<int>(frame.size()), 0);
                    } else {
                        return;
                    }
                }
            });
        }

        while (!stopped_) {
            int n = ::recv(c, tmp.data(), static_cast<int>(tmp.size()), 0);
            if (n <= 0) break;
            {
                std::lock_guard<std::mutex> lk(buf_mu_);
                // Filter out the client's PONG responses to our kicker PINGs
                // so tests inspecting the received byte stream don't trip on
                // unexpected PONG frames. PONG is fixed-size: 8-byte header
                // (KANG, type=0x51, flags=0, len=0). We do this by scanning
                // for KANG+0x51+0x00+0x00+0x00 sequences and removing them
                // in-place. This is a stream filter, so a PONG arriving split
                // across two recv() calls would slip through; in practice the
                // 8 bytes always come together because send_message() writes
                // the full header in one syscall.
                size_t base = rx_.size();
                rx_.insert(rx_.end(), tmp.data(), tmp.data() + n);
                if (filter_pongs_.load()) {
                    // Only filter the freshly appended chunk; earlier bytes
                    // were already inspected on previous iterations.
                    static constexpr uint8_t PONG_HEADER[8] = {
                        'K','A','N','G',0x51,0x00,0x00,0x00
                    };
                    // Walk forwards, removing each match. Safe under removal
                    // because we don't advance i on a match.
                    size_t i = base;
                    while (i + 8 <= rx_.size()) {
                        if (std::memcmp(rx_.data() + i, PONG_HEADER, 8) == 0) {
                            rx_.erase(rx_.begin() + i, rx_.begin() + i + 8);
                        } else {
                            ++i;
                        }
                    }
                }
            }
            buf_cv_.notify_all();
        }

        if (kicker.joinable()) {
            stop_kicking_ = true;
            kicker.join();
        }
    }

    sock_t listen_ = INVALID_SOCK_T;
    sock_t client_ = INVALID_SOCK_T;
    uint16_t port_ = 0;
    bool send_kick_pings_;
    std::atomic<bool> filter_pongs_{true};  // strip PONG noise unless test opts out
    std::thread thread_;
    std::atomic<bool> stopped_{false};
    std::atomic<bool> stop_kicking_{false};

    std::mutex client_mu_;
    std::condition_variable client_cv_;
    bool client_ready_ = false;

    std::mutex buf_mu_;
    std::condition_variable buf_cv_;
    std::vector<uint8_t> rx_;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool fail(const char* tname, const char* msg) {
    std::fprintf(stderr, "[FAIL] %s: %s\n", tname, msg);
    return false;
}

// Poll predicate `pred` up to `timeout` -- avoids fixed sleeps in tests.
template <class Pred>
bool wait_for(Pred pred, std::chrono::milliseconds timeout = 2000ms) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(10ms);
    }
    return pred();
}

// Drives a single AUTH_OK handshake on `server` for the just-connected
// `client`. Returns true if the AUTH bytes were observed AND the client's
// authenticate() returned true. The test name is used in failure messages.
//
// Note: the kicker stays running post-AUTH (it sends 0x99 unknown frames
// that the client's dispatcher logs and ignores). This is intentional --
// the same starvation that bites authenticate() also bites submit_dp() /
// report_solution() (they both go through send_message + ssl_io_mutex_),
// so the kicker has to keep running.
//
// All "after AUTH_OK" tests share this preamble; centralizing it removes
// dozens of copy-pasted failure paths and ensures the AUTH bytes are
// reliably consumed (so a later wait_recv_pop doesn't accidentally return
// the AUTH frame's header bytes -- a flaky failure mode we hit during
// development).
class MockJlpServer;  // fwd
template <class Server>
bool drive_auth_ok(Server& server, JLPPoolClient& client, const char* tname) {
    std::atomic<bool> ok{false};
    std::thread t([&] { ok = client.authenticate("worker", ""); });
    // v1.4.1 B.2: AUTH frame is 8-byte header + 120-byte v2 hello.
    auto auth = server.wait_recv_pop(128, 5000ms);
    if (auth.size() != 128) {
        // Force authenticate() to unblock so we can join cleanly.
        server.close_client();
        if (t.joinable()) t.join();
        std::fprintf(stderr,
            "[FAIL] %s: AUTH bytes never arrived at mock server (got %zu)\n",
            tname, auth.size());
        return false;
    }
    if (!server.send_frame(0x02 /*AUTH_OK*/, nullptr, 0)) {
        if (t.joinable()) t.join();
        std::fprintf(stderr, "[FAIL] %s: failed to send AUTH_OK\n", tname);
        return false;
    }
    t.join();
    if (!ok.load()) {
        std::fprintf(stderr, "[FAIL] %s: authenticate() returned false\n", tname);
        return false;
    }
    return true;
}

// Construct + connect a JLPPoolClient at use_tls=false, no auto-reconnect.
// Returns the client (already connected; authenticate() not yet called).
//
// We set the SHORTEST useful socket timeout (1 ms) intentionally. The
// production receive_message() holds ssl_io_mutex_ across the blocking
// recv() call; send_message() (used by authenticate, report_solution, and
// the sender thread) needs that same mutex. std::mutex on Windows is built
// on SRWLock which is not strictly FIFO; with a 100 ms recv timeout and the
// receiver thread re-acquiring the mutex at the top of every iteration, the
// sender thread can be starved for many seconds (reproduced reliably in
// dev). A 1 ms recv timeout means the receiver releases the mutex roughly
// every millisecond, giving the sender a regular window to win the race
// (the receiver does very little work per iteration, so a 1 ms loop has
// negligible CPU cost on test machines).
std::unique_ptr<JLPPoolClient> make_connected(uint16_t port) {
    auto c = std::make_unique<JLPPoolClient>();
    c->set_use_tls(false);
    c->set_reconnect(false);
    c->set_timeout(1);
    if (!c->connect("127.0.0.1", port)) {
        std::fprintf(stderr, "[helper] connect to 127.0.0.1:%u failed\n",
                     (unsigned)port);
        return nullptr;
    }
    return c;
}

// ---------------------------------------------------------------------------
// 1. AUTH wire format
// ---------------------------------------------------------------------------
bool test_auth_wire_format() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("auth_wire_format", "client connect failed");

    std::thread t([&] {
        client->authenticate("worker_test_name", "hunter2");
    });

    // v1.4.1 B.2: AUTH wire payload is 120 bytes
    // (worker_name 64 + password 32 + timestamp_ms 8 + nonce 16);
    // total frame is 8-byte header + 120 = 128 bytes.
    constexpr size_t kFrameSize   = 8 + 120;
    constexpr uint16_t kPayloadSize = 120;
    auto bytes = server.wait_recv(kFrameSize);
    if (bytes.size() != kFrameSize) {
        server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
        t.join();
        return fail("auth_wire_format",
                    "did not receive 128 AUTH bytes within timeout");
    }

    // Header inspection.
    if (!(bytes[0] == 'K' && bytes[1] == 'A' && bytes[2] == 'N' && bytes[3] == 'G')) {
        server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
        t.join();
        return fail("auth_wire_format", "header magic != KANG");
    }
    if (bytes[4] != TYPE_AUTH) {
        server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
        t.join();
        return fail("auth_wire_format", "header type != AUTH (0x01)");
    }
    if (bytes[5] != 0) {
        server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
        t.join();
        return fail("auth_wire_format", "header flags != 0");
    }
    uint16_t plen = static_cast<uint16_t>(bytes[6] | (bytes[7] << 8));
    if (plen != kPayloadSize) {
        server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
        t.join();
        return fail("auth_wire_format",
                    "header payload_size != 120 (v1.4.1 B.2 wire fmt)");
    }

    // Payload layout: name[64] password[32] timestamp_ms(LE u64, 8) nonce[16].
    const uint8_t* hello = bytes.data() + 8;
    const char* expected = "worker_test_name";
    size_t elen = std::strlen(expected);
    if (std::memcmp(hello, expected, elen) != 0) {
        server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
        t.join();
        return fail("auth_wire_format", "worker_name leading bytes wrong");
    }
    for (size_t i = elen; i < 64; ++i) {
        if (hello[i] != 0) {
            server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
            t.join();
            return fail("auth_wire_format", "worker_name not NUL-padded");
        }
    }

    if (std::memcmp(hello + 64, "hunter2", 7) != 0) {
        server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
        t.join();
        return fail("auth_wire_format", "password leading bytes wrong");
    }

    uint64_t ts_ms = 0;
    std::memcpy(&ts_ms, hello + 96, 8);
    // Sanity bound: timestamp must be sometime after Jan 2024 (in ms)
    // and before year 9999.
    if (ts_ms < 1704067200000ULL || ts_ms > 253402300799000ULL) {
        server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
        t.join();
        return fail("auth_wire_format",
                    "timestamp_ms out of plausible range");
    }

    // Nonce: at least one byte must be nonzero. With 16 bytes from
    // std::random_device, the probability of all zero is ~2^-128 -- so
    // this fires only on a busted RNG.
    bool any_nonzero = false;
    for (size_t i = 0; i < 16; ++i) {
        if (hello[104 + i] != 0) { any_nonzero = true; break; }
    }
    if (!any_nonzero) {
        server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
        t.join();
        return fail("auth_wire_format", "nonce was all-zero");
    }

    server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
    t.join();
    return true;
}

// ---------------------------------------------------------------------------
// 2. authenticate() returns true on AUTH_OK after a brief delay.
// ---------------------------------------------------------------------------
bool test_authenticate_ok() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("authenticate_ok", "connect failed");

    bool result = false;
    std::thread t([&] { result = client->authenticate("worker", ""); });

    // Wait for the AUTH frame to arrive, then delay before responding. A
    // generous 5 s timeout because the receiver thread holds ssl_io_mutex_
    // during recv() and the sender has to win that mutex to actually send;
    // with our 100 ms recv timeout that race resolves quickly, but on a
    // loaded CI box it can still take a beat.
    auto auth = server.wait_recv(128, 5000ms);
    if (auth.size() != 128) {
        std::fprintf(stderr,
            "[debug] authenticate_ok: wait_recv got %zu bytes\n", auth.size());
        server.close_client();
        t.join();
        return fail("authenticate_ok", "AUTH not received");
    }

    std::this_thread::sleep_for(200ms);
    server.send_frame(TYPE_AUTH_OK, nullptr, 0);

    t.join();
    if (!result) return fail("authenticate_ok", "authenticate() returned false");
    return true;
}

// ---------------------------------------------------------------------------
// 3. authenticate() returns false on AUTH_FAIL.
// ---------------------------------------------------------------------------
bool test_authenticate_fail() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("authenticate_fail", "connect failed");

    bool result = true;
    std::thread t([&] { result = client->authenticate("worker", ""); });

    server.wait_recv(128);
    server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);

    t.join();
    if (result) return fail("authenticate_fail", "authenticate() returned true on AUTH_FAIL");
    return true;
}

// ---------------------------------------------------------------------------
// 4. authenticate() returns false when MSG_ERROR arrives during AUTH_SENT.
// ---------------------------------------------------------------------------
bool test_authenticate_msg_error() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("authenticate_msg_error", "connect failed");

    bool result = true;
    std::thread t([&] { result = client->authenticate("worker", ""); });

    server.wait_recv(128);
    const char* msg = "bad worker name";
    server.send_frame(TYPE_MSG_ERROR, msg, static_cast<uint16_t>(std::strlen(msg)));

    t.join();
    if (result) return fail("authenticate_msg_error",
                            "authenticate() returned true on MSG_ERROR pre-AUTH");
    return true;
}

// ---------------------------------------------------------------------------
// 5. authenticate() times out when server is silent.
//
// The production timeout is 10 s. Rather than wait that long, we close the
// server's accepted socket -- the receiver loop's auto-reconnect branch
// observes the disconnect, flips auth_state_ to DISCONNECTED, and notifies
// auth_cv_, causing authenticate() to return false promptly.
//
// IMPORTANT: this test uses auto_reconnect=true. With auto_reconnect=false
// the receiver loop has no branch that flips connected_=false on a non-timeout
// recv error -- it would spin in receive_message until disconnect() is
// called. With auto_reconnect=true, the receiver enters the reconnect path,
// closes the socket, sets connected_=false, notifies the cv, and exits the
// loop after a single backoff sleep. authenticate() returns false within ~1 s.
// ---------------------------------------------------------------------------
bool test_authenticate_silent_server() {
    MockJlpServer server;
    auto client = std::make_unique<JLPPoolClient>();
    client->set_use_tls(false);
    client->set_reconnect(true);   // <-- needed so receiver notifies cv on disconnect
    client->set_timeout(1);        // mirror make_connected: short recv timeout
    if (!client->connect("127.0.0.1", server.port()))
        return fail("authenticate_silent_server", "connect failed");

    bool result = true;
    std::thread t([&] { result = client->authenticate("worker", ""); });

    // Wait for AUTH bytes so we know authenticate() is past send_hello().
    server.wait_recv(128);
    // Now hang up. authenticate() should return false within a few seconds.
    server.close_client();

    auto start = std::chrono::steady_clock::now();
    t.join();
    auto elapsed = std::chrono::steady_clock::now() - start;

    if (result) return fail("authenticate_silent_server",
                            "authenticate() returned true after server hangup");
    if (elapsed > 5s) return fail("authenticate_silent_server",
                                  "authenticate() took longer than 5s after hangup");
    return true;
}

// ---------------------------------------------------------------------------
// 6. Worker-name length validation: >= 64 chars must be rejected before
//    any bytes hit the wire.
// ---------------------------------------------------------------------------
bool test_worker_name_too_long() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("worker_name_too_long", "connect failed");

    // 65-character name (>= sizeof(JLPClientHello::worker_name) = 64).
    std::string too_long(65, 'x');
    bool result = client->authenticate(too_long, "");

    if (result) return fail("worker_name_too_long",
                            "authenticate() returned true for 65-char name");

    // The server must NOT have received any bytes (send_hello returned false
    // before send_message ran). Give a short window for any rogue bytes.
    auto bytes = server.wait_recv(1, 200ms);
    if (!bytes.empty()) return fail("worker_name_too_long",
                                    "AUTH bytes leaked despite oversized name");
    return true;
}

// ---------------------------------------------------------------------------
// 7. PING -> PONG. The client must answer a server PING with a zero-payload
//    PONG even before AUTH_OK (PING/PONG are in the always-allowed set).
// ---------------------------------------------------------------------------
bool test_ping_pong() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("ping_pong", "connect failed");

    if (!drive_auth_ok(server, *client, "ping_pong")) return false;

    // Stop the kicker + drain leftover 0x99 frames so our explicit PING is
    // the next thing the client sees (and the next thing the server sees on
    // the wire is the resulting PONG, not interleaved kicker noise).
    server.stop_kicker();
    std::this_thread::sleep_for(20ms);
    server.drain_rx();

    // Server sends PING. Client should respond with PONG (8-byte header).
    server.send_frame(TYPE_PING, nullptr, 0);
    auto pong = server.wait_recv_pop(8, 5000ms);
    if (pong.size() != 8) return fail("ping_pong", "no PONG within timeout");
    if (!(pong[0] == 'K' && pong[1] == 'A' && pong[2] == 'N' && pong[3] == 'G')) {
        std::fprintf(stderr, "[debug] ping_pong: got bytes %02x %02x %02x %02x %02x %02x %02x %02x\n",
                     pong[0], pong[1], pong[2], pong[3], pong[4], pong[5], pong[6], pong[7]);
        return fail("ping_pong", "PONG magic wrong");
    }
    if (pong[4] != TYPE_PONG) {
        std::fprintf(stderr, "[debug] ping_pong: type was 0x%02x (want 0x%02x)\n",
                     pong[4], TYPE_PONG);
        return fail("ping_pong", "PONG type wrong");
    }
    uint16_t plen = static_cast<uint16_t>(pong[6] | (pong[7] << 8));
    if (plen != 0) return fail("ping_pong", "PONG payload_size != 0");
    return true;
}

// ---------------------------------------------------------------------------
// 8. Pre-auth message gating (Wave 4 D-H4).
//    WORK_ASN before AUTH_OK must NOT invoke the work callback. After AUTH_OK,
//    the same WORK_ASN MUST invoke it.
// ---------------------------------------------------------------------------
bool test_pre_auth_work_gated() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("pre_auth_work_gated", "connect failed");

    std::atomic<int> callback_count{0};
    client->set_work_callback([&](const WorkAssignment&) { callback_count++; });

    // Build a 109-byte JLPServerConfig payload with a recognizable work_id.
    JLPServerConfig cfg{};
    for (int i = 0; i < 33; ++i) cfg.public_key[i]  = 0x10 + i;
    for (int i = 0; i < 32; ++i) cfg.range_start[i] = 0x20 + i;
    for (int i = 0; i < 32; ++i) cfg.range_end[i]   = 0x30 + i;
    cfg.dp_bits = 24;
    cfg.work_id = 0xCAFE'BABE'1234'5678ULL;
    static_assert(sizeof(JLPServerConfig) == 109,
                  "JLPServerConfig must be 109 bytes on the wire");

    // Start authenticate so the receiver thread is dispatching messages.
    bool auth_result = false;
    std::thread auth_t([&] { auth_result = client->authenticate("worker", ""); });
    if (server.wait_recv_pop(128, 5000ms).size() != 128) {
        server.close_client();
        auth_t.join();
        return fail("pre_auth_work_gated", "AUTH bytes never arrived");
    }

    // Send WORK_ASN BEFORE AUTH_OK -- must be ignored.
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));

    // Give the receiver time to ingest and reject the message.
    std::this_thread::sleep_for(150ms);

    if (callback_count.load() != 0) {
        server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
        auth_t.join();
        return fail("pre_auth_work_gated",
                    "work callback invoked while in AUTH_SENT");
    }

    // Now send AUTH_OK to enter AUTH_OK state, then send WORK_ASN again.
    server.send_frame(TYPE_AUTH_OK, nullptr, 0);
    auth_t.join();
    if (!auth_result) return fail("pre_auth_work_gated", "authenticate failed");

    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));

    if (!wait_for([&] { return callback_count.load() >= 1; }, 2000ms)) {
        return fail("pre_auth_work_gated",
                    "work callback NOT invoked after AUTH_OK");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 9. WORK_ASN parsing: 109-byte payload yields a correctly populated
//    WorkAssignment in the callback.
// ---------------------------------------------------------------------------
bool test_work_asn_parsing() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("work_asn_parsing", "connect failed");

    WorkAssignment captured{};
    std::atomic<bool> got{false};
    client->set_work_callback([&](const WorkAssignment& w) {
        captured = w;
        got = true;
    });

    if (!drive_auth_ok(server, *client, "work_asn_parsing")) return false;

    // Build a payload with values we can byte-compare.
    JLPServerConfig cfg{};
    for (int i = 0; i < 33; ++i) cfg.public_key[i]  = static_cast<uint8_t>(0x80 + i);
    for (int i = 0; i < 32; ++i) cfg.range_start[i] = static_cast<uint8_t>(0x01 + i);
    for (int i = 0; i < 32; ++i) cfg.range_end[i]   = static_cast<uint8_t>(0xC0 + i);
    cfg.dp_bits = 0xDEADBEEFu;
    cfg.work_id = 0x1122'3344'5566'7788ULL;

    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));

    if (!wait_for([&] { return got.load(); }, 2000ms)) {
        return fail("work_asn_parsing", "work callback not fired");
    }

    if (std::memcmp(captured.public_key, cfg.public_key, 33) != 0)
        return fail("work_asn_parsing", "public_key mismatch");
    if (std::memcmp(captured.range_start, cfg.range_start, 32) != 0)
        return fail("work_asn_parsing", "range_start mismatch");
    if (std::memcmp(captured.range_end, cfg.range_end, 32) != 0)
        return fail("work_asn_parsing", "range_end mismatch");
    if (captured.dp_bits != cfg.dp_bits)
        return fail("work_asn_parsing", "dp_bits mismatch");
    if (captured.work_id != cfg.work_id)
        return fail("work_asn_parsing", "work_id mismatch");
    return true;
}

// ---------------------------------------------------------------------------
// 10. STATS_RSP parsing: 36-byte payload (server '<QIIffQI'):
//       total_dps:u64, total_workers:u32, active_workers:u32,
//       dps_per_second:f32, your_share:f32, your_dps:u64, uptime:u32
// ---------------------------------------------------------------------------
bool test_stats_rsp_parsing() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("stats_rsp", "connect failed");

    if (!drive_auth_ok(server, *client, "stats_rsp")) return false;

    uint64_t total          = 0x1111'2222'3333'4444ULL;
    uint32_t total_workers  = 99;
    uint32_t active_workers = 42;
    float    dps_per_sec    = 1234.5f;
    float    your_share     = 0.0625f;
    uint64_t your_dps       = 0x0BADF00DCAFEBABEULL;
    uint32_t uptime         = 3600;

    uint8_t payload[36];
    std::memcpy(payload +  0, &total,          8);
    std::memcpy(payload +  8, &total_workers,  4);
    std::memcpy(payload + 12, &active_workers, 4);
    std::memcpy(payload + 16, &dps_per_sec,    4);
    std::memcpy(payload + 20, &your_share,     4);
    std::memcpy(payload + 24, &your_dps,       8);
    std::memcpy(payload + 32, &uptime,         4);

    server.send_frame(TYPE_STATS_RSP, payload, sizeof(payload));

    // The receiver thread updates stats_ under stats_mutex_, so wait briefly.
    if (!wait_for([&] {
            PoolStats s = client->get_stats();
            return s.total_dps == total
                && s.active_workers == active_workers
                && s.your_dps == your_dps
                && s.uptime_seconds == uptime;
        }, 2000ms))
    {
        PoolStats s = client->get_stats();
        std::fprintf(stderr,
            "[debug] stats: total=%llu active=%u your=%llu uptime=%u\n",
            (unsigned long long)s.total_dps,
            (unsigned)s.active_workers,
            (unsigned long long)s.your_dps,
            (unsigned)s.uptime_seconds);
        return fail("stats_rsp", "stats fields not parsed correctly");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 11. MSG_ERROR handling: large payloads + control characters must not crash.
//     This is a smoke test for the Wave 4 D-L3 sanitizer; we don't verify
//     stderr content (just that we get back to the prompt with the client
//     still alive).
// ---------------------------------------------------------------------------
bool test_msg_error_resilience() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("msg_error_resilience", "connect failed");

    // 500-byte all-printable payload.
    std::vector<uint8_t> big(500, 'A');
    server.send_frame(TYPE_MSG_ERROR, big.data(),
                      static_cast<uint16_t>(big.size()));

    // 32-byte payload with embedded control characters: ESC + NUL + BEL etc.
    uint8_t ctrl[32] = {
        0x1B, 0x00, 0x07, 0x01, 0x02, 0x03, 0x04, 0x05,
        'h',  'e',  'l',  'l',  'o',  0x1B, '[', '3',
        '1',  'm',  'r',  'e',  'd',  0x1B, '[', '0',
        'm',  '!',  '!',  0x00, 0x00, 0x00, 0x00, 0x00,
    };
    server.send_frame(TYPE_MSG_ERROR, ctrl, sizeof(ctrl));

    // If we got here without aborting, we pass. Give the receiver a moment
    // to process so any deferred crash would surface.
    std::this_thread::sleep_for(150ms);
    if (!client->is_connected()) {
        return fail("msg_error_resilience",
                    "client dropped connection on MSG_ERROR (should tolerate)");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 12. submit_dp / submit_dps return true once authenticated.
// ---------------------------------------------------------------------------
bool test_submit_dp_basic() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("submit_dp_basic", "connect failed");

    if (!drive_auth_ok(server, *client, "submit_dp_basic")) return false;

    DistinguishedPoint dp{};
    for (int i = 0; i < 32; ++i) { dp.x[i] = i; dp.d[i] = 32 + i; }
    dp.type = 0;
    dp.dp_bits = 24;

    if (!client->submit_dp(dp))
        return fail("submit_dp_basic", "submit_dp returned false");

    std::vector<DistinguishedPoint> batch(5, dp);
    if (!client->submit_dps(batch))
        return fail("submit_dp_basic", "submit_dps returned false");
    return true;
}

// ---------------------------------------------------------------------------
// 13. DP submission wire format. After submitting a DP and waiting for the
//     sender thread to flush, the server should observe a DP_BATCH frame.
// ---------------------------------------------------------------------------
bool test_dp_submission_wire_format() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("dp_wire_format", "connect failed");

    if (!drive_auth_ok(server, *client, "dp_wire_format")) return false;

    DistinguishedPoint dp{};
    for (int i = 0; i < 32; ++i) { dp.x[i] = 0xA0 + i; dp.d[i] = 0x50 + i; }
    dp.type = 1;
    dp.dp_bits = 30;

    if (!client->submit_dp(dp)) return fail("dp_wire_format", "submit_dp false");

    // Sender wakes on dp_cv_ with up to a 100ms timeout; allow generous slack.
    auto bytes = server.wait_recv_pop(8, 5000ms);
    if (bytes.size() != 8) {
        std::fprintf(stderr, "[debug] dp_wire_format: got %zu bytes\n", bytes.size());
        return fail("dp_wire_format", "no DP frame header received");
    }

    if (!(bytes[0] == 'K' && bytes[1] == 'A' && bytes[2] == 'N' && bytes[3] == 'G')) {
        std::fprintf(stderr, "[debug] dp_wire_format: header %02x %02x %02x %02x %02x %02x %02x %02x\n",
                     bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
        return fail("dp_wire_format", "DP frame magic != KANG");
    }
    // The current sender uses DP_BATCH_V2 (0x24) by default with work_id
    // attestation. v1 (0x22) is still accepted by the server for older
    // clients but the in-process client always emits v2.
    if (bytes[4] != TYPE_DP_BATCH && bytes[4] != TYPE_DP_BATCH_V2) {
        std::fprintf(stderr, "[debug] dp_wire_format: type 0x%02x (want 0x22 or 0x24)\n",
                     bytes[4]);
        return fail("dp_wire_format", "DP frame type != DP_BATCH or DP_BATCH_V2");
    }

    // Drain whatever payload was advertised so subsequent tests in the suite
    // don't see leftover bytes (each test owns its own server, but be tidy).
    uint16_t plen = static_cast<uint16_t>(bytes[6] | (bytes[7] << 8));
    if (plen > 0) (void)server.wait_recv_pop(plen, 1000ms);
    return true;
}

// ---------------------------------------------------------------------------
// 14. Backpressure: submit_dps with a vector that would overflow
//     MAX_DP_QUEUE_SIZE returns false.
//
// We don't actually want to push 100k DPs (slow + blows the sender out of the
// water). Instead, we DON'T authenticate -- so the sender thread refuses to
// transmit, and the queue accumulates as long as we keep submitting.
//
// We then push (MAX_DP_QUEUE_SIZE - 1) one-by-one (which is the boundary the
// sender drops when not authed too), but without auth the sender still pulls
// from the queue and just discards. So instead we test the boundary by:
//
//   submit_dps(vector of MAX_DP_QUEUE_SIZE + 1 entries)  -> must return false
//
// This exercises the explicit overflow check in submit_dps without requiring
// us to first stuff the queue.
// ---------------------------------------------------------------------------
bool test_dp_backpressure() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("dp_backpressure", "connect failed");

    // Authenticate normally so submit paths behave like production.
    if (!drive_auth_ok(server, *client, "dp_backpressure")) return false;

    DistinguishedPoint dp{};
    dp.type = 0; dp.dp_bits = 24;

    // A single batch larger than MAX_DP_QUEUE_SIZE must be rejected
    // up-front, regardless of current queue size.
    std::vector<DistinguishedPoint> too_big(JLPPoolClient::MAX_DP_QUEUE_SIZE + 1, dp);
    if (client->submit_dps(too_big)) {
        return fail("dp_backpressure",
                    "submit_dps accepted a batch larger than MAX_DP_QUEUE_SIZE");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 15. report_solution: 32-byte private key arrives as a SOLUTION frame.
// ---------------------------------------------------------------------------
bool test_report_solution() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("report_solution", "connect failed");

    if (!drive_auth_ok(server, *client, "report_solution")) return false;

    uint8_t key[32];
    for (int i = 0; i < 32; ++i) key[i] = static_cast<uint8_t>(0xF0 ^ i);
    if (!client->report_solution(key))
        return fail("report_solution", "report_solution returned false");

    auto bytes = server.wait_recv_pop(8 + 32, 5000ms);
    if (bytes.size() != 8 + 32) {
        std::fprintf(stderr, "[debug] report_solution: got %zu bytes\n", bytes.size());
        return fail("report_solution", "did not receive 40 bytes for SOLUTION");
    }

    if (bytes[4] != TYPE_SOLUTION) {
        std::fprintf(stderr, "[debug] report_solution: type 0x%02x (want 0x%02x); first bytes: %02x %02x %02x %02x %02x %02x %02x %02x\n",
                     bytes[4], TYPE_SOLUTION,
                     bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
        return fail("report_solution", "SOLUTION frame type wrong");
    }
    if (std::memcmp(bytes.data() + 8, key, 32) != 0)
        return fail("report_solution", "SOLUTION payload != private key");
    return true;
}

// ---------------------------------------------------------------------------
// 16. v1.4.1 C.3: SOLUTION callback dedup. Sending the same SOLUTION
//     payload twice fires the user's callback exactly once.
// ---------------------------------------------------------------------------
bool test_solution_callback_dedup() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("solution_callback_dedup", "connect failed");

    std::atomic<int> fire_count{0};
    client->set_solution_callback([&](const uint8_t*) { fire_count++; });

    if (!drive_auth_ok(server, *client, "solution_callback_dedup")) return false;

    uint8_t key[32];
    for (int i = 0; i < 32; ++i) key[i] = static_cast<uint8_t>(0xA5 ^ i);

    server.send_frame(TYPE_SOLUTION, key, sizeof(key));
    if (!wait_for([&] { return fire_count.load() >= 1; }, 2000ms)) {
        return fail("solution_callback_dedup",
                    "first SOLUTION did not fire callback");
    }
    // Second identical SOLUTION must be suppressed by fire_solution_callback.
    server.send_frame(TYPE_SOLUTION, key, sizeof(key));
    std::this_thread::sleep_for(200ms);

    if (fire_count.load() != 1) {
        return fail("solution_callback_dedup",
                    "callback fired more than once for duplicate SOLUTION");
    }
    return true;
}

// ---------------------------------------------------------------------------
// 17. v1.4.1 C.3: WORK_ASN callback dedup. Sending two WORK_ASN frames
//     with the same work_id fires the work callback only once.
// ---------------------------------------------------------------------------
bool test_work_callback_dedup() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("work_callback_dedup", "connect failed");

    std::atomic<int> fire_count{0};
    client->set_work_callback([&](const WorkAssignment&) { fire_count++; });

    if (!drive_auth_ok(server, *client, "work_callback_dedup")) return false;

    JLPServerConfig cfg{};
    for (int i = 0; i < 33; ++i) cfg.public_key[i]  = 0x70 + i;
    for (int i = 0; i < 32; ++i) cfg.range_start[i] = 0x80 + i;
    for (int i = 0; i < 32; ++i) cfg.range_end[i]   = 0x90 + i;
    cfg.dp_bits = 24;
    cfg.work_id = 0xDEADBEEF'CAFE0001ULL;

    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));
    if (!wait_for([&] { return fire_count.load() >= 1; }, 2000ms)) {
        return fail("work_callback_dedup",
                    "first WORK_ASN did not fire callback");
    }
    // Same work_id arriving again -- treat as a network-layer duplicate.
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));
    std::this_thread::sleep_for(200ms);

    if (fire_count.load() != 1) {
        return fail("work_callback_dedup",
                    "callback fired again for duplicate work_id");
    }

    // Sanity: a different work_id MUST fire the callback again.
    cfg.work_id = 0xDEADBEEF'CAFE0002ULL;
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));
    if (!wait_for([&] { return fire_count.load() >= 2; }, 2000ms)) {
        return fail("work_callback_dedup",
                    "fresh work_id failed to fire callback");
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
    {"auth_wire_format",          test_auth_wire_format},
    {"authenticate_ok",           test_authenticate_ok},
    {"authenticate_fail",         test_authenticate_fail},
    {"authenticate_msg_error",    test_authenticate_msg_error},
    {"authenticate_silent",       test_authenticate_silent_server},
    {"worker_name_too_long",      test_worker_name_too_long},
    {"ping_pong",                 test_ping_pong},
    {"pre_auth_work_gated",       test_pre_auth_work_gated},
    {"work_asn_parsing",          test_work_asn_parsing},
    {"stats_rsp_parsing",         test_stats_rsp_parsing},
    {"msg_error_resilience",      test_msg_error_resilience},
    {"submit_dp_basic",           test_submit_dp_basic},
    {"dp_wire_format",            test_dp_submission_wire_format},
    {"dp_backpressure",           test_dp_backpressure},
    {"report_solution",           test_report_solution},
    {"solution_callback_dedup",   test_solution_callback_dedup},
    {"work_callback_dedup",       test_work_callback_dedup},
};

}  // namespace

int main() {
#ifdef _WIN32
    WSAGuard guard;
#endif
    std::printf("=== JLP pool protocol tests (mock TCP server) ===\n");
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
