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
//    9.  WORK_ASN parsing: 126-byte JLPServerConfig payload populates the
//        WorkAssignment fields exactly (v1.5: extended from 109 to 126
//        by the asymmetric tame/wild fields at the tail).
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

#include <algorithm>
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
// task #34: WSACleanup() in the destructor raced with ntdll's
// process-exit cleanup of any sockets a JLPPoolClient's receiver
// thread still held a reference to. Even though each test fully
// destructs its client (which joins the receiver and closes the
// socket), Windows' kernel-level socket-handle release can lag the
// closesocket() call by a few microseconds; if WSACleanup() fires in
// that gap the ntdll RtlpFreezeTimeBias cleanup path access-violates
// from a stale per-thread WinSock state. Per MSDN, "WSACleanup()
// should be called only if WSAStartup completes successfully" and
// "an application is not required to call WSACleanup; the system
// releases all per-process WinSock state at process exit." Dropping
// the explicit WSACleanup() lets the OS do the right thing without
// the cross-thread race.
struct WSAGuard {
    WSAGuard()  { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); }
    ~WSAGuard() { /* see above; intentionally no WSACleanup() */ }
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
constexpr uint8_t TYPE_MAINTENANCE = 0x60;  // v1.5.4 back-off signal
constexpr uint8_t TYPE_MSG_ERROR = 0xFF;

// Build [magic=KANG][type][flags=PROTOCOL_VERSION][len:LE u16][payload].
// v1.4.2 B.5: flags now carries PROTOCOL_VERSION. The client rejects
// non-matching flags as a protocol version mismatch, so mock-server
// frames must use the same value.
// v1.5 (protocol_version=3): bumped from 2 to 3 to track
// jlp_wire::PROTOCOL_VERSION; the tests below verify behaviors that
// are protocol-version agnostic once the handshake completes (work
// callback firing, dedup, dp_sequence anti-replay, etc.).
constexpr uint8_t MOCK_PROTOCOL_VERSION = 4;  // v1.5.4: header flags = PROTOCOL_VERSION 4
std::vector<uint8_t> build_frame(uint8_t type, const void* payload, uint16_t len) {
    std::vector<uint8_t> out;
    out.reserve(8 + len);
    out.push_back('K'); out.push_back('A'); out.push_back('N'); out.push_back('G');
    out.push_back(type);
    out.push_back(MOCK_PROTOCOL_VERSION);              // v1.4.2 B.5: flags = PROTOCOL_VERSION
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

    // TP-10 regression aid: send the same frame byte-by-byte with a
    // short inter-byte delay. Stresses the stream-resync helper
    // (jlp_pool_client::receive_message + header_partial_ buffer) by
    // GUARANTEEING the receiver hits SO_RCVTIMEO partial-read in the
    // middle of every multi-byte field. If the resync code is wrong,
    // this test misaligns the stream and the receiver eventually
    // tries to dispatch garbage.
    bool send_frame_fragmented(uint8_t type, const void* payload,
                               uint16_t len,
                               std::chrono::microseconds inter_byte_delay =
                                   std::chrono::microseconds(500)) {
        if (!wait_for_client()) return false;
        auto bytes = build_frame(type, payload, len);
        std::lock_guard<std::mutex> lk(client_mu_);
        if (client_ == INVALID_SOCK_T) return false;
        const char* p = reinterpret_cast<const char*>(bytes.data());
        for (size_t i = 0; i < bytes.size(); ++i) {
            int n = ::send(client_, p + i, 1, 0);
            if (n <= 0) return false;
            // Brief pause so the client's recv() drains the single byte
            // and tries again, hitting SO_RCVTIMEO mid-frame.
            std::this_thread::sleep_for(inter_byte_delay);
        }
        return true;
    }

    // Close the accepted client socket (simulates server-side disconnect).
    //
    // The shutdown(SD_BOTH) call before closesocket is load-bearing on
    // Windows. MSDN explicitly documents that calling closesocket on a
    // socket while another thread is blocked in recv() on the same
    // handle leaves the recv in an UNDEFINED state. With our run()
    // thread parked in ::recv(c) waiting for client bytes, a bare
    // closesocket from this thread reliably reproduced an
    // execute-at-NULL access violation in the recv-blocked thread on
    // the order of 1-in-3 runs of test_jlp_pool_protocol (crash sites
    // distributed across multiple unrelated subtests because the race
    // window opens on every test fixture teardown). shutdown(SD_BOTH)
    // signals a graceful close, so the blocked recv returns 0 BEFORE
    // closesocket actually frees the handle, eliminating the
    // unspecified-behaviour window. POSIX shutdown(SHUT_RDWR) has the
    // same semantics, so the call works unmodified on Linux + macOS.
    void close_client() {
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

    void stop() {
        if (stopped_.exchange(true)) return;
        // Signal the kicker BEFORE closing anything so it stops trying
        // to send into a soon-to-be-half-shut socket. The kicker checks
        // stopped_/stop_kicking_ at the top of every iteration; setting
        // both here means the kicker bails out on its next 2ms wakeup
        // (or in the middle of the current iteration if it has not yet
        // entered ::send). Without this, stop() could race the kicker
        // into a window where the kicker holds client_mu_ blocked in
        // ::send while close_client() spins waiting for the same lock,
        // serializing teardown against the kernel's send-buffer drain.
        stop_kicking_ = true;
        // Close listen so a pending accept() unblocks.
        if (listen_ != INVALID_SOCK_T) {
            CLOSE_SOCK(listen_);
            listen_ = INVALID_SOCK_T;
        }
        // shutdown + close on the accepted client socket. The
        // shutdown(SD_BOTH) is what unblocks the recv-blocked run()
        // thread without invoking the closesocket-during-recv UB; see
        // the comment on close_client() above.
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
        // Apply a short recv timeout to the accepted socket so the recv
        // loop below can poll stopped_ on its own without depending on
        // close_client() to unblock it. The previous design relied on
        // close_client() to wake recv via a graceful shutdown(SD_BOTH)
        // plus closesocket, which still left a narrow Windows-specific
        // window where the recv-blocked thread could observe an
        // inconsistent socket state during teardown. A 50ms poll keeps
        // CPU cost negligible while letting the run() thread exit on
        // its own clock rather than racing against the teardown thread.
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
            if (n == 0) {
                // Graceful FIN from peer.
                break;
            }
            if (n < 0) {
                // Distinguish a benign recv timeout (SO_RCVTIMEO above)
                // from a real socket error. On timeout we loop back to
                // check stopped_. On any other error the socket is gone
                // and we exit.
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
                // Filter out the client's PONG responses to our kicker PINGs
                // so tests inspecting the received byte stream don't trip on
                // unexpected PONG frames. PONG is fixed-size: 8-byte header
                // (KANG, type=0x51, flags=PROTOCOL_VERSION, len=0). v1.4.2 B.5
                // landed flags=PROTOCOL_VERSION on every client send, so the
                // 6th byte is no longer 0; track MOCK_PROTOCOL_VERSION so the
                // filter actually matches (pre-fix the filter silently no-op'd
                // because byte 5 didn't match).
                size_t base = rx_.size();
                rx_.insert(rx_.end(), tmp.data(), tmp.data() + n);
                if (filter_pongs_.load()) {
                    // Only filter the freshly appended chunk; earlier bytes
                    // were already inspected on previous iterations.
                    static const uint8_t PONG_HEADER[8] = {
                        'K','A','N','G',0x51,MOCK_PROTOCOL_VERSION,0x00,0x00
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
    // task #34: was set_timeout(1). The 1ms tight loop on recv timeouts
    // combined with the 1ms partial-read window on Windows
    // SO_RCVTIMEO+MSG_WAITALL corrupted stream alignment downstream.
    // The wire-side fix is in jlp_pool_client::receive_message
    // (stream-resync helper), but matching the test's timeout to a
    // realistic 50ms also removes the per-test high-frequency
    // teardown-race window so the matching wire-side guard does not
    // have to do all the work alone.
    c->set_timeout(50);
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
    if (bytes[5] != MOCK_PROTOCOL_VERSION) {
        // v1.4.2 B.5: client must set flags = PROTOCOL_VERSION (=2).
        server.send_frame(TYPE_AUTH_FAIL, nullptr, 0);
        t.join();
        return fail("auth_wire_format", "header flags != PROTOCOL_VERSION");
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
    client->set_timeout(50);       // mirror make_connected: see task #34 rationale
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
    // Callback captures live LONGER than the client (declared first, destroyed
    // last). Pre-fix the client was declared before `callback_count`, so the
    // atomic went out of scope first; the receiver thread joined inside
    // ~JLPPoolClient could still be mid fire_work_callback (copy made under
    // callbacks_mutex_, callback invoked outside the lock) writing through a
    // stale capture reference. The receiver join inside disconnect() ensures
    // no callback fires after `client` is destroyed, so as long as
    // `callback_count` outlives `client`, the lambda's reference is valid for
    // the entire duration the callback can be invoked.
    std::atomic<int> callback_count{0};
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("pre_auth_work_gated", "connect failed");

    client->set_work_callback([&](const WorkAssignment&) { callback_count++; });

    // v1.5: JLPServerConfig is now 126 bytes (was 109 in v1.4.x). The
    // pre-auth gate fires before the payload is parsed, so the test
    // outcome is unchanged; just update the sizeof contract.
    JLPServerConfig cfg{};
    for (int i = 0; i < 33; ++i) cfg.public_key[i]  = 0x10 + i;
    for (int i = 0; i < 32; ++i) cfg.range_start[i] = 0x20 + i;
    for (int i = 0; i < 32; ++i) cfg.range_end[i]   = 0x30 + i;
    cfg.dp_bits = 24;
    cfg.work_id = 0xCAFE'BABE'1234'5678ULL;
    cfg.kangaroo_type  = 1;   // TAME_ONLY (any non-zero satisfies v1.5)
    cfg.start_offset_a = 0;
    cfg.start_offset_b = 0;
    static_assert(sizeof(JLPServerConfig) == 126,
                  "v1.5: JLPServerConfig must be 126 bytes on the wire");

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
    // Tear the client down EXPLICITLY before the function returns so
    // the receiver thread is fully joined before `callback_count` even
    // begins its scoped lifetime exit. Relying on unique_ptr's implicit
    // destruction at function return is correct in principle (the
    // capture is declared first, destroyed last), but explicit reset
    // moves the join out of the stack-unwind window where Windows
    // exit-cleanup races are most visible. Also clears the std::function
    // first so even a stray late dispatch attempt is a no-op.
    client->set_work_callback(nullptr);
    client.reset();
    return true;
}

// ---------------------------------------------------------------------------
// 9. WORK_ASN parsing: 126-byte v1.5 payload yields a correctly populated
//    WorkAssignment in the callback. Pre-v1.5 the payload was 109 bytes;
//    the trailing 17 bytes carry kangaroo_type + start_offset_a +
//    start_offset_b.
// ---------------------------------------------------------------------------
bool test_work_asn_parsing() {
    // Callback captures (`captured`, `got`) must outlive `client`. See the
    // matching comment in test_pre_auth_work_gated for the full lifetime
    // rationale: the receiver thread is joined inside ~JLPPoolClient, so once
    // `client` is destroyed no callback can fire; declaring captures first
    // (destroyed last) keeps their addresses valid for the lambda's entire
    // observable lifetime.
    WorkAssignment captured{};
    std::atomic<bool> got{false};
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("work_asn_parsing", "connect failed");

    client->set_work_callback([&](const WorkAssignment& w) {
        captured = w;
        got = true;
    });

    if (!drive_auth_ok(server, *client, "work_asn_parsing")) return false;

    // Build a payload with values we can byte-compare. dp_bits must be in
    // the validated [8..32] window (the WORK_ASN handler drops the message
    // otherwise; see tests/test_jlp_pool_dp_bits_validation.cpp). Use 26 --
    // mid-range, still byte-distinctive (it is 0x0000001A on the wire).
    //
    // v1.5: the WORK_ASN payload now carries kangaroo_type +
    // start_offset_a/_b at the tail. Fill them with byte-distinctive
    // values so the round-trip assertions below catch any drift in the
    // wire decode path.
    JLPServerConfig cfg{};
    for (int i = 0; i < 33; ++i) cfg.public_key[i]  = static_cast<uint8_t>(0x80 + i);
    for (int i = 0; i < 32; ++i) cfg.range_start[i] = static_cast<uint8_t>(0x01 + i);
    for (int i = 0; i < 32; ++i) cfg.range_end[i]   = static_cast<uint8_t>(0xC0 + i);
    cfg.dp_bits        = 26u;
    cfg.work_id        = 0x1122'3344'5566'7788ULL;
    cfg.kangaroo_type  = 2;                                // WILD_ONLY
    cfg.start_offset_a = 0x0123'4567'89AB'CDEFULL;
    cfg.start_offset_b = 0x0123'4567'89AB'CDEFULL + 4096ULL;

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
    if (captured.kangaroo_type != cfg.kangaroo_type)
        return fail("work_asn_parsing", "v1.5 kangaroo_type mismatch");
    if (captured.start_offset_a != cfg.start_offset_a)
        return fail("work_asn_parsing", "v1.5 start_offset_a mismatch");
    if (captured.start_offset_b != cfg.start_offset_b)
        return fail("work_asn_parsing", "v1.5 start_offset_b mismatch");
    // Explicit teardown: clear the callback first (so no late dispatch
    // can touch the `captured`/`got` captures even via a copied
    // std::function inside fire_work_callback) and then reset the
    // unique_ptr so the receiver/sender threads are joined before the
    // function frame begins unwinding. See test_pre_auth_work_gated for
    // the full rationale on why moving the join out of the implicit
    // destructor window improves teardown determinism on Windows.
    client->set_work_callback(nullptr);
    client.reset();
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
            PoolStatsLocal s = client->get_stats();
            return s.total_dps == total
                && s.active_workers == active_workers
                && s.your_dps == your_dps
                && s.uptime_seconds == uptime;
        }, 2000ms))
    {
        PoolStatsLocal s = client->get_stats();
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
// 15. report_solution: DELETED in v1.5.
//
// The v1.4.x client-to-server SOLUTION upload path was the theft surface
// v1.5 was designed to eliminate (worker self-solved, key in worker
// memory + on disk for up to 24h before pool ever saw it). The
// JLPPoolClient::report_solution() method, the PoolClient interface
// declaration, the PoolManager wrapper, the recovered_keys/<ts>.json
// persistence, the 24-hour retry uploader thread, and the SecureBuffer
// key staging were all removed together. SOLUTION is now strictly
// server-to-client (the broadcast path is still covered by
// test_solution_callback_dedup below). See
// .claude/tasks/v1.5-asymmetric-kangaroo.md.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 16. v1.4.1 C.3: SOLUTION callback dedup. Sending the same SOLUTION
//     payload twice fires the user's callback exactly once.
// ---------------------------------------------------------------------------
bool test_solution_callback_dedup() {
    // Capture must outlive client. See test_pre_auth_work_gated for the
    // lifetime rationale.
    std::atomic<int> fire_count{0};
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("solution_callback_dedup", "connect failed");

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
    // Explicit teardown: see test_pre_auth_work_gated. Clearing the
    // callback first guarantees that even a late receiver-thread copy
    // of the std::function cannot invoke the lambda after this point;
    // resetting the client then joins all background threads before
    // `fire_count` begins scoped destruction.
    client->set_solution_callback(nullptr);
    client.reset();
    return true;
}

// ---------------------------------------------------------------------------
// 17. v1.4.2 Pool-F6: WORK_ASN callback dedup MOVED from JLPPoolClient
//     to PoolManager. The supervisor recreates JLPPoolClient on every
//     reconnect, so client-instance dedup state (last_work_id_fired_)
//     reset to numeric_limits::max() after each reconnect and re-fired
//     the same work_id to the host. The manager-level dedup outlives
//     client churn. This test now verifies the OPPOSITE: the JLPPoolClient
//     ALWAYS forwards every WORK_ASN. It is no longer the dedup
//     authority. The PoolManager-level dedup is exercised by
//     tests/test_jlp_pool_manager.cpp (not this file).
// ---------------------------------------------------------------------------
bool test_work_callback_dedup() {
    // Capture must outlive client. See test_pre_auth_work_gated for the
    // lifetime rationale.
    std::atomic<int> fire_count{0};
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("work_callback_dedup", "connect failed");

    client->set_work_callback([&](const WorkAssignment&) { fire_count++; });

    if (!drive_auth_ok(server, *client, "work_callback_dedup")) return false;

    JLPServerConfig cfg{};
    for (int i = 0; i < 33; ++i) cfg.public_key[i]  = 0x70 + i;
    for (int i = 0; i < 32; ++i) cfg.range_start[i] = 0x80 + i;
    for (int i = 0; i < 32; ++i) cfg.range_end[i]   = 0x90 + i;
    cfg.dp_bits        = 24;
    cfg.work_id        = 0xDEADBEEF'CAFE0001ULL;
    cfg.kangaroo_type  = 1;   // TAME_ONLY (v1.5 requires non-zero)
    cfg.start_offset_a = 0;
    cfg.start_offset_b = 0;

    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));
    if (!wait_for([&] { return fire_count.load() >= 1; }, 2000ms)) {
        return fail("work_callback_dedup",
                    "first WORK_ASN did not fire callback");
    }
    // Same work_id arriving again. Post-Pool-F6 the JLPPoolClient
    // forwards every WORK_ASN (dedup is the manager's job); we expect
    // fire_count to reach 2.
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));
    if (!wait_for([&] { return fire_count.load() >= 2; }, 2000ms)) {
        return fail("work_callback_dedup",
                    "duplicate WORK_ASN should re-fire at client level "
                    "(dedup moved to PoolManager in v1.4.2 Pool-F6)");
    }

    // Sanity: a different work_id MUST also fire the callback.
    cfg.work_id = 0xDEADBEEF'CAFE0002ULL;
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));
    if (!wait_for([&] { return fire_count.load() >= 3; }, 2000ms)) {
        return fail("work_callback_dedup",
                    "fresh work_id failed to fire callback");
    }
    // Explicit teardown: see test_pre_auth_work_gated. Clear the
    // callback first so a late dispatch is a no-op, then reset the
    // client so its receiver/sender threads finish joining before
    // `fire_count` enters its scoped destruction.
    client->set_work_callback(nullptr);
    client.reset();
    return true;
}

// ---------------------------------------------------------------------------
// 18. v1.4.1 B.1 anti-replay: sequence numbers on DP_BATCH_V2.
//
// The v2 wire grew a 4-byte LE sequence field per DP (DistinguishedPointV2
// is 78 bytes = work_id(8) + sequence(4) + x(32) + d(32) + type(1) + dp_bits(1)).
// The server tracks an expected window and rejects out-of-window sequences
// (replays of captured DP_BATCHes, late duplicates from a misbehaving
// client).
//
// What this test pins (CLIENT-side anti-replay properties):
//
//   a. Per-WorkAssignment monotonic sequence: every DP the client emits
//      under a single work_id carries a sequence strictly greater than
//      every previously-emitted DP for that work_id. Pre-fix, a sender
//      restart could reset the counter to 0 mid-chunk, producing
//      replay-looking sequences a malicious server could exploit.
//
//   b. Sequence reset on WORK_ASN: when the server assigns a new work_id,
//      the client's sequence counter restarts at 0 for the new chunk.
//      Mixing sequences across work_ids would let a captured DP_BATCH be
//      replayed against a different chunk.
//
//   c. No sequence is reused: across the entire DP stream we observe, no
//      two DPs with the same (work_id, sequence) tuple appear on the wire.
//
// The mock-server-rejection scenario in the task plan (server REJECTS a
// stale sequence) actually lives in the SERVER, not the client. The
// JLP wire is unidirectional for DP submission: client -> server. The
// equivalent client-side property is "the client never emits a stale
// sequence in the first place" -- which is what we test here.
// ---------------------------------------------------------------------------
bool test_dp_sequence_anti_replay() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("dp_sequence_anti_replay", "connect failed");

    if (!drive_auth_ok(server, *client, "dp_sequence_anti_replay")) return false;

    // Assign the first work_id BEFORE submitting any DPs. Without a work
    // assignment, the client's current_work_.work_id stays 0, and the
    // sequence-reset-on-WORK_ASN property cannot be observed.
    JLPServerConfig cfg{};
    for (int i = 0; i < 33; ++i) cfg.public_key[i]  = 0x40 + i;
    for (int i = 0; i < 32; ++i) cfg.range_start[i] = 0x50 + i;
    for (int i = 0; i < 32; ++i) cfg.range_end[i]   = 0x60 + i;
    cfg.dp_bits        = 24;
    cfg.work_id        = 0xAAAA'BBBB'CCCC'0001ULL;
    cfg.kangaroo_type  = 1;   // TAME_ONLY (v1.5 requires non-zero)
    cfg.start_offset_a = 0;
    cfg.start_offset_b = 0;
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));

    // Brief sleep so the client's receiver thread dispatches the WORK_ASN
    // before we submit any DPs. Without this the first DP can be tagged
    // with work_id=0 (snapshotted before the WORK_ASN landed).
    std::this_thread::sleep_for(150ms);
    // Stop the kicker and drain leftover bytes so the next wait_recv_pop
    // returns the actual DP_BATCH frame instead of unknown-type filler.
    server.stop_kicker();
    std::this_thread::sleep_for(30ms);
    server.drain_rx();

    // Helper: drain one DP_BATCH_V2 frame from the wire and return the
    // 4-byte LE sequence of EACH DP in the batch (in submission order).
    // The frame is [magic:4][type:1][flags:1][len:u16 LE][count:u32 LE]
    // [dp(78) x count]. Sequence sits at offset 8 within each 78-byte DP
    // entry (after work_id at offset 0).
    auto pop_batch_sequences = [&](std::chrono::milliseconds timeout)
        -> std::vector<uint32_t> {
        std::vector<uint32_t> out;
        auto header = server.wait_recv_pop(8, timeout);
        if (header.size() != 8) return out;
        if (header[4] != TYPE_DP_BATCH_V2) {
            std::fprintf(stderr,
                "[debug] dp_sequence_anti_replay: expected DP_BATCH_V2 (0x24), got 0x%02x\n",
                header[4]);
            return out;
        }
        uint16_t plen = static_cast<uint16_t>(header[6] | (header[7] << 8));
        if (plen < 4) return out;
        auto payload = server.wait_recv_pop(plen, 1000ms);
        if (payload.size() != plen) return out;
        uint32_t count =
            static_cast<uint32_t>(payload[0]) |
            (static_cast<uint32_t>(payload[1]) << 8) |
            (static_cast<uint32_t>(payload[2]) << 16) |
            (static_cast<uint32_t>(payload[3]) << 24);
        // Sanity: payload size must equal 4 + count * 78.
        if (payload.size() != 4ULL + static_cast<size_t>(count) * 78) {
            std::fprintf(stderr,
                "[debug] dp_sequence_anti_replay: payload size %zu != 4 + count(%u)*78\n",
                payload.size(), count);
            return out;
        }
        for (uint32_t i = 0; i < count; ++i) {
            // Per-DP layout: work_id(8) | sequence(4) | x(32) | d(32) | type(1) | dp_bits(1).
            const uint8_t* dp = payload.data() + 4 + i * 78;
            uint32_t seq =
                static_cast<uint32_t>(dp[8]) |
                (static_cast<uint32_t>(dp[9]) << 8) |
                (static_cast<uint32_t>(dp[10]) << 16) |
                (static_cast<uint32_t>(dp[11]) << 24);
            out.push_back(seq);
        }
        return out;
    };

    auto make_dp = [](uint8_t marker) {
        DistinguishedPoint dp{};
        for (int i = 0; i < 32; ++i) {
            dp.x[i] = static_cast<uint8_t>(marker ^ i);
            dp.d[i] = static_cast<uint8_t>(0x80 + (marker ^ i));
        }
        dp.type = 0;
        dp.dp_bits = 24;
        return dp;
    };

    // ----- Phase 1: submit 3 batches under work_id #1 -----
    // Each batch is 3 DPs. The sender thread drains up to 100 DPs per
    // iteration, but the client allocates sequences in submission order
    // regardless. Across the three batches we expect 9 monotonically-
    // increasing sequences starting at 0: [0,1,2,3,4,5,6,7,8].
    std::vector<DistinguishedPoint> batch_a;
    for (int i = 0; i < 3; ++i) batch_a.push_back(make_dp(0x10 + i));
    if (!client->submit_dps(batch_a))
        return fail("dp_sequence_anti_replay", "submit_dps batch_a failed");

    std::vector<uint32_t> work1_seqs;
    // First flush may bring just batch_a (3 DPs) or batch_a + later submissions
    // if the sender hasn't woken yet; poll up to 9 total sequences (across
    // potentially multiple DP_BATCH_V2 frames) under work_id #1.
    auto collect_until = [&](size_t want, std::vector<uint32_t>& acc,
                             std::chrono::milliseconds total_timeout) {
        auto deadline = std::chrono::steady_clock::now() + total_timeout;
        while (acc.size() < want &&
               std::chrono::steady_clock::now() < deadline) {
            auto s = pop_batch_sequences(200ms);
            for (uint32_t v : s) acc.push_back(v);
            if (acc.size() < want && s.empty()) {
                std::this_thread::sleep_for(20ms);
            }
        }
    };

    collect_until(3, work1_seqs, 5000ms);
    if (work1_seqs.size() < 3) {
        std::fprintf(stderr,
            "[debug] dp_sequence_anti_replay: got only %zu seqs from batch_a\n",
            work1_seqs.size());
        return fail("dp_sequence_anti_replay",
                    "batch_a sequences never reached the wire");
    }

    // Submit two more batches under the SAME work_id. Each should
    // continue the sequence monotonically.
    std::vector<DistinguishedPoint> batch_b;
    for (int i = 0; i < 3; ++i) batch_b.push_back(make_dp(0x20 + i));
    if (!client->submit_dps(batch_b))
        return fail("dp_sequence_anti_replay", "submit_dps batch_b failed");

    std::vector<DistinguishedPoint> batch_c;
    for (int i = 0; i < 3; ++i) batch_c.push_back(make_dp(0x30 + i));
    if (!client->submit_dps(batch_c))
        return fail("dp_sequence_anti_replay", "submit_dps batch_c failed");

    collect_until(9, work1_seqs, 5000ms);
    if (work1_seqs.size() < 9) {
        std::fprintf(stderr,
            "[debug] dp_sequence_anti_replay: got %zu/9 work1 seqs\n",
            work1_seqs.size());
        return fail("dp_sequence_anti_replay",
                    "fewer than 9 sequences seen under work_id #1");
    }

    // Property (a): per-work_id strict monotonicity.
    for (size_t i = 1; i < work1_seqs.size(); ++i) {
        if (work1_seqs[i] <= work1_seqs[i-1]) {
            std::fprintf(stderr,
                "[debug] dp_sequence_anti_replay: non-monotonic work1: seq[%zu]=%u <= seq[%zu]=%u\n",
                i, work1_seqs[i], i-1, work1_seqs[i-1]);
            return fail("dp_sequence_anti_replay",
                        "work_id #1 sequence is not strictly increasing (replay window)");
        }
    }

    // Property (c) for the work1 stream: every seq is unique.
    {
        std::vector<uint32_t> sorted = work1_seqs;
        std::sort(sorted.begin(), sorted.end());
        for (size_t i = 1; i < sorted.size(); ++i) {
            if (sorted[i] == sorted[i-1]) {
                std::fprintf(stderr,
                    "[debug] dp_sequence_anti_replay: duplicate seq %u in work1 stream\n",
                    sorted[i]);
                return fail("dp_sequence_anti_replay",
                            "work_id #1 emitted a duplicate sequence number");
            }
        }
    }

    // The first sequence under a brand-new work_id should be 0 (or close
    // to it; some implementations seed > 0 from the manager-supplied
    // dp_sequence_next_). We only assert "starts low" here -- a brand-new
    // worker should never start mid-window for the first chunk it sees.
    if (work1_seqs.front() > 4u) {
        std::fprintf(stderr,
            "[debug] dp_sequence_anti_replay: first work1 seq = %u (expected near 0)\n",
            work1_seqs.front());
        return fail("dp_sequence_anti_replay",
                    "first sequence under a fresh work_id is not near zero");
    }

    // ----- Phase 2: switch work_id, observe sequence reset -----
    cfg.work_id = 0xAAAA'BBBB'CCCC'0002ULL;
    server.send_frame(TYPE_WORK_ASN, &cfg, sizeof(cfg));

    // Wait briefly so the WORK_ASN dispatcher latches the new work_id
    // before we drain new DPs. Without this the next submit_dp could
    // still tag DPs with the OLD work_id (stale current_work_).
    std::this_thread::sleep_for(150ms);

    std::vector<DistinguishedPoint> batch_d;
    for (int i = 0; i < 3; ++i) batch_d.push_back(make_dp(0x40 + i));
    if (!client->submit_dps(batch_d))
        return fail("dp_sequence_anti_replay", "submit_dps batch_d failed");

    std::vector<uint32_t> work2_seqs;
    collect_until(3, work2_seqs, 5000ms);
    if (work2_seqs.size() < 3) {
        return fail("dp_sequence_anti_replay",
                    "batch_d sequences never reached the wire");
    }

    // Property (b): sequence reset on WORK_ASN. The first sequence under
    // work_id #2 should be SMALLER than the last sequence emitted under
    // work_id #1. If the counter did not reset, we'd see continued
    // monotonic growth and a malicious server could replay a captured
    // work_id #1 DP under work_id #2 to forge credit.
    if (!(work2_seqs.front() < work1_seqs.back())) {
        std::fprintf(stderr,
            "[debug] dp_sequence_anti_replay: work2 first seq %u >= work1 last seq %u (no reset)\n",
            work2_seqs.front(), work1_seqs.back());
        return fail("dp_sequence_anti_replay",
                    "sequence did NOT reset on WORK_ASN (anti-replay broken)");
    }

    // Property (a) again for work2: strict monotonicity within the new chunk.
    for (size_t i = 1; i < work2_seqs.size(); ++i) {
        if (work2_seqs[i] <= work2_seqs[i-1]) {
            return fail("dp_sequence_anti_replay",
                        "work_id #2 sequence is not strictly increasing");
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

// TP-10 regression: stream-resync helper survives maximum
// fragmentation. Sends a real WORK_ASN frame byte-by-byte with a 500us
// pause between bytes so every recv() call hits the 50ms SO_RCVTIMEO
// mid-multi-byte-field. If the resync code (per-client header_partial_
// + payload_partial_ buffers in receive_message) regresses, the
// receiver mis-aligns the stream and the work callback either never
// fires or fires with garbage dp_bits.
bool test_stream_resync_fragmented() {
    std::atomic<int> callback_count{0};
    WorkAssignment captured{};
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("stream_resync_fragmented", "connect failed");

    client->set_work_callback([&](const WorkAssignment& w) {
        captured = w;
        ++callback_count;
    });

    if (!drive_auth_ok(server, *client, "stream_resync_fragmented")) {
        return false;
    }

    JLPServerConfig cfg{};
    for (int i = 0; i < 33; ++i) cfg.public_key[i]  = static_cast<uint8_t>(0xA0 + i);
    for (int i = 0; i < 32; ++i) cfg.range_start[i] = static_cast<uint8_t>(0x10 + i);
    for (int i = 0; i < 32; ++i) cfg.range_end[i]   = static_cast<uint8_t>(0xC0 + i);
    cfg.dp_bits        = 24;
    cfg.work_id        = 0xDEADBEEF'BABEF00DULL;
    cfg.kangaroo_type  = 1;
    cfg.start_offset_a = 0x1111'2222'3333'4444ULL;
    cfg.start_offset_b = cfg.start_offset_a + 0x10000ULL;

    if (!server.send_frame_fragmented(TYPE_WORK_ASN, &cfg, sizeof(cfg))) {
        return fail("stream_resync_fragmented", "fragmented send failed");
    }

    if (!wait_for([&] { return callback_count.load() >= 1; }, 5000ms)) {
        return fail("stream_resync_fragmented",
                    "callback never fired despite 5s wait (resync regression?)");
    }
    if (captured.dp_bits != 24u) {
        return fail("stream_resync_fragmented",
                    "captured dp_bits != 24 (stream misalignment?)");
    }
    if (captured.work_id != cfg.work_id) {
        return fail("stream_resync_fragmented",
                    "captured work_id mismatch (stream misalignment?)");
    }
    if (std::memcmp(captured.public_key, cfg.public_key, 33) != 0) {
        return fail("stream_resync_fragmented",
                    "captured public_key mismatch (stream misalignment?)");
    }
    if (captured.start_offset_a != cfg.start_offset_a ||
        captured.start_offset_b != cfg.start_offset_b) {
        return fail("stream_resync_fragmented",
                    "captured start_offsets mismatch (stream misalignment?)");
    }

    client->set_work_callback(nullptr);
    client.reset();
    return true;
}

// ---------------------------------------------------------------------------
// v1.5.4/v1.5.5: AUTH_OK update advert parsing. The server completes AUTH then
// sends an AUTH_OK whose payload is a 388-byte AuthOkPayload (latest_version,
// download_url, nonzero sha256, flags=update_available, manifest_sig). The
// client must parse it and expose it via get_update_advert(). We do NOT
// trigger a real self-update here: this unit context only inspects the parsed
// advert, so the manifest_sig is a deterministic non-verifying pattern (the
// signature gate is exercised by test_self_update_signing).
// ---------------------------------------------------------------------------
bool test_authok_advert_parsing() {
    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("authok_advert", "connect failed");

    // Build the 388-byte AuthOkPayload by hand (matches
    // jlp_wire_generated.hpp: latest[16] min[16] flags[1] reserved[3]
    // url[256] sha256[32] manifest_sig[64]).
    uint8_t payload[388];
    std::memset(payload, 0, sizeof(payload));
    const char* latest = "1.5.9";
    const char* minv   = "1.5.0";
    const char* url    = "https://collisionprotocol.com/download/collider.exe";
    std::memcpy(payload + 0, latest, std::strlen(latest));
    std::memcpy(payload + 16, minv, std::strlen(minv));
    payload[32] = 0x01;  // flags bit0 = update_available
    std::memcpy(payload + 36, url, std::strlen(url));
    // Nonzero sha256 (deterministic pattern).
    for (int i = 0; i < 32; ++i) payload[36 + 256 + i] = static_cast<uint8_t>(0xA0 + i);
    // Nonzero manifest_sig (deterministic pattern); parsing-only, not a valid
    // Ed25519 signature.
    for (int i = 0; i < 64; ++i) payload[36 + 256 + 32 + i] = static_cast<uint8_t>(0x10 + i);

    std::atomic<bool> ok{false};
    std::thread t([&] { ok = client->authenticate("worker", ""); });
    auto auth = server.wait_recv_pop(128, 5000ms);
    if (auth.size() != 128) {
        server.close_client();
        if (t.joinable()) t.join();
        return fail("authok_advert", "AUTH bytes never arrived");
    }
    if (!server.send_frame(TYPE_AUTH_OK, payload, sizeof(payload))) {
        if (t.joinable()) t.join();
        return fail("authok_advert", "failed to send AUTH_OK advert");
    }
    t.join();
    if (!ok.load()) return fail("authok_advert", "authenticate() returned false");

    // The advert is parsed inside handle_auth_ok on the receiver thread,
    // which runs before authenticate() observes AUTH_OK -- but to be safe
    // against any ordering, poll for the parsed value.
    JLPPoolClient::UpdateAdvert advert;
    bool got = wait_for([&] {
        advert = client->get_update_advert();
        return advert.present;
    }, 2000ms);
    if (!got) return fail("authok_advert", "advert not marked present");
    if (advert.latest_version != "1.5.9")
        return fail("authok_advert", "latest_version mismatch");
    if (advert.min_version != "1.5.0")
        return fail("authok_advert", "min_version mismatch");
    if (advert.download_url != url)
        return fail("authok_advert", "download_url mismatch");
    if (!advert.update_available)
        return fail("authok_advert", "update_available flag not set");
    bool sha_ok = true;
    for (int i = 0; i < 32; ++i) {
        if (advert.sha256[static_cast<size_t>(i)] != static_cast<uint8_t>(0xA0 + i)) {
            sha_ok = false;
            break;
        }
    }
    if (!sha_ok) return fail("authok_advert", "sha256 mismatch");
    bool sig_ok = true;
    for (int i = 0; i < 64; ++i) {
        if (advert.manifest_sig[static_cast<size_t>(i)] != static_cast<uint8_t>(0x10 + i)) {
            sig_ok = false;
            break;
        }
    }
    if (!sig_ok) return fail("authok_advert", "manifest_sig mismatch");

    client.reset();
    return true;
}

// ---------------------------------------------------------------------------
// v1.5.4: MAINTENANCE frame handling. After AUTH_OK the server sends a
// MAINTENANCE (0x60) frame with active=1, retry=30, message "upgrading".
// The client must parse it and fire the maintenance callback.
// ---------------------------------------------------------------------------
bool test_maintenance_frame() {
    std::atomic<bool> fired{false};
    std::atomic<bool> got_active{false};
    std::atomic<uint32_t> got_retry{0};
    std::string got_msg;
    std::mutex msg_mu;

    MockJlpServer server;
    auto client = make_connected(server.port());
    if (!client) return fail("maintenance_frame", "connect failed");

    client->set_maintenance_callback(
        [&](bool active, uint32_t retry, std::string message) {
            got_active.store(active);
            got_retry.store(retry);
            {
                std::lock_guard<std::mutex> lk(msg_mu);
                got_msg = std::move(message);
            }
            fired.store(true);
        });

    if (!drive_auth_ok(server, *client, "maintenance_frame")) return false;

    // Build the 262-byte MaintenancePayload: active[1] reserved[1]
    // retry_after_secs[4 LE] message[256].
    uint8_t payload[262];
    std::memset(payload, 0, sizeof(payload));
    payload[0] = 1;  // active
    uint32_t retry = 30;
    std::memcpy(payload + 2, &retry, 4);
    const char* msg = "upgrading";
    std::memcpy(payload + 6, msg, std::strlen(msg));

    if (!server.send_frame(TYPE_MAINTENANCE, payload, sizeof(payload))) {
        return fail("maintenance_frame", "failed to send MAINTENANCE");
    }

    if (!wait_for([&] { return fired.load(); }, 2000ms)) {
        return fail("maintenance_frame", "maintenance callback never fired");
    }
    if (!got_active.load())
        return fail("maintenance_frame", "active flag not true");
    if (got_retry.load() != 30)
        return fail("maintenance_frame", "retry_after_secs mismatch");
    {
        std::lock_guard<std::mutex> lk(msg_mu);
        if (got_msg != "upgrading")
            return fail("maintenance_frame", "message mismatch");
    }

    client->set_maintenance_callback(nullptr);
    client.reset();
    return true;
}

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
    // {"report_solution", ...} removed in v1.5: client-to-server SOLUTION
    // upload path was deleted as the v1.4.x theft surface. See the
    // explanatory block where test_report_solution() used to live.
    {"solution_callback_dedup",   test_solution_callback_dedup},
    {"work_callback_dedup",       test_work_callback_dedup},
    {"dp_sequence_anti_replay",   test_dp_sequence_anti_replay},
    {"stream_resync_fragmented",  test_stream_resync_fragmented},
    {"authok_advert_parsing",     test_authok_advert_parsing},
    {"maintenance_frame",         test_maintenance_frame},
};

}  // namespace

#ifdef _WIN32
#include <DbgHelp.h>
#pragma comment(lib, "dbghelp.lib")
LONG WINAPI test_seh_filter(EXCEPTION_POINTERS* ep) {
    CONTEXT* ctx_in = ep->ContextRecord;
    std::fprintf(stderr,
                 "\n=== UNHANDLED EXCEPTION ===\n"
                 "tid       = %lu\n"
                 "code      = 0x%08lX\n"
                 "addr      = %p\n"
                 "param[0]  = 0x%016llX (access type)\n"
                 "param[1]  = 0x%016llX (faulting va)\n"
                 "RIP       = 0x%016llX\n"
                 "RSP       = 0x%016llX\n"
                 "RBP       = 0x%016llX\n",
                 GetCurrentThreadId(),
                 ep->ExceptionRecord->ExceptionCode,
                 ep->ExceptionRecord->ExceptionAddress,
                 (unsigned long long)(ep->ExceptionRecord->NumberParameters > 0
                                      ? ep->ExceptionRecord->ExceptionInformation[0] : 0),
                 (unsigned long long)(ep->ExceptionRecord->NumberParameters > 1
                                      ? ep->ExceptionRecord->ExceptionInformation[1] : 0),
                 (unsigned long long)ctx_in->Rip,
                 (unsigned long long)ctx_in->Rsp,
                 (unsigned long long)ctx_in->Rbp);

    // Walk the stack starting from the faulting context. When RIP is zero
    // (execute-at-NULL) StackWalk64 cannot resolve the first frame, so seed
    // the walk from the return address sitting on top of RSP instead. That
    // recovers the call site that jumped to the null pointer.
    HANDLE proc = GetCurrentProcess();
    SymInitialize(proc, NULL, TRUE);

    CONTEXT ctx = *ctx_in;
    if (ctx.Rip == 0 && ctx.Rsp != 0) {
        // Best-effort: peek the qword at RSP. If it lives in committed
        // memory we treat it as a candidate return address. We swallow
        // any access fault here so the SEH filter does not recursively
        // crash on a bad RSP.
        __try {
            uint64_t candidate = *reinterpret_cast<uint64_t*>(ctx.Rsp);
            ctx.Rip = candidate;
            ctx.Rsp += 8;
            std::fprintf(stderr,
                         "(RIP was 0; seeding stack walk from return "
                         "address 0x%016llX)\n",
                         (unsigned long long)candidate);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            std::fprintf(stderr,
                         "(RIP was 0; RSP %p is not readable, cannot seed "
                         "stack walk)\n", (void*)ctx_in->Rsp);
        }
    }

    STACKFRAME64 frame{};
    frame.AddrPC.Offset    = ctx.Rip;
    frame.AddrPC.Mode      = AddrModeFlat;
    frame.AddrFrame.Offset = ctx.Rbp;
    frame.AddrFrame.Mode   = AddrModeFlat;
    frame.AddrStack.Offset = ctx.Rsp;
    frame.AddrStack.Mode   = AddrModeFlat;

    fprintf(stderr, "stack:\n");
    for (int i = 0; i < 30; ++i) {
        if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, proc, GetCurrentThread(),
                         &frame, &ctx, NULL, SymFunctionTableAccess64,
                         SymGetModuleBase64, NULL)) break;
        if (frame.AddrPC.Offset == 0) break;

        DWORD64 disp = 0;
        char buf[sizeof(SYMBOL_INFO) + 256];
        SYMBOL_INFO* sym = (SYMBOL_INFO*)buf;
        sym->SizeOfStruct = sizeof(SYMBOL_INFO);
        sym->MaxNameLen   = 256;
        const char* name = "?";
        if (SymFromAddr(proc, frame.AddrPC.Offset, &disp, sym)) name = sym->Name;

        char modname[MAX_PATH] = "?";
        HMODULE hmod = NULL;
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR)frame.AddrPC.Offset, &hmod) && hmod) {
            GetModuleFileNameA(hmod, modname, MAX_PATH);
        }

        fprintf(stderr, "  %p  %s+0x%llx  [%s]\n",
                (void*)frame.AddrPC.Offset, name, (unsigned long long)disp, modname);
    }
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

int main() {
#ifdef _WIN32
    WSAGuard guard;
    SetUnhandledExceptionFilter(test_seh_filter);
    // Vectored handler catches exceptions before per-thread handlers, even
    // for thread shutdown crashes that bypass the unhandled exception filter.
    AddVectoredExceptionHandler(1, [](EXCEPTION_POINTERS* ep) -> LONG {
        // Only handle access violations (not breakpoints / single-step).
        if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION)
            return EXCEPTION_CONTINUE_SEARCH;
        return test_seh_filter(ep);
    });
#endif
    std::printf("=== JLP pool protocol tests (mock TCP server) ===\n");
    int failures = 0;
    for (const auto& t : TESTS) {
        std::fprintf(stderr, "[ ENTER ] %s\n", t.name);
        std::fflush(stderr);
        std::printf("[ run ] %s\n", t.name);
        std::fflush(stdout);
        bool ok = t.fn();  // ALL locals of t.fn() are destructed by the time this returns
        std::fprintf(stderr, "[ DTORDONE ] %s ok=%d\n", t.name, (int)ok);
        std::fflush(stderr);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::fprintf(stderr, "[ POSTSLEEP ] %s\n", t.name);
        std::fflush(stderr);
        if (ok) {
            std::printf("[ ok  ] %s\n", t.name);
            std::fflush(stdout);
        } else {
            std::printf("[FAIL ] %s\n", t.name);
            ++failures;
        }
    }
    std::printf("\n%zu tests, %d failures\n",
                sizeof(TESTS) / sizeof(TESTS[0]), failures);
    return failures == 0 ? 0 : 1;
}
