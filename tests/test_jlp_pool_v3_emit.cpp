// test_jlp_pool_v3_emit.cpp
//
// v1.5.5 checkpoint-replay anti-cheat (task #9): end-to-end test of the LIVE
// DP-sender V3 emit path. Drives a REAL JLPPoolClient (real sender + receiver
// threads) against a frame-aware mock pool server and asserts:
//
//   1. v3_emit_when_capture_available: a client with a loaded worker identity
//      (= negotiated protocol v4) AND set_checkpoint_capture_available(true),
//      given a DP carrying a committable checkpoint chain, emits a DP_BATCH_V3
//      (0x26) frame -- NOT a DP_BATCH_V2 (0x24). The mock then issues a
//      CHALLENGE for that work_id; the client answers with a CHALLENGE_RSP
//      (0x33) built from the retained walk. This is the exact server->client
//      shadow-mode flow the live pool runs.
//
//   2. v2_fallback_without_capture: the SAME committable DP, but with
//      set_checkpoint_capture_available(false), ships as a plain DP_BATCH_V2
//      (0x24) and no V3 frame is ever sent. This pins the "V2 is the default,
//      no fabricated commitment" invariant.
//
// The mock does not verify signatures or replay the walk (that is the Python
// server's job, exercised by test_checkpoint_session + the server suite); it
// only confirms the CLIENT puts the right frames on the wire and answers a
// challenge. Self-contained mock (no linkage to the reconnect test's harness)
// matching the convention in test_jlp_pool_reconnect.cpp.

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
#include "pool/jlp_wire_generated.hpp"       // jlp_wire::PROTOCOL_VERSION
#include "core/checkpoint_commit.hpp"        // build_root for the expected root
#include "core/worker_identity.hpp"          // load_from_wif (v4 identity)
#include "core/wif.hpp"                       // wif::encode for a test key

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
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

constexpr uint8_t TYPE_AUTH         = 0x01;
constexpr uint8_t TYPE_AUTH_OK      = 0x02;
constexpr uint8_t TYPE_WORK_REQ     = 0x10;
constexpr uint8_t TYPE_WORK_ASN     = 0x11;
constexpr uint8_t TYPE_DP_BATCH_V2  = 0x24;
constexpr uint8_t TYPE_DP_BATCH_V3  = 0x26;
constexpr uint8_t TYPE_CHALLENGE    = 0x32;
constexpr uint8_t TYPE_CHALLENGE_RSP= 0x33;

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

std::vector<uint8_t> build_work_asn(uint64_t work_id) {
    JLPServerConfig cfg{};
    cfg.public_key[0] = 0x02;
    for (int i = 1; i < 33; ++i) cfg.public_key[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 32; ++i) {
        cfg.range_start[i] = 0x00;
        cfg.range_end[i]   = (i == 31) ? 0xFF : 0x00;
    }
    cfg.dp_bits        = 24;
    cfg.work_id        = work_id;
    cfg.kangaroo_type  = 1;   // TAME_ONLY -> permits tame (type 0) DPs
    cfg.start_offset_a = 0;
    cfg.start_offset_b = 1;
    static_assert(sizeof(JLPServerConfig) == 126,
                  "WORK_ASN payload must be 126 bytes on the wire");
    std::vector<uint8_t> out(sizeof(JLPServerConfig));
    std::memcpy(out.data(), &cfg, sizeof(JLPServerConfig));
    return out;
}

// Build a CHALLENGE payload <Q work_id><8s nonce><H count><I idx>*count, the
// exact layout CheckpointSession::decode_challenge consumes (matches the
// Python encode_challenge). Challenge a single segment index 0.
std::vector<uint8_t> build_challenge(uint64_t work_id, uint32_t seg_idx) {
    std::vector<uint8_t> p;
    for (int i = 0; i < 8; ++i)
        p.push_back(static_cast<uint8_t>((work_id >> (8 * i)) & 0xFF));
    for (int i = 0; i < 8; ++i) p.push_back(static_cast<uint8_t>(0xA0 + i));  // nonce
    const uint16_t count = 1;
    p.push_back(static_cast<uint8_t>(count & 0xFF));
    p.push_back(static_cast<uint8_t>(count >> 8));
    for (int i = 0; i < 4; ++i)
        p.push_back(static_cast<uint8_t>((seg_idx >> (8 * i)) & 0xFF));
    return p;
}

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
        if (::bind(listen_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
            std::abort();
        sockaddr_in bound{};
        socklen_t blen = sizeof(bound);
        ::getsockname(listen_, reinterpret_cast<sockaddr*>(&bound), &blen);
        port_ = ntohs(bound.sin_port);
        ::listen(listen_, 8);
        thread_ = std::thread(&MockPoolServer::run, this);
    }
    ~MockPoolServer() { stop(); }

    uint16_t port() const { return port_; }
    void set_work_id(uint64_t id) { work_id_.store(id); }
    // When true, on the first DP_BATCH_V3 the mock issues a CHALLENGE.
    void set_challenge_on_v3(bool v) { challenge_on_v3_.store(v); }

    int v2_frames() const { return v2_frames_.load(); }
    int v3_frames() const { return v3_frames_.load(); }
    int challenge_rsp_frames() const { return challenge_rsp_frames_.load(); }
    uint64_t last_v3_work_id() const { return last_v3_work_id_.load(); }

    void stop() {
        if (stopped_.exchange(true)) return;
        if (listen_ != INVALID_SOCK_T) { CLOSE_SOCK(listen_); listen_ = INVALID_SOCK_T; }
        {
            std::lock_guard<std::mutex> lk(client_mu_);
            if (current_client_ != INVALID_SOCK_T) {
                CLOSE_SOCK(current_client_);
                current_client_ = INVALID_SOCK_T;
            }
        }
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
            sock_t c = ::accept(listen_, nullptr, nullptr);
            if (c == INVALID_SOCK_T) break;
#ifdef _WIN32
            DWORD rt = 50;
            setsockopt(c, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char*>(&rt), sizeof(rt));
#else
            struct timeval tv; tv.tv_sec = 0; tv.tv_usec = 50000;
            setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif
            {
                std::lock_guard<std::mutex> lk(client_mu_);
                current_client_ = c;
            }
            handle_connection(c);
            {
                std::lock_guard<std::mutex> lk(client_mu_);
                if (current_client_ == c) {
                    CLOSE_SOCK(c);
                    current_client_ = INVALID_SOCK_T;
                }
            }
        }
    }

    void handle_connection(sock_t c) {
        std::vector<uint8_t> rx;
        std::vector<char> tmp(4096);
        const uint64_t wid = work_id_.load();
        bool challenged = false;
        while (!stopped_) {
            int n = ::recv(c, tmp.data(), static_cast<int>(tmp.size()), 0);
            if (n == 0) break;
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

            size_t off = 0;
            while (rx.size() - off >= 8) {
                const uint8_t type = rx[off + 4];
                const uint16_t plen =
                    static_cast<uint16_t>(rx[off + 6]) |
                    (static_cast<uint16_t>(rx[off + 7]) << 8);
                if (rx.size() - off < static_cast<size_t>(8) + plen) break;
                const uint8_t* body = rx.data() + off + 8;

                if (type == TYPE_AUTH) {
                    if (!send_all(c, build_frame(TYPE_AUTH_OK, nullptr, 0))) return;
                    auto asn = build_work_asn(wid);
                    if (!send_all(c, build_frame(TYPE_WORK_ASN, asn.data(),
                                                 static_cast<uint16_t>(asn.size()))))
                        return;
                } else if (type == TYPE_WORK_REQ) {
                    auto asn = build_work_asn(wid);
                    if (!send_all(c, build_frame(TYPE_WORK_ASN, asn.data(),
                                                 static_cast<uint16_t>(asn.size()))))
                        return;
                } else if (type == TYPE_DP_BATCH_V2) {
                    v2_frames_.fetch_add(1);
                } else if (type == TYPE_DP_BATCH_V3) {
                    v3_frames_.fetch_add(1);
                    // V3 payload: [count u32 LE][dp 114]; work_id is the first
                    // 8 bytes of the first DP record.
                    if (plen >= 4 + 8) {
                        uint64_t got = 0;
                        for (int i = 0; i < 8; ++i)
                            got |= static_cast<uint64_t>(body[4 + i]) << (8 * i);
                        last_v3_work_id_.store(got);
                    }
                    if (challenge_on_v3_.load() && !challenged) {
                        challenged = true;
                        auto ch = build_challenge(wid, 0);
                        if (!send_all(c, build_frame(TYPE_CHALLENGE, ch.data(),
                                                     static_cast<uint16_t>(ch.size()))))
                            return;
                    }
                } else if (type == TYPE_CHALLENGE_RSP) {
                    challenge_rsp_frames_.fetch_add(1);
                }
                off += static_cast<size_t>(8) + plen;
            }
            if (off > 0) rx.erase(rx.begin(), rx.begin() + off);
        }
    }

    sock_t   listen_         = INVALID_SOCK_T;
    sock_t   current_client_ = INVALID_SOCK_T;
    uint16_t port_           = 0;
    std::atomic<uint64_t> work_id_{777};
    std::atomic<bool>     challenge_on_v3_{false};
    std::atomic<int>      v2_frames_{0};
    std::atomic<int>      v3_frames_{0};
    std::atomic<int>      challenge_rsp_frames_{0};
    std::atomic<uint64_t> last_v3_work_id_{0};
    std::thread       thread_;
    std::atomic<bool> stopped_{false};
    std::mutex        client_mu_;
};

int g_pass = 0;
int g_fail = 0;

bool fail(const char* tname, const char* msg) {
    std::fprintf(stderr, "[FAIL] %s: %s\n", tname, msg);
    ++g_fail;
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

std::array<uint8_t, 32> priv_from_hex_tail(uint8_t tail) {
    std::array<uint8_t, 32> p{};
    for (auto& b : p) b = 0x42;
    p[31] = tail;
    return p;
}

// Build a DistinguishedPoint carrying a committable 3-checkpoint chain (tame
// type, matching the WORK_ASN's TAME_ONLY assignment). n_segments == 2.
DistinguishedPoint make_committable_dp() {
    DistinguishedPoint dp{};
    for (int i = 0; i < 32; ++i) { dp.x[i] = static_cast<uint8_t>(i); dp.d[i] = static_cast<uint8_t>(0x80 + i); }
    dp.type = 0;       // tame (TAME_ONLY work)
    dp.dp_bits = 24;
    dp.ckpt_distances.resize(3);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < 32; ++i)
            dp.ckpt_distances[c][i] = static_cast<uint8_t>((c + 1) * 0x10 + i);
    dp.ckpt_l1s2 = {0, 0, 0};
    return dp;
}

// Load a real v4 worker identity from a deterministic test key so the client's
// v3_commit_emit_active() gate (which requires worker_identity_) is satisfied.
std::shared_ptr<collider::identity::WorkerIdentity> make_test_identity(
        std::string& bech32_out) {
    auto priv = priv_from_hex_tail(0x07);
    std::string wif = collider::wif::encode(priv, /*compressed=*/true);
    auto id = collider::identity::load_from_wif(wif, "bc");
    if (!id) return nullptr;
    bech32_out = id->bech32_address();
    return std::make_shared<collider::identity::WorkerIdentity>(std::move(*id));
}

bool run_v3_emit_and_challenge() {
    const char* T = "v3_emit_when_capture_available";
    MockPoolServer mock;
    mock.set_work_id(777);
    mock.set_challenge_on_v3(true);

    std::string bech32;
    auto identity = make_test_identity(bech32);
    if (!identity) return fail(T, "load_from_wif returned null (OpenSSL?)");

    JLPPoolClient client;
    client.set_use_tls(false);
    client.set_reconnect(false);
    client.set_worker_identity(identity);
    client.set_checkpoint_capture_available(true);  // capture built + v4

    if (!client.connect("127.0.0.1", mock.port()))
        return fail(T, "connect failed");
    if (!client.authenticate(bech32, ""))
        return fail(T, "authenticate failed");

    // Wait for the WORK_ASN to land so the sender stamps the right work_id.
    WorkAssignment work{};
    if (!wait_for([&] { return client.request_work(work) && work.work_id == 777; },
                  3000ms))
        return fail(T, "WORK_ASN(work_id=777) not received");

    // Submit a committable DP -> sender must emit DP_BATCH_V3.
    client.submit_dp(make_committable_dp());

    if (!wait_for([&] { return mock.v3_frames() >= 1; }, 3000ms))
        return fail(T, "no DP_BATCH_V3 frame emitted");
    if (mock.v2_frames() != 0)
        return fail(T, "a DP_BATCH_V2 frame leaked when V3 should be used");
    if (mock.last_v3_work_id() != 777)
        return fail(T, "V3 DP carried the wrong work_id");

    // The mock issued a CHALLENGE on the V3; the client must answer it.
    if (!wait_for([&] { return mock.challenge_rsp_frames() >= 1; }, 3000ms))
        return fail(T, "CHALLENGE not answered with CHALLENGE_RSP");

    client.disconnect();
    std::printf("[PASS] %s (v3=%d, v2=%d, challenge_rsp=%d)\n",
                T, mock.v3_frames(), mock.v2_frames(), mock.challenge_rsp_frames());
    ++g_pass;
    return true;
}

bool run_v2_fallback_without_capture() {
    const char* T = "v2_fallback_without_capture";
    MockPoolServer mock;
    mock.set_work_id(888);
    mock.set_challenge_on_v3(false);

    std::string bech32;
    auto identity = make_test_identity(bech32);
    if (!identity) return fail(T, "load_from_wif returned null (OpenSSL?)");

    JLPPoolClient client;
    client.set_use_tls(false);
    client.set_reconnect(false);
    client.set_worker_identity(identity);
    // capture NOT available -> V2 fallback even though the DP is committable.
    client.set_checkpoint_capture_available(false);

    if (!client.connect("127.0.0.1", mock.port()))
        return fail(T, "connect failed");
    if (!client.authenticate(bech32, ""))
        return fail(T, "authenticate failed");

    WorkAssignment work{};
    if (!wait_for([&] { return client.request_work(work) && work.work_id == 888; },
                  3000ms))
        return fail(T, "WORK_ASN(work_id=888) not received");

    client.submit_dp(make_committable_dp());

    if (!wait_for([&] { return mock.v2_frames() >= 1; }, 3000ms))
        return fail(T, "no DP_BATCH_V2 frame emitted");
    if (mock.v3_frames() != 0)
        return fail(T, "a DP_BATCH_V3 frame was emitted with capture unavailable");

    client.disconnect();
    std::printf("[PASS] %s (v2=%d, v3=%d)\n", T, mock.v2_frames(), mock.v3_frames());
    ++g_pass;
    return true;
}

}  // namespace

int main() {
#ifdef _WIN32
    WSAGuard wsa;
#endif
    std::printf("=== test_jlp_pool_v3_emit (task #9 live V3 emit) ===\n");
    run_v3_emit_and_challenge();
    run_v2_fallback_without_capture();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
