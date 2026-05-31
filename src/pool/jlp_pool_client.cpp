// jlp_pool_client.cpp: JeanLucPons Kangaroo protocol client implementation.
// Implements TLS auth, hostname verification, the AuthState gate, bounded
// reconnect with jittered backoff, and the concurrent read/write SSL
// mutex split.
// See docs/internals/jlp-pool-client.md for the hardening summary and refs.

#include "jlp_pool_client.hpp"
#include "jlp_wire_generated.hpp"   // PROTOCOL_VERSION constant
#include "stats_sanitize.hpp"       // sanitize_stats_rsp_floats
#include "../core/worker_identity.hpp"  // B1 wire-v4 signed AUTH
#include "../core/byte_codec.hpp"
#include "../core/secure_write.hpp" // secure_open_ofstream for DP backlog file
#include "../core/session_log.hpp"  // milestone() / update_session_state() for
                                    // pool connect/auth/work/dp events.
#include <cmath>                    // std::isfinite for STATS_RSP
#include <cstdlib>                  // (historical: was used for std::atexit
                                    // pairing of WSACleanup; the atexit hook
                                    // was reverted -- see init_sockets() comment
                                    // -- but the header stays for portability
                                    // of related cstdlib helpers used elsewhere
                                    // in the TU.)
#include <cstring>
#include <ctime>                    // gmtime_s / gmtime_r for AUTH clock-sanity
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <mutex>                    // for std::once_flag, std::call_once
#include <condition_variable>
#include <filesystem>               // persistence paths
#include <fstream>                  // atomic file write
#include <sstream>                  // timestamp + filename
#include <iomanip>                  // hex formatting

#ifdef COLLIDER_HAS_OPENSSL
#include <openssl/x509v3.h>
#endif

// Pentest CLIENT-LIE-1: platform-native CSPRNG for AUTH nonce.
#if defined(_WIN32)
#  include <bcrypt.h>
#  pragma comment(lib, "bcrypt.lib")
#elif defined(__linux__)
#  include <sys/random.h>
#elif defined(__APPLE__)
#  include <stdlib.h>      // arc4random_buf
#endif

// Windows TLS verification requires bridging
// the OS root cert store into OpenSSL. vcpkg's OpenSSL on Windows has no
// usable default verify path, so SSL_CTX_set_default_verify_paths() returns
// success but loads zero anchors -- every public cert (incl. Let's Encrypt)
// then fails with X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY. We enumerate
// the "ROOT" system store via wincrypt and add each cert to OpenSSL's
// X509_STORE. Linked via crypt32.lib (set in CMakeLists.txt). This mirrors
// the bridge already used by tests/test_jlp_pool_handshake.cpp; if the two
// drift, the handshake test fails first.
#if defined(_WIN32) && defined(COLLIDER_HAS_OPENSSL)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    // wincrypt.h must come AFTER windows.h; pulls in CertOpenSystemStoreA /
    // CertEnumCertificatesInStore / CertCloseStore.
    #include <wincrypt.h>
    #include <openssl/err.h>
#endif

namespace collider {
namespace pool {

namespace {

// v1.5: the hex32() helper that formatted a 32-byte private key for the
// recovered_keys/*.json file is gone together with the file itself.

// Atomic file write: write into a sibling tempfile, fsync (POSIX best
// effort), then rename. On Windows std::filesystem::rename overwrites
// implicitly; on POSIX it is atomic against concurrent readers.
//
// Permissions: the tempfile is created via secure_open_ofstream with
// SecureWriteOnFailure::FailHard. Owner-only mode 0600 on POSIX / owner-
// only DACL on Windows. The rename preserves the mode of the source, so
// the destination ends up owner-only as well. This is the path used for
// the DP backlog file persist_dp_queue_to_disk() writes during shutdown
// (and previously for recovered_keys/*.json; that surface was deleted
// in v1.5 since the worker no longer handles private keys). FailHard
// refuses to create the tmp file at all if the owner-only ACL cannot
// be constructed; atomic_write returns false on that path and the
// caller logs the failure.
bool atomic_write(const std::filesystem::path& path,
                  const std::string& content) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    if (ec && !fs::exists(path.parent_path())) {
        return false;
    }
    fs::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream ofs = collider::secure_open_ofstream(
            tmp, std::ios::binary | std::ios::trunc,
            collider::SecureWriteOnFailure::FailHard);
        if (!ofs.is_open()) return false;
        ofs.write(content.data(),
                  static_cast<std::streamsize>(content.size()));
        if (!ofs.good()) return false;
        ofs.flush();
    }
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(path, ec);
        fs::rename(tmp, path, ec);
        if (ec) {
            return false;
        }
    }
    return true;
}

// Process-wide DP-submission counters maintained by the session-log
// wire-in. These live at TU scope (NOT on JLPPoolClient) because the
// supervisor in PoolManager tears down and recreates the client across
// reconnects; per-client counters would zero on every reconnect and the
// "total DPs this run" snapshot in session_state.json would jitter. The
// counters are atomics so the sender thread can bump them without taking
// a lock.
//
// Note: writes to the SessionState (via update_session_state) are
// debounced internally to ~5s; bumping these atomics on every DP send is
// cheap, the disk I/O is bounded.
std::atomic<uint64_t> g_dp_total_submitted{0};
std::atomic<uint64_t> g_dp_submitted_this_work{0};
std::atomic<uint64_t> g_current_work_id{0};
std::atomic<uint32_t> g_last_dp_seq{0};

// sender_loop timing constants. KEEPALIVE_SECONDS bounds the gap
// between any outgoing message; the server's per-message read timeout
// is 30s, so without traffic the server tears down our connection
// before a slow worker (dp_bits=35) ever finds a DP. STATS_INTERVAL_SECONDS
// drives the periodic STATS_REQ that refreshes pool-wide stats for the
// "Your: X / Pool: Y (Z%)" UI. Both are scoped here so the helper
// methods invoked from sender_loop see them without re-declaration.
constexpr auto kSenderKeepaliveSeconds  = std::chrono::seconds(20);
constexpr auto kSenderStatsIntervalSeconds = std::chrono::seconds(10);
// Maximum DPs per batched send. The server caps batch count at 10000,
// but 100 DPs gives us ~6.6 KB of payload per send, fitting within
// typical MTU after framing and bounding the time we hold dp_mutex_
// during the drain copy.
constexpr size_t kSenderMaxBatchDps = 100;

// Build a SessionState seed populated only with the pool-mode fields;
// callers patch additional fields and pass to update_session_state.
::collider::log::SessionState build_pool_state_seed(
    const std::string& endpoint,
    bool connected) {
    ::collider::log::SessionState s;
    s.mode = "pool";
    s.pool_endpoint = endpoint;
    s.connected = connected;
    uint64_t wid = g_current_work_id.load(std::memory_order_acquire);
    if (wid != 0) s.current_work_id = wid;
    s.dp_count_submitted_total =
        g_dp_total_submitted.load(std::memory_order_acquire);
    s.dp_count_submitted_this_work =
        g_dp_submitted_this_work.load(std::memory_order_acquire);
    s.dp_seq_last = g_last_dp_seq.load(std::memory_order_acquire);
    s.last_dp_submit_at = std::chrono::system_clock::now();
    return s;
}

}  // anonymous namespace

bool JLPPoolClient::sockets_initialized_ = false;

#ifdef COLLIDER_HAS_OPENSSL
// Thread-safe OpenSSL initialization using std::call_once
static std::once_flag ssl_init_flag;
static bool ssl_init_success = false;

static void init_openssl_once() {
    SSL_library_init();
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();
    ssl_init_success = true;
}

bool JLPPoolClient::init_tls() {
    // Thread-safe one-time initialization
    std::call_once(ssl_init_flag, init_openssl_once);
    if (!ssl_init_success) {
        std::cerr << "[Pool] OpenSSL initialization failed" << std::endl;
        return false;
    }

    // Create SSL context
    ssl_ctx_ = SSL_CTX_new(TLS_client_method());
    if (!ssl_ctx_) {
        std::cerr << "[Pool] Failed to create SSL context" << std::endl;
        return false;
    }

    // Set minimum TLS version to 1.2
    SSL_CTX_set_min_proto_version(ssl_ctx_, TLS1_2_VERSION);

    // load the platform default CA trust store so chain
    // verification can find a trust anchor (Let's Encrypt etc.). Without this,
    // SSL_VERIFY_PEER fails on any cert whose root the program does not know.
    // See OpenSSL: SSL_CTX_set_default_verify_paths(3).
    // on Windows, OpenSSL's default verify
    // paths are useless (no usable system bundle), so we additionally bridge
    // the OS "ROOT" cert store into OpenSSL's X509_STORE via wincrypt. This
    // is the same bridge used by tests/test_jlp_pool_handshake.cpp.
    if (verify_cert_) {
        // track whether ANY trust anchor mechanism
        // succeeded. If verify_cert is on but no trust store could be
        // loaded, init_tls must FAIL HARD so operators see a clean
        // startup diagnostic. Pre-1.4.1 we logged a warning and
        // continued -- subsequent cert verification would then fail
        // with a cryptic OpenSSL error during the first connection
        // attempt, which made the actual root cause (no CA bundle
        // available on this host) needlessly hard to diagnose.
        bool trust_loaded = false;

#ifdef _WIN32
        {
            HCERTSTORE hStore = CertOpenSystemStoreA(0, "ROOT");
            if (hStore) {
                X509_STORE* store = SSL_CTX_get_cert_store(ssl_ctx_);
                int added = 0;
                PCCERT_CONTEXT pCtx = nullptr;
                while ((pCtx = CertEnumCertificatesInStore(hStore, pCtx)) != nullptr) {
                    const unsigned char* p = pCtx->pbCertEncoded;
                    X509* x509 = d2i_X509(nullptr, &p, (long)pCtx->cbCertEncoded);
                    if (x509) {
                        if (X509_STORE_add_cert(store, x509) == 1) {
                            ++added;
                        } else {
                            // Duplicate or other non-fatal error; clear so a
                            // later real OpenSSL call does not see this stale
                            // entry in the error queue.
                            ERR_clear_error();
                        }
                        X509_free(x509);
                    }
                }
                CertCloseStore(hStore, 0);
                if (added == 0) {
                    std::cerr << "[Pool] Warning: Windows ROOT store enumerated "
                                 "0 certs; falling back to OpenSSL default verify paths."
                              << std::endl;
                    trust_loaded = (SSL_CTX_set_default_verify_paths(ssl_ctx_) == 1);
                } else {
                    trust_loaded = true;
                }
            } else {
                std::cerr << "[Pool] Warning: CertOpenSystemStoreA(ROOT) failed (err="
                          << (unsigned long)GetLastError()
                          << "); falling back to OpenSSL default verify paths "
                             "(likely empty on Windows)." << std::endl;
                trust_loaded = (SSL_CTX_set_default_verify_paths(ssl_ctx_) == 1);
            }
        }
#else
        trust_loaded = (SSL_CTX_set_default_verify_paths(ssl_ctx_) == 1);
#endif

        if (!trust_loaded) {
            std::cerr << "[Pool] FATAL: TLS verification enabled but no trust "
                         "anchors could be loaded. Install a CA bundle (set "
                         "SSL_CERT_FILE / SSL_CERT_DIR, or install ca-certificates "
                         "on Linux), or pass verify_cert=false to skip "
                         "verification (NOT recommended for production)."
                      << std::endl;
            SSL_CTX_free(ssl_ctx_);
            ssl_ctx_ = nullptr;
            return false;
        }
        SSL_CTX_set_verify(ssl_ctx_, SSL_VERIFY_PEER, nullptr);
    } else {
        // Explicit opt-out (e.g., self-signed test cert). Loud warning.
        std::cerr << "[Pool] WARNING: TLS certificate verification DISABLED "
                     "(verify_cert=false). MITM attack possible." << std::endl;
        SSL_CTX_set_verify(ssl_ctx_, SSL_VERIFY_NONE, nullptr);
    }

    // Create SSL connection
    ssl_ = SSL_new(ssl_ctx_);
    if (!ssl_) {
        std::cerr << "[Pool] Failed to create SSL connection" << std::endl;
        SSL_CTX_free(ssl_ctx_);
        ssl_ctx_ = nullptr;
        return false;
    }

    // SNI + hostname verification. These are SSL-level (not CTX)
    // because the hostname is per-connection.
    //   * SSL_set_tlsext_host_name      - SNI: tells server which vhost we want.
    //                                     Required for any modern multi-tenant
    //                                     TLS terminator (nginx, ALB, etc.).
    //                                     RFC 6066 sec 3.
    //   * X509_VERIFY_PARAM_set1_host   - tells OpenSSL to actually check that
    //                                     the cert's SAN/CN matches the host we
    //                                     dialed. Without this, OpenSSL only
    //                                     verifies the chain, NOT the identity:
    //                                     a Let's Encrypt cert for ANY host
    //                                     would be accepted. RFC 6125 sec 6.
    if (verify_cert_) {
        if (SSL_set_tlsext_host_name(ssl_, host_.c_str()) != 1) {
            std::cerr << "[Pool] SSL_set_tlsext_host_name (SNI) failed for host: "
                      << host_ << std::endl;
            SSL_free(ssl_);
            SSL_CTX_free(ssl_ctx_);
            ssl_ = nullptr;
            ssl_ctx_ = nullptr;
            return false;
        }

        X509_VERIFY_PARAM* param = SSL_get0_param(ssl_);
        if (!param) {
            std::cerr << "[Pool] SSL_get0_param returned null" << std::endl;
            SSL_free(ssl_);
            SSL_CTX_free(ssl_ctx_);
            ssl_ = nullptr;
            ssl_ctx_ = nullptr;
            return false;
        }
        // Reject partial wildcards (e.g., "f*.example.com"). Standard hardening.
        X509_VERIFY_PARAM_set_hostflags(param, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
        if (X509_VERIFY_PARAM_set1_host(param, host_.c_str(), 0) != 1) {
            std::cerr << "[Pool] X509_VERIFY_PARAM_set1_host failed for host: "
                      << host_ << std::endl;
            SSL_free(ssl_);
            SSL_CTX_free(ssl_ctx_);
            ssl_ = nullptr;
            ssl_ctx_ = nullptr;
            return false;
        }
        // Belt-and-suspenders: also enforce at SSL level (CTX-level was set above).
        SSL_set_verify(ssl_, SSL_VERIFY_PEER, nullptr);
    } else {
        // Even when verification is off, set SNI - many pools require it just
        // to route the handshake to the right vhost.
        SSL_set_tlsext_host_name(ssl_, host_.c_str());
    }

    // Attach socket to SSL
    SSL_set_fd(ssl_, static_cast<int>(socket_));

    // Perform TLS handshake
    int ret = SSL_connect(ssl_);
    if (ret != 1) {
        int err = SSL_get_error(ssl_, ret);
        std::cerr << "[Pool] TLS handshake failed: " << err << std::endl;
        // Drain the OpenSSL error queue so we report the underlying cause
        // (e.g., hostname mismatch, expired cert) and not a stale prior error.
        unsigned long e;
        while ((e = ERR_get_error()) != 0) {
            char buf[256];
            ERR_error_string_n(e, buf, sizeof(buf));
            std::cerr << "[Pool] SSL error: " << buf << std::endl;
        }
        // If the failure was specifically certificate verification, surface that.
        long verify_result = SSL_get_verify_result(ssl_);
        if (verify_result != X509_V_OK) {
            std::cerr << "[Pool] Cert verification failed: "
                      << X509_verify_cert_error_string(verify_result)
                      << " (peer host: " << host_ << ")" << std::endl;
        }
        SSL_free(ssl_);
        SSL_CTX_free(ssl_ctx_);
        ssl_ = nullptr;
        ssl_ctx_ = nullptr;
        return false;
    }

    std::cout << "[Pool] TLS connection established (" << SSL_get_version(ssl_)
              << ", host=" << host_ << ", verify="
              << (verify_cert_ ? "on" : "off") << ")" << std::endl;
    return true;
}

void JLPPoolClient::cleanup_tls() {
    if (ssl_) {
        SSL_shutdown(ssl_);
        SSL_free(ssl_);
        ssl_ = nullptr;
    }
    if (ssl_ctx_) {
        SSL_CTX_free(ssl_ctx_);
        ssl_ctx_ = nullptr;
    }
}

int JLPPoolClient::ssl_send(const void* data, size_t size) {
    if (!ssl_) return -1;
    return SSL_write(ssl_, data, static_cast<int>(size));
}

int JLPPoolClient::ssl_recv(void* data, size_t size) {
    if (!ssl_) return -1;
    return SSL_read(ssl_, data, static_cast<int>(size));
}
#endif

bool JLPPoolClient::init_sockets() {
#ifdef _WIN32
    // WSAStartup is called at most once per process via call_once. We
    // deliberately do NOT pair it with std::atexit(WSACleanup):
    //
    // The Windows image loader unloads ws2_32.dll during process exit
    // in an order that can run BEFORE the C runtime fires registered
    // atexit lambdas. When that happens, the atexit lambda calls
    // WSACleanup against an already-unloaded DLL and a null-pointer
    // access violation is the result. The deterministic SEH segfault
    // in test_jlp_pool_protocol::stats_rsp_parsing surfaced exactly
    // this teardown race.
    //
    // The "leak" the original W1-A audit flagged is the per-process
    // WSAStartup refcount sitting at 1 instead of 0 at process exit.
    // The OS reaps the entire process address space on termination, so
    // the refcount is moot. The cure (atexit) was worse than the
    // disease (none).
    //
    // sockets_initialized_ remains a cheap fast-path flag so the
    // call_once predicate stays out of the hot path on re-init.
    static std::once_flag startup_once;
    bool startup_ok = true;
    std::call_once(startup_once, [&]() {
        WSADATA wsa_data;
        if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
            startup_ok = false;
            return;
        }
        sockets_initialized_ = true;
    });
    if (!startup_ok) {
        return false;
    }
#endif
    return true;
}

void JLPPoolClient::cleanup_sockets() {
#ifdef _WIN32
    // Idempotent. Reachable from:
    //   - Direct test-harness calls (legacy; no production caller
    //     invokes this directly, verified via grep across the tree).
    //
    // Note: there is intentionally NO std::atexit hook here. The prior
    // atexit pairing was reverted to fix the deterministic SEH
    // segfault in test_jlp_pool_protocol::stats_rsp_parsing -- the
    // Windows image loader could unload ws2_32.dll BEFORE the CRT
    // fired the atexit lambda, causing the lambda to call WSACleanup
    // against an already-unloaded DLL. See init_sockets() above for
    // the full rationale.
    if (sockets_initialized_) {
        WSACleanup();
        sockets_initialized_ = false;
    }
#endif
}

// safely re-assign a std::thread. std::thread::operator=
// invokes std::terminate if the LHS is joinable. Always join (or detach) the
// previous thread before overwriting.
// IMPORTANT: do not call this from inside the thread that `t` represents - that
// would self-join and deadlock. The receiver path that calls this for
// receiver_thread_ must therefore happen from a different thread. Currently
// only disconnect() (caller thread) re-creates these objects.
void JLPPoolClient::replace_thread(std::thread& t, std::thread new_thread) noexcept {
    // noexcept. The body catches every potential throw
    // (std::system_error from join/detach, anything from std::cerr)
    // and std::thread::operator= is itself noexcept per the standard.
    // Static-asserted below for future-proofing.
    static_assert(
        std::is_nothrow_move_assignable<std::thread>::value,
        "std::thread::operator=(std::thread&&) must be noexcept for "
        "replace_thread to honor its noexcept declaration");
    if (t.joinable()) {
        // hard-enforce the docstring's self-join warning. The
        // alternative is std::system_error("resource_deadlock_would_occur")
        // thrown from inside the join() below, which we'd then swallow into
        // a detach() that succeeds -- but the thread (this thread) keeps
        // running off a dangling thread object. Detach explicitly + log.
        if (t.get_id() == std::this_thread::get_id()) {
            try {
                std::cerr << "[Pool] replace_thread called from inside the "
                             "target thread; detaching to avoid self-join "
                             "deadlock (this is a programming error and "
                             "leaks the std::thread). Caller: "
                          << std::this_thread::get_id() << std::endl;
            } catch (...) {}
            try { t.detach(); } catch (...) {}
            t = std::move(new_thread);
            return;
        }

        // Last-ditch safety. In well-formed code the caller has already signaled
        // running_=false and the old thread has exited, so this is fast.
        try {
            t.join();
        } catch (const std::system_error& e) {
            // We can't `throw`, can't `terminate` quietly. Best
            // effort: log and detach so the destructor of the
            // outgoing thread object doesn't terminate.
            try {
                std::cerr << "[Pool] thread join failed in replace_thread: "
                          << e.what() << std::endl;
            } catch (...) {
                // std::cerr can technically throw if exceptions are
                // enabled on it; swallow to honor noexcept.
            }
            try { t.detach(); } catch (...) {}
        } catch (...) {
            // Any other exception: detach to avoid terminate.
            try { t.detach(); } catch (...) {}
        }
    }
    t = std::move(new_thread);
}

JLPPoolClient::JLPPoolClient()
    : socket_(INVALID_SOCK)
    , port_(17403)
    , timeout_ms_(30000)  // 30 second timeout (was 3 seconds)
    , auto_reconnect_(true)
    , connected_(false)
    , running_(false)
    , last_receive_was_timeout_(false)
{
    init_sockets();
    memset(&stats_, 0, sizeof(stats_));
}

JLPPoolClient::~JLPPoolClient() {
    disconnect();
}

bool JLPPoolClient::connect(const std::string& host, uint16_t port) {
    if (connected_) {
        disconnect();
    }

    host_ = host;
    port_ = port;
    auth_state_.store(AuthState::CONNECTING);

    // emit the plaintext warning BEFORE the TCP socket is
    // opened. AUTH payload carries the worker name (= payout address) and
    // password; the operator should have a chance to abort BEFORE any
    // credential lands on a plaintext socket. parse_pool_url already prints
    // the same warning at config time; printing it here too covers callers
    // that build PoolConfig programmatically and bypass parse_pool_url.
    if (!use_tls_) {
        warn_if_plaintext(host, port);
    }

    // resolve via AF_UNSPEC so IPv6-only pools (or dual-stack
    // pools whose A record times out) are reachable. Iterate addrinfo list
    // and try each candidate; succeed on the first one that connects.
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result) != 0) {
        std::cerr << "[Pool] Failed to resolve hostname: " << host << std::endl;
        auth_state_.store(AuthState::DISCONNECTED);
        return false;
    }

    socket_ = INVALID_SOCK;
    for (struct addrinfo* ai = result; ai != nullptr; ai = ai->ai_next) {
        socket_ = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (socket_ == INVALID_SOCK) continue;

        // Apply timeouts before connect so a stalled SYN doesn't hang
        // indefinitely. Same values used for both directions.
#ifdef _WIN32
        DWORD timeout = timeout_ms_;
        setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
        setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
#else
        struct timeval tv;
        tv.tv_sec = timeout_ms_ / 1000;
        tv.tv_usec = (timeout_ms_ % 1000) * 1000;
        setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif

        if (::connect(socket_, ai->ai_addr, (int)ai->ai_addrlen) != SOCK_ERROR) {
            break;  // success
        }
        closesocket(socket_);
        socket_ = INVALID_SOCK;
    }
    freeaddrinfo(result);

    if (socket_ == INVALID_SOCK) {
        std::cerr << "[Pool] Failed to connect to " << host << ":" << port
                  << " (no resolved address answered)" << std::endl;
        auth_state_.store(AuthState::DISCONNECTED);
        // Session log: a connect failure leaves the supervisor to
        // schedule a retry; without this milestone the only evidence
        // of the failure in the session log would be the symmetric
        // "disconnect" emitted from JLPPoolClient::disconnect() which
        // does not distinguish "connect refused" from "graceful
        // teardown".
        ::collider::log::milestone(
            "pool_connect_failed",
            host + ":" + std::to_string(port));
        return false;
    }

    // Initialize TLS if enabled
#ifdef COLLIDER_HAS_OPENSSL
    if (use_tls_) {
        if (!init_tls()) {
            closesocket(socket_);
            socket_ = INVALID_SOCK;
            auth_state_.store(AuthState::DISCONNECTED);
            return false;
        }
    }
#else
    if (use_tls_) {
        std::cerr << "[Pool] TLS requested but OpenSSL not available" << std::endl;
        closesocket(socket_);
        socket_ = INVALID_SOCK;
        auth_state_.store(AuthState::DISCONNECTED);
        return false;
    }
#endif

    connected_ = true;
    running_ = true;

    std::cout << "[Pool] Connected to " << host << ":" << port;
    if (use_tls_) {
        std::cout << " (TLS)";
    }
    std::cout << std::endl;

    // Session log: record the successful TCP/TLS connect. The endpoint
    // string is repeated in the SessionState so the JSON snapshot has it
    // standalone (without having to grep the log for the matching event).
    {
        std::string endpoint = host + ":" + std::to_string(port) +
                               (use_tls_ ? " (TLS)" : "");
        ::collider::log::milestone("pool_connect", endpoint);
        ::collider::log::update_session_state(
            build_pool_state_seed(endpoint, /*connected=*/true));
    }

    // plaintext warning moved to the top of connect()
    // (and parse_pool_url) so it fires before the TCP handshake. Pre-fix
    // the warning printed here, AFTER the connection succeeded and just
    // before the receiver/sender threads spun up, which made it easy to
    // miss in the operator's log and gave no chance to abort before bytes
    // hit the wire.

    // safely (re)assign the thread objects. If a previous
    // disconnect didn't fully clean up (which shouldn't happen, but defense in
    // depth), join the existing thread before overwriting it. NEVER call this
    // from inside the receiver/sender thread itself.
    replace_thread(receiver_thread_,
                   std::thread(&JLPPoolClient::receiver_loop, this));
    replace_thread(sender_thread_,
                   std::thread(&JLPPoolClient::sender_loop, this));

    return true;
}

void JLPPoolClient::disconnect() {
    // clean shutdown drain + correct TLS teardown order.
    // Pre-fix order:
    //   1. running_=false, connected_=false
    //   2. raw closesocket() (cuts the SSL session mid-flight)
    //   3. join threads
    //   4. cleanup_tls() / SSL_shutdown
    // The sender_loop drained the queue once after running_ flipped, but
    // because connected_ was ALSO flipped before the drain ran the batch
    // was discarded via `batch.clear()`: silent DP loss on every clean
    // shutdown. And tearing down the raw socket before SSL_shutdown means
    // the server never receives a close_notify, so it logs a "dirty"
    // teardown for every disconnect (and on the wire it cannot tell a
    // crashed client from one that quit gracefully).
    // Fix:
    //   1. Signal drain (drain_requested_) so the sender flushes the
    //      remaining DPs while connected_ is still true and auth_state_
    //      is still AUTH_OK.
    //   2. Wait up to DRAIN_TIMEOUT_MS for dp_queue_ to empty.
    //   3. THEN set running_=false / connected_=false.
    //   4. SSL_shutdown FIRST (sends close_notify if SSL is still up).
    //   5. THEN raw closesocket().
    //   6. THEN join threads (they unblock from the raw socket close).
    //   7. THEN free SSL_CTX/SSL.
    // Reentry from the receiver/sender thread itself still detaches
    // instead of self-joining, just like before.

    // Step 1: signal sender to flush, but keep connected_/auth_state_ as
    // they are so the in-flight batch can actually be sent.
    if (connected_.load() && auth_state_.load() == AuthState::AUTH_OK) {
        drain_requested_.store(true, std::memory_order_release);
        dp_cv_.notify_all();

        // Step 2: wait for the queue to empty, bounded.
        std::unique_lock<std::mutex> lk(dp_mutex_);
        dp_cv_.wait_for(lk, std::chrono::milliseconds(DRAIN_TIMEOUT_MS),
                        [this] { return dp_queue_.empty() || !running_; });
        if (!dp_queue_.empty()) {
            std::cerr << "[Pool] disconnect: drain timed out with "
                      << dp_queue_.size() << " DP(s) still queued"
                      << std::endl;
        }
    }

    // Step 3: tear down state. running_ first so the sender stops looping
    // on its next wait_for; connected_ next so any in-progress send sees
    // the flag.
    running_ = false;
    connected_ = false;
    auth_state_.store(AuthState::DISCONNECTED);

    // Wake the sender (drain-or-exit) and any authenticate() waiter.
    dp_cv_.notify_all();
    auth_cv_.notify_all();

    // Step 4: send close_notify BEFORE we yank the raw socket. SSL_shutdown
    // is a no-op if ssl_ is null; we guard explicitly to keep the code
    // readable. Note: doing this BEFORE joining the receiver thread is
    // safe because we hold ssl_write_mutex_ here. The receiver's pending
    // SSL_read holds only ssl_read_mutex_; per OpenSSL's threading model,
    // one concurrent reader + one concurrent writer on the same SSL is
    // supported.
#ifdef COLLIDER_HAS_OPENSSL
    if (use_tls_ && ssl_) {
        std::lock_guard<std::mutex> wlock(ssl_write_mutex_);
        // Best-effort: a half-open socket or already-shut-down peer may
        // return WANT_READ / WANT_WRITE / 0; we ignore and continue.
        SSL_shutdown(ssl_);
    }
#endif

    // Step 5: close the raw socket. This unblocks any in-progress
    // SSL_read / recv in the receiver (causing it to return an error
    // and exit) and any blocking write in the sender.
    if (socket_ != INVALID_SOCK) {
        closesocket(socket_);
        socket_ = INVALID_SOCK;
    }

    // Step 6: join both threads BEFORE touching ssl_. Once they have
    // exited, no thread is using ssl_ and it is safe to free.
    if (receiver_thread_.joinable()) {
        if (std::this_thread::get_id() != receiver_thread_.get_id()) {
            try { receiver_thread_.join(); } catch (const std::system_error& e) {
                std::cerr << "[Pool] receiver join failed: " << e.what() << std::endl;
                try { receiver_thread_.detach(); } catch (...) {}
            }
        } else {
            std::cerr << "[Pool] WARNING: disconnect() called from receiver "
                         "thread; detaching to avoid self-join" << std::endl;
            receiver_thread_.detach();
        }
    }
    if (sender_thread_.joinable()) {
        if (std::this_thread::get_id() != sender_thread_.get_id()) {
            try { sender_thread_.join(); } catch (const std::system_error& e) {
                std::cerr << "[Pool] sender join failed: " << e.what() << std::endl;
                try { sender_thread_.detach(); } catch (...) {}
            }
        } else {
            std::cerr << "[Pool] WARNING: disconnect() called from sender "
                         "thread; detaching" << std::endl;
            sender_thread_.detach();
        }
    }

    // v1.5: the "Step 6b" block that stopped and joined the
    // solution-upload retry threads was deleted. The RetryThread vector
    // and the report_solution() retry uploader that populated it are
    // both gone -- the pool server is the sole key-computer, the worker
    // never holds a recovered key, and there is nothing to retry-upload.

    // Step 7: free TLS objects (both I/O threads have exited).
#ifdef COLLIDER_HAS_OPENSSL
    if (use_tls_) {
        cleanup_tls();
    }
#endif

    // Reset the drain flag so a subsequent connect+disconnect doesn't
    // think a stale drain is in progress.
    drain_requested_.store(false, std::memory_order_release);

    // "[Pool] Disconnected" used to land on stdout here and was
    // visible to the operator twice on every pool teardown (the
    // supervisor's session disconnect plus the destructor's
    // teardown disconnect both pass through this path). With the
    // interactive menu loop the lines flashed between TUI exit
    // and menu re-entry and read like an error. The session log
    // milestone below preserves the disconnect record for post-
    // mortem; no operator-facing stdout line is needed.

    // Session log: record the disconnect AFTER stdout so the order in
    // the log matches the operator-visible sequence. The endpoint
    // string is host:port (no TLS suffix; not relevant for disconnect).
    //
    // Suppress the milestone when host_ is empty: that is the
    // destructor-from-never-connected path (JLPPoolClient created but
    // connect() never called, or connect() failed before host_ was
    // assigned). Logging "disconnect" with an empty endpoint would
    // produce a confusing line; logging twice when the supervisor
    // teardown calls disconnect after a failed connect would
    // duplicate. The host_-non-empty guard catches both.
    if (!host_.empty()) {
        ::collider::log::milestone(
            "disconnect",
            host_ + ":" + std::to_string(port_));
        auto seed = build_pool_state_seed(
            host_ + ":" + std::to_string(port_), /*connected=*/false);
        // After disconnect, force a state flush regardless of throttle
        // so the JSON snapshot reflects connected=false before any
        // subsequent reconnect attempt fires.
        ::collider::log::update_session_state(seed);
        ::collider::log::flush_session_state();
    }
}

bool JLPPoolClient::is_connected() const {
    return connected_;
}

bool JLPPoolClient::authenticate(const std::string& worker_name,
                                 const std::string& password) {
    ip_banned_  = false;
    ban_reason_.clear();
    worker_name_ = worker_name;
    // AUTH wire format (JLPClientHelloV2) carries a real
    // password slot. Pre-1.4.1 we logged a warning that --pool-password
    // was being silently ignored; that warning is now obsolete.
    // Stage into a SecureString so the bytes are zeroed when this
    // function exits. The caller's std::string copy is unaffected; the
    // supervisor (PoolManager) holds the long-lived credential and
    // calls authenticate() again on reconnect.
    password_.assign(password.data(), password.size());

    // Scope guard: wipe password_ on every return path (timeout, failure,
    // and success alike). The post-handshake state is "no longer needs
    // the secret"; only the supervisor's persistent copy keeps the
    // credential available for the next reconnect.
    struct PasswordWipeGuard {
        ::collider::SecureString* p;
        ~PasswordWipeGuard() { if (p) p->wipe(); }
    } pw_guard{&password_};

    // pool server validates the AUTH timestamp against its own
    // wall clock with a +/-30s window (AUTH_CLOCK_DRIFT_SECS). If the
    // operator's clock is wildly off (CMOS battery dead, fresh container
    // before NTP sync, etc.), every reconnect attempt will fail with a
    // confusing "AUTH_FAILED" instead of a clear cause. Print a one-line
    // warning when the local clock looks implausible so the message is
    // visible BEFORE the wire failure.
    {
        auto now = std::chrono::system_clock::now();
        auto t = std::chrono::system_clock::to_time_t(now);
        std::tm tm{};
#ifdef _WIN32
        gmtime_s(&tm, &t);
#else
        gmtime_r(&t, &tm);
#endif
        int year = tm.tm_year + 1900;
        if (year < 2024 || year > 2099) {
            std::cerr << "[Pool] WARNING: system clock year is " << year
                      << "; pool AUTH timestamps are validated within "
                      << jlp_wire::AUTH_CLOCK_DRIFT_SECS
                      << "s of the server's wall clock. Set the clock "
                         "before connecting if AUTH keeps failing.\n";
        }
    }

    // Move to AUTH_SENT before transmitting so the receiver can correctly
    // accept AUTH_OK / AUTH_FAIL when they arrive.
    auth_state_.store(AuthState::AUTH_SENT);

    if (!send_hello()) {
        auth_state_.store(AuthState::DISCONNECTED);
        return false;
    }

    // actually wait for AUTH_OK / AUTH_FAIL / MSG_ERROR or timeout.
    // The receiver thread updates auth_state_ and notifies auth_cv_.
    std::unique_lock<std::mutex> lock(auth_cv_mutex_);
    bool ok = auth_cv_.wait_for(
        lock,
        std::chrono::milliseconds(AUTH_RESPONSE_TIMEOUT_MS),
        [this] {
            AuthState s = auth_state_.load();
            return s == AuthState::AUTH_OK
                || s == AuthState::AUTH_FAILED
                || s == AuthState::DISCONNECTED;
        });

    if (!ok) {
        std::cerr << "[Pool] Authentication timed out after "
                  << AUTH_RESPONSE_TIMEOUT_MS << "ms (no AUTH_OK / AUTH_FAIL "
                     "from server)" << std::endl;
        return false;
    }

    AuthState final_state = auth_state_.load();
    if (final_state == AuthState::AUTH_OK) {
        return true;
    }
    if (final_state == AuthState::AUTH_FAILED) {
        std::cerr << "[Pool] Authentication rejected by server (AUTH_FAIL)"
                  << std::endl;
        return false;
    }
    // DISCONNECTED or anything else
    std::cerr << "[Pool] Authentication failed (state="
              << static_cast<int>(final_state) << ")" << std::endl;
    return false;
}

bool JLPPoolClient::send_hello() {
    // send the v2 AUTH wire format. Pre-1.4.1 we sent
    // JLPClientHello (76 bytes) but the Python server has always
    // decoded 96 bytes (name + password); gpu_count/speed silently
    // landed on the password slot. The v2 layout adds a real password
    // field plus a timestamp_ms and 16-byte nonce so the server can
    // refuse replays of captured AUTH packets and bound client clock
    // drift. The pre-existing gpu_count_/speed_ telemetry is dropped
    // from AUTH because it isn't read on the server side and never was.
    JLPClientHelloV2 hello;
    std::memset(&hello, 0, sizeof(hello));
    if (worker_name_.size() >= sizeof(hello.worker_name)) {
        std::cerr << "[Pool] Worker name too long: " << worker_name_.size()
                  << " bytes (max " << (sizeof(hello.worker_name) - 1)
                  << "). Refusing to send truncated identity." << std::endl;
        return false;
    }
    if (password_.size() >= sizeof(hello.password)) {
        std::cerr << "[Pool] Pool password too long: " << password_.size()
                  << " bytes (max " << (sizeof(hello.password) - 1)
                  << "). Refusing to send truncated credentials." << std::endl;
        return false;
    }
    std::memcpy(hello.worker_name, worker_name_.data(), worker_name_.size());
    if (!password_.empty()) {
        std::memcpy(hello.password, password_.data(), password_.size());
    }

    // Stamp wall-clock time in ms LE. The server tolerates +/- 30s of
    // skew; clients with a clock more than that off should be fixed.
    using namespace std::chrono;
    const auto now_ms = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()
    ).count();
    hello.timestamp_ms = static_cast<uint64_t>(now_ms);

    // Pentest CLIENT-LIE-1 fix (2026-05-23 deep audit): historical
    // std::random_device was assumed CSPRNG-grade, but several MinGW
    // builds shipped a Mersenne Twister seeded from time() as their
    // random_device implementation. A predictable nonce lets an
    // attacker pre-compute AUTH replays. Use platform-native CSPRNG
    // directly: BCryptGenRandom on Windows, getrandom(2) on Linux,
    // arc4random_buf on macOS. std::random_device stays only as the
    // last-resort fallback path if the native call fails, with a
    // loud stderr warning so the operator notices.
    {
        bool ok = false;
#if defined(_WIN32)
        // BCryptGenRandom with BCRYPT_USE_SYSTEM_PREFERRED_RNG returns
        // 0 (STATUS_SUCCESS) on success. The system-preferred RNG is
        // the Cryptography Next Generation provider; CSPRNG-grade.
        if (BCryptGenRandom(nullptr,
                            reinterpret_cast<PUCHAR>(hello.nonce),
                            static_cast<ULONG>(sizeof(hello.nonce)),
                            BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0) {
            ok = true;
        }
#elif defined(__linux__)
        ssize_t n = getrandom(hello.nonce, sizeof(hello.nonce), 0);
        if (n == static_cast<ssize_t>(sizeof(hello.nonce))) ok = true;
#elif defined(__APPLE__)
        arc4random_buf(hello.nonce, sizeof(hello.nonce));
        ok = true;
#endif
        if (!ok) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::cerr
                    << "[AUTH] WARNING: platform CSPRNG unavailable; "
                       "AUTH nonce falls back to std::random_device "
                       "(CLIENT-LIE-1 fix degraded). On MinGW builds "
                       "std::random_device may be PRNG-seeded.\n";
            }
            using WordEngine =
                std::independent_bits_engine<std::random_device,
                                             32, uint32_t>;
            WordEngine eng;
            for (size_t i = 0; i < sizeof(hello.nonce);
                 i += sizeof(uint32_t)) {
                uint32_t w = eng();
                for (size_t b = 0; b < sizeof(uint32_t); ++b) {
                    hello.nonce[i + b] =
                        static_cast<uint8_t>((w >> (8 * b)) & 0xFFu);
                }
            }
        }
    }

    // B1 wire-v4: if a worker identity is loaded (--worker-key flow),
    // upgrade the AUTH frame in-place. The v4 layout is the v3 prefix
    // (name + password + timestamp + nonce) followed by 33-byte pubkey
    // and 64-byte raw r||s signature over the canonical AUTH message
    // (see worker_identity.hpp + jlp_protocol.py
    // build_auth_canonical_message). header.flags is bumped to
    // PROTOCOL_VERSION_V4 so the server routes to its v4 decoder.
    //
    // The compressed pubkey's hash160 in bech32 form MUST equal
    // worker_name (the server checks); we don't re-derive here because
    // load_from_wif already populated bech32_address_ and the operator
    // is expected to have set --worker = identity.bech32_address().
    // Mismatch will surface as AUTH_FAIL on the server side.
    // v1.5.2 auth-flow diagnostics. Every observable transition gets
    // a milestone so the session log can localize an auth failure to
    // ONE microsecond: did the signing block? was send_message slow?
    // did the wire ack come back empty? Without these breadcrumbs,
    // the only visible events were [pool_connect] and [disconnect],
    // which forced cross-referencing the server log every time auth
    // misbehaved (user-reported 2026-05-25 5s pool death).
    auto auth_milestone = [](const char* event, const std::string& detail) {
        ::collider::log::milestone(event, detail);
    };
    // header.flags carries this client's PROTOCOL_VERSION on EVERY send (the
    // v1.4.2 B.5 contract). This is a v1.5.4 client, so it always advertises
    // version 4 regardless of whether the AUTH body is the signed v4 identity
    // form or the unsigned 120-byte form. The version (flags) and the
    // signed/unsigned body shape are orthogonal axes; downgrading the
    // unsigned path to 3 would get this client refused by a strict-mode pool
    // (floor 4). The "wire=" label reflects the actual flags byte sent.
    const uint8_t auth_wire_flags =
        static_cast<uint8_t>(jlp_wire::PROTOCOL_VERSION);
    auth_milestone("auth_build_begin",
                   std::string("wire=v") +
                       std::to_string(static_cast<int>(auth_wire_flags)));

    if (worker_identity_) {
        JLPClientHelloV4 hv4;
        std::memset(&hv4, 0, sizeof(hv4));
        std::memcpy(hv4.worker_name, hello.worker_name,
                    sizeof(hello.worker_name));
        // hv4.password_padding stays zero-filled.
        hv4.timestamp_ms = hello.timestamp_ms;
        std::memcpy(hv4.nonce, hello.nonce, sizeof(hello.nonce));
        std::memcpy(hv4.pubkey_compressed,
                    worker_identity_->pubkey_compressed().data(), 33);
        auto canonical = auth_v4::build_canonical_message(
            static_cast<uint8_t>(jlp_wire::PROTOCOL_VERSION_V4),
            hv4.timestamp_ms, hv4.nonce, worker_name_);
        // Signing is the most likely place for a multi-second stall
        // (FIPS provider init, smart-card prompt, AV scan of the key
        // file). Bracket it so the milestone gap shows the cost.
        auth_milestone("auth_sign_begin",
                       std::string("canonical_bytes=") +
                           std::to_string(canonical.size()));
        auto sig = worker_identity_->sign_message(
            canonical.data(), canonical.size());
        if (!sig) {
            auth_milestone("auth_sign_failed", "worker_identity returned null signature");
            std::cerr << "[Pool] wire-v4 signing failed; refusing to send "
                         "AUTH (worker identity error)\n";
            ::collider::secure_wipe(hello.password, sizeof(hello.password));
            return false;
        }
        auth_milestone("auth_sign_end", "");
        std::memcpy(hv4.signature_raw, sig->data(), 64);
        auth_milestone("auth_send_begin",
                       std::string("payload_bytes=") +
                           std::to_string(sizeof(hv4)));
        const bool sent = send_message(
            JLPMessageType::AUTH, &hv4, sizeof(hv4), auth_wire_flags);
        auth_milestone(sent ? "auth_send_end" : "auth_send_failed",
                       sent ? std::string("payload_bytes=") +
                                  std::to_string(sizeof(hv4))
                            : "send_message returned false");
        ::collider::secure_wipe(hello.password, sizeof(hello.password));
        return sent;
    }

    auth_milestone("auth_send_begin",
                   std::string("payload_bytes=") +
                       std::to_string(sizeof(hello)));
    // Stamp header.flags = v3 explicitly. The default would apply
    // PROTOCOL_VERSION (now 4), which would mislabel this v3-shaped
    // JLPClientHelloV2 body as v4 and make the server route it to the v4
    // decoder (size + signature mismatch -> AUTH_FAIL).
    const bool sent = send_message(
        JLPMessageType::AUTH, &hello, sizeof(hello), auth_wire_flags);
    auth_milestone(sent ? "auth_send_end" : "auth_send_failed",
                   sent ? std::string("payload_bytes=") +
                              std::to_string(sizeof(hello))
                        : "send_message returned false");

    // Wipe the stack copy of the password before this frame is popped.
    // Without an explicit wipe the bytes linger in whatever the OS does
    // with the previous stack page until something else overwrites it,
    // and a core dump captured between authenticate() and the next
    // function call would still show the credential.
    ::collider::secure_wipe(hello.password, sizeof(hello.password));

    return sent;
}

void JLPPoolClient::set_worker_identity(
    std::shared_ptr<collider::identity::WorkerIdentity> id) {
    worker_identity_ = std::move(id);
}

namespace auth_v4 {
std::vector<uint8_t> build_canonical_message(
    uint8_t protocol_version,
    uint64_t timestamp_ms,
    const uint8_t nonce[16],
    const std::string& worker_name) {
    // Layout MUST match collision-protocol/src/jlp_protocol.py
    // JLPProtocol.build_auth_canonical_message exactly. Any drift
    // breaks v4 AUTH cross-impl.
    //   AUTH_SIG_PREFIX(24) || u8(proto_ver) || u64_le(ts_ms) ||
    //   nonce16 || u8(name_len) || name_bytes
    static const std::string kPrefix = "COLLIDER-WORKER-AUTH-v1\n";
    std::vector<uint8_t> out;
    out.reserve(kPrefix.size() + 1 + 8 + 16 + 1 + worker_name.size());
    out.insert(out.end(), kPrefix.begin(), kPrefix.end());
    out.push_back(protocol_version);
    // Little-endian u64 timestamp.
    for (int i = 0; i < 8; ++i) {
        out.push_back(static_cast<uint8_t>((timestamp_ms >> (8 * i)) & 0xFF));
    }
    out.insert(out.end(), nonce, nonce + 16);
    // Worker name capped at 255 bytes; Python's len() byte prefix
    // tops out at 0xFF so anything longer would silently truncate.
    // The pool side caps worker_name at 64 bytes (BTC address fits in
    // 42-62), so this is defensive only.
    const size_t name_len = (worker_name.size() > 255) ? 255 : worker_name.size();
    out.push_back(static_cast<uint8_t>(name_len));
    out.insert(out.end(), worker_name.begin(),
               worker_name.begin() + name_len);
    return out;
}
}  // namespace auth_v4

bool JLPPoolClient::request_work(WorkAssignment& work) {
    // Send work request
    if (!send_message(JLPMessageType::WORK_REQ, nullptr, 0)) {
        return false;
    }

    // Wait for work (with timeout). Use work_received_ (not work_id != 0)
    // because chunk ID 0 is a valid first assignment from the pool.
    auto start = std::chrono::steady_clock::now();
    while (running_) {
        {
            std::lock_guard<std::mutex> lock(work_mutex_);
            if (work_received_) {
                work = current_work_;
                return true;
            }
        }

        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > timeout_ms_) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return false;
}

bool JLPPoolClient::submit_dp(const DistinguishedPoint& dp) {
    std::lock_guard<std::mutex> lock(dp_mutex_);

    // Backpressure: reject if queue is full to prevent OOM
    if (dp_queue_.size() >= MAX_DP_QUEUE_SIZE) {
        // Queue full - caller should slow down or retry
        return false;
    }

    dp_queue_.push_back(dp);
    dp_cv_.notify_one();
    return true;
}

bool JLPPoolClient::submit_dps(const std::vector<DistinguishedPoint>& dps) {
    std::lock_guard<std::mutex> lock(dp_mutex_);

    // Backpressure: reject entire batch if it would overflow
    if (dp_queue_.size() + dps.size() > MAX_DP_QUEUE_SIZE) {
        return false;
    }

    for (const auto& dp : dps) {
        dp_queue_.push_back(dp);
    }
    dp_cv_.notify_one();
    return true;
}

PoolStatsLocal JLPPoolClient::get_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

// v1.5: JLPPoolClient::report_solution() and its supporting infrastructure
// (24-hour retry uploader, recovered_keys/<ts>.json atomic write,
// SecureBuffer key staging) were DELETED. The pool server (collision-
// protocol) is the sole key-computer; the SOLUTION wire message is
// strictly server-to-client. There is no longer any code path on the
// worker that handles a 32-byte recovered private key.
// See .claude/tasks/v1.5-asymmetric-kangaroo.md.

void JLPPoolClient::set_solution_callback(SolutionCallback cb) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    solution_callback_ = cb;
}

void JLPPoolClient::set_work_callback(WorkCallback cb) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    work_callback_ = cb;
}

// dp_sequence_next_ seeding from PoolManager so a
// supervisor-driven reconnect doesn't reset it to 0.
void JLPPoolClient::seed_dp_sequence(uint64_t work_id, uint32_t next_seq) {
    std::lock_guard<std::mutex> lock(work_mutex_);
    // Seed both fields together so the snapshot below is consistent.
    current_work_.work_id = work_id;
    dp_sequence_next_ = next_seq;
    // Note: work_received_ stays as-is. A real WORK_ASN must still
    // arrive from the server to set work_received_=true and populate
    // the public_key / range_* fields.
}

void JLPPoolClient::set_dp_sequence_progress_callback(DpSeqProgressCallback cb) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    dp_seq_progress_callback_ = std::move(cb);
}

void JLPPoolClient::set_work_id_assigned_callback(WorkIdAssignedCallback cb) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    work_id_assigned_callback_ = std::move(cb);
}

void JLPPoolClient::set_kangaroo_type_changed_callback(
        KangarooTypeChangedCallback cb) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    kangaroo_type_changed_callback_ = std::move(cb);
}

void JLPPoolClient::set_maintenance_callback(MaintenanceCallback cb) {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    maintenance_callback_ = std::move(cb);
}

void JLPPoolClient::snapshot_dp_sequence(uint64_t& out_work_id,
                                          uint32_t& out_next_seq) const {
    std::lock_guard<std::mutex> lock(work_mutex_);
    out_work_id  = current_work_.work_id;
    out_next_seq = dp_sequence_next_;
}

// print a clear plaintext warning. Idempotent on
// jlps://: caller is expected to gate on !use_tls before calling.
void JLPPoolClient::warn_if_plaintext(const std::string& host,
                                       uint16_t port) {
    std::cerr << "[Pool] WARNING: connecting over plaintext jlp:// to "
              << host << ":" << port
              << ". The AUTH payload carries your worker name (= payout "
                 "address) and password. Use jlps:// in production."
              << std::endl;
}

// snapshot the queued DPs so caller can persist them
// to disk before the client is destroyed.
std::vector<DistinguishedPoint> JLPPoolClient::snapshot_dp_queue() const {
    std::vector<DistinguishedPoint> out;
    std::lock_guard<std::mutex> lock(dp_mutex_);
    out.reserve(dp_queue_.size());
    for (const auto& dp : dp_queue_) {
        out.push_back(dp);
    }
    return out;
}

// re-queue DPs that were on disk from a prior shutdown.
// Called before authenticate() so the first batch sent post-AUTH_OK
// includes the backlog.
void JLPPoolClient::preload_dp_queue(std::vector<DistinguishedPoint>&& persisted) {
    if (persisted.empty()) return;
    std::lock_guard<std::mutex> lock(dp_mutex_);
    // Preserve original order: persisted[0] is the oldest, should be
    // first off the queue. Re-queued DPs go at the front so they
    // precede any new DPs that arrived between preload and AUTH_OK
    // (though in practice the kernel callback isn't firing yet at this
    // point in the connect path).
    // We cap at MAX_DP_QUEUE_SIZE; anything beyond that is dropped with
    // a stderr note. This is a recovery path, not a steady-state path;
    // exceeding the cap means the prior session was very productive
    // and the operator deserves to know.
    size_t take = std::min(persisted.size(), MAX_DP_QUEUE_SIZE);
    if (take < persisted.size()) {
        std::cerr << "[Pool] preload_dp_queue: persisted queue had "
                  << persisted.size() << " DPs but cap is "
                  << MAX_DP_QUEUE_SIZE << "; dropping "
                  << (persisted.size() - take) << "." << std::endl;
    }
    for (size_t i = 0; i < take; ++i) {
        dp_queue_.push_back(persisted[i]);
    }
    dp_cv_.notify_one();
}

// write queued DPs to a binary length-prefixed file so
// the next process start can replay them. Format:
//   [u32 magic 'CDPQ'][u32 count][count * (32 x, 32 d, u8 type, u64 dp_bits)]
//   = 8 + count * 73 bytes
bool JLPPoolClient::persist_dp_queue_to_disk(const std::string& path) const {
    std::vector<DistinguishedPoint> snap = snapshot_dp_queue();
    if (snap.empty()) {
        // Nothing to persist; remove any stale file from a prior
        // shutdown so we don't re-replay it next time.
        std::error_code ec;
        std::filesystem::remove(path, ec);
        return true;
    }
    std::string buf;
    buf.reserve(8 + snap.size() * 73);
    auto put_u32 = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) {
            buf.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
        }
    };
    auto put_u64 = [&](uint64_t v) {
        for (int i = 0; i < 8; ++i) {
            buf.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
        }
    };
    // Magic 'CDPQ' little-endian
    buf.push_back('C');
    buf.push_back('D');
    buf.push_back('P');
    buf.push_back('Q');
    put_u32(static_cast<uint32_t>(snap.size()));
    for (const auto& dp : snap) {
        buf.append(reinterpret_cast<const char*>(dp.x), 32);
        buf.append(reinterpret_cast<const char*>(dp.d), 32);
        buf.push_back(static_cast<char>(dp.type));
        put_u64(dp.dp_bits);
    }
    return atomic_write(std::filesystem::path(path), buf);
}

// dedup + safe-fire helpers.
// Both helpers consolidate the copy-callback-under-lock-then-call-
// outside pattern that pre-1.4.1 was hand-coded only at the
// WORK_ASN site (and skipped entirely at the SOLUTION site). The
// dedup keys:
//   * fire_solution_callback: 32-byte solution payload (server-
//     broadcast on SOLUTION). v1.5: this is the pool-computed
//     recovered key bytes, but the worker treats them as opaque
//     stop-signal metadata -- the dedup exists so a server retransmit
//     doesn't fire the host's shutdown latch twice. Parameter is named
//     `pubkey_bytes` for historical reasons; functionally it is now
//     "the bytes the server broadcast in the SOLUTION frame".
//   * fire_work_callback: WorkAssignment.work_id. The server should
//     not re-issue an identical work_id, but it's cheap defense and
//     prevents duplicate kangaroo-restarts on a flaky link.
// Both lock on the same mutex as set_*_callback so the function
// pointer can never tear and the dedup state stays consistent.
void JLPPoolClient::fire_solution_callback(const uint8_t* pubkey_bytes) {
    SolutionCallback cb;
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        if (solution_fired_
            && std::memcmp(last_solution_pubkey_.data(), pubkey_bytes, 32) == 0) {
            // Same pubkey already delivered; suppress.
            return;
        }
        std::memcpy(last_solution_pubkey_.data(), pubkey_bytes, 32);
        solution_fired_ = true;
        cb = solution_callback_;
    }
    if (cb) {
        cb(pubkey_bytes);
    }
}

void JLPPoolClient::fire_work_callback(const WorkAssignment& work) {
    // client-instance work_id dedup was removed because
    // the supervisor recreates JLPPoolClient on reconnect, which reset
    // any last_work_id_fired_ sentinel back to numeric_limits::max() and
    // re-fired the same work_id to the host after every reconnect.
    // PoolManager now owns the dedup state (last_work_id_seen_) and
    // outlives client recreation. Also fire the work_id_assigned
    // callback so the manager's per-(worker, work_id) DP sequence map
    // resets on genuine new chunks.
    WorkCallback cb;
    WorkIdAssignedCallback wid_cb;
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        cb = work_callback_;
        wid_cb = work_id_assigned_callback_;
    }
    if (wid_cb) {
        wid_cb(work.work_id);
    }
    if (cb) {
        cb(work);
    }
}

bool JLPPoolClient::send_message(JLPMessageType type, const void* data, size_t size,
                                 uint8_t flags_override) {
    if (!connected_ || socket_ == INVALID_SOCK) {
        return false;
    }

    // New JLP protocol format: [MAGIC:4][TYPE:1][FLAGS:1][LENGTH:2] = 8 bytes
    // the `flags` byte carries PROTOCOL_VERSION. Senders MUST
    // set it; receivers MUST validate it. Pre-fix this was hard-coded 0,
    // which the spec/IDL has now repurposed as "v0 / legacy".
    // B1 wire-v4: flags_override lets the AUTH path bump to
    // PROTOCOL_VERSION_V4 without affecting all other messages.
    JLPHeader header;
    header.magic[0] = 'K';
    header.magic[1] = 'A';
    header.magic[2] = 'N';
    header.magic[3] = 'G';
    header.type = static_cast<uint8_t>(type);
    header.flags = (flags_override == 0xFF)
        ? static_cast<uint8_t>(jlp_wire::PROTOCOL_VERSION)
        : flags_override;
    header.payload_size = static_cast<uint16_t>(size);  // 2-byte little-endian

    // Serialize the header+payload write so two writers (sender_loop and
    // main thread) cannot interleave their bytes. Reads run concurrently
    // on a separate mutex; this is deliberate -- a single shared mutex
    // would deadlock against the receiver's blocking SSL_read.
    std::lock_guard<std::mutex> io_lock(ssl_write_mutex_);

    // Send header
#ifdef COLLIDER_HAS_OPENSSL
    if (use_tls_ && ssl_) {
        if (ssl_send(&header, sizeof(header)) != (int)sizeof(header)) {
            connected_ = false;
            return false;
        }
    } else
#endif
    {
        if (send(socket_, (const char*)&header, sizeof(header), 0) != sizeof(header)) {
            connected_ = false;
            return false;
        }
    }

    // Send payload
    if (size > 0 && data != nullptr) {
#ifdef COLLIDER_HAS_OPENSSL
        if (use_tls_ && ssl_) {
            if (ssl_send(data, size) != (int)size) {
                connected_ = false;
                return false;
            }
        } else
#endif
        {
            if (send(socket_, (const char*)data, (int)size, 0) != (int)size) {
                connected_ = false;
                return false;
            }
        }
    }

    return true;
}

bool JLPPoolClient::receive_message(JLPHeader& header, std::vector<uint8_t>& payload) {
    if (!connected_ || socket_ == INVALID_SOCK) {
        return false;
    }

    // Serialize only against other readers (only the receiver_loop reads
    // in this design, so this is defensive). Crucially, we do NOT share a
    // mutex with the writer side: the receiver may block here for minutes
    // waiting on the next server frame, and the main thread must remain
    // free to send WORK_REQ / DP_BATCH during that wait.
    // We hold the lock around BOTH the header read and the payload read so
    // a future second reader could not splice frames; the SSL object
    // tolerates concurrent SSL_read and SSL_write per OpenSSL's threading
    // contract.
    std::unique_lock<std::mutex> io_lock(ssl_read_mutex_);

    // STREAM-RESYNC FIX (task #34, v1.5.x): when SO_RCVTIMEO fires
    // mid-message Windows recv() returns the PARTIAL bytes received so
    // far. The previous code returned that partial read as "timeout, try
    // again" -- but those bytes were already drained from the socket's
    // read queue, so the next iteration started at a byte offset inside
    // the original frame. With the 1ms timeout the JLP test harness uses
    // (set_timeout(1) in make_connected) this misalignment fired on
    // nearly every frame, eventually corrupting the stream badly enough
    // that the resulting bogus header crashed downstream via __fastfail
    // with FAST_FAIL_RANGE_CHECK_FAILURE.
    // Fix: accumulate partial reads into a per-call buffer that persists
    // ACROSS timeout returns within the SAME receive_message invocation.
    // The byte-count progress is kept on the JLPPoolClient member fields
    // (header_progress_ / payload_progress_) so a return-on-timeout
    // preserves the offset for the next call; the resync invariant is
    // "we never lose a byte once it has been recv'd off the socket."
    auto wait_for_bytes = [&](uint8_t* dst, size_t want, size_t& progress,
                              bool* out_timed_out) -> bool {
        // progress < want on entry; recv up to (want - progress) bytes,
        // accumulate, return true on full fill; out_timed_out=true means
        // partial but recoverable (call again); false return + out_timed
        // _out=false means real connection error.
        *out_timed_out = false;
        while (progress < want) {
            if (!connected_ || socket_ == INVALID_SOCK) {
                return false;
            }
            int got = 0;
#ifdef COLLIDER_HAS_OPENSSL
            if (use_tls_ && ssl_) {
                got = ssl_recv(dst + progress,
                               static_cast<int>(want - progress));
                if (got <= 0) {
                    int ssl_err = SSL_get_error(ssl_, got);
                    if (ssl_err == SSL_ERROR_WANT_READ ||
                        ssl_err == SSL_ERROR_WANT_WRITE) {
                        *out_timed_out = true;
                        return false;
                    }
                    if (ssl_err == SSL_ERROR_SYSCALL) {
#ifdef _WIN32
                        int se = WSAGetLastError();
                        if (se == WSAETIMEDOUT || se == WSAEWOULDBLOCK) {
                            *out_timed_out = true;
                            return false;
                        }
#else
                        if (errno == EAGAIN || errno == EWOULDBLOCK ||
                            errno == ETIMEDOUT) {
                            *out_timed_out = true;
                            return false;
                        }
#endif
                    }
                    return false;
                }
            } else
#endif
            {
                got = recv(socket_, reinterpret_cast<char*>(dst + progress),
                           static_cast<int>(want - progress), 0);
                if (got <= 0) {
#ifdef _WIN32
                    int err = WSAGetLastError();
                    if (err == WSAETIMEDOUT || err == WSAEWOULDBLOCK) {
                        *out_timed_out = true;
                        return false;
                    }
#else
                    if (errno == EAGAIN || errno == EWOULDBLOCK ||
                        errno == ETIMEDOUT) {
                        *out_timed_out = true;
                        return false;
                    }
#endif
                    return false;
                }
            }
            progress += static_cast<size_t>(got);
        }
        return true;
    };

    // Receive header into the per-client partial buffer so a timeout
    // resumes from the right offset on the next call.
    {
        bool timed_out = false;
        if (!wait_for_bytes(reinterpret_cast<uint8_t*>(&header_partial_),
                            sizeof(header_partial_),
                            header_partial_progress_,
                            &timed_out)) {
            last_receive_was_timeout_ = timed_out;
            return false;
        }
        header = header_partial_;
        header_partial_progress_ = 0;  // consumed; reset for next message
    }
    last_receive_was_timeout_ = false;

    // Validate magic
    if (header.magic[0] != 'K' || header.magic[1] != 'A' ||
        header.magic[2] != 'N' || header.magic[3] != 'G') {
        std::cerr << "[Pool] Invalid message magic" << std::endl;
        return false;
    }

    // validate protocol version. The `flags` byte now carries
    // PROTOCOL_VERSION. A mismatch indicates the peer is running a different
    // wire version (legacy v0 / future v3) and we should disconnect rather
    // than silently mis-decode. The MSG_ERROR/protocol_version_mismatch
    // (0x10) response is the correct server reply but we cannot send it
    // here because the receiver is one-direction; we drop the connection
    // and rely on the supervisor to log + retry.
    if (header.flags != jlp_wire::PROTOCOL_VERSION) {
        std::cerr << "[Pool] Protocol version mismatch: server sent flags="
                  << (int)header.flags << ", expected "
                  << (int)jlp_wire::PROTOCOL_VERSION
                  << " (PROTOCOL_VERSION). Disconnecting.\n";
        return false;
    }

    // payload_size is uint16_t on the wire so the upper
    // bound is 65535 by construction; the previous 65000 cap was an
    // inherited magic number with no semantic meaning. We still defer to
    // MAX_MESSAGE_SIZE in the IDL (1 MiB) when it is smaller, but in
    // practice the wire field width is the binding constraint for any
    // single JLP message. The check remains defensive against a future
    // wire-field-width change.
    constexpr uint32_t U16_MAX_PAYLOAD = 65535u;
    constexpr uint32_t MAX_PAYLOAD_SIZE =
        jlp_wire::MAX_MESSAGE_SIZE < U16_MAX_PAYLOAD
            ? jlp_wire::MAX_MESSAGE_SIZE
            : U16_MAX_PAYLOAD;
    if (static_cast<uint32_t>(header.payload_size) > MAX_PAYLOAD_SIZE) {
        std::cerr << "[Pool] Payload size exceeds limit: " << header.payload_size
                  << " bytes (cap " << MAX_PAYLOAD_SIZE << ")" << std::endl;
        return false;
    }

    // Receive payload, also using the partial-accumulating helper. The
    // payload uses its own per-client member buffer so a timeout
    // during the payload phase preserves the bytes already drained.
    // payload_partial_progress_ resets to 0 only when the FULL payload
    // has been received; until then, every receive_message call resumes
    // exactly where the previous call left off.
    if (header.payload_size > 0) {
        if (payload_partial_.size() != header.payload_size) {
            payload_partial_.resize(header.payload_size);
        }
        bool timed_out = false;
        if (!wait_for_bytes(payload_partial_.data(),
                            header.payload_size,
                            payload_partial_progress_,
                            &timed_out)) {
            last_receive_was_timeout_ = timed_out;
            // CRITICAL: do NOT reset header_partial_progress_ or
            // payload_partial_progress_ here. The next call must resume
            // payload-phase reads on this same header. Stash the header
            // back into header_partial_ so the next call skips the
            // header read (header_partial_progress_ = sizeof()).
            header_partial_ = header;
            header_partial_progress_ = sizeof(header_partial_);
            return false;
        }
        payload = std::move(payload_partial_);
        payload_partial_.clear();
        payload_partial_progress_ = 0;
    }

    return true;
}

void JLPPoolClient::receiver_loop() {
    // the previous in-receiver-thread reconnect block was
    // dead code. It logged "reconnect attempt N" then ALWAYS returned
    // after a single sleep, with the actual reconnect deferred to the
    // PoolManager supervisor. RECONNECT_BACKOFF_MULTIPLIER never ran more
    // than once, MAX_AUTH_FAIL_ATTEMPTS was unreachable from here (the
    // AUTH_FAIL handler exits the loop via running_=false in
    // handle_server_message), and the misleading constants suggested a
    // retry policy that didn't actually exist.
    // The supervisor (pool_manager.cpp PoolManager::supervisor_loop) is
    // the single reconnect driver. On socket loss we simply tear down our
    // own state and exit; the supervisor's 500ms probe will see
    // !is_connected() and drive a fresh connect()+authenticate() from a
    // safe thread context.

    while (running_ && connected_) {
        JLPHeader header;
        std::vector<uint8_t> payload;

        if (receive_message(header, payload)) {
            handle_server_message(header, payload);
        } else if (last_receive_was_timeout_) {
            // Just a timeout, not a real disconnect; keep waiting. This is
            // normal when the server has nothing to send and the keepalive
            // PING from sender_loop drives liveness.
            continue;
        } else {
            // Actual connection loss. Flip state to disconnected and exit.
            // The PoolManager supervisor takes over from here (recreates the
            // client, calls connect()+authenticate() with a jittered backoff,
            // and respects MAX_AUTH_FAIL_ATTEMPTS / MAX_RECONNECT_BACKOFF_MS).
            connected_ = false;
            auth_state_.store(AuthState::DISCONNECTED);
            // Wake any authenticate() waiter so it returns promptly with
            // the disconnected state instead of timing out.
            auth_cv_.notify_all();
            // Wake the sender so it observes the state change and exits
            // (or finishes its drain) instead of blocking another full
            // wait_for cycle.
            dp_cv_.notify_all();
            running_ = false;
            return;
        }
    }
}

void JLPPoolClient::sender_loop() {
    // v2 wire format: every DP carries the worker's claimed work_id so the
    // server can match it to the chunk currently assigned to this worker.
    // The server still accepts v1 submissions from older clients deployed
    // in the field; freshly-built binaries always emit v2.
    std::vector<JLPDistinguishedPointV2> batch;
    batch.reserve(kSenderMaxBatchDps);
    // payload is the staging buffer for DP_BATCH_V2 emission. Reusing it
    // across iterations (and across send_dp_batch / perform_final_drain_pass
    // calls) avoids re-allocating on every drain.
    std::vector<uint8_t> payload;
    payload.reserve(4 + kSenderMaxBatchDps * sizeof(JLPDistinguishedPointV2));

    auto last_send      = std::chrono::steady_clock::now();
    auto last_stats_req = std::chrono::steady_clock::now()
                          - kSenderStatsIntervalSeconds;  // fire once on first iteration

    while (running_) {
        wait_for_send_signal();

        // Snapshot current_work_id AFTER the wait so a WORK_ASN that
        // arrived during the wait_for is picked up. Lock order
        // (work_mutex_ then dp_mutex_) mirrors PoolManager::dp_callback_hook
        // to avoid a deadlock cycle.
        uint64_t current_work_id = 0;
        uint32_t seq_start = 0;
        uint8_t expected_type = kAnyDpType;
        {
            std::lock_guard<std::mutex> wlock(work_mutex_);
            current_work_id = current_work_.work_id;
            seq_start = dp_sequence_next_;
            // Derive the permitted wire type from kangaroo_type (KANG_MODE):
            // 1=TAME_ONLY -> 0 (tame), 2=WILD_ONLY -> 1 (wild). Anything
            // else (0/illegal/unset) disables the filter.
            expected_type = kang_type_to_dp_type(current_work_.kangaroo_type);
        }

        drain_dp_queue_into_batch(current_work_id, seq_start, expected_type, batch);

        if (!batch.empty()) {
            // Optimistically advance the per-chunk sequence counter to
            // cover the batch we just drafted. The send-or-requeue path
            // below rolls back the advance if AUTH wasn't ready.
            advance_dp_sequence_if_unchanged(
                current_work_id,
                seq_start + static_cast<uint32_t>(batch.size()));

            const bool auth_ok =
                (auth_state_.load() == AuthState::AUTH_OK);
            const bool conn = connected_.load();

            if (conn && auth_ok) {
                send_dp_batch(current_work_id, seq_start, batch, payload, last_send);
                batch.clear();
            } else if (running_.load()) {
                requeue_unauth_batch(current_work_id, seq_start, batch);
                batch.clear();
            } else {
                // Genuine shutdown: running_ has flipped to false. The
                // disconnect() drain path runs BEFORE running_ flips,
                // so reaching here means we are past the drain window
                // and the queue still wasn't empty. Count the loss for
                // visibility but do not retry.
                shutdown_drop_count_.fetch_add(batch.size(),
                                               std::memory_order_relaxed);
                std::cerr << "[Pool] sender: dropping " << batch.size()
                          << " DP(s) on shutdown" << std::endl;
                batch.clear();
            }
        }

        if (drain_requested_.load(std::memory_order_acquire)) {
            perform_final_drain_pass(batch, payload, last_send);
        }

        send_periodic_stats_and_keepalive(last_send, last_stats_req);
    }
}

void JLPPoolClient::wait_for_send_signal() {
    // Wait for queue activity (releases dp_mutex_ during the wait). Also
    // wakes promptly if disconnect() asked for a drain or running_ flipped
    // to false. 100ms upper bound caps keepalive-window jitter.
    std::unique_lock<std::mutex> lock(dp_mutex_);
    dp_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
        return !dp_queue_.empty() || !running_
            || drain_requested_.load(std::memory_order_acquire);
    });
}

size_t JLPPoolClient::drain_dp_queue_into_batch(
        uint64_t current_work_id,
        uint32_t seq_start,
        uint8_t expected_type,
        std::vector<JLPDistinguishedPointV2>& batch) {
    // Drain up to kSenderMaxBatchDps DPs under dp_mutex_ alone. The lock
    // is held only long enough to copy out the batch and pop the source
    // entries; producers (submit_dp / submit_dps) can proceed as soon as
    // we return. Wire is little-endian on x86/ARM-LE; convert explicitly
    // so this remains correct on big-endian builds. Manual byte assembly
    // avoids the missing-htole64 portability headache (Windows lacks it).
    const size_t before = batch.size();
    uint64_t dropped_this_call = 0;
    std::lock_guard<std::mutex> lock(dp_mutex_);
    while (!dp_queue_.empty() && batch.size() < kSenderMaxBatchDps) {
        const auto& dp = dp_queue_.front();

        // v1.5 client-side type-mismatch safety net. The currently-assigned
        // kangaroo_type permits exactly one DP wire type (0=tame, 1=wild).
        // A DP carrying the other type can only be a stale leak from the
        // prior herd or a backend bug; shipping it makes the server drop and
        // penalize us, so we drop it locally instead. expected_type ==
        // kAnyDpType disables the filter (no asymmetric assignment active),
        // preserving legacy behavior. Dropped DPs do NOT consume a sequence
        // number: only batch.size() (the shipped count) drives seq, so the
        // shipped stream stays contiguous.
        if (expected_type != kAnyDpType && dp.type != expected_type) {
            dp_queue_.pop_front();
            ++dropped_this_call;
            continue;
        }

        JLPDistinguishedPointV2 jlp_dp;
        {
            uint8_t* p = reinterpret_cast<uint8_t*>(&jlp_dp.work_id);
            for (int i = 0; i < 8; ++i) {
                p[i] = static_cast<uint8_t>((current_work_id >> (8 * i)) & 0xFFu);
            }
        }
        {
            // per-(worker, work_id) sequence (LE).
            uint32_t seq = seq_start + static_cast<uint32_t>(batch.size());
            uint8_t* p = reinterpret_cast<uint8_t*>(&jlp_dp.sequence);
            for (int i = 0; i < 4; ++i) {
                p[i] = static_cast<uint8_t>((seq >> (8 * i)) & 0xFFu);
            }
        }
        memcpy(jlp_dp.x, dp.x, 32);
        memcpy(jlp_dp.d, dp.d, 32);
        jlp_dp.type = dp.type;
        jlp_dp.dp_bits = static_cast<uint8_t>(dp.dp_bits & 0xFF);
        batch.push_back(jlp_dp);
        dp_queue_.pop_front();
    }

    if (dropped_this_call != 0) {
        const uint64_t total =
            dropped_type_mismatch_.fetch_add(dropped_this_call,
                                             std::memory_order_relaxed)
            + dropped_this_call;
        // Throttled WARN (every Nth drop, power-of-two mask) so a sustained
        // mismatch storm cannot flood the log.
        constexpr uint64_t kWarnEvery = 256;
        if ((total & (kWarnEvery - 1)) < dropped_this_call) {
            std::cerr << "[Pool] WARN dropped " << dropped_this_call
                      << " DP(s) with type != assigned (expected="
                      << static_cast<int>(expected_type)
                      << "); " << total
                      << " total since startup. Client-side safety net for "
                         "v1.5 asymmetric assignment; these never reached the "
                         "server." << std::endl;
        }
    }
    return batch.size() - before;
}

void JLPPoolClient::advance_dp_sequence_if_unchanged(uint64_t current_work_id,
                                                     uint32_t new_next) {
    // Compare-and-set: only write if the chunk under work_mutex_ still
    // matches the one we drafted the batch under. If WORK_ASN raced our
    // drain (work_id changed mid-flight), the DPs we drained are tagged
    // with the OLD work_id and will be rejected by the server's work_id
    // check anyway. We must NOT stomp the new chunk's counter, which
    // WORK_ASN reset to 0. Done OUTSIDE dp_mutex_ to keep the lock order
    // (work_mutex_ then dp_mutex_).
    std::lock_guard<std::mutex> wlock(work_mutex_);
    if (current_work_.work_id == current_work_id) {
        dp_sequence_next_ = new_next;
    }
}

bool JLPPoolClient::send_dp_batch(
        uint64_t current_work_id,
        uint32_t seq_start,
        const std::vector<JLPDistinguishedPointV2>& batch,
        std::vector<uint8_t>& payload,
        std::chrono::steady_clock::time_point& last_send) {
    // Wire format expected by the pool: [count:u32 LE][dp1:78][dp2:78]...
    // (DP_BATCH_V2 uses 78-byte v2 entries; the leading u32 count is
    // capped at 10000 server-side, so without it the first 4 bytes of
    // the first DP get interpreted as the count and the server tears
    // down the connection with "Batch count N exceeds max 10000".)
    payload.clear();
    uint32_t count = static_cast<uint32_t>(batch.size());
    payload.push_back(static_cast<uint8_t>( count        & 0xFFu));
    payload.push_back(static_cast<uint8_t>((count >>  8) & 0xFFu));
    payload.push_back(static_cast<uint8_t>((count >> 16) & 0xFFu));
    payload.push_back(static_cast<uint8_t>((count >> 24) & 0xFFu));
    payload.insert(payload.end(),
                   reinterpret_cast<const uint8_t*>(batch.data()),
                   reinterpret_cast<const uint8_t*>(batch.data())
                       + batch.size() * sizeof(JLPDistinguishedPointV2));
    const bool ok = send_message(JLPMessageType::DP_BATCH_V2,
                                 payload.data(), payload.size());
    if (!ok) {
        return false;
    }
    last_send = std::chrono::steady_clock::now();

    // Mirror the advanced sequence counter back to PoolManager so a
    // supervisor-driven reconnect that recreates this client seeds the
    // new instance with the post-batch value (not 0).
    DpSeqProgressCallback cb;
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        cb = dp_seq_progress_callback_;
    }
    if (cb) {
        cb(current_work_id,
           seq_start + static_cast<uint32_t>(batch.size()));
    }

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.your_dps += batch.size();
    }

    // Session log: bump the lifetime + per-work DP counters, then push
    // an UPDATED state snapshot. The update is throttled internally
    // (~5s between disk writes), so even a 1000-DP/s sender does at
    // most ~12 fsyncs/minute.
    uint64_t batch_n = batch.size();
    g_dp_total_submitted.fetch_add(batch_n, std::memory_order_acq_rel);
    g_dp_submitted_this_work.fetch_add(batch_n, std::memory_order_acq_rel);
    g_last_dp_seq.store(
        seq_start + static_cast<uint32_t>(batch.size()),
        std::memory_order_release);

    // dp_batch_submitted heartbeat. Emits one milestone every 10 batches
    // so a forensic reader can detect a worker that successfully sent
    // some batches then hung (no follow-on heartbeat in the log) without
    // having to ingest the per-batch session_state.json stream. The
    // counters are process-static atomics so a supervisor-driven
    // reconnect that recreates this client keeps the running totals
    // monotonic across the recreation boundary. Throttled at 10 batches
    // so even a 100-batch/s sender adds at most 10 milestone lines/sec.
    //
    // TP-15: the update_session_state(build_pool_state_seed(...)) call
    // used to fire on EVERY batch. The session-state writer itself was
    // throttled to 5s internally, but the seed builder (string concat,
    // atomic loads, SessionState struct construction) ran every time
    // -- ~100x/sec on a saturated sender. Folded inside the 10-batch
    // milestone block so the seed builds + flushes share a cadence;
    // the operator-visible session_state.json freshness goes from
    // "every batch" (stale by up to 5s anyway) to "every 10 batches"
    // (stale by up to ~1s at production rates), which matches the
    // milestone heartbeat granularity the forensic reader already uses.
    {
        static std::atomic<uint64_t> batches_local{0};
        static std::atomic<uint64_t> dps_local{0};
        const uint64_t b = batches_local.fetch_add(1, std::memory_order_relaxed) + 1;
        const uint64_t d = dps_local.fetch_add(batch_n, std::memory_order_relaxed) + batch_n;
        if ((b % 10) == 0) {
            std::ostringstream m;
            m << "batches=" << b
              << " dps=" << d
              << " work_id=" << current_work_id;
            ::collider::log::milestone("dp_batch_submitted", m.str());
            ::collider::log::update_session_state(
                build_pool_state_seed(
                    host_ + ":" + std::to_string(port_),
                    /*connected=*/true));
        }
    }
    return true;
}

void JLPPoolClient::requeue_unauth_batch(
        uint64_t current_work_id,
        uint32_t seq_start,
        const std::vector<JLPDistinguishedPointV2>& batch) {
    // We drained DPs from the queue but the server hasn't accepted us
    // yet (mid-reconnect or AUTH still pending). Pre-fix, the batch was
    // silently dropped via batch.clear() and the sequence counter had
    // ALREADY been advanced, so the next batch sent post-AUTH_OK would
    // skip those sequence numbers and appear as a gap on the server
    // side. Correct behavior: roll the sequence counter back, then push
    // the DPs BACK onto the front of dp_queue_ in reverse order so
    // their order on the wire is preserved. They'll be re-drained on
    // the next iteration after AUTH_OK.
    {
        std::lock_guard<std::mutex> wlock(work_mutex_);
        if (current_work_.work_id == current_work_id) {
            // Undo the optimistic advance done by the caller.
            dp_sequence_next_ = seq_start;
        }
    }
    {
        std::lock_guard<std::mutex> lock(dp_mutex_);
        // Push back in reverse so the original head ends up at the
        // front of the deque again. We rebuild a DistinguishedPoint
        // out of the JLP wire struct so the queue's element type stays
        // consistent; fields are identical in semantics, just narrower
        // (dp_bits is uint8 vs uint64 in the abstract type).
        for (auto it = batch.rbegin(); it != batch.rend(); ++it) {
            DistinguishedPoint dp{};
            memcpy(dp.x, it->x, 32);
            memcpy(dp.d, it->d, 32);
            dp.type = it->type;
            dp.dp_bits = it->dp_bits;
            dp_queue_.push_front(dp);
        }
    }
    // Rate-limited warning so a long pre-AUTH stall is visible in
    // operator logs.
    static std::atomic<uint64_t> warn_counter{0};
    if ((warn_counter.fetch_add(1) & 31) == 0) {
        const bool conn = connected_.load();
        std::cerr << "[Pool] sender: AUTH not yet OK; "
                  << batch.size()
                  << " DP(s) re-queued for retry (state="
                  << static_cast<int>(auth_state_.load())
                  << ", connected=" << (conn ? 1 : 0)
                  << ")" << std::endl;
    }
}

void JLPPoolClient::perform_final_drain_pass(
        std::vector<JLPDistinguishedPointV2>& batch,
        std::vector<uint8_t>& payload,
        std::chrono::steady_clock::time_point& last_send) {
    // disconnect() asked for a drain: ship whatever remains in the
    // queue WHILE we still have connected_ + AUTH_OK. Exits when the
    // queue is empty, running_ flips false, AUTH state slips, or a
    // send fails mid-drain. Each iteration re-samples work_id under
    // work_mutex_ in case WORK_ASN landed between iterations.
    while (running_.load()
           && connected_.load()
           && auth_state_.load() == AuthState::AUTH_OK) {
        uint64_t wid = 0;
        uint32_t seq = 0;
        uint8_t expected_type = kAnyDpType;
        {
            std::lock_guard<std::mutex> wlock(work_mutex_);
            wid = current_work_.work_id;
            seq = dp_sequence_next_;
            expected_type = kang_type_to_dp_type(current_work_.kangaroo_type);
        }
        drain_dp_queue_into_batch(wid, seq, expected_type, batch);
        if (batch.empty()) break;
        advance_dp_sequence_if_unchanged(
            wid, seq + static_cast<uint32_t>(batch.size()));
        if (!send_dp_batch(wid, seq, batch, payload, last_send)) {
            batch.clear();
            break;  // Network failed mid-drain; abandon
        }
        batch.clear();
    }
    // Notify any waiter (disconnect) that the queue is now empty (or
    // we couldn't drain further).
    dp_cv_.notify_all();
}

void JLPPoolClient::send_periodic_stats_and_keepalive(
        std::chrono::steady_clock::time_point& last_send,
        std::chrono::steady_clock::time_point& last_stats_req) {
    // Both STATS_REQ and PING are gated on AUTH_OK so we don't talk to
    // a server that hasn't accepted us yet. STATS_REQ refreshes the
    // pool-aggregated counters (your_dps across all machines sharing
    // this worker name). PING resets the server's per-message read
    // timeout. STATS_REQ counts as activity for keepalive purposes, so
    // the PING is skipped if STATS_REQ went out inside the same window.
    if (!connected_ || auth_state_.load() != AuthState::AUTH_OK) {
        return;
    }
    auto now = std::chrono::steady_clock::now();
    if (now - last_stats_req >= kSenderStatsIntervalSeconds) {
        send_message(JLPMessageType::STATS_REQ, nullptr, 0);
        last_stats_req = now;
        last_send = now;
    }
    if (now - last_send >= kSenderKeepaliveSeconds) {
        send_message(JLPMessageType::PING, nullptr, 0);
        last_send = now;
    }
}

void JLPPoolClient::handle_server_message(const JLPHeader& header,
                                          const std::vector<uint8_t>& payload) {
    JLPMessageType msg_type = static_cast<JLPMessageType>(header.type);

    // gate work-affecting messages on AUTH_OK. A malicious or
    // misbehaving pool MUST NOT be able to inject WORK_ASN / SOLUTION / DP_ACK
    // / STATS_RSP before authentication completes.
    // Allowed pre-AUTH_OK: AUTH_OK, AUTH_FAIL, MSG_ERROR, PING/PONG.
    // Anything else: log + ignore (do NOT crash, do NOT disconnect on
    // single-message basis. the receiver loop handles real disconnects).
    AuthState s = auth_state_.load();
    const bool authed = (s == AuthState::AUTH_OK);
    bool always_allowed =
        (msg_type == JLPMessageType::AUTH_OK)   ||
        (msg_type == JLPMessageType::AUTH_FAIL) ||
        (msg_type == JLPMessageType::MSG_ERROR) ||
        (msg_type == JLPMessageType::PING)      ||
        (msg_type == JLPMessageType::PONG);

    if (!authed && !always_allowed) {
        std::cerr << "[Pool] Rejecting message type 0x" << std::hex
                  << (int)header.type << std::dec
                  << " before AUTH_OK (current state="
                  << static_cast<int>(s) << ")" << std::endl;
        return;
    }

    // Per-message dispatch. Each handler owns parsing, validation,
    // side-effects, and callback firing for its message type. MSG_ERROR
    // additionally needs the pre-dispatch AuthState value (captured above
    // as `s`) so it can detect the AUTH_SENT case without re-reading the
    // atomic after our own writes may have moved it.
    switch (msg_type) {
        case JLPMessageType::WORK_ASN:   handle_work_asn(header, payload);            break;
        case JLPMessageType::STATS_RSP:  handle_stats_rsp(header, payload);           break;
        case JLPMessageType::AUTH_OK:    handle_auth_ok(header, payload);             break;
        case JLPMessageType::AUTH_FAIL:  handle_auth_fail(header, payload);           break;
        case JLPMessageType::DP_ACK:     handle_dp_ack(header, payload);              break;
        case JLPMessageType::SOLUTION:   handle_solution(header, payload);            break;
        case JLPMessageType::PING:       handle_ping(header, payload);                break;
        case JLPMessageType::MAINTENANCE: handle_maintenance(header, payload);        break;
        case JLPMessageType::MSG_ERROR:  handle_msg_error(header, payload, s);        break;
        default:                         handle_default_unknown(header, payload);     break;
    }
}

void JLPPoolClient::handle_work_asn(const JLPHeader& /*header*/,
                                    const std::vector<uint8_t>& payload) {
    // Debug output only when enabled
    if (debug_mode_) {
        std::cerr << "[DEBUG] SERVER_WORK payload size: " << payload.size()
                  << " (expected: " << sizeof(JLPServerConfig) << ")" << std::endl;
        std::cerr << "[DEBUG] Raw payload (first 128 bytes): ";
        for (size_t i = 0; i < std::min(payload.size(), (size_t)128); i++) {
            char buf[4];
            snprintf(buf, sizeof(buf), "%02x", payload[i]);
            std::cerr << buf;
            if ((i + 1) % 33 == 0) std::cerr << " | ";
        }
        std::cerr << std::endl;
    }

    // Require an EXACT-size payload (audit pool HIGH-2). A '<' bound silently
    // accepted an oversized WORK_ASN and ignored the trailing bytes, which
    // violates the project's own doctrine that a decoder seeing a longer-than-
    // expected payload must reject it as malformed (a future field-growing
    // WORK_ASN should bump the version, not be silently truncated here).
    if (payload.size() != sizeof(JLPServerConfig)) {
        return;
    }

    const JLPServerConfig* config =
        reinterpret_cast<const JLPServerConfig*>(payload.data());

    if (reject_work_asn_dp_bits(*config)) {
        return;
    }

    // Debug: show parsed public key (only when debug enabled)
    if (debug_mode_) {
        char pk_hex[67];
        ::collider::hex_encode_lower(config->public_key, 33, pk_hex);
        std::cerr << "[DEBUG] Parsed pubkey: " << pk_hex << std::endl;
    }

    WorkAssignment work_copy;
    apply_work_asn_assignment(*config, work_copy);

    std::cout << "[Pool] Received work assignment (ID: " << config->work_id
              << ", DP bits: " << config->dp_bits << ")" << std::endl;

    log_work_asn_milestone(*config, work_copy);

    fire_work_callback(work_copy);
}

bool JLPPoolClient::reject_work_asn_dp_bits(const JLPServerConfig& config) {
    // Validate dp_bits before accepting the assignment. A buggy
    // or malicious server can set dp_bits to a value the
    // kangaroo solver will never satisfy (a distinguished
    // point is X with N leading-zero bits; at 255 leading zeros
    // the probability per step is 2^-255, so the solver
    // produces zero DPs and the GPU burns power forever
    // without making progress).
    // 8..32 is the supportable window: below 8 the DP rate
    // floods the wire (millions per second), and above 32 the
    // expected time between DPs at a few hundred MOps/s is
    // measured in days. Production server-side defaults sit
    // in 14..28 depending on workload bit width. Anything
    // outside the validated window is treated as a protocol
    // violation: log, drop the WORK_ASN, and disconnect so
    // the supervisor can pick a different server.
    constexpr uint32_t kMinDpBits = 8;
    constexpr uint32_t kMaxDpBits = 32;
    if (config.dp_bits >= kMinDpBits && config.dp_bits <= kMaxDpBits) {
        return false;
    }
    std::cerr << "[Pool] Rejecting WORK_ASN with "
              << "dp_bits=" << config.dp_bits
              << " (valid range: "
              << kMinDpBits << ".." << kMaxDpBits
              << "). Disconnecting from server."
              << std::endl;
    // Tear down our side of the connection from inside
    // the receiver thread by flipping the same state
    // bits the natural connection-loss path uses
    // (receiver_loop checks running_ && connected_).
    // The PoolManager supervisor sees !is_connected()
    // on its 500ms probe and decides whether to retry
    // against the same misbehaving server or rotate to
    // another. Cutting the socket here also unblocks
    // any in-progress recv() in receive_message.
    connected_ = false;
    auth_state_.store(AuthState::DISCONNECTED);
    auth_cv_.notify_all();
    dp_cv_.notify_all();
    running_ = false;
    if (socket_ != INVALID_SOCK) {
        // jlp_pool_client.hpp aliases closesocket to
        // POSIX close() on non-Windows builds.
        closesocket(socket_);
        socket_ = INVALID_SOCK;
    }
    return true;
}

void JLPPoolClient::apply_work_asn_assignment(const JLPServerConfig& config,
                                              WorkAssignment& out_work_copy) {
    // snapshot the work assignment under
    // work_mutex_, then delegate to fire_work_callback,
    // which dedups by work_id and copies the callback
    // pointer under callbacks_mutex_ before calling
    // outside any lock. Pre-1.4.1 this was inlined here
    // and only protected by work_mutex_; a concurrent
    // set_work_callback could race with the read.
    // v1.5 type-mismatch epoch race: when the server reassigns this worker
    // to the OTHER half (TAME_ONLY <-> WILD_ONLY), every DP still queued
    // from the prior herd carries the old type byte. Shipping those under
    // the new work_id makes the server drop and penalize us. We detect the
    // change under work_mutex_ and flush our own dp_queue_, then notify
    // PoolManager (which owns the reconnect-window buffer) AFTER releasing
    // every lock. prev_kangaroo_type==0 is the first assignment (no prior
    // herd) and is not treated as a change.
    bool type_changed = false;
    uint8_t prev_kangaroo_type = 0;
    size_t flushed_in_flight = 0;
    {
        std::lock_guard<std::mutex> lock(work_mutex_);
        // Capture the previously-applied kangaroo_type BEFORE the overwrite.
        prev_kangaroo_type = current_work_.kangaroo_type;
        memcpy(current_work_.public_key, config.public_key, 33);
        memcpy(current_work_.range_start, config.range_start, 32);
        memcpy(current_work_.range_end, config.range_end, 32);
        current_work_.dp_bits = config.dp_bits;
        current_work_.work_id = config.work_id;
        // v1.5 asymmetric assignment fields. The backend
        // (CudaRCKangarooBackend::initialize) maps kangaroo_type to a
        // KANG_MODE_* and rejects 0/BOTH in pool mode (server bug if it
        // ever arrives). start_offset_a/_b are propagated for future
        // per-worker chunk-slicing and recorded in the milestone log.
        current_work_.kangaroo_type  = config.kangaroo_type;
        current_work_.start_offset_a = config.start_offset_a;
        current_work_.start_offset_b = config.start_offset_b;
        work_received_ = true;  // chunk ID 0 is valid; don't use work_id as sentinel
        // per-chunk DP sequence resets at the
        // start of every assignment. The server expects
        // sequences to start at 0 and grow monotonically
        // for each (worker, work_id) pair.
        dp_sequence_next_ = 0;
        out_work_copy = current_work_;

        type_changed = (prev_kangaroo_type != 0 &&
                        prev_kangaroo_type != config.kangaroo_type);
        if (type_changed) {
            // dp_queue_ is ours; cleared here under dp_mutex_, taken AFTER
            // work_mutex_ per the documented lock order (work_mutex_ then
            // dp_mutex_).
            std::lock_guard<std::mutex> dlock(dp_mutex_);
            flushed_in_flight = dp_queue_.size();
            dp_queue_.clear();
        }
    }

    if (type_changed) {
        std::cout << "[Pool] INFO kangaroo_type changed "
                  << static_cast<int>(prev_kangaroo_type) << "->"
                  << static_cast<int>(config.kangaroo_type)
                  << "; flushed " << flushed_in_flight
                  << " stale-type in-flight DP(s) for new work_id="
                  << config.work_id << std::endl;

        // Notify PoolManager to flush its reconnect-window buffer. Copy the
        // callback pointer under callbacks_mutex_ and invoke OUTSIDE every
        // other lock (matches fire_work_callback's discipline) so the
        // manager callback can never deadlock against work_mutex_/dp_mutex_.
        KangarooTypeChangedCallback cb_copy;
        {
            std::lock_guard<std::mutex> clock(callbacks_mutex_);
            cb_copy = kangaroo_type_changed_callback_;
        }
        if (cb_copy) {
            cb_copy(prev_kangaroo_type, config.kangaroo_type);
        }
    }
}

void JLPPoolClient::log_work_asn_milestone(const JLPServerConfig& config,
                                           const WorkAssignment& work_copy) {
    // Session log: the work assignment is the central pool-mode
    // event. Reset per-work counters (g_dp_submitted_this_work,
    // g_last_dp_seq) BEFORE the SessionState seed is built so the
    // snapshot reflects the new work_id with a fresh counter
    // baseline; otherwise a same-pid reassignment would carry the
    // prior chunk's count forward.
    g_current_work_id.store(config.work_id, std::memory_order_release);
    g_dp_submitted_this_work.store(0, std::memory_order_release);
    g_last_dp_seq.store(0, std::memory_order_release);

    char range_start_hex[65];
    char range_end_hex[65];
    ::collider::hex_encode_lower(work_copy.range_start, 32, range_start_hex);
    ::collider::hex_encode_lower(work_copy.range_end,   32, range_end_hex);

    // v1.5 asymmetric fields land in the milestone too so operators
    // tailing collider.log can see which half (TAME/WILD) the server
    // assigned this worker; a stuck round-robin would surface as all
    // recent assignments showing the same kangaroo_type.
    const char* type_name = "BOTH(illegal)";
    if (config.kangaroo_type == 1) type_name = "TAME_ONLY";
    else if (config.kangaroo_type == 2) type_name = "WILD_ONLY";
    std::ostringstream detail;
    detail << "work_id=" << config.work_id
           << " dp_bits=" << config.dp_bits
           << " range_start=" << range_start_hex
           << " range_end=" << range_end_hex
           << " kangaroo_type=" << static_cast<int>(config.kangaroo_type)
           << "(" << type_name << ")"
           << " start_offset_a=" << config.start_offset_a
           << " start_offset_b=" << config.start_offset_b;
    ::collider::log::milestone("work_received", detail.str());

    auto seed = build_pool_state_seed(
        host_ + ":" + std::to_string(port_), /*connected=*/true);
    seed.work_dp_bits = static_cast<int>(config.dp_bits);
    seed.work_range_start_hex = range_start_hex;
    seed.work_range_end_hex = range_end_hex;
    seed.work_started_at = std::chrono::system_clock::now();
    ::collider::log::update_session_state(seed);
}

void JLPPoolClient::handle_stats_rsp(const JLPHeader& /*header*/,
                                     const std::vector<uint8_t>& payload) {
    // Server wire format ('<QIIffQI', 36 bytes):
    //   [0..8)    uint64 total_dps        (LE)
    //   [8..12)   uint32 total_workers    (LE)
    //   [12..16)  uint32 active_workers   (LE)
    //   [16..20)  float  dps_per_second   (IEEE-754 LE)
    //   [20..24)  float  your_share       (IEEE-754 LE; fraction 0..1)
    //   [24..32)  uint64 your_dps         (LE; AGGREGATE across all
    //                                       machines using this BTC
    //                                       address as worker name)
    //   [32..36)  uint32 uptime_seconds   (LE)
    if (payload.size() < 36) {
        return;
    }
    std::lock_guard<std::mutex> lock(stats_mutex_);
    memcpy(&stats_.total_dps,       payload.data() +  0, 8);
    memcpy(&stats_.total_workers,   payload.data() +  8, 4);
    memcpy(&stats_.active_workers,  payload.data() + 12, 4);
    memcpy(&stats_.dps_per_second,  payload.data() + 16, 4);
    memcpy(&stats_.your_share,      payload.data() + 20, 4);
    memcpy(&stats_.your_dps,        payload.data() + 24, 8);
    memcpy(&stats_.uptime_seconds,  payload.data() + 32, 4);

    // sanitize wire floats before downstream
    // uint64_t cast (which is UB on NaN / +/-Inf / >2^64).
    sanitize_stats_rsp_floats(stats_.dps_per_second, stats_.your_share);
    // Clamp the wire uint64 totals so a buggy or hostile server can't
    // feed absurd values into the UI panels / JSON exports.
    sanitize_stats_rsp_uints(stats_.your_dps, stats_.total_dps);

    // Legacy aliases for older callers.
    stats_.connected_workers = stats_.active_workers;
    stats_.pool_speed = static_cast<uint64_t>(stats_.dps_per_second);
}

void JLPPoolClient::handle_auth_ok(const JLPHeader& /*header*/,
                                   const std::vector<uint8_t>& payload) {
    std::cout << "[Pool] Authentication successful" << std::endl;
    // the supervisor in PoolManager resets its
    // own backoff and consecutive_failures counters on every
    // successful reconnect. The previously-client-local
    // counters (reconnect_delay_ms_ / reconnect_attempts_ /
    // consecutive_auth_failures_) were deleted because the
    // in-receiver-thread reconnect block they fed was dead code.
    // Session log: AUTH_OK is the milestone the operator cares
    // about for "are we actually working?". worker_name_ holds
    // the payout BTC address (= JLP "worker" identifier).
    ::collider::log::milestone("auth_ok", worker_name_);

    // v1.5.4: parse the AUTH_OK update advert (324-byte AuthOkPayload).
    // A legacy server sends a zero-payload AUTH_OK; treat anything
    // shorter than the advert struct as "no advert" and clear the cached
    // value. A longer-than-expected payload reads the leading 324 bytes
    // (additive growth is reserved via the struct's reserved[] field).
    UpdateAdvert advert;  // present=false by default
    if (payload.size() >= sizeof(jlp_wire::AuthOkPayload)) {
        jlp_wire::AuthOkPayload p;
        std::memcpy(&p, payload.data(), sizeof(p));

        // Helper: read a null-padded ASCII field into a std::string,
        // stopping at the first NUL (or the field end).
        const auto read_cstr = [](const uint8_t* buf, size_t cap) {
            size_t n = 0;
            while (n < cap && buf[n] != 0) ++n;
            return std::string(reinterpret_cast<const char*>(buf), n);
        };

        advert.present          = true;
        advert.latest_version   = read_cstr(p.latest_version,
                                            sizeof(p.latest_version));
        advert.min_version      = read_cstr(p.min_version,
                                            sizeof(p.min_version));
        advert.download_url     = read_cstr(p.download_url,
                                            sizeof(p.download_url));
        std::memcpy(advert.sha256.data(), p.sha256, advert.sha256.size());
        advert.update_available   = (p.flags & 0x01) != 0;
        advert.maintenance_active = (p.flags & 0x02) != 0;
    }

    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        update_advert_ = advert;
    }

    if (advert.present) {
        std::string detail =
            "latest=" + advert.latest_version +
            " min=" + advert.min_version +
            " update=" + (advert.update_available ? "1" : "0") +
            " maint=" + (advert.maintenance_active ? "1" : "0") +
            " url_present=" + (advert.download_url.empty() ? "0" : "1");
        ::collider::log::milestone("authok_advert", detail);
    }

    // Publish AUTH_OK only AFTER the advert is parsed and stored. The
    // release-store here pairs with the acquire-load in the authenticate()
    // waiter, so any consumer that observes AUTH_OK (e.g. pool_solver
    // reading get_update_advert() right after connect() returns) is
    // guaranteed to see the fully-written advert. Storing/notifying before
    // the parse let the consumer read a stale empty advert and silently skip
    // a self-update the server had advertised.
    auth_state_.store(AuthState::AUTH_OK);
    auth_cv_.notify_all();
}

JLPPoolClient::UpdateAdvert JLPPoolClient::get_update_advert() const {
    std::lock_guard<std::mutex> lock(callbacks_mutex_);
    return update_advert_;
}

void JLPPoolClient::handle_maintenance(const JLPHeader& /*header*/,
                                       const std::vector<uint8_t>& payload) {
    // Require the full MaintenancePayload (262 bytes). Reject anything
    // shorter as malformed (do not act on a partial frame).
    if (payload.size() < sizeof(jlp_wire::MaintenancePayload)) {
        return;
    }
    jlp_wire::MaintenancePayload p;
    std::memcpy(&p, payload.data(), sizeof(p));

    const bool active = (p.active != 0);
    const uint32_t retry = p.retry_after_secs;

    // Decode the operator note as printable ASCII, stopping at the first
    // NUL and replacing control characters (same anti-injection approach
    // as handle_auth_fail / handle_msg_error).
    std::string message;
    message.reserve(sizeof(p.message));
    for (size_t i = 0; i < sizeof(p.message) && p.message[i] != 0; ++i) {
        unsigned char c = p.message[i];
        message.push_back((c >= 0x20 && c < 0x7F) ? static_cast<char>(c) : '.');
    }

    std::string detail =
        "active=" + std::string(active ? "1" : "0") +
        " retry=" + std::to_string(retry) +
        " msg=" + message;
    ::collider::log::milestone("pool_maintenance", detail);

    // Copy the callback out under the lock, fire it OUTSIDE the lock so
    // the PoolManager hook (which takes its own mutexes) can never
    // deadlock against callbacks_mutex_. Mirrors the kangaroo-type and
    // solution callback patterns.
    MaintenanceCallback cb_copy;
    {
        std::lock_guard<std::mutex> lock(callbacks_mutex_);
        cb_copy = maintenance_callback_;
    }
    if (cb_copy) {
        cb_copy(active, retry, message);
    }
}

void JLPPoolClient::handle_auth_fail(const JLPHeader& /*header*/,
                                     const std::vector<uint8_t>& payload) {
    // Decode payload as printable ASCII (same safe approach as handle_msg_error:
    // cap length and replace control characters to prevent ANSI injection).
    constexpr size_t MAX_PRINT = 256;
    std::string reason;
    reason.reserve(std::min(payload.size(), MAX_PRINT));
    for (size_t i = 0; i < payload.size() && i < MAX_PRINT; ++i) {
        unsigned char c = payload[i];
        reason.push_back((c >= 0x20 && c < 0x7F) ? static_cast<char>(c) : '.');
    }

    if (reason.rfind("IP banned:", 0) == 0) {
        ip_banned_  = true;
        ban_reason_ = reason;
        std::cerr << "[Pool] Your IP is BANNED by the pool server.\n"
                  << "       Reason: " << reason << "\n"
                  << "       Wait for the ban to expire, then restart."
                  << std::endl;
        ::collider::log::milestone("auth_fail", "ip_banned: " + reason);
    } else if (reason.find("upgrade required: v1.5 asymmetric protocol")
               != std::string::npos) {
        // v1.5 pool servers reject v1.4.x clients to lock out the
        // theft-vulnerable code path. Server's verbatim reason string
        // (pinned by pool-server-engineer as
        // KangarooPoolServer.UPGRADE_REQUIRED_REASON):
        //   "upgrade required: v1.5 asymmetric protocol; v1.4.x clients
        //    are theft-vulnerable and refused"
        // Match-string and render a clear operator-facing message
        // rather than a generic auth failure.
        std::cerr << "[Pool] UPGRADE REQUIRED -- the pool server rejected "
                     "this client because v1.4.x is theft-vulnerable.\n"
                  << "[Pool] Download the v1.5+ binary from "
                     "https://collisionprotocol.com/download and re-run."
                  << std::endl;
        ::collider::log::milestone("auth_fail", "upgrade_required: " + reason);
    } else {
        std::cerr << "[Pool] Authentication rejected by server";
        if (!reason.empty()) std::cerr << ": " << reason;
        std::cerr << std::endl;
        ::collider::log::milestone("auth_fail", reason.empty()
                                       ? "server rejected credentials"
                                       : reason);
    }

    auth_state_.store(AuthState::AUTH_FAILED);
    auth_cv_.notify_all();
    connected_ = false;
}

void JLPPoolClient::handle_dp_ack(const JLPHeader& /*header*/,
                                  const std::vector<uint8_t>& /*payload*/) {
    // DPs were received by server. nothing to do.
}

void JLPPoolClient::handle_solution(const JLPHeader& /*header*/,
                                    const std::vector<uint8_t>& payload) {
    // v1.5: SOLUTION is now strictly a server-to-client BROADCAST.
    // The pool server (collision-protocol's collision_detector) is the
    // sole entity that sees DPs of both tame and wild type and the
    // sole entity that computes the recovered private key. When the
    // server detects a collision and finishes its own automated sweep
    // of the puzzle funds, it broadcasts SOLUTION to all connected
    // workers as a "stop grinding, the pool solved this chunk" signal.
    //
    // The 32-byte payload of this message MAY carry the recovered
    // private key bytes (server's choice for transparency / audit), but
    // the v1.5 worker DOES NOT STORE THEM:
    //   - No recovered_keys/<ts>.json file is ever written (that surface
    //     was deleted).
    //   - The bytes are not echoed to stdout/stderr in a copy-paste
    //     friendly form.
    //   - The bytes are passed through fire_solution_callback purely so
    //     dedup against retransmits stays robust. The caller (PoolManager
    //     -> run_pool_mode) is responsible for treating the bytes as
    //     stop-signal metadata only and triggering shutdown.
    //
    // See .claude/tasks/v1.5-asymmetric-kangaroo.md.
    std::cout << "[Pool] SOLUTION FOUND BY POOL -- work stopped."
              << std::endl;
    ::collider::log::milestone(
        "solution_received",
        "work_id=" +
            std::to_string(g_current_work_id.load(std::memory_order_acquire)));
    if (payload.size() >= 32) {
        fire_solution_callback(payload.data());
    }
}

void JLPPoolClient::handle_ping(const JLPHeader& /*header*/,
                                const std::vector<uint8_t>& /*payload*/) {
    // Server ping. respond with pong.
    send_message(JLPMessageType::PONG, nullptr, 0);
}

void JLPPoolClient::handle_msg_error(const JLPHeader& /*header*/,
                                     const std::vector<uint8_t>& payload,
                                     AuthState pre_dispatch_state) {
    // cap printed length and strip control characters so a
    // malicious server cannot inject ANSI escape sequences or megabytes
    // of attacker-controlled text into operator logs.
    constexpr size_t MAX_PRINT = 256;
    std::string error;
    error.reserve(std::min(payload.size(), MAX_PRINT));
    for (size_t i = 0; i < payload.size() && i < MAX_PRINT; ++i) {
        unsigned char c = payload[i];
        // Allow printable ASCII; replace controls (incl. ESC=0x1B) with '.'
        error.push_back((c >= 0x20 && c < 0x7F) ? static_cast<char>(c) : '.');
    }
    std::cerr << "[Pool] Server error: " << error;
    if (payload.size() > MAX_PRINT) {
        std::cerr << "...(truncated, " << payload.size() << " bytes total)";
    }
    std::cerr << std::endl;

    // If we receive an error before AUTH completes, treat it as auth
    // failure so authenticate() returns instead of timing out. We use
    // the AuthState captured by the dispatcher BEFORE any handler ran,
    // so a same-message handler reordering cannot change the decision.
    if (pre_dispatch_state == AuthState::AUTH_SENT) {
        auth_state_.store(AuthState::AUTH_FAILED);
        auth_cv_.notify_all();
    }
}

void JLPPoolClient::handle_default_unknown(const JLPHeader& header,
                                           const std::vector<uint8_t>& /*payload*/) {
    // PONG falls through to here: no client-side reaction is required.
    // For genuinely unknown message types, log only when debug is on so a
    // protocol version skew does not spam operator logs.
    if (debug_mode_) {
        std::cerr << "[DEBUG] Unknown message type: 0x"
                  << std::hex << (int)header.type << std::dec << std::endl;
    }
}

// DistinguishedPoint serialization
std::vector<uint8_t> DistinguishedPoint::serialize() const {
    std::vector<uint8_t> data(65 + 8);
    memcpy(data.data(), x, 32);
    memcpy(data.data() + 32, d, 32);
    data[64] = type;
    memcpy(data.data() + 65, &dp_bits, 8);
    return data;
}

DistinguishedPoint DistinguishedPoint::deserialize(const uint8_t* data, size_t len) {
    DistinguishedPoint dp;
    if (len >= 65) {
        memcpy(dp.x, data, 32);
        memcpy(dp.d, data + 32, 32);
        dp.type = data[64];
        if (len >= 73) {
            memcpy(&dp.dp_bits, data + 65, 8);
        }
    }
    return dp;
}

// Factory function
std::unique_ptr<PoolClient> create_pool_client(const std::string& type) {
    if (type == POOL_TYPE_JLP) {
        return std::make_unique<JLPPoolClient>();
    }
    // HTTP pool path has been removed; only JLP remains.
    return nullptr;
}

} // namespace pool
} // namespace collider
