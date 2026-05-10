// jlp_pool_client.cpp - JeanLucPons Kangaroo protocol implementation
//
// Wave 4 (2026-05-04 security review) hardening summary:
//   D-H1  TLS hostname verification + SNI + default trust store         init_tls()
//   D-H2  verify_cert default flipped to true (in header)
//   D-H4  AuthState machine gates work-affecting messages on AUTH_OK    handle_server_message()
//   D-H5  Bounded reconnect after AUTH_FAIL + jittered backoff          receiver_loop()
//   D-M2  ssl_write_mutex_ / ssl_read_mutex_ split (concurrent r/w)     send_message() / receive_message()
//   D-M5  authenticate() actually waits for AUTH_OK / AUTH_FAIL         authenticate()
//   B-LOW-6 thread::operator= safety on reconnect                       replace_thread()
//   B-MED-1 pool_speed type fixed to uint64_t (in pool_client.hpp)      handle_server_message() STATS_RSP
//
// References:
//   - RFC 6125 ("Representation and Verification of Domain-Based Application
//     Service Identity within Internet PKI Using X.509 Certs")
//   - RFC 6066 sec 3 (Server Name Indication TLS extension)
//   - OpenSSL 1.1.1+ docs: SSL_set_tlsext_host_name(3),
//     X509_VERIFY_PARAM_set1_host(3), SSL_CTX_set_default_verify_paths(3)
//   - C++23 [thread.thread.assign]/p2 (assigning to joinable thread terminates)

#include "jlp_pool_client.hpp"
#include "../core/byte_codec.hpp"
#include <cstring>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <mutex>  // for std::once_flag, std::call_once

#ifdef COLLIDER_HAS_OPENSSL
#include <openssl/x509v3.h>
#endif

// Wave 4 followup (2026-05-04): Windows TLS verification requires bridging
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

    // Wave 4 D-H1: load the platform default CA trust store so chain
    // verification can find a trust anchor (Let's Encrypt etc.). Without this,
    // SSL_VERIFY_PEER fails on any cert whose root the program does not know.
    // See OpenSSL: SSL_CTX_set_default_verify_paths(3).
    //
    // Wave 4 D-H1 followup (2026-05-04): on Windows, OpenSSL's default verify
    // paths are useless (no usable system bundle), so we additionally bridge
    // the OS "ROOT" cert store into OpenSSL's X509_STORE via wincrypt. This
    // is the same bridge used by tests/test_jlp_pool_handshake.cpp.
    if (verify_cert_) {
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
                                 "0 certs; TLS verification will fail." << std::endl;
                }
            } else {
                std::cerr << "[Pool] Warning: CertOpenSystemStoreA(ROOT) failed (err="
                          << (unsigned long)GetLastError()
                          << "); falling back to OpenSSL default verify paths "
                             "(likely empty on Windows)." << std::endl;
                if (SSL_CTX_set_default_verify_paths(ssl_ctx_) != 1) {
                    std::cerr << "[Pool] Warning: SSL_CTX_set_default_verify_paths "
                                 "also failed; TLS verification will fail." << std::endl;
                }
            }
        }
#else
        if (SSL_CTX_set_default_verify_paths(ssl_ctx_) != 1) {
            // Non-fatal but worth surfacing - hostname verification will
            // probably fail without trust anchors loaded.
            std::cerr << "[Pool] Warning: SSL_CTX_set_default_verify_paths "
                         "failed (system trust store unavailable). TLS "
                         "verification will likely fail." << std::endl;
        }
#endif
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

    // Wave 4 D-H1: SNI + hostname verification. These are SSL-level (not CTX)
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
    if (!sockets_initialized_) {
        WSADATA wsa_data;
        if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
            return false;
        }
        sockets_initialized_ = true;
    }
#endif
    return true;
}

void JLPPoolClient::cleanup_sockets() {
#ifdef _WIN32
    if (sockets_initialized_) {
        WSACleanup();
        sockets_initialized_ = false;
    }
#endif
}

// Wave 4 B-LOW-6: safely re-assign a std::thread. std::thread::operator=
// invokes std::terminate if the LHS is joinable. Always join (or detach) the
// previous thread before overwriting.
//
// IMPORTANT: do not call this from inside the thread that `t` represents - that
// would self-join and deadlock. The receiver path that calls this for
// receiver_thread_ must therefore happen from a different thread. Currently
// only disconnect() (caller thread) re-creates these objects.
void JLPPoolClient::replace_thread(std::thread& t, std::thread new_thread) {
    if (t.joinable()) {
        // Last-ditch safety. In well-formed code the caller has already signaled
        // running_=false and the old thread has exited, so this is fast.
        try {
            t.join();
        } catch (const std::system_error& e) {
            std::cerr << "[Pool] thread join failed in replace_thread: "
                      << e.what() << std::endl;
            // Detach so destructor doesn't terminate.
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
    , gpu_count_(1)
    , speed_(0)
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

    // Resolve hostname
    struct addrinfo hints, *result;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &result) != 0) {
        std::cerr << "[Pool] Failed to resolve hostname: " << host << std::endl;
        auth_state_.store(AuthState::DISCONNECTED);
        return false;
    }

    // Create socket
    socket_ = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (socket_ == INVALID_SOCK) {
        freeaddrinfo(result);
        std::cerr << "[Pool] Failed to create socket" << std::endl;
        auth_state_.store(AuthState::DISCONNECTED);
        return false;
    }

    // Set timeout
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

    // Connect
    if (::connect(socket_, result->ai_addr, (int)result->ai_addrlen) == SOCK_ERROR) {
        freeaddrinfo(result);
        closesocket(socket_);
        socket_ = INVALID_SOCK;
        std::cerr << "[Pool] Failed to connect to " << host << ":" << port << std::endl;
        auth_state_.store(AuthState::DISCONNECTED);
        return false;
    }

    freeaddrinfo(result);

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
    if (use_tls_) std::cout << " (TLS)";
    std::cout << std::endl;

    // Wave 4 B-LOW-6: safely (re)assign the thread objects. If a previous
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
    running_ = false;
    connected_ = false;
    auth_state_.store(AuthState::DISCONNECTED);

    // Wake up sender thread and any authenticate() waiters.
    dp_cv_.notify_all();
    auth_cv_.notify_all();

    // Close the raw socket FIRST. This unblocks any in-progress SSL_read /
    // recv in the receiver thread (causing it to return an error and exit),
    // and wakes the sender thread from any blocking write. We must NOT call
    // cleanup_tls() yet: the receiver may still be inside ssl_recv() using
    // ssl_. Calling SSL_free() underneath it causes heap corruption
    // ("pointer being freed was not allocated" on macOS).
    if (socket_ != INVALID_SOCK) {
        closesocket(socket_);
        socket_ = INVALID_SOCK;
    }

    // Join both threads BEFORE touching ssl_. Once they have exited, no
    // thread is using ssl_ and it is safe to free it.
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

    // Now safe to free TLS objects -- both I/O threads have exited.
#ifdef COLLIDER_HAS_OPENSSL
    if (use_tls_) {
        cleanup_tls();
    }
#endif

    std::cout << "[Pool] Disconnected" << std::endl;
}

bool JLPPoolClient::is_connected() const {
    return connected_;
}

bool JLPPoolClient::authenticate(const std::string& worker_name,
                                 const std::string& password) {
    worker_name_ = worker_name;

    // Wave 4 D-M5 / I: the JLP wire ClientHello has no password field. Warn
    // loudly if the operator passed one so they don't believe they are
    // authenticated by password when in fact only their worker name is sent.
    if (!password.empty()) {
        static std::once_flag pw_warn_once;
        std::call_once(pw_warn_once, []() {
            std::cerr << "[Pool] WARNING: --pool-password was set but JLP "
                         "protocol authenticates by worker name only. The "
                         "password is being IGNORED. Remove it to silence this "
                         "warning, or use a JLP server that supports the AUTH "
                         "flags reserved for future credential payloads."
                      << std::endl;
        });
    }

    // Move to AUTH_SENT before transmitting so the receiver can correctly
    // accept AUTH_OK / AUTH_FAIL when they arrive.
    auth_state_.store(AuthState::AUTH_SENT);

    if (!send_hello()) {
        auth_state_.store(AuthState::DISCONNECTED);
        return false;
    }

    // Wave 4 D-M5: actually wait for AUTH_OK / AUTH_FAIL / MSG_ERROR or timeout.
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
    JLPClientHello hello;
    memset(&hello, 0, sizeof(hello));
    // Wave 4 D-M6: validate worker_name length to avoid silent truncation.
    // Bitcoin Bech32m addresses can run up to ~90 chars; if the user passed a
    // longer one, refuse rather than silently mangle it (rewards go nowhere).
    if (worker_name_.size() >= sizeof(hello.worker_name)) {
        std::cerr << "[Pool] Worker name too long: " << worker_name_.size()
                  << " bytes (max " << (sizeof(hello.worker_name) - 1)
                  << "). Refusing to send truncated identity." << std::endl;
        return false;
    }
    // memset above zero-fills, so the trailing NUL is guaranteed even though
    // strncpy wouldn't write it for max-length input.
    std::memcpy(hello.worker_name, worker_name_.data(), worker_name_.size());
    hello.gpu_count = gpu_count_;
    hello.speed = speed_;

    return send_message(JLPMessageType::AUTH, &hello, sizeof(hello));
}

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

    dp_queue_.push(dp);
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
        dp_queue_.push(dp);
    }
    dp_cv_.notify_one();
    return true;
}

PoolStats JLPPoolClient::get_stats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool JLPPoolClient::report_solution(const uint8_t* private_key) {
    return send_message(JLPMessageType::SOLUTION, private_key, 32);
}

void JLPPoolClient::set_solution_callback(SolutionCallback cb) {
    solution_callback_ = cb;
}

void JLPPoolClient::set_work_callback(WorkCallback cb) {
    work_callback_ = cb;
}

bool JLPPoolClient::send_message(JLPMessageType type, const void* data, size_t size) {
    if (!connected_ || socket_ == INVALID_SOCK) {
        return false;
    }

    // New JLP protocol format: [MAGIC:4][TYPE:1][FLAGS:1][LENGTH:2] = 8 bytes
    JLPHeader header;
    header.magic[0] = 'K';
    header.magic[1] = 'A';
    header.magic[2] = 'N';
    header.magic[3] = 'G';
    header.type = static_cast<uint8_t>(type);
    header.flags = 0;  // No flags
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

    int received;

    // Serialize only against other readers (only the receiver_loop reads
    // in this design, so this is defensive). Crucially, we do NOT share a
    // mutex with the writer side: the receiver may block here for minutes
    // waiting on the next server frame, and the main thread must remain
    // free to send WORK_REQ / DP_BATCH during that wait.
    //
    // We hold the lock around BOTH the header read and the payload read so
    // a future second reader could not splice frames; the SSL object
    // tolerates concurrent SSL_read and SSL_write per OpenSSL's threading
    // contract.
    std::unique_lock<std::mutex> io_lock(ssl_read_mutex_);

    // Receive header
#ifdef COLLIDER_HAS_OPENSSL
    if (use_tls_ && ssl_) {
        received = ssl_recv(&header, sizeof(header));
        if (received != sizeof(header)) {
            int ssl_err = SSL_get_error(ssl_, received);
            if (ssl_err == SSL_ERROR_WANT_READ || ssl_err == SSL_ERROR_WANT_WRITE) {
                last_receive_was_timeout_ = true;
                return false;
            }
            // SSL_ERROR_SYSCALL with a recoverable system errno is also a
            // "no data, try again" -- not a fatal connection loss.
            if (ssl_err == SSL_ERROR_SYSCALL) {
#ifdef _WIN32
                int sock_err = WSAGetLastError();
                if (sock_err == WSAETIMEDOUT || sock_err == WSAEWOULDBLOCK) {
                    last_receive_was_timeout_ = true;
                    return false;
                }
#else
                if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ETIMEDOUT) {
                    last_receive_was_timeout_ = true;
                    return false;
                }
#endif
            }
            last_receive_was_timeout_ = false;
            return false;
        }
    } else
#endif
    {
        received = recv(socket_, (char*)&header, sizeof(header), MSG_WAITALL);
        if (received != sizeof(header)) {
            // Check if it's a timeout vs actual disconnect
#ifdef _WIN32
            int err = WSAGetLastError();
            if (err == WSAETIMEDOUT || err == WSAEWOULDBLOCK) {
                last_receive_was_timeout_ = true;
                return false;
            }
#else
            if (errno == EAGAIN || errno == EWOULDBLOCK || errno == ETIMEDOUT) {
                last_receive_was_timeout_ = true;
                return false;
            }
#endif
            last_receive_was_timeout_ = false;
            return false;
        }
    }
    last_receive_was_timeout_ = false;

    // Validate magic
    if (header.magic[0] != 'K' || header.magic[1] != 'A' ||
        header.magic[2] != 'N' || header.magic[3] != 'G') {
        std::cerr << "[Pool] Invalid message magic" << std::endl;
        return false;
    }

    // New JLP protocol has no version field - FLAGS field is at byte 5
    // No version validation needed

    // Validate payload size (max 64KB with uint16_t, but cap at reasonable limit)
    constexpr uint16_t MAX_PAYLOAD_SIZE = 65000;
    if (header.payload_size > MAX_PAYLOAD_SIZE) {
        std::cerr << "[Pool] Payload size exceeds limit: " << header.payload_size << " bytes" << std::endl;
        return false;
    }

    // Receive payload
    if (header.payload_size > 0) {
        payload.resize(header.payload_size);
#ifdef COLLIDER_HAS_OPENSSL
        if (use_tls_ && ssl_) {
            received = ssl_recv(payload.data(), header.payload_size);
        } else
#endif
        {
            received = recv(socket_, (char*)payload.data(), header.payload_size, MSG_WAITALL);
        }
        if (received != (int)header.payload_size) {
            return false;
        }
    }

    return true;
}

void JLPPoolClient::receiver_loop() {
    // Wave 4 D-H5: jittered exponential backoff. Use a thread-local RNG so we
    // don't synchronize a fleet of workers (thundering herd against the pool).
    std::mt19937 rng{std::random_device{}()};
    auto jitter_delay = [&](uint32_t base_ms) -> uint32_t {
        std::uniform_real_distribution<double> jitter(0.75, 1.25);  // +/-25%
        return static_cast<uint32_t>(base_ms * jitter(rng));
    };

    while (running_ && connected_) {
        JLPHeader header;
        std::vector<uint8_t> payload;

        if (receive_message(header, payload)) {
            handle_server_message(header, payload);
        } else if (last_receive_was_timeout_) {
            // Just a timeout, not a real disconnect - continue waiting
            // This is normal when server has nothing to send
            continue;
        } else if (connected_ && auto_reconnect_) {
            // Wave 4 D-L4: clean up TLS BEFORE closing the underlying socket
            // so we don't leak SSL_CTX/SSL on reconnect.
#ifdef COLLIDER_HAS_OPENSSL
            if (use_tls_) {
                cleanup_tls();
            }
#endif
            // Actual connection loss - try to reconnect with exponential backoff
            closesocket(socket_);
            socket_ = INVALID_SOCK;
            connected_ = false;
            auth_state_.store(AuthState::DISCONNECTED);
            // Wake any thread waiting in authenticate() so it can return
            // promptly with the disconnected state instead of timing out.
            auth_cv_.notify_all();

            while (running_ && auto_reconnect_) {
                // Wave 4 D-H5: enforce a hard cap on consecutive AUTH_FAILs.
                // Without this, a stale credential would keep retrying forever
                // (1, 2, 4, ... 60 seconds, looking like cred-stuffing to the
                // pool's WAF).
                if (consecutive_auth_failures_ >= MAX_AUTH_FAIL_ATTEMPTS) {
                    std::cerr << "[Pool] Reached " << MAX_AUTH_FAIL_ATTEMPTS
                              << " consecutive AUTH_FAILs; giving up. "
                              << "Check worker name / credentials and reconnect "
                                 "manually." << std::endl;
                    running_ = false;
                    return;
                }

                reconnect_attempts_++;
                uint32_t delay = jitter_delay(reconnect_delay_ms_);
                std::cerr << "[Pool] Connection lost, reconnect attempt "
                          << reconnect_attempts_
                          << " in " << (delay / 1000.0) << "s..." << std::endl;

                std::this_thread::sleep_for(std::chrono::milliseconds(delay));

                if (!running_) break;

                // NOTE: we are inside the receiver thread. connect() restarts
                // the receiver thread via replace_thread() - which would
                // self-join us and std::terminate. Instead, do the reconnect
                // work inline here:
                //   1. Reopen socket + TLS (cannot call connect() helper).
                //   2. Re-authenticate (will set worker_name_ from prev value).
                //
                // Because reopening involves DNS + TLS handshake which is
                // non-trivial code already in connect(), we defer the actual
                // connect to a brief escape: set a flag and let the supervisor
                // handle it. For now, exit the receiver loop and let
                // pool_manager's caller drive reconnection externally.
                //
                // Simpler and safer: stop trying from inside the receiver
                // thread. The pool manager (or main loop) can detect
                // !is_connected() and call connect()+authenticate() from a
                // safe thread context.
                std::cerr << "[Pool] Receiver exiting; external supervisor "
                             "must call connect()/authenticate() to recover."
                          << std::endl;
                running_ = false;
                return;
            }
        }
    }
}

void JLPPoolClient::sender_loop() {
    // v2 wire format: every DP carries the worker's claimed work_id so the
    // server can match it to the chunk currently assigned to this worker.
    // The server still accepts v1 submissions from older clients deployed
    // in the field; freshly-built binaries always emit v2.
    std::vector<JLPDistinguishedPointV2> batch;
    batch.reserve(100);
    // Hoisted out of the per-iteration body (Gemini PR review): payload
    // is the staging buffer for DP_BATCH_V2 emission. Reusing it across
    // iterations avoids re-allocating on every drain.
    std::vector<uint8_t> payload;
    payload.reserve(4 + 100 * sizeof(JLPDistinguishedPointV2));

    // Keepalive: the server's per-message read timeout is 30s. With
    // dp_bits=35, a single CPU worker may take many minutes to find a DP,
    // so without periodic traffic the server disconnects us before we
    // ever produce one. Send a PING every KEEPALIVE_SECONDS to reset the
    // server's read timeout. Counts any send_message() call as activity.
    constexpr auto KEEPALIVE_SECONDS = std::chrono::seconds(20);
    auto last_send = std::chrono::steady_clock::now();
    // Periodically poll the pool for aggregated stats (your_dps across
    // ALL machines sharing this worker name, total pool DPs, etc.). Drives
    // the "Your: X / Pool: Y (Z%)" display in run_pool_mode. Cheap: empty
    // STATS_REQ + 36-byte STATS_RSP.
    constexpr auto STATS_INTERVAL_SECONDS = std::chrono::seconds(10);
    auto last_stats_req = std::chrono::steady_clock::now()
                          - STATS_INTERVAL_SECONDS;  // fire once on first iteration

    while (running_) {
        // 1. Wait for queue activity (releases dp_mutex_ during the wait).
        {
            std::unique_lock<std::mutex> lock(dp_mutex_);
            dp_cv_.wait_for(lock, std::chrono::milliseconds(100), [this] {
                return !dp_queue_.empty() || !running_;
            });
        }

        // 2. Snapshot the current work_id AFTER the wait (Gemini PR
        //    review): a pre-wait snapshot becomes stale if WORK_ASN
        //    arrives during the 100ms wait, leading to DPs being tagged
        //    with the previous chunk's work_id and rejected by the
        //    server. Lock order: PoolManager::dp_callback_hook acquires
        //    (work_mutex_ then dp_mutex_); we mirror that order here to
        //    avoid a deadlock cycle.
        uint64_t current_work_id = 0;
        {
            std::lock_guard<std::mutex> wlock(work_mutex_);
            current_work_id = current_work_.work_id;
        }

        // 3. Drain the queue under dp_mutex_ alone.
        {
            std::lock_guard<std::mutex> lock(dp_mutex_);
            while (!dp_queue_.empty() && batch.size() < 100) {
                const auto& dp = dp_queue_.front();
                JLPDistinguishedPointV2 jlp_dp;
                // Wire is little-endian little-end on x86/ARM-LE; convert
                // explicitly so this remains correct on big-endian builds
                // (Gemini PR review). Manual byte assembly avoids the
                // missing-htole64 portability headache (Windows lacks it).
                {
                    uint8_t* p = reinterpret_cast<uint8_t*>(&jlp_dp.work_id);
                    for (int i = 0; i < 8; ++i) {
                        p[i] = static_cast<uint8_t>((current_work_id >> (8 * i)) & 0xFFu);
                    }
                }
                memcpy(jlp_dp.x, dp.x, 32);
                memcpy(jlp_dp.d, dp.d, 32);
                jlp_dp.type = dp.type;
                jlp_dp.dp_bits = static_cast<uint8_t>(dp.dp_bits & 0xFF);
                batch.push_back(jlp_dp);
                dp_queue_.pop();
            }
        }

        // Send batch (only after AUTH_OK; before then the server will reject
        // these and we'd be sending plaintext DPs to an unauthenticated channel).
        if (!batch.empty() && connected_
            && auth_state_.load() == AuthState::AUTH_OK) {
            // Wire format expected by the pool: [count:u32 LE][dp1:74][dp2:74]...
            // (DP_BATCH_V2 uses 74-byte v2 entries; the leading u32 count is
            // capped at 10000 server-side, so without it the first 4 bytes of
            // the first DP get interpreted as the count and the server tears
            // down the connection with "Batch count N exceeds max 10000".)
            payload.clear();
            uint32_t count = static_cast<uint32_t>(batch.size());
            // Explicit little-endian count (Gemini PR review).
            payload.push_back(static_cast<uint8_t>( count        & 0xFFu));
            payload.push_back(static_cast<uint8_t>((count >>  8) & 0xFFu));
            payload.push_back(static_cast<uint8_t>((count >> 16) & 0xFFu));
            payload.push_back(static_cast<uint8_t>((count >> 24) & 0xFFu));
            payload.insert(payload.end(),
                           reinterpret_cast<uint8_t*>(batch.data()),
                           reinterpret_cast<uint8_t*>(batch.data())
                               + batch.size() * sizeof(JLPDistinguishedPointV2));
            send_message(JLPMessageType::DP_BATCH_V2, payload.data(), payload.size());
            last_send = std::chrono::steady_clock::now();

            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.your_dps += batch.size();

            batch.clear();
        } else if (!batch.empty()) {
            // Drop batch if not authenticated. Don't accumulate forever.
            batch.clear();
        }

        // Periodic STATS_REQ to refresh server-aggregated stats (your_dps
        // across all machines using this worker name). Both this and the
        // PING below are gated on AUTH_OK so we don't talk to a server
        // that hasn't accepted us yet.
        if (connected_ && auth_state_.load() == AuthState::AUTH_OK) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_stats_req >= STATS_INTERVAL_SECONDS) {
                send_message(JLPMessageType::STATS_REQ, nullptr, 0);
                last_stats_req = now;
                last_send = now;  // STATS_REQ is also activity for keepalive purposes
            }
        }

        // Periodic keepalive PING. Skipped if STATS_REQ already counted as
        // activity within the keepalive window.
        if (connected_ && auth_state_.load() == AuthState::AUTH_OK) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_send >= KEEPALIVE_SECONDS) {
                send_message(JLPMessageType::PING, nullptr, 0);
                last_send = now;
            }
        }
    }
}

void JLPPoolClient::handle_server_message(const JLPHeader& header,
                                          const std::vector<uint8_t>& payload) {
    JLPMessageType msg_type = static_cast<JLPMessageType>(header.type);

    // Wave 4 D-H4: gate work-affecting messages on AUTH_OK. A malicious or
    // misbehaving pool MUST NOT be able to inject WORK_ASN / SOLUTION / DP_ACK
    // / STATS_RSP before authentication completes.
    //
    // Allowed pre-AUTH_OK: AUTH_OK, AUTH_FAIL, MSG_ERROR, PING/PONG.
    // Anything else: log + ignore (do NOT crash, do NOT disconnect on
    // single-message basis -- the receiver loop handles real disconnects).
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

    switch (msg_type) {
        case JLPMessageType::WORK_ASN: {  // Work assignment from server
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

            if (payload.size() >= sizeof(JLPServerConfig)) {
                const JLPServerConfig* config =
                    reinterpret_cast<const JLPServerConfig*>(payload.data());

                // Debug: show parsed public key (only when debug enabled)
                if (debug_mode_) {
                    char pk_hex[67];
                    ::collider::hex_encode_lower(config->public_key, 33, pk_hex);
                    std::cerr << "[DEBUG] Parsed pubkey: " << pk_hex << std::endl;
                }

                // Copy work data and callback pointer INSIDE lock, then call OUTSIDE
                // This prevents deadlock if callback tries to acquire any lock
                WorkAssignment work_copy;
                WorkCallback cb_copy;
                {
                    std::lock_guard<std::mutex> lock(work_mutex_);
                    memcpy(current_work_.public_key, config->public_key, 33);
                    memcpy(current_work_.range_start, config->range_start, 32);
                    memcpy(current_work_.range_end, config->range_end, 32);
                    current_work_.dp_bits = config->dp_bits;
                    current_work_.work_id = config->work_id;
                    work_received_ = true;  // chunk ID 0 is valid; don't use work_id as sentinel
                    work_copy = current_work_;
                    cb_copy = work_callback_;
                }

                std::cout << "[Pool] Received work assignment (ID: " << config->work_id
                          << ", DP bits: " << config->dp_bits << ")" << std::endl;

                // Call callback OUTSIDE the lock to prevent deadlock
                if (cb_copy) {
                    cb_copy(work_copy);
                }
            }
            break;
        }

        case JLPMessageType::STATS_RSP: {  // Statistics response
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
            if (payload.size() >= 36) {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                memcpy(&stats_.total_dps,       payload.data() +  0, 8);
                memcpy(&stats_.total_workers,   payload.data() +  8, 4);
                memcpy(&stats_.active_workers,  payload.data() + 12, 4);
                memcpy(&stats_.dps_per_second,  payload.data() + 16, 4);
                memcpy(&stats_.your_share,      payload.data() + 20, 4);
                memcpy(&stats_.your_dps,        payload.data() + 24, 8);
                memcpy(&stats_.uptime_seconds,  payload.data() + 32, 4);
                // Legacy aliases for older callers.
                stats_.connected_workers = stats_.active_workers;
                stats_.pool_speed = static_cast<uint64_t>(stats_.dps_per_second);
            }
            break;
        }

        case JLPMessageType::AUTH_OK: {  // Authentication accepted
            std::cout << "[Pool] Authentication successful" << std::endl;
            auth_state_.store(AuthState::AUTH_OK);
            consecutive_auth_failures_ = 0;
            // Reset reconnect backoff on a successful auth.
            reconnect_delay_ms_ = RECONNECT_BASE_DELAY_MS;
            reconnect_attempts_ = 0;
            auth_cv_.notify_all();
            break;
        }

        case JLPMessageType::AUTH_FAIL: {  // Authentication rejected
            std::cerr << "[Pool] Authentication failed" << std::endl;
            auth_state_.store(AuthState::AUTH_FAILED);
            consecutive_auth_failures_++;
            auth_cv_.notify_all();
            // Mark connection unhealthy so the receiver_loop reconnect path
            // (or external supervisor) sees it. Per D-H5, after
            // MAX_AUTH_FAIL_ATTEMPTS the receiver gives up entirely.
            connected_ = false;
            break;
        }

        case JLPMessageType::DP_ACK: {  // DP acknowledged
            // DPs were received by server - nothing to do
            break;
        }

        case JLPMessageType::SOLUTION: {  // Solution found (bidirectional)
            std::cout << "[Pool] SOLUTION FOUND!" << std::endl;
            if (solution_callback_ && payload.size() >= 32) {
                solution_callback_(payload.data());
            }
            break;
        }

        case JLPMessageType::PING: {  // Server ping - respond with pong
            send_message(JLPMessageType::PONG, nullptr, 0);
            break;
        }

        case JLPMessageType::MSG_ERROR: {  // Server error
            // Wave 4 D-L3: cap printed length and strip control characters so a
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
            // failure so authenticate() returns instead of timing out.
            if (s == AuthState::AUTH_SENT) {
                auth_state_.store(AuthState::AUTH_FAILED);
                auth_cv_.notify_all();
            }
            break;
        }

        default:
            if (debug_mode_) {
                std::cerr << "[DEBUG] Unknown message type: 0x"
                          << std::hex << (int)header.type << std::dec << std::endl;
            }
            break;
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
    // HTTP pool path was deleted in Wave 4 (D-C1). No other types supported.
    return nullptr;
}

} // namespace pool
} // namespace collider
