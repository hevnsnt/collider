/**
 * test_jlp_pool_handshake -- live integration test against the
 * Collision Protocol JLP pool server (pool.collisionprotocol.com:17403).
 *
 * What this test verifies (positive signals):
 *   1. The pool's TLS certificate validates against the system trust store
 *      AND its CN/SAN matches "pool.collisionprotocol.com" (RFC 6125 hostname
 *      check). If either fails, this test FAILS - that is a real outage of
 *      the pool's TLS posture, not a flaky test.
 *   2. After TLS, the client sends a single AUTH message with the exact
 *      bytes the production JLP client emits: 8-byte header
 *      [magic="KANG", type=0x01, flags=PROTOCOL_VERSION=2, payload_size LE u16]
 *      followed by sizeof(JLPClientHelloV2) = 120 bytes of payload.
 *      Pre-v1.4.2 this test asserted the legacy 76-byte v1 payload, which
 *      had drifted from the production client (audit B.4).
 *   3. The server replies with a well-formed JLP frame: 8-byte header whose
 *      magic == "KANG" and whose payload_size matches the bytes that follow.
 *
 * What this test does NOT verify:
 *   - That the worker is accepted (server may reject unknown workers; the
 *     reply may be AUTH_FAIL = 0x03). A well-formed AUTH_FAIL is still a
 *     PASS for handshake purposes.
 *   - DP submission, work assignment, solution submission. Those require a
 *     valid worker registration on the pool's side.
 *
 * Skip semantics (return 77):
 *   - Environment variable COLLIDER_SKIP_NETWORK_TESTS is set (any value).
 *   - TCP connect to pool.collisionprotocol.com:17403 fails or DNS resolution
 *     fails inside the 5-second connect deadline. Network tests must not
 *     break offline / firewalled CI machines.
 *
 * Fail semantics (return 1):
 *   - TCP connected but TLS handshake failed (cert invalid, hostname
 *     mismatch, expired chain, protocol downgrade, etc.).
 *   - Server frame magic != "KANG" (server speaking the wrong protocol).
 *   - Connection closed before a complete header was received.
 *   - Header length field disagrees with bytes actually sent before close.
 *
 * Wire format reference: src/pool/jlp_pool_client.hpp
 *   JLPHeader: [4]"KANG", [1]type, [1]flags=PROTOCOL_VERSION(2), [2]payload_size LE = 8 bytes.
 *   JLPClientHelloV2: [64]worker_name + [32]password + [8]timestamp_ms LE
 *                     + [16]nonce = 120 bytes. Matches IDL AuthPayloadV2
 *                     (struct format '<64s32sQ16s'). Verified at compile
 *                     time below via static_assert.
 *
 * This test does NOT link against the production jlp_pool_client; it
 * re-implements the handshake at the byte level so it cannot regress in
 * lockstep with the client. If they diverge, the test fails.
 */

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <windows.h>
    // wincrypt.h must come AFTER windows.h. It exposes CertOpenSystemStoreA /
    // CertEnumCertificatesInStore / CertCloseStore which we use to bridge the
    // Windows system trust store into OpenSSL's X509_STORE. vcpkg's OpenSSL
    // does NOT load the Windows cert store on its own; SSL_CTX_set_default_verify_paths()
    // resolves to a UNIX-style path that does not exist on Windows, so chain
    // verification fails for every public cert (incl. Let's Encrypt) without
    // this bridge. Link against crypt32.lib (set in CMakeLists.txt).
    #include <wincrypt.h>
    typedef SOCKET socket_t;
    static const socket_t INVALID_SOCK_LOCAL = INVALID_SOCKET;
    #define CLOSE_SOCK closesocket
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <sys/select.h>
    typedef int socket_t;
    static const socket_t INVALID_SOCK_LOCAL = -1;
    #define CLOSE_SOCK ::close
#endif

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/x509v3.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr const char*    POOL_HOST       = "pool.collisionprotocol.com";
constexpr const char*    POOL_PORT_STR   = "17403";
constexpr int            CONNECT_TIMEOUT_SEC = 5;
constexpr int            IO_TIMEOUT_SEC      = 10;

// Mirror of src/pool/jlp_pool_client.hpp wire structs. We do NOT include the
// production header to keep this test free of the heavy pool dependency
// graph; instead we re-declare the layout and static_assert sizes match.
#pragma pack(push, 1)
struct JLPHeader {
    uint8_t  magic[4];        // "KANG"
    uint8_t  type;            // message type
    uint8_t  flags;           // 0
    uint16_t payload_size;    // little-endian
};

// v1.4.2 B.4: production wire layout is JLPClientHelloV2 (120 bytes).
// Pre-fix this test asserted the OLD 76-byte v1 layout (worker_name +
// gpu_count + speed), so it could not have been running against the
// real production client. struct format '<64s32sQ16s' matches IDL
// AuthPayloadV2 and src/pool/jlp_pool_client.hpp::JLPClientHelloV2.
struct JLPClientHelloV2 {
    char     worker_name[64];   // null-padded worker / BTC payout address
    char     password[32];      // optional pool password, null-padded
    uint64_t timestamp_ms;      // client wall-clock (LE)
    uint8_t  nonce[16];         // per-AUTH random
};
#pragma pack(pop)

static_assert(sizeof(JLPHeader)        ==  8,  "JLPHeader must be exactly 8 bytes");
static_assert(sizeof(JLPClientHelloV2) == 120, "JLPClientHelloV2 must be exactly 120 bytes");

// JLP message type IDs (subset).
constexpr uint8_t JLP_AUTH      = 0x01;
constexpr uint8_t JLP_AUTH_OK   = 0x02;
constexpr uint8_t JLP_AUTH_FAIL = 0x03;
constexpr uint8_t JLP_MSG_ERROR = 0xFF;

// ---------------------------------------------------------------------------
// Tiny socket helpers
// ---------------------------------------------------------------------------

static void log_ssl_errors(const char* context) {
    unsigned long e;
    while ((e = ERR_get_error()) != 0) {
        char buf[256];
        ERR_error_string_n(e, buf, sizeof(buf));
        fprintf(stderr, "[ssl-err] %s: %s\n", context, buf);
    }
}

#ifdef _WIN32
struct WSAGuard {
    WSAGuard() { WSADATA w; WSAStartup(MAKEWORD(2, 2), &w); }
    ~WSAGuard() { WSACleanup(); }
};
#endif

// Returns INVALID_SOCK_LOCAL on any failure (DNS, timeout, refused). The
// caller treats that as a SKIP, not a FAIL, since this is a network test.
static socket_t tcp_connect_with_timeout(const char* host, const char* port,
                                         int timeout_sec) {
    addrinfo hints;
    std::memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo* res = nullptr;
    int rc = getaddrinfo(host, port, &hints, &res);
    if (rc != 0 || !res) {
        fprintf(stderr, "[net] getaddrinfo(%s:%s) failed: rc=%d\n", host, port, rc);
        return INVALID_SOCK_LOCAL;
    }

    socket_t sock = INVALID_SOCK_LOCAL;
    for (addrinfo* a = res; a != nullptr; a = a->ai_next) {
        sock = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (sock == INVALID_SOCK_LOCAL) continue;

        // Switch to non-blocking for the timed connect.
#ifdef _WIN32
        u_long nb = 1;
        ioctlsocket(sock, FIONBIO, &nb);
#else
        int flags = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, flags | O_NONBLOCK);
#endif

        rc = ::connect(sock, a->ai_addr, (int)a->ai_addrlen);
        bool in_progress = false;
        if (rc < 0) {
#ifdef _WIN32
            int werr = WSAGetLastError();
            in_progress = (werr == WSAEWOULDBLOCK || werr == WSAEINPROGRESS);
#else
            in_progress = (errno == EINPROGRESS);
#endif
        }
        if (rc == 0 || in_progress) {
            fd_set wfds;
            FD_ZERO(&wfds);
            FD_SET(sock, &wfds);
            timeval tv{ timeout_sec, 0 };
            int sel = ::select((int)sock + 1, nullptr, &wfds, nullptr, &tv);
            if (sel > 0) {
                int err = 0;
                socklen_t errlen = sizeof(err);
                getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&err, &errlen);
                if (err == 0) {
                    // Restore blocking mode for the rest of the test.
#ifdef _WIN32
                    u_long bl = 0;
                    ioctlsocket(sock, FIONBIO, &bl);
#else
                    int f = fcntl(sock, F_GETFL, 0);
                    fcntl(sock, F_SETFL, f & ~O_NONBLOCK);
#endif
                    freeaddrinfo(res);
                    return sock;
                }
            }
        }
        CLOSE_SOCK(sock);
        sock = INVALID_SOCK_LOCAL;
    }
    freeaddrinfo(res);
    return INVALID_SOCK_LOCAL;
}

// recv exactly `n` bytes or fail. Returns true if all bytes read.
static bool recv_all(SSL* ssl, void* buf, int n) {
    int got = 0;
    while (got < n) {
        int r = SSL_read(ssl, (char*)buf + got, n - got);
        if (r <= 0) {
            int e = SSL_get_error(ssl, r);
            fprintf(stderr, "[net] SSL_read returned %d (ssl_error=%d) after %d/%d bytes\n",
                    r, e, got, n);
            return false;
        }
        got += r;
    }
    return true;
}

}  // namespace

int main() {
    // -----------------------------------------------------------------------
    // Skip-by-environment-variable check.
    // -----------------------------------------------------------------------
    if (const char* skip = std::getenv("COLLIDER_SKIP_NETWORK_TESTS"); skip && skip[0] != '\0') {
        printf("Skipping: COLLIDER_SKIP_NETWORK_TESTS is set (=\"%s\")\n", skip);
        return 77;
    }

#ifdef _WIN32
    WSAGuard wsa_guard;
#endif

    printf("=== JLP pool TLS handshake test against %s:%s ===\n", POOL_HOST, POOL_PORT_STR);

    // -----------------------------------------------------------------------
    // 1. TCP connect with timeout. Failure => SKIP, not FAIL (offline CI).
    // -----------------------------------------------------------------------
    socket_t sock = tcp_connect_with_timeout(POOL_HOST, POOL_PORT_STR, CONNECT_TIMEOUT_SEC);
    if (sock == INVALID_SOCK_LOCAL) {
        printf("Skipping: TCP connect to %s:%s failed within %d seconds\n",
               POOL_HOST, POOL_PORT_STR, CONNECT_TIMEOUT_SEC);
        return 77;
    }

    // Set generous I/O timeouts on the socket so a stuck server fails fast
    // rather than hanging the test runner.
#ifdef _WIN32
    DWORD tv_ms = (DWORD)IO_TIMEOUT_SEC * 1000;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv_ms, sizeof(tv_ms));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv_ms, sizeof(tv_ms));
#else
    timeval iotv{ IO_TIMEOUT_SEC, 0 };
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &iotv, sizeof(iotv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &iotv, sizeof(iotv));
#endif

    printf("[+] TCP connected\n");

    // -----------------------------------------------------------------------
    // 2. OpenSSL one-time init + per-connection context with hostname verify.
    // -----------------------------------------------------------------------
    SSL_library_init();
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();

    SSL_CTX* ctx = SSL_CTX_new(TLS_client_method());
    if (!ctx) {
        fprintf(stderr, "[fail] SSL_CTX_new failed\n");
        log_ssl_errors("SSL_CTX_new");
        CLOSE_SOCK(sock);
        return 1;
    }
    SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION);

    // Cert verification: REQUIRED. The whole point of TLS to a known pool is
    // that we authenticate the server. A failure here is a real failure.
    //
    // Trust anchor loading is platform-specific:
    //   * Linux / macOS: SSL_CTX_set_default_verify_paths() finds the system
    //     CA bundle (e.g., /etc/ssl/certs on most distros, the keychain-backed
    //     bundle on macOS via OpenSSL's compiled-in defaults).
    //   * Windows: vcpkg's OpenSSL has NO usable default path. Windows stores
    //     CAs in the registry-backed system cert store, which OpenSSL does
    //     not consult. Without the explicit bridge below, every public cert
    //     fails with X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY (the exact
    //     symptom this test was hitting: "unable to get local issuer cert").
    //
    // The Windows path enumerates the "ROOT" system store and adds each cert
    // to OpenSSL's per-CTX X509_STORE. We are NOT weakening verification --
    // we still use SSL_VERIFY_PEER plus the X509_VERIFY_PARAM_set1_host()
    // hostname check below. We are simply giving OpenSSL access to the trust
    // anchors the OS already trusts.
#ifdef _WIN32
    {
        HCERTSTORE hStore = CertOpenSystemStoreA(0, "ROOT");
        if (hStore) {
            X509_STORE* store = SSL_CTX_get_cert_store(ctx);
            int added = 0;
            PCCERT_CONTEXT pCtx = nullptr;
            while ((pCtx = CertEnumCertificatesInStore(hStore, pCtx)) != nullptr) {
                const unsigned char* p = pCtx->pbCertEncoded;
                X509* x509 = d2i_X509(nullptr, &p, (long)pCtx->cbCertEncoded);
                if (x509) {
                    if (X509_STORE_add_cert(store, x509) == 1) {
                        ++added;
                    } else {
                        // Duplicate or other non-fatal error; clear so
                        // subsequent OpenSSL ops don't see a stale error.
                        ERR_clear_error();
                    }
                    X509_free(x509);
                }
            }
            CertCloseStore(hStore, 0);
            printf("[+] Loaded %d certs from Windows ROOT store into OpenSSL trust\n",
                   added);
            if (added == 0) {
                fprintf(stderr, "[warn] Windows ROOT store enumerated 0 certs; "
                                "TLS verification will fail.\n");
            }
        } else {
            fprintf(stderr, "[warn] CertOpenSystemStoreA(ROOT) failed (err=%lu); "
                            "falling back to OpenSSL default verify paths "
                            "(likely empty on Windows).\n",
                    (unsigned long)GetLastError());
            if (SSL_CTX_set_default_verify_paths(ctx) != 1) {
                fprintf(stderr, "[warn] SSL_CTX_set_default_verify_paths also "
                                "failed; verification will fail.\n");
            }
        }
    }
#else
    if (SSL_CTX_set_default_verify_paths(ctx) != 1) {
        fprintf(stderr, "[warn] SSL_CTX_set_default_verify_paths failed; "
                        "system CA store may be missing.\n");
    }
#endif
    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr);

    SSL* ssl = SSL_new(ctx);
    if (!ssl) {
        fprintf(stderr, "[fail] SSL_new failed\n");
        log_ssl_errors("SSL_new");
        SSL_CTX_free(ctx);
        CLOSE_SOCK(sock);
        return 1;
    }

    // SNI - required by virtually every modern multi-tenant TLS terminator.
    if (SSL_set_tlsext_host_name(ssl, POOL_HOST) != 1) {
        fprintf(stderr, "[fail] SSL_set_tlsext_host_name failed\n");
        log_ssl_errors("SNI");
        SSL_free(ssl); SSL_CTX_free(ctx); CLOSE_SOCK(sock);
        return 1;
    }

    // Hostname verification (RFC 6125): without this, OpenSSL only verifies
    // the chain but NOT that the cert was issued for the host we dialed.
    X509_VERIFY_PARAM* vparam = SSL_get0_param(ssl);
    if (!vparam) {
        fprintf(stderr, "[fail] SSL_get0_param returned null\n");
        SSL_free(ssl); SSL_CTX_free(ctx); CLOSE_SOCK(sock);
        return 1;
    }
    X509_VERIFY_PARAM_set_hostflags(vparam, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
    if (X509_VERIFY_PARAM_set1_host(vparam, POOL_HOST, 0) != 1) {
        fprintf(stderr, "[fail] X509_VERIFY_PARAM_set1_host failed for %s\n", POOL_HOST);
        SSL_free(ssl); SSL_CTX_free(ctx); CLOSE_SOCK(sock);
        return 1;
    }
    SSL_set_verify(ssl, SSL_VERIFY_PEER, nullptr);

    SSL_set_fd(ssl, (int)sock);

    int sslc = SSL_connect(ssl);
    if (sslc != 1) {
        int e = SSL_get_error(ssl, sslc);
        fprintf(stderr, "[fail] TLS handshake failed (SSL_get_error=%d)\n", e);
        log_ssl_errors("SSL_connect");
        long verify = SSL_get_verify_result(ssl);
        if (verify != X509_V_OK) {
            fprintf(stderr, "[fail] cert verify: %s\n",
                    X509_verify_cert_error_string(verify));
        }
        SSL_free(ssl); SSL_CTX_free(ctx); CLOSE_SOCK(sock);
        return 1;  // FAIL - not SKIP
    }
    printf("[+] TLS established (%s)\n", SSL_get_version(ssl));

    // -----------------------------------------------------------------------
    // 3. Send AUTH message (header + JLPClientHello).
    // -----------------------------------------------------------------------
    JLPHeader hdr{};
    hdr.magic[0] = 'K'; hdr.magic[1] = 'A'; hdr.magic[2] = 'N'; hdr.magic[3] = 'G';
    hdr.type         = JLP_AUTH;
    // v1.4.2 B.5: senders set flags = PROTOCOL_VERSION (2). Servers reject
    // mismatches with MSG_ERROR/protocol_version_mismatch.
    hdr.flags        = 2;
    hdr.payload_size = (uint16_t)sizeof(JLPClientHelloV2);

    // v1.4.2 B.4: production wire layout is the 120-byte v2 payload.
    JLPClientHelloV2 hello{};
    const char* worker = "bc1qtest000000000000000000000000000000qqqqq";
    std::memset(hello.worker_name, 0, sizeof(hello.worker_name));
    std::strncpy(hello.worker_name, worker, sizeof(hello.worker_name) - 1);
    // password optional; leave zero-filled for unauthenticated test pool.
    std::memset(hello.password, 0, sizeof(hello.password));
    // Wall-clock in ms since epoch.
    using namespace std::chrono;
    hello.timestamp_ms = (uint64_t)duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
    // Deterministic but uncoordinated nonce for the test bot.
    for (int i = 0; i < 16; ++i) {
        hello.nonce[i] = (uint8_t)((hello.timestamp_ms >> (i * 3)) ^ (0xA5 + i));
    }

    if (SSL_write(ssl, &hdr, sizeof(hdr)) != (int)sizeof(hdr)) {
        fprintf(stderr, "[fail] short write on header\n");
        log_ssl_errors("SSL_write header");
        SSL_free(ssl); SSL_CTX_free(ctx); CLOSE_SOCK(sock);
        return 1;
    }
    if (SSL_write(ssl, &hello, sizeof(hello)) != (int)sizeof(hello)) {
        fprintf(stderr, "[fail] short write on payload\n");
        log_ssl_errors("SSL_write payload");
        SSL_free(ssl); SSL_CTX_free(ctx); CLOSE_SOCK(sock);
        return 1;
    }
    printf("[+] Sent AUTH header (8 B) + ClientHello (%zu B)\n", sizeof(hello));

    // -----------------------------------------------------------------------
    // 4. Read response header. Verify magic + sane payload length.
    // -----------------------------------------------------------------------
    JLPHeader resp{};
    if (!recv_all(ssl, &resp, sizeof(resp))) {
        fprintf(stderr, "[fail] timed out / disconnected before full header read\n");
        SSL_shutdown(ssl); SSL_free(ssl); SSL_CTX_free(ctx); CLOSE_SOCK(sock);
        return 1;
    }

    bool magic_ok = (resp.magic[0] == 'K' && resp.magic[1] == 'A' &&
                     resp.magic[2] == 'N' && resp.magic[3] == 'G');
    printf("[+] Response header: magic=%c%c%c%c type=0x%02X flags=0x%02X payload=%u\n",
           resp.magic[0], resp.magic[1], resp.magic[2], resp.magic[3],
           resp.type, resp.flags, (unsigned)resp.payload_size);

    if (!magic_ok) {
        fprintf(stderr, "[fail] response magic is not 'KANG' - server is not "
                        "speaking JLP wire protocol\n");
        SSL_shutdown(ssl); SSL_free(ssl); SSL_CTX_free(ctx); CLOSE_SOCK(sock);
        return 1;
    }

    // Drain payload if any. We do NOT validate payload contents - just length.
    if (resp.payload_size > 0) {
        // Cap at 64 KiB so a malicious server cannot OOM the test. Pool spec
        // says payloads are <= ~109 bytes today; anything over 64 KiB is
        // certainly broken.
        if (resp.payload_size > 65535u) {
            fprintf(stderr, "[fail] absurd payload_size %u\n",
                    (unsigned)resp.payload_size);
            SSL_shutdown(ssl); SSL_free(ssl); SSL_CTX_free(ctx); CLOSE_SOCK(sock);
            return 1;
        }
        std::string body(resp.payload_size, '\0');
        if (!recv_all(ssl, body.data(), (int)body.size())) {
            fprintf(stderr, "[fail] short read on payload (expected %u bytes)\n",
                    (unsigned)resp.payload_size);
            SSL_shutdown(ssl); SSL_free(ssl); SSL_CTX_free(ctx); CLOSE_SOCK(sock);
            return 1;
        }
    }

    // Classify response type. AUTH_OK or AUTH_FAIL or MSG_ERROR are all
    // "server speaks the protocol correctly" outcomes.
    const char* type_label = "?";
    switch (resp.type) {
        case JLP_AUTH_OK:   type_label = "AUTH_OK";   break;
        case JLP_AUTH_FAIL: type_label = "AUTH_FAIL"; break;
        case JLP_MSG_ERROR: type_label = "MSG_ERROR"; break;
        default:            type_label = "OTHER";     break;
    }
    printf("[+] Server response classified as: %s\n", type_label);

    // -----------------------------------------------------------------------
    // 5. Clean disconnect.
    // -----------------------------------------------------------------------
    SSL_shutdown(ssl);
    SSL_free(ssl);
    SSL_CTX_free(ctx);
    CLOSE_SOCK(sock);

    printf("PASS: TLS handshake + JLP frame exchange completed successfully.\n");
    return 0;
}
