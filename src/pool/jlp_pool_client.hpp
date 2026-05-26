// jlp_pool_client.hpp - JeanLucPons Kangaroo protocol compatible pool client
// Compatible with pools running JLP Kangaroo server (port 17403)

#pragma once

// Windows: Must define NOMINMAX before any includes to prevent min/max macro conflicts
#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
#endif

#include "pool_client.hpp"
#include "pool_config.hpp"
#include "../core/secure_buffer.hpp"  // SecureString for the pool password
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <atomic>
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

// TLS support via OpenSSL
#ifdef COLLIDER_HAS_OPENSSL
    #include <openssl/ssl.h>
    #include <openssl/err.h>
#endif

namespace collider {
namespace identity {
class WorkerIdentity;
}  // namespace identity
}  // namespace collider

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET socket_t;
    #define INVALID_SOCK INVALID_SOCKET
    #define SOCK_ERROR SOCKET_ERROR
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    typedef int socket_t;
    #define INVALID_SOCK -1
    #define SOCK_ERROR -1
    #define closesocket close
#endif

namespace collider {
namespace pool {

// JLP Protocol message types - must match collision-protocol server
enum class JLPMessageType : uint8_t {
    // Authentication
    AUTH      = 0x01,
    AUTH_OK   = 0x02,
    AUTH_FAIL = 0x03,

    // Work distribution
    WORK_REQ  = 0x10,
    WORK_ASN  = 0x11,

    // Distinguished points (v1 - no chunk attestation)
    DP_SUBMIT = 0x20,
    DP_ACK    = 0x21,
    DP_BATCH  = 0x22,

    // Distinguished points v2 - identical fields plus an 8-byte work_id
    // prefix attesting which assigned chunk the DP came from. The server
    // verifies the claimed work_id matches this worker's current
    // assignment and rejects the submission otherwise (catches the
    // wrong-chunk submission attack class). v2 wire layout matches
    // collision-protocol/src/jlp_protocol.py DistinguishedPointV2:
    //   struct.pack('<Q32s32sBB', work_id, x, d, type, dp_bits)
    DP_SUBMIT_V2 = 0x23,
    DP_BATCH_V2  = 0x24,

    // Statistics
    STATS_REQ = 0x30,
    STATS_RSP = 0x31,

    // Solution
    SOLUTION  = 0x40,

    // Keepalive
    PING      = 0x50,
    PONG      = 0x51,

    // Error (named MSG_ERROR to avoid Windows ERROR macro conflict)
    MSG_ERROR = 0xFF
};

// JLP Protocol header - MUST match server format exactly:
// Python: struct.pack('<4sBBH', b'KANG', msg_type, flags, len(payload))
// [MAGIC:4][TYPE:1][FLAGS:1][LENGTH:2][PAYLOAD]
#pragma pack(push, 1)
struct JLPHeader {
    uint8_t magic[4];        // "KANG" (4 bytes)
    uint8_t type;            // Message type (1 byte)
    uint8_t flags;           // Flags, typically 0 (1 byte)
    uint16_t payload_size;   // Size of payload (2 bytes, little-endian)
};  // Total: 8 bytes

// AUTH wire format (120 bytes). A legacy JLPClientHello (76 bytes) sent
// by the client was decoded as 96 bytes (name + password) by the Python
// server and silently mis-aligned gpu_count/speed onto the password
// slot. The current format is name + password + a timestamp_ms + a
// 16-byte nonce so the server can reject replays of captured AUTH
// packets and enforce client-clock drift bounds.
// Field order matches collision-protocol/src/jlp_protocol.py
// (struct.pack equivalent: '<64s32sQ16s').
struct JLPClientHelloV2 {
    char     worker_name[64];   // Worker identifier (Bitcoin address)
    char     password[32];      // Optional pool password
    uint64_t timestamp_ms;      // Wall-clock time in ms (LE)
    uint8_t  nonce[16];         // Per-AUTH random nonce
};
static_assert(sizeof(JLPClientHelloV2) == 120,
              "JLPClientHelloV2 must be 120 bytes on the wire");

// B1 wire-v4 AUTH frame (217 bytes). Extends v3 with the compressed
// public key and the 64-byte raw r||s ECDSA signature over the
// canonical AUTH message (see worker_identity.hpp + jlp_protocol.py
// build_auth_canonical_message). The password slot is preserved at
// offset 64..96 to keep the v3 prefix bit-for-bit compatible with
// servers that probe by slot index; v4 zeros it because v4 replaces
// the credential model with signed identity.
// Field order matches collision-protocol/src/jlp_protocol.py
// encode_auth_v4 / decode_auth_v4 layout exactly.
//
// Defined OUTSIDE the IDL-generated jlp_wire_generated.hpp because the
// codegen tool refuses to round-trip if hand edits land there. The IDL
// (protocol/jlp.yaml) will absorb v4 when the server fully cuts over
// (PROTOCOL_VERSION_MIN bumped to 4); until then v4 is an opt-in
// client capability gated on a loaded WorkerIdentity.
namespace jlp_wire {
// header.flags byte for v4 frames. Servers route to decode_auth_v4
// when they see this value; servers pinned to v3 reject as wire
// mismatch (which is fine because operators only flip --worker-key on
// after the server-side cutover).
constexpr int PROTOCOL_VERSION_V4 = 4;
}  // namespace jlp_wire

namespace auth_v4 {
// Build the canonical signed message body that wire-v4 AUTH covers.
// MUST match collision-protocol/src/jlp_protocol.py
// JLPProtocol.build_auth_canonical_message byte-for-byte. Layout:
//   "COLLIDER-WORKER-AUTH-v1\n" || u8(proto_ver) || u64_le(ts_ms) ||
//   nonce16 || u8(name_len) || name_bytes.
// Free function so the KAT test can exercise it without touching the
// JLPPoolClient class. Pure: no I/O, no state.
std::vector<uint8_t> build_canonical_message(
    uint8_t protocol_version,
    uint64_t timestamp_ms,
    const uint8_t nonce[16],
    const std::string& worker_name);
}  // namespace auth_v4

struct JLPClientHelloV4 {
    char     worker_name[64];          // Worker identifier (must be bech32(hash160(pubkey)))
    char     password_padding[32];     // Zero-filled for v4
    uint64_t timestamp_ms;             // Wall-clock time in ms (LE)
    uint8_t  nonce[16];                // Per-AUTH random nonce
    uint8_t  pubkey_compressed[33];    // secp256k1 compressed pubkey
    uint8_t  signature_raw[64];        // ECDSA over canonical msg, raw r||s (NOT DER)
};
static_assert(sizeof(JLPClientHelloV4) == 217,
              "JLPClientHelloV4 must be 217 bytes on the wire");

// Work assignment structure - must match collision-protocol/src/jlp_protocol.py ServerConfig
// v1.5 (protocol_version=3): payload grew from 109 to 126 bytes. The three
// new trailing fields (kangaroo_type, start_offset_a, start_offset_b)
// carry the asymmetric tame/wild assignment that denies any single worker
// the data needed to compute the recovered private key locally. The
// pre-existing fields keep their identical byte offsets and little-endian
// encoding; old (109-byte) decoders see the longer payload and reject
// it as malformed -- there is intentionally no v1.4 -> v1.5 interop.
// Python: struct.pack('<33s32s32sIQBQQ',
//   public_key, range_start, range_end, dp_bits, work_id,
//   kangaroo_type, start_offset_a, start_offset_b)
//
// Sizeof equality with jlp_wire::WorkAssignment is enforced by static_assert
// in tests/test_jlp_wire_generated.cpp; any future drift between this
// handwritten struct and the auto-generated wire struct breaks the build.
struct JLPServerConfig {
    uint8_t  public_key[33];  // 33 bytes - Compressed public key
    uint8_t  range_start[32]; // 32 bytes - Range start (big-endian)
    uint8_t  range_end[32];   // 32 bytes - Range end (big-endian)
    uint32_t dp_bits;         // 4 bytes  - DP bits (little-endian)
    uint64_t work_id;         // 8 bytes  - Work identifier (little-endian)
    // v1.5 fields (offset 109..125):
    uint8_t  kangaroo_type;   // 1 byte   - 0=BOTH (illegal in pool), 1=TAME_ONLY, 2=WILD_ONLY
    uint64_t start_offset_a;  // 8 bytes  - inclusive lower bound (low 64 bits) of worker's offset window
    uint64_t start_offset_b;  // 8 bytes  - exclusive upper bound (low 64 bits) of worker's offset window
    // Total: 126 bytes
};

struct JLPDistinguishedPoint {
    uint8_t x[32];           // X coordinate
    uint8_t d[32];           // Distance
    uint8_t type;            // Tame (0) or Wild (1)
    uint8_t dp_bits;         // Number of leading-zero bits used (matches server)
};
// Wire format must be 66 bytes to match collision-protocol/src/jlp_protocol.py
// DistinguishedPoint.to_bytes() (struct.pack('<32s32sBB', x, d, type, dp_bits)).
// DP_BATCH payload is then [count:u32 LE][dp1:66][dp2:66]...
static_assert(sizeof(JLPDistinguishedPoint) == 66,
              "JLPDistinguishedPoint must be 66 bytes on the wire");

// v2 distinguished point: 78-byte wire layout. The 8-byte work_id prefix
// attests which assigned chunk the DP came from; the 4-byte sequence is
// a per-(worker, work_id) monotonic counter. The server tracks an
// expected window and rejects out-of-window sequences as replays.
// Matches the server's DistinguishedPointV2.to_bytes()
// (struct.pack('<QI32s32sBB', work_id, sequence, x, d, type, dp_bits)).
// DP_BATCH_V2 payload is [count:u32 LE][dp1:78][dp2:78]...
struct JLPDistinguishedPointV2 {
    uint64_t work_id;        // 8 bytes  - chunk id from WorkAssignment (LE)
    uint32_t sequence;       // 4 bytes  - monotonic per-chunk counter (LE)
    uint8_t  x[32];          // 32 bytes - X coordinate
    uint8_t  d[32];          // 32 bytes - Distance
    uint8_t  type;           // 1 byte   - Tame (0) or Wild (1)
    // Wire format pins this to 1 byte (struct.pack 'B' on the server side
    // gives 78-byte total). The abstract DistinguishedPoint::dp_bits in
    // pool_client.hpp is uint64_t for API ergonomics; the sender narrows
    // to uint8_t here. Realistic values are 1..50, so truncation is safe.
    // (Gemini PR review pointed out the type difference; deliberate.)
    uint8_t  dp_bits;        // 1 byte   - Number of leading-zero bits used
};
static_assert(sizeof(JLPDistinguishedPointV2) == 78,
              "JLPDistinguishedPointV2 must be 78 bytes on the wire");
#pragma pack(pop)

// Connection / authentication state machine.
// The receiver MUST gate work-affecting messages on AUTH_OK so a
// malicious or misbehaving server cannot inject WORK_ASN / SOLUTION /
// DP_ACK / STATS_RSP before the client has authenticated.
enum class AuthState : uint8_t {
    DISCONNECTED = 0,  // Not connected (initial / after disconnect)
    CONNECTING   = 1,  // TCP/TLS handshake in progress
    AUTH_SENT    = 2,  // AUTH message sent, waiting for AUTH_OK / AUTH_FAIL
    AUTH_OK      = 3,  // Authentication accepted; full message dispatch enabled
    AUTH_FAILED  = 4   // Authentication rejected; do not auto-retry indefinitely
};

class JLPPoolClient : public PoolClient {
public:
    // Queue limits to prevent unbounded memory growth
    static constexpr size_t MAX_DP_QUEUE_SIZE = 100000;  // ~6.5MB of DPs max
    static constexpr uint8_t SUPPORTED_PROTOCOL_VERSION = 1;

    // Bound the time we wait for an AUTH_OK / AUTH_FAIL response after
    // sending the AUTH message.
    static constexpr uint32_t AUTH_RESPONSE_TIMEOUT_MS = 10000;  // 10s

    // max time disconnect() waits for the sender to drain
    // dp_queue_ before forcing the threads to exit. Was 500 ms originally;
    // bumped to 2000 ms after Wave J reproduced shutdown_drains_dp_queue
    // flaking on loaded Windows hosts where the OS scheduler can park
    // the sender for >500 ms during process exit. 2 s still lets a wedged
    // sender give up well before any operator-perceived hang and matches
    // the AUTH_RESPONSE_TIMEOUT_MS / 5 budget proportion.
    // For the full lifecycle + cross-references see
    // docs/internals/jlp-v1.5-pool-protocol.md "Drain timeout" section.
    static constexpr uint32_t DRAIN_TIMEOUT_MS = 2000;

    JLPPoolClient();
    ~JLPPoolClient() override;

    // PoolClient interface
    bool connect(const std::string& host, uint16_t port) override;
    void disconnect() override;
    bool is_connected() const override;

    // actually waits for AUTH_OK / AUTH_FAIL / MSG_ERROR or timeout.
    // Returns false on AUTH_FAIL, MSG_ERROR, or timeout. The `password` parameter
    // is currently ignored by the JLP wire protocol (no password field in the
    // ClientHello struct); it is accepted for interface compatibility with
    // PoolClient but a non-empty value will produce a one-time warning.
    bool authenticate(const std::string& worker_name,
                     const std::string& password = "") override;

    bool request_work(WorkAssignment& work) override;
    bool submit_dp(const DistinguishedPoint& dp) override;
    bool submit_dps(const std::vector<DistinguishedPoint>& dps) override;

    PoolStatsLocal get_stats() override;

    // v1.5: client-to-server report_solution() was DELETED. No code path
    // on the worker may possess the puzzle's private key. The SOLUTION
    // wire message is now strictly server-to-client (broadcast on solve).
    // The recovered_keys/<ts>.json persistence file, the 24-hour retry
    // thread infrastructure, and the SecureBuffer key staging were all
    // removed together. The server (collision-protocol) is the sole
    // entity that sees DPs of both tame and wild type and the sole
    // entity that can compute the recovered key. See
    // .claude/tasks/v1.5-asymmetric-kangaroo.md.

    void set_solution_callback(SolutionCallback cb) override;
    void set_work_callback(WorkCallback cb) override;

    std::string get_pool_type() const override { return POOL_TYPE_JLP; }

    // JLP-specific settings
    void set_timeout(uint32_t timeout_ms) { timeout_ms_ = timeout_ms; }
    void set_reconnect(bool auto_reconnect) { auto_reconnect_ = auto_reconnect; }
    void set_debug_mode(bool debug) { debug_mode_ = debug; }
    void set_use_tls(bool use_tls) { use_tls_ = use_tls; }
    void set_verify_cert(bool verify) { verify_cert_ = verify; }

    // B1 wire-v4: attach a loaded worker identity (--worker-key). When
    // set, send_hello emits a 217-byte v4 AUTH frame signed by the
    // identity's privkey and bumps header.flags to PROTOCOL_VERSION_V4.
    // When nullptr (default) the client emits v3 AUTH unchanged. The
    // shared_ptr lets the identity outlive any single reconnect cycle
    // without forcing the PoolManager to re-load the key on every
    // supervisor-driven JLPPoolClient re-creation.
    void set_worker_identity(
        std::shared_ptr<collider::identity::WorkerIdentity> id);

    // dp_sequence_next_ ownership lives in PoolManager
    // so a supervisor-driven reconnect that recreates the JLPPoolClient
    // does NOT reset the counter to 0 (which would self-ban the worker
    // against the server's _dp_seq_high watermark). On every client
    // construction the supervisor calls seed_dp_sequence(seq, work_id)
    // with the persisted next-seq for that (worker, work_id) pair.
    //   * seed_dp_sequence: write-only seed for the per-chunk counter.
    //   * dp_sequence_progress_callback: mirror counter mutations back
    //     so the manager-side map and the on-disk snapshot stay current.
    //   * work_id_progress_callback: mirror work_id changes back so
    //     the manager can detect genuine reassignment and reset the
    //     counter for the new key (Pool-B1 final clause).
    using DpSeqProgressCallback =
        std::function<void(uint64_t work_id, uint32_t next_seq)>;
    using WorkIdAssignedCallback =
        std::function<void(uint64_t work_id)>;

    void seed_dp_sequence(uint64_t work_id, uint32_t next_seq);
    void set_dp_sequence_progress_callback(DpSeqProgressCallback cb);
    void set_work_id_assigned_callback(WorkIdAssignedCallback cb);

    // emit the plaintext warning before any bytes are
    // written to the wire (AUTH carries credentials). Called by
    // parse_pool_url + the supervisor when it spins up a fresh client.
    // No-op for jlps://.
    static void warn_if_plaintext(const std::string& url_scheme_or_host,
                                  uint16_t port);

    // Returns true if the last authenticate() failure was an IP ban
    // (server sent AUTH_FAIL with an "IP banned:" payload). When true,
    // ban_reason() holds the server's ban message. The supervisor uses
    // this to give up immediately instead of burning through retries.
    bool is_ip_banned() const { return ip_banned_; }
    const std::string& ban_reason() const { return ban_reason_; }

    // pre-load DPs that were on disk from a prior shutdown.
    // Called by PoolManager BEFORE authenticate() so the first DP_BATCH_V2
    // includes the persisted backlog. The vector is moved in to avoid an
    // extra copy of a large queue.
    void preload_dp_queue(std::vector<DistinguishedPoint>&& persisted);

    // serialize the current dp_queue_ to disk so the next
    // process start can replay it. Called by PoolManager::disconnect()
    // BEFORE the underlying JLPPoolClient is destroyed.
    bool persist_dp_queue_to_disk(const std::string& path) const;

    // snapshot the per-chunk DP counters held inside the
    // client. After the supervisor decides to recreate the client (e.g.
    // on disconnect) it should call this just before destroying the old
    // instance so any in-flight increments are mirrored back to its map.
    void snapshot_dp_sequence(uint64_t& out_work_id,
                              uint32_t& out_next_seq) const;

private:
    bool debug_mode_ = false;

    // Set by handle_auth_fail() when the server's AUTH_FAIL payload begins
    // with "IP banned:". Checked by the supervisor to short-circuit retries.
    bool ip_banned_ = false;
    std::string ban_reason_;

    // Network
    socket_t socket_;
    std::string host_;
    uint16_t port_;
    uint32_t timeout_ms_;
    bool auto_reconnect_;
    std::atomic<bool> connected_;
    std::atomic<bool> running_;
    std::atomic<bool> last_receive_was_timeout_;  // Track if last recv was timeout vs disconnect

    // task #34 stream-resync state. Receive_message accumulates partial
    // recv() into these per-client buffers so a SO_RCVTIMEO mid-message
    // does not lose stream alignment (the Heisenbug behind v1.5.x
    // JLPPoolProtocol crashes on Windows: 1ms timeout + MSG_WAITALL +
    // multi-byte messages = silent partial-byte drops). Both fields are
    // touched ONLY from the receiver thread while ssl_read_mutex_ is
    // held, so no atomicity is needed.
    JLPHeader header_partial_{};
    size_t    header_partial_progress_  = 0;
    std::vector<uint8_t> payload_partial_;
    size_t    payload_partial_progress_ = 0;

    // connection-state machine. atomic so the receiver and main
    // threads can both read it without taking a lock on the hot dispatch path.
    std::atomic<AuthState> auth_state_{AuthState::DISCONNECTED};

    // condition variable + mutex used by authenticate() to wait
    // for an AUTH_OK / AUTH_FAIL / MSG_ERROR transition.
    std::mutex auth_cv_mutex_;
    std::condition_variable auth_cv_;

    // the in-receiver-thread reconnect block (which
    // counted consecutive_auth_failures_ against MAX_AUTH_FAIL_ATTEMPTS
    // and applied an exponential backoff with RECONNECT_BACKOFF_MULTIPLIER)
    // was deleted because it ALWAYS returned after a single sleep, so the
    // counters and constants were dead code suggesting behavior that never
    // ran. The PoolManager supervisor (pool_manager.cpp) is now the only
    // reconnect driver. The auth-fail cap moved to pool_config.hpp as
    // MAX_AUTH_FAIL_ATTEMPTS and is enforced by the supervisor.

    // TLS support
    bool use_tls_ = false;
    bool verify_cert_ = true;  // default to verify (was false / fail-open)
#ifdef COLLIDER_HAS_OPENSSL
    SSL_CTX* ssl_ctx_ = nullptr;
    SSL* ssl_ = nullptr;
    bool init_tls();
    void cleanup_tls();
    int ssl_send(const void* data, size_t size);
    int ssl_recv(void* data, size_t size);
#endif

    // Separate read and write mutexes for concurrent I/O on the SSL object.
    // OpenSSL is documented to support one concurrent reader and one
    // concurrent writer on the same SSL session, provided each direction is
    // serialized internally. A SINGLE mutex covering both sides deadlocks:
    // the receiver thread blocks in SSL_read holding the mutex, and any
    // main-thread send_message() (e.g. WORK_REQ) cannot acquire the mutex
    // to send -- so the request never goes on the wire and the worker
    // appears idle to the pool until it times out.
    // We therefore split into:
    //   ssl_write_mutex_  -- serializes sender_loop and main-thread sends
    //   ssl_read_mutex_   -- serializes only the receiver_loop (defensive;
    //                        only one thread reads in this design)
    // Renegotiation / shutdown / SSL_clear must be done with both threads
    // joined (see disconnect()) -- those are NOT covered by these mutexes.
    std::mutex ssl_write_mutex_;
    std::mutex ssl_read_mutex_;

    // Worker info
    std::string worker_name_;
    // Pool password. Held as a SecureString so the bytes are wiped on
    // destruction (zero-on-free) and can be wiped explicitly the moment
    // the handshake completes. The supervisor (PoolManager) keeps its
    // own copy so reconnect can re-authenticate without prompting; that
    // is the storage owner of record. The client's copy exists only for
    // the duration of one authenticate() call.
    ::collider::SecureString password_;       // optional pool password
    // historical gpu_count_ / speed_ members lived here for ClientHello
    // telemetry. v2 wire dropped both fields; deleted 2026-05-23 (audit
    // finding #23) since nothing transmits them and nothing reads them.

    // B1 wire-v4 (2026-05-23): optional signed-identity material. When
    // set, send_hello() emits a 217-byte v4 AUTH frame and bumps the
    // header.flags byte to PROTOCOL_VERSION_V4. nullptr (default) keeps
    // the existing v3 wire path. Owned by the pool supervisor; shared_ptr
    // because the same identity outlives any single reconnect.
    std::shared_ptr<collider::identity::WorkerIdentity> worker_identity_;

    // Current work
    WorkAssignment current_work_;
    bool work_received_ = false;  // true once WORK_ASN arrives (work_id==0 is a valid chunk)
    // per-(worker, work_id) monotonic DP counter. Reset to
    // 0 every time the server hands out a new chunk; incremented on
    // each DP submitted in DP_BATCH_V2. The server tracks the expected
    // window per (worker_name, work_id) and rejects out-of-window
    // sequences as replays. The counter lives next to current_work_
    // because both are reset together when WORK_ASN arrives.
    uint32_t dp_sequence_next_ = 0;
    // Mutable so const accessors (snapshot_dp_sequence) can lock without
    // const_cast. The mutex itself is conceptually metadata, not state.
    mutable std::mutex work_mutex_;

    // Statistics
    PoolStatsLocal stats_;
    std::mutex stats_mutex_;

    // Callbacks. Protected by callbacks_mutex_ so that set_*_callback
    // can race against an inbound message handler without tearing the
    // function pointer, and so the dedup state (last_solution_pubkey_,
    // last_work_id_fired_) is consistent across threads. An earlier
    // unlocked design had the SOLUTION handler read solution_callback_
    // directly and call the callback inside the read; if a SOLUTION
    // message arrived twice (server retry, bug, malicious replay) the
    // callback fired twice, which on a privacy-sensitive pool would
    // expose the recovered key to two writers (e.g., logged + filed
    // twice).
    SolutionCallback solution_callback_;
    WorkCallback     work_callback_;
    mutable std::mutex callbacks_mutex_;
    std::array<uint8_t, 32> last_solution_pubkey_{};  // zero = none seen
    bool     solution_fired_ = false;
    // client-instance work_id dedup was removed because
    // the supervisor recreates JLPPoolClient on reconnect, resetting any
    // last_work_id_fired_ sentinel and re-firing the same work_id back to
    // the host. Dedup now lives on PoolManager (which outlives client
    // recreation). This client always forwards every WORK_ASN; the
    // manager-side filter decides whether to surface it to the host.
    DpSeqProgressCallback   dp_seq_progress_callback_;
    WorkIdAssignedCallback  work_id_assigned_callback_;
    void fire_solution_callback(const uint8_t* pubkey_bytes);
    void fire_work_callback(const WorkAssignment& work);

    // Receiver thread
    std::thread receiver_thread_;
    void receiver_loop();

    // v1.5: the RetryThread infrastructure that backed
    // JLPPoolClient::report_solution() (24-hour retry uploader for
    // recovered private keys) was DELETED together with report_solution
    // itself. The pool server is now the sole key-computer; the worker
    // never holds a recovered key, so there is nothing to retry-upload.
    // See .claude/tasks/v1.5-asymmetric-kangaroo.md.

    // DP queue for batched submission. std::deque is used (vs
    // std::queue) so the sender can push the head of an un-sendable
    // batch BACK onto the front of the queue if it drained 100 DPs but
    // the AUTH state hadn't reached AUTH_OK yet. Dropping those DPs
    // silently loses work; pushing them to a side queue would re-order
    // against fresh DPs.
    std::deque<DistinguishedPoint> dp_queue_;
    // Mutable so const accessors (snapshot_dp_queue) can lock without
    // const_cast. The mutex itself is conceptually metadata, not state.
    mutable std::mutex dp_mutex_;
    std::condition_variable dp_cv_;
    std::thread sender_thread_;
    std::atomic<bool> drain_requested_{false};  // flush flag
    // DPs that the sender had to drop on a genuine shutdown
    // (running_=false before AUTH_OK). Surfaced via get_shutdown_drop_count
    // so PoolManager can roll it into its own dropped_count_.
    std::atomic<uint64_t> shutdown_drop_count_{0};
    void sender_loop();

    // sender_loop sub-steps. The loop body is naturally five phases:
    // wait for activity, drain the queue into a wire-ready batch, send
    // it (or roll back the sequence counter and re-queue if AUTH isn't
    // ready), do a final drain pass if disconnect() asked for one, and
    // run periodic STATS_REQ + PING keepalives. Splitting the phases
    // into private methods keeps sender_loop a thin orchestrator and
    // collapses the duplicate drain-and-send code that previously
    // appeared verbatim in both the main path and the disconnect
    // cleanup block.
    void wait_for_send_signal();
    // Drain up to 100 DPs from dp_queue_ into `batch`, encoded for the
    // wire under current_work_id / seq_start. The caller has already
    // sampled current_work_id and dp_sequence_next_ under work_mutex_;
    // we hold dp_mutex_ alone for the drain. Returns the number drained.
    size_t drain_dp_queue_into_batch(uint64_t current_work_id,
                                     uint32_t seq_start,
                                     std::vector<JLPDistinguishedPointV2>& batch);
    // Compare-and-set dp_sequence_next_ for current_work_id: only write
    // if the chunk under work_mutex_ still matches (a racing WORK_ASN
    // resets the counter for the new chunk, and we must not stomp it).
    void advance_dp_sequence_if_unchanged(uint64_t current_work_id,
                                          uint32_t new_next);
    // Best-effort send of a batch under AUTH_OK. Wire format is
    // [count:u32 LE][dp1:78][dp2:78]... (matches DP_BATCH_V2). On send
    // success, mirrors the new sequence counter back to PoolManager,
    // bumps stats / session-log counters, and updates last_send for
    // keepalive bookkeeping. payload is a hoisted scratch buffer the
    // caller owns to avoid per-call reallocation.
    bool send_dp_batch(uint64_t current_work_id,
                       uint32_t seq_start,
                       const std::vector<JLPDistinguishedPointV2>& batch,
                       std::vector<uint8_t>& payload,
                       std::chrono::steady_clock::time_point& last_send);
    // Roll back the sequence advance and push the drained DPs back onto
    // the FRONT of dp_queue_ in reverse so wire order is preserved.
    // Called when we drained DPs but the connection wasn't AUTH_OK yet,
    // so the batch must wait for the next iteration.
    void requeue_unauth_batch(uint64_t current_work_id,
                              uint32_t seq_start,
                              const std::vector<JLPDistinguishedPointV2>& batch);
    // Final drain pass after disconnect() asked for a flush. Drains as
    // many DPs as possible WHILE we still have connected_ + AUTH_OK,
    // sending them in 100-DP batches. Exits when the queue is empty,
    // running_ flips false, AUTH state slips, or a send fails. Returns
    // when no more progress is possible. `batch` and `payload` are
    // scratch buffers borrowed from the caller.
    void perform_final_drain_pass(std::vector<JLPDistinguishedPointV2>& batch,
                                  std::vector<uint8_t>& payload,
                                  std::chrono::steady_clock::time_point& last_send);
    // Send STATS_REQ every STATS_INTERVAL_SECONDS and PING every
    // KEEPALIVE_SECONDS if no activity has gone out in that window.
    // Both are gated on AUTH_OK so we don't talk to a server that
    // hasn't accepted us yet. last_send is updated when either send
    // happens.
    void send_periodic_stats_and_keepalive(
        std::chrono::steady_clock::time_point& last_send,
        std::chrono::steady_clock::time_point& last_stats_req);

public:
    uint64_t get_shutdown_drop_count() const {
        return shutdown_drop_count_.load(std::memory_order_relaxed);
    }
    // caller copies all queued DPs out so they can be
    // serialized to disk before the client is destroyed. Returns the
    // copy; original queue is left intact (caller decides whether to
    // discard).
    std::vector<DistinguishedPoint> snapshot_dp_queue() const;
private:

    // Protocol helpers. send_message defaults header.flags to the
    // currently-active PROTOCOL_VERSION (v3 today). Callers that need
    // to emit a different wire-version frame (e.g. v4 AUTH) pass
    // flags_override explicitly.
    bool send_message(JLPMessageType type, const void* data, size_t size,
                      uint8_t flags_override = 0xFF);
    bool receive_message(JLPHeader& header, std::vector<uint8_t>& payload);
    bool send_hello();
    void handle_server_message(const JLPHeader& header, const std::vector<uint8_t>& payload);

    // Per-message handlers. handle_server_message is a thin dispatcher:
    // it enforces the pre-AUTH_OK gate and forwards to one of these. Each
    // handler owns its own payload parsing, validation, side-effects, and
    // callback firing. Splitting them out keeps the dispatcher small and
    // lets each protocol case be reasoned about (and unit-tested) in
    // isolation. handle_default_unknown covers the switch default branch
    // (debug-only log of unrecognized message types) plus PONG, which has
    // no client-side reaction beyond the keepalive bookkeeping already
    // done by the receiver loop's last-receive timestamp.
    void handle_auth_ok(const JLPHeader& header, const std::vector<uint8_t>& payload);
    void handle_auth_fail(const JLPHeader& header, const std::vector<uint8_t>& payload);
    void handle_msg_error(const JLPHeader& header,
                          const std::vector<uint8_t>& payload,
                          AuthState pre_dispatch_state);
    void handle_work_asn(const JLPHeader& header, const std::vector<uint8_t>& payload);
    void handle_solution(const JLPHeader& header, const std::vector<uint8_t>& payload);
    void handle_dp_ack(const JLPHeader& header, const std::vector<uint8_t>& payload);
    void handle_stats_rsp(const JLPHeader& header, const std::vector<uint8_t>& payload);
    void handle_ping(const JLPHeader& header, const std::vector<uint8_t>& payload);
    void handle_default_unknown(const JLPHeader& header,
                                const std::vector<uint8_t>& payload);

    // WORK_ASN sub-steps. handle_work_asn() composes these: validate
    // dp_bits and tear the connection down on a violation, snapshot the
    // assignment under work_mutex_, emit the session-log milestone and
    // SessionState snapshot. Splitting them out keeps the WORK_ASN
    // handler at dispatcher length and lets the parts be reasoned about
    // (and tested) independently. reject_work_asn_dp_bits returns true
    // when the assignment was rejected (caller must stop further
    // processing); the helper is responsible for ALL the connection
    // teardown bookkeeping previously inlined in the switch case.
    bool reject_work_asn_dp_bits(const JLPServerConfig& config);
    void apply_work_asn_assignment(const JLPServerConfig& config,
                                   WorkAssignment& out_work_copy);
    void log_work_asn_milestone(const JLPServerConfig& config,
                                const WorkAssignment& work_copy);

    // Safely (re)assign a std::thread by joining the existing one if
    // joinable. Plain `t = std::thread(...)` on a joinable thread
    // invokes std::terminate per [thread.thread.assign]/p2.
    // Marked noexcept: the body catches every exception path (join's
    // std::system_error, detach's std::system_error, anything else from
    // std::cerr) and std::thread::operator= is itself noexcept per the
    // standard, so the function never propagates an exception. The
    // noexcept marker lets the compiler tighten code generation in the
    // reconnect-error path that called this; an implicit
    // "potentially-throwing" classification on a function used during
    // error recovery is the kind of latent hazard that can cause
    // std::terminate during cleanup.
    static void replace_thread(std::thread& t, std::thread new_thread) noexcept;

    // Platform init
    static bool init_sockets();
    static void cleanup_sockets();
    static bool sockets_initialized_;
};

} // namespace pool
} // namespace collider
