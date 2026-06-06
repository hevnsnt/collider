# JLP v1.5 Pool Protocol Reference

Authoritative reference for the v1.5 asymmetric kangaroo pool protocol.
JLP v4 (shipped in collider v1.5.4) is the current wire version; JLP v3
(clients v1.5.0 to 1.5.3) is accepted in compatibility mode. The v3
asymmetric assignment, frame format, and DP_BATCH_V2 path below are
unchanged in v4; the v4 delta (in-band auto update, maintenance mode,
checkpoint-walk DP commitment, and the replay challenge) is in the "JLP
v4 delta" section near the end. Companion to
`src/pool/jlp_pool_client.hpp` + `src/pool/jlp_pool_client.cpp` (client
side) and the Python collision detector in the `collision-protocol`
repo (server side). Any wire change MUST touch BOTH sides + the
mock-server tests in `tests/test_jlp_pool_protocol.cpp` and
`tests/test_jlp_pool_dp_bits_validation.cpp`.

## Frame Format

All frames share the same 8-byte header:

```
+--------+--------+--------+--------+--------+--------+--------+--------+
| 'K'    | 'A'    | 'N'    | 'G'    | type   | flags  | length (LE u16) |
+--------+--------+--------+--------+--------+--------+--------+--------+
```

- `magic`: literal `"KANG"`, ASCII.
- `type`: message type (see table below).
- `flags`: protocol version byte. v1.5.4 = `0x04`; v1.5.0 to 1.5.3 = `0x03`.
- `length`: little-endian uint16 payload length (0..65535).

Receiver validates magic, version, and that `length <= MAX_MESSAGE_SIZE`
before reading the payload. The server negotiates per connection: at
AUTH it reads the client `flags` and sets
`negotiated_version = min(client_flags, 4)`, then stamps that version
into the `flags` byte of every outbound frame, so a v3 client (flags=3)
is answered in v3 and a v4 client (flags=4) in v4. There is a
non-configurable security floor of `3` (config key `protocol_mode`:
`compatibility` floors at v3 and accepts v3 + v4; `strict` floors at
v4). A client below the floor (v1.4.x and older) is refused at AUTH
with `MSG_ERROR/protocol_version_mismatch` (0x10); that error now
applies only below the floor, not to v3 in compatibility mode.

## Message Types

| Type | Name            | Dir | Payload   | Notes                                                                        |
| ---- | --------------- | --- | --------- | ---------------------------------------------------------------------------- |
| 0x01 | `AUTH`          | C→S | 120 B     | Worker name + password + timestamp_ms + nonce                                |
| 0x02 | `AUTH_OK`       | S→C | 0 / 324 B | Authentication accepted. v4: `AuthOkPayload` (324 B) advert; v3: 0 B         |
| 0x03 | `AUTH_FAIL`     | S→C | ≤ 256 B   | Reason string (ascii, control-chars stripped)                                |
| 0x10 | `MSG_ERROR`     | S→C | ≤ 256 B   | Generic protocol error (server-side)                                         |
| 0x11 | `WORK_ASN`      | S→C | 126 B     | Work assignment (see WORK_ASN section)                                       |
| 0x12 | `STATS_RSP`     | S→C | 36 B      | Pool statistics tick                                                         |
| 0x20 | `DP_SUBMIT_V2`  | C→S | 78 B      | Single distinguished point (v2 wire fmt)                                     |
| 0x24 | `DP_BATCH_V2`   | C→S | 4 + n×78  | Batched v2 DPs; u32 LE count prefix + n × DP. v3 path; never challenged      |
| 0x25 | `DP_SUBMIT_V3`  | C→S | 114 B     | **v4.** Single v3 DP (`DistinguishedPointV3`) with checkpoint commitment     |
| 0x26 | `DP_BATCH_V3`   | C→S | 4 + n×114 | **v4.** Batched v3 DPs; u32 LE count prefix + n × DP                         |
| 0x30 | `DP_ACK`        | S→C | 8 B       | Server ack of received DP batch                                              |
| 0x32 | `CHALLENGE`     | S→C | variable  | **v4.** Checkpoint-replay challenge. Gated to v4 clients only                |
| 0x33 | `CHALLENGE_RSP` | C→S | variable  | **v4.** Endpoint checkpoint distances + Merkle proofs (hand coded)           |
| 0x40 | `SOLUTION`      | S→C | 32 B      | Pool solved chunk; payload is recovered key bytes                            |
| 0x50 | `PING`          | S→C | 0 B       | Server keepalive                                                             |
| 0x51 | `PONG`          | C→S | 0 B       | Client keepalive response                                                    |
| 0x60 | `MAINTENANCE`   | S→C | 262 B     | **v4.** `MaintenancePayload`; back off + auto resume. v3 gets AUTH_FAIL note |

## WORK_ASN Payload (126 bytes, v1.5)

```
offset  size  field
0       33    public_key       compressed (0x02/0x03 prefix + X)
33      32    range_start      big-endian; inclusive
65      32    range_end        big-endian; exclusive
97       4    dp_bits          little-endian u32, validated 8..32
101      8    work_id          little-endian u64
109      1    kangaroo_type    0=BOTH (illegal in pool), 1=TAME_ONLY, 2=WILD_ONLY
110      8    start_offset_a   little-endian u64; inclusive lower bound of worker window
118      8    start_offset_b   little-endian u64; exclusive upper bound
```

### v1.5 asymmetric fields

The pool server is the **sole key-computer** in v1.5. Workers never
hold a recovered private key. The server assigns each worker a half:
TAME_ONLY (1) walks forward from `range_start`, WILD_ONLY (2) walks
forward from `range_end - N*step` (server's choice). The mathematical
guarantee is that a TAME-walking kangaroo and a WILD-walking kangaroo
collide at a distinguished point; the server sees both sides of the
collision and computes the key.

`start_offset_a` + `start_offset_b` carve a disjoint sub-range from
the assigned chunk for THIS worker. Reserved for future per-worker
chunk-slicing; currently propagated end-to-end and logged but not yet
enforced by the worker.

### kangaroo_type validation

`kangaroo_type == 0` is the only value the client REJECTS at the
backend layer (`CudaRCKangarooBackend::initialize` returns an error).
0 means "BOTH" — a worker that walks both tame and wild can compute
the recovered key locally, which is exactly the v1.4 theft surface
v1.5 eliminated. A server bug that sends 0 stops the worker; it does
not silently degrade to local key recovery.

### dp_bits validation

`receive_message → reject_work_asn_dp_bits` rejects outside `[8, 32]`:

- `< 8`: DPs flood the wire at millions per second
- `> 32`: expected time between DPs is days at production rates

Outside-window assignments trigger a clean disconnect + supervisor retry.

## DP_BATCH_V2 Payload

```
offset  size  field
0        4    count        little-endian u32, capped at 10000 per batch
4        78   dp[0]        first distinguished point
82       78   dp[1]
...      78   dp[count-1]
```

Each DP is the v2 78-byte layout:

```
offset  size  field
0        8    work_id      little-endian u64, attests to assigned chunk
8        4    sequence     little-endian u32, per-(worker, work_id) monotonic
12       32   x            big-endian distinguished-point X coord
44       32   d            big-endian distance scalar
76       1    type         0 = tame, 1 = wild
77       1    dp_bits      number of leading-zero bits (matches server)
```

### Anti-replay / dedup

- `work_id` ensures the server can match a DP to the chunk currently
  assigned to that worker.
- `sequence` is the per-(worker, work_id) counter the worker increments
  monotonically. Server caches the last observed sequence per worker;
  out-of-window sequences are rejected as replays.
- Pre-AUTH_OK DPs are queued + flushed back into the queue if the
  client reconnects (see `requeue_unauth_batch` in jlp_pool_client.cpp).

## Connection Lifecycle Cheat Sheet

```
state      ┃ event            ┃ next state    ┃ side effect
───────────╋──────────────────╋───────────────╋──────────────────────────
DISCONNECTED┃ connect() ok    ┃ CONNECTING    ┃ socket open, TLS up
CONNECTING ┃ send AUTH        ┃ AUTH_SENT     ┃ password wiped from heap
AUTH_SENT  ┃ recv AUTH_OK     ┃ AUTH_OK       ┃ dp_queue drain enabled
AUTH_SENT  ┃ recv AUTH_FAIL   ┃ DISCONNECTED  ┃ supervisor backoff
AUTH_SENT  ┃ recv MSG_ERROR   ┃ AUTH_FAILED   ┃ stop auto-retry
AUTH_OK    ┃ recv WORK_ASN    ┃ AUTH_OK       ┃ backend re-init for chunk
AUTH_OK    ┃ disconnect()     ┃ DISCONNECTED  ┃ drain dp_queue (2s cap)
```

### Drain timeout

`DRAIN_TIMEOUT_MS = 2000` (bumped from 500 in Wave J after observing
~1.4s drain windows on saturated 10gig links). The window bounds
clean shutdown so a wedged sender does not block process exit
indefinitely; in practice the queue drains in <100ms for healthy
workers. If draining times out, the client logs "drain timed out with
N DP(s) still queued" and exits anyway — the supervisor's next
reconnect cycle re-attempts the orphaned DPs.

## Reconnect Supervisor

`PoolManager::supervisor_loop` polls `is_connected()` every 500ms.
On observed disconnect:

```
attempt 1  -> sleep [500ms, 1000ms]      jittered (backoff/2 .. backoff)
attempt 2  -> sleep [1000ms, 2000ms]
attempt 3  -> sleep [2000ms, 4000ms]
...
attempt N  -> sleep [16s, 32s]            capped at MAX_RECONNECT_BACKOFF_MS=32000
```

Hard caps:

- `MAX_RECONNECT_ATTEMPTS = 16` — after this, supervisor gives up and
  the solve loop exits cleanly.
- `MAX_AUTH_FAIL_ATTEMPTS = 5` — credential-spray defense; AUTH_FAIL
  responses count separately from socket-loss reconnects.
- IP ban detection (AUTH_FAIL payload starts with `ip_banned:`)
  short-circuits the supervisor immediately. No retry, no backoff,
  exit with operator-facing message.

## JLP v4 delta (v1.5.4)

v4 is the current wire version. Everything above (frame format, AUTH,
WORK_ASN asymmetric assignment, DP_BATCH_V2, reconnect supervisor) is
unchanged. v4 layers the following additions on top, all negotiated to
v4 clients only; a v3 client (v1.5.0 to 1.5.3) keeps mining via the v3
path and is answered in v3.

### Version negotiation

The server sets `negotiated_version = min(client_flags, 4)` at AUTH and
stamps that version into the `flags` byte of every outbound frame. A
v3 client gets flags=3 and a zero-payload AUTH_OK; a v4 client gets
flags=4 and the 324-byte AUTH_OK advert. Security floor is 3 (config
`protocol_mode`: `compatibility` accepts v3 + v4, `strict` v4 only);
v1.4.x and older are refused at AUTH.

### AUTH_OK advert and in-band auto update

For a v4 client, AUTH_OK (0x02) carries `AuthOkPayload` (324 bytes,
struct pack `<16s16sB3s256s32s`):

```
offset  size  field
0       16    latest_version   ASCII semver, null-padded (e.g. "1.5.4")
16      16    min_version      ASCII semver of the minimum supported client
32      1     flags            bit0 = update_available, bit1 = maintenance_active
33      3     reserved
36      256   download_url     HTTPS URL of the latest signed binary, null-padded;
                               all-zero disables auto update
292     32    sha256           raw SHA-256 of the binary at download_url
```

The client compares its own version to `latest_version` and, if older
and a `download_url` is present, fetches the binary over HTTPS,
verifies it against `sha256`, then replaces itself and relaunches.

### Maintenance mode

MAINTENANCE (0x60) carries `MaintenancePayload` (262 bytes, struct pack
`<BBI256s`):

```
offset  size  field
0       1     active             1 = maintenance in effect (back off), 0 = resume
1       1     reserved
2       4     retry_after_secs   little-endian u32; base reconnect backoff, client adds jitter
6       256   message            operator note, null-padded ASCII
```

The server sends this after AUTH_OK to a worker that connects while the
pool is in maintenance, or broadcasts it to live workers when an
operator toggles maintenance on. The client shows the note and backs
off gracefully, then auto resumes. A v3 client (no 0x60 frame) instead
receives an AUTH_FAIL carrying a maintenance note.

### DistinguishedPointV3 and DP_BATCH_V3

v4 workers submit DPs via DP_BATCH_V3 (0x26), N x `DistinguishedPointV3`
(114 bytes, struct pack `<QI32s32sBB32sI`):

```
offset  size  field
0       8     work_id      little-endian u64, attests to assigned chunk
8       4     sequence     little-endian u32, per-(worker, work_id) monotonic
12      32    x            big-endian distinguished-point X coord
44      32    d            big-endian distance scalar
76      1     type         0 = tame, 1 = wild
77      1     dp_bits      number of leading-zero bits (matches server)
78      32    ckpt_root    Merkle root over the walk's checkpoint distances
                           (one checkpoint every CHECKPOINT_INTERVAL = 65536 jumps)
110     4     n_segments   little-endian u32, committed segment count (checkpoint_count - 1)
```

It is a v2 superset plus the checkpoint-walk commitment. In
compatibility mode the server also accepts the older DP_BATCH_V2 frames
and credits them identically; a v2 frame is simply never challenged.

### Checkpoint-replay challenge

CHALLENGE (0x32, server to client, variable length): work_id u64 LE,
nonce 8 bytes, count u16 LE, then count x segment index u32 LE. The
worker replies with CHALLENGE_RSP (0x33, variable length, hand coded)
revealing the endpoint checkpoint distances plus their Merkle proofs so
the server can replay the forward jumps and confirm the segment links
up. The challenge is gated to v4 clients only; a v3 client is never
challenged. The verifier ships off (`challenge_mode`: `off`, `shadow`,
`enforce`) and is enabled only after it is validated against a real
walk, so honest workers are never falsely penalized during the rollout.

### Type mismatch is recoverable (no longer a ban)

As of v1.5.4 a wrong-type DP (a tame DP under WILD_ONLY or vice versa)
is treated as a recoverable event, not cheating and not a ban. The
server rejects the DP and asks the worker to re request work (the stale
work_id path); only repeated stale or wrong submissions beyond a
disconnect limit cause a clean disconnect that forces a fresh AUTH and
WORK_REQ, never a permanent IP ban. An epoch race where a worker is
reassigned between tame and wild mid flight can emit a stale-type DP
with no malice, so banning on it was incorrect. Every worker still
gets the asymmetric tame/wild split plus DP shadow verification; v4
workers are additionally eligible for the checkpoint replay challenge
when it is enabled, and v3 workers are never challenged.

## See Also

- `src/pool/jlp_pool_client.hpp/.cpp` — wire-level implementation
- `src/pool/pool_manager.hpp/.cpp` — reconnect supervisor + dedup
- `tests/test_jlp_pool_protocol.cpp` — 18 unit + integration cases
  (including TP-10 byte-fragmented stream-resync regression)
- `tests/test_jlp_pool_dp_bits_validation.cpp` — S1 dp_bits boundary
- `tests/test_jlp_pool_handshake.cpp` — full TLS handshake roundtrip
- `tests/test_jlp_pool_manager.cpp` — supervisor lifecycle
- `tests/test_jlp_pool_reconnect.cpp` — backoff + jitter + cap
- `.claude/tasks/v1.5-asymmetric-kangaroo.md` — design doc for v1.5
  asymmetric protocol rationale
