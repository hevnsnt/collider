# JLP v1.5 Pool Protocol Reference

Authoritative reference for the v1.5 asymmetric kangaroo pool protocol
(JLP v3 on the wire). Companion to `src/pool/jlp_pool_client.hpp` +
`src/pool/jlp_pool_client.cpp` (client side) and the Python collision
detector in the `collision-protocol` repo (server side). Any wire
change MUST touch BOTH sides + the mock-server tests in
`tests/test_jlp_pool_protocol.cpp` and `tests/test_jlp_pool_dp_bits_validation.cpp`.

## Frame Format

All frames share the same 8-byte header:

```
+--------+--------+--------+--------+--------+--------+--------+--------+
| 'K'    | 'A'    | 'N'    | 'G'    | type   | flags  | length (LE u16) |
+--------+--------+--------+--------+--------+--------+--------+--------+
```

- `magic`: literal `"KANG"`, ASCII.
- `type`: message type (see table below).
- `flags`: protocol version byte. v1.5 = `0x03`.
- `length`: little-endian uint16 payload length (0..65535).

Receiver validates magic, version, and that `length <= MAX_MESSAGE_SIZE`
before reading the payload. A version mismatch causes a clean
disconnect with a `MSG_ERROR/protocol_version_mismatch` (0x10) reply if
auth is complete; pre-auth it just disconnects.

## Message Types

| Type | Name           | Dir | Payload  | Notes                                             |
| ---- | -------------- | --- | -------- | ------------------------------------------------- |
| 0x01 | `AUTH`         | C→S | 120 B    | Worker name + password + timestamp_ms + nonce     |
| 0x02 | `AUTH_OK`      | S→C | 0 B      | Authentication accepted                           |
| 0x03 | `AUTH_FAIL`    | S→C | ≤ 256 B  | Reason string (ascii, control-chars stripped)     |
| 0x10 | `MSG_ERROR`    | S→C | ≤ 256 B  | Generic protocol error (server-side)              |
| 0x11 | `WORK_ASN`     | S→C | 126 B    | Work assignment (see WORK_ASN section)            |
| 0x12 | `STATS_RSP`    | S→C | 36 B     | Pool statistics tick                              |
| 0x20 | `DP_SUBMIT_V2` | C→S | 78 B     | Single distinguished point (v2 wire fmt)          |
| 0x24 | `DP_BATCH_V2`  | C→S | 4 + n×78 | Batched DPs; u32 LE count prefix + n × DP         |
| 0x30 | `DP_ACK`       | S→C | 8 B      | Server ack of received DP batch                   |
| 0x40 | `SOLUTION`     | S→C | 32 B     | Pool solved chunk; payload is recovered key bytes |
| 0x50 | `PING`         | S→C | 0 B      | Server keepalive                                  |
| 0x51 | `PONG`         | C→S | 0 B      | Client keepalive response                         |

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
