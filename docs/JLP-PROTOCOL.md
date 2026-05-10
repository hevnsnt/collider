# JLP Wire Protocol Reference

JLP is the wire protocol spoken between theCollider's pool client and a JLP pool server (the reference server is `collision-protocol`, available at <https://collisionprotocol.com>). This document is intended to be sufficient for a third-party implementer to write a conformant client or alternative server.

The IDL source of truth is `protocol/jlp.yaml`. The C++ generated header `src/pool/jlp_wire_generated.hpp` and the Python generated module `data/protocol/jlp_protocol_generated.py` are produced from that YAML by `tools/codegen/jlp_codegen.py`. **Hand-editing the generated files is forbidden.** A third-party client should derive its own bindings from `protocol/jlp.yaml`, not from any single language binding.

The reference C++ client is `src/pool/jlp_pool_client.cpp`; the reference Python codegen lives at `tools/codegen/jlp_codegen.py`.

> Note for readers of the public Free repo: `protocol/jlp.yaml` and `tools/codegen/` are excluded from the public sync. The C++ generated header at `src/pool/jlp_wire_generated.hpp` is the only artifact in the Free repo, and it carries the same byte layout. This document mirrors the IDL exactly.

---

## 1. Protocol version

`protocol_version: 2`

The protocol version increments only on breaking wire changes. Backward-compatible additions (new message types, new optional fields) do not bump the version.

---

## 2. Endianness and packing

- **All multi-byte integer fields are little-endian.**
- **All structs are packed.** No padding. The generated C++ header uses `#pragma pack(push, 1)` and asserts each struct's `wire_size` at compile time.
- Opaque byte fields (such as range bounds and EC coordinates) are **big-endian** to match canonical Bitcoin / cryptographic convention. This is documented per-field below.

| Field type | Encoding                                                |
| ---------- | ------------------------------------------------------- |
| `uint8`    | 1 byte.                                                 |
| `uint16`   | 2 bytes, little-endian.                                 |
| `uint32`   | 4 bytes, little-endian.                                 |
| `uint64`   | 8 bytes, little-endian.                                 |
| `float32`  | 4 bytes, IEEE 754 single, little-endian.                |
| `bytes[N]` | N bytes, opaque (per-field comment specifies encoding). |

---

## 3. Frame header

Every JLP message starts with the same 8-byte header.

| Offset | Size | Field          | Type       | Notes                                              |
| -----: | ---: | -------------- | ---------- | -------------------------------------------------- |
|      0 |    4 | `magic`        | `bytes[4]` | Always the four ASCII bytes `K`, `A`, `N`, `G`.    |
|      4 |    1 | `type`         | `uint8`    | One of the message-type constants in section 4.    |
|      5 |    1 | `flags`        | `uint8`    | Reserved. Currently `0`.                           |
|      6 |    2 | `payload_size` | `uint16`   | Number of payload bytes following the header (LE). |

Header total: **8 bytes**. Payload follows immediately. Total frame size on the wire is `8 + payload_size`.

`MAX_MESSAGE_SIZE = 1048576` (1 MiB). Frames whose `payload_size` exceeds this cap MUST be rejected before reading the payload.

---

## 4. Message types

| Name           | Code | Direction        | Notes                                                     |
| -------------- | ---- | ---------------- | --------------------------------------------------------- |
| `AUTH`         | 0x01 | Client -> Server | First frame after connect.                                |
| `AUTH_OK`      | 0x02 | Server -> Client | Authentication accepted.                                  |
| `AUTH_FAIL`    | 0x03 | Server -> Client | Authentication rejected. Connection closes.               |
| `WORK_REQ`     | 0x10 | Client -> Server | Request a chunk assignment.                               |
| `WORK_ASN`     | 0x11 | Server -> Client | Chunk assignment.                                         |
| `DP_SUBMIT`    | 0x20 | Client -> Server | Single distinguished-point submission (v1, deprecated).   |
| `DP_ACK`       | 0x21 | Server -> Client | Acknowledges a `DP_SUBMIT` or `DP_BATCH_V2`.              |
| `DP_BATCH`     | 0x22 | Client -> Server | Batch of v1 DPs (deprecated).                             |
| `DP_SUBMIT_V2` | 0x23 | Client -> Server | Single v2 DP (rare; clients normally batch).              |
| `DP_BATCH_V2`  | 0x24 | Client -> Server | Batch of v2 DPs (preferred).                              |
| `STATS_REQ`    | 0x30 | Client -> Server | Optional explicit pull. Server also pushes periodically.  |
| `STATS_RSP`    | 0x31 | Server -> Client | Pool statistics snapshot.                                 |
| `SOLUTION`     | 0x40 | Server -> Client | A solution was found in this pool. Pushed asynchronously. |
| `PING`         | 0x50 | Client -> Server | Keepalive.                                                |
| `PONG`         | 0x51 | Server -> Client | Keepalive reply.                                          |
| `MSG_ERROR`    | 0xFF | Either           | Out-of-band error.                                        |

The constant is named `MSG_ERROR` rather than `ERROR` because `ERROR` collides with the Windows `<winerror.h>` macro. Server and client implementations should follow the same convention.

---

## 5. Struct layouts

All structs are packed. Offsets are zero-based from the start of the **payload** (i.e. after the 8-byte header). Struct-level wire sizes are documented and asserted at codegen time.

### 5.1 `AuthPayload` (96 bytes)

Payload of an `AUTH` (0x01) frame.

| Offset | Size | Field         | Type        | Notes                                                                            |
| -----: | ---: | ------------- | ----------- | -------------------------------------------------------------------------------- |
|      0 |   64 | `worker_name` | `bytes[64]` | UTF-8 worker name. Doubles as the BTC payout address. Null-padded to 64 bytes.   |
|     64 |   32 | `password`    | `bytes[32]` | Optional shared pool password. Null-padded to 32 bytes. Empty if not configured. |

Python `struct.pack` format: `<64s32s`.

`worker_name` MUST be the Bitcoin address that will receive payouts; the server uses this string verbatim as the per-worker credit key.

### 5.2 `WorkAssignment` (109 bytes)

Payload of a `WORK_ASN` (0x11) frame.

| Offset | Size | Field         | Type        | Notes                                                               |
| -----: | ---: | ------------- | ----------- | ------------------------------------------------------------------- |
|      0 |   33 | `public_key`  | `bytes[33]` | Compressed secp256k1 pubkey of the puzzle target (`02...`/`03...`). |
|     33 |   32 | `range_start` | `bytes[32]` | Big-endian 32-byte chunk start.                                     |
|     65 |   32 | `range_end`   | `bytes[32]` | Big-endian 32-byte chunk end.                                       |
|     97 |    4 | `dp_bits`     | `uint32`    | Distinguished-point bit threshold.                                  |
|    101 |    8 | `work_id`     | `uint64`    | Low 64 bits of the chunk identifier.                                |

Python `struct.pack` format: `<33s32s32sIQ`.

The C++ struct in `src/pool/jlp_pool_client.hpp` is named `JLPServerConfig` for historical reasons. The Python side names it `WorkAssignment`. The wire layout is identical.

`work_id` is what the client must echo back in every DP submission for this chunk (see `DistinguishedPointV2`).

### 5.3 `DistinguishedPoint` (66 bytes, deprecated)

Payload of a `DP_SUBMIT` (0x20) frame, or one element of a `DP_BATCH` (0x22) array.

| Offset | Size | Field     | Type        | Notes                                                        |
| -----: | ---: | --------- | ----------- | ------------------------------------------------------------ |
|      0 |   32 | `x`       | `bytes[32]` | Big-endian X coordinate.                                     |
|     32 |   32 | `d`       | `bytes[32]` | Big-endian walked distance (signed; sign-extended into 32B). |
|     64 |    1 | `type`    | `uint8`     | `0` = tame, `1` = wild.                                      |
|     65 |    1 | `dp_bits` | `uint8`     | Leading-zero bit count required of `x`.                      |

Python `struct.pack` format: `<32s32sBB`.

This struct lacks any work-id attestation. The server can only do cryptographic-math verification (does the X coordinate have the required leading zeros, etc.), not chunk-binding. **Do not use `DistinguishedPoint` (v1) in new clients.** It is preserved on the wire for v1.2.x deployed clients only and is rate-limited or refused in current servers.

### 5.4 `DistinguishedPointV2` (78 bytes)

Payload of a `DP_SUBMIT_V2` (0x23) frame, or one element of a `DP_BATCH_V2` (0x24) array.

| Offset | Size | Field      | Type        | Notes                                                                     |
| -----: | ---: | ---------- | ----------- | ------------------------------------------------------------------------- |
|      0 |    8 | `work_id`  | `uint64`    | Chunk id from the assigned `WorkAssignment`. LE.                          |
|      8 |    4 | `sequence` | `uint32`    | Per-`(worker, work_id)` monotonic counter, starts at 0. LE. (v1.4.1 B.1.) |
|     12 |   32 | `x`        | `bytes[32]` | Big-endian X coordinate.                                                  |
|     44 |   32 | `d`        | `bytes[32]` | Big-endian walked distance (signed; sign-extended into 32B).              |
|     76 |    1 | `type`     | `uint8`     | `0` = tame, `1` = wild.                                                   |
|     77 |    1 | `dp_bits`  | `uint8`     | Leading-zero bit count required of `x`.                                   |

Python `struct.pack` format: `<QI32s32sBB`.

**Sequence rules.** The `sequence` field is per `(worker_name, work_id)`, monotonic, starting at `0` for the first DP of each chunk. The server tracks a sliding window of expected sequence numbers; sequences far below the high-water mark are rejected as replays. A client that abandons a chunk and is reassigned the same `work_id` later starts back at `0`. Out-of-order delivery within the window is acceptable; the window is only used to reject obvious replay attempts.

**`work_id` rules.** Clients MUST set `work_id` to the most recent `WorkAssignment.work_id` for the chunk this DP came from. Submitting a DP whose `work_id` does not match the worker's currently-assigned chunk is treated as an anti-cheat infraction (see section 9).

### 5.5 `PoolStats` (36 bytes)

Payload of a `STATS_RSP` (0x31) frame.

| Offset | Size | Field            | Type      | Notes                                                                  |
| -----: | ---: | ---------------- | --------- | ---------------------------------------------------------------------- |
|      0 |    8 | `total_dps`      | `uint64`  | Cumulative pool DP count. LE.                                          |
|      8 |    4 | `total_workers`  | `uint32`  | Distinct worker names ever seen. LE.                                   |
|     12 |    4 | `active_workers` | `uint32`  | Workers active in the last 5 minutes. LE.                              |
|     16 |    4 | `dps_per_second` | `float32` | Pool aggregate DP rate. IEEE 754 LE.                                   |
|     20 |    4 | `your_share`     | `float32` | Fraction of `total_dps` attributable to this worker name. IEEE 754 LE. |
|     24 |    8 | `your_dps`       | `uint64`  | DPs credited to this worker name. LE.                                  |
|     32 |    4 | `uptime_seconds` | `uint32`  | Server uptime in seconds. LE.                                          |

Python `struct.pack` format: `<QIIffQI`.

`your_share` is summed across all machines using the same `worker_name`, not just this connection.

---

## 6. Connection lifecycle

```
Client                                    Server
  |                                          |
  |---- TCP connect (TLS for jlps://) ------>|
  |                                          |
  |---- AUTH (within AUTH_TIMEOUT_SECS=30) ->|
  |                                          |
  |<--- AUTH_OK -or- AUTH_FAIL --------------|
  |                                          |
  |---- WORK_REQ ---------------------------->|
  |                                          |
  |<--- WORK_ASN ----------------------------|
  |                                          |
  |  (compute on assigned chunk)             |
  |                                          |
  |---- DP_BATCH_V2 ------------------------>|
  |<--- DP_ACK ------------------------------|
  |                                          |
  |---- PING (every KEEPALIVE_SECS=20) ----->|
  |<--- PONG --------------------------------|
  |                                          |
  |<--- STATS_RSP (server-pushed) -----------|
  |<--- SOLUTION (server-pushed when found) -|
  |                                          |
  |  (on chunk completion: WORK_REQ again)   |
```

### 6.1 Connect

Clients connect over TCP for `jlp://` URLs and over TLS for `jlps://` URLs. TLS connections MUST validate against the system trust store; the reference client uses `X509_VERIFY_PARAM_set1_host` with `NO_PARTIAL_WILDCARDS` plus SNI plus the default `verify_paths`. A failed trust-store load aborts at init (do not silently fall back to no-verify).

### 6.2 AUTH

Within `AUTH_TIMEOUT_SECS = 30` seconds of the TCP/TLS handshake, the client MUST send an `AUTH` frame. Servers drop connections that do not complete AUTH within this window.

The server replies with exactly one of:

- `AUTH_OK` (0x02). Empty payload. The connection is now in the "authenticated" state.
- `AUTH_FAIL` (0x03). The connection MUST be closed by the server. Clients SHOULD bound their reconnect attempts (the reference client uses 3 attempts with jittered exponential backoff).

Until `AUTH_OK` is received, the client MUST NOT send `WORK_REQ`, `DP_*`, or `STATS_REQ`. The reference server's state machine rejects any of those before AUTH and treats them as anti-cheat infractions.

### 6.3 WORK_REQ / WORK_ASN

After `AUTH_OK`, the client sends `WORK_REQ` (empty payload) and the server replies with a `WorkAssignment` payload. The chunk is now bound to this `(worker_name, work_id)` pair until the client signals completion (next `WORK_REQ` after sending the last DP of the previous chunk) or the server reissues it after a stale-chunk timeout.

A client MAY hold only one outstanding chunk at a time. Multiple parallel chunks per connection are not supported in protocol v2.

### 6.4 DP submission

Clients SHOULD batch DPs in `DP_BATCH_V2` frames. Each frame contains 1 to `MAX_BATCH_SIZE = 10000` `DistinguishedPointV2` records, packed end-to-end with no separators. The frame's `payload_size` must equal `count * 78`.

The reference client debounces submissions on a fixed cadence (typically every few seconds) and flushes immediately when the batch fills up.

The server replies with `DP_ACK` (payload format: TBD; the reference implementation echoes the count of accepted DPs).

### 6.5 Keepalive

Clients SHOULD send `PING` every `KEEPALIVE_SECS = 20` seconds when no other client-to-server traffic is in flight. The server replies with `PONG`. Both frames have empty payloads.

A client that has not sent any frame for several `KEEPALIVE_SECS` periods may be disconnected by the server.

### 6.6 Server-pushed messages

`STATS_RSP` and `SOLUTION` are pushed by the server asynchronously. Clients MUST be prepared to receive them at any time after `AUTH_OK`. The reference client decouples its read loop from its write loop for exactly this reason.

`SOLUTION` payload format is documented elsewhere (it carries the recovered private key plus the `work_id` that produced it). When a `SOLUTION` is received, the client SHOULD treat the current pool round as over and either disconnect or wait for a new `WorkAssignment`.

---

## 7. AUTH replay protection

The server rejects replay-style AUTH attempts (the same `AUTH` payload bytes captured and replayed from another IP). The mechanism is server-validated using a timestamp plus a random nonce embedded in the AUTH payload's password slot for clients that opt into replay protection, with a drift bound of plus or minus 30 seconds.

The exact wire layout of the timestamp / nonce subfield within the `password` slot is server-implementation-specific in protocol v2 and is documented as required-but-spec-pending pending v1.4.2. New third-party clients SHOULD assume that some form of replay protection exists and that submitting captured AUTH bytes from another IP will be rejected; the reference C++ client implementation in `src/pool/jlp_pool_client.cpp` is the authoritative example for the current scheme.

---

## 8. Anti-cheat rules

A client MUST follow these rules to remain unbanned. The thresholds below are the reference server's defaults (from `protocol/jlp.yaml::anti_cheat`); a third-party server may tighten them.

| Rule                                     | Threshold                                         |
| ---------------------------------------- | ------------------------------------------------- |
| Invalid DPs per IP                       | 100 per `invalid_dp_window_secs = 3600` (1 hour). |
| `work_id` mismatch                       | Counts as an invalid DP.                          |
| Out-of-window `sequence`                 | Counts as an invalid DP.                          |
| Cryptographically inconsistent X / d     | Counts as an invalid DP.                          |
| AUTH after the 30-second window          | Disconnect, no ban (transient).                   |
| Sending non-AUTH frames before `AUTH_OK` | Counts as an invalid DP-equivalent infraction.    |

Ban escalation, per IP, within `ban_count_window_secs = 2592000` (30 days):

1. First infraction: 1 hour (`3600` sec).
2. Second: 6 hours (`21600` sec).
3. Third: 1 day (`86400` sec).
4. Fourth: 7 days (`604800` sec).
5. Fifth and beyond: permanent.

Old infraction records are pruned after `invalid_dp_retention_secs = 1209600` (14 days).

A reasonable client never trips any of these rules in normal operation. The thresholds exist for actively-misbehaving (or buggy) clients.

---

## 9. Constants summary

These constants are defined in the IDL (`protocol/jlp.yaml::constants`) and reproduced in both generated bindings.

| Name                | Value     | Description                                             |
| ------------------- | --------- | ------------------------------------------------------- |
| `MAX_MESSAGE_SIZE`  | `1048576` | 1 MiB hard cap per frame, enforced before payload read. |
| `MAX_BATCH_SIZE`    | `10000`   | Maximum DPs per `DP_BATCH` or `DP_BATCH_V2` frame.      |
| `AUTH_TIMEOUT_SECS` | `30`      | Worker must send AUTH within this window.               |
| `KEEPALIVE_SECS`    | `20`      | Recommended client PING frequency.                      |

---

## 10. Worked examples

### 10.1 Building an `AUTH` frame

Pseudo-code, building the frame for `worker_name = "1MyBtcAddress..."` with no password.

```
worker_name_bytes = utf8("1MyBtcAddress...").ljust(64, b"\x00")    # exactly 64 bytes
password_bytes    = b"\x00" * 32                                    # exactly 32 bytes
auth_payload      = worker_name_bytes + password_bytes              # 96 bytes total

header  = b"KANG"                          # magic
header += bytes([0x01])                    # type = AUTH
header += bytes([0x00])                    # flags = 0
header += (96).to_bytes(2, "little")       # payload_size = 96 LE

frame = header + auth_payload              # 8 + 96 = 104 bytes

socket.sendall(frame)
```

### 10.2 Building a `DP_BATCH_V2` frame

Building a batch of three `DistinguishedPointV2` records for the assigned `work_id = 0xDEADBEEF`, starting at `sequence = 0`.

```
def encode_dp_v2(work_id, seq, x_be32, d_be32, dp_type, dp_bits):
    return (
        work_id.to_bytes(8, "little") +           # 0..7
        seq.to_bytes(4, "little") +               # 8..11
        x_be32 +                                  # 12..43 (caller provides 32 BE bytes)
        d_be32 +                                  # 44..75 (caller provides 32 BE bytes)
        bytes([dp_type, dp_bits])                 # 76..77
    )
    # = 78 bytes total

dp0 = encode_dp_v2(0xDEADBEEF, 0, x0_be, d0_be, 0, 24)
dp1 = encode_dp_v2(0xDEADBEEF, 1, x1_be, d1_be, 1, 24)
dp2 = encode_dp_v2(0xDEADBEEF, 2, x2_be, d2_be, 0, 24)

batch_payload = dp0 + dp1 + dp2                   # 234 bytes

header  = b"KANG"
header += bytes([0x24])                            # type = DP_BATCH_V2
header += bytes([0x00])                            # flags = 0
header += (234).to_bytes(2, "little")              # payload_size = 234 LE

frame = header + batch_payload                     # 8 + 234 = 242 bytes
socket.sendall(frame)
```

The next batch on the same chunk would start at `sequence = 3`.

### 10.3 Reading a `WORK_ASN`

```
header = recv_exactly(8)
assert header[0:4] == b"KANG"
assert header[4]  == 0x11                          # type = WORK_ASN
payload_size = int.from_bytes(header[6:8], "little")
assert payload_size == 109                         # WorkAssignment is fixed-size
payload = recv_exactly(payload_size)

public_key  = payload[0:33]                        # compressed pubkey
range_start = payload[33:65]                       # 32 BE bytes
range_end   = payload[65:97]                       # 32 BE bytes
dp_bits     = int.from_bytes(payload[97:101], "little")
work_id     = int.from_bytes(payload[101:109], "little")
```

---

## 11. Implementation notes for third parties

- **Single source of truth.** Implement against `protocol/jlp.yaml` from the dev tree. The C++ header `jlp_wire_generated.hpp` and the Python module `jlp_protocol_generated.py` are byte-for-byte equivalent and either is acceptable as a cross-check.
- **Codegen.** If you do not have access to the IDL (e.g. you are working from the public Free repo's `jlp_wire_generated.hpp`), the layouts in section 5 are the authoritative wire format and match the C++ header byte-for-byte.
- **Endianness boundary.** The most common implementation bug is sending the `x` and `d` fields in little-endian. They are big-endian on the wire to match Bitcoin's canonical 32-byte big-endian integer convention. The fixed-width integer fields (`work_id`, `sequence`, `dp_bits`, `payload_size`) are little-endian.
- **Padding.** Do not pad. The structs are packed; `WorkAssignment` is exactly 109 bytes, not 112.
- **Strings.** `worker_name` and `password` are null-padded fixed-width fields. They are not null-terminated C strings. Do not strip trailing nulls before comparison; pad your input on the way out and accept padded values on the way in.
- **One in-flight write at a time.** TLS implementations require serialization across `SSL_write` and (in rare cases) `SSL_read`. The reference client uses a `ssl_io_mutex_`; a third-party client should do the same or use a single-threaded I/O model.
- **Frame boundaries.** TCP does not preserve message boundaries. Always read the 8-byte header in full first, then read exactly `payload_size` bytes, then process. Never assume a single `recv()` returns one full frame.

---

## 12. Reference implementations

| Implementation                            | Path                                                                      | Status                                 |
| ----------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------- |
| C++ pool client (theCollider)             | `src/pool/jlp_pool_client.cpp`                                            | Authoritative client.                  |
| C++ generated wire types                  | `src/pool/jlp_wire_generated.hpp`                                         | Generated; do not hand-edit.           |
| Python codegen + generated wire types     | `tools/codegen/jlp_codegen.py`, `data/protocol/jlp_protocol_generated.py` | Generated; do not hand-edit.           |
| Python pool server (`collision-protocol`) | <https://github.com/hevnsnt/collision-protocol>                           | Reference server.                      |
| IDL (single source of truth)              | `protocol/jlp.yaml`                                                       | Edit here, regenerate everything else. |

Wire-format changes MUST update the IDL plus regenerate both bindings plus update the protocol drift round-trip test (`tests/protocol/`) plus update the matching pool-server code in `collision-protocol` in lockstep. The protocol drift test catches silent skew between Python and C++ codegen; CI runs it on every push.

---

## 13. Versioning

| Protocol version | First binary release | Notes                                                                                                                                                      |
| ---------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1`              | v1.2.x               | `DP_SUBMIT` and `DP_BATCH` only; no `work_id` attestation.                                                                                                 |
| `2`              | v1.4.0               | Adds `DP_SUBMIT_V2`, `DP_BATCH_V2`, `WorkAssignment.work_id`. Backward compatible read of v1 DPs.                                                          |
| `2` (v1.4.1)     | v1.4.1               | `DistinguishedPointV2` gains a 4-byte `sequence` field; wire size 74 -> 78. Servers running v1.4.0 reject v1.4.1 DPs (size mismatch); upgrade in lockstep. |

A client SHOULD NOT mix `DP_SUBMIT` (v1) and `DP_SUBMIT_V2` (v2) on the same connection. Pick one and stick with it for the lifetime of the connection.
