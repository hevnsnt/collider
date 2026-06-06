# JLP Wire Protocol Reference

JLP is the wire protocol spoken between theCollider's pool client and a JLP pool server (the reference server is `collision-protocol`, available at <https://collisionprotocol.com>). This document is intended to be sufficient for a third-party implementer to write a conformant client or alternative server.

The IDL source of truth is `protocol/jlp.yaml`. The C++ generated header `src/pool/jlp_wire_generated.hpp` and the Python generated module `data/protocol/jlp_protocol_generated.py` are produced from that YAML by `tools/codegen/jlp_codegen.py`. **Hand-editing the generated files is forbidden.** A third-party client should derive its own bindings from `protocol/jlp.yaml`, not from any single language binding.

The reference C++ client is `src/pool/jlp_pool_client.cpp`; the reference Python codegen lives at `tools/codegen/jlp_codegen.py`.

> Note for readers of the public Free repo: `protocol/jlp.yaml` and `tools/codegen/` are excluded from the public sync. The C++ generated header at `src/pool/jlp_wire_generated.hpp` is the only artifact in the Free repo, and it carries the same byte layout. This document mirrors the IDL exactly.

---

## 1. Protocol version

`protocol_version: 4` (shipped in collider v1.5.4).

The protocol version increments only on breaking wire changes. Backward-compatible additions (new message types, new optional fields) do not bump the version.

### 1.1 Version history and negotiation

| `protocol_version` | Shipped in      | Status                             |
| ------------------ | --------------- | ---------------------------------- |
| `4`                | v1.5.4          | Current.                           |
| `3`                | v1.5.0 to 1.5.3 | Accepted in compatibility mode.    |
| `2`                | v1.4.0 to 1.4.4 | Refused at AUTH (below the floor). |
| `1`                | v1.2.x to 1.4.0 | Refused at AUTH (below the floor). |

The header `flags` byte carries the sender's protocol version. There is a non-configurable security floor of `3`: any client whose `flags` are below `3` (v1.4.x and older) is refused at AUTH with an upgrade-required reason.

The server negotiates the version per connection. At AUTH it reads the client `flags` and sets `negotiated_version = min(client_flags, 4)`, then stamps that version into the `flags` byte of every outbound frame. So a v3 client (flags=3) is answered with flags=3 on every server frame and a zero-payload `AUTH_OK` (the v3 form), while a v4 client (flags=4) is answered with flags=4 and the 324-byte `AuthOkPayload` advert. Senders set `flags` to their own protocol version (`4` for v1.5.4).

The server config key `protocol_mode` selects the floor:

- `compatibility` (default): floor `3`, accepts both v3 and v4 clients.
- `strict`: floor `4`, accepts v4 clients only.

The `protocol_version_mismatch` error (0x10) now applies only below the floor; v3 and v4 are negotiated, not rejected.

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

| Offset | Size | Field          | Type       | Notes                                                                                                                                                                                                                                                 |
| -----: | ---: | -------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|      0 |    4 | `magic`        | `bytes[4]` | Always the four ASCII bytes `K`, `A`, `N`, `G`.                                                                                                                                                                                                       |
|      4 |    1 | `type`         | `uint8`    | One of the message-type constants in section 4.                                                                                                                                                                                                       |
|      5 |    1 | `flags`        | `uint8`    | Protocol version. Senders set their own version (`4` in v1.5.4, `3` for v1.5.0 to 1.5.3). The server negotiates per connection and stamps `negotiated_version = min(client_flags, 4)` into the `flags` byte of every outbound frame. See section 1.1. |
|      6 |    2 | `payload_size` | `uint16`   | Number of payload bytes following the header (LE).                                                                                                                                                                                                    |

Header total: **8 bytes**. Payload follows immediately. Total frame size on the wire is `8 + payload_size`.

`MAX_MESSAGE_SIZE = 1048576` (1 MiB). Frames whose `payload_size` exceeds this cap MUST be rejected before reading the payload.

---

## 4. Message types

| Name            | Code | Direction        | Notes                                                                                                     |
| --------------- | ---- | ---------------- | --------------------------------------------------------------------------------------------------------- |
| `AUTH`          | 0x01 | Client -> Server | First frame after connect.                                                                                |
| `AUTH_OK`       | 0x02 | Server -> Client | Authentication accepted. For a v4 client, carries `AuthOkPayload` (324 B); for a v3 client, zero payload. |
| `AUTH_FAIL`     | 0x03 | Server -> Client | Authentication rejected. Connection closes.                                                               |
| `WORK_REQ`      | 0x10 | Client -> Server | Request a chunk assignment.                                                                               |
| `WORK_ASN`      | 0x11 | Server -> Client | Chunk assignment.                                                                                         |
| `DP_SUBMIT`     | 0x20 | Client -> Server | Single distinguished-point submission (v1, deprecated).                                                   |
| `DP_ACK`        | 0x21 | Server -> Client | Acknowledges a DP submission.                                                                             |
| `DP_BATCH`      | 0x22 | Client -> Server | Batch of v1 DPs (deprecated).                                                                             |
| `DP_SUBMIT_V2`  | 0x23 | Client -> Server | Single v2 DP (rare; clients normally batch).                                                              |
| `DP_BATCH_V2`   | 0x24 | Client -> Server | Batch of v2 DPs. v3 clients submit here; accepted in compatibility mode and never challenged.             |
| `DP_SUBMIT_V3`  | 0x25 | Client -> Server | **v4.** Single v3 DP, carries `DistinguishedPointV3`.                                                     |
| `DP_BATCH_V3`   | 0x26 | Client -> Server | **v4.** `uint32` count + N x `DistinguishedPointV3`.                                                      |
| `STATS_REQ`     | 0x30 | Client -> Server | Optional explicit pull. Server also pushes periodically.                                                  |
| `STATS_RSP`     | 0x31 | Server -> Client | Pool statistics snapshot.                                                                                 |
| `CHALLENGE`     | 0x32 | Server -> Client | **v4.** Checkpoint-replay challenge (variable length). Gated to v4 clients only.                          |
| `CHALLENGE_RSP` | 0x33 | Client -> Server | **v4.** Reveals endpoint checkpoint distances plus Merkle proofs (variable length, hand coded).           |
| `SOLUTION`      | 0x40 | Server -> Client | A solution was found in this pool. Pushed asynchronously.                                                 |
| `PING`          | 0x50 | Client -> Server | Keepalive.                                                                                                |
| `PONG`          | 0x51 | Server -> Client | Keepalive reply.                                                                                          |
| `MAINTENANCE`   | 0x60 | Server -> Client | **v4.** Carries `MaintenancePayload` (262 B); tells the worker to back off and auto resume.               |
| `MSG_ERROR`     | 0xFF | Either           | Out-of-band error.                                                                                        |

The constant is named `MSG_ERROR` rather than `ERROR` because `ERROR` collides with the Windows `<winerror.h>` macro. Server and client implementations should follow the same convention.

Message types marked **v4** are negotiated to v4 clients only. A v3 client never sees `DP_SUBMIT_V3`, `DP_BATCH_V3`, `CHALLENGE`, or `MAINTENANCE`; it submits DPs via `DP_BATCH_V2`, gets a zero-payload `AUTH_OK`, and (if the pool is in maintenance) receives an `AUTH_FAIL` carrying a maintenance note instead of a `MAINTENANCE` frame.

---

## 5. Struct layouts

All structs are packed. Offsets are zero-based from the start of the **payload** (i.e. after the 8-byte header). Struct-level wire sizes are documented and asserted at codegen time.

### 5.1 `AuthPayloadV2` (120 bytes)

Payload of an `AUTH` (0x01) frame in v3 and v4. This is the current AUTH layout.

| Offset | Size | Field          | Type        | Notes                                                                            |
| -----: | ---: | -------------- | ----------- | -------------------------------------------------------------------------------- |
|      0 |   64 | `worker_name`  | `bytes[64]` | UTF-8 worker name. Doubles as the BTC payout address. Null-padded to 64 bytes.   |
|     64 |   32 | `password`     | `bytes[32]` | Optional shared pool password. Null-padded to 32 bytes. Empty if not configured. |
|     96 |    8 | `timestamp_ms` | `uint64`    | Client wall-clock time in milliseconds. LE.                                      |
|    104 |   16 | `nonce`        | `bytes[16]` | Per-AUTH random nonce.                                                           |

Python `struct.pack` format: `<64s32sQ16s`.

`worker_name` MUST be the Bitcoin address that will receive payouts; the server uses this string verbatim as the per-worker credit key.

`timestamp_ms` must be within plus or minus 30 seconds of the server clock (`auth_clock_skew (0x12)` on violation). `nonce` uniqueness is enforced in a short server-side seen-set (`auth_nonce_reuse (0x13)` on replay). The header `flags` byte carries the client's protocol version; a client below the floor of `3` is refused with `protocol_version_mismatch (0x10)`.

### 5.1.1 `AuthPayload` (96 bytes, deprecated)

The legacy v1/v2 AUTH layout was `worker_name[64]` + `password[32]`, Python format `<64s32s`. It carried no timestamp or nonce. Clients sending it (flags below the floor) are refused at AUTH; it is documented only for historical reference.

### 5.2 `WorkAssignment` (126 bytes)

Payload of a `WORK_ASN` (0x11) frame in v3 and v4.

| Offset | Size | Field            | Type        | Notes                                                                              |
| -----: | ---: | ---------------- | ----------- | ---------------------------------------------------------------------------------- |
|      0 |   33 | `public_key`     | `bytes[33]` | Compressed secp256k1 pubkey of the puzzle target (`02...`/`03...`).                |
|     33 |   32 | `range_start`    | `bytes[32]` | Big-endian 32-byte chunk start.                                                    |
|     65 |   32 | `range_end`      | `bytes[32]` | Big-endian 32-byte chunk end.                                                      |
|     97 |    4 | `dp_bits`        | `uint32`    | Distinguished-point bit threshold. LE. Validated `8 <= dp_bits <= 32`.             |
|    101 |    8 | `work_id`        | `uint64`    | Low 64 bits of the chunk identifier. LE.                                           |
|    109 |    1 | `kangaroo_type`  | `uint8`     | `1` = TAME_ONLY, `2` = WILD_ONLY. `0` = BOTH is reserved and illegal in pool mode. |
|    110 |    8 | `start_offset_a` | `uint64`    | Inclusive low 64 bits of the worker's offset window. LE.                           |
|    118 |    8 | `start_offset_b` | `uint64`    | Exclusive upper bound (`b > a`). LE.                                               |

Python `struct.pack` format: `<33s32s32sIQBQQ`.

The C++ struct in `src/pool/jlp_pool_client.hpp` is named `JLPServerConfig` for historical reasons. The Python side names it `WorkAssignment`. The wire layout is identical.

`work_id` is what the client must echo back in every DP submission for this chunk (see `DistinguishedPointV2` / `DistinguishedPointV3`).

The trailing three fields were added in v3 for asymmetric TAME-only / WILD-only assignment. `kangaroo_type` tracks the RCKangaroo `KANG_MODE_*` enum (`third_party/RCKangaroo/defs.h`) and is distinct from the per-DP `type` byte. `[start_offset_a, start_offset_b)` is the worker's disjoint sub-window inside `[range_start, range_end)`; the server guarantees no two same-type workers overlap. A client that receives `dp_bits` outside `8..32` MUST disconnect (anti-DoS guard). The wire size is identical in v3 and v4.

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

**Actual replay-defence scheme (v1.4.2).** The reference server (`collision-protocol/src/pool_server.py`, `_check_dp_sequence`) implements the window as: per `(worker_name, work_id)` it tracks `_dp_seq_high`, the highest sequence ever accepted. A new submission is accepted iff `sequence > _dp_seq_high - SEQ_REPLAY_WINDOW` (with `SEQ_REPLAY_WINDOW = 1024`); otherwise it is rejected as `dp_sequence_out_of_window (0x21)`. The server does NOT separately track every individual seen sequence, so:

- **An attacker who replays a captured DP_BATCH_V2 with sequences within 1024 of the current high-water mark is NOT rejected by this check alone.** The server's cryptographic consistency check on the DP fields (X / d / type) is the actual defence against replayed real DPs; the sequence window only defends against very-old captures.
- **A client that crashes mid-batch and restarts MUST resume its counter from the last persisted value, not from 0.** Reusing earlier sequence values within the window does not collide (server admits `sequence > high - 1024`) but using values below the floor triggers `dp_sequence_out_of_window`. Persisting the counter across restarts is the client-side responsibility implemented by `PoolManager` (v1.4.2 Pool-B1, `~/.collider/pool_dp_seq.dat`).
- **Out-of-order DPs within the window slip through the same-sequence check.** This is a known limitation. The v1.4.2 dependency map: server's protocol-level guard is the lookback bound; the cryptographic-consistency check is what actually rejects forged DPs; the client-side persistence prevents accidental self-banning on flaky networks.

The pre-1.4.2 client header `src/pool/dp_seq_window.hpp` documented a stricter per-`(work_id)` set-of-seen-sequences scheme. Neither end implemented that scheme; it was a stale spec. The header has been deleted.

**`work_id` rules.** Clients MUST set `work_id` to the most recent `WorkAssignment.work_id` for the chunk this DP came from. Submitting a DP whose `work_id` does not match the worker's currently-assigned chunk is treated as an anti-cheat infraction (see section 9).

**`type` rules.** The per-DP `type` must match the worker's assigned `kangaroo_type` (TAME_ONLY workers emit `type=0` only; WILD_ONLY workers emit `type=1` only). As of v1.5.4 a wrong-type DP is a **recoverable event**, not cheating and not a ban: the server rejects the DP and asks the worker to re request work (the stale `work_id` path). Only repeated stale or wrong submissions beyond a disconnect limit cause a clean disconnect that forces a fresh AUTH and WORK_REQ, never a permanent IP ban. An epoch race where a worker is reassigned between tame and wild mid flight can emit a stale-type DP with no malice, so banning on it was incorrect (see section 9).

### 5.4.1 `DistinguishedPointV3` (114 bytes)

Payload of a `DP_SUBMIT_V3` (0x25) frame, or one element of a `DP_BATCH_V3` (0x26) array. Introduced in v4 (v1.5.4).

| Offset | Size | Field        | Type        | Notes                                                                                                        |
| -----: | ---: | ------------ | ----------- | ------------------------------------------------------------------------------------------------------------ |
|      0 |    8 | `work_id`    | `uint64`    | Chunk id from the assigned `WorkAssignment`. LE.                                                             |
|      8 |    4 | `sequence`   | `uint32`    | Per-`(worker, work_id)` monotonic counter, starts at 0. LE.                                                  |
|     12 |   32 | `x`          | `bytes[32]` | Big-endian X coordinate.                                                                                     |
|     44 |   32 | `d`          | `bytes[32]` | Big-endian walked distance (signed; sign-extended into 32B).                                                 |
|     76 |    1 | `type`       | `uint8`     | `0` = tame, `1` = wild.                                                                                      |
|     77 |    1 | `dp_bits`    | `uint8`     | Leading-zero bit count required of `x`.                                                                      |
|     78 |   32 | `ckpt_root`  | `bytes[32]` | Merkle root over the walk's checkpoint distances (one checkpoint every `CHECKPOINT_INTERVAL = 65536` jumps). |
|    110 |    4 | `n_segments` | `uint32`    | Committed segment count (`checkpoint_count - 1`). LE.                                                        |

Python `struct.pack` format: `<QI32s32sBB32sI`.

`DistinguishedPointV3` is a v2 superset plus the checkpoint-walk commitment: the first six fields are byte-identical to `DistinguishedPointV2`, and `ckpt_root` / `n_segments` carry the proof-of-walk commitment that the `CHALLENGE` machinery verifies. In compatibility mode the server also accepts the older `DP_BATCH_V2` frames and credits them identically; a v2 frame is simply never challenged. A `DP_BATCH_V3` is `uint32 count` + N x `DistinguishedPointV3`, wire size `4 + count * 114` bytes, capped at `MAX_BATCH_SIZE = 10000`.

### 5.4.2 `AuthOkPayload` (324 bytes)

Payload of an `AUTH_OK` (0x02) frame sent to a v4 client. A v3 client receives a zero-payload `AUTH_OK` (the v3 form), so the advert is fully backward compatible.

| Offset | Size | Field            | Type         | Notes                                                                              |
| -----: | ---: | ---------------- | ------------ | ---------------------------------------------------------------------------------- |
|      0 |   16 | `latest_version` | `bytes[16]`  | ASCII semver of the latest client, null-padded (e.g. `"1.5.4"`).                   |
|     16 |   16 | `min_version`    | `bytes[16]`  | ASCII semver of the minimum supported client.                                      |
|     32 |    1 | `flags`          | `uint8`      | bit0 = `update_available`, bit1 = `maintenance_active`.                            |
|     33 |    3 | `reserved`       | `bytes[3]`   | Reserved.                                                                          |
|     36 |  256 | `download_url`   | `bytes[256]` | HTTPS URL of the latest signed binary, null-padded; all-zero disables auto update. |
|    292 |   32 | `sha256`         | `bytes[32]`  | Raw SHA-256 of the binary at `download_url`.                                       |

Python `struct.pack` format: `<16s16sB3s256s32s`.

This drives in-band client auto update. The client compares its own version to `latest_version` and, if older and a `download_url` is present, fetches the binary over HTTPS, verifies it against `sha256`, then replaces itself and relaunches.

### 5.4.3 `MaintenancePayload` (262 bytes)

Payload of a `MAINTENANCE` (0x60) frame. Introduced in v4.

| Offset | Size | Field              | Type         | Notes                                                                |
| -----: | ---: | ------------------ | ------------ | -------------------------------------------------------------------- |
|      0 |    1 | `active`           | `uint8`      | `1` = maintenance in effect (back off), `0` = all clear (resume).    |
|      1 |    1 | `reserved`         | `uint8`      | Reserved.                                                            |
|      2 |    4 | `retry_after_secs` | `uint32`     | Suggested base reconnect backoff in seconds. LE; client adds jitter. |
|      6 |  256 | `message`          | `bytes[256]` | Operator note, null-padded ASCII.                                    |

Python `struct.pack` format: `<BBI256s`.

The server sends this after `AUTH_OK` to a worker that connects while the pool is in maintenance, or broadcasts it to live workers when an operator toggles maintenance on. The client shows the note and backs off gracefully, then auto resumes. A v3 client (which has no 0x60 frame) instead receives an `AUTH_FAIL` carrying a maintenance note.

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
  |---- AUTH (flags=4, AuthPayloadV2) ------>|
  |     (within AUTH_TIMEOUT_SECS=30)        |
  |                                          |
  |<--- AUTH_OK (AuthOkPayload advert for    |
  |     a v4 client; zero payload for v3)    |
  |     -or- AUTH_FAIL                       |
  |                                          |
  |<--- MAINTENANCE (if pool is paused) -----|
  |                                          |
  |---- WORK_REQ ---------------------------->|
  |                                          |
  |<--- WORK_ASN ----------------------------|
  |                                          |
  |  (compute on assigned chunk)             |
  |                                          |
  |---- DP_BATCH_V3 (v4) / DP_BATCH_V2 (v3) >|
  |<--- DP_ACK ------------------------------|
  |                                          |
  |<--- CHALLENGE (v4 only, when enabled) ---|
  |---- CHALLENGE_RSP ---------------------->|
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

At AUTH the server negotiates `negotiated_version = min(client_flags, 4)` and stamps it into every outbound frame (section 1.1). It then replies with one of:

- `AUTH_OK` (0x02). For a v4 client this carries `AuthOkPayload` (324 B, the update advert); for a v3 client it is a zero-payload frame. The connection is now in the "authenticated" state.
- `AUTH_FAIL` (0x03). The connection MUST be closed by the server. Clients SHOULD bound their reconnect attempts (the reference client uses jittered exponential backoff). A client below the security floor of `3` (v1.4.x and older) is refused here with an upgrade-required reason.

If the pool is in maintenance, a v4 client receives a `MAINTENANCE` (0x60) frame after `AUTH_OK` and backs off; a v3 client instead receives an `AUTH_FAIL` carrying a maintenance note.

Until `AUTH_OK` is received, the client MUST NOT send `WORK_REQ`, `DP_*`, or `STATS_REQ`. The reference server's state machine rejects any of those before AUTH and treats them as anti-cheat infractions.

### 6.3 WORK_REQ / WORK_ASN

After `AUTH_OK`, the client sends `WORK_REQ` (empty payload) and the server replies with a `WorkAssignment` payload. The chunk is now bound to this `(worker_name, work_id)` pair until the client signals completion (next `WORK_REQ` after sending the last DP of the previous chunk) or the server reissues it after a stale-chunk timeout.

A client MAY hold only one outstanding chunk at a time. Multiple parallel chunks per connection are not supported.

### 6.4 DP submission

A v4 client batches DPs in `DP_BATCH_V3` (0x26) frames, each containing 1 to `MAX_BATCH_SIZE = 10000` `DistinguishedPointV3` records (`payload_size = 4 + count * 114`). A v3 client batches in `DP_BATCH_V2` (0x24) frames of `DistinguishedPointV2` records (`payload_size = 4 + count * 78`); these are accepted in compatibility mode and never challenged. Records are packed end-to-end with no separators.

The reference client debounces submissions on a fixed cadence (typically every few seconds) and flushes immediately when the batch fills up.

The server replies with `DP_ACK`, which echoes the count of accepted DPs.

### 6.4.1 Checkpoint-replay challenge (v4)

A `DistinguishedPointV3` carries a `ckpt_root` Merkle commitment over the walk's checkpoint distances. The server may issue a `CHALLENGE` (0x32) for a random subset of segments; the worker replies with `CHALLENGE_RSP` (0x33) revealing the endpoint checkpoint distances plus their Merkle proofs so the server can replay the forward jumps and confirm the segment links up. The challenge is gated to v4 clients only; a v3 client is never challenged. The verifier ships off (`challenge_mode: off`; values are `off`, `shadow`, `enforce`) and is enabled only after it is validated against a real walk, so honest workers are never falsely penalized during the rollout.

### 6.5 Keepalive

Clients SHOULD send `PING` every `KEEPALIVE_SECS = 20` seconds when no other client-to-server traffic is in flight. The server replies with `PONG`. Both frames have empty payloads.

A client that has not sent any frame for several `KEEPALIVE_SECS` periods may be disconnected by the server.

### 6.6 Server-pushed messages

`STATS_RSP` and `SOLUTION` are pushed by the server asynchronously. Clients MUST be prepared to receive them at any time after `AUTH_OK`. The reference client decouples its read loop from its write loop for exactly this reason.

`SOLUTION` payload format is documented elsewhere (it carries the recovered private key plus the `work_id` that produced it). When a `SOLUTION` is received, the client SHOULD treat the current pool round as over and either disconnect or wait for a new `WorkAssignment`.

---

## 7. AUTH replay protection

The server rejects replay-style AUTH attempts (the same `AUTH` payload bytes captured and replayed from another IP). As of v3 the `AuthPayloadV2` layout (section 5.1) carries dedicated `timestamp_ms` and `nonce` fields for this:

- `timestamp_ms` must be within plus or minus 30 seconds of the server clock, otherwise the AUTH is rejected with `auth_clock_skew (0x12)`.
- `nonce` is a per-AUTH random 16-byte value. The server keeps a short seen-set; a replayed nonce is rejected with `auth_nonce_reuse (0x13)`.

A captured AUTH frame replayed later fails the clock-skew bound, and one replayed quickly fails the nonce check, so submitting captured AUTH bytes from another IP is rejected. The reference C++ client implementation in `src/pool/jlp_pool_client.cpp` is the authoritative example.

---

## 8. Anti-cheat rules

A client MUST follow these rules to remain unbanned. The thresholds below are the reference server's defaults (from `protocol/jlp.yaml::anti_cheat`); a third-party server may tighten them.

| Rule                                     | Threshold                                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Invalid DPs per IP                       | 100 per `invalid_dp_window_secs = 3600` (1 hour).                                                |
| `work_id` mismatch                       | Counts as an invalid DP.                                                                         |
| Out-of-window `sequence`                 | Counts as an invalid DP.                                                                         |
| Cryptographically inconsistent X / d     | Counts as an invalid DP.                                                                         |
| Wrong-type DP (tame under WILD_ONLY etc) | Recoverable, not a ban. Server rejects the DP and asks the worker to re request work. See below. |
| AUTH after the 30-second window          | Disconnect, no ban (transient).                                                                  |
| Sending non-AUTH frames before `AUTH_OK` | Counts as an invalid DP-equivalent infraction.                                                   |

Ban escalation, per IP, within `ban_count_window_secs = 2592000` (30 days):

1. First infraction: 1 hour (`3600` sec).
2. Second: 6 hours (`21600` sec).
3. Third: 1 day (`86400` sec).
4. Fourth: 7 days (`604800` sec).
5. Fifth and beyond: permanent.

Old infraction records are pruned after `invalid_dp_retention_secs = 1209600` (14 days).

**Type mismatch is recoverable (v1.5.4).** A worker submitting a wrong-type DP under a TAME_ONLY / WILD_ONLY assignment is no longer banned. The server rejects the DP and asks the worker to re request work (the stale `work_id` path); only repeated stale or wrong submissions beyond a disconnect limit cause a clean disconnect that forces a fresh AUTH and WORK_REQ, never a permanent IP ban. An epoch race where a worker is reassigned between tame and wild mid flight can emit a stale-type DP with no malice, so banning on it was incorrect.

**Per-client-version policy.** Every worker gets the asymmetric tame/wild split plus DP shadow verification (`dp_verify_mode` defaults to `shadow`: the server samples DPs for a cryptographic re-check at random; values are `off`, `shadow`, `enforce`). v4 workers are additionally eligible for the checkpoint-replay challenge (section 6.4.1) when it is enabled; v3 workers are never challenged.

A reasonable client never trips any of these rules in normal operation. The thresholds exist for actively-misbehaving (or buggy) clients.

---

## 9. Constants summary

These constants are defined in the IDL (`protocol/jlp.yaml::constants`) and reproduced in both generated bindings.

| Name                  | Value     | Description                                                                            |
| --------------------- | --------- | -------------------------------------------------------------------------------------- |
| `MAX_MESSAGE_SIZE`    | `1048576` | 1 MiB hard cap per frame, enforced before payload read.                                |
| `MAX_BATCH_SIZE`      | `10000`   | Maximum DPs per `DP_BATCH`, `DP_BATCH_V2`, or `DP_BATCH_V3` frame.                     |
| `AUTH_TIMEOUT_SECS`   | `30`      | Worker must send AUTH within this window.                                              |
| `KEEPALIVE_SECS`      | `20`      | Recommended client PING frequency.                                                     |
| `CHECKPOINT_INTERVAL` | `65536`   | Jumps between checkpoint distances committed in `DistinguishedPointV3.ckpt_root` (v4). |

---

## 10. Worked examples

### 10.1 Building an `AUTH` frame

Pseudo-code, building the `AuthPayloadV2` frame for `worker_name = "1MyBtcAddress..."` with no password. A v4 client sets `flags = 4`; a v3 client sets `flags = 3`.

```
import os, time

worker_name_bytes = utf8("1MyBtcAddress...").ljust(64, b"\x00")    # exactly 64 bytes
password_bytes    = b"\x00" * 32                                    # exactly 32 bytes
timestamp_ms      = int(time.time() * 1000).to_bytes(8, "little")   # 8 bytes LE
nonce             = os.urandom(16)                                  # 16 bytes
auth_payload      = (worker_name_bytes + password_bytes
                     + timestamp_ms + nonce)                        # 120 bytes total

header  = b"KANG"                          # magic
header += bytes([0x01])                    # type = AUTH
header += bytes([0x04])                    # flags = 4 (protocol version)
header += (120).to_bytes(2, "little")      # payload_size = 120 LE

frame = header + auth_payload              # 8 + 120 = 128 bytes

socket.sendall(frame)
```

### 10.2 Building a `DP_BATCH_V2` frame (v3 client)

Building a batch of three `DistinguishedPointV2` records for the assigned `work_id = 0xDEADBEEF`, starting at `sequence = 0`. A v3 client sets `flags = 3`.

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

count_prefix  = (3).to_bytes(4, "little")          # 0..3 (DP count)
batch_payload = count_prefix + dp0 + dp1 + dp2     # 4 + 3*78 = 238 bytes

header  = b"KANG"
header += bytes([0x24])                            # type = DP_BATCH_V2
header += bytes([0x03])                            # flags = 3 (protocol version)
header += (238).to_bytes(2, "little")              # payload_size = 238 LE

frame = header + batch_payload                     # 8 + 238 = 246 bytes
socket.sendall(frame)
```

The next batch on the same chunk would start at `sequence = 3`.

### 10.2.1 Building a `DP_BATCH_V3` frame (v4 client)

Same shape, but each record is a 114-byte `DistinguishedPointV3` carrying the checkpoint-walk commitment, and the frame type is `DP_BATCH_V3` (0x26) with `flags = 4`.

```
def encode_dp_v3(work_id, seq, x_be32, d_be32, dp_type, dp_bits,
                 ckpt_root_32, n_segments):
    return (
        work_id.to_bytes(8, "little") +           # 0..7
        seq.to_bytes(4, "little") +               # 8..11
        x_be32 +                                  # 12..43 (32 BE bytes)
        d_be32 +                                  # 44..75 (32 BE bytes)
        bytes([dp_type, dp_bits]) +               # 76..77
        ckpt_root_32 +                            # 78..109 (32-byte Merkle root)
        n_segments.to_bytes(4, "little")          # 110..113
    )
    # = 114 bytes total

dp0 = encode_dp_v3(0xDEADBEEF, 0, x0_be, d0_be, 0, 24, root0, n0)

count_prefix  = (1).to_bytes(4, "little")          # 0..3 (DP count)
batch_payload = count_prefix + dp0                 # 4 + 1*114 = 118 bytes

header  = b"KANG"
header += bytes([0x26])                            # type = DP_BATCH_V3
header += bytes([0x04])                            # flags = 4 (protocol version)
header += (118).to_bytes(2, "little")              # payload_size = 118 LE

frame = header + batch_payload                     # 8 + 118 = 126 bytes
socket.sendall(frame)
```

### 10.3 Reading a `WORK_ASN`

```
header = recv_exactly(8)
assert header[0:4] == b"KANG"
assert header[4]  == 0x11                          # type = WORK_ASN
payload_size = int.from_bytes(header[6:8], "little")
assert payload_size == 126                         # WorkAssignment is fixed-size (v3 / v4)
payload = recv_exactly(payload_size)

public_key     = payload[0:33]                     # compressed pubkey
range_start    = payload[33:65]                    # 32 BE bytes
range_end      = payload[65:97]                    # 32 BE bytes
dp_bits        = int.from_bytes(payload[97:101], "little")
work_id        = int.from_bytes(payload[101:109], "little")
kangaroo_type  = payload[109]                      # 1=TAME_ONLY, 2=WILD_ONLY
start_offset_a = int.from_bytes(payload[110:118], "little")
start_offset_b = int.from_bytes(payload[118:126], "little")
```

---

## 11. Implementation notes for third parties

- **Single source of truth.** Implement against `protocol/jlp.yaml` from the dev tree. The C++ header `jlp_wire_generated.hpp` and the Python module `jlp_protocol_generated.py` are byte-for-byte equivalent and either is acceptable as a cross-check.
- **Codegen.** If you do not have access to the IDL (e.g. you are working from the public Free repo's `jlp_wire_generated.hpp`), the layouts in section 5 are the authoritative wire format and match the C++ header byte-for-byte.
- **Endianness boundary.** The most common implementation bug is sending the `x` and `d` fields in little-endian. They are big-endian on the wire to match Bitcoin's canonical 32-byte big-endian integer convention. The fixed-width integer fields (`work_id`, `sequence`, `dp_bits`, `payload_size`) are little-endian.
- **Padding.** Do not pad. The structs are packed; `WorkAssignment` is exactly 126 bytes (v3 / v4), `DistinguishedPointV3` is exactly 114 bytes.
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

| Protocol version | First binary release | Notes                                                                                                                                                                                                                                                                                                                                                             |
| ---------------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `1`              | v1.2.x               | `DP_SUBMIT` and `DP_BATCH` only; no `work_id` attestation.                                                                                                                                                                                                                                                                                                        |
| `2`              | v1.4.0               | Adds `DP_SUBMIT_V2`, `DP_BATCH_V2`, `WorkAssignment.work_id`. Backward compatible read of v1 DPs.                                                                                                                                                                                                                                                                 |
| `2` (v1.4.1)     | v1.4.1               | `DistinguishedPointV2` gains a 4-byte `sequence` field; wire size 74 -> 78. Servers running v1.4.0 reject v1.4.1 DPs (size mismatch); upgrade in lockstep.                                                                                                                                                                                                        |
| `3`              | v1.5.0 to 1.5.3      | `AuthPayloadV2` (120 B) adds `timestamp_ms` + `nonce`. `WorkAssignment` grows 109 -> 126 B with `kangaroo_type` + `start_offset_a/b` for asymmetric TAME-only / WILD-only assignment. `SOLUTION` is strictly server to client. v1.4.x and older are refused at AUTH (security floor of 3).                                                                        |
| `4`              | v1.5.4               | `AUTH_OK` carries `AuthOkPayload` (324 B) update advert (in-band client auto update). Adds `MAINTENANCE` (262 B), `DP_SUBMIT_V3` / `DP_BATCH_V3` (114 B per DP) with a checkpoint-walk Merkle commitment, and `CHALLENGE` / `CHALLENGE_RSP` (proof-of-walk). Per-connection version negotiation keeps v3 (v1.5.0 to 1.5.3) clients working in compatibility mode. |

A client SHOULD NOT mix DP variants (`DP_SUBMIT` v1, `DP_SUBMIT_V2`, `DP_SUBMIT_V3`) on the same connection. Pick the one matching the negotiated version and stick with it for the lifetime of the connection.
