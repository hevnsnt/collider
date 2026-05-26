# Pool Mode Operator Guide

How to join the Collision Protocol pool, what your worker does on the wire, how share-of-pool credit accrues, and what counts as good worker etiquette. This document is for operators running theCollider in pool mode and want more than the README's quick-start gives.

For the wire format itself (frame layouts, byte offsets, struct sizes), see [JLP-PROTOCOL.md](JLP-PROTOCOL.md). This document is the operator-facing companion.

> **v1.5 worker note.** Starting in v1.5, the pool assigns you ONE side (TAME or WILD) on each connect. You do not pick. The pool round-robins across connections so the network stays balanced. You never see the puzzle's private key in pool mode; the algorithm itself denies any single worker the data needed to compute it. Payouts arrive at the Bitcoin address in your `--worker` argument after the operator triggers them through the admin payout UI. See [`v1.5 migration guide`](MIGRATION-v1.5.md) for the upgrade procedure; v1.4.x clients are refused by v1.5 servers with `AUTH_FAIL: UPGRADE_REQUIRED`.

---

## Table of contents

- [Why a pool exists](#why-a-pool-exists)
- [Joining the public pool in 30 seconds](#joining-the-public-pool-in-30-seconds)
- [What your worker does on the wire](#what-your-worker-does-on-the-wire)
- [Share-of-pool accrual and payout](#share-of-pool-accrual-and-payout)
- [Picking a good worker address](#picking-a-good-worker-address)
- [Telemetry and monitoring your worker](#telemetry-and-monitoring-your-worker)
- [Reconnect, suspend, resume](#reconnect-suspend-resume)
- [Worker etiquette and anti-cheat](#worker-etiquette-and-anti-cheat)
- [Running a non-default pool](#running-a-non-default-pool)
- [Troubleshooting](#troubleshooting)
- [Where to go next](#where-to-go-next)

---

## Why a pool exists

Pollard's Kangaroo recovers a private key from a pubkey plus a search range in roughly the square root of the range size group operations. For puzzle #135 (a 135-bit range), that is roughly `2^67.5` group operations. A single RTX 4090 doing 8 GKeys/s would, in isolation, take roughly 730 years to cover that work. The math is tractable; the wall-clock time for any one machine is not.

The kangaroo algorithm has a useful structural property: tame and wild walks deposit "distinguished points" (DPs, points whose X coordinate has a configurable number of leading zero bits) into a shared store. A collision between any tame and any wild DP, contributed by any worker, recovers the key. The work parallelizes cleanly: every worker contributes DPs to a common pool, and any collision finishes the puzzle.

The Collision Protocol pool is the shared DP store plus the orchestration around it: chunk assignment, anti-cheat verification, share-of-pool accounting, and the broadcast when a solution is reconstructed.

---

## Joining the public pool in 30 seconds

```bash
./collider --pool jlps://collisionprotocol.com:17403 \
           --worker 1YourBitcoinAddressForRewards
```

That is the entire setup. The first time you connect successfully, guided mode persists your worker address to `~/.collider/config.yml` so you do not have to repeat it on every launch.

Required:

- `--pool <url>` (or `pool.url:` in `config.yml`). Use `jlps://` for TLS (recommended for any pool you do not host yourself).
- `--worker <addr>` (or `pool.worker:` in `config.yml`). A real Bitcoin address that you control. **When the pool solves a puzzle, your share of the reward is sent to exactly this address.** It is also your identity on the pool for share-of-pool accrual. See [Picking a good worker address](#picking-a-good-worker-address) below for how to validate you actually control it.

Not required:

- `--pool-password`. The public pool ignores it. Private pools that throttle by password set their own value.
- `--pool-api-key`. Only used by HTTP-only pool variants.

---

## What your worker does on the wire

Once connected, the client runs the following loop. The full state machine is documented in [JLP-PROTOCOL.md](JLP-PROTOCOL.md) section 6.

```
1. TCP / TLS handshake.
2. Send AUTH (within 30 seconds of connect) -> AUTH_OK or AUTH_FAIL.
3. Send WORK_REQ -> receive WORK_ASN
     (v1.5: pubkey + range + dp_bits + work_id + kangaroo_type + start_offset_a + start_offset_b).
4. Compute kangaroo on the chunk. v1.5 workers run ONLY tame OR ONLY wild
   kangaroos based on the type field; the host-side cross-collision detection is
   disabled in pool mode. Whenever a DP is produced, queue it.
5. Every few seconds, flush queued DPs as a DP_BATCH_V2 frame -> receive DP_ACK.
6. Every 20 seconds with no other traffic, send PING -> receive PONG.
7. On chunk completion (the kangaroo finishes its assigned range): send WORK_REQ again.
8. Eventually: server pushes SOLUTION when any TAME worker's DPs collide with any
   WILD worker's DPs in the pool aggregator. The puzzle private key is computed
   on the server, never on a worker.
```

Submitting a DP whose type does not match your assigned `kangaroo_type` is treated as binary modification and results in a permanent IP ban on first occurrence (audit finding P2). A well-behaved v1.5 client cannot trigger this; the type bit is set automatically by the backend at `WORK_ASN` time.

The connection is long-lived. theCollider holds one socket for the full session and multiplexes the read and write loops behind a TLS-safe mutex. Reconnects are jittered and bounded (3 attempts on `AUTH_FAIL`, exponential backoff with a unified cap).

Two anti-cheat properties of the wire are worth understanding as an operator:

1. **`work_id` attestation.** Every DP your client submits carries the `work_id` of the chunk it came from. The server rejects DPs whose `work_id` does not match the worker's currently-assigned chunk. You cannot collect DPs off a previous chunk, abandon it, and resubmit them later under a new chunk; the server detects the mismatch and counts it as an infraction.
2. **Per-DP `sequence` nonce.** Every DP has a 4-byte monotonic counter, per `(worker, work_id)`, starting at 0. The server tracks a sliding window of expected sequence numbers and rejects sequences far below the high-water mark as replays. Capturing a `DP_BATCH_V2` frame from your own client and replaying it later does not work; the second submission is dropped at the window check.

Neither rule bites a well-behaved client. Both exist because misbehaving clients (or buggy custom clients) have historically tried both.

---

## Share-of-pool accrual and payout

The pool credits DPs to the `worker_name` field in your AUTH payload. That field is the literal Bitcoin address you authenticated with; the server uses the string verbatim as the per-worker accrual key.

> The same `worker_name` (Bitcoin address) summed across every machine you run is the credit unit. If you have a desktop on `--worker bc1qABC...` and a Mac laptop on `--worker bc1qABC...`, both feed the same accrual bucket. If they use different addresses, they are two separate workers with two separate buckets.

`STATS_RSP` frames (pushed by the server periodically) report:

| Field            | Meaning                                                                |
| ---------------- | ---------------------------------------------------------------------- |
| `total_dps`      | Cumulative pool DP count, across every worker, since pool inception.   |
| `total_workers`  | Distinct worker names ever seen.                                       |
| `active_workers` | Workers active in the last 5 minutes.                                  |
| `dps_per_second` | Pool aggregate DP rate right now.                                      |
| `your_share`     | Your share of `total_dps` as a fraction `[0, 1]`.                      |
| `your_dps`       | DPs credited to your worker_name (summed across all your connections). |
| `uptime_seconds` | How long the server has been up.                                       |

`your_share` is the field operators look at. It is the fraction of the pool's accumulated work attributable to your Bitcoin address.

### Payout policy

Payout is server-side and outside the scope of this document. The reference Collision Protocol policy is "proportional credit at puzzle solve time, paid to the worker address on file" but exact rates, fees, vesting, and dispute resolution are owned by the pool operator. For the current public-pool policy, see [collisionprotocol.com](https://collisionprotocol.com).

What the client guarantees:

- Every DP your worker submits, accepted by the server with a `DP_ACK`, is credited to your `worker_name`.
- The server's `STATS_RSP` is the authoritative count.
- Your client logs the rolling `your_dps` and `your_share` every stats interval; you can verify against the pool's public dashboard.

What the client does not guarantee:

- That a solve will happen in a useful timeframe.
- That a partial-credit payout will be processed against a specific schedule.
- That the pool will not change its payout policy. Operators should read the pool's terms before committing significant hash time.

---

## Picking a good worker address

This is the single most important decision you will make as a pool operator. **The address you put in `--worker` is the address the pool will send your share of the reward to when a puzzle is solved.** If you pick an address whose private key you do not hold, your share is unrecoverable and there is nothing the pool operator can do about it. Get this right before you start submitting DPs.

The `worker_name` field is 64 bytes on the wire, null-padded if shorter. Any valid Bitcoin address fits, including:

- Legacy P2PKH (`1...`, ~34 chars).
- P2SH (`3...`, ~34 chars).
- Native SegWit bech32 (`bc1q...`, ~42 chars).
- Taproot bech32m (`bc1p...`, ~62 chars).

All four work on the wire. The server does not validate the address format (so a typo will not bounce AUTH); it just uses the string as a key.

### Validating that you control the address

Before you use an address as your worker, confirm you can spend from it. The pool cannot do this for you; the only thing it sees is a string.

**Use an address from a wallet you own and run yourself.** Acceptable sources include:

- Bitcoin Core (you control the wallet file plus passphrase).
- A hardware wallet you physically possess (Ledger, Trezor, Coldcard, BitBox).
- A self-custodied software wallet (Electrum, Sparrow, BlueWallet, Wasabi, Samourai) where you hold the seed phrase.

**Do not use:**

- A deposit address copied from an exchange (Coinbase, Binance, Kraken). Exchanges rotate deposit addresses, treat them as ephemeral routing hints, and may not credit a future payout to one. Some exchanges will refuse "unexpected" inbound transactions outright.
- An address you do not have a wallet file or seed phrase for. "I generated it on a website once" is not control. "I have it on a piece of paper from years ago" is not control unless you have tested spending from it.
- An address from a custodial brain-wallet service or vanity-address generator that retained the private key.

**The definitive test:** send a tiny amount of BTC (a few thousand sats) to the address, then send it back out to a different address you control. If the outbound transaction confirms, you have proven you can sign for the address. If you cannot complete that round trip, do not use the address as your worker.

Practical follow-ups:

- Use a dedicated address that you do not share with hot wallets or exchanges. If the pool ever has to coordinate a payout, you do not want it landing in a custodial wallet that has rotated its deposit addresses.
- Treat the BTC address as a worker identity, not a secret. It is sent in plaintext over TLS to the pool and shows up on the public stats dashboard (if the pool runs one). The address being public does not affect your ability to spend from it.
- Once chosen, do not change it casually. Switching addresses mid-puzzle splits your accrual across two buckets, and the pool's payout policy is unlikely to merge them retroactively.
- If you lose access to the wallet behind the address (drive failure, lost seed phrase), your accrued share becomes unrecoverable the same way a regular lost wallet is unrecoverable. Back up your seed phrase before you commit serious hash time.

---

## Telemetry and monitoring your worker

theCollider prints rolling telemetry to the terminal: GPU step rate, DPs submitted, DPs acknowledged, current chunk, and the last `STATS_RSP` snapshot.

For headless operation:

```bash
./collider --pool jlps://collisionprotocol.com:17403 \
           --worker bc1qYourBtcAddress \
           --verbose 2>&1 | tee -a collider.log
```

`--verbose` adds per-DP submission lines (useful for debugging, noisy for steady-state). `--debug` adds the resolved configuration dump at startup so you can see exactly which config / CLI fields took effect.

For a quick health check without watching the terminal: the pool's public dashboard (when the operator runs one) shows `your_share` and `your_dps` keyed by your Bitcoin address. The dashboard is the same data the server pushes to your client; it is just a read-only HTML view.

---

## Reconnect, suspend, resume

The pool client is designed to survive flaky networks without operator intervention.

- **TLS or TCP drop.** The client reconnects with jittered exponential backoff up to the unified `MAX_RECONNECT_BACKOFF_MS` cap. After reconnect plus AUTH, the server reissues your in-flight chunk (matched on worker name).
- **`AUTH_FAIL`.** Bounded 3-attempt retry, then the client exits with a clear error. `AUTH_FAIL` typically means the server rejected your AUTH bytes (clock drift, bad worker name format, IP ban). Check the server's anti-cheat status; see [JLP-PROTOCOL.md](JLP-PROTOCOL.md) section 8.
- **Local crash mid-chunk.** On restart, the client requests a fresh chunk. Any DPs you computed but did not flush before the crash are lost. DPs that the server already `DP_ACK`-ed are credited.
- **Suspend and resume (laptop closes lid).** TLS sessions do not survive system suspend. On resume, the client detects the dead socket on next write, reconnects, and resumes work. Expect a one-time chunk reissue.

For deliberate suspension (you want to stop, then resume later without losing your accrual):

- Stop the client with SIGINT (Ctrl+C) or SIGTERM. The client flushes any queued DPs before exit (best-effort) and closes the socket cleanly.
- On restart with the same `--worker`, the server treats you as the same worker. Your accrued `your_dps` is preserved on the server side and continues from where it left off.

There is no client-side state that needs to persist for pool resumption to work; the server is the authority for what you have contributed.

---

## Worker etiquette and anti-cheat

The full anti-cheat policy is in [JLP-PROTOCOL.md](JLP-PROTOCOL.md) section 8. Operator-facing summary:

| Behavior                                                           | Cost                                   |
| ------------------------------------------------------------------ | -------------------------------------- |
| Submit a DP with leading-zero count below `dp_bits`                | Invalid DP, counts toward IP rate cap. |
| Submit a DP with `work_id` that does not match your assigned chunk | Invalid DP.                            |
| Submit a DP with `sequence` outside the server's expected window   | Invalid DP.                            |
| Submit a cryptographically inconsistent DP (X and d do not match)  | Invalid DP.                            |
| Send `WORK_REQ`, `DP_*`, or `STATS_REQ` before `AUTH_OK`           | Invalid-DP-equivalent infraction.      |
| Miss AUTH within the 30-second window                              | Disconnect, no ban. Just reconnect.    |

The IP-level rate cap is 100 invalid DPs per hour (server default). A reasonable client never trips it. Bans escalate per IP, in a 30-day rolling window:

| Infraction number | Ban duration |
| ----------------- | ------------ |
| 1                 | 1 hour       |
| 2                 | 6 hours      |
| 3                 | 1 day        |
| 4                 | 7 days       |
| 5+                | Permanent    |

What this means in practice: do not run patched or homebrew clients that bypass the local sanity checks. If you write a third-party client, run it against a private test pool first (see "Running a non-default pool" below) before pointing it at the public pool. A bug that generates a flood of malformed DPs at 8 GKeys/s can ban your IP in seconds.

If your worker is banned in error (genuine client bug, fixed and updated), contact the pool operator. Bans are reversible at the operator's discretion.

---

## Running a non-default pool

theCollider's pool client is server-agnostic. Any server that speaks the JLP wire protocol works.

```bash
./collider --pool jlps://your-pool.example.com:17403 --worker bc1qYourAddr
./collider --pool jlp://10.0.0.5:17403 --worker bc1qYourAddr            # plaintext, LAN
./collider --pool http://api.your-pool.example.com --worker bc1qYourAddr # HTTP variant
```

Schemes:

- `jlps://` (TLS over TCP). Recommended for anything beyond your LAN. The client validates against the system trust store (`X509_VERIFY_PARAM_set1_host` with `NO_PARTIAL_WILDCARDS` plus SNI). A trust-store load failure aborts at init in v1.4.1; there is no silent fallback to plaintext or no-verify.
- `jlp://` (plaintext TCP). Acceptable on a LAN or in a test environment. Do not use over the public internet; AUTH frames travel in cleartext.
- `http://` (HTTP variant). The client speaks a non-JLP REST variant; some legacy private pools use this. Wire-level features (batch sequencing, work_id attestation) are degraded.

For running your own server: the reference implementation is [github.com/hevnsnt/collision-protocol](https://github.com/hevnsnt/collision-protocol), a Python pool server. The wire IDL at `protocol/jlp.yaml` plus the generated bindings (C++ `jlp_wire_generated.hpp`, Python `jlp_protocol_generated.py`) is the single source of truth; any conformant server reading the IDL can drive any conformant client.

---

## Troubleshooting

### `AUTH_FAIL` on first connect

Most common cause: clock drift. The server validates an AUTH timestamp within plus or minus 30 seconds. Sync your clock:

```bash
# Linux / macOS:
sudo systemctl restart systemd-timesyncd       # systemd
sudo sntp -sS pool.ntp.org                     # macOS / manual

# Windows (admin PowerShell):
w32tm /resync
```

Second-most-common: malformed worker name. Verify it is a real Bitcoin address by pasting it into a block explorer.

### TLS error mentioning missing trust anchor

`jlps://` requires a system CA bundle. v1.4.1 fails hard at init rather than falling back.

- **Linux**: `sudo apt install ca-certificates` (Ubuntu) or `sudo dnf install ca-certificates` (Fedora).
- **macOS**: trust anchors ship with the system; if missing, reinstall Xcode Command Line Tools (`xcode-select --install`).
- **Windows**: trust anchors come from the OS certificate store; verify Windows Update is current. If you have a non-standard CA path, set `SSL_CERT_FILE` or `SSL_CERT_DIR`.

### `your_share` is zero after several minutes of running

Diagnose:

```bash
./collider --pool jlps://collisionprotocol.com:17403 \
           --worker bc1qYourBtcAddress \
           --verbose --debug
```

Look for:

- `DP_ACK` lines. If you see `DP_BATCH_V2` going out but no `DP_ACK` coming back, the server is rejecting them (check anti-cheat counter on the dashboard).
- "Conflicting search modes" or "Pro feature" errors. The pool client never reaches AUTH if a mode-mutex error trips during arg parsing.
- "no work assignment" or stuck on `WORK_REQ`. The server may be queueing chunks; wait 1-2 minutes.

### Worker shows in dashboard under a different address

You authenticated with a different `--worker` than the one in the dashboard. Common causes:

- A `pool.worker:` value in `config.yml` is overriding what you think is the CLI value. Use `--debug` to see the resolved value.
- A typo. The pool does not validate address format; a one-character typo creates a new bucket.

Fix by stopping the client, correcting the address, and restarting. The bad-address bucket remains in the dashboard with whatever DPs you accrued under it; contact the pool operator if you want them merged.

### Network latency makes `dps_per_second` look unstable

Your local stats line is computed from `DP_ACK` round trips. High RTT to the pool jitters it. The server-side `your_dps` in `STATS_RSP` is the authoritative number; use that for accrual checks.

---

## Where to go next

- [JLP-PROTOCOL.md](JLP-PROTOCOL.md) - wire format, frame layouts, anti-cheat rules in detail.
- [CONFIGURATION.md](CONFIGURATION.md) - the `pool` section of `config.yml`, plus precedence between CLI and config.
- [ARCHITECTURE.md](ARCHITECTURE.md) - where the pool client lives in the source tree (`src/pool/`).
- [collisionprotocol.com](https://collisionprotocol.com) - public-pool dashboard, payout policy, status page.
- [github.com/hevnsnt/collision-protocol](https://github.com/hevnsnt/collision-protocol) - reference Python server.
