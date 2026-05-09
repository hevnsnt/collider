# theCollider

**A GPU-accelerated solver for the Bitcoin Puzzle Challenge.** Distributed pool mining with cryptographic anti-cheat, K=1.15 Pollard's Kangaroo on secp256k1, native CUDA + Metal.

```
$ ./collider --worker bc1qYourBitcoinAddress
+==============================================================+
|                      theCollider v1.4.0                      |
+==============================================================+
[*] GPU: NVIDIA GeForce RTX 4090
[*] Backend: CUDA 12.4
[*] Connecting to pool: collisionprotocol.com:17403
[*] Authenticated as bc1q... -- chunk 0x4e7a13... assigned

Pool: 47.2% | Speed: 8.14 GKeys/s | DPs: 1.2M | Sent: 847K | ETA: ~2.3 years
```

The Free version is fully functional for pool mining. [Pro features](docs/PRO-FEATURES.md) extend the same binary for opportunistic brain-wallet sweeps and weak-PRNG attacks.

---

## Table of contents

- [The Bitcoin Puzzle Challenge](#the-bitcoin-puzzle-challenge)
- [Algorithm: Pollard's Kangaroo](#algorithm-pollards-kangaroo)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Pool mining](#pool-mining)
- [Solo mode](#solo-mode)
- [Benchmark](#benchmark)
- [Configuration](#configuration)
- [CLI reference](#cli-reference)
- [The JLP protocol](#the-jlp-protocol)
- [Security and anti-cheat](#security-and-anti-cheat)
- [What happens when the pool solves a puzzle](#what-happens-when-the-pool-solves-a-puzzle)
- [Performance](#performance)
- [Architecture](#architecture)
- [Pro features](#pro-features)
- [Operational security](#operational-security)
- [Contributing](#contributing)
- [License](#license)

---

## The Bitcoin Puzzle Challenge

In 2015 someone funded 160 Bitcoin addresses with private keys of increasing difficulty: puzzle **n** has a 256-bit private key whose top `(n-1)` bits are zero, leaving a search space of `2^(n-1)`. Puzzles 1-70 have been solved. Then the creator of the puzzle in mid-2023 swept a third of the funds (every puzzle past 70 that had been solved was emptied — leaving 65, 75, 80, 85, ..., 160 untouched, each multiple of 5).

The remaining unsolved puzzles total **~970 BTC** at the time of writing. Puzzle 135 alone holds **13.5 BTC**.

For the unsolved puzzles in the stride-of-5 set, the **public key has been exposed** by previous spend transactions. That changes everything.

| Puzzle | Range | Public key exposed | Algorithm |
|---|---|---|---|
| 71-130 (every 5th still funded) | `2^(n-1)` | **yes** | Pollard's Kangaroo: `O(sqrt(2^(n-1)))` |
| 135, 140, ..., 160 | `2^(n-1)` | **yes** | Pollard's Kangaroo |

Brute force at 10 GKeys/s on puzzle 135 takes longer than the age of the universe. Kangaroo on the same GPU takes about 2-3 years. A pool of a few hundred GPUs takes weeks.

That's the math that makes this project feasible.

## Algorithm: Pollard's Kangaroo

Two herds of "kangaroos" walk the keyspace:

- **Tame kangaroos** start at known points and record their path.
- **Wild kangaroos** start from the target public key.

When a wild kangaroo lands on a tame kangaroo's path, the difference between their walks is the private key. The trick is **distinguished points** (DPs) — points whose x-coordinate has a configurable number of leading-zero bits. We only store and compare these, which keeps the DP database tractable.

Our implementation uses [RCKangaroo](https://github.com/RetiredCoder/RCKangaroo) at the kernel level (K=1.15, the best public Kangaroo implementation we've measured) wrapped behind the JLP pool client.

`K` is the efficiency factor: K=1.0 is the theoretical minimum, K=1.15 means we do ~15% more EC operations than information-theoretically required. Most public Kangaroo implementations sit at K=1.6-2.0.

## Quick start

```bash
# Download the binary for your platform from GitHub Releases:
#   https://github.com/hevnsnt/collider/releases/latest

# Connect to the pool (your Bitcoin address is your worker identity)
./collider --worker bc1qYourBitcoinAddress
```

That is the entire setup for pool mining. Your address determines reward attribution; no registration, no API keys.

## Installation

### Pre-built binaries

[GitHub Releases](https://github.com/hevnsnt/collider/releases) carry binaries for:

- **Windows**: `collider.exe`
- **Linux**: `collider`
- **macOS (Apple Silicon)**: `collider-macos-arm64`

The website at [collisionprotocol.com](https://collisionprotocol.com) auto-links to the latest release.

### Build from source

Platform-specific build guides live in `docs/`:

- [Build on Windows](docs/BUILD-WINDOWS.md)
- [Build on Linux](docs/BUILD-LINUX.md)
- [Build on macOS](docs/BUILD-MACOS.md)

Common requirements: CMake 3.20+, Ninja, a C++20 compiler. CUDA 12.x for NVIDIA GPUs; Apple Silicon for Metal. CPU fallback exists but is impractical for real puzzles.

## Pool mining

Pool mining is the primary use case for the Free version. It distributes the search across many GPUs and pays out proportionally based on contributed work.

### Why a pool

A single RTX 4090 working alone on puzzle 135 has an expected solve time on the order of years. A pool with the equivalent of ~500 RTX 4090s solves it in ~2 weeks. The pool keeps a shared **distinguished-point database** that turns concurrent searches into a single shared search.

The pool itself does not hold your funds, your keys, or any solution material. It coordinates DP submissions and detects collisions; the eventual private key is broadcast back to all workers and the on-chain spend is constructed transparently (see [What happens when the pool solves a puzzle](#what-happens-when-the-pool-solves-a-puzzle)).

### Reward model

When a puzzle is solved, the prize BTC is split:

- **5% pool fee** to cover infrastructure (the JLP server runs on dedicated hardware with hot/cold backups).
- **95% prize** distributed proportionally across all workers in the credited DP pool, weighted by each worker's contribution since the puzzle work began.

Each Bitcoin address (worker name) is its own credit bucket. Two workers with the same `--worker` value share credit; two workers with different addresses keep them separate. There is no minimum payout threshold.

### Connecting

```bash
# Default: connects to collisionprotocol.com:17403 over TLS
./collider --worker bc1qYourBitcoinAddress

# Or with explicit pool URL
./collider --pool jlp://collisionprotocol.com:17403 --worker bc1q...
```

Optional: `--gpus 0,2,3` to bind specific GPUs; `--password <secret>` if the pool ever introduces gated access (currently open).

## Solo mode

You can solve directly without the pool. This is useful for benchmarking, verifying behavior, or attacking smaller puzzles (1-70 range).

```bash
./collider --puzzle 32      # tractable in seconds on any GPU
./collider --puzzle 75      # roughly weeks on a single 4090
./collider --puzzle 135     # not realistic on a single GPU; use pool
```

The DP database lives in `./dps.db` (SQLite); resume a run by re-launching with the same `--puzzle` argument. Solo mode autosaves every 1M passphrases checked.

## Benchmark

```bash
./collider --benchmark
```

Reports a sustained EC-multiply rate over 30 seconds. Compare against the table below; large deviations usually mean a thermal-throttled GPU or a stale CUDA driver.

## Configuration

A `config.yml` in the working directory replaces most CLI flags:

```yaml
pool:
  url: "jlp://collisionprotocol.com:17403"
  worker: "bc1qYourBitcoinAddress"

gpu:
  devices: []          # empty = all
  threads_per_block: 256

logging:
  level: info          # trace, debug, info, warn, error
  file: ./collider.log
```

CLI flags always override file values.

## CLI reference

```
Usage: collider [options]

Pool options:
  --pool, -p <url>            Pool URL (default: jlp://collisionprotocol.com:17403)
  --worker, -w <address>      Bitcoin address for reward attribution (required)
  --password <secret>         Optional pool password

Solo options:
  --puzzle <n>                Target puzzle 1-160 (solo mode)
  --range-start <hex>         Override start of search range
  --range-end <hex>           Override end of search range
  --dp-bits <n>               Distinguished-point bit count (auto by default)
  --kangaroo                  Force Kangaroo (default when pubkey exposed)

GPU options:
  --gpus, -g <ids>            Comma-separated GPU IDs (default: all)
  --batch-size <n>            Per-batch passphrase count

Other:
  --benchmark                 30s sustained-rate test
  --config, -c <file>         Load config.yml from path
  --analyze                   Show puzzle list with current pool state
  --resume                    Resume from saved state in current dir
  --verbose, -v               Verbose logging
  --debug                     Trace-level logging (very noisy)
  --help, -h                  This message
  --version                   Print version and exit
```

[Pro flags](docs/PRO-FEATURES.md) (`--brainwallet`, `--bloom`, `--puzzle-only-v2`, `--schemes`, `--addr-types`, etc.) are present in the Free binary but reject with the message "I'm sorry, but this is a pro function..." when invoked without a license.

## The JLP protocol

JLP (originally Jean-Luc Pons's wire format, extended in v2 with anti-cheat) is the protocol between the client and the pool server. It runs over TCP, optionally wrapped in TLS 1.3.

### Wire format

Every message starts with the same 8-byte header:

```
+--------+--------+--------+--------+--------+--------+--------+--------+
|        magic = "KANG"             |  type  |  flags |   payload_len   |
|       (4 bytes)                   | (1 byte)| (1 byte)| (2 bytes, LE) |
+--------+--------+--------+--------+--------+--------+--------+--------+
```

Followed by `payload_len` bytes of message-specific payload. Multi-byte integers are little-endian.

### Message types

| Hex   | Name                                | Direction       | Purpose                                  |
|-------|-------------------------------------|-----------------|------------------------------------------|
| 0x01  | `AUTH`                              | client -> server | Worker authentication                    |
| 0x02  | `AUTH_OK`                           | server -> client | Authentication accepted                  |
| 0x03  | `AUTH_FAIL`                         | server -> client | Authentication rejected                  |
| 0x10  | `WORK_REQ`                          | client -> server | Request a chunk of the keyspace          |
| 0x11  | `WORK_ASN`                          | server -> client | Chunk assignment with `work_id`          |
| 0x20  | `DP_SUBMIT`                         | client -> server | Submit a single DP (v1, no attestation)  |
| 0x21  | `DP_ACK`                            | server -> client | DP credited                              |
| 0x22  | `DP_BATCH`                          | client -> server | Batch of DPs (v1, no attestation)        |
| 0x23  | `DP_SUBMIT_V2`                      | client -> server | Submit a DP **with `work_id` attestation** |
| 0x24  | `DP_BATCH_V2`                       | client -> server | Batch v2                                 |
| 0x30  | `STATS_REQ` / `STATS_RSP` (0x31)   | both             | Pool stats                               |
| 0x40  | `SOLUTION`                          | server -> client | Puzzle solved; private key broadcast      |
| 0x50/0x51 | `PING` / `PONG`                | both             | Keepalive                                |
| 0xFF  | `MSG_ERROR`                         | both             | Diagnostic / error                        |

The full wire definition (struct layouts, IDL hash) lives in [`protocol/jlp.yaml`](protocol/jlp.yaml) — both the C++ client and the Python pool server are generated from this file.

### Work assignment lifecycle

```
client          server
  |   AUTH ----> |       client identifies as bc1q... (worker name)
  | <-- AUTH_OK  |
  |              |
  | WORK_REQ --> |
  | <-- WORK_ASN |       109-byte payload: pubkey + range + dp_bits + work_id
  |              |
  |   ...search a 2^dp_bits subspace, find DPs...
  |              |
  | DP_BATCH_V2->|       each DP carries the work_id of the chunk it came from
  | <-- DP_ACK   |
  |              |
  | WORK_REQ --> |       pull next chunk
  | <-- WORK_ASN |
```

### v1 vs v2 DP submissions

Earlier deployments used `DP_SUBMIT` / `DP_BATCH` (0x20 / 0x22). v1.4.0 clients prefer `DP_SUBMIT_V2` / `DP_BATCH_V2` (0x23 / 0x24), which prepend an 8-byte `work_id` to every DP. The server rejects any v2 DP whose `work_id` does not match the chunk currently assigned to that worker. v1 messages are still accepted for backwards compatibility with deployed pre-1.4.0 clients.

### TLS

Default port 17403 expects TLS 1.3 with the pool's certificate signed by Let's Encrypt. The client validates the cert chain using the system root store. Pinning is not currently enforced; if you want it, set `pool.cert_pin` in `config.yml` to a SHA-256 hex of the expected leaf.

## Security and anti-cheat

The pool's job is to coordinate honest workers. The only failure mode that costs other workers real money is a **cheating worker that fabricates DPs** — they would inflate their share of the eventual prize without contributing real computation. v1.4.0 has three layers of defense, all enabled by default:

### 1. Cryptographic math verification (the rckangaroo model)

Every accepted DP is checked against the secp256k1 EC equation that an honestly-walked kangaroo must satisfy. For a tame DP at distance `d`, the X coordinate must equal `(d * G).x`. For a wild DP, it must equal `((Q - r_start*G) ± d*G).x`, where `Q` is the puzzle's public key and `r_start` is the chunk's range start. A worker that fabricates DPs without performing the EC operations has to solve the discrete log problem on secp256k1 — which is what we're trying to do in the first place.

The verifier runs in `shadow` mode by default (logs deviations without acting on them) for the first hour after a server upgrade, then escalates to `enforce`. Operators can flip this in the server config.

### 2. Work-ID attestation (v2 wire format)

Each chunk handed out by the server has a unique 64-bit `work_id`. Honest v2 clients prepend that `work_id` to every DP they submit from that chunk. The server cross-checks against the worker's currently-assigned chunk. A worker submitting DPs for a chunk it was never assigned (the "wrong-range DPs" attack) is unambiguously cheating; the DP is dropped and the IP routes to the ban pipeline.

### 3. IP-based ban escalation

Bans are by IP, not by Bitcoin address. The shared-payout-address model means banning an address punishes legitimate co-mining workers when one bad actor uses the same address. The bad actor's source IP is the right pivot.

The escalation ladder, configurable in the server's `config.yaml` under `security:`:

| Offense | Ban duration |
|---|---|
| 1st | 1 hour |
| 2nd | 6 hours |
| 3rd | 1 day |
| 4th | 7 days |
| 5th+ | Permanent (admin-only clear) |

A "ban-eligible event" is `>= 100` invalid DPs from a single IP within a rolling 1-hour window. Prior bans are counted within a 30-day sliding window; older offenses fall off and the ladder restarts.

The server also keeps a 14-day audit log of every invalid DP for forensic review. Cleared bans (admin-unbanned) move to a separate audit table rather than being deleted.

The default thresholds are deliberately conservative — a healthy worker on a flaky network connection (transient corruption, restart loops) will not trip the threshold. The `100/hour` window is calibrated against historical false-positive rates.

### What gets logged

The pool server retains:

- Every accepted DP (worker name + IP + work_id, hashed for storage)
- Every rejected DP and the reason
- Every ban event and its escalation level
- TLS handshake metadata (cipher, cert fingerprint)

It does NOT retain:

- Source IPs longer than 30 days unless attached to an active ban
- Any decoded passphrase, priv-key candidate, or wordlist content
- Anything tying a Bitcoin address to a real-world identity

## What happens when the pool solves a puzzle

This is the part most workers want documented. Step by step:

### Step 1 — Detection

Every accepted DP goes into a SQLite-indexed database. On every insert the server checks for a **collision**: does this DP's X coordinate match a DP from the opposite herd (one tame, one wild) in the same chunk? Most inserts find nothing; a successful collision triggers the solver.

### Step 2 — Verification

A collision is a strong signal but not yet a proven solution. The server runs the rckangaroo recovery: from the two distance values, compute the candidate private key, then check `priv * G == puzzle_pubkey`. If yes, this is the real key. If not (extremely rare; usually a hash collision in the DP storage path, not an EC collision), the candidate is dropped and DPs continue accumulating.

### Step 3 — Broadcast

Verified solutions are broadcast to every connected worker via a `SOLUTION` message (0x40). The payload is the 32-byte private key. Workers stop searching for that puzzle and report the solve to their local UI.

A workers seeing `SOLUTION` for the puzzle they're working will:

```
[!!!] PUZZLE 135 SOLVED
[!!!] Private key: <32 bytes hex>
[!!!] Pool will distribute rewards within 24 hours.
[*]   Disconnecting from this puzzle's chunk pool.
```

The client does NOT auto-spend the puzzle. The pool's bonded address signs the spend (see step 4) so the broadcast key is provably useless to a malicious worker who tries to front-run — by the time they receive `SOLUTION`, the pool has already submitted the spend transaction with its priority gas price.

### Step 4 — On-chain settlement

The pool operator's bonded Bitcoin address constructs and broadcasts a transaction that:

1. Spends the puzzle's UTXO using the recovered private key.
2. Sends the prize to the pool's payout address (held in cold storage between solves).
3. Pays an aggressive priority fee to land in the next block; this defeats the small risk of a worker racing the broadcast.

Pool operator's bonded address is published in advance on the website and never changes. The bond is currently 25 BTC; if the operator ever attempts to abscond, the bond is forfeited to the contributing workers via a pre-signed timelock contract. (Whitepaper draft on `collisionprotocol.com/whitepaper`.)

### Step 5 — Reward distribution

Within 24 hours of the on-chain confirmation:

1. The 5% pool fee is sent to the operator's fee address.
2. The 95% prize is distributed across all workers credited with at least one accepted DP for that puzzle. Each worker's share is `(their_credited_DPs / total_credited_DPs) * prize_amount`.
3. Each worker receives a Bitcoin transaction to the address they used as their `--worker` value.

Distribution transactions are batched (multiple recipients per tx) to minimize on-chain fees. Confirmation typically lands within 6 blocks. Every distribution tx is published with a corresponding receipt at `collisionprotocol.com/payouts/<puzzle_n>` that lists worker addresses and credited DPs (anonymized except for the address).

### Step 6 — Audit period

For 30 days after a solve, the full DP-credit ledger is published as a CSV under `collisionprotocol.com/audit/<puzzle_n>.csv`. Workers can reconcile their share against the public ledger. Disputes go through `disputes@collisionprotocol.com` and are reviewed by a community auditor (currently a rotating volunteer).

## Performance

Sustained EC-multiply rate (Kangaroo, no checkpointing overhead):

| GPU | Keys/Second |
|---|---|
| RTX Pro 6000 (Blackwell, 96GB) | ~14 GKeys/s |
| RTX 5090 | ~12 GKeys/s |
| RTX 4090 | ~8 GKeys/s |
| RTX 3090 | ~4 GKeys/s |
| RTX 3060 | ~1.5 GKeys/s |
| Apple M3 Max | ~1.2 GKeys/s |
| Apple M2 | ~400 MKeys/s |
| Apple M1 | ~200 MKeys/s |

Multi-GPU scales linearly up to PCIe lane saturation (typically 4-8 GPUs per host before memory bandwidth becomes the bottleneck).

The hot-path EC kernel is in `src/gpu/secp256k1.cu`; we use projective coordinates and batch inversions to amortize the expensive modular inverse.

## Architecture

```
src/
├── main.cpp              # CLI parsing + dispatch
├── core/                 # Config, puzzle DB, runtime types
├── gpu/                  # CUDA kernels + Metal shaders
│   ├── secp256k1.cu      # EC point arithmetic
│   ├── kangaroo_kernel.cu# Kangaroo walk
│   └── ...
├── pool/                 # JLP client (TLS + state machine)
│   ├── jlp_pool_client.cpp
│   └── jlp_wire_generated.hpp   # auto-generated from protocol/jlp.yaml
├── platform/             # CUDA / Metal / CPU HAL
└── ui/                   # Banner, interactive menu, progress display

protocol/
└── jlp.yaml              # Single source of truth for the JLP wire format

third_party/
└── RCKangaroo/           # The actual Kangaroo kernel (GPLv3, vendored)
```

## Pro features

The Free version above is fully functional for pool mining and small-puzzle solo mode. The Pro version adds opportunistic and dedicated brain-wallet attack capabilities to the same binary.

See [docs/PRO-FEATURES.md](docs/PRO-FEATURES.md) for the full feature list, including:

- Multi-scheme + multi-address brain-wallet sweeps
- Weak-PRNG kernels (libbitcoin "Milk Sad" CVE-2023-39910, Profanity CVE-2022-40769, Trust Wallet CVE-2023-31290, glibc / MSVC / Java Random)
- Puzzle-only mode that short-circuits before the EC multiply
- Encoding mutations (UTF-16-LE, UTF-32, Latin-1, etc.)
- Modular legacy KDF framework
- Electrum v1/v2 mnemonic seed kernels
- Historical CVE sweep (Debian OpenSSL 2008, Android SecureRandom 2013)

Pro is a one-time license purchase. License gating is enforced at the binary; the same binary handles both Free and Pro paths, so no separate download.

## Operational security

A few things worth noting if you intend to run this for real money:

- The pool connection is TLS-encrypted by default. **Your private keys are never transmitted** — only Distinguished Points, which are public information about the search progress.
- The pool **cannot** solve puzzles without contributors. It is a coordination mechanism, not a key escrow. The DP database alone is not a private key.
- **Do not target Bitcoin addresses you don't own** outside the puzzle set. The puzzle addresses are explicitly an open challenge with no claimant; arbitrary addresses are not. Using cryptographic tools against systems without authorization is illegal in most jurisdictions and unethical everywhere.
- The Pro brain-wallet path is intended for security research and recovering your own forgotten passphrases. Do not run it against passphrase wordlists from systems you don't own.
- Pool mining puts your machine on the public internet talking to a TLS endpoint. Standard hygiene applies: keep the binary updated, run on a non-privileged user, don't reuse the same Bitcoin address as your hot wallet.

## Contributing

Pull requests welcome. Particularly valuable:

- OpenCL backend for AMD GPUs.
- More aggressive PCIe-bandwidth-aware batching for high-GPU hosts.
- Additional secp256k1 architecture-specific optimization (RDNA3, Hopper).
- Alternative pool-protocol implementations (this codebase is JLP-only today).

For any change touching the wire protocol, please open an issue first; protocol drift is the worst kind of bug because it's invisible until a worker silently submits unauthenticated DPs.

## License

MIT. See [LICENSE](LICENSE).

The third-party RCKangaroo kernel under `third_party/RCKangaroo/` is GPLv3 and stays that way; the rest of the codebase is MIT.

## Acknowledgments

- [RetiredCoder](https://github.com/RetiredCoder) — the RCKangaroo implementation that makes K=1.15 possible.
- [Jean-Luc Pons](https://github.com/JeanLucPons) — the original GPU Kangaroo work and the JLP protocol.
- bitcoin-core/secp256k1 — the reference EC implementation we cross-validate against.
- Everyone who has run a pool worker since the project started.

---

*"The puzzle is the prize."*
