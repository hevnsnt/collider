# theCollider

> A GPU-accelerated solver for the [Bitcoin Puzzle Challenge](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx), built for the unsolved keys at the hard end of the curve.

theCollider is the most complete open-source toolkit for the Bitcoin Puzzle Challenge today: solo Pollard's Kangaroo and brute-force search on CUDA (Windows, Linux) and Metal (macOS Apple Silicon), a theft-resistant pool client for distributed solving via the [Collision Protocol](https://collisionprotocol.com) pool, and the canonical bundled record of every solved puzzle to date. A paid Pro edition adds an opportunistic brain-wallet pipeline, a hashcat-style rule engine, a BIP-39 / BIP-32 / BIP-44 / BIP-49 / BIP-84 mnemonic scanner, and a multi-scheme weak-PRNG kernel on top of the same core.

```
$ ./collider --puzzle 135 --kangaroo
```

That command, today, on commodity hardware, takes a real swing at puzzle #135, currently the smallest unsolved kangaroo-able puzzle in the challenge.

---

## The puzzle, briefly

In 2015, the author of Bitcoin Core funded 160 addresses on the Bitcoin mainnet. The N-th address holds a private key drawn from the range `[2^(N-1), 2^N)`. The total reward across all 160 addresses approaches 1000 BTC. The addresses are public; the keys are not. The challenge is to find them.

Every puzzle from 1 to 70 has been solved, plus every multiple of 5 from 75 to 130 (82 in total; bundled in `data/puzzle_history.json`). The remaining unsolved puzzles split into two populations with very different mathematics:

| Bit range                                          | Status                                    | Solvable how                                                 |
| -------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| 1 to 65 (contiguous)                               | Solved                                    | Bundled, for record and reference                            |
| 66 to 70 (contiguous)                              | Solved                                    | Bundled                                                      |
| 75, 80, 85 ... 130 (mult. of 5)                    | Solved (partially-spent, pubkey revealed) | Bundled                                                      |
| 135, 140, 145, 150, 155, 160                       | **Unsolved, pubkey revealed**             | Pollard's Kangaroo, distributed                              |
| 71, 72, 73, 74, 76, 77, ... non-mult-of-5 above 70 | **Unsolved, pubkey unknown**              | Brute force only (computationally infeasible above ~64 bits) |

That second-to-last row is the tractable frontier. Puzzles #135 through #160 (the multiples of 5) were each originally funded with a reward proportional to the puzzle number; #135 held an original deposit on the order of 13.5 BTC, #160 on the order of 16 BTC. Live balances drift as the original wallet broadcasts further partial-spend transactions; the `--puzzle` banner queries mempool.space at solve time to report the current balance. The public keys for these targets are knowable today because the original wallet has, at some point, broadcast a partial-spend transaction, exposing the pubkey on-chain.

The bottom row is mathematically locked. Without a spending transaction, the pubkey is unknown to anyone, anywhere, and SHA-256 plus RIPEMD-160 keep it that way. Brute force is the only remaining option, and at 71+ bits, brute force on consumer hardware is multi-decade work.

theCollider concentrates on what is solvable: kangaroo against revealed pubkeys, alone or as part of a pool.

---

## Quick start

> Three OS commands, one `./collider` invocation. No SDK install, no account, no auth.

### Linux (CUDA)

```bash
curl -fsSL -o collider \
  https://github.com/hevnsnt/collider/releases/latest/download/collider-linux-x64-cuda
chmod +x collider
./collider
```

### Windows (CUDA)

```powershell
iwr -OutFile collider.exe `
  https://github.com/hevnsnt/collider/releases/latest/download/collider-windows-x64-cuda.exe
.\collider.exe
```

### macOS (Apple Silicon, Metal)

```bash
curl -fsSL -o collider \
  https://github.com/hevnsnt/collider/releases/latest/download/collider-macos-arm64-metal
chmod +x collider
./collider
```

The default `./collider` invocation:

1. Detects every visible GPU and calibrates batch size on first launch.
2. ROI-ranks the unsolved puzzles by reward divided by expected operations.
3. Picks the best target, fires kangaroo if the pubkey is bundled, falls back to brute force otherwise.

To target a specific puzzle:

```bash
./collider --puzzle 135 --kangaroo
```

To join the public pool and distribute the work:

```bash
./collider --pool jlps://collisionprotocol.com:17403 \
           --worker 1YourBitcoinAddress...
```

The full CLI reference is below.

---

## How it compares

theCollider is not the first GPU solver for this problem. Several open-source projects have moved the state of the art forward, and theCollider builds directly on one of them.

### RCKangaroo (RetiredCoder)

[RCKangaroo](https://github.com/RetiredCoder/RCKangaroo) is a state-of-the-art kangaroo implementation. It introduced the SOTA method with K=1.15 (versus K=2.1 for classic three-way kangaroo, ~1.8x fewer operations and ~1.8x less DP storage), and it benchmarks at roughly 8 GKeys/s on an RTX 4090. **theCollider uses RCKangaroo as its CUDA kangaroo backend.** Credit where it belongs: the kernel doing the heavy lifting on Windows and Linux is RetiredCoder's, redistributed here under its GPLv3 license in `third_party/RCKangaroo/`.

What theCollider adds on top:

| Capability                                                     | RCKangaroo               | theCollider                                                 |
| -------------------------------------------------------------- | ------------------------ | ----------------------------------------------------------- |
| SOTA kangaroo on CUDA (Windows, Linux)                         | Yes (the implementation) | Yes (links RCKangaroo)                                      |
| Apple Silicon / Metal kangaroo                                 | No (CUDA only)           | Yes (native Jacobian Metal port, v1.4.1)                    |
| Apple Silicon / Metal brute force                              | No                       | Yes (v1.4.1)                                                |
| Distributed / pool solving                                     | No (solo only)           | Yes (JLP wire protocol over TCP+TLS)                        |
| Bundled puzzle metadata (addresses, ranges, pubkeys)           | Manual per run           | All 82 solved puzzles plus revealed-pubkey unsolved bundled |
| Automatic puzzle selection (ROI ranking)                       | No                       | Yes                                                         |
| Graceful kangaroo to brute-force fallback on no-pubkey puzzles | No                       | Yes (v1.4.1)                                                |
| Live BTC balance on the solved banner (mempool.space)          | No                       | Yes (v1.4.1)                                                |
| Brain-wallet + funded-address scanner (100M+ addresses)        | No                       | Yes, Pro edition                                            |
| Hashcat-style rule engine + bloom-filter lookup                | No                       | Yes, Pro edition                                            |
| License                                                        | GPLv3                    | MIT (Free), commercial (Pro)                                |

On raw CUDA kangaroo throughput the two are identical, because theCollider links RCKangaroo's kernel directly (see `third_party/RCKangaroo/` and `src/gpu/rckangaroo_wrapper.cu`). RCKangaroo gives you the solver kernel; theCollider gives you the solver kernel plus Apple Silicon support (Metal kangaroo, our own implementation), a pool client and JLP protocol, multi-GPU orchestration, brute-force kernels for the smaller puzzles, and the operator tooling around all of it. The two are complementary, not competitive: theCollider is "RCKangaroo plus everything around it that you would otherwise have to build yourself."

### Other solvers in the space

- [BSGS / Pollard's Rho variants](https://github.com/JeanLucPons/BSGS) tend to be 32- to 64-bit ceilings; they are not viable at puzzle #135 scale.
- [Kangaroo (JLP)](https://github.com/JeanLucPons/Kangaroo) is the canonical older kangaroo solver; predates the SOTA method.
- Web-based "checkers" are not solvers; they query precomputed databases.

theCollider treats kangaroo, brute force, and the brain-wallet path (Pro) as one product with shared scaffolding (CLI, config, multi-GPU calibration, pool, telemetry, checkpointing), instead of three separate tools.

---

## Pool mode (Collision Protocol)

For puzzles above #135, no single machine has the compute budget to finish in a reasonable time. The pool solves this by sharding the search range across many workers and letting them share distinguished points (DPs). When a collision is detected on the server, the private key falls out of the math.

```bash
./collider --pool jlps://collisionprotocol.com:17403 --worker 1YourBitcoinAddressForRewards
```

What happens after AUTH:

1. The server assigns your worker a chunk of the kangaroo search range, a `work_id` attestation token, AND a `kangaroo_type` (TAME-only or WILD-only).
2. Your GPU runs kangaroo on that chunk, producing distinguished points (X coordinate with a configurable bit-count of leading zeros) of the assigned type ONLY. The host-side cross-type collision detector is disabled in pool mode.
3. Your client batches DPs into `DP_BATCH_V2` frames every few seconds and submits.
4. The server tracks the cumulative DP count per worker. The Bitcoin address you authenticated with is the reward-accrual key.
5. When the server matches a TAME DP from one worker against a WILD DP from another, it reconstructs the private key, broadcasts a hot-wallet sweep transaction, waits for cross-provider mempool attestation, then pushes a `SOLUTION` frame to every connected worker.
6. Workers who contributed valid DPs accrue share-of-pool credit, redeemable when a puzzle is solved per the pool's payout policy. See [docs/POOL.md](docs/POOL.md) for the mechanics.

### TAME/WILD asymmetric assignment (v1.5.0 theft-resistance)

In v1.4.x, a pool worker who happened to find the cross-collision locally computed the puzzle's private key on their machine and could sweep the funds before the pool ever saw the solution. v1.5.0 closes this window architecturally:

- **Asymmetric work**: each worker runs ONLY tame kangaroos OR ONLY wild kangaroos. The math requires both types to recover the key. A single worker, no matter how lucky, cannot collide a TAME against a WILD locally.
- **No host-side hashtable in pool mode**: the worker's cross-type DP table is disabled. DPs flow straight to the network. The `result.found = true` codepath is removed; the kangaroo loop exits only on external stop.
- **Server is the sole solver**: the pool server is the only place where TAME and WILD DPs aggregate. It reconstructs the key, broadcasts the sweep, waits for a SECOND mempool provider to observe the sweep tx, and only then notifies workers that the puzzle is solved.
- **`report_solution` removed from the client**. The v1.5 worker never computes nor handles a private key in pool mode. SOLUTION is server-to-client only.
- **Type-mismatch is a permanent ban**: a worker submitting a WILD DP while assigned TAME (or vice versa) is unambiguous binary modification, not honest-but-buggy behavior. The pool server bans the originating IP permanently on first occurrence.

The v1.5 wire protocol bumps to **JLP v3** to enforce this. v1.4.x clients are refused at AUTH with `UPGRADE_REQUIRED`; there is no compatibility shim. Standalone (non-pool) mode keeps the symmetric `BOTH` kangaroo mode and is unchanged.

### Other anti-cheat (carried forward from v1.4.x)

Every DP carries a `work_id` attestation (you cannot resubmit captured DPs against a different chunk) and a per-DP monotonic `sequence` nonce (you cannot replay a captured `DP_BATCH`). AUTH replay is blocked by a 30-second timestamp drift window. Server `dp_bits` is validated to `[8, 32]` (a malicious `dp_bits=255` would otherwise burn GPU cycles indefinitely). Invalid DPs cost reputation per IP with bans escalating from 1 hour to permanent.

The reference server is the [Collision Protocol](https://github.com/hevnsnt/collision-protocol) project, deployed at `collisionprotocol.com`. Third-party servers can implement the same protocol (it is documented in full at [docs/JLP-PROTOCOL.md](docs/JLP-PROTOCOL.md)).

---

## Free vs. Pro

theCollider ships in two editions from one source tree.

| Capability                                                              | Free (MIT) | Pro (commercial) |
| ----------------------------------------------------------------------- | ---------- | ---------------- |
| Pollard's Kangaroo (CUDA, Metal)                                        | Yes        | Yes              |
| Brute-force puzzle search (CUDA, Metal)                                 | Yes        | Yes              |
| Pool client (JLP, TLS)                                                  | Yes        | Yes              |
| Bundled puzzle history (all 82 solved)                                  | Yes        | Yes              |
| Multi-GPU support with calibration                                      | Yes        | Yes              |
| Save / resume checkpoints                                               | Yes        | Yes              |
| GPU benchmark                                                           | Yes        | Yes              |
| Brain-wallet passphrase pipeline + 100M+ funded-address bloom filter    |            | Yes              |
| Hashcat-style rule engine on GPU (mutate any wordlist into billions)    |            | Yes              |
| PCFG + Markov passphrase generators (high-probability candidates first) |            | Yes              |
| BIP-39 / BIP-32 mnemonic scanner (BIP-44 / 49 / 84 derivation, v1.5.0)  |            | Yes              |
| v2 multi-scheme kernel: weak-PRNG sweeps (Milk Sad, Profanity, etc.)    |            | Yes              |
| WarpWallet brain-wallet scheme (`--warpwallet-salt`)                    |            | Yes              |
| Multi-panel FTXUI dashboard (GPU panel, range coverage, sparklines)     |            | Yes              |
| License gating (HMAC-SHA256, offline cache, 24h TTL)                    |            | Yes              |

Pro licenses are available at [collisionprotocol.com/pro](https://collisionprotocol.com/pro). The Pro source tree is not public; the Free build is the same source minus the Pro modules, generated automatically by GitHub and published at [github.com/hevnsnt/collider](https://github.com/hevnsnt/collider).

---

## Pro: every key is a lottery ticket

Free is a complete puzzle solver. **Pro turns the same binary into a treasure hunter.**

Your GPU is grinding billions of private keys to find puzzle 135. In Free, every key is checked against exactly one target. In Pro, **every key is also checked against a 100 million entry funded-address bloom filter**, automatically, in the same GPU kernel pass, at essentially zero marginal cost.

### The killer feature: opportunistic scanning during pool and standalone work

```
Free pool mode:
  GPU computes pubkey -> derives DP -> sends DP to pool server.
  (One target. Puzzle 135. That is the whole story.)

Pro pool mode:
  GPU computes pubkey -> derives DP -> hashes to Bitcoin address ->
    queries bloom filter for 100M+ funded addresses ->
    hit? log to bloom_hits.txt for verification.
  Sends DP to pool server.
  (Still mining puzzle 135. AND scanning every key against every
   funded Bitcoin wallet that has ever existed.)
```

You bought the hardware. You pay the power bill. Pro makes sure every key your GPU computes is doing as much work for you as it possibly can. Tail `bloom_hits.txt` in another terminal and walk away.

### Plus: dedicated brain-wallet, BIP, and weak-PRNG modes

When you are not pool-mining, Pro flips into dedicated treasure-hunter modes:

- **Brain wallets**: PCFG + Markov + hashcat-style rule engine drive a fused GPU pipeline (SHA-256 -> secp256k1 -> hash160 -> bloom probe) at tens to hundreds of millions of candidate checks per second on Ampere through Blackwell GPUs. Measure your card with `./collider --benchmark`. `SHA256("correct horse battery staple")` is a real private key; tens of thousands of these wallets were created between 2011 and 2014, and many still hold real Bitcoin today.
- **BIP-39 / BIP-32 mnemonic scanner (v1.5.0)**: takes a wordlist of candidate seed phrases (or exhaustive entropy enumeration), validates each against BIP-39 checksum, derives the BIP-32 master key, and walks every historical and modern derivation path: pre-BIP-44 wallets (Electrum 2.x, MultiBit HD, blockchain.info, early Bitcoin Core HD), BIP-44 P2PKH, BIP-49 P2SH-P2WPKH, BIP-84 native segwit P2WPKH. Each child key's `hash160` is probed against the same 100M+ funded-address bloom filter. Hit-driven mnemonic recovery — useful for partial-recall, leaked-phrase, and combinatorial-permutation scenarios. See `--bip-scan`, `--bip-scan-wordlist`, `--bip-combinatorial` in the CLI reference.
- **Weak-PRNG sweeps (v2 multi-scheme kernel)**: scans for keys produced by historically-broken random number generators. Milk Sad (CVE-2023-39910), Profanity (CVE-2022-40769), Trust Wallet, glibc, MSVC, Java. Five Bitcoin address types per candidate seed, eight scheme variants per pass.

Same install. Same UI. License-gated via HMAC-SHA256 with a 24h offline-verifiable cache.

### **See [docs/PRO.md](docs/PRO.md) for the full Pro pitch, including the math behind opportunistic scanning, generator details, the bloom-filter internals, and the pricing tiers.**

Pro licenses are at [collisionprotocol.com/pro](https://collisionprotocol.com/pro).

---

## CLI reference

Full list: `./collider --help`. Highlights below; the source of truth is [`src/cli/cli_parser.cpp`](src/cli/cli_parser.cpp).

### Puzzle mode

| Flag                                         | Effect                                                                                          |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `--puzzle N`, `-P`                           | Target puzzle N. Default: ROI-pick the best unsolved puzzle.                                    |
| `--kangaroo`                                 | Force Pollard's Kangaroo. Demotes to brute force gracefully when no pubkey is bundled (v1.4.1). |
| `--all-unsolved`                             | Auto-progress through unsolved puzzles in turn.                                                 |
| `--auto-next`                                | After a solve, advance to the next puzzle automatically.                                        |
| `--puzzle-min-bits N`, `--puzzle-max-bits N` | Bound `--all-unsolved` to a bit range.                                                          |
| `--puzzle-target <addr>`                     | Override target Bitcoin address (independent of `--puzzle N`).                                  |
| `--puzzle-start <hex>`                       | Override range start (`0x...`).                                                                 |
| `--puzzle-end <hex>`                         | Override range end.                                                                             |
| `--pubkey <hex>`                             | 33-byte compressed pubkey. Only needed for non-bundled targets.                                 |
| `--puzzle-checkpoint <file>`                 | Save / resume search state across runs.                                                         |
| `--dp-bits N`                                | Distinguished-point bits for kangaroo. Default auto. Manual: 16 to 28.                          |
| `--random` / `--sequential`                  | Search direction within the range. Default: random.                                             |
| `--analyze`                                  | Print ROI ranking and exit. No search.                                                          |
| `--no-smart`                                 | Disable ROI-based auto-selection; pick the lowest-numbered unsolved instead.                    |

### Pool mode

| Flag                     | Effect                                                              |
| ------------------------ | ------------------------------------------------------------------- |
| `--pool <url>`, `-p`     | `jlps://` (TLS), `jlp://` (plaintext), or `http://` (HTTP variant). |
| `--worker <addr>`, `-w`  | Bitcoin address for pool rewards. Required for `--pool`.            |
| `--pool-password <pass>` | Optional. Collision Protocol does not require it.                   |
| `--pool-api-key <key>`   | Optional, for HTTP pools.                                           |

### GPU and tuning

| Flag                     | Effect                                                                |
| ------------------------ | --------------------------------------------------------------------- |
| `--gpus 0,1,3`, `-g`     | Specific GPU IDs. Default: all detected.                              |
| `--batch-size N`         | Keys per batch. Default 4M. Tune with `--calibrate`.                  |
| `--calibrate`            | Run batch-size calibration (also runs automatically on first launch). |
| `--force-calibrate`      | Force re-calibration even if a saved value exists.                    |
| `--benchmark`            | Synthetic GPU benchmark. Default 30 seconds.                          |
| `--benchmark-time <sec>` | Benchmark duration override.                                          |

### General

| Flag                    | Effect                                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| `--config <file>`, `-c` | Config file path. Default search: `./config.yml`, `./config.yaml`, `~/.collider/config.yml`. |
| `--verbose`, `-v`       | Verbose output.                                                                              |
| `--debug`               | Debug output.                                                                                |
| `--help`, `-h`          | Help.                                                                                        |

### Pro mode (license-gated) **(PRO VERSION ONLY)**

These flags belong to the brain-wallet pipeline that ships only in the Pro edition. The Free parser accepts them but the runner short-circuits with a Pro-feature hint, so they are visible in `./collider --help` everywhere; the actual work happens in a Pro build.

| Flag                            | Effect                                                                                                           |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `--activate <KEY>`              | **(PRO VERSION ONLY)** Activate a Pro license key (run once after purchase).                                     |
| `--brainwallet`                 | **(PRO VERSION ONLY)** Brain-wallet pipeline. Requires `--bloom`.                                                |
| `--brainwallet-v2`              | **(PRO VERSION ONLY)** Multi-address bloom probe (compressed + uncompressed P2PKH + P2SH-P2WPKH per passphrase). |
| `--brainwallet-setup`           | **(PRO VERSION ONLY)** Interactive setup wizard for the brain-wallet pipeline.                                   |
| `--bloom <file.blf>`            | **(PRO VERSION ONLY)** Bloom filter of funded addresses (built with the `build_bloom` tool).                     |
| `--bloom-tight <file.blf>`      | **(PRO VERSION ONLY)** Tight CPU-side bloom that re-probes empty-hit candidates from the loose GPU bloom.        |
| `--verify-set <file.uvrf>`      | **(PRO VERSION ONLY)** UVRF verify-set that rejects bloom false positives before they reach the user.            |
| `--warpwallet-salt <salt>`      | **(PRO VERSION ONLY)** WarpWallet brain-wallet scheme (scrypt + PBKDF2, Keybase-style).                          |
| `--bip-scan` **(v1.5.0)**       | **(PRO VERSION ONLY)** BIP-39 mnemonic scanner. Requires `--bip-scan-wordlist` or `--bip-combinatorial`.         |
| `--bip-scan-wordlist <file>`    | **(PRO VERSION ONLY)** Wordlist of candidate mnemonic phrases (one per line, whitespace-separated words).        |
| `--bip-combinatorial`           | **(PRO VERSION ONLY)** Exhaustive BIP-39 entropy-space enumeration; implies `--bip-scan`.                        |
| `--bip-combinatorial-words {N}` | **(PRO VERSION ONLY)** Mnemonic length for combinatorial mode. One of 12, 15, 18, 21, 24. Default 12.            |
| `--no-bip-gpu`                  | **(PRO VERSION ONLY)** Force CPU-side BIP scanner (default: GPU-accelerated PBKDF2-HMAC-SHA512 + EC derivation). |
| `--puzzle-only-v2`              | **(PRO VERSION ONLY)** v2 multi-scheme weak-PRNG kernel (Milk Sad, Profanity, Trust Wallet, glibc, MSVC, Java).  |
| `--schemes <list>`              | **(PRO VERSION ONLY)** Comma-separated weak-PRNG schemes for `--puzzle-only-v2`.                                 |
| `--resume`                      | **(PRO VERSION ONLY)** Resume the brain-wallet scan from the last checkpoint.                                    |
| `--save-interval N`             | **(PRO VERSION ONLY)** Save state every N candidates.                                                            |
| `--cpu-rules`                   | **(PRO VERSION ONLY)** Force CPU-side rule expansion.                                                            |

Pro license activation: pass `--activate <KEY>` once after purchase. A binary launched without an activated license drops to the interactive flow and directs you to [collisionprotocol.com/pro](https://collisionprotocol.com/pro). The activated key is HMAC-cached at `~/.collider/license.cache` with a 24-hour TTL; subsequent runs verify offline against that cache and re-validate against the issuer endpoint over TLS when the cache expires.

---

## Configuration

CLI flags always win. Anything not on the command line falls through to `config.yml` (in the working directory or `~/.collider/`), then to the built-in defaults.

A documented example with every section is at [`example-config.yml`](example-config.yml). The full schema is in [`src/core/yaml_config.hpp`](src/core/yaml_config.hpp).

Minimum pool config:

```yaml
pool:
  worker: "1YourBitcoinAddressForRewards"
  url: "jlps://collisionprotocol.com:17403"
```

Standalone with a custom range:

```yaml
puzzle:
  number: 71 # Used for record-keeping; range below overrides.
  kangaroo: false # No bundled pubkey for arbitrary addresses.
  target: "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"
  start: "0x40000000000000000"
  end: "0x7ffffffffffffffff"
```

Full schema and precedence rules: [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

---

## Build from source

Prebuilt binaries are the path of least resistance ([GitHub Releases](https://github.com/hevnsnt/collider/releases)). For source builds, the canonical two-command path that works on a fresh clone with no extra flags:

- **Linux**: `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- **Windows**: same, from the "x64 Native Tools Command Prompt for VS 2022". vcpkg auto-bootstraps for OpenSSL.
- **macOS**: `./build_macos.sh free` (sets `OPENSSL_ROOT_DIR` from Homebrew and runs Ninja).

The free build defaults to producing **only** the `collider` binary. Tests, benchmarks, and CLI tools (`build_bloom`, `generate_license`) are opt-in via `-DCOLLIDER_BUILD_TESTS=ON`, `-DCOLLIDER_BUILD_BENCHMARKS=ON`, `-DCOLLIDER_BUILD_TOOLS=ON`. The shorter `-DBUILD_TESTS=ON` / `-DBUILD_BENCHMARKS=ON` / `-DBUILD_TOOLS=ON` are accepted as aliases.

Full per-platform guides:

- [docs/INSTALL.md](docs/INSTALL.md) - prerequisites, CUDA setup, troubleshooting.
- [docs/BUILD-MACOS.md](docs/BUILD-MACOS.md) - Metal specifics, embedded shaders, build flags.

CUDA architecture defaults: `75;86;89;120` (Turing, Ampere, Ada, Blackwell desktop). Override with `-DCMAKE_CUDA_ARCHITECTURES="89"` (or whichever SM matches your card) for ~3x faster compile times. Note: sm_120 is desktop Blackwell (RTX 5090, RTX PRO 6000); sm_100 is datacenter Blackwell (B100/B200) and is not in the default.

---

## Performance

Benchmarked per-GPU numbers live in the [release notes](https://github.com/hevnsnt/collider/releases) for each tagged version. Throughput depends strongly on driver version, batch size, and which kernel path is exercised; static numbers in a README go stale within weeks of the next driver release.

For a fresh number on your hardware:

```bash
./collider --benchmark
./collider --benchmark --benchmark-time 60
```

The Free benchmark measures CPU and GPU SHA-256 throughput so operators can validate that the hardware is reachable and approximately matches what other tools see on the same silicon. The Pro benchmark runs the full brain-wallet fused kernel (SHA-256 -> secp256k1 -> hash160 -> bloom probe) and reports per-stage and end-to-end rates. For the standalone benchmark driver with the full per-stage table:

```bash
./bench_gpu_pipeline --time 30 --gpu 0
```

---

## Status

Current release: **v1.5.0 (stable, public)** -- Theft-Resistance Architecture (Mainnet).

v1.5.0 is a pool-architecture rewrite that closes the v1.4.x worker self-solve theft window. In v1.4.x a pool worker who happened to find the cross-collision computed the puzzle's private key locally and could sweep the funds before the pool ever saw the solution. v1.5.0 denies any single worker the data needed to compute the key (TAME-only or WILD-only assignment, no host-side cross-type hashtable, server-only solving). The full v1.5 security audit cleared all five mainnet-blocking findings. 305 tests pass across collider-pro and collision-protocol, 2 skipped, 0 failures. The wire-format bump to **JLP v3** is a hard cutover: v1.4.x clients are refused at AUTH with `UPGRADE_REQUIRED`.

This release also adds the **BIP-39 / BIP-32 mnemonic scanner (Pro)**: a partial-recall and combinatorial mnemonic recovery tool that walks every historical and modern derivation path (Electrum 2.x, MultiBit HD, blockchain.info, early Bitcoin Core HD, BIP-44, BIP-49, BIP-84) against the same 100M+ funded-address bloom filter the brain-wallet runner uses.

### v1.5.0 highlights vs v1.4.x

- **Theft-resistant pool architecture**: asymmetric TAME-only / WILD-only kangaroo assignment, host-side cross-type hashtable disabled in pool mode, `report_solution` removed from the client, server-only key reconstruction with cross-provider sweep attestation before SOLUTION broadcast.
- **JLP wire protocol bumped to v3.** `WORK_ASN` adds `kangaroo_type`, `start_offset_a`, `start_offset_b`. v1.4.x workers are refused at AUTH; the network upgrades together.
- **Single-strike permanent IP ban** for type-mismatched DP submissions. Wrong-type DP is unambiguous binary modification, not honest-but-buggy behavior.
- **Hot wallet hardening** (server side, collision-protocol): argon2id KDF (m=64 MiB, t=3, p=4) + AES-256-GCM, 77-byte encrypted file at mode 0600. SOLUTION broadcast is gated on the sweep tx being observed by a SECOND mempool provider.
- **BIP-39 / BIP-32 mnemonic scanner (Pro)** with BIP-44 / BIP-49 / BIP-84 + pre-BIP-44 historical derivation profiles. `--bip-scan`, `--bip-scan-wordlist`, `--bip-combinatorial`.
- **`--brainwallet-v2` multi-address bloom probe** per passphrase: compressed P2PKH, uncompressed P2PKH, P2SH-P2WPKH.
- Brain-wallet sizing math hardened (max_product capped at pipeline capacity, no more "exceeds buffer capacity" floods on high `--max-words`).
- TUI warm-up overlay before the dashboard renders; honest throughput counter (no inflated batch-total counting refused overflow dispatches); UTF-8-safe row truncation; multi-GPU EC table init reliable on secondary devices.
- Migration guide at [docs/MIGRATION-v1.5.md](docs/MIGRATION-v1.5.md).

### Performance expectations (v1.5.0)

theCollider's CUDA secp256k1 + bloom pipeline currently runs at roughly **30 to 50% of the state-of-the-art** (RCKangaroo, libsecp256k1, brainflayer) on the same silicon. The kangaroo path links RCKangaroo directly and inherits its throughput; the brain-wallet fused pipeline does not yet share that field arithmetic and is the gap.

The gap is concentrated in three known areas, queued for v1.6.0:

- 32-bit limb PTX field arithmetic where SOTA uses hand-tuned 64-bit limbs.
- No GLV decomposition on the brain-wallet scalar multiply path (`src/gpu/glv_decompose.cuh` exists but is not wired into `fused_pipeline.cu`).
- Single-threaded host generators that can starve a 4090/5090 pipeline.

The v1.5.0 release window was spent on theft-resistance and BIP scanner correctness; the crypto-pipeline rewrite originally scoped for v1.5.0 deferred to v1.6.0. Use the built-in benchmark for fresh numbers on your hardware:

```bash
./collider --benchmark
./collider --benchmark --benchmark-time 60
```

### Scheduled work

- **v1.6.0 (perf): crypto pipeline rewrite.** 64-bit limb PTX field arithmetic, GLV + Strauss-Shamir simultaneous double-scalar mul on the brain-wallet fused kernel, batched Montgomery inverse, host-generator thread pool. Target: 2.5 to 3.5x throughput vs v1.5.0 on the same GPU.
- **v1.6.0 (anticipated): puzzle and pool TUI parallel overhaul.** The multi-panel TUI shipped in v1.4.2 covers brain-wallet mode only; puzzle and pool modes still use single-line flat progress. v1.6.0 brings the same multi-panel treatment (range coverage, GPU panel, connection state, DP sparkline, hotkeys) to those modes.
- **v1.5.1 (anticipated): BIP scanner GPU fan-out.** The v1.5.0 BIP scanner offloads PBKDF2-HMAC-SHA512 and EC derivation to the GPU but the candidate-phrase loop itself is single-threaded for clarity. Multi-threaded fan-out is embarrassingly parallel and is the obvious next lift.

### Known limitations in v1.5.0

- **Mac binary**: `Build and Release (Free)` ships Linux + Windows only at v1.5.0. The standalone Apple Silicon Metal kangaroo and brute-force ship in source but the CI macOS link path has a pre-existing `bsgs_solve` undefined-symbol issue on arm64; tracked separately, not a v1.5.0 regression.
- **Standalone puzzle kangaroo save/resume**: works. Auto-saved to `~/.collider/state/kangaroo_herd_puzzle_<N>.kang` on shutdown; resumed with `--resume-kangaroo`. Routes through a patched RCKangaroo (`third_party/RCKangaroo/.patches/save-load-state.patch`).
- **Pool kangaroo save/resume**: works. Auto-saved to `~/.collider/state/kangaroo_herd_<work_id>.kang` and resumed on next chunk assignment.
- **Puzzle and pool TUI**: still flat-line single-line progress; multi-panel TUI is v1.6.0 scope.
- **CUDA crypto pipeline throughput**: see "Performance expectations" above; v1.6.0 closes this gap.
- **AMD ROCm**: no port planned.

## Changelog

Full release history with breaking changes, security fixes, and migration notes lives at **[docs/CHANGELOG.md](docs/CHANGELOG.md)**.

| Release | Date       | Headline                                                                                          |
| ------- | ---------- | ------------------------------------------------------------------------------------------------- |
| v1.5.0  | 2026-05-21 | Theft-Resistance Architecture (Mainnet). JLP v3. TAME/WILD asymmetric assignment. BIP scanner.    |
| v1.4.4  | 2026-05-20 | Cross-platform build + CI hardening (Linux libcurl guardrail, Pro/Free CI split).                 |
| v1.4.3  | 2026-05-19 | Pool DP big-endian fix. Client-side ban detection. Edition-aware CI.                              |
| v1.4.2  | 2026-05-17 | A-tier stabilization. Full-pipeline benchmark. SecureBuffer. Per-GPU work balancer. Bloom FP pin. |
| v1.4.1  | 2026-05-10 | Quality lift. Apple Silicon Metal complete. DP_BATCH_V2 sequence nonce. Server resilience.        |
| v1.4.0  | 2026-05-04 | Adversarial-review-driven major. GPU byte-swap fix. TLS hostname verify. AUTH state machine.      |
| 1.3.x   | --         | Pre-public private-only history.                                                                  |

JLP wire-format protocol version tracks independently of the binary version. Wire-format breaking changes bump the protocol version and the binary major version together.

| Platform          | GPU backend            | Status        |
| ----------------- | ---------------------- | ------------- |
| Linux x64         | CUDA 12.x              | Supported     |
| Windows x64       | CUDA 12.x              | Supported     |
| macOS arm64       | Metal (M1, M2, M3, M4) | Supported     |
| macOS x64 (Intel) | none                   | Not supported |

---

## Where to go from here

| For                                         | See                                                                            |
| ------------------------------------------- | ------------------------------------------------------------------------------ |
| Install on a new machine                    | [docs/INSTALL.md](docs/INSTALL.md)                                             |
| Build options, CMake flags, troubleshooting | [docs/INSTALL.md](docs/INSTALL.md), [docs/BUILD-MACOS.md](docs/BUILD-MACOS.md) |
| `config.yml` schema and precedence          | [docs/CONFIGURATION.md](docs/CONFIGURATION.md)                                 |
| Source tree map for contributors            | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)                                   |
| Joining the pool, share accrual, etiquette  | [docs/POOL.md](docs/POOL.md)                                                   |
| JLP wire protocol (third-party clients)     | [docs/JLP-PROTOCOL.md](docs/JLP-PROTOCOL.md)                                   |
| GPU crypto correctness tests                | [docs/CRYPTO-VALIDATION.md](docs/CRYPTO-VALIDATION.md)                         |
| v1.4.x to v1.5.0 upgrade path               | [docs/MIGRATION-v1.5.md](docs/MIGRATION-v1.5.md)                               |
| Full release history                        | [docs/CHANGELOG.md](docs/CHANGELOG.md)                                         |

---

## License and community

The Free edition is **MIT-licensed**. See [LICENSE](LICENSE).

`third_party/RCKangaroo/` is GPLv3-licensed (RetiredCoder, 2024). Builds that link RCKangaroo carry its license forward in the binary.

The Pro edition is a separate, license-gated build. Pro source is **not** MIT; binaries are delivered to paying customers at [collisionprotocol.com/pro](https://collisionprotocol.com/pro).

Issues, bug reports, and feature requests:

- Public (Free edition): [github.com/hevnsnt/collider/issues](https://github.com/hevnsnt/collider/issues)
- Pool / protocol questions: see [github.com/hevnsnt/collision-protocol](https://github.com/hevnsnt/collision-protocol)
