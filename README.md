# theCollider

> A GPU-accelerated solver for the [Bitcoin Puzzle Challenge](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx), built for the unsolved keys at the hard end of the curve.

theCollider is the most complete open-source toolkit for the Bitcoin Puzzle Challenge today: solo Pollard's Kangaroo and brute-force search on CUDA (Windows, Linux) and Metal (macOS Apple Silicon), a pool client for distributed solving via the [Collision Protocol](https://collisionprotocol.com) pool, and the canonical bundled record of every solved puzzle to date. A paid Pro edition adds a brain-wallet opportunistic pipeline and Brainwallet cracker on top of the same core.

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

[RCKangaroo](https://github.com/RetiredCoder/RCKangaroo) is the current state-of-the-art kangaroo implementation. It introduced the SOTA method with K=1.15 (versus K=2.1 for classic three-way kangaroo, ~1.8x fewer operations and ~1.8x less DP storage), and it benchmarks at roughly 8 GKeys/s on an RTX 4090. **theCollider uses RCKangaroo as its CUDA kangaroo backend.** Credit where it belongs: the kernel doing the heavy lifting on Windows and Linux is RetiredCoder's, redistributed here under its GPLv3 license in `third_party/RCKangaroo/`.

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
| Brain-wallet passphrase pipeline                               | No                       | Yes, Pro edition                                            |
| Hashcat-style rule engine + bloom-filter lookup                | No                       | Yes, Pro edition                                            |
| License                                                        | GPLv3                    | MIT (Free), commercial (Pro)                                |

RCKangaroo wins on raw CUDA kangaroo benchmark tuning. theCollider wins on platform coverage, distributed-solving infrastructure, and operator ergonomics. The two are complementary, not competitive; theCollider is "RCKangaroo plus everything around it that you would otherwise have to build yourself".

### Other solvers in the space

- [BSGS / Pollard's Rho variants](https://github.com/JeanLucPons/BSGS) tend to be 32- to 64-bit ceilings; they are not viable at puzzle #135 scale.
- [Kangaroo (JLP)](https://github.com/JeanLucPons/Kangaroo) is the canonical older kangaroo solver; predates the SOTA method.
- Web-based "checkers" are not solvers; they query precomputed databases.

theCollider treats kangaroo, brute force, and the brain-wallet path (Pro) as one product with shared scaffolding (CLI, config, multi-GPU calibration, pool, telemetry, checkpointing), instead of three separate tools.

---

## Pool mode (Collision Protocol)

For puzzles above #135, no single machine has the compute budget to finish in a reasonable time. The pool solves this by sharding the search range across many workers and letting them share distinguished points (DPs). When a collision is detected on the server, the private key falls out of the math.

```bash
./collider --pool jlps://collisionprotocol.com:17403 \
           --worker 1YourBitcoinAddressForRewards
```

What happens after AUTH:

1. The server assigns your worker a chunk of the kangaroo search range plus a `work_id` attestation token.
2. Your GPU runs kangaroo on that chunk, producing distinguished points (X coordinate with a configurable bit-count of leading zeros).
3. Your client batches DPs into `DP_BATCH_V2` frames every few seconds and submits.
4. The server tracks the cumulative DP count per worker. The Bitcoin address you authenticated with is the reward-accrual key.
5. When two DPs collide (one tame, one wild), the server reconstructs the private key and pushes a `SOLUTION` frame to every connected worker.
6. Workers who contributed valid DPs accrue share-of-pool credit, redeemable when a puzzle is solved per the pool's payout policy. See [docs/POOL.md](docs/POOL.md) for the mechanics.

The wire protocol has anti-cheat: every DP carries a `work_id` attestation (you cannot resubmit captured DPs against a different chunk) and a per-DP monotonic `sequence` nonce (you cannot replay a captured `DP_BATCH`). AUTH replay is blocked by a 30-second timestamp drift window. Invalid DPs cost you reputation per IP, with bans escalating from 1 hour to permanent.

The reference server is the [Collision Protocol](https://github.com/hevnsnt/collision-protocol) project, deployed at `collisionprotocol.com`. Third-party servers can implement the same protocol (it is documented in full at [docs/JLP-PROTOCOL.md](docs/JLP-PROTOCOL.md)).

---

## Free vs. Pro

theCollider ships in two editions from one source tree.

| Capability                                           | Free (MIT) | Pro (commercial) |
| ---------------------------------------------------- | ---------- | ---------------- |
| Pollard's Kangaroo (CUDA, Metal)                     | Yes        | Yes              |
| Brute-force puzzle search (CUDA, Metal)              | Yes        | Yes              |
| Pool client (JLP, TLS)                               | Yes        | Yes              |
| Bundled puzzle history (all 82 solved)               | Yes        | Yes              |
| Multi-GPU support with calibration                   | Yes        | Yes              |
| Save / resume checkpoints                            | Yes        | Yes              |
| GPU benchmark                                        | Yes        | Yes              |
| Brain-wallet passphrase pipeline                     |            | Yes              |
| Hashcat-style rule engine (GPU)                      |            | Yes              |
| Funded-address bloom filter (100+ million addresses) |            | Yes              |
| PCFG + Markov passphrase generators                  |            | Yes              |
| v2 puzzle-mode multi-scheme kernel                   |            | Yes              |
| License gating (Ed25519, offline-verifiable)         |            | Yes              |

Pro licenses are available at [collisionprotocol.com/pro](https://collisionprotocol.com/pro). The Pro source tree is not public; the Free build is the same source minus the Pro modules, generated automatically by `scripts/sync-to-free.sh` and published at [github.com/hevnsnt/collider](https://github.com/hevnsnt/collider).

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

| Flag                  | Effect                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------- |
| `--brainwallet`       | **(PRO VERSION ONLY)** Brain-wallet pipeline. Requires `--bloom`.                            |
| `--brainwallet-setup` | **(PRO VERSION ONLY)** Interactive setup wizard for the brain-wallet pipeline.               |
| `--bloom <file.blf>`  | **(PRO VERSION ONLY)** Bloom filter of funded addresses (built with the `build_bloom` tool). |
| `--resume`            | **(PRO VERSION ONLY)** Resume the brain-wallet scan from the last checkpoint.                |
| `--save-interval N`   | **(PRO VERSION ONLY)** Save state every N candidates.                                        |
| `--cpu-rules`         | **(PRO VERSION ONLY)** Force CPU-side rule expansion.                                        |

Pro license activation is handled at first run via the interactive flow, not a separate flag. A binary that requires activation will prompt and direct you to [collisionprotocol.com/pro](https://collisionprotocol.com/pro) on first launch.

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

Prebuilt binaries are the path of least resistance ([GitHub Releases](https://github.com/hevnsnt/collider/releases)). For source builds, the canonical entry points are:

- **Linux**: `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel`
- **Windows**: same, from the "x64 Native Tools Command Prompt for VS 2022". vcpkg auto-bootstraps for OpenSSL.
- **macOS**: `./build_macos.sh free` (sets `OPENSSL_ROOT_DIR` from Homebrew and runs Ninja).

Full per-platform guides:

- [docs/INSTALL.md](docs/INSTALL.md) - prerequisites, CUDA setup, troubleshooting.
- [docs/BUILD-MACOS.md](docs/BUILD-MACOS.md) - Metal specifics, embedded shaders, build flags.

CUDA architecture defaults: `86;89;100` (Ampere, Ada, Blackwell). Override with `-DCMAKE_CUDA_ARCHITECTURES="89"` (or whichever SM matches your card) for ~3x faster compile times.

---

## Performance

Benchmarked per-GPU numbers live in the [release notes](https://github.com/hevnsnt/collider/releases) for each tagged version. Throughput depends strongly on driver version, batch size, and which kernel path is exercised; static numbers in a README go stale within weeks of the next driver release.

For a fresh number on your hardware:

```bash
./collider --benchmark
./collider --benchmark --benchmark-time 60
```

The benchmark prints kangaroo step rate and EC-multiply throughput per GPU.

---

## Status

Latest release: **v1.4.1** (2026-05-10). Highlights:

- Apple Silicon kangaroo and brute-force pipelines (no more CPU fallback on Mac).
- Per-DP sequence-nonce replay protection on the pool wire (`DP_BATCH_V2`).
- All 82 confirmed-solved puzzles bundled (1 to 70 plus every multiple of 5 from 75 to 130).
- Graceful kangaroo to brute-force demotion when no pubkey is available.
- Live BTC balance on solved banners via mempool.space.
- BTC payout address persistence (`~/.collider/config.yml`) from guided mode.

Full history: [docs/CHANGELOG.md](docs/CHANGELOG.md).

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
| Full release history                        | [docs/CHANGELOG.md](docs/CHANGELOG.md)                                         |

---

## License and community

The Free edition is **MIT-licensed**. See [LICENSE](LICENSE).

`third_party/RCKangaroo/` is GPLv3-licensed (RetiredCoder, 2024). Builds that link RCKangaroo carry its license forward in the binary.

The Pro edition is a separate, license-gated build. Pro source is **not** MIT; binaries are delivered to paying customers at [collisionprotocol.com/pro](https://collisionprotocol.com/pro).

Issues, bug reports, and feature requests:

- Public (Free edition): [github.com/hevnsnt/collider/issues](https://github.com/hevnsnt/collider/issues)
- Pool / protocol questions: see [github.com/hevnsnt/collision-protocol](https://github.com/hevnsnt/collision-protocol)
