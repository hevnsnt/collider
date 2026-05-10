# theCollider

GPU-accelerated solver for the [Bitcoin Puzzle Challenge](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx) (1000 BTC distributed across 160 secp256k1 ranges of progressively larger size, designed by the Bitcoin author to be solvable only at scale).

**theCollider** is two products with a shared core:

- **theCollider** (free, MIT). Pollard's Kangaroo and brute-force scanners for the Bitcoin Puzzle Challenge, plus a JLP pool client for distributed solving.
- **theCollider Pro**. Adds the brain-wallet pipeline (passphrase generation plus bloom-filter lookup against funded address sets), license-gated. See [collisionprotocol.com/pro](https://collisionprotocol.com/pro).

This README covers both. Pro-only features are tagged **(PRO VERSION ONLY)**.

---

## Background you actually need

### Which puzzles are kangaroo-able

Pollard's Kangaroo recovers a private key from a **public key plus range** in roughly the square root of n group operations. It cannot run on an address alone. Addresses are SHA-256+RIPEMD-160 of the pubkey, and that hash is one-way.

A pubkey is only knowable for an address that has had a _spending_ transaction (the input script reveals it). For the Bitcoin Puzzle Challenge:

- **Solved puzzles** (1 to 70 plus every multiple of 5 up to 130, 82 in total). Pubkey known by definition (someone spent them). Bundled in `data/puzzle_history.json`.
- **Multiples of 5 in 71 to 160** (75, 80, 85, ..., 160). Partial-solved and spent at some point. Pubkey revealed. Bundled.
- **Non-multiples of 5 above 71** (71, 72, 73, 74, 76, 77, ...). Pure addresses, never spent. **The pubkey is mathematically unknown to anyone, anywhere**, until someone partial-solves and broadcasts a spending transaction. Until then: brute force is the only option.

`./collider --puzzle N --kangaroo` does the right thing in v1.4.1 when puzzle N has no bundled pubkey:

- In `--all-unsolved` or `--auto-next` worklist mode: silently demote that puzzle to brute force and continue.
- In single-puzzle interactive mode (TTY): prompt for a pubkey, with ENTER falling back to brute force.
- In single-puzzle non-interactive mode (piped / CI): silently demote and log it.

`--kangaroo` on a known-pubkey puzzle works on Windows and Linux (CUDA backend) and on macOS (Metal backend, with the Jacobian rewrite shipped in v1.4.1).

### Brute force vs. Kangaroo, in one paragraph

Kangaroo runs in roughly the square root of n group operations. Brute force runs in n. For a 71-bit puzzle, that is the difference between roughly 2^35 and 2^71 operations. If a puzzle's pubkey is known, kangaroo is the only sane choice. If it is not, brute force is the only choice. On consumer hardware, anything above roughly 64 bits is multi-decade brute-force territory, so attention concentrates on the kangaroo-able multiples of 5.

---

## Quick Start

### Get a binary

[GitHub releases](https://github.com/hevnsnt/collider/releases) (free edition). Linux x64 plus CUDA, Windows x64 plus CUDA, macOS arm64 plus Metal. **(PRO VERSION ONLY)** builds are issued per-license to paying customers via the dashboard.

### Run the easiest unsolved puzzle

```
./collider
```

Default behavior: ROI-rank all puzzles by reward divided by expected ops, pick the best, fire kangaroo if the pubkey is bundled, fall back to brute force otherwise.

### Run a specific puzzle with kangaroo

```
./collider --puzzle 75 --kangaroo
```

(Puzzle 75 has a revealed pubkey; bundled. Kangaroo on the 75-bit range targets roughly 2^37 steps.)

### Pool mode (distributed)

```
./collider --pool jlps://collisionprotocol.com:17403 --worker 1YourBitcoinAddress
```

You receive work assignments (chunk plus work_id), submit distinguished points, and earn shares. The pool's anti-cheat verifier (math plus work_id attestation plus AUTH replay protection plus per-DP sequence nonce in v1.4.1) is documented at [collisionprotocol.com](https://collisionprotocol.com) and in [docs/JLP-PROTOCOL.md](docs/JLP-PROTOCOL.md).

---

## CLI reference

Full list: `./collider --help`. The most useful flags:

| Flag                                          | Effect                                                                                                             |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `--puzzle N`, `-P`                            | Target puzzle N. Default: ROI-pick the best unsolved puzzle.                                                       |
| `--kangaroo`                                  | Force kangaroo. Requires a known pubkey for the target. Demotes to brute force if no pubkey is available (v1.4.1). |
| `--puzzle-target <addr>`                      | Override target Bitcoin address (independent of `--puzzle N`).                                                     |
| `--puzzle-start <hex>`                        | Override range start. Format: `0x...`.                                                                             |
| `--puzzle-end <hex>`                          | Override range end.                                                                                                |
| `--pubkey <hex>`                              | 33-byte compressed pubkey (02.../03...). Only needed when scanning a target whose pubkey isn't bundled (rare).     |
| `--all-unsolved`                              | Auto-progress through unsolved puzzles in turn.                                                                    |
| `--puzzle-min-bits N` / `--puzzle-max-bits N` | Bound `--all-unsolved` to a bit-range.                                                                             |
| `--auto-next`                                 | After solving, advance to the next puzzle automatically.                                                           |
| `--puzzle-checkpoint <file>`                  | Save / resume search state.                                                                                        |
| `--dp-bits N`                                 | Distinguished-point bits for kangaroo. Default: auto. Manual override: 16 to 28.                                   |
| `--random` / `--sequential`                   | Search direction within the range. Default: random.                                                                |
| `--analyze`                                   | Print the puzzle ROI ranking and exit. No search.                                                                  |
| `--no-smart`                                  | Disable ROI-based puzzle auto-selection (when no `--puzzle N`). Picks the lowest-numbered unsolved puzzle first.   |
| `--gpus 0,1,3`, `-g`                          | Specific GPU IDs. Default: all detected.                                                                           |
| `--batch-size N`                              | Keys per batch. Default 4M. Tune with `--calibrate`.                                                               |
| `--calibrate`                                 | Run the GPU batch-size calibration (also runs automatically on first launch).                                      |
| `--force-calibrate`                           | Force re-calibration even if a saved value exists.                                                                 |
| `--pool <url>`, `-p`                          | Pool mode. `jlps://` for TLS, `jlp://` for plaintext, `http://` for HTTP variant. `--worker <addr>` is required.   |
| `--worker <addr>`, `-w`                       | Bitcoin address for pool rewards.                                                                                  |
| `--pool-password <pass>`                      | Optional, for pools that require it (Collision Protocol does not).                                                 |
| `--pool-api-key <key>`                        | Optional, for HTTP pools that require API-key auth.                                                                |
| `--config <file>`, `-c`                       | Use a non-default config file. Default search: `./config.yml`, `./config.yaml`, `~/.collider/config.yml`.          |
| `--benchmark`                                 | Run a synthetic GPU benchmark (default: 30s). Prints throughput.                                                   |
| `--benchmark-time <sec>`                      | Override the benchmark duration.                                                                                   |
| `--verbose`, `-v`                             | Verbose output.                                                                                                    |
| `--debug`                                     | Debug output for troubleshooting.                                                                                  |
| `--help`, `-h`                                | Show help.                                                                                                         |

**(PRO VERSION ONLY)** flags (license-gated; ignored or hint-printed in the free build):

| Flag                  | Effect                                                                           |
| --------------------- | -------------------------------------------------------------------------------- |
| `--brainwallet`       | Brain-wallet mode. Requires `--bloom`.                                           |
| `--brainwallet-setup` | Interactive setup wizard for the brain-wallet pipeline.                          |
| `--bloom <file.blf>`  | Bloom filter of funded addresses (built with the `build_bloom` tool).            |
| `--resume`            | Resume the brain-wallet scan from the last checkpoint.                           |
| `--save-interval N`   | Save state every N candidates.                                                   |
| `--cpu-rules`         | Force CPU-side rule expansion (allows multi-GPU parallelism on certain targets). |
| `--activate <KEY>`    | Activate the Pro license key (run once after purchase).                          |

---

## config.yml

`config.yml` lives in the working directory or in `~/.collider/`. CLI flags always win over config values when both are set.

A documented example ships at the repo root: [`example-config.yml`](example-config.yml). The full schema is in [`src/core/yaml_config.hpp`](src/core/yaml_config.hpp).

Minimum useful pool config:

```yaml
pool:
  worker: "1YourBitcoinAddressForRewards"
  url: "jlps://collisionprotocol.com:17403"
```

Standalone with a custom range:

```yaml
puzzle:
  number: 71 # Used for record-keeping; range below overrides.
  kangaroo: false # Brute force (no pubkey for arbitrary addresses).
  target: "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"
  start: "0x40000000000000000"
  end: "0x7ffffffffffffffff"
```

Standalone with a non-bundled but known pubkey (rare):

```yaml
puzzle:
  number: 71
  kangaroo: true
  pubkey: "02ABCDEF...your 33-byte compressed pubkey hex..."
  start: "0x40000000000000000"
  end: "0x7ffffffffffffffff"
```

---

## Pool mode usage

The pool client speaks the JLP wire protocol over TCP (or TCP+TLS via `jlps://`). v1.4.1 wire properties:

- **AUTH timing**. Worker must send AUTH within 30 seconds of TCP connect or the server drops.
- **DP work_id attestation**. Every distinguished point you submit cryptographically attests which assigned chunk it came from. Server rejects mismatches.
- **Per-DP sequence nonce** (v1.4.1 B.1). Monotonic counter per (worker, work_id) starting at 0. The server tracks an expected window and rejects out-of-window sequences (replays of captured DP_BATCHes).
- **TLS**. When the URL is `jlps://`, certificates are validated against the system trust store. v1.4.1 fails hard at init if no trust anchors are loadable (was a silent fallback in v1.4.0).

Optional pool authentication:

- `--pool-password`. Passes a password in the AUTH frame's password slot. Collision Protocol's public pool ignores this. Pools that need it (private pools, throttled-tier pools) advertise it in their docs.
- `--pool-api-key`. For HTTP-only pools that require a header-based API key. Not used over JLP. Kept for compatibility with non-JLP pool integrations.

Most users on Collision Protocol only need `--pool jlps://collisionprotocol.com:17403 --worker <btc-addr>`.

The full wire format is documented in [docs/JLP-PROTOCOL.md](docs/JLP-PROTOCOL.md). Third-party clients can implement against that document plus `protocol/jlp.yaml` (the IDL).

---

## Build from source

### Prerequisites

- CMake 3.20 or newer
- A C++20 compiler (MSVC 2022, GCC 11 or newer, Apple Clang 14 or newer)
- **CUDA backend** (Linux / Windows): CUDA Toolkit 12.x
- **Metal backend** (macOS): Apple Silicon, Xcode Command Line Tools
- OpenSSL (optional but recommended; enables TLS pool connections)
- vcpkg (Windows; auto-bootstraps from the repo root)

### Free build

```
git clone https://github.com/hevnsnt/collider.git
cd collider
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target collider --parallel
```

Output: `build/collider` (or `build/collider.exe` on Windows).

For macOS, the canonical entry point is `./build_macos.sh` (configures Metal, sets `OPENSSL_ROOT_DIR` from Homebrew, runs Ninja). See [docs/BUILD-MACOS.md](docs/BUILD-MACOS.md).

CUDA architectures default to `86;89;100` (Ampere, Ada, Blackwell). To target a single architecture for faster builds, pass `-DCMAKE_CUDA_ARCHITECTURES="89"` (or whichever SM matches your card).

---

## Performance

Benchmarked numbers per GPU live in the [GitHub release notes](https://github.com/hevnsnt/collider/releases) for each tagged version. Numbers depend strongly on driver version, batch size, and which kernel path is exercised. Anything quoted statically here would go stale within weeks of the next driver release.

If you want a fresh number on your hardware:

```
./collider --benchmark
```

(30-second synthetic run by default; prints kangaroo step rate and EC-multiply throughput. Use `--benchmark-time <sec>` to change the duration.)

---

## Troubleshooting

### Kangaroo demoted to brute force

The puzzle you targeted does not have its pubkey bundled. v1.4.1 handles this gracefully:

- **Worklist mode** (`--all-unsolved` or `--auto-next`): the puzzle is silently downgraded to brute force and the run continues.
- **Single-puzzle interactive**: you are prompted for a pubkey. Press ENTER to fall back to brute force, or paste a 33-byte compressed pubkey if you have one from an external source.
- **Single-puzzle non-interactive** (piped, CI): silently demoted with a log line.

See "Which puzzles are kangaroo-able" above for which puzzles have bundled pubkeys.

### Pool client cannot connect

- Verify the URL. `jlps://` is TLS, `jlp://` is plaintext, `http://` is the HTTP-API variant.
- Verify the worker address is a real Bitcoin address (the pool's reward-payment field).
- Check the local time. The AUTH frame is timestamped on the server side; severe clock drift may produce auth failures.
- If you see a TLS error referencing a missing trust anchor: install `ca-certificates` (Linux) or run from a shell that has `SSL_CERT_FILE` or `SSL_CERT_DIR` set.

### "Brain wallet scanning is a Pro feature" in `--help`

Correct. That is the FREE build. Brain-wallet scanning is **(PRO VERSION ONLY)**, available at [collisionprotocol.com/pro](https://collisionprotocol.com/pro). The free edition handles the Bitcoin Puzzle Challenge (kangaroo plus brute force plus pool client).

### Reporting bugs

Issues at [github.com/hevnsnt/collider/issues](https://github.com/hevnsnt/collider/issues). Please include:

- Output of `./collider --help` (the build identifies its edition there).
- GPU model plus driver version (`nvidia-smi` on Linux/Windows, `system_profiler SPDisplaysDataType` on macOS).
- The exact command line you ran.
- A copy-paste of the output (not a screenshot when possible).

---

## License

MIT. See [LICENSE](LICENSE).

The Pro edition is a separate, license-gated build of the same source tree (with `-DCOLLIDER_PRO=ON` plus the brain-wallet sources that aren't in the public repo). Pro source is **not** MIT and is delivered as binaries to paying customers only.
