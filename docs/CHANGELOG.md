# Changelog

All notable changes to theCollider are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This changelog covers both the Free and **(PRO VERSION ONLY)** editions. Pro-only changes are tagged inline.

---

## [1.4.1] - 2026-05-10

Quality lift over v1.4.0. Server resilience, wire format hardening, macOS Metal completeness, and a CLI/runtime split that makes the codebase materially easier to extend.

### Added

- **macOS Metal kangaroo (standalone)**. v1.4.1 D.1 ships a Jacobian-coordinate rewrite of the kangaroo kernel for Apple Silicon; D.3 wires it into the standalone puzzle dispatch path. Mac users can now run `--puzzle <N> --kangaroo` natively without falling back to CPU.
- **macOS Metal brute force (standalone)**. v1.4.1 D.3 brings up the brute-force puzzle pipeline on Metal. The Mac binary no longer falls back to CPU for brute-force puzzle search.
- **Per-DP sequence nonce in `DP_BATCH_V2`**. v1.4.1 B.1 adds a 4-byte little-endian monotonic counter per `(worker, work_id)` to every distinguished-point submission. The server tracks an expected window and rejects out-of-window sequences (replays of captured `DP_BATCH`es). Wire size of `DistinguishedPointV2` grows from 74 to 78 bytes; the v1 `DistinguishedPoint` (66 bytes, no work_id) is still accepted for compatibility with deployed v1.2.x clients.
- **Live BTC balance on solved banner**. When a puzzle is solved, the banner shows the live BTC balance via mempool.space (libcurl, 5s timeout). Falls back to "balance unavailable (offline)" on network failure.
- **`--pubkey <hex>` CLI flag plus `puzzle.pubkey` config field**. Allow scanning a target whose pubkey is not in the bundled `data/puzzle_history.json` (rare; for newly revealed pubkeys or research targets).
- **`--puzzle-target`, `--puzzle-start`, `--puzzle-end` CLI flags plus matching config fields**. Per-run override of address and range without modifying the bundled puzzle data.
- **PuzzleDatabase coverage**. All 82 confirmed-solved puzzles are present (1 to 70 plus every multiple of 5 from 75 to 130). `--puzzle 32` works.
- **Protocol drift round-trip test**. v1.4.1 C7 round-trips every wire struct between the Python codegen and the C++ generated header to catch silent IDL/binding skew.
- **TLS hardening**. Pool TLS init fail-hards if no trust anchors are loadable (was a silent fallback in v1.4.0).
- **Server-side resilience batch**. C3 stale-chunk min-heap, C6/C8 batch processing, signal handler awaits shutdown, solution-broadcast retry/TaskGroup. Server changes ship in the `collision-protocol` sibling repo; client visibility is "fewer chunk-reissue surprises and fewer dropped solutions on server restart".

### Changed

- **`--kangaroo` no longer hard-fails on a no-pubkey puzzle.** New v1.4.1 behavior:
  - In `--all-unsolved` or `--auto-next` worklist mode: silently demote that puzzle to brute force and continue.
  - In single-puzzle interactive mode (TTY): prompt for a pubkey, with ENTER falling back to brute force.
  - In single-puzzle non-interactive mode: silently demote and log it.
- **`src/main.cpp` split into runtime modules**. v1.4.1 A.3 extracts `puzzle_solver`, `pool_solver`, `brain_wallet_runner`, `gpu_detection`, and `license_gate` from the monolithic `main.cpp`. The brain-wallet runner becomes Pro-only (excluded from the public Free repo).
- **CLI parser extracted to `src/cli/cli_parser.cpp`**. Verbatim move from `main.cpp`; no behavior change. Single source of truth for "what flags exist".
- **SHA-256 round logic unified across `.cu` kernels** (v1.4.1 D.2-SHA). One implementation, called from every kernel that needs SHA-256.
- **RIPEMD-160 round logic unified** (v1.4.1 D.2-RIPEMD). One implementation, shared.
- **secp256k1 modular-inverse implementations consolidated** (v1.4.1 BW-DEDUP-4). Audit and consolidation; all kernels now go through a single `mod_inv` implementation using literal `p-2` exponent.
- **Pimpl on heavy headers**. `streaming_brain_wallet.hpp` (C.1) and `brain_wallet_engine.hpp` (C.2) get a pimpl, dropping public ABI surface.

### Fixed

- **Multi-GPU EC-table race in `mega_fused_kernel.cu`** (v1.4.1 BW-B3) **(PRO VERSION ONLY)**. Per-batch pinned host buffers in `MultiGPUBrainWallet::process_batch` eliminated the F-01 aliasing race.
- **`replace_thread` noexcept** (v1.4.1 P-T2). Thread-replacement codepath in the pool reconnector cannot throw across the boundary.
- **Pool `MAX_RECONNECT_BACKOFF_MS`** unified into a single header so client and reconnector agree on the cap.
- **Bloom flush durability** in `dp_store.py` (v1.4.1 S-B4, server side). Commit DB before `bloom.add` so a crash mid-flush does not silently drop DPs.
- **Dead `semaphore.locked()` check** removed from `pool_server.py` (v1.4.1 S-B1, server side).

### Removed

- **`src/runtime/brain_wallet_runner.{cpp,hpp}` from the public Free repo**. v1.4.1 A.3 moved these to the `PRO_PATHS` exclusion list. Free builds short-circuit `--brainwallet` at the runner boundary instead.
- **`bloom_filter.cu`** consolidated into `h160_bloom_filter.cu` (v1.4.1 BW-DEDUP-3).

---

## [1.4.0] - 2026-05-04

Major release driven by a multi-track adversarial review on the v1.3.0-dev branch. The review found 87 issues across the GPU brain-wallet pipeline, JLP pool client, CLI/config layer, and concurrency primitives. v1.4.0 consolidates the fixes for every Critical and most High findings.

### Headline correctness fixes

- **GPU brain-wallet pipeline byte-swap bug** **(PRO VERSION ONLY)**. SHA-256 to uint256 scalar conversion was writing into `limbs[i]` instead of `limbs[7-i]` in five places, producing hash160 values that could not match real funded Bitcoin addresses. Latent since the kernel was written. GPU pubkeys now match canonical Bitcoin convention.
- **`secp256k1.cu`'s `mod_inv` rewritten**. The previous hand-written addition chain used a wrong exponent; replaced with right-to-left binary exponentiation against literal `p-2`.
- **`fused_pipeline.cu`'s `sha256_short` rewritten** to handle messages of any length (was silently truncating past 55 bytes).
- **TLS hostname verification in JLP pool client**. SNI plus `X509_VERIFY_PARAM_set1_host` with `NO_PARTIAL_WILDCARDS` plus default trust store.
- **AUTH state machine in JLP pool client**. Rejects `WORK_ASN`, `SOLUTION`, `DP_ACK`, `STATS_RSP` before `AUTH_OK`.
- **Bounded reconnect on `AUTH_FAIL`**: 3 attempts, jittered exponential backoff. Was unbounded.
- **`ssl_io_mutex_`** serializing `SSL_write` and `SSL_read`.
- **`validate_mode_mutex()`** in `parse_args` rejects mutually exclusive modes (`--brainwallet --pool` and similar).
- **`MegaWorkQueue::head/tail/completed`** promoted from `uint32_t` to `unsigned long long`. Persistent kernel uses 64-bit `atomicAdd`.
- **`__launch_bounds__(256, 4)`** on top-level kernels.

### Removed

- **`http_pool_client.{cpp,hpp}`** deleted. Silently leaked credentials when configured `https://` (D-C1). Use `jlps://` for TLS instead.
- **Wrong GLV decomposition** (`glv_decompose`, `ec_mul_glv`, `ec_add_glv_affine`) in `secp256k1.cu`. Decomposition was wrong; deleted to remove the silent landmine. Re-enabling requires porting libsecp256k1's `secp256k1_scalar_split_lambda`.

### Added

- GPU correctness test suite (`test_secp256k1_inv.cu`, `test_ec_table_consistency.cu`, `test_gpu_hash160.cu`, `test_cli_parser.cpp`). CUDA tests skip cleanly with code 77 on hosts without a GPU.
- `COLLIDER_PRO` build option. Default OFF (Free).
- Per-batch pinned host buffers in `MultiGPUBrainWallet::process_batch` **(PRO VERSION ONLY)**.
- `MEGA_FUSED_MAX_ITEMS_PER_LAUNCH = UINT32_MAX` cap **(PRO VERSION ONLY)**.

### Phase 9 v2 puzzle-mode kernel **(PRO VERSION ONLY)**

- New `--puzzle-only-v2` plus `--puzzle-keys`, `--schemes`, `--addr-types` flags drive a v2 brain-wallet pipeline targeting the Bitcoin Puzzle Challenge address set.
- macOS Metal port of the v2 kernel ships in this release.

---

## [1.3.x] and earlier

Pre-public history. The v1.3.x line was the last private-only series; v1.4.0 is the first release with a synchronized public Free edition at <https://github.com/hevnsnt/collider>.

---

## Version policy

- **Major** version (`1.x.0` -> `2.0.0`): breaking changes to the JLP wire format, the `config.yml` schema, or the CLI surface (flag rename, mode removed).
- **Minor** version (`1.4.x` -> `1.5.0`): new features, new CLI flags, new wire-format messages with backward compatibility, new GPU backends.
- **Patch** version (`1.4.0` -> `1.4.1`): bug fixes, internal refactors, performance work without API changes.

The wire-format protocol version (`protocol_version` in `protocol/jlp.yaml`) tracks independently of the binary version. Wire-format breaking changes bump the protocol version and the binary major version together.
