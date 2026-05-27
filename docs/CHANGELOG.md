# Changelog

All notable changes to theCollider are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This changelog covers both the Free and **(PRO VERSION ONLY)** editions. Pro-only changes are tagged inline.

---

## [1.5.0] - 2026-05-21: Theft-Resistance Architecture (Mainnet)

Pool architecture rewrite that closes the v1.4.x worker self-solve theft window. In v1.4.x a pool worker who found the cross-collision computed the puzzle's private key locally and could sweep the funds before the pool ever saw the solution. In v1.5 the algorithm itself denies any single worker the data needed to compute the key: each worker runs ONLY tame kangaroos OR ONLY wild kangaroos, the host-side collision detection is disabled in pool mode, and the pool server is the sole place where cross-type DPs aggregate. The server detects the collision, computes the key, broadcasts a hot-wallet sweep transaction, waits for cross-provider attestation that the sweep has propagated, and only then notifies workers that the puzzle is solved.

The full v1.5 security audit (see `collision-protocol/docs/v1.5-security-audit-report.md` in the pool server repo) cleared all five mainnet-blocking findings. Audit summary: 305 tests pass across both repos, 2 skipped, 0 failures, including the Wave 9 C1-specific end-to-end scenario.

### BREAKING

- **JLP protocol version bump to v3.** v1.4.x clients are refused at AUTH with reason `UPGRADE_REQUIRED`. There is no compatibility shim. The wire-layer reason is theft-resistance: a v1.4.x worker connecting to a v1.5 server would still receive a v3 `WORK_ASN` it cannot interpret, and a v1.4.x worker connecting to a v1.4.x server still computes keys locally. The clean break forces the network to upgrade together.
- **`report_solution()` removed from the client.** `JLPPoolClient::report_solution`, `PoolManager::report_solution`, the `recovered_keys/` JSON persistence, the SecureBuffer-staged private key in the retry uploader, and the `cb.on_solution` lambda in `pool_solver.cpp` are all deleted. The client never computes nor handles a private key in pool mode. SOLUTION is a server-to-client message only; the server's `_handle_solution` inbound path is removed.
- **Server-side: `sweep_service.MempoolClient.broadcast()` return type changed** (collision-protocol). Previously returned a bare `str` (the txid hex). v1.5 returns `tuple[str, str] = (txid_hex, broadcast_url)`. The second element identifies which provider accepted the broadcast so the cross-provider attestation step can route its propagation poll to a DIFFERENT provider (the audit C1 fix). Operators with a custom `MempoolClient` stub, a private mempool integration, or any other consumer of the broadcast return value MUST update their call sites to unpack the tuple. Code that does `txid = client.broadcast(raw_hex)` will now bind a tuple to `txid` and fail downstream; rewrite as `txid, broadcast_url = client.broadcast(raw_hex)`. This is a server-only change; no impact on the v1.5 worker client.

### New (PRO VERSION ONLY and Free, pool runtime)

- **Asymmetric tame/wild work assignment.** `RCGpuKang::KangarooMode { BOTH, TAME_ONLY, WILD_ONLY }` in the forked RCKangaroo. `BOTH` is rejected by `CudaRCKangarooBackend::initialize` in pool mode with an explicit "would re-enable v1.4.x theft-vulnerable local solve path" error. The host-side DP hashtable is disabled in type-only modes; DPs flow straight to the network. The `result.found = true` path is removed; the kangaroo loop exits only on external stop. Standalone (non-pool) mode keeps `BOTH` and is unchanged.
- **Wire schema v3 `WORK_ASN`.** Adds `kangaroo_type: u8`, `start_offset_a: u64`, `start_offset_b: u64`. The protocol IDL is regenerated to Python and C++ codecs from `protocol/jlp.yaml`. Standalone mode does not touch the new fields.
- **Single-strike permanent ban for type-mismatched DPs.** A v1.5 worker submitting a wild DP while assigned tame (or vice versa) is unambiguous binary modification, not honest-but-buggy behavior. The pool server bans the originating IP permanently on first occurrence, bypassing the rolling-window invalid-DP escalation ladder. The type check fires before bloom insertion and before the pending buffer, so a malicious DP cannot poison the collision search.

### Security

- **All five mainnet-blocking audit findings closed.** The full audit lives in `collision-protocol/docs/v1.5-security-audit-report.md`. Summary:
  - **C1 (BLOCKING)**: sweep propagation is now gated on cross-provider attestation. The pool server broadcasts the sweep tx via one provider, polls the OTHER provider with a hard timeout, and refuses to release the SOLUTION wire message until the second provider observes the tx. Blind 3-second sleep is gone.
  - **S1**: legacy P2PKH SIGHASH_ALL signing pinned against the Bitcoin Wiki canonical KAT.
  - **S2**: BIP-143 P2WPKH signing pinned against the BIP-143 spec KAT.
  - **A1**: strict integer parsing applied at both the Next.js admin route gate and the Python service layer.
  - **O1**: `SOLUTION.txt` writes now route through `ServerConfig.data_dir` instead of a hard-coded `./data/SOLUTION.txt`.
- **Sweep before SOLUTION ordering**. The pool server signs and broadcasts the sweep tx, confirms propagation via the other provider, AND THEN broadcasts SOLUTION. Workers receive a private key whose puzzle UTXOs are already moving (or already moved) and cannot win the race even if they extract the key from the SOLUTION payload.
- **Cross-provider attestation is symmetric and pinned by name.** The set-subtraction logic that picks the attest provider guarantees broadcast and attestation hit DIFFERENT providers regardless of which one served as primary. The regression tests `test_propagation_attests_via_FALLBACK_when_broadcast_used_PRIMARY` and `test_propagation_attests_via_PRIMARY_when_broadcast_used_FALLBACK` are pinned by name and must not be renamed without auditor sign-off.
- **Hot wallet hardening (collision-protocol).** argon2id KDF (m=64 MiB, t=3, p=4) + AES-256-GCM, 77-byte encrypted file at mode 0600. The sweep service refuses to broadcast SOLUTION if the wallet cannot decrypt at startup. Per-puzzle key rotation is the recommended deployment posture; see `collision-protocol/docs/HOT-WALLET.md`.

### Operator-facing

- **Migration guide**: `docs/MIGRATION-v1.5.md` documents the v1.4.x to v1.5 upgrade path. Workers must upgrade to v1.5 to keep mining. Pool operators must provision a hot wallet, configure two DISTINCT mempool API providers, and seed at least one Firebase Auth admin before the admin payout UI is functional.
- **Test coverage**: 304 tests pass across collider-pro plus collision-protocol, 2 skipped, 0 failures. New pinned tests cover cross-provider attestation by name (do not rename without auditor sign-off).
- **`docs/PRO.md`, `docs/POOL.md` updated** with the v1.5 worker experience: "you are assigned TAME or WILD on connect; you never see the puzzle's private key; payouts are operator-triggered via the admin UI to your registered Bitcoin address."

### Removed

- **`report_solution` plumbing in `JLPPoolClient` and `PoolManager`** including the retry uploader jthread, the SecureBuffer key staging, and the `recovered_keys/*.json` persistence path. No code path in the v1.5 client receives or computes a private key in pool mode.
- **Legacy `_handle_solution` inbound path in the pool server**. SOLUTION is now server-originated only.

---

## [1.4.4] - 2026-05-20

Cross-platform build + CI hardening. No runtime behavior changes; mainly unblocks the free Linux build and tightens the Pro / Free CI split.

### Fixed

- Linux CI now installs `libcurl` so the free build links cleanly. Added a CMake guardrail that fails configure with an actionable message when libcurl is missing on Linux.
- Cross-platform build fixes for Linux GCC and macOS clang (template instantiation differences and missing `<cstddef>` / `<algorithm>` includes that MSVC tolerated).
- Pro feature reachability: a Pro-only codepath that was unreachable from the free interactive menu but compiled into the binary is gated correctly now.

### Changed

- CI: gate Pro build steps on `pcfg.hpp` presence in the public repo (the file is in `PRO_PATHS` and excluded from free, so its absence is the canonical signal for "this is a free checkout").
- CI: macOS release binaries dropped from the release matrix; macOS users build from source via `./build_macos.sh free`.

---

## [1.4.3] - 2026-05-19

Pool mode reliability + edition-aware CI. Driven by a field-reported DP submission bug that surfaced once big-endian x coordinates were exercised against the new pool server.

### Fixed

- Pool DP submission: send the full big-endian x coordinate in CUDA DP submissions. Previous CUDA-side serialization truncated the high half on certain DP-bit configurations, causing the server to reject otherwise-valid DPs.
- macOS: `std::jthread` replaced with a portable `RetryThread` (libc++ on macOS arm64 was still missing `<stop_token>` at the toolchain version we targeted). `-DCOLLIDER_PRO` flag plumbing fixed in the macOS build path.

### Added

- Pool ban-detection on the client side: when the server signals a ban (rate-limit, invalid-DP escalation, or permanent ban), the client surfaces an explicit message rather than reconnecting in a tight loop.
- Separate CI/CD pipelines for the Free and Pro editions. Pro builds run in the private repo; the public free repo's CI is now fully edition-aware (skips macOS on plain `main`-branch pushes to conserve minutes; full matrix on tag push).

---

## [1.4.2] - 2026-05-17

A-tier stabilization release. Three waves of fixes across performance honesty, security posture, code quality, and build/sync hygiene. Driven by a six-reviewer adversarial validation pass on the v1.4.2 line. The headline correctness work is the full-pipeline benchmark, the dynamic per-GPU work balancer, the kangaroo `cudaMemcpy` hoisting, and the secret-handling cleanup (`SecureBuffer`, `secure_open_ofstream`, constant-time license compare).

### Added

- **`bench_gpu_pipeline.exe` plus `--benchmark` full-pipeline microbench** **(PRO VERSION ONLY)**. Drives the fused brainwallet kernel end to end (SHA-256 to secp256k1 to RIPEMD-160 to bloom) and reports per-stage throughput. Replaces the SHA-only benchmark that the v1.4.1 README advertised but did not implement. `docs/PRO.md` now ships a measured RTX 3060 table.
- **Dynamic per-GPU work balancer in the brainwallet runner** **(PRO VERSION ONLY)**. EMA throughput tracking (alpha=0.25), proportional split, eliminates the sync-stall that was idling the faster GPU at 26% utilization in mixed-GPU rigs. Measured 1.5x to 2x throughput lift on heterogeneous setups.
- **Kangaroo `cudaMemcpy` hoisting** **(PRO VERSION ONLY)**. Per-kangaroo save path collapsed from O(n^2) (~540 GiB transferred per save) to O(n) (~8.25 MiB per save call). Roughly 65,000x reduction in bytes moved per save. Closes the Q11 TODO in `src/gpu/kangaroo_kernel.cu` with a real profile-driven hoist.
- **`--pool-password-file <path>` CLI flag**. Reads the pool password from a file instead of argv to keep it out of `ps` and Task Manager. `--pool-password` now emits a deprecation warning; removal scheduled for v1.6.0. `read_password_file` rejects world-readable or group-readable files (POSIX 0600 check, Windows DACL Everyone/Authenticated Users/Users SID check).
- **`secure_open_ofstream` utility for all recovered-key files**. POSIX `0600` plus Windows owner-only DACL. Applied at 8 sites including the pool's `recovered_keys/*.json` (highest-stakes file in the system: it contains a plaintext private key).
- **`SecureBuffer<T>` and `SecureString` RAII types**. `OPENSSL_cleanse` on destruction. Applied to license HMAC key plus tag, pool password (`PasswordWipeGuard` in `authenticate()`), per-iteration brainwallet private keys (`PrivateKeyWipeGuard`), per-solve puzzle private keys, and BIP39 seeds.
- **`src/core/paths.hpp`**. Consolidates 17 HOME/USERPROFILE-resolution copies across 15 files into `collider_home()`, `collider_state_dir()`, `collider_config_dir()`.
- **`src/cli/flag_spec.hpp` flag registry**. Replaces the 328-line if/else chain in `cli_parser.cpp` with a 44-line dispatch loop. 87% collapse. New flags add ~3 lines instead of ~10.
- **`tests/test_kangaroo_dxz_fuzz.cu`**. Injects synthetic kangaroo states that drive the dx-computation to zero; asserts the SIMT guard skips the step without ECC errors. Closes T5.
- **`tests/test_kangaroo_work_file::dp_round_trip`**. Writes a work-file with N synthetic DPs, reloads, asserts byte-equivalent DP slice. Closes T6.
- **`tests/test_bloom_fp_rate.cpp`**. Builds the loose GPU bloom at 1e-5 and the tight CPU bloom at 1e-7, queries 10M random non-member 20-byte hashes, asserts the empirical FP rate is within 2x of target. Closes T7.
- **`tests/test_jlp_pool_protocol::dp_sequence_anti_replay`**. Verifies the v1.4.1 B.1 DP v2 4-byte sequence nonce actually causes stale-sequence DPs to be rejected on the server side. Closes T8.
- **`tests/test_jlp_pool_dp_bits_validation.cpp`**. Server `WORK_ASN` with `dp_bits` outside `[8, 32]` must be rejected and the connection dropped. Closes S1.
- **`tests/test_work_balancer.cpp`**. Validates the EMA-driven proportional split converges under representative mixed-GPU throughput patterns.

### Changed

- **HMAC license verification uses `CRYPTO_memcmp`** for constant-time tag compare. Was `memcmp`; closed the timing side-channel. The HMAC hex output buffer is now wiped on free (was returning an unwiped `std::string`).
- **TUI redesign for the brainwallet runner** **(PRO VERSION ONLY)**. Carbon-fiber palette (phosphor green plus amber). Animated COLLIDER block-letter boot banner with a ~1.1s shine-wipe. Compact one-row-per-GPU layout, single sparkline, big-label metrics. Static column widths everywhere in the GPU detail row, no more layout shift as values change width. Opaque modal backdrops; help, wordlist picker, and bloom picker no longer bleed through.
- **TUI rolling-mean smoothing**. GPU util, power, and PCIe Gen now use a 60s rolling mean (was per-render NVML reads, flickered). THROUGHPUT line uses a 30s rolling mean (was per-batch instantaneous, jumped 2x to 5x between rule-engine and crypto batches).
- **Brainwallet runner extracted 14 helper functions**, ~942 lines moved out of the god-function. Inner scan loop, GPU-rules dispatch, and hit handler still resist clean extraction (15+ shared mutable locals); deferred to the v1.5.0 crypto pipeline rewrite. Net main body: 3635 lines to 2693 lines.
- **`puzzle_solver.cpp` decomposed**. Orchestrator dropped from 2509 to 866 lines plus 4 new helper translation units.
- **GPU rules dispatch gate** folded to a single `gpu_rules_active` flag (was three coupled booleans).
- **Detached `std::thread` calls converted to `std::jthread` with `std::stop_token`**. Closes Q6 at the `jlp_pool_client` retry uploader; also applied to `balance.cpp`, `pipeline.cu`, and 2 TUI panels.
- **`const_cast<std::mutex&>` removed in `jlp_pool_client`**; the offending members are now `mutable`.
- **`report_solution` jthread closure**: the 32-byte private key passed through the closure is now a `SecureBuffer<uint8_t>` rather than `std::array`, so it is wiped on lambda exit.
- **CMake CUDA SM defaults aligned to `75;86;89;120`** (Turing through Blackwell desktop). Local default was missing Turing; closes B2.
- **`build_pro.bat` and `build_free.bat` produce truly different binaries.** Two CMake configures (`build_free/` with `-DCOLLIDER_PRO=OFF`, `build_pro/` with `-DCOLLIDER_PRO=ON`). The v1.4.1 `copy /Y build\collider.exe build\collider_pro.exe` step is gone. The Free binary no longer ships the entire Pro feature set. Closes L4.
- **`scripts/sync-to-free.sh` PRO_PATHS extended** with 13 missing paths: `license_gate`, `brainwallet_setup`, `runtime/scan_state`, `runtime/runtime_control`, `runtime/perf_instrumentation`, `runtime/empty_hit_writer`, `runtime/runtime_config_yaml`, `plugins/`, `ui/tui/`, `gpu/v2/`, `platform/nvml_query`, `platform/gpu_telemetry_*`. Closes B3.
- **`test_cli_parser` now links the real parser**. The in-test `parse_args_for_test` mirror is gone; the test now exercises the production symbol via the `src/cli/flag_spec.hpp` registry. Closes T2.
- **`test_hash_vectors` rewired to test production crypto**. The in-test `namespace cpu_ref { sha256, ripemd160 }` is gone; the test now includes `crypto_cpu.hpp` and asserts NIST KATs against `collider::cpu::sha256` and `collider::cpu::ripemd160`. Closes T1.
- **`test_warpwallet_kat` renamed to `test_warpwallet_properties`** (later restored). The test was originally property-based (determinism, Hamming distance, non-zero output) and never carried real Keybase KAT vectors. The intermediate rename stopped lying about coverage. Closes T4. **Restored to `test_warpwallet_kat` in v1.4.2 T3.2 (A-tier repair wave 1)** with three pinned Keybase reference vectors from the upstream `spec.json` (https://github.com/keybase/warpwallet) alongside the existing property checks. The CMake target renamed `WarpWalletProperties` -> `WarpWalletKAT` accordingly.
- **`CLAUDE.md`, `README.md`, `docs/PRO.md`, `docs/ARCHITECTURE.md`**: dropped the "Ed25519 signature verification" claim. Replaced with an honest "HMAC-SHA256 verification with a private shared key compiled into the binary." Also dropped the "256-byte patchable license slot" claim, which never matched the actual implementation (libcurl HTTP activation plus 24h HMAC'd local cache). Closes M1.
- **`docs/PRO.md`**: dropped the "hundreds of millions of passphrase checks/s" unbenched claim. Replaced with a measured RTX 3060 table plus a softened "tens to hundreds of millions on Ampere through Blackwell" qualifier. Closes M2 and M3.
- **`docs/PRO.md`**: dropped the "no phone-home, offline-verifiable" claim, which contradicted `license_check.cpp`. Replaced with an honest "first activation POSTs to the issuer over TLS, then a 24h HMAC'd local cache. No per-run check inside the cache window."

### Fixed

- **`secure_buffer.hpp` regression** where a `#if 0` block had silently disabled `OPENSSL_cleanse` and shipped only the volatile-store fallback. The cleanse is now unconditionally compiled in.
- **`secure_open_ofstream` Windows silent-fallback closed**. The function now emits a loud warning if `build_owner_only_sa` fails (previous behavior was a silent fallback to the inherited DACL, which defeated the whole point).
- **Pool wire `dp_bits` validation** (`8 <= dp_bits <= 32`). Rejects buggy or malicious server `WORK_ASN` messages that would otherwise burn GPU cycles indefinitely (a `dp_bits=255` `WORK_ASN` previously caused the kangaroo solver to never emit a DP). Closes S1.
- **Multi-puzzle handling**: `--all-unsolved` no longer prematurely terminates after the first puzzle on the RCKangaroo, MultiGPU Kangaroo, and GPU brute paths. All three returned 0 instead of continuing the worklist.
- **Brute-mode `brute_lengths_csv` written at all 4 save sites**. Was only the explicit `'s'`-key save site, so paused, periodic, and final saves silently dropped the field, and resume refused to load the resulting work-file.
- **Bloom FP-rate empirical pin**. The new `test_bloom_fp_rate` test catches drift in the bloom hash distribution or sizing math that would silently inflate empty-hit accounting.

### Security

- **Constant-time license HMAC compare** via `CRYPTO_memcmp`. Closes S2.
- **`SecureBuffer<T>` plus `SecureString`** applied to license key material, pool password, recovered private-key buffers, per-iteration brainwallet private keys, and BIP39 seeds. Heap residue of secrets after free is now wiped. Closes S6.
- **`secure_open_ofstream` for recovered-key files**. POSIX `0600` plus Windows owner-only DACL at 8 sites including `recovered_keys/*.json`. Closes S5.
- **`--pool-password-file <path>`** removes the pool password from argv. Closes S4.
- **`dp_bits` server-input validation** stops a DoS vector where a malicious or buggy server burns GPU cycles forever. Closes S1.

### Removed

- **`--addr-types` flag removed end to end**. The v1.4.0 introduction shipped as a half-wired placeholder that printed "not wired through the orchestrator yet (Phase 4 follow-up). Use the legacy brain-wallet pipeline for now." Closes Q10.
- **Four dead `.cpp` files deleted**: `src/generators/priority_queue.cpp`, `src/generators/passphrase_generator.cpp`, `src/runtime/scan_state.cpp`, `src/runtime/runtime_control.cpp`. All were empty stubs or single-singleton-accessor essays. Header-only contracts stand. Closes Q4.
- **Three `#if 0` "reserved for future" blocks deleted**: `src/gpu/v2/weak_prng_kernel.cu:96-143`, three commented `__constant__` arrays in `src/gpu/secp256k1.cu:36-93`, and the stale GLV-stub comment at `src/gpu/fused_pipeline.cu:750`. Git remembers; v1.5.0 plan covers the actual replacement. Closes Q5.
- **File-header design essays moved out of source**. 60-line headers in 5 files (`src/runtime/empty_hit_writer.hpp`, `src/runtime/scan_state.hpp`, `src/runtime/perf_instrumentation.hpp`, `src/gpu/secp256k1_field.cuh`, `src/pool/jlp_pool_client.cpp`) replaced with 5-line summaries plus a pointer to `docs/internals/`. Closes Q12.
- **700+ phase, wave, and audit-ID metadata stamps stripped from production source.** Things like `v1.4.x`, `phase N`, `wave N`, `Repair R\d+`, `task #\d+`, `Tier C`, `Pool-F5`, `R-B10`, `D-H1`, `Track-f F-03`. Net dominant AI-tell dropped from the public-facing source. Closes Q2 (first pass; see Known limitations).

### Known limitations

- The brainwallet runner main body is still 2693 lines (down from 3635). The inner scan loop, GPU-rules dispatch, and hit handler have 15+ shared mutable locals and resist clean extraction without a data-flow rework. Further decomposition is scheduled alongside the v1.5.0 crypto pipeline rewrite.
- Roughly 350 inline metadata stamps remain in production source, concentrated in `src/gpu/mega_fused_kernel.cu` and `src/gpu/kangaroo_kernel.cu`. A second-pass strip is in flight; the kernel files were left for last because their phase tags are intermixed with kernel-region markers that need to be replaced with proper section headers rather than just deleted.

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
