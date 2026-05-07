# Changelog

All notable changes to theCollider are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.0-dev] - 2026-05-04 (full-opto branch, post-adversarial-review)

A multi-track adversarial review on 2026-05-04 found 87 issues (13 Critical, 22 High, 33 Medium, 19 Low) across the GPU brain wallet pipeline, JLP pool client, CLI/config layer, and concurrency primitives. Headline finding: the GPU brain wallet pipeline byte-swapped the SHA256 output when packing it into the uint256 scalar, producing hash160 values that could not match real funded Bitcoin addresses. Latent since the kernel was written. Full report at `docs/review-2026-05-04/00-SUMMARY.md`.

This release consolidates fixes for every Critical and most High findings.

### Added

- GPU correctness test suite: `tests/test_secp256k1_inv.cu`, `tests/test_ec_table_consistency.cu`, `tests/test_gpu_hash160.cu`, `tests/test_cli_parser.cpp`. CUDA tests gated on `COLLIDER_BACKEND=CUDA`; skip cleanly with code 77 on hosts without a GPU. See `docs/CRYPTO-VALIDATION.md`.
- Test wrappers `secp256k1_test_inverse_correctness` and `secp256k1_test_table_on_curve` in `src/gpu/secp256k1.cu` exposing `__device__` primitives to host validation.
- `fused_validate_scalar()` in `src/gpu/fused_pipeline.cu` rejects scalar 0 or scalar ≥ n before EC multiply.
- TLS hostname verification in JLP pool client (SNI, `X509_VERIFY_PARAM_set1_host` with `NO_PARTIAL_WILDCARDS`, default trust store).
- AUTH state machine in JLP pool client (rejects `WORK_ASN`, `SOLUTION`, `DP_ACK`, `STATS_RSP` before AUTH_OK).
- Bounded reconnect on AUTH_FAIL (3 attempts, jittered exponential backoff).
- `ssl_io_mutex_` serializing `SSL_write` / `SSL_read`.
- CLI parser unit tests (67 rows / 101 EXPECT assertions).
- `validate_mode_mutex()` in `parse_args` rejects `--brainwallet --pool` and similar combinations.
- `COLLIDER_PRO` build option (Wave 6, see `docs/PRO-MIGRATION.md`). Default ON; flips to OFF when `pro/` submodule lands.
- Per-batch pinned host buffers in `MultiGPUBrainWallet::process_batch` (eliminates F-01 aliasing race).
- `mega_compute_grid_blocks` / `fused_compute_grid_blocks` helpers (computed in `unsigned long long`, refuse oversized launches).
- `MEGA_FUSED_MAX_ITEMS_PER_LAUNCH = UINT32_MAX` cap.
- `__launch_bounds__(256, 4)` on top-level kernels.
- `mega_compress_pubkey()` helper.
- Validation scripts: `scripts/wave-0-windows-validate.bat` (pre-fix expected-failure runner) and `scripts/wave-1-windows-validate.bat` (post-fix all-pass runner).

### Changed (correctness fixes)

- SHA256 → uint256 scalar conversion writes into `limbs[7-i]` instead of `limbs[i]` (5 sites). GPU pubkeys now match canonical Bitcoin convention.
- `secp256k1.cu`'s `mod_inv` replaced from a hand-written addition chain (wrong exponent) to right-to-left binary exponentiation using literal p−2.
- `fused_pipeline.cu`'s `sha256_short` rewritten to handle messages of any length (was silently truncating past 55 bytes).
- `MegaWorkQueue::head/tail/completed` promoted from `uint32_t` to `unsigned long long`. Persistent kernel uses 64-bit `atomicAdd`.
- `mega_get_progress` returns `uint64_t`.
- `mega_sha256_registers` length encoding uses full 64-bit FIPS 180-4 (`W[14]:W[15]`).
- `mega_shared_kernel`'s `__shared__ s_words` sized to `MEGA_MAX_PASSPHRASE` (was 64).
- `sha256_33` calls `mega_sha256_block` (single SHA256 implementation; eliminates SM_90 nvlink overflow).
- `streaming_brain_wallet.hpp` `rule_engine_` is `std::shared_ptr<HashcatRuleEngine>` swapped under mutex.
- `Logger::log_file_` is `shared_ptr<std::ofstream>`, snapshotted under mutex.
- `signal_handler` body is two atomic stores; logging deferred to main-thread context.
- `localtime_r`/`localtime_s` instead of `std::localtime` (thread-safety).
- CLIFlags refactored to explicit `*_set` bits set during parse (was inferred-from-args).
- `--puzzle-min-bits`, `--puzzle-max-bits`, `--save-interval` honor CLI values (were unconditionally clobbered by config).
- `PoolStats.pool_speed` type changed `double` → `uint64_t` to match server's `<Q` packing.
- Pool factory hard-rejects `http://` URLs with a migration message.

### Removed

- `src/pool/http_pool_client.{cpp,hpp}` deleted (silently leaked credentials when configured `https://`; D-C1).
- `secp256k1.cu`'s `glv_decompose`, `ec_mul_glv`, `ec_add_glv_affine` deleted (decomposition was wrong; silent landmine for future "30% GLV speedup" attempts; to re-enable, port libsecp256k1's `secp256k1_scalar_split_lambda`).
- Pre-existing wrong expected value for "Puzzle 2 pubkey" hash160 in `tests/test_hash_vectors.cpp` (was `91b24bf...`, correct is `06afd46b...`; had been masking real test signal).

### Fixed (Critical and High findings)

C-CRIT-1, C-CRIT-2, C-CRIT-3, C-CRIT-4, A-CRIT-1, A-CRIT-2, A-HIGH-1, A-HIGH-2, A-HIGH-3, A-HIGH-5, F-01, F-02, F-03, F-04, F-08, F-11, F-12, F-17, D-C1, D-H1, D-H4, D-H5, D-M2, D-M5, B-LOW-6, B-MED-1, E-CRIT-1, E-HIGH-1, E-HIGH-2, M1, M2, M3, M5.

See `docs/review-2026-05-04/00-SUMMARY.md` for finding descriptions.

### Known leftover (deferred follow-ups)

- F-05/F-06: `check_balance_async` thread detachment / unbounded thread spawn.
- F-07: `gpu_rule_engines.emplace_back` realloc invalidation in main.cpp.
- D-M1: receiver-thread reconnect self-join hazard (partially mitigated; full fix needs supervisor).
- M4: silent-skip telemetry in MegaFusedResult.
- Pro/Free source split (Wave 6 partial). Build flag landed; sources still in public repo. See `docs/PRO-MIGRATION.md`.
- Historical brain wallet code in git log of public repo. Not rewriting history.

### Required user actions

1. **Restore GitHub Actions billing.** Every track of the review cited absent CI as a contributor.
2. **Validate on Windows CUDA.** Run `scripts\wave-1-windows-validate.bat` from `x64 Native Tools Command Prompt for VS 2022`. Expect: 8/8 tests pass.
3. **Create `hevnsnt/collider-pro` private repo** to complete Wave 6. See `docs/PRO-MIGRATION.md`.

---

## [1.2.0-dev] - 2026-01-08 (full-opto branch)

### MEGA FUSED KERNEL - Ultimate Performance Optimization

This experimental branch implements the ultimate optimization: a single mega-fused kernel that processes the entire brainwallet pipeline in registers with NO intermediate global memory writes.

#### New Architecture

**Single Kernel Pipeline:**

```
word + rule → passphrase (registers) → SHA256 → EC_MUL → SHA256 → RIPEMD160 → Bloom
```

**Memory Savings:**

- Eliminates intermediate passphrase buffer (~934 MB per batch for 3.65M passphrases)
- All data flows through registers between stages
- Estimated 2-5x memory bandwidth improvement

#### Optimizations Implemented

**1. Lookup Table Rule Engine (eliminates switch divergence)**

- 256-entry tolower/toupper tables in constant memory
- Branchless character transformations
- `#pragma unroll` on all character loops
- Full hashcat rule coverage (l, u, c, C, t, r, d, f, [, ], {, }, $, ^, T, D, ', @, s, i, o)
- Estimated 3-5x improvement over switch-based engine

**2. Embedded Cryptographic Operations**

- Complete SHA256 implementation (variable length + 33-byte pubkey variants)
- Complete RIPEMD160 implementation (fully unrolled 80 rounds)
- Complete secp256k1 EC operations (field arithmetic + point operations)
- Windowed EC scalar multiplication (5-bit windows, 52 lookups)
- Optimized modular inverse (Fermat's theorem with addition chain)

**3. Persistent Kernel with Work Stealing**

- Eliminates kernel launch overhead for continuous processing
- Warp-cooperative work distribution
- Atomic work queue with configurable chunk size
- Better SM utilization through load balancing
- Progress tracking for real-time monitoring

**4. Shared Memory Optimized Variant**

- Caches 4 words per block in shared memory
- Cooperative loading by first 4 threads
- Reduces global memory pressure for repeated word access
- Beneficial when same word processed with many rules

**5. Warp-Level Optimizations**

- `__shfl_sync` for warp-uniform data distribution
- All 32 lanes process different items simultaneously
- Minimized warp divergence through branchless operations

#### New Files

- `src/gpu/mega_fused_kernel.cu` - Complete mega kernel implementation (~1800 lines)
- `src/gpu/mega_fused_kernel.hpp` - Public API header

#### API

Three kernel variants available:

1. **Standard Mega Kernel** - `mega_fused_brainwallet_batch()`
   - Best for large batches with moderate rule counts

2. **Shared Memory Kernel** - `mega_fused_shared_batch()`
   - Best when same words processed with many rules

3. **Persistent Kernel** - `mega_fused_persistent_launch()`
   - Best for continuous streaming workloads
   - Lowest latency for incremental results

#### Expected Performance

Theoretical improvements (requires benchmarking on CUDA hardware):

- Memory bandwidth: 2-5x (eliminated intermediate buffer)
- Rule engine: 3-5x (lookup tables vs switch)
- Kernel launch: 10-100x (persistent kernel)
- Overall: 5-10x improvement expected over v1.1.1

#### Known Limitations

- Requires CUDA hardware (not available on Metal backend)
- Maximum passphrase length: 128 bytes
- Maximum rule length: depends on rule complexity

---

## [1.1.1] - 2026-01-08

### Performance - Brain Wallet Scanner

#### Benchmark Results (Dual RTX 4090)

| Mode                        | Speed   | Notes                                  |
| --------------------------- | ------- | -------------------------------------- |
| Direct benchmark (no rules) | ~1.8B/s | SHA256→EC→SHA256→RIPEMD160→Bloom       |
| With GPU rules (73 rules)   | ~5.2M/s | Rule application is current bottleneck |

**Note:** The rule application kernel has significant thread divergence due to switch statements. Future optimization should target the rule engine.

#### Optimizations Implemented

**1. Windowed EC Scalar Multiplication (~5x faster)**

- Replaced naive 256-iteration double-and-add with 5-bit windowed method
- Precomputed tables: 52 windows × 32 points per window
- Per-GPU table allocation for multi-GPU safety
- `cudaMemcpyToSymbol` for device-local table pointers

**2. SHA256 Optimization (~20-30% faster)**

- Circular buffer message schedule: `W[16]` instead of `W[64]`
- First 16 rounds fully unrolled with `SHA256_ROUND` macro
- Rounds 16-63 compute W on-the-fly: `W[i & 15]`
- Eliminated intermediate arrays, direct register usage

**3. RIPEMD160 Optimization (~30-50% faster)**

- Fully unrolled 80 rounds (160 macro invocations for left+right paths)
- Branchless round functions: `ripemd_f0` through `ripemd_f4`
- Direct loading into X[] array (no intermediate `block[64]`)
- Round-specific `RIPEMD_ROUND_L`/`RIPEMD_ROUND_R` macros

#### Bug Fixes

**Startup "invalid argument" Error**

- Root cause: Pending CUDA errors from async operations incorrectly attributed to kernel launch
- Fix: Added `cudaGetLastError()` to clear pending errors before kernel launches
- Added safety checks for 0-block kernel launches in `gpu_apply_rules_cross_product()` and `gpu_apply_single_rule()`

### Changed

- `src/gpu/secp256k1.cu` - Per-GPU precomputed table management
- `src/gpu/fused_pipeline.cu` - Optimized SHA256 and RIPEMD160 implementations
- `src/gpu/gpu_rules.cu` - Safety checks and error clearing

### Technical Notes

- EC multiplication is now ~30% of pipeline time (was ~70%)
- New bottleneck is rule application for high rule counts
- Future optimization targets: GLV endomorphism, rule kernel optimization

---

## [1.1.0] - 2025-01-06

### Added

#### PCFG (Probabilistic Context-Free Grammar) Training

- Full implementation of password pattern learning from wordlists
- Extracts structure patterns (L=lowercase, U=uppercase, D=digit, S=symbol)
- Calculates probability weights for each pattern
- Generates candidates in probability order (most likely first)
- Saves trained models to `.pcfg` files for reuse

#### WarpWallet/Scrypt Support

- Complete scrypt implementation with Salsa20/8 core and BlockMix
- HMAC-SHA256 and PBKDF2-SHA256 for secondary derivation
- Standard WarpWallet key derivation: `s1 XOR s2`
- Email-as-salt support for WarpWallet format
- `WarpWalletProcessor` class for batch processing

#### Markov Chain Generator

- Character-level Markov chain for password generation
- `TransitionMatrix` class for storing transition probabilities
- `Trainer` class for learning from password corpus
- `Generator` class for probability-ordered enumeration
- `MarkovSource` PassphraseSource implementation
- Save/load functionality for trained models

#### Performance Optimizations

- **Parallel Bloom Filter Loading**: Multi-GPU simultaneous copy
  - ~N-1x speedup for N GPUs
  - Progress reporting during load
- **True Double Buffering**: Overlap CPU/GPU work
  - CPU prepares batch N+1 while GPU processes batch N
  - Up to 2x throughput improvement

#### New Files

- `src/core/warpwallet.hpp` - WarpWallet/Scrypt implementation
- `src/generators/markov.hpp` - Markov chain generator
- `docs/CHANGELOG.md` - This changelog

### Fixed

#### Critical Bug Fixes

- **Mode Selection Bug**: Brainwallet mode incorrectly activated pool mode when pool config existed
  - Root cause: `pool_mode` flag inherited from config and never reset
  - Fix: Explicitly set `pool_mode = false` when brainwallet selected
- **MSVC Compilation Error (C2598)**: `extern "C"` linkage inside function body
  - Root cause: C++ requires linkage specifications at global scope
  - Fix: Moved `extern "C"` declarations to file scope

- **Tames Generation**: Kangaroo tames generation returned `false`
  - Root cause: Stub implementation
  - Fix: Full implementation with proper jump table initialization

#### Warnings Fixed

- Removed unused `MAX_WORD_LEN` and `MAX_RULE_LEN` constants (warning #177-D)
- Removed dead L2 cache attribute code (warning #550-D)
- Fixed unsigned char comparison in puzzle_config.hpp

### Changed

- `src/ui/brainwallet_setup.hpp` - Integrated PCFG training UI
- `src/gpu/rckangaroo_wrapper.cu` - Full tames generation implementation
- `src/gpu/brain_wallet_gpu.cpp` - Parallel loading, double buffering, extern "C" fix
- `src/gpu/h160_bloom_filter.cu` - Added `h160_bloom_set_config` function
- `src/generators/pcfg.hpp` - Fixed C++ default member initializer issue

### Build Compatibility

- Verified on Windows 10/11 with Visual Studio 2022 + CUDA 12.9
- Verified on macOS (Apple Silicon) with Metal backend
- Verified on Linux with GCC 11+ and CUDA 12.x

---

## [1.0.0] - 2025-01-04

### Added

- Initial release of theCollider
- RCKangaroo integration (K=1.15 optimal)
- Fused GPU brain wallet pipeline
- Bloom filter checking against ~50M funded addresses
- Hashcat-compatible rule engine (35+ operations)
- JLP pool protocol client
- Interactive mode with setup wizard
- Multi-GPU support with automatic detection
- YAML configuration file support
- macOS Metal backend support

### Core Features

- **Kangaroo Solver**: Pollard's Kangaroo for ECDLP with SOTA method
- **Brain Wallet Scanner**: SHA256 → secp256k1 → SHA256 → RIPEMD160 → Bloom
- **Pool Mode**: Distributed solving via Collision Protocol
- **Opportunistic Scanning**: Check DPs against bloom filter during Kangaroo solving

### Performance

- RTX 4090: 8 GKeys/s (Kangaroo), 1.8B keys/s (Brain Wallet)
- RTX 3090: 4 GKeys/s (Kangaroo), 1.0B keys/s (Brain Wallet)
- Multi-GPU scaling with near-linear performance

---

## Version History Summary

| Version | Date       | Highlights                                           |
| ------- | ---------- | ---------------------------------------------------- |
| 1.1.1   | 2026-01-08 | EC/SHA256/RIPEMD160 optimizations, startup error fix |
| 1.1.0   | 2025-01-06 | PCFG, WarpWallet, Markov chains, parallel loading    |
| 1.0.0   | 2025-01-04 | Initial release                                      |

---

## Upcoming

### Planned for 1.2.0

- [ ] Collision Protocol web dashboard
- [ ] Worker registration portal
- [ ] AMD GPU support via HIP/ROCm
- [ ] CUDA Graphs optimization
- [ ] Neural network passphrase prediction
