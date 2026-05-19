# theCollider Architecture

A high-level map of the codebase: layered view, CMake targets, source tree, backend selection, free/Pro split, pool-client lifecycle, testing layout, and dependencies. Intended for new contributors and third-party readers who want the lay of the land without grepping the tree.

This document describes the **free** edition. **(PRO VERSION ONLY)** modules (brain-wallet pipeline, license verification, scrapers, the v2 puzzle-mode kernel) live in this private dev tree but are excluded from the public Free repo by `scripts/sync-to-free.sh`. They are referenced here only at a high level.

---

## Table of contents

- [Layered view](#layered-view)
- [CMake library targets](#cmake-library-targets)
- [Source tree, top-down](#source-tree-top-down)
- [Backend selection](#backend-selection)
- [Free / Pro split](#free--pro-split)
- [Pool client lifecycle](#pool-client-lifecycle)
- [Testing layout](#testing-layout)
- [Dependency summary](#dependency-summary)
- [Where to go next](#where-to-go-next)

---

## Layered view

```
+-----------------------------------------------------------+
|                     CLI / runtime                         |
|   src/cli/        src/runtime/        src/main.cpp        |
+-----------------------------------------------------------+
|                     Solver layer                          |
|   src/core/       src/generators/(*)  src/rules/(*)       |
+-----------------------------------------------------------+
|                     GPU compute                           |
|   src/gpu/        third_party/RCKangaroo/                 |
+-----------------------------------------------------------+
|                     Platform HAL                          |
|   src/platform/   (CUDA / Metal / CPU backends)           |
+-----------------------------------------------------------+
|                     Pool networking                       |
|   src/pool/       (JLP wire + HTTP fallback + TLS)        |
+-----------------------------------------------------------+

(*) Pro-only.
```

The build produces one of two executables from this tree:

- `collider` (free): puzzle solver (kangaroo + brute force) plus JLP pool client plus benchmark.
- `collider_pro` **(PRO VERSION ONLY)**: same plus brain-wallet pipeline plus license-gated features.

Both are linked from the same CMake project, gated by `-DCOLLIDER_PRO=ON|OFF`.

---

## CMake library targets

The build is split into focused static libraries so that platforms missing a backend (e.g. macOS without CUDA) can drop the relevant target without #ifdef sprawl.

| Target              | Purpose                                                                                                                                                                      |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `collider_platform` | Platform abstraction over CUDA / Metal / CPU. Backend selection at configure time.                                                                                           |
| `collider_core`     | CPU-side solver logic: rule engine, priority queue, host orchestration.                                                                                                      |
| `collider_gpu`      | GPU kernels: SHA-256, secp256k1, RIPEMD-160, bloom filter, kangaroo, puzzle solver.                                                                                          |
| `collider_pool`     | JLP wire client, HTTP fallback, pool manager. Links OpenSSL for TLS.                                                                                                         |
| `collider_license`  | **(PRO VERSION ONLY)** License verification: HMAC-SHA256 cache validation, with remote re-validation against the issuer endpoint when the local cache is missing or expired. |
| `rckangaroo`        | Third-party Kangaroo solver (GPLv3, in `third_party/RCKangaroo/`).                                                                                                           |

Compile definitions follow the platform: `COLLIDER_USE_CUDA=1`, `COLLIDER_USE_METAL=1`, or `COLLIDER_USE_CPU=1`, plus exactly one of `COLLIDER_PLATFORM_WINDOWS`, `COLLIDER_PLATFORM_LINUX`, `COLLIDER_PLATFORM_MACOS`. `COLLIDER_PRO=1` is set for Pro builds.

---

## Source tree, top-down

### `src/main.cpp` and `src/cli/`

Entry point and command-line parser. The parser (`cli_parser.cpp`) is the single source of truth for which flags exist; every doc that mentions a flag is checked against it. The parser itself is intentionally small (one `for` loop over `argv`) and pure: parsing produces an `Arguments` struct plus a `CLIFlags` bitmask of "this flag was explicitly set" markers.

`validate_mode_mutex()` enforces that the user picks exactly one of `--brainwallet`, `--pool`, `--puzzle [N] [--kangaroo]`.

### `src/runtime/`

Mode dispatchers split out from `main.cpp` during the v1.4.1 A.3 refactor. One file per mode, plus shared globals.

| File                    | Role                                                                           |
| ----------------------- | ------------------------------------------------------------------------------ |
| `puzzle_solver.cpp`     | Standalone puzzle (kangaroo or brute force, with v1.4.1 graceful demote).      |
| `pool_solver.cpp`       | Pool client driver: connect, AUTH, request work, submit DPs, handle solutions. |
| `brain_wallet_runner.*` | **(PRO VERSION ONLY)** Brain-wallet pipeline driver.                           |
| `gpu_detection.cpp`     | Enumerate visible GPUs, query memory, assign device indices.                   |
| `license_gate.cpp`      | **(PRO VERSION ONLY)** Verify HMAC-SHA256 license signature at startup.        |
| `runtime_globals.hpp`   | Globals shared across runtime modules (signal handlers, logger).               |

### `src/core/`

CPU-side solver logic that does not depend on a specific GPU backend.

| File                | Role                                                                               |
| ------------------- | ---------------------------------------------------------------------------------- |
| `yaml_config.hpp`   | `AppConfig` plus `apply_config_to_args()`. The config schema is documented inline. |
| `edition.hpp`       | Edition string macros for the banner.                                              |
| `rule_engine.*`     | Hashcat-style rule application (CPU side).                                         |
| `priority_queue.*`  | Probability-ordered passphrase queue (used by Pro generators).                     |
| `puzzle_database.*` | Bundled puzzle metadata (addresses, ranges, pubkeys for revealed-pubkey puzzles).  |

### `src/gpu/`

All GPU kernels and their host-side dispatchers. The directory is "fat" by design: one place to look when chasing a kernel-side bug.

| File                              | Role                                                                                         |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| `secp256k1.cu` / `.metal`         | Field arithmetic, EC point ops, scalar multiply, modular inverse.                            |
| `sha256.cu` / `sha256.metal`      | SHA-256 round logic, unified across kernels in v1.4.1 D.2.                                   |
| `ripemd160.cu` / `.metal`         | RIPEMD-160 round logic, unified across kernels in v1.4.1 D.2.                                |
| `kangaroo.cu` / `kangaroo.metal`  | Pollard's Kangaroo kernel (pool client and Pro standalone).                                  |
| `kangaroo_metal.mm`               | macOS Metal dispatcher for the kangaroo kernel.                                              |
| `puzzle.metal`, `puzzle_metal.mm` | macOS Metal standalone puzzle kernel (v1.4.1 D.3, brute-force pipeline).                     |
| `mega_fused_kernel.cu`            | **(PRO VERSION ONLY)** Fused brain-wallet kernel (SHA-256 + secp256k1 + RIPEMD-160 + bloom). |
| `fused_pipeline.cu`               | **(PRO VERSION ONLY)** Older fused brain-wallet pipeline kept for compatibility.             |
| `h160_bloom_filter.cu`            | **(PRO VERSION ONLY)** GPU bloom filter against funded address H160s.                        |
| `gpu_rule_kernel.cu`              | **(PRO VERSION ONLY)** GPU rule expansion.                                                   |
| `metal_multi_gpu_puzzle.mm`       | macOS multi-GPU adapter for the puzzle Metal pipeline.                                       |

The Metal shaders are embedded in the binary at build time via `cmake/kangaroo_metal_source.h.in` and `cmake/puzzle_metal_source.h.in`. There is no filesystem dependency at runtime; a deployed binary cannot be missing or out of sync with its shaders.

### `src/pool/`

JLP wire-protocol client and pool manager. See [JLP-PROTOCOL.md](JLP-PROTOCOL.md) for the wire format.

| File                        | Role                                                                              |
| --------------------------- | --------------------------------------------------------------------------------- |
| `jlp_pool_client.{hpp,cpp}` | Reference C++ client for the JLP wire protocol.                                   |
| `jlp_wire_generated.hpp`    | C++ packed structs generated from `protocol/jlp.yaml`. Do not hand-edit.          |
| `pool_manager.{hpp,cpp}`    | Reconnect logic, stats aggregation, work-queue management.                        |
| `http_pool_client.*`        | HTTP fallback (kept for non-JLP integrations).                                    |
| `pool_tls.*`                | OpenSSL TLS init plus hostname verification (`X509_VERIFY_PARAM_set1_host`, SNI). |

### `src/platform/`

Backend selection. Each backend implements the same `IPlatform` interface but only one is linked per build.

| File                      | Role                                         |
| ------------------------- | -------------------------------------------- |
| `platform_cuda.{hpp,cpp}` | CUDA backend (Linux, Windows).               |
| `platform_metal.{hpp,mm}` | Metal backend (macOS).                       |
| `platform_cpu.{hpp,cpp}`  | CPU fallback for hosts with neither toolkit. |

### `src/ui/`

Interactive terminal UI: banner, ROI table, brain-wallet setup wizard. No business logic; pure presentation.

### `src/license/` **(PRO VERSION ONLY)**

License validation. On first run with `--activate KEY`, the binary POSTs the key to the issuer's license endpoint over TLS and, on success, persists the result in `~/.collider/license.cache` as a JSON blob authenticated with HMAC-SHA256 against an embedded shared key. Subsequent runs verify the cache HMAC offline (constant-time compare via `CRYPTO_memcmp`) and re-validate against the endpoint after the cache TTL (24 hours) expires.

### `src/generators/` **(PRO VERSION ONLY)**

Passphrase generation: PCFG, Markov, priority queues. Drives the brain-wallet pipeline.

### `src/rules/` **(PRO VERSION ONLY)**

Rule engine plumbing for `gpu_rules.{cpp,cu,hpp}`. Hashcat-style rule files in `rules/` at repo root.

### `src/scrapers/` **(PRO VERSION ONLY)**

Lyrics / quotes scrapers used to seed brain-wallet wordlists.

---

## Backend selection

CMake auto-selects exactly one backend at configure time, in this priority order:

1. **Metal** if `APPLE` and `COLLIDER_USE_METAL=ON`.
2. **CUDA** if `find_package(CUDAToolkit)` succeeds and `COLLIDER_USE_CUDA=ON`.
3. **CPU** fallback otherwise.

The detected backend is written to `COLLIDER_BACKEND` (string: `"CUDA"` / `"METAL"` / `"CPU"`) and exported as a compile definition. Code that needs to branch on backend uses these definitions (`#ifdef COLLIDER_USE_CUDA`, etc.); host orchestration code is backend-agnostic and goes through the platform HAL.

CUDA-specific compile flags (Release):

```
-O3 --use_fast_math --extra-device-vectorization --restrict --fmad=true -lineinfo
--extended-lambda --expt-relaxed-constexpr
```

CUDA runtime library is **Shared** (`cudart.lib`) on Windows to avoid duplicate-symbol errors with the static runtime when linking multiple .cu translation units.

Default CUDA architectures: `75;86;89;120` (Turing, Ampere, Ada, Blackwell desktop). Override with `-DCMAKE_CUDA_ARCHITECTURES`. Note: sm_120 is desktop Blackwell (RTX 5090, RTX PRO 6000); sm_100 is datacenter Blackwell (B100/B200) and is not in the default.

---

## Free / Pro split

### What is gated

`-DCOLLIDER_PRO=ON` flips three things:

1. The `COLLIDER_PRO=1` compile definition is exposed to every target.
2. Brain-wallet sources (`src/generators/`, `src/rules/`, `src/scrapers/`, `src/license/`, several `.cu` kernels) are added to the build.
3. The compiled banner says "PRO".

When `COLLIDER_PRO=OFF` (the public Free repo's default):

- Pro source files are not compiled.
- `#ifdef COLLIDER_PRO` blocks compile out cleanly.
- Pro CLI flags are still parsed (so the user gets a clear error message instead of "unknown flag"), but the runner short-circuits with a "Pro feature" hint.

### How the public repo stays clean

`scripts/sync-to-free.sh` copies the private working tree into a fresh clone of the public Free repo, applying a `PRO_PATHS` exclusion list. Public-only files (LICENSE, the public CI workflow, `build_macos.sh`) are preserved out of `PRESERVE_PATHS`. A defense-in-depth check enumerates `PRO_PATHS` against the staged Free tree and aborts before push if any Pro path leaked. See the script header for the full procedure.

`PRO_PATHS` is the single source of truth for what is private. New Pro source files must be added there in the same commit, otherwise the next sync leaks.

---

## Pool client lifecycle

The JLP pool client is a worker that maintains one TCP connection (or TLS connection over `jlps://`) to the server. It runs in its own thread, marshalling DPs from the GPU onto the wire.

```
TCP/TLS connect
     |
     v
AUTH (within 30s) -----> AUTH_OK / AUTH_FAIL
     |                          |
     | AUTH_OK                  | AUTH_FAIL: bounded reconnect
     v                          | (3 attempts, jittered backoff)
WORK_REQ -----> WORK_ASN
     |
     | DP discovered on GPU
     v
DP_BATCH_V2 -----> DP_ACK / MSG_ERROR
     |
     v
PING every 20s -----> PONG
     |
     v
SOLUTION (server-pushed when found)
```

State-machine details and the wire format are in [JLP-PROTOCOL.md](JLP-PROTOCOL.md).

The client serializes `SSL_write` and `SSL_read` behind a mutex (one connection, one in-flight read or write at a time). DP submission is debounced and batched up to `MAX_BATCH_SIZE` (10000) per `DP_BATCH_V2` frame.

---

## Testing layout

| Directory                                | What lives there                                                    |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `tests/`                                 | Unit tests, runnable via `ctest` from `build/`.                     |
| `tests/test_cli_parser.cpp`              | CLI parser regression suite (every flag, every error path).         |
| `tests/test_secp256k1_inv.cu`            | GPU correctness: modular inverse round-trip tests.                  |
| `tests/test_ec_table_consistency.cu`     | EC point precomputed-table sanity.                                  |
| `tests/test_gpu_hash160.cu`              | SHA-256 and RIPEMD-160 KAT vectors against `data/test_vectors.txt`. |
| `tests/test_hash_vectors.cpp`            | CPU-side hash KATs.                                                 |
| `tests/test_rule_engine`                 | Rule-engine regression.                                             |
| `tests/test_priority_queue`              | Priority queue regression.                                          |
| `tests/test_platform`                    | Platform HAL smoke tests.                                           |
| `tests/protocol/` **(PRO VERSION ONLY)** | Protocol-drift round-trip tests (Python codegen vs. C++).           |

CUDA tests skip cleanly with code 77 on hosts without a GPU. Tests are gated by `-DCOLLIDER_BUILD_TESTS=ON` (default ON).

---

## Dependency summary

| Dependency   | Version       | Source                               | Purpose                                                                     |
| ------------ | ------------- | ------------------------------------ | --------------------------------------------------------------------------- |
| CMake        | 3.20+         | (system)                             | Build system.                                                               |
| OpenSSL      | 1.1+ or 3.x   | vcpkg (Windows) or distro pkg        | TLS for `jlps://`, optional.                                                |
| xxHash       | v0.8.2        | CMake FetchContent                   | Bloom filter hashing.                                                       |
| CUDA Toolkit | 12.x          | NVIDIA                               | GPU backend on Linux / Windows.                                             |
| Apple Metal  | macOS 13+ SDK | Xcode CLT                            | GPU backend on macOS.                                                       |
| RCKangaroo   | bundled       | `third_party/RCKangaroo/`            | Reference Kangaroo solver (GPLv3).                                          |
| PyYAML       | optional      | pip                                  | Build-time JLP codegen (falls back to committed generated files if absent). |
| vcpkg        | bundled       | `vcpkg/` (auto-bootstrap on Windows) | Dependency manager (Windows).                                               |

C++ standard: C++20. CUDA standard: CUDA C++ 20.

---

## Contributor walks

If you are debugging a feature, walk down from the entry point: `src/main.cpp` -> `src/runtime/<mode>_solver.cpp` -> the GPU dispatcher -> the kernel. Most code paths terminate in one or two kernels.

If you are adding a CLI flag: edit `src/cli/cli_parser.cpp`, add a matching `CLIFlags` bit in `src/core/yaml_config.hpp`, propagate it in `apply_config_to_args()`, then update [README.md](../README.md) and [CONFIGURATION.md](CONFIGURATION.md). The flag must appear in `print_usage()` in `cli_parser.cpp` (gated by `COLLIDER_PRO` if Pro-only).

If you are touching the wire protocol: edit `protocol/jlp.yaml`, regenerate via `tools/codegen/jlp_codegen.py`, update both the C++ and Python sides in lockstep, and update [JLP-PROTOCOL.md](JLP-PROTOCOL.md). Wire changes also require a corresponding update in the `collision-protocol` repo (the Python pool server) plus a passing protocol-drift round-trip test.

---

## Where to go next

| For                                                     | See                                                        |
| ------------------------------------------------------- | ---------------------------------------------------------- |
| User-facing CLI surface and quick start                 | [README.md](../README.md)                                  |
| Building from source on each platform                   | [INSTALL.md](INSTALL.md), [BUILD-MACOS.md](BUILD-MACOS.md) |
| `config.yml` schema and precedence                      | [CONFIGURATION.md](CONFIGURATION.md)                       |
| Wire format (third-party clients, alternative servers)  | [JLP-PROTOCOL.md](JLP-PROTOCOL.md)                         |
| Pool operator concerns (etiquette, accrual, anti-cheat) | [POOL.md](POOL.md)                                         |
| GPU crypto correctness tests and how to extend them     | [CRYPTO-VALIDATION.md](CRYPTO-VALIDATION.md)               |
| Release history                                         | [CHANGELOG.md](CHANGELOG.md)                               |
