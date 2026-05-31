# Building theCollider on macOS

theCollider on macOS uses **Apple Metal** for GPU compute. CUDA is unavailable on Mac. Apple Silicon (M1, M2, M3, M4) is required. Intel Macs are not supported.

This document covers both the Free and **(PRO VERSION ONLY)** editions. The same source tree builds either; the `-DCOLLIDER_PRO=ON|OFF` flag selects which.

## 1. Prerequisites (one-time)

```bash
# Xcode Command Line Tools (provides clang, Metal headers, frameworks).
xcode-select --install

# Homebrew (skip if already installed).
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Build tools and OpenSSL.
brew install cmake ninja openssl@3
```

Verify:

```bash
cmake --version       # 3.20 or newer
ninja --version       # any
clang++ --version     # Apple clang 14+
brew --prefix openssl@3   # should print a path
```

## 2. Clone and build

For the Free edition (public repo):

```bash
git clone https://github.com/hevnsnt/collider.git
cd collider
./build_macos.sh free
```

For the **(PRO VERSION ONLY)** edition (private repo, license required):

```bash
git clone git@github.com:hevnsnt/collider-pro.git
cd collider-pro
./build_macos.sh pro
```

`build_macos.sh` is the canonical macOS build entry point. It sets `OPENSSL_ROOT_DIR` from Homebrew, configures Metal, and runs Ninja with all CPU cores. The argument `pro`, `free`, and `clean` are order-independent.

Other invocations:

```bash
./build_macos.sh                 # Pro, incremental
./build_macos.sh free            # Free, incremental
./build_macos.sh clean           # Pro, wipe build/ and reconfigure
./build_macos.sh free clean      # Free, wipe build/ and reconfigure
```

## 3. What the script sets

| CMake option                | Value      | Why                                                             |
| --------------------------- | ---------- | --------------------------------------------------------------- |
| `COLLIDER_PRO`              | `ON`/`OFF` | `ON` for Pro, `OFF` for Free.                                   |
| `COLLIDER_USE_METAL`        | `ON`       | Apple GPU compute.                                              |
| `COLLIDER_USE_CUDA`         | `OFF`      | No CUDA on Mac.                                                 |
| `COLLIDER_BUILD_TESTS`      | `ON`       | Run `ctest` after build to validate.                            |
| `COLLIDER_BUILD_BENCHMARKS` | `OFF`      | Not needed for the standard Mac build.                          |
| `COLLIDER_BUILD_TOOLS`      | `OFF`      | License generator and bloom builder are Linux/Windows-targeted. |
| `COLLIDER_DISABLE_IPO`      | `ON`       | Predictable build output; matches Linux/Windows.                |
| `CMAKE_BUILD_TYPE`          | `Release`  | Optimized.                                                      |

**Do NOT pass** `-DCOLLIDER_EDITION_PRO=ON`. That option name does not exist (older build scripts had this typo); CMake silently ignores it and builds Free. The correct name is `COLLIDER_PRO`.

## 4. Artifacts

After a successful build:

```
build/collider          # Free or Pro binary (one or the other; not both per build)
build/collider_pro      # PRO VERSION ONLY: copied alongside `collider` for release packaging
```

Both are native Mach-O for arm64 (Apple Silicon).

For Free builds, `build/collider` is the only artifact. For Pro builds, the script also copies it to `build/collider_pro` so the release pipeline has consistent artifact basenames across Windows, Linux, and Mac.

## 5. Run

### Standalone puzzle (Free or Pro)

```bash
# Show banner and ROI-rank to the easiest unsolved puzzle.
./build/collider

# Specific puzzle with kangaroo (works on Metal in v1.4.1).
./build/collider --puzzle 75 --kangaroo

# Brute force (works on Metal in v1.4.1 D.3).
./build/collider --puzzle 71 --random
```

### Pool mode (Free or Pro)

```bash
./build/collider --pool jlps://pool.collisionprotocol.com:17403 \
                 --worker bc1qYourBitcoinAddress
```

### Brain wallet **(PRO VERSION ONLY)**

```bash
./build/collider_pro --brainwallet --bloom funded_addresses.blf
```

### v2 puzzle-mode kernel **(PRO VERSION ONLY)**

```bash
./build/collider_pro --puzzle-only-v2 \
                     --schemes all \
                     --puzzle-keys ./data/puzzle_history.json
```

## 6. Tests

```bash
cd build
ctest --output-on-failure
```

Pool-protocol tests open loopback sockets; if any time out, retry once.

## 7. Metal-specific notes

Metal shaders in this build are **embedded into the binary at build time** via CMake-generated headers. The CMake configure step reads each `.metal` source, wraps it in a raw string literal, and writes a `*_metal_source.h` under `build/generated/`. The host dispatcher hands the string directly to `[device newLibraryWithSource:]`. There is no filesystem lookup at runtime; a deployed binary cannot be missing or out of sync with its shaders.

Embedded shaders:

- `src/gpu/kangaroo.metal` (pool kangaroo and standalone kangaroo, v1.4.1 D.1 Jacobian rewrite).
- `src/gpu/puzzle.metal` (standalone puzzle brute-force pipeline, v1.4.1 D.3).

Filesystem-loaded shaders **(PRO VERSION ONLY)**:

- `src/gpu/sha256_metal_bench.mm`.
- `src/gpu/v2/v2_metal_dispatch.mm` (drives `brain_wallet_v2.metal`).

These are loaded from the filesystem at first use because they are part of the v2 brain-wallet pipeline that runs out of the source tree during Pro development. Run the binary from the repo root, or copy the matching `.metal` file next to the binary, if you see "could not locate" messages.

## 8. Troubleshooting

**"Could NOT find OpenSSL"**

Set `OPENSSL_ROOT_DIR` from the Homebrew prefix:

```bash
export OPENSSL_ROOT_DIR=$(brew --prefix openssl@3)
./build_macos.sh clean
```

**"Metal.framework not found" or linker cannot resolve `MTLCreateSystemDefaultDevice`**

Reinstall the Xcode Command Line Tools:

```bash
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install
```

**Build fails on Intel Mac**

Intel Macs are not supported. Apple Silicon (M1, M2, M3, M4) only.

**`ctest -R V2Orchestrator` segfault or "could not open file"** **(PRO VERSION ONLY)**

The test writes JSON files under `$TMPDIR/tmp_v2_orch/`. If `$TMPDIR` is unset, set it:

```bash
export TMPDIR=/tmp
```

## 9. Performance

For benchmarked numbers on each chip (M1, M1 Max, M1 Ultra, M2 family, M3 family, M4 family), see the GitHub release notes for each tagged version. Numbers depend strongly on macOS version, batch size, and kernel path; anything quoted statically in the docs would go stale within weeks of the next OS release.

To measure your specific machine:

```bash
./build/collider --benchmark
./build/collider --benchmark --benchmark-time 60
```

## 10. Releasing

Pro releases on Mac follow the same flow as Windows and Linux:

1. Tag `v1.4.1-pro` on `collider-pro/main` after merging.
2. CI builds the artifact for distribution.
3. Free release follows via `scripts/sync-to-free.sh v1.4.1-free`.
4. Website auto-pulls from `github.com/hevnsnt/collider/releases/latest`.

---

## Where to go next

| For                                      | See                                          |
| ---------------------------------------- | -------------------------------------------- |
| First-time install (Xcode CLT, Homebrew) | [INSTALL.md](INSTALL.md)                     |
| Linux build reference                    | [BUILD-LINUX.md](BUILD-LINUX.md)             |
| Windows build reference                  | [BUILD-WINDOWS.md](BUILD-WINDOWS.md)         |
| CLI surface and runtime usage            | [README.md](../README.md)                    |
| `config.yml` schema and precedence       | [CONFIGURATION.md](CONFIGURATION.md)         |
| Source-tree map                          | [ARCHITECTURE.md](ARCHITECTURE.md)           |
| GPU correctness test strategy            | [CRYPTO-VALIDATION.md](CRYPTO-VALIDATION.md) |
