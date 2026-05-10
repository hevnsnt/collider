# Building theCollider on Linux

Platform-specific build reference for Linux x64 with the CUDA backend. For a step-by-step first-time install (distro packages, CUDA Toolkit install, environment setup), see [INSTALL.md](INSTALL.md). This document is the reference for CMake options, build flags, multi-GPU layouts, and Linux-specific troubleshooting.

This document covers both the Free and **(PRO VERSION ONLY)** editions. The same source tree builds either; the `-DCOLLIDER_PRO=ON|OFF` flag selects which.

---

## Table of contents

- [Quick build](#quick-build)
- [CMake options](#cmake-options)
- [CUDA architecture selection](#cuda-architecture-selection)
- [Build types and optimization](#build-types-and-optimization)
- [Tests](#tests)
- [Multi-GPU systems](#multi-gpu-systems)
- [Linux-specific troubleshooting](#linux-specific-troubleshooting)
- [Release packaging](#release-packaging)
- [Where to go next](#where-to-go-next)

---

## Quick build

```bash
git clone https://github.com/hevnsnt/collider.git
cd collider

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Output: `build/collider`.

For the **(PRO VERSION ONLY)** edition (private repo, license required):

```bash
git clone git@github.com:hevnsnt/collider-pro.git
cd collider-pro

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCOLLIDER_PRO=ON
cmake --build build --parallel
```

Outputs: `build/collider` and `build/collider_pro` (Pro builds produce both binaries for release packaging).

---

## CMake options

Options recognized by the project (defaults in `CMakeLists.txt`):

| Option                      | Default     | Description                                                             |
| --------------------------- | ----------- | ----------------------------------------------------------------------- |
| `COLLIDER_USE_CUDA`         | `ON`        | Enable the CUDA backend. Required for GPU compute on Linux.             |
| `COLLIDER_USE_METAL`        | `ON`        | Enable the Metal backend (no effect on Linux).                          |
| `COLLIDER_USE_CPU`          | `ON`        | Enable the CPU fallback backend.                                        |
| `COLLIDER_BUILD_TESTS`      | `ON`        | Build unit tests (`ctest` runs from `build/`).                          |
| `COLLIDER_BUILD_BENCHMARKS` | `ON`        | Build the standalone benchmark targets.                                 |
| `COLLIDER_BUILD_TOOLS`      | `ON`        | Build CLI tools (`build_bloom`, `generate_license`, ...).               |
| `COLLIDER_PRO`              | `OFF`       | Enable the Pro brain-wallet pipeline. Requires the private source tree. |
| `CMAKE_BUILD_TYPE`          | `Release`   | `Release`, `Debug`, or `RelWithDebInfo`.                                |
| `CMAKE_CUDA_ARCHITECTURES`  | `86;89;100` | Target SM versions. Override for faster local builds.                   |

Override at configure time:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
                       -DCOLLIDER_BUILD_TESTS=OFF \
                       -DCMAKE_CUDA_ARCHITECTURES="89"
```

---

## CUDA architecture selection

The default architecture list (`86;89;100`) covers Ampere, Ada, and Blackwell. Compiling for every architecture roughly triples NVCC's work; for local development, restrict to your card.

| GPU series           | Compute capability | `CMAKE_CUDA_ARCHITECTURES` value |
| -------------------- | ------------------ | -------------------------------- |
| RTX 2000 (Turing)    | sm_75              | `75`                             |
| RTX 3000 (Ampere)    | sm_86              | `86`                             |
| RTX 4000 (Ada)       | sm_89              | `89`                             |
| RTX 5000 (Blackwell) | sm_100             | `100`                            |
| H100 (Hopper)        | sm_90              | `90`                             |

Example (RTX 4090 only):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
                       -DCMAKE_CUDA_ARCHITECTURES="89"
```

Multi-card hosts with mixed architectures should list every architecture present:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
                       -DCMAKE_CUDA_ARCHITECTURES="86;89"
```

Minimum supported compute capability is **7.5** (Turing). Pre-Turing cards lack the required PTX features for `--use_fast_math` plus the lambda-in-device patterns used by the kernels.

---

## Build types and optimization

### Release (default)

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

NVCC flags applied to GPU code:

```
-O3 --use_fast_math --extra-device-vectorization --restrict --fmad=true -lineinfo
--extended-lambda --expt-relaxed-constexpr
```

GCC/Clang host flags include LTO and architecture-aware optimization. The build links against the **shared** CUDA runtime (`libcudart.so`) to avoid duplicate-symbol errors when multiple `.cu` translation units are linked.

### RelWithDebInfo

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Same optimization as Release plus DWARF debug info. Use when you need `cuda-gdb` or `nvprof` against an optimized binary.

### Debug

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

Disables optimization, enables device-side assertions. Significantly slower; suitable only for correctness debugging.

---

## Tests

```bash
cd build
ctest --output-on-failure
```

CUDA tests skip cleanly with code 77 on hosts without a GPU. Run a single suite:

```bash
ctest --output-on-failure -R GpuHash160
ctest --output-on-failure -R secp256k1_inv
ctest --output-on-failure -R cli_parser
```

For details on the GPU correctness test strategy, see [CRYPTO-VALIDATION.md](CRYPTO-VALIDATION.md).

---

## Multi-GPU systems

CUDA enumerates devices via `cudaGetDeviceCount`; the order matches `nvidia-smi -L` unless `CUDA_VISIBLE_DEVICES` is set.

By default, theCollider uses every visible GPU. To restrict:

```bash
./collider --gpus 0,2          # CLI: skip GPU 1
```

Or in `config.yml`:

```yaml
gpu:
  devices: [0, 2]
```

Or via the environment (also hides them from `nvidia-smi`):

```bash
CUDA_VISIBLE_DEVICES=0,2 ./collider
```

Each GPU runs an independent kangaroo walk with its own DP queue. Cross-GPU work distribution is automatic; you do not need to partition the search range manually.

---

## Linux-specific troubleshooting

### `CUDA_ERROR_NO_DEVICE` despite `nvidia-smi` working

The running user is not in the `video` group, or the NVIDIA character devices are not present.

```bash
sudo usermod -aG video $USER
ls -l /dev/nvidia*
```

Log out and back in for group membership to take effect.

### `nvcc fatal: Unsupported gpu architecture 'compute_XX'`

Your CUDA Toolkit is too old for the architecture you requested. CUDA 12.0 added Hopper (sm_90); CUDA 12.4 added Blackwell (sm_100). Upgrade to a Toolkit that supports your card or drop the unsupported architecture from `CMAKE_CUDA_ARCHITECTURES`.

### `ptxas error: Entry function uses too much local data`

Register pressure on a specific kernel exceeds the per-thread cap. Most often hits older CUDA Toolkit versions; current CUDA 12.x compiles every kernel cleanly. Workaround:

```bash
cmake -B build -DCMAKE_CUDA_FLAGS="-maxrregcount=128"
```

### Linker out of memory during build

CUDA linking with full parallelism is memory-hungry. Reduce parallel jobs:

```bash
cmake --build build --parallel 2
```

Or add swap space; CUDA 12.x linker peaks around 4 to 6 GB per concurrent translation unit on full-architecture builds.

### `libcudart.so: cannot open shared object file`

The CUDA runtime is not on the dynamic loader path.

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Or add it permanently in `/etc/ld.so.conf.d/cuda.conf` and run `sudo ldconfig`.

### OpenSSL not found at configure time

```bash
sudo apt install libssl-dev          # Debian / Ubuntu
sudo dnf install openssl-devel       # Fedora / RHEL
sudo pacman -S openssl               # Arch
```

The build is configured to find OpenSSL automatically; specifying a custom path is rarely needed. If you have OpenSSL in a non-standard prefix:

```bash
cmake -B build -DOPENSSL_ROOT_DIR=/opt/openssl
```

### TLS handshake failure on `jlps://` URLs at runtime

The system trust store is empty or missing. v1.4.1 fails hard at TLS init rather than falling back to no-verify.

```bash
sudo apt install ca-certificates     # Debian / Ubuntu
sudo dnf install ca-certificates     # Fedora / RHEL
sudo update-ca-certificates
```

---

## Release packaging

Linux Free release artifacts are built in CI on tag push (`v*`). The local equivalent:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
                       -DCMAKE_CUDA_ARCHITECTURES="75;86;89;100"
cmake --build build --parallel
strip build/collider
sha256sum build/collider > build/collider.sha256
```

The release artifact bundles every supported CUDA architecture in one binary; CUDA's runtime picks the closest match at launch.

Pro releases (binary-only, license-gated) go through the private CI plus the per-customer license signing pipeline; the operator workflow is documented in the private repo.

---

## Where to go next

| For                                                | See                                          |
| -------------------------------------------------- | -------------------------------------------- |
| First-time install (distro packages, CUDA Toolkit) | [INSTALL.md](INSTALL.md)                     |
| Windows build reference                            | [BUILD-WINDOWS.md](BUILD-WINDOWS.md)         |
| macOS build reference                              | [BUILD-MACOS.md](BUILD-MACOS.md)             |
| CLI surface and runtime usage                      | [README.md](../README.md)                    |
| `config.yml` schema and precedence                 | [CONFIGURATION.md](CONFIGURATION.md)         |
| Source-tree map                                    | [ARCHITECTURE.md](ARCHITECTURE.md)           |
| GPU correctness test strategy                      | [CRYPTO-VALIDATION.md](CRYPTO-VALIDATION.md) |
