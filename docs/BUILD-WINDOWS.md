# Building theCollider on Windows

Platform-specific build reference for Windows x64 with the CUDA backend. For a step-by-step first-time install (Visual Studio, CUDA Toolkit, dependency setup), see [INSTALL.md](INSTALL.md). This document is the reference for CMake options, build flags, vcpkg bootstrap, and Windows-specific troubleshooting.

This document covers both the Free and **(PRO VERSION ONLY)** editions. The same source tree builds either; the `-DCOLLIDER_PRO=ON|OFF` flag selects which.

---

## Table of contents

- [Quick build](#quick-build)
- [Build environment](#build-environment)
- [CMake options](#cmake-options)
- [CUDA architecture selection](#cuda-architecture-selection)
- [vcpkg and OpenSSL](#vcpkg-and-openssl)
- [Build types and optimization](#build-types-and-optimization)
- [Tests](#tests)
- [Multi-GPU systems](#multi-gpu-systems)
- [Windows-specific troubleshooting](#windows-specific-troubleshooting)
- [Release packaging](#release-packaging)
- [Where to go next](#where-to-go-next)

---

## Quick build

From "x64 Native Tools Command Prompt for VS 2022":

```cmd
git clone https://github.com/hevnsnt/collider.git
cd collider

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Output: `build\collider.exe`.

For the **(PRO VERSION ONLY)** edition (private repo, license required):

```cmd
git clone git@github.com:hevnsnt/collider-pro.git
cd collider-pro
build_pro.bat
```

`build_pro.bat` calls `vcvars64.bat`, configures CMake with Ninja, and builds with 24 parallel jobs. Outputs: `build\collider_pro.exe` and `build\collider.exe`.

---

## Build environment

CUDA on Windows requires the MSVC toolchain to be available in the same shell that runs CMake. The two reliable ways to set this up:

### Option A: x64 Native Tools Command Prompt

The Visual Studio installer creates a shortcut named **x64 Native Tools Command Prompt for VS 2022**. Launching this gives you a `cmd.exe` shell with `vcvars64.bat` already sourced: MSVC, the Windows SDK, plus the CUDA Toolkit's `nvcc` are on `PATH`.

This is the recommended entry point for ad-hoc builds.

### Option B: scripted vcvars

Bash and PowerShell shells do not have a way to source `vcvars64.bat` in-place. The workaround is a thin batch wrapper:

```batch
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

`build_pro.bat` in the repo root is the production version of this pattern.

PowerShell and Git Bash users who want to call CMake directly should launch from the x64 Native Tools Command Prompt or use the batch-wrapper pattern. **Running `cmake` from a plain `cmd` or PowerShell prompt without `vcvars64.bat` loaded will fail** with `nvcc` errors or with the CMake C++ compiler test failing.

---

## CMake options

Options recognized by the project (defaults in `CMakeLists.txt`):

| Option                      | Default      | Description                                                               |
| --------------------------- | ------------ | ------------------------------------------------------------------------- |
| `COLLIDER_USE_CUDA`         | `ON`         | Enable the CUDA backend. Required for GPU compute on Windows.             |
| `COLLIDER_USE_METAL`        | `ON`         | Enable the Metal backend (no effect on Windows).                          |
| `COLLIDER_USE_CPU`          | `ON`         | Enable the CPU fallback backend.                                          |
| `COLLIDER_BUILD_TESTS`      | `ON`         | Build unit tests (`ctest` runs from `build\`).                            |
| `COLLIDER_BUILD_BENCHMARKS` | `ON`         | Build the standalone benchmark targets.                                   |
| `COLLIDER_BUILD_TOOLS`      | `ON`         | Build CLI tools (`build_bloom`, `generate_license`, ...).                 |
| `COLLIDER_PRO`              | `OFF`        | Enable the Pro brain-wallet pipeline. Requires the private source tree.   |
| `CMAKE_BUILD_TYPE`          | `Release`    | `Release`, `Debug`, or `RelWithDebInfo`.                                  |
| `CMAKE_CUDA_ARCHITECTURES`  | (no default) | Target SM versions. The shipped release uses `"75;86;89;120"`. See below. |

Override at configure time:

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
                       -DCOLLIDER_BUILD_TESTS=OFF ^
                       -DCMAKE_CUDA_ARCHITECTURES="89"
```

(The `^` is the cmd.exe line-continuation character.)

---

## CUDA architecture selection

The shipped release (built by `.github/workflows/build-release.yml`) targets `"75;86;89;120"` on CUDA Toolkit 12.8. Those four SMs cover every consumer card from RTX 20 through RTX 50 plus the RTX 6000 Ada and RTX PRO 6000 Blackwell workstation cards. Compiling for every architecture roughly multiplies NVCC's work by the number of SMs; for local development, restrict to your card.

| GPU                                          | Architecture         | SM    | Required CUDA toolkit |
| -------------------------------------------- | -------------------- | ----- | --------------------- |
| RTX 20 series (2060 / 2070 / 2080 / 2080 Ti) | Turing               | `75`  | 12.x                  |
| RTX 30 series (3060 / 3070 / 3080 / 3090)    | Ampere consumer      | `86`  | 12.x                  |
| RTX 40 series + RTX 6000 Ada Workstation     | Ada Lovelace         | `89`  | 12.x                  |
| RTX 50 series + RTX PRO 6000 Blackwell       | Blackwell consumer   | `120` | **12.8+**             |
| GTX 10 series (Pascal)                       | Pascal               | `61`  | 12.x                  |
| A100 / A30                                   | Ampere datacenter    | `80`  | 12.x                  |
| H100 / H200                                  | Hopper               | `90`  | 12.0+                 |
| B100 / B200                                  | Blackwell datacenter | `100` | 12.4+                 |

### Easiest: auto-detect this machine

If you are building on the same machine the binary will run on:

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
                       -DCMAKE_CUDA_ARCHITECTURES=native
```

`native` queries each installed GPU's compute capability via the CUDA driver and builds for exactly those SMs. Requires CMake 3.24+. Fastest compile, but the binary is not portable to other machines.

### Specific architecture

For a single card type (faster compile than building for many):

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
                       -DCMAKE_CUDA_ARCHITECTURES="89"
```

### Multiple architectures (portable binary)

For a binary that runs on several different cards:

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
                       -DCMAKE_CUDA_ARCHITECTURES="75;86;89;120"
```

Each architecture adds roughly 10 to 15 MB of SASS to the binary and increases compile time roughly linearly. For forward compatibility with future GPUs, append `-virtual` to the highest entry (for example `"75;86;89;120-virtual"`) so newer cards can JIT-compile from PTX at first launch.

Minimum supported compute capability is **7.5** (Turing). Pre-Turing cards lack `--use_fast_math` plus lambda-in-device patterns used by the kernels.

---

## vcpkg and OpenSSL

theCollider on Windows uses [vcpkg](https://github.com/microsoft/vcpkg) to provide OpenSSL (for `jlps://` TLS pool connections). The repo includes a `vcpkg.json` manifest that lists `openssl` as a dependency.

The CMake configure step auto-bootstraps vcpkg in `.\vcpkg\` if `VCPKG_ROOT` is not already set. This downloads, builds, and caches OpenSSL on the first run; subsequent configures reuse the cache.

First-build expectations:

- Cold configure: 5 to 15 minutes (vcpkg compiles OpenSSL).
- Warm configure: under 30 seconds.

To use a pre-existing vcpkg installation instead of the bundled bootstrap, set `VCPKG_ROOT` before configuring:

```cmd
set VCPKG_ROOT=C:\src\vcpkg
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

To disable vcpkg entirely (and provide OpenSSL another way):

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=C:\path\to\openssl
```

The build degrades gracefully without OpenSSL: TLS pool connections (`jlps://`) are unavailable, but plaintext `jlp://` and the standalone solver still work.

---

## Build types and optimization

### Release (default)

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

NVCC flags applied to GPU code:

```
-O3 --use_fast_math --extra-device-vectorization --restrict --fmad=true -lineinfo
--extended-lambda --expt-relaxed-constexpr
```

MSVC host flags: `/O2 /arch:AVX2 /GL` with link-time code generation (`/LTCG`).

The build links against the **shared** CUDA runtime (`cudart.lib`) to avoid duplicate-symbol errors when multiple `.cu` translation units are linked. This means deployed `collider.exe` requires `cudart64_12.dll` on the target machine (shipped in the CUDA Toolkit redistributable).

### RelWithDebInfo

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Same optimization as Release plus PDB debug info. Use when you need `cuda-gdb`, NVIDIA Nsight, or Visual Studio against an optimized binary.

### Debug

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug
```

Disables optimization, enables device-side assertions. Significantly slower; suitable only for correctness debugging.

---

## Tests

From "x64 Native Tools Command Prompt for VS 2022":

```cmd
cd build
ctest --output-on-failure
```

CUDA tests skip cleanly with code 77 on hosts without a GPU. Run a single suite:

```cmd
ctest --output-on-failure -R GpuHash160
ctest --output-on-failure -R secp256k1_inv
ctest --output-on-failure -R cli_parser
```

For details on the GPU correctness test strategy, see [CRYPTO-VALIDATION.md](CRYPTO-VALIDATION.md).

---

## Multi-GPU systems

CUDA on Windows enumerates devices via `cudaGetDeviceCount`; the order matches `nvidia-smi -L`. Both consumer GeForce drivers and the data-center (`Tesla`) driver branch are supported.

By default, theCollider uses every visible GPU. To restrict:

```cmd
.\collider --gpus 0,2
```

Or in `config.yml`:

```yaml
gpu:
  devices: [0, 2]
```

Or via the environment:

```cmd
set CUDA_VISIBLE_DEVICES=0,2
.\collider
```

Each GPU runs an independent kangaroo walk with its own DP queue. Cross-GPU work distribution is automatic.

### WDDM vs TCC driver mode

Consumer GeForce cards run in WDDM (Windows Display Driver Model). Data-center cards (Tesla, A100, H100) can be switched to TCC for lower-latency CUDA. theCollider works in either mode; TCC typically gives a modest throughput uplift on data-center cards due to the absent display compositor, though the magnitude is workload-dependent. Measure with `--benchmark` on your own hardware before relying on a specific number.

```cmd
nvidia-smi -i 0 -dm 1                  # WDDM = 0, TCC = 1
```

A reboot is required after the mode change.

---

## Windows-specific troubleshooting

### `'nvcc' is not recognized as an internal or external command`

`vcvars64.bat` is not loaded in the current shell. Open "x64 Native Tools Command Prompt for VS 2022", or use a batch wrapper that sources it.

### `LINK : fatal error LNK1181: cannot open input file 'cudart_static.lib'`

This indicates a static-CUDA-runtime link configuration that the project does not support. The build is configured to link against the **shared** CUDA runtime; clean and reconfigure:

```cmd
rmdir /s /q build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
```

### `CMake Error: Could not find CUDAToolkit`

The CUDA Toolkit is not installed or its environment variables are not set. Verify:

```cmd
nvcc --version
echo %CUDA_PATH%
```

If `CUDA_PATH` is unset, the CUDA installer did not complete cleanly. Reinstall the CUDA Toolkit.

### `vcpkg integration failed` or vcpkg bootstrap hangs

vcpkg's first-run bootstrap downloads dependencies from GitHub and Microsoft CDNs. Corporate firewalls sometimes block these. Workarounds:

- Set `HTTPS_PROXY` and `HTTP_PROXY` before configuring.
- Use a pre-bootstrapped vcpkg from another machine and set `VCPKG_ROOT`.
- Skip vcpkg entirely by providing OpenSSL via `-DOPENSSL_ROOT_DIR=...`.

### `cudart64_12.dll was not found` at runtime

The CUDA runtime DLL is not on `PATH` when launching `collider.exe`. The CUDA Toolkit installer adds it to `PATH` system-wide; verify:

```cmd
where cudart64_12.dll
```

If absent, reinstall the CUDA Toolkit or copy `cudart64_12.dll` next to `collider.exe`.

### `MSVC error C2039: 'ERROR': is not a member of 'X'`

This usually indicates a third-party header has `#define ERROR ...` that conflicts with a project symbol. theCollider uses `MSG_ERROR` (not `ERROR`) for its JLP message-type constant for exactly this reason; if you see the error, suspect a recently added header. The Windows `<winerror.h>` macro is the most common culprit.

### TLS handshake failure on `jlps://` URLs at runtime

The system trust store is the Windows certificate store; v1.4.1 fails hard at TLS init rather than falling back to no-verify. If the cert store is empty (unusual on a managed install), set a custom CA bundle:

```cmd
set SSL_CERT_FILE=C:\path\to\cacert.pem
```

### Builds slow despite `--parallel`

Windows Defender real-time scanning adds latency to every NVCC invocation. Add the build directory and the CUDA Toolkit directory to the Defender exclusion list (Settings -> Update and Security -> Windows Security -> Virus and threat protection -> Exclusions).

---

## Release packaging

Windows Free release artifacts are built in CI on tag push (`v*`). The local equivalent:

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
                       -DCMAKE_CUDA_ARCHITECTURES="75;86;89;120"
cmake --build build --parallel
certutil -hashfile build\collider.exe SHA256 > build\collider.exe.sha256
```

The release artifact bundles every supported CUDA architecture in one binary; CUDA's runtime picks the closest match at launch. SM 120 (RTX 50 / RTX PRO 6000 Blackwell) requires CUDA Toolkit 12.8 or newer; older toolkits will fail at configure time.

Pro releases (binary-only, license-gated) go through the private CI plus the per-customer license signing pipeline; the operator workflow is documented in the private repo.

---

## Where to go next

| For                                              | See                                          |
| ------------------------------------------------ | -------------------------------------------- |
| First-time install (Visual Studio, CUDA Toolkit) | [INSTALL.md](INSTALL.md)                     |
| Linux build reference                            | [BUILD-LINUX.md](BUILD-LINUX.md)             |
| macOS build reference                            | [BUILD-MACOS.md](BUILD-MACOS.md)             |
| CLI surface and runtime usage                    | [README.md](../README.md)                    |
| `config.yml` schema and precedence               | [CONFIGURATION.md](CONFIGURATION.md)         |
| Source-tree map                                  | [ARCHITECTURE.md](ARCHITECTURE.md)           |
| GPU correctness test strategy                    | [CRYPTO-VALIDATION.md](CRYPTO-VALIDATION.md) |
