# Installation Guide

This guide covers building theCollider from source on Linux, Windows, and macOS. For prebuilt binaries see the [GitHub Releases page](https://github.com/hevnsnt/collider/releases).

This document covers the **free** edition. **(PRO VERSION ONLY)** builds are issued per-license to paying customers; see [collisionprotocol.com/pro](https://collisionprotocol.com/pro).

---

## Table of Contents

- [GPU support in the shipped release](#gpu-support-in-the-shipped-release)
- [Prerequisites](#prerequisites)
- [Linux](#linux)
- [Windows](#windows)
- [macOS](#macos)
- [Build Options](#build-options)
- [Verifying the Build](#verifying-the-build)
- [Troubleshooting](#troubleshooting)

---

## GPU support in the shipped release

The binaries published at [github.com/hevnsnt/collider/releases](https://github.com/hevnsnt/collider/releases) are built for these NVIDIA GPUs (one fat binary, runtime-selected per device):

| Marketing line                        | Architecture       | SM      |
| ------------------------------------- | ------------------ | ------- |
| RTX 20 (2060 / 2070 / 2080 / 2080 Ti) | Turing             | **75**  |
| RTX 30 (3060 / 3070 / 3080 / 3090)    | Ampere consumer    | **86**  |
| RTX 40 (4060 / 4070 / 4080 / 4090)    | Ada Lovelace       | **89**  |
| RTX 6000 Ada Workstation              | Ada Lovelace       | **89**  |
| RTX 50 (5060 / 5070 / 5080 / 5090)    | Blackwell consumer | **120** |
| RTX PRO 6000 Blackwell Workstation    | Blackwell consumer | **120** |

If your card is in this table, **download the release binary and skip the build steps below**. If your card is not in this table (older Pascal / Volta cards, datacenter Ampere A100 / A30, Hopper H100 / H200, datacenter Blackwell B100 / B200, or anything pre-Turing), see [Build Options](#build-options) below for how to compile from source with a custom `CMAKE_CUDA_ARCHITECTURES` value.

Apple Silicon (M1 / M2 / M3 / M4) uses Metal instead of CUDA and is also supported by the shipped macOS arm64 release.

---

## Prerequisites

### All platforms

| Requirement  | Version       | Notes                                     |
| ------------ | ------------- | ----------------------------------------- |
| CMake        | 3.20 or newer | Build system                              |
| Git          | any           | Source checkout                           |
| C++ compiler | C++20         | MSVC 2022, GCC 11+, or Apple Clang 14+    |
| Ninja        | recommended   | Faster builds than Make on every platform |

### GPU backends

theCollider auto-detects one of three backends. Auto-detection runs in this order:

1. **Metal** on Apple Silicon (macOS arm64).
2. **CUDA** on Linux or Windows when the CUDA Toolkit is found.
3. **CPU** fallback (slow; correctness only).

**NVIDIA (CUDA backend):**

- NVIDIA GPU with Compute Capability 7.5 or higher (Turing or newer).
- CUDA Toolkit 12.x. The codebase targets `--use_fast_math`, lambda-in-device support, and relaxed-constexpr; toolkits older than 12.0 will not compile every kernel.
- Latest stable NVIDIA driver for your CUDA version.

**Apple Silicon (Metal backend):**

- Apple M1, M2, M3, or M4 (any tier).
- macOS 13 or newer (the Metal API surface used by the Jacobian kangaroo rewrite is 13+).
- Xcode Command Line Tools (`xcode-select --install`).

Intel Macs with eGPUs are **not supported**. Apple Silicon only.

---

## Linux

### Step 1: install system dependencies

**Ubuntu / Debian:**

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build git libssl-dev
```

**Fedora / RHEL:**

```bash
sudo dnf install -y gcc-c++ cmake ninja-build git openssl-devel
```

**Arch Linux:**

```bash
sudo pacman -S base-devel cmake ninja git openssl
```

### Step 2: install CUDA Toolkit

**Option A: NVIDIA repository (Ubuntu / Debian).**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-4
```

**Option B: NVIDIA installer.** Download from <https://developer.nvidia.com/cuda-downloads> and follow the on-screen instructions.

Verify:

```bash
nvcc --version
nvidia-smi
```

### Step 3: clone and build

```bash
git clone https://github.com/hevnsnt/collider.git
cd collider

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

To target a single CUDA architecture for faster compile times:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build build --parallel
```

When `CMAKE_CUDA_ARCHITECTURES` is not specified, CMake selects a fallback default that depends on your CUDA toolkit version and may not include your GPU. For a build that runs everywhere the shipped release runs, pass `"75;86;89;120"` (and use CUDA Toolkit 12.8+ for SM 120). For a build that targets exactly this machine, pass `native` (CMake 3.24+). See [Build Options](#build-options) for the full mapping.

Output: `build/collider`.

### Step 4: run the test suite

```bash
cd build
ctest --output-on-failure
```

---

## Windows

### Step 1: install Visual Studio

Install Visual Studio 2022 (Community or Build Tools) with the **Desktop development with C++** workload. Required components:

- MSVC v143 build tools.
- Windows 10 / 11 SDK.
- C++ CMake tools for Windows.

### Step 2: install the CUDA Toolkit

Download CUDA 12.x from <https://developer.nvidia.com/cuda-downloads>, run the installer, reboot when prompted.

Verify from "x64 Native Tools Command Prompt for VS 2022":

```cmd
nvcc --version
```

### Step 3: install Git and CMake

```cmd
winget install Git.Git
winget install Kitware.CMake
winget install Ninja-build.Ninja
```

### Step 4: clone and build

From "x64 Native Tools Command Prompt for VS 2022":

```cmd
git clone https://github.com/hevnsnt/collider.git
cd collider

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The CMake configure step auto-bootstraps vcpkg in `./vcpkg/` if `VCPKG_ROOT` is not already set. This downloads OpenSSL on first run.

Output: `build\collider.exe`.

### Step 5: run the test suite

```cmd
cd build
ctest --output-on-failure
```

---

## macOS

theCollider on macOS uses **Apple Metal**. Apple Silicon (M1, M2, M3, M4) is required.

### Step 1: install Xcode Command Line Tools

```bash
xcode-select --install
```

### Step 2: install Homebrew dependencies

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install cmake ninja openssl@3
```

### Step 3: clone and build

The canonical entry point on macOS is `./build_macos.sh`, which sets `OPENSSL_ROOT_DIR` from Homebrew, configures Metal, and runs Ninja with all CPU cores.

```bash
git clone https://github.com/hevnsnt/collider.git
cd collider
./build_macos.sh free
```

For a clean rebuild:

```bash
./build_macos.sh free clean
```

Output: `build/collider`.

For a deeper Mac-specific reference (CMake options, embedded Metal shaders, troubleshooting), see [BUILD-MACOS.md](BUILD-MACOS.md).

---

## Build Options

CMake options recognized by the project:

| Option                      | Default      | Description                                                                         |
| --------------------------- | ------------ | ----------------------------------------------------------------------------------- |
| `COLLIDER_USE_CUDA`         | `ON`         | Enable the CUDA backend (Linux / Windows).                                          |
| `COLLIDER_USE_METAL`        | `ON`         | Enable the Metal backend (macOS).                                                   |
| `COLLIDER_USE_CPU`          | `ON`         | Enable the CPU fallback backend.                                                    |
| `COLLIDER_BUILD_TESTS`      | `ON`         | Build unit tests (`ctest` runs from `build/`).                                      |
| `COLLIDER_BUILD_BENCHMARKS` | `ON`         | Build the benchmark targets.                                                        |
| `COLLIDER_BUILD_TOOLS`      | `ON`         | Build CLI tools (`build_bloom`, `generate_license`, etc.).                          |
| `CMAKE_BUILD_TYPE`          | `Release`    | `Release`, `Debug`, or `RelWithDebInfo`.                                            |
| `CMAKE_CUDA_ARCHITECTURES`  | (no default) | Target SM versions. See the table below. The shipped release uses `"75;86;89;120"`. |

### CUDA architecture selection

Pick the SM value(s) matching your hardware. The shipped release covers the first four rows; for anything else, build from source with the matching value.

| GPU                                          | Architecture         | SM    | Required CUDA toolkit |
| -------------------------------------------- | -------------------- | ----- | --------------------- |
| RTX 20 series (2060 / 2070 / 2080 / 2080 Ti) | Turing               | `75`  | 12.x                  |
| RTX 30 series (3060 / 3070 / 3080 / 3090)    | Ampere consumer      | `86`  | 12.x                  |
| RTX 40 series + RTX 6000 Ada Workstation     | Ada Lovelace         | `89`  | 12.x                  |
| RTX 50 series + RTX PRO 6000 Blackwell       | Blackwell consumer   | `120` | **12.8+**             |
| GTX 10 series (1060 / 1070 / 1080)           | Pascal               | `61`  | 12.x                  |
| A100 / A30                                   | Ampere datacenter    | `80`  | 12.x                  |
| H100 / H200                                  | Hopper               | `90`  | 12.0+                 |
| B100 / B200                                  | Blackwell datacenter | `100` | 12.4+                 |

#### Easiest: auto-detect this machine

If the build will only run on the same machine it was compiled on:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
                       -DCMAKE_CUDA_ARCHITECTURES=native
```

`native` queries each installed GPU's compute capability via the CUDA driver and builds for exactly those SMs. Requires CMake 3.24+. Fastest compile, but the binary is not portable to other machines.

#### Specific architecture

For one card type (faster compile than building for many):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
                       -DCMAKE_CUDA_ARCHITECTURES="89"
```

#### Multiple architectures (portable binary)

For a binary that runs on several different cards:

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
                       -DCMAKE_CUDA_ARCHITECTURES="75;86;89;120"
```

Each architecture adds roughly 10 to 15 MB of SASS to the binary and increases compile time roughly linearly. For forward compatibility with future GPUs, append `-virtual` to the highest entry (for example `"75;86;89;120-virtual"`) so newer cards can JIT-compile from PTX at first launch.

---

## Verifying the Build

```bash
# Quick benchmark (defaults to 30 seconds).
./collider --benchmark

# Help / CLI surface.
./collider --help

# Full unit-test suite.
cd build
ctest --output-on-failure
```

---

## Troubleshooting

### CUDA not found

```
CMake Error: Could not find CUDAToolkit
```

1. Verify `nvcc --version` runs.
2. Add CUDA to `PATH` and `LD_LIBRARY_PATH`:

   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

3. Set `CUDAToolkit_ROOT` if CUDA is in a non-standard location:

   ```bash
   cmake -B build -DCUDAToolkit_ROOT=/opt/cuda
   ```

### GPU not detected at runtime

```
No CUDA-capable device detected
```

1. Verify `nvidia-smi` reports your GPU.
2. Check Compute Capability is 7.5 or higher (Turing, Ampere, Ada, Blackwell, Hopper).
3. Update the NVIDIA driver to a version compatible with your CUDA Toolkit.
4. Reboot after driver installation.

### Build fails with "ptxas: too much local memory"

```
ptxas error: Entry function uses too much local data
```

Reduce register pressure by capping the per-thread register allocation:

```bash
cmake -B build -DCMAKE_CUDA_FLAGS="-maxrregcount=128"
```

### Out of memory during build

```
c++: fatal error: Killed signal terminated program
```

The CUDA kernels link-stage can use significant RAM with parallel jobs. Reduce parallelism:

```bash
cmake --build build --parallel 2
```

Add swap space if the host has under 16 GB RAM.

### Metal backend issues (macOS)

```
Metal framework not found
```

1. Verify Xcode Command Line Tools: `xcode-select -p`.
2. Reinstall if missing: `xcode-select --install`.
3. Accept the Xcode license: `sudo xcodebuild -license accept`.

### Windows: nvcc not on PATH

```
'nvcc' is not recognized as an internal or external command
```

Open "x64 Native Tools Command Prompt for VS 2022" (this sets the MSVC plus CUDA environment in one step). Direct invocation from a plain `cmd` or PowerShell prompt will not have `vcvars64.bat` loaded.

### Linux: no CUDA device despite nvidia-smi working

```
CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
```

Add the running user to the `video` group:

```bash
sudo usermod -aG video $USER
```

Log out and back in.

---

## Getting Help

If a build issue is not covered here, open an issue at <https://github.com/hevnsnt/collider/issues> with:

- Operating system and version.
- GPU model plus driver version.
- CUDA Toolkit version (Linux / Windows) or macOS version (Mac).
- Complete CMake configure log.
- Steps to reproduce.

---

## Where to go next

| For                                     | See                                  |
| --------------------------------------- | ------------------------------------ |
| CLI surface and quick-start examples    | [README.md](../README.md)            |
| `config.yml` schema and precedence      | [CONFIGURATION.md](CONFIGURATION.md) |
| Pool client setup, accrual, etiquette   | [POOL.md](POOL.md)                   |
| Wire format (third-party clients)       | [JLP-PROTOCOL.md](JLP-PROTOCOL.md)   |
| macOS Metal specifics, embedded shaders | [BUILD-MACOS.md](BUILD-MACOS.md)     |
| Release history                         | [CHANGELOG.md](CHANGELOG.md)         |
