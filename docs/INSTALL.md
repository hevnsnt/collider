# Installation Guide

This guide covers building and installing thePuzzler on Linux, Windows, and macOS.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Linux Installation](#linux-installation)
- [Windows Installation](#windows-installation)
- [macOS Installation](#macos-installation)
- [Build Options](#build-options)
- [Post-Installation Setup](#post-installation-setup)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### All Platforms

| Requirement | Version | Notes |
|-------------|---------|-------|
| CMake | 3.20+ | Build system |
| Git | 2.0+ | Source control |
| C++ Compiler | C++20 support | GCC 10+, Clang 12+, MSVC 2019+ |

### GPU Requirements

**NVIDIA (CUDA backend):**
- NVIDIA GPU with Compute Capability 6.0 or higher
- CUDA Toolkit 11.0 or newer (12.x recommended)
- Latest NVIDIA drivers

**Apple Silicon (Metal backend):**
- Apple M1, M2, M3, or newer
- macOS 12.0 (Monterey) or newer
- Xcode Command Line Tools

### Recommended Hardware

| Configuration | GPU | VRAM | System RAM |
|---------------|-----|------|------------|
| Minimum | GTX 1060 | 6 GB | 16 GB |
| Recommended | RTX 3080 | 10 GB | 32 GB |
| Optimal | RTX 4090 | 24 GB | 64 GB |
| Multi-GPU | 4x RTX 4090/5090 | 96 GB+ | 128 GB |

---

## Linux Installation

### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y build-essential cmake git libssl-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install -y gcc-c++ cmake git openssl-devel
```

**Arch Linux:**
```bash
sudo pacman -S base-devel cmake git openssl
```

### Step 2: Install CUDA Toolkit

**Option A: Package Manager (Ubuntu/Debian)**
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA Toolkit
sudo apt install -y cuda-toolkit-12-4
```

**Option B: NVIDIA Website**

1. Visit https://developer.nvidia.com/cuda-downloads
2. Select your Linux distribution
3. Download and run the installer
4. Follow the on-screen instructions

**Verify CUDA Installation:**
```bash
nvcc --version
nvidia-smi
```

Expected output:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.4, V12.4.xxx
```

### Step 3: Clone and Build

```bash
# Clone the repository
git clone https://github.com/yourusername/thepuzzler.git
cd thepuzzler

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build (use all available cores)
make -j$(nproc)

# Verify build
./thepuzzler --version
```

**Expected CMake output:**
```
-- Detected NVIDIA CUDA
-- CUDA Version: 12.4
-- Compiling for CUDA architectures: 60;70;75;80;86;89;90
-- thePuzzler backend: CUDA
-- Configuring done
-- Build files have been written to: .../thepuzzler/build
```

### Step 4: Install (Optional)

```bash
# Install to system (requires sudo)
sudo make install

# Or install to user directory
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make install
```

### Step 5: Run Tests

```bash
# Run the test suite
ctest --output-on-failure

# Run a quick benchmark
./thepuzzler --benchmark
```

---

## Windows Installation

### Step 1: Install Visual Studio

1. Download Visual Studio 2019 or 2022 from https://visualstudio.microsoft.com/
2. Run the installer
3. Select "Desktop development with C++" workload
4. Ensure these components are selected:
   - MSVC v143 (or v142) C++ build tools
   - Windows 10/11 SDK
   - C++ CMake tools for Windows

### Step 2: Install CUDA Toolkit

1. Visit https://developer.nvidia.com/cuda-downloads
2. Select Windows, your version (10 or 11), and exe (local)
3. Download the installer (~3 GB)
4. Run the installer with default options
5. Restart your computer when prompted

**Verify Installation:**

Open "x64 Native Tools Command Prompt for VS 2022" and run:
```cmd
nvcc --version
```

### Step 3: Install Git and CMake

**Option A: Winget (Windows 11)**
```cmd
winget install Git.Git
winget install Kitware.CMake
```

**Option B: Manual Download**
- Git: https://git-scm.com/download/win
- CMake: https://cmake.org/download/

### Step 4: Clone and Build

**Using Developer PowerShell for VS 2022:**

```powershell
# Clone the repository
git clone https://github.com/yourusername/thepuzzler.git
cd thepuzzler

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release --parallel

# Verify build
.\Release\thepuzzler.exe --version
```

**Using Visual Studio IDE:**

1. Open Visual Studio
2. Select "Clone a repository"
3. Enter: `https://github.com/yourusername/thepuzzler.git`
4. Visual Studio will detect CMakeLists.txt automatically
5. Select "Release" configuration from the dropdown
6. Build > Build All (Ctrl+Shift+B)

### Step 5: Run Tests

```powershell
# From build directory
ctest -C Release --output-on-failure

# Quick benchmark
.\Release\thepuzzler.exe --benchmark
```

---

## macOS Installation

### Apple Silicon (M1/M2/M3)

thePuzzler uses Metal for GPU acceleration on Apple Silicon Macs.

### Step 1: Install Xcode Command Line Tools

```bash
xcode-select --install
```

Click "Install" when prompted.

### Step 2: Install Homebrew Dependencies

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake git openssl
```

### Step 3: Clone and Build

```bash
# Clone the repository
git clone https://github.com/yourusername/thepuzzler.git
cd thepuzzler

# Create build directory
mkdir build && cd build

# Configure with CMake (auto-detects Metal)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(sysctl -n hw.ncpu)

# Verify build
./thepuzzler --version
```

**Expected CMake output:**
```
-- Detected Apple Silicon (arm64)
-- Metal framework found
-- thePuzzler backend: METAL
-- Configuring done
-- Build files have been written to: .../thepuzzler/build
```

### Step 4: Run Tests

```bash
ctest --output-on-failure
./thepuzzler --benchmark
```

### Intel Mac with NVIDIA eGPU

If you have an Intel Mac with an external NVIDIA GPU:

1. Install CUDA drivers (may require older macOS versions)
2. Build with CUDA backend:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DSUPERFLAYER_USE_METAL=OFF
```

**Note:** NVIDIA support on macOS is limited to older macOS versions (pre-Catalina) and requires third-party drivers.

---

## Build Options

### CMake Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `SUPERFLAYER_USE_CUDA` | ON | Enable CUDA backend |
| `SUPERFLAYER_USE_METAL` | ON | Enable Metal backend (macOS only) |
| `SUPERFLAYER_USE_CPU` | ON | Enable CPU fallback |
| `SUPERFLAYER_BUILD_TESTS` | ON | Build unit tests |
| `SUPERFLAYER_BUILD_TOOLS` | ON | Build CLI tools |
| `SUPERFLAYER_LTO` | ON | Enable Link-Time Optimization |
| `CMAKE_BUILD_TYPE` | Release | Build type (Release/Debug/RelWithDebInfo) |

### Example: Custom Build

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DSUPERFLAYER_BUILD_TESTS=OFF \
  -DSUPERFLAYER_LTO=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86;89"
```

### CUDA Architecture Selection

By default, thePuzzler compiles for multiple GPU architectures. For faster builds targeting specific hardware:

| GPU Series | Architecture | CMake Setting |
|------------|--------------|---------------|
| GTX 1000 | sm_60, sm_61 | `60;61` |
| RTX 2000 | sm_75 | `75` |
| RTX 3000 | sm_86 | `86` |
| RTX 4000 | sm_89 | `89` |
| RTX 5000 | sm_100 | `100` |

Example for RTX 4090 only:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="89"
```

---

## Post-Installation Setup

### 1. Verify GPU Detection

```bash
./thepuzzler --list-gpus
```

Expected output:
```
Detected GPUs:
  [0] NVIDIA GeForce RTX 4090 (24576 MB, SM 8.9, 128 SMs)
  [1] NVIDIA GeForce RTX 3090 (24576 MB, SM 8.6, 82 SMs)

Total: 2 GPU(s) available
```

### 2. Build a Bloom Filter

Before scanning for brain wallets, you need a bloom filter of target addresses:

```bash
# Download UTXO data (requires Bitcoin Core or utxo-dump)
# See USAGE.md for detailed instructions

# Build bloom filter from address list
./build_bloom -i addresses.csv -o addresses.blf -m 100000

# Or use pre-built filters from community sources
```

### 3. Prepare Wordlists

thePuzzler includes a preprocessing script for wordlists:

```bash
# Process raw wordlist data
python3 scripts/preprocess_data.py --data-dir data --output-dir processed

# Output structure:
# processed/
#   passwords.txt    (15M passwords)
#   phrases.txt      (1.1M phrases)
#   wordlists.txt    (7.4M words)
#   names.txt        (8.2M names)
#   crypto.txt       (19K crypto terms)
```

### 4. Test Your Setup

```bash
# Run a quick brain wallet test
echo "test passphrase" | ./thepuzzler --bloom addresses.blf --wordlist -

# Run Kangaroo benchmark
./thepuzzler --kangaroo --benchmark

# Run full benchmark suite
./thepuzzler --benchmark --all
```

---

## Troubleshooting

### CUDA Not Found

**Symptom:**
```
CMake Error: Could not find CUDA
```

**Solution:**
1. Verify CUDA is installed: `nvcc --version`
2. Add CUDA to PATH:
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
3. Set CUDA_HOME:
   ```bash
   export CUDA_HOME=/usr/local/cuda
   ```

### GPU Not Detected

**Symptom:**
```
No CUDA-capable device detected
```

**Solution:**
1. Verify drivers: `nvidia-smi`
2. Check GPU compatibility (Compute Capability 6.0+)
3. Update NVIDIA drivers to latest version
4. Reboot after driver installation

### Build Fails with Register Pressure

**Symptom:**
```
ptxas error: Entry function uses too much local data
```

**Solution:**

Reduce optimization level or increase register allocation:
```bash
cmake .. -DCMAKE_CUDA_FLAGS="-maxrregcount=128"
```

### Out of Memory During Build

**Symptom:**
```
c++: fatal error: Killed signal terminated program
```

**Solution:**

1. Reduce parallel jobs:
   ```bash
   make -j2  # Instead of -j$(nproc)
   ```

2. Add swap space:
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Metal Backend Issues (macOS)

**Symptom:**
```
Metal framework not found
```

**Solution:**
1. Verify Xcode Command Line Tools: `xcode-select -p`
2. Install if missing: `xcode-select --install`
3. Accept Xcode license: `sudo xcodebuild -license accept`

### Windows: CUDA Path Issues

**Symptom:**
```
'nvcc' is not recognized as an internal or external command
```

**Solution:**
1. Add CUDA to system PATH:
   - Open System Properties > Environment Variables
   - Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin`
2. Restart command prompt/IDE

### Permission Denied on Linux

**Symptom:**
```
CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
```

**Solution:**

Add user to video group:
```bash
sudo usermod -aG video $USER
# Log out and back in
```

---

## Getting Help

If you encounter issues not covered here:

1. Check the GitHub Issues page
2. Search the Bitcoin Talk thread
3. Review NVIDIA CUDA documentation
4. Open a new issue with:
   - Operating system and version
   - GPU model and driver version
   - CUDA version
   - Complete error message
   - Steps to reproduce

---

## Next Steps

After successful installation:

1. Read the **[Usage Guide](USAGE.md)** to learn thePuzzler commands
2. Review **[BITCOIN-PUZZLE-STRATEGY.md](BITCOIN-PUZZLE-STRATEGY.md)** for puzzle solving tactics
3. Build your bloom filter and start scanning

---

*Installation complete. Time to solve some puzzles.*
