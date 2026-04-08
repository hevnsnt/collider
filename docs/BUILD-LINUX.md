# Building collider on Linux

## Prerequisites

### Ubuntu / Debian

```bash
# Build essentials and OpenSSL
sudo apt update
sudo apt install -y build-essential cmake git libssl-dev

# CUDA Toolkit (12.x or newer)
# Download from: https://developer.nvidia.com/cuda-downloads
# Select Linux > x86_64 > your distro > deb (network)
# Follow the install instructions on that page
```

### Fedora / RHEL

```bash
sudo dnf install -y gcc-c++ cmake git openssl-devel
# Install CUDA from: https://developer.nvidia.com/cuda-downloads
# Select Linux > x86_64 > Fedora > rpm (network)
```

### Arch Linux

```bash
sudo pacman -S base-devel cmake git openssl cuda
```

### Verify Prerequisites

```bash
cmake --version    # Should be 3.18+
nvcc --version     # Should be 12.x+
g++ --version      # Should support C++17
```

## Build Steps

```bash
git clone https://github.com/hevnsnt/collider.git
cd collider
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOLLIDER_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

cmake --build . --target collider -- -j$(nproc)
```

The executable will be at: `build/collider`

## CUDA Architecture Guide

Set `-DCMAKE_CUDA_ARCHITECTURES` based on your GPU:

| GPU Family | Architecture |
|------------|-------------|
| GTX 1060/1070/1080 | 61 |
| RTX 2060/2070/2080 | 75 |
| RTX 3060/3070/3080/3090 | 86 |
| RTX 4060/4070/4080/4090 | 89 |
| RTX 5090 | 100 |

You can specify multiple: `-DCMAKE_CUDA_ARCHITECTURES="75;86;89"`

To check your GPU's compute capability:
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

## Troubleshooting

**"nvcc not found"**
Add CUDA to your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```
Add these lines to `~/.bashrc` to make permanent.

**"Could NOT find OpenSSL"**
```bash
sudo apt install libssl-dev    # Debian/Ubuntu
sudo dnf install openssl-devel # Fedora/RHEL
```

**"unsupported GNU version" from nvcc**
Your GCC may be too new for your CUDA version. Install an older GCC:
```bash
sudo apt install gcc-12 g++-12
cmake .. -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 ...
```

**Build uses too much memory**
Reduce parallel jobs: `make -j4` instead of `make -j$(nproc)`

## Running

```bash
./collider --worker bc1qYourBitcoinAddress
```

Or for solo mode:
```bash
./collider --puzzle 135
```

To run in the background:
```bash
nohup ./collider --worker bc1qYourAddress > collider.log 2>&1 &
```
