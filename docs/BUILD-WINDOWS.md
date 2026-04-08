# Building collider on Windows

## Prerequisites

1. **Visual Studio 2022** (Community, Pro, or Build Tools)
   - Install the "Desktop development with C++" workload
   - Download: https://visualstudio.microsoft.com/downloads/

2. **CUDA Toolkit 12.x or newer**
   - Download: https://developer.nvidia.com/cuda-downloads
   - Select Windows > x86_64 > your Windows version
   - After install, verify: `nvcc --version`

3. **CMake 3.18+** (included with Visual Studio)
   - Verify: `cmake --version`

## Build Steps

Open **x64 Native Tools Command Prompt for VS 2022** (search for it in Start menu). All commands must be run from this prompt, not a regular cmd or PowerShell.

```cmd
git clone https://github.com/hevnsnt/collider.git
cd collider
mkdir build
cd build

cmake .. -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCOLLIDER_USE_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

ninja collider
```

The executable will be at: `build\collider.exe`

OpenSSL DLLs are automatically copied next to the executable after build.

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

## Troubleshooting

**"cl.exe not found"**
You need to run from the VS Developer Command Prompt, not a regular cmd/PowerShell. Search for "x64 Native Tools Command Prompt for VS 2022" in the Start menu.

**"nvcc not found"**
CUDA Toolkit isn't in PATH. Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` to your PATH, or reinstall CUDA with "Add to PATH" checked.

**OpenSSL not found**
CMake should find OpenSSL automatically via vcpkg. If it fails, install OpenSSL manually:
```cmd
vcpkg install openssl:x64-windows
```
Then add `-DOPENSSL_ROOT_DIR="C:/vcpkg/installed/x64-windows"` to your cmake command.

**vcpkg errors ("Unable to find Visual Studio instance")**
The bundled vcpkg may be outdated. Disable it and point to an existing OpenSSL:
```cmd
cmake .. -G Ninja -DVCPKG_MANIFEST_MODE=OFF -DOPENSSL_ROOT_DIR="path/to/openssl" ...
```

## Running

```cmd
collider.exe --worker bc1qYourBitcoinAddress
```

Or for solo mode:
```cmd
collider.exe --puzzle 135
```
