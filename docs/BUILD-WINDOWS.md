# Building collider on Windows

## Prerequisites

1. **Visual Studio 2022** (Community, Pro, or Build Tools)
   - Install the "Desktop development with C++" workload
   - Download: https://visualstudio.microsoft.com/downloads/

2. **CUDA Toolkit 12.x or newer**
   - Download: https://developer.nvidia.com/cuda-downloads
   - Select Windows > x86_64 > your Windows version
   - After install, verify: `nvcc --version`

3. **CMake 3.18+** (usually included with Visual Studio)
   - Verify: `cmake --version`

4. **vcpkg** (optional, for OpenSSL -- Visual Studio 2022 includes it)
   - If not bundled: https://github.com/microsoft/vcpkg

## Build Steps

### Option A: Command Line (Recommended)

Open **x64 Native Tools Command Prompt for VS 2022** (search for it in Start menu).

```cmd
git clone https://github.com/hevnsnt/collider.git
cd collider
mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCOLLIDER_USE_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES="75;86;89"

cmake --build . --config Release --target collider
```

The executable will be at: `build\Release\collider.exe`

### Option B: Using Ninja (Faster Builds)

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
vcpkg should handle this automatically. If not:
```cmd
vcpkg install openssl:x64-windows
```
Then add `-DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake` to your cmake command.

**"CUDA_ARCHITECTURES" error**
Your CMake version may be too old. Update to CMake 3.18+.

## Running

```cmd
cd build\Release
collider.exe --worker bc1qYourBitcoinAddress
```

Or for solo mode:
```cmd
collider.exe --puzzle 135
```
