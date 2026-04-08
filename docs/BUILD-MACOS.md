# Building collider on macOS

## Important Notes

- collider uses **Metal** on macOS (not CUDA)
- Only **Apple Silicon** (M1/M2/M3/M4) Macs are supported
- Performance is roughly 1/10th of equivalent NVIDIA hardware due to Metal's compute limitations
- Intel Macs are not supported (no Metal compute, no CUDA)

## Prerequisites

```bash
# Install Xcode Command Line Tools (includes clang/clang++)
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install CMake and OpenSSL
brew install cmake openssl
```

### Verify Prerequisites

```bash
cmake --version     # Should be 3.18+
clang++ --version   # Should support C++17
brew --prefix openssl  # Should return a path
```

## Build Steps

```bash
git clone https://github.com/hevnsnt/collider.git
cd collider
mkdir build && cd build

export OPENSSL_ROOT_DIR=$(brew --prefix openssl)

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOLLIDER_USE_CUDA=OFF \
  -DCOLLIDER_USE_METAL=ON \
  -DOPENSSL_ROOT_DIR=$OPENSSL_ROOT_DIR

make -j$(sysctl -n hw.ncpu)
```

The executable will be at: `build/collider`

## Troubleshooting

**"Could NOT find OpenSSL"**
Homebrew installs OpenSSL in a non-standard location. Set the path explicitly:
```bash
export OPENSSL_ROOT_DIR=$(brew --prefix openssl)
```
Then re-run the cmake command.

**"Metal.framework not found"**
Make sure you have Xcode Command Line Tools installed:
```bash
xcode-select --install
```

**"Unsupported architecture" or build fails on Intel Mac**
Intel Macs are not supported. The Metal compute backend requires Apple Silicon.

## Expected Performance

| Chip | Approximate Speed |
|------|-------------------|
| M1 | ~200 MKeys/s |
| M2 | ~400 MKeys/s |
| M3 | ~500 MKeys/s |
| M4 | ~600 MKeys/s |
| M1 Max/Ultra | ~400-800 MKeys/s |

For comparison, an RTX 4090 does ~8 GKeys/s. Pool mining is still worthwhile on Apple Silicon, but the contribution per GPU is lower.

## Running

```bash
./collider --worker bc1qYourBitcoinAddress
```

Or for solo mode:
```bash
./collider --puzzle 135
```
