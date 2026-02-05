# collider

A GPU-accelerated solver for the Bitcoin Puzzle Challenge, implementing Pollard's Kangaroo algorithm on secp256k1.

```
$ ./collider --worker bc1qYourAddress
[*] GPU: NVIDIA GeForce RTX 4090
[*] Backend: CUDA 12.4
[*] Connecting to pool: collisionprotocol.com:17403

Pool: 47.2% | Speed: 8.14 GKeys/s | DPs: 1.2M | Sent: 847K | ETA: ~2.3 years
```

## The Bitcoin Puzzle

In 2015, someone created 256 Bitcoin addresses with private keys of increasing difficulty. The first 70 have been solved. Puzzle #135 holds **13.5 BTC** (~$1.3M at current prices) and requires finding a 135-bit private key.

The math: 135-bit keyspace = 2^135 possibilities. Brute force at 10 billion keys/second would take longer than the age of the universe.

**Pollard's Kangaroo** reduces this to O(√n) operations. Instead of 2^135 operations, we need ~2^67.5. Still astronomical for one GPU, but tractable for a distributed pool.

That's where collider comes in.

## How It Works

The Kangaroo algorithm works by launching two types of "kangaroos" that jump around the keyspace:

1. **Tame kangaroos** start from a known point and record their path
2. **Wild kangaroos** start from the target public key

When a wild kangaroo lands on a tame kangaroo's path, we can compute the private key from the collision. The trick is using "Distinguished Points" (DPs), points with special properties that we actually store and compare.

Our implementation achieves **K=1.15** efficiency, meaning we solve puzzles in approximately 1.15x the theoretical minimum operations. Most public implementations run at K=1.6 or worse.

## Features

- **CUDA acceleration** for NVIDIA GPUs (Compute 6.0+)
- **Metal backend** for Apple Silicon
- **Multi-GPU support** with linear scaling
- **Pool mining** via the JLP protocol
- **Solo mode** via command line flags
- **Benchmark mode** to test your hardware

## Building

### Requirements

| Component | Version |
|-----------|---------|
| CUDA Toolkit | 12.0+ |
| CMake | 3.18+ |
| C++ Compiler | C++17 support |
| OpenSSL | 1.1+ (for TLS) |

### Linux

```bash
git clone https://github.com/hevnsnt/collider.git
cd collider
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Windows

Prerequisites:
- Visual Studio 2022 with C++ workload
- CUDA Toolkit 12.x
- vcpkg for dependencies

```powershell
git clone https://github.com/hevnsnt/collider.git
cd collider

# Install dependencies via vcpkg
vcpkg install openssl:x64-windows

# Configure and build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake
cmake --build . --config Release
```

Or use the provided batch file:
```powershell
.\build.bat
```

### macOS (Apple Silicon)

```bash
git clone https://github.com/hevnsnt/collider.git
cd collider
./build_macos.sh
```

The Metal backend is used automatically on Apple Silicon. Performance is roughly 1/10th of equivalent NVIDIA hardware due to Metal's less mature compute ecosystem, but it works.

## Usage

### Pool Mining (Recommended)

Solo solving Puzzle #135 with a single RTX 4090 would take approximately 195 years. Pool mining distributes the work across many GPUs.

```bash
# Basic pool connection
./collider --worker bc1qYourBitcoinAddress

# Or with explicit pool URL
./collider --pool collisionprotocol.com:17403 --worker bc1qYourBitcoinAddress
```

Your Bitcoin address is your identity and payout destination. When the puzzle is solved, rewards are distributed proportionally based on Distinguished Points contributed.

**Pool fee:** 5% (covers infrastructure, development, and coordination)

### Solo Mode

If you want to mine directly without the pool:

```bash
# Target a specific puzzle
./collider --puzzle 135

# With Kangaroo algorithm (required for puzzles with known public keys)
./collider --puzzle 135 --kangaroo
```

Note: Solo mode is technically possible but economically impractical for high-bit puzzles.

### Benchmarking

```bash
./collider --benchmark
```

Expected performance (Kangaroo mode):

| GPU | Keys/Second |
|-----|-------------|
| RTX 5090 | ~12 GKeys/s |
| RTX 4090 | ~8 GKeys/s |
| RTX 3090 | ~4 GKeys/s |
| RTX 3060 | ~1.5 GKeys/s |
| Apple M2 | ~400 MKeys/s |

### Configuration File

Create `config.yml` for persistent settings:

```yaml
pool:
  url: "collisionprotocol.com:17403"
  worker: "bc1qYourBitcoinAddress"

gpu:
  devices: []  # Empty = use all available GPUs
```

```bash
./collider --config config.yml
```

## Command Reference

```
Usage: collider [options]

Pool Options:
  --worker, -w <address>    Your Bitcoin address for rewards
  --pool <host:port>        Pool server (default: collisionprotocol.com:17403)

Solo Options:
  --puzzle, -P <number>     Target puzzle number
  --kangaroo                Use Kangaroo algorithm
  --dp-bits <n>             Distinguished point bits (default: auto)

GPU Options:
  --gpus, -g <ids>          GPU device IDs (default: all)

Other:
  --benchmark               Run performance benchmark
  --config, -c <file>       Load configuration file
  --verbose, -v             Verbose output
  --help, -h                Show help
```

## Security Considerations

**This software is designed for legitimate puzzle solving and security research.**

The Bitcoin Puzzle addresses have no legitimate owner claiming them. They were created specifically as a cryptographic challenge. The same cannot be said for other Bitcoin addresses.

**Do not use this tool to:**
- Attempt to crack wallets you don't own
- Target addresses with known owners
- Engage in any form of theft

Using cryptographic tools against systems without authorization is illegal in most jurisdictions.

**Operational security:**
- The pool connection uses TLS encryption
- Your private keys are never transmitted, only Distinguished Points
- The pool cannot solve puzzles without contributors; it's a coordination mechanism, not a key escrow

## Architecture

```
src/
├── main.cpp              # Entry point and CLI handling
├── core/                 # Core types, config, puzzle database
├── gpu/                  # CUDA kernels and GPU management
│   ├── kangaroo_*.cu     # Kangaroo algorithm implementation
│   └── rckangaroo/       # RCKangaroo backend (K=1.15)
├── pool/                 # Pool client (JLP protocol)
├── platform/             # Platform abstraction (CUDA/Metal)
└── ui/                   # Terminal UI and progress display
```

The hot path is the GPU kernel that performs elliptic curve point additions. We use projective coordinates and batch inversions to minimize expensive modular inversions.

## Protocol

collider implements the JLP (Jean-Luc Pons) protocol for pool communication:

- Binary protocol over TCP with optional TLS
- 8-byte header: `KANG` magic + type + flags + length
- Message types: AUTH, WORK_REQ, WORK_ASN, DP_SUBMIT, DP_BATCH

Full protocol specification: [collisionprotocol.com/protocol](https://collisionprotocol.com/protocol)

## Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss.

Areas where contributions would be particularly valuable:
- OpenCL backend for AMD GPUs
- Performance optimizations for specific GPU architectures
- Additional test coverage

## Pro Version

Need more features? [collider pro](https://collisionprotocol.com/pro) includes:

- **Opportunistic brainwallet**: Every key generated during pool mining is checked against 50M+ funded addresses for free bonus hits
- **Dedicated brain wallet cracker**: Targeted passphrase cracking mode
- **Bloom filter**: 50M+ funded address database included
- **PCFG generation**: Learn password patterns from wordlists
- **Markov chains**: Character-level probability models
- **Rule engine**: Transform candidates (leetspeak, case mutations, etc.)
- **Interactive menu**: Auto-configuration and guided setup

One-time purchase: $49.99 per major version.

## Links

- Website: [collisionprotocol.com](https://collisionprotocol.com)
- Pool stats: [collisionprotocol.com/pool](https://collisionprotocol.com/pool)
- Protocol spec: [collisionprotocol.com/protocol](https://collisionprotocol.com/protocol)
- Pro version: [collisionprotocol.com/pro](https://collisionprotocol.com/pro)

## Acknowledgments

- [RetiredCoder](https://github.com/RetiredCoder): RCKangaroo implementation
- [Jean-Luc Pons](https://github.com/JeanLucPons): Original Kangaroo GPU work
- bitcoin-core/secp256k1: Reference implementation

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*"The puzzle is the prize."*
