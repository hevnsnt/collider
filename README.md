# collider

**GPU-accelerated Bitcoin Puzzle solver using Pollard's Kangaroo algorithm.**

> K=1.15 efficiency — 40-80% fewer operations than competitors.

## What is this?

`collider` is a free, open-source GPU solver for the [1000 BTC Bitcoin Puzzle Challenge](https://collisionprotocol.com). It uses Pollard's Kangaroo algorithm on the secp256k1 elliptic curve to hunt for private keys via Distinguished Points.

Join the **Collision Protocol pool** and contribute your GPU power. When a puzzle is solved, rewards are split proportionally based on your Distinguished Point contributions.

## Features

- 🚀 **CUDA GPU acceleration** — RTX 4090: ~8 GKeys/s, RTX 3090: ~4 GKeys/s
- ⚡ **K=1.15 efficiency** — State-of-the-art Kangaroo implementation
- 🖥️ **Multi-GPU support** — Linear scaling across GPUs
- 🌐 **Pool mining** — Connect to any JLP-compatible pool
- 📊 **Benchmark mode** — Test your GPU performance
- 🍎 **Cross-platform** — Linux, Windows, macOS (CUDA + Metal)
- 📝 **YAML config** — Flexible configuration

## Quick Start

```bash
# Build
git clone https://github.com/hevnsnt/collider.git
cd collider
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Join the pool
./collider --pool collisionprotocol.com:17403 --worker YourBTCAddress

# Benchmark
./collider --benchmark
```

## Requirements

- **NVIDIA GPU** with CUDA 12.0+ (or Apple Silicon for Metal)
- CMake 3.18+
- C++17 compiler
- CUDA Toolkit 12.0+

## Pool Mining

Connect to the Collision Protocol pool:

```bash
./collider --pool collisionprotocol.com:17403 --worker <your-btc-address>
```

Your share of the reward = (Your DPs / Total DPs) × Prize BTC. Pool fee: 5%.

## Configuration

Create a `config.yaml`:

```yaml
pool:
  url: collisionprotocol.com:17403
  worker: YourBTCAddress

gpu:
  device: 0          # GPU index (or "all")
  grid_size: 0       # 0 = auto
  block_size: 256
```

Run with config: `./collider --config config.yaml`

## Pro Version

Need solo puzzle solving, brain wallet cracking, bloom filters, PCFG, or rule engines?

Check out [**collider pro**](https://collisionprotocol.com/pro) — $49.99 per major version.

## Links

- 🌐 [collisionprotocol.com](https://collisionprotocol.com)
- 📖 [Documentation](https://collisionprotocol.com/docs)
- 📊 [Pool Stats](https://collisionprotocol.com/pool)
- 🛒 [Get Pro](https://collisionprotocol.com/pro)

## License

MIT — see [LICENSE](LICENSE).

## Contributing

PRs welcome! Please open an issue first for major changes.
