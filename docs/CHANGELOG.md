# Changelog

All notable changes to theCollider are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2025-01-06

### Added

#### PCFG (Probabilistic Context-Free Grammar) Training
- Full implementation of password pattern learning from wordlists
- Extracts structure patterns (L=lowercase, U=uppercase, D=digit, S=symbol)
- Calculates probability weights for each pattern
- Generates candidates in probability order (most likely first)
- Saves trained models to `.pcfg` files for reuse

#### WarpWallet/Scrypt Support
- Complete scrypt implementation with Salsa20/8 core and BlockMix
- HMAC-SHA256 and PBKDF2-SHA256 for secondary derivation
- Standard WarpWallet key derivation: `s1 XOR s2`
- Email-as-salt support for WarpWallet format
- `WarpWalletProcessor` class for batch processing

#### Markov Chain Generator
- Character-level Markov chain for password generation
- `TransitionMatrix` class for storing transition probabilities
- `Trainer` class for learning from password corpus
- `Generator` class for probability-ordered enumeration
- `MarkovSource` PassphraseSource implementation
- Save/load functionality for trained models

#### Performance Optimizations
- **Parallel Bloom Filter Loading**: Multi-GPU simultaneous copy
  - ~N-1x speedup for N GPUs
  - Progress reporting during load
- **True Double Buffering**: Overlap CPU/GPU work
  - CPU prepares batch N+1 while GPU processes batch N
  - Up to 2x throughput improvement

#### New Files
- `src/core/warpwallet.hpp` - WarpWallet/Scrypt implementation
- `src/generators/markov.hpp` - Markov chain generator
- `docs/CHANGELOG.md` - This changelog

### Fixed

#### Critical Bug Fixes
- **Mode Selection Bug**: Brainwallet mode incorrectly activated pool mode when pool config existed
  - Root cause: `pool_mode` flag inherited from config and never reset
  - Fix: Explicitly set `pool_mode = false` when brainwallet selected
  
- **MSVC Compilation Error (C2598)**: `extern "C"` linkage inside function body
  - Root cause: C++ requires linkage specifications at global scope
  - Fix: Moved `extern "C"` declarations to file scope

- **Tames Generation**: Kangaroo tames generation returned `false`
  - Root cause: Stub implementation
  - Fix: Full implementation with proper jump table initialization

#### Warnings Fixed
- Removed unused `MAX_WORD_LEN` and `MAX_RULE_LEN` constants (warning #177-D)
- Removed dead L2 cache attribute code (warning #550-D)
- Fixed unsigned char comparison in puzzle_config.hpp

### Changed
- `src/ui/brainwallet_setup.hpp` - Integrated PCFG training UI
- `src/gpu/rckangaroo_wrapper.cu` - Full tames generation implementation
- `src/gpu/brain_wallet_gpu.cpp` - Parallel loading, double buffering, extern "C" fix
- `src/gpu/h160_bloom_filter.cu` - Added `h160_bloom_set_config` function
- `src/generators/pcfg.hpp` - Fixed C++ default member initializer issue

### Build Compatibility
- Verified on Windows 10/11 with Visual Studio 2022 + CUDA 12.9
- Verified on macOS (Apple Silicon) with Metal backend
- Verified on Linux with GCC 11+ and CUDA 12.x

---

## [1.0.0] - 2025-01-04

### Added
- Initial release of theCollider
- RCKangaroo integration (K=1.15 optimal)
- Fused GPU brain wallet pipeline
- Bloom filter checking against ~50M funded addresses
- Hashcat-compatible rule engine (35+ operations)
- JLP pool protocol client
- Interactive mode with setup wizard
- Multi-GPU support with automatic detection
- YAML configuration file support
- macOS Metal backend support

### Core Features
- **Kangaroo Solver**: Pollard's Kangaroo for ECDLP with SOTA method
- **Brain Wallet Scanner**: SHA256 → secp256k1 → SHA256 → RIPEMD160 → Bloom
- **Pool Mode**: Distributed solving via Collision Protocol
- **Opportunistic Scanning**: Check DPs against bloom filter during Kangaroo solving

### Performance
- RTX 4090: 8 GKeys/s (Kangaroo), 1.8B keys/s (Brain Wallet)
- RTX 3090: 4 GKeys/s (Kangaroo), 1.0B keys/s (Brain Wallet)
- Multi-GPU scaling with near-linear performance

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.1.0 | 2025-01-06 | PCFG, WarpWallet, Markov chains, parallel loading |
| 1.0.0 | 2025-01-04 | Initial release |

---

## Upcoming

### Planned for 1.2.0
- [ ] Collision Protocol web dashboard
- [ ] Worker registration portal
- [ ] AMD GPU support via HIP/ROCm
- [ ] CUDA Graphs optimization
- [ ] Neural network passphrase prediction
