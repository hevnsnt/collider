# Superflayer Research Analysis

## Executive Summary

This document analyzes five key projects and one research paper to inform the development of Superflayer, a GPU-accelerated brainflayer port targeting 4x RTX 5090 GPUs. The goal is to achieve 10B+ keys/sec throughput by combining the best architectural patterns from existing implementations with novel optimizations from recent research.

---

## 1. Brainflayer (Original Implementation)

**Repository:** https://github.com/ryancdotorg/brainflayer
**Language:** C
**Platform:** Linux x86_64 only

### Architecture Overview

Brainflayer is a CPU-based proof-of-concept cracker for cryptocurrency brainwallets. It follows Unix philosophy with a modular pipeline design:

```
Input (stdin/file) -> Hash Generation -> EC Point Multiplication -> Address Generation -> Bloom Filter Lookup
```

### Key Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| `brainflayer.c` | Main execution engine | Single-threaded, stdin processing |
| `ec_pubkey_fast.c` | Accelerated EC operations | Precomputed multiplication tables |
| `bloom.c/h` | Space-efficient lookups | Memory-mapped bloom filters |
| `mmapf.c/h` | Memory mapping | Shared memory across processes |
| `hex2blf` | Bloom filter generator | Preprocesses target addresses |
| `ecmtabgen` | EC table generator | Precomputes multiplication tables |

### Hash Generation Methods

| Method | Flag | Use Case | Performance |
|--------|------|----------|-------------|
| SHA256 | `-t sha256` | Standard Bitcoin brainwallet | Fast |
| Keccak256 | `-t keccak` | Ethereum passphrases | Fast |
| WarpWallet | `-t warp` | Salted scrypt KDF | Slow |
| BrainV2 | `-t brainv2` | Scrypt-based | Very slow |
| Rushwallet | `-t rush` | URL fragment scheme | Medium |
| Raw privkey | `-t priv` | Direct key input | Fastest |

### Performance Characteristics

- **Single-threaded design**: Requires external workload distribution via `-n` and `-k` options
- **Hyperthreading benefits**: Significant gains from parallel process execution
- **EC optimization**: 4x faster than DEFCON release via precomputed tables
- **Bloom filter efficiency**: Millions of hash160 lookups per second

### Key Insights for Superflayer

1. **Modular pipeline**: Separate passphrase input from cryptographic operations
2. **Precomputed tables**: Essential for EC acceleration
3. **Memory-mapped data**: Enables efficient shared memory across processes/GPUs
4. **Bloom filters**: Space-efficient target matching (vs. full hash table)

### Limitations to Address

- Single-threaded CPU bottleneck
- Platform-specific memory optimizations
- No GPU acceleration
- Sequential key derivation inefficient for massive parallelism

---

## 2. CudaBrainSecp

**Repository:** https://github.com/XopMC/CudaBrainSecp
**Language:** C++/CUDA
**Focus:** Brain wallet recovery with full point multiplication

### CUDA Kernel Architecture

CudaBrainSecp implements a unique approach where each GPU thread performs **full elliptic curve point multiplication** rather than incremental key derivation:

```cuda
// Kernel entry point
CudaRunSecp256k1()
  └── _PointMultiSecp256k1()  // Full EC multiplication per thread
       └── GTable lookups      // 16 chunks x 64 bytes = 1024 bytes/key
```

### Precomputed Point Table Strategy

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Chunks | 16 | Divide 256-bit scalar into 16x16-bit segments |
| Points per chunk | 65,536 | 2^16 precomputed points |
| Total points | 1,048,576 | Complete multiplication table |
| Memory footprint | ~67 MB | GTable resident in global memory |

**Algorithm:**
1. Split private key into 16 chunks of 2 bytes each
2. Look up precomputed point for each chunk
3. Add all 16 points together
4. Perform single modular inverse

### Operational Modes

**ModeBooks (Recommended for wordlists):**
```
Small prime wordlist + Large affix wordlist
Each thread: one affix word x all prime words
Benefits: GPU Global Memory Coalescing
```

**ModeCombo (Permutation-based):**
```
Thread loads starting state by thread ID
Generates permutations internally
Less memory-efficient but flexible
```

### Performance Analysis

| GPU | Performance | Notes |
|-----|-------------|-------|
| RTX 3060 | Faster (CC 8.6) | Better than 2070 |
| RTX 2070 | Baseline (CC 7.5) | Reference |
| vs Hashcat | 20-30x faster | Due to precomputed tables |

**Bottlenecks identified:**
- Register pressure from EC arithmetic
- Non-coalesced GTable access (random parts of table)
- Global memory bandwidth for table lookups

### Key Insights for Superflayer

1. **Full multiplication per thread**: Enables independent key processing
2. **Precomputed tables**: Trade memory for compute efficiency
3. **Wordlist optimization**: ModeBooks pattern for passphrase processing
4. **Register pressure**: Major concern for kernel optimization

### Limitations

- Excels only for non-derivable keys (brain wallets, hashed seeds)
- Not suitable for sequential key spaces (Bitcoin Puzzle)
- Memory coalescing issues with table lookups

---

## 3. BitCrack

**Repository:** https://github.com/brichard19/BitCrack
**Language:** C++/CUDA/OpenCL
**Focus:** Bitcoin private key brute-force search

### Architecture

BitCrack demonstrates production-quality multi-GPU implementation with modular design:

```
┌─────────────────────────────────────────────────────────────┐
│                      BitCrack Architecture                   │
├─────────────────────────────────────────────────────────────┤
│  CmdParse    │  Logger     │  AddrGen    │  AddressUtil    │
├─────────────────────────────────────────────────────────────┤
│             KeyFinderLib (Abstract Interface)                │
├─────────────────────────────────────────────────────────────┤
│  CudaKeySearchDevice    │    CLKeySearchDevice              │
├─────────────────────────────────────────────────────────────┤
│        secp256k1lib     │      CryptoUtil                   │
└─────────────────────────────────────────────────────────────┘
```

### Multi-GPU Work Distribution

**Strategy 1: Keyspace Sharing (`--share M/N`)**
```
GPU 0: Keys 0 to N/4
GPU 1: Keys N/4 to N/2
GPU 2: Keys N/2 to 3N/4
GPU 3: Keys 3N/4 to N
```

**Strategy 2: Stride-based (`--stride NUMBER`)**
```
GPU 0: Keys 0, 4, 8, 12...
GPU 1: Keys 1, 5, 9, 13...
GPU 2: Keys 2, 6, 10, 14...
GPU 3: Keys 3, 7, 11, 15...
```

**Strategy 3: Progress Persistence (`--continue FILE`)**
- Checkpoint-based resume
- Enables distributed multi-machine searches
- Fault-tolerant execution

### CUDA Kernel Design

| Parameter | Default | Purpose |
|-----------|---------|---------|
| Blocks | 32 | Match compute units |
| Threads/Block | 256 | Multiple of warp size (32) |
| Keys/Thread | 256 | Amortize kernel overhead |

**Total keys per kernel launch:** 32 x 256 x 256 = 2,097,152

### Memory Hierarchy Optimization

| Memory Type | Usage | Access Pattern |
|-------------|-------|----------------|
| Constant | Target addresses | Read-only, cached |
| Shared | Per-block EC workspace | High bandwidth |
| Global | Input/output, key ranges | Minimized |
| Registers | EC arithmetic | Maximized |

### Performance

- **GeForce GT 640**: ~10.33 MKey/s (reference)
- **RTX 3090**: ~1 GKey/s (modern benchmark)
- **Scaling**: Near-linear with GPU count

### Key Insights for Superflayer

1. **Modular design**: Separate platform-specific code from algorithms
2. **Multiple GPU strategies**: Different approaches for different use cases
3. **Keys per thread**: Amortize overhead via batching
4. **Memory hierarchy**: Strategic placement of data
5. **Progress persistence**: Essential for long-running searches

---

## 4. Secp256k1-CUDA-ecc

**Repository:** https://github.com/8891689/Secp256k1-CUDA-ecc
**Language:** C++/CUDA
**Focus:** High-performance batch EC operations

### Montgomery Ladder Implementation

The Montgomery ladder algorithm provides constant-time scalar multiplication, essential for security and consistent performance:

```cuda
kernel_montgomery_ladder_batch_optimized<<<blocks, threads>>>(
    scalars,       // Array of private keys
    base_point,    // Generator point G
    results,       // Output public keys
    count          // Batch size
);
```

**Algorithm (per thread):**
```
R0 = O (point at infinity)
R1 = G (base point)
for i = 255 downto 0:
    if bit[i] == 0:
        R1 = R0 + R1
        R0 = R0 + R0  // double
    else:
        R0 = R0 + R1
        R1 = R1 + R1  // double
return R0
```

### Batch Processing Architecture

```
Input: N private keys
┌─────────────────────────────────────────┐
│         CUDA Kernel Dispatch            │
├─────────────────────────────────────────┤
│  Thread 0: key[0] → Montgomery ladder   │
│  Thread 1: key[1] → Montgomery ladder   │
│  ...                                    │
│  Thread N-1: key[N-1] → Montgomery      │
├─────────────────────────────────────────┤
│         Batch Modular Inverse           │
│    (Montgomery's Trick: 1 inv + 3N mul) │
├─────────────────────────────────────────┤
│           Output: N public keys         │
└─────────────────────────────────────────┘
```

### Key Optimizations

1. **Montgomery arithmetic**: Efficient modular multiplication without division
2. **Batch inversion**: Single expensive inversion amortized across batch
3. **Constant-time execution**: No timing side channels
4. **Coalesced memory access**: Aligned data structures

### Key Insights for Superflayer

1. **Montgomery ladder**: Standard for secure, fast EC multiplication
2. **Batch inversion**: Critical optimization (see gECC paper)
3. **Memory alignment**: Essential for coalesced access
4. **Constant-time**: Important for consistent GPU utilization

---

## 5. gECC Research Paper

**Paper:** "gECC: A GPU-based high-throughput framework for Elliptic Curve Cryptography"
**Source:** https://arxiv.org/html/2501.03245v1
**Claims:** 4-5x faster than prior GPU ECC implementations

### Novel Optimization Techniques

#### 1. Batch EC Operations with Montgomery's Trick

Convert N modular inversions to 1 inversion + 3N multiplications:

```
Standard: N inversions @ 256 multiplications each = 256N muls
Optimized: 1 inversion + 3N muls = 3N + 256 muls

For N=1,000,000:
Standard: 256,000,000 equivalent operations
Optimized: 3,000,256 equivalent operations
Speedup: ~85x for inversion-heavy workloads
```

#### 2. Gather-Apply-Scatter (GAS) Mechanism

**Problem:** 32n threads performing independent inversions causes warp divergence.

**Solution:** Reduce inversion operations from 32n to sp (streaming processors):
```
Phase 1 (Gather):    Collect operands from threads to SP-local storage
Phase 2 (Apply):     Single SP performs batched inversion
Phase 3 (Scatter):   Distribute results back to threads
```

**Benefit:** Eliminates warp divergence in inversion, reduces by 32x.

#### 3. Data-Locality-Aware Kernel Fusion

**"Find-then-recompute" method:**
```
Pass 1: Compute, discard intermediate results (save memory)
Pass 2: Recompute only for final output (when needed)
```

**Trade-off:** Slight compute overhead vs. significant memory savings, improving cache efficiency.

#### 4. IMAD Replacement Strategy

**Observation:** IMAD (Integer Multiply-Add) has 4-cycle issue interval.

**Optimization:**
- Replace expensive IMAD with predicate registers for carry propagation
- Use IADD3 (2-cycle issue interval) instead of IMAD where possible
- Leverage instruction-level parallelism

```
Standard: IMAD r0, r1, r2, r3  // 4 cycles
Optimized: IADD3 r0, r1, r2, r3 + predicate carry  // 2 cycles
```

### Performance Results

| Operation | Speedup vs Prior Work |
|-----------|----------------------|
| ECDSA Verification | 5.56x |
| ECDSA Signing | 4.18x |
| Point Multiplication (unknown) | 4.94x |
| Point Multiplication (fixed) | 4.04x |
| SM2 Modular Arithmetic | 1.72x |

### GPU Architecture Considerations (A100)

| Component | Specification | Usage |
|-----------|--------------|-------|
| Streaming Processors | 432 | Batch processing units |
| Shared Memory + L1 | 20 MB combined | Intermediate storage |
| L2 Persistent Cache | 40 MB (75% configurable) | Point table caching |
| Memory Layout | Column-major | Coalesced warp access |

### Key Insights for Superflayer

1. **Montgomery's Trick**: Essential for batch key generation
2. **GAS Mechanism**: Adapt for RTX 5090 SM architecture
3. **Kernel Fusion**: Trade memory for compute efficiency
4. **Instruction optimization**: Profile and optimize PTX code
5. **Cache management**: Leverage Blackwell's large L2

---

## 6. RTX 5090 Blackwell Architecture

### Hardware Specifications

| Specification | RTX 5090 | RTX 4090 | RTX 3090 | Improvement |
|---------------|----------|----------|----------|-------------|
| CUDA Cores | 21,760 | 16,384 | 10,496 | +33% vs 4090 |
| SMs | 170 | 128 | 82 | +33% vs 4090 |
| Base Clock | 2.01 GHz | 2.23 GHz | 1.40 GHz | - |
| Boost Clock | 2.41 GHz | 2.52 GHz | 1.70 GHz | - |
| VRAM | 32 GB GDDR7 | 24 GB GDDR6X | 24 GB GDDR6X | +33% |
| Memory Bandwidth | 1,792 GB/s | 1,008 GB/s | 936 GB/s | +78% |
| Memory Interface | 512-bit | 384-bit | 384-bit | +33% |
| PCIe | 5.0 x16 | 4.0 x16 | 4.0 x16 | 2x |
| TDP | 575W | 450W | 350W | +28% |

### Blackwell SM Architecture Changes

| Feature | Blackwell | Hopper | Impact |
|---------|-----------|--------|--------|
| Shared/L1 Memory per SM | 128 KB | 256 KB | Reduced local storage |
| L2 Cache (Total) | 65 MB monolithic | 50 MB (2x25 MB) | Larger unified cache |
| Distributed Shared Memory | Enhanced | Basic | Better cross-SM communication |
| Tensor Cores | 5th gen | 4th gen | FP4 support |

### Optimization Implications for Superflayer

**L2 Cache Strategy:**
```
65 MB L2 cache / 4 GPUs = 65 MB per GPU
Precomputed EC table: ~67 MB (CudaBrainSecp approach)

Strategy: Fit critical portion of EC table in L2 persistent cache
Configure: 75% L2 persistence = 48.75 MB per GPU
Result: Hot path table lookups hit L2, not global memory
```

**Shared Memory Reduction:**
```
Hopper: 256 KB/SM for working set
Blackwell: 128 KB/SM

Mitigation:
1. Use registers more aggressively (high register count per thread)
2. Leverage L2 cache for spillover
3. Reduce threads/block to increase per-thread shared memory
```

**Memory Bandwidth:**
```
1,792 GB/s theoretical bandwidth
For 4x GPUs: 7,168 GB/s aggregate

Key generation bottleneck analysis:
- EC point multiplication: ~1000 multiplications/key
- Each multiplication: ~256-bit operands
- Memory-to-compute ratio favors compute-bound kernels
```

### Multi-GPU Considerations

| Feature | Benefit for Superflayer |
|---------|------------------------|
| PCIe 5.0 x16 | 64 GB/s per GPU bidirectional |
| Peer-to-peer | Direct GPU-to-GPU memory access |
| NVLink (if available) | 900 GB/s inter-GPU bandwidth |

---

## Synthesis: Key Findings

### Performance Baseline

| Implementation | Hardware | Performance | Keys/Second |
|----------------|----------|-------------|-------------|
| Brainflayer | Modern CPU | Baseline | ~1M/sec |
| CudaBrainSecp | RTX 3060 | 20-30x Hashcat | ~50M/sec (est) |
| BitCrack | RTX 3090 | Optimized | ~1B/sec |
| gECC | A100 | 4-5x prior | ~4B/sec (est) |

### Critical Optimization Techniques

1. **Precomputed EC Tables**: 20-30x speedup (CudaBrainSecp)
2. **Montgomery's Trick for Batch Inversion**: 85x for N=1M (gECC)
3. **Keys per Thread Batching**: Amortize overhead (BitCrack)
4. **Memory Hierarchy Optimization**: L2 cache persistence (Blackwell)
5. **Instruction-level Optimization**: IMAD replacement (gECC)

### Architectural Patterns to Adopt

```
┌─────────────────────────────────────────────────────────────────┐
│                    Superflayer Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│  Input Layer                                                     │
│  ├── Wordlist streaming (from brainflayer)                      │
│  └── Passphrase batching (from CudaBrainSecp ModeBooks)         │
├─────────────────────────────────────────────────────────────────┤
│  Hash Generation Layer                                           │
│  ├── SHA256/Keccak256 parallel hashing                          │
│  └── Configurable hash modes (from brainflayer)                 │
├─────────────────────────────────────────────────────────────────┤
│  EC Multiplication Layer                                         │
│  ├── Precomputed point tables (from CudaBrainSecp)              │
│  ├── Montgomery ladder (from Secp256k1-CUDA-ecc)                │
│  └── Batch inversion with GAS (from gECC)                       │
├─────────────────────────────────────────────────────────────────┤
│  Address Generation Layer                                        │
│  └── RIPEMD160(SHA256(pubkey)) or Keccak (Ethereum)             │
├─────────────────────────────────────────────────────────────────┤
│  Lookup Layer                                                    │
│  ├── GPU-resident bloom filter                                  │
│  └── Batch match verification                                   │
├─────────────────────────────────────────────────────────────────┤
│  Multi-GPU Coordination                                          │
│  ├── Work distribution (from BitCrack)                          │
│  ├── Progress persistence                                        │
│  └── Dynamic load balancing                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Register pressure | High | Medium | Careful kernel tuning, reduce threads/block |
| Memory coalescing issues | Medium | High | Align data structures, column-major layout |
| Multi-GPU synchronization overhead | Medium | Medium | Minimize cross-GPU communication |
| Blackwell shared memory reduction | Certain | Medium | Leverage L2 cache, adjust algorithms |
| Wordlist I/O bottleneck | Low | High | Async streaming, prefetching |

---

## Conclusion

Superflayer should combine:

1. **Brainflayer's** modular pipeline and passphrase handling
2. **CudaBrainSecp's** precomputed EC tables and wordlist modes
3. **BitCrack's** multi-GPU architecture and work distribution
4. **Secp256k1-CUDA-ecc's** Montgomery ladder implementation
5. **gECC's** batch inversion and instruction-level optimizations

Target performance of 10B+ keys/sec across 4x RTX 5090 is achievable given:
- RTX 3090 achieves ~1B keys/sec
- RTX 5090 has ~2x compute capacity
- 4 GPUs provide ~4x parallelism
- gECC optimizations provide 4-5x improvement

**Conservative estimate:** 4 x 2 x 1B = 8B keys/sec
**Optimistic estimate:** 4 x 2 x 1B x 1.5 (optimizations) = 12B keys/sec
