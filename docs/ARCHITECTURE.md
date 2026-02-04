# Superflayer System Architecture

## Overview

Superflayer is a GPU-accelerated cryptocurrency brainwallet recovery tool designed for 4x RTX 5090 GPUs. The architecture optimizes for maximum throughput by combining proven patterns from existing implementations with novel optimizations from recent research.

**Target:** 10B+ keys/second across 4x RTX 5090 GPUs

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           SUPERFLAYER SYSTEM                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │                        HOST CONTROLLER                               │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │     │
│  │  │   Wordlist   │  │     Work     │  │   Progress   │               │     │
│  │  │   Manager    │  │  Distributor │  │   Tracker    │               │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │     │
│  │  │    Config    │  │    Result    │  │     I/O      │               │     │
│  │  │   Manager    │  │  Collector   │  │   Pipeline   │               │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                                    │                                          │
│                    ┌───────────────┼───────────────┐                          │
│                    │               │               │                          │
│                    ▼               ▼               ▼                          │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                      GPU DEVICE LAYER (4x RTX 5090)                   │    │
│  │                                                                       │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │    │
│  │  │   GPU 0     │  │   GPU 1     │  │   GPU 2     │  │   GPU 3     │  │    │
│  │  │             │  │             │  │             │  │             │  │    │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │    │
│  │  │ │Hash Eng │ │  │ │Hash Eng │ │  │ │Hash Eng │ │  │ │Hash Eng │ │  │    │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │    │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │    │
│  │  │ │EC Engine│ │  │ │EC Engine│ │  │ │EC Engine│ │  │ │EC Engine│ │  │    │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │    │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │    │
│  │  │ │ Matcher │ │  │ │ Matcher │ │  │ │ Matcher │ │  │ │ Matcher │ │  │    │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │    │
│  │                                                                       │    │
│  │  ┌───────────────────────────────────────────────────────────────┐   │    │
│  │  │                    SHARED GPU RESOURCES                        │   │    │
│  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │   │    │
│  │  │  │ EC Point     │  │ Bloom Filter │  │   Work       │         │   │    │
│  │  │  │ Tables (L2)  │  │ (Global Mem) │  │   Queues     │         │   │    │
│  │  │  └──────────────┘  └──────────────┘  └──────────────┘         │   │    │
│  │  └───────────────────────────────────────────────────────────────┘   │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Host Controller Layer

The host controller manages overall system coordination running on CPU:

```
┌────────────────────────────────────────────────────────────────────┐
│                       HOST CONTROLLER                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    CONFIGURATION MANAGER                     │  │
│  │                                                              │  │
│  │  • Hash mode selection (SHA256, Keccak, WarpWallet, etc.)   │  │
│  │  • Address type (Bitcoin compressed/uncompressed, Ethereum) │  │
│  │  • Target addresses/bloom filter path                       │  │
│  │  • GPU allocation and thread configuration                  │  │
│  │  • Checkpoint interval and output format                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                     WORDLIST MANAGER                         │  │
│  │                                                              │  │
│  │  ┌───────────┐   ┌───────────┐   ┌───────────┐             │  │
│  │  │  File     │   │  Memory   │   │  Async    │             │  │
│  │  │  Reader   │──▶│  Buffer   │──▶│  Streamer │             │  │
│  │  └───────────┘   └───────────┘   └───────────┘             │  │
│  │                                                              │  │
│  │  Modes:                                                      │  │
│  │  • Single wordlist: stdin or file                           │  │
│  │  • ModeBooks: prime_list × affix_list                       │  │
│  │  • Combination: word1 + word2 + ... + wordN                 │  │
│  │  • Permutation: all orderings of word set                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    WORK DISTRIBUTOR                          │  │
│  │                                                              │  │
│  │  Strategy 1: Static Partitioning                            │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │ Wordlist split into 4 equal chunks at startup          │ │  │
│  │  │ GPU N processes chunk N (no runtime coordination)      │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                              │  │
│  │  Strategy 2: Work Stealing                                  │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │ Central work queue with batch dispensing               │ │  │
│  │  │ Idle GPUs steal work from busy GPUs                    │ │  │
│  │  │ Dynamic load balancing for uneven workloads            │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    PROGRESS TRACKER                          │  │
│  │                                                              │  │
│  │  • Keys processed per GPU (atomic counters)                 │  │
│  │  • Checkpointing: save position every N million keys        │  │
│  │  • ETA calculation based on wordlist size                   │  │
│  │  • Match logging with private key recovery                  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 2. GPU Processing Pipeline

Each GPU runs an identical, independent pipeline:

```
┌────────────────────────────────────────────────────────────────────┐
│                    GPU PROCESSING PIPELINE                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Stage 1: PASSPHRASE BATCH INGESTION                               │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                                                                 ││
│  │  Host Memory              PCIe 5.0              GPU Global Mem ││
│  │  ┌─────────┐              64 GB/s              ┌─────────┐     ││
│  │  │Batch[N] │  ═══════════════════════════════▶ │Batch[N] │     ││
│  │  └─────────┘              Async                └─────────┘     ││
│  │                           Transfer                              ││
│  │  Batch size: 1M-4M passphrases                                 ││
│  │  Double buffering: transfer batch N+1 while processing batch N ││
│  └────────────────────────────────────────────────────────────────┘│
│                                    │                                │
│                                    ▼                                │
│  Stage 2: HASH GENERATION                                          │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                                                                 ││
│  │  ┌─────────────────────────────────────────────────────────┐   ││
│  │  │                   HASH ENGINE KERNEL                     │   ││
│  │  │                                                          │   ││
│  │  │  Thread Configuration:                                   │   ││
│  │  │  • Blocks: 170 (match SM count)                         │   ││
│  │  │  • Threads/Block: 512                                   │   ││
│  │  │  • Passphrases/Thread: 8-16                            │   ││
│  │  │                                                          │   ││
│  │  │  Operations per passphrase:                             │   ││
│  │  │  SHA256:    passphrase → 32-byte private key            │   ││
│  │  │  Keccak:    passphrase → 32-byte private key            │   ││
│  │  │  WarpWallet: scrypt(passphrase, salt) [slow mode]       │   ││
│  │  └─────────────────────────────────────────────────────────┘   ││
│  │                                                                 ││
│  │  Output: Array of 256-bit private keys (scalars)               ││
│  └────────────────────────────────────────────────────────────────┘│
│                                    │                                │
│                                    ▼                                │
│  Stage 3: ELLIPTIC CURVE MULTIPLICATION                           │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                                                                 ││
│  │  ┌─────────────────────────────────────────────────────────┐   ││
│  │  │              EC MULTIPLICATION ENGINE                    │   ││
│  │  │                                                          │   ││
│  │  │  Algorithm: Montgomery Ladder with Precomputed Tables    │   ││
│  │  │                                                          │   ││
│  │  │  ┌────────────────────────────────────────────────────┐  │   ││
│  │  │  │              PRECOMPUTED POINT TABLE               │  │   ││
│  │  │  │                                                    │  │   ││
│  │  │  │  Structure: 16 chunks × 65,536 points each        │  │   ││
│  │  │  │  Storage: L2 cache (48 MB persistent) + Global    │  │   ││
│  │  │  │                                                    │  │   ││
│  │  │  │  Lookup: scalar[i:i+16] → precomputed_point       │  │   ││
│  │  │  └────────────────────────────────────────────────────┘  │   ││
│  │  │                                                          │   ││
│  │  │  ┌────────────────────────────────────────────────────┐  │   ││
│  │  │  │               BATCH POINT ADDITION                 │  │   ││
│  │  │  │                                                    │  │   ││
│  │  │  │  Per thread: Sum 16 looked-up points              │  │   ││
│  │  │  │  Batch inversion: Montgomery's Trick              │  │   ││
│  │  │  │    1 inversion + 3N multiplications               │  │   ││
│  │  │  │                                                    │  │   ││
│  │  │  │  GAS Pattern:                                     │  │   ││
│  │  │  │    Gather → Apply (per-SM inversion) → Scatter   │  │   ││
│  │  │  └────────────────────────────────────────────────────┘  │   ││
│  │  │                                                          │   ││
│  │  │  Output: Array of 512-bit public keys (x, y)            │   ││
│  │  └─────────────────────────────────────────────────────────┘   ││
│  └────────────────────────────────────────────────────────────────┘│
│                                    │                                │
│                                    ▼                                │
│  Stage 4: ADDRESS GENERATION                                       │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                                                                 ││
│  │  Bitcoin: RIPEMD160(SHA256(pubkey))                            ││
│  │  ├── Compressed: 33 bytes (0x02/0x03 + x)                      ││
│  │  └── Uncompressed: 65 bytes (0x04 + x + y)                     ││
│  │                                                                 ││
│  │  Ethereum: Keccak256(pubkey)[12:32]                            ││
│  │  └── 20 bytes from 64-byte uncompressed point (no prefix)      ││
│  │                                                                 ││
│  │  Output: Array of 20-byte address hashes (hash160)             ││
│  └────────────────────────────────────────────────────────────────┘│
│                                    │                                │
│                                    ▼                                │
│  Stage 5: TARGET MATCHING                                          │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │                                                                 ││
│  │  ┌─────────────────────────────────────────────────────────┐   ││
│  │  │                   BLOOM FILTER LOOKUP                    │   ││
│  │  │                                                          │   ││
│  │  │  GPU-resident bloom filter in global memory              │   ││
│  │  │  False positive rate: ~0.1%                             │   ││
│  │  │  Size: 8 GB for 100M addresses                          │   ││
│  │  │                                                          │   ││
│  │  │  Parallel lookup: each thread checks its address         │   ││
│  │  └─────────────────────────────────────────────────────────┘   ││
│  │                         │                                       ││
│  │                         ▼ (on potential match)                  ││
│  │  ┌─────────────────────────────────────────────────────────┐   ││
│  │  │                   EXACT MATCH VERIFICATION               │   ││
│  │  │                                                          │   ││
│  │  │  Secondary check against sorted address list             │   ││
│  │  │  Eliminates false positives                              │   ││
│  │  │  Triggered only for bloom filter hits                    │   ││
│  │  └─────────────────────────────────────────────────────────┘   ││
│  │                         │                                       ││
│  │                         ▼ (on confirmed match)                  ││
│  │  ┌─────────────────────────────────────────────────────────┐   ││
│  │  │                   RESULT OUTPUT                          │   ││
│  │  │                                                          │   ││
│  │  │  Write to result buffer:                                 │   ││
│  │  │  • Private key (256 bits)                               │   ││
│  │  │  • Matched address                                       │   ││
│  │  │  • Original passphrase                                   │   ││
│  │  │  • Timestamp                                             │   ││
│  │  └─────────────────────────────────────────────────────────┘   ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 3. Memory Architecture

Optimized for RTX 5090 Blackwell architecture:

```
┌────────────────────────────────────────────────────────────────────┐
│                  GPU MEMORY HIERARCHY (per GPU)                     │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                         REGISTERS                            │  │
│  │                      (255 per thread)                        │  │
│  │                                                              │  │
│  │  Usage:                                                      │  │
│  │  • EC point coordinates (x, y): 16 × 32-bit = 512 bits      │  │
│  │  • Working values for multiplication: 32 × 32-bit           │  │
│  │  • Loop counters, temp storage                              │  │
│  │                                                              │  │
│  │  Strategy: Maximize register usage, minimize shared memory   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                    │                                │
│                                    ▼                                │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    SHARED MEMORY (128 KB/SM)                 │  │
│  │                                                              │  │
│  │  Allocation per block (512 threads):                        │  │
│  │  • Batch inversion workspace: 32 KB                         │  │
│  │  • Intermediate point storage: 64 KB                        │  │
│  │  • Bloom filter hash workspace: 16 KB                       │  │
│  │  • Reserved: 16 KB                                          │  │
│  │                                                              │  │
│  │  Strategy: Use for per-block coordination, GAS pattern      │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                    │                                │
│                                    ▼                                │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    L2 CACHE (65 MB unified)                  │  │
│  │                                                              │  │
│  │  L2 Persistent Region (48 MB, 75% configurable):            │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  EC Point Table (hot portion): 48 MB                   │ │  │
│  │  │  • Most frequently accessed table chunks               │ │  │
│  │  │  • 16-bit index coverage: chunks 0-7                   │ │  │
│  │  │  • Reduces global memory pressure by ~50%              │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                              │  │
│  │  L2 Standard Region (17 MB):                                │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  Working data, spillover, dynamic caching              │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                    │                                │
│                                    ▼                                │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                 GLOBAL MEMORY (32 GB GDDR7)                  │  │
│  │                                                              │  │
│  │  Static Allocations:                                        │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  EC Point Table (complete): 67 MB                      │ │  │
│  │  │  Bloom Filter: 8 GB                                    │ │  │
│  │  │  Sorted Address List (verification): 2 GB              │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                              │  │
│  │  Dynamic Allocations (double-buffered):                     │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │  Input Buffer A/B: 2 × 128 MB (4M passphrases each)   │ │  │
│  │  │  Private Key Buffer A/B: 2 × 128 MB                   │ │  │
│  │  │  Public Key Buffer A/B: 2 × 256 MB                    │ │  │
│  │  │  Address Buffer A/B: 2 × 64 MB                        │ │  │
│  │  │  Result Buffer: 16 MB                                 │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                              │  │
│  │  Total Static: ~10 GB                                       │  │
│  │  Total Dynamic: ~1.2 GB                                     │  │
│  │  Available for future use: ~20 GB                           │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 4. Multi-GPU Coordination

```
┌────────────────────────────────────────────────────────────────────┐
│                   MULTI-GPU COORDINATION                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    WORK DISTRIBUTION                         │  │
│  │                                                              │  │
│  │         ┌─────────────────────────────────────┐             │  │
│  │         │        WORDLIST/KEY SPACE           │             │  │
│  │         │   [0 ─────────────────────────── N] │             │  │
│  │         └─────────────────────────────────────┘             │  │
│  │                          │                                   │  │
│  │              ┌───────────┼───────────┐                       │  │
│  │              │           │           │                       │  │
│  │              ▼           ▼           ▼                       │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │  │
│  │  │  Segment 0  │ │  Segment 1  │ │  Segment 2  │ │Segment3│ │  │
│  │  │   GPU 0     │ │   GPU 1     │ │   GPU 2     │ │ GPU 3  │ │  │
│  │  │  [0, N/4)   │ │ [N/4, N/2)  │ │[N/2, 3N/4)  │ │[3N/4,N)│ │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └────────┘ │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                  SYNCHRONIZATION POINTS                      │  │
│  │                                                              │  │
│  │  Initialization:                                             │  │
│  │  ├── All GPUs load EC point table (parallel)                │  │
│  │  ├── All GPUs load bloom filter (parallel)                  │  │
│  │  └── Barrier: wait for all GPU initialization               │  │
│  │                                                              │  │
│  │  Runtime (minimal sync):                                     │  │
│  │  ├── Progress reporting (async, non-blocking)               │  │
│  │  ├── Result collection (async queue)                        │  │
│  │  └── Checkpointing (periodic, staggered)                    │  │
│  │                                                              │  │
│  │  Termination:                                                │  │
│  │  ├── Any GPU finds match → signal all                       │  │
│  │  └── All GPUs complete → aggregate results                  │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                INTERCONNECT TOPOLOGY                         │  │
│  │                                                              │  │
│  │  Option 1: PCIe 5.0 Only                                    │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │                      CPU                                │ │  │
│  │  │                       │                                 │ │  │
│  │  │           ┌───────────┼───────────┐                     │ │  │
│  │  │           │           │           │                     │ │  │
│  │  │         GPU0       GPU1        GPU2       GPU3          │ │  │
│  │  │                                                         │ │  │
│  │  │  Bandwidth: 64 GB/s per GPU (bidirectional)            │ │  │
│  │  │  Latency: ~1-2 μs                                      │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  │                                                              │  │
│  │  Option 2: NVLink (if available)                            │  │
│  │  ┌────────────────────────────────────────────────────────┐ │  │
│  │  │         GPU0 ═══════════════════ GPU1                  │ │  │
│  │  │           ║                        ║                    │ │  │
│  │  │           ║                        ║                    │ │  │
│  │  │         GPU2 ═══════════════════ GPU3                  │ │  │
│  │  │                                                         │ │  │
│  │  │  Bandwidth: 900 GB/s total                             │ │  │
│  │  │  Latency: ~0.1-0.2 μs                                  │ │  │
│  │  └────────────────────────────────────────────────────────┘ │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Kernel Architecture

### Hash Generation Kernel

```cuda
// Simplified kernel structure
__global__ void hash_generation_kernel(
    const char* __restrict__ passphrases,      // Input passphrases
    const uint32_t* __restrict__ lengths,      // Passphrase lengths
    uint256_t* __restrict__ private_keys,      // Output private keys
    const uint32_t batch_size,
    const HashMode mode                        // SHA256, Keccak, etc.
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = tid; i < batch_size; i += stride) {
        const char* phrase = passphrases + i * MAX_PHRASE_LEN;
        const uint32_t len = lengths[i];

        switch (mode) {
            case SHA256:
                sha256_hash(phrase, len, &private_keys[i]);
                break;
            case KECCAK:
                keccak256_hash(phrase, len, &private_keys[i]);
                break;
            // Other modes...
        }
    }
}
```

### EC Multiplication Kernel (Core)

```cuda
// Montgomery ladder with precomputed table lookup
__global__ void ec_multiply_batch_kernel(
    const uint256_t* __restrict__ scalars,     // Private keys
    const ECPoint* __restrict__ point_table,   // Precomputed 1M points
    ECPoint* __restrict__ public_keys,         // Output public keys
    const uint32_t batch_size
) {
    // Shared memory for batch inversion (GAS pattern)
    __shared__ uint256_t inv_workspace[THREADS_PER_BLOCK * 2];

    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t local_tid = threadIdx.x;

    if (tid >= batch_size) return;

    uint256_t scalar = scalars[tid];
    ECPoint result = POINT_AT_INFINITY;

    // Table-based multiplication: 16 chunks of 16 bits each
    #pragma unroll
    for (int chunk = 0; chunk < 16; chunk++) {
        uint32_t index = extract_16bits(scalar, chunk);
        ECPoint chunk_point = point_table[chunk * 65536 + index];

        // Accumulate without immediate inversion (defer)
        result = point_add_defer_inversion(result, chunk_point);
    }

    // GAS: Gather denominators for batch inversion
    inv_workspace[local_tid] = result.denominator;
    __syncthreads();

    // Apply: Block-level batch inversion using Montgomery's trick
    if (local_tid == 0) {
        batch_invert_montgomery(inv_workspace, THREADS_PER_BLOCK);
    }
    __syncthreads();

    // Scatter: Apply inverted denominator
    result = apply_inversion(result, inv_workspace[local_tid]);

    public_keys[tid] = result;
}
```

### Address Generation and Matching Kernel

```cuda
__global__ void address_match_kernel(
    const ECPoint* __restrict__ public_keys,
    const uint8_t* __restrict__ bloom_filter,
    const uint32_t bloom_size_bits,
    const uint256_t* __restrict__ scalars,     // For recovery
    const char* __restrict__ passphrases,      // For recovery
    MatchResult* __restrict__ results,
    uint32_t* __restrict__ result_count,
    const uint32_t batch_size,
    const AddressMode mode                     // Bitcoin/Ethereum
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    ECPoint pubkey = public_keys[tid];
    uint8_t address[20];

    // Generate address based on mode
    switch (mode) {
        case BITCOIN_COMPRESSED:
            generate_bitcoin_address_compressed(pubkey, address);
            break;
        case BITCOIN_UNCOMPRESSED:
            generate_bitcoin_address_uncompressed(pubkey, address);
            break;
        case ETHEREUM:
            generate_ethereum_address(pubkey, address);
            break;
    }

    // Bloom filter check (fast path)
    if (bloom_check(bloom_filter, bloom_size_bits, address)) {
        // Potential match - write to results for host verification
        uint32_t idx = atomicAdd(result_count, 1);
        if (idx < MAX_RESULTS) {
            results[idx].private_key = scalars[tid];
            memcpy(results[idx].address, address, 20);
            results[idx].passphrase_idx = tid;
        }
    }
}
```

---

## Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                        DATA FLOW DIAGRAM                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  TIME ───────────────────────────────────────────────────────────▶ │
│                                                                    │
│  Batch 0    Batch 1    Batch 2    Batch 3    Batch 4    ...       │
│                                                                    │
│  ┌────────┐                                                        │
│  │TRANSFER│ ───────────────────────────────────────────────────▶   │
│  └────────┘  ┌────────┐                                            │
│              │TRANSFER│ ───────────────────────────────────────▶   │
│  ┌────────┐  └────────┘  ┌────────┐                                │
│  │ HASH   │              │TRANSFER│ ─────────────────────────▶     │
│  └────────┘  ┌────────┐  └────────┘                                │
│              │ HASH   │              ┌────────┐                    │
│  ┌────────┐  └────────┘  ┌────────┐  │TRANSFER│                    │
│  │EC MULT │              │ HASH   │  └────────┘                    │
│  └────────┘  ┌────────┐  └────────┘                                │
│              │EC MULT │              ┌────────┐                    │
│  ┌────────┐  └────────┘  ┌────────┐  │ HASH   │                    │
│  │ADDRESS │              │EC MULT │  └────────┘                    │
│  └────────┘  ┌────────┐  └────────┘                                │
│              │ADDRESS │              ┌────────┐                    │
│  ┌────────┐  └────────┘  ┌────────┐  │EC MULT │                    │
│  │ MATCH  │              │ADDRESS │  └────────┘                    │
│  └────────┘  ┌────────┐  └────────┘                                │
│              │ MATCH  │              ┌────────┐                    │
│              └────────┘  ┌────────┐  │ADDRESS │                    │
│                          │ MATCH  │  └────────┘                    │
│                          └────────┘              ┌────────┐        │
│                                                  │ MATCH  │        │
│                                                  └────────┘        │
│                                                                    │
│  ════════════════════════════════════════════════════════════════  │
│  Pipeline achieves full utilization after 4 batch warm-up          │
│  Each kernel overlaps with next batch's transfer                   │
└────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Parameters

### Compile-Time Constants

```cpp
// Thread configuration
constexpr uint32_t THREADS_PER_BLOCK = 512;
constexpr uint32_t BLOCKS_PER_GPU = 170;        // Match RTX 5090 SM count
constexpr uint32_t KEYS_PER_THREAD = 8;

// Memory configuration
constexpr size_t MAX_PHRASE_LENGTH = 256;       // bytes
constexpr size_t BATCH_SIZE = 4 * 1024 * 1024;  // 4M passphrases
constexpr size_t BLOOM_SIZE_BYTES = 8ULL * 1024 * 1024 * 1024;  // 8 GB

// EC table configuration
constexpr uint32_t TABLE_CHUNKS = 16;
constexpr uint32_t POINTS_PER_CHUNK = 65536;    // 2^16
constexpr size_t POINT_SIZE = 64;               // 2 × 32 bytes
constexpr size_t TABLE_SIZE = TABLE_CHUNKS * POINTS_PER_CHUNK * POINT_SIZE;

// L2 cache configuration (Blackwell)
constexpr size_t L2_PERSISTENT_SIZE = 48 * 1024 * 1024;  // 48 MB (75%)
```

### Runtime Configuration

```yaml
# config.yaml
input:
  mode: "wordlist"           # wordlist, modebooks, combination, raw
  wordlist_path: "./words.txt"
  affix_path: "./affixes.txt"  # for modebooks mode

hash:
  algorithm: "sha256"        # sha256, keccak, warp, brainv2

address:
  type: "bitcoin_compressed" # bitcoin_compressed, bitcoin_uncompressed, ethereum

targets:
  bloom_filter: "./targets.blf"
  verification_list: "./targets_sorted.hex"

gpus:
  devices: [0, 1, 2, 3]
  threads_per_block: 512
  batch_size: 4194304

output:
  results_file: "./results.txt"
  checkpoint_interval: 100000000  # keys
  checkpoint_path: "./checkpoint.bin"
```

---

## Performance Considerations

### Bottleneck Analysis

| Stage | Operations | Bottleneck | Optimization |
|-------|------------|------------|--------------|
| Hash Generation | SHA256/Keccak | Compute | Warp-level parallelism |
| EC Multiplication | 16 table lookups + 15 point adds | Memory bandwidth | L2 cache persistence |
| Batch Inversion | 1 inv + 3N muls | Compute | GAS pattern |
| Address Generation | 2 hashes (SHA256 + RIPEMD160) | Compute | Fused kernel |
| Bloom Lookup | Hash + bit check | Memory latency | Prefetching |

### Expected Throughput

```
Per GPU (RTX 5090):
  Theoretical peak: 21,760 cores × 2.41 GHz = 52.4 TFLOPS

  EC multiplication (dominant cost):
    ~1000 32-bit multiplications per key
    Effective throughput: ~50 Gkeys/sec theoretical
    Realistic with overhead: ~2.5 Gkeys/sec

  4x GPUs: ~10 Gkeys/sec aggregate target
```

---

## Error Handling

```cpp
enum class SuperflayerError {
    SUCCESS = 0,
    GPU_INIT_FAILED,
    OUT_OF_MEMORY,
    TABLE_LOAD_FAILED,
    BLOOM_LOAD_FAILED,
    WORDLIST_ERROR,
    CHECKPOINT_FAILED,
    CUDA_KERNEL_ERROR
};

// Each GPU maintains independent error state
// Host polls GPU error buffers asynchronously
// Critical errors trigger graceful shutdown with checkpoint
```

---

## Future Extensions

1. **Additional Hash Modes**: Argon2, bcrypt for modern brainwallets
2. **Distributed Mode**: Multiple machines via MPI or custom protocol
3. **Real-time Target Updates**: Hot-reload bloom filter
4. **Hardware Wallet Patterns**: BIP39/BIP44 derivation paths
5. **Memory Optimization**: Streaming for larger-than-VRAM wordlists
