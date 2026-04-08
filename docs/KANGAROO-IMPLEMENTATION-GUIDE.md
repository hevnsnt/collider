# Pollard's Kangaroo Algorithm - Implementation Guide

## Executive Summary

Pollard's Kangaroo (also called Pollard's Lambda) algorithm is the optimal approach for solving the Elliptic Curve Discrete Logarithm Problem (ECDLP) when you have a **known public key** and need to find the private key within a bounded range. It reduces complexity from O(N) to O(sqrt(N)), making it feasible to solve Bitcoin puzzles that would otherwise be intractable.

**Key Finding**: The existing `/src/core/kangaroo.hpp` implements the basic algorithm correctly but is CPU-only. For Puzzle #135 (13.5 BTC), a high-performance CUDA implementation similar to JeanLucPons/Kangaroo is needed.

---

## 1. Algorithm Deep Dive

### 1.1 Core Concept

The algorithm exploits the birthday paradox on elliptic curve points. Two "kangaroos" (random walkers) jump through the curve. When they land on the same point (collision), we can compute the private key.

**Mathematical Foundation**:
```
Given:
  - Target public key Q = k*G (we want to find k)
  - Range [a, b] containing k

Setup:
  - Tame kangaroo starts at known position: T = (a + (b-a)/2) * G
  - Wild kangaroo starts at unknown position: W = Q (the target)

Both jump using deterministic function:
  - f(P) = P + d[i] * G, where i = hash(P.x) mod 32

When T and W land on same point P:
  - T traveled distance d_t from start
  - W traveled distance d_w from Q
  - Therefore: k = d_t - d_w + (a + (b-a)/2)
```

### 1.2 Wild vs Tame Kangaroos

| Kangaroo | Starting Position | Purpose |
|----------|------------------|---------|
| **Tame** | Known point in range center: `(a + (b-a)/2) * G` | Reference walker with known scalar |
| **Wild** | Target public key Q | Searches for collision with tame |

**Why it works**: Expected collision after O(sqrt(N)) steps each, due to birthday paradox.

### 1.3 Distinguished Points (DP)

**Problem**: Checking every point for collision requires O(sqrt(N)) storage.

**Solution**: Only record "distinguished points" - points where the X coordinate has k trailing zero bits.

```
DP criteria: x.d[0] & ((1 << dp_bits) - 1) == 0

Example with dp_bits = 20:
  - Only 1 in 2^20 (~1M) points are distinguished
  - Storage reduced 1,000,000x
  - Trade-off: Need extra steps between DPs (~2^dp_bits/2 on average)
```

**DP bits selection** (from existing code):
```cpp
dp_bits = std::max(10, std::min(28, range_bits / 4));
```

For Puzzle #135 (135-bit range):
- dp_bits = 135/4 = 33... clamped to 28
- Practical: 20-24 bits balances memory vs steps

### 1.4 Jump Table Design

```cpp
static constexpr int NUM_JUMPS = 32;  // Power of 2 for efficient modulo

// Jump distances span from sqrt(sqrt(range)) to sqrt(range)
for (int i = 0; i < NUM_JUMPS; i++) {
    int shift = sqrt_bits / 4 + (i * sqrt_bits * 3) / (4 * NUM_JUMPS);
    distances[i] = 1 << shift;
}
```

**Key insight**: Variable jump sizes provide better mixing than uniform jumps.

### 1.5 Complexity Analysis

| Range Bits | Brute Force | Kangaroo | Speedup |
|------------|-------------|----------|---------|
| 66 | 2^65 ops | 2^33 ops | ~4 billion x |
| 70 | 2^69 ops | 2^35 ops | ~68 billion x |
| 135 | 2^134 ops | 2^67.5 ops | Astronomical |

**Puzzle #135 specifics**:
- Range: 2^134 to 2^135 (135-bit key)
- Kangaroo ops: sqrt(2^135) = 2^67.5 = ~200 quintillion
- At 10B ops/sec: 200 quintillion / 10^10 = 20 million seconds = **231 days**
- With 256 GPUs: ~1 day

---

## 2. JeanLucPons/Kangaroo Implementation Analysis

### 2.1 Architecture

JeanLucPons/Kangaroo is the reference GPU implementation used to solve Bitcoin puzzles #75-130.

```
Main Components:
├── Kangaroo.cpp         # Main entry, coordination
├── GPU/GPUEngine.cu     # CUDA kernel management
├── GPU/GPUMath.h        # Optimized 256-bit GPU arithmetic
├── GPU/GPUHash.h        # Hash functions for DP detection
├── Kangaroo.h           # Algorithm structures
├── SECPK1/              # secp256k1 implementation
│   ├── Point.cpp/h      # EC point operations
│   └── Int.cpp/h        # Big integer arithmetic
├── Timer.cpp/h          # Performance measurement
└── HashTable.cpp/h      # DP storage and collision detection
```

### 2.2 GPU Kernel Structure

```cuda
__global__ void kangaroo_kernel(
    uint64_t* px,           // X coordinates (64-bit words * 4)
    uint64_t* py,           // Y coordinates
    uint64_t* distance,     // Scalar distances traveled
    int* status,            // DP found flags
    uint64_t* jump_table,   // Precomputed jump scalars
    ECPoint* jump_points,   // Precomputed jump points
    int dp_bits             // Distinguished point threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread maintains one kangaroo
    uint256 x, y, d;
    load_kangaroo(tid, px, py, distance, &x, &y, &d);

    for (int iter = 0; iter < STEPS_PER_KERNEL; iter++) {
        // Select jump based on x coordinate
        int jump_idx = x.d[0] & 31;

        // Point addition: P += jump_points[jump_idx]
        ec_add_affine(&x, &y, jump_points[jump_idx]);

        // Update distance
        d += jump_table[jump_idx];

        // Check for distinguished point
        if ((x.d[0] & ((1ULL << dp_bits) - 1)) == 0) {
            status[tid] = 1;
            break;
        }
    }

    store_kangaroo(tid, px, py, distance, x, y, d);
}
```

### 2.3 Key Optimizations

| Optimization | Technique | Speedup |
|--------------|-----------|---------|
| **Batch Point Addition** | Montgomery's trick for batch inversion | 10-50x |
| **GLV Endomorphism** | Exploit secp256k1 endomorphism | 1.5x |
| **Precomputed Jump Points** | 32 points in constant memory | Reduces EC muls |
| **Efficient DP Check** | Single 64-bit AND operation | Negligible cost |
| **Coalesced Memory** | Structure-of-arrays layout | Better bandwidth |
| **Register Optimization** | Minimize spills to local memory | Reduced latency |

### 2.4 Performance Benchmarks (Historical)

| Puzzle | Hardware | Time | Implementation |
|--------|----------|------|----------------|
| #85 | 4x V100 | 25 min | JeanLucPons |
| #110 | 256x V100 | 2.1 days | JeanLucPons |
| #115 | 256x V100 | 13 days | JeanLucPons |
| #120 | Pool (1000+ GPUs) | ~1 month | Collaborative |
| #130 | Large pool | Several months | Collaborative |

---

## 3. Implementation Requirements

### 3.1 Public Key Input Format

```cpp
struct KangarooTarget {
    uint8_t compressed_pubkey[33];    // 02/03 + X coordinate
    uint8_t uncompressed_pubkey[65];  // 04 + X + Y
    uint256 x, y;                     // Internal representation
};

void parse_compressed_pubkey(const uint8_t* data, uint256& x, uint256& y) {
    // Copy X (bytes 1-32, big-endian to little-endian)
    for (int i = 0; i < 32; i++) {
        x.bytes[31-i] = data[i+1];
    }

    // Compute Y: y^2 = x^3 + 7
    uint256 x3, y2;
    mod_mul(x3, x, x);       // x^2
    mod_mul(x3, x3, x);      // x^3
    mod_add(y2, x3, SEVEN);  // x^3 + 7
    mod_sqrt(y, y2);         // y = sqrt(x^3 + 7)

    // Check parity
    bool is_even = data[0] == 0x02;
    if ((y.d[0] & 1) != (is_even ? 0 : 1)) {
        mod_neg(y, y);
    }
}
```

### 3.2 Memory Requirements

| Component | Size (per GPU) | Notes |
|-----------|----------------|-------|
| **Kangaroo State** | 64 bytes * N | x(32B) + y(32B) per kangaroo |
| **Distance Scalars** | 32 bytes * N | 256-bit scalars |
| **Jump Table** | 32 * 64 = 2 KB | 32 jump points |
| **Local DP Buffer** | 16 MB typical | Before sync to host |

**Example for RTX 5090 (32 GB VRAM)**:
- N_kangaroos = 100 million
- State: 6.4 GB + Distance: 3.2 GB = ~10 GB working set

### 3.3 Distinguished Point Storage

```cpp
struct DistinguishedPoint {
    uint64_t x_hash;      // First 64 bits of X
    uint32_t extra_bits;  // Additional X bits for verification
    bool is_tame;         // Tame or wild kangaroo
    uint8_t padding[3];
    uint256 distance;     // Scalar distance traveled
};  // 48 bytes per DP

class DPHashTable {
    std::unordered_map<uint64_t, std::vector<DistinguishedPoint>> table;
    std::mutex mutex;

    bool insert_and_check(const DistinguishedPoint& dp, uint256* result_key);
};
```

---

## 4. Command Line Usage (JeanLucPons)

```bash
# Basic usage with public key
./Kangaroo -t 02abcd...  # Compressed pubkey (hex)

# Specify search range
./Kangaroo -t 02abc... -r 134:135  # Range 2^134 to 2^135

# Multi-GPU
./Kangaroo -t 02abc... -gpu 0,1,2,3

# With work file for resume
./Kangaroo -t 02abc... -w work.txt

# Load previous work and continue
./Kangaroo -t 02abc... -i work.txt

# Server mode (for distributed solving)
./Kangaroo -t 02abc... -s -p 17403
```

---

## 5. Integration Recommendations

### 5.1 Short-term (1-2 weeks)

1. **Use JeanLucPons/Kangaroo as external tool**
   - Clone and build the repo
   - Add wrapper in superflayer CLI
   - Focus on Puzzle #135 immediately

2. **Enhance CPU Kangaroo**
   - Add multi-threading with std::thread
   - Add work file save/resume

### 5.2 Medium-term (1-3 months)

1. **Port Kangaroo kernel to CUDA codebase**
   - Reuse existing secp256k1.cu
   - Add kangaroo_kernel.cu
   - Integrate with multi-GPU infrastructure

2. **Add GLV endomorphism**
   - 1.5x speedup potential
   - Well-documented in libsecp256k1

### 5.3 Files to Modify

| File | Modification |
|------|--------------|
| `src/core/kangaroo.hpp` | Add multi-threading, work files |
| `src/gpu/secp256k1.cu` | Maybe add GLV |
| **(NEW)** `src/gpu/kangaroo_kernel.cu` | GPU Kangaroo implementation |
| **(NEW)** `src/core/kangaroo_gpu.hpp` | Host-side GPU management |

---

## 6. GLV Endomorphism (1.5x Speedup)

secp256k1 has a special structure that allows faster multiplication:

```
Endomorphism: lambda * P = (beta * x, y) where
  lambda^3 = 1 (mod n)
  beta^3 = 1 (mod p)

For scalar k, decompose: k = k1 + k2 * lambda (mod n)
where |k1|, |k2| < sqrt(n)

Then: k*G = k1*G + k2*(lambda*G) = k1*G + k2*G'
  where G' = (beta * G.x, G.y)

Result: 2 half-size multiplications instead of 1 full
```

Constants for secp256k1:
```cpp
// Beta: cube root of 1 mod p
static const uint256 BETA = {
    0x7ae96a2b657c0710ULL, 0x6e64479eac3434e9ULL,
    0x9cf0497512f58995ULL, 0x851695d49a83f8efULL
};

// Lambda: cube root of 1 mod n
static const uint256 LAMBDA = {
    0xdf02967c1b23bd72ULL, 0x122e22ea20816678ULL,
    0xa5261c028812645aULL, 0x5363ad4cc05c30e0ULL
};
```

---

## Summary

Pollard's Kangaroo is the optimal path to solving Bitcoin Puzzle #135 (13.5 BTC). The existing codebase is well-prepared:

1. **CPU Kangaroo** (`kangaroo.hpp`) - Correct algorithm, needs GPU port
2. **secp256k1 CUDA** (`secp256k1.cu`) - Excellent, has batch inversion
3. **Infrastructure** - Multi-GPU ready

**Recommended immediate action**:
- Build JeanLucPons/Kangaroo and start Puzzle #135 attack
- In parallel, port the kernel to this codebase for unified tool
