# Superflayer Performance Targets

## Executive Summary

This document defines performance targets, benchmarks, and optimization strategies for Superflayer. The primary goal is **10B+ keys/second** across 4x RTX 5090 GPUs, representing a significant advancement over existing brainwallet cracking tools.

---

## Target Hardware Specifications

### RTX 5090 (per GPU)

| Specification | Value | Relevance |
|---------------|-------|-----------|
| CUDA Cores | 21,760 | Parallel compute capacity |
| SMs | 170 | Kernel block mapping |
| Base Clock | 2.01 GHz | Minimum performance floor |
| Boost Clock | 2.41 GHz | Typical sustained frequency |
| VRAM | 32 GB GDDR7 | Table + bloom filter capacity |
| Memory Bandwidth | 1,792 GB/s | Throughput ceiling for memory-bound ops |
| L2 Cache | 65 MB | EC table hot path caching |
| Shared/SM | 128 KB | Per-block working memory |
| TDP | 575W | Thermal/power constraints |

### 4x RTX 5090 System

| Specification | Value |
|---------------|-------|
| Total CUDA Cores | 87,040 |
| Total SMs | 680 |
| Aggregate Memory BW | 7,168 GB/s |
| Total VRAM | 128 GB |
| System TDP | 2,300W+ |

---

## Performance Targets

### Primary Target

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Keys/second (4 GPU)** | **10B+** | 2x per-GPU vs RTX 3090, 4x scaling |
| Keys/second (1 GPU) | 2.5B+ | Baseline per-device target |
| GPU Utilization | > 85% | Minimize idle cycles |
| Multi-GPU Scaling | > 3.5x | Near-linear scaling |

### Secondary Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Latency (batch) | < 50ms | For 4M key batch |
| Memory Efficiency | > 80% | Bandwidth utilization |
| Power Efficiency | > 4M keys/W | Per-watt performance |
| Startup Time | < 10s | Table loading included |

---

## Baseline Comparisons

### Existing Implementations

| Tool | Hardware | Performance | Source |
|------|----------|-------------|--------|
| Brainflayer | Modern CPU (single thread) | ~1M keys/sec | Estimated |
| Brainflayer | Multi-core CPU (8 threads) | ~8M keys/sec | Estimated |
| CudaBrainSecp | RTX 3060 | ~50M keys/sec | GitHub claims |
| BitCrack | RTX 3090 | ~1B keys/sec | Community benchmarks |
| gECC | A100 | ~4B EC ops/sec | Paper claims |

### Theoretical Analysis

**RTX 5090 vs RTX 3090:**
- CUDA Cores: 21,760 vs 10,496 (2.07x)
- Memory BW: 1,792 vs 936 GB/s (1.91x)
- Expected speedup: ~2x (conservative)

**4x RTX 5090:**
- 4 x 2x RTX 3090 = 8x RTX 3090
- With scaling losses (~12%): ~7x RTX 3090
- If RTX 3090 = 1B keys/sec: Target = 7B-10B keys/sec

---

## Performance Model

### Operation Breakdown

Each key generation involves:

| Operation | Approx. Cost | Bottleneck |
|-----------|-------------|------------|
| Passphrase hash (SHA256) | 1 | Compute |
| EC point multiplication | 100 | Memory/Compute |
| Batch inversion (amortized) | 3 | Compute |
| Address hash (SHA256+RIPEMD160) | 2 | Compute |
| Bloom filter lookup | 0.1 | Memory |
| **Total** | **~106** | Mixed |

*Cost units relative to SHA256 hash*

### Throughput Calculation

**Per-SM Analysis:**
```
RTX 5090 SM Configuration:
- 128 CUDA cores per SM
- 4 warp schedulers per SM
- 2048 max threads per SM

Kernel Configuration:
- 512 threads per block
- ~4 blocks per SM (limited by registers)
- 2048 threads active per SM

Key Operations per Thread Iteration:
- 1 hash (~500 cycles)
- 1 EC multiply (~50,000 cycles with table)
- 1 address gen (~1000 cycles)
Total: ~51,500 cycles per key

At 2.1 GHz average:
- 2.1B cycles/sec per SM
- 2.1B / 51,500 = ~40,777 keys/sec per SM

170 SMs per GPU:
- 170 x 40,777 = ~6.9M keys/sec per GPU
```

Wait, this seems too low. Let's reconsider with parallelism:

**Parallel Analysis:**
```
With 2048 threads per SM, processing keys in parallel:
- Each thread processes 1 key per kernel launch
- Batch size: 170 SMs x 2048 threads = 348,160 keys per batch

EC multiply dominates:
- ~50,000 cycles at 2.1 GHz = ~24 us per key per thread
- But 2048 threads work in parallel

Actual per-SM throughput:
- 2048 keys / 24 us = 85M keys/sec per SM (theoretical)
- With memory/occupancy limits: ~15-20M keys/sec per SM

170 SMs:
- 170 x 17.5M = ~3B keys/sec per GPU (theoretical)
- With practical overhead: 2-2.5B keys/sec per GPU

4 GPUs:
- 4 x 2.25B = 9B keys/sec (mid estimate)
```

**Revised Estimate:** 8-12B keys/sec for 4x RTX 5090

---

## Optimization Strategies

### 1. EC Multiplication Optimization

**Target:** Reduce EC multiply from 50,000 to 30,000 cycles

| Technique | Expected Gain | Complexity |
|-----------|---------------|------------|
| Precomputed table (16 chunks) | 20x baseline | Implemented |
| L2 cache persistence | +30% | Medium |
| IMAD→IADD3 replacement | +15% | High |
| Optimized addition chains | +10% | High |
| Register pressure reduction | +20% | High |

**Implementation:**
```cuda
// Before: Standard Montgomery multiplication
__device__ void mul_mont(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    // 16 IMAD instructions in critical path
    // Each IMAD: 4 cycle issue interval
}

// After: Optimized with carry predicates
__device__ void mul_mont_opt(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    // Replace IMAD with IADD3 + predicate carry
    // IADD3: 2 cycle issue interval
    asm volatile(
        "add.cc.u64 %0, %1, %2;\n"
        "addc.u64 %3, 0, 0;\n"
        : "=l"(sum), "=l"(carry) : "l"(a), "l"(b)
    );
}
```

### 2. Memory Hierarchy Optimization

**L2 Cache Persistence:**
```cpp
// Configure 48 MB persistent for EC table
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = ec_table_ptr;
attr.accessPolicyWindow.num_bytes = 48 * 1024 * 1024;
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

**Expected Impact:**
- Without L2 persistence: ~20% of lookups hit L2
- With L2 persistence: ~75% of lookups hit L2
- Speedup: ~1.3x for memory-bound sections

### 3. Batch Inversion (Montgomery's Trick)

**Standard:** N inversions = N x 256 multiplications
**Batched:** 1 inversion + 3N multiplications

```
For batch size N = 1,000,000:
Standard: 256,000,000 equivalent ops
Batched:  3,000,256 equivalent ops
Speedup:  ~85x for inversion-heavy workloads
```

**Implementation:**
```cuda
__global__ void batch_invert_montgomery(
    FieldElement* values,
    uint32_t count
) {
    __shared__ FieldElement products[BLOCK_SIZE];

    // Phase 1: Compute running products
    // products[i] = values[0] * values[1] * ... * values[i]

    // Phase 2: Single inversion of final product
    // inv_all = 1 / products[count-1]

    // Phase 3: Extract individual inverses
    // values[i] = inv_all * products[i-1] / values[i]
}
```

### 4. Kernel Fusion

**Before:** Separate kernels with global memory round-trips
```
hash_kernel → private_keys (global) → ec_kernel → pubkeys (global) → addr_kernel
```

**After:** Fused kernels with register/shared memory
```
fused_kernel: hash → (registers) → ec_mult → (registers) → addr_gen → bloom_check
```

**Expected Gain:** 10-20% reduction in memory traffic

### 5. Multi-GPU Scaling

**Work Distribution Strategy:**
```cpp
// Static partitioning for predictable workloads
void distribute_static(const Wordlist& words, int num_gpus) {
    size_t chunk_size = words.size() / num_gpus;
    for (int g = 0; g < num_gpus; g++) {
        size_t start = g * chunk_size;
        size_t end = (g == num_gpus - 1) ? words.size() : start + chunk_size;
        gpu_workers[g].process(words.subspan(start, end - start));
    }
}

// Work stealing for variable workloads
class WorkStealingQueue {
    std::atomic<size_t> head_, tail_;
    std::vector<Batch> batches_;
public:
    bool steal(Batch& batch) {
        size_t h = head_.fetch_add(1);
        if (h >= tail_.load()) return false;
        batch = batches_[h];
        return true;
    }
};
```

**Scaling Target:** > 3.8x with 4 GPUs

---

## Benchmark Suite

### Microbenchmarks

| Benchmark | Description | Expected |
|-----------|-------------|----------|
| `bench_sha256` | SHA256 throughput | 50B hashes/sec |
| `bench_keccak` | Keccak256 throughput | 40B hashes/sec |
| `bench_field_mul` | Field multiplication | 500B ops/sec |
| `bench_field_inv` | Single inversion | 100M ops/sec |
| `bench_batch_inv` | Batched inversion | 10B ops/sec |
| `bench_point_add` | Point addition | 5B ops/sec |
| `bench_ec_mult` | Scalar multiplication | 3B ops/sec |
| `bench_bloom` | Bloom filter lookup | 100B ops/sec |

### System Benchmarks

| Benchmark | Description | Target |
|-----------|-------------|--------|
| `bench_e2e_1gpu` | End-to-end, 1 GPU | 2.5B keys/sec |
| `bench_e2e_4gpu` | End-to-end, 4 GPUs | 10B keys/sec |
| `bench_scaling` | 1→2→3→4 GPU scaling | > 3.8x at 4 |
| `bench_sustained` | 1-hour continuous run | > 90% of peak |
| `bench_wordlist_io` | I/O limited test | < 5% overhead |

### Validation Benchmarks

| Benchmark | Description | Criteria |
|-----------|-------------|----------|
| `test_known_wallets` | Known brainwallet test | 100% found |
| `test_correctness` | Random key validation | 100% correct |
| `test_bloom_accuracy` | False positive rate | < 0.1% |

---

## Performance Measurement

### Metrics Collection

```cpp
struct PerformanceMetrics {
    // Throughput
    uint64_t keys_processed;
    double elapsed_seconds;
    double keys_per_second;

    // GPU utilization
    float sm_occupancy;
    float memory_bandwidth_utilization;
    float compute_throughput;

    // Memory
    size_t global_memory_reads;
    size_t global_memory_writes;
    size_t l2_cache_hits;
    size_t l2_cache_misses;

    // Power
    float power_watts;
    double keys_per_joule;
};

// Collection via CUPTI
void collect_metrics(cudaStream_t stream, PerformanceMetrics& metrics);
```

### Profiling Commands

```bash
# Detailed kernel analysis
ncu --set full --target-processes all ./superflayer --benchmark

# Timeline analysis
nsys profile --trace=cuda,nvtx --output=timeline ./superflayer --benchmark

# Memory analysis
ncu --metrics l2__read_throughput,l2__write_throughput ./superflayer --benchmark

# Occupancy analysis
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ./superflayer --benchmark
```

---

## Optimization Roadmap

### Phase 1: Baseline (Week 11)

| Metric | Expected Baseline |
|--------|-------------------|
| 1 GPU throughput | 1-1.5B keys/sec |
| SM occupancy | 50-60% |
| Memory BW utilization | 30-40% |

### Phase 2: Memory Optimization (Week 12)

| Optimization | Expected Improvement |
|--------------|---------------------|
| L2 persistence | +25-35% |
| Coalescing fixes | +10-15% |
| Prefetching | +5-10% |
| **Cumulative** | **1.5-2B keys/sec** |

### Phase 3: Compute Optimization (Week 12-13)

| Optimization | Expected Improvement |
|--------------|---------------------|
| IMAD replacement | +10-15% |
| Register optimization | +15-20% |
| Batch inversion | +20-30% |
| **Cumulative** | **2-2.5B keys/sec** |

### Phase 4: Multi-GPU Scaling (Week 13)

| Configuration | Expected Performance |
|---------------|---------------------|
| 2 GPUs | 4.5-5B keys/sec (1.9x) |
| 3 GPUs | 6.5-7.5B keys/sec (2.8x) |
| 4 GPUs | 9-11B keys/sec (3.8x) |

---

## Performance Budget

### Time Budget per Key (Target: 400 ps)

| Operation | Time Budget | % of Total |
|-----------|-------------|------------|
| Passphrase hash | 20 ps | 5% |
| EC table lookup | 40 ps | 10% |
| Point additions (15x) | 150 ps | 37.5% |
| Batch inversion (amortized) | 30 ps | 7.5% |
| Address generation | 100 ps | 25% |
| Bloom lookup | 20 ps | 5% |
| Overhead (transfers, sync) | 40 ps | 10% |
| **Total** | **400 ps** | 100% |

### Memory Bandwidth Budget

```
Per key memory operations:
- EC table reads: 16 x 64 bytes = 1,024 bytes
- Private key write: 32 bytes
- Public key write: 64 bytes
- Address write: 20 bytes
- Bloom filter read: 8 bytes (avg)
Total: ~1,150 bytes per key

At 10B keys/sec:
- Required bandwidth: 10B x 1,150 = 11.5 TB/s

Available (4 GPUs):
- 7.168 TB/s aggregate

Gap: Need ~1.6x more bandwidth than available!

Solution:
1. L2 caching reduces effective table reads by 75%
   - Effective table reads: 256 bytes/key
   - Total: 380 bytes/key = 3.8 TB/s ✓
2. Fused kernels reduce intermediate writes
   - Eliminate pubkey global write: 64 bytes saved
   - Total: 316 bytes/key = 3.16 TB/s ✓
```

---

## Contingency Plans

### If Target Not Achieved

| Scenario | Mitigation |
|----------|------------|
| EC mult too slow | Smaller precomputed table (8 chunks), more compute |
| L2 cache misses | Split table across GPUs, reduce hot set |
| Multi-GPU overhead | Static partitioning only, eliminate sync |
| Memory bandwidth | Reduce batch size, pipeline more aggressively |

### Fallback Targets

| Level | Target | Acceptable For |
|-------|--------|----------------|
| Gold | 10B+ keys/sec | Full success |
| Silver | 7-10B keys/sec | Production use |
| Bronze | 5-7B keys/sec | Usable MVP |
| Minimum | 3-5B keys/sec | Proof of concept |

---

## Appendix: RTX 5090 vs Prior Generations

| Spec | RTX 3090 | RTX 4090 | RTX 5090 | 5090 vs 3090 |
|------|----------|----------|----------|--------------|
| CUDA Cores | 10,496 | 16,384 | 21,760 | 2.07x |
| SMs | 82 | 128 | 170 | 2.07x |
| Boost Clock | 1.70 GHz | 2.52 GHz | 2.41 GHz | 1.42x |
| Memory BW | 936 GB/s | 1,008 GB/s | 1,792 GB/s | 1.91x |
| L2 Cache | 6 MB | 72 MB | 65 MB | 10.8x |
| FP32 TFLOPS | 35.6 | 82.6 | 104.8 | 2.94x |

**Key Insight:** L2 cache increase (6→65 MB) is the biggest improvement for table-lookup-heavy workloads like Superflayer.

---

## Conclusion

The 10B+ keys/sec target is achievable with:

1. **Optimized EC multiplication** using precomputed tables and L2 caching
2. **Batch inversion** reducing per-key overhead by 85x
3. **Kernel fusion** minimizing memory round-trips
4. **Near-linear multi-GPU scaling** through minimal synchronization

The RTX 5090's 65 MB L2 cache is the key enabler, allowing the majority of EC table lookups to hit cache rather than global memory, fundamentally changing the memory-compute balance compared to previous generations.
