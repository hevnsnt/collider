# Superflayer Implementation Plan

## Overview

This document outlines the phased development plan for Superflayer, a GPU-accelerated cryptocurrency brainwallet recovery tool. The implementation is divided into 6 phases with clear milestones, complexity scores, and success criteria.

**Total Estimated Duration:** 14-18 weeks
**Target:** 10B+ keys/sec across 4x RTX 5090 GPUs

---

## Phase Summary

| Phase | Name | Duration | Complexity | Dependencies |
|-------|------|----------|------------|--------------|
| 1 | Foundation & Scaffolding | 2 weeks | Medium | None |
| 2 | Core Cryptographic Kernels | 4 weeks | Very High | Phase 1 |
| 3 | Multi-GPU Coordination | 2 weeks | High | Phase 2 |
| 4 | I/O Pipeline & Matching | 2 weeks | Medium | Phase 2 |
| 5 | Optimization & Profiling | 3 weeks | Very High | Phase 3, 4 |
| 6 | Testing & Hardening | 2 weeks | Medium | Phase 5 |

---

## Phase 1: Foundation & Scaffolding

**Duration:** 2 weeks
**Complexity Score:** 5/10
**Goal:** Establish project structure, build system, and basic infrastructure

### Week 1: Project Setup

#### Task 1.1: Repository & Build System
**Complexity:** 3/10 | **Duration:** 2 days

- [ ] Initialize Git repository with .gitignore
- [ ] Create CMakeLists.txt with CUDA support
- [ ] Configure for CUDA 12.x+ and SM 100 (Blackwell)
- [ ] Set up directory structure per ARCHITECTURE.md
- [ ] Add dependency management (FetchContent or vcpkg)

```cmake
# Target CMake configuration
cmake_minimum_required(VERSION 3.25)
project(superflayer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "100")  # Blackwell

find_package(CUDAToolkit REQUIRED)
```

#### Task 1.2: RAII CUDA Infrastructure
**Complexity:** 4/10 | **Duration:** 2 days

- [ ] Implement `CudaBuffer<T>` template (GPU memory RAII)
- [ ] Implement `CudaStream` wrapper (async operations)
- [ ] Implement `CudaEvent` wrapper (synchronization)
- [ ] Create `CudaContext` for device management
- [ ] Add error checking macros (`CUDA_CHECK`, `CUDA_ASSERT`)

```cpp
// Core RAII types to implement
template<typename T> class CudaBuffer;
class CudaStream;
class CudaEvent;
class CudaContext;

#define CUDA_CHECK(call) /* ... */
```

#### Task 1.3: Logging & Configuration
**Complexity:** 3/10 | **Duration:** 1 day

- [ ] Integrate spdlog for logging
- [ ] Create configuration parser (CLI11 + YAML)
- [ ] Define configuration schema per ARCHITECTURE.md
- [ ] Implement validation for GPU device IDs

### Week 2: Basic Host Framework

#### Task 1.4: Device Enumeration & Initialization
**Complexity:** 4/10 | **Duration:** 2 days

- [ ] Enumerate available CUDA devices
- [ ] Query device properties (memory, SM count, compute capability)
- [ ] Verify Blackwell architecture (SM 100)
- [ ] Initialize multi-GPU context
- [ ] Implement device selection logic

#### Task 1.5: Memory Pool & Allocation
**Complexity:** 5/10 | **Duration:** 2 days

- [ ] Create memory pool for GPU allocations
- [ ] Implement double-buffering infrastructure
- [ ] Add L2 cache persistence configuration
- [ ] Create pinned host memory allocator
- [ ] Implement memory usage tracking

#### Task 1.6: Basic Test Infrastructure
**Complexity:** 3/10 | **Duration:** 1 day

- [ ] Set up Catch2 test framework
- [ ] Create GPU kernel test harness
- [ ] Add benchmark timing infrastructure
- [ ] Implement simple smoke tests

### Phase 1 Deliverables

- [ ] Compiling project with CUDA support
- [ ] RAII wrappers for all CUDA resources
- [ ] Multi-GPU initialization working
- [ ] Configuration loading from file/CLI
- [ ] Test framework running

---

## Phase 2: Core Cryptographic Kernels

**Duration:** 4 weeks
**Complexity Score:** 9/10
**Goal:** Implement high-performance CUDA kernels for EC operations

### Week 3: Field Arithmetic

#### Task 2.1: 256-bit Integer Arithmetic
**Complexity:** 7/10 | **Duration:** 3 days

- [ ] Port field element representation from secp256k1
- [ ] Implement addition with carry (CUDA-optimized)
- [ ] Implement subtraction with borrow
- [ ] Implement comparison operations
- [ ] Add unit tests with known test vectors

```cpp
// Target struct
struct alignas(32) FieldElement {
    uint64_t d[4];  // 256 bits as 4x64-bit limbs
};
```

#### Task 2.2: Montgomery Multiplication
**Complexity:** 8/10 | **Duration:** 3 days

- [ ] Implement Montgomery reduction for secp256k1 prime
- [ ] Optimize using PTX IMAD instructions
- [ ] Implement modular squaring
- [ ] Add conversion to/from Montgomery form
- [ ] Validate against reference implementation

#### Task 2.3: Modular Inversion
**Complexity:** 8/10 | **Duration:** 2 days

- [ ] Implement extended Euclidean algorithm
- [ ] Optimize using addition chain for secp256k1
- [ ] Create batch inversion (Montgomery's trick)
- [ ] Validate inversion correctness

### Week 4: Point Operations

#### Task 2.4: Point Representation
**Complexity:** 5/10 | **Duration:** 1 day

- [ ] Define affine point structure
- [ ] Define Jacobian/projective point structure
- [ ] Implement point validity checks
- [ ] Create conversion functions

```cpp
struct AffinePoint {
    FieldElement x, y;
};

struct JacobianPoint {
    FieldElement x, y, z;
};
```

#### Task 2.5: Point Addition & Doubling
**Complexity:** 8/10 | **Duration:** 3 days

- [ ] Implement Jacobian point addition
- [ ] Implement Jacobian point doubling
- [ ] Optimize for register usage
- [ ] Handle edge cases (infinity, doubling)
- [ ] Add deferred inversion variants

#### Task 2.6: Precomputed Table Generation
**Complexity:** 7/10 | **Duration:** 2 days

- [ ] Generate 16-chunk precomputed table (CPU)
- [ ] Serialize table to binary format
- [ ] Create table loader for GPU
- [ ] Validate table correctness

### Week 5: Scalar Multiplication

#### Task 2.7: Montgomery Ladder Kernel
**Complexity:** 9/10 | **Duration:** 3 days

- [ ] Implement Montgomery ladder algorithm
- [ ] Integrate precomputed table lookups
- [ ] Add batch processing (multiple keys per kernel)
- [ ] Optimize memory access patterns
- [ ] Profile register usage (<= 128 per thread)

```cuda
__global__ void montgomery_ladder_batch(
    const uint256_t* scalars,
    const ECPoint* table,
    ECPoint* results,
    uint32_t count
);
```

#### Task 2.8: Batch Inversion Integration
**Complexity:** 9/10 | **Duration:** 2 days

- [ ] Implement GAS (Gather-Apply-Scatter) pattern
- [ ] Integrate Montgomery's trick for batch inversion
- [ ] Optimize shared memory usage
- [ ] Reduce warp divergence
- [ ] Profile and measure speedup

### Week 6: Hash & Address Kernels

#### Task 2.9: SHA256 CUDA Kernel
**Complexity:** 6/10 | **Duration:** 2 days

- [ ] Port optimized SHA256 to CUDA
- [ ] Batch multiple hashes per kernel
- [ ] Optimize for throughput vs latency
- [ ] Add unit tests with NIST vectors

#### Task 2.10: Keccak256 CUDA Kernel
**Complexity:** 6/10 | **Duration:** 1 day

- [ ] Port Keccak-256 to CUDA
- [ ] Optimize permutation function
- [ ] Add unit tests

#### Task 2.11: RIPEMD160 CUDA Kernel
**Complexity:** 5/10 | **Duration:** 1 day

- [ ] Port RIPEMD-160 to CUDA
- [ ] Create fused SHA256+RIPEMD160 kernel
- [ ] Add unit tests

#### Task 2.12: Address Generation Kernel
**Complexity:** 5/10 | **Duration:** 1 day

- [ ] Implement Bitcoin address generation (compressed/uncompressed)
- [ ] Implement Ethereum address generation
- [ ] Create configurable address mode
- [ ] Add unit tests

### Phase 2 Deliverables

- [ ] All field arithmetic operations tested
- [ ] Point multiplication generating correct public keys
- [ ] Batch inversion providing expected speedup
- [ ] All hash functions validated
- [ ] Single-GPU end-to-end key generation working

---

## Phase 3: Multi-GPU Coordination

**Duration:** 2 weeks
**Complexity Score:** 7/10
**Goal:** Scale to 4 GPUs with efficient work distribution

### Week 7: Work Distribution

#### Task 3.1: Work Partitioning Strategy
**Complexity:** 6/10 | **Duration:** 2 days

- [ ] Implement static keyspace partitioning
- [ ] Add stride-based distribution option
- [ ] Create work queue for dynamic balancing
- [ ] Handle uneven workload distribution

#### Task 3.2: GPU Context Per-Device
**Complexity:** 6/10 | **Duration:** 2 days

- [ ] Create per-GPU worker threads
- [ ] Initialize independent CUDA contexts
- [ ] Load EC table to each GPU
- [ ] Set up independent streams per GPU

#### Task 3.3: Async Host-Device Communication
**Complexity:** 7/10 | **Duration:** 2 days

- [ ] Implement async batch transfer
- [ ] Set up double buffering per GPU
- [ ] Create stream synchronization logic
- [ ] Profile PCIe transfer overhead

### Week 8: Coordination & Progress

#### Task 3.4: Progress Aggregation
**Complexity:** 5/10 | **Duration:** 2 days

- [ ] Implement atomic progress counters
- [ ] Create per-GPU statistics
- [ ] Aggregate throughput metrics
- [ ] Add real-time progress display

#### Task 3.5: Result Collection
**Complexity:** 5/10 | **Duration:** 1 day

- [ ] Create result queue (lock-free)
- [ ] Implement host-side result verification
- [ ] Add result logging with full details

#### Task 3.6: Checkpointing
**Complexity:** 6/10 | **Duration:** 2 days

- [ ] Define checkpoint file format
- [ ] Implement periodic checkpoint saving
- [ ] Create checkpoint loading/resume
- [ ] Add crash recovery logic

### Phase 3 Deliverables

- [ ] 4-GPU execution working
- [ ] Near-linear scaling (> 3.5x for 4 GPUs)
- [ ] Checkpointing and resume functional
- [ ] Progress reporting accurate

---

## Phase 4: I/O Pipeline & Matching

**Duration:** 2 weeks
**Complexity Score:** 6/10
**Goal:** Complete input/output pipeline and target matching

### Week 9: Input Processing

#### Task 4.1: Wordlist Parser
**Complexity:** 4/10 | **Duration:** 2 days

- [ ] Implement file-based wordlist loading
- [ ] Add stdin streaming support
- [ ] Create memory-mapped file option
- [ ] Handle various line endings

#### Task 4.2: Passphrase Generation Modes
**Complexity:** 6/10 | **Duration:** 2 days

- [ ] Implement ModeBooks (prime x affix)
- [ ] Implement combination mode
- [ ] Implement permutation mode
- [ ] Add passphrase batching for GPU transfer

#### Task 4.3: Async Streaming Pipeline
**Complexity:** 6/10 | **Duration:** 2 days

- [ ] Create producer-consumer queue
- [ ] Implement prefetching
- [ ] Handle backpressure
- [ ] Profile I/O vs compute ratio

### Week 10: Target Matching

#### Task 4.4: Bloom Filter Implementation
**Complexity:** 6/10 | **Duration:** 2 days

- [ ] Port bloom filter to GPU
- [ ] Create bloom filter builder (from address list)
- [ ] Implement parallel bloom lookup
- [ ] Tune false positive rate

```cpp
// Target interface
class GpuBloomFilter {
    size_t bits_;
    uint8_t* filter_;  // GPU memory
public:
    void build(const std::vector<Hash160>& addresses);
    __device__ bool check(const Hash160& addr);
};
```

#### Task 4.5: Exact Match Verification
**Complexity:** 4/10 | **Duration:** 1 day

- [ ] Load sorted address list to GPU
- [ ] Implement binary search on bloom hit
- [ ] Reduce false positives to zero

#### Task 4.6: Result Output
**Complexity:** 3/10 | **Duration:** 1 day

- [ ] Format matched results (privkey, address, passphrase)
- [ ] Write to output file
- [ ] Add optional JSON output
- [ ] Implement real-time match notification

### Phase 4 Deliverables

- [ ] Complete end-to-end pipeline functional
- [ ] All passphrase modes working
- [ ] Bloom filter with < 0.1% false positive rate
- [ ] Results correctly identifying known test targets

---

## Phase 5: Optimization & Profiling

**Duration:** 3 weeks
**Complexity Score:** 9/10
**Goal:** Achieve target 10B+ keys/sec performance

### Week 11: Profiling & Baseline

#### Task 5.1: Comprehensive Profiling
**Complexity:** 7/10 | **Duration:** 3 days

- [ ] Profile with Nsight Compute
- [ ] Identify kernel bottlenecks
- [ ] Measure memory bandwidth utilization
- [ ] Analyze register pressure
- [ ] Create performance baseline

```
Target metrics to collect:
- Achieved occupancy
- Memory throughput (global, shared, L2)
- Compute throughput (FLOPS)
- Warp stall reasons
- Register spills to local memory
```

#### Task 5.2: PTX Analysis
**Complexity:** 8/10 | **Duration:** 2 days

- [ ] Examine generated PTX code
- [ ] Identify suboptimal instruction sequences
- [ ] Measure instruction mix (IMAD vs IADD3)
- [ ] Plan inline PTX optimizations

### Week 12: Kernel Optimization

#### Task 5.3: EC Multiplication Optimization
**Complexity:** 9/10 | **Duration:** 3 days

- [ ] Apply gECC IMAD replacement technique
- [ ] Optimize table lookup coalescing
- [ ] Reduce register pressure
- [ ] Experiment with occupancy vs registers trade-off

#### Task 5.4: L2 Cache Optimization
**Complexity:** 8/10 | **Duration:** 2 days

- [ ] Configure L2 persistent cache for EC table
- [ ] Profile cache hit rate
- [ ] Tune persistent region size
- [ ] Measure bandwidth improvement

```cpp
// L2 persistence configuration
cudaStreamAttrValue attr;
attr.accessPolicyWindow.base_ptr = ec_table;
attr.accessPolicyWindow.num_bytes = 48 * 1024 * 1024;  // 48 MB
attr.accessPolicyWindow.hitRatio = 1.0f;
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
```

### Week 13: Pipeline Optimization

#### Task 5.5: Kernel Fusion
**Complexity:** 8/10 | **Duration:** 2 days

- [ ] Fuse hash + EC multiply where beneficial
- [ ] Fuse address generation + bloom lookup
- [ ] Measure kernel launch overhead reduction

#### Task 5.6: Memory Transfer Optimization
**Complexity:** 6/10 | **Duration:** 2 days

- [ ] Optimize double buffering timing
- [ ] Use CUDA graphs for repeated patterns
- [ ] Minimize synchronization points
- [ ] Profile PCIe utilization

#### Task 5.7: Final Performance Validation
**Complexity:** 7/10 | **Duration:** 2 days

- [ ] Run full benchmark suite
- [ ] Validate 10B+ keys/sec target
- [ ] Document optimization results
- [ ] Create performance regression tests

### Phase 5 Deliverables

- [ ] Performance target achieved (10B+ keys/sec)
- [ ] Profiling documentation complete
- [ ] All optimizations documented with measurements
- [ ] Performance regression test suite

---

## Phase 6: Testing & Hardening

**Duration:** 2 weeks
**Complexity Score:** 5/10
**Goal:** Production-ready quality and reliability

### Week 14: Comprehensive Testing

#### Task 6.1: Unit Test Coverage
**Complexity:** 4/10 | **Duration:** 2 days

- [ ] Achieve > 80% code coverage
- [ ] Add edge case tests
- [ ] Test error handling paths
- [ ] Add fuzzing for input parsing

#### Task 6.2: Integration Tests
**Complexity:** 5/10 | **Duration:** 2 days

- [ ] End-to-end test with known brainwallets
- [ ] Multi-GPU stress test
- [ ] Long-running stability test
- [ ] Memory leak detection

#### Task 6.3: Regression Tests
**Complexity:** 4/10 | **Duration:** 1 day

- [ ] Automated performance benchmarks
- [ ] Correctness validation suite
- [ ] CI/CD pipeline setup

### Week 15: Documentation & Release

#### Task 6.4: Documentation
**Complexity:** 3/10 | **Duration:** 2 days

- [ ] Complete README with usage examples
- [ ] API documentation (Doxygen)
- [ ] Performance tuning guide
- [ ] Troubleshooting guide

#### Task 6.5: Security Review
**Complexity:** 5/10 | **Duration:** 2 days

- [ ] Review for memory safety issues
- [ ] Check for timing side channels
- [ ] Validate cryptographic correctness
- [ ] Document security considerations

#### Task 6.6: Release Preparation
**Complexity:** 3/10 | **Duration:** 1 day

- [ ] Create release build configuration
- [ ] Test on fresh systems
- [ ] Prepare release notes
- [ ] Tag v1.0.0 release

### Phase 6 Deliverables

- [ ] Test coverage > 80%
- [ ] No known bugs or memory leaks
- [ ] Complete documentation
- [ ] Security review complete
- [ ] Release-ready v1.0.0

---

## Risk Mitigation

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Register pressure limits occupancy | High | High | Tune threads/block, split kernels |
| L2 cache insufficient for table | Medium | High | Partial table caching, recomputation |
| PCIe bottleneck | Low | Medium | Larger batches, compression |
| Multi-GPU scaling < 3.5x | Medium | High | Minimize sync, work stealing |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| EC optimization harder than expected | Medium | +2 weeks | Allow buffer in Phase 5 |
| Blackwell quirks | Low | +1 week | Early hardware testing |
| Dependency issues | Low | +1 week | Pin versions, vendoring |

---

## Success Criteria

### Phase Gates

| Phase | Gate Criteria |
|-------|---------------|
| 1 | Project compiles, multi-GPU init works |
| 2 | Single-GPU correctness validated |
| 3 | 4-GPU scaling > 3.5x |
| 4 | End-to-end pipeline finds test targets |
| 5 | 10B+ keys/sec achieved |
| 6 | All tests pass, documentation complete |

### Final Acceptance

- [ ] 10B+ keys/sec on 4x RTX 5090
- [ ] Correctness verified against reference implementations
- [ ] All passphrase modes functional
- [ ] Multi-day stability test passed
- [ ] Documentation complete
- [ ] Release v1.0.0 tagged

---

## Resource Requirements

### Hardware

- 4x NVIDIA RTX 5090 GPUs
- Host system with PCIe 5.0 x16 slots
- 128+ GB system RAM
- NVMe storage for wordlists

### Software

- CUDA Toolkit 12.8+
- CMake 3.25+
- GCC 13+ or Clang 17+
- Nsight Compute/Systems

### Team

- 1-2 CUDA/cryptography specialists
- ~40-60 person-hours per week

---

## Appendix: Task Dependency Graph

```
Phase 1 ─────────────────────────────────────────────────────────────▶
    │
    ▼
Phase 2 ─────────────────────────────────────────────────────────────▶
    │                                                      │
    ├──────────────────────────┐                          │
    ▼                          ▼                          ▼
Phase 3                    Phase 4                    (parallel)
    │                          │
    └──────────────────────────┴──────────────────────────┐
                                                          ▼
                                                      Phase 5
                                                          │
                                                          ▼
                                                      Phase 6
```

All tasks within a phase can be parallelized by multiple developers where dependencies allow.
