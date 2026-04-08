# Language Decision: Superflayer Implementation

## Executive Summary

**Recommendation: C++20 with CUDA C++**

After evaluating Rust, Go, and C++ for implementing Superflayer, C++ with CUDA emerges as the optimal choice for a GPU-intensive cryptographic application targeting 4x RTX 5090 GPUs. While Rust offers compelling safety guarantees, the maturity of CUDA tooling for C++, zero FFI overhead, and the ability to leverage extensive existing codebases make C++ the pragmatic choice for maximum performance.

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| GPU Performance | 35% | Raw CUDA kernel throughput, memory management |
| Development Velocity | 20% | Time to working implementation |
| Code Reuse | 15% | Ability to port existing implementations |
| Maintainability | 15% | Long-term code quality and safety |
| Ecosystem Maturity | 15% | Tooling, libraries, community support |

---

## Option 1: C++ with CUDA C++

### Overview

The traditional approach using native CUDA with modern C++20 features.

### Technical Analysis

**GPU Performance: 10/10**

```cpp
// Native CUDA kernel - zero abstraction overhead
__global__ void ec_multiply_kernel(
    const uint256_t* __restrict__ scalars,
    ECPoint* __restrict__ results,
    const ECPoint* __restrict__ table
) {
    // Direct PTX generation, maximum optimization potential
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Inline PTX for critical sections
    asm volatile(
        "madc.hi.u32 %0, %1, %2, %3;\n"
        : "=r"(result) : "r"(a), "r"(b), "r"(c)
    );
}
```

- Direct access to all CUDA features (cooperative groups, async copy, tensor cores)
- Full control over memory hierarchy (shared, L2 persistence, registers)
- Inline PTX assembly for micro-optimization
- Mature profiling tools (Nsight, nvprof)

**Development Velocity: 7/10**

```cpp
// Modern C++20 provides ergonomic host code
auto process_batch = [&](std::span<Passphrase> batch) -> std::vector<Result> {
    // RAII CUDA memory management
    CudaBuffer<uint256_t> d_keys(batch.size());

    // Structured bindings, ranges, concepts
    auto [grid, block] = compute_launch_config(batch.size());

    hash_kernel<<<grid, block, 0, stream>>>(d_keys.get(), batch.size());
    return collect_results(d_keys);
};
```

- C++20 features significantly improve code quality
- Still requires manual CUDA memory management
- Compilation times are reasonable with proper structure
- Debugging requires CUDA-specific tools

**Code Reuse: 10/10**

All reference implementations are in C/C++ with CUDA:
- CudaBrainSecp: Direct port of kernel architecture
- BitCrack: Entire secp256k1 library reusable
- Secp256k1-CUDA-ecc: Montgomery ladder implementation
- gECC: Optimization patterns directly applicable

```cpp
// Can directly include existing headers
#include "secp256k1/field.h"
#include "secp256k1/group.h"
#include "secp256k1/ecmult.h"
```

**Maintainability: 6/10**

- Memory safety requires discipline
- Modern C++ (smart pointers, RAII) mitigates risks
- AddressSanitizer/CUDA-memcheck available
- Still prone to subtle bugs (use-after-free, race conditions)

**Ecosystem Maturity: 10/10**

| Component | Status | Notes |
|-----------|--------|-------|
| CUDA Toolkit | Stable | Full Blackwell support (12.x+) |
| cuBLAS/cuFFT | Stable | Optimized primitives |
| Thrust | Stable | Parallel algorithms |
| CUB | Stable | Block-level primitives |
| libsecp256k1 | Stable | Bitcoin reference implementation |
| OpenSSL/BoringSSL | Stable | SHA256/Keccak implementations |

### Pros
- Maximum performance with zero abstraction overhead
- Direct access to all CUDA features
- Extensive existing codebase to leverage
- Mature tooling and debugging support
- Large community and documentation

### Cons
- Memory safety is developer responsibility
- Verbose compared to modern languages
- Build system complexity (CMake + CUDA)
- Header-heavy code slows compilation

---

## Option 2: Rust with rust-gpu/cudarc

### Overview

Rust ecosystem for GPU compute: `rustc_codegen_nvvm` for kernels, `cudarc` for host-side CUDA runtime.

### Technical Analysis

**GPU Performance: 7/10**

```rust
// Device-side kernel (rustc_codegen_nvvm)
#[no_mangle]
pub extern "ptx-kernel" fn ec_multiply_kernel(
    scalars: *const U256,
    results: *mut ECPoint,
    table: *const ECPoint,
    count: u32,
) {
    let tid = cuda::thread_idx_x() + cuda::block_idx_x() * cuda::block_dim_x();
    if tid >= count { return; }

    // LLVM NVVM backend generates PTX
    // Performance typically within 5-15% of hand-tuned CUDA
}

// Host-side (cudarc)
fn main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let module = ctx.load_ptx(include_str!("kernel.ptx"))?;
    let kernel = module.get_function("ec_multiply_kernel")?;

    // Type-safe kernel launch
    kernel.launch(&[&d_scalars, &d_results, &d_table, &count])?;
}
```

**Limitations:**
- rustc_codegen_nvvm is actively developed but not production-stable
- Some CUDA features not yet exposed (cooperative groups, async copy)
- PTX quality depends on LLVM NVVM backend
- No inline PTX assembly support

**Development Velocity: 6/10**

The Rust CUDA project was rebooted in January 2025 after 3+ years dormant:

```
Timeline (from Rust GPU blog):
- Jan 2025: Reboot announced
- Mar 2025: Initial project update
- May 2025: Continued progress
- Aug 2025: Significant improvements
```

Current state:
- Basic kernels work
- Memory management improving
- Still missing advanced features
- Documentation sparse

**Code Reuse: 3/10**

- Cannot directly use C/C++ CUDA codebases
- Must reimplement all kernels in Rust
- Some pure-Rust crypto libraries exist but not GPU-optimized
- Would need to port secp256k1 entirely

```rust
// Would need to reimplement from scratch
mod secp256k1 {
    pub struct FieldElement([u64; 4]);
    pub struct AffinePoint { x: FieldElement, y: FieldElement }

    // ~2000 lines of careful cryptographic code
    impl FieldElement {
        pub fn mul(&self, other: &Self) -> Self { /* ... */ }
        pub fn inv(&self) -> Self { /* ... */ }
    }
}
```

**Maintainability: 9/10**

- Memory safety guaranteed by compiler
- No data races possible
- Fearless concurrency for multi-GPU coordination
- Strong type system catches errors at compile time

**Ecosystem Maturity: 4/10**

| Component | Status | Notes |
|-----------|--------|-------|
| rustc_codegen_nvvm | Active development | Not stable |
| cudarc | Stable | Host-side runtime wrapper |
| cuda-std | Experimental | Device-side stdlib |
| Crypto crates | Stable (CPU) | Not GPU-optimized |
| Big integer | Limited | No GPU-native options |

### Pros
- Memory safety and fearless concurrency
- Modern language ergonomics
- Excellent error handling
- Growing community interest

### Cons
- GPU toolchain not production-ready
- Significant development effort to reimplement kernels
- Performance gap vs native CUDA (5-15%)
- Missing advanced CUDA features
- Limited debugging tools for GPU code

---

## Option 3: Go with gorgonia/cu

### Overview

Go's approach uses CGO bindings to CUDA runtime, with gorgonia providing higher-level abstractions.

### Technical Analysis

**GPU Performance: 5/10**

```go
// Go cannot write GPU kernels directly
// Must use pre-compiled CUDA kernels via CGO

// #cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcuda
// #include <cuda_runtime.h>
// extern void ec_multiply_kernel_wrapper(void* scalars, void* results, int count);
import "C"

func ProcessBatch(scalars []uint256, results []ECPoint) error {
    // CGO overhead on every call
    C.ec_multiply_kernel_wrapper(
        unsafe.Pointer(&scalars[0]),
        unsafe.Pointer(&results[0]),
        C.int(len(scalars)),
    )
    return nil
}
```

**Limitations:**
- Cannot write kernels in Go
- CGO overhead (~100-200ns per call)
- Must maintain separate C++/CUDA codebase
- Memory management split between Go GC and CUDA

**Development Velocity: 5/10**

- Host code in Go is pleasant
- Kernel development still requires C++/CUDA
- CGO debugging is challenging
- Two languages to maintain

**Code Reuse: 6/10**

- Can wrap existing CUDA code
- But must write CGO bindings manually
- Memory ownership is complex
- Serialization overhead for complex types

```go
// gorgonia/cu provides some abstractions
import "gorgonia.org/cu"

func main() {
    dev, _ := cu.GetDevice(0)
    ctx, _ := dev.MakeContext(cu.SchedAuto)
    defer cu.DestroyContext(&ctx)

    // Still need external kernel compilation
    mod, _ := cu.LoadModule("kernels.ptx")
    fn, _ := mod.Function("ec_multiply_kernel")

    // Launch
    fn.Launch(gridDim, blockDim, sharedMem, stream, args...)
}
```

**Maintainability: 7/10**

- Go code is clean and maintainable
- But CUDA kernels still in C++ (same issues)
- CGO boundary is a source of bugs
- GC pauses could impact latency

**Ecosystem Maturity: 5/10**

| Component | Status | Notes |
|-----------|--------|-------|
| gorgonia/cu | Maintained | Basic CUDA driver API |
| gorgonia/gorgonia | Active | ML focus, not crypto |
| CGO | Stable | Overhead concerns |
| Big integer (Go) | Good | math/big, but CPU only |

### Pros
- Simple, clean Go code for coordination
- Good for distributed systems
- Strong concurrency primitives
- Can wrap existing CUDA code

### Cons
- Cannot write GPU kernels in Go
- CGO overhead and complexity
- Must maintain two codebases (Go + CUDA)
- Memory management across boundary is error-prone
- Not suited for low-level GPU optimization

---

## Comparative Analysis

### Performance Benchmark (Estimated)

| Language | Kernel Performance | Host Overhead | Multi-GPU Coord | Overall |
|----------|-------------------|---------------|-----------------|---------|
| C++/CUDA | 100% (baseline) | ~0 | Good | 100% |
| Rust | 85-95% | ~0 | Excellent | 90% |
| Go/CGO | 100% (same kernels) | 5-10% | Excellent | 93% |

### Development Effort

| Task | C++ | Rust | Go |
|------|-----|------|-----|
| Core CUDA kernels | 4 weeks | 8 weeks | 4 weeks (C++) |
| Multi-GPU coordination | 2 weeks | 1 week | 1 week |
| CLI/Configuration | 1 week | 0.5 weeks | 0.5 weeks |
| Testing infrastructure | 1 week | 0.5 weeks | 1 week |
| Debugging/Optimization | 2 weeks | 4 weeks | 3 weeks |
| **Total** | **10 weeks** | **14 weeks** | **9.5 weeks** |

### Risk Assessment

| Risk | C++ | Rust | Go |
|------|-----|------|-----|
| Memory bugs | Medium | Very Low | Medium (CGO) |
| Performance ceiling | Very Low | Low-Medium | Low |
| Toolchain stability | Very Low | Medium-High | Low |
| Maintenance burden | Medium | Low | Medium-High |
| Hiring/Contributors | Easy | Medium | Medium |

---

## Decision Matrix

| Criterion | Weight | C++ | Rust | Go |
|-----------|--------|-----|------|-----|
| GPU Performance | 35% | 10 | 7 | 8 |
| Development Velocity | 20% | 7 | 6 | 7 |
| Code Reuse | 15% | 10 | 3 | 6 |
| Maintainability | 15% | 6 | 9 | 7 |
| Ecosystem Maturity | 15% | 10 | 4 | 5 |
| **Weighted Score** | 100% | **8.55** | **5.95** | **6.85** |

---

## Recommendation: C++ with CUDA C++

### Rationale

1. **Performance is paramount**: Superflayer's value proposition is maximum throughput. C++ provides the best path to peak performance with zero abstraction overhead.

2. **Code reuse accelerates development**: Direct use of CudaBrainSecp, BitCrack, and secp256k1-CUDA-ecc code saves weeks of development.

3. **Mature tooling**: Nsight profiler, cuda-memcheck, and extensive documentation enable efficient optimization.

4. **Risk mitigation**: Proven technology with known behavior. No surprises from evolving toolchains.

### Modern C++ Mitigations

Address C++ weaknesses with modern practices:

```cpp
// 1. RAII for all CUDA resources
template<typename T>
class CudaBuffer {
    T* ptr_ = nullptr;
    size_t size_ = 0;
public:
    CudaBuffer(size_t n) : size_(n) {
        CUDA_CHECK(cudaMalloc(&ptr_, n * sizeof(T)));
    }
    ~CudaBuffer() { if (ptr_) cudaFree(ptr_); }

    // Move-only semantics
    CudaBuffer(CudaBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    T* get() { return ptr_; }
    size_t size() const { return size_; }
};

// 2. std::span for safe array access
void process_batch(std::span<const Passphrase> input,
                   std::span<Result> output);

// 3. Concepts for type safety
template<typename T>
concept CudaType = std::is_trivially_copyable_v<T> &&
                   alignof(T) >= 4;

template<CudaType T>
void copy_to_device(CudaBuffer<T>& dest, std::span<const T> src);

// 4. Expected for error handling
std::expected<Result, CudaError> launch_kernel(/*...*/);
```

### Build System

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.25)
project(superflayer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES "100")  # Blackwell

# Enable sanitizers for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(superflayer PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>
    )
endif()

# CUDA-specific optimizations
target_compile_options(superflayer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        --extra-device-vectorization
        -Xptxas -v  # Show register usage
    >
)
```

### Hybrid Approach (Future Consideration)

If Rust CUDA matures, consider a hybrid:
- **Rust**: Host-side coordination, CLI, configuration
- **C++/CUDA**: Performance-critical kernels

```rust
// Rust host code
use superflayer_cuda::ffi;

fn main() {
    let config = Config::from_args();
    let ctx = ffi::CudaContext::new(config.gpus)?;

    // Call C++ kernel wrapper
    let results = ctx.process_wordlist(&wordlist)?;
}
```

This allows eventual migration while preserving kernel performance.

---

## Implementation Notes

### Directory Structure

```
superflayer/
├── CMakeLists.txt
├── src/
│   ├── main.cpp                 # Entry point
│   ├── config/                  # Configuration parsing
│   ├── host/                    # CPU-side coordination
│   │   ├── work_distributor.cpp
│   │   ├── progress_tracker.cpp
│   │   └── result_collector.cpp
│   ├── cuda/                    # GPU kernels
│   │   ├── kernels/
│   │   │   ├── hash.cu          # SHA256, Keccak
│   │   │   ├── ec_multiply.cu   # Point multiplication
│   │   │   ├── address.cu       # Address generation
│   │   │   └── bloom.cu         # Bloom filter lookup
│   │   ├── memory/
│   │   │   ├── buffer.hpp       # RAII wrappers
│   │   │   └── pool.hpp         # Memory pool
│   │   └── secp256k1/           # EC primitives (ported)
│   │       ├── field.cuh
│   │       ├── group.cuh
│   │       └── ecmult.cuh
│   └── io/                      # File I/O
│       ├── wordlist.cpp
│       └── bloom_filter.cpp
├── tests/                       # Unit and integration tests
├── benchmarks/                  # Performance tests
└── third_party/                 # Dependencies
    ├── secp256k1/               # Bitcoin Core library
    └── xxhash/                  # Fast hashing
```

### Key Dependencies

| Library | Purpose | License |
|---------|---------|---------|
| CUDA Toolkit 12.x+ | GPU runtime | Proprietary (free) |
| libsecp256k1 | EC reference | MIT |
| xxHash | Fast hashing | BSD |
| CLI11 | Argument parsing | BSD |
| spdlog | Logging | MIT |
| nlohmann/json | Configuration | MIT |
| Catch2 | Testing | BSL |

---

## Conclusion

C++ with CUDA C++ is the recommended implementation language for Superflayer. The combination of maximum performance, extensive code reuse opportunities, and mature tooling outweighs the safety benefits of Rust for this performance-critical application. Modern C++20 practices and tooling mitigate traditional C++ risks while preserving the ability to achieve the 10B+ keys/sec performance target.

Future consideration: As Rust CUDA tooling matures (2026+), evaluate a hybrid approach with Rust host code and C++/CUDA kernels.
