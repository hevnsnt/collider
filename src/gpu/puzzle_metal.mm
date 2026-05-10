/**
 * Apple Metal fused brute-force puzzle dispatcher -- implementation.
 *
 * See puzzle_metal.hpp for the public C++ surface. This Obj-C++ file holds
 * the Metal-API state behind an Impl pimpl so the public header stays
 * C++ (callable from MultiGPUPuzzleSolver and the runtime puzzle solver
 * without dragging Foundation / Metal headers into the rest of the build).
 *
 * The dispatcher mirrors the kangaroo_metal.mm pattern: same MSL-source
 * embedding (kangaroo_metal_source.h is shared by both kernels via the
 * generated/ include directory), same MTLCompileOptions math-mode shim,
 * same ARC ownership, same @autoreleasepool around device-touching calls.
 * If you change one, audit the other for drift.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "puzzle_metal.hpp"
#include "../core/byte_codec.hpp"
#include "../core/crypto_cpu.hpp"
#include "puzzle_metal_source.h"

#include <cstring>
#include <sstream>
#include <vector>

namespace collider {
namespace gpu {

namespace {

// Build the 32-window 8-bit-per-window precomputed G table on the CPU. The
// host computes each entry as `d * 2^(8*w) * G` via cpu::ec_mul +
// cpu::ec_to_affine, then packs them in the LE-by-limb form the kernel reads.
//
// Row 0 of every window (d == 0) is the identity, encoded as all-zero
// (X=0, Y=0); the kernel skips zero windows so this row is never dereferenced
// at runtime. We still allocate the row so the indexing math in the kernel
// (`(w * 256 + d) * 8`) lands in-bounds for every byte value.
//
// One-time cost on host: 32 * 255 = 8160 EC scalar multiplications. cpu::ec_mul
// is a simple double-and-add (~256 doublings + ~128 adds per call). On an M1
// at ~2us per ec_mul the total table-build time is ~16 ms. That's a one-time
// init cost, not in the hot path.
//
// Production-grade build pattern: instead of N independent k*G calls, we
// chain incrementally per window:
//   for w in 0..32:
//     base = (1 << (8*w)) * G               // computed once via ec_mul
//     point = identity
//     for d in 1..256:
//       point = point + base                 // mixed-coord add
//       table[w*256 + d] = (X, Y) of point
// This drops the cost from O(W * D * log scalar) to O(W + W * D), a ~256x
// reduction. Still <1 ms on M1.
void build_g_table(std::vector<uint64_t>& out)
{
    out.assign(kPuzzleMetalGTableUlongs, 0ull);

    for (uint32_t w = 0; w < kPuzzleMetalGTableWindows; ++w) {
        // base_w = 2^(8*w) * G
        cpu::uint256_t base_scalar{};
        const uint32_t bit = 8u * w;
        const uint32_t limb = bit / 64u;
        const uint32_t off  = bit % 64u;
        base_scalar.d[limb] = (uint64_t)1 << off;

        cpu::ECPoint base_point;
        cpu::ec_mul(base_point, base_scalar);
        cpu::uint256_t base_x, base_y;
        cpu::ec_to_affine(base_x, base_y, base_point);

        // d=0: identity (already zeroed by out.assign).
        // Start a Jacobian accumulator at (base_x, base_y, 1) for d=1; then
        // accumulate `+= (base_x, base_y)` for d=2..255.
        cpu::ECPoint acc;
        acc.X = base_x;
        acc.Y = base_y;
        acc.Z = cpu::uint256_t(1);

        for (uint32_t d = 1; d < kPuzzleMetalGTableEntries; ++d) {
            cpu::uint256_t ax, ay;
            cpu::ec_to_affine(ax, ay, acc);
            const size_t entry_base = (size_t)(w * kPuzzleMetalGTableEntries + d) * 8u;
            for (int i = 0; i < 4; ++i) out[entry_base + i] = ax.d[i];
            for (int i = 0; i < 4; ++i) out[entry_base + 4 + i] = ay.d[i];

            if (d + 1 < kPuzzleMetalGTableEntries) {
                cpu::ECPoint next;
                cpu::ec_add(next, acc, base_x, base_y);
                acc = next;
            }
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// PuzzleMetalSolver::Impl
// ---------------------------------------------------------------------------

struct PuzzleMetalSolver::Impl {
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;
    id<MTLLibrary>              lib      = nil;
    id<MTLComputePipelineState> pipe     = nil;

    // Persistent device buffers. All MTLResourceStorageModeShared so the host
    // can write/read directly without explicit cudaMemcpy-style sync. On
    // Apple silicon the GPU and CPU share the same physical RAM; "shared"
    // storage mode is the natural fit and avoids the blit-encoder dance
    // that "managed" mode requires.
    id<MTLBuffer> b_gtable    = nil;   // 512 KiB precomputed G table
    id<MTLBuffer> b_target    = nil;   // 20-byte target hash160
    id<MTLBuffer> b_match_lo  = nil;   // 8-byte uint64 (atomic)
    id<MTLBuffer> b_match_hi  = nil;   // 8-byte uint64 (atomic)
    id<MTLBuffer> b_match_fnd = nil;   // 4-byte uint32 (atomic)

    uint64_t batch = kPuzzleMetalDefaultBatchSize;
    std::string device_name_str;
};

PuzzleMetalSolver::PuzzleMetalSolver() : impl_(new Impl{}) {}

PuzzleMetalSolver::~PuzzleMetalSolver() {
    @autoreleasepool {
        delete impl_;
        impl_ = nullptr;
    }
}

std::string PuzzleMetalSolver::device_name() const {
    return impl_ ? impl_->device_name_str : std::string{};
}

uint64_t PuzzleMetalSolver::batch_size() const {
    return impl_ ? impl_->batch : 0ull;
}

uint64_t PuzzleMetalSolver::set_batch_size(uint64_t bs) {
    if (!impl_) return 0ull;
    if (bs == 0) bs = kPuzzleMetalDefaultBatchSize;
    // Round up to a multiple of the threadgroup width so the dispatch grid
    // doesn't spawn partial threadgroups. The kernel itself bounds-checks
    // `gid >= total_keys`, but a partial trailing threadgroup wastes
    // scheduling slots; this rounding keeps dispatch even.
    const uint64_t tg = kPuzzleMetalThreadgroupWidth;
    const uint64_t rounded = ((bs + tg - 1) / tg) * tg;
    impl_->batch = rounded;
    return rounded;
}

bool PuzzleMetalSolver::init() {
    @autoreleasepool {
        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) { error_ = "no Metal device"; return false; }
        impl_->device_name_str = std::string([impl_->device.name UTF8String]);

        NSString* source = [NSString stringWithUTF8String:kPuzzleMetalSource];
        if (!source) {
            error_ = "embedded MSL source is not valid UTF-8 (build corrupt)";
            return false;
        }
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion2_4;
        // Same SDK guard as kangaroo_metal.mm: MTLMathModeFast lives only in
        // macOS 15+ headers. Building against an older SDK (Sonoma CI) means
        // the symbol isn't declared even though @available would skip the
        // call at runtime, so we gate at compile time too.
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15.0, iOS 18.0, *)) {
            opts.mathMode = MTLMathModeFast;
        } else {
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = YES;
            #pragma clang diagnostic pop
        }
#else
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = YES;
        #pragma clang diagnostic pop
#endif
        NSError* err = nil;
        impl_->lib = [impl_->device newLibraryWithSource:source options:opts error:&err];
        if (!impl_->lib) {
            error_ = std::string("MSL compile failed: ")
                   + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }

        id<MTLFunction> fn = [impl_->lib newFunctionWithName:@"puzzle_search"];
        if (!fn) { error_ = "puzzle_search kernel missing from compiled library"; return false; }
        impl_->pipe = [impl_->device newComputePipelineStateWithFunction:fn error:&err];
        if (!impl_->pipe) {
            error_ = std::string("puzzle_search pipeline create failed: ")
                   + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }

        impl_->queue = [impl_->device newCommandQueue];
        if (!impl_->queue) { error_ = "command queue create failed"; return false; }

        // Allocate persistent buffers.
        impl_->b_gtable    = [impl_->device newBufferWithLength:kPuzzleMetalGTableBytes
                                                        options:MTLResourceStorageModeShared];
        impl_->b_target    = [impl_->device newBufferWithLength:20
                                                        options:MTLResourceStorageModeShared];
        impl_->b_match_lo  = [impl_->device newBufferWithLength:sizeof(uint64_t)
                                                        options:MTLResourceStorageModeShared];
        impl_->b_match_hi  = [impl_->device newBufferWithLength:sizeof(uint64_t)
                                                        options:MTLResourceStorageModeShared];
        impl_->b_match_fnd = [impl_->device newBufferWithLength:sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
        if (!impl_->b_gtable || !impl_->b_target || !impl_->b_match_lo
            || !impl_->b_match_hi || !impl_->b_match_fnd)
        {
            error_ = "Metal buffer allocation failed";
            return false;
        }

        // Build the 32-window precomputed G table on the host and copy it
        // into the device buffer. ~16 ms one-time cost on M1.
        std::vector<uint64_t> gtable_host;
        build_g_table(gtable_host);
        std::memcpy([impl_->b_gtable contents], gtable_host.data(), kPuzzleMetalGTableBytes);

        return true;
    }
}

bool PuzzleMetalSolver::set_target(const std::array<uint8_t, 20>& hash160) {
    if (!impl_->b_target) { error_ = "init() not called"; return false; }
    std::memcpy([impl_->b_target contents], hash160.data(), 20);
    return true;
}

bool PuzzleMetalSolver::search_batch(uint64_t start_lo, uint64_t start_hi,
                                     uint64_t batch_size,
                                     uint64_t& found_lo, uint64_t& found_hi)
{
    if (!impl_->pipe) { error_ = "init() not called"; return false; }
    if (batch_size == 0) batch_size = impl_->batch;
    if (batch_size == 0) { error_ = "batch size is zero"; return false; }
    // The kernel's `gid` is a 32-bit thread_position_in_grid value;
    // total_keys must therefore fit in uint32. Real callers pass at
    // most ~4M keys per dispatch (matches CUDA default), so this is
    // never a hot-path limitation. Reject the bad config explicitly
    // rather than silently truncating to UINT32_MAX, which would skip
    // the upper portion of an oversized batch.
    if (batch_size > (uint64_t)UINT32_MAX) {
        error_ = "batch_size exceeds 2^32; reduce per-dispatch size";
        return false;
    }

    // Reset match-result slots before the dispatch.
    *(uint64_t*)[impl_->b_match_lo  contents] = 0ull;
    *(uint64_t*)[impl_->b_match_hi  contents] = 0ull;
    *(uint32_t*)[impl_->b_match_fnd contents] = 0u;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:impl_->pipe];

        const uint64_t total_keys = batch_size;
        [enc setBuffer:impl_->b_gtable    offset:0 atIndex:0];
        [enc setBuffer:impl_->b_target    offset:0 atIndex:1];
        [enc setBytes:&start_lo   length:sizeof(start_lo)   atIndex:2];
        [enc setBytes:&start_hi   length:sizeof(start_hi)   atIndex:3];
        [enc setBytes:&total_keys length:sizeof(total_keys) atIndex:4];
        [enc setBuffer:impl_->b_match_lo  offset:0 atIndex:5];
        [enc setBuffer:impl_->b_match_hi  offset:0 atIndex:6];
        [enc setBuffer:impl_->b_match_fnd offset:0 atIndex:7];

        const NSUInteger tg_pref = kPuzzleMetalThreadgroupWidth;
        const NSUInteger maxTPT  = [impl_->pipe maxTotalThreadsPerThreadgroup];
        const NSUInteger tg      = (tg_pref <= maxTPT) ? tg_pref : maxTPT;

        // batch_size is rounded up to a multiple of tg in set_batch_size.
        // For ad-hoc callers passing an unrounded batch_size we still
        // dispatch the exact thread count and rely on the kernel's
        // `gid >= total_keys` early-exit.
        [enc dispatchThreads:MTLSizeMake(total_keys, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }

    const uint32_t found = *(const uint32_t*)[impl_->b_match_fnd contents];
    if (found != 0u) {
        found_lo = *(const uint64_t*)[impl_->b_match_lo contents];
        found_hi = *(const uint64_t*)[impl_->b_match_hi contents];
        return true;
    }
    found_lo = 0;
    found_hi = 0;
    return false;
}

bool PuzzleMetalSolver::verify_one(uint64_t priv_lo, uint64_t priv_hi,
                                   const std::array<uint8_t, 20>& target_h160)
{
    // Run a single-key dispatch with start = (priv_lo, priv_hi) and
    // batch_size = 1 (rounded up to the threadgroup width by the kernel
    // dispatch). The kernel will hash only thread 0; threads 1..tg-1
    // early-exit on `gid >= total_keys`.
    if (!set_target(target_h160)) return false;
    uint64_t flo = 0, fhi = 0;
    return search_batch(priv_lo, priv_hi, 1, flo, fhi);
}

}  // namespace gpu
}  // namespace collider
