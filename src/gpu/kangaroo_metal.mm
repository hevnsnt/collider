/**
 * Apple Metal Pollard's Kangaroo dispatcher -- implementation.
 *
 * See kangaroo_metal.hpp for the public C++ surface. This Obj-C++ file
 * holds the Metal-API state behind an Impl pimpl so the public header
 * stays C++ (callable from main.cpp / pool client without Foundation).
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "kangaroo_metal.hpp"
#include "../core/byte_codec.hpp"
#include "kangaroo_metal_source.h"

#include <chrono>
#include <cstring>

namespace collider {
namespace gpu {

// Pimpl payload. All `id<MTL...>` fields are strong references owned by
// ARC -- this file is compiled with -fobjc-arc so when ~Impl runs
// (via `delete impl_;` in ~KangarooMetalSolver), each strong ref is
// auto-released. Do NOT downgrade this file to plain C++ compilation;
// the strong refs would leak. Wrapping the destructor body in
// @autoreleasepool keeps any released-by-ARC objects from accumulating
// in the surrounding pool.
struct KangarooMetalSolver::Impl {
    id<MTLDevice>               device      = nil;
    id<MTLCommandQueue>         queue       = nil;
    id<MTLLibrary>              lib         = nil;
    id<MTLComputePipelineState> pipe_step   = nil;
    id<MTLComputePipelineState> pipe_priv2pub = nil;

    KangarooMetalConfig cfg;
    uint64_t work_id_current = 0;

    // Persistent device buffers.
    id<MTLBuffer> b_x        = nil;   // num_kangaroos * 4 ulong
    id<MTLBuffer> b_y        = nil;
    id<MTLBuffer> b_d        = nil;
    id<MTLBuffer> b_type     = nil;   // num_kangaroos uchar
    id<MTLBuffer> b_jump_x   = nil;   // 32 * 4 ulong
    id<MTLBuffer> b_jump_y   = nil;
    id<MTLBuffer> b_jump_d   = nil;
    // DP records and counters are double-buffered so the GPU can start
    // round N+1 while the host is still reading round N's DPs out of
    // the previous slot. Host alternates between slots 0 and 1; the
    // active slot is `slot_active` below. `pending_cmd` holds the
    // pre-submitted next-round command buffer; on the next step_round
    // call we wait on it instead of submitting fresh, then pre-submit
    // the round after that. State buffers (b_x/b_y/b_d) are NOT
    // double-buffered because round N+1 strictly reads round N's
    // final state -- pre-submitting only overlaps DP-extraction with
    // GPU compute, not state-mutation.
    id<MTLBuffer> b_dp_recs[2]  = { nil, nil };
    id<MTLBuffer> b_dp_count[2] = { nil, nil };
    id<MTLCommandBuffer> pending_cmd = nil;
    int slot_active = 0;          // slot the pending_cmd will write into

    std::string device_name_str;

    // Submit one kangaroo step round to the queue, writing DP records
    // to the given slot (0 or 1). Returns the committed command buffer.
    // Used by the pipelined step_round() to overlap GPU compute with
    // host-side DP extraction.
    id<MTLCommandBuffer> submit_step(int slot) {
        *(uint32_t*)[b_dp_count[slot] contents] = 0;

        const uint32_t cnt   = cfg.num_kangaroos;
        const uint32_t steps = cfg.steps_per_round;
        const uint32_t dp    = cfg.dp_bits;
        const uint64_t wid   = work_id_current;
        const uint32_t dpmax = cfg.dp_max_per_round;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe_step];
        [enc setBuffer:b_x       offset:0 atIndex:0];
        [enc setBuffer:b_y       offset:0 atIndex:1];
        [enc setBuffer:b_d       offset:0 atIndex:2];
        [enc setBuffer:b_type    offset:0 atIndex:3];
        [enc setBuffer:b_jump_x  offset:0 atIndex:4];
        [enc setBuffer:b_jump_y  offset:0 atIndex:5];
        [enc setBuffer:b_jump_d  offset:0 atIndex:6];
        [enc setBytes:&cnt   length:sizeof(cnt)   atIndex:7];
        [enc setBytes:&steps length:sizeof(steps) atIndex:8];
        [enc setBytes:&dp    length:sizeof(dp)    atIndex:9];
        [enc setBytes:&wid   length:sizeof(wid)   atIndex:10];
        [enc setBuffer:b_dp_recs[slot]  offset:0 atIndex:11];
        [enc setBuffer:b_dp_count[slot] offset:0 atIndex:12];
        [enc setBytes:&dpmax length:sizeof(dpmax) atIndex:13];

        const NSUInteger tg = MIN((NSUInteger)32,
                                  [pipe_step maxTotalThreadsPerThreadgroup]);
        [enc dispatchThreads:MTLSizeMake(cnt, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        return cmd;
    }
};

KangarooMetalSolver::KangarooMetalSolver() : impl_(new Impl{}) {}

KangarooMetalSolver::~KangarooMetalSolver() {
    @autoreleasepool {
        // Drain any pending pre-submitted round before tearing down
        // the buffers it references. Otherwise ARC could release the
        // command queue while the GPU still holds references to its
        // buffers, leading to use-after-free on the device side.
        if (impl_ && impl_->pending_cmd) {
            [impl_->pending_cmd waitUntilCompleted];
            impl_->pending_cmd = nil;
        }
        delete impl_;
        impl_ = nullptr;
    }
}

std::string KangarooMetalSolver::device_name() const {
    return impl_ ? impl_->device_name_str : std::string{};
}

bool KangarooMetalSolver::init(const KangarooMetalConfig& cfg) {
    @autoreleasepool {
        impl_->cfg = cfg;
        impl_->work_id_current = cfg.work_id;

        // Auto-size dp_max_per_round if the caller left it at 0. A round
        // produces (num_kangaroos * steps_per_round) walk ops; the
        // expected DP count is that count divided by 2^dp_bits. Reserve
        // 4x the expected to absorb statistical bursts; clamp to the
        // floor so tiny configurations still allocate a usable buffer.
        if (impl_->cfg.dp_max_per_round == 0) {
            const uint64_t round_ops =
                static_cast<uint64_t>(impl_->cfg.num_kangaroos) *
                static_cast<uint64_t>(impl_->cfg.steps_per_round);
            const uint32_t shift = impl_->cfg.dp_bits;
            uint64_t expected = (shift >= 64) ? 0 : (round_ops >> shift);
            uint64_t reserve  = expected * 4;
            if (reserve > UINT32_MAX) reserve = UINT32_MAX;
            if (reserve < kMinDpMaxPerRound) reserve = kMinDpMaxPerRound;
            impl_->cfg.dp_max_per_round = static_cast<uint32_t>(reserve);
        }

        impl_->device = MTLCreateSystemDefaultDevice();
        if (!impl_->device) { error_ = "no Metal device"; return false; }
        impl_->device_name_str = std::string([impl_->device.name UTF8String]);

        NSString* source = [NSString stringWithUTF8String:kKangarooMetalSource];
        if (!source) {
            error_ = "embedded MSL source is not valid UTF-8 (build corrupt)";
            return false;
        }
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion2_4;
        // SDK guard: MTLMathModeFast lives only in macOS 15+ headers.
        // Building against an older SDK (Mac CI runs Sonoma) means the
        // symbol isn't declared even though @available would skip the
        // call at runtime. Gate at compile time too.
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

        id<MTLFunction> fn_step = [impl_->lib newFunctionWithName:@"kangaroo_step"];
        if (!fn_step) { error_ = "kangaroo_step kernel missing"; return false; }
        impl_->pipe_step = [impl_->device newComputePipelineStateWithFunction:fn_step
                                                                         error:&err];
        if (!impl_->pipe_step) {
            error_ = std::string("kangaroo_step pipeline create failed: ")
                   + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }

        id<MTLFunction> fn_pub = [impl_->lib newFunctionWithName:@"priv_to_pub"];
        if (fn_pub) {
            impl_->pipe_priv2pub = [impl_->device newComputePipelineStateWithFunction:fn_pub
                                                                                  error:&err];
        }

        impl_->queue = [impl_->device newCommandQueue];

        // Sizes derived from impl_->cfg (which may have been auto-sized
        // above) rather than the original `cfg`, so subsequent rounds
        // see the same buffer dimensions.
        const size_t N        = impl_->cfg.num_kangaroos;
        const size_t kJumps   = kJumpTableSize;          // 32, see hpp
        const size_t kDpBytes = 8 + 32 + 32 + 1 + 1;     // 74, JLP DP_BATCH_V2 layout
        const size_t kLimbBytes = 4 * 8;                 // uint64_t[4]

        impl_->b_x        = [impl_->device newBufferWithLength:N * kLimbBytes
                                                        options:MTLResourceStorageModeShared];
        impl_->b_y        = [impl_->device newBufferWithLength:N * kLimbBytes
                                                        options:MTLResourceStorageModeShared];
        impl_->b_d        = [impl_->device newBufferWithLength:N * kLimbBytes
                                                        options:MTLResourceStorageModeShared];
        impl_->b_type     = [impl_->device newBufferWithLength:N
                                                        options:MTLResourceStorageModeShared];
        impl_->b_jump_x   = [impl_->device newBufferWithLength:kJumps * kLimbBytes
                                                        options:MTLResourceStorageModeShared];
        impl_->b_jump_y   = [impl_->device newBufferWithLength:kJumps * kLimbBytes
                                                        options:MTLResourceStorageModeShared];
        impl_->b_jump_d   = [impl_->device newBufferWithLength:kJumps * kLimbBytes
                                                        options:MTLResourceStorageModeShared];
        // Double-buffered DP records + counters for command-buffer pipelining.
        for (int s = 0; s < 2; ++s) {
            impl_->b_dp_recs[s]  = [impl_->device newBufferWithLength:impl_->cfg.dp_max_per_round * kDpBytes
                                                              options:MTLResourceStorageModeShared];
            impl_->b_dp_count[s] = [impl_->device newBufferWithLength:sizeof(uint32_t)
                                                              options:MTLResourceStorageModeShared];
        }
        return true;
    }
}

bool KangarooMetalSolver::set_jump_table(const std::array<KangarooSeed, kJumpTableSize>& jumps)
{
    if (!impl_->b_jump_x) { error_ = "init() not called"; return false; }
    uint64_t* jx = (uint64_t*)[impl_->b_jump_x contents];
    uint64_t* jy = (uint64_t*)[impl_->b_jump_y contents];
    uint64_t* jd = (uint64_t*)[impl_->b_jump_d contents];
    for (size_t i = 0; i < kJumpTableSize; ++i) {
        ::collider::be32_to_limbs_le(jumps[i].x.data(), jx + i * 4);
        ::collider::be32_to_limbs_le(jumps[i].y.data(), jy + i * 4);
        ::collider::be32_to_limbs_le(jumps[i].d.data(), jd + i * 4);
    }
    return true;
}

bool KangarooMetalSolver::seed_kangaroos(const std::vector<KangarooSeed>& seeds)
{
    if (!impl_->b_x) { error_ = "init() not called"; return false; }
    if ((uint32_t)seeds.size() != impl_->cfg.num_kangaroos) {
        error_ = "seed count != num_kangaroos";
        return false;
    }
    uint64_t* x = (uint64_t*)[impl_->b_x contents];
    uint64_t* y = (uint64_t*)[impl_->b_y contents];
    uint64_t* d = (uint64_t*)[impl_->b_d contents];
    uint8_t*  t = (uint8_t*) [impl_->b_type contents];
    for (size_t i = 0; i < seeds.size(); ++i) {
        ::collider::be32_to_limbs_le(seeds[i].x.data(), x + i * 4);
        ::collider::be32_to_limbs_le(seeds[i].y.data(), y + i * 4);
        ::collider::be32_to_limbs_le(seeds[i].d.data(), d + i * 4);
        t[i] = seeds[i].type;
    }
    return true;
}

bool KangarooMetalSolver::replace_seed(uint32_t index, const KangarooSeed& seed) {
    if (!impl_->b_x) { error_ = "init() not called"; return false; }
    if (index >= impl_->cfg.num_kangaroos) {
        error_ = "replace_seed index out of range";
        return false;
    }
    uint64_t* x = (uint64_t*)[impl_->b_x contents];
    uint64_t* y = (uint64_t*)[impl_->b_y contents];
    uint64_t* d = (uint64_t*)[impl_->b_d contents];
    uint8_t*  t = (uint8_t*) [impl_->b_type contents];
    ::collider::be32_to_limbs_le(seed.x.data(), x + index * 4);
    ::collider::be32_to_limbs_le(seed.y.data(), y + index * 4);
    ::collider::be32_to_limbs_le(seed.d.data(), d + index * 4);
    t[index] = seed.type;
    return true;
}

bool KangarooMetalSolver::find_dead_kangaroos(std::vector<uint32_t>& out_dead) {
    if (!impl_->b_x) { error_ = "init() not called"; return false; }
    out_dead.clear();
    const uint64_t* x = (const uint64_t*)[impl_->b_x contents];
    const uint64_t* y = (const uint64_t*)[impl_->b_y contents];
    const uint32_t n = impl_->cfg.num_kangaroos;
    // A kangaroo is "dead" only when both (x, y) are entirely zero --
    // matches the identity arm of point_op() in kangaroo.metal:375.
    // (0, 0) is not on the curve so it's safe to use as a sentinel; a
    // legitimate walk will never visit it.
    for (uint32_t i = 0; i < n; ++i) {
        const bool x_zero = (x[i*4+0] == 0) && (x[i*4+1] == 0)
                         && (x[i*4+2] == 0) && (x[i*4+3] == 0);
        const bool y_zero = (y[i*4+0] == 0) && (y[i*4+1] == 0)
                         && (y[i*4+2] == 0) && (y[i*4+3] == 0);
        if (x_zero && y_zero) out_dead.push_back(i);
    }
    return true;
}

void KangarooMetalSolver::set_work_id(uint64_t work_id) {
    impl_->work_id_current = work_id;
}

std::vector<KangarooMetalDP> KangarooMetalSolver::step_round()
{
    std::vector<KangarooMetalDP> out;
    if (!impl_->pipe_step) {
        // init() either was never called or failed; without the pipeline
        // there's nothing to dispatch. Set a sticky error so a caller that
        // forgets to check msolver.init()'s bool sees something on the
        // next inspection of error() instead of an endless stream of
        // empty rounds.
        error_ = "step_round called before successful init()";
        return out;
    }

    @autoreleasepool {
        // Pipelined dispatch: if a previous step_round pre-submitted a
        // round (pending_cmd != nil), wait on that one. Otherwise
        // submit fresh. Either way, after this block `cmd` is the
        // committed-and-finished round whose DPs we extract.
        id<MTLCommandBuffer> cmd;
        int read_slot;
        if (impl_->pending_cmd) {
            cmd = impl_->pending_cmd;
            read_slot = impl_->slot_active;
            impl_->pending_cmd = nil;
        } else {
            // Cold start: no pre-submitted round. Submit and wait inline.
            read_slot = 0;
            impl_->slot_active = 0;
            cmd = impl_->submit_step(read_slot);
        }
        [cmd waitUntilCompleted];

        // Pre-submit the NEXT round to the alternate slot. State
        // buffers (b_x/b_y/b_d) are shared, but the kernel writes
        // them only after reading them, and the queue serializes
        // command buffers, so the pre-submitted round will see the
        // post-`cmd` state. DP records go to the alternate slot so
        // the host can read this round's DPs from `read_slot` while
        // GPU writes the next round's DPs to `1 - read_slot`.
        const int next_slot = 1 - read_slot;
        impl_->slot_active = next_slot;
        impl_->pending_cmd = impl_->submit_step(next_slot);

        const uint32_t dpmax = impl_->cfg.dp_max_per_round;
        const uint32_t raw_found =
            *(uint32_t*)[impl_->b_dp_count[read_slot] contents];
        uint32_t found = raw_found;
        if (found > dpmax) {
            // Kernel atomic ran past dp_max_per_round; the records past
            // the cap were silently discarded by the kernel's
            // `if (slot < dp_max)` guard. Surface this as a sticky error
            // so the caller can grow the buffer for the next round
            // (e.g., via a future reconfigure path) instead of reading
            // empty rate from the pool. Truncate to keep memory access
            // in bounds.
            std::ostringstream oss;
            oss << "DP buffer overflow: " << raw_found
                << " produced, " << dpmax << " absorbed (dp_max_per_round)";
            error_ = oss.str();
            found = dpmax;
        }
        // Wire layout from the kernel struct DPRecord:
        //   ulong work_id; uchar x_be[32]; uchar d_be[32]; uchar type; uchar dp_bits;
        // Total: 8 + 32 + 32 + 1 + 1 = 74 bytes (matches JLPDistinguishedPointV2).
        const uint8_t* recs = (const uint8_t*)[impl_->b_dp_recs[read_slot] contents];
        out.reserve(found);
        for (uint32_t i = 0; i < found; ++i) {
            KangarooMetalDP dp_out;
            const uint8_t* rec = recs + (size_t)i * 74;
            std::memcpy(&dp_out.work_id, rec, 8);
            std::memcpy(dp_out.x_be, rec + 8, 32);
            std::memcpy(dp_out.d_be, rec + 40, 32);
            dp_out.type    = rec[72];
            dp_out.dp_bits = rec[73];
            out.push_back(dp_out);
        }
    }
    return out;
}

}  // namespace gpu
}  // namespace collider
