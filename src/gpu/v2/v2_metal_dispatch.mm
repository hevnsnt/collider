/**
 * Brain Wallet v2 -- Apple Metal runtime dispatch.
 *
 * Compiled on macOS (Pro builds) to drive the brain_wallet_v2.metal
 * shaders from the host. The CUDA equivalent lives in v2_orchestrator.cpp's
 * `__CUDA__` branch; this file is the Mac analogue.
 *
 * Public C++ entry point (declared in v2_metal_dispatch.hpp):
 *   bool v2_metal_run_puzzle_only(...)
 *
 * Algorithm:
 *   1. Load brain_wallet_v2.metal from disk (the unmangled MSL source --
 *      we compile it at runtime via newLibraryWithSource so we don't need
 *      a build-time metallib step).
 *   2. Create a compute pipeline for `v2_puzzle_only_multi_scheme`.
 *   3. Encode an MTLBuffer per kernel arg (passphrases, offsets, lengths,
 *      targets, target_count, matches, match_count, pw_count, scheme_id).
 *   4. For each enabled scheme bit, set scheme_id = bit_index and dispatch
 *      one batch of `count` threads; barrier between dispatches.
 *   5. Read back the match buffer + count.
 *
 * Validation status: written from the metal_platform.mm patterns + the
 * v2 .metal kernel signature; not yet exercised on a Mac (development
 * box is Windows). Mac validation is the integration test.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "brain_wallet_v2.hpp"
#include "v2_metal_dispatch.hpp"

namespace collider {
namespace gpu {
namespace v2 {
namespace metal {

namespace {

// Read brain_wallet_v2.metal from disk. We resolve relative to the
// executable's directory so a deployed app can carry the .metal next
// to its binary (or under a Resources/ subdir on a real .app bundle).
NSString* load_shader_source() {
    NSBundle* main = [NSBundle mainBundle];
    NSString* candidates[] = {
        [main pathForResource:@"brain_wallet_v2" ofType:@"metal"],
        [[main bundlePath] stringByAppendingPathComponent:@"brain_wallet_v2.metal"],
        @"./brain_wallet_v2.metal",
        @"./src/gpu/v2/brain_wallet_v2.metal",
    };
    for (NSString* p : candidates) {
        if (!p) continue;
        NSError* err = nil;
        NSString* src = [NSString stringWithContentsOfFile:p
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
        if (src) return src;
    }
    return nil;
}

// Iterate set bits of `mask`, calling cb(bit_index) for each.
template <class F>
void for_each_set_bit(uint32_t mask, F&& cb) {
    while (mask) {
        uint32_t b = __builtin_ctz(mask);
        cb(b);
        mask &= mask - 1;
    }
}

}  // namespace

bool v2_metal_run_puzzle_only(
    const std::vector<PuzzleTarget>& targets,
    const std::vector<uint8_t>& passphrases_packed,
    const std::vector<uint32_t>& offsets,
    const std::vector<uint32_t>& lengths,
    uint32_t scheme_mask,
    std::vector<V2MatchRecord>& matches_out,
    std::string& error_out)
{
    error_out.clear();
    matches_out.clear();
    if (offsets.size() != lengths.size()) {
        error_out = "offsets / lengths size mismatch";
        return false;
    }
    if (targets.empty() || offsets.empty() || scheme_mask == 0) {
        error_out = "empty inputs";
        return false;
    }

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            error_out = "no Metal device";
            return false;
        }

        NSString* source = load_shader_source();
        if (!source) {
            error_out = "could not locate brain_wallet_v2.metal";
            return false;
        }

        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion2_4;
        opts.fastMathEnabled = YES;
        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&err];
        if (!lib) {
            error_out = std::string("MSL compile failed: ")
                      + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }

        id<MTLFunction> fn =
            [lib newFunctionWithName:@"v2_puzzle_only_multi_scheme"];
        if (!fn) {
            error_out = "kernel v2_puzzle_only_multi_scheme not found in lib";
            return false;
        }

        id<MTLComputePipelineState> pipe =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) {
            error_out = std::string("pipeline create failed: ")
                      + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return false;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Buffer 0: passphrases (packed bytes)
        id<MTLBuffer> b_pass = [device newBufferWithBytes:passphrases_packed.data()
                                                   length:passphrases_packed.size()
                                                  options:MTLResourceStorageModeShared];
        // Buffer 1, 2: offsets / lengths
        id<MTLBuffer> b_off = [device newBufferWithBytes:offsets.data()
                                                  length:offsets.size() * sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_len = [device newBufferWithBytes:lengths.data()
                                                  length:lengths.size() * sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];
        // Buffer 3, 4: targets + count
        id<MTLBuffer> b_tgt = [device newBufferWithBytes:targets.data()
                                                  length:targets.size() * sizeof(PuzzleTarget)
                                                 options:MTLResourceStorageModeShared];
        uint32_t target_count = (uint32_t)targets.size();
        id<MTLBuffer> b_tgtc = [device newBufferWithBytes:&target_count
                                                   length:sizeof(uint32_t)
                                                  options:MTLResourceStorageModeShared];
        // Buffer 5: matches output
        const uint32_t kMaxMatches = V2_MAX_MATCHES_PER_BATCH;
        id<MTLBuffer> b_match = [device newBufferWithLength:kMaxMatches * sizeof(V2MatchRecord)
                                                    options:MTLResourceStorageModeShared];
        std::memset([b_match contents], 0, kMaxMatches * sizeof(V2MatchRecord));
        // Buffer 6: match counter (uint32 atomic)
        id<MTLBuffer> b_count = [device newBufferWithLength:sizeof(uint32_t)
                                                    options:MTLResourceStorageModeShared];
        *(uint32_t*)[b_count contents] = 0;
        // Buffer 7: pw_count
        uint32_t pw_count = (uint32_t)offsets.size();
        id<MTLBuffer> b_pwc = [device newBufferWithBytes:&pw_count
                                                  length:sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];

        // Per scheme: set buffer 8 = scheme_id and dispatch.
        //
        // Pipelining: every kernel writes into the SAME match-count
        // atomic and matches buffer, but the dispatches are otherwise
        // independent. Commit each command buffer immediately and only
        // call waitUntilCompleted on the LAST one -- Metal then chains
        // the GPU work asynchronously rather than serializing CPU
        // round-trips per scheme. (Gemini PR #15 MED finding.)
        bool any_dispatched = false;
        std::vector<id<MTLCommandBuffer>> in_flight;
        // Hold the per-scheme scheme_id buffer alive until the matching
        // command buffer completes; without this the autorelease pool
        // could free it while the GPU is still reading it.
        std::vector<id<MTLBuffer>> live_sid_buffers;
        for_each_set_bit(scheme_mask, [&](uint32_t bit) {
            uint32_t scheme_id = bit;
            id<MTLBuffer> b_sid = [device newBufferWithBytes:&scheme_id
                                                      length:sizeof(uint32_t)
                                                     options:MTLResourceStorageModeShared];
            live_sid_buffers.push_back(b_sid);

            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:b_pass   offset:0 atIndex:0];
            [enc setBuffer:b_off    offset:0 atIndex:1];
            [enc setBuffer:b_len    offset:0 atIndex:2];
            [enc setBuffer:b_tgt    offset:0 atIndex:3];
            [enc setBuffer:b_tgtc   offset:0 atIndex:4];
            [enc setBuffer:b_match  offset:0 atIndex:5];
            [enc setBuffer:b_count  offset:0 atIndex:6];
            [enc setBuffer:b_pwc    offset:0 atIndex:7];
            [enc setBuffer:b_sid    offset:0 atIndex:8];

            // 32 threads per threadgroup matches CUDA's tighter MT-block
            // sizing; SHA-256 schemes are register-light.
            MTLSize tg = MTLSizeMake(32, 1, 1);
            MTLSize grid = MTLSizeMake(pw_count, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cmd commit];
            in_flight.push_back(cmd);
            any_dispatched = true;
        });

        // Wait once for all commit'd buffers to finish. Iterating
        // in commit order is not strictly required (Metal serializes
        // them through the queue), but waiting on the last one is
        // enough -- waitUntilCompleted is a barrier on the queue's
        // entire history before that command buffer.
        if (!in_flight.empty()) {
            [in_flight.back() waitUntilCompleted];
        }
        // live_sid_buffers' MTLBuffers go out of scope at function exit
        // (after we've memcpy'd matches out below); ARC releases them
        // there.

        if (!any_dispatched) {
            error_out = "no scheme bits set";
            return false;
        }

        uint32_t hits = *(uint32_t*)[b_count contents];
        if (hits > kMaxMatches) hits = kMaxMatches;
        matches_out.resize(hits);
        if (hits > 0) {
            std::memcpy(matches_out.data(), [b_match contents],
                        hits * sizeof(V2MatchRecord));
        }
    }
    return true;
}

}  // namespace metal
}  // namespace v2
}  // namespace gpu
}  // namespace collider
