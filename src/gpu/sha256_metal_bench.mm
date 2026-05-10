/**
 * Metal SHA-256 throughput benchmark - Obj-C++ host side.
 *
 * Loads src/gpu/sha256.metal at runtime via newLibraryWithSource,
 * dispatches the `sha256_bench` kernel in a tight loop, measures
 * sustained H/s. Free-licensed; available in both Free and Pro Mac
 * builds.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "sha256_metal_bench.hpp"

#include <chrono>
#include <fstream>
#include <sstream>
#include <vector>

namespace collider {
namespace gpu {

namespace {

NSString* load_shader_source(std::string& err_out) {
    NSBundle* main = [NSBundle mainBundle];
    NSString* candidates[] = {
        [main pathForResource:@"sha256" ofType:@"metal"],
        [[main bundlePath] stringByAppendingPathComponent:@"sha256.metal"],
        @"./sha256.metal",
        @"./src/gpu/sha256.metal",
        @"./build/sha256.metal",
    };
    for (NSString* p : candidates) {
        if (!p) continue;
        NSError* err = nil;
        NSString* src = [NSString stringWithContentsOfFile:p
                                                  encoding:NSUTF8StringEncoding
                                                     error:&err];
        if (src) return src;
    }
    err_out = "could not locate sha256.metal (looked in bundle, ./, ./src/gpu/, ./build/)";
    return nil;
}

}  // namespace

Sha256MetalBenchResult run_sha256_metal_benchmark(int seconds, uint32_t batch_size)
{
    Sha256MetalBenchResult res;
    if (seconds <= 0) seconds = 5;
    if (batch_size == 0) batch_size = 1u << 18;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { res.error = "no Metal device"; return res; }
        res.device_name = std::string([device.name UTF8String]);

        NSString* source = load_shader_source(res.error);
        if (!source) return res;

        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion2_4;
        // macOS 15 deprecated `fastMathEnabled` in favor of `mathMode`. The
        // `@available(...)` check below gates the call at runtime, but the
        // SYMBOL itself (MTLMathModeFast) only exists in the macOS 15+ SDK
        // headers -- compiling against a 14- SDK fails before we ever reach
        // the runtime check. Wrap the new-API arm in
        // __MAC_OS_X_VERSION_MAX_ALLOWED so older SDKs (e.g. CI runners on
        // Sonoma) skip the symbol entirely. (Caught by Mac CI on PR #15.)
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
        // Pre-macOS-15 SDK: the new API isn't declared, so just use the
        // (still-functional) deprecated fastMathEnabled property.
        #pragma clang diagnostic push
        #pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = YES;
        #pragma clang diagnostic pop
#endif
        NSError* err = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:source options:opts error:&err];
        if (!lib) {
            res.error = std::string("MSL compile failed: ")
                      + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return res;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"sha256_bench"];
        if (!fn) { res.error = "kernel sha256_bench not found"; return res; }
        id<MTLComputePipelineState> pipe =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) {
            res.error = std::string("pipeline create failed: ")
                      + (err ? [[err localizedDescription] UTF8String] : "unknown");
            return res;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // 64 bytes per input; pre-fill with a deterministic pattern.
        std::vector<uint8_t> inputs((size_t)batch_size * 64);
        for (size_t i = 0; i < inputs.size(); ++i)
            inputs[i] = (uint8_t)(i & 0xff);

        id<MTLBuffer> b_in  = [device newBufferWithBytes:inputs.data()
                                                  length:inputs.size()
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_out = [device newBufferWithLength:(size_t)batch_size * 32
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_n   = [device newBufferWithBytes:&batch_size
                                                  length:sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];

        // Warmup dispatch (avoids first-launch cost in the measured window).
        {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:b_in  offset:0 atIndex:0];
            [enc setBuffer:b_out offset:0 atIndex:1];
            [enc setBuffer:b_n   offset:0 atIndex:2];
            const NSUInteger tg = MIN((NSUInteger)64, [pipe maxTotalThreadsPerThreadgroup]);
            [enc dispatchThreads:MTLSizeMake(batch_size, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        const auto t0 = std::chrono::steady_clock::now();
        const auto bench_dur = std::chrono::seconds(seconds);
        uint64_t hashes = 0;
        while (std::chrono::steady_clock::now() - t0 < bench_dur) {
            id<MTLCommandBuffer> cmd = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            [enc setBuffer:b_in  offset:0 atIndex:0];
            [enc setBuffer:b_out offset:0 atIndex:1];
            [enc setBuffer:b_n   offset:0 atIndex:2];
            const NSUInteger tg = MIN((NSUInteger)64, [pipe maxTotalThreadsPerThreadgroup]);
            [enc dispatchThreads:MTLSizeMake(batch_size, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            hashes += batch_size;
        }
        const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();

        res.ok = true;
        res.total_hashes = hashes;
        res.elapsed_seconds = (double)elapsed_ms / 1000.0;
        res.hashes_per_second = res.elapsed_seconds > 0
            ? (double)hashes / res.elapsed_seconds
            : 0.0;
    }
    return res;
}

}  // namespace gpu
}  // namespace collider
