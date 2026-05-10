/**
 * Metal secp256k1 known-answer test.
 *
 * Runs the priv_to_pub kernel from src/gpu/kangaroo.metal against a
 * small batch of well-known scalars (priv = 1, 2, 3) and checks the
 * output bytes against published sipa/secp256k1 vectors.
 *
 * If any of these fails, the field-arithmetic primitives in the .metal
 * file are wrong; the Kangaroo walk built on top of them cannot be
 * trusted. So this test is the gate for shipping the Metal Kangaroo.
 *
 * Mac-only. ctest target name: MetalSecp256k1KAT.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/core/byte_codec.hpp"
#include "kangaroo_metal_source.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct KAT {
    const char* priv_be;     // 32-byte hex
    const char* expect_x_be; // 32-byte hex
    const char* expect_y_be; // 32-byte hex
};

// sipa/secp256k1 published vectors.
static const KAT kats[] = {
    {  // priv = 1 -> G
        "0000000000000000000000000000000000000000000000000000000000000001",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8"
    },
    {  // priv = 2 -> 2G
        "0000000000000000000000000000000000000000000000000000000000000002",
        "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
        "1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a"
    },
    {  // priv = 3 -> 3G
        "0000000000000000000000000000000000000000000000000000000000000003",
        "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",
        "388f7b0f632de8140fe337e62a37f3566500a99934c2231b6cb9fd7584b8e672"
    },
};

void hex_to_bytes(const char* hex, uint8_t out[32]) {
    for (int i = 0; i < 32; ++i) {
        char hi = hex[i*2], lo = hex[i*2+1];
        auto h = [](char c) -> uint8_t {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
            if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
            return 0;
        };
        out[i] = (h(hi) << 4) | h(lo);
    }
}

std::string bytes_to_hex(const uint8_t b[32]) {
    char out[65];
    for (int i = 0; i < 32; ++i) std::snprintf(out + i*2, 3, "%02x", b[i]);
    out[64] = '\0';
    return std::string(out);
}

}  // namespace

// Bring the shared codec into the unqualified scope used by main().
using ::collider::be32_to_limbs_le;
using ::collider::limbs_le_to_be32;

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { std::fprintf(stderr, "no Metal device\n"); return 77; }

        NSString* src = [NSString stringWithUTF8String:
                            ::collider::gpu::kKangarooMetalSource];
        if (!src) {
            std::fprintf(stderr, "embedded MSL source is not valid UTF-8 (build corrupt)\n");
            return 1;
        }
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion2_4;
        // SDK guard: MTLMathModeFast is macOS 15+ only.
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
        id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
        if (!lib) {
            std::fprintf(stderr, "MSL compile failed: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "?");
            return 1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"priv_to_pub"];
        if (!fn) { std::fprintf(stderr, "priv_to_pub kernel missing\n"); return 1; }
        id<MTLComputePipelineState> pipe =
            [device newComputePipelineStateWithFunction:fn error:&err];
        if (!pipe) {
            std::fprintf(stderr, "pipeline failed: %s\n",
                         err ? [[err localizedDescription] UTF8String] : "?");
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // secp256k1 generator (well-known).
        const char* GX_HEX = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
        const char* GY_HEX = "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";
        uint8_t gx_be[32], gy_be[32];
        hex_to_bytes(GX_HEX, gx_be);
        hex_to_bytes(GY_HEX, gy_be);
        uint64_t gx_l[4], gy_l[4];
        be32_to_limbs_le(gx_be, gx_l);
        be32_to_limbs_le(gy_be, gy_l);

        const int N = sizeof(kats) / sizeof(kats[0]);
        // Pack scalars into a buffer.
        std::vector<uint64_t> priv_buf((size_t)N * 4);
        for (int i = 0; i < N; ++i) {
            uint8_t k[32];
            hex_to_bytes(kats[i].priv_be, k);
            be32_to_limbs_le(k, priv_buf.data() + i * 4);
        }
        id<MTLBuffer> b_priv = [device newBufferWithBytes:priv_buf.data()
                                                   length:priv_buf.size() * 8
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_pubx = [device newBufferWithLength:N * 4 * 8
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_puby = [device newBufferWithLength:N * 4 * 8
                                                   options:MTLResourceStorageModeShared];
        uint32_t count = (uint32_t)N;

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:b_priv offset:0 atIndex:0];
        [enc setBuffer:b_pubx offset:0 atIndex:1];
        [enc setBuffer:b_puby offset:0 atIndex:2];
        [enc setBytes:gx_l length:sizeof(gx_l) atIndex:3];
        [enc setBytes:gy_l length:sizeof(gy_l) atIndex:4];
        [enc setBytes:&count length:sizeof(count) atIndex:5];
        [enc dispatchThreads:MTLSizeMake(N, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(MIN((NSUInteger)N,
                                                    [pipe maxTotalThreadsPerThreadgroup]),
                                                1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        const uint64_t* px = (const uint64_t*)[b_pubx contents];
        const uint64_t* py = (const uint64_t*)[b_puby contents];

        int failures = 0;
        for (int i = 0; i < N; ++i) {
            uint8_t got_x[32], got_y[32];
            limbs_le_to_be32(px + i * 4, got_x);
            limbs_le_to_be32(py + i * 4, got_y);
            std::string sx = bytes_to_hex(got_x);
            std::string sy = bytes_to_hex(got_y);
            bool ok = (sx == kats[i].expect_x_be) && (sy == kats[i].expect_y_be);
            std::printf("[%s] priv=%s\n", ok ? "ok  " : "FAIL", kats[i].priv_be);
            if (!ok) {
                std::printf("       got_x=%s\n  expected_x=%s\n",
                            sx.c_str(), kats[i].expect_x_be);
                std::printf("       got_y=%s\n  expected_y=%s\n",
                            sy.c_str(), kats[i].expect_y_be);
                ++failures;
            }
        }
        if (failures != 0) {
            std::fprintf(stderr, "test_metal_secp256k1: %d/%d failed\n", failures, N);
            return 1;
        }
        std::printf("test_metal_secp256k1: %d/%d PASS\n", N, N);
        return 0;
    }
}
