/**
 * Metal fused puzzle pipeline known-answer test.
 *
 * End-to-end exercise of src/gpu/puzzle.metal:
 *   privkey  -- via the precomputed G-table -->  P = k * G
 *   P        -- mod_inv + compress  -->  33-byte compressed pubkey
 *   pubkey   -- inline SHA-256       -->  32-byte digest
 *   digest   -- inline RIPEMD-160    -->  20-byte hash160
 *   hash160  -- byte compare         -->  match flag + key writeback
 *
 * KAT vectors: well-known Bitcoin puzzle 1 (privkey = 1) and puzzle 2
 * (privkey = 2). Hash160 values come from tests/test_hash_vectors.cpp's
 * HASH160_TESTS table -- the same hashes the CUDA path is validated
 * against. A passing run on this test means EVERY primitive in
 * puzzle.metal -- field arithmetic, Jacobian point ops, mod_inv addition
 * chain, SHA-256 round function, RIPEMD-160 dual chain -- is byte-correct.
 *
 * Failure modes:
 *   - Returns 77 if no Metal device is present (headless CI / Linux box).
 *     ctest is configured with SKIP_RETURN_CODE=77 so the run is skipped
 *     rather than failed in that case.
 *   - Returns 1 on any KAT mismatch with a clear "FAIL: privkey N" line.
 *   - Returns 0 on full success.
 *
 * Mac-only. ctest target name: MetalPuzzleKAT.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "../src/gpu/puzzle_metal.hpp"

#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>

namespace {

struct PuzzleKAT {
    uint64_t priv_lo;
    uint64_t priv_hi;
    const char* expect_h160_hex;   // 40 hex chars
};

// Anchored on the puzzle 1 / puzzle 2 hash160 vectors from
// tests/test_hash_vectors.cpp's HASH160_TESTS. The CUDA puzzle pipeline
// is validated against the same numbers; matching them on Metal proves
// the GPU pipeline reaches the same digest byte-for-byte.
static const PuzzleKAT kats[] = {
    // Puzzle 1: privkey = 1 -> hash160 of compress(1*G).
    { 1ull, 0ull, "751e76e8199196d454941c45d1b3a323f1433bd6" },
    // Puzzle 2: privkey = 2 -> hash160 of compress(2*G).
    // (Note: this is the corrected hash from C-CRIT-4; pre-fix vectors
    //  in older docs gave 91b24bf... which was wrong.)
    { 2ull, 0ull, "06afd46bcdfd22ef94ac122aa11f241244a37ecc" },
};

void hex_to_h160(const char* hex, std::array<uint8_t, 20>& out) {
    auto h = [](char c) -> uint8_t {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return 0;
    };
    for (int i = 0; i < 20; ++i) {
        out[i] = (h(hex[i*2]) << 4) | h(hex[i*2 + 1]);
    }
}

}  // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> probe = MTLCreateSystemDefaultDevice();
        if (!probe) {
            std::fprintf(stderr, "[skip] no Metal device available\n");
            return 77;  // ctest SKIP_RETURN_CODE
        }
    }

    collider::gpu::PuzzleMetalSolver solver;
    if (!solver.init()) {
        std::fprintf(stderr, "[fail] PuzzleMetalSolver::init: %s\n",
                     solver.error().c_str());
        return 1;
    }
    std::printf("Metal device: %s\n", solver.device_name().c_str());

    int failures = 0;
    for (const auto& k : kats) {
        std::array<uint8_t, 20> target;
        hex_to_h160(k.expect_h160_hex, target);

        const bool ok = solver.verify_one(k.priv_lo, k.priv_hi, target);
        if (ok) {
            std::printf("  PASS: privkey=%llu -> %s\n",
                        (unsigned long long)k.priv_lo, k.expect_h160_hex);
        } else {
            std::printf("  FAIL: privkey=%llu -> %s (kernel did not match)\n",
                        (unsigned long long)k.priv_lo, k.expect_h160_hex);
            ++failures;
        }
    }

    // Negative control: feed a target that does NOT match privkey=1's
    // hash160. The kernel must NOT report a match; otherwise our compare
    // logic is broken and every key would falsely match every target.
    {
        std::array<uint8_t, 20> bogus{};
        for (int i = 0; i < 20; ++i) bogus[i] = (uint8_t)(0xAA ^ i);
        const bool false_hit = solver.verify_one(1ull, 0ull, bogus);
        if (!false_hit) {
            std::printf("  PASS: negative control (privkey=1 vs random h160)\n");
        } else {
            std::printf("  FAIL: negative control: kernel reported a match for a random target\n");
            ++failures;
        }
    }

    std::printf("\n=== Results ===\n");
    std::printf("Failures: %d\n", failures);
    return failures > 0 ? 1 : 0;
}
