/**
 * test_sha256_batch_kat: Known-Answer Test for the GPU sha256_batch
 * host wrapper. Regression guard for the truncation bug fixed in
 * T0.3.c (A-tier wave 1, 2026-05-17).
 *
 * Background
 * ==========
 * Pre-T0.3.c, `sha256_batch_kernel` called `sha256_hash(msg, len, out)`
 * which performed a single-block compression with the copy loop
 * `for (size_t i = 0; i < len && i < 55; i++) block[i] = message[i];`
 * For len > 55 the message was truncated at byte 55 but the bit-length
 * padding at the tail of the block (last 8 bytes) still encoded the
 * full `len`. Result: silently-wrong SHA-256 digests for any
 * passphrase / message at or above 55 bytes.
 *
 * The matching bug in the fused brain-wallet path was patched as Wave-1
 * A-CRIT-1 a year ago; the standalone copy in src/gpu/sha256.cu never
 * got the matching fix and was still wired into `sha256_batch` via
 * `sha256_batch_kernel`. Live callers (bench_pipeline.cpp stage 1 and
 * puzzle_solver_benchmark.cpp's standalone CUDA sha256 rate) drove
 * the kernel with 32-byte to 64-byte inputs, so:
 *   - len=32 happened to land in the "always fits single-block" regime
 *     where the truncation cap (55) is never hit, so digests were
 *     accidentally correct.
 *   - len=64 hit the truncation hard: the kernel hashed bytes [0..54]
 *     then claimed it had hashed 64 bytes, producing a digest that
 *     differed from FIPS 180-4 for every input.
 *
 * T0.3.c deleted the truncating helper entirely and rewired
 * `sha256_batch_kernel` to use the multi-block `sha256_hash_long`
 * (FIPS 180-4-correct for any length).
 *
 * Test vectors
 * ============
 * Vectors below are bit-identical to the canonical SHA-256 KATs in
 * FIPS 180-4 Appendix B (empty string, "abc") plus single-byte "a"
 * and a series of zero-byte inputs covering the boundary the bug
 * lived at:
 *
 *   len=0:   FIPS 180-4 Appendix B.1
 *   len=1:   "a", well-known canonical
 *   len=3:   "abc", FIPS 180-4 Appendix B.1
 *   len=32:  hashlib.sha256(b'\\x00'*32), Bitcoin reference
 *   len=55:  hashlib.sha256(b'\\x00'*55), last single-block input
 *   len=56:  hashlib.sha256(b'\\x00'*56), FIRST multi-block input;
 *            the boundary that lit up the bug
 *   len=63:  hashlib.sha256(b'\\x00'*63), one byte short of block fill
 *   len=64:  hashlib.sha256(b'\\x00'*64), exactly one block of data
 *   len=65:  hashlib.sha256(b'\\x00'*65), one byte into the 3rd-block
 *            length region
 *   len=100: hashlib.sha256(b'\\x00'*100), well past the boundary
 *
 * All non-canonical vectors verified against Python `hashlib.sha256`
 * (CPython 3.11, OpenSSL backend) on 2026-05-17.
 *
 * Returns 77 (CTest skip) if no CUDA device, 0 on pass, 1 on fail.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Production API from src/gpu/sha256.cu
extern "C" {
    cudaError_t sha256_batch(
        const uint8_t* d_passphrases,
        const uint32_t* d_offsets,
        const uint32_t* d_lengths,
        uint8_t* d_hashes,
        size_t count,
        cudaStream_t stream
    );
}

// hex string to 32 bytes for canonical digests
static bool hex_to_bytes(const char* hex, uint8_t* out, size_t out_len) {
    for (size_t i = 0; i < out_len; ++i) {
        char hi = hex[i * 2];
        char lo = hex[i * 2 + 1];
        auto nyb = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
            if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
            return -1;
        };
        int h = nyb(hi), l = nyb(lo);
        if (h < 0 || l < 0) return false;
        out[i] = (uint8_t)((h << 4) | l);
    }
    return true;
}

struct Sha256KAT {
    const char* label;
    std::vector<uint8_t> input;
    const char* expected_hex;   // 64 hex chars = 32 bytes
};

static std::vector<Sha256KAT> build_vectors() {
    std::vector<Sha256KAT> v;
    v.push_back({"len=0 (empty)", {},
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"});
    v.push_back({"len=1 (\"a\")", {'a'},
        "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"});
    v.push_back({"len=3 (\"abc\")", {'a','b','c'},
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"});
    v.push_back({"len=32 (zeros)", std::vector<uint8_t>(32, 0),
        "66687aadf862bd776c8fc18b8e9f8e20089714856ee233b3902a591d0d5f2925"});
    v.push_back({"len=55 (zeros, last single-block)", std::vector<uint8_t>(55, 0),
        "02779466cdec163811d078815c633f21901413081449002f24aa3e80f0b88ef7"});
    v.push_back({"len=56 (zeros, FIRST multi-block, bug boundary)",
        std::vector<uint8_t>(56, 0),
        "d4817aa5497628e7c77e6b606107042bbba3130888c5f47a375e6179be789fbb"});
    v.push_back({"len=63 (zeros, one byte before block fill)",
        std::vector<uint8_t>(63, 0),
        "c7723fa1e0127975e49e62e753db53924c1bd84b8ac1ac08df78d09270f3d971"});
    v.push_back({"len=64 (zeros, exact block)", std::vector<uint8_t>(64, 0),
        "f5a5fd42d16a20302798ef6ed309979b43003d2320d9f0e8ea9831a92759fb4b"});
    v.push_back({"len=65 (zeros, first byte past block)",
        std::vector<uint8_t>(65, 0),
        "98ce42deef51d40269d542f5314bef2c7468d401ad5d85168bfab4c0108f75f7"});
    v.push_back({"len=100 (zeros, well past boundary)", std::vector<uint8_t>(100, 0),
        "cd00e292c5970d3c5e2f0ffa5171e555bc46bfc4faddfb4a418b6840b86e79a3"});
    return v;
}

int main() {
    int dev_count = 0;
    cudaError_t err = cudaGetDeviceCount(&dev_count);
    if (err != cudaSuccess || dev_count == 0) {
        std::fprintf(stderr, "[skip] no CUDA device available\n");
        return 77;  // CTest SKIP
    }

    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    const auto vectors = build_vectors();
    const size_t N = vectors.size();

    // Pack all input bytes into a single contiguous buffer with per-message
    // offsets and lengths, matching the production sha256_batch signature.
    std::vector<uint8_t> packed;
    std::vector<uint32_t> h_offsets(N), h_lengths(N);
    for (size_t i = 0; i < N; ++i) {
        h_offsets[i] = (uint32_t)packed.size();
        h_lengths[i] = (uint32_t)vectors[i].input.size();
        packed.insert(packed.end(), vectors[i].input.begin(), vectors[i].input.end());
    }
    // Pack at least 1 byte so cudaMalloc(0) does not bite anyone
    if (packed.empty()) packed.push_back(0);

    uint8_t* d_in = nullptr;
    uint32_t* d_offsets = nullptr;
    uint32_t* d_lengths = nullptr;
    uint8_t* d_out = nullptr;

    auto cleanup = [&]() {
        if (d_in)      cudaFree(d_in);
        if (d_offsets) cudaFree(d_offsets);
        if (d_lengths) cudaFree(d_lengths);
        if (d_out)     cudaFree(d_out);
    };

    err = cudaMalloc(&d_in, packed.size());
    if (err == cudaSuccess) err = cudaMalloc(&d_offsets, N * sizeof(uint32_t));
    if (err == cudaSuccess) err = cudaMalloc(&d_lengths, N * sizeof(uint32_t));
    if (err == cudaSuccess) err = cudaMalloc(&d_out, N * 32);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        cleanup();
        return 1;
    }

    err = cudaMemcpy(d_in, packed.data(), packed.size(), cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(d_offsets, h_offsets.data(), N * sizeof(uint32_t),
                         cudaMemcpyHostToDevice);
    if (err == cudaSuccess)
        err = cudaMemcpy(d_lengths, h_lengths.data(), N * sizeof(uint32_t),
                         cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy H to D failed: %s\n", cudaGetErrorString(err));
        cleanup();
        return 1;
    }

    err = sha256_batch(d_in, d_offsets, d_lengths, d_out, N, /*stream*/0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "sha256_batch launch failed: %s\n",
                     cudaGetErrorString(err));
        cleanup();
        return 1;
    }
    cudaDeviceSynchronize();

    std::vector<uint8_t> h_out(N * 32);
    err = cudaMemcpy(h_out.data(), d_out, N * 32, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy D to H failed: %s\n",
                     cudaGetErrorString(err));
        cleanup();
        return 1;
    }
    cleanup();

    int failures = 0;
    for (size_t i = 0; i < N; ++i) {
        uint8_t expected[32];
        if (!hex_to_bytes(vectors[i].expected_hex, expected, 32)) {
            std::fprintf(stderr, "internal error: bad hex literal for [%s]\n",
                         vectors[i].label);
            ++failures;
            continue;
        }
        const uint8_t* actual = &h_out[i * 32];
        if (std::memcmp(actual, expected, 32) != 0) {
            std::fprintf(stderr, "FAIL [%s] (len=%u)\n  expected: ",
                         vectors[i].label, h_lengths[i]);
            for (int b = 0; b < 32; ++b)
                std::fprintf(stderr, "%02x", expected[b]);
            std::fprintf(stderr, "\n  actual:   ");
            for (int b = 0; b < 32; ++b)
                std::fprintf(stderr, "%02x", actual[b]);
            std::fprintf(stderr, "\n");
            ++failures;
        } else {
            std::fprintf(stdout, "PASS [%s] (len=%u)\n",
                         vectors[i].label, h_lengths[i]);
        }
    }

    if (failures != 0) {
        std::fprintf(stderr, "\n%d/%zu vectors failed\n", failures, N);
        return 1;
    }
    std::fprintf(stdout, "\nAll %zu vectors passed\n", N);
    return 0;
}
