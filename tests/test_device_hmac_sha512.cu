/**
 * Device HMAC-SHA512 known-answer test.
 *
 * Validates that the streaming HMAC-SHA-512 implementation in
 * src/gpu/v2/device_hashes.cuh produces RFC 4231-compliant output. The
 * device version is independent from the CPU reference in
 * src/gpu/v2/sha512_cpu.hpp (which has its own KAT in
 * tests/v2/test_pbkdf2_cpu.cpp); this test catches regressions in the
 * device-only code that the CPU test cannot see.
 *
 * Required after the v1.4.0 stack-reduction refactor of hmac_sha512:
 * the streaming form (separate sha512_compress calls instead of a
 * single sha512(buffer)) preserves the algorithm but rearranges the
 * intermediate buffer flow, so an algebraic regression would silently
 * produce wrong digests for S7/S8 brain-wallet schemes. This test pins
 * the output bit-for-bit.
 *
 * Coverage:
 *   * RFC 4231 TC1: 20-byte 0x0b key, "Hi There" message
 *   * RFC 4231 TC2: short ASCII key, ASCII message
 *   * RFC 4231 TC4: 25-byte counting key, 50-byte 0xcd message
 *   * Long key (>128 bytes) -- exercises the key-hashing branch
 *
 * Returns 77 (ctest skip) if no CUDA device is available.
 */

#include <cuda_runtime.h>
#include "../src/gpu/v2/device_hashes.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

using namespace collider::gpu::v2;

namespace {

// Each test case captures the inputs and the expected 64-byte output.
struct HmacCase {
    const char* name;
    const uint8_t* key;
    uint32_t      key_len;
    const uint8_t* msg;
    uint32_t      msg_len;
    const char*   expected_hex;  // 128-char lowercase hex
};

// Single-thread kernel: run hmac_sha512 once on a per-case input pair.
__global__ void run_hmac_kernel(
    const uint8_t* key, uint32_t key_len,
    const uint8_t* msg, uint32_t msg_len,
    uint8_t* out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        device::hmac_sha512(key, key_len, msg, msg_len, out);
    }
}

std::string hex_encode(const uint8_t* b, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s; s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(d[(b[i] >> 4) & 0xF]);
        s.push_back(d[b[i] & 0xF]);
    }
    return s;
}

bool run_case(const HmacCase& c) {
    uint8_t* d_key = nullptr;
    uint8_t* d_msg = nullptr;
    uint8_t* d_out = nullptr;
    cudaMalloc(&d_key, c.key_len > 0 ? c.key_len : 1);
    cudaMalloc(&d_msg, c.msg_len > 0 ? c.msg_len : 1);
    cudaMalloc(&d_out, 64);
    if (c.key_len > 0) cudaMemcpy(d_key, c.key, c.key_len, cudaMemcpyHostToDevice);
    if (c.msg_len > 0) cudaMemcpy(d_msg, c.msg, c.msg_len, cudaMemcpyHostToDevice);

    run_hmac_kernel<<<1, 1>>>(d_key, c.key_len, d_msg, c.msg_len, d_out);
    cudaError_t rc = cudaDeviceSynchronize();
    bool ok = (rc == cudaSuccess);

    uint8_t out[64];
    if (ok) cudaMemcpy(out, d_out, 64, cudaMemcpyDeviceToHost);

    cudaFree(d_key);
    cudaFree(d_msg);
    cudaFree(d_out);

    if (!ok) {
        std::fprintf(stderr, "[FAIL] %s: kernel error %s\n",
                     c.name, cudaGetErrorString(rc));
        return false;
    }
    const std::string got = hex_encode(out, 64);
    if (got != c.expected_hex) {
        std::fprintf(stderr, "[FAIL] %s\n  got      = %s\n  expected = %s\n",
                     c.name, got.c_str(), c.expected_hex);
        return false;
    }
    std::fprintf(stdout, "[ok  ] %s\n", c.name);
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::fprintf(stderr, "no CUDA device, skipping\n");
        return 77;  // ctest SKIP
    }
    cudaSetDevice(0);

    // ---- TC1: 20-byte 0x0b key, "Hi There" -----------------------------
    static const uint8_t tc1_key[20] = {
        0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,
        0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,0x0b,
    };
    static const char tc1_msg[] = "Hi There";

    // ---- TC2: short ASCII key, ASCII message ---------------------------
    // RFC 4231 TC2: key = "Jefe", msg = "what do ya want for nothing?"
    static const char tc2_key[] = "Jefe";
    static const char tc2_msg[] = "what do ya want for nothing?";

    // ---- TC4: 25-byte counting key, 50-byte 0xcd message ---------------
    static const uint8_t tc4_key[25] = {
        0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,
        0x0b,0x0c,0x0d,0x0e,0x0f,0x10,0x11,0x12,0x13,0x14,
        0x15,0x16,0x17,0x18,0x19,
    };
    static uint8_t tc4_msg[50];
    for (int i = 0; i < 50; ++i) tc4_msg[i] = 0xcd;

    // ---- RFC 4231 TC6: 131-byte 0xaa key, ASCII message ---------------
    // Exercises the "key longer than block size (128)" branch which
    // hashes the key down to 64 bytes via sha512(key, key_len, k0).
    static uint8_t tc6_key[131];
    for (int i = 0; i < 131; ++i) tc6_key[i] = 0xaa;
    static const char tc6_msg[] =
        "Test Using Larger Than Block-Size Key - Hash Key First";

    HmacCase cases[] = {
        { "RFC 4231 TC1 (key=20*0x0b, msg='Hi There')",
          tc1_key, 20, (const uint8_t*)tc1_msg, 8,
          "87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cde"
          "daa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854" },
        { "RFC 4231 TC2 (key='Jefe', msg='what do ya want for nothing?')",
          (const uint8_t*)tc2_key, 4, (const uint8_t*)tc2_msg, 28,
          "164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea250554"
          "9758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737" },
        { "RFC 4231 TC4 (counting key, 50 bytes 0xcd)",
          tc4_key, 25, tc4_msg, 50,
          "b0ba465637458c6990e5a8c5f61d4af7e576d97ff94b872de76f8050361ee3db"
          "a91ca5c11aa25eb4d679275cc5788063a5f19741120c4f2de2adebeb10a298dd" },
        { "RFC 4231 TC6 (131-byte key, exercises hashed-key branch)",
          tc6_key, 131, (const uint8_t*)tc6_msg,
          (uint32_t)(sizeof(tc6_msg) - 1),
          "80b24263c7c1a3ebb71493c1dd7be8b49b46d1f41b4aeec1121b013783f8f352"
          "6b56d037e05f2598bd0fd2215d6a1e5295e64f73f63f0aec8b915a985d786598" },
    };

    int failures = 0;
    for (const auto& c : cases) {
        if (!run_case(c)) ++failures;
    }
    if (failures) {
        std::fprintf(stderr, "test_device_hmac_sha512: %d/%d failed\n",
                     failures, (int)(sizeof(cases)/sizeof(cases[0])));
        return 1;
    }
    std::fprintf(stdout, "test_device_hmac_sha512: %d/%d PASS\n",
                 (int)(sizeof(cases)/sizeof(cases[0])),
                 (int)(sizeof(cases)/sizeof(cases[0])));
    return 0;
}
