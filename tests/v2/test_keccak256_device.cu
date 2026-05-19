/**
 * test_keccak256_device.cu -- task F stage 2.
 *
 * Validates that the device-side Keccak-256 produces byte-identical output
 * to the CPU reference. Both must be in sync; this test is the safety net
 * for that contract.
 */

#include "src/gpu/v2/keccak256_cpu.hpp"
#include "src/gpu/v2/keccak256_device.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

using collider::gpu::v2::keccak256::keccak256;
namespace dev = collider::gpu::v2::device;

__global__ void run_keccak256(
    const uint8_t* __restrict__ in,
    uint32_t in_len,
    uint8_t* __restrict__ out)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dev::keccak256(in, in_len, out);
    }
}

static std::string hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s; s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(d[p[i] >> 4]);
        s.push_back(d[p[i] & 0xF]);
    }
    return s;
}

static int failures = 0;

static void check_input(const char* label, const uint8_t* in, uint32_t len) {
    // CPU reference.
    uint8_t cpu_out[32];
    keccak256(in, len, cpu_out);

    // GPU device.
    uint8_t* d_in = nullptr;
    uint8_t* d_out = nullptr;
    if (cudaMalloc(&d_in, len ? len : 1) != cudaSuccess ||
        cudaMalloc(&d_out, 32) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s -- cudaMalloc\n", label);
        ++failures;
        return;
    }
    if (len > 0) {
        cudaMemcpy(d_in, in, len, cudaMemcpyHostToDevice);
    }
    run_keccak256<<<1, 1>>>(d_in, len, d_out);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s -- kernel launch: %s\n",
                     label, cudaGetErrorString(e));
        ++failures;
        cudaFree(d_in); cudaFree(d_out);
        return;
    }
    uint8_t gpu_out[32];
    cudaMemcpy(gpu_out, d_out, 32, cudaMemcpyDeviceToHost);
    cudaFree(d_in); cudaFree(d_out);

    if (std::memcmp(cpu_out, gpu_out, 32) != 0) {
        std::fprintf(stderr,
            "FAIL: %s\n  cpu: %s\n  gpu: %s\n",
            label, hex(cpu_out, 32).c_str(), hex(gpu_out, 32).c_str());
        ++failures;
    }
}

static void check_str(const char* label, const char* s) {
    check_input(label, reinterpret_cast<const uint8_t*>(s),
                (uint32_t)std::strlen(s));
}

int main() {
    // Trust-anchor inputs -- if CPU passes its own KATs (separate test
    // target), and these match CPU output bit-for-bit, the device port is
    // correct.
    check_str("empty",         "");
    check_str("abc",           "abc");
    check_str("hello",         "hello");
    check_str("longer string", "The quick brown fox jumps over the lazy dog");

    // Block-boundary cases at the rate (136 bytes).
    {
        std::vector<uint8_t> buf(135, 0xAA);
        check_input("135 x 0xAA", buf.data(), 135);
    }
    {
        std::vector<uint8_t> buf(136, 0xAA);
        check_input("136 x 0xAA", buf.data(), 136);
    }
    {
        std::vector<uint8_t> buf(137, 0xAA);
        check_input("137 x 0xAA", buf.data(), 137);
    }
    {
        std::vector<uint8_t> buf(272, 0x55);  // exactly 2 blocks
        check_input("272 x 0x55", buf.data(), 272);
    }

    // ETH-pubkey-style input (64 bytes).
    {
        const uint8_t pubkey_xy[64] = {
            0x79,0xbe,0x66,0x7e,0xf9,0xdc,0xbb,0xac,
            0x55,0xa0,0x62,0x95,0xce,0x87,0x0b,0x07,
            0x02,0x9b,0xfc,0xdb,0x2d,0xce,0x28,0xd9,
            0x59,0xf2,0x81,0x5b,0x16,0xf8,0x17,0x98,
            0x48,0x3a,0xda,0x77,0x26,0xa3,0xc4,0x65,
            0x5d,0xa4,0xfb,0xfc,0x0e,0x11,0x08,0xa8,
            0xfd,0x17,0xb4,0x48,0xa6,0x85,0x54,0x19,
            0x9c,0x47,0xd0,0x8f,0xfb,0x10,0xd4,0xb8,
        };
        check_input("ETH G_x||G_y", pubkey_xy, 64);
    }

    if (failures != 0) {
        std::fprintf(stderr,
            "test_keccak256_device: %d device/cpu mismatch(es)\n", failures);
        return 1;
    }
    std::printf("test_keccak256_device: PASS\n");
    return 0;
}
