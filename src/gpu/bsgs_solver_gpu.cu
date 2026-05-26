// bsgs_solver_gpu.cu -- Phase F2 first cut.
//
// GPU-accelerated BSGS: the heavy scalar multiplications (baby table
// + giant unshifted points) run on the GPU via the existing
// secp256k1_batch_mul_simple primitive. The collision check
// (subtract H, sort, binary search) is still host-side; a follow-up
// commit ports the sort+lookup to GPU radix-sort + parallel binary
// search to remove the host bottleneck. The baseline here is a
// correct end-to-end pipeline so the operator can A/B BSGS vs
// Kangaroo today on bounded ranges.
#include "gpu/bsgs_solver_gpu.hpp"

#include <cuda_runtime.h>

#include "core/crypto_cpu.hpp"
#include "core/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

// Forward decl of the host wrapper exposed from secp256k1.cu. The
// header for the GPU EC ops is not public; the wrapper is exposed via
// extern "C" so any .cu / .cpp can call it without pulling in the
// kernel-private __constant__ tables.
extern "C" cudaError_t secp256k1_batch_mul_simple(const void* d_private_keys,
                                                  void* d_public_keys,
                                                  size_t count,
                                                  cudaStream_t stream);
extern "C" cudaError_t secp256k1_init_table(cudaStream_t stream);

namespace collider::gpu::bsgs {

namespace {

// Bit-equivalent of secp256k1.cu's uint256 (uint32_t[8], little-endian
// limbs) and ECPointAffine (X, Y). The kernel writes into raw bytes
// of this exact layout; we mirror it here so the host can read /
// compare without including the .cu file's private headers.
struct GpuUint256 {
    uint32_t limbs[8];
};

struct GpuPointAffine {
    GpuUint256 x;
    GpuUint256 y;
};

// Convert host uint256_t (8 x uint64_t little-endian) into the
// GPU uint256 layout (8 x uint32_t little-endian). crypto_cpu's
// uint256_t::d[i] is the i-th 64-bit limb little-endian; the GPU
// stores 8 x 32-bit little-endian, so d[0] low half -> limbs[0],
// d[0] high half -> limbs[1], etc.
void uint256_to_gpu(const cpu::uint256_t& src, GpuUint256& dst) {
    for (int i = 0; i < 4; ++i) {
        const uint64_t v = src.d[i];
        dst.limbs[2 * i + 0] = static_cast<uint32_t>(v);
        dst.limbs[2 * i + 1] = static_cast<uint32_t>(v >> 32);
    }
}

// Inverse of uint256_to_gpu.
void uint256_from_gpu(const GpuUint256& src, cpu::uint256_t& dst) {
    for (int i = 0; i < 4; ++i) {
        dst.d[i] =
            (static_cast<uint64_t>(src.limbs[2 * i + 1]) << 32) |
             static_cast<uint64_t>(src.limbs[2 * i + 0]);
    }
}

// BE 32-byte -> uint256_t (host LE limb layout).
void be_to_uint256(const uint8_t* be, cpu::uint256_t& out) {
    for (int i = 0; i < 4; ++i) {
        uint64_t v = 0;
        for (int b = 0; b < 8; ++b) {
            v = (v << 8) | be[i * 8 + b];
        }
        // BE bytes [0..7] are the most-significant 64-bit limb, which
        // is d[3] in LE limb order.
        out.d[3 - i] = v;
    }
}

// uint256_t (host LE limbs) -> BE 32 bytes.
void uint256_to_be(const cpu::uint256_t& in, uint8_t* be) {
    for (int i = 0; i < 4; ++i) {
        const uint64_t v = in.d[3 - i];
        for (int b = 0; b < 8; ++b) {
            be[i * 8 + b] = static_cast<uint8_t>((v >> ((7 - b) * 8)) & 0xFF);
        }
    }
}

// Big-endian raw 32-byte X compare. Each baby entry is stored as
// (be_x[32], baby_index). std::lower_bound over the sorted vector
// then resolves to the matching index in O(log m).
struct BabyEntry {
    std::array<uint8_t, 32> x_be;
    uint64_t baby_index;
    bool operator<(const BabyEntry& o) const { return x_be < o.x_be; }
};

// Helper: compute candidate = H + (-V) where V is an EC point in
// affine. We negate V.y modulo p, then call cpu::ec_add(H_jac, V_neg).
// Returns the affine X of the result.
cpu::uint256_t host_sub_x(const cpu::ECPoint& H_jac,
                          const cpu::uint256_t& Vx,
                          const cpu::uint256_t& Vy) {
    // -V.y = (p - V.y) mod p
    cpu::uint256_t neg_y;
    cpu::mod_sub(neg_y, cpu::SECP256K1_P, Vy);

    cpu::ECPoint result;
    cpu::ec_add(result, H_jac, Vx, neg_y);
    cpu::uint256_t out_x, out_y;
    cpu::ec_to_affine(out_x, out_y, result);
    return out_x;
}

// Convert uint256_t to a 32-byte BE array for sorted-key comparison.
std::array<uint8_t, 32> uint256_to_be_array(const cpu::uint256_t& v) {
    std::array<uint8_t, 32> out{};
    uint256_to_be(v, out.data());
    return out;
}

// Approximate sqrt for the baby-table sizing. ceil(sqrt(N)) where N
// is small enough to fit in uint64_t (bits <= 64 caller-checked).
uint64_t ceil_sqrt(uint64_t n) {
    if (n == 0) return 0;
    long double s = std::sqrt(static_cast<long double>(n));
    uint64_t r = static_cast<uint64_t>(s);
    // Tighten the bounds: r*r may have lost precision.
    while (r > 0 && r * r > n) --r;
    while ((r + 1) * (r + 1) <= n) ++r;
    return r + (r * r < n ? 1 : 0);
}

}  // namespace

BsgsResult bsgs_solve(const BsgsConfig& cfg) {
    BsgsResult out;

    if (cfg.bits <= 0 || cfg.bits > kMaxBits) {
        out.kind = BsgsResultKind::OutOfRange;
        out.error_message = "bits out of [1, " +
                            std::to_string(kMaxBits) + "]";
        return out;
    }

    // Decode the range. range_end - range_start must fit in uint64_t
    // because the bits cap above guarantees N <= 2^48.
    cpu::uint256_t b_full, e_full;
    be_to_uint256(cfg.range_start_be, b_full);
    be_to_uint256(cfg.range_end_be,   e_full);
    // Sanity: only the low 64 bits of b/e should differ for the
    // bits<=kMaxBits guarantee. We pull them out as plain uint64_t.
    if (b_full.d[1] != e_full.d[1] || b_full.d[2] != e_full.d[2] ||
        b_full.d[3] != e_full.d[3]) {
        out.kind = BsgsResultKind::OutOfRange;
        out.error_message =
            "range_end - range_start exceeds the 64-bit window the "
            "first-cut driver supports";
        return out;
    }
    const uint64_t b_low = b_full.d[0];
    const uint64_t e_low = e_full.d[0];
    if (e_low <= b_low) {
        out.kind = BsgsResultKind::NotInRange;
        out.error_message = "range_end <= range_start";
        return out;
    }
    const uint64_t N = e_low - b_low;
    const uint64_t m = std::max<uint64_t>(1, ceil_sqrt(N));
    const uint64_t num_giants = (N + m - 1) / m;

    out.baby_table_size = m;

    // Device init.
    if (cfg.device_id >= 0) {
        cudaError_t serr = cudaSetDevice(cfg.device_id);
        if (serr != cudaSuccess) {
            out.kind = BsgsResultKind::GpuError;
            out.error_message = "cudaSetDevice: ";
            out.error_message += cudaGetErrorString(serr);
            return out;
        }
    }
    cudaStream_t stream = nullptr;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        out.kind = BsgsResultKind::GpuError;
        out.error_message = "cudaStreamCreate: ";
        out.error_message += cudaGetErrorString(err);
        return out;
    }
    err = secp256k1_init_table(stream);
    if (err != cudaSuccess) {
        out.kind = BsgsResultKind::GpuError;
        out.error_message = "secp256k1_init_table: ";
        out.error_message += cudaGetErrorString(err);
        cudaStreamDestroy(stream);
        return out;
    }

    // Baby table: scalars = [0..m). One thread per scalar.
    std::vector<GpuUint256> baby_scalars(m, GpuUint256{});
    for (uint64_t j = 0; j < m; ++j) {
        baby_scalars[j].limbs[0] = static_cast<uint32_t>(j & 0xFFFFFFFFu);
        baby_scalars[j].limbs[1] = static_cast<uint32_t>(j >> 32);
    }
    GpuUint256* d_baby_scalars = nullptr;
    GpuPointAffine* d_baby_pts = nullptr;
    cudaMalloc(&d_baby_scalars, m * sizeof(GpuUint256));
    cudaMalloc(&d_baby_pts,     m * sizeof(GpuPointAffine));
    cudaMemcpyAsync(d_baby_scalars, baby_scalars.data(),
                    m * sizeof(GpuUint256),
                    cudaMemcpyHostToDevice, stream);
    err = secp256k1_batch_mul_simple(d_baby_scalars, d_baby_pts, m, stream);
    if (err != cudaSuccess) {
        out.kind = BsgsResultKind::GpuError;
        out.error_message = "baby batch_mul: ";
        out.error_message += cudaGetErrorString(err);
        cudaFree(d_baby_scalars); cudaFree(d_baby_pts);
        cudaStreamDestroy(stream);
        return out;
    }
    std::vector<GpuPointAffine> baby_pts(m);
    cudaMemcpyAsync(baby_pts.data(), d_baby_pts,
                    m * sizeof(GpuPointAffine),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Build sorted baby table for O(log m) lookups.
    std::vector<BabyEntry> sorted_baby;
    sorted_baby.reserve(m);
    for (uint64_t j = 0; j < m; ++j) {
        cpu::uint256_t bx;
        uint256_from_gpu(baby_pts[j].x, bx);
        BabyEntry e;
        e.x_be = uint256_to_be_array(bx);
        e.baby_index = j;
        sorted_baby.push_back(e);
    }
    std::sort(sorted_baby.begin(), sorted_baby.end());

    // Free baby device buffers (host copy retained).
    cudaFree(d_baby_scalars);
    cudaFree(d_baby_pts);

    // Giant scalars: (b + i*m) for i in [0, num_giants).
    std::vector<GpuUint256> giant_scalars(num_giants, GpuUint256{});
    for (uint64_t i = 0; i < num_giants; ++i) {
        const uint64_t s = b_low + i * m;  // b_low + i*m fits in uint64 (bits<=48)
        giant_scalars[i].limbs[0] = static_cast<uint32_t>(s & 0xFFFFFFFFu);
        giant_scalars[i].limbs[1] = static_cast<uint32_t>(s >> 32);
    }
    GpuUint256* d_giant_scalars = nullptr;
    GpuPointAffine* d_giant_pts = nullptr;
    cudaMalloc(&d_giant_scalars, num_giants * sizeof(GpuUint256));
    cudaMalloc(&d_giant_pts,     num_giants * sizeof(GpuPointAffine));
    cudaMemcpyAsync(d_giant_scalars, giant_scalars.data(),
                    num_giants * sizeof(GpuUint256),
                    cudaMemcpyHostToDevice, stream);
    err = secp256k1_batch_mul_simple(d_giant_scalars, d_giant_pts,
                                     num_giants, stream);
    if (err != cudaSuccess) {
        out.kind = BsgsResultKind::GpuError;
        out.error_message = "giant batch_mul: ";
        out.error_message += cudaGetErrorString(err);
        cudaFree(d_giant_scalars); cudaFree(d_giant_pts);
        cudaStreamDestroy(stream);
        return out;
    }
    std::vector<GpuPointAffine> giant_pts(num_giants);
    cudaMemcpyAsync(giant_pts.data(), d_giant_pts,
                    num_giants * sizeof(GpuPointAffine),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_giant_scalars);
    cudaFree(d_giant_pts);
    cudaStreamDestroy(stream);

    // Host-side collision search. For each giant point V_i =
    // (b + i*m)*G, compute candidate = H - V_i; look up candidate.X
    // in the sorted baby table. On hit verify (b + i*m + j)*G == H
    // because two EC points can share an X (the point and -P).
    cpu::uint256_t Hx, Hy;
    be_to_uint256(cfg.target_pubkey_x_be, Hx);
    be_to_uint256(cfg.target_pubkey_y_be, Hy);
    cpu::ECPoint H_jac;
    H_jac.X = Hx;
    H_jac.Y = Hy;
    H_jac.Z = cpu::uint256_t(1);

    auto last_progress = std::chrono::steady_clock::now();
    for (uint64_t i = 0; i < num_giants; ++i) {
        if (cfg.progress &&
            std::chrono::steady_clock::now() - last_progress >
                std::chrono::milliseconds(500)) {
            if (!cfg.progress(i, cfg.progress_user)) {
                out.kind = BsgsResultKind::Cancelled;
                out.giant_steps_completed = i;
                return out;
            }
            last_progress = std::chrono::steady_clock::now();
        }

        cpu::uint256_t Vx, Vy;
        uint256_from_gpu(giant_pts[i].x, Vx);
        uint256_from_gpu(giant_pts[i].y, Vy);
        cpu::uint256_t candidate_x = host_sub_x(H_jac, Vx, Vy);
        std::array<uint8_t, 32> key = uint256_to_be_array(candidate_x);

        BabyEntry probe;
        probe.x_be = key;
        probe.baby_index = 0;
        auto it = std::lower_bound(sorted_baby.begin(),
                                   sorted_baby.end(),
                                   probe);
        if (it != sorted_baby.end() && it->x_be == key) {
            // Candidate hit. Verify k*G == H to rule out the
            // X-only collision (i.e. we matched the negated point).
            const uint64_t k_low = b_low + i * m + it->baby_index;
            cpu::uint256_t k_full;
            k_full.d[0] = k_low;
            k_full.d[1] = b_full.d[1];
            k_full.d[2] = b_full.d[2];
            k_full.d[3] = b_full.d[3];
            cpu::ECPoint check_jac;
            cpu::ec_mul(check_jac, k_full);
            cpu::uint256_t check_x, check_y;
            cpu::ec_to_affine(check_x, check_y, check_jac);
            if (check_x == Hx && check_y == Hy) {
                out.kind = BsgsResultKind::Found;
                uint256_to_be(k_full, out.recovered_key_be);
                out.giant_steps_completed = i + 1;
                return out;
            }
            // false positive (X collision); keep searching.
        }
    }

    out.kind = BsgsResultKind::NotInRange;
    out.giant_steps_completed = num_giants;
    return out;
}

}  // namespace collider::gpu::bsgs
