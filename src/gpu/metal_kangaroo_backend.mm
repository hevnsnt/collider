/**
 * Apple Metal Kangaroo backend adapter -- implementation.
 *
 * Builds the jump table + initial kangaroo seeds on the CPU using
 * cpu::ec_mul, hands them to KangarooMetalSolver, then drives the
 * synchronous step_round() loop. DPs are forwarded to
 * BackendCallbacks::on_dp in the wire-format byte layout the pool
 * client expects.
 *
 * Compiled as Obj-C++ with -fobjc-arc; ARC manages Metal buffer
 * lifetimes inside KangarooMetalSolver. This file itself uses no
 * Foundation/Metal types directly; the host loop is plain C++.
 */

#import <Foundation/Foundation.h>

#include "metal_kangaroo_backend.hpp"

#include "../core/byte_codec.hpp"
#include "../core/crypto_cpu.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>

namespace collider {
namespace kangaroo {

namespace {

// Decompress the 33-byte pool pubkey into (X, Y).
bool decompress_target_pubkey(const uint8_t pubkey33[33],
                              cpu::uint256_t& x,
                              cpu::uint256_t& y,
                              std::string& error_out)
{
    char hex[67];
    ::collider::hex_encode_lower(pubkey33, 33, hex);
    if (!cpu::decompress_pubkey(x, y, std::string(hex))) {
        error_out = std::string("failed to decompress pool target pubkey: ") + hex;
        return false;
    }
    return true;
}

// Build a 32-entry jump table: jumps[i] = ((1 << avg_pow) + i + 1) * G.
// Distances are clustered around 2^(dp_bits/2 + 2) for a balanced K
// factor; the +i offset ensures every entry is distinct so kangaroos
// don't fall into cycles.
std::array<gpu::KangarooSeed, gpu::kJumpTableSize>
build_jump_table(uint32_t dp_bits)
{
    std::array<gpu::KangarooSeed, gpu::kJumpTableSize> jumps{};
    const uint32_t avg_pow = std::max<uint32_t>(2u, (dp_bits / 2u) + 2u);
    for (size_t i = 0; i < gpu::kJumpTableSize; ++i) {
        cpu::uint256_t jd;
        jd.d[0] = (1ULL << avg_pow) + static_cast<uint64_t>(i + 1);
        jd.d[1] = jd.d[2] = jd.d[3] = 0;

        cpu::ECPoint jp;
        cpu::ec_mul(jp, jd);
        cpu::uint256_t jx, jy;
        cpu::ec_to_affine(jx, jy, jp);

        ::collider::limbs_le_to_be32(jx.d, jumps[i].x.data());
        ::collider::limbs_le_to_be32(jy.d, jumps[i].y.data());
        ::collider::limbs_le_to_be32(jd.d, jumps[i].d.data());
        jumps[i].type = 0;
    }
    return jumps;
}

// Build a single kangaroo seed. Tames (i < half) start at known scalar
// offsets in the chunk range; wilds (i >= half) root at the target Q
// plus a random offset. Factored out of build_seeds() so the host's
// identity-recovery loop can re-seed one kangaroo at a time when the
// kernel signals identity (P == -Q -> (0,0)) on a thread.
gpu::KangarooSeed
build_one_seed(bool is_tame,
               const cpu::uint256_t& off,
               const cpu::uint256_t& range_start,
               const cpu::uint256_t& target_x,
               const cpu::uint256_t& target_y)
{
    gpu::KangarooSeed s{};
    if (is_tame) {
        cpu::uint256_t scalar;
        cpu::add256(scalar, range_start, off);
        cpu::ECPoint p;
        cpu::ec_mul(p, scalar);
        cpu::uint256_t px, py;
        cpu::ec_to_affine(px, py, p);
        ::collider::limbs_le_to_be32(px.d,    s.x.data());
        ::collider::limbs_le_to_be32(py.d,    s.y.data());
        ::collider::limbs_le_to_be32(scalar.d, s.d.data());
        s.type = 0;
    } else {
        cpu::ECPoint op;
        cpu::ec_mul(op, off);
        cpu::uint256_t ox, oy;
        cpu::ec_to_affine(ox, oy, op);
        cpu::ECPoint wp;
        wp.X = target_x;
        wp.Y = target_y;
        wp.Z = cpu::uint256_t(1);
        cpu::ec_add(wp, wp, ox, oy);
        cpu::uint256_t wx, wy;
        cpu::ec_to_affine(wx, wy, wp);
        ::collider::limbs_le_to_be32(wx.d, s.x.data());
        ::collider::limbs_le_to_be32(wy.d, s.y.data());
        ::collider::limbs_le_to_be32(off.d, s.d.data());
        s.type = 1;
    }
    return s;
}

// Seed N kangaroos. First half are tames; second half are wilds.
// Random offsets keep workers diverse across boots.
std::vector<gpu::KangarooSeed>
build_seeds(uint32_t num_kangaroos,
            const cpu::uint256_t& range_start,
            const cpu::uint256_t& target_x,
            const cpu::uint256_t& target_y,
            uint64_t work_id)
{
    std::vector<gpu::KangarooSeed> seeds;
    seeds.reserve(num_kangaroos);

    std::mt19937_64 rng(static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count()) ^ work_id);

    const uint32_t half = num_kangaroos / 2u;
    for (uint32_t i = 0; i < num_kangaroos; ++i) {
        cpu::uint256_t off{};
        off.d[0] = rng();
        seeds.push_back(build_one_seed(i < half, off, range_start,
                                        target_x, target_y));
    }
    return seeds;
}

}  // namespace

bool MetalKangarooBackend::initialize(const collider::pool::WorkAssignment& work) {
    error_.clear();

    cpu::uint256_t target_x, target_y;
    if (!decompress_target_pubkey(work.public_key, target_x, target_y, error_)) {
        return false;
    }

    cpu::uint256_t range_start;
    ::collider::be32_to_limbs_le(work.range_start, range_start.d);

    // Cache range_start + target so solve()'s identity-recovery loop
    // can re-seed dead kangaroos without re-running the pubkey decode.
    range_start_ = range_start;
    target_x_    = target_x;
    target_y_    = target_y;

    cfg_         = gpu::KangarooMetalConfig{};
    cfg_.dp_bits = work.dp_bits;
    cfg_.work_id = work.work_id;
    work_id_     = work.work_id;

    if (!solver_.init(cfg_)) {
        error_ = "Metal Kangaroo init failed: " + solver_.error();
        return false;
    }

    auto jumps = build_jump_table(work.dp_bits);
    if (!solver_.set_jump_table(jumps)) {
        error_ = "Metal Kangaroo set_jump_table failed: " + solver_.error();
        return false;
    }

    auto seeds = build_seeds(cfg_.num_kangaroos, range_start,
                             target_x, target_y, work.work_id);
    if (!solver_.seed_kangaroos(seeds)) {
        error_ = "Metal Kangaroo seed_kangaroos failed: " + solver_.error();
        return false;
    }

    initialized_ = true;
    return true;
}

void MetalKangarooBackend::solve(BackendCallbacks cb) {
    if (!initialized_) {
        error_ = "MetalKangarooBackend::solve called before initialize";
        return;
    }

    const auto loop_start = std::chrono::steady_clock::now();
    uint64_t total_dp        = 0;
    uint64_t total_ops       = 0;
    uint64_t total_re_seeded = 0;

    // RNG used to mint fresh offsets for re-seeded kangaroos. Seed once
    // per solve() call rather than per round so two simultaneous
    // re-seeds (back-to-back rounds with overlapping dead lists) get
    // different offsets.
    std::mt19937_64 reseed_rng(static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count())
        ^ (work_id_ * 0x9E3779B97F4A7C15ULL));

    const uint32_t half = cfg_.num_kangaroos / 2u;
    std::vector<uint32_t> dead;
    dead.reserve(16);

    while (cb.should_continue()) {
        solver_.set_work_id(work_id_);
        auto dps = solver_.step_round();
        total_ops += static_cast<uint64_t>(cfg_.num_kangaroos)
                   * cfg_.steps_per_round;
        for (const auto& dp : dps) {
            cb.on_dp(dp.x_be, dp.d_be, dp.type, dp.dp_bits);
            ++total_dp;
        }

        // Identity recovery: scan the kangaroo state buffer for any
        // (0, 0) point -- the kernel writes that on P == -Q hits via
        // point_op() in kangaroo.metal. Without re-seeding, those
        // threads emit garbage DPs every subsequent round and poison
        // the pool's collision detection.
        if (solver_.find_dead_kangaroos(dead)) {
            for (uint32_t idx : dead) {
                cpu::uint256_t off{};
                off.d[0] = reseed_rng();
                const bool is_tame = (idx < half);
                const gpu::KangarooSeed s = build_one_seed(
                    is_tame, off, range_start_, target_x_, target_y_);
                solver_.replace_seed(idx, s);
                ++total_re_seeded;
            }
            if (!dead.empty()) {
                std::cerr << "[Metal] Re-seeded " << dead.size()
                          << " kangaroo(s) that hit identity (running total: "
                          << total_re_seeded << ")\n";
            }
        }

        const auto   now     = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(now - loop_start).count();
        const double rate    = (elapsed > 0.0)
            ? static_cast<double>(total_ops) / elapsed
            : 0.0;
        if (!cb.on_progress(rate, total_dp)) break;
    }
    // Metal backend never produces a local solution -- DPs go to the
    // pool server, which detects the cross-worker collision and tells
    // the pool. cb.on_solution is intentionally never invoked here.
}

std::string MetalKangarooBackend::device_summary() const {
    return solver_.device_name();
}

}  // namespace kangaroo
}  // namespace collider
