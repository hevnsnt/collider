/**
 * MultiGPUKangarooManager -- Metal backend for standalone puzzle solving.
 *
 * Pre-1.4.1 the Mac build of `collider --puzzle N --kangaroo` hit the
 * no-CUDA stub of MultiGPUKangarooManager (kangaroo_solver_gpu.hpp) and
 * silently fell back to the CPU kangaroo, even though the Metal kernel
 * was already shipping for pool mode. This file closes that gap by
 * implementing the same surface (`init/set_range/set_target_pubkey/
 * solve/...`) on top of KangarooMetalSolver.
 *
 * Architecture:
 *   - One KangarooMetalSolver instance (M-series machines have one
 *     unified GPU; gpu_ids is ignored beyond reporting num_gpus()==1).
 *   - Jump table + seed-construction reuses the exact same logic as
 *     metal_kangaroo_backend.mm so the two paths produce statistically
 *     identical kangaroo walks. The two files intentionally share
 *     build_jump_table() and build_one_seed() semantics; if you change
 *     one, change the other.
 *   - Standalone collision detection: KangarooMetalDPs flow into a
 *     std::unordered_map<x_be, (d_be, type)>. When the same X appears
 *     with a tame entry on one side and a wild entry on the other, the
 *     private key is k = (d_tame - d_wild) mod n. Verified by
 *     reconstructing k*G and checking it matches the target pubkey.
 *   - Pool mode does this collision detection on the SERVER (across
 *     all workers, via dp_store.add_dp); standalone has to do it
 *     locally because there is no server.
 *
 * Compiled only on APPLE && COLLIDER_USE_METAL.
 */

#import <Foundation/Foundation.h>

#include "kangaroo_solver_gpu.hpp"
#include "kangaroo_metal.hpp"

#include "../core/byte_codec.hpp"
#include "../core/crypto_cpu.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace collider {
namespace gpu {

namespace {

// ---------------------------------------------------------------------------
// Jump-table + seed builders -- mirror metal_kangaroo_backend.mm exactly.
// Kept private to this TU; the pool backend has its own copy because the
// two modules build under different translation units and there's no
// shared header that owns these helpers (yet -- v1.5 candidate for
// `metal_kangaroo_seeding.{hpp,mm}`).
// ---------------------------------------------------------------------------

std::array<KangarooSeed, kJumpTableSize>
build_jump_table(uint32_t dp_bits)
{
    std::array<KangarooSeed, kJumpTableSize> jumps{};
    const uint32_t avg_pow = std::max<uint32_t>(2u, (dp_bits / 2u) + 2u);
    for (size_t i = 0; i < kJumpTableSize; ++i) {
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

KangarooSeed
build_one_seed(bool is_tame,
               const cpu::uint256_t& off,
               const cpu::uint256_t& range_start,
               const cpu::uint256_t& target_x,
               const cpu::uint256_t& target_y)
{
    KangarooSeed s{};
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

// k = (d_tame - d_wild) mod n. cpu::sub256 returns the borrow; on
// underflow we add n by computing n - (d_wild - d_tame) instead.
// Final reduce loop handles the (extremely rare) case where the
// raw difference still exceeds n.
cpu::uint256_t
mod_n_sub(const cpu::uint256_t& d_tame, const cpu::uint256_t& d_wild)
{
    cpu::uint256_t k;
    if (d_tame >= d_wild) {
        cpu::sub256(k, d_tame, d_wild);
    } else {
        cpu::uint256_t diff;
        cpu::sub256(diff, d_wild, d_tame);
        cpu::sub256(k, cpu::SECP256K1_N, diff);
    }
    while (!(k < cpu::SECP256K1_N)) {
        cpu::uint256_t tmp;
        cpu::sub256(tmp, k, cpu::SECP256K1_N);
        k = tmp;
    }
    return k;
}

// Verify a recovered scalar k against the target pubkey. Returns true
// iff k*G == (target_x, target_y).
bool
verify_key(const cpu::uint256_t& k,
           const cpu::uint256_t& target_x,
           const cpu::uint256_t& target_y)
{
    if (k.d[0] == 0 && k.d[1] == 0 && k.d[2] == 0 && k.d[3] == 0) {
        return false;  // 0*G is the identity, not a valid recovery.
    }
    cpu::ECPoint p;
    cpu::ec_mul(p, k);
    cpu::uint256_t px, py;
    cpu::ec_to_affine(px, py, p);
    return (px == target_x) && (py == target_y);
}

// Map key for the host-side DP table. unordered_map<std::string, ...>
// is the simplest collision-correct hashable container for 32-byte
// keys without pulling in xxHash; X coordinates are uniform so the
// stdlib's string hash is fine here.
inline std::string x_be_key(const uint8_t x_be[32]) {
    return std::string(reinterpret_cast<const char*>(x_be), 32);
}

}  // namespace

// ---------------------------------------------------------------------------
// MultiGPUKangarooManager::Impl -- Metal backend
// ---------------------------------------------------------------------------

struct MultiGPUKangarooManager::Impl {
    KangarooMetalSolver solver;
    KangarooMetalConfig cfg;

    cpu::uint256_t range_start{};
    cpu::uint256_t range_end{};
    cpu::uint256_t target_x{};
    cpu::uint256_t target_y{};
    bool target_set = false;

    // Collision table: x_be -> (d_be, type). Two entries with the same
    // X but opposite types -> a recoverable collision. We keep ALL DPs
    // (not just opposites) because the same X can appear repeatedly
    // for the same kangaroo path; we resolve at insert time.
    struct StoredDP {
        std::array<uint8_t, 32> d_be;
        uint8_t type;
    };
    std::unordered_map<std::string, StoredDP> dp_table;
};

MultiGPUKangarooManager::MultiGPUKangarooManager()
    : impl_(new Impl()) {}

MultiGPUKangarooManager::~MultiGPUKangarooManager() {
    delete impl_;
}

bool MultiGPUKangarooManager::init(const std::vector<int>& /*gpu_ids*/) {
    // Mac has a single integrated GPU per device; gpu_ids is honored
    // for API parity but the solver only ever uses [0]. If the caller
    // requested specific IDs we still log it for visibility.
    impl_->cfg = KangarooMetalConfig{};
    impl_->cfg.dp_bits         = dp_bits;
    impl_->cfg.num_kangaroos   = static_cast<uint32_t>(num_kangaroos_per_gpu);
    impl_->cfg.steps_per_round = static_cast<uint32_t>(steps_per_round);
    impl_->cfg.work_id         = 0;  // standalone path doesn't use work_id semantics

    if (!impl_->solver.init(impl_->cfg)) {
        std::cerr << "[Metal] MultiGPUKangarooManager init failed: "
                  << impl_->solver.error() << "\n";
        return false;
    }
    return true;
}

int MultiGPUKangarooManager::num_gpus() const {
    return 1;
}

void MultiGPUKangarooManager::set_range(const UInt256& start, const UInt256& end) {
    // UInt256::parts and cpu::uint256_t::d are both 4xu64 little-endian
    // limb arrays with identical encoding -- direct copy is correct.
    for (int i = 0; i < 4; ++i) {
        impl_->range_start.d[i] = start.parts[i];
        impl_->range_end.d[i]   = end.parts[i];
    }
}

void MultiGPUKangarooManager::set_target_h160(const std::array<uint8_t, 20>& /*h160*/) {
    // Kangaroo intrinsically requires the public key, not just the
    // address. The caller is expected to follow set_target_h160 with
    // set_target_pubkey before calling solve(); this stub exists for
    // API parity with the CUDA path's pubkey-resolution flow.
}

void MultiGPUKangarooManager::set_target_pubkey(
    const cpu::uint256_t& x, const cpu::uint256_t& y)
{
    impl_->target_x   = x;
    impl_->target_y   = y;
    impl_->target_set = true;
}

GPUKangarooResult MultiGPUKangarooManager::solve() {
    GPUKangarooResult result{};
    result.found = false;

    if (!impl_->target_set) {
        std::cerr << "[Metal] solve() called before set_target_pubkey -- "
                     "kangaroo requires the target public key, not just "
                     "an address.\n";
        return result;
    }

    // Push current dp_bits + sizing back onto the solver if the caller
    // changed them after init() (puzzle_solver.cpp does -- it computes
    // optimal dp_bits AFTER init() returns). The solver's dp_bits is
    // stored as the immutable config; KangarooMetalSolver doesn't
    // currently expose a setter, so we re-init when it changed.
    if (impl_->cfg.dp_bits != dp_bits) {
        impl_->cfg.dp_bits = dp_bits;
        if (!impl_->solver.init(impl_->cfg)) {
            std::cerr << "[Metal] re-init for new dp_bits failed: "
                      << impl_->solver.error() << "\n";
            return result;
        }
    }

    // Build jump table and seed kangaroos. Use a per-solve work_id
    // derived from steady_clock so re-runs against the same target
    // get distinct walk seeds.
    const uint64_t work_id = static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    impl_->cfg.work_id = work_id;
    impl_->solver.set_work_id(work_id);

    auto jumps = build_jump_table(impl_->cfg.dp_bits);
    if (!impl_->solver.set_jump_table(jumps)) {
        std::cerr << "[Metal] set_jump_table failed: "
                  << impl_->solver.error() << "\n";
        return result;
    }

    std::mt19937_64 rng(work_id ^ 0x9E3779B97F4A7C15ULL);
    std::vector<KangarooSeed> seeds;
    seeds.reserve(impl_->cfg.num_kangaroos);
    const uint32_t half = impl_->cfg.num_kangaroos / 2u;
    for (uint32_t i = 0; i < impl_->cfg.num_kangaroos; ++i) {
        cpu::uint256_t off{};
        off.d[0] = rng();
        seeds.push_back(build_one_seed(i < half, off,
                                       impl_->range_start,
                                       impl_->target_x, impl_->target_y));
    }
    if (!impl_->solver.seed_kangaroos(seeds)) {
        std::cerr << "[Metal] seed_kangaroos failed: "
                  << impl_->solver.error() << "\n";
        return result;
    }

    impl_->dp_table.clear();
    impl_->dp_table.reserve(1u << 20);

    const auto loop_start = std::chrono::steady_clock::now();
    uint64_t total_dp  = 0;
    uint64_t total_ops = 0;
    std::vector<uint32_t> dead;
    dead.reserve(16);

    while (!stop_flag.load(std::memory_order_relaxed)) {
        auto dps = impl_->solver.step_round();
        total_ops += static_cast<uint64_t>(impl_->cfg.num_kangaroos)
                   * impl_->cfg.steps_per_round;

        for (const auto& dp : dps) {
            ++total_dp;
            std::string key = x_be_key(dp.x_be);

            auto it = impl_->dp_table.find(key);
            if (it == impl_->dp_table.end()) {
                Impl::StoredDP s{};
                std::memcpy(s.d_be.data(), dp.d_be, 32);
                s.type = dp.type;
                impl_->dp_table.emplace(std::move(key), s);
                continue;
            }

            // Same X collided with a stored DP. If types match it's the
            // same kangaroo's path -- ignore (refresh d to the newer
            // distance to keep the table fresh, though this rarely
            // matters for correctness).
            if (it->second.type == dp.type) {
                std::memcpy(it->second.d_be.data(), dp.d_be, 32);
                continue;
            }

            // Opposite-type collision -- candidate recovery.
            cpu::uint256_t d_new, d_old;
            ::collider::be32_to_limbs_le(dp.d_be, d_new.d);
            ::collider::be32_to_limbs_le(it->second.d_be.data(), d_old.d);

            cpu::uint256_t d_tame, d_wild;
            if (dp.type == 0) {  // new is tame, stored is wild
                d_tame = d_new;
                d_wild = d_old;
            } else {              // new is wild, stored is tame
                d_tame = d_old;
                d_wild = d_new;
            }

            cpu::uint256_t k = mod_n_sub(d_tame, d_wild);
            if (verify_key(k, impl_->target_x, impl_->target_y)) {
                result.found        = true;
                result.private_key  = k;
                result.total_steps  = total_ops;
                result.dp_count     = total_dp;
                const auto end_time = std::chrono::steady_clock::now();
                result.elapsed_seconds =
                    std::chrono::duration<double>(end_time - loop_start).count();
                return result;
            }
            // Mismatch survives verification: a hash-only collision
            // that wasn't an actual recovery (extremely rare on a
            // proper kangaroo walk -- typically signals a corrupt
            // distance value). Keep the new entry and move on.
            std::memcpy(it->second.d_be.data(), dp.d_be, 32);
            it->second.type = dp.type;
        }

        // Identity guard -- same as pool backend.
        if (impl_->solver.find_dead_kangaroos(dead)) {
            for (uint32_t idx : dead) {
                cpu::uint256_t off{};
                off.d[0] = rng();
                const bool is_tame = (idx < half);
                impl_->solver.replace_seed(
                    idx,
                    build_one_seed(is_tame, off, impl_->range_start,
                                   impl_->target_x, impl_->target_y));
            }
        }

        if (progress_callback) {
            const auto   now     = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - loop_start).count();
            const double rate    = (elapsed > 0.0)
                ? static_cast<double>(total_ops) / elapsed
                : 0.0;
            if (!progress_callback(total_ops, total_dp, rate)) {
                stop_flag.store(true, std::memory_order_relaxed);
            }
        }
    }

    const auto end_time = std::chrono::steady_clock::now();
    result.elapsed_seconds =
        std::chrono::duration<double>(end_time - loop_start).count();
    result.total_steps = total_ops;
    result.dp_count    = total_dp;
    return result;
}

}  // namespace gpu
}  // namespace collider
