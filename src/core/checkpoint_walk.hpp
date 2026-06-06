// CPU-reference checkpoint-walk generator for the v1.5.4 checkpoint-replay
// anti-cheat (client side, task #9).
//
// This is the canonical, hardware-independent model of one kangaroo walk. It
// produces, for a walk that starts at a known birth distance, the ordered
// sequence of checkpoint DISTANCE scalars (one every CHECKPOINT_INTERVAL
// jumps) plus the loop-state (L1S2) bit at each checkpoint. Those distances
// are exactly what checkpoint_commit.hpp commits to (Merkle root) and what
// challenge_response.hpp reveals; the pool server replays the same forward
// jumps to verify a challenged segment.
//
// It is a byte-for-byte mirror of the SERVER reference
// (collision-protocol/src/jump_table.py + src/checkpoint_replay.py:walk_step):
//   * build_jump_table: std::mt19937_64 seeded 0, jmp1 then jmp2, each
//     JMP_CNT entries = minjump + RndMax(minjump) with the low bit cleared.
//   * walk_step: h = x mod 512 indexes the table the loop-state bit selects
//     (jmp2 when set, else jmp1); odd y subtracts the jump distance
//     (endomorphism), even y adds; the L1S2 transition reproduces
//     RCGpuCore.cu:218-228 with the asymmetric INV_FLAG convention.
//
// WHY a CPU reference exists alongside the GPU capture: the live GPU kernel
// (third_party/RCKangaroo/RCGpuCore.cu) walks millions of kangaroos in
// parallel and does not retain per-kangaroo distance history, so true
// per-DP checkpoint capture is a kernel change that needs on-hardware
// validation (see rckangaroo_wrapper.cu, COLLIDER_CHECKPOINT_CAPTURE). This
// CPU model is the verifiable core: it lets the commit / emit / challenge /
// proof path be exercised end-to-end on CPU and cross-checked against the
// Python verifier without a GPU, and it is the oracle the GPU capture must
// reproduce once hardware validation is possible.
//
// Cross-checked against the Python reference by tests/test_checkpoint_walk.cpp
// (jump-table KAT + walk-distance KAT + honest round-trip the Python
// CheckpointReplayVerifier accepts).

#pragma once

#include "core/crypto_cpu.hpp"
#include "core/checkpoint_commit.hpp"

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace collider {
namespace checkpoint_walk {

// JMP_CNT = 512 (third_party/RCKangaroo/defs.h:54). h(P) = x mod JMP_CNT.
constexpr int kJmpCnt = 512;
// INV_FLAG = 0x4000 (third_party/RCKangaroo/defs.h:113); used only in the
// L1S2 loop-detect comparison, exactly as the kernel and the Python do.
constexpr uint32_t kInvFlag = 0x4000;
// Jumps per committed checkpoint. MUST equal jlp_protocol.py CHECKPOINT_INTERVAL
// and jlp_wire PROTOCOL CHECKPOINT_INTERVAL. Tests override with a small value.
constexpr uint64_t kCheckpointInterval = 65536;

// --- std::mt19937_64 reproduction -----------------------------------------
//
// The jump distances must be byte-for-byte identical to the client's
// RCKangaroo generation and the server's jump_table.py. The C++ standard
// pins std::mt19937_64's algorithm and constants, and jump_table.py's
// MT19937_64 was validated against it; we use the standard library engine
// directly so there is one canonical generator, not a hand-rolled copy.
// EcInt::RndBits / RndMax (Ec.cpp) draw 64-bit limbs from this stream.

// 256-bit unsigned scalar, little-endian limbs. Jump distances are well under
// 2^192 but the walk distance is tracked mod n (a full 256-bit modulus), so
// the type is a full 256-bit value. Reuses the field uint256_t representation
// from crypto_cpu so it interops with the EC point routines.
using Scalar = cpu::uint256_t;

// secp256k1 group order n (little-endian limbs); distances are reduced mod n
// exactly as checkpoint_replay.py uses ec.n.
inline const Scalar& curve_order() { return cpu::SECP256K1_N; }

// RndBits(nbits): fill ceil(nbits/64) limbs from the engine, mask the top
// limb to nbits%64. Mirrors jump_table.py::_rnd_bits / Ec.cpp:704-714.
inline Scalar rnd_bits(std::mt19937_64& rng, int nbits) {
    if (nbits > 256) nbits = 256;
    Scalar v;
    const int limbs = (nbits + 63) / 64;
    for (int i = 0; i < limbs; ++i) v.d[i] = rng();
    const int top = nbits / 64;
    const int rem = nbits % 64;
    if (rem) v.d[top] &= ((uint64_t(1) << rem) - 1);
    return v;
}

// Bit length of a scalar (position of the highest set bit + 1).
inline int bit_length(const Scalar& s) {
    for (int limb = 3; limb >= 0; --limb) {
        if (s.d[limb] == 0) continue;
        uint64_t v = s.d[limb];
        int b = 0;
        while (v) { v >>= 1; ++b; }
        return limb * 64 + b;
    }
    return 0;
}

// RndMax(bound): uniform in [0, bound) by masked rejection sampling on the
// bound's bit length. Mirrors jump_table.py::_rnd_max / Ec.cpp:717-736.
inline Scalar rnd_max(std::mt19937_64& rng, const Scalar& bound) {
    if (bound.is_zero()) return Scalar(0);
    const int bits = bit_length(bound);
    Scalar draw;
    do {
        draw = rnd_bits(rng, bits);
    } while (!(draw < bound));
    return draw;
}

// One jump table: JMP_CNT scalar distances. jmp1 and jmp2 are drawn from the
// SAME engine in order, so they must be generated together.
struct JumpTable {
    int range_bits = 0;
    std::array<Scalar, kJmpCnt> jmp1{};
    std::array<Scalar, kJmpCnt> jmp2{};
};

// minjump = 1 << shift, as a Scalar (shift < 256).
inline Scalar one_shifted(int shift) {
    Scalar s(0);
    s.d[shift / 64] = uint64_t(1) << (shift % 64);
    return s;
}

// build_jump_table: seed 0, jmp1 (minjump 2^(range_bits/2+3)) then jmp2
// (minjump 2^(range_bits-10)), each entry minjump + RndMax(minjump) with the
// low bit cleared. Byte-for-byte mirror of jump_table.py::build_jump_table.
inline JumpTable build_jump_table(int range_bits) {
    std::mt19937_64 rng(0);
    JumpTable jt;
    jt.range_bits = range_bits;
    auto gen = [&](std::array<Scalar, kJmpCnt>& out, int min_shift) {
        const Scalar minjump = one_shifted(min_shift);
        for (int i = 0; i < kJmpCnt; ++i) {
            Scalar dist;
            cpu::add256(dist, minjump, rnd_max(rng, minjump));  // minjump + r
            dist.d[0] &= ~uint64_t(1);                          // force even
            out[i] = dist;
        }
    };
    gen(jt.jmp1, range_bits / 2 + 3);
    gen(jt.jmp2, range_bits - 10);
    return jt;
}

// h(P) = low 9 bits of x = x mod 512 (RCGpuCore.cu:158 x[0] % JMP_CNT).
inline int jump_index(const Scalar& x) {
    return static_cast<int>(x.d[0] % kJmpCnt);
}

// --- scalar arithmetic mod n -----------------------------------------------
//
// The walk distance d is a signed scalar tracked mod n: an odd-y step
// SUBTRACTS the jump distance, an even-y step ADDS it. checkpoint_replay.py
// uses Python big-int (d - dist / d + dist) then reduces mod n only when it
// records the checkpoint. We keep d reduced in [0, n) at every step, which is
// equivalent mod n and matches the recorded checkpoint values exactly.

// r = (a + b) mod n.
inline Scalar add_mod_n(const Scalar& a, const Scalar& b) {
    Scalar r;
    const uint64_t carry = cpu::add256(r, a, b);
    const Scalar& n = curve_order();
    if (carry || r >= n) {
        Scalar t;
        cpu::sub256(t, r, n);
        r = t;
    }
    return r;
}

// r = (a - b) mod n, with a, b already in [0, n).
inline Scalar sub_mod_n(const Scalar& a, const Scalar& b) {
    Scalar r;
    const uint64_t borrow = cpu::sub256(r, a, b);
    if (borrow) {
        Scalar t;
        cpu::add256(t, r, curve_order());
        r = t;
    }
    return r;
}

// reduce a (assumed < 2*n, i.e. a single jump distance well under n) mod n.
inline Scalar reduce_mod_n(const Scalar& a) {
    const Scalar& n = curve_order();
    if (a >= n) {
        Scalar r;
        cpu::sub256(r, a, n);
        return r;
    }
    return a;
}

// --- the canonical walk step -----------------------------------------------
//
// State carried between steps: the affine point P (x, y), the distance d in
// [0, n), and the loop-state bit l1s2. Mirrors
// checkpoint_replay.py:walk_step / RCGpuCore.cu:155-228 for one kangaroo.
struct WalkState {
    cpu::uint256_t px;   // affine x
    cpu::uint256_t py;   // affine y
    Scalar d;            // distance in [0, n)
    int l1s2 = 0;        // loop-state bit
};

// Build the affine point for a distance scalar on a TAME walk (d*G) or a WILD
// walk (PntA + d*G). The caller supplies PntA for wild walks (its affine
// coords); for tame walks PntA is ignored. Mirrors
// CheckpointReplayVerifier.reconstruct_checkpoint_point.
inline void point_for_distance(const Scalar& d, bool wild,
                               const cpu::uint256_t& pnta_x,
                               const cpu::uint256_t& pnta_y,
                               cpu::uint256_t& out_x, cpu::uint256_t& out_y) {
    cpu::ECPoint dG;
    cpu::ec_mul(dG, reduce_mod_n(d));
    if (wild) {
        cpu::ECPoint sum;
        cpu::ec_add(sum, dG, pnta_x, pnta_y);
        cpu::ec_to_affine(out_x, out_y, sum);
    } else {
        cpu::ec_to_affine(out_x, out_y, dG);
    }
}

// Advance one kernel step. Updates `st` in place. `jt` is the walk's jump
// table (built once for the run's range_bits).
inline void walk_step(WalkState& st, const JumpTable& jt) {
    const int h = jump_index(st.px);
    const Scalar dist = st.l1s2 ? jt.jmp2[h] : jt.jmp1[h];

    // jp = (dist mod n) * G; odd y negates (subtract), even y adds.
    cpu::ECPoint jp;
    cpu::ec_mul(jp, reduce_mod_n(dist));
    cpu::uint256_t jpx, jpy;
    cpu::ec_to_affine(jpx, jpy, jp);

    const bool y_odd = (st.py.d[0] & 1) != 0;
    if (y_odd) {
        // negate jp.y: p - jpy
        cpu::uint256_t neg_y;
        cpu::mod_sub(neg_y, cpu::SECP256K1_P, jpy);
        jpy = neg_y;
        st.d = sub_mod_n(st.d, reduce_mod_n(dist));
    } else {
        st.d = add_mod_n(st.d, reduce_mod_n(dist));
    }

    // P2 = P + jp (P in affine; lift to Jacobian Z=1 then mixed-add).
    cpu::ECPoint P;
    P.X = st.px;
    P.Y = st.py;
    P.Z = cpu::uint256_t(1);
    cpu::ECPoint P2;
    cpu::ec_add(P2, P, jpx, jpy);
    cpu::uint256_t nx, ny;
    cpu::ec_to_affine(nx, ny, P2);

    // L1S2 transition (RCGpuCore.cu:218-228, checkpoint_replay.py:91-102).
    int l1s2_next;
    if (st.l1s2) {
        l1s2_next = 0;  // jmp2 / loop-escape step always clears the bit.
    } else {
        const uint32_t jmp_ind = static_cast<uint32_t>(h)
            | (y_odd ? kInvFlag : 0u);
        const bool ny_odd = (ny.d[0] & 1) != 0;
        const uint32_t jmp_next = static_cast<uint32_t>(jump_index(nx))
            | (ny_odd ? 0u : kInvFlag);
        l1s2_next = (jmp_ind == jmp_next) ? 1 : 0;
    }

    st.px = nx;
    st.py = ny;
    st.l1s2 = l1s2_next;
}

// One captured checkpoint: the distance scalar (the leaf the Merkle tree
// commits) and the loop-state bit at that checkpoint (revealed in a challenge
// so the server's replay starts from the correct table).
struct Checkpoint {
    checkpoint_commit::Distance d_be{};  // 32-byte big-endian distance mod n
    int l1s2 = 0;
};

// Encode a Scalar (mod-n, little-endian limbs) as a 32-byte big-endian
// Distance, the on-wire / Merkle-leaf form.
inline checkpoint_commit::Distance to_be(const Scalar& s) {
    checkpoint_commit::Distance out{};
    for (int limb = 0; limb < 4; ++limb) {
        const uint64_t v = s.d[limb];
        // limb 0 is least significant -> rightmost 8 bytes.
        uint8_t* p = out.data() + (3 - limb) * 8;
        for (int b = 0; b < 8; ++b)
            p[b] = static_cast<uint8_t>((v >> (8 * (7 - b))) & 0xFF);
    }
    return out;
}

// Generate the ordered checkpoint sequence for a walk of `segments`
// CHECKPOINT_INTERVAL-jump segments, starting from birth distance `start_d`.
// `wild` selects the tame (d*G) or wild (PntA + d*G) birth point; PntA is the
// wild birth point (Q - range_start*G) in affine coords, ignored for tame.
//
// Returns segments+1 checkpoints: index 0 is the birth, index i is after
// i*interval jumps. The walk is deterministic in (jt, start_d, wild, PntA),
// so it reproduces the GPU walk's distances exactly once the GPU capture is
// validated to feed the same inputs.
inline std::vector<Checkpoint> generate_checkpoints(
    const JumpTable& jt, const Scalar& start_d, bool wild,
    const cpu::uint256_t& pnta_x, const cpu::uint256_t& pnta_y,
    uint64_t segments, uint64_t interval = kCheckpointInterval) {
    WalkState st;
    st.d = reduce_mod_n(start_d);
    st.l1s2 = 0;
    point_for_distance(st.d, wild, pnta_x, pnta_y, st.px, st.py);

    std::vector<Checkpoint> cps;
    cps.reserve(segments + 1);
    cps.push_back({to_be(st.d), st.l1s2});
    for (uint64_t seg = 0; seg < segments; ++seg) {
        for (uint64_t j = 0; j < interval; ++j) walk_step(st, jt);
        cps.push_back({to_be(st.d), st.l1s2});
    }
    return cps;
}

// Pull just the distance leaves (for checkpoint_commit::build_root /
// build_challenge_rsp_payload, which take a vector<Distance>).
inline std::vector<checkpoint_commit::Distance> distances_of(
    const std::vector<Checkpoint>& cps) {
    std::vector<checkpoint_commit::Distance> out;
    out.reserve(cps.size());
    for (const auto& c : cps) out.push_back(c.d_be);
    return out;
}

}  // namespace checkpoint_walk
}  // namespace collider
