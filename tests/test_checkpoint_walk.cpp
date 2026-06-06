/**
 * test_checkpoint_walk -- the v1.5.4 client CPU-reference checkpoint-walk
 * generator (src/core/checkpoint_walk.hpp) reproduces the SERVER reference
 * (collision-protocol/src/jump_table.py + src/checkpoint_replay.py) exactly,
 * and the checkpoints it emits commit + reveal cleanly through
 * checkpoint_commit.hpp / challenge_response.hpp.
 *
 * The golden values below were generated from the Python reference on the
 * fixed fixture and pasted here so the C++ side is checked against the SAME
 * silicon-validated model the pool server replays (checkpoint_replay.py:30-34
 * notes that model was cross-checked byte-for-byte against a real GPU kernel
 * trace). Regenerate with:
 *   cd collision-protocol && python -c "
 *     from src.jump_table import build_jump_table
 *     from src.checkpoint_replay import CheckpointReplayVerifier
 *     from src import checkpoint_commit as cc
 *     from src.ec_backend import load_ec_backend
 *     RANGE_BITS=80; RANGE_START=1<<(RANGE_BITS-1); INTERVAL=16; SEGMENTS=8
 *     START_D=(1<<40)|0x12345; ec=load_ec_backend(); n=ec.n
 *     jt=build_jump_table(RANGE_BITS)
 *     comp=lambda P:(b'\\x02' if P.y%2==0 else b'\\x03')+P.x.to_bytes(32,'big')
 *     v=CheckpointReplayVerifier(RANGE_START,comp(ec.G).hex(),INTERVAL,ec)
 *     P=(START_D%n)*ec.G; d=START_D; l=0; cps=[d%n]; ll=[l]
 *     for _ in range(SEGMENTS):
 *       for _ in range(INTERVAL): P,d,l=v.walk_step(P,d,l)
 *       cps.append(d%n); ll.append(l)
 *     print(cc.build_root(cps).hex())"
 *
 * Exit: 0 pass, 1 fail.
 */

#include "core/checkpoint_walk.hpp"
#include "core/checkpoint_commit.hpp"
#include "pool/challenge_response.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace cw = collider::checkpoint_walk;
namespace cc = collider::checkpoint_commit;

constexpr int kRangeBits = 80;
constexpr uint64_t kInterval = 16;
constexpr uint64_t kSegments = 8;

std::string to_hex(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(d[p[i] >> 4]);
        s.push_back(d[p[i] & 0xF]);
    }
    return s;
}

// 64-hex-char big-endian -> Distance.
cc::Distance from_hex64(const char* h) {
    cc::Distance d{};
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return 0;
    };
    for (int i = 0; i < 32; ++i)
        d[i] = static_cast<uint8_t>((nib(h[2 * i]) << 4) | nib(h[2 * i + 1]));
    return d;
}

// Scalar (little-endian limbs) -> 64-char big-endian hex.
std::string scalar_hex(const cw::Scalar& s) {
    cc::Distance be = cw::to_be(s);
    return to_hex(be.data(), 32);
}

struct GoldenCp { const char* d_hex; int l1s2; };

// --- golden values from the Python reference (see header comment) ---
const char* kJmp1_0   = "00000000000000000000000000000000000000000000000000000fc5cb41dc3e";
const char* kJmp1_1   = "00000000000000000000000000000000000000000000000000000a17f032e8b8";
const char* kJmp1_2   = "000000000000000000000000000000000000000000000000000008133b0725ac";
const char* kJmp1_511 = "00000000000000000000000000000000000000000000000000000f4a0ab51378";
const char* kJmp2_0   = "00000000000000000000000000000000000000000000007c5c0fdfe84b7a98ec";
const char* kJmp2_1   = "00000000000000000000000000000000000000000000004170b9847d8a65a8f8";
const char* kJmp2_511 = "00000000000000000000000000000000000000000000006bc65e18614959616e";

const GoldenCp kGolden[] = {
    {"0000000000000000000000000000000000000000000000000000010000012345", 0},
    {"0000000000000000000000000000000000000000000000000000061752d1536f", 0},
    {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd22494ec5b4c50", 0},
    {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd1d1fc20ef4f4c", 0},
    {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd1ee6d6a62ddaa", 0},
    {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd221ce5cc57b40", 0},
    {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd242267c401864", 0},
    {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd256094fccd492", 0},
    {"fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd1f0e1059f0438", 0},
};
const char* kRoot =
    "d85de3ddfff41e80987a962a4ff39d790e3b3baa43ba46ddf5354755fcf25725";

bool eq_distance(const cc::Distance& a, const char* hex64) {
    return a == from_hex64(hex64);
}

}  // namespace

int main() {
    // 1. Jump table KAT: the C++ MT19937_64 + RndMax reproduction matches the
    //    Python jump_table.py for the sampled entries (both tables, the order
    //    matters because they share one RNG stream).
    cw::JumpTable jt = cw::build_jump_table(kRangeBits);
    struct { int tab; int idx; const char* hex; } jkat[] = {
        {1, 0, kJmp1_0}, {1, 1, kJmp1_1}, {1, 2, kJmp1_2}, {1, 511, kJmp1_511},
        {2, 0, kJmp2_0}, {2, 1, kJmp2_1}, {2, 511, kJmp2_511},
    };
    for (const auto& k : jkat) {
        const cw::Scalar& s = (k.tab == 1) ? jt.jmp1[k.idx] : jt.jmp2[k.idx];
        if (scalar_hex(s) != std::string(k.hex)) {
            std::fprintf(stderr,
                "FAIL: jmp%d[%d]\n  got %s\n  exp %s\n",
                k.tab, k.idx, scalar_hex(s).c_str(), k.hex);
            return 1;
        }
    }

    // 2. Walk KAT: the canonical walk reproduces the server's checkpoint
    //    distances + loop-state bits step-for-step. START_D = (1<<40)|0x12345.
    cw::Scalar start_d(0);
    start_d.d[0] = (uint64_t(1) << 40) | 0x12345ull;
    collider::cpu::uint256_t pnta_x, pnta_y;  // unused for a tame walk
    std::vector<cw::Checkpoint> cps = cw::generate_checkpoints(
        jt, start_d, /*wild=*/false, pnta_x, pnta_y, kSegments, kInterval);

    const size_t expected = sizeof(kGolden) / sizeof(kGolden[0]);
    if (cps.size() != expected) {
        std::fprintf(stderr, "FAIL: got %zu checkpoints, expected %zu\n",
                     cps.size(), expected);
        return 1;
    }
    for (size_t i = 0; i < cps.size(); ++i) {
        if (!eq_distance(cps[i].d_be, kGolden[i].d_hex)) {
            std::fprintf(stderr,
                "FAIL: checkpoint %zu distance\n  got %s\n  exp %s\n",
                i, to_hex(cps[i].d_be.data(), 32).c_str(), kGolden[i].d_hex);
            return 1;
        }
        if (cps[i].l1s2 != kGolden[i].l1s2) {
            std::fprintf(stderr, "FAIL: checkpoint %zu l1s2 got %d exp %d\n",
                         i, cps[i].l1s2, kGolden[i].l1s2);
            return 1;
        }
    }

    // 3. The Merkle root over the generated distances matches the Python root
    //    (cross-check checkpoint_walk -> checkpoint_commit against the server).
    std::vector<cc::Distance> dists = cw::distances_of(cps);
    cc::Hash root = cc::build_root(dists);
    if (to_hex(root.data(), 32) != std::string(kRoot)) {
        std::fprintf(stderr, "FAIL: root\n  got %s\n  exp %s\n",
                     to_hex(root.data(), 32).c_str(), kRoot);
        return 1;
    }

    // 4. Build a CHALLENGE_RSP over every segment and confirm each revealed
    //    endpoint's Merkle proof verifies against the committed root (the
    //    check the server runs before replaying). This is the full client
    //    commit -> challenge -> proof path on the generated walk.
    std::vector<uint32_t> indices;
    for (uint32_t i = 0; i + 1 < dists.size(); ++i) indices.push_back(i);
    std::array<uint8_t, 8> nonce = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint8_t> payload;
    if (!collider::pool::build_challenge_rsp_payload(
            0xABCDEF01u, nonce, dists, indices, payload)) {
        std::fprintf(stderr, "FAIL: build_challenge_rsp_payload over walk\n");
        return 1;
    }
    for (uint32_t idx : indices) {
        auto ps = cc::build_proof(dists, idx);
        auto pe = cc::build_proof(dists, idx + 1);
        if (!cc::verify_proof(root, dists[idx], idx, ps) ||
            !cc::verify_proof(root, dists[idx + 1], idx + 1, pe)) {
            std::fprintf(stderr, "FAIL: walk proof idx=%u\n", idx);
            return 1;
        }
    }

    // 5. NEGATIVE: a fabricated (off-walk) checkpoint set produces a DIFFERENT
    //    root, and the honest proofs do NOT verify against it. This is the
    //    structural property that defeats a grinder: it cannot fabricate a
    //    consistent committed chain it did not actually walk.
    std::vector<cc::Distance> forged = dists;
    forged[2][31] ^= 0x01;  // perturb one checkpoint
    cc::Hash forged_root = cc::build_root(forged);
    if (forged_root == root) {
        std::fprintf(stderr, "FAIL: forged root collided with honest root\n");
        return 1;
    }
    // The honest proof for leaf 2 must NOT verify the forged distance against
    // the honest root (the leaf hash differs).
    auto p2 = cc::build_proof(dists, 2);
    if (cc::verify_proof(root, forged[2], 2, p2)) {
        std::fprintf(stderr, "FAIL: forged checkpoint verified against honest root\n");
        return 1;
    }

    std::printf("PASS: checkpoint_walk reproduces the server jump table + "
                "walk (%zu checkpoints), root matches Python, commit/challenge "
                "round-trips, forged set rejected.\n", cps.size());
    return 0;
}
