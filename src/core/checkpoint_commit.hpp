// Client-side checkpoint Merkle commitment for the v1.5.4 checkpoint-replay
// anti-cheat. Byte-for-byte mirror of the server's
// collision-protocol/src/checkpoint_commit.py so a root built here verifies
// against the proofs the server checks.
//
// A checkpoint is the walk DISTANCE scalar (mod n) at that point, encoded
// 32-byte big-endian. Leaf = SHA256(0x00 || dist_be32); internal node =
// SHA256(0x01 || left || right); odd levels duplicate the last node. The
// domain-separation prefixes (0x00 leaf / 0x01 node) prevent leaf/node
// confusion, matching the Python.
//
// Cross-validate with tools/merkle_dump.cpp against the Python build_root.

#pragma once

#include "core/crypto_cpu.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace collider {
namespace checkpoint_commit {

using Hash = std::array<uint8_t, 32>;
using Distance = std::array<uint8_t, 32>;  // 32-byte big-endian scalar mod n

inline Hash leaf_hash(const Distance& dist_be) {
    uint8_t buf[33];
    buf[0] = 0x00;
    std::memcpy(buf + 1, dist_be.data(), 32);
    return cpu::SHA256::hash(buf, 33);
}

inline Hash node_hash(const Hash& a, const Hash& b) {
    uint8_t buf[65];
    buf[0] = 0x01;
    std::memcpy(buf + 1, a.data(), 32);
    std::memcpy(buf + 33, b.data(), 32);
    return cpu::SHA256::hash(buf, 65);
}

// Merkle root over the ordered checkpoint distances. Empty input -> all-zero
// root (matches the Python).
inline Hash build_root(const std::vector<Distance>& distances) {
    if (distances.empty()) {
        Hash zero{};
        return zero;
    }
    std::vector<Hash> level;
    level.reserve(distances.size());
    for (const auto& d : distances) level.push_back(leaf_hash(d));
    while (level.size() > 1) {
        if (level.size() % 2) level.push_back(level.back());  // dup last
        std::vector<Hash> next;
        next.reserve(level.size() / 2);
        for (size_t i = 0; i < level.size(); i += 2)
            next.push_back(node_hash(level[i], level[i + 1]));
        level.swap(next);
    }
    return level[0];
}

// One authentication-path element: a sibling hash and whether it sits to the
// RIGHT of the current node (so it is concatenated after). Matches the
// (sibling, sibling_is_right) tuples the Python emits and the wire codec packs.
struct ProofStep {
    Hash sibling;
    bool sibling_is_right;
};

inline std::vector<ProofStep> build_proof(const std::vector<Distance>& distances,
                                          size_t index) {
    std::vector<Hash> level;
    level.reserve(distances.size());
    for (const auto& d : distances) level.push_back(leaf_hash(d));
    std::vector<ProofStep> path;
    size_t idx = index;
    while (level.size() > 1) {
        // Defensive bounds: a caller index past the leaf count (or a non-power-
        // of-2 checkpoint set that walked idx out of range) must not read OOB.
        // Bail with the partial path built so far rather than indexing past
        // level.end(). For balanced trees idx is always in range here, so this
        // guard never fires and the happy-path output is byte-identical.
        if (idx >= level.size()) return path;
        if (level.size() % 2) level.push_back(level.back());  // dup last
        // After padding the level is even-sized, so the sibling of an in-range
        // idx is always in range; assert it explicitly and fail safe if not.
        if (idx % 2 == 0) {
            if (idx + 1 >= level.size()) return path;
            path.push_back({level[idx + 1], true});
        } else {
            if (idx == 0) return path;  // unreachable for in-range odd idx
            path.push_back({level[idx - 1], false});
        }
        std::vector<Hash> next;
        next.reserve(level.size() / 2);
        for (size_t i = 0; i < level.size(); i += 2)
            next.push_back(node_hash(level[i], level[i + 1]));
        level.swap(next);
        idx /= 2;
    }
    return path;
}

// Confirm `dist_be` is the committed checkpoint at `index` under `root`.
// Mirror of the Python verify_proof; used by the KAT and available to any
// client-side self-check before sending a CHALLENGE_RSP.
inline bool verify_proof(const Hash& root, const Distance& dist_be,
                         size_t index, const std::vector<ProofStep>& path) {
    Hash node = leaf_hash(dist_be);
    for (const auto& step : path) {
        node = step.sibling_is_right ? node_hash(node, step.sibling)
                                     : node_hash(step.sibling, node);
    }
    (void)index;  // index is implicit in the sibling-side flags, kept for
                  // signature parity with the Python verifier.
    return node == root;
}

}  // namespace checkpoint_commit
}  // namespace collider
