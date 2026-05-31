// Client-side CHALLENGE_RSP construction for the v1.5.4 checkpoint-replay
// anti-cheat. Byte-for-byte mirror of JLPProtocol.encode_challenge_rsp +
// _pack_proof in collision-protocol/src/jlp_protocol.py, so a payload built
// here decodes and verifies on the server unchanged.
//
// On receiving a CHALLENGE (work_id, nonce, [segment indices]), the worker
// reveals, for each requested segment idx, the two endpoint checkpoint
// distances (d_start = checkpoints[idx], d_end = checkpoints[idx+1]) and their
// Merkle authentication paths against the committed root. The server replays
// CHECKPOINT_INTERVAL forward jumps to confirm the segment links up.
//
// This builds the CHALLENGE_RSP PAYLOAD; the caller frames it with the JLP
// message header (MAGIC|type|flags|length) the rest of the pool client uses.

#pragma once

#include "core/checkpoint_commit.hpp"  // Distance, ProofStep, build_proof

#include <array>
#include <cstdint>
#include <vector>

namespace collider {
namespace pool {

// DoS caps mirror jlp_protocol.py (MAX_CHALLENGE_SEGMENTS / MAX_PROOF_DEPTH).
constexpr uint16_t kMaxChallengeSegments = 64;
constexpr uint16_t kMaxProofDepth        = 64;

namespace detail {

inline void put_u16le(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}
inline void put_u32le(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
inline void put_u64le(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

// <H depth> then depth x [<B sibling_is_right><32s sibling>]. Mirrors
// JLPProtocol._pack_proof. Returns false if the path is implausibly deep.
inline bool append_proof(std::vector<uint8_t>& out,
                         const std::vector<checkpoint_commit::ProofStep>& proof) {
    if (proof.size() > kMaxProofDepth) return false;
    put_u16le(out, static_cast<uint16_t>(proof.size()));
    for (const auto& step : proof) {
        out.push_back(step.sibling_is_right ? 1u : 0u);
        out.insert(out.end(), step.sibling.begin(), step.sibling.end());
    }
    return true;
}

}  // namespace detail

// Build the CHALLENGE_RSP payload for `indices` against the walk's ordered
// checkpoint `distances` (32-byte big-endian scalars). Each segment idx must
// satisfy idx + 1 < distances.size(). Returns false (and leaves `out`
// unspecified) on any out-of-range index, too many segments, or an
// over-deep proof. On success `out` is the exact payload the server's
// JLPProtocol.decode_challenge_rsp consumes.
inline bool build_challenge_rsp_payload(
    uint64_t work_id,
    const std::array<uint8_t, 8>& nonce,
    const std::vector<checkpoint_commit::Distance>& distances,
    const std::vector<uint32_t>& indices,
    std::vector<uint8_t>& out) {
    if (indices.size() > kMaxChallengeSegments) return false;

    out.clear();
    detail::put_u64le(out, work_id);
    out.insert(out.end(), nonce.begin(), nonce.end());
    detail::put_u16le(out, static_cast<uint16_t>(indices.size()));

    for (uint32_t idx : indices) {
        // Segment idx spans checkpoints[idx] .. checkpoints[idx+1]; both must
        // exist (the server verifies d_end at leaf idx+1).
        if (static_cast<size_t>(idx) + 1 >= distances.size()) return false;

        detail::put_u32le(out, idx);
        const auto& d_start = distances[idx];
        const auto& d_end   = distances[idx + 1];
        out.insert(out.end(), d_start.begin(), d_start.end());
        out.insert(out.end(), d_end.begin(), d_end.end());

        if (!detail::append_proof(out, checkpoint_commit::build_proof(distances, idx)))
            return false;
        if (!detail::append_proof(out, checkpoint_commit::build_proof(distances, idx + 1)))
            return false;
    }
    return true;
}

}  // namespace pool
}  // namespace collider
