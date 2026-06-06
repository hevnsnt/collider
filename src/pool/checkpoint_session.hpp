// Client-side checkpoint-replay session state + wire helpers (v1.5.4, task #9).
//
// Ties together the three already-existing pieces -- the Merkle commit
// (core/checkpoint_commit.hpp), the CHALLENGE_RSP builder
// (pool/challenge_response.hpp), and the canonical CPU walk
// (core/checkpoint_walk.hpp) -- into the per-connection bookkeeping the live
// pool client needs:
//
//   * DP_BATCH_V3 encoding (count u32 + 114-byte V3 records), so the client
//     can emit a checkpoint commitment alongside each DP when the server
//     speaks protocol_version >= 4.
//   * a retained store of the most recently committed walk's checkpoint
//     distances (keyed by work_id), so when the server later sends CHALLENGE
//     the client can reveal the requested segments + Merkle proofs.
//   * the CHALLENGE -> CHALLENGE_RSP transform (decode the challenge, look up
//     the retained checkpoints, build the response payload).
//
// All functions are pure over explicit state (a CheckpointSession plus the
// raw bytes) so they are unit-testable without a socket, a GPU, or the
// 3000-line JLPPoolClient. The wire byte layouts are pinned to the Python
// server codecs (collision-protocol/src/jlp_protocol.py) and cross-checked by
// tests/test_checkpoint_session.cpp + the Python round-trip test.

#pragma once

#include "core/checkpoint_commit.hpp"
#include "pool/challenge_response.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <vector>

namespace collider {
namespace pool {

// A committed walk: the ordered checkpoint distances (32-byte big-endian
// scalars), the Merkle root the client put on the wire, and the work_id the
// DP belonged to. n_segments == distances.size() - 1.
struct CommittedWalk {
    uint64_t work_id = 0;
    std::vector<checkpoint_commit::Distance> distances;
    checkpoint_commit::Hash root{};

    uint32_t n_segments() const {
        return distances.empty()
                   ? 0u
                   : static_cast<uint32_t>(distances.size() - 1);
    }
    bool empty() const { return distances.size() < 2; }
};

// Per-connection checkpoint-replay session. Holds the most recently committed
// walk so an incoming CHALLENGE can be answered. The server issues at most one
// outstanding challenge per worker and keys it to a work_id + nonce, so a
// single retained walk per work_id is sufficient; we keep the latest.
//
// Thread-safety: the live client commits walks from the sender thread and
// answers challenges from the receiver thread, so the store is mutex-guarded.
class CheckpointSession {
public:
    // Record the walk that produced a DP we are about to emit as V3. Stored so
    // a later CHALLENGE for this work_id can be answered. Overwrites any prior
    // retained walk for the same work_id (latest commitment wins, matching the
    // server's single-outstanding-challenge-per-worker model).
    void retain(const CommittedWalk& walk) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_ = walk;
    }

    // Drop any retained walk (e.g. on work_id change or disconnect).
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_.reset();
    }

    bool has_commitment() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return latest_.has_value() && !latest_->empty();
    }

    // Build a CHALLENGE_RSP payload for a received CHALLENGE payload. Returns
    // false (and leaves `out` cleared) if there is no retained walk, the
    // work_id does not match, an index is out of range, or the proof build
    // fails. On success `out` is the exact payload the server's
    // decode_challenge_rsp consumes (NOT framed; the caller adds the JLP
    // header). The work_id + nonce are echoed back from the challenge.
    bool answer_challenge(const std::vector<uint8_t>& challenge_payload,
                          std::vector<uint8_t>& out) const {
        out.clear();
        uint64_t work_id = 0;
        std::array<uint8_t, 8> nonce{};
        std::vector<uint32_t> indices;
        if (!decode_challenge(challenge_payload, work_id, nonce, indices))
            return false;

        std::lock_guard<std::mutex> lock(mutex_);
        if (!latest_.has_value() || latest_->empty()) return false;
        if (latest_->work_id != work_id) return false;
        // Every requested index must reference a committed segment.
        for (uint32_t idx : indices) {
            if (static_cast<size_t>(idx) + 1 >= latest_->distances.size())
                return false;
        }
        return build_challenge_rsp_payload(work_id, nonce, latest_->distances,
                                           indices, out);
    }

    // Decode a CHALLENGE payload: <Q work_id><8s nonce><H count><I idx>*count.
    // Mirrors JLPProtocol.decode_challenge. Returns false on truncation or a
    // count above the DoS cap (kMaxChallengeSegments).
    static bool decode_challenge(const std::vector<uint8_t>& p,
                                 uint64_t& work_id,
                                 std::array<uint8_t, 8>& nonce,
                                 std::vector<uint32_t>& indices) {
        indices.clear();
        if (p.size() < 18) return false;
        work_id = 0;
        for (int i = 0; i < 8; ++i)
            work_id |= static_cast<uint64_t>(p[i]) << (8 * i);
        std::memcpy(nonce.data(), p.data() + 8, 8);
        const uint16_t count =
            static_cast<uint16_t>(p[16]) | (static_cast<uint16_t>(p[17]) << 8);
        if (count > kMaxChallengeSegments) return false;
        const size_t need = 18 + static_cast<size_t>(count) * 4;
        if (p.size() < need) return false;
        indices.reserve(count);
        size_t off = 18;
        for (uint16_t i = 0; i < count; ++i) {
            uint32_t idx = 0;
            for (int b = 0; b < 4; ++b)
                idx |= static_cast<uint32_t>(p[off + b]) << (8 * b);
            indices.push_back(idx);
            off += 4;
        }
        return true;
    }

private:
    mutable std::mutex mutex_;
    std::optional<CommittedWalk> latest_;
};

// Encode a DP_BATCH_V3 payload from V3 DP records. Wire format matches
// JLPProtocol.encode_dp_batch_v3: [count u32 LE][dp 114]*count, where each DP
// is the little-endian JLPDistinguishedPointV3 struct laid out by
// struct.pack('<QI32s32sBB32sI'). The struct is #pragma pack(1) and the host
// is little-endian, so a direct memcpy of the struct is byte-identical to the
// Python pack; we assemble field-by-field anyway to stay endianness-explicit
// and independent of struct padding assumptions.
template <typename V3Dp>
inline void encode_dp_batch_v3(const std::vector<V3Dp>& dps,
                               std::vector<uint8_t>& out) {
    out.clear();
    const uint32_t count = static_cast<uint32_t>(dps.size());
    for (int i = 0; i < 4; ++i)
        out.push_back(static_cast<uint8_t>((count >> (8 * i)) & 0xFF));
    for (const auto& dp : dps) {
        for (int i = 0; i < 8; ++i)
            out.push_back(static_cast<uint8_t>((dp.work_id >> (8 * i)) & 0xFF));
        for (int i = 0; i < 4; ++i)
            out.push_back(static_cast<uint8_t>((dp.sequence >> (8 * i)) & 0xFF));
        out.insert(out.end(), dp.x, dp.x + 32);
        out.insert(out.end(), dp.d, dp.d + 32);
        out.push_back(dp.type);
        out.push_back(dp.dp_bits);
        out.insert(out.end(), dp.ckpt_root, dp.ckpt_root + 32);
        for (int i = 0; i < 4; ++i)
            out.push_back(static_cast<uint8_t>((dp.n_segments >> (8 * i)) & 0xFF));
    }
}

}  // namespace pool
}  // namespace collider
