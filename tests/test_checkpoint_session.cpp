/**
 * test_checkpoint_session -- the client CheckpointSession (the per-connection
 * checkpoint-replay bookkeeping wired into the live pool client) decodes a
 * server CHALLENGE, answers it with a CHALLENGE_RSP byte-for-byte identical to
 * the Python server's encode_challenge_rsp, and encodes DP_BATCH_V3 frames
 * byte-for-byte identical to encode_dp_batch_v3.
 *
 * Golden bytes were produced from collision-protocol/src/jlp_protocol.py on
 * the fixture distances=[1,2,3,4], work_id=0x1122334455667788,
 * nonce=01..08, challenge indices=[0,2]. See header comments for the exact
 * regeneration command.
 *
 * Exit: 0 pass, 1 fail.
 */

#include "pool/checkpoint_session.hpp"
#include "pool/jlp_pool_client.hpp"  // collider::pool::JLPDistinguishedPointV3
#include "core/checkpoint_commit.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

namespace cc = collider::checkpoint_commit;
using collider::pool::CheckpointSession;
using collider::pool::CommittedWalk;

std::vector<uint8_t> from_hex(const std::string& h) {
    std::vector<uint8_t> out;
    out.reserve(h.size() / 2);
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return 0;
    };
    for (size_t i = 0; i + 1 < h.size(); i += 2)
        out.push_back(static_cast<uint8_t>((nib(h[i]) << 4) | nib(h[i + 1])));
    return out;
}

std::string to_hex(const std::vector<uint8_t>& v) {
    static const char* d = "0123456789abcdef";
    std::string s;
    s.reserve(v.size() * 2);
    for (uint8_t b : v) { s.push_back(d[b >> 4]); s.push_back(d[b & 0xF]); }
    return s;
}

cc::Distance be_distance(uint64_t v) {
    cc::Distance d{};
    for (int i = 0; i < 8; ++i)
        d[31 - i] = static_cast<uint8_t>((v >> (8 * i)) & 0xFF);
    return d;
}

const char* kChallenge =
    "8877665544332211010203040506070802000000000002000000";
const char* kChallengeRsp =
    "887766554433221101020304050607080200000000000000000000000000000000000000"
    "000000000000000000000000000000000001000000000000000000000000000000000000"
    "000000000000000000000000000202000158cc2f44d3a27866874701fbad573da9ad1cfd"
    "88fa3145531c822f20a58beea101be808de894637e4e4e4f6b169152b682ae0a7a4b050d"
    "ee8be36555b76b6388f90200001fd4247443c9440cb3c48c28851937196bc156032d70a9"
    "6c98e127ecb347e45f01be808de894637e4e4e4f6b169152b682ae0a7a4b050dee8be365"
    "55b76b6388f9020000000000000000000000000000000000000000000000000000000000"
    "000000000003000000000000000000000000000000000000000000000000000000000000"
    "000402000182f02cf2ac0074619e6d747c35e08b29431a16943ddf81cfd9065c004ee636"
    "4a000971c8a1ce81287ccbc95aa4f171a5f807fb13ea2118f56b99769459a64906ad0200"
    "00d9cf8add8675a1b25627d7b0ec33bc177cb3930b0b6e995d79c386b980b2f4d6000971"
    "c8a1ce81287ccbc95aa4f171a5f807fb13ea2118f56b99769459a64906ad";
const char* kDpBatchV3 =
    "01000000887766554433221107000000000102030405060708090a0b0c0d0e0f10111213"
    "1415161718191a1b1c1d1e1f00000000000000000000000000000000000000000000000000"
    "00000000000099011445f385494f9f6116ec5530e7e9e24e2fdf6388c47d72e29657ce7b86"
    "0c1484c303000000";

}  // namespace

int main() {
    namespace cwk = collider::pool;
    const uint64_t work_id = 0x1122334455667788ull;
    std::vector<cc::Distance> distances = {
        be_distance(1), be_distance(2), be_distance(3), be_distance(4)};

    // 1. Retain the committed walk, decode the server CHALLENGE, and produce a
    //    CHALLENGE_RSP byte-for-byte identical to the Python server.
    CommittedWalk walk;
    walk.work_id = work_id;
    walk.distances = distances;
    walk.root = cc::build_root(distances);

    CheckpointSession session;
    session.retain(walk);
    if (!session.has_commitment()) {
        std::fprintf(stderr, "FAIL: session reports no commitment after retain\n");
        return 1;
    }

    std::vector<uint8_t> chal = from_hex(kChallenge);
    std::vector<uint8_t> rsp;
    if (!session.answer_challenge(chal, rsp)) {
        std::fprintf(stderr, "FAIL: answer_challenge returned false\n");
        return 1;
    }
    if (to_hex(rsp) != std::string(kChallengeRsp)) {
        std::fprintf(stderr,
            "FAIL: CHALLENGE_RSP mismatch\n  got %s\n  exp %s\n",
            to_hex(rsp).c_str(), kChallengeRsp);
        return 1;
    }

    // 2. A CHALLENGE for the WRONG work_id is refused (no false reveal).
    std::vector<uint8_t> bad_chal = chal;
    bad_chal[0] ^= 0xFF;  // corrupt work_id
    std::vector<uint8_t> bad_rsp;
    if (session.answer_challenge(bad_chal, bad_rsp)) {
        std::fprintf(stderr, "FAIL: answered a challenge for the wrong work_id\n");
        return 1;
    }

    // 3. After clear(), there is no commitment to reveal.
    session.clear();
    if (session.has_commitment() || session.answer_challenge(chal, rsp)) {
        std::fprintf(stderr, "FAIL: session answered after clear()\n");
        return 1;
    }

    // 4. DP_BATCH_V3 encoder matches the Python encode_dp_batch_v3 wire bytes.
    collider::pool::JLPDistinguishedPointV3 dp{};
    dp.work_id = work_id;
    dp.sequence = 7;
    for (int i = 0; i < 32; ++i) dp.x[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 32; ++i) dp.d[i] = 0;
    dp.d[31] = 0x99;
    dp.type = 1;
    dp.dp_bits = 20;
    cc::Hash root = cc::build_root(distances);
    for (int i = 0; i < 32; ++i) dp.ckpt_root[i] = root[i];
    dp.n_segments = 3;

    std::vector<collider::pool::JLPDistinguishedPointV3> dps = {dp};
    std::vector<uint8_t> batch;
    cwk::encode_dp_batch_v3(dps, batch);
    if (to_hex(batch) != std::string(kDpBatchV3)) {
        std::fprintf(stderr,
            "FAIL: DP_BATCH_V3 mismatch\n  got %s\n  exp %s\n",
            to_hex(batch).c_str(), kDpBatchV3);
        return 1;
    }

    std::printf("PASS: CheckpointSession decodes CHALLENGE, answers byte-exact "
                "CHALLENGE_RSP, refuses wrong work_id / post-clear, and encodes "
                "DP_BATCH_V3 matching the Python server.\n");
    return 0;
}
