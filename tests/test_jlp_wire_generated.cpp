// test_jlp_wire_generated.cpp
//
// Drift detector. Confirms that the legacy JLPHeader/JLPMessageType/
// JLPDistinguishedPoint/JLPServerConfig/JLPDistinguishedPointV2 types
// in src/pool/jlp_pool_client.hpp stay byte-equivalent to the
// auto-generated counterparts in src/pool/jlp_wire_generated.hpp
// (and therefore byte-equivalent to the Python bindings derived from
// the same IDL).
//
// This is a static_assert + runtime equality test, not a network test.
// If anyone hand-edits the legacy structs without updating jlp.yaml
// (or vice versa) this fails at build time.

#include "pool/jlp_pool_client.hpp"
#include "pool/jlp_wire_generated.hpp"

#include <cassert>
#include <cstddef>   // offsetof
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace collider::pool;

// ---- compile-time: sizes ---------------------------------------------------

static_assert(sizeof(JLPHeader) == sizeof(jlp_wire::Header),
              "JLPHeader and jlp_wire::Header have diverged");
static_assert(sizeof(jlp_wire::Header) == 8,
              "JLP header is 8 bytes by spec");

static_assert(sizeof(JLPDistinguishedPoint) == sizeof(jlp_wire::DistinguishedPoint),
              "DP v1 layout has diverged");
static_assert(sizeof(jlp_wire::DistinguishedPoint) == 66,
              "DP v1 wire size is 66 by spec");

static_assert(sizeof(JLPDistinguishedPointV2) == sizeof(jlp_wire::DistinguishedPointV2),
              "DP v2 layout has diverged");
// v1.4.1 B.1: bumped from 74 to 78 by adding a 4-byte LE sequence
// field between work_id and x for per-(worker, work_id) replay
// detection. See protocol/jlp.yaml DistinguishedPointV2 wire_size.
static_assert(sizeof(jlp_wire::DistinguishedPointV2) == 78,
              "DP v2 wire size is 78 by spec (post-1.4.1 B.1 sequence)");

// v1.5 (protocol_version=3): WorkAssignment grew to 126 bytes with
// three trailing asymmetric-assignment fields (kangaroo_type,
// start_offset_a, start_offset_b). The handwritten JLPServerConfig
// in jlp_pool_client.hpp was extended by client-runtime (Wave 2 task #5)
// to match the new 126-byte layout; the equality drift detector below
// is reactivated. Both struct sizes are pinned to 126 by the second
// assert so any future schema change without a parallel struct update
// trips at build time.
static_assert(sizeof(JLPServerConfig) == sizeof(jlp_wire::WorkAssignment),
              "WorkAssignment layout has diverged");
static_assert(sizeof(jlp_wire::WorkAssignment) == 126,
              "v1.5: WorkAssignment wire size is 126 by spec "
              "(109 head + 1 kangaroo_type + 8 start_offset_a + 8 start_offset_b)");
static_assert(sizeof(JLPServerConfig) == 126,
              "v1.5: JLPServerConfig must be 126 bytes after task #5 extension");

// Field-by-field offset checks for the new v1.5 fields so a future
// reorder is caught by both sizeof equality AND member-position equality.
static_assert(offsetof(JLPServerConfig, public_key) ==
              offsetof(jlp_wire::WorkAssignment, public_key),
              "WorkAssignment.public_key offset drifted");
static_assert(offsetof(JLPServerConfig, range_start) ==
              offsetof(jlp_wire::WorkAssignment, range_start),
              "WorkAssignment.range_start offset drifted");
static_assert(offsetof(JLPServerConfig, range_end) ==
              offsetof(jlp_wire::WorkAssignment, range_end),
              "WorkAssignment.range_end offset drifted");
static_assert(offsetof(JLPServerConfig, dp_bits) ==
              offsetof(jlp_wire::WorkAssignment, dp_bits),
              "WorkAssignment.dp_bits offset drifted");
static_assert(offsetof(JLPServerConfig, work_id) ==
              offsetof(jlp_wire::WorkAssignment, work_id),
              "WorkAssignment.work_id offset drifted");
static_assert(offsetof(JLPServerConfig, kangaroo_type) ==
              offsetof(jlp_wire::WorkAssignment, kangaroo_type),
              "WorkAssignment.kangaroo_type offset drifted");
static_assert(offsetof(JLPServerConfig, start_offset_a) ==
              offsetof(jlp_wire::WorkAssignment, start_offset_a),
              "WorkAssignment.start_offset_a offset drifted");
static_assert(offsetof(JLPServerConfig, start_offset_b) ==
              offsetof(jlp_wire::WorkAssignment, start_offset_b),
              "WorkAssignment.start_offset_b offset drifted");

// v1.4.2 B.3: AUTH drift detector. Pre-fix this was missing entirely, so
// the handwritten JLPClientHelloV2 (120 bytes) had drifted from the IDL
// (which still said AuthPayload was 96 bytes). With AuthPayloadV2 now in
// the IDL, any future drift between the handwritten struct and the
// codegen-emitted one will break the build.
static_assert(sizeof(JLPClientHelloV2) == sizeof(jlp_wire::AuthPayloadV2),
              "JLPClientHelloV2 and jlp_wire::AuthPayloadV2 have diverged");
static_assert(sizeof(jlp_wire::AuthPayloadV2) == 120,
              "AuthPayloadV2 wire size is 120 by spec");
// Field-by-field offset checks so a future reorder is caught by sizeof
// equality + member-position equality (sizeof alone misses same-total
// reorderings).
static_assert(offsetof(JLPClientHelloV2, worker_name) ==
              offsetof(jlp_wire::AuthPayloadV2, worker_name),
              "AuthPayloadV2.worker_name offset drifted");
static_assert(offsetof(JLPClientHelloV2, password) ==
              offsetof(jlp_wire::AuthPayloadV2, password),
              "AuthPayloadV2.password offset drifted");
static_assert(offsetof(JLPClientHelloV2, timestamp_ms) ==
              offsetof(jlp_wire::AuthPayloadV2, timestamp_ms),
              "AuthPayloadV2.timestamp_ms offset drifted");
static_assert(offsetof(JLPClientHelloV2, nonce) ==
              offsetof(jlp_wire::AuthPayloadV2, nonce),
              "AuthPayloadV2.nonce offset drifted");

// ---- compile-time: message-type values match ------------------------------

static_assert(static_cast<uint8_t>(JLPMessageType::AUTH) ==
              static_cast<uint8_t>(jlp_wire::MessageType::AUTH));
static_assert(static_cast<uint8_t>(JLPMessageType::AUTH_OK) ==
              static_cast<uint8_t>(jlp_wire::MessageType::AUTH_OK));
static_assert(static_cast<uint8_t>(JLPMessageType::AUTH_FAIL) ==
              static_cast<uint8_t>(jlp_wire::MessageType::AUTH_FAIL));
static_assert(static_cast<uint8_t>(JLPMessageType::WORK_REQ) ==
              static_cast<uint8_t>(jlp_wire::MessageType::WORK_REQ));
static_assert(static_cast<uint8_t>(JLPMessageType::WORK_ASN) ==
              static_cast<uint8_t>(jlp_wire::MessageType::WORK_ASN));
static_assert(static_cast<uint8_t>(JLPMessageType::DP_SUBMIT) ==
              static_cast<uint8_t>(jlp_wire::MessageType::DP_SUBMIT));
static_assert(static_cast<uint8_t>(JLPMessageType::DP_BATCH) ==
              static_cast<uint8_t>(jlp_wire::MessageType::DP_BATCH));
static_assert(static_cast<uint8_t>(JLPMessageType::DP_SUBMIT_V2) ==
              static_cast<uint8_t>(jlp_wire::MessageType::DP_SUBMIT_V2));
static_assert(static_cast<uint8_t>(JLPMessageType::DP_BATCH_V2) ==
              static_cast<uint8_t>(jlp_wire::MessageType::DP_BATCH_V2));
static_assert(static_cast<uint8_t>(JLPMessageType::SOLUTION) ==
              static_cast<uint8_t>(jlp_wire::MessageType::SOLUTION));
static_assert(static_cast<uint8_t>(JLPMessageType::PING) ==
              static_cast<uint8_t>(jlp_wire::MessageType::PING));
static_assert(static_cast<uint8_t>(JLPMessageType::PONG) ==
              static_cast<uint8_t>(jlp_wire::MessageType::PONG));
static_assert(static_cast<uint8_t>(JLPMessageType::MSG_ERROR) ==
              static_cast<uint8_t>(jlp_wire::MessageType::MSG_ERROR));

// ---- compile-time: protocol constants -------------------------------------

static_assert(jlp_wire::PROTOCOL_VERSION == 3,
              "v1.5: protocol bumped to 3 for asymmetric tame/wild work assignment");
static_assert(jlp_wire::MAX_BATCH_SIZE == 10000);
static_assert(jlp_wire::MAX_MESSAGE_SIZE == 1048576);

// ---- runtime: round-trip pack/unpack via memcpy on both sides --------------

namespace {

int failures = 0;

#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        std::fprintf(stderr, "FAIL: %s   (%s:%d)\n",                \
                     msg, __FILE__, __LINE__);                      \
        ++failures;                                                 \
    }                                                               \
} while (0)

void test_header_layout() {
    jlp_wire::Header h{};
    h.magic[0] = 'K'; h.magic[1] = 'A'; h.magic[2] = 'N'; h.magic[3] = 'G';
    h.type = static_cast<uint8_t>(jlp_wire::MessageType::PING);
    h.flags = 0;
    h.payload_size = 0x1234;

    uint8_t buf[sizeof(jlp_wire::Header)];
    std::memcpy(buf, &h, sizeof(buf));

    CHECK(buf[0] == 'K' && buf[1] == 'A' && buf[2] == 'N' && buf[3] == 'G',
          "header magic bytes");
    CHECK(buf[4] == 0x50, "header type byte for PING");
    CHECK(buf[5] == 0x00, "header flags byte");
    // little-endian uint16_t
    CHECK(buf[6] == 0x34 && buf[7] == 0x12, "header payload_size LE");
}

void test_dp_v2_layout() {
    jlp_wire::DistinguishedPointV2 dp{};
    dp.work_id = 0x1122334455667788ULL;
    // v1.4.1 B.1: sequence is a uint32 LE at offset 8, sliding x/d/
    // type/dp_bits forward by 4 bytes (DP V2 wire is now 78 bytes).
    dp.sequence = 0xDEADBEEFu;
    for (int i = 0; i < 32; ++i) dp.x[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 32; ++i) dp.d[i] = static_cast<uint8_t>(0x80 | i);
    dp.type = 1;
    dp.dp_bits = 24;

    uint8_t buf[sizeof(jlp_wire::DistinguishedPointV2)];
    std::memcpy(buf, &dp, sizeof(buf));

    // work_id must be the first 8 bytes, little-endian.
    CHECK(buf[0] == 0x88 && buf[1] == 0x77 && buf[2] == 0x66 && buf[3] == 0x55 &&
          buf[4] == 0x44 && buf[5] == 0x33 && buf[6] == 0x22 && buf[7] == 0x11,
          "dp_v2 work_id position + endianness");
    // sequence: 4 bytes LE between work_id and x (offsets 8..11).
    CHECK(buf[8] == 0xEF && buf[9] == 0xBE && buf[10] == 0xAD && buf[11] == 0xDE,
          "dp_v2 sequence position + endianness (v1.4.1 B.1)");
    CHECK(buf[12] == 0x00 && buf[13] == 0x01, "dp_v2 x[] follows sequence");
    CHECK(buf[44] == 0x80, "dp_v2 d[] follows x[]");
    CHECK(buf[76] == 1, "dp_v2 type at offset 76");
    CHECK(buf[77] == 24, "dp_v2 dp_bits at offset 77");
}

void test_auth_payload_v2_layout() {
    // v1.4.2 B.3: verify the byte layout that the Python pool server
    // actually decodes. struct format '<64s32sQ16s' means worker_name +
    // password + timestamp_ms LE + nonce.
    jlp_wire::AuthPayloadV2 ap{};
    for (int i = 0; i < 64; ++i) ap.worker_name[i] = (uint8_t)('A' + (i % 26));
    for (int i = 0; i < 32; ++i) ap.password[i] = (uint8_t)('a' + (i % 26));
    ap.timestamp_ms = 0x0123456789ABCDEFULL;
    for (int i = 0; i < 16; ++i) ap.nonce[i] = (uint8_t)(0xF0 | (i & 0x0F));

    uint8_t buf[sizeof(jlp_wire::AuthPayloadV2)];
    std::memcpy(buf, &ap, sizeof(buf));

    // worker_name at offset 0..63
    CHECK(buf[0] == 'A' && buf[25] == 'Z' && buf[26] == 'A',
          "AuthPayloadV2.worker_name at offset 0");
    // password at offset 64..95
    CHECK(buf[64] == 'a' && buf[89] == 'z',
          "AuthPayloadV2.password at offset 64");
    // timestamp_ms LE at offset 96..103
    CHECK(buf[96]  == 0xEF && buf[97]  == 0xCD &&
          buf[98]  == 0xAB && buf[99]  == 0x89 &&
          buf[100] == 0x67 && buf[101] == 0x45 &&
          buf[102] == 0x23 && buf[103] == 0x01,
          "AuthPayloadV2.timestamp_ms LE at offset 96");
    // nonce at offset 104..119
    CHECK(buf[104] == 0xF0 && buf[119] == 0xFF,
          "AuthPayloadV2.nonce at offset 104");
}

void test_work_assignment_v3_layout() {
    // v1.5 (protocol_version=3): WORK_ASN payload extends from 109 to
    // 126 bytes with kangaroo_type (u8) + start_offset_a (u64 LE) +
    // start_offset_b (u64 LE) at the tail. Verify the new fields land
    // at the right offsets and round-trip via memcpy.
    jlp_wire::WorkAssignment w{};
    for (int i = 0; i < 33; ++i) w.public_key[i] = 0x01;
    for (int i = 0; i < 32; ++i) w.range_start[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 32; ++i) w.range_end[i] = static_cast<uint8_t>(0xC0 | i);
    w.dp_bits = 28;
    w.work_id = 42;
    w.kangaroo_type = 1;  // wild
    w.start_offset_a = 0x00000000DEADBEEFULL;
    w.start_offset_b = 0x00000000DEADBEEFULL + 1024ULL;

    uint8_t buf[sizeof(jlp_wire::WorkAssignment)];
    std::memcpy(buf, &w, sizeof(buf));

    // Offset 0..32: public_key
    CHECK(buf[0] == 0x01 && buf[32] == 0x01, "WORK_ASN public_key at offset 0");
    // Offset 33..64: range_start
    CHECK(buf[33] == 0x00 && buf[64] == 31, "WORK_ASN range_start at offset 33");
    // Offset 65..96: range_end
    CHECK(buf[65] == 0xC0 && buf[96] == (0xC0 | 31), "WORK_ASN range_end at offset 65");
    // Offset 97..100: dp_bits LE (28 = 0x1C)
    CHECK(buf[97] == 0x1C && buf[98] == 0x00 && buf[99] == 0x00 && buf[100] == 0x00,
          "WORK_ASN dp_bits LE at offset 97");
    // Offset 101..108: work_id LE (42 = 0x2A)
    CHECK(buf[101] == 0x2A && buf[102] == 0x00 && buf[108] == 0x00,
          "WORK_ASN work_id LE at offset 101");
    // Offset 109: kangaroo_type
    CHECK(buf[109] == 0x01, "WORK_ASN kangaroo_type at offset 109");
    // Offset 110..117: start_offset_a LE (0xDEADBEEF)
    CHECK(buf[110] == 0xEF && buf[111] == 0xBE && buf[112] == 0xAD && buf[113] == 0xDE &&
          buf[114] == 0x00 && buf[115] == 0x00 && buf[116] == 0x00 && buf[117] == 0x00,
          "WORK_ASN start_offset_a LE at offset 110");
    // Offset 118..125: start_offset_b LE (0xDEADBEEF + 1024 = 0xDEADC2EF)
    CHECK(buf[118] == 0xEF && buf[119] == 0xC2 && buf[120] == 0xAD && buf[121] == 0xDE &&
          buf[122] == 0x00 && buf[123] == 0x00 && buf[124] == 0x00 && buf[125] == 0x00,
          "WORK_ASN start_offset_b LE at offset 118");

    // Round-trip via memcpy.
    jlp_wire::WorkAssignment w2{};
    std::memcpy(&w2, buf, sizeof(w2));
    CHECK(w2.dp_bits == 28, "round-trip dp_bits");
    CHECK(w2.work_id == 42ULL, "round-trip work_id");
    CHECK(w2.kangaroo_type == 1, "round-trip kangaroo_type");
    CHECK(w2.start_offset_a == 0x00000000DEADBEEFULL, "round-trip start_offset_a");
    CHECK(w2.start_offset_b == 0x00000000DEADBEEFULL + 1024ULL, "round-trip start_offset_b");
    CHECK(std::memcmp(w2.public_key, w.public_key, 33) == 0, "round-trip public_key");
    CHECK(std::memcmp(w2.range_start, w.range_start, 32) == 0, "round-trip range_start");
    CHECK(std::memcmp(w2.range_end, w.range_end, 32) == 0, "round-trip range_end");
}

void test_legacy_to_generated_memcpy() {
    // A legacy JLPDistinguishedPointV2 must be byte-equivalent to the
    // codegen one. v1.4.1 B.1 added a sequence field on both sides
    // (struct.pack '<QI32s32sBB'), so memcpy preserves all six fields.
    JLPDistinguishedPointV2 legacy{};
    legacy.work_id = 0x1122334455667788ULL;
    legacy.sequence = 0xCAFEBABEu;
    for (int i = 0; i < 32; ++i) legacy.x[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 32; ++i) legacy.d[i] = static_cast<uint8_t>(0x80 | i);
    legacy.type = 1;
    legacy.dp_bits = 24;

    jlp_wire::DistinguishedPointV2 gen{};
    std::memcpy(&gen, &legacy, sizeof(gen));

    CHECK(gen.work_id == legacy.work_id, "memcpy preserves work_id");
    CHECK(gen.sequence == legacy.sequence, "memcpy preserves sequence (v1.4.1 B.1)");
    CHECK(gen.type == legacy.type, "memcpy preserves type");
    CHECK(gen.dp_bits == legacy.dp_bits, "memcpy preserves dp_bits");
    CHECK(std::memcmp(gen.x, legacy.x, 32) == 0, "memcpy preserves x[]");
    CHECK(std::memcmp(gen.d, legacy.d, 32) == 0, "memcpy preserves d[]");
}

}  // namespace

int main() {
    test_header_layout();
    test_dp_v2_layout();
    test_auth_payload_v2_layout();
    test_work_assignment_v3_layout();
    test_legacy_to_generated_memcpy();

    if (failures != 0) {
        std::fprintf(stderr, "test_jlp_wire_generated: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_jlp_wire_generated: PASS\n");
    return 0;
}
