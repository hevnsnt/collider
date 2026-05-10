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
static_assert(sizeof(jlp_wire::DistinguishedPointV2) == 74,
              "DP v2 wire size is 74 by spec");

static_assert(sizeof(JLPServerConfig) == sizeof(jlp_wire::WorkAssignment),
              "WorkAssignment layout has diverged");
static_assert(sizeof(jlp_wire::WorkAssignment) == 109,
              "WorkAssignment wire size is 109 by spec");

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

static_assert(jlp_wire::PROTOCOL_VERSION == 2,
              "Phase 0 IDL fixes the version at 2");
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
    for (int i = 0; i < 32; ++i) dp.x[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 32; ++i) dp.d[i] = static_cast<uint8_t>(0x80 | i);
    dp.type = 1;
    dp.dp_bits = 24;

    uint8_t buf[sizeof(jlp_wire::DistinguishedPointV2)];
    std::memcpy(buf, &dp, sizeof(buf));

    // work_id must be the first 8 bytes, little-endian
    CHECK(buf[0] == 0x88 && buf[1] == 0x77 && buf[2] == 0x66 && buf[3] == 0x55 &&
          buf[4] == 0x44 && buf[5] == 0x33 && buf[6] == 0x22 && buf[7] == 0x11,
          "dp_v2 work_id position + endianness");
    CHECK(buf[8] == 0x00 && buf[9] == 0x01, "dp_v2 x[] follows work_id");
    CHECK(buf[40] == 0x80, "dp_v2 d[] follows x[]");
    CHECK(buf[72] == 1, "dp_v2 type at offset 72");
    CHECK(buf[73] == 24, "dp_v2 dp_bits at offset 73");
}

void test_legacy_to_generated_memcpy() {
    // A legacy JLPDistinguishedPointV2 must be byte-equivalent.
    JLPDistinguishedPointV2 legacy{};
    legacy.work_id = 0x1122334455667788ULL;
    for (int i = 0; i < 32; ++i) legacy.x[i] = static_cast<uint8_t>(i);
    for (int i = 0; i < 32; ++i) legacy.d[i] = static_cast<uint8_t>(0x80 | i);
    legacy.type = 1;
    legacy.dp_bits = 24;

    jlp_wire::DistinguishedPointV2 gen{};
    std::memcpy(&gen, &legacy, sizeof(gen));

    CHECK(gen.work_id == legacy.work_id, "memcpy preserves work_id");
    CHECK(gen.type == legacy.type, "memcpy preserves type");
    CHECK(gen.dp_bits == legacy.dp_bits, "memcpy preserves dp_bits");
    CHECK(std::memcmp(gen.x, legacy.x, 32) == 0, "memcpy preserves x[]");
    CHECK(std::memcmp(gen.d, legacy.d, 32) == 0, "memcpy preserves d[]");
}

}  // namespace

int main() {
    test_header_layout();
    test_dp_v2_layout();
    test_legacy_to_generated_memcpy();

    if (failures != 0) {
        std::fprintf(stderr, "test_jlp_wire_generated: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_jlp_wire_generated: PASS\n");
    return 0;
}
