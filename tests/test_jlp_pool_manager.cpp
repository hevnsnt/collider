// test_jlp_pool_manager.cpp
//
// Pure-logic tests for D:\theCollider\src\pool\pool_manager.cpp and the
// DistinguishedPoint serializer in src/pool/jlp_pool_client.cpp. No sockets,
// no threads -- everything in this file runs in main().
//
// Coverage target (from the test plan, Priority 2):
//   - parse_pool_url:     valid scheme variants (jlp://, jlps://, bare host)
//   - parse_pool_url:     scheme rejection (http:// must hard-fail with a
//                         migration hint); we capture stderr to confirm)
//   - parse_pool_url:     port validation (0, 65536, non-numeric, valid max)
//   - DistinguishedPoint: serialize() / deserialize() round-trip + size invariant
//   - PoolStats:          default value-init zeroes numeric fields
//
// No CTest framework: each check returns early with a printf + non-zero exit
// code on failure. Same style as tests/test_jlp_pool_handshake.cpp.

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
#endif

#include "pool/pool_manager.hpp"
#include "pool/pool_client.hpp"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

using collider::pool::PoolConfig;
using collider::pool::parse_pool_url;
using collider::pool::DistinguishedPoint;
using collider::pool::PoolStats;
using collider::pool::POOL_TYPE_JLP;

namespace {

// Tiny test-failure helper. Returns false to let callers short-circuit.
bool fail(const char* test_name, const char* msg) {
    std::fprintf(stderr, "[FAIL] %s: %s\n", test_name, msg);
    return false;
}

// ---------------------------------------------------------------------------
// parse_pool_url -- valid URLs
// ---------------------------------------------------------------------------

bool test_parse_jlp_url_with_port() {
    PoolConfig cfg{};
    if (!parse_pool_url("jlp://myhost:1234", cfg)) {
        return fail("parse_jlp_url_with_port", "parse returned false");
    }
    if (cfg.type != POOL_TYPE_JLP) return fail("parse_jlp_url_with_port", "type != jlp");
    if (cfg.host != "myhost")      return fail("parse_jlp_url_with_port", "host != myhost");
    if (cfg.port != 1234)          return fail("parse_jlp_url_with_port", "port != 1234");
    if (cfg.use_tls)               return fail("parse_jlp_url_with_port", "use_tls true for jlp://");
    return true;
}

bool test_parse_jlps_url_with_port() {
    PoolConfig cfg{};
    if (!parse_pool_url("jlps://myhost:1234", cfg)) {
        return fail("parse_jlps_url_with_port", "parse returned false");
    }
    if (cfg.type != POOL_TYPE_JLP) return fail("parse_jlps_url_with_port", "type != jlp");
    if (cfg.host != "myhost")      return fail("parse_jlps_url_with_port", "host != myhost");
    if (cfg.port != 1234)          return fail("parse_jlps_url_with_port", "port != 1234");
    if (!cfg.use_tls)              return fail("parse_jlps_url_with_port", "use_tls false for jlps://");
    return true;
}

bool test_parse_bare_host_with_port() {
    // Shorthand: no scheme should default to jlp:// (no TLS).
    PoolConfig cfg{};
    if (!parse_pool_url("myhost:1234", cfg)) {
        return fail("parse_bare_host_with_port", "parse returned false");
    }
    if (cfg.type != POOL_TYPE_JLP) return fail("parse_bare_host_with_port", "type != jlp");
    if (cfg.host != "myhost")      return fail("parse_bare_host_with_port", "host != myhost");
    if (cfg.port != 1234)          return fail("parse_bare_host_with_port", "port != 1234");
    if (cfg.use_tls)               return fail("parse_bare_host_with_port", "use_tls should be false (no scheme)");
    return true;
}

bool test_parse_bare_host_no_port_uses_default() {
    // No scheme + no port should give the JLP default (17403).
    PoolConfig cfg{};
    if (!parse_pool_url("myhost", cfg)) {
        return fail("parse_bare_host_no_port", "parse returned false");
    }
    if (cfg.host != "myhost") return fail("parse_bare_host_no_port", "host != myhost");
    if (cfg.port != 17403)    return fail("parse_bare_host_no_port", "default port != 17403");
    return true;
}

// ---------------------------------------------------------------------------
// parse_pool_url -- rejection of http:// (Wave 4 D-C1 migration)
// ---------------------------------------------------------------------------

bool test_parse_http_url_rejected() {
    // http:// must hard-fail with a migration message. We capture stderr to
    // confirm the operator gets the migration hint, not a silent rejection.
    std::stringstream captured;
    auto* old = std::cerr.rdbuf(captured.rdbuf());

    PoolConfig cfg{};
    bool ok = parse_pool_url("http://host:8080", cfg);

    std::cerr.rdbuf(old);

    if (ok) return fail("parse_http_url_rejected",
                        "parse_pool_url(http://...) returned true (must reject)");

    // Look for either "deprecated", "migration", or "removed" in the captured
    // message so an operator can tell why parsing failed.
    const std::string text = captured.str();
    if (text.find("deprecated") == std::string::npos &&
        text.find("Migration")  == std::string::npos &&
        text.find("removed")    == std::string::npos) {
        std::fprintf(stderr, "[FAIL] parse_http_url_rejected: "
                             "stderr does not mention deprecation/migration. "
                             "Got: %s\n", text.c_str());
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// parse_pool_url -- port validation
// ---------------------------------------------------------------------------

bool test_parse_port_zero_rejected() {
    // Port 0 is technically reserved + meaningless for a server.
    std::stringstream sink;
    auto* old = std::cerr.rdbuf(sink.rdbuf());

    PoolConfig cfg{};
    bool ok = parse_pool_url("host:0", cfg);

    std::cerr.rdbuf(old);

    if (ok) return fail("parse_port_zero_rejected", "port=0 should be rejected");
    return true;
}

bool test_parse_port_overflow_rejected() {
    // 65536 is one past the uint16_t range.
    std::stringstream sink;
    auto* old = std::cerr.rdbuf(sink.rdbuf());

    PoolConfig cfg{};
    bool ok = parse_pool_url("host:65536", cfg);

    std::cerr.rdbuf(old);

    if (ok) return fail("parse_port_overflow_rejected", "port=65536 should be rejected");
    return true;
}

bool test_parse_port_nonnumeric_rejected() {
    // The regex itself enforces \d+ so a non-numeric port should fail to match.
    // Either way the function must return false.
    std::stringstream sink;
    auto* old = std::cerr.rdbuf(sink.rdbuf());

    PoolConfig cfg{};
    bool ok = parse_pool_url("host:notanumber", cfg);

    std::cerr.rdbuf(old);

    if (ok) return fail("parse_port_nonnumeric_rejected",
                        "non-numeric port should be rejected");
    return true;
}

bool test_parse_port_default_jlp_accepted() {
    // 17403 is the documented default JLP port; it must round-trip cleanly.
    PoolConfig cfg{};
    if (!parse_pool_url("host:17403", cfg)) {
        return fail("parse_port_default_jlp_accepted",
                    "parse returned false for valid port 17403");
    }
    if (cfg.port != 17403) return fail("parse_port_default_jlp_accepted",
                                       "port not preserved");
    return true;
}

// ---------------------------------------------------------------------------
// DistinguishedPoint serialization round-trip
// ---------------------------------------------------------------------------

bool test_dp_serialize_roundtrip() {
    DistinguishedPoint dp;
    // Fill with a known pattern so a memcpy bug (offset error, length mismatch)
    // shows up as a byte-level diff.
    for (int i = 0; i < 32; ++i) {
        dp.x[i] = static_cast<uint8_t>(0xA0 + i);
        dp.d[i] = static_cast<uint8_t>(0x40 + i);
    }
    dp.type    = 1;            // wild
    dp.dp_bits = 0x0123456789ABCDEFULL;

    std::vector<uint8_t> bytes = dp.serialize();
    if (bytes.size() != 73) {
        std::fprintf(stderr,
            "[FAIL] dp_serialize_roundtrip: serialized size %zu (want 73)\n",
            bytes.size());
        return false;
    }

    DistinguishedPoint round = DistinguishedPoint::deserialize(bytes.data(), bytes.size());
    if (std::memcmp(round.x, dp.x, 32) != 0)
        return fail("dp_serialize_roundtrip", "x[] mismatch");
    if (std::memcmp(round.d, dp.d, 32) != 0)
        return fail("dp_serialize_roundtrip", "d[] mismatch");
    if (round.type != dp.type)
        return fail("dp_serialize_roundtrip", "type mismatch");
    if (round.dp_bits != dp.dp_bits)
        return fail("dp_serialize_roundtrip", "dp_bits mismatch");
    return true;
}

// ---------------------------------------------------------------------------
// PoolStats default values
// ---------------------------------------------------------------------------

bool test_poolstats_default_values() {
    // Value-initialize -- this is what the rest of the codebase uses, e.g.
    // PoolManager::get_stats() returns `PoolStats{}` when no client is set.
    PoolStats s{};
    if (s.total_dps        != 0) return fail("poolstats_default", "total_dps != 0");
    if (s.your_dps         != 0) return fail("poolstats_default", "your_dps != 0");
    if (s.your_share       != 0.0) return fail("poolstats_default", "your_share != 0");
    if (s.connected_workers != 0) return fail("poolstats_default", "connected_workers != 0");
    if (s.pool_speed       != 0) return fail("poolstats_default", "pool_speed != 0");
    if (!s.status.empty()) return fail("poolstats_default", "status not empty");
    return true;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

struct TestCase {
    const char* name;
    bool (*fn)();
};

const TestCase TESTS[] = {
    {"parse_jlp_url_with_port",          test_parse_jlp_url_with_port},
    {"parse_jlps_url_with_port",         test_parse_jlps_url_with_port},
    {"parse_bare_host_with_port",        test_parse_bare_host_with_port},
    {"parse_bare_host_no_port_default",  test_parse_bare_host_no_port_uses_default},
    {"parse_http_url_rejected",          test_parse_http_url_rejected},
    {"parse_port_zero_rejected",         test_parse_port_zero_rejected},
    {"parse_port_overflow_rejected",     test_parse_port_overflow_rejected},
    {"parse_port_nonnumeric_rejected",   test_parse_port_nonnumeric_rejected},
    {"parse_port_default_jlp_accepted",  test_parse_port_default_jlp_accepted},
    {"dp_serialize_roundtrip",           test_dp_serialize_roundtrip},
    {"poolstats_default_values",         test_poolstats_default_values},
};

}  // namespace

int main() {
    std::printf("=== JLP pool manager unit tests ===\n");
    int failures = 0;
    for (const auto& t : TESTS) {
        std::printf("[ run ] %s\n", t.name);
        if (t.fn()) {
            std::printf("[ ok  ] %s\n", t.name);
        } else {
            std::printf("[FAIL ] %s\n", t.name);
            ++failures;
        }
    }
    std::printf("\n%zu tests, %d failures\n",
                sizeof(TESTS) / sizeof(TESTS[0]), failures);
    return failures == 0 ? 0 : 1;
}
