/**
 * test_stats_sanitize.cpp -- v1.4.2 B.6 regression test.
 *
 * Pre-fix the STATS_RSP handler did `static_cast<uint64_t>(float)` on
 * floats read directly from the wire. For NaN, +/-Inf, or out-of-range
 * floats this is undefined behaviour - in practice on x86-64 it returns
 * INT64_MIN, on ARM64 it returns 0; either way the downstream display
 * lies and downstream arithmetic may corrupt.
 */

#include "../src/pool/stats_sanitize.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>

using collider::pool::sanitize_wire_float;
using collider::pool::sanitize_stats_rsp_floats;

namespace {

int g_pass = 0;
int g_fail = 0;

#define EXPECT_EQ(a, b, label)                                              \
    do {                                                                    \
        if ((a) == (b)) {                                                   \
            ++g_pass;                                                       \
        } else {                                                            \
            ++g_fail;                                                       \
            std::cerr << "[FAIL] " << (label) << ": expected " << (b)       \
                      << ", got " << (a) << "\n";                           \
        }                                                                   \
    } while (0)

#define EXPECT_FINITE(v, label)                                             \
    do {                                                                    \
        if (std::isfinite(v)) {                                             \
            ++g_pass;                                                       \
        } else {                                                            \
            ++g_fail;                                                       \
            std::cerr << "[FAIL] " << (label) << ": not finite\n";          \
        }                                                                   \
    } while (0)

}  // namespace

int main() {
    std::cout << "test_stats_sanitize (v1.4.2 B.6 regression suite)\n";

    // ---- sanitize_wire_float identity cases ----
    EXPECT_EQ(sanitize_wire_float(0.5f, 0.0f, 1.0f), 0.5f, "passthrough in range");
    EXPECT_EQ(sanitize_wire_float(0.0f, 0.0f, 1.0f), 0.0f, "low bound");
    EXPECT_EQ(sanitize_wire_float(1.0f, 0.0f, 1.0f), 1.0f, "high bound");

    // ---- clamping ----
    EXPECT_EQ(sanitize_wire_float(-1.0f, 0.0f, 1.0f), 0.0f, "clamp negative to low");
    EXPECT_EQ(sanitize_wire_float(2.0f,  0.0f, 1.0f), 1.0f, "clamp >hi to hi");
    EXPECT_EQ(sanitize_wire_float(1e30f, 0.0f, 1.0e18f), 1.0e18f, "clamp 1e30 to 1e18");

    // ---- NaN / Inf normalization to zero ----
    EXPECT_EQ(sanitize_wire_float(std::numeric_limits<float>::quiet_NaN(),
                                  0.0f, 1.0f), 0.0f, "NaN -> 0");
    EXPECT_EQ(sanitize_wire_float(std::numeric_limits<float>::infinity(),
                                  0.0f, 1.0f), 0.0f, "+Inf -> 0");
    EXPECT_EQ(sanitize_wire_float(-std::numeric_limits<float>::infinity(),
                                  0.0f, 1.0f), 0.0f, "-Inf -> 0");

    // ---- sanitize_stats_rsp_floats integration ----
    {
        float dps = std::numeric_limits<float>::quiet_NaN();
        float share = -0.5f;
        sanitize_stats_rsp_floats(dps, share);
        EXPECT_FINITE(dps, "stats: NaN dps becomes finite");
        EXPECT_EQ(dps,   0.0f, "stats: NaN dps clamped to 0");
        EXPECT_EQ(share, 0.0f, "stats: negative share clamped to 0");
    }
    {
        float dps = 1e20f;     // a buggy server reports impossible rate
        float share = 5.0f;    // > 100% credit (cannot happen)
        sanitize_stats_rsp_floats(dps, share);
        EXPECT_EQ(dps,   1.0e18f, "stats: 1e20 dps clamped to 1e18");
        EXPECT_EQ(share, 1.0f,    "stats: 5.0 share clamped to 1.0");
    }
    {
        // Downstream UB check: static_cast<uint64_t>(NaN) is UB pre-fix.
        // After sanitize, the cast is guaranteed defined.
        float dps = std::numeric_limits<float>::quiet_NaN();
        float share = 0.5f;
        sanitize_stats_rsp_floats(dps, share);
        volatile uint64_t cast_target = static_cast<uint64_t>(dps);
        EXPECT_EQ(cast_target, 0ULL, "downstream uint64 cast is defined");
    }

    std::cout << "Summary: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
