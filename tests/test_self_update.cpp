// test_self_update.cpp
//
// Unit tests for collider::update::semver_less (the version-comparison
// primitive that decides whether a pool-advertised client version is newer
// than the running one). Pure string parsing; no network, no filesystem.

#include "runtime/self_update.hpp"

#include <cstdio>
#include <string>

using collider::update::semver_less;

namespace {

int g_failures = 0;

void check(bool cond, const char* expr) {
    if (!cond) {
        std::fprintf(stderr, "[FAIL] %s\n", expr);
        ++g_failures;
    }
}

#define CHECK(expr) check((expr), #expr)

}  // namespace

int main() {
    std::printf("=== self_update semver_less tests ===\n");

    // Basic ordering: patch, minor, major bumps.
    CHECK(semver_less("1.5.4", "1.5.5"));   // patch
    CHECK(semver_less("1.5.4", "1.6.0"));   // minor
    CHECK(semver_less("1.5.4", "2.0.0"));   // major

    // Equal is NOT less (strict).
    CHECK(!semver_less("1.5.4", "1.5.4"));

    // Reverse direction is not less.
    CHECK(!semver_less("1.5.5", "1.5.4"));
    CHECK(!semver_less("1.6.0", "1.5.4"));
    CHECK(!semver_less("2.0.0", "1.5.4"));

    // Numeric (not lexical) comparison: 1.5.10 > 1.5.9.
    CHECK(!semver_less("1.5.10", "1.5.9"));
    CHECK(semver_less("1.5.9", "1.5.10"));
    // And 1.10.0 > 1.9.0 (would fail under lexical compare).
    CHECK(semver_less("1.9.0", "1.10.0"));
    CHECK(!semver_less("1.10.0", "1.9.0"));

    // Malformed-suffix tolerance: a non-numeric suffix on a component is
    // ignored, so "1.5.4-rc1" parses as {1,5,4}.
    CHECK(!semver_less("1.5.4-rc1", "1.5.4"));   // equal core -> not less
    CHECK(semver_less("1.5.4-rc1", "1.5.5"));    // 1.5.4 < 1.5.5
    CHECK(semver_less("1.5.4", "1.5.5+build7")); // suffix on b ignored
    CHECK(!semver_less("1.5.5+build7", "1.5.4"));

    // Missing components default to 0: "1.5" == "1.5.0".
    CHECK(!semver_less("1.5", "1.5.0"));
    CHECK(!semver_less("1.5.0", "1.5"));
    CHECK(semver_less("1.5", "1.5.1"));
    CHECK(semver_less("1", "1.0.1"));

    // Empty / garbage strings parse as all-zero; should not crash.
    CHECK(!semver_less("", ""));
    CHECK(semver_less("", "0.0.1"));
    CHECK(!semver_less("0.0.1", ""));

    if (g_failures == 0) {
        std::printf("all semver_less tests passed\n");
    } else {
        std::printf("%d semver_less test(s) FAILED\n", g_failures);
    }
    return g_failures == 0 ? 0 : 1;
}
