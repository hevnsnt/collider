// Phase 2 (builder-gpu): NVML wrapper unit test.
//
// Coverage:
//   1. Happy path: init() returns true on a CI machine with an NVIDIA
//      driver, false otherwise. Sanity-check a handful of accessors.
//   2. Forced-missing path: COLLIDER_FORCE_NVML_MISSING=1 makes init()
//      return false even when nvml.dll loads cleanly; every accessor
//      returns nullopt.
//   3. Out-of-range index: device_name(99999) is nullopt without crash.
//   4. Shutdown idempotence: calling shutdown() twice does not crash;
//      is_available() returns false afterwards.
//
// Handcrafted PASS/FAIL output. Exit 0 = pass, non-zero = fail. Designed
// to match the in-tree style (no gtest, no Catch2).

#include "platform/nvml_query.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
    #include <stdlib.h>  // _putenv_s
#else
    #include <stdlib.h>
#endif

namespace {

int g_fail_count = 0;

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

void unset_env(const char* name) {
#if defined(_WIN32)
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);\
        ++g_fail_count;                                             \
    }                                                               \
} while (0)

// ---------------------------------------------------------------------
// Test 1: happy path. If init() succeeds, exercise the major
// accessors against device 0 and confirm the values are in plausible
// ranges. If init() fails, confirm every accessor still returns
// nullopt cleanly (the no-NVIDIA-driver code path).
// ---------------------------------------------------------------------
void test_happy_path() {
    std::printf("test_happy_path: ");

    unset_env("COLLIDER_FORCE_NVML_MISSING");
    namespace nv = collider::platform::nvml;

    const bool ok = nv::init();
    if (!ok) {
        // No driver / no NVML on this machine. Confirm degraded contract.
        CHECK(!nv::is_available(), "is_available() must be false when init() fails");
        CHECK(!nv::device_count().has_value(), "device_count() must be nullopt without NVML");
        CHECK(!nv::device_name(0).has_value(), "device_name(0) must be nullopt without NVML");
        CHECK(!nv::temperature_c(0).has_value(), "temperature_c(0) must be nullopt without NVML");
        CHECK(!nv::util_gpu_pct(0).has_value(), "util_gpu_pct(0) must be nullopt without NVML");
        CHECK(!nv::pci_info(0).has_value(), "pci_info(0) must be nullopt without NVML");
        std::printf("PASS (no NVML on host; degraded path verified)\n");
        nv::shutdown();
        return;
    }

    CHECK(nv::is_available(), "is_available() must be true after successful init()");

    auto count = nv::device_count();
    CHECK(count.has_value(), "device_count() must return a value when init() succeeded");
    if (count) {
        CHECK(*count >= 1, "device_count() must be >= 1 on a system with init() success");
    }

    if (count && *count >= 1) {
        // Device 0 should expose a name and either a real temperature
        // or, at minimum, never crash. We allow nullopt on individual
        // queries (some virtualized GPUs hide telemetry) but the name
        // and PCI info should always work.
        auto name = nv::device_name(0);
        CHECK(name.has_value() && !name->empty(),
              "device_name(0) must return a non-empty string");
        if (name) {
            std::printf("[gpu0=%s] ", name->c_str());
        }

        auto pci = nv::pci_info(0);
        CHECK(pci.has_value(), "pci_info(0) must return a value");
        if (pci) {
            CHECK(!pci->bus_id.empty(), "pci_info(0).bus_id must be non-empty");
        }

        auto temp = nv::temperature_c(0);
        if (temp) {
            // Real GPUs sit between ambient (~15 C) and thermal cutoff
            // (~110 C). Anything outside this is a sensor glitch.
            CHECK(*temp >= 10.0f && *temp <= 120.0f,
                  "temperature_c(0) must be in plausible range");
            std::printf("[temp=%.0fC] ", *temp);
        }

        auto util = nv::util_gpu_pct(0);
        if (util) {
            CHECK(*util >= 0 && *util <= 100, "util_gpu_pct(0) must be 0..100");
            std::printf("[util=%d%%] ", *util);
        }

        auto mem_total = nv::memory_total_bytes(0);
        if (mem_total) {
            CHECK(*mem_total > 0, "memory_total_bytes(0) must be positive");
        }

        auto cuda_idx = nv::cuda_to_nvml_index(0);
        // cuda_to_nvml_index may legitimately be nullopt if the CUDA
        // runtime isn't loaded or the mapping fails on this host;
        // either way it must not crash.
        if (cuda_idx) {
            CHECK(*cuda_idx < *count, "cuda_to_nvml_index(0) must be in range");
        }
    }

    nv::shutdown();
    CHECK(!nv::is_available(), "is_available() must be false after shutdown()");
    std::printf("PASS\n");
}

// ---------------------------------------------------------------------
// Test 2: forced-missing. Set the env var, init() must return false,
// every accessor nullopt.
// ---------------------------------------------------------------------
void test_forced_missing() {
    std::printf("test_forced_missing: ");

    namespace nv = collider::platform::nvml;
    // Ensure we are starting from a clean slate after test_happy_path's
    // shutdown.
    set_env("COLLIDER_FORCE_NVML_MISSING", "1");

    const bool ok = nv::init();
    CHECK(!ok, "init() must return false when COLLIDER_FORCE_NVML_MISSING=1");
    CHECK(!nv::is_available(), "is_available() must be false under forced-missing");

    CHECK(!nv::device_count().has_value(), "device_count() must be nullopt under forced-missing");
    CHECK(!nv::cuda_to_nvml_index(0).has_value(),
          "cuda_to_nvml_index(0) must be nullopt under forced-missing");
    CHECK(!nv::device_name(0).has_value(),
          "device_name(0) must be nullopt under forced-missing");
    CHECK(!nv::pci_info(0).has_value(),
          "pci_info(0) must be nullopt under forced-missing");
    CHECK(!nv::temperature_c(0).has_value(),
          "temperature_c(0) must be nullopt under forced-missing");
    CHECK(!nv::power_milliwatts(0).has_value(),
          "power_milliwatts(0) must be nullopt under forced-missing");
    CHECK(!nv::power_limit_milliwatts(0).has_value(),
          "power_limit_milliwatts(0) must be nullopt under forced-missing");
    CHECK(!nv::util_gpu_pct(0).has_value(),
          "util_gpu_pct(0) must be nullopt under forced-missing");
    CHECK(!nv::util_mem_pct(0).has_value(),
          "util_mem_pct(0) must be nullopt under forced-missing");
    CHECK(!nv::sm_clock_mhz(0).has_value(),
          "sm_clock_mhz(0) must be nullopt under forced-missing");
    CHECK(!nv::mem_clock_mhz(0).has_value(),
          "mem_clock_mhz(0) must be nullopt under forced-missing");
    CHECK(!nv::fan_pct(0).has_value(),
          "fan_pct(0) must be nullopt under forced-missing");
    CHECK(!nv::memory_used_bytes(0).has_value(),
          "memory_used_bytes(0) must be nullopt under forced-missing");
    CHECK(!nv::memory_total_bytes(0).has_value(),
          "memory_total_bytes(0) must be nullopt under forced-missing");

    nv::shutdown();
    unset_env("COLLIDER_FORCE_NVML_MISSING");
    std::printf("PASS\n");
}

// ---------------------------------------------------------------------
// Test 3: out-of-range index. NVML returns an error for index 99999;
// the wrapper must convert it to nullopt without crashing. Run both
// with and without NVML available (i.e. independent of test 1's
// outcome).
// ---------------------------------------------------------------------
void test_out_of_range_index() {
    std::printf("test_out_of_range_index: ");

    namespace nv = collider::platform::nvml;
    unset_env("COLLIDER_FORCE_NVML_MISSING");
    nv::init();  // result doesn't matter; we only care that the call below is safe

    constexpr unsigned int bad = 99999u;
    CHECK(!nv::device_name(bad).has_value(), "device_name(99999) must be nullopt");
    CHECK(!nv::pci_info(bad).has_value(), "pci_info(99999) must be nullopt");
    CHECK(!nv::temperature_c(bad).has_value(), "temperature_c(99999) must be nullopt");
    CHECK(!nv::power_milliwatts(bad).has_value(),
          "power_milliwatts(99999) must be nullopt");
    CHECK(!nv::util_gpu_pct(bad).has_value(), "util_gpu_pct(99999) must be nullopt");
    CHECK(!nv::sm_clock_mhz(bad).has_value(), "sm_clock_mhz(99999) must be nullopt");
    CHECK(!nv::fan_pct(bad).has_value(), "fan_pct(99999) must be nullopt");
    CHECK(!nv::memory_total_bytes(bad).has_value(),
          "memory_total_bytes(99999) must be nullopt");

    nv::shutdown();
    std::printf("PASS\n");
}

// =====================================================================
// Test 5: per-symbol force-missing (FAN). With
// COLLIDER_FORCE_NVML_MISSING_FAN=1 set, init() still succeeds when
// NVML is available; the fan symbol is short-circuited to nullptr so
// fan_pct() returns nullopt while sibling accessors (temperature, util,
// memory) keep working. Models the realistic older-driver case where
// nvml.dll loads but a single function pointer is unresolvable.
//
// On hosts where NVML is unavailable the whole library fails to load
// regardless of the per-symbol hook, so the test gracefully degrades
// to a no-op (init() returns false, fan_pct nullopt is the standard
// no-NVML behavior).
// =====================================================================
void test_force_missing_fan() {
    std::printf("test_force_missing_fan: ");

    namespace nv = collider::platform::nvml;
    // Reset wrapper state so init() will re-resolve symbols under the
    // new env-var setup.
    nv::shutdown();
    unset_env("COLLIDER_FORCE_NVML_MISSING");
    set_env("COLLIDER_FORCE_NVML_MISSING_FAN", "1");

    const bool ok = nv::init();
    if (!ok) {
        // No NVML on this host; the per-symbol hook is moot but the
        // contract still holds (every accessor returns nullopt). Confirm
        // and report.
        CHECK(!nv::fan_pct(0).has_value(),
              "fan_pct(0) must be nullopt when NVML is unavailable");
        unset_env("COLLIDER_FORCE_NVML_MISSING_FAN");
        std::printf("PASS (no NVML on host; hook path not exercised)\n");
        return;
    }

    // NVML loaded; the FAN symbol pointer must be null after init() so
    // fan_pct() returns nullopt regardless of the underlying card's
    // fan presence.
    CHECK(!nv::fan_pct(0).has_value(),
          "fan_pct(0) must be nullopt under FORCE_NVML_MISSING_FAN");

    // Sibling accessors MUST keep working. We accept nullopt only when
    // the underlying card legitimately does not expose the sensor (some
    // virtualized GPUs hide temp + util); the assertion is that the
    // accessor does not crash and the wrapper is still marked
    // available.
    CHECK(nv::is_available(),
          "is_available() must remain true with only FAN forced missing");
    auto temp = nv::temperature_c(0);
    auto util = nv::util_gpu_pct(0);
    // If both temp and util are nullopt the card is too restrictive
    // for the assertion to mean anything. On the developer dev box and
    // CI the host has at least one of them.
    const bool sibling_alive = temp.has_value() || util.has_value();
    CHECK(sibling_alive,
          "temperature_c or util_gpu_pct must work with only FAN forced missing");

    nv::shutdown();
    unset_env("COLLIDER_FORCE_NVML_MISSING_FAN");
    std::printf("PASS\n");
}

// =====================================================================
// Test 6: per-symbol force-missing (POWER). Same shape as test 5 but
// with the power-usage symbol nulled out. Confirms the env-var hook
// independently disables individual symbols rather than acting as an
// all-or-nothing switch.
// =====================================================================
void test_force_missing_power() {
    std::printf("test_force_missing_power: ");

    namespace nv = collider::platform::nvml;
    nv::shutdown();
    unset_env("COLLIDER_FORCE_NVML_MISSING");
    unset_env("COLLIDER_FORCE_NVML_MISSING_FAN");
    set_env("COLLIDER_FORCE_NVML_MISSING_POWER", "1");

    const bool ok = nv::init();
    if (!ok) {
        CHECK(!nv::power_milliwatts(0).has_value(),
              "power_milliwatts(0) must be nullopt when NVML is unavailable");
        unset_env("COLLIDER_FORCE_NVML_MISSING_POWER");
        std::printf("PASS (no NVML on host; hook path not exercised)\n");
        return;
    }

    CHECK(!nv::power_milliwatts(0).has_value(),
          "power_milliwatts(0) must be nullopt under FORCE_NVML_MISSING_POWER");
    CHECK(nv::is_available(),
          "is_available() must remain true with only POWER forced missing");

    auto temp = nv::temperature_c(0);
    auto util = nv::util_gpu_pct(0);
    const bool sibling_alive = temp.has_value() || util.has_value();
    CHECK(sibling_alive,
          "temperature_c or util_gpu_pct must work with only POWER forced missing");

    nv::shutdown();
    unset_env("COLLIDER_FORCE_NVML_MISSING_POWER");
    std::printf("PASS\n");
}

// ---------------------------------------------------------------------
// Test 4: shutdown idempotence. shutdown() must be safe to call any
// number of times; is_available() returns false after.
// ---------------------------------------------------------------------
void test_shutdown_idempotence() {
    std::printf("test_shutdown_idempotence: ");

    namespace nv = collider::platform::nvml;
    unset_env("COLLIDER_FORCE_NVML_MISSING");
    nv::init();
    nv::shutdown();
    CHECK(!nv::is_available(), "is_available() must be false after first shutdown()");
    nv::shutdown();  // must not crash
    CHECK(!nv::is_available(), "is_available() must remain false after second shutdown()");
    nv::shutdown();  // and a third time, for good measure
    CHECK(!nv::is_available(), "is_available() must remain false after third shutdown()");
    std::printf("PASS\n");
}

}  // namespace

int main() {
    std::printf("=== test_nvml_wrapper (Phase 2 NVML wrapper) ===\n");

    test_happy_path();
    test_forced_missing();
    test_out_of_range_index();
    test_force_missing_fan();
    test_force_missing_power();
    test_shutdown_idempotence();

    if (g_fail_count == 0) {
        std::printf("=== All NVML wrapper tests passed ===\n");
        return 0;
    }
    std::fprintf(stderr, "=== %d NVML wrapper assertion(s) FAILED ===\n", g_fail_count);
    return 1;
}
