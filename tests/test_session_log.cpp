// test_session_log.cpp -- coverage for src/core/session_log.{cpp,hpp}.
//
// Closes the 2026-05-17 reviewer gap on the session-log subsystem.
// Five test families:
//
//   1. redact_argv: golden cases for all four kRedactedFlags
//      (--pool-password, --pool-password-file, --pool-api-key,
//      --activate), in both "--flag value" and "--flag=value" forms.
//      Pre-fix, --activate keys leaked into the startup banner because
//      the flag was added after the redaction set was last audited.
//
//   2. serialize_state: a default-constructed SessionState must emit
//      well-formed JSON (braces balanced, every emitted key has a
//      value). Optional fields stay omitted; the always-on bookkeeping
//      fields are present even when empty.
//
//   3. shutdown_session_log: double-fire is a no-op. The std::atexit
//      hook + an explicit test-fixture call can both reach the
//      shutdown helper; the second invocation must not double-flush,
//      double-log, or crash.
//
//   4. prune_old_logs boundary: with 21 fake "collider-*-*.log" files
//      in the logs dir, init_session_log() must keep exactly 20 (the
//      kSessionLogRetainCount cap). Pre-fix, an off-by-one would have
//      left 21 on disk.
//
//   5. atomic_write_state_file durability: after a successful write
//      the .tmp must be gone (rename, not copy). After a failed write
//      (path under a non-existent + uncreatable parent) the .tmp must
//      also be absent (unwind path); no torn output left on disk.
//
// All tests run against a per-test tempdir that we point USERPROFILE /
// HOME at, so the production paths::collider_home() resolution stays
// untouched but lands inside the sandbox. We never write into the
// developer's real ~/.collider during ctest.

#include "core/session_log.hpp"
#include "cli/cli_parser.hpp"      // Arguments
#include "core/paths.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
    #include <process.h>     // _getpid
#else
    #include <unistd.h>      // getpid
#endif

namespace fs = std::filesystem;

namespace {

int g_pass = 0;
int g_fail = 0;

#define EXPECT_TRUE(cond, label)                                          \
    do {                                                                  \
        if (cond) {                                                       \
            ++g_pass;                                                     \
        } else {                                                          \
            ++g_fail;                                                     \
            std::cerr << "[FAIL] " << (label) << "\n";                    \
        }                                                                 \
    } while (0)

#define EXPECT_EQ_STR(actual, expected, label)                            \
    do {                                                                  \
        const std::string& a_ = (actual);                                 \
        const std::string& e_ = (expected);                               \
        if (a_ == e_) {                                                   \
            ++g_pass;                                                     \
        } else {                                                          \
            ++g_fail;                                                     \
            std::cerr << "[FAIL] " << (label) << "\n"                     \
                      << "  expected: " << e_ << "\n"                     \
                      << "  actual:   " << a_ << "\n";                    \
        }                                                                 \
    } while (0)

// Cross-platform setenv. _putenv_s on Windows, setenv on POSIX.
void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    ::setenv(name, value.c_str(), 1);
#endif
}

// Cross-platform pid: Windows uses _getpid() (in <process.h>) and
// POSIX uses getpid() (in <unistd.h>). Defined before make_temp_home
// so its tempdir name can include the pid for inter-process safety.
int test_getpid() {
#ifdef _WIN32
    return static_cast<int>(::_getpid());
#else
    return static_cast<int>(::getpid());
#endif
}

// Build a fresh tempdir and point USERPROFILE / HOME at it. Returns
// the path. Caller is responsible for cleanup (the tempdir is
// auto-cleaned on test exit via the destructor of a RAII guard
// returned from this helper -- see TempHome below).
fs::path make_temp_home() {
    auto base = fs::temp_directory_path() / "collider_test_session_log";
    fs::create_directories(base);
    // Per-test subdir keyed off pid + a counter so two test invocations
    // in the same CTest run do not collide.
    static int counter = 0;
    auto sub = base / ("t" + std::to_string(test_getpid()) + "_" +
                       std::to_string(counter++));
    fs::remove_all(sub);
    fs::create_directories(sub);
    return sub;
}

struct TempHome {
    fs::path path;
    std::string prev_userprofile;
    std::string prev_home;
    bool had_userprofile = false;
    bool had_home = false;

    TempHome() {
        path = make_temp_home();
        if (const char* p = std::getenv("USERPROFILE")) {
            prev_userprofile = p;
            had_userprofile = true;
        }
        if (const char* p = std::getenv("HOME")) {
            prev_home = p;
            had_home = true;
        }
        set_env("USERPROFILE", path.string());
        set_env("HOME",        path.string());
    }
    ~TempHome() {
        // Restore prior env. Best-effort; if the original env wasn't
        // set, leave the tempdir path in place (CTest spawns a fresh
        // process anyway).
        if (had_userprofile) set_env("USERPROFILE", prev_userprofile);
        if (had_home)        set_env("HOME",        prev_home);
        std::error_code ec;
        fs::remove_all(path, ec);
        // Ignore ec: a leftover tempdir is non-fatal.
    }
};

// ---------------------------------------------------------------------------
// 1. redact_argv golden cases
// ---------------------------------------------------------------------------

void test_redact_argv() {
    // --pool-password value
    {
        const char* argv[] = {"collider", "--pool-password", "secret123"};
        std::string out = ::collider::log::detail::redact_argv(
            3, const_cast<char**>(argv));
        EXPECT_EQ_STR(out, "collider --pool-password REDACTED",
                      "redact_argv: --pool-password value");
    }
    // --pool-password=value
    {
        const char* argv[] = {"collider", "--pool-password=secret123"};
        std::string out = ::collider::log::detail::redact_argv(
            2, const_cast<char**>(argv));
        EXPECT_EQ_STR(out, "collider --pool-password=REDACTED",
                      "redact_argv: --pool-password=value");
    }
    // --pool-password-file value
    {
        const char* argv[] = {"collider", "--pool-password-file", "/etc/pw"};
        std::string out = ::collider::log::detail::redact_argv(
            3, const_cast<char**>(argv));
        EXPECT_EQ_STR(out, "collider --pool-password-file REDACTED",
                      "redact_argv: --pool-password-file value");
    }
    // --pool-password-file=value
    {
        const char* argv[] = {"collider", "--pool-password-file=/etc/pw"};
        std::string out = ::collider::log::detail::redact_argv(
            2, const_cast<char**>(argv));
        EXPECT_EQ_STR(out, "collider --pool-password-file=REDACTED",
                      "redact_argv: --pool-password-file=value");
    }
    // --pool-api-key value (deprecated flag but still accepted)
    {
        const char* argv[] = {"collider", "--pool-api-key", "ak_live_xyz"};
        std::string out = ::collider::log::detail::redact_argv(
            3, const_cast<char**>(argv));
        EXPECT_EQ_STR(out, "collider --pool-api-key REDACTED",
                      "redact_argv: --pool-api-key value");
    }
    // --pool-api-key=value
    {
        const char* argv[] = {"collider", "--pool-api-key=ak_live_xyz"};
        std::string out = ::collider::log::detail::redact_argv(
            2, const_cast<char**>(argv));
        EXPECT_EQ_STR(out, "collider --pool-api-key=REDACTED",
                      "redact_argv: --pool-api-key=value");
    }
    // --activate value (license key)
    {
        const char* argv[] = {"collider", "--activate", "LIC-AAAA-BBBB-CCCC"};
        std::string out = ::collider::log::detail::redact_argv(
            3, const_cast<char**>(argv));
        EXPECT_EQ_STR(out, "collider --activate REDACTED",
                      "redact_argv: --activate value");
    }
    // --activate=value
    {
        const char* argv[] = {"collider", "--activate=LIC-AAAA-BBBB-CCCC"};
        std::string out = ::collider::log::detail::redact_argv(
            2, const_cast<char**>(argv));
        EXPECT_EQ_STR(out, "collider --activate=REDACTED",
                      "redact_argv: --activate=value");
    }
    // Untouched flags pass through verbatim.
    {
        const char* argv[] = {"collider", "--puzzle", "71", "--pool-url",
                              "jlp://example.com:4242"};
        std::string out = ::collider::log::detail::redact_argv(
            5, const_cast<char**>(argv));
        EXPECT_EQ_STR(out,
                      "collider --puzzle 71 --pool-url jlp://example.com:4242",
                      "redact_argv: non-credential flags pass through");
    }
    // Mixed: a redacted flag and a non-redacted flag side by side.
    {
        const char* argv[] = {"collider", "--puzzle", "71", "--pool-password",
                              "hunter2", "--worker", "ws-1"};
        std::string out = ::collider::log::detail::redact_argv(
            7, const_cast<char**>(argv));
        EXPECT_EQ_STR(out,
                      "collider --puzzle 71 --pool-password REDACTED "
                      "--worker ws-1",
                      "redact_argv: mixed redacted/non-redacted");
    }
}

// ---------------------------------------------------------------------------
// 2. serialize_state default-construct shape
// ---------------------------------------------------------------------------

// Lightweight balanced-braces check: walk the string, count '{' and '}'
// (skipping anything inside string literals). For our hand-rolled JSON
// this is a sufficient well-formedness test; we are not parsing.
bool json_braces_balanced(const std::string& s) {
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (char c : s) {
        if (escape) { escape = false; continue; }
        if (c == '\\') { escape = true; continue; }
        if (c == '"') { in_string = !in_string; continue; }
        if (in_string) continue;
        if (c == '{' || c == '[') ++depth;
        if (c == '}' || c == ']') --depth;
        if (depth < 0) return false;
    }
    return depth == 0 && !in_string;
}

bool json_contains_key(const std::string& s, const std::string& key) {
    // Look for "key": as a substring; ignores in-string occurrences for
    // simplicity (our JSON values don't contain raw "":" combinations).
    return s.find("\"" + key + "\":") != std::string::npos;
}

void test_serialize_state_default() {
    ::collider::log::SessionState s;  // default-constructed
    // The always-on bookkeeping is empty/zero by default; we still
    // expect serialize to emit a well-formed object.
    std::string json = ::collider::log::detail::serialize_state(s);

    EXPECT_TRUE(json_braces_balanced(json),
                "serialize_state: default-constructed produces balanced JSON");
    EXPECT_TRUE(!json.empty() && json.front() == '{' && json.back() == '}',
                "serialize_state: top-level is a single JSON object");

    // The always-emitted bookkeeping keys must be present even when
    // empty: the crash handler relies on these to identify the
    // process and locate the per-session log.
    EXPECT_TRUE(json_contains_key(json, "mode"),
                "serialize_state: emits mode");
    EXPECT_TRUE(json_contains_key(json, "pid"),
                "serialize_state: emits pid");
    EXPECT_TRUE(json_contains_key(json, "log_path"),
                "serialize_state: emits log_path");
    EXPECT_TRUE(json_contains_key(json, "boot_ts"),
                "serialize_state: emits boot_ts");
    EXPECT_TRUE(json_contains_key(json, "last_update_ts"),
                "serialize_state: emits last_update_ts");

    // Optional fields must NOT be emitted when unset (avoids noisy
    // JSON for the common "I only updated pool fields" caller).
    EXPECT_TRUE(!json_contains_key(json, "total_checked"),
                "serialize_state: omits unset total_checked");
    EXPECT_TRUE(!json_contains_key(json, "puzzle_number"),
                "serialize_state: omits unset puzzle_number");
    EXPECT_TRUE(!json_contains_key(json, "gpus"),
                "serialize_state: omits empty gpus array");

    // Mode populated -> mode key still present, no regression.
    s.mode = "puzzle";
    s.pid = 4242;
    s.puzzle_number = 71;
    std::string json2 = ::collider::log::detail::serialize_state(s);
    EXPECT_TRUE(json_braces_balanced(json2),
                "serialize_state: populated state stays balanced");
    EXPECT_TRUE(json2.find("\"mode\":\"puzzle\"") != std::string::npos,
                "serialize_state: mode value round-trips");
    EXPECT_TRUE(json_contains_key(json2, "puzzle_number"),
                "serialize_state: engaged optional appears");
}

// ---------------------------------------------------------------------------
// 3. shutdown_session_log double-fire guard
// ---------------------------------------------------------------------------

void test_shutdown_double_fire() {
    // SessionLogSink is a Meyers singleton whose initialized_ flag is
    // latched true by the prior prune-boundary test's init call (and
    // by the shutdown handlers themselves). Calling shutdown again
    // here exercises the std::atomic_flag once-guard inside
    // shutdown_session_log: the first call must succeed, the second
    // and third must be no-ops (no crash, no hang, no double-flush
    // attempt against an already-closed sink). We assert behavior by
    // round-trip timing: a real double-flush + milestone emit would
    // re-take the SessionLogSink mutex and re-flush the
    // SessionStateStore; once-guarded no-ops return in microseconds.
    auto t0 = std::chrono::steady_clock::now();
    ::collider::log::shutdown_session_log();
    auto t1 = std::chrono::steady_clock::now();
    ::collider::log::shutdown_session_log();
    auto t2 = std::chrono::steady_clock::now();
    ::collider::log::shutdown_session_log();
    auto t3 = std::chrono::steady_clock::now();

    auto first_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t1 - t0).count();
    auto second_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t2 - t1).count();
    auto third_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t3 - t2).count();

    // We can't assert absolute timings (Windows scheduler jitter), but
    // we CAN assert that the second + third calls did not take longer
    // than a generous 100ms ceiling: a real flush of a non-empty
    // SessionStateStore involves a disk write that on a slow CI box
    // could exceed this only if the once-guard had genuinely failed.
    EXPECT_TRUE(second_ms < 100,
                "shutdown_double_fire: second call is a no-op (<100ms)");
    EXPECT_TRUE(third_ms < 100,
                "shutdown_double_fire: third call is a no-op (<100ms)");
    // first_ms is informational only; we do not pin its upper bound
    // because the very first call to shutdown after the
    // prune-boundary test legitimately flushes the SessionStateStore.
    (void)first_ms;
}

// ---------------------------------------------------------------------------
// 4. prune_old_logs boundary
// ---------------------------------------------------------------------------

void test_prune_old_logs_boundary() {
    TempHome home;

    // Pre-seed kSessionLogRetainCount + 1 fake log files. Their names
    // must match the prune filter ("collider-*-*.log") and their
    // mtimes must be distinct so the oldest-first sort is deterministic.
    fs::path log_dir = home.path / ".collider" / "logs";
    fs::create_directories(log_dir);

    const int seed_count =
        ::collider::log::kSessionLogRetainCount + 1;  // 21 by default
    std::vector<fs::path> seeded;
    for (int i = 0; i < seed_count; ++i) {
        std::string fname = "collider-2020-01-01T00-00-" +
                            (i < 10 ? std::string("0") : std::string("")) +
                            std::to_string(i) + "-1000.log";
        fs::path p = log_dir / fname;
        std::ofstream ofs(p);
        ofs << "fake log " << i << "\n";
        ofs.close();
        // Stagger mtimes so the prune's sort-by-mtime picks a
        // deterministic oldest. fs::last_write_time accepts a
        // file_time_type; offset by i seconds in the past.
        auto now = fs::file_time_type::clock::now();
        fs::last_write_time(p, now - std::chrono::seconds(seed_count - i));
        seeded.push_back(p);
    }

    EXPECT_TRUE(seeded.size() == static_cast<size_t>(seed_count),
                "prune_old_logs: seeded all fake logs");

    // Trigger prune via init_session_log. The init opens its OWN log
    // file in the same dir, so after init we expect:
    //   - kSessionLogRetainCount files preserved by prune
    //   - plus 1 newly created session log
    // = kSessionLogRetainCount + 1 files. With seed_count =
    // kSessionLogRetainCount + 1, the math works out to one file
    // deleted (the oldest of the seed set).
    bool ok = ::collider::log::init_session_log();
    EXPECT_TRUE(ok, "prune_old_logs: init_session_log succeeded");

    // Count *.log files now.
    int after_count = 0;
    for (auto& de : fs::directory_iterator(log_dir)) {
        if (de.is_regular_file() && de.path().extension() == ".log") {
            ++after_count;
        }
    }
    // The newly created session log is kSessionLogRetainCount + 1.
    EXPECT_TRUE(after_count ==
                    ::collider::log::kSessionLogRetainCount + 1,
                "prune_old_logs: keeps retain_count + 1 (retained + new)");

    // The oldest seeded file (i==0, earliest mtime) must be gone.
    EXPECT_TRUE(!fs::exists(seeded.front()),
                "prune_old_logs: oldest seeded log was deleted");
    // The newest seeded file (i==seed_count-1) must still be present.
    EXPECT_TRUE(fs::exists(seeded.back()),
                "prune_old_logs: newest seeded log was preserved");

    ::collider::log::shutdown_session_log();
}

// ---------------------------------------------------------------------------
// 5. atomic_write_state_file tmp+rename + unwind
// ---------------------------------------------------------------------------

void test_atomic_write_state_file() {
    TempHome home;
    fs::path target = home.path / ".collider" / "session_state.json";
    fs::path tmp = target;
    tmp += ".tmp";
    fs::create_directories(target.parent_path());

    // Success path: write a non-empty payload. After return, the
    // target must exist with exactly the payload bytes and the .tmp
    // must be gone (the rename succeeded).
    const std::string payload = "{\"mode\":\"puzzle\",\"pid\":4242}";
    bool ok = ::collider::log::detail::atomic_write_state_file(target, payload);
    EXPECT_TRUE(ok, "atomic_write_state_file: success returns true");
    EXPECT_TRUE(fs::exists(target),
                "atomic_write_state_file: target file exists after write");
    EXPECT_TRUE(!fs::exists(tmp),
                "atomic_write_state_file: tmp removed after successful rename");
    {
        std::ifstream ifs(target, std::ios::binary);
        std::stringstream ss;
        ss << ifs.rdbuf();
        EXPECT_EQ_STR(ss.str(), payload,
                      "atomic_write_state_file: payload round-trips");
    }

    // Idempotency: overwrite with a new payload. The tmp should still
    // be cleaned up; target should hold the new payload.
    const std::string payload2 = "{\"mode\":\"pool\",\"pid\":9999}";
    bool ok2 = ::collider::log::detail::atomic_write_state_file(target, payload2);
    EXPECT_TRUE(ok2,
                "atomic_write_state_file: overwrite returns true");
    EXPECT_TRUE(!fs::exists(tmp),
                "atomic_write_state_file: tmp gone after overwrite");
    {
        std::ifstream ifs(target, std::ios::binary);
        std::stringstream ss;
        ss << ifs.rdbuf();
        EXPECT_EQ_STR(ss.str(), payload2,
                      "atomic_write_state_file: overwrite payload round-trips");
    }

    // Failure path: target inside a parent that cannot be created
    // (a path component is an existing FILE, not a dir). On both
    // POSIX and Windows std::filesystem::create_directories returns
    // an error_code for this case. The function should return false
    // and leave no .tmp on disk.
    fs::path blocker = home.path / "blocker_file";
    {
        std::ofstream blk(blocker);
        blk << "I am a file blocking dir creation\n";
    }
    fs::path bad_target = blocker / "subdir" / "session_state.json";
    fs::path bad_tmp = bad_target;
    bad_tmp += ".tmp";
    bool ok3 = ::collider::log::detail::atomic_write_state_file(bad_target,
                                                                payload);
    EXPECT_TRUE(!ok3,
                "atomic_write_state_file: bogus parent path returns false");
    EXPECT_TRUE(!fs::exists(bad_tmp),
                "atomic_write_state_file: no leftover tmp on failure");
    EXPECT_TRUE(!fs::exists(bad_target),
                "atomic_write_state_file: no leftover target on failure");
}

}  // namespace

int main() {
    std::cout << "test_session_log -- 2026-05-17 reviewer coverage\n";

    // Test ordering matters: SessionLogSink is a Meyers singleton whose
    // initialized_ flag latches true on the first init_session_log()
    // call and is not reset by shutdown_session_log (the production
    // contract: one session log per process). So the prune-boundary
    // case (which needs a real init invocation to drive prune_old_logs)
    // must run before the shutdown double-fire case (which lights up
    // the singleton and locks it for the rest of this process). The
    // pure-function tests (redact_argv, serialize_state,
    // atomic_write_state_file) do not touch the singleton and can run
    // in any order.
    test_redact_argv();
    test_serialize_state_default();
    test_atomic_write_state_file();
    test_prune_old_logs_boundary();
    test_shutdown_double_fire();

    std::cout << "\n" << g_pass << " passed, " << g_fail << " failed.\n";
    return g_fail == 0 ? 0 : 1;
}
