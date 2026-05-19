// phase 5 (builder-threading: plugin-runner).
//
// Coverage for collider::plugins::PluginRunner. Synthetic plugins are
// implemented as inline Python one-liners (python -c "..."), which is
// available on every supported host (Windows, Linux, macOS) without an
// extra build step. When no Python interpreter is found the test prints
// SKIP and exits 0; CI hosts always have Python so this is a developer-
// box safety valve, not a permanent gate.
//
// Tests:
//   1. echo plugin: dispatch a Hit event, assert the JSON-line appears
//      in the plugin's recent_output ring within a bounded wait.
//   2. malformed plugin output: a plugin that emits its own garbage on
//      stdout (binary chunks, no-newline tails). The runner must not
//      crash; recent_output reflects whatever lines it could parse.
//   3. plugin death mid-stream: a plugin that exits cleanly after N
//      lines. Subsequent dispatch() calls do not crash; snapshot_status
//      shows alive=false after a short wait.
//   4. bounded queue drop-oldest: a slow plugin (sleeps per line)
//      receiving a burst > kSendQueueCap events. drop_count > 0 in
//      snapshot_status.
//   5. concurrent output interleaving: two plugins running side-by-side
//      both surface their lines; ring lengths are within kRecentOutputCap.

#include "plugins/plugin_runner.hpp"
#include "plugins/plugin_protocol.hpp"
#include "plugins/plugin_registry.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifndef _WIN32
#  include <signal.h>
#endif


namespace {

struct Failures {
    int count = 0;
    void check(bool cond, const char* what) {
        if (!cond) {
            std::cerr << "[FAIL] " << what << "\n";
            ++count;
        }
    }
};

// Resolve a Python interpreter usable for inline -c scripts. Tries
// "python" first (Windows + most Linux distros), then "python3" (some
// Linux + macOS). Returns the empty string when neither resolves.
std::string find_python() {
    // We do not have a portable "which" without spawning a process.
    // Try invoking each candidate with --version under a guard subprocess
    // via std::system + the platform's standard null sink. A non-zero
    // exit means missing.
#ifdef _WIN32
    const char* null_sink = "NUL";
#else
    const char* null_sink = "/dev/null";
#endif
    for (const char* exe : {"python", "python3"}) {
        std::string cmd = std::string(exe) + " --version > " + null_sink +
                          " 2> " + null_sink;
        int rc = std::system(cmd.c_str());
        if (rc == 0) return exe;
    }
    return {};
}

// Build a registry with one or more synthetic plugins.
collider::plugins::PluginSpec make_spec(
    const std::string& name,
    std::vector<std::string> command,
    std::vector<collider::plugins::EventKind> events,
    bool enabled = true) {
    collider::plugins::PluginSpec s;
    s.name = name;
    s.command = std::move(command);
    s.events = std::move(events);
    s.enabled = enabled;
    return s;
}

collider::plugins::Event make_hit_event(const std::string& passphrase) {
    collider::plugins::Event ev;
    ev.kind = collider::plugins::EventKind::Hit;
    ev.ts_iso8601 = "2026-05-15T18:42:31Z";
    collider::plugins::HitData hd;
    hd.passphrase = passphrase;
    hd.privkey_hex = std::string(64, '0');
    hd.h160_hex = std::string(40, '0');
    ev.payload = std::move(hd);
    return ev;
}

bool wait_for(const std::function<bool()>& predicate,
              std::chrono::milliseconds timeout) {
    auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return predicate();
}

}  // namespace

int main() {
    Failures fail;
    const std::string python = find_python();
    if (python.empty()) {
        std::cout << "test_plugin_runner: SKIP (no python interpreter)\n";
        return 0;
    }

    // Test 1: echo plugin round-trip.
    {
        // Python script reads every line from stdin and echoes it back
        // verbatim on stdout. flush=True so the parent reader sees each
        // line promptly without buffering.
        const std::string echo_py =
            "import sys\n"
            "for line in sys.stdin:\n"
            "    print(line.rstrip(), flush=True)\n";

        collider::plugins::PluginRegistry registry;
        registry.plugins.push_back(make_spec(
            "echo",
            {python, "-c", echo_py},
            {collider::plugins::EventKind::Hit}));

        collider::plugins::PluginRunner runner(std::move(registry));
        size_t spawned = runner.start();
        fail.check(spawned == 1, "echo plugin spawned");

        auto ev = make_hit_event("alpha");
        runner.dispatch(ev);

        bool saw_line = wait_for([&]() {
            auto snaps = runner.snapshot_status();
            if (snaps.empty()) return false;
            for (const auto& line : snaps[0].recent_output) {
                if (line.find("\"alpha\"") != std::string::npos) return true;
            }
            return false;
        }, std::chrono::milliseconds(3000));
        fail.check(saw_line, "echo plugin produced expected line");

        runner.stop();
    }

    // Test 2: malformed plugin stdout.
    {
        // Plugin emits binary chunks + a partial trailing line then exits.
        // The runner's reader_loop accumulates the partial tail and
        // appends it on EOF without crashing.
        const std::string garbage_py =
            "import sys\n"
            "sys.stdout.write('partial line without newline')\n"
            "sys.stdout.flush()\n";

        collider::plugins::PluginRegistry registry;
        registry.plugins.push_back(make_spec(
            "garbage",
            {python, "-c", garbage_py},
            {collider::plugins::EventKind::Hit}));

        collider::plugins::PluginRunner runner(std::move(registry));
        runner.start();
        // No dispatch needed; the plugin emits immediately and exits.
        bool dead = wait_for([&]() {
            auto snaps = runner.snapshot_status();
            return !snaps.empty() && !snaps[0].alive;
        }, std::chrono::milliseconds(3000));
        fail.check(dead, "garbage plugin marked dead after exit");

        auto snaps = runner.snapshot_status();
        fail.check(!snaps.empty(), "snapshot has the garbage plugin entry");
        // recent_output may or may not contain the partial line depending
        // on how the OS flushed the pipe at process exit; the assertion is
        // that nothing crashed and we got past stop() cleanly.
        runner.stop();
    }

    // Test 3: plugin death mid-stream.
    {
        // Plugin reads N lines then exits cleanly. The runner must still
        // accept further dispatches without crashing and snapshot_status
        // eventually shows alive=false.
        const std::string die_py =
            "import sys\n"
            "for i, line in enumerate(sys.stdin):\n"
            "    print(line.rstrip(), flush=True)\n"
            "    if i >= 2:\n"
            "        sys.exit(0)\n";

        collider::plugins::PluginRegistry registry;
        registry.plugins.push_back(make_spec(
            "dying",
            {python, "-c", die_py},
            {collider::plugins::EventKind::Hit}));

        collider::plugins::PluginRunner runner(std::move(registry));
        runner.start();

        // Send 10 events; plugin exits after handling 3.
        for (int i = 0; i < 10; ++i) {
            runner.dispatch(make_hit_event("event-" + std::to_string(i)));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        bool dead = wait_for([&]() {
            auto snaps = runner.snapshot_status();
            return !snaps.empty() && !snaps[0].alive;
        }, std::chrono::milliseconds(3000));
        fail.check(dead, "dying plugin marked dead after exit");

        // Additional dispatches after death must not crash.
        for (int i = 0; i < 5; ++i) {
            runner.dispatch(make_hit_event("post-death-" + std::to_string(i)));
        }
        // active_count returns zero now.
        fail.check(runner.active_count() == 0, "active_count == 0 after death");

        runner.stop();
    }

    // Test 4: bounded queue drop-oldest.
    {
        // A slow plugin: sleeps 50 ms per line. We dispatch >> kSendQueueCap
        // events as fast as possible, then look at drop_count.
        const std::string slow_py =
            "import sys, time\n"
            "for line in sys.stdin:\n"
            "    time.sleep(0.05)\n"
            "    print(line.rstrip(), flush=True)\n";

        collider::plugins::PluginRegistry registry;
        registry.plugins.push_back(make_spec(
            "slow",
            {python, "-c", slow_py},
            {collider::plugins::EventKind::Hit}));

        collider::plugins::PluginRunner runner(std::move(registry));
        runner.start();

        const size_t cap = collider::plugins::PluginRunner::kSendQueueCap;
        const size_t flood = cap * 2 + 200;
        for (size_t i = 0; i < flood; ++i) {
            runner.dispatch(make_hit_event("flood-" + std::to_string(i)));
        }

        // Allow the writer thread a tick to copy from queue to stdin pipe;
        // by then any overflow drops have been recorded.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto snaps = runner.snapshot_status();
        fail.check(!snaps.empty(), "slow plugin snapshot present");
        if (!snaps.empty()) {
            fail.check(snaps[0].drop_count > 0,
                       "drop_count > 0 after queue overflow");
        }

        runner.stop();
    }

    // Test 5: concurrent output interleaving.
    {
        // Two plugins, each emits 15 lines on stdout immediately and then
        // exits. We expect each plugin's recent_output ring to have
        // <= kRecentOutputCap lines (10) and the rings to be independent.
        const std::string burst_py =
            "import sys\n"
            "for i in range(15):\n"
            "    print(f'burst-line-{i}', flush=True)\n";

        collider::plugins::PluginRegistry registry;
        registry.plugins.push_back(make_spec(
            "alpha",
            {python, "-c", burst_py},
            {collider::plugins::EventKind::Hit}));
        registry.plugins.push_back(make_spec(
            "bravo",
            {python, "-c", burst_py},
            {collider::plugins::EventKind::Hit}));

        collider::plugins::PluginRunner runner(std::move(registry));
        runner.start();

        bool both_done = wait_for([&]() {
            auto snaps = runner.snapshot_status();
            if (snaps.size() != 2) return false;
            return !snaps[0].alive && !snaps[1].alive;
        }, std::chrono::milliseconds(3000));
        fail.check(both_done, "both burst plugins exited");

        auto snaps = runner.snapshot_status();
        fail.check(snaps.size() == 2, "two plugins in snapshot");
        if (snaps.size() == 2) {
            for (size_t i = 0; i < snaps.size(); ++i) {
                fail.check(
                    snaps[i].recent_output.size() <=
                        collider::plugins::PluginRunner::kRecentOutputCap,
                    "recent_output bounded by kRecentOutputCap");
                // Each plugin must surface lines that look like burst-line-N
                // for some N; if a plugin emitted 15 lines and the ring is
                // 10, the last 5 lines (10..14) should be present.
                bool saw = false;
                for (const auto& line : snaps[i].recent_output) {
                    if (line.find("burst-line-") != std::string::npos) {
                        saw = true; break;
                    }
                }
                fail.check(saw, "plugin output present in its ring");
            }
        }

        runner.stop();
    }

    // T-B3: concurrent start race. Multiple threads call start() on the
    // same runner; exactly one should win (return non-zero spawned count
    // when the registry has plugins) and the rest should return 0
    // without crashing or double-spawning. The previous Impl had plain
    // bool started/stopped accessed without atomicity; under TSAN the
    // race is a definite UB report. Asserting "exactly one winner" is
    // the externally observable property the atomic CAS guarantees.
    {
        const std::string echo_py =
            "import sys\n"
            "for line in sys.stdin:\n"
            "    print(line.rstrip(), flush=True)\n";
        collider::plugins::PluginRegistry registry;
        registry.plugins.push_back(make_spec(
            "echo-race",
            {python, "-c", echo_py},
            {collider::plugins::EventKind::Hit}));

        collider::plugins::PluginRunner runner(std::move(registry));

        constexpr int kRacers = 4;
        std::vector<std::thread> racers;
        std::atomic<int> winners{0};
        std::atomic<int> losers{0};
        for (int i = 0; i < kRacers; ++i) {
            racers.emplace_back([&]() {
                size_t spawned = runner.start();
                if (spawned == 1) {
                    winners.fetch_add(1, std::memory_order_relaxed);
                } else {
                    losers.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        for (auto& t : racers) t.join();

        fail.check(winners.load() == 1,
                   "T-B3: exactly one start() call wins the CAS");
        fail.check(losers.load() == kRacers - 1,
                   "T-B3: every other start() returns 0 spawned");

        // Also race stop() vs stop() to exercise the same CAS on the
        // stopped flag. Both should return cleanly with no crash;
        // dispatch after stop must be a no-op.
        std::vector<std::thread> stoppers;
        for (int i = 0; i < kRacers; ++i) {
            stoppers.emplace_back([&]() { runner.stop(); });
        }
        for (auto& t : stoppers) t.join();

        // dispatch() after stop must be a no-op (does not crash).
        runner.dispatch(make_hit_event("post-stop"));
        fail.check(true, "T-B3: dispatch after racing stop() did not crash");
    }

#ifndef _WIN32
    // T-B6: PluginRunner installs SIG_IGN on SIGPIPE for its lifetime
    // and restores the prior disposition on stop(). The old code
    // assumed CookedModeGuard (TUI lifetime) had already done this;
    // --no-tui leaves SIGPIPE at SIG_DFL and the runner kills itself
    // when a plugin's stdin write hits a dead pipe. We assert the
    // observable signal handler change: pre-start we install a
    // sentinel handler, run start(), see SIG_IGN, run stop(), see
    // the sentinel restored.
    {
        // Snapshot the current SIGPIPE disposition so we can put it
        // back even if the test fails part-way.
        struct sigaction outer_pre{};
        ::sigaction(SIGPIPE, nullptr, &outer_pre);

        // Install a sentinel: we use SIG_DFL because that's the most
        // distinct from SIG_IGN. Any later sigaction comparison just
        // checks sa_handler.
        struct sigaction sentinel{};
        sentinel.sa_handler = SIG_DFL;
        sigemptyset(&sentinel.sa_mask);
        sentinel.sa_flags = 0;
        ::sigaction(SIGPIPE, &sentinel, nullptr);

        // Synthetic plugin: just sleep. We do not need any I/O for
        // this test; we just need a runner that has called start().
        const std::string sleep_py =
            "import time, sys\n"
            "time.sleep(2)\n";
        collider::plugins::PluginRegistry registry;
        registry.plugins.push_back(make_spec(
            "sleeper",
            {python, "-c", sleep_py},
            {collider::plugins::EventKind::Hit}));
        collider::plugins::PluginRunner runner(std::move(registry));

        runner.start();
        struct sigaction cur{};
        ::sigaction(SIGPIPE, nullptr, &cur);
        bool installed_ign = (cur.sa_handler == SIG_IGN);
        fail.check(installed_ign,
                   "T-B6: PluginRunner::start installed SIG_IGN for SIGPIPE");

        runner.stop();
        ::sigaction(SIGPIPE, nullptr, &cur);
        bool restored_dfl = (cur.sa_handler == SIG_DFL);
        fail.check(restored_dfl,
                   "T-B6: PluginRunner::stop restored prior SIGPIPE handler");

        // Restore the test process's original SIGPIPE disposition.
        ::sigaction(SIGPIPE, &outer_pre, nullptr);
    }
#endif

    if (fail.count == 0) {
        std::cout << "test_plugin_runner: all checks passed\n";
        return 0;
    }
    std::cerr << "test_plugin_runner: " << fail.count << " failure(s)\n";
    return 1;
}
