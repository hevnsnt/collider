/**
 * test_cli_parser.cpp - CLI parser + config override matrix tests.
 *
 * Validates the CLIFlags-aware override behaviour in apply_config_to_args() and
 * the mode-mutex check in parse_args_core(). Each test row exercises one cell
 * of the (CLI flag x config field) matrix from
 * docs/review-2026-05-04/track-e-cli-config.md.
 *
 * The test links the real parser (src/cli/cli_parser.cpp via the registry in
 * src/cli/flag_spec.hpp) directly. parse_args_mirror() below is a one-line
 * adapter that turns vector<string> argv into char*[] and calls
 * parse_args_core() through src/cli/cli_parser.hpp. There is no longer any
 * in-test reimplementation of flag parsing. A regression in the production
 * parser fails this test, which was the documented goal of the T2 wiring
 * fix in v1.4.2-final-validation-blockers.md.
 *
 * If you add a CLI flag, register it in src/cli/cli_parser.cpp's kFlagsRaw[]
 * table. No test-side mirror update is needed.
 */

#include "cli/cli_parser.hpp"     // ::Arguments, parse_args_core, ::collider::CLIFlags
#include "core/yaml_config.hpp"   // collider::AppConfig, apply_config_to_args

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Adapter: turn vector<string> argv -> char*[] and call the real parser.
// Slot 0 is a synthetic program name (parse_args_core skips it the same way
// it skips argv[0] in production).
// ---------------------------------------------------------------------------
int parse_args_mirror(const std::vector<std::string>& argv,
                      Arguments& args,
                      collider::CLIFlags& cli,
                      std::string& err_msg) {
    std::vector<std::string> storage;
    storage.reserve(argv.size() + 1);
    storage.emplace_back("test_collider");
    for (auto& s : argv) storage.push_back(s);

    std::vector<char*> argv_c;
    argv_c.reserve(storage.size());
    for (auto& s : storage) argv_c.push_back(s.data());

    int argc = static_cast<int>(argv_c.size());
    return parse_args_core(argc, argv_c.data(), args, cli, err_msg);
}

// ---------------------------------------------------------------------------
// Tiny test harness
// ---------------------------------------------------------------------------
struct TestStats {
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failures;
};
TestStats g_stats;

#define EXPECT_TRUE(cond, name) do { \
    if (!(cond)) { \
        g_stats.failed++; \
        g_stats.failures.emplace_back(std::string(name) + ": EXPECT_TRUE(" #cond ") failed at line " + std::to_string(__LINE__)); \
        std::cerr << "[FAIL] " << name << " line " << __LINE__ << ": " #cond "\n"; \
    } else { \
        g_stats.passed++; \
    } \
} while (0)

#define EXPECT_EQ(actual, expected, name) do { \
    auto _eq_actual = (actual); auto _eq_expected = (expected); \
    if (!(_eq_actual == _eq_expected)) { \
        g_stats.failed++; \
        std::ostringstream _eq_oss; _eq_oss << name << ": EXPECT_EQ(" #actual ", " #expected ") failed: got " << _eq_actual << " want " << _eq_expected << " line " << __LINE__; \
        g_stats.failures.emplace_back(_eq_oss.str()); \
        std::cerr << "[FAIL] " << _eq_oss.str() << "\n"; \
    } else { \
        g_stats.passed++; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// Test rows: each row asserts the resulting Arguments after argv -> parse ->
// apply_config_to_args(). Comments reference track-e matrix cells.
// ---------------------------------------------------------------------------

void run_all() {
    using collider::AppConfig;
    using collider::CLIFlags;
    using collider::apply_config_to_args;

    // -----------------------------------------------------------------------
    // Group A: parse_args alone (per-flag CLIFlags bit accuracy)
    // -----------------------------------------------------------------------
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--batch-size", "4000000"}, a, c, err), 0, "A.01.batch-size-default-value");
        EXPECT_TRUE(c.batch_size_set, "A.01.bit-set");
        EXPECT_EQ(a.batch_size, (size_t)4000000, "A.01.value");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({}, a, c, err), 0, "A.02.no-flags");
        EXPECT_TRUE(!c.batch_size_set, "A.02.no-bit-set");
        EXPECT_EQ(a.batch_size, (size_t)4000000, "A.02.default-value");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--puzzle", "0"}, a, c, err), 0, "A.03.puzzle-zero");
        EXPECT_TRUE(c.puzzle_number_set, "A.03.bit-set-on-explicit-zero");
        EXPECT_EQ(a.puzzle_number, 0, "A.03.value-zero");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--puzzle"}, a, c, err), 0, "A.04.puzzle-no-arg");
        EXPECT_TRUE(!c.puzzle_number_set, "A.04.no-bit-without-N");
        EXPECT_TRUE(a.puzzle_mode, "A.04.puzzle-mode-on");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--random"}, a, c, err), 0, "A.05.random");
        EXPECT_TRUE(c.puzzle_random_set, "A.05.bit-set");
        EXPECT_TRUE(a.puzzle_random, "A.05.value-true");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--sequential"}, a, c, err), 0, "A.06.sequential");
        EXPECT_TRUE(c.puzzle_random_set, "A.06.bit-set");
        EXPECT_TRUE(!a.puzzle_random, "A.06.value-false");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--no-smart"}, a, c, err), 0, "A.07.no-smart");
        EXPECT_TRUE(c.smart_select_set, "A.07.bit-set");
        EXPECT_TRUE(!a.smart_select, "A.07.value-false");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--brainwallet"}, a, c, err), 0, "A.08.brainwallet");
        EXPECT_TRUE(c.brainwallet_set, "A.08.bit-set");
        EXPECT_TRUE(a.brainwallet_mode, "A.08.value");
        EXPECT_TRUE(!a.pool_mode, "A.08.pool-cleared");
    }

    // -----------------------------------------------------------------------
    // Group B: Mode mutex (E.1)
    // -----------------------------------------------------------------------
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--brainwallet", "--pool", "jlp://x:1"}, a, c, err);
        EXPECT_EQ(rc, -1, "B.01.brainwallet-then-pool-rejected");
        EXPECT_TRUE(!err.empty(), "B.01.err-msg-set");
    }
    {
        // Order test: --pool then --brainwallet; brainwallet always wins inside
        // parse_args_core (clears pool), so this should NOT trigger mutex.
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--pool", "jlp://x:1", "--brainwallet"}, a, c, err);
        EXPECT_EQ(rc, 0, "B.02.pool-then-brainwallet-allowed");
        EXPECT_TRUE(a.brainwallet_mode, "B.02.brainwallet-on");
        EXPECT_TRUE(!a.pool_mode, "B.02.pool-cleared");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--puzzle", "71", "--kangaroo"}, a, c, err);
        EXPECT_EQ(rc, 0, "B.03.puzzle-with-kangaroo-allowed");
        EXPECT_TRUE(a.puzzle_kangaroo, "B.03.kangaroo-on");
        EXPECT_EQ(a.puzzle_number, 71, "B.03.puzzle");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--brainwallet", "--kangaroo"}, a, c, err);
        EXPECT_EQ(rc, -1, "B.04.brainwallet-kangaroo-rejected");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--pool", "jlp://x:1", "--puzzle", "71"}, a, c, err);
        EXPECT_EQ(rc, -1, "B.05.pool-and-explicit-puzzle-rejected");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--brainwallet", "--all-unsolved"}, a, c, err);
        EXPECT_EQ(rc, -1, "B.06.brainwallet-all-unsolved-rejected");
    }
    {
        // Standalone --kangaroo (no brainwallet/pool) is allowed.
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--kangaroo"}, a, c, err);
        EXPECT_EQ(rc, 0, "B.07.kangaroo-alone-allowed");
        EXPECT_TRUE(a.puzzle_kangaroo, "B.07.value");
    }

    // -----------------------------------------------------------------------
    // Group C: Override matrix - value flags previously broken by sentinel
    // collisions (E.2 / E.HIGH).
    // -----------------------------------------------------------------------
    // C.01 --batch-size 4000000 (== old sentinel) vs config.batch_size = 8M
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--batch-size", "4000000"}, a, c, err);
        AppConfig cfg; cfg.batch_size = 8'000'000;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.batch_size, (size_t)4'000'000, "C.01.cli-default-value-NOT-clobbered");
    }
    // C.02 --batch-size unset, config = 8M -> config wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.batch_size = 8'000'000;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.batch_size, (size_t)8'000'000, "C.02.config-applied-when-unset");
    }
    // C.03 --puzzle 0 explicit, config.puzzle_number = 71 -> CLI 0 wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--puzzle", "0"}, a, c, err);
        AppConfig cfg; cfg.puzzle_number = 71;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.puzzle_number, 0, "C.03.cli-explicit-zero-wins");
    }
    // C.04 --puzzle unset, config 71 -> config wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.puzzle_number = 71;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.puzzle_number, 71, "C.04.config-puzzle-applied");
    }
    // C.05 --random vs config.random_search=false -> CLI random wins (was the
    // Scenario C bug)
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--random"}, a, c, err);
        AppConfig cfg; cfg.random_search = false;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.puzzle_random, "C.05.random-cli-not-flipped-by-config");
    }
    // C.06 --sequential vs config.random_search=true -> CLI sequential wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--sequential"}, a, c, err);
        AppConfig cfg; cfg.random_search = true;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(!a.puzzle_random, "C.06.sequential-cli-wins");
    }
    // C.07 --benchmark-time 30 (== default) vs config.benchmark_seconds = 60
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--benchmark-time", "30"}, a, c, err);
        AppConfig cfg; cfg.benchmark_seconds = 60;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.benchmark_seconds, 30, "C.07.benchmark-time-explicit-default-wins");
    }
    // C.08 --save-interval 1000000 (== default) vs config.save_interval = 5M
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--save-interval", "1000000"}, a, c, err);
        AppConfig cfg; cfg.save_interval = 5'000'000;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.save_interval, (size_t)1'000'000, "C.08.save-interval-explicit-default-wins");
    }

    // -----------------------------------------------------------------------
    // Group D: forgotten-flag migration (E.3)
    // -----------------------------------------------------------------------
    // D.01 --puzzle-min-bits 100 vs config.min_bits = 50 -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--puzzle-min-bits", "100"}, a, c, err);
        AppConfig cfg; cfg.min_bits = 50;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.puzzle_min_bits, 100, "D.01.min-bits-cli-wins");
    }
    // D.02 unset CLI, config 50 -> config wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.min_bits = 50;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.puzzle_min_bits, 50, "D.02.min-bits-config-applies");
    }
    // D.03 --puzzle-max-bits 80 vs config.max_bits = 120 -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--puzzle-max-bits", "80"}, a, c, err);
        AppConfig cfg; cfg.max_bits = 120;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.puzzle_max_bits, 80, "D.03.max-bits-cli-wins");
    }
    // D.04 --save-interval 5M vs config 7M -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--save-interval", "5000000"}, a, c, err);
        AppConfig cfg; cfg.save_interval = 7'000'000;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.save_interval, (size_t)5'000'000, "D.04.save-interval-cli-wins");
    }

    // -----------------------------------------------------------------------
    // Group E: Critical findings 1 + 2 (mode-leak)
    // -----------------------------------------------------------------------
    // E.01 --pool then --brainwallet => brainwallet wins, pool URL cleared
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--pool", "jlp://leaks.example:1", "--brainwallet"}, a, c, err);
        EXPECT_TRUE(a.brainwallet_mode, "E.01.brainwallet-on");
        EXPECT_TRUE(!a.pool_mode, "E.01.pool-mode-off");
        EXPECT_TRUE(a.pool_url.empty(), "E.01.pool-url-cleared");
    }
    // E.02 --brainwallet, config.pool_url set -> brainwallet wins, pool stays off
    //      (the production-bug Scenario A)
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--brainwallet", "--bloom", "x.blf"}, a, c, err);
        AppConfig cfg; cfg.pool_url = "jlp://pool.example:8888";
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.brainwallet_mode, "E.02.brainwallet-on");
        EXPECT_TRUE(!a.pool_mode, "E.02.pool-mode-off");
        // pool_url WILL get populated because nothing CLI-set the bit, but
        // pool_mode should remain false because brainwallet is set. This is
        // the documented compromise: callers must check pool_mode, not pool_url.
    }
    // E.03 --pool, config.brainwallet_enabled = true -> NOT dual mode
    //      (Scenario B Critical regression case)
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--pool", "jlp://x:1"}, a, c, err);
        AppConfig cfg; cfg.brainwallet_enabled = true;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.pool_mode, "E.03.pool-mode-stays-on");
        EXPECT_TRUE(!a.brainwallet_mode, "E.03.brainwallet-NOT-flipped-by-config");
    }
    // E.04 No CLI mode, config has brainwallet_enabled=true and no pool URL
    //      -> brainwallet on (config-only mode select)
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.brainwallet_enabled = true;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.brainwallet_mode, "E.04.config-brainwallet-applied");
        EXPECT_TRUE(!a.pool_mode, "E.04.no-pool");
    }
    // E.05 No CLI, config has both pool.url and brainwallet.enabled -> pool URL
    //      gets applied first AND would auto-enable pool, so brainwallet should
    //      NOT also flip on (apply gates on !args.pool_mode)
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg;
        cfg.pool_url = "jlp://p:1";
        cfg.brainwallet_enabled = true;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.pool_mode, "E.05.pool-on-from-config-url");
        EXPECT_TRUE(!a.brainwallet_mode, "E.05.brainwallet-NOT-also-on");
    }

    // -----------------------------------------------------------------------
    // Group F: Per-flag override coverage of remaining matrix cells (E.4)
    // -----------------------------------------------------------------------
    // F.01 --gpus vs config.gpu_devices -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--gpus", "0,1"}, a, c, err);
        AppConfig cfg; cfg.gpu_devices = {2, 3};
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.gpu_ids.size(), (size_t)2, "F.01.gpu-cli-wins-size");
        EXPECT_EQ(a.gpu_ids[0], 0, "F.01.gpu-cli-wins-0");
        EXPECT_EQ(a.gpu_ids[1], 1, "F.01.gpu-cli-wins-1");
    }
    // F.02 --gpus unset, config -> config applied
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.gpu_devices = {2, 3};
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.gpu_ids.size(), (size_t)2, "F.02.gpu-config-size");
        EXPECT_EQ(a.gpu_ids[0], 2, "F.02.gpu-config-0");
    }
    // F.03 --bloom vs config.bloom_file -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--bloom", "cli.blf"}, a, c, err);
        AppConfig cfg; cfg.bloom_file = "cfg.blf";
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.bloom_file, std::string("cli.blf"), "F.03.bloom-cli-wins");
    }
    // F.04 --pool with worker, config.pool_worker -> CLI worker wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--pool", "jlp://x:1", "--worker", "1A_cli"}, a, c, err);
        AppConfig cfg; cfg.pool_worker = "1Z_cfg";
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.pool_worker, std::string("1A_cli"), "F.04.worker-cli-wins");
    }
    // F.05 --pool-password CLI vs config -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--pool", "jlp://x:1", "--pool-password", "cli_pw"}, a, c, err);
        AppConfig cfg; cfg.pool_password.assign("cfg_pw", 6);
        apply_config_to_args(a, cfg, c);
        // pool_password is a SecureString (move-only, no operator==);
        // compare by extracting the bytes into a transient std::string.
        EXPECT_EQ(std::string(a.pool_password.data(), a.pool_password.size()),
                  std::string("cli_pw"), "F.05.password-cli-wins");
    }
    // F.06 --pool-api-key CLI vs config -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--pool", "jlp://x:1", "--pool-api-key", "cli_key"}, a, c, err);
        AppConfig cfg; cfg.pool_api_key = "cfg_key";
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.pool_api_key, std::string("cli_key"), "F.06.apikey-cli-wins");
    }
    // F.07 --kangaroo vs config.kangaroo=false -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--kangaroo"}, a, c, err);
        AppConfig cfg; cfg.kangaroo = false;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.puzzle_kangaroo, "F.07.kangaroo-cli-overrides-cfg-false");
    }
    // F.08 No --kangaroo CLI, config kangaroo=false -> config wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.kangaroo = false;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(!a.puzzle_kangaroo, "F.08.kangaroo-cfg-false-applies");
    }
    // F.09 --auto-next vs config.auto_next=false -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--auto-next"}, a, c, err);
        AppConfig cfg; cfg.auto_next = false;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.puzzle_auto_next, "F.09.auto-next-cli-wins");
    }
    // F.10 --resume CLI, config.resume=false -> resume on
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--resume"}, a, c, err);
        AppConfig cfg; cfg.resume = false;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.resume, "F.10.resume-cli");
    }
    // F.11 No CLI resume, config.resume=true -> resume on
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.resume = true;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.resume, "F.11.resume-cfg");
    }
    // F.12 --dp-bits 20 vs config.dp_bits=24 -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--dp-bits", "20"}, a, c, err);
        AppConfig cfg; cfg.dp_bits = 24;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.dp_bits, 20, "F.12.dpbits-cli-wins");
    }
    // F.13 No CLI dp-bits, config 24 -> config wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.dp_bits = 24;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.dp_bits, 24, "F.13.dpbits-cfg-applies");
    }
    // F.14 --puzzle-checkpoint vs config.checkpoint -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--puzzle-checkpoint", "cli.ckpt"}, a, c, err);
        AppConfig cfg; cfg.checkpoint = "cfg.ckpt";
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.puzzle_checkpoint, std::string("cli.ckpt"), "F.14.ckpt-cli-wins");
    }
    // F.15 --no-smart vs config.smart_select=true -> CLI wins (false)
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--no-smart"}, a, c, err);
        AppConfig cfg; cfg.smart_select = true;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(!a.smart_select, "F.15.no-smart-cli-wins");
    }
    // F.16 --verbose, config.verbose=false -> CLI verbose wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--verbose"}, a, c, err);
        AppConfig cfg; cfg.verbose = false;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.verbose, "F.16.verbose-cli");
    }
    // F.17 No --verbose CLI, config.verbose=true -> verbose on
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.verbose = true;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.verbose, "F.17.verbose-cfg-applies");
    }
    // F.18 --debug, config.debug=false -> debug on
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--debug"}, a, c, err);
        AppConfig cfg; cfg.debug = false;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.debug, "F.18.debug-cli");
    }
    // F.18a (2026-05-16) --perf-instrument opt-in. The brain-wallet runner
    // gates perf::set_enabled(true) on args.perf_instrument so the
    // per-kernel cudaEvent collector stays off by default; without the
    // flag, production multi-GPU scans avoid the cross-device event-ring
    // edge case discovered during the GPU-cascade investigation.
    // No-flag default: false. With flag: true. No config.yml override
    // path (the flag is operator-only; we don't want it pinned by a
    // shared config file).
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        EXPECT_TRUE(!a.perf_instrument, "F.18a.perf-instrument-default-off");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--perf-instrument"}, a, c, err);
        EXPECT_TRUE(a.perf_instrument, "F.18a.perf-instrument-cli");
    }
    // F.19 --force-calibrate, config.force_calibrate=false -> on
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--force-calibrate"}, a, c, err);
        AppConfig cfg; cfg.force_calibrate = false;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.force_calibrate, "F.19.force-calibrate-cli");
        EXPECT_TRUE(a.calibrate, "F.19.calibrate-implied");
    }
    // F.20 No CLI calibrate, config.force_calibrate=true -> on
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.force_calibrate = true;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.force_calibrate, "F.20.force-calibrate-cfg");
        EXPECT_TRUE(a.calibrate, "F.20.calibrate-implied-cfg");
    }
    // F.21 --pool jlp:// CLI, config.pool_url different -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--pool", "jlp://cli:1"}, a, c, err);
        AppConfig cfg; cfg.pool_url = "jlp://cfg:9";
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.pool_url, std::string("jlp://cli:1"), "F.21.pool-url-cli-wins");
        EXPECT_TRUE(a.pool_mode, "F.21.pool-mode-on");
    }
    // F.22 No --pool, no config.pool_url -> pool stays off
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg;  // empty
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(!a.pool_mode, "F.22.pool-off");
        EXPECT_TRUE(a.pool_url.empty(), "F.22.pool-url-empty");
    }
    // F.23 --benchmark-time 60 vs config.benchmark_seconds=30 -> CLI 60 wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--benchmark-time", "60"}, a, c, err);
        AppConfig cfg; cfg.benchmark_seconds = 30;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.benchmark_seconds, 60, "F.23.benchmark-time-cli-wins");
    }
    // F.24 No CLI, config.benchmark_seconds=120 -> 120 applied
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.benchmark_seconds = 120;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.benchmark_seconds, 120, "F.24.benchmark-time-cfg-applies");
    }
    // F.25 No CLI, config.benchmark_seconds=30 (default) -> stays 30
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg;  // benchmark_seconds defaults to 30
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.benchmark_seconds, 30, "F.25.benchmark-time-default");
    }
    // F.26 --batch-size 0 (intentional zero, e.g. test) vs config 8M -> CLI 0 wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--batch-size", "0"}, a, c, err);
        AppConfig cfg; cfg.batch_size = 8'000'000;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.batch_size, (size_t)0, "F.26.batch-size-zero-cli-wins");
    }
    // F.27 No --pool-password, config.pool_password set -> config applied
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.pool_password.assign("cfg_pw", 6);
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(std::string(a.pool_password.data(), a.pool_password.size()),
                  std::string("cfg_pw"), "F.27.password-cfg-applies");
    }
    // F.28 No --bloom, config bloom -> applied
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg; cfg.bloom_file = "cfg.blf";
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.bloom_file, std::string("cfg.blf"), "F.28.bloom-cfg-applies");
    }
    // F.29 --auto-next vs config.auto_next=true -> stays on (no flip-down)
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--auto-next"}, a, c, err);
        AppConfig cfg; cfg.auto_next = true;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.puzzle_auto_next, "F.29.auto-next-both-on");
    }
    // F.30 No CLI auto-next, config.auto_next=false (default) -> off
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({}, a, c, err);
        AppConfig cfg;  // auto_next defaults to false
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(!a.puzzle_auto_next, "F.30.auto-next-default-off");
    }
    // F.31 --puzzle-min-bits 0 explicit vs config 50 -> CLI 0 wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--puzzle-min-bits", "0"}, a, c, err);
        AppConfig cfg; cfg.min_bits = 50;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.puzzle_min_bits, 0, "F.31.min-bits-explicit-zero-wins");
    }
    // F.32 --puzzle-max-bits 160 explicit (== default) vs config 120 -> CLI wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--puzzle-max-bits", "160"}, a, c, err);
        AppConfig cfg; cfg.max_bits = 120;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.puzzle_max_bits, 160, "F.32.max-bits-explicit-default-wins");
    }
    // F.33 --dp-bits -1 explicit (== default) vs config 24 -> CLI -1 wins
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--dp-bits", "-1"}, a, c, err);
        AppConfig cfg; cfg.dp_bits = 24;
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.dp_bits, -1, "F.33.dpbits-explicit-default-wins");
    }

    // -----------------------------------------------------------------------
    // Group G: TUI flag pair + tui.enabled config override.
    //
    // The runner's render path selects between the new multi-panel TUI and
    // the v1.4.2 flat-line status block. CLI --no-tui / --tui pin the choice;
    // tui.enabled in config.yml provides the same pin with lower precedence;
    // when neither is set the runner falls through to TTY auto-detect. Each
    // row below proves one cell of that resolution matrix.
    // -----------------------------------------------------------------------

    // G.01 --no-tui alone -> no_tui=true and user_set bit pinned
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--brainwallet", "--no-tui"}, a, c, err), 0,
                  "G.01.parse-ok");
        EXPECT_TRUE(a.no_tui, "G.01.no-tui-true");
        EXPECT_TRUE(a.no_tui_user_set, "G.01.user-set-true");
    }

    // G.02 --tui alone -> no_tui=false and user_set bit pinned
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--brainwallet", "--tui"}, a, c, err), 0,
                  "G.02.parse-ok");
        EXPECT_TRUE(!a.no_tui, "G.02.no-tui-false");
        EXPECT_TRUE(a.no_tui_user_set, "G.02.user-set-true");
    }

    // G.03 No CLI flag -> user_set stays false, runner falls through to
    // its own isatty auto-detect. Default no_tui value is false.
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--brainwallet"}, a, c, err), 0,
                  "G.03.parse-ok");
        EXPECT_TRUE(!a.no_tui_user_set, "G.03.user-set-default-false");
        EXPECT_TRUE(!a.no_tui, "G.03.no-tui-default-false");
    }

    // G.04 No CLI flag, config.tui_enabled=0 -> no_tui=true via config and
    // user_set pinned so the runner skips TTY auto-detect.
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--brainwallet"}, a, c, err);
        AppConfig cfg; cfg.tui_enabled = 0;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.no_tui, "G.04.cfg-disables-tui");
        EXPECT_TRUE(a.no_tui_user_set, "G.04.user-set-via-cfg");
    }

    // G.05 No CLI flag, config.tui_enabled=1 -> no_tui=false explicitly.
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--brainwallet"}, a, c, err);
        AppConfig cfg; cfg.tui_enabled = 1;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(!a.no_tui, "G.05.cfg-enables-tui");
        EXPECT_TRUE(a.no_tui_user_set, "G.05.user-set-via-cfg");
    }

    // G.06 CLI --tui overrides config.tui_enabled=0 (CLI always wins).
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--brainwallet", "--tui"}, a, c, err);
        AppConfig cfg; cfg.tui_enabled = 0;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(!a.no_tui, "G.06.cli-tui-beats-cfg-disable");
        EXPECT_TRUE(a.no_tui_user_set, "G.06.user-set-true");
    }

    // G.07 CLI --no-tui overrides config.tui_enabled=1 (CLI always wins).
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--brainwallet", "--no-tui"}, a, c, err);
        AppConfig cfg; cfg.tui_enabled = 1;
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(a.no_tui, "G.07.cli-no-tui-beats-cfg-enable");
        EXPECT_TRUE(a.no_tui_user_set, "G.07.user-set-true");
    }

    // G.08 No CLI flag, config.tui_enabled=-1 (default) -> user_set stays
    // false so the runner's isatty branch runs.
    {
        Arguments a; CLIFlags c; std::string err;
        parse_args_mirror({"--brainwallet"}, a, c, err);
        AppConfig cfg;  // tui_enabled defaults to -1
        apply_config_to_args(a, cfg, c);
        EXPECT_TRUE(!a.no_tui_user_set, "G.08.cfg-unset-leaves-user-set-false");
    }

    // -----------------------------------------------------------------------
    // Group H: Real-parser-only coverage added with the T2 wiring fix.
    //
    // The previous in-test mirror was missing rows for several flags that
    // the production parser supports. Now that the test links the real
    // parser, those flags fall under test for the first time. Each row
    // below documents which behavior the registry refactor (Q9) must
    // continue to preserve.
    // -----------------------------------------------------------------------

    // H.01 --pool-password-file: file source wins over plain --pool-password
    // ordering. Test writes a temp file, then asserts the password and the
    // path are both populated and the pool_password_file_set bit is on.
    {
        // Write a temp file with the secret on the first line plus trailing
        // whitespace that the production parser must strip.
        char tmp_path[L_tmpnam];
        std::tmpnam(tmp_path);
        {
            std::ofstream f(tmp_path);
            f << "secret_from_file  \r\n";  // CR + trailing spaces
        }
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--pool-password-file", tmp_path}, a, c, err);
        EXPECT_EQ(rc, 0, "H.01.parse-ok");
        EXPECT_EQ(std::string(a.pool_password.data(), a.pool_password.size()),
                  std::string("secret_from_file"),
                  "H.01.password-read-and-trimmed");
        EXPECT_EQ(a.pool_password_file, std::string(tmp_path),
                  "H.01.path-recorded");
        EXPECT_TRUE(c.pool_password_file_set, "H.01.file-bit-set");
        EXPECT_TRUE(c.pool_password_set,
                    "H.01.password-bit-set-too-so-yaml-cannot-clobber");
        std::remove(tmp_path);
    }

    // H.02 --pool-password-file missing path -> -1 with err_msg populated.
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror(
            {"--pool-password-file", "/nonexistent/path/should/not/exist.txt"},
            a, c, err);
        EXPECT_EQ(rc, -1, "H.02.missing-file-rejected");
        EXPECT_TRUE(!err.empty(), "H.02.err-msg-set");
    }

    // H.03 --brute valid lengths -> brainwallet_mode flips on and lengths land
    // in args.brute_lengths in declared order.
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--brute", "5", "6", "7"}, a, c, err);
        EXPECT_EQ(rc, 0, "H.03.parse-ok");
        EXPECT_TRUE(a.brainwallet_mode, "H.03.brainwallet-implied");
        EXPECT_EQ(a.brute_lengths.size(), (size_t)3, "H.03.three-lengths");
        EXPECT_EQ(a.brute_lengths[0], 5, "H.03.first");
        EXPECT_EQ(a.brute_lengths[1], 6, "H.03.second");
        EXPECT_EQ(a.brute_lengths[2], 7, "H.03.third");
    }

    // H.04 --brute with out-of-range length is rejected.
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--brute", "17"}, a, c, err);
        EXPECT_EQ(rc, -1, "H.04.length-17-rejected");
        EXPECT_TRUE(!err.empty(), "H.04.err-msg-set");
    }

    // H.05 --brute with zero positive args is rejected.
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--brute"}, a, c, err);
        EXPECT_EQ(rc, -1, "H.05.no-length-rejected");
    }

    // H.06 --brute + --brainwallet-v2 mutex (declared at parse time).
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--brainwallet-v2", "--brute", "6"}, a, c, err);
        EXPECT_EQ(rc, -1, "H.06.brute-vs-v2-rejected");
    }

    // H.07 --brute + --brainwallet-warpwallet mutex.
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror(
            {"--brainwallet-warpwallet", "user@example.com", "--brute", "6"},
            a, c, err);
        EXPECT_EQ(rc, -1, "H.07.brute-vs-warpwallet-rejected");
    }

    // H.08 --pubkey: sets puzzle_pubkey AND its CLIFlags bit.
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror(
            {"--pubkey",
             "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"},
            a, c, err);
        EXPECT_EQ(rc, 0, "H.08.parse-ok");
        EXPECT_TRUE(c.puzzle_pubkey_set, "H.08.bit-set");
        EXPECT_EQ(a.puzzle_pubkey.size(), (size_t)66, "H.08.value-len");
    }

    // H.09 Unknown flag is silently skipped (preserves legacy behavior).
    {
        Arguments a; CLIFlags c; std::string err;
        int rc = parse_args_mirror({"--this-flag-does-not-exist", "--puzzle", "71"},
                                   a, c, err);
        EXPECT_EQ(rc, 0, "H.09.unknown-flag-ignored");
        EXPECT_EQ(a.puzzle_number, 71, "H.09.subsequent-flag-still-parses");
    }

    // H.10 Short alias coverage: -P, -g, -v, -h, -p, -w, -c.
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"-P", "71"}, a, c, err), 0, "H.10a.parse-ok");
        EXPECT_EQ(a.puzzle_number, 71, "H.10a.puzzle-via-short");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"-g", "0,2"}, a, c, err), 0, "H.10b.parse-ok");
        EXPECT_EQ(a.gpu_ids.size(), (size_t)2, "H.10b.gpu-via-short");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"-v"}, a, c, err), 0, "H.10c.parse-ok");
        EXPECT_TRUE(a.verbose, "H.10c.verbose-via-short");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"-h"}, a, c, err), 0, "H.10d.parse-ok");
        EXPECT_TRUE(a.help, "H.10d.help-via-short");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"-p", "jlp://h:1", "-w", "1WORKER"}, a, c, err),
                  0, "H.10e.parse-ok");
        EXPECT_TRUE(a.pool_mode, "H.10e.pool-via-short");
        EXPECT_EQ(a.pool_worker, std::string("1WORKER"),
                  "H.10e.worker-via-short");
    }
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"-c", "alt.yml"}, a, c, err), 0,
                  "H.10f.parse-ok");
        EXPECT_EQ(a.config_file, std::string("alt.yml"),
                  "H.10f.config-via-short");
    }

    // --- M8: --worker-unsafe-allow-any must be a real top-level flag, not a
    //     global argv sweep. ---

    // An invalid worker name (contains '!') is rejected at parse time.
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--pool", "jlp://h:1", "--worker", "bad!name"},
                                    a, c, err),
                  -1, "M8.01.bad-worker-rejected");
    }
    // The escape hatch, passed legitimately as a top-level flag, allows it.
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--pool", "jlp://h:1", "--worker", "bad!name",
                                     "--worker-unsafe-allow-any"},
                                    a, c, err),
                  0, "M8.02.escape-hatch-allows");
        EXPECT_TRUE(a.worker_unsafe_allow_any, "M8.02.flag-set");
        EXPECT_EQ(a.pool_worker, std::string("bad!name"), "M8.02.worker-kept");
    }
    // Order independence: the escape hatch works even when it precedes --worker.
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--worker-unsafe-allow-any", "--pool",
                                     "jlp://h:1", "--worker", "bad!name"},
                                    a, c, err),
                  0, "M8.03.escape-hatch-before-worker");
    }
    // The string appearing only as ANOTHER flag's VALUE must NOT disable the
    // validator. Here it is smuggled in as the --pool URL value, so it is
    // consumed by --pool and never seen as a top-level flag. With an invalid
    // --worker name and no genuine escape-hatch flag, parsing is rejected.
    // (Under the old global-argv-sweep this token would have flipped the
    // unsafe bit and silently accepted the bad worker name.)
    {
        Arguments a; CLIFlags c; std::string err;
        EXPECT_EQ(parse_args_mirror({"--pool", "--worker-unsafe-allow-any",
                                     "--worker", "bad!name"},
                                    a, c, err),
                  -1, "M8.04.value-smuggle-rejected");
        EXPECT_TRUE(!a.worker_unsafe_allow_any, "M8.04.flag-not-set");
    }
}

}  // namespace

int main() {
    std::cout << "[*] Running CLI parser + config override tests\n";
    run_all();
    std::cout << "\n=== test_cli_parser results ===\n";
    std::cout << "Passed: " << g_stats.passed << "\n";
    std::cout << "Failed: " << g_stats.failed << "\n";
    if (g_stats.failed > 0) {
        std::cout << "\nFailures:\n";
        for (auto& f : g_stats.failures) std::cout << "  - " << f << "\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
