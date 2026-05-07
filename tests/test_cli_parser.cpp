/**
 * test_cli_parser.cpp - Wave 5 (track-e) CLI parser + config override matrix tests.
 *
 * Validates the CLIFlags-aware override behaviour in apply_config_to_args() and
 * the mode-mutex check in parse_args(). Each test row exercises one cell of the
 * (CLI flag x config field) matrix from docs/review-2026-05-04/track-e-cli-config.md.
 *
 * Note: parse_args() lives in src/main.cpp and would pull in the entire main
 * binary (with detect_gpus, run_pool_mode, etc). To keep the test target self
 * contained we replicate a minimal Arguments + parse_args_for_test that mirrors
 * the production parser. The two parsers MUST stay in sync; if you add a flag
 * to main.cpp::parse_args_core, mirror it here.
 *
 * The override-matrix half of these tests calls collider::apply_config_to_args
 * directly (template, header-only) -- no main.cpp link required.
 */

#include "../src/core/yaml_config.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Local Arguments mirror -- must match src/main.cpp::Arguments fields used by
// apply_config_to_args(). We keep ONLY the fields the override path touches.
// ---------------------------------------------------------------------------
struct Arguments {
    std::vector<int> gpu_ids = {};
    size_t batch_size = 4'000'000;
    bool verbose = false;
    bool help = false;

    bool benchmark = false;
    int benchmark_seconds = 30;

    bool puzzle_mode = true;
    int puzzle_number = 0;
    std::string puzzle_target;
    std::string puzzle_range_start;
    std::string puzzle_range_end;
    bool puzzle_random = true;
    std::string puzzle_checkpoint;
    bool puzzle_auto_next = false;
    bool puzzle_all_unsolved = false;
    int puzzle_min_bits = 0;
    int puzzle_max_bits = 160;
    bool puzzle_kangaroo = false;
    bool use_rckangaroo = true;
    int dp_bits = -1;
    std::string bloom_file;

    bool brainwallet_mode = false;
    bool brainwallet_setup = false;
    std::string wordlist_file;
    bool resume = false;
    size_t save_interval = 1000000;
    bool cpu_rules = false;

    bool calibrate = false;
    bool force_calibrate = false;

    bool analyze_puzzles = false;
    bool smart_select = true;

    bool debug = false;

    bool pool_mode = false;
    std::string pool_url;
    std::string pool_worker;
    std::string pool_password;
    std::string pool_api_key;

    std::string config_file;
};

// ---------------------------------------------------------------------------
// validate_mode_mutex -- must mirror main.cpp::validate_mode_mutex exactly.
// ---------------------------------------------------------------------------
int validate_mode_mutex(const Arguments& args, std::string& msg) {
    int active = 0;
    std::vector<std::string> chosen;
    if (args.brainwallet_mode) { active++; chosen.emplace_back("--brainwallet"); }
    if (args.pool_mode)        { active++; chosen.emplace_back("--pool <url>"); }
    if (args.puzzle_kangaroo && (args.brainwallet_mode || args.pool_mode)) {
        active++;
        chosen.emplace_back("--kangaroo");
    }
    if ((args.puzzle_number > 0 || args.puzzle_all_unsolved) &&
        (args.brainwallet_mode || args.pool_mode)) {
        active++;
        if (args.puzzle_all_unsolved) chosen.emplace_back("--all-unsolved");
        else chosen.emplace_back("--puzzle " + std::to_string(args.puzzle_number));
    }
    if (active <= 1) return 0;
    msg = "[!] Conflicting search modes: ";
    for (size_t i = 0; i < chosen.size(); ++i) {
        msg += chosen[i];
        if (i + 1 < chosen.size()) msg += ", ";
    }
    return -1;
}

// ---------------------------------------------------------------------------
// parse_args mirror (no exit, no stderr). Must stay in sync with
// src/main.cpp::parse_args_core. Tests pass argv arrays through this.
// ---------------------------------------------------------------------------
int parse_args_mirror(const std::vector<std::string>& argv,
                      Arguments& args,
                      collider::CLIFlags& cli,
                      std::string& err_msg) {
    args = Arguments{};
    cli = collider::CLIFlags{};
    int argc = static_cast<int>(argv.size());

    // Build argv[] with a fake program name at slot 0 (parse_args skips i=0).
    std::vector<std::string> storage;
    storage.reserve(argc + 1);
    storage.emplace_back("test_collider");
    for (auto& s : argv) storage.push_back(s);

    auto get = [&](int i) -> const char* { return storage[i].c_str(); };

    int n = static_cast<int>(storage.size());
    for (int i = 1; i < n; i++) {
        std::string arg = get(i);

        if (arg == "--help" || arg == "-h") {
            args.help = true;
        } else if (arg == "--verbose" || arg == "-v") {
            args.verbose = true; cli.verbose_set = true;
        } else if ((arg == "--gpus" || arg == "-g") && i + 1 < n) {
            args.gpu_ids.clear();
            std::string gpus = get(++i);
            size_t pos = 0;
            while ((pos = gpus.find(',')) != std::string::npos) {
                args.gpu_ids.push_back(std::stoi(gpus.substr(0, pos)));
                gpus.erase(0, pos + 1);
            }
            args.gpu_ids.push_back(std::stoi(gpus));
            cli.gpu_ids_set = true;
        } else if (arg == "--batch-size" && i + 1 < n) {
            args.batch_size = std::stoull(get(++i)); cli.batch_size_set = true;
        } else if (arg == "--benchmark") {
            args.benchmark = true;
        } else if (arg == "--benchmark-time" && i + 1 < n) {
            args.benchmark_seconds = std::stoi(get(++i)); cli.benchmark_seconds_set = true;
        } else if (arg == "--puzzle" || arg == "-P") {
            args.puzzle_mode = true;
            if (i + 1 < n && storage[i + 1][0] != '-') {
                args.puzzle_number = std::stoi(get(++i)); cli.puzzle_number_set = true;
            }
        } else if (arg == "--puzzle-target" && i + 1 < n) {
            args.puzzle_target = get(++i);
        } else if (arg == "--puzzle-start" && i + 1 < n) {
            args.puzzle_range_start = get(++i);
        } else if (arg == "--puzzle-end" && i + 1 < n) {
            args.puzzle_range_end = get(++i);
        } else if (arg == "--sequential") {
            args.puzzle_random = false; cli.puzzle_random_set = true;
        } else if (arg == "--random") {
            args.puzzle_random = true;  cli.puzzle_random_set = true;
        } else if (arg == "--puzzle-checkpoint" && i + 1 < n) {
            args.puzzle_checkpoint = get(++i); cli.puzzle_checkpoint_set = true;
        } else if (arg == "--auto-next") {
            args.puzzle_auto_next = true; cli.puzzle_auto_next_set = true;
        } else if (arg == "--all-unsolved") {
            args.puzzle_all_unsolved = true;
        } else if (arg == "--puzzle-min-bits" && i + 1 < n) {
            args.puzzle_min_bits = std::stoi(get(++i)); cli.puzzle_min_bits_set = true;
        } else if (arg == "--puzzle-max-bits" && i + 1 < n) {
            args.puzzle_max_bits = std::stoi(get(++i)); cli.puzzle_max_bits_set = true;
        } else if (arg == "--kangaroo") {
            args.puzzle_kangaroo = true; cli.puzzle_kangaroo_set = true;
        } else if (arg == "--dp-bits" && i + 1 < n) {
            args.dp_bits = std::stoi(get(++i)); cli.dp_bits_set = true;
        } else if (arg == "--bloom" && i + 1 < n) {
            args.bloom_file = get(++i); cli.bloom_file_set = true;
        } else if (arg == "--brainwallet") {
            args.brainwallet_mode = true;
            args.pool_mode = false;
            args.pool_url.clear();
            cli.brainwallet_set = true;
        } else if (arg == "--brainwallet-setup") {
            args.brainwallet_setup = true;
        } else if (arg == "--resume") {
            args.resume = true; cli.resume_set = true;
        } else if (arg == "--cpu-rules") {
            args.cpu_rules = true;
        } else if (arg == "--save-interval" && i + 1 < n) {
            args.save_interval = std::stoull(get(++i)); cli.save_interval_set = true;
        } else if (arg == "--calibrate") {
            args.calibrate = true;
        } else if (arg == "--force-calibrate") {
            args.calibrate = true; args.force_calibrate = true; cli.force_calibrate_set = true;
        } else if (arg == "--debug") {
            args.debug = true; cli.debug_set = true;
        } else if (arg == "--analyze") {
            args.analyze_puzzles = true;
        } else if (arg == "--no-smart") {
            args.smart_select = false; cli.smart_select_set = true;
        } else if ((arg == "--pool" || arg == "-p") && i + 1 < n) {
            args.pool_mode = true; args.pool_url = get(++i); cli.pool_url_set = true;
        } else if ((arg == "--worker" || arg == "-w") && i + 1 < n) {
            args.pool_worker = get(++i); cli.pool_worker_set = true;
        } else if (arg == "--pool-password" && i + 1 < n) {
            args.pool_password = get(++i); cli.pool_password_set = true;
        } else if (arg == "--pool-api-key" && i + 1 < n) {
            args.pool_api_key = get(++i); cli.pool_api_key_set = true;
        } else if ((arg == "--config" || arg == "-c") && i + 1 < n) {
            args.config_file = get(++i);
        }
    }

    return validate_mode_mutex(args, err_msg);
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
    // Group A: parse_args alone -- per-flag CLIFlags bit accuracy
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
        // Order test: --pool then --brainwallet -- brainwallet always wins inside
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
        // pool_url WILL get populated because nothing CLI-set the bit -- but
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
        AppConfig cfg; cfg.pool_password = "cfg_pw";
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.pool_password, std::string("cli_pw"), "F.05.password-cli-wins");
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
        AppConfig cfg; cfg.pool_password = "cfg_pw";
        apply_config_to_args(a, cfg, c);
        EXPECT_EQ(a.pool_password, std::string("cfg_pw"), "F.27.password-cfg-applies");
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
}

}  // namespace

int main() {
    std::cout << "[*] Running CLI parser + config override tests (Wave 5 / track-e)\n";
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
