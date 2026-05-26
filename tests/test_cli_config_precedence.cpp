// test_cli_config_precedence.cpp -- TP-9.
//
// The "CLI > config.yml" precedence rule (documented in main.cpp:211-212
// and in the FlagSpec design comment) is the single most important
// merge invariant in the configuration layer. An operator who hits
// `--batch-size 10M` MUST see that value reach the runner regardless
// of what config.yml said. This file pins that contract via a direct
// apply_config_to_args() KAT.
//
// Implementation note: we don't parse a YAML file from disk; we
// construct the AppConfig + CLIFlags + Arguments structs directly
// to isolate the merge logic from YAML parsing concerns (which have
// their own tests in test_runtime_yaml).

#include "cli/cli_parser.hpp"
#include "core/yaml_config.hpp"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_passes   = 0;

void fail(const char* tag, const std::string& msg) {
    std::fprintf(stderr, "[FAIL] %s: %s\n", tag, msg.c_str());
    ++g_failures;
}
void pass(const char* tag) {
    std::printf("[ ok  ] %s\n", tag);
    ++g_passes;
}

}  // namespace

int main() {
    using ::collider::AppConfig;
    using ::collider::CLIFlags;
    using ::collider::apply_config_to_args;
    std::printf("=== test_cli_config_precedence (TP-9) ===\n");

    // -------------------------------------------------------------------
    // Case 1: CLI set, config also set -> CLI wins.
    // -------------------------------------------------------------------
    {
        Arguments args;
        args.batch_size = 10'000'000;   // operator typed --batch-size 10M
        AppConfig cfg;
        cfg.batch_size = 5'000'000;     // config.yml said 5M
        CLIFlags cli;
        cli.batch_size_set = true;      // CLI parser noted operator override

        apply_config_to_args(args, cfg, cli);
        if (args.batch_size != 10'000'000) {
            fail("cli_overrides_config/batch_size",
                 "expected 10M, got " + std::to_string(args.batch_size));
        } else {
            pass("cli_overrides_config/batch_size");
        }
    }

    // -------------------------------------------------------------------
    // Case 2: CLI NOT set, config set -> config value flows through.
    // -------------------------------------------------------------------
    {
        Arguments args;
        args.batch_size = 4'000'000;     // Arguments default
        AppConfig cfg;
        cfg.batch_size = 8'000'000;      // config.yml says 8M
        CLIFlags cli;                    // batch_size_set stays false

        apply_config_to_args(args, cfg, cli);
        if (args.batch_size != 8'000'000) {
            fail("config_fills_when_cli_silent/batch_size",
                 "expected 8M, got " + std::to_string(args.batch_size));
        } else {
            pass("config_fills_when_cli_silent/batch_size");
        }
    }

    // -------------------------------------------------------------------
    // Case 3: GPU IDs -- vector field; CLI wins.
    // (AppConfig field is `gpu_devices`; merge target on args is gpu_ids.)
    // -------------------------------------------------------------------
    {
        Arguments args;
        args.gpu_ids = {0, 2};            // operator typed --gpus 0,2
        AppConfig cfg;
        cfg.gpu_devices = {1, 3, 5};      // config wants different GPUs
        CLIFlags cli;
        cli.gpu_ids_set = true;

        apply_config_to_args(args, cfg, cli);
        if (args.gpu_ids != std::vector<int>{0, 2}) {
            fail("cli_overrides_config/gpu_ids",
                 "vector did not preserve CLI values");
        } else {
            pass("cli_overrides_config/gpu_ids");
        }
    }

    // -------------------------------------------------------------------
    // Case 4: GPU IDs -- config fills when CLI silent.
    // -------------------------------------------------------------------
    {
        Arguments args;  // gpu_ids defaults to empty
        AppConfig cfg;
        cfg.gpu_devices = {1, 3, 5};
        CLIFlags cli;    // gpu_ids_set stays false

        apply_config_to_args(args, cfg, cli);
        if (args.gpu_ids != std::vector<int>{1, 3, 5}) {
            fail("config_fills_when_cli_silent/gpu_ids",
                 "config gpu_devices did not flow through to args.gpu_ids");
        } else {
            pass("config_fills_when_cli_silent/gpu_ids");
        }
    }

    // -------------------------------------------------------------------
    // Case 6: pool URL -- CLI wins; auto-enable pool_mode side effect
    // does NOT fire when CLI URL was given (since pool_url_set is true).
    // -------------------------------------------------------------------
    {
        Arguments args;
        args.pool_url = "jlp://operator.example:8333";  // CLI
        args.pool_mode = false;
        AppConfig cfg;
        cfg.pool_url = "jlp://config.example:9333";     // config
        CLIFlags cli;
        cli.pool_url_set = true;

        apply_config_to_args(args, cfg, cli);
        if (args.pool_url != "jlp://operator.example:8333") {
            fail("cli_overrides_config/pool_url",
                 "expected operator URL, got '" + args.pool_url + "'");
        } else {
            pass("cli_overrides_config/pool_url");
        }
    }

    // -------------------------------------------------------------------
    // Case 7: config provides pool URL when CLI is silent; pool_mode
    // auto-enable kicks in (the documented side-effect at
    // yaml_config.hpp:494).
    // -------------------------------------------------------------------
    {
        Arguments args;
        args.pool_url.clear();
        args.pool_mode = false;
        args.brainwallet_mode = false;
        AppConfig cfg;
        cfg.pool_url = "jlp://config.example:9333";
        CLIFlags cli;  // pool_url_set stays false

        apply_config_to_args(args, cfg, cli);
        if (args.pool_url != "jlp://config.example:9333") {
            fail("config_fills_pool_url",
                 "expected config URL, got '" + args.pool_url + "'");
        } else {
            pass("config_fills_pool_url");
        }
        if (!args.pool_mode) {
            fail("config_auto_enables_pool_mode",
                 "pool_mode should auto-enable when config supplies URL "
                 "and brainwallet is not requested");
        } else {
            pass("config_auto_enables_pool_mode");
        }
    }

    std::printf("\n%d passes, %d failures\n", g_passes, g_failures);
    return g_failures == 0 ? 0 : 1;
}
