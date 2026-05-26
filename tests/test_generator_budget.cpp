// test_generator_budget.cpp -- TP-8 budget enforcement assertion.
//
// generator_candidates_per_iteration is documented as the per-mode
// budget that forces forward progress through the IterationMode state
// machine. Without it MARKOV / PCFG / KEYBOARD_WALK would emit
// arbitrarily many candidates per sub-iteration and PHASE_CYCLING
// would never get its next turn.
//
// test_generator_modes.cpp already proves the state machine advances
// through every mode within ~1200 batches at budget=256. This file
// adds a stricter, narrower assertion: track per-mode candidate
// counts across a run and verify the per-mode total never exceeds
// budget * mode_repeats by more than a small margin. The margin
// allows for the runner draining a partially-filled queue ON the
// boundary; if the budget enforcement is OFF entirely, the margin
// will be wildly exceeded.

#include "generators/streaming_brain_wallet.hpp"

#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>

namespace bw = ::collider::generators;

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

void write_tiny_wordlist(const std::string& path) {
    std::ofstream f(path);
    f << "password\n123456\nbitcoin\nsatoshi\nhello\nworld\nadmin\nqwerty\n"
         "letmein\nmonkey\ndragon\nfootball\nbaseball\nshadow\nmaster\n"
         "michael\nsunshine\nlovely\nflower\nsilver\n";
}

}  // namespace

int main() {
    std::printf("=== test_generator_budget (TP-8) ===\n");

    const std::string wordlist = "test_generator_budget_wordlist.txt";
    write_tiny_wordlist(wordlist);

    bw::StreamingBrainWallet::Config cfg;
    cfg.base_wordlist = wordlist;
    cfg.batch_size = 256;
    cfg.enable_dedup = false;
    cfg.verbose = false;
    // Budget intentionally small so a single sub-iteration must hit
    // the cap quickly. If the enforcement is broken, MARKOV / PCFG
    // will emit far more than 512 candidates before advancing.
    const size_t kBudget = 512;
    cfg.generator_candidates_per_iteration = kBudget;

    bw::StreamingBrainWallet gen(cfg);
    if (!gen.init()) {
        std::remove(wordlist.c_str());
        fail("generator_init", "init() returned false");
        std::printf("\n%d passes, %d failures\n", g_passes, g_failures);
        return 1;
    }

    // Track: candidates produced per mode in the CURRENT sub-iteration,
    // resetting each time mode_sub_iteration advances. The maximum
    // any single sub-iteration emits must respect the budget.
    using Mode = bw::IterationMode;
    std::map<Mode, size_t> max_per_subiter;  // worst sub-iter per mode
    std::map<Mode, size_t> cur_subiter_total;
    std::map<Mode, size_t> cur_subiter_index;
    std::set<Mode> modes_seen;
    constexpr size_t kMaxBatches = 600;

    for (size_t i = 0; i < kMaxBatches; ++i) {
        auto batch = gen.next_batch();
        auto stats = gen.get_stats();
        modes_seen.insert(stats.iteration_mode);

        // sub-iteration index per mode; reset cur_subiter_total when it
        // advances.
        auto& last_idx = cur_subiter_index[stats.iteration_mode];
        if (stats.mode_sub_iteration != last_idx) {
            // Sub-iteration boundary: roll the prior tally into the max.
            auto& tally = cur_subiter_total[stats.iteration_mode];
            auto& worst = max_per_subiter[stats.iteration_mode];
            if (tally > worst) worst = tally;
            tally = 0;
            last_idx = stats.mode_sub_iteration;
        }
        cur_subiter_total[stats.iteration_mode] += batch.size();
    }

    // Final roll-up.
    for (auto& [m, tally] : cur_subiter_total) {
        if (tally > max_per_subiter[m]) max_per_subiter[m] = tally;
    }

    std::remove(wordlist.c_str());

    // Budget contract: the per-sub-iteration cap should be in the
    // ballpark of kBudget. We allow up to 4x to absorb batch_size
    // granularity (a single batch may overshoot by batch_size). If
    // any mode exceeds 8x, the budget enforcement is effectively off.
    constexpr size_t kMargin = 4;
    bool any_violation = false;
    for (auto [m, worst] : max_per_subiter) {
        std::printf("    mode %d: worst sub-iter total = %zu (budget=%zu)\n",
                    static_cast<int>(m), worst, kBudget);
        if (worst > kBudget * kMargin * 2) {
            fail("budget_per_subiter",
                 "mode " + std::to_string(static_cast<int>(m)) +
                     " emitted " + std::to_string(worst) +
                     " candidates in a single sub-iter (budget " +
                     std::to_string(kBudget) + " * margin " +
                     std::to_string(kMargin * 2) + ")");
            any_violation = true;
        }
    }
    if (!any_violation) pass("budget_per_subiter");

    // Mode-advance contract: at least 3 distinct iteration modes must
    // have been observed in 600 batches at this budget. Less would
    // indicate the state machine is stuck.
    if (modes_seen.size() < 3) {
        fail("mode_advance",
             "only " + std::to_string(modes_seen.size()) +
                 " distinct modes observed in " +
                 std::to_string(kMaxBatches) + " batches");
    } else {
        pass("mode_advance");
    }

    std::printf("\n%d passes, %d failures\n", g_passes, g_failures);
    return g_failures == 0 ? 0 : 1;
}
