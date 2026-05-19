/**
 * test_generator_modes -- v1.4.2 A-tier coverage for the brain-wallet
 * iteration modes that were previously stubbed out.
 *
 * The state machine inside StreamingBrainWallet can traverse several
 * IterationMode transitions within a single next_batch() call (an iteration
 * with no productive candidates is burned through quickly). So this test
 * does NOT assert "mode X is observed at batch N then mode Y at batch M";
 * it asserts the stronger but more pragmatic claim:
 *
 *   1. Across a long enough run, every Phase-4 mode is OBSERVED at least
 *      once via Stats::iteration_mode. This locks down the Phase 4 wiring
 *      against accidental reversion to PHASE_CYCLING-as-fallback.
 *
 *   2. MARKOV, PCFG, and KEYBOARD_WALK each contribute non-zero candidate
 *      production in at least one batch where they are the active mode.
 *      A regression that wires PCFG to a null Generator would still
 *      pass test 1 but fail test 2.
 *
 *   3. The total candidate count is non-zero -- the generator never
 *      deadlocks.
 */

#include "src/generators/streaming_brain_wallet.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>

namespace bw = ::collider::generators;

namespace {

void write_tiny_wordlist(const std::string& path) {
    std::ofstream f(path);
    // 32 short common English words. Long enough for the Markov trainer
    // and PCFG grammar to build meaningful state, short enough that the
    // test runs in well under a second.
    const char* words[] = {
        "satoshi", "bitcoin", "wallet", "password", "secret", "hello",
        "world", "private", "public", "address", "puzzle", "treasure",
        "hunter", "scanner", "kangaroo", "blockchain", "crypto", "ledger",
        "nakamoto", "consensus", "transaction", "mining", "mempool",
        "halving", "difficulty", "merkle", "genesis", "signature",
        "trapdoor", "elliptic", "modulus", "prime",
    };
    for (const char* w : words) f << w << '\n';
}

int fail(const char* msg, int code) {
    std::fprintf(stderr, "[FAIL] %s\n", msg);
    return code;
}

const char* mode_str(bw::IterationMode m) {
    switch (m) {
        case bw::IterationMode::PHASE_CYCLING:  return "PHASE_CYCLING";
        case bw::IterationMode::RULE_STACKING:  return "RULE_STACKING";
        case bw::IterationMode::HYBRID_MASK:    return "HYBRID_MASK";
        case bw::IterationMode::COMBINATOR:     return "COMBINATOR";
        case bw::IterationMode::MARKOV:         return "MARKOV";
        case bw::IterationMode::PCFG:           return "PCFG";
        case bw::IterationMode::KEYBOARD_WALK:  return "KEYBOARD_WALK";
    }
    return "?";
}

}  // namespace

int main() {
    const std::string wordlist_path = "test_generator_modes_wordlist.txt";
    write_tiny_wordlist(wordlist_path);

    bw::StreamingBrainWallet::Config config;
    config.base_wordlist = wordlist_path;
    config.batch_size = 128;            // small batches: each one is a single mode-observation point
    config.enable_dedup = false;
    config.verbose = false;
    // Tight budget so the state machine advances through every mode
    // within a few hundred batches instead of millions.
    config.generator_candidates_per_iteration = 256;

    bw::StreamingBrainWallet gen(config);
    if (!gen.init()) {
        std::remove(wordlist_path.c_str());
        return fail("StreamingBrainWallet::init() returned false", 1);
    }

    std::set<bw::IterationMode> modes_observed;
    std::set<bw::IterationMode> modes_with_production;
    size_t total_candidates = 0;
    size_t empty_batch_streak = 0;
    // v1.4.2 C.3: RULE_STACKING now actually consumes stacked rules and
    // respects the generator_candidates_per_iteration budget (256 in this
    // test), bounding each sub-iteration so MARKOV/PCFG/KEYBOARD_WALK get
    // their turn within the run.
    constexpr size_t kMaxBatches = 1200;
    constexpr size_t kMaxEmptyStreak = 64;

    for (size_t i = 0; i < kMaxBatches; i++) {
        auto batch = gen.next_batch();
        auto stats = gen.get_stats();

        modes_observed.insert(stats.iteration_mode);
        if (!batch.empty()) {
            modes_with_production.insert(stats.iteration_mode);
            empty_batch_streak = 0;
        } else {
            empty_batch_streak++;
            if (empty_batch_streak > kMaxEmptyStreak) {
                std::remove(wordlist_path.c_str());
                return fail("generator deadlocked: too many empty batches in a row",
                            2);
            }
        }
        total_candidates += batch.size();

        // Early-out: we've seen all four advanced modes producing candidates.
        if (modes_with_production.count(bw::IterationMode::RULE_STACKING) &&
            modes_with_production.count(bw::IterationMode::MARKOV) &&
            modes_with_production.count(bw::IterationMode::PCFG) &&
            modes_with_production.count(bw::IterationMode::KEYBOARD_WALK)) {
            std::printf("All advanced modes producing candidates after %zu batches\n",
                        i + 1);
            break;
        }
    }

    std::remove(wordlist_path.c_str());

    std::printf("Total candidates generated: %zu\n", total_candidates);
    std::printf("Modes observed (active at end of some batch):\n");
    for (auto m : modes_observed) std::printf("  - %s\n", mode_str(m));
    std::printf("Modes that produced candidates:\n");
    for (auto m : modes_with_production) std::printf("  - %s\n", mode_str(m));

    if (total_candidates == 0) {
        return fail("generator produced zero candidates over the entire run", 3);
    }

    // Required modes: the three v1.4.2 Phase-4 additions plus
    // RULE_STACKING (consumption path wired in v1.4.2 C.3).
    const std::set<bw::IterationMode> required = {
        bw::IterationMode::RULE_STACKING,
        bw::IterationMode::MARKOV,
        bw::IterationMode::PCFG,
        bw::IterationMode::KEYBOARD_WALK,
    };

    for (auto m : required) {
        if (!modes_observed.count(m)) {
            std::fprintf(stderr,
                         "[FAIL] never observed mode %s -- the state machine "
                         "skipped this iteration mode entirely\n",
                         mode_str(m));
            return 4;
        }
        if (!modes_with_production.count(m)) {
            std::fprintf(stderr,
                         "[FAIL] mode %s was observed but never produced any "
                         "candidates -- generator is likely null or returning "
                         "nullopt on every call\n",
                         mode_str(m));
            return 5;
        }
    }

    std::printf("PASS: all three v1.4.2 Phase-4 iteration modes are alive "
                "and producing candidates.\n");
    return 0;
}
