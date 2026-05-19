/**
 * test_resume_iteration_mode: Repair R1 coverage.
 *
 * Before R1, StreamingBrainWallet::restore_state() copied iteration_mode_
 * from the snapshot but never called the corresponding init_*_mode().
 * That left the mode-owned generator state (markov_matrix_,
 * pcfg_grammar_, stacked_rules_, hybrid_mask_plans_, keyboard_walk_seeds_)
 * empty after resume, so next_candidate() returned "" forever until the
 * per-iteration budget burned out and restart_with_next_iteration() fired
 * on the next phase wrap.
 *
 * This test constructs a fresh generator, then for each advanced mode:
 *   1. Builds a StateSnapshot pointing at that mode mid-iteration.
 *   2. Calls restore_state.
 *   3. Pulls one batch.
 *   4. Asserts the batch is non-empty.
 *
 * A regression that drops the init_*_mode dispatch will fail step 4 for
 * MARKOV / PCFG / RULE_STACKING / HYBRID_MASK / KEYBOARD_WALK.
 */

#include "src/generators/streaming_brain_wallet.hpp"
#include "src/core/search_state.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

namespace bw = ::collider::generators;

namespace {

void write_wordlist(const std::string& path) {
    std::ofstream f(path);
    // Same set as test_generator_modes; 32 words is enough for Markov
    // trigram fallback and PCFG structure extraction to produce output.
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

const char* mode_name(bw::IterationMode m) {
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

// Build a fresh generator + restore into `mode`, then assert non-empty
// production. Returns 0 on pass, non-zero on fail (with stderr message).
int probe_resume_into(const std::string& wordlist_path,
                     bw::IterationMode mode,
                     size_t phase_iteration_for_mode) {
    bw::StreamingBrainWallet::Config config;
    config.base_wordlist = wordlist_path;
    config.batch_size = 64;
    config.enable_dedup = false;
    config.verbose = false;
    // Large enough budget that the freshly-restored mode can produce at
    // least one batch before the per-iteration counter exhausts.
    config.generator_candidates_per_iteration = 4096;

    bw::StreamingBrainWallet gen(config);
    if (!gen.init()) {
        std::fprintf(stderr, "[%s] init failed\n", mode_name(mode));
        return 10;
    }

    bw::StreamingBrainWallet::StateSnapshot snap;
    snap.current_phase = 0;          // any valid phase index
    snap.current_word_idx = 0;
    snap.current_rule_idx = 0;
    snap.phase_iteration = phase_iteration_for_mode;
    snap.total_generated = 0;
    snap.wordlist_size = 0;          // 0 disables the size-match gate
    snap.rules_count = 0;
    snap.iteration_mode = mode;
    snap.mode_sub_iteration = 0;
    snap.stacked_rule_depth = 2;
    snap.mask_length = 1;
    snap.combinator_word2_idx = 0;
    snap.is_brute_mode = false;

    if (!gen.restore_state(snap)) {
        std::fprintf(stderr, "[%s] restore_state returned false\n",
                     mode_name(mode));
        return 11;
    }

    // Pull a few batches. Resuming into a "thin" mode (e.g. COMBINATOR
    // with a 32-word list) can produce empty intermediate batches as the
    // iterator advances; assert that at least one of the first few
    // batches produces something. A regression that nulls out the
    // mode-specific state will produce empty batches uniformly.
    constexpr size_t kProbeBatches = 8;
    size_t total = 0;
    for (size_t i = 0; i < kProbeBatches; i++) {
        auto batch = gen.next_batch();
        total += batch.size();
        if (total > 0) break;
    }

    if (total == 0) {
        std::fprintf(stderr,
                     "[%s] FAIL: produced zero candidates across the first %zu "
                     "batches after restore. The init_*_mode() dispatch in "
                     "restore_state() is broken.\n",
                     mode_name(mode), kProbeBatches);
        return 12;
    }

    std::printf("[%s] PASS: produced %zu candidates after restore\n",
                mode_name(mode), total);
    return 0;
}

}  // namespace

// Repair R2 cover: confirm the GPU-incompatible Combinator phase gets
// skipped by advance_past_gpu_incompatible_phases() and the generator
// lands on a phase whose `gpu_compatible` flag is true. Without R2 this
// helper does not exist on the generator; with R2 it exists and skips
// past phase 4 (Combinator) to phase 5 (Deep Dive).
int probe_advance_past_gpu_incompatible(const std::string& wordlist_path) {
    bw::StreamingBrainWallet::Config config;
    config.base_wordlist = wordlist_path;
    config.batch_size = 64;
    config.enable_dedup = false;
    config.verbose = false;
    config.generator_candidates_per_iteration = 4096;

    bw::StreamingBrainWallet gen(config);
    if (!gen.init()) {
        std::fprintf(stderr, "[advance_past] init failed\n");
        return 20;
    }

    // Restore into the Combinator phase (phase 3; the slot that R2
    // marks gpu_compatible=false). Iteration 0 keeps us in PHASE_CYCLING
    // so the phase-index path is what gets exercised (not the iteration-
    // mode path).
    bw::StreamingBrainWallet::StateSnapshot snap;
    snap.current_phase = 3;            // Combinator phase
    snap.current_word_idx = 0;
    snap.current_rule_idx = 0;
    snap.phase_iteration = 0;
    snap.total_generated = 0;
    snap.wordlist_size = 0;
    snap.rules_count = 0;
    snap.iteration_mode = bw::IterationMode::PHASE_CYCLING;
    snap.is_brute_mode = false;

    if (!gen.restore_state(snap)) {
        std::fprintf(stderr, "[advance_past] restore_state returned false\n");
        return 21;
    }

    if (gen.current_phase_name() != "Combinator") {
        std::fprintf(stderr,
                     "[advance_past] expected to land in Combinator after restore, "
                     "got \"%s\"\n",
                     gen.current_phase_name().c_str());
        return 22;
    }

    if (gen.current_phase_gpu_compatible()) {
        std::fprintf(stderr,
                     "[advance_past] FAIL: Combinator should be marked "
                     "gpu_compatible=false but it isn't.\n");
        return 23;
    }

    if (!gen.advance_past_gpu_incompatible_phases()) {
        std::fprintf(stderr,
                     "[advance_past] FAIL: advance returned false on a "
                     "live (non-brute) generator.\n");
        return 24;
    }

    if (!gen.current_phase_gpu_compatible()) {
        std::fprintf(stderr,
                     "[advance_past] FAIL: after advance, landed in another "
                     "gpu_compatible=false phase (\"%s\").\n",
                     gen.current_phase_name().c_str());
        return 25;
    }

    std::printf("[advance_past] PASS: skipped \"Combinator\" -> \"%s\"\n",
                gen.current_phase_name().c_str());
    return 0;
}

// v1.4.2 R-B8 coverage: PuzzleSearchState v3 -> v4 file migration.
//
// Before R-B8, the puzzle search state file format pinned the position
// to two 64-bit limbs (position_lo, position_hi). v4 added
// position_full[0..3]. Old v3 files must still load cleanly -- their
// position migrates into position_full[0..1] with [2..3] zeroed. This
// probe writes a hand-crafted v3 file on disk, loads it through the
// current SearchStateManager, and checks both the migrated position and
// the loaded_version reporting field.
int probe_v3_to_v4_migration() {
    using collider::PuzzleSearchState;
    using collider::SearchStateManager;

    // Redirect ~/.collider/state to a test-local directory. mirrors the
    // pattern in test_brute_resume_state.cpp.
#ifdef _WIN32
    const std::string tmp_home =
        (std::filesystem::temp_directory_path() /
         "collider_puzzle_state_v3_migration").string();
    _putenv_s("USERPROFILE", tmp_home.c_str());
#else
    const std::string tmp_home =
        (std::filesystem::temp_directory_path() /
         "collider_puzzle_state_v3_migration").string();
    setenv("HOME", tmp_home.c_str(), 1);
#endif
    std::filesystem::create_directories(tmp_home + "/.collider/state");

    // Build a v3-format state file by hand. The v3 checksum domain mixed
    // (puzzle_number, position_lo, position_hi, total_checked); we use
    // compute_checksum_v3 to produce a checksum the loader will accept.
    //
    // Puzzle 75 covers [2^74, 2^75); the private key fits in 75 bits, so
    // position_hi must fit in 11 bits (high_bits = ((75-1) % 64) + 1).
    // 0x500 is comfortably inside that envelope while staying recognizable.
    PuzzleSearchState v3_view;
    v3_view.puzzle_number = 75;
    v3_view.position_lo   = 0xCAFEBABEDEADBEEFULL;
    v3_view.position_hi   = 0x500ULL;
    v3_view.total_checked = 4242424242ULL;
    const uint32_t v3_checksum =
        SearchStateManager::compute_checksum_v3(v3_view);

    const std::string state_path =
        SearchStateManager::get_puzzle_state_path(v3_view.puzzle_number);
    {
        std::ofstream f(state_path, std::ios::out | std::ios::trunc);
        if (!f.is_open()) {
            std::fprintf(stderr,
                         "[migration] could not create v3 state file at %s\n",
                         state_path.c_str());
            return 30;
        }
        // Header banner advertises v3; loader picks parsed_version = 3
        // because the (newer) state_version key is absent. The legacy
        // file shape -- no position_full_* fields -- triggers the
        // migration branch.
        f << "# Collider Puzzle Search State v3\n";
        f << "# Do not modify manually - checksum protected\n\n";
        f << "puzzle_number=" << v3_view.puzzle_number << "\n";
        f << "position_lo="   << v3_view.position_lo   << "\n";
        f << "position_hi="   << v3_view.position_hi   << "\n";
        f << "total_checked=" << v3_view.total_checked << "\n";
        f << "timestamp=2026-05-15 00:00:00\n";
        f << "checksum="      << v3_checksum           << "\n";
    }

    // Load through the manager and validate the migration.
    PuzzleSearchState loaded =
        SearchStateManager::load_puzzle_state(v3_view.puzzle_number);

    int rc = 0;
    auto fail = [&](const char* what) {
        std::fprintf(stderr, "[migration] FAIL: %s\n", what);
        if (rc == 0) rc = 31;
    };

    if (!loaded.valid) fail("loaded state should be valid for a clean v3 file");
    if (loaded.position_full[0] != v3_view.position_lo)
        fail("position_full[0] should equal v3 position_lo");
    if (loaded.position_full[1] != v3_view.position_hi)
        fail("position_full[1] should equal v3 position_hi");
    if (loaded.position_full[2] != 0)
        fail("position_full[2] should be zero after migration");
    if (loaded.position_full[3] != 0)
        fail("position_full[3] should be zero after migration");
    if (loaded.position_lo != v3_view.position_lo)
        fail("position_lo mirror should survive the migration");
    if (loaded.position_hi != v3_view.position_hi)
        fail("position_hi mirror should survive the migration");
    if (loaded.total_checked != v3_view.total_checked)
        fail("total_checked should survive the migration");
    if (loaded.loaded_version != 3u)
        fail("loaded_version should report 3 (the on-disk version)");

    // Cleanup: erase the migrated file so subsequent test runs start fresh.
    SearchStateManager::clear_puzzle_state(v3_view.puzzle_number);

    if (rc == 0) {
        std::printf(
            "[migration] PASS: v3 puzzle state file loads as v4 in-memory\n");
    }
    return rc;
}

int main() {
    const std::string wordlist_path = "test_resume_iteration_mode_wordlist.txt";
    write_wordlist(wordlist_path);

    // Map each mode to a phase_iteration in its range so the restored
    // iteration_mode is internally consistent with get_iteration_mode().
    // The mapping mirrors streaming_brain_wallet.cpp:71-85.
    struct Case {
        bw::IterationMode mode;
        size_t phase_iteration;
    };
    const Case cases[] = {
        { bw::IterationMode::RULE_STACKING,  3  },
        { bw::IterationMode::HYBRID_MASK,    8  },
        { bw::IterationMode::COMBINATOR,     13 },
        { bw::IterationMode::MARKOV,         18 },
        { bw::IterationMode::PCFG,           23 },
        { bw::IterationMode::KEYBOARD_WALK,  28 },
    };

    int rc = 0;
    for (const auto& c : cases) {
        int r = probe_resume_into(wordlist_path, c.mode, c.phase_iteration);
        if (r != 0 && rc == 0) rc = r;
    }

    int adv_rc = probe_advance_past_gpu_incompatible(wordlist_path);
    if (adv_rc != 0 && rc == 0) rc = adv_rc;

    // v1.4.2 R-B8: puzzle state file migration.
    int migrate_rc = probe_v3_to_v4_migration();
    if (migrate_rc != 0 && rc == 0) rc = migrate_rc;

    std::remove(wordlist_path.c_str());

    if (rc == 0) {
        std::printf("All resume-into-iteration-mode probes passed.\n");
    }
    return rc;
}
