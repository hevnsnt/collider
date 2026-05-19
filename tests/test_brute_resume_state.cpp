/**
 * test_brute_resume_state: task #5 coverage for the
 * BrainWalletSearchState serialization round-trip in --brute mode.
 *
 * Pins:
 *   1. A brute-mode state with (length_idx, position, lengths_csv) saves
 *      and loads with all fields preserved.
 *   2. validate_state() accepts a brute-mode snapshot without requiring
 *      wordlist info (wordlist_size = 0, wordlist_hash = 0 are fine).
 *   3. compute_checksum() is sensitive to the new brute fields; two
 *      identical states differing only in brute_position must produce
 *      different checksums (otherwise resume from the wrong position
 *      would not trip the checksum mismatch).
 *
 * Does not exercise GPU code or the BrainWalletStateManager save_state
 * file-I/O path: those are integration concerns. This is a pure
 * serialization / validation contract test.
 */

#include "src/core/brainwallet_state.hpp"
#include "src/core/search_state.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using collider::BrainWalletSearchState;
using collider::BrainWalletStateManager;
using collider::PuzzleSearchState;
using collider::SearchStateManager;

namespace {

int g_fails = 0;

void expect_true(bool cond, const char* label) {
    if (!cond) {
        std::fprintf(stderr, "[FAIL] %s\n", label);
        ++g_fails;
    }
}

template <typename A, typename B>
void expect_eq(const A& got, const B& want, const char* label) {
    if (!(got == want)) {
        std::fprintf(stderr, "[FAIL] %s\n", label);
        ++g_fails;
    }
}

// Build a minimal brute-mode state; the wordlist/phase fields are
// untouched (they're ignored in brute mode by design).
BrainWalletSearchState make_brute_state() {
    BrainWalletSearchState s;
    s.is_brute_mode = true;
    s.brute_length_idx = 1;
    s.brute_position = 1234567;
    s.brute_lengths_csv = "7,8,9";
    s.total_checked = 9876543;
    s.unique_tested = 9876543;
    s.hits_found = 0;
    s.session_id = "bw_brute_resume_test";
    s.state_version = 3;
    return s;
}

}  // namespace

int main() {
    // --- 1. validate_state accepts a brute-mode snapshot without wordlist info ---
    {
        BrainWalletSearchState s = make_brute_state();
        std::string err = BrainWalletStateManager::validate_state(s);
        expect_true(err.empty(),
                    "validate_state accepts brute-mode snapshot with no wordlist");
    }

    // --- 2. checksum sensitivity to brute_position ---
    {
        BrainWalletSearchState a = make_brute_state();
        BrainWalletSearchState b = make_brute_state();
        b.brute_position += 1;  // Only the position differs.
        uint32_t hash_a = BrainWalletStateManager::compute_checksum(a);
        uint32_t hash_b = BrainWalletStateManager::compute_checksum(b);
        expect_true(hash_a != hash_b,
                    "checksum differs when brute_position changes");
    }

    // --- 3. checksum sensitivity to brute_lengths_csv ---
    {
        BrainWalletSearchState a = make_brute_state();
        BrainWalletSearchState b = make_brute_state();
        b.brute_lengths_csv = "6,7,8";  // Different list.
        uint32_t hash_a = BrainWalletStateManager::compute_checksum(a);
        uint32_t hash_b = BrainWalletStateManager::compute_checksum(b);
        expect_true(hash_a != hash_b,
                    "checksum differs when brute_lengths_csv changes");
    }

    // --- 4. checksum sensitivity to is_brute_mode flag itself ---
    {
        BrainWalletSearchState a = make_brute_state();
        BrainWalletSearchState b = make_brute_state();
        b.is_brute_mode = false;
        uint32_t hash_a = BrainWalletStateManager::compute_checksum(a);
        uint32_t hash_b = BrainWalletStateManager::compute_checksum(b);
        expect_true(hash_a != hash_b,
                    "checksum differs when is_brute_mode flips");
    }

    // --- 5. Round-trip via on-disk save/load preserves brute fields ---
    {
        // Redirect ~/.collider/state to a test-local directory so we
        // don't stomp the user's real state file. The manager reads
        // USERPROFILE / HOME at every call; setting the right env var
        // for this OS is enough.
#ifdef _WIN32
        std::string tmp_home = (std::filesystem::temp_directory_path() /
                                "collider_brute_resume_test").string();
        _putenv_s("USERPROFILE", tmp_home.c_str());
#else
        std::string tmp_home = (std::filesystem::temp_directory_path() /
                                "collider_brute_resume_test").string();
        setenv("HOME", tmp_home.c_str(), 1);
#endif
        std::filesystem::create_directories(tmp_home + "/.collider/state");

        BrainWalletSearchState in = make_brute_state();
        bool save_ok = BrainWalletStateManager::save_state(in);
        expect_true(save_ok, "save_state succeeds for brute-mode snapshot");

        BrainWalletSearchState out = BrainWalletStateManager::load_state();
        expect_true(out.valid, "load_state returns valid for brute snapshot");
        expect_eq(out.is_brute_mode, true, "round-trip: is_brute_mode");
        expect_eq(out.brute_length_idx, (size_t)1,
                  "round-trip: brute_length_idx");
        expect_eq(out.brute_position, (uint64_t)1234567,
                  "round-trip: brute_position");
        expect_eq(out.brute_lengths_csv, std::string("7,8,9"),
                  "round-trip: brute_lengths_csv");
        expect_eq(out.total_checked, (uint64_t)9876543,
                  "round-trip: total_checked");

        // Cleanup.
        BrainWalletStateManager::clear_state();
    }

    // ------------------------------------------------------------------
    // v1.4.2 R-B8: PuzzleSearchState v4 on-disk round-trip.
    //
    // The puzzle search state file format moved to v4 (R-B8) to carry the
    // full UInt256 position rather than just position_lo / position_hi.
    // This block exercises:
    //   - The save -> load round-trip preserves all four position_full
    //     limbs (and the legacy lo/hi mirror).
    //   - validate_state accepts a state whose puzzle bit-length pins the
    //     position to the bottom 128 bits (the v3-compatible case).
    //   - validate_state correctly rejects a state whose position_full
    //     overflows the puzzle's allowed bit window.
    //   - compute_checksum is sensitive to position_full[2] / [3] (a v3
    //     -> v4 silent regression would forget those bytes and the
    //     checksum would not catch tampering of the upper limbs).
    // ------------------------------------------------------------------
    {
        // Use a per-test HOME so the real ~/.collider/state never gets
        // stomped. The brain-wallet block above already did this; we set
        // it again with a distinct directory for clarity.
#ifdef _WIN32
        std::string puzzle_home =
            (std::filesystem::temp_directory_path() /
             "collider_puzzle_state_v4_test").string();
        _putenv_s("USERPROFILE", puzzle_home.c_str());
#else
        std::string puzzle_home =
            (std::filesystem::temp_directory_path() /
             "collider_puzzle_state_v4_test").string();
        setenv("HOME", puzzle_home.c_str(), 1);
#endif
        std::filesystem::create_directories(puzzle_home + "/.collider/state");

        // Build a v4 state with all four limbs populated. puzzle_number
        // 200 would overflow even uint256, so we deliberately stay
        // inside the supported 1-160 window.
        PuzzleSearchState in;
        in.puzzle_number       = 95;  // 95-bit puzzle
        in.position_full[0]    = 0x0123456789ABCDEFULL;
        in.position_full[1]    = 0x7FFFFFFFULL;  // top 31 bits of the high limb
        in.position_full[2]    = 0;              // 95-bit: limbs 2-3 must be 0
        in.position_full[3]    = 0;
        in.position_lo         = in.position_full[0];
        in.position_hi         = in.position_full[1];
        in.total_checked       = 1'234'567'890ULL;

        // 1. Round-trip via on-disk save / load.
        bool save_ok = SearchStateManager::save_puzzle_state(in);
        expect_true(save_ok, "PuzzleSearchState v4: save returns true");

        PuzzleSearchState out =
            SearchStateManager::load_puzzle_state(in.puzzle_number);
        expect_true(out.valid, "PuzzleSearchState v4: load returns valid");
        expect_eq(out.position_full[0], in.position_full[0],
                  "PuzzleSearchState v4: position_full[0] preserved");
        expect_eq(out.position_full[1], in.position_full[1],
                  "PuzzleSearchState v4: position_full[1] preserved");
        expect_eq(out.position_full[2], (uint64_t)0,
                  "PuzzleSearchState v4: position_full[2] preserved (zero)");
        expect_eq(out.position_full[3], (uint64_t)0,
                  "PuzzleSearchState v4: position_full[3] preserved (zero)");
        expect_eq(out.position_lo, in.position_full[0],
                  "PuzzleSearchState v4: position_lo mirrors position_full[0]");
        expect_eq(out.position_hi, in.position_full[1],
                  "PuzzleSearchState v4: position_hi mirrors position_full[1]");
        expect_eq(out.total_checked, in.total_checked,
                  "PuzzleSearchState v4: total_checked preserved");
        expect_eq(out.loaded_version, collider::kSearchStateVersion,
                  "PuzzleSearchState v4: loaded_version reports v4");

        // 2. validate_state rejects an overflow in the high limb. For
        // puzzle 95 the allowed bits are [0..94], which fits in
        // position_full[0..1] with position_full[1] <= 0x7FFFFFFF; set
        // limb 2 non-zero and we expect a rejection.
        PuzzleSearchState bad = in;
        bad.position_full[2] = 1;  // way past the 95-bit window
        std::string err = SearchStateManager::validate_state(bad);
        expect_true(!err.empty(),
                    "PuzzleSearchState v4: validate rejects bits above the "
                    "puzzle's range");

        // 3. Checksum sensitivity to position_full[2] and [3]. A v3 ->
        // v4 regression that "forgot" to mix the upper limbs would let
        // a tampered file pass; this test pins the FNV mix domain.
        PuzzleSearchState a = in;
        PuzzleSearchState b = in;
        b.position_full[2] = 0x42;
        uint32_t ha = SearchStateManager::compute_checksum(a);
        uint32_t hb = SearchStateManager::compute_checksum(b);
        expect_true(ha != hb,
                    "PuzzleSearchState v4: checksum is sensitive to "
                    "position_full[2]");

        PuzzleSearchState c = in;
        c.position_full[3] = 0xDEADBEEFULL;
        uint32_t hc = SearchStateManager::compute_checksum(c);
        expect_true(ha != hc,
                    "PuzzleSearchState v4: checksum is sensitive to "
                    "position_full[3]");

        // 4. Cleanup the file we just wrote.
        SearchStateManager::clear_puzzle_state(in.puzzle_number);
    }

    if (g_fails) {
        std::fprintf(stderr, "\n%d failure(s).\n", g_fails);
        return 1;
    }
    std::printf("All brute-resume state tests passed.\n");
    return 0;
}
