/**
 * test_pcfg_probability_order
 *
 * Covers the v1.5.x PCFG improvements:
 *   1. Generator emits candidates in non-increasing probability order with no
 *      duplicate phrases over the first N outputs (Weir pivot enumeration over
 *      a log-space max-heap).
 *   2. Log-space scoring (Grammar::score_log) is internally consistent: the
 *      score of an emitted phrase equals the priority the generator reported,
 *      and the structure log-prob plus terminal log-probs add up.
 *   3. Multithreaded training produces identical counts to single-threaded.
 *   4. wordlist_dedup keeps first occurrence, exact, order-preserving.
 *
 * Pure CPU, no GPU dependency.
 */

#include "../src/generators/pcfg.hpp"
#include "../src/generators/wordlist_dedup.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace pcfg = ::collider::pcfg;
namespace gen = ::collider::generators;

namespace {

int fail(const char* msg, int code) {
    std::fprintf(stderr, "[FAIL] %s\n", msg);
    return code;
}

// Build a tiny hand-made grammar so the expected ordering is deterministic and
// independent of any corpus. Two structures, a couple of terminals each, all
// in log space.
pcfg::Grammar make_tiny_grammar() {
    pcfg::Grammar g;

    // Structure "L5 D2" with linear prob 0.7, "L5" with linear prob 0.3.
    g.structures.push_back({{"L5", "D2"}, pcfg::to_log_prob(0.7)});
    g.structures.push_back({{"L5"}, pcfg::to_log_prob(0.3)});
    std::sort(g.structures.begin(), g.structures.end(),
              [](const auto& a, const auto& b) { return a.log_prob > b.log_prob; });

    pcfg::NonTerminal l5;
    l5.type = "L";
    l5.length = 5;
    l5.terminals.push_back({"satos", pcfg::to_log_prob(0.6)});
    l5.terminals.push_back({"bitco", pcfg::to_log_prob(0.4)});
    g.non_terminals["L5"] = l5;

    pcfg::NonTerminal d2;
    d2.type = "D";
    d2.length = 2;
    d2.terminals.push_back({"21", pcfg::to_log_prob(0.8)});
    d2.terminals.push_back({"09", pcfg::to_log_prob(0.2)});
    g.non_terminals["D2"] = d2;

    return g;
}

void write_corpus(const std::string& path, const std::vector<std::string>& lines) {
    std::ofstream f(path, std::ios::binary);
    for (const auto& l : lines) f << l << '\n';
}

}  // namespace

int main() {
    // --- Test 1 + 2: probability order, no dupes, score consistency --------
    {
        pcfg::Grammar g = make_tiny_grammar();
        pcfg::Generator generator(g);

        std::set<std::string> seen;
        double prev = std::numeric_limits<double>::infinity();
        size_t count = 0;
        const size_t kWant = 200;

        for (size_t i = 0; i < kWant; ++i) {
            auto c = generator.next();
            if (!c.has_value()) break;  // grammar exhausted (small grammar)
            ++count;

            // No duplicate phrases.
            if (!seen.insert(c->phrase).second) {
                return fail("duplicate phrase emitted by Generator", 10);
            }

            // Non-increasing probability: priority carries the log-prob.
            double lp = static_cast<double>(c->priority);
            if (lp > prev + 1e-6) {
                std::fprintf(stderr,
                             "[FAIL] phrase '%s' log-prob %.6f > previous %.6f\n",
                             c->phrase.c_str(), lp, prev);
                return 11;
            }
            prev = lp;

            // Score consistency: re-scoring the emitted phrase under the
            // grammar must match the reported priority (within float slack).
            double rescored = g.score_log(c->phrase);
            if (std::abs(rescored - lp) > 1e-3) {
                std::fprintf(stderr,
                             "[FAIL] score_log('%s')=%.6f != priority %.6f\n",
                             c->phrase.c_str(), rescored, lp);
                return 12;
            }
        }

        // The tiny grammar has 2*2 + 2 = 6 distinct phrases; the generator
        // must enumerate exactly those with no omissions or repeats.
        if (count != 6) {
            std::fprintf(stderr,
                         "[FAIL] expected 6 distinct phrases, got %zu\n", count);
            return 13;
        }

        // Highest-probability phrase must be the all-zero config of the most
        // probable structure: "satos" + "21".
        // (0.7 * 0.6 * 0.8 = 0.336 beats the L5-only 0.3 * 0.6 = 0.18.)
        // We re-run to grab the first emission.
        pcfg::Generator g2(g);
        auto first = g2.next();
        if (!first.has_value() || first->phrase != "satos21") {
            return fail("first emission was not the most probable phrase satos21",
                        14);
        }
    }

    // --- Test 3: multithreaded training == single-threaded -----------------
    {
        const std::string corpus = "test_pcfg_corpus.txt";
        std::vector<std::string> lines;
        // Mix of structures, years, keyboard walks, long lowercase runs.
        for (int i = 0; i < 50; ++i) {
            lines.push_back("password1999");
            lines.push_back("satoshi2009");
            lines.push_back("qwerty");
            lines.push_back("correcthorsebattery");
            lines.push_back("hunter2");
            lines.push_back("Bitcoin21");
        }
        write_corpus(corpus, lines);

        pcfg::TrainerConfig st_cfg;
        st_cfg.num_threads = 1;
        pcfg::Trainer st(st_cfg);
        st.train(corpus);
        auto g_single = st.build_grammar();
        auto stats_single = st.get_training_stats();

        pcfg::TrainerConfig mt_cfg;
        mt_cfg.num_threads = 4;
        pcfg::Trainer mt(mt_cfg);
        mt.train(corpus);
        auto g_multi = mt.build_grammar();
        auto stats_multi = mt.get_training_stats();

        std::remove(corpus.c_str());

        if (stats_single.total_passwords != stats_multi.total_passwords) {
            std::fprintf(stderr,
                         "[FAIL] passwords single=%llu multi=%llu\n",
                         (unsigned long long)stats_single.total_passwords,
                         (unsigned long long)stats_multi.total_passwords);
            return 20;
        }
        if (stats_single.unique_structures != stats_multi.unique_structures) {
            return fail("structure count differs single vs multi-threaded", 21);
        }
        if (stats_single.total_non_terminals != stats_multi.total_non_terminals) {
            return fail("non-terminal count differs single vs multi-threaded", 22);
        }
        if (g_single.structures.size() != g_multi.structures.size()) {
            return fail("grammar structure size differs single vs multi", 23);
        }
        // Year detector should have fired: there must be a Y4 non-terminal.
        if (g_single.non_terminals.find("Y4") == g_single.non_terminals.end()) {
            return fail("year detector did not produce a Y4 non-terminal", 24);
        }
        // Keyboard detector should have fired on "qwerty" (K6).
        if (g_single.non_terminals.find("K6") == g_single.non_terminals.end()) {
            return fail("keyboard detector did not produce a K6 non-terminal", 25);
        }
        if (stats_single.keyboard_patterns_detected == 0) {
            return fail("keyboard_patterns_detected stayed zero", 26);
        }
    }

    // --- Test 3b: train/score/generate parity for a K + Y password ---------
    // Locks the fix for the score_log parity gap: a password trained with the
    // keyboard ('K') and year ('Y') detectors must re-score to a finite
    // log-prob (NOT kLogZero) using the SAME shared segmentation, and that
    // log-prob must match the priority the Generator assigns when it emits the
    // identical phrase.
    {
        const std::string corpus = "test_pcfg_parity_corpus.txt";
        std::vector<std::string> lines;
        // "qwerty2020" segments to "K6 Y4" under the shared detectors. Repeat
        // it so it survives the min_terminal_prob floor and dominates its
        // structure (the generator will emit it as the top config of K6 Y4).
        for (int i = 0; i < 40; ++i) {
            lines.push_back("qwerty2020");
        }
        // A little noise so the grammar has more than one structure.
        for (int i = 0; i < 5; ++i) {
            lines.push_back("hello99");
        }
        write_corpus(corpus, lines);

        pcfg::TrainerConfig cfg;
        cfg.num_threads = 1;
        cfg.min_length = 4;
        pcfg::Trainer tr(cfg);
        tr.train(corpus);
        auto grammar = tr.build_grammar();
        std::remove(corpus.c_str());

        // The structure "K6 Y4" must exist (proves K + Y segmentation during
        // training).
        bool has_ky = false;
        for (const auto& s : grammar.structures) {
            if (s.pattern.size() == 2 && s.pattern[0] == "K6" &&
                s.pattern[1] == "Y4") {
                has_ky = true;
                break;
            }
        }
        if (!has_ky) {
            return fail("training did not produce the K6 Y4 structure", 40);
        }

        // Re-score the exact password. With the parity fix this must be finite
        // (the old inline classify() would parse it as L6 D4 -> structure not
        // found -> kLogZero).
        double scored = grammar.score_log("qwerty2020");
        if (scored == pcfg::kLogZero || !std::isfinite(scored)) {
            return fail("score_log('qwerty2020') returned kLogZero (parity gap)",
                        41);
        }

        // The Generator must emit "qwerty2020", and its priority (log-prob)
        // must match score_log within float slack.
        pcfg::Generator generator(grammar);
        bool found = false;
        for (int i = 0; i < 1000; ++i) {
            auto c = generator.next();
            if (!c.has_value()) break;
            if (c->phrase == "qwerty2020") {
                double gen_lp = static_cast<double>(c->priority);
                if (std::abs(gen_lp - scored) > 1e-3) {
                    std::fprintf(stderr,
                                 "[FAIL] generate/score parity: gen=%.6f "
                                 "score=%.6f\n", gen_lp, scored);
                    return 42;
                }
                found = true;
                break;
            }
        }
        if (!found) {
            return fail("Generator never emitted the trained phrase qwerty2020",
                        43);
        }
    }

    // --- Test 4: exact wordlist dedup, first-occurrence order --------------
    {
        std::vector<std::string> words = {
            "alpha", "beta", "alpha", "gamma", "beta", "delta", "alpha",
        };
        gen::DedupStats stats;
        auto unique = gen::dedup_copy(words, &stats);

        std::vector<std::string> expected = {"alpha", "beta", "gamma", "delta"};
        if (unique != expected) {
            return fail("dedup did not preserve first-occurrence order", 30);
        }
        if (stats.input_count != 7 || stats.unique_count != 4 ||
            stats.dropped_count != 3) {
            std::fprintf(stderr,
                         "[FAIL] dedup stats in=%zu uniq=%zu drop=%zu\n",
                         stats.input_count, stats.unique_count,
                         stats.dropped_count);
            return 31;
        }

        // In-place overload should agree.
        std::vector<std::string> inplace = words;
        gen::dedup_in_place(inplace);
        if (inplace != expected) {
            return fail("dedup_in_place result differs from dedup_copy", 32);
        }

        // File-to-file dedup round-trip.
        const std::string in_path = "test_pcfg_dedup_in.txt";
        const std::string out_path = "test_pcfg_dedup_out.txt";
        {
            std::ofstream f(in_path, std::ios::binary);
            for (const auto& w : words) f << w << '\n';
        }
        gen::DedupStats fstats;
        if (!gen::dedup_file(in_path, out_path, &fstats)) {
            std::remove(in_path.c_str());
            return fail("dedup_file returned false", 33);
        }
        std::vector<std::string> file_out;
        {
            std::ifstream f(out_path, std::ios::binary);
            std::string l;
            while (std::getline(f, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                file_out.push_back(l);
            }
        }
        std::remove(in_path.c_str());
        std::remove(out_path.c_str());
        if (file_out != expected) {
            return fail("dedup_file output differs from expected unique set", 34);
        }
    }

    std::printf("PASS: PCFG log-space order, scoring, MT training, and dedup all OK.\n");
    return 0;
}
