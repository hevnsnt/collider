/**
 * test_markov_model
 *
 * Regression tests for the Markov generator's training counter and binary
 * model loader (audit H1 + H2).
 *
 *   H1: Trainer::total_passwords_ must count each accepted password EXACTLY
 *       once. The pre-fix code incremented once per password AND once per
 *       train() call, double-counting one extra per file. This test trains a
 *       known number of passwords across multiple files and asserts the count
 *       is exact.
 *
 *   H2: TransitionMatrix::load must reject a malformed / truncated / hostile
 *       .mkvc file by THROWING std::runtime_error, never by over-allocating or
 *       reading out of bounds. We forge files with absurd counts and with
 *       truncated bodies and assert each throws.
 *
 * Pure CPU, no GPU dependency.
 */

// markov.hpp declares MarkovSource : public PassphraseSource, so the base
// class definition must be visible first (production TUs include it before
// markov.hpp; this standalone test must too).
#include "../src/generators/passphrase_generator.hpp"
#include "../src/generators/markov.hpp"

#include <cstdio>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace mk = ::collider::markov;

namespace {

int fail(const char* msg, int code) {
    std::fprintf(stderr, "[FAIL] %s\n", msg);
    return code;
}

void write_lines(const std::string& path, const std::vector<std::string>& lines) {
    std::ofstream f(path, std::ios::binary);
    for (const auto& l : lines) f << l << '\n';
}

template <typename T>
void put(std::ofstream& f, T v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

bool throws_runtime_error(const std::string& path) {
    try {
        mk::TransitionMatrix::load(path);
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

}  // namespace

int main() {
    // --- Test 1 (H1): total_passwords_ counts each password exactly once -----
    {
        // Three files, 10 + 20 + 5 = 35 passwords total. Every line is >=
        // min_length (4) and <= max_length so none are skipped.
        const std::string f1 = "test_markov_h1_a.txt";
        const std::string f2 = "test_markov_h1_b.txt";
        const std::string f3 = "test_markov_h1_c.txt";

        std::vector<std::string> a(10, "passwordA");
        std::vector<std::string> b(20, "passwordB");
        std::vector<std::string> c(5, "passwordC");
        write_lines(f1, a);
        write_lines(f2, b);
        write_lines(f3, c);

        mk::Trainer trainer;
        trainer.train_multiple({f1, f2, f3});
        auto stats = trainer.get_stats();

        std::remove(f1.c_str());
        std::remove(f2.c_str());
        std::remove(f3.c_str());

        // EXACT: 35 passwords, no per-file inflation. The pre-fix code would
        // report 35 + 3 = 38 (one extra per train() call).
        if (stats.total_passwords != 35) {
            std::fprintf(stderr,
                         "[FAIL] total_passwords=%llu, expected 35 (H1 "
                         "double-count regressed)\n",
                         (unsigned long long)stats.total_passwords);
            return 1;
        }

        // Single-file path must also be exact.
        const std::string f4 = "test_markov_h1_d.txt";
        write_lines(f4, std::vector<std::string>(7, "passwordD"));
        mk::Trainer single;
        single.train(f4);
        std::remove(f4.c_str());
        if (single.get_stats().total_passwords != 7) {
            return fail("single-file total_passwords != 7 (H1 regressed)", 2);
        }
    }

    // --- Test 2 (H2): a well-formed round-trip still loads -------------------
    {
        const std::string model = "test_markov_h2_ok.mkvc";
        std::vector<std::string> lines(40, "satoshi");
        const std::string corpus = "test_markov_h2_corpus.txt";
        write_lines(corpus, lines);

        mk::Trainer trainer;
        trainer.train(corpus);
        auto matrix = trainer.build_matrix();
        matrix.save(model);
        std::remove(corpus.c_str());

        // Sanity: a real saved model loads without throwing and has contexts.
        try {
            auto loaded = mk::TransitionMatrix::load(model);
            if (loaded.num_contexts() == 0) {
                std::remove(model.c_str());
                return fail("round-trip model loaded with zero contexts", 10);
            }
        } catch (const std::exception& e) {
            std::remove(model.c_str());
            std::fprintf(stderr, "[FAIL] valid model failed to load: %s\n",
                         e.what());
            return 11;
        }
        std::remove(model.c_str());
    }

    // --- Test 3 (H2): absurd context count must throw, not over-allocate -----
    {
        const std::string bad = "test_markov_h2_bigcount.mkvc";
        {
            std::ofstream f(bad, std::ios::binary);
            f.write("MKVC", 4);
            put<uint32_t>(f, 1);            // version
            put<uint32_t>(f, 2);            // order
            put<uint32_t>(f, 0xFFFFFFFFu);  // num_ctx: ~4 billion, file is tiny
        }
        bool threw = throws_runtime_error(bad);
        std::remove(bad.c_str());
        if (!threw) {
            return fail("absurd Markov context count did not throw (H2)", 12);
        }
    }

    // --- Test 4 (H2): truncated body (count says 1, no data) must throw ------
    {
        const std::string bad = "test_markov_h2_trunc.mkvc";
        {
            std::ofstream f(bad, std::ios::binary);
            f.write("MKVC", 4);
            put<uint32_t>(f, 1);  // version
            put<uint32_t>(f, 2);  // order
            put<uint32_t>(f, 1);  // num_ctx = 1
            put<uint32_t>(f, 8);  // ctx_len = 8 but no context bytes follow
        }
        bool threw = throws_runtime_error(bad);
        std::remove(bad.c_str());
        if (!threw) {
            return fail("truncated Markov context body did not throw (H2)", 13);
        }
    }

    // --- Test 5 (H2): absurd transition count for a context must throw -------
    {
        const std::string bad = "test_markov_h2_bigprobs.mkvc";
        {
            std::ofstream f(bad, std::ios::binary);
            f.write("MKVC", 4);
            put<uint32_t>(f, 1);            // version
            put<uint32_t>(f, 2);            // order
            put<uint32_t>(f, 1);            // num_ctx = 1
            put<uint32_t>(f, 2);            // ctx_len = 2
            f.write("ab", 2);               // context bytes
            put<uint32_t>(f, 0xFFFFFFFFu);  // num_probs: absurd
        }
        bool threw = throws_runtime_error(bad);
        std::remove(bad.c_str());
        if (!threw) {
            return fail("absurd Markov transition count did not throw (H2)", 14);
        }
    }

    // --- Test 6 (H2): bad magic must throw -----------------------------------
    {
        const std::string bad = "test_markov_h2_magic.mkvc";
        {
            std::ofstream f(bad, std::ios::binary);
            f.write("XXXX", 4);
            put<uint32_t>(f, 1);
        }
        bool threw = throws_runtime_error(bad);
        std::remove(bad.c_str());
        if (!threw) {
            return fail("bad Markov magic did not throw (H2)", 15);
        }
    }

    std::printf("PASS: Markov H1 count exactness + H2 hardened loader all OK.\n");
    return 0;
}
