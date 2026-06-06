/**
 * test_pcfg_load_hardening
 *
 * Regression tests for Grammar::load against malformed / truncated / hostile
 * .pcfg files (audit H2). The loader must THROW std::runtime_error rather than
 * over-allocate (a 4-billion-entry count on a 20-byte file) or read out of
 * bounds when a length field outruns the file.
 *
 * Also confirms a valid round-trip still loads, so the hardening did not break
 * the happy path.
 *
 * Pure CPU, no GPU dependency.
 */

#include "../src/generators/pcfg.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

namespace pcfg = ::collider::pcfg;

namespace {

int fail(const char* msg, int code) {
    std::fprintf(stderr, "[FAIL] %s\n", msg);
    return code;
}

template <typename T>
void put(std::ofstream& f, T v) {
    f.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

void write_header(std::ofstream& f, uint32_t version) {
    f.write("PCFG", 4);
    put<uint32_t>(f, version);
}

bool throws_runtime_error(const std::string& path) {
    try {
        pcfg::Grammar::load(path);
    } catch (const std::runtime_error&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

}  // namespace

int main() {
    // --- Test 1: valid round-trip still loads --------------------------------
    {
        pcfg::Grammar g;
        g.structures.push_back({{"L5", "D2"}, pcfg::to_log_prob(0.7)});
        pcfg::NonTerminal l5;
        l5.type = "L";
        l5.length = 5;
        l5.terminals.push_back({"satos", pcfg::to_log_prob(0.6)});
        g.non_terminals["L5"] = l5;
        pcfg::NonTerminal d2;
        d2.type = "D";
        d2.length = 2;
        d2.terminals.push_back({"21", pcfg::to_log_prob(0.8)});
        g.non_terminals["D2"] = d2;

        const std::string ok = "test_pcfg_h2_ok.pcfg";
        g.save(ok);
        try {
            auto loaded = pcfg::Grammar::load(ok);
            if (loaded.structures.size() != 1 ||
                loaded.non_terminals.size() != 2) {
                std::remove(ok.c_str());
                return fail("round-trip grammar lost structures/non-terminals",
                            1);
            }
        } catch (const std::exception& e) {
            std::remove(ok.c_str());
            std::fprintf(stderr, "[FAIL] valid grammar failed to load: %s\n",
                         e.what());
            return 2;
        }
        std::remove(ok.c_str());
    }

    // --- Test 2: absurd structure count must throw ---------------------------
    {
        const std::string bad = "test_pcfg_h2_bigstruct.pcfg";
        {
            std::ofstream f(bad, std::ios::binary);
            write_header(f, 2);
            put<uint32_t>(f, 0xFFFFFFFFu);  // num_structures: absurd
        }
        bool threw = throws_runtime_error(bad);
        std::remove(bad.c_str());
        if (!threw) {
            return fail("absurd PCFG structure count did not throw (H2)", 3);
        }
    }

    // --- Test 3: absurd pattern size must throw ------------------------------
    {
        const std::string bad = "test_pcfg_h2_bigpattern.pcfg";
        {
            std::ofstream f(bad, std::ios::binary);
            write_header(f, 2);
            put<uint32_t>(f, 1);            // num_structures = 1
            put<uint32_t>(f, 0xFFFFFFFFu);  // pattern_size: absurd
        }
        bool threw = throws_runtime_error(bad);
        std::remove(bad.c_str());
        if (!threw) {
            return fail("absurd PCFG pattern size did not throw (H2)", 4);
        }
    }

    // --- Test 4: non-terminal name length outruns file must throw ------------
    {
        const std::string bad = "test_pcfg_h2_bignt.pcfg";
        {
            std::ofstream f(bad, std::ios::binary);
            write_header(f, 2);
            put<uint32_t>(f, 1);            // num_structures = 1
            put<uint32_t>(f, 1);            // pattern_size = 1
            put<uint32_t>(f, 0xFFFFFFFFu);  // nt_len: absurd, no bytes follow
        }
        bool threw = throws_runtime_error(bad);
        std::remove(bad.c_str());
        if (!threw) {
            return fail("absurd PCFG non-terminal name length did not throw "
                        "(H2)", 5);
        }
    }

    // --- Test 5: absurd terminal count must throw ----------------------------
    {
        const std::string bad = "test_pcfg_h2_bigterm.pcfg";
        {
            std::ofstream f(bad, std::ios::binary);
            write_header(f, 2);
            put<uint32_t>(f, 0);            // num_structures = 0
            put<uint32_t>(f, 1);            // num_nts = 1
            put<uint32_t>(f, 2);            // name_len = 2
            f.write("L5", 2);               // name
            f.put('L');                     // type
            put<uint32_t>(f, 5);            // length
            put<uint32_t>(f, 0xFFFFFFFFu);  // num_terminals: absurd
        }
        bool threw = throws_runtime_error(bad);
        std::remove(bad.c_str());
        if (!threw) {
            return fail("absurd PCFG terminal count did not throw (H2)", 6);
        }
    }

    // --- Test 6: truncated terminal value length outruns file must throw -----
    {
        const std::string bad = "test_pcfg_h2_bigval.pcfg";
        {
            std::ofstream f(bad, std::ios::binary);
            write_header(f, 2);
            put<uint32_t>(f, 0);            // num_structures = 0
            put<uint32_t>(f, 1);            // num_nts = 1
            put<uint32_t>(f, 2);            // name_len = 2
            f.write("L5", 2);               // name
            f.put('L');                     // type
            put<uint32_t>(f, 5);            // length
            put<uint32_t>(f, 1);            // num_terminals = 1
            put<uint32_t>(f, 0xFFFFFFFFu);  // val_len: absurd, no bytes follow
        }
        bool threw = throws_runtime_error(bad);
        std::remove(bad.c_str());
        if (!threw) {
            return fail("absurd PCFG terminal value length did not throw (H2)",
                        7);
        }
    }

    // --- Test 7: bad magic + unsupported version must throw ------------------
    {
        const std::string bad_magic = "test_pcfg_h2_magic.pcfg";
        {
            std::ofstream f(bad_magic, std::ios::binary);
            f.write("XXXX", 4);
            put<uint32_t>(f, 2);
        }
        bool m = throws_runtime_error(bad_magic);
        std::remove(bad_magic.c_str());
        if (!m) return fail("bad PCFG magic did not throw (H2)", 8);

        const std::string bad_ver = "test_pcfg_h2_version.pcfg";
        {
            std::ofstream f(bad_ver, std::ios::binary);
            write_header(f, 99);  // unsupported version
        }
        bool v = throws_runtime_error(bad_ver);
        std::remove(bad_ver.c_str());
        if (!v) return fail("unsupported PCFG version did not throw (H2)", 9);
    }

    std::printf("PASS: PCFG load hardening (H2) all OK.\n");
    return 0;
}
