/**
 * test_bip39_validate -- v1.4.2 Phase L.2 regression test.
 *
 * Reference vectors from the official BIP-39 test vector set
 * (https://github.com/trezor/python-mnemonic/blob/master/vectors.json).
 * Each entry is (entropy_hex, mnemonic) -- we validate the mnemonic and
 * compare the recovered ENT bytes byte-for-byte against the published
 * entropy.
 *
 * The test also exercises rejection paths:
 *   - bad word count
 *   - unknown word
 *   - flipped checksum bit
 */

#include "../src/core/bip39.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

void expect(bool cond, const char* label) {
    if (cond) {
        ++g_pass;
        std::printf("[PASS] %s\n", label);
    } else {
        ++g_fail;
        std::fprintf(stderr, "[FAIL] %s\n", label);
    }
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> out(hex.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        unsigned int v;
        std::sscanf(hex.c_str() + i * 2, "%02x", &v);
        out[i] = static_cast<uint8_t>(v);
    }
    return out;
}

std::string locate_wordlist() {
    // Try a handful of plausible paths so the test runs both from the
    // build dir and from the repo root.
    const std::vector<std::string> candidates = {
        "data/crypto/bip39_english.txt",
        "../data/crypto/bip39_english.txt",
        "../../data/crypto/bip39_english.txt",
        "../../../data/crypto/bip39_english.txt",
    };
    for (const auto& p : candidates) {
        if (std::filesystem::exists(p)) return p;
    }
    return "";
}

}  // namespace

int main() {
    using namespace collider::bip39;

    std::printf("test_bip39_validate (v1.4.2 Phase L.2)\n");

    std::string wordlist_path = locate_wordlist();
    if (wordlist_path.empty()) {
        std::fprintf(stderr,
                     "[SKIP] could not find bip39_english.txt from build "
                     "dir; this test is launched without the data/ tree "
                     "accessible.\n");
        return 77;  // CTest skip code
    }
    std::printf("Wordlist: %s\n", wordlist_path.c_str());

    WordlistEnglish wl;
    try {
        wl.load(wordlist_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] load: %s\n", e.what());
        return 1;
    }
    expect(wl.ready(), "WordlistEnglish::ready() after load");
    expect(wl.index("abandon") == 0, "first word indexes to 0");
    expect(wl.index("zoo") == 2047, "last word indexes to 2047");
    expect(wl.index("zzz") == -1, "unknown word returns -1");

    // Official BIP-39 test vectors (Trezor python-mnemonic, vectors.json).
    struct Vec { const char* entropy_hex; const char* mnemonic; };
    const Vec vectors[] = {
        {"00000000000000000000000000000000",
         "abandon abandon abandon abandon abandon abandon abandon abandon "
         "abandon abandon abandon about"},
        {"7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f",
         "legal winner thank year wave sausage worth useful legal winner "
         "thank yellow"},
        {"80808080808080808080808080808080",
         "letter advice cage absurd amount doctor acoustic avoid letter "
         "advice cage above"},
        {"ffffffffffffffffffffffffffffffff",
         "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong"},
        {"000000000000000000000000000000000000000000000000",
         "abandon abandon abandon abandon abandon abandon abandon abandon "
         "abandon abandon abandon abandon abandon abandon abandon abandon "
         "abandon agent"},
        {"0000000000000000000000000000000000000000000000000000000000000000",
         "abandon abandon abandon abandon abandon abandon abandon abandon "
         "abandon abandon abandon abandon abandon abandon abandon abandon "
         "abandon abandon abandon abandon abandon abandon abandon art"},
        {"ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
         "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo "
         "zoo zoo zoo zoo zoo zoo zoo vote"},
    };

    for (const auto& v : vectors) {
        auto expected = hex_to_bytes(v.entropy_hex);
        auto words = split_words(v.mnemonic);
        std::vector<uint8_t> recovered;
        bool ok = validate_mnemonic(words, wl, &recovered);
        bool match = ok && recovered == expected;

        char label[256];
        std::snprintf(label, sizeof(label),
                      "vector: %zu words / %zu-bit entropy",
                      words.size(), expected.size() * 8);
        expect(match, label);
        if (!match) {
            std::fprintf(stderr, "  expected: %s\n", v.entropy_hex);
            std::fprintf(stderr, "  ok=%d, recovered_size=%zu\n",
                         (int)ok, recovered.size());
        }
    }

    // --- Rejection: wrong word count -------------------------------------
    {
        std::vector<std::string> ten = {
            "abandon","abandon","abandon","abandon","abandon",
            "abandon","abandon","abandon","abandon","abandon"};
        expect(!validate_mnemonic(ten, wl), "10-word mnemonic rejected");
    }

    // --- Rejection: unknown word -----------------------------------------
    {
        auto words = split_words(vectors[0].mnemonic);
        words.back() = "satoshi";  // not in BIP-39
        expect(!validate_mnemonic(words, wl),
               "unknown word rejected");
    }

    // --- Rejection: flipped checksum -------------------------------------
    {
        // Vector 0 ends in "about" which encodes the correct CS for all
        // zeros entropy. Swap to "abandon" (idx 0) and the CS will flip.
        auto words = split_words(vectors[0].mnemonic);
        words.back() = "abandon";
        expect(!validate_mnemonic(words, wl),
               "flipped-checksum mnemonic rejected");
    }

    std::printf("\nSummary: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
