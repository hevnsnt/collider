/**
 * test_bip_scan_runner_smoke -- T1-D (v1.5.0 audit follow-up).
 *
 * End-to-end smoke test for the BIP scanner derivation + bloom probe
 * chain. Builds a tiny in-memory bloom seeded with the address that
 * derives from the BIP-49 spec mnemonic at m/49'/0'/0'/0/0, then
 * iterates a 5-phrase candidate list and asserts:
 *   * The seeded phrase produces exactly one bloom hit at the
 *     expected profile + path.
 *   * The 4 non-matching phrases produce zero hits.
 *
 * Catches regressions in:
 *   * BIP-39 -> seed PBKDF2 (covered by T1-E too; redundancy is fine)
 *   * BIP-32 master + derive_path
 *   * hash160_p2sh_p2wpkh (covered by T1-F too)
 *   * MurmurHash3 + bloom probe
 *   * end-to-end glue across all of the above
 *
 * Does NOT spin up the TUI or the file-based runner; mirrors the
 * per-phrase logic inline so the test stays under 1 second of ctest
 * runtime and doesn't depend on the full bip_scanner_runner harness.
 */

#include "../src/core/bip32.hpp"
#include "../src/core/bip39.hpp"
#include "../src/runtime/bip_address.hpp"
#include "../src/tools/utxo_bloom_builder.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

int g_pass = 0;
int g_fail = 0;

void check(const char* tag, bool ok) {
    if (ok) {
        ++g_pass;
        std::printf("[ ok  ] %s\n", tag);
    } else {
        ++g_fail;
        std::fprintf(stderr, "[FAIL] %s\n", tag);
    }
}

std::string locate_wordlist() {
    const std::vector<std::string> candidates = {
        "data/bip39/english.txt",
        "../data/bip39/english.txt",
        "../../data/bip39/english.txt",
        "data/crypto/bip39_english.txt",
        "../data/crypto/bip39_english.txt",
        "../../data/crypto/bip39_english.txt",
    };
    for (const auto& p : candidates) {
        if (std::filesystem::exists(p)) return p;
    }
    return "";
}

// Mirror of the inner per-phrase loop in bip_scanner_runner.cpp.
// Returns the number of bloom hits this phrase produced.
int count_hits_for_phrase(
    const std::string& phrase,
    const collider::bip39::WordlistEnglish& wordlist,
    const collider::utxo::UTXOBloomBuilder& bloom,
    const std::vector<std::pair<std::string, bool /*is_p2sh*/>>& profiles,
    int addrs_per_profile) {
    auto words = collider::bip39::split_words(phrase);
    std::vector<uint8_t> entropy;
    if (!collider::bip39::validate_mnemonic(words, wordlist, &entropy)) {
        return 0;  // bad checksum
    }
    std::array<uint8_t, 64> seed;
    try {
        seed = collider::bip32::mnemonic_to_seed(phrase, std::string{});
    } catch (...) {
        return 0;
    }
    collider::bip32::ExtKey master;
    try {
        master =
            collider::bip32::master_from_seed(seed.data(), seed.size());
    } catch (...) {
        return 0;
    }

    int hits = 0;
    for (const auto& [path_tpl, is_p2sh] : profiles) {
        for (int i = 0; i < addrs_per_profile; ++i) {
            std::string path = path_tpl;
            size_t pos = path.find("{idx}");
            if (pos != std::string::npos) {
                path.replace(pos, 5, std::to_string(i));
            }
            collider::bip32::ExtKey child;
            try {
                auto parsed = collider::bip32::parse_path(path);
                child = collider::bip32::derive_path(master, parsed);
            } catch (...) {
                continue;
            }
            auto pub =
                collider::bip32::detail::priv_to_pub(child.key.data());
            std::array<uint8_t, 20> h160 =
                is_p2sh
                    ? collider::bip_address::hash160_p2sh_p2wpkh(pub.data())
                    : collider::bip_address::hash160_pubkey(pub.data());
            collider::utxo::H160 q{};
            std::memcpy(q.data, h160.data(), 20);
            if (bloom.probably_contains(q)) {
                ++hits;
            }
        }
    }
    return hits;
}

}  // namespace

int main() {
    using namespace collider;

    std::printf("=== test_bip_scan_runner_smoke (T1-D v1.5.0) ===\n");

    std::string wordlist_path = locate_wordlist();
    if (wordlist_path.empty()) {
        std::fprintf(stderr,
                     "[SKIP] BIP-39 english.txt not found from cwd\n");
        return 77;  // ctest skip
    }
    bip39::WordlistEnglish wordlist;
    try {
        wordlist.load(wordlist_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] wordlist load: %s\n", e.what());
        return 1;
    }

    // Build a tiny bloom (1024 bits, 4 hash fns) and seed it with the
    // hash160_p2sh_p2wpkh of the BIP-49 spec mainnet account 0 first
    // receiving pubkey derived from the spec abandon-about mnemonic
    // at m/49'/0'/0'/0/0.
    const char* seed_mnemonic =
        "abandon abandon abandon abandon abandon abandon "
        "abandon abandon abandon abandon abandon about";
    utxo::UTXOBloomBuilder::Config bcfg{};
    bcfg.expected_elements = 100;        // small bloom; sized for one entry
    bcfg.target_fp_rate    = 0.00001;
    utxo::UTXOBloomBuilder bloom(bcfg);

    {
        std::array<uint8_t, 64> seed;
        seed = bip32::mnemonic_to_seed(seed_mnemonic, std::string{});
        auto master = bip32::master_from_seed(seed.data(), seed.size());
        auto path = bip32::parse_path("m/49'/0'/0'/0/0");
        auto child = bip32::derive_path(master, path);
        auto pub = bip32::detail::priv_to_pub(child.key.data());
        auto h160 = bip_address::hash160_p2sh_p2wpkh(pub.data());
        utxo::H160 q{};
        std::memcpy(q.data, h160.data(), 20);
        bloom.add_h160(q);
    }

    // Subset of the bip_scanner profile list (the full set is 11
    // profiles x 20 addrs = 220 probes per phrase; for the smoke
    // test, 2 profiles x 1 addr suffices).
    const std::vector<std::pair<std::string, bool>> profiles = {
        {"m/49'/0'/0'/0/{idx}", true},   // BIP-49 P2SH-P2WPKH
        {"m/44'/0'/0'/0/{idx}", false},  // BIP-44 P2PKH (negative ctrl)
    };

    // 5-phrase candidate list: 1 matching + 4 non-matching, all
    // BIP-39-valid 12-word famous test mnemonics.
    const std::vector<std::pair<std::string, int>> candidates = {
        {"abandon abandon abandon abandon abandon abandon "
         "abandon abandon abandon abandon abandon about",
         /*expected_hits=*/1},  // seeded
        {"legal winner thank year wave sausage worth useful "
         "legal winner thank yellow",
         0},
        {"letter advice cage absurd amount doctor acoustic "
         "avoid letter advice cage above",
         0},
        {"zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong",
         0},
        {"jelly better achieve collect unaware mountain thought "
         "cargo oxygen act hood bridge",
         0},
    };

    int total_hits = 0;
    int seeded_phrase_hits = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& [phrase, want_hits] = candidates[i];
        int got = count_hits_for_phrase(
            phrase, wordlist, bloom, profiles, /*addrs_per_profile=*/1);
        if (i == 0) seeded_phrase_hits = got;
        total_hits += got;
        char tag[96];
        std::snprintf(tag, sizeof(tag),
                      "phrase[%zu]_hits_match (got=%d want=%d)",
                      i, got, want_hits);
        check(tag, got == want_hits);
    }
    char total_tag[96];
    std::snprintf(total_tag, sizeof(total_tag),
                  "total_hits_eq_seeded_phrase_hits (total=%d seeded=%d)",
                  total_hits, seeded_phrase_hits);
    check(total_tag, total_hits == seeded_phrase_hits && total_hits >= 1);

    std::printf("\n%d passes, %d failures\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
