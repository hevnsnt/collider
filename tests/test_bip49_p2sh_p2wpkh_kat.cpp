/**
 * test_bip49_p2sh_p2wpkh_kat -- T1-F (v1.5.0 audit follow-up).
 *
 * Pin the BIP-49 P2SH-P2WPKH address derivation against the spec
 * reference vectors so a regression in hash160_p2sh_p2wpkh's redeem-
 * script construction (0x00 || 0x14 || hash160(pubkey)) cannot land
 * silently. The bip_scanner_runner probes ~190 addresses per phrase
 * via this exact helper; wrong bytes here mean every BIP-49 hit
 * silently misses.
 *
 * Reference: BIP-49 mediawiki test vector
 * https://github.com/bitcoin/bips/blob/master/bip-0049.mediawiki
 *
 *   Mnemonic:  abandon abandon abandon abandon abandon abandon
 *              abandon abandon abandon abandon abandon about
 *   Passphrase: ""  (the spec uses empty passphrase, NOT "TREZOR")
 *   Path: m/49'/1'/0'/0/0   (testnet account 0 first receiving)
 *   Pubkey: 03a1af804ac108a8a51782198c2d034b28bf90c8803f5a53f76276fa69a4eae77f
 *   Address (testnet P2SH): 2Mww8dCYPUpKHofjgcXcBCEGmniw9CoaiD2
 *   Decoded address hash160: 336caa13e08b96080a32b5d818d59b4ab3b36742
 *
 * The hash160_p2sh_p2wpkh(pubkey) MUST equal 336caa13...3b36742.
 *
 * Independent of the trezor seed-vector test (T1-E) which exercises
 * the PBKDF2 path. This test pins the redeem-script + double-hash
 * piece specifically.
 */

#include "../src/core/bip32.hpp"
#include "../src/runtime/bip_address.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <string>

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

std::string hex_lower(const uint8_t* p, size_t n) {
    static const char* h = "0123456789abcdef";
    std::string out;
    out.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(h[(p[i] >> 4) & 0xF]);
        out.push_back(h[p[i] & 0xF]);
    }
    return out;
}

std::vector<uint8_t> hex_to_bytes(const char* hex) {
    std::vector<uint8_t> out;
    size_t n = std::strlen(hex);
    out.reserve(n / 2);
    for (size_t i = 0; i + 1 < n; i += 2) {
        unsigned int v = 0;
        std::sscanf(hex + i, "%02x", &v);
        out.push_back(static_cast<uint8_t>(v));
    }
    return out;
}

}  // namespace

int main() {
    using namespace collider;

    std::printf("=== test_bip49_p2sh_p2wpkh_kat (T1-F v1.5.0) ===\n");

    // Direct vector: feed the spec pubkey to hash160_p2sh_p2wpkh and
    // verify the redeem-script-hash matches the address's decoded
    // hash160 byte sequence.
    {
        auto pubkey = hex_to_bytes(
            "03a1af804ac108a8a51782198c2d034b28bf90c8803f5a53f76276fa69a4eae77f");
        if (pubkey.size() != 33) {
            std::fprintf(stderr, "[FAIL] pubkey hex decode wrong length\n");
            return 1;
        }
        auto h160 = bip_address::hash160_p2sh_p2wpkh(pubkey.data());
        std::string got = hex_lower(h160.data(), 20);
        const char* want = "336caa13e08b96080a32b5d818d59b4ab3b36742";
        if (got == want) {
            check("direct_pubkey_to_p2sh_p2wpkh_h160_matches_spec", true);
        } else {
            check("direct_pubkey_to_p2sh_p2wpkh_h160_matches_spec", false);
            std::fprintf(stderr, "      got:  %s\n", got.c_str());
            std::fprintf(stderr, "      want: %s\n", want);
        }
    }

    // End-to-end: derive the pubkey from the spec mnemonic via BIP-32
    // path m/49'/1'/0'/0/0 and re-run the hash160_p2sh_p2wpkh check.
    // This catches a regression in EITHER the derivation pipeline OR
    // the hash160 step.
    {
        const char* mnemonic =
            "abandon abandon abandon abandon abandon abandon "
            "abandon abandon abandon abandon abandon about";
        std::array<uint8_t, 64> seed;
        try {
            seed = bip32::mnemonic_to_seed(mnemonic, std::string{});
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                "[FAIL] mnemonic_to_seed: %s\n", e.what());
            return 1;
        }
        bip32::ExtKey master;
        try {
            master = bip32::master_from_seed(seed.data(), seed.size());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[FAIL] master_from_seed: %s\n", e.what());
            return 1;
        }
        std::vector<uint32_t> path;
        try {
            path = bip32::parse_path("m/49'/1'/0'/0/0");
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[FAIL] parse_path: %s\n", e.what());
            return 1;
        }
        bip32::ExtKey child;
        try {
            child = bip32::derive_path(master, path);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[FAIL] derive_path: %s\n", e.what());
            return 1;
        }
        auto pub = bip32::detail::priv_to_pub(child.key.data());
        std::string pub_hex = hex_lower(pub.data(), 33);
        const char* want_pub =
            "03a1af804ac108a8a51782198c2d034b28bf90c8803f5a53f76276fa69a4eae77f";
        if (pub_hex == want_pub) {
            check("e2e_derived_pubkey_matches_spec", true);
        } else {
            check("e2e_derived_pubkey_matches_spec", false);
            std::fprintf(stderr, "      got pub:  %s\n", pub_hex.c_str());
            std::fprintf(stderr, "      want pub: %s\n", want_pub);
        }
        auto h160 = bip_address::hash160_p2sh_p2wpkh(pub.data());
        std::string h_hex = hex_lower(h160.data(), 20);
        const char* want_h = "336caa13e08b96080a32b5d818d59b4ab3b36742";
        if (h_hex == want_h) {
            check("e2e_p2sh_h160_matches_spec", true);
        } else {
            check("e2e_p2sh_h160_matches_spec", false);
            std::fprintf(stderr, "      got h160:  %s\n", h_hex.c_str());
            std::fprintf(stderr, "      want h160: %s\n", want_h);
        }
    }

    std::printf("\n%d passes, %d failures\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
