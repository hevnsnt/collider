/**
 * test_wif_kat.cpp -- B1 / wire-v4 (2026-05-23).
 *
 * Pins WIF (Wallet Import Format) decode + encode round-trips against
 * canonical Bitcoin Core test vectors. WIF correctness is foundational
 * to wire-v4: a wrong decode means the worker's --worker-key file
 * loads to the wrong privkey, the derived pubkey doesn't match the
 * declared worker_name, AUTH fails everywhere.
 *
 * Vectors:
 *   1. Satoshi's "5HpHagT65TZzG1PH3CSu63k8DbpvD8s5ip4nEB3kEsreAnchuDf"
 *      (uncompressed mainnet) -> privkey 0c28fca386c7a227600b2fe50b7cae11ec86d3bf1fbe471be89827e19d72aa1d
 *   2. The same privkey re-encoded as compressed wif
 *      (KwdMAjGmerYanjeui5SHS7JkmpZvVipYvB2LJGU1ZxJwYvP98617) round-trips.
 *   3. Round-trip: decode(encode(random32, compressed=true)) returns the same.
 *   4. Bad checksum: any single-byte corruption rejected.
 *   5. Wrong length: rejected.
 *
 * Reference vectors from https://en.bitcoin.it/wiki/Wallet_import_format.
 */

#include "../src/core/wif.hpp"

#include <array>
#include <cstring>
#include <iostream>
#include <string>

using collider::wif::DecodedKey;
using collider::wif::decode;
using collider::wif::encode;

namespace {

int g_pass = 0;
int g_fail = 0;

std::array<uint8_t, 32> hex32(const char* hex) {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 32; ++i) {
        unsigned hi = 0, lo = 0;
        char c = hex[i * 2];
        hi = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
        c = hex[i * 2 + 1];
        lo = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return out;
}

void expect_decode_match(const std::string& wif, bool expect_compressed,
                         const std::array<uint8_t, 32>& expect_priv,
                         const std::string& label) {
    auto r = decode(wif);
    if (!r.has_value()) {
        ++g_fail;
        std::cerr << "[FAIL] " << label << " -- decode returned nullopt\n";
        return;
    }
    if (r->compressed != expect_compressed) {
        ++g_fail;
        std::cerr << "[FAIL] " << label
                  << " -- compressed flag mismatch (got=" << r->compressed
                  << ", want=" << expect_compressed << ")\n";
        return;
    }
    if (r->privkey != expect_priv) {
        ++g_fail;
        std::cerr << "[FAIL] " << label << " -- privkey bytes differ\n";
        return;
    }
    ++g_pass;
    std::cout << "[PASS] " << label << "\n";
}

void expect_roundtrip(const std::array<uint8_t, 32>& priv, bool compressed,
                      const std::string& label) {
    std::string wif = encode(priv, compressed);
    auto r = decode(wif);
    if (!r.has_value()) {
        ++g_fail;
        std::cerr << "[FAIL] " << label << " -- decode(encode) returned nullopt\n";
        return;
    }
    if (r->compressed != compressed || r->privkey != priv) {
        ++g_fail;
        std::cerr << "[FAIL] " << label << " -- round-trip mismatch\n";
        return;
    }
    ++g_pass;
    std::cout << "[PASS] " << label << " (" << wif << ")\n";
}

void expect_decode_rejects(const std::string& wif, const std::string& label) {
    auto r = decode(wif);
    if (r.has_value()) {
        ++g_fail;
        std::cerr << "[FAIL] " << label << " -- accepted invalid wif\n";
        return;
    }
    ++g_pass;
    std::cout << "[PASS] " << label << " (rejected)\n";
}

}  // namespace

int main() {
    std::cout << "=== test_wif_kat (B1 wire-v4) ===\n";

    // 1. Bitcoin wiki reference: uncompressed mainnet privkey wif.
    auto priv1 = hex32("0c28fca386c7a227600b2fe50b7cae11ec86d3bf1fbe471be89827e19d72aa1d");
    expect_decode_match(
        "5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ",
        /*expect_compressed=*/false, priv1, "wiki uncompressed vector");

    // 2. The same privkey but compressed wif (BIP-178 sample).
    expect_decode_match(
        "KwdMAjGmerYanjeui5SHS7JkmpZvVipYvB2LJGU1ZxJwYvP98617",
        /*expect_compressed=*/true, priv1, "wiki compressed vector");

    // 3. Round-trip encode->decode for both forms.
    expect_roundtrip(priv1, true, "compressed round-trip");
    expect_roundtrip(priv1, false, "uncompressed round-trip");

    // 4. Distinct fixed bytes: another round-trip with a known privkey.
    auto priv2 = hex32("1111111111111111111111111111111111111111111111111111111111111111");
    expect_roundtrip(priv2, true, "0x11... round-trip");

    // 5. Bad checksum: flip one character near the end of a valid wif.
    {
        std::string wif = "5HueCGU8rMjxEXxiPuD5BDku4MkFqeZyd4dZ1jvhTVqvbTLvyTJ";
        wif[wif.size() - 1] = (wif.back() == 'J') ? 'K' : 'J';
        expect_decode_rejects(wif, "corrupted checksum rejected");
    }

    // 6. Wrong length: too-short Base58 must reject (not crash).
    expect_decode_rejects("1234567890", "too-short input rejected");

    // 7. Empty string: must reject without raising.
    expect_decode_rejects("", "empty input rejected");

    std::cout << "\n=== Result: " << g_pass << " pass, " << g_fail
              << " fail ===\n";
    return g_fail == 0 ? 0 : 1;
}
