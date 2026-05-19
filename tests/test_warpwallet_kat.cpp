/**
 * test_warpwallet_kat -- known-answer test for the WarpWallet brainwallet
 * derivation (Keybase spec: scrypt(N=2^18, r=8, p=1) +
 * PBKDF2-SHA256(c=2^16), XOR).
 *
 * T3.2 (2026-05-17): file renamed from test_warpwallet_properties.cpp back
 * to its original name now that real Keybase reference vectors are pinned
 * (the previous rename was forced because the file shipped property-only
 * checks and the _kat name overstated what it tested).
 *
 * Vector source: https://github.com/keybase/warpwallet/blob/master/test/spec.json
 *   - Generated 2013-11-20 by the reference Keybase WarpWallet generator.
 *   - Params: N=2^18, p=1, r=8, dkLen=32, pbkdf2c=2^16 (matches the
 *     constants in src/core/warpwallet.hpp).
 *   - seeds[0] = scrypt(passphrase+"\x01", salt+"\x01")
 *     seeds[1] = pbkdf2_sha256(passphrase+"\x02", salt+"\x02")
 *     seeds[2] = seeds[0] XOR seeds[1]  -- THIS is the final 32-byte
 *                                          private key.
 *
 * Coverage:
 *
 *   KAT-1..3: derive_key on three pinned Keybase vectors must produce the
 *             exact seeds[2] byte sequence. A wrong-but-deterministic
 *             reimplementation (incorrect scrypt block layout, swapped
 *             suffix bytes, mis-ordered XOR endianness, off-by-one
 *             PBKDF2 iteration count) is now visible as a byte mismatch
 *             on at least one vector.
 *
 * Plus the original property-based checks (kept verbatim from the
 * preceding test_warpwallet_properties.cpp -- properties are still useful
 * for catching defects that don't land on any single KAT, e.g. a fresh
 * RNG path that produces different output on alternating runs):
 *
 *   PROP-1.  derive_key is deterministic across runs.
 *   PROP-2.  derive_key output is exactly 32 bytes and not all-zero.
 *   PROP-3.  passphrase-bit sensitivity (Hamming distance > 64 bits).
 *   PROP-4.  salt-bit sensitivity (Hamming distance > 64 bits).
 *   PROP-5.  compute_hash160 / derive_key + cpu::compute_hash160 agree.
 *   PROP-6.  verify() round-trip.
 *
 * Runtime budget: scrypt(N=2^18,r=8,p=1) is intentionally slow; expect
 * ~1-3s per derivation. Three KAT vectors + three property derivations
 * = ~6-18 seconds; the CMake target keeps a 60s TIMEOUT to absorb slow
 * CI runners.
 */

#include "../src/core/warpwallet.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

size_t hamming_distance_bytes(const uint8_t* a, const uint8_t* b, size_t n) {
    size_t d = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t x = a[i] ^ b[i];
        while (x) { d += (x & 1); x >>= 1; }
    }
    return d;
}

// Decode a 64-char ASCII hex string into a 32-byte array. The Keybase
// spec.json stores the seeds[2] values as lowercase hex with no 0x prefix.
std::array<uint8_t, 32> hex_to_bytes_32(const char* hex) {
    std::array<uint8_t, 32> out{};
    auto nibble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return -1;
    };
    for (size_t i = 0; i < 32; ++i) {
        int hi = nibble(hex[2 * i]);
        int lo = nibble(hex[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            std::fprintf(stderr,
                "hex_to_bytes_32: bad nibble at offset %zu in '%s'\n",
                2 * i, hex);
            std::abort();
        }
        out[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return out;
}

void dump_hex(const char* label, const uint8_t* p, size_t n) {
    std::fprintf(stderr, "  %s: ", label);
    for (size_t i = 0; i < n; ++i) std::fprintf(stderr, "%02x", p[i]);
    std::fputc('\n', stderr);
}

// ============================================================================
// Pinned Keybase WarpWallet vectors. Source:
//   https://github.com/keybase/warpwallet/blob/master/test/spec.json
//
// Three vectors chosen from the upstream 12-vector set: the first three
// short-passphrase entries. We keep three (not all 12) because scrypt(N=2^18)
// is ~1-3 seconds per call; running all twelve would exhaust the 60s
// CMake timeout on slow CI runners. Three vectors is enough to catch the
// classes of bug a single KAT could miss: scrypt block ordering, PBKDF2
// iteration count, XOR endianness, salt/passphrase suffix bytes.
// ============================================================================
struct KeybaseVector {
    const char* passphrase;
    const char* salt;
    const char* expected_privkey_hex;   // seeds[2] = scrypt XOR pbkdf2
};

const KeybaseVector KAT_VECTORS[] = {
    {
        "ER8FT+HFjk0",
        "7DpniYifN6c",
        "6f2552e159f2a1e1e26c2262da459818fd56c81c363fcc70b94c423def42e59f",
    },
    {
        "YqIDBApDYME",
        "G34HqIgjrIc",
        "da009602a5781a8795d55c6e68a4b4d52969a75955ea70255869dd17c3398592",
    },
    {
        "FPdAxCygMJg",
        "X+qaSwhUYXw",
        "2f6af9ad997b831963f4de48278c044e687ff3cecc25739d1564985b929cb3dd",
    },
};

}  // namespace

int main() {
    using namespace collider::warpwallet;

    std::printf("test_warpwallet_kat\n");
    std::printf("Note: scrypt(N=2^18,r=8,p=1) is intentionally slow; expect ~1-3s/derivation.\n");
    std::printf("KAT vectors: %zu pinned from Keybase spec.json + property suite.\n",
                sizeof(KAT_VECTORS) / sizeof(KAT_VECTORS[0]));

    // --- KAT vectors --------------------------------------------------------
    for (size_t i = 0; i < sizeof(KAT_VECTORS) / sizeof(KAT_VECTORS[0]); ++i) {
        const auto& v = KAT_VECTORS[i];
        char label[96];
        std::snprintf(label, sizeof(label),
            "KAT[%zu]: derive_key(\"%s\", \"%s\") matches Keybase reference",
            i, v.passphrase, v.salt);

        const auto expected = hex_to_bytes_32(v.expected_privkey_hex);
        const auto computed = derive_key(v.passphrase, v.salt);

        const bool ok = (computed == expected);
        expect(ok, label);
        if (!ok) {
            dump_hex("expected", expected.data(), 32);
            dump_hex("computed", computed.data(), 32);
        }
    }

    // --- Properties (kept verbatim from the former _properties suite) -------
    const std::string p1 = "satoshi nakamoto bitcoin";
    const std::string s1 = "warpwallet@example.com";
    auto k1 = derive_key(p1, s1);
    auto k1_again = derive_key(p1, s1);
    expect(k1 == k1_again, "PROP-1: deterministic same (passphrase, salt) -> same key");

    expect(k1.size() == 32, "PROP-2a: output is exactly 32 bytes");
    bool any_nonzero = false;
    for (uint8_t b : k1) if (b != 0) { any_nonzero = true; break; }
    expect(any_nonzero, "PROP-2b: output is not all-zero");

    auto k_diff_pass = derive_key("satoshi nakamoto bitcoiN", s1);  // 1-bit flip
    size_t hd_pass = hamming_distance_bytes(k1.data(), k_diff_pass.data(), 32);
    expect(hd_pass > 64,
           "PROP-3: passphrase sensitivity (~100-128 bits diff for 1-char change)");
    std::printf("  passphrase Hamming distance: %zu bits (of 256)\n", hd_pass);

    auto k_diff_salt = derive_key(p1, "warpwallet@example.coM");
    size_t hd_salt = hamming_distance_bytes(k1.data(), k_diff_salt.data(), 32);
    expect(hd_salt > 64,
           "PROP-4: salt sensitivity (~100-128 bits diff for 1-char change)");
    std::printf("  salt Hamming distance: %zu bits (of 256)\n", hd_salt);

    auto h160_a = compute_hash160(p1, s1);
    expect(h160_a.size() == 20,
           "PROP-5a: compute_hash160 returns 20 bytes");

    auto privkey = derive_key(p1, s1);
    auto h160_b = collider::cpu::compute_hash160(privkey.data());
    expect(h160_a == h160_b,
           "PROP-5b: compute_hash160 == derive_key + cpu::compute_hash160 pipeline");

    expect(verify(p1, s1, h160_a),
           "PROP-6a: verify() accepts correct (passphrase, salt) for an h160");
    expect(!verify("wrong passphrase", s1, h160_a),
           "PROP-6b: verify() rejects an incorrect passphrase");

    std::printf("\nSummary: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
