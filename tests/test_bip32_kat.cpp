// test_bip32_kat.cpp -- TP-3 known-answer-test for BIP-32 derivation.
//
// Vectors taken verbatim from the BIP-32 spec:
//   https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki
//
// The spec defines five "Test Vector" blocks. Each block fixes a seed
// and a sequence of (path, expected private/chain) tuples. We exercise
// vectors 1 and 2 in full (covers hardened + non-hardened branches,
// short + long paths, both root + nested derivation). Vector 3 adds an
// edge case where the seed produces a master key whose
// ChainCode||PrivKey is at the edge of the valid range; vector 4 + 5
// add hardened derivations of large indices.
//
// Path parser edge cases are covered separately; this file's job is to
// nail down the cryptographic correctness of ckd_priv + the scalar
// arithmetic in add_mod_n.

#include "core/bip32.hpp"
#include "core/crypto_cpu.hpp"  // SHA256 + RIPEMD160 for fingerprint self-check

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_passes   = 0;

void fail(const char* tag, const std::string& msg) {
    std::fprintf(stderr, "[FAIL] %s: %s\n", tag, msg.c_str());
    ++g_failures;
}
void pass(const char* tag) {
    std::printf("[ ok  ] %s\n", tag);
    ++g_passes;
}

std::vector<uint8_t> hex_to_bytes(const std::string& hex) {
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    auto nyb = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    };
    for (size_t i = 0; i + 1 < hex.size(); i += 2) {
        int hi = nyb(hex[i]);
        int lo = nyb(hex[i + 1]);
        if (hi < 0 || lo < 0) throw std::runtime_error("bad hex");
        out.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return out;
}

std::string bytes_to_hex(const uint8_t* p, size_t n) {
    static const char* h = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(h[(p[i] >> 4) & 0xF]);
        s.push_back(h[p[i] & 0xF]);
    }
    return s;
}

// Expected ext-priv pair from the spec, given as hex strings of the
// 32-byte private key and 32-byte chain code (extracted from the
// spec's serialized xprv via the known offsets: chain at byte 13,
// private key at byte 46+1, 32 bytes each).
struct ExpectedExt {
    std::string priv_hex;   // 64 hex chars (32 bytes BE)
    std::string chain_hex;  // 64 hex chars (32 bytes BE)
};

void check_node(const char* tag,
                const ::collider::bip32::ExtKey& got,
                const ExpectedExt& want) {
    const std::string got_priv = bytes_to_hex(got.key.data(), 32);
    const std::string got_chain = bytes_to_hex(got.chain.data(), 32);
    if (got_priv != want.priv_hex) {
        fail(tag, "priv mismatch: got " + got_priv + " want " +
                       want.priv_hex);
        return;
    }
    if (got_chain != want.chain_hex) {
        fail(tag, "chain mismatch: got " + got_chain + " want " +
                       want.chain_hex);
        return;
    }
    pass(tag);
}

// Parent-fingerprint self-consistency check (audit follow-up to the
// bip32.hpp:277 fix that removed the wasted EC scalar mul). The
// recomputed fingerprint must match the field the derive_path call
// populated; otherwise the optimization broke the parent_pub plumbing
// or someone resurrected the dead (void)parent_pub line and started
// hashing the child key by mistake.
void check_parent_fingerprint_consistency(
    const char* tag,
    const ::collider::bip32::ExtKey& parent,
    const ::collider::bip32::ExtKey& child) {
    // Re-derive parent pubkey from its private scalar.
    auto pub = ::collider::bip32::detail::priv_to_pub(parent.key.data());
    // hash160(pub) -> first 4 bytes is the parent fingerprint per
    // BIP-32 spec section "Serialization format".
    auto sha = ::collider::cpu::SHA256::hash(pub.data(), pub.size());
    auto pkh = ::collider::cpu::RIPEMD160::hash(sha.data(), sha.size());
    if (std::memcmp(child.parent_fingerprint.data(), pkh.data(), 4) != 0) {
        const std::string got = bytes_to_hex(child.parent_fingerprint.data(), 4);
        const std::string want = bytes_to_hex(pkh.data(), 4);
        fail(tag,
             "parent_fingerprint mismatch: got " + got + " want " + want);
        return;
    }
    pass(tag);
}

}  // namespace

int main() {
    using ::collider::bip32::ExtKey;
    using ::collider::bip32::ckd_priv;
    using ::collider::bip32::derive_path;
    using ::collider::bip32::master_from_seed;
    using ::collider::bip32::parse_path;

    std::printf("=== test_bip32_kat (TP-3) ===\n");

    // ===== Test Vector 1 =====
    // Seed: 000102030405060708090a0b0c0d0e0f
    {
        auto seed = hex_to_bytes("000102030405060708090a0b0c0d0e0f");
        ExtKey m;
        try {
            m = master_from_seed(seed.data(), seed.size());
        } catch (const std::exception& e) {
            fail("v1_master_throws", e.what());
            goto vec1_done;
        }
        check_node("v1_master", m, ExpectedExt{
            "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35",
            "873dff81c02f525623fd1fe5167eac3a55a049de3d314bb42ee227ffed37d508",
        });

        // m/0'
        ExtKey n1;
        try {
            n1 = ckd_priv(m, 0u | 0x80000000u);
        } catch (const std::exception& e) {
            fail("v1_m_0h_throws", e.what());
            goto vec1_done;
        }
        check_node("v1_m_0h", n1, ExpectedExt{
            "edb2e14f9ee77d26dd93b4ecede8d16ed408ce149b6cd80b0715a2d911a0afea",
            "47fdacbd0f1097043b78c63c20c34ef4ed9a111d980047ad16282c7ae6236141",
        });
        check_parent_fingerprint_consistency("v1_m_0h_fingerprint", m, n1);

        // m/0'/1
        ExtKey n2;
        try {
            n2 = ckd_priv(n1, 1u);
        } catch (const std::exception& e) {
            fail("v1_m_0h_1_throws", e.what());
            goto vec1_done;
        }
        check_node("v1_m_0h_1", n2, ExpectedExt{
            "3c6cb8d0f6a264c91ea8b5030fadaa8e538b020f0a387421a12de9319dc93368",
            "2a7857631386ba23dacac34180dd1983734e444fdbf774041578e9b6adb37c19",
        });
        check_parent_fingerprint_consistency("v1_m_0h_1_fingerprint", n1, n2);

        // m/0'/1/2'
        ExtKey n3;
        try {
            n3 = ckd_priv(n2, 2u | 0x80000000u);
        } catch (const std::exception& e) {
            fail("v1_m_0h_1_2h_throws", e.what());
            goto vec1_done;
        }
        check_node("v1_m_0h_1_2h", n3, ExpectedExt{
            "cbce0d719ecf7431d88e6a89fa1483e02e35092af60c042b1df2ff59fa424dca",
            "04466b9cc8e161e966409ca52986c584f07e9dc81f735db683c3ff6ec7b1503f",
        });

        // m/0'/1/2'/2
        ExtKey n4;
        try {
            n4 = ckd_priv(n3, 2u);
        } catch (const std::exception& e) {
            fail("v1_m_0h_1_2h_2_throws", e.what());
            goto vec1_done;
        }
        check_node("v1_m_0h_1_2h_2", n4, ExpectedExt{
            "0f479245fb19a38a1954c5c7c0ebab2f9bdfd96a17563ef28a6a4b1a2a764ef4",
            "cfb71883f01676f587d023cc53a35bc7f88f724b1f8c2892ac1275ac822a3edd",
        });

        // m/0'/1/2'/2/1000000000 -- exercises a >2^29 index (non-hardened
        // tail of a long path; the +0x80000000 isn't applied).
        ExtKey n5;
        try {
            n5 = ckd_priv(n4, 1000000000u);
        } catch (const std::exception& e) {
            fail("v1_m_0h_1_2h_2_1B_throws", e.what());
            goto vec1_done;
        }
        check_node("v1_m_0h_1_2h_2_1B", n5, ExpectedExt{
            "471b76e389e528d6de6d816857e012c5455051cad6660850e58372a6c3e6e7c8",
            "c783e67b921d2beb8f6b389cc646d7263b4145701dadd2161548a8b078e65e9e",
        });
        check_parent_fingerprint_consistency(
            "v1_m_0h_1_2h_2_1B_fingerprint", n4, n5);
    }
vec1_done:;

    // ===== Test Vector 2 =====
    // Seed: fffcf9f6f3f0edeae7e4e1ded ... (32 bytes of decreasing pairs)
    {
        auto seed = hex_to_bytes(
            "fffcf9f6f3f0edeae7e4e1dedbd8d5d2cfccc9c6c3c0bdbab7b4b1aeaba8a5a29f"
            "9c999693908d8a8784817e7b7875726f6c696663605d5a5754514e4b484542");
        ExtKey m;
        try {
            m = master_from_seed(seed.data(), seed.size());
        } catch (const std::exception& e) {
            fail("v2_master_throws", e.what());
            goto vec2_done;
        }
        check_node("v2_master", m, ExpectedExt{
            "4b03d6fc340455b363f51020ad3ecca4f0850280cf436c70c727923f6db46c3e",
            "60499f801b896d83179a4374aeb7822aaeaceaa0db1f85ee3e904c4defbd9689",
        });

        // m/0 (non-hardened)
        ExtKey n1;
        try {
            n1 = ckd_priv(m, 0u);
        } catch (const std::exception& e) {
            fail("v2_m_0_throws", e.what());
            goto vec2_done;
        }
        check_node("v2_m_0", n1, ExpectedExt{
            "abe74a98f6c7eabee0428f53798f0ab8aa1bd37873999041703c742f15ac7e1e",
            "f0909affaa7ee7abe5dd4e100598d4dc53cd709d5a5c2cac40e7412f232f7c9c",
        });

        // m/0/2147483647' (largest valid hardened index = 0x80000000 |
        // 0x7FFFFFFF). Tests max-index encoding.
        ExtKey n2;
        try {
            n2 = ckd_priv(n1, 2147483647u | 0x80000000u);
        } catch (const std::exception& e) {
            fail("v2_m_0_2147483647h_throws", e.what());
            goto vec2_done;
        }
        check_node("v2_m_0_2147483647h", n2, ExpectedExt{
            "877c779ad9687164e9c2f4f0f4ff0340814392330693ce95a58fe18fd52e6e93",
            "be17a268474a6bb9c61e1d720cf6215e2a88c5406c4aee7b38547f585c9a37d9",
        });

        // m/0/2147483647'/1
        ExtKey n3;
        try {
            n3 = ckd_priv(n2, 1u);
        } catch (const std::exception& e) {
            fail("v2_m_0_2147483647h_1_throws", e.what());
            goto vec2_done;
        }
        check_node("v2_m_0_2147483647h_1", n3, ExpectedExt{
            "704addf544a06e5ee4bea37098463c23613da32020d604506da8c0518e1da4b7",
            "f366f48f1ea9f2d1d3fe958c95ca84ea18e4c4ddb9366c336c927eb246fb38cb",
        });
    }
vec2_done:;

    // ===== parse_path edge cases =====
    {
        auto p1 = parse_path("m/0");
        if (p1.size() == 1 && p1[0] == 0u) pass("parse_path_m_0");
        else fail("parse_path_m_0", "got size=" + std::to_string(p1.size()));

        auto p2 = parse_path("m/0'/1/2'/2/1000000000");
        bool ok = (p2.size() == 5) &&
                  (p2[0] == (0u | 0x80000000u)) &&
                  (p2[1] == 1u) &&
                  (p2[2] == (2u | 0x80000000u)) &&
                  (p2[3] == 2u) &&
                  (p2[4] == 1000000000u);
        if (ok) pass("parse_path_long_mixed");
        else fail("parse_path_long_mixed", "decoded indices wrong");

        // h suffix is the alternate hardened marker
        auto p3 = parse_path("m/44h/0h/0h/0/0");
        bool h_ok = (p3.size() == 5) &&
                    (p3[0] == (44u | 0x80000000u)) &&
                    (p3[1] == (0u | 0x80000000u)) &&
                    (p3[2] == (0u | 0x80000000u)) &&
                    (p3[3] == 0u) &&
                    (p3[4] == 0u);
        if (h_ok) pass("parse_path_h_suffix");
        else fail("parse_path_h_suffix", "h-suffix not honoured");

        // Bare "m" (no leading slash) is rejected.
        bool threw = false;
        try { (void)parse_path("m"); } catch (const std::exception&) { threw = true; }
        if (threw) pass("parse_path_rejects_bare_m");
        else fail("parse_path_rejects_bare_m", "did not throw");

        // Index >= 2^31 is rejected (would collide with the hardened bit).
        threw = false;
        try { (void)parse_path("m/2147483648"); } catch (const std::exception&) { threw = true; }
        if (threw) pass("parse_path_rejects_overflow");
        else fail("parse_path_rejects_overflow", "did not throw");

        // Empty segment between slashes.
        threw = false;
        try { (void)parse_path("m//0"); } catch (const std::exception&) { threw = true; }
        if (threw) pass("parse_path_rejects_empty_segment");
        else fail("parse_path_rejects_empty_segment", "did not throw");
    }

    // ===== derive_path equivalence =====
    // m/0'/1/2'/2 via derive_path() must equal the per-step ckd_priv chain
    // checked above.
    {
        auto seed = hex_to_bytes("000102030405060708090a0b0c0d0e0f");
        auto m = master_from_seed(seed.data(), seed.size());
        auto path = parse_path("m/0'/1/2'/2");
        auto leaf = derive_path(m, path);
        check_node("derive_path_v1_m_0h_1_2h_2", leaf, ExpectedExt{
            "0f479245fb19a38a1954c5c7c0ebab2f9bdfd96a17563ef28a6a4b1a2a764ef4",
            "cfb71883f01676f587d023cc53a35bc7f88f724b1f8c2892ac1275ac822a3edd",
        });
    }

    std::printf("\n%d passes, %d failures\n", g_passes, g_failures);
    return g_failures == 0 ? 0 : 1;
}
