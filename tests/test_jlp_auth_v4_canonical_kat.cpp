/**
 * test_jlp_auth_v4_canonical_kat.cpp -- B1 wire-v4 (2026-05-23).
 *
 * Pins the C++ JLPPoolClient::build_auth_canonical_message output
 * against the byte string produced by the Python server's
 * JLPProtocol.build_auth_canonical_message for an identical input.
 * The two must agree byte-for-byte; otherwise the C++ client signs a
 * different message than the server verifies, and every v4 AUTH
 * silently fails on production.
 *
 * The Python reference bytes were captured by running:
 *
 *   python -c "from src.jlp_protocol import JLPProtocol, PROTOCOL_VERSION_V4 ; \
 *              from src.auth_v4 import encode_p2wpkh_address, hash160 ; \
 *              from coincurve import PrivateKey ; \
 *              priv = PrivateKey(bytes.fromhex('c0' * 31 + '01')) ; \
 *              pub = priv.public_key.format(compressed=True) ; \
 *              name = encode_p2wpkh_address(hash160(pub), 'bc') ; \
 *              canonical = JLPProtocol.build_auth_canonical_message( \
 *                  PROTOCOL_VERSION_V4, 1700000000123, bytes(range(16)), name) ; \
 *              print(canonical.hex())"
 *
 * Output captured 2026-05-23.
 */

#include "../src/pool/jlp_pool_client.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

using collider::pool::auth_v4::build_canonical_message;
using collider::pool::jlp_wire::PROTOCOL_VERSION_V4;

namespace {

std::string to_hex(const uint8_t* p, size_t n) {
    static const char* H = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        s.push_back(H[p[i] >> 4]);
        s.push_back(H[p[i] & 0xf]);
    }
    return s;
}

}  // namespace

int main() {
    std::cout << "=== test_jlp_auth_v4_canonical_kat (B1 wire-v4) ===\n";

    const std::string name =
        "bc1q2hs02hlnt27w33a9r78yx435m2gagv76mxqag9";
    const uint64_t ts = 1700000000123ULL;
    uint8_t nonce[16];
    for (int i = 0; i < 16; ++i) nonce[i] = static_cast<uint8_t>(i);

    auto canonical = build_canonical_message(
        static_cast<uint8_t>(PROTOCOL_VERSION_V4), ts, nonce, name);

    const std::string expected_hex =
        "434f4c4c494445522d574f524b45522d415554482d76310a"  // "COLLIDER-WORKER-AUTH-v1\n"
        "04"                                                  // proto_ver = 4
        "7b68e5cf8b010000"                                    // ts_ms = 1700000000123 LE u64 (8 bytes)
        "000102030405060708090a0b0c0d0e0f"                    // nonce (16 bytes starting at 0x00)
        "2a"                                                  // name_len = 42
        "626331713268733032686c6e743237773333613972373879783433356d326761677637366d7871616739";  // name

    std::string got = to_hex(canonical.data(), canonical.size());
    if (got != expected_hex) {
        std::cerr << "[FAIL] canonical message mismatch:\n"
                  << "  got  " << got << "\n"
                  << "  want " << expected_hex << "\n";
        return 1;
    }
    std::cout << "[PASS] canonical message matches Python "
              << "JLPProtocol.build_auth_canonical_message (len="
              << canonical.size() << ")\n";

    // Sanity: prefix byte-for-byte and the proto_ver byte at offset 24.
    if (canonical.size() < 25 || canonical[24] != 0x04) {
        std::cerr << "[FAIL] proto_ver byte not at offset 24\n";
        return 1;
    }
    std::cout << "[PASS] proto_ver=4 at offset 24\n";

    std::cout << "\n=== Result: ok ===\n";
    return 0;
}
