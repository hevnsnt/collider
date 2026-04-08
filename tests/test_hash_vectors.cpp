/**
 * SHA256 and RIPEMD160 Test Vectors
 *
 * Verifies hash implementations against NIST test vectors and known Bitcoin values.
 * This file provides CPU reference implementations to validate GPU kernels.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <array>
#include <vector>

// =============================================================================
// CPU Reference Implementations (for validation)
// =============================================================================

namespace cpu_ref {

// SHA256 Constants
static const uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint32_t sigma0(uint32_t x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
inline uint32_t sigma1(uint32_t x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }
inline uint32_t gamma0(uint32_t x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
inline uint32_t gamma1(uint32_t x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }

void sha256(const uint8_t* msg, size_t len, uint8_t* hash) {
    uint32_t H[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Prepare padded message
    size_t padded_len = ((len + 8) / 64 + 1) * 64;
    std::vector<uint8_t> padded(padded_len, 0);
    memcpy(padded.data(), msg, len);
    padded[len] = 0x80;

    // Length in bits (big endian)
    uint64_t bit_len = len * 8;
    for (int i = 0; i < 8; i++) {
        padded[padded_len - 1 - i] = (bit_len >> (i * 8)) & 0xFF;
    }

    // Process blocks
    for (size_t block = 0; block < padded_len; block += 64) {
        uint32_t W[64];

        // Load message block
        for (int i = 0; i < 16; i++) {
            W[i] = ((uint32_t)padded[block + i*4] << 24) |
                   ((uint32_t)padded[block + i*4 + 1] << 16) |
                   ((uint32_t)padded[block + i*4 + 2] << 8) |
                   ((uint32_t)padded[block + i*4 + 3]);
        }

        // Extend
        for (int i = 16; i < 64; i++) {
            W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16];
        }

        // Compress
        uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
        uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h + sigma1(e) + ch(e, f, g) + SHA256_K[i] + W[i];
            uint32_t t2 = sigma0(a) + maj(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        H[0] += a; H[1] += b; H[2] += c; H[3] += d;
        H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }

    // Output
    for (int i = 0; i < 8; i++) {
        hash[i*4] = (H[i] >> 24) & 0xFF;
        hash[i*4 + 1] = (H[i] >> 16) & 0xFF;
        hash[i*4 + 2] = (H[i] >> 8) & 0xFF;
        hash[i*4 + 3] = H[i] & 0xFF;
    }
}

// RIPEMD160 Constants
inline uint32_t rotl32(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

void ripemd160(const uint8_t* msg, size_t len, uint8_t* hash) {
    static const uint32_t K[5] = { 0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E };
    static const uint32_t KK[5] = { 0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000 };

    static const int R[80] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
        3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
        1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
        4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
    };
    static const int RR[80] = {
        5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
        6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
        15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
        8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
        12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
    };
    static const int S[80] = {
        11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
        7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
        11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
        11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
        9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
    };
    static const int SS[80] = {
        8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
        9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
        9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
        15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
        8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
    };

    uint32_t H[5] = { 0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0 };

    // Pad message
    size_t padded_len = ((len + 8) / 64 + 1) * 64;
    std::vector<uint8_t> padded(padded_len, 0);
    memcpy(padded.data(), msg, len);
    padded[len] = 0x80;

    // Length in bits (little endian)
    uint64_t bit_len = len * 8;
    for (int i = 0; i < 8; i++) {
        padded[padded_len - 8 + i] = (bit_len >> (i * 8)) & 0xFF;
    }

    // Process blocks
    for (size_t block = 0; block < padded_len; block += 64) {
        uint32_t X[16];
        for (int i = 0; i < 16; i++) {
            X[i] = ((uint32_t)padded[block + i*4]) |
                   ((uint32_t)padded[block + i*4 + 1] << 8) |
                   ((uint32_t)padded[block + i*4 + 2] << 16) |
                   ((uint32_t)padded[block + i*4 + 3] << 24);
        }

        uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4];
        uint32_t aa = H[0], bb = H[1], cc = H[2], dd = H[3], ee = H[4];

        for (int j = 0; j < 80; j++) {
            uint32_t f, ff;
            int round = j / 16;

            if (round == 0) { f = b ^ c ^ d; ff = bb ^ (cc | ~dd); }
            else if (round == 1) { f = (b & c) | (~b & d); ff = (bb & dd) | (cc & ~dd); }
            else if (round == 2) { f = (b | ~c) ^ d; ff = (bb | ~cc) ^ dd; }
            else if (round == 3) { f = (b & d) | (c & ~d); ff = (bb & cc) | (~bb & dd); }
            else { f = b ^ (c | ~d); ff = bb ^ cc ^ dd; }

            uint32_t t = rotl32(a + f + X[R[j]] + K[round], S[j]) + e;
            a = e; e = d; d = rotl32(c, 10); c = b; b = t;

            t = rotl32(aa + ff + X[RR[j]] + KK[round], SS[j]) + ee;
            aa = ee; ee = dd; dd = rotl32(cc, 10); cc = bb; bb = t;
        }

        uint32_t t = H[1] + c + dd;
        H[1] = H[2] + d + ee;
        H[2] = H[3] + e + aa;
        H[3] = H[4] + a + bb;
        H[4] = H[0] + b + cc;
        H[0] = t;
    }

    // Output (little endian)
    for (int i = 0; i < 5; i++) {
        hash[i*4] = H[i] & 0xFF;
        hash[i*4 + 1] = (H[i] >> 8) & 0xFF;
        hash[i*4 + 2] = (H[i] >> 16) & 0xFF;
        hash[i*4 + 3] = (H[i] >> 24) & 0xFF;
    }
}

void hash160(const uint8_t* msg, size_t len, uint8_t* hash) {
    uint8_t sha[32];
    sha256(msg, len, sha);
    ripemd160(sha, 32, hash);
}

} // namespace cpu_ref

// =============================================================================
// Test Vectors
// =============================================================================

struct TestVector {
    const char* name;
    const char* input_hex;
    const char* expected_hex;
};

// SHA256 Test Vectors (NIST)
static const TestVector SHA256_TESTS[] = {
    { "Empty string", "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
    { "abc", "616263", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
    { "448 bits", "6162636462636465636465666465666765666768666768696768696a68696a6b696a6b6c6a6b6c6d6b6c6d6e6c6d6e6f6d6e6f706e6f7071",
      "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
    { "satoshi", "7361746f736869", "da2876b3eb31edb4436fa4650673fc6f01f90de2f1793c4ec332b2387b09726f" },
};

// RIPEMD160 Test Vectors
static const TestVector RIPEMD160_TESTS[] = {
    { "Empty string", "", "9c1185a5c5e9fc54612808977ee8f548b2258d31" },
    { "a", "61", "0bdc9d2d256b3ee9daae347be6f4dc835a467ffe" },
    { "abc", "616263", "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc" },
    { "message digest", "6d65737361676520646967657374", "5d0689ef49d2fae572b881b123a85ffa21595f36" },
};

// Hash160 Test Vectors (Bitcoin pubkey -> address)
// These use known Bitcoin addresses to verify the full pipeline
static const TestVector HASH160_TESTS[] = {
    // Compressed pubkey for private key 1 (puzzle 1)
    { "Puzzle 1 pubkey", "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
      "751e76e8199196d454941c45d1b3a323f1433bd6" },
    // Compressed pubkey for private key 2 (puzzle 2)
    { "Puzzle 2 pubkey", "02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
      "91b24bf9f5288532960ac687abb035127b1d28a5" },
};

// =============================================================================
// Test Helpers
// =============================================================================

bool hex_to_bytes(const char* hex, uint8_t* bytes, size_t* len) {
    *len = strlen(hex) / 2;
    for (size_t i = 0; i < *len; i++) {
        unsigned int val;
        if (sscanf(hex + i*2, "%2x", &val) != 1) return false;
        bytes[i] = (uint8_t)val;
    }
    return true;
}

void bytes_to_hex(const uint8_t* bytes, size_t len, char* hex) {
    for (size_t i = 0; i < len; i++) {
        sprintf(hex + i*2, "%02x", bytes[i]);
    }
    hex[len*2] = '\0';
}

bool compare_hex(const uint8_t* result, size_t len, const char* expected_hex) {
    char result_hex[128];
    bytes_to_hex(result, len, result_hex);
    return strcmp(result_hex, expected_hex) == 0;
}

// =============================================================================
// Main Test Runner
// =============================================================================

int main() {
    int passed = 0, failed = 0;
    uint8_t input[256], result[32];
    size_t input_len;

    std::cout << "=== SHA256 Test Vectors ===\n";
    for (const auto& test : SHA256_TESTS) {
        hex_to_bytes(test.input_hex, input, &input_len);
        cpu_ref::sha256(input, input_len, result);

        if (compare_hex(result, 32, test.expected_hex)) {
            std::cout << "  PASS: " << test.name << "\n";
            passed++;
        } else {
            char got[65];
            bytes_to_hex(result, 32, got);
            std::cout << "  FAIL: " << test.name << "\n";
            std::cout << "    Expected: " << test.expected_hex << "\n";
            std::cout << "    Got:      " << got << "\n";
            failed++;
        }
    }

    std::cout << "\n=== RIPEMD160 Test Vectors ===\n";
    for (const auto& test : RIPEMD160_TESTS) {
        hex_to_bytes(test.input_hex, input, &input_len);
        cpu_ref::ripemd160(input, input_len, result);

        if (compare_hex(result, 20, test.expected_hex)) {
            std::cout << "  PASS: " << test.name << "\n";
            passed++;
        } else {
            char got[41];
            bytes_to_hex(result, 20, got);
            std::cout << "  FAIL: " << test.name << "\n";
            std::cout << "    Expected: " << test.expected_hex << "\n";
            std::cout << "    Got:      " << got << "\n";
            failed++;
        }
    }

    std::cout << "\n=== Hash160 (Bitcoin) Test Vectors ===\n";
    for (const auto& test : HASH160_TESTS) {
        hex_to_bytes(test.input_hex, input, &input_len);
        cpu_ref::hash160(input, input_len, result);

        if (compare_hex(result, 20, test.expected_hex)) {
            std::cout << "  PASS: " << test.name << "\n";
            passed++;
        } else {
            char got[41];
            bytes_to_hex(result, 20, got);
            std::cout << "  FAIL: " << test.name << "\n";
            std::cout << "    Expected: " << test.expected_hex << "\n";
            std::cout << "    Got:      " << got << "\n";
            failed++;
        }
    }

    std::cout << "\n=== Results ===\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";

    return failed > 0 ? 1 : 0;
}
