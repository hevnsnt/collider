/**
 * WarpWallet Implementation
 *
 * WarpWallet is a deterministic brain wallet that uses scrypt + PBKDF2-SHA256
 * for key derivation, making it resistant to GPU/ASIC brute-force attacks.
 *
 * Algorithm:
 *   s1 = scrypt(key=passphrase+"\x01", salt=salt+"\x01", N=2^18, r=8, p=1, dkLen=32)
 *   s2 = pbkdf2(key=passphrase+"\x02", salt=salt+"\x02", c=2^16, dkLen=32, prf=SHA256)
 *   privkey = s1 XOR s2
 *
 * Reference: https://keybase.io/warp
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <array>
#include <vector>

#include "crypto_cpu.hpp"

namespace collider {
namespace warpwallet {

// ============================================================================
// HMAC-SHA256 for PBKDF2
// ============================================================================

class HMAC_SHA256 {
public:
    static constexpr size_t BLOCK_SIZE = 64;
    static constexpr size_t HASH_SIZE = 32;

    static std::array<uint8_t, HASH_SIZE> compute(
        const uint8_t* key, size_t key_len,
        const uint8_t* data, size_t data_len
    ) {
        std::array<uint8_t, BLOCK_SIZE> k_ipad = {};
        std::array<uint8_t, BLOCK_SIZE> k_opad = {};

        // If key is longer than block size, hash it
        if (key_len > BLOCK_SIZE) {
            auto hashed_key = cpu::SHA256::hash(key, key_len);
            std::memcpy(k_ipad.data(), hashed_key.data(), HASH_SIZE);
            std::memcpy(k_opad.data(), hashed_key.data(), HASH_SIZE);
        } else {
            std::memcpy(k_ipad.data(), key, key_len);
            std::memcpy(k_opad.data(), key, key_len);
        }

        // XOR key with ipad and opad constants
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
            k_ipad[i] ^= 0x36;
            k_opad[i] ^= 0x5c;
        }

        // Inner hash: SHA256(k_ipad || data)
        std::vector<uint8_t> inner_input(BLOCK_SIZE + data_len);
        std::memcpy(inner_input.data(), k_ipad.data(), BLOCK_SIZE);
        std::memcpy(inner_input.data() + BLOCK_SIZE, data, data_len);
        auto inner_hash = cpu::SHA256::hash(inner_input.data(), inner_input.size());

        // Outer hash: SHA256(k_opad || inner_hash)
        std::vector<uint8_t> outer_input(BLOCK_SIZE + HASH_SIZE);
        std::memcpy(outer_input.data(), k_opad.data(), BLOCK_SIZE);
        std::memcpy(outer_input.data() + BLOCK_SIZE, inner_hash.data(), HASH_SIZE);

        return cpu::SHA256::hash(outer_input.data(), outer_input.size());
    }
};

// ============================================================================
// PBKDF2-SHA256
// ============================================================================

/**
 * PBKDF2 with HMAC-SHA256
 *
 * @param password  The password
 * @param salt      The salt
 * @param c         Iteration count
 * @param dkLen     Desired key length
 * @return          Derived key
 */
inline std::vector<uint8_t> pbkdf2_sha256(
    const uint8_t* password, size_t password_len,
    const uint8_t* salt, size_t salt_len,
    uint32_t c,
    size_t dkLen
) {
    const size_t hLen = 32;  // SHA256 output size
    size_t num_blocks = (dkLen + hLen - 1) / hLen;

    std::vector<uint8_t> result(dkLen);
    std::vector<uint8_t> salt_with_index(salt_len + 4);
    std::memcpy(salt_with_index.data(), salt, salt_len);

    for (size_t block = 1; block <= num_blocks; block++) {
        // Append block index (big-endian)
        salt_with_index[salt_len + 0] = (block >> 24) & 0xff;
        salt_with_index[salt_len + 1] = (block >> 16) & 0xff;
        salt_with_index[salt_len + 2] = (block >> 8) & 0xff;
        salt_with_index[salt_len + 3] = block & 0xff;

        // U_1 = PRF(password, salt || INT_32_BE(i))
        auto U = HMAC_SHA256::compute(password, password_len,
                                       salt_with_index.data(), salt_with_index.size());

        std::array<uint8_t, 32> T = U;

        // U_j = PRF(password, U_{j-1}), T = T XOR U_j
        for (uint32_t j = 2; j <= c; j++) {
            U = HMAC_SHA256::compute(password, password_len, U.data(), U.size());
            for (size_t k = 0; k < hLen; k++) {
                T[k] ^= U[k];
            }
        }

        // Copy T to result
        size_t offset = (block - 1) * hLen;
        size_t to_copy = std::min(hLen, dkLen - offset);
        std::memcpy(result.data() + offset, T.data(), to_copy);
    }

    return result;
}

// ============================================================================
// Scrypt
// ============================================================================

namespace detail {

/**
 * Salsa20/8 core function
 */
inline void salsa20_8(uint32_t B[16]) {
    uint32_t x[16];
    std::memcpy(x, B, 64);

    auto R = [](uint32_t a, int b) -> uint32_t {
        return (a << b) | (a >> (32 - b));
    };

    for (int i = 0; i < 4; i++) {
        x[ 4] ^= R(x[ 0] + x[12],  7);
        x[ 8] ^= R(x[ 4] + x[ 0],  9);
        x[12] ^= R(x[ 8] + x[ 4], 13);
        x[ 0] ^= R(x[12] + x[ 8], 18);

        x[ 9] ^= R(x[ 5] + x[ 1],  7);
        x[13] ^= R(x[ 9] + x[ 5],  9);
        x[ 1] ^= R(x[13] + x[ 9], 13);
        x[ 5] ^= R(x[ 1] + x[13], 18);

        x[14] ^= R(x[10] + x[ 6],  7);
        x[ 2] ^= R(x[14] + x[10],  9);
        x[ 6] ^= R(x[ 2] + x[14], 13);
        x[10] ^= R(x[ 6] + x[ 2], 18);

        x[ 3] ^= R(x[15] + x[11],  7);
        x[ 7] ^= R(x[ 3] + x[15],  9);
        x[11] ^= R(x[ 7] + x[ 3], 13);
        x[15] ^= R(x[11] + x[ 7], 18);

        x[ 1] ^= R(x[ 0] + x[ 3],  7);
        x[ 2] ^= R(x[ 1] + x[ 0],  9);
        x[ 3] ^= R(x[ 2] + x[ 1], 13);
        x[ 0] ^= R(x[ 3] + x[ 2], 18);

        x[ 6] ^= R(x[ 5] + x[ 4],  7);
        x[ 7] ^= R(x[ 6] + x[ 5],  9);
        x[ 4] ^= R(x[ 7] + x[ 6], 13);
        x[ 5] ^= R(x[ 4] + x[ 7], 18);

        x[11] ^= R(x[10] + x[ 9],  7);
        x[ 8] ^= R(x[11] + x[10],  9);
        x[ 9] ^= R(x[ 8] + x[11], 13);
        x[10] ^= R(x[ 9] + x[ 8], 18);

        x[12] ^= R(x[15] + x[14],  7);
        x[13] ^= R(x[12] + x[15],  9);
        x[14] ^= R(x[13] + x[12], 13);
        x[15] ^= R(x[14] + x[13], 18);
    }

    for (int i = 0; i < 16; i++) {
        B[i] += x[i];
    }
}

/**
 * scryptBlockMix (for r = 8)
 */
inline void block_mix(uint32_t* B, uint32_t* Y, int r) {
    // 2 * r blocks of 64 bytes each = 2 * r * 16 uint32_t
    const int block_count = 2 * r;
    uint32_t X[16];

    // Copy last block to X
    std::memcpy(X, &B[(block_count - 1) * 16], 64);

    for (int i = 0; i < block_count; i++) {
        // X = X XOR B[i]
        for (int j = 0; j < 16; j++) {
            X[j] ^= B[i * 16 + j];
        }

        // X = Salsa20/8(X)
        salsa20_8(X);

        // Even blocks go to first half, odd to second
        if (i % 2 == 0) {
            std::memcpy(&Y[(i / 2) * 16], X, 64);
        } else {
            std::memcpy(&Y[(r + i / 2) * 16], X, 64);
        }
    }
}

/**
 * Integerify: extracts j from block B[2*r-1]
 * Returns the first 64 bits of the last block, mod N
 */
inline uint64_t integerify(const uint32_t* B, int r, uint64_t N) {
    // B[2*r-1] starts at offset (2*r - 1) * 16
    const uint32_t* last_block = &B[(2 * r - 1) * 16];
    // Little-endian 64-bit
    uint64_t result = static_cast<uint64_t>(last_block[0]) |
                      (static_cast<uint64_t>(last_block[1]) << 32);
    return result & (N - 1);  // Assumes N is power of 2
}

}  // namespace detail

/**
 * scrypt key derivation function
 *
 * @param password      Password bytes
 * @param password_len  Password length
 * @param salt          Salt bytes
 * @param salt_len      Salt length
 * @param N             CPU/memory cost parameter (must be power of 2)
 * @param r             Block size parameter
 * @param p             Parallelism parameter
 * @param dkLen         Desired key length
 * @return              Derived key
 */
inline std::vector<uint8_t> scrypt(
    const uint8_t* password, size_t password_len,
    const uint8_t* salt, size_t salt_len,
    uint64_t N, uint32_t r, uint32_t p,
    size_t dkLen
) {
    // B = PBKDF2-SHA256(password, salt, 1, p * 128 * r)
    const size_t MFLen = 128 * r;  // Size of one block in bytes
    std::vector<uint8_t> B = pbkdf2_sha256(password, password_len,
                                            salt, salt_len,
                                            1, p * MFLen);

    // Allocate memory for V (N blocks of 128*r bytes each)
    const size_t block_words = 32 * r;  // 128*r bytes = 32*r uint32_t
    std::vector<uint32_t> V(N * block_words);
    std::vector<uint32_t> X(block_words);
    std::vector<uint32_t> Y(block_words);

    // Process each 128*r byte block
    for (uint32_t i = 0; i < p; i++) {
        // Convert B[i] to uint32_t array (little-endian)
        uint8_t* Bi = B.data() + i * MFLen;
        for (size_t j = 0; j < block_words; j++) {
            X[j] = static_cast<uint32_t>(Bi[j * 4 + 0]) |
                   (static_cast<uint32_t>(Bi[j * 4 + 1]) << 8) |
                   (static_cast<uint32_t>(Bi[j * 4 + 2]) << 16) |
                   (static_cast<uint32_t>(Bi[j * 4 + 3]) << 24);
        }

        // Phase 1: Fill V with expensive memory-hard operations
        for (uint64_t j = 0; j < N; j++) {
            std::memcpy(&V[j * block_words], X.data(), MFLen);
            detail::block_mix(X.data(), Y.data(), r);
            std::swap(X, Y);
        }

        // Phase 2: Random memory access
        for (uint64_t j = 0; j < N; j++) {
            uint64_t idx = detail::integerify(X.data(), r, N);
            for (size_t k = 0; k < block_words; k++) {
                X[k] ^= V[idx * block_words + k];
            }
            detail::block_mix(X.data(), Y.data(), r);
            std::swap(X, Y);
        }

        // Convert X back to bytes (little-endian)
        for (size_t j = 0; j < block_words; j++) {
            Bi[j * 4 + 0] = X[j] & 0xff;
            Bi[j * 4 + 1] = (X[j] >> 8) & 0xff;
            Bi[j * 4 + 2] = (X[j] >> 16) & 0xff;
            Bi[j * 4 + 3] = (X[j] >> 24) & 0xff;
        }
    }

    // Final PBKDF2 to produce output
    return pbkdf2_sha256(password, password_len,
                         B.data(), B.size(),
                         1, dkLen);
}

// ============================================================================
// WarpWallet Key Derivation
// ============================================================================

/**
 * WarpWallet key derivation
 *
 * Derives a Bitcoin private key from a passphrase and email salt.
 *
 * @param passphrase    The passphrase (without suffix)
 * @param salt          The salt (typically email address, without suffix)
 * @return              32-byte private key
 */
inline std::array<uint8_t, 32> derive_key(
    const std::string& passphrase,
    const std::string& salt
) {
    // WarpWallet parameters
    constexpr uint64_t SCRYPT_N = 1ULL << 18;  // 2^18 = 262144
    constexpr uint32_t SCRYPT_R = 8;
    constexpr uint32_t SCRYPT_P = 1;
    constexpr uint32_t PBKDF2_C = 1ULL << 16;  // 2^16 = 65536
    constexpr size_t KEY_LEN = 32;

    // Prepare inputs with suffixes
    std::string pass_scrypt = passphrase + "\x01";
    std::string salt_scrypt = salt + "\x01";
    std::string pass_pbkdf2 = passphrase + "\x02";
    std::string salt_pbkdf2 = salt + "\x02";

    // s1 = scrypt(key=passphrase+"\x01", salt=salt+"\x01", N=2^18, r=8, p=1, dkLen=32)
    auto s1 = scrypt(
        reinterpret_cast<const uint8_t*>(pass_scrypt.data()), pass_scrypt.size(),
        reinterpret_cast<const uint8_t*>(salt_scrypt.data()), salt_scrypt.size(),
        SCRYPT_N, SCRYPT_R, SCRYPT_P, KEY_LEN
    );

    // s2 = pbkdf2(key=passphrase+"\x02", salt=salt+"\x02", c=2^16, dkLen=32)
    auto s2 = pbkdf2_sha256(
        reinterpret_cast<const uint8_t*>(pass_pbkdf2.data()), pass_pbkdf2.size(),
        reinterpret_cast<const uint8_t*>(salt_pbkdf2.data()), salt_pbkdf2.size(),
        PBKDF2_C, KEY_LEN
    );

    // privkey = s1 XOR s2
    std::array<uint8_t, 32> privkey;
    for (size_t i = 0; i < KEY_LEN; i++) {
        privkey[i] = s1[i] ^ s2[i];
    }

    return privkey;
}

/**
 * Compute Hash160 from WarpWallet passphrase and salt
 *
 * @param passphrase    The passphrase
 * @param salt          The salt (email)
 * @return              20-byte Hash160
 */
inline std::array<uint8_t, 20> compute_hash160(
    const std::string& passphrase,
    const std::string& salt
) {
    auto privkey = derive_key(passphrase, salt);
    return cpu::compute_hash160(privkey.data());
}

/**
 * Verify a WarpWallet passphrase against a known Hash160
 *
 * @param passphrase    The passphrase to test
 * @param salt          The salt (email)
 * @param target_hash   The target 20-byte Hash160
 * @return              True if match
 */
inline bool verify(
    const std::string& passphrase,
    const std::string& salt,
    const std::array<uint8_t, 20>& target_hash
) {
    auto computed = compute_hash160(passphrase, salt);
    return computed == target_hash;
}

/**
 * WarpWallet Configuration for batch processing
 */
struct WarpWalletConfig {
    std::string salt;  // Email used as salt
    bool enabled = false;
};

/**
 * WarpWallet batch processor (CPU-based)
 *
 * Note: WarpWallet is intentionally CPU-intensive (memory-hard scrypt).
 * GPU acceleration for WarpWallet is limited due to memory requirements.
 * This implementation is for verification and small-scale testing.
 */
class WarpWalletProcessor {
public:
    explicit WarpWalletProcessor(const WarpWalletConfig& config)
        : config_(config) {}

    /**
     * Process a single passphrase
     */
    std::array<uint8_t, 20> process(const std::string& passphrase) const {
        return compute_hash160(passphrase, config_.salt);
    }

    /**
     * Check if a passphrase matches a target hash
     */
    bool check(const std::string& passphrase, const std::array<uint8_t, 20>& target) const {
        return verify(passphrase, config_.salt, target);
    }

    /**
     * Get the configured salt
     */
    const std::string& salt() const { return config_.salt; }

private:
    WarpWalletConfig config_;
};

}  // namespace warpwallet
}  // namespace collider
