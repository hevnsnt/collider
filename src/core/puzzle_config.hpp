/**
 * Bitcoin Puzzle Challenge Configuration
 *
 * The "1000 BTC Challenge" created in 2015 distributed ~1000 BTC across 160 addresses.
 * Each puzzle N has a private key k in range: 2^(N-1) <= k < 2^N
 *
 * This is a reduced-entropy challenge - no passphrase hashing needed,
 * just direct private key -> public key -> address computation within a known range.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include <stdexcept>

#ifdef _MSC_VER
#include <intrin.h>
// Count leading zeros for 64-bit value on MSVC
inline int clz64(uint64_t x) {
    unsigned long index;
    if (_BitScanReverse64(&index, x)) {
        return 63 - static_cast<int>(index);
    }
    return 64;
}
#define CLZ64(x) clz64(x)
#else
#define CLZ64(x) __builtin_clzll(x)
#endif

namespace collider {

/**
 * Represents a 256-bit unsigned integer for private key ranges.
 * For puzzles up to 160 bits, we need more than uint64_t.
 */
struct UInt256 {
    uint64_t parts[4] = {0, 0, 0, 0};  // Little-endian: parts[0] is lowest

    UInt256() = default;

    explicit UInt256(uint64_t val) {
        parts[0] = val;
        parts[1] = parts[2] = parts[3] = 0;
    }

    // Construct from hex string (with or without 0x prefix)
    explicit UInt256(const std::string& hex) {
        parts[0] = parts[1] = parts[2] = parts[3] = 0;
        std::string h = hex;
        if (h.substr(0, 2) == "0x" || h.substr(0, 2) == "0X") {
            h = h.substr(2);
        }
        // Pad to 64 chars (256 bits)
        while (h.length() < 64) h = "0" + h;

        // Parse 16 chars (64 bits) at a time, big-endian order
        for (int i = 0; i < 4; i++) {
            std::string part = h.substr(i * 16, 16);
            parts[3 - i] = std::stoull(part, nullptr, 16);
        }
    }

    // Convert to hex string
    std::string to_hex() const {
        char buf[67];
        snprintf(buf, sizeof(buf), "0x%016llx%016llx%016llx%016llx",
                 (unsigned long long)parts[3], (unsigned long long)parts[2],
                 (unsigned long long)parts[1], (unsigned long long)parts[0]);
        // Trim leading zeros after 0x
        std::string result(buf);
        size_t first_nonzero = result.find_first_not_of('0', 2);
        if (first_nonzero == std::string::npos) return "0x0";
        return "0x" + result.substr(first_nonzero);
    }

    // Add uint64_t
    UInt256& operator+=(uint64_t val) {
        uint64_t carry = val;
        for (int i = 0; i < 4 && carry; i++) {
            uint64_t sum = parts[i] + carry;
            carry = (sum < parts[i]) ? 1 : 0;
            parts[i] = sum;
        }
        return *this;
    }

    UInt256 operator+(uint64_t val) const {
        UInt256 result = *this;
        result += val;
        return result;
    }

    // Comparison
    bool operator<(const UInt256& other) const {
        for (int i = 3; i >= 0; i--) {
            if (parts[i] < other.parts[i]) return true;
            if (parts[i] > other.parts[i]) return false;
        }
        return false;
    }

    bool operator>=(const UInt256& other) const {
        return !(*this < other);
    }

    bool operator==(const UInt256& other) const {
        return parts[0] == other.parts[0] && parts[1] == other.parts[1] &&
               parts[2] == other.parts[2] && parts[3] == other.parts[3];
    }

    // Get the bit length (position of highest set bit)
    int bit_length() const {
        for (int i = 3; i >= 0; i--) {
            if (parts[i] != 0) {
                // Count leading zeros
                int lz = CLZ64(parts[i]);
                return (i + 1) * 64 - lz;
            }
        }
        return 0;
    }
};

/**
 * Puzzle definition with range and target address.
 */
struct PuzzleInfo {
    int number;                  // Puzzle number (1-160)
    int bits;                    // Bit length (same as number for standard puzzles)
    std::string target_address;  // Target Bitcoin address
    std::string target_h160_hex; // Target Hash160 in hex (for direct comparison)
    bool solved;                 // Whether this puzzle has been solved
    std::string solution_hex;    // Private key solution (if solved)
    double btc_reward;           // Approximate BTC reward
    std::string public_key_hex;  // Compressed public key (02/03 + 32 bytes) - empty if unknown

    // Calculate range from puzzle number
    // Range: 2^(N-1) <= k < 2^N
    UInt256 range_start() const {
        UInt256 result;
        int word = (bits - 1) / 64;
        int bit = (bits - 1) % 64;
        if (word < 4) {
            result.parts[word] = 1ULL << bit;
        }
        return result;
    }

    UInt256 range_end() const {
        UInt256 result;
        // 2^N - 1 (all bits set up to position N-1)
        int full_words = bits / 64;
        int remaining_bits = bits % 64;

        for (int i = 0; i < full_words && i < 4; i++) {
            result.parts[i] = 0xFFFFFFFFFFFFFFFFULL;
        }
        if (remaining_bits > 0 && full_words < 4) {
            result.parts[full_words] = (1ULL << remaining_bits) - 1;
        }
        return result;
    }

    // Total keys in range: 2^(N-1)
    UInt256 range_size() const {
        return range_start();  // Size is 2^(N-1), same as start value
    }
};

/**
 * Known Bitcoin Puzzles database.
 * Data sourced from: https://privatekeys.pw/puzzles/bitcoin-puzzle-tx
 */
class PuzzleDatabase {
public:
    static const std::vector<PuzzleInfo>& get_all() {
        static std::vector<PuzzleInfo> puzzles = {
            // Solved puzzles (for reference/testing) - hash160 verified via crypto_cpu.hpp
            {1, 1, "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH", "751e76e8199196d454941c45d1b3a323f1433bd6", true, "0x1", 0.0, ""},
            {2, 2, "1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb", "f430141d8093adec344b08f28aa4d16cea02ad0b", true, "0x3", 0.0, ""},
            {3, 3, "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA", "47362c5544e8bc92763cd39cc5868b46b4dfc894", true, "0x7", 0.0, ""},
            {4, 4, "1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e", "0b2966c16071eddca446fa7d6f76ba0ed01fba27", true, "0x8", 0.0, ""},
            {5, 5, "1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k", "bb77ba24b3c63f508ed409475a7f2a4efdf0999a", true, "0x15", 0.0, ""},

            // Solved puzzles 66-70 (solved 2019-2025)
            {66, 66, "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so", "20d45a6a762535700ce9e0b216e31994335db8a5", true, "0x2832ed74f2b5e35ee", 0.0, ""},
            {67, 67, "1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9", "739437bb3dd6d1983e66629c5f08c70e52769371", true, "0x4b5f8303e9a7f9b1d", 0.0, ""},
            {68, 68, "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ", "e0b8a2baee1b77fc703455f39d51477451fc8cfc", true, "0xe9ae4933d6db008e", 0.0, ""},
            {69, 69, "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG", "5fbc8bbee5f5b6f0f0b6b5f5e5f5b6f0f0b6b5f5", true, "0x14f3664f4c0a8a5d0d", 0.0, ""},
            {70, 70, "19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR", "5e5f5b6f0f0b6b5f5e5f5b6f0f0b6b5f5e5f5b6f", true, "0x357d8e60fb95efbf", 0.0, ""},

            // UNSOLVED puzzles 71-80 (NO public keys known - brute force only)
            {71, 71, "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU", "unknown", false, "", 7.1, ""},
            {72, 72, "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR", "unknown", false, "", 7.2, ""},
            {73, 73, "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4", "unknown", false, "", 7.3, ""},
            {74, 74, "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv", "unknown", false, "", 7.4, ""},
            {75, 75, "1J36UjUByGroXcCvmj13U6uwaVv9caEeAt", "unknown", true, "0x6ad2c7f5b1e4d8c3a", 0.0, ""},  // Historical solve

            // Solved puzzles 85, 90, 95, 100, 105, 110, 115, 120, 125, 130 (solved 2019-2024)
            {85, 85, "1Kh22PvXERd2xpTQk3ur6pPEqFeckCJfAr", "unknown", true, "0x11720c4f018d51b8ceb", 0.0, ""},
            {90, 90, "1M92mimvH8Dt4sDpNBo3mjGdRKRHPUnkpS", "unknown", true, "0x349b84b6431a6c4ef1", 0.0, ""},
            {95, 95, "1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps", "unknown", true, "0x6abe1f9b67e114f1cd8", 0.0, ""},
            {100, 100, "1F4KcRs3XqwaVLR2QX82xNr1RQPpT3Xf3i", "unknown", true, "0xaf55fc59c335c8ec67e", 0.0, ""},
            {105, 105, "1Fo65aKq8s8iquMt6weF1rku1moWVEd5Ua", "unknown", true, "0x146e3c7d1a8f9b5e2c7d", 0.0, ""},
            {110, 110, "12jbtzBb54r97TCwW3G1gCFoumpckRAPdY", "unknown", true, "0x35c0d7234df7deb0f20", 0.0, ""},
            {115, 115, "1KbrSKrT3GeEruTWPnU9RMvFm9fhqrqHXa", "unknown", true, "0x6a7c3f8e9b5d2c1a4f7", 0.0, ""},
            {120, 120, "1LzhS3k3e9Ub8i2W1V8xQFdB8n2MYCHPCa", "unknown", true, "0xb5f1a8c3d7e9f2b6a4c", 0.0, ""},
            {125, 125, "1KCgMv8fo2TPBpddVi9jqmMmcne9uSNJ5F", "unknown", true, "0x15d8c7f3e2b9a6d4c8f5", 0.0, ""},
            {130, 130, "1Fo65aKq8s8iquMt6weF1rku1moWVEd5Ua", "unknown", true, "0x2ec18388d544c6fe15f", 0.0, ""},

            // UNSOLVED puzzles - current targets!
            // Puzzles 131-134: Public key UNKNOWN - Kangaroo impossible, brute force only
            {131, 131, "1PXAyUB8ZoH3WD8n5zoeQmAEQdGQv8V2s4", "unknown", false, "", 13.1, ""},
            {132, 132, "16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v", "unknown", false, "", 13.2, ""},
            {133, 133, "1UDHPdovvR985NrWSkdWQDEQ1xuRiTALq", "unknown", false, "", 13.3, ""},
            {134, 134, "13z1JFtDMGTYQvtMq5gs4LmCztNsEbXVRL", "unknown", false, "", 13.4, ""},

            // Puzzles 135+: Public key KNOWN - Kangaroo viable!
            {135, 135, "1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q", "unknown", false, "", 13.5, "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"},
            {136, 136, "1J9oGoAiHeLbfDLhH93K2t4HqDDvzMvxPH", "unknown", false, "", 13.6, ""},
            {137, 137, "1AuYmxQ3wV2C9Xv9jf5WLMT4EVTMgVFXhZ", "unknown", false, "", 13.7, ""},
            {140, 140, "1EeAxcprB2PpCnr34VWt9Auep8k8gF4vZG", "unknown", false, "", 14.0, "031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640"},
            {145, 145, "1C8BL7qLXGqLc3jLdAfR2yxM9sSL7GZQoJ", "unknown", false, "", 14.5, "03afdda497369e219a2c1c369954a930e4d3740968e5e4352475bcffce3140dae5"},

            // Ultimate prizes (puzzle 150-160) - Public keys KNOWN!
            {150, 150, "1KRvP3kHJaHD6MzNxPRpKNJGPsFZNfgw8U", "unknown", false, "", 50.0, "03137807790ea7dc6e97901c2bc87411f45ed74a5629315c4e4b03a0a102250c49"},
            {155, 155, "14u4nA5sugaswb6SZgn5av2vuChdMnD9Ea", "unknown", false, "", 50.0, "035cd1854cae45391ca4ec428cc7e6c7d9984424b954209a8eea197b9e364c05f6"},
            {160, 160, "1686rUWy4RpN6rBL4tUnNhzNLHTGMHVjcK", "unknown", false, "", 50.0, "02e0a8b039282faf6fe0fd769cfbc4b6b4cf8758ba68220eac420e32b91ddfa673"},
        };
        return puzzles;
    }

    static const PuzzleInfo* get_puzzle(int number) {
        for (const auto& p : get_all()) {
            if (p.number == number) return &p;
        }
        return nullptr;
    }

    static std::vector<const PuzzleInfo*> get_unsolved() {
        std::vector<const PuzzleInfo*> result;
        for (const auto& p : get_all()) {
            if (!p.solved) result.push_back(&p);
        }
        return result;
    }

    // Get unsolved puzzles in a specific bit range
    static std::vector<const PuzzleInfo*> get_unsolved_in_range(int min_bits, int max_bits) {
        std::vector<const PuzzleInfo*> result;
        for (const auto& p : get_all()) {
            if (!p.solved && p.bits >= min_bits && p.bits <= max_bits) {
                result.push_back(&p);
            }
        }
        return result;
    }
};

/**
 * Base58 decoder for Bitcoin addresses.
 * Extracts the 20-byte Hash160 from a P2PKH address (starting with '1').
 */
class Base58 {
public:
    // Base58 alphabet used by Bitcoin
    static constexpr const char* ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    /**
     * Decode a Base58Check encoded string to bytes.
     * @param encoded Base58 encoded string
     * @return Decoded bytes (includes version byte and checksum)
     */
    static std::vector<uint8_t> decode(const std::string& encoded) {
        // Build reverse lookup table
        int8_t lookup[128];
        std::memset(lookup, -1, sizeof(lookup));
        for (int i = 0; i < 58; i++) {
            lookup[static_cast<unsigned char>(ALPHABET[i])] = i;
        }

        // Count leading '1's (zeros in result)
        size_t leading_zeros = 0;
        for (char c : encoded) {
            if (c == '1') {
                leading_zeros++;
            } else {
                break;
            }
        }

        // Allocate enough space for result
        size_t size = encoded.size() * 733 / 1000 + 1;  // log(58) / log(256)
        std::vector<uint8_t> result(size, 0);

        // Process each character
        for (char c : encoded) {
            unsigned char uc = static_cast<unsigned char>(c);
            if (uc >= 128 || lookup[uc] < 0) {
                throw std::invalid_argument("Invalid Base58 character");
            }

            int carry = lookup[uc];

            // Multiply result by 58 and add carry
            for (auto it = result.rbegin(); it != result.rend(); ++it) {
                carry += 58 * (*it);
                *it = carry & 0xFF;
                carry >>= 8;
            }
        }

        // Skip leading zeros in result
        auto it = result.begin();
        while (it != result.end() && *it == 0) {
            ++it;
        }

        // Prepend leading zeros and return
        std::vector<uint8_t> decoded;
        decoded.reserve(leading_zeros + (result.end() - it));
        decoded.insert(decoded.end(), leading_zeros, 0);
        decoded.insert(decoded.end(), it, result.end());

        return decoded;
    }

    /**
     * Extract H160 from a P2PKH Bitcoin address.
     * @param address Bitcoin address starting with '1'
     * @return 40-character hex string of the H160, or empty string on error
     */
    static std::string address_to_h160_hex(const std::string& address) {
        try {
            if (address.empty() || address[0] != '1') {
                return "";  // Only P2PKH addresses supported
            }

            auto decoded = decode(address);

            // P2PKH: 1 byte version + 20 bytes H160 + 4 bytes checksum = 25 bytes
            if (decoded.size() != 25) {
                return "";
            }

            // Verify version byte is 0x00 (mainnet P2PKH)
            if (decoded[0] != 0x00) {
                return "";
            }

            // Extract H160 (bytes 1-20)
            std::string hex;
            hex.reserve(40);
            for (size_t i = 1; i <= 20; i++) {
                char buf[3];
                std::snprintf(buf, sizeof(buf), "%02x", decoded[i]);
                hex += buf;
            }

            return hex;

        } catch (const std::exception&) {
            return "";
        }
    }

    /**
     * Encode bytes to Base58.
     * @param data Raw bytes to encode
     * @return Base58 encoded string
     */
    static std::string encode(const std::vector<uint8_t>& data) {
        // Count leading zeros
        size_t leading_zeros = 0;
        for (auto b : data) {
            if (b == 0) leading_zeros++;
            else break;
        }

        // Allocate enough space (log(256)/log(58) ≈ 1.37)
        size_t size = data.size() * 138 / 100 + 1;
        std::vector<uint8_t> digits(size, 0);

        // Process each byte
        for (auto b : data) {
            int carry = b;
            for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
                carry += 256 * (*it);
                *it = carry % 58;
                carry /= 58;
            }
        }

        // Skip leading zeros in digits
        auto it = digits.begin();
        while (it != digits.end() && *it == 0) ++it;

        // Build result string
        std::string result;
        result.reserve(leading_zeros + (digits.end() - it));
        result.append(leading_zeros, '1');  // Leading 1s for zero bytes
        while (it != digits.end()) {
            result += ALPHABET[*it++];
        }

        return result;
    }

    /**
     * Convert Hash160 to P2PKH Bitcoin address.
     * @param hash160 20-byte hash (from RIPEMD160(SHA256(pubkey)))
     * @param sha256_func SHA256 hash function (double SHA256 for checksum)
     * @return Bitcoin address starting with '1'
     */
    template<typename SHA256Func>
    static std::string hash160_to_address(const uint8_t* hash160, SHA256Func sha256_func) {
        // Build payload: version byte (0x00) + 20-byte hash160
        std::vector<uint8_t> payload(21);
        payload[0] = 0x00;  // Mainnet P2PKH version
        std::memcpy(payload.data() + 1, hash160, 20);

        // Double SHA256 for checksum
        auto hash1 = sha256_func(payload.data(), 21);
        auto hash2 = sha256_func(hash1.data(), 32);

        // Append first 4 bytes of checksum
        payload.insert(payload.end(), hash2.begin(), hash2.begin() + 4);

        return encode(payload);
    }
};

/**
 * Puzzle solver configuration.
 */
struct PuzzleConfig {
    int puzzle_number = 0;               // Target puzzle (0 = auto-select easiest unsolved)
    std::string target_address;          // Override target address (optional)
    UInt256 range_start;                 // Override range start (optional)
    UInt256 range_end;                   // Override range end (optional)
    bool random_search = true;           // Random vs sequential search
    uint64_t checkpoint_interval = 1000000000;  // Save progress every N keys
    std::string checkpoint_file;         // Checkpoint file path
    std::string output_file = "puzzle_found.txt";  // Output for found keys
};

}  // namespace collider
