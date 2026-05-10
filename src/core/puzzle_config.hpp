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
#include "byte_codec.hpp"

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

    // Total keys in range: 2^(N-1).
    //
    // Algebraic identity: for puzzle N the range is the inclusive
    // interval [2^(N-1), 2^N - 1], whose size is
    //
    //     2^N - 1 - 2^(N-1) + 1 = 2^(N-1)
    //
    // which is exactly range_start(). So returning range_start() is
    // correct for ALL canonical Bitcoin puzzles -- it is NOT a bug.
    // (Audited 2026-05-09: confirmed alongside the related
    // range_bits_from_be inclusive-end fix in src/core/byte_codec.hpp.)
    UInt256 range_size() const {
        return range_start();
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
            // ALL data below verified 2026-05-09 against the canonical
            // export at https://privatekeys.pw/puzzles/bitcoin-puzzle-tx
            // (status=solved and status=unsolved CSV endpoints). Pre-1.4
            // versions of this table contained:
            //   - wrong hash160 fields for puzzles 2-5 and 69-70
            //     (didn't match the listed addresses)
            //   - fabricated solution_hex for 67, 68, 69, 70, 75, 85,
            //     90, 95, 100, 105, 110, 115, 120, 125, 130
            //   - wrong target_address for puzzles 90, 95, 100, 105,
            //     110, 115, 120, 125, 131, 132, 133, 134, 135, 136, 137,
            //     140, 145 (mostly off by one or shifted)
            // Any code or test that branched on these fields produced
            // wrong results. Re-derive from the canonical CSV when
            // updating; do not edit one field without verifying the row.

            // Solved puzzles 1-5 (solved at puzzle creation; sequential
            // private keys 1, 3, 7, 8, 0x15 = 21).
            {1, 1, "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH", "751e76e8199196d454941c45d1b3a323f1433bd6", true, "0x1", 0.0, "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"},
            {2, 2, "1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb", "7dd65592d0ab2fe0d0257d571abf032cd9db93dc", true, "0x3", 0.0, "02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"},
            {3, 3, "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA", "5dedfbf9ea599dd4e3ca6a80b333c472fd0b3f69", true, "0x7", 0.0, "025cbdf0646e5db4eaa398f365f2ea7a0e3d419b7e0330e39ce92bddedcac4f9bc"},
            {4, 4, "1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e", "9652d86bedf43ad264362e6e6eba6eb764508127", true, "0x8", 0.0, "022f01e5e15cca351daff3843fb70f3c2f0a1bdd05e5af888a67784ef3e10a2a01"},
            {5, 5, "1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k", "8f9dff39a81ee4abcbad2ad8bafff090415a2be8", true, "0x15", 0.0, "02352bbf4a4cdd12564f93fa332ce333301d9ad40271f8107181340aef25be59d5"},

            // Solved puzzles 66-70, 75, 80, 85-130. These all had
            // outgoing transactions that exposed the public key, making
            // Kangaroo ECDLP cracking viable. Solve dates from the
            // canonical CSV are: 66=2024-09-12, 67=2025-02-21,
            // 68=2025-04-06, 69=2025-04-30, 70=2019-06-09,
            // 75=2019-06-10, 80=2019-06-11, 85=2019-06-17, 90=2019-07-01,
            // 95=2019-07-06, 100=2019-07-08, 105=2019-09-23,
            // 110=2020-05-30, 115=2020-06-16, 120=2023-02-27,
            // 125=2023-07-09, 130=2024-09-23.
            {66, 66, "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so", "20d45a6a762535700ce9e0b216e31994335db8a5", true, "0x2832ed74f2b5e35ee", 0.0, "024ee2be2d4e9f92d2f5a4a03058617dc45befe22938feed5b7a6b7282dd74cbdd"},
            {67, 67, "1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9", "739437bb3dd6d1983e66629c5f08c70e52769371", true, "0x730fc235c1942c1ae", 0.0, "0212209f5ec514a1580a2937bd833979d933199fc230e204c6cdc58872b7d46f75"},
            {68, 68, "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ", "e0b8a2baee1b77fc703455f39d51477451fc8cfc", true, "0xbebb3940cd0fc1491", 0.0, "031fe02f1d740637a7127cdfe8a77a8a0cfc6435f85e7ec3282cb6243c0a93ba1b"},
            {69, 69, "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG", "61eb8a50c86b0584bb727dd65bed8d2400d6d5aa", true, "0x101d83275fb2bc7e0c", 0.0, "024babadccc6cfd5f0e5e7fd2a50aa7d677ce0aa16fdce26a0d0882eed03e7ba53"},
            {70, 70, "19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR", "5db8cda53a6a002db10365967d7f85d19e171b10", true, "0x349b84b6431a6c4ef1", 0.0, "0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483"},
            {75, 75, "1J36UjUByGroXcCvmj13U6uwaVv9caEeAt", "badf8b0d34289e679ec65c6c61d3a974353be5cf", true, "0x4c5ce114686a1336e07", 0.0, "03726b574f193e374686d8e12bc6e4142adeb06770e0a2856f5e4ad89f66044755"},
            {80, 80, "1BCf6rHUW6m3iH2ptsvnjgLruAiPQQepLe", "6fe5a36eef0684af0b91f3b6cfc972d68c4f6fab", true, "0xea1a5c66dcc11b5ad180", 0.0, "037e1238f7b1ce757df94faa9a2eb261bf0aeb9f84dbf81212104e78931c2a19dc"},
            {85, 85, "1Kh22PvXERd2xpTQk3ur6pPEqFeckCJfAr", "cd03c1e6268ce9b89e3c3eeab8d0f1b6e8cac281", true, "0x11720c4f018d51b8cebba8", 0.0, "0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a"},
            {90, 90, "1L12FHH2FHjvTviyanuiFVfmzCy46RRATU", "d06b6e206691295ec345782d7ea0686969d8674b", true, "0x2ce00bb2136a445c71e85bf", 0.0, "035c38bd9ae4b10e8a250857006f3cfd98ab15a6196d9f4dfd25bc7ecc77d788d5"},
            {95, 95, "19eVSDuizydXxhohGh8Ki9WY9KsHdSwoQC", "5ed822125365274262191d2b77e88d436dd56d88", true, "0x527a792b183c7f64a0e8b1f4", 0.0, "02967a5905d6f3b420959a02789f96ab4c3223a2c4d2762f817b7895c5bc88a045"},
            {100, 100, "1KCgMv8fo2TPBpddVi9jqmMmcne9uSNJ5F", "c7a7b23f6bd98b8aaf527beb724dda9460b1bc6e", true, "0xaf55fc59c335c8ec67ed24826", 0.0, "03d2063d40402f030d4cc71331468827aa41a8a09bd6fd801ba77fb64f8e67e617"},
            {105, 105, "1CMjscKB3QW7SDyQ4c3C3DEUHiHRhiZVib", "7c957db6fdd0733bb83bc6d6d747711263ba50b0", true, "0x16f14fc2054cd87ee6396b33df3", 0.0, "03bcf7ce887ffca5e62c9cabbdb7ffa71dc183c52c04ff4ee5ee82e0c55c39d77b"},
            {110, 110, "12JzYkkN76xkwvcPT6AWKZtGX6w2LAgsJg", "0e5f3c406397442996825fd395543514fd06f207", true, "0x35c0d7234df7deb0f20cf7062444", 0.0, "0309976ba5570966bf889196b7fdf5a0f9a1e9ab340556ec29f8bb60599616167d"},
            {115, 115, "1NLbHuJebVwUZ1XqDjsAyfTRUPwDQbemfv", "ea0f2b7576bd098921fce9bfebe37f6383e639a4", true, "0x60f4d11574f5deee49961d9609ac6", 0.0, "0248d313b0398d4923cdca73b8cfa6532b91b96703902fc8b32fd438a3b7cd7f55"},
            {120, 120, "17s2b9ksz5y7abUm92cHwG8jEPCzK3dLnT", "4b46e10a541aeec6be3fac709c256fb7da69308e", true, "0xb10f22572c497a836ea187f2e1fc23", 0.0, "02ceb6cbbcdbdf5ef7150682150f4ce2c6f4807b349827dcdbdd1f2efa885a2630"},
            {125, 125, "1PXAyUB8ZoH3WD8n5zoAthYjN15yN5CVq5", "f7079256aa027dc437cbb539f955472416725fc8", true, "0x1c533b6bb7f0804e09960225e44877ac", 0.0, "0233709eb11e0d4439a729f21c2c443dedb727528229713f0065721ba8fa46f00e"},
            {130, 130, "1Fo65aKq8s8iquMt6weF1rku1moWVEd5Ua", "a24922852051a9002ebf4c864a55acb75bb4cf75", true, "0x33e7665705359f04f28b88cf897c603c9", 0.0, "03633cbe3ec02b9401c5effa144c5b4d22f87940259634858fc7e59b1c09937852"},

            // UNSOLVED 71-74. Plain addresses with no outgoing tx, so
            // no public key exposed -- pure brute force only, Kangaroo
            // not viable.
            {71, 71, "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU", "f6f5431d25bbf7b12e8add9af5e3475c44a0a5b8", false, "", 7.1, ""},
            {72, 72, "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR", "bf7413e8df4e7a34ce9dc13e2f2648783ec54adb", false, "", 7.2, ""},
            {73, 73, "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4", "105b7f253f0ebd7843adaebbd805c944bfb863e4", false, "", 7.3, ""},
            {74, 74, "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv", "9f1adb20baeacc38b3f49f3df6906a0e48f2df3d", false, "", 7.4, ""},

            // UNSOLVED puzzles 131-134: Public key UNKNOWN. Kangaroo
            // impossible, brute force only.
            {131, 131, "16zRPnT8znwq42q7XeMkZUhb1bKqgRogyy", "41b4b36a6c036568972380177eca2916cacd71de", false, "", 13.1, ""},
            {132, 132, "1KrU4dHE5WrW8rhWDsTRjR21r8t3dsrS3R", "cecd3ca4319651bd3afd1e23ab66e111ed38d16d", false, "", 13.2, ""},
            {133, 133, "17uDfp5r4n441xkgLFmhNoSW1KWp6xVLD",  "014e15e4ea6da460cc7835e262676baa37988e4f", false, "", 13.3, ""},
            {134, 134, "13A3JrvXmvg5w9XGvyyR4JEJqiLz8ZySY3", "17a5ebfaf62e73f149e33ba674836801f13a80b9", false, "", 13.4, ""},

            // UNSOLVED puzzles with KNOWN public keys (creator's
            // outgoing tx exposed pubkey). Kangaroo viable. These are
            // the live targets for v1.4.0 pool work.
            {135, 135, "16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v", "3b6f58a75a54bfd85d1bc6c51180fdc732992326", false, "", 13.5, "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"},
            {136, 136, "1UDHPdovvR985NrWSkdWQDEQ1xuRiTALq",  "05257be4b57ee43fc09762d5d3a9ad4a6e1a0364", false, "", 13.6, ""},
            {137, 137, "15nf31J46iLuK1ZkTnqHo7WgN5cARFK3RA", "3482f8986e13c018692053a784481c63a3554c9c", false, "", 13.7, ""},
            {140, 140, "1QKBaU6WAeycb3DbKbLBkX7vJiaS8r42Xo", "ffbb35a7bb9bbe16c1aa2534f7ff11d59c8e3d1a", false, "", 14.0, "031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640"},
            {145, 145, "19GpszRNUej5yYqxXoLnbZWKew3KdVLkXg", "5abf369388deb8072741b4eb43ef10fa9388a729", false, "", 14.5, "03afdda497369e219a2c1c369954a930e4d3740968e5e4352475bcffce3140dae5"},

            // Ultimate prizes (puzzle 150, 155, 160). Public keys KNOWN
            // (creator outgoing tx). btc_reward fields pre-1.4 were a
            // hand-typed "50.0" placeholder that didn't match reality;
            // canonical export shows 15, 15.5, 16 BTC respectively.
            // Addresses + hash160s also corrected from corrupt values.
            {150, 150, "1MUJSJYtGPVGkBCTqGspnxyHahpt5Te8jy", "e08c4d3bc9cf2b3e2cb88de2bfaa4fe8c7aa3f24", false, "", 15.0, "03137807790ea7dc6e97901c2bc87411f45ed74a5629315c4e4b03a0a102250c49"},
            {155, 155, "1AoeP37TmHdFh8uN72fu9AqgtLrUwcv2wJ", "6b8b7830f73c5bf9e8beb9f161ad82b3bde992e4", false, "", 15.5, "035cd1854cae45391ca4ec428cc7e6c7d9984424b954209a8eea197b9e364c05f6"},
            {160, 160, "1NBC8uXJy1GiJ6drkiZa1WuKn51ps7EPTv", "e84818e1bf7f699aa6e28ef9edfb582099099292", false, "", 16.0, "02e0a8b039282faf6fe0fd769cfbc4b6b4cf8758ba68220eac420e32b91ddfa673"},
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
            char buf[41];
            ::collider::hex_encode_lower(decoded.data() + 1, 20, buf);
            return std::string(buf, 40);

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
