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
#include <cstdio>
#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include <stdexcept>
#include <iostream>
#include <cctype>

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
 * Data sourced from: https://btcpuzzle.info/
 */
class PuzzleDatabase {
public:
    static const std::vector<PuzzleInfo>& get_all() {
        static std::vector<PuzzleInfo> puzzles = {
            // Solved puzzles 1-70 (all verified)
            {1, 1, "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH", "", true, "0x1", 0.0, ""},
            {2, 2, "1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb", "", true, "0x3", 0.0, ""},
            {3, 3, "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA", "", true, "0x7", 0.0, ""},
            {4, 4, "1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e", "", true, "0x8", 0.0, ""},
            {5, 5, "1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k", "", true, "0x15", 0.0, ""},
            {6, 6, "1PitScNLyp2HCygzadCh7FveTnfmpPbfp8", "", true, "0x31", 0.0, ""},
            {7, 7, "1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC", "", true, "0x4c", 0.0, ""},
            {8, 8, "1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK", "", true, "0xe0", 0.0, ""},
            {9, 9, "1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV", "", true, "0x1d3", 0.0, ""},
            {10, 10, "1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe", "", true, "0x202", 0.0, ""},
            {11, 11, "1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu", "", true, "0x483", 0.0, ""},
            {12, 12, "1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot", "", true, "0xa7b", 0.0, ""},
            {13, 13, "1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1", "", true, "0x1460", 0.0, ""},
            {14, 14, "1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk", "", true, "0x2930", 0.0, ""},
            {15, 15, "1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW", "", true, "0x68f3", 0.0, ""},
            {16, 16, "1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY", "", true, "0xc936", 0.0, ""},
            {17, 17, "1HduPEXZRdG26SUT5Yk83mLkPyjnZuJ7Bm", "", true, "0x1764f", 0.0, ""},
            {18, 18, "1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE", "", true, "0x3080d", 0.0, ""},
            {19, 19, "1NWmZRpHH4XSPwsW6dsS3nrNWfL1yrJj4w", "", true, "0x5749f", 0.0, ""},
            {20, 20, "1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum", "", true, "0xd2c55", 0.0, ""},
            {21, 21, "14oFNXucftsHiUMY8uctg6N487riuyXs4h", "", true, "0x1ba534", 0.0, ""},
            {22, 22, "1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv", "", true, "0x2de40f", 0.0, ""},
            {23, 23, "1L2GM8eE7mJWLdo3HZS6su1832NX2txaac", "", true, "0x556e52", 0.0, ""},
            {24, 24, "1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7", "", true, "0xdc2a04", 0.0, ""},
            {25, 25, "15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP", "", true, "0x1fa5ee5", 0.0, ""},
            {26, 26, "1JVnST957hGztonaWK6FougdtjxzHzRMMg", "", true, "0x340326e", 0.0, ""},
            {27, 27, "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k", "", true, "0x6ac3875", 0.0, ""},
            {28, 28, "12jbtzBb54r97TCwW3G1gCFoumpckRAPdY", "", true, "0xd916ce8", 0.0, ""},
            {29, 29, "19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT", "", true, "0x17e2551e", 0.0, ""},
            {30, 30, "1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps", "", true, "0x3d94cd64", 0.0, ""},
            {31, 31, "1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE", "", true, "0x7d4fe747", 0.0, ""},
            {32, 32, "1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR", "", true, "0xb862a62e", 0.0, ""},
            {33, 33, "187swFMjz1G54ycVU56B7jZFHFTNVQFDiu", "", true, "0x1a96ca8d8", 0.0, ""},
            {34, 34, "1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q", "", true, "0x34a65911d", 0.0, ""},
            {35, 35, "1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb", "", true, "0x4aed21170", 0.0, ""},
            {36, 36, "1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1", "", true, "0x9de820a7c", 0.0, ""},
            {37, 37, "14iXhn8bGajVWegZHJ18vJLHhntcpL4dex", "", true, "0x1757756a93", 0.0, ""},
            {38, 38, "1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2", "", true, "0x22382facd0", 0.0, ""},
            {39, 39, "122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8", "", true, "0x4b5f8303e9", 0.0, ""},
            {40, 40, "1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv", "", true, "0xe9ae4933d6", 0.0, ""},
            {41, 41, "1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E", "", true, "0x153869acc5b", 0.0, ""},
            {42, 42, "1E32GPWgDyeyQac4aJxm9HVoLrrEYPnM4N", "", true, "0x2a221c58d8f", 0.0, ""},
            {43, 43, "1PiFuqGpG8yGM5v6rNHWS3TjsG6awgEGA1", "", true, "0x6bd3b27c591", 0.0, ""},
            {44, 44, "1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF", "", true, "0xe02b35a358f", 0.0, ""},
            {45, 45, "1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk", "", true, "0x122fca143c05", 0.0, ""},
            {46, 46, "1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP", "", true, "0x2ec18388d544", 0.0, ""},
            {47, 47, "1Pd8VvT49sHKsmqrQiP61RsVwmXCZ6ay7Z", "", true, "0x6cd610b53cba", 0.0, ""},
            {48, 48, "1DFYhaB2J9q1LLZJWKTnscPWos9VBqDHzv", "", true, "0xade6d7ce3b9b", 0.0, ""},
            {49, 49, "12CiUhYVTTH33w3SPUBqcpMoqnApAV4WCF", "", true, "0x174176b015f4d", 0.0, ""},
            {50, 50, "1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk", "", true, "0x22bd43c2e9354", 0.0, ""},
            {51, 51, "1NpnQyZ7x24ud82b7WiRNvPm6N8bqGQnaS", "", true, "0x75070a1a009d4", 0.0, ""},
            {52, 52, "15z9c9sVpu6fwNiK7dMAFgMYSK4GqsGZim", "", true, "0xefae164cb9e3c", 0.0, ""},
            {53, 53, "15K1YKJMiJ4fpesTVUcByoz334rHmknxmT", "", true, "0x180788e47e326c", 0.0, ""},
            {54, 54, "1KYUv7nSvXx4642TKeuC2SNdTk326uUpFy", "", true, "0x236fb6d5ad1f43", 0.0, ""},
            {55, 55, "1LzhS3k3e9Ub8i2W1V8xQFdB8n2MYCHPCa", "", true, "0x6abe1f9b67e114", 0.0, ""},
            {56, 56, "17aPYR1m6pVAacXg1PTDDU7XafvK1dxvhi", "", true, "0x9d18b63ac4ffdf", 0.0, ""},
            {57, 57, "15c9mPGLku1HuW9LRtBf4jcHVpBUt8txKz", "", true, "0x1eb25c90795d61c", 0.0, ""},
            {58, 58, "1Dn8NF8qDyyfHMktmuoQLGyjWmZXgvosXf", "", true, "0x2c675b852189a21", 0.0, ""},
            {59, 59, "1HAX2n9Uruu9YDt4cqRgYcvtGvZj1rbUyt", "", true, "0x7496cbb87cab44f", 0.0, ""},
            {60, 60, "1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su", "", true, "0xfc07a1825367bbe", 0.0, ""},
            {61, 61, "1AVJKwzs9AskraJLGHAZPiaZcrpDr1U6AB", "", true, "0x13c96a3742f64906", 0.0, ""},
            {62, 62, "1Me6EfpwZK5kQziBwBfvLiHjaPGxCKLoJi", "", true, "0x363d541eb611abee", 0.0, ""},
            {63, 63, "1NpYjtLira16LfGbGwZJ5JbDPh3ai9bjf4", "", true, "0x7cce5efdaccf6808", 0.0, ""},

            // Solved puzzles 64-70
            {64, 64, "16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN", "", true, "0xf7051f27b09112d4", 0.0, ""},
            {65, 65, "18ZMbwUFLMHoZBbfpCjUJQTCMCbktshgpe", "", true, "0x1a838b13505b26867", 0.0, "0230210c23b1a047bc9bdbb13448e67deddc108946de6de639bcc75d47c0216b1b"},
            {66, 66, "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so", "", true, "0x2832ed74f2b5e35ee", 0.0, ""},
            {67, 67, "1BY8GQbnueYofwSuFAT3USAhGjPrkxDdW9", "", true, "0x730fc235c1942c1ae", 0.0, ""},
            {68, 68, "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ", "", true, "0xbebb3940cd0fc1491", 0.0, ""},
            {69, 69, "19vkiEajfhuZ8bs8Zu2jgmC6oqZbWqhxhG", "", true, "0x101d83275fb2bc7e0c", 0.0, ""},
            {70, 70, "19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR", "", true, "0x349b84b6431a6c4ef1", 0.0, "0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483"},

            // Solved "every 5th" puzzles (75-130)
            {75, 75, "1J36UjUByGroXcCvmj13U6uwaVv9caEeAt", "", true, "0x4c5ce114686a1336e07", 0.0, "03726b574f193e374686d8e12bc6e4142adeb06770e0a2856f5e4ad89f66044755"},
            {80, 80, "1BCf6rHUW6m3iH2ptsvnjgLruAiPQQepLe", "", true, "0xea1a5c66dcc11b5ad180", 0.0, "037e1238f7b1ce757df94faa9a2eb261bf0aeb9f84dbf81212104e78931c2a19dc"},
            {85, 85, "1Kh22PvXERd2xpTQk3ur6pPEqFeckCJfAr", "", true, "0x11720c4f018d51b8cebba8", 0.0, "0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a"},
            {90, 90, "1L12FHH2FHjvTviyanuiFVfmzCy46RRATU", "", true, "0x2ce00bb2136a445c71e85bf", 0.0, "035c38bd9ae4b10e8a250857006f3cfd98ab15a6196d9f4dfd25bc7ecc77d788d5"},
            {95, 95, "19eVSDuizydXxhohGh8Ki9WY9KsHdSwoQC", "", true, "0x527a792b183c7f64a0e8b1f4", 0.0, "02967a5905d6f3b420959a02789f96ab4c3223a2c4d2762f817b7895c5bc88a045"},
            {100, 100, "1KCgMv8fo2TPBpddVi9jqmMmcne9uSNJ5F", "", true, "0xaf55fc59c335c8ec67ed24826", 0.0, "03d2063d40402f030d4cc71331468827aa41a8a09bd6fd801ba77fb64f8e67e617"},
            {105, 105, "1CMjscKB3QW7SDyQ4c3C3DEUHiHRhiZVib", "", true, "0x16f14fc2054cd87ee6396b33df3", 0.0, "03bcf7ce887ffca5e62c9cabbdb7ffa71dc183c52c04ff4ee5ee82e0c55c39d77b"},
            {110, 110, "12JzYkkN76xkwvcPT6AWKZtGX6w2LAgsJg", "", true, "0x35c0d7234df7deb0f20cf7062444", 0.0, "0309976ba5570966bf889196b7fdf5a0f9a1e9ab340556ec29f8bb60599616167d"},
            {115, 115, "1NLbHuJebVwUZ1XqDjsAyfTRUPwDQbemfv", "", true, "0x60f4d11574f5deee49961d9609ac6", 0.0, "0248d313b0398d4923cdca73b8cfa6532b91b96703902fc8b32fd438a3b7cd7f55"},
            {120, 120, "17s2b9ksz5y7abUm92cHwG8jEPCzK3dLnT", "", true, "0xb10f22572c497a836ea187f2e1fc23", 0.0, "02ceb6cbbcdbdf5ef7150682150f4ce2c6f4807b349827dcdbdd1f2efa885a2630"},
            {125, 125, "1PXAyUB8ZoH3WD8n5zoAthYjN15yN5CVq5", "", true, "0x1c533b6bb7f0804e09960225e44877ac", 0.0, "0233709eb11e0d4439a729f21c2c443dedb727528229713f0065721ba8fa46f00e"},
            {130, 130, "1Fo65aKq8s8iquMt6weF1rku1moWVEd5Ua", "", true, "0x33e7665705359f04f28b88cf897c603c9", 0.0, "03633cbe3ec02b9401c5effa144c5b4d22f87940259634858fc7e59b1c09937852"},

            // ============================================================
            // UNSOLVED puzzles - current targets!
            // ============================================================

            // Unsolved 71-74 (no public key)
            {71, 71, "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU", "", false, "", 7.1, ""},
            {72, 72, "1JTK7s9YVYywfm5XUH7RNhHJH1LshCaRFR", "", false, "", 7.2, ""},
            {73, 73, "12VVRNPi4SJqUTsp6FmqDqY5sGosDtysn4", "", false, "", 7.3, ""},
            {74, 74, "1FWGcVDK3JGzCC3WtkYetULPszMaK2Jksv", "", false, "", 7.4, ""},
            // Unsolved 76-79
            {76, 76, "1DJh2eHFYQfACPmrvpyWc8MSTYKh7w9eRF", "", false, "", 7.6, ""},
            {77, 77, "1Bxk4CQdqL9p22JEtDfdXMsng1XacifUtE", "", false, "", 7.7, ""},
            {78, 78, "15qF6X51huDjqTmF9BJgxXdt1xcj46Jmhb", "", false, "", 7.8, ""},
            {79, 79, "1ARk8HWJMn8js8tQmGUJeQHjSE7KRkn2t8", "", false, "", 7.9, ""},
            // Unsolved 81-84
            {81, 81, "15qsCm78whspNQFydGJQk5rexzxTQopnHZ", "", false, "", 8.1, ""},
            {82, 82, "13zYrYhhJxp6Ui1VV7pqa5WDhNWM45ARAC", "", false, "", 8.2, ""},
            {83, 83, "14MdEb4eFcT3MVG5sPFG4jGLuHJSnt1Dk2", "", false, "", 8.3, ""},
            {84, 84, "1CMq3SvFcVEcpLMuuH8PUcNiqsK1oicG2D", "", false, "", 8.4, ""},
            // Unsolved 86-89
            {86, 86, "1K3x5L6G57Y494fDqBfrojD28UJv4s5JcK", "", false, "", 8.6, ""},
            {87, 87, "1PxH3K1Shdjb7gSEoTX7UPDZ6SH4qGPrvq", "", false, "", 8.7, ""},
            {88, 88, "16AbnZjZZipwHMkYKBSfswGWKDmXHjEpSf", "", false, "", 8.8, ""},
            {89, 89, "19QciEHbGVNY4hrhfKXmcBBCrJSBZ6TaVt", "", false, "", 8.9, ""},
            // Unsolved 91-94
            {91, 91, "1EzVHtmbN4fs4MiNk3ppEnKKhsmXYJ4s74", "", false, "", 9.1, ""},
            {92, 92, "1AE8NzzgKE7Yhz7BWtAcAAxiFMbPo82NB5", "", false, "", 9.2, ""},
            {93, 93, "17Q7tuG2JwFFU9rXVj3uZqRtioH3mx2Jad", "", false, "", 9.3, ""},
            {94, 94, "1K6xGMUbs6ZTXBnhw1pippqwK6wjBWtNpL", "", false, "", 9.4, ""},
            // Unsolved 96-99
            {96, 96, "15ANYzzCp5BFHcCnVFzXqyibpzgPLWaD8b", "", false, "", 9.6, ""},
            {97, 97, "18ywPwj39nGjqBrQJSzZVq2izR12MDpDr8", "", false, "", 9.7, ""},
            {98, 98, "1CaBVPrwUxbQYYswu32w7Mj4HR4maNoJSX", "", false, "", 9.8, ""},
            {99, 99, "1JWnE6p6UN7ZJBN7TtcbNDoRcjFtuDWoNL", "", false, "", 9.9, ""},
            // Unsolved 101-104
            {101, 101, "1CKCVdbDJasYmhswB6HKZHEAnNaDpK7W4n", "", false, "", 10.1, ""},
            {102, 102, "1PXv28YxmYMaB8zxrKeZBW8dt2HK7RkRPX", "", false, "", 10.2, ""},
            {103, 103, "1AcAmB6jmtU6AiEcXkmiNE9TNVPsj9DULf", "", false, "", 10.3, ""},
            {104, 104, "1EQJvpsmhazYCcKX5Au6AZmZKRnzarMVZu", "", false, "", 10.4, ""},
            // Unsolved 106-109
            {106, 106, "18KsfuHuzQaBTNLASyj15hy4LuqPUo1FNB", "", false, "", 10.6, ""},
            {107, 107, "15EJFC5ZTs9nhsdvSUeBXjLAuYq3SWaxTc", "", false, "", 10.7, ""},
            {108, 108, "1HB1iKUqeffnVsvQsbpC6dNi1XKbyNuqao", "", false, "", 10.8, ""},
            {109, 109, "1GvgAXVCbA8FBjXfWiAms4ytFeJcKsoyhL", "", false, "", 10.9, ""},
            // Unsolved 111-114
            {111, 111, "1824ZJQ7nKJ9QFTRBqn7z7dHV5EGpzUpH3", "", false, "", 11.1, ""},
            {112, 112, "18A7NA9FTsnJxWgkoFfPAFbQzuQxpRtCos", "", false, "", 11.2, ""},
            {113, 113, "1NeGn21dUDDeqFQ63xb2SpgUuXuBLA4WT4", "", false, "", 11.3, ""},
            {114, 114, "174SNxfqpdMGYy5YQcfLbSTK3MRNZEePoy", "", false, "", 11.4, ""},
            // Unsolved 116-119
            {116, 116, "1MnJ6hdhvK37VLmqcdEwqC3iFxyWH2PHUV", "", false, "", 11.6, ""},
            {117, 117, "1KNRfGWw7Q9Rmwsc6NT5zsdvEb9M2Wkj5Z", "", false, "", 11.7, ""},
            {118, 118, "1PJZPzvGX19a7twf5HyD2VvNiPdHLzm9F6", "", false, "", 11.8, ""},
            {119, 119, "1GuBBhf61rnvRe4K8zu8vdQB3kHzwFqSy7", "", false, "", 11.9, ""},
            // Unsolved 121-124
            {121, 121, "1GDSuiThEV64c166LUFC9uDcVdGjqkxKyh", "", false, "", 12.1, ""},
            {122, 122, "1Me3ASYt5JCTAK2XaC32RMeH34PdprrfDx", "", false, "", 12.2, ""},
            {123, 123, "1CdufMQL892A69KXgv6UNBD17ywWqYpKut", "", false, "", 12.3, ""},
            {124, 124, "1BkkGsX9ZM6iwL3zbqs7HWBV7SvosR6m8N", "", false, "", 12.4, ""},
            // Unsolved 126-129
            {126, 126, "1AWCLZAjKbV1P7AHvaPNCKiB7ZWVDMxFiz", "", false, "", 12.6, ""},
            {127, 127, "1G6EFyBRU86sThN3SSt3GrHu1sA7w7nzi4", "", false, "", 12.7, ""},
            {128, 128, "1MZ2L1gFrCtkkn6DnTT2e4PFUTHw9gNwaj", "", false, "", 12.8, ""},
            {129, 129, "1Hz3uv3nNZzBVMXLGadCucgjiCs5W9vaGz", "", false, "", 12.9, ""},
            // Unsolved 131-160
            {131, 131, "16zRPnT8znwq42q7XeMkZUhb1bKqgRogyy", "", false, "", 13.1, ""},
            {132, 132, "1KrU4dHE5WrW8rhWDsTRjR21r8t3dsrS3R", "", false, "", 13.2, ""},
            {133, 133, "17uDfp5r4n441xkgLFmhNoSW1KWp6xVLD", "", false, "", 13.3, ""},
            {134, 134, "13A3JrvXmvg5w9XGvyyR4JEJqiLz8ZySY3", "", false, "", 13.4, ""},
            {135, 135, "16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v", "", false, "", 13.5, "02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"},
            {136, 136, "1UDHPdovvR985NrWSkdWQDEQ1xuRiTALq", "", false, "", 13.6, ""},
            {137, 137, "15nf31J46iLuK1ZkTnqHo7WgN5cARFK3RA", "", false, "", 13.7, ""},
            {138, 138, "1Ab4vzG6wEQBDNQM1B2bvUz4fqXXdFk2WT", "", false, "", 13.8, ""},
            {139, 139, "1Fz63c775VV9fNyj25d9Xfw3YHE6sKCxbt", "", false, "", 13.9, ""},
            {140, 140, "1QKBaU6WAeycb3DbKbLBkX7vJiaS8r42Xo", "", false, "", 14.0, "031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640"},
            {141, 141, "1CD91Vm97mLQvXhrnoMChhJx4TP9MaQkJo", "", false, "", 14.1, ""},
            {142, 142, "15MnK2jXPqTMURX4xC3h4mAZxyCcaWWEDD", "", false, "", 14.2, ""},
            {143, 143, "13N66gCzWWHEZBxhVxG18P8wyjEWF9Yoi1", "", false, "", 14.3, ""},
            {144, 144, "1NevxKDYuDcCh1ZMMi6ftmWwGrZKC6j7Ux", "", false, "", 14.4, ""},
            {145, 145, "19GpszRNUej5yYqxXoLnbZWKew3KdVLkXg", "", false, "", 14.5, "03afdda497369e219a2c1c369954a930e4d3740968e5e4352475bcffce3140dae5"},
            {146, 146, "1M7ipcdYHey2Y5RZM34MBbpugghmjaV89P", "", false, "", 14.6, ""},
            {147, 147, "18aNhurEAJsw6BAgtANpexk5ob1aGTwSeL", "", false, "", 14.7, ""},
            {148, 148, "1FwZXt6EpRT7Fkndzv6K4b4DFoT4trbMrV", "", false, "", 14.8, ""},
            {149, 149, "1CXvTzR6qv8wJ7eprzUKeWxyGcHwDYP1i2", "", false, "", 14.9, ""},
            {150, 150, "1MUJSJYtGPVGkBCTqGspnxyHahpt5Te8jy", "", false, "", 15.0, "03137807790ea7dc6e97901c2bc87411f45ed74a5629315c4e4b03a0a102250c49"},
            {151, 151, "13Q84TNNvgcL3HJiqQPvyBb9m4hxjS3jkV", "", false, "", 15.1, ""},
            {152, 152, "1LuUHyrQr8PKSvbcY1v1PiuGuqFjWpDumN", "", false, "", 15.2, ""},
            {153, 153, "18192XpzzdDi2K11QVHR7td2HcPS6Qs5vg", "", false, "", 15.3, ""},
            {154, 154, "1NgVmsCCJaKLzGyKLFJfVequnFW9ZvnMLN", "", false, "", 15.4, ""},
            {155, 155, "1AoeP37TmHdFh8uN72fu9AqgtLrUwcv2wJ", "", false, "", 15.5, "035cd1854cae45391ca4ec428cc7e6c7d9984424b954209a8eea197b9e364c05f6"},
            {156, 156, "1FTpAbQa4h8trvhQXjXnmNhqdiGBd1oraE", "", false, "", 15.6, ""},
            {157, 157, "14JHoRAdmJg3XR4RjMDh6Wed6ft6hzbQe9", "", false, "", 15.7, ""},
            {158, 158, "19z6waranEf8CcP8FqNgdwUe1QRxvUNKBG", "", false, "", 15.8, ""},
            {159, 159, "14u4nA5sugaswb6SZgn5av2vuChdMnD9E5", "", false, "", 15.9, ""},
            {160, 160, "1NBC8uXJy1GiJ6drkiZa1WuKn51ps7EPTv", "", false, "", 16.0, "02e0a8b039282faf6fe0fd769cfbc4b6b4cf8758ba68220eac420e32b91ddfa673"},
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

    /**
     * Check if a puzzle has been solved by querying its address balance.
     * Uses mempool.space API. Returns true if balance is 0 (funds moved = solved).
     * Returns false if balance > 0 or if the API call fails (fail-open: assume unsolved).
     */
    static bool check_if_solved(const std::string& address) {
        // Build URL: https://mempool.space/api/address/{address}
        // Response JSON contains "chain_stats.funded_txo_sum" and "chain_stats.spent_txo_sum"
        // If funded == spent, balance is 0, puzzle is solved

#ifdef _WIN32
        std::string cmd = "curl.exe -sf --max-time 10 https://mempool.space/api/address/" + address + " 2>NUL";
        FILE* pipe = _popen(cmd.c_str(), "r");
#else
        std::string cmd = "curl -sf --max-time 10 https://mempool.space/api/address/" + address + " 2>/dev/null";
        FILE* pipe = popen(cmd.c_str(), "r");
#endif
        if (!pipe) return false;  // Can't check, assume unsolved

        std::string result;
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe)) {
            result += buffer;
        }
#ifdef _WIN32
        int status = _pclose(pipe);
#else
        int status = pclose(pipe);
#endif
        if (status != 0 || result.empty()) return false;  // API failed, assume unsolved

        // Parse JSON manually (no JSON library dependency)
        // Look for "funded_txo_sum":N and "spent_txo_sum":M
        // If N == M and N > 0, the balance is 0 (all funds spent = solved)
        auto extract_value = [](const std::string& json, const std::string& key) -> int64_t {
            size_t pos = json.find("\"" + key + "\"");
            if (pos == std::string::npos) return -1;
            pos = json.find(':', pos);
            if (pos == std::string::npos) return -1;
            pos++; // skip ':'
            while (pos < json.size() && json[pos] == ' ') pos++;
            std::string num;
            while (pos < json.size() && (std::isdigit(json[pos]) || json[pos] == '-')) {
                num += json[pos++];
            }
            if (num.empty()) return -1;
            return std::stoll(num);
        };

        // Find the chain_stats section
        size_t chain_stats_pos = result.find("\"chain_stats\"");
        if (chain_stats_pos == std::string::npos) return false;
        std::string chain_section = result.substr(chain_stats_pos);

        int64_t funded = extract_value(chain_section, "funded_txo_sum");
        int64_t spent = extract_value(chain_section, "spent_txo_sum");

        if (funded <= 0) return false;  // No funds ever received, can't determine

        return (funded == spent);  // All funds spent = puzzle solved
    }

    /**
     * Verify a puzzle is still unsolved before starting work.
     * Prints a warning and returns false if the puzzle appears solved.
     */
    static bool verify_unsolved(int puzzle_number) {
        const auto* puzzle = get_puzzle(puzzle_number);
        if (!puzzle) return false;

        if (puzzle->solved) {
            std::cerr << "[!] Puzzle #" << puzzle_number << " is already marked as solved in the database.\n";
            return false;
        }

        std::cout << "[*] Checking if puzzle #" << puzzle_number << " is still unsolved...\n";

        if (check_if_solved(puzzle->target_address)) {
            std::cerr << "[!] Puzzle #" << puzzle_number << " appears to have been SOLVED!\n";
            std::cerr << "    Address " << puzzle->target_address << " has zero balance.\n";
            std::cerr << "    The funds have been moved. This puzzle is no longer worth mining.\n";
            return false;
        }

        std::cout << "[+] Puzzle #" << puzzle_number << " confirmed unsolved (balance > 0).\n";
        return true;
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
