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

            // ALL solved puzzles 1-65 plus 66-70 plus 75/80/85/.../130
            // (every multiple of 5 up to 130). 82 entries total. The
            // canonical export at https://privatekeys.pw/puzzles/bitcoin-
            // puzzle-tx tracks 82 confirmed solves; this list mirrors
            // it. Source of truth: data/puzzle_history.json with the
            // private keys; addresses + h160s + compressed pubkeys are
            // DERIVED from the keys via scripts/gen_puzzle_table.py.
            // Regenerate by running:
            //   python scripts/gen_puzzle_table.py
            // and pasting the output here. v1.4.1 expanded the bundle
            // from "spot solved entries" to "every confirmed solve" so
            // `--puzzle N` works for any N in [1, 65] without needing
            // --puzzle-start / --puzzle-end overrides.
            {1, 1, "1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH", "751e76e8199196d454941c45d1b3a323f1433bd6", true, "0x1", 0.0, "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"},
            {2, 2, "1CUNEBjYrCn2y1SdiUMohaKUi4wpP326Lb", "7dd65592d0ab2fe0d0257d571abf032cd9db93dc", true, "0x3", 0.0, "02f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9"},
            {3, 3, "19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA", "5dedfbf9ea599dd4e3ca6a80b333c472fd0b3f69", true, "0x7", 0.0, "025cbdf0646e5db4eaa398f365f2ea7a0e3d419b7e0330e39ce92bddedcac4f9bc"},
            {4, 4, "1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e", "9652d86bedf43ad264362e6e6eba6eb764508127", true, "0x8", 0.0, "022f01e5e15cca351daff3843fb70f3c2f0a1bdd05e5af888a67784ef3e10a2a01"},
            {5, 5, "1E6NuFjCi27W5zoXg8TRdcSRq84zJeBW3k", "8f9dff39a81ee4abcbad2ad8bafff090415a2be8", true, "0x15", 0.0, "02352bbf4a4cdd12564f93fa332ce333301d9ad40271f8107181340aef25be59d5"},
            {6, 6, "1PitScNLyp2HCygzadCh7FveTnfmpPbfp8", "f93ec34e9e34a8f8ff7d600cdad83047b1bcb45c", true, "0x31", 0.0, "03f2dac991cc4ce4b9ea44887e5c7c0bce58c80074ab9d4dbaeb28531b7739f530"},
            {7, 7, "1McVt1vMtCC7yn5b9wgX1833yCcLXzueeC", "e2192e8a7dd8dd1c88321959b477968b941aa973", true, "0x4c", 0.0, "0296516a8f65774275278d0d7420a88df0ac44bd64c7bae07c3fe397c5b3300b23"},
            {8, 8, "1M92tSqNmQLYw33fuBvjmeadirh1ysMBxK", "dce76b2613052ea012204404a97b3c25eac31715", true, "0xe0", 0.0, "0308bc89c2f919ed158885c35600844d49890905c79b357322609c45706ce6b514"},
            {9, 9, "1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV", "7d0f6c64afb419bbd7e971e943d7404b0e0daab4", true, "0x1d3", 0.0, "0243601d61c836387485e9514ab5c8924dd2cfd466af34ac95002727e1659d60f7"},
            {10, 10, "1LeBZP5QCwwgXRtmVUvTVrraqPUokyLHqe", "d7729816650e581d7462d52ad6f732da0e2ec93b", true, "0x202", 0.0, "03a7a4c30291ac1db24b4ab00c442aa832f7794b5a0959bec6e8d7fee802289dcd"},
            {11, 11, "1PgQVLmst3Z314JrQn5TNiys8Hc38TcXJu", "f8c698da3164ef8fa4258692d118cc9a902c5acc", true, "0x483", 0.0, "038b05b0603abd75b0c57489e451f811e1afe54a8715045cdf4888333f3ebc6e8b"},
            {12, 12, "1DBaumZxUkM4qMQRt2LVWyFJq5kDtSZQot", "85a1f9ba4da24c24e582d9b891dacbd1b043f971", true, "0xa7b", 0.0, "038b00fcbfc1a203f44bf123fc7f4c91c10a85c8eae9187f9d22242b4600ce781c"},
            {13, 13, "1Pie8JkxBT6MGPz9Nvi3fsPkr2D8q3GBc1", "f932d0188616c964416b91fb9cf76ba9790a921e", true, "0x1460", 0.0, "03aadaaab1db8d5d450b511789c37e7cfeb0eb8b3e61a57a34166c5edc9a4b869d"},
            {14, 14, "1ErZWg5cFCe4Vw5BzgfzB74VNLaXEiEkhk", "97f9281a1383879d72ac52a6a3e9e8b9a4a4f655", true, "0x2930", 0.0, "03b4f1de58b8b41afe9fd4e5ffbdafaeab86c5db4769c15d6e6011ae7351e54759"},
            {15, 15, "1QCbW9HWnwQWiQqVo5exhAnmfqKRrCRsvW", "fe7c45126731f7384640b0b0045fd40bac72e2a2", true, "0x68f3", 0.0, "02fea58ffcf49566f6e9e9350cf5bca2861312f422966e8db16094beb14dc3df2c"},
            {16, 16, "1BDyrQ6WoF8VN3g9SAS1iKZcPzFfnDVieY", "7025b4efb3ff42eb4d6d71fab6b53b4f4967e3dd", true, "0xc936", 0.0, "029d8c5d35231d75eb87fd2c5f05f65281ed9573dc41853288c62ee94eb2590b7a"},
            {17, 17, "1HduPEXZRdG26SUT5Yk83mLkPyjnZuJ7Bm", "b67cb6edeabc0c8b927c9ea327628e7aa63e2d52", true, "0x1764f", 0.0, "033f688bae8321b8e02b7e6c0a55c2515fb25ab97d85fda842449f7bfa04e128c3"},
            {18, 18, "1GnNTmTVLZiqQfLbAdp9DVdicEnB5GoERE", "ad1e852b08eba53df306ec9daa8c643426953f94", true, "0x3080d", 0.0, "020ce4a3291b19d2e1a7bf73ee87d30a6bdbc72b20771e7dfff40d0db755cd4af1"},
            {19, 19, "1NWmZRpHH4XSPwsW6dsS3nrNWfL1yrJj4w", "ebfbe6819fcdebab061732ce91df7d586a037dee", true, "0x5749f", 0.0, "0385663c8b2f90659e1ccab201694f4f8ec24b3749cfe5030c7c3646a709408e19"},
            {20, 20, "1HsMJxNiV7TLxmoF6uJNkydxPFDog4NQum", "b907c3a2a3b27789dfb509b730dd47703c272868", true, "0xd2c55", 0.0, "033c4a45cbd643ff97d77f41ea37e843648d50fd894b864b0d52febc62f6454f7c"},
            {21, 21, "14oFNXucftsHiUMY8uctg6N487riuyXs4h", "29a78213caa9eea824acf08022ab9dfc83414f56", true, "0x1ba534", 0.0, "031a746c78f72754e0be046186df8a20cdce5c79b2eda76013c647af08d306e49e"},
            {22, 22, "1CfZWK1QTQE3eS9qn61dQjV89KDjZzfNcv", "7ff45303774ef7a52fffd8011981034b258cb86b", true, "0x2de40f", 0.0, "023ed96b524db5ff4fe007ce730366052b7c511dc566227d929070b9ce917abb43"},
            {23, 23, "1L2GM8eE7mJWLdo3HZS6su1832NX2txaac", "d0a79df189fe1ad5c306cc70497b358415da579e", true, "0x556e52", 0.0, "03f82710361b8b81bdedb16994f30c80db522450a93e8e87eeb07f7903cf28d04b"},
            {24, 24, "1rSnXMr63jdCuegJFuidJqWxUPV7AtUf7", "0959e80121f36aea13b3bad361c15dac26189e2f", true, "0xdc2a04", 0.0, "036ea839d22847ee1dce3bfc5b11f6cf785b0682db58c35b63d1342eb221c3490c"},
            {25, 25, "15JhYXn6Mx3oF4Y7PcTAv2wVVAuCFFQNiP", "2f396b29b27324300d0c59b17c3abc1835bd3dbb", true, "0x1fa5ee5", 0.0, "03057fbea3a2623382628dde556b2a0698e32428d3cd225f3bd034dca82dd7455a"},
            {26, 26, "1JVnST957hGztonaWK6FougdtjxzHzRMMg", "bfebb73562d4541b32a02ba664d140b5a574792f", true, "0x340326e", 0.0, "024e4f50a2a3eccdb368988ae37cd4b611697b26b29696e42e06d71368b4f3840f"},
            {27, 27, "128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k", "0c7aaf6caa7e5424b63d317f0f8f1f9fa40d5560", true, "0x6ac3875", 0.0, "031a864bae3922f351f1b57cfdd827c25b7e093cb9c88a72c1cd893d9f90f44ece"},
            {28, 28, "12jbtzBb54r97TCwW3G1gCFoumpckRAPdY", "1306b9e4ff56513a476841bac7ba48d69516b1da", true, "0xd916ce8", 0.0, "03e9e661838a96a65331637e2a3e948dc0756e5009e7cb5c36664d9b72dd18c0a7"},
            {29, 29, "19EEC52krRUK1RkUAEZmQdjTyHT7Gp1TYT", "5a416cc9148f4a377b672c8ae5d3287adaafadec", true, "0x17e2551e", 0.0, "026caad634382d34691e3bef43ed4a124d8909a8a3362f91f1d20abaaf7e917b36"},
            {30, 30, "1LHtnpd8nU5VHEMkG2TMYYNUjjLc992bps", "d39c4704664e1deb76c9331e637564c257d68a08", true, "0x3d94cd64", 0.0, "030d282cf2ff536d2c42f105d0b8588821a915dc3f9a05bd98bb23af67a2e92a5b"},
            {31, 31, "1LhE6sCTuGae42Axu1L1ZB7L96yi9irEBE", "d805f6f251f7479ebd853b3d0f4b9b2656d92f1d", true, "0x7d4fe747", 0.0, "0387dc70db1806cd9a9a76637412ec11dd998be666584849b3185f7f9313c8fd28"},
            {32, 32, "1FRoHA9xewq7DjrZ1psWJVeTer8gHRqEvR", "9e42601eeaedc244e15f17375adb0e2cd08efdc9", true, "0xb862a62e", 0.0, "0209c58240e50e3ba3f833c82655e8725c037a2294e14cf5d73a5df8d56159de69"},
            {33, 33, "187swFMjz1G54ycVU56B7jZFHFTNVQFDiu", "4e15e5189752d1eaf444dfd6bff399feb0443977", true, "0x1a96ca8d8", 0.0, "03a355aa5e2e09dd44bb46a4722e9336e9e3ee4ee4e7b7a0cf5785b283bf2ab579"},
            {34, 34, "1PWABE7oUahG2AFFQhhvViQovnCr4rEv7Q", "f6d67d7983bf70450f295c9cb828daab265f1bfa", true, "0x34a65911d", 0.0, "033cdd9d6d97cbfe7c26f902faf6a435780fe652e159ec953650ec7b1004082790"},
            {35, 35, "1PWCx5fovoEaoBowAvF5k91m2Xat9bMgwb", "f6d8ce225ffbdecec170f8298c3fc28ae686df25", true, "0x4aed21170", 0.0, "02f6a8148a62320e149cb15c544fe8a25ab483a0095d2280d03b8a00a7feada13d"},
            {36, 36, "1Be2UF9NLfyLFbtm3TCbmuocc9N1Kduci1", "74b1e012be1521e5d8d75e745a26ced845ea3d37", true, "0x9de820a7c", 0.0, "02b3e772216695845fa9dda419fb5daca28154d8aa59ea302f05e916635e47b9f6"},
            {37, 37, "14iXhn8bGajVWegZHJ18vJLHhntcpL4dex", "28c30fb9118ed1da72e7c4f89c0164756e8a021d", true, "0x1757756a93", 0.0, "027d2c03c3ef0aec70f2c7e1e75454a5dfdd0e1adea670c1b3a4643c48ad0f1255"},
            {38, 38, "1HBtApAFA9B2YZw3G2YKSMCtb3dVnjuNe2", "b190e2d40cfdeee2cee072954a2be89e7ba39364", true, "0x22382facd0", 0.0, "03c060e1e3771cbeccb38e119c2414702f3f5181a89652538851d2e3886bdd70c6"},
            {39, 39, "122AJhKLEfkFBaGAd84pLp1kfE7xK3GdT8", "0b304f2a79a027270276533fe1ed4eff30910876", true, "0x4b5f8303e9", 0.0, "022d77cd1467019a6bf28f7375d0949ce30e6b5815c2758b98a74c2700bc006543"},
            {40, 40, "1EeAxcprB2PpCnr34VfZdFrkUWuxyiNEFv", "95a156cd21b4a69de969eb6716864f4c8b82a82a", true, "0xe9ae4933d6", 0.0, "03a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4"},
            {41, 41, "1L5sU9qvJeuwQUdt4y1eiLmquFxKjtHr3E", "d1562eb37357f9e6fc41cb2359f4d3eda4032329", true, "0x153869acc5b", 0.0, "03b357e68437da273dcf995a474a524439faad86fc9effc300183f714b0903468b"},
            {42, 42, "1E32GPWgDyeyQac4aJxm9HVoLrrEYPnM4N", "8efb85f9c5b5db2d55973a04128dc7510075ae23", true, "0x2a221c58d8f", 0.0, "03eec88385be9da803a0d6579798d977a5d0c7f80917dab49cb73c9e3927142cb6"},
            {43, 43, "1PiFuqGpG8yGM5v6rNHWS3TjsG6awgEGA1", "f92044c7924e5525c61207972c253c9fc9f086f7", true, "0x6bd3b27c591", 0.0, "02a631f9ba0f28511614904df80d7f97a4f43f02249c8909dac92276ccf0bcdaed"},
            {44, 44, "1CkR2uS7LmFwc3T2jV8C1BhWb5mQaoxedF", "80df54e1f612f2fc5bdc05c9d21a83aa8d20791e", true, "0xe02b35a358f", 0.0, "025e466e97ed0e7910d3d90ceb0332df48ddf67d456b9e7303b50a3d89de357336"},
            {45, 45, "1NtiLNGegHWE3Mp9g2JPkgx6wUg4TW7bbk", "f0225bfc68a6e17e87cd8b5e60ae3be18f120753", true, "0x122fca143c05", 0.0, "026ecabd2d22fdb737be21975ce9a694e108eb94f3649c586cc7461c8abf5da71a"},
            {46, 46, "1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP", "9a012260d01c5113df66c8a8438c9f7a1e3d5dac", true, "0x2ec18388d544", 0.0, "03fd5487722d2576cb6d7081426b66a3e2986c1ce8358d479063fb5f2bb6dd5849"},
            {47, 47, "1Pd8VvT49sHKsmqrQiP61RsVwmXCZ6ay7Z", "f828005d41b0f4fed4c8dca3b06011072cfb07d4", true, "0x6cd610b53cba", 0.0, "023a12bd3caf0b0f77bf4eea8e7a40dbe27932bf80b19ac72f5f5a64925a594196"},
            {48, 48, "1DFYhaB2J9q1LLZJWKTnscPWos9VBqDHzv", "8661cb56d9df0a61f01328b55af7e56a3fe7a2b2", true, "0xade6d7ce3b9b", 0.0, "0291bee5cf4b14c291c650732faa166040e4c18a14731f9a930c1e87d3ec12debb"},
            {49, 49, "12CiUhYVTTH33w3SPUBqcpMoqnApAV4WCF", "0d2f533966c6578e1111978ca698f8add7fffdf3", true, "0x174176b015f4d", 0.0, "02591d682c3da4a2a698633bf5751738b67c343285ebdc3492645cb44658911484"},
            {50, 50, "1MEzite4ReNuWaL5Ds17ePKt2dCxWEofwk", "de081b76f840e462fa2cdf360173dfaf4a976a47", true, "0x22bd43c2e9354", 0.0, "03f46f41027bbf44fafd6b059091b900dad41e6845b2241dc3254c7cdd3c5a16c6"},
            {51, 51, "1NpnQyZ7x24ud82b7WiRNvPm6N8bqGQnaS", "ef6419cffd7fad7027994354eb8efae223c2dbe7", true, "0x75070a1a009d4", 0.0, "028c6c67bef9e9eebe6a513272e50c230f0f91ed560c37bc9b033241ff6c3be78f"},
            {52, 52, "15z9c9sVpu6fwNiK7dMAFgMYSK4GqsGZim", "36af659edbe94453f6344e920d143f1778653ae7", true, "0xefae164cb9e3c", 0.0, "0374c33bd548ef02667d61341892134fcf216640bc2201ae61928cd0874f6314a7"},
            {53, 53, "15K1YKJMiJ4fpesTVUcByoz334rHmknxmT", "2f4870ef54fa4b048c1365d42594cc7d3d269551", true, "0x180788e47e326c", 0.0, "020faaf5f3afe58300a335874c80681cf66933e2a7aeb28387c0d28bb048bc6349"},
            {54, 54, "1KYUv7nSvXx4642TKeuC2SNdTk326uUpFy", "cb66763cf7fde659869ae7f06884d9a0f879a092", true, "0x236fb6d5ad1f43", 0.0, "034af4b81f8c450c2c870ce1df184aff1297e5fcd54944d98d81e1a545ffb22596"},
            {55, 55, "1LzhS3k3e9Ub8i2W1V8xQFdB8n2MYCHPCa", "db53d9bbd1f3a83b094eeca7dd970bd85b492fa2", true, "0x6abe1f9b67e114", 0.0, "0385a30d8413af4f8f9e6312400f2d194fe14f02e719b24c3f83bf1fd233a8f963"},
            {56, 56, "17aPYR1m6pVAacXg1PTDDU7XafvK1dxvhi", "48214c5969ae9f43f75070cea1e2cb41d5bdcccd", true, "0x9d18b63ac4ffdf", 0.0, "033f2db2074e3217b3e5ee305301eeebb1160c4fa1e993ee280112f6348637999a"},
            {57, 57, "15c9mPGLku1HuW9LRtBf4jcHVpBUt8txKz", "328660ef43f66abe2653fa178452a5dfc594c2a1", true, "0x1eb25c90795d61c", 0.0, "02a521a07e98f78b03fc1e039bc3a51408cd73119b5eb116e583fe57dc8db07aea"},
            {58, 58, "1Dn8NF8qDyyfHMktmuoQLGyjWmZXgvosXf", "8c2a6071f89c90c4dab5ab295d7729d1b54ea60f", true, "0x2c675b852189a21", 0.0, "0311569442e870326ceec0de24eb5478c19e146ecd9d15e4666440f2f638875f42"},
            {59, 59, "1HAX2n9Uruu9YDt4cqRgYcvtGvZj1rbUyt", "b14ed3146f5b2c9bde1703deae9ef33af8110210", true, "0x7496cbb87cab44f", 0.0, "0241267d2d7ee1a8e76f8d1546d0d30aefb2892d231cee0dde7776daf9f8021485"},
            {60, 60, "1Kn5h2qpgw9mWE5jKpk8PP4qvvJ1QVy8su", "cdf8e5c7503a9d22642e3ecfc87817672787b9c5", true, "0xfc07a1825367bbe", 0.0, "0348e843dc5b1bd246e6309b4924b81543d02b16c8083df973a89ce2c7eb89a10d"},
            {61, 61, "1AVJKwzs9AskraJLGHAZPiaZcrpDr1U6AB", "68133e19b2dfb9034edf9830a200cfdf38c90cbd", true, "0x13c96a3742f64906", 0.0, "0249a43860d115143c35c09454863d6f82a95e47c1162fb9b2ebe0186eb26f453f"},
            {62, 62, "1Me6EfpwZK5kQziBwBfvLiHjaPGxCKLoJi", "e26646db84b0602f32b34b5a62ca3cae1f91b779", true, "0x363d541eb611abee", 0.0, "03231a67e424caf7d01a00d5cd49b0464942255b8e48766f96602bdfa4ea14fea8"},
            {63, 63, "1NpYjtLira16LfGbGwZJ5JbDPh3ai9bjf4", "ef58afb697b094423ce90721fbb19a359ef7c50e", true, "0x7cce5efdaccf6808", 0.0, "0365ec2994b8cc0a20d40dd69edfe55ca32a54bcbbaa6b0ddcff36049301a54579"},
            {64, 64, "16jY7qLJnxb7CHZyqBP8qca9d51gAjyXQN", "3ee4133d991f52fdf6a25c9834e0745ac74248a4", true, "0xf7051f27b09112d4", 0.0, "03100611c54dfef604163b8358f7b7fac13ce478e02cb224ae16d45526b25d9d4d"},
            {65, 65, "18ZMbwUFLMHoZBbfpCjUJQTCMCbktshgpe", "52e763a7ddc1aa4fa811578c491c1bc7fd570137", true, "0x1a838b13505b26867", 0.0, "0230210c23b1a047bc9bdbb13448e67deddc108946de6de639bcc75d47c0216b1b"},
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
