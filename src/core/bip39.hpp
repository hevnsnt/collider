/**
 * bip39.hpp -- BIP-39 mnemonic validation (v1.4.2 Phase L.2).
 *
 * Loads the standard 2048-word English BIP-39 wordlist from
 * data/crypto/bip39_english.txt (or a caller-supplied path) and provides
 * lookup + checksum-verified mnemonic validation.
 *
 * Scope of this header (v1.4.2 increment):
 *   - WordlistEnglish::load(path)        -- read words from disk
 *   - WordlistEnglish::index(word)       -- O(log n) bisect lookup
 *   - validate_mnemonic(words, list)     -- checksum-verified validation
 *     supports 12/15/18/21/24-word counts per BIP-39 spec
 *
 * Out of scope for this header (defer to v1.5):
 *   - mnemonic -> seed via PBKDF2-HMAC-SHA512 (needs the HMAC-SHA512
 *     primitive that already exists in src/gpu/v2/device_hashes.cuh; a
 *     CPU mirror is the v1.5 task)
 *   - BIP-32 master/child key derivation
 *   - Diceware 7776-word generator
 *
 * Spec: https://github.com/bitcoin/bips/blob/master/bip-0039.mediawiki
 */

#pragma once

#include "crypto_cpu.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace collider {
namespace bip39 {

class WordlistEnglish {
public:
    static constexpr size_t kCount = 2048;
    static constexpr size_t kIndexBits = 11;  // log2(2048)

    /**
     * Load the BIP-39 English wordlist from disk. One word per line, sorted
     * lexicographically per spec. Throws on missing file / wrong line count.
     */
    void load(const std::string& path) {
        std::ifstream f(path);
        if (!f) {
            throw std::runtime_error("BIP-39 wordlist not found: " + path);
        }
        words_.clear();
        words_.reserve(kCount);
        std::string line;
        while (std::getline(f, line)) {
            // Trim trailing CR (Windows line endings on a POSIX text).
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ' ||
                                       line.back() == '\t')) {
                line.pop_back();
            }
            if (line.empty()) continue;
            words_.push_back(line);
        }
        if (words_.size() != kCount) {
            throw std::runtime_error("BIP-39 wordlist length mismatch: got " +
                                     std::to_string(words_.size()) + " (want " +
                                     std::to_string(kCount) + ")");
        }
        // Spec guarantees sorted; verify so a corrupted file fails loudly.
        for (size_t i = 1; i < words_.size(); i++) {
            if (words_[i] <= words_[i - 1]) {
                throw std::runtime_error("BIP-39 wordlist not sorted at index " +
                                         std::to_string(i));
            }
        }
    }

    /**
     * Return the 0-based 11-bit index of `word`, or -1 if not in the list.
     * Bisect-based on the spec-guaranteed sorted order.
     */
    [[nodiscard]] int index(const std::string& word) const noexcept {
        if (words_.size() != kCount) return -1;
        auto it = std::lower_bound(words_.begin(), words_.end(), word);
        if (it == words_.end() || *it != word) return -1;
        return static_cast<int>(it - words_.begin());
    }

    [[nodiscard]] const std::string& word(size_t idx) const {
        if (idx >= words_.size()) {
            throw std::out_of_range("BIP-39 wordlist index out of range");
        }
        return words_[idx];
    }

    [[nodiscard]] bool ready() const noexcept { return words_.size() == kCount; }

private:
    std::vector<std::string> words_;
};

/**
 * Split a mnemonic string into individual words on whitespace. NFKD
 * normalization is NOT applied here -- the BIP-39 spec requires it for
 * non-ASCII wordlists (Japanese, Chinese), but for the English wordlist
 * the spec mandates ASCII so a simple whitespace split is conformant.
 */
[[nodiscard]] inline std::vector<std::string> split_words(const std::string& mnemonic) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : mnemonic) {
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!cur.empty()) {
                out.push_back(std::move(cur));
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(std::move(cur));
    return out;
}

/**
 * Validate a mnemonic string against the supplied wordlist.
 *
 * Per BIP-39:
 *   - Word count must be in {12, 15, 18, 21, 24}.
 *   - Each word maps to its 11-bit index in the wordlist.
 *   - Concatenated bits = ENT (entropy) || CS (checksum) where CS is the
 *     first ENT/32 bits of SHA-256(ENT).
 *   - ENT lengths: 128 / 160 / 192 / 224 / 256 bits.
 *   - CS lengths : 4 / 5 / 6 / 7 / 8 bits.
 *
 * @return true iff the mnemonic is structurally valid AND the checksum
 *         matches. `entropy_out`, if non-null, receives the ENT bytes on
 *         success (size matches the word count's ENT/8).
 */
inline bool validate_mnemonic(const std::vector<std::string>& words,
                              const WordlistEnglish& wordlist,
                              std::vector<uint8_t>* entropy_out = nullptr) {
    if (!wordlist.ready()) return false;
    const size_t n = words.size();
    if (n != 12 && n != 15 && n != 18 && n != 21 && n != 24) return false;

    const size_t total_bits = n * WordlistEnglish::kIndexBits;
    const size_t cs_bits = total_bits / 33;
    const size_t ent_bits = total_bits - cs_bits;
    if (cs_bits == 0 || ent_bits % 8 != 0) return false;
    const size_t ent_bytes = ent_bits / 8;

    // Reconstruct the ENT||CS bitstream from word indices.
    std::vector<uint8_t> stream((total_bits + 7) / 8, 0);
    size_t bit_pos = 0;
    for (const auto& w : words) {
        int idx = wordlist.index(w);
        if (idx < 0) return false;
        // Emit 11 bits MSB-first into stream.
        for (int b = 10; b >= 0; b--) {
            uint8_t bit = static_cast<uint8_t>((idx >> b) & 1u);
            stream[bit_pos / 8] |= static_cast<uint8_t>(bit << (7 - (bit_pos % 8)));
            bit_pos++;
        }
    }

    // Slice out ENT and (left-aligned) CS byte.
    std::vector<uint8_t> ent(stream.begin(), stream.begin() + ent_bytes);
    uint8_t cs_observed = stream[ent_bytes];  // top byte of CS

    // Expected CS = top cs_bits of SHA-256(ENT).
    auto hash = collider::cpu::SHA256::hash(ent.data(), ent.size());
    uint8_t cs_expected = hash[0];
    const uint8_t mask = static_cast<uint8_t>(0xFFu << (8 - cs_bits));

    if ((cs_observed & mask) != (cs_expected & mask)) {
        return false;
    }

    if (entropy_out) {
        *entropy_out = std::move(ent);
    }
    return true;
}

}  // namespace bip39
}  // namespace collider
