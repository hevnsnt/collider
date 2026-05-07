/**
 * UTXO Bloom Filter Builder
 *
 * Builds a Bloom filter from utxo-dump CSV output for high-speed
 * GPU-based address matching.
 *
 * Pipeline:
 *   utxo-dump CSV -> H160 extraction -> Bloom filter -> .blf file
 *
 * Optimized for:
 *   - ~50M target addresses
 *   - 0.001% false positive rate
 *   - GPU memory alignment (128-byte boundaries)
 *   - Direct memory mapping for VRAM loading
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <array>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>

// Portable popcount
#ifdef _MSC_VER
#include <intrin.h>
#define POPCOUNT(x) __popcnt(x)
#else
#define POPCOUNT(x) __builtin_popcount(x)
#endif

namespace collider {
namespace utxo {

// MurmurHash3 finalization mix
inline uint64_t fmix64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

/**
 * MurmurHash3 128-bit hash for H160 inputs.
 * Returns two 64-bit values for double hashing scheme.
 */
inline std::pair<uint64_t, uint64_t> murmurhash3_128(
    const uint8_t* data,
    size_t len,
    uint32_t seed = 0
) {
    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;

    uint64_t h1 = seed;
    uint64_t h2 = seed;

    // Body
    const uint64_t* blocks = reinterpret_cast<const uint64_t*>(data);
    size_t nblocks = len / 16;

    for (size_t i = 0; i < nblocks; i++) {
        uint64_t k1 = blocks[i * 2];
        uint64_t k2 = blocks[i * 2 + 1];

        k1 *= c1;
        k1 = (k1 << 31) | (k1 >> 33);
        k1 *= c2;
        h1 ^= k1;

        h1 = (h1 << 27) | (h1 >> 37);
        h1 += h2;
        h1 = h1 * 5 + 0x52dce729;

        k2 *= c2;
        k2 = (k2 << 33) | (k2 >> 31);
        k2 *= c1;
        h2 ^= k2;

        h2 = (h2 << 31) | (h2 >> 33);
        h2 += h1;
        h2 = h2 * 5 + 0x38495ab5;
    }

    // Tail
    const uint8_t* tail = data + nblocks * 16;
    uint64_t k1 = 0;
    uint64_t k2 = 0;

    switch (len & 15) {
        case 15: k2 ^= uint64_t(tail[14]) << 48; [[fallthrough]];
        case 14: k2 ^= uint64_t(tail[13]) << 40; [[fallthrough]];
        case 13: k2 ^= uint64_t(tail[12]) << 32; [[fallthrough]];
        case 12: k2 ^= uint64_t(tail[11]) << 24; [[fallthrough]];
        case 11: k2 ^= uint64_t(tail[10]) << 16; [[fallthrough]];
        case 10: k2 ^= uint64_t(tail[9]) << 8; [[fallthrough]];
        case 9:  k2 ^= uint64_t(tail[8]);
                 k2 *= c2;
                 k2 = (k2 << 33) | (k2 >> 31);
                 k2 *= c1;
                 h2 ^= k2;
                 [[fallthrough]];
        case 8:  k1 ^= uint64_t(tail[7]) << 56; [[fallthrough]];
        case 7:  k1 ^= uint64_t(tail[6]) << 48; [[fallthrough]];
        case 6:  k1 ^= uint64_t(tail[5]) << 40; [[fallthrough]];
        case 5:  k1 ^= uint64_t(tail[4]) << 32; [[fallthrough]];
        case 4:  k1 ^= uint64_t(tail[3]) << 24; [[fallthrough]];
        case 3:  k1 ^= uint64_t(tail[2]) << 16; [[fallthrough]];
        case 2:  k1 ^= uint64_t(tail[1]) << 8; [[fallthrough]];
        case 1:  k1 ^= uint64_t(tail[0]);
                 k1 *= c1;
                 k1 = (k1 << 31) | (k1 >> 33);
                 k1 *= c2;
                 h1 ^= k1;
    }

    // Finalization
    h1 ^= len;
    h2 ^= len;
    h1 += h2;
    h2 += h1;
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 += h2;
    h2 += h1;

    return {h1, h2};
}

/**
 * H160 (RIPEMD160 hash of public key) - 20 bytes.
 */
struct H160 {
    uint8_t data[20];

    bool operator==(const H160& other) const {
        return std::memcmp(data, other.data, 20) == 0;
    }

    // Parse from hex string
    static H160 from_hex(const std::string& hex) {
        H160 result;
        if (hex.size() != 40) {
            throw std::invalid_argument("H160 hex must be 40 characters");
        }

        for (size_t i = 0; i < 20; i++) {
            unsigned int byte;
            std::sscanf(hex.c_str() + i * 2, "%02x", &byte);
            result.data[i] = static_cast<uint8_t>(byte);
        }

        return result;
    }

    // Convert to hex string
    std::string to_hex() const {
        char hex[41];
        for (size_t i = 0; i < 20; i++) {
            std::snprintf(hex + i * 2, 3, "%02x", data[i]);
        }
        hex[40] = '\0';
        return std::string(hex);
    }
};

/**
 * Bloom filter file header.
 * Aligned to 128 bytes for GPU memory mapping.
 */
struct alignas(128) BloomFilterHeader {
    char magic[4] = {'B', 'L', 'F', '1'};  // Magic + version
    uint32_t version = 1;
    uint64_t num_bits;           // Total bits in filter
    uint32_t num_hashes;         // Number of hash functions (k)
    uint32_t seed;               // MurmurHash3 seed
    uint64_t num_elements;       // Number of elements inserted
    double target_fp_rate;       // Target false positive rate
    uint64_t data_offset;        // Offset to bit array (128-byte aligned)
    uint8_t reserved[80];        // Padding to 128 bytes
};

static_assert(sizeof(BloomFilterHeader) == 128, "Header must be 128 bytes");

/**
 * UTXO Bloom Filter Builder
 *
 * Builds an optimized Bloom filter from utxo-dump CSV output.
 */
class UTXOBloomBuilder {
public:
    struct Config {
        double target_fp_rate = 0.00001;     // 0.001% false positive rate
        uint64_t expected_elements = 50000000;  // ~50M addresses
        uint64_t min_satoshis = 100000;      // 0.001 BTC minimum balance
        uint32_t seed = 0x5F3759DF;          // MurmurHash3 seed
        bool include_p2pkh = true;           // Legacy addresses (1...)
        bool include_p2sh = true;            // Script hash addresses (3...)
        bool include_p2wpkh = true;          // Native SegWit (bc1q...)

        Config() = default;
    };

    UTXOBloomBuilder() : config_() {
        calculate_parameters();
        allocate_filter();
    }

    explicit UTXOBloomBuilder(const Config& config)
        : config_(config) {
        calculate_parameters();
        allocate_filter();
    }

    /**
     * Process utxo-dump CSV file.
     * Expected format: txid,vout,address,amount
     * or: address,amount (simplified)
     */
    void process_csv(const std::string& path, bool show_progress = true) {
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("Cannot open CSV file: " + path);
        }

        // Get file size for progress calculation
        file.seekg(0, std::ios::end);
        uint64_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::string line;
        bool first_line = true;
        uint64_t lines_processed = 0;
        uint64_t bytes_processed = 0;
        uint64_t addresses_added = 0;
        uint64_t addresses_skipped = 0;

        auto start_time = std::chrono::steady_clock::now();
        auto last_update = start_time;
        const auto update_interval = std::chrono::milliseconds(500);

        while (std::getline(file, line)) {
            bytes_processed += line.size() + 1;  // +1 for newline

            if (line.empty() || line[0] == '#') continue;

            // Skip header
            if (first_line && (line.find("txid") != std::string::npos ||
                               line.find("address") != std::string::npos)) {
                first_line = false;
                continue;
            }
            first_line = false;

            uint64_t before = elements_added_;
            process_line(line);
            if (elements_added_ > before) {
                addresses_added++;
            } else {
                addresses_skipped++;
            }
            lines_processed++;

            // Update progress display
            if (show_progress) {
                auto now = std::chrono::steady_clock::now();
                if (now - last_update >= update_interval) {
                    last_update = now;
                    display_progress(bytes_processed, file_size, lines_processed,
                                   addresses_added, addresses_skipped, start_time);
                }
            }
        }

        // Final progress update
        if (show_progress) {
            display_progress(bytes_processed, file_size, lines_processed,
                           addresses_added, addresses_skipped, start_time, true);
            std::cout << "\n";
        }
    }

private:
    void display_progress(uint64_t bytes_done, uint64_t bytes_total,
                         uint64_t lines, uint64_t added, uint64_t skipped,
                         std::chrono::steady_clock::time_point start_time,
                         bool is_final = false) {
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - start_time).count();

        double progress = (bytes_total > 0) ?
            (static_cast<double>(bytes_done) / bytes_total) : 0.0;
        double percent = progress * 100.0;

        // Calculate speeds
        double lines_per_sec = (elapsed_sec > 0) ? (lines / elapsed_sec) : 0;
        (void)lines_per_sec;  // Suppress unused variable warning
        double mb_per_sec = (elapsed_sec > 0) ?
            (static_cast<double>(bytes_done) / (1024*1024) / elapsed_sec) : 0;

        // ETA calculation
        double eta_sec = 0;
        if (progress > 0.01 && progress < 1.0) {
            eta_sec = (elapsed_sec / progress) * (1.0 - progress);
        }

        // Format ETA string
        std::string eta_str;
        if (eta_sec > 3600) {
            int hours = static_cast<int>(eta_sec / 3600);
            int mins = static_cast<int>((static_cast<int>(eta_sec) % 3600) / 60);
            eta_str = std::to_string(hours) + "h " + std::to_string(mins) + "m";
        } else if (eta_sec > 60) {
            int mins = static_cast<int>(eta_sec / 60);
            int secs = static_cast<int>(eta_sec) % 60;
            eta_str = std::to_string(mins) + "m " + std::to_string(secs) + "s";
        } else if (eta_sec > 0) {
            eta_str = std::to_string(static_cast<int>(eta_sec)) + "s";
        } else {
            eta_str = "calculating...";
        }

        // Build progress bar
        const int bar_width = 30;
        int filled = static_cast<int>(progress * bar_width);
        std::string bar(filled, '#');
        bar += std::string(bar_width - filled, '-');

        // Build output string first, then pad to fixed width
        std::ostringstream oss;
        oss << "  [" << bar << "] "
            << std::fixed << std::setprecision(1) << percent << "%";

        // Show stats
        oss << "  " << format_number(added) << " addresses";
        if (skipped > 0) {
            oss << " (" << format_number(skipped) << " below min)";
        }

        if (!is_final) {
            oss << "  ETA: " << eta_str;
            oss << "  [" << std::fixed << std::setprecision(1)
                << mb_per_sec << " MB/s]";
        } else {
            oss << "  Done in " << format_duration(elapsed_sec);
        }

        // Pad to fixed width to overwrite any previous longer output
        std::string output = oss.str();
        const size_t min_width = 120;  // Wider to handle all variations
        if (output.size() < min_width) {
            output.append(min_width - output.size(), ' ');
        }

        std::cout << "\r" << output << std::flush;
    }

    static std::string format_number(uint64_t n) {
        if (n >= 1000000) {
            return std::to_string(n / 1000000) + "." +
                   std::to_string((n % 1000000) / 100000) + "M";
        } else if (n >= 1000) {
            return std::to_string(n / 1000) + "." +
                   std::to_string((n % 1000) / 100) + "K";
        }
        return std::to_string(n);
    }

    static std::string format_duration(double seconds) {
        if (seconds >= 3600) {
            int hours = static_cast<int>(seconds / 3600);
            int mins = static_cast<int>((static_cast<int>(seconds) % 3600) / 60);
            return std::to_string(hours) + "h " + std::to_string(mins) + "m";
        } else if (seconds >= 60) {
            int mins = static_cast<int>(seconds / 60);
            int secs = static_cast<int>(seconds) % 60;
            return std::to_string(mins) + "m " + std::to_string(secs) + "s";
        }
        return std::to_string(static_cast<int>(seconds)) + "s";
    }

public:
    /**
     * Add a single H160 hash to the filter.
     */
    void add_h160(const H160& h160) {
        auto [h1, h2] = murmurhash3_128(h160.data, 20, config_.seed);

        for (uint32_t i = 0; i < num_hashes_; i++) {
            uint64_t hash = h1 + i * h2;
            uint64_t bit_idx = hash % num_bits_;
            set_bit(bit_idx);
        }

        elements_added_++;
    }

    /**
     * Check if an H160 might be in the filter.
     */
    bool probably_contains(const H160& h160) const {
        auto [h1, h2] = murmurhash3_128(h160.data, 20, config_.seed);

        for (uint32_t i = 0; i < num_hashes_; i++) {
            uint64_t hash = h1 + i * h2;
            uint64_t bit_idx = hash % num_bits_;
            if (!get_bit(bit_idx)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Save filter to .blf file.
     * Format: 128-byte header + bit array (128-byte aligned)
     */
    void save(const std::string& path) const {
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot create BLF file: " + path);
        }

        // Prepare header
        BloomFilterHeader header;
        header.num_bits = num_bits_;
        header.num_hashes = num_hashes_;
        header.seed = config_.seed;
        header.num_elements = elements_added_;
        header.target_fp_rate = config_.target_fp_rate;
        header.data_offset = sizeof(BloomFilterHeader);

        // Write header
        file.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // Write bit array
        file.write(reinterpret_cast<const char*>(bits_.data()), bits_.size());

        // Pad to 128-byte boundary
        size_t total = sizeof(header) + bits_.size();
        size_t padding = (128 - (total % 128)) % 128;
        if (padding > 0) {
            std::vector<uint8_t> pad(padding, 0);
            file.write(reinterpret_cast<const char*>(pad.data()), padding);
        }
    }

    /**
     * Load filter from .blf file.
     */
    static UTXOBloomBuilder load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open BLF file: " + path);
        }

        BloomFilterHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(header));

        if (std::string(header.magic, 4) != "BLF1") {
            throw std::runtime_error("Invalid BLF file format");
        }

        Config config;
        config.target_fp_rate = header.target_fp_rate;
        config.expected_elements = header.num_elements;
        config.seed = header.seed;

        UTXOBloomBuilder builder(config);
        builder.num_bits_ = header.num_bits;
        builder.num_hashes_ = header.num_hashes;
        builder.elements_added_ = header.num_elements;

        // Resize and read bit array
        builder.bits_.resize((header.num_bits + 7) / 8);
        file.seekg(header.data_offset);
        file.read(reinterpret_cast<char*>(builder.bits_.data()), builder.bits_.size());

        return builder;
    }

    // Getters
    uint64_t num_bits() const { return num_bits_; }
    uint32_t num_hashes() const { return num_hashes_; }
    uint64_t elements_added() const { return elements_added_; }
    size_t size_bytes() const { return bits_.size(); }

    double estimated_fp_rate() const {
        // p = (1 - e^(-k*n/m))^k
        double k = num_hashes_;
        double n = elements_added_;
        double m = num_bits_;
        return std::pow(1.0 - std::exp(-k * n / m), k);
    }

    const std::vector<uint8_t>& data() const { return bits_; }

    /**
     * Get statistics for display.
     */
    struct Stats {
        uint64_t num_bits;
        uint32_t num_hashes;
        uint64_t elements_added;
        size_t size_mb;
        double estimated_fp_rate;
        double fill_ratio;
    };

    Stats get_stats() const {
        Stats s;
        s.num_bits = num_bits_;
        s.num_hashes = num_hashes_;
        s.elements_added = elements_added_;
        s.size_mb = bits_.size() / (1024 * 1024);
        s.estimated_fp_rate = estimated_fp_rate();

        // Count set bits
        uint64_t set_bits = 0;
        for (uint8_t byte : bits_) {
            set_bits += POPCOUNT(byte);
        }
        s.fill_ratio = static_cast<double>(set_bits) / num_bits_;

        return s;
    }

private:
    Config config_;
    std::vector<uint8_t> bits_;
    uint64_t num_bits_;
    uint32_t num_hashes_;
    uint64_t elements_added_ = 0;

    void calculate_parameters() {
        // Optimal number of bits: m = -n * ln(p) / (ln(2)^2)
        double n = config_.expected_elements;
        double p = config_.target_fp_rate;
        double ln2_sq = 0.693147 * 0.693147;

        num_bits_ = static_cast<uint64_t>(-n * std::log(p) / ln2_sq);

        // Round up to 64-bit boundary for GPU efficiency
        num_bits_ = ((num_bits_ + 63) / 64) * 64;

        // Optimal number of hash functions: k = (m/n) * ln(2)
        num_hashes_ = static_cast<uint32_t>(
            std::round((static_cast<double>(num_bits_) / n) * 0.693147)
        );

        // Clamp to reasonable range
        num_hashes_ = std::max(1u, std::min(32u, num_hashes_));
    }

    void allocate_filter() {
        size_t bytes = (num_bits_ + 7) / 8;
        // Round up to 128-byte boundary for GPU
        bytes = ((bytes + 127) / 128) * 128;
        bits_.resize(bytes, 0);
    }

    void set_bit(uint64_t idx) {
        bits_[idx / 8] |= (1 << (idx % 8));
    }

    bool get_bit(uint64_t idx) const {
        return (bits_[idx / 8] >> (idx % 8)) & 1;
    }

    void process_line(const std::string& line) {
        // Parse CSV line
        std::vector<std::string> fields;
        std::istringstream iss(line);
        std::string field;

        while (std::getline(iss, field, ',')) {
            // Trim whitespace
            field.erase(0, field.find_first_not_of(" \t"));
            field.erase(field.find_last_not_of(" \t\r\n") + 1);
            fields.push_back(field);
        }

        if (fields.size() < 2) return;

        // Determine format
        std::string address;
        uint64_t satoshis = 0;

        try {
            if (fields.size() >= 6) {
                // bitcoin-utxo-dump format: count,txid,vout,amount,type,address
                address = fields[5];
                satoshis = std::stoull(fields[3]);
            } else if (fields.size() >= 4) {
                // Alternative format: txid,vout,address,amount
                address = fields[2];
                satoshis = std::stoull(fields[3]);
            } else {
                // Simple format: address,amount
                address = fields[0];
                satoshis = std::stoull(fields[1]);
            }
        } catch (const std::exception&) {
            // Skip lines with invalid numbers (e.g., header row)
            return;
        }

        // Filter by minimum balance
        if (satoshis < config_.min_satoshis) return;

        // Extract H160 from address
        try {
            H160 h160 = decode_address(address);
            add_h160(h160);
        } catch (const std::exception&) {
            // Skip invalid addresses (unsupported types like P2WSH, P2TR)
        }
    }

    /**
     * Decode Bitcoin address to H160.
     * Supports P2PKH (1...), P2SH (3...), and P2WPKH (bc1q...).
     */
    H160 decode_address(const std::string& address) const {
        H160 result;

        if (address.empty()) {
            throw std::invalid_argument("Empty address");
        }

        // Check address type
        if (address[0] == '1' && config_.include_p2pkh) {
            // P2PKH: Base58Check decode
            result = base58check_decode(address);
        } else if (address[0] == '3' && config_.include_p2sh) {
            // P2SH: Base58Check decode
            result = base58check_decode(address);
        } else if (address.substr(0, 4) == "bc1q" && config_.include_p2wpkh) {
            // P2WPKH: Bech32 decode
            result = bech32_decode(address);
        } else if (address.size() == 40) {
            // Raw H160 hex
            result = H160::from_hex(address);
        } else {
            throw std::invalid_argument("Unsupported address format");
        }

        return result;
    }

    // Base58 alphabet
    static constexpr const char* BASE58_ALPHABET =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    H160 base58check_decode(const std::string& address) const {
        // Decode base58
        std::vector<uint8_t> decoded;
        decoded.reserve(25);

        // Build value
        std::vector<uint8_t> num;
        for (char c : address) {
            const char* pos = std::strchr(BASE58_ALPHABET, c);
            if (!pos) {
                throw std::invalid_argument("Invalid base58 character");
            }

            int carry = pos - BASE58_ALPHABET;
            for (auto& byte : num) {
                int val = byte * 58 + carry;
                byte = val & 0xff;
                carry = val >> 8;
            }
            while (carry) {
                num.push_back(carry & 0xff);
                carry >>= 8;
            }
        }

        // Count leading zeros
        size_t leading_zeros = 0;
        for (char c : address) {
            if (c == '1') leading_zeros++;
            else break;
        }

        // Build result (reverse)
        decoded.resize(leading_zeros, 0);
        for (auto it = num.rbegin(); it != num.rend(); ++it) {
            decoded.push_back(*it);
        }

        // Should be 25 bytes: version (1) + H160 (20) + checksum (4)
        if (decoded.size() != 25) {
            throw std::invalid_argument("Invalid address length");
        }

        // Extract H160 (skip version byte, ignore checksum)
        H160 result;
        std::memcpy(result.data, decoded.data() + 1, 20);

        return result;
    }

    H160 bech32_decode(const std::string& address) const {
        // Simplified bech32 decode for bc1q addresses
        // bc1q addresses are: hrp(bc) + separator(1) + data(witness version + H160)

        if (address.size() < 14 || address.substr(0, 4) != "bc1q") {
            throw std::invalid_argument("Invalid bech32 address");
        }

        static const char* BECH32_ALPHABET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

        // Decode data part (skip "bc1")
        std::vector<uint8_t> data;
        for (size_t i = 3; i < address.size(); i++) {
            const char* pos = std::strchr(BECH32_ALPHABET, std::tolower(address[i]));
            if (!pos) {
                throw std::invalid_argument("Invalid bech32 character");
            }
            data.push_back(pos - BECH32_ALPHABET);
        }

        // Convert from 5-bit to 8-bit groups
        // First byte is witness version (0 for P2WPKH)
        // Rest is the H160 in 5-bit encoding

        std::vector<uint8_t> converted;
        uint32_t acc = 0;
        int bits = 0;

        for (size_t i = 1; i < data.size() - 6; i++) {  // Skip version and checksum
            acc = (acc << 5) | data[i];
            bits += 5;
            while (bits >= 8) {
                bits -= 8;
                converted.push_back((acc >> bits) & 0xff);
            }
        }

        if (converted.size() != 20) {
            throw std::invalid_argument("Invalid witness program length");
        }

        H160 result;
        std::memcpy(result.data, converted.data(), 20);

        return result;
    }
};

}  // namespace utxo
}  // namespace collider
