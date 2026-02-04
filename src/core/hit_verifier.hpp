/**
 * Hit Verifier - False Positive Filtering
 *
 * Verifies bloom filter hits against the actual UTXO database.
 * Uses a memory-mapped index for O(1) lookups on verified addresses.
 *
 * Pipeline:
 *   GPU Bloom Hit -> Host Verification -> Confirmed Match -> Output
 *
 * The bloom filter is tuned for 0.001% FP rate, meaning ~1 in 100,000
 * bloom hits will be false positives. This verifier eliminates them.
 */

#pragma once

#include "types.hpp"
#include "../tools/utxo_bloom_builder.hpp"
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <mutex>
#include <atomic>
#include <chrono>

namespace collider {

/**
 * UTXO entry with balance information.
 */
struct UTXOEntry {
    utxo::H160 h160;
    uint64_t satoshis;
    std::string address;  // Original address string
};

/**
 * Verified hit result.
 */
struct VerifiedHit {
    std::string passphrase;
    uint8_t private_key[32];
    utxo::H160 h160;
    uint64_t satoshis;
    std::string address;
    std::chrono::system_clock::time_point timestamp;
};

/**
 * Hit Verifier
 *
 * Maintains an in-memory hash set of target H160s for O(1) verification.
 * Falls back to disk-based index for very large datasets.
 */
class HitVerifier {
public:
    struct Config {
        size_t max_memory_entries = 100'000'000;  // Max entries in memory
        bool use_disk_fallback = true;            // Use disk index if > max
        std::string disk_index_path;              // Path for disk index
        bool log_false_positives = false;         // Track FP stats

        Config() = default;
    };

    HitVerifier() : config_() {}
    explicit HitVerifier(const Config& config) : config_(config) {}

    /**
     * Load UTXO data from CSV file.
     * @param path Path to utxo-dump CSV
     * @param min_satoshis Minimum balance filter
     */
    void load_from_csv(const std::string& path, uint64_t min_satoshis = 100000) {
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("Cannot open UTXO CSV: " + path);
        }

        std::string line;
        bool first_line = true;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            // Skip header
            if (first_line && (line.find("txid") != std::string::npos ||
                               line.find("address") != std::string::npos)) {
                first_line = false;
                continue;
            }
            first_line = false;

            try {
                auto entry = parse_csv_line(line, min_satoshis);
                if (entry.satoshis >= min_satoshis) {
                    add_entry(entry);
                }
            } catch (const std::exception&) {
                // Skip invalid lines
            }
        }
    }

    /**
     * Add a single UTXO entry.
     */
    void add_entry(const UTXOEntry& entry) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Store in hash set for O(1) lookup
        H160Key key;
        std::memcpy(key.data, entry.h160.data, 20);
        entries_[key] = entry;

        total_entries_++;
    }

    /**
     * Verify a bloom filter hit.
     * @param h160 The H160 hash to verify
     * @return Entry if verified, nullopt if false positive
     */
    std::optional<UTXOEntry> verify(const utxo::H160& h160) const {
        std::lock_guard<std::mutex> lock(mutex_);

        H160Key key;
        std::memcpy(key.data, h160.data, 20);

        auto it = entries_.find(key);
        if (it != entries_.end()) {
            verified_hits_++;
            return it->second;
        }

        false_positives_++;
        return std::nullopt;
    }

    /**
     * Batch verify multiple H160s.
     * @param h160s Array of H160 hashes
     * @param count Number of H160s
     * @return Vector of verified entries
     */
    std::vector<UTXOEntry> verify_batch(
        const utxo::H160* h160s,
        size_t count
    ) const {
        std::vector<UTXOEntry> results;
        results.reserve(count / 100);  // Expect ~1% hits at most

        std::lock_guard<std::mutex> lock(mutex_);

        for (size_t i = 0; i < count; i++) {
            H160Key key;
            std::memcpy(key.data, h160s[i].data, 20);

            auto it = entries_.find(key);
            if (it != entries_.end()) {
                results.push_back(it->second);
                verified_hits_++;
            } else {
                false_positives_++;
            }
        }

        return results;
    }

    /**
     * Get verification statistics.
     */
    struct Stats {
        uint64_t total_entries;
        uint64_t verified_hits;
        uint64_t false_positives;
        double fp_rate;
        size_t memory_bytes;
    };

    Stats get_stats() const {
        Stats s;
        s.total_entries = total_entries_;
        s.verified_hits = verified_hits_;
        s.false_positives = false_positives_;
        s.fp_rate = (verified_hits_ + false_positives_) > 0 ?
            static_cast<double>(false_positives_) /
            (verified_hits_ + false_positives_) : 0.0;
        s.memory_bytes = entries_.size() * (sizeof(H160Key) + sizeof(UTXOEntry));
        return s;
    }

    /**
     * Check if an H160 is in the verification set.
     */
    bool contains(const utxo::H160& h160) const {
        std::lock_guard<std::mutex> lock(mutex_);

        H160Key key;
        std::memcpy(key.data, h160.data, 20);

        return entries_.find(key) != entries_.end();
    }

    /**
     * Get total value of all tracked UTXOs.
     */
    uint64_t total_satoshis() const {
        std::lock_guard<std::mutex> lock(mutex_);

        uint64_t total = 0;
        for (const auto& [key, entry] : entries_) {
            total += entry.satoshis;
        }
        return total;
    }

    /**
     * Get entries sorted by balance (highest first).
     */
    std::vector<UTXOEntry> get_top_entries(size_t n) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<UTXOEntry> all;
        all.reserve(entries_.size());

        for (const auto& [key, entry] : entries_) {
            all.push_back(entry);
        }

        std::partial_sort(
            all.begin(),
            all.begin() + std::min(n, all.size()),
            all.end(),
            [](const UTXOEntry& a, const UTXOEntry& b) {
                return a.satoshis > b.satoshis;
            }
        );

        if (all.size() > n) {
            all.resize(n);
        }

        return all;
    }

    /**
     * Save verification set to binary file.
     */
    void save(const std::string& path) const {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot create verification file: " + path);
        }

        // Header
        file.write("UVRF", 4);  // Magic
        uint32_t version = 1;
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));

        uint64_t count = entries_.size();
        file.write(reinterpret_cast<const char*>(&count), sizeof(count));

        // Entries
        for (const auto& [key, entry] : entries_) {
            file.write(reinterpret_cast<const char*>(key.data), 20);
            file.write(reinterpret_cast<const char*>(&entry.satoshis), sizeof(entry.satoshis));

            uint32_t addr_len = entry.address.size();
            file.write(reinterpret_cast<const char*>(&addr_len), sizeof(addr_len));
            file.write(entry.address.data(), addr_len);
        }
    }

    /**
     * Load verification set from binary file.
     */
    void load(const std::string& path) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open verification file: " + path);
        }

        // Header
        char magic[4];
        file.read(magic, 4);
        if (std::string(magic, 4) != "UVRF") {
            throw std::runtime_error("Invalid verification file format");
        }

        uint32_t version;
        file.read(reinterpret_cast<char*>(&version), sizeof(version));

        uint64_t count;
        file.read(reinterpret_cast<char*>(&count), sizeof(count));

        entries_.clear();
        entries_.reserve(count);

        // Entries
        for (uint64_t i = 0; i < count && file.good(); i++) {
            UTXOEntry entry;
            H160Key key;

            file.read(reinterpret_cast<char*>(key.data), 20);
            std::memcpy(entry.h160.data, key.data, 20);

            file.read(reinterpret_cast<char*>(&entry.satoshis), sizeof(entry.satoshis));

            uint32_t addr_len;
            file.read(reinterpret_cast<char*>(&addr_len), sizeof(addr_len));
            entry.address.resize(addr_len);
            file.read(entry.address.data(), addr_len);

            entries_[key] = std::move(entry);
        }

        total_entries_ = entries_.size();
    }

    size_t size() const { return entries_.size(); }

private:
    Config config_;

    // H160 key for unordered_map
    struct H160Key {
        uint8_t data[20];

        bool operator==(const H160Key& other) const {
            return std::memcmp(data, other.data, 20) == 0;
        }
    };

    struct H160Hash {
        size_t operator()(const H160Key& key) const {
            // Use first 8 bytes as hash (H160 is already a hash)
            return *reinterpret_cast<const uint64_t*>(key.data);
        }
    };

    std::unordered_map<H160Key, UTXOEntry, H160Hash> entries_;
    mutable std::mutex mutex_;

    std::atomic<uint64_t> total_entries_{0};
    mutable std::atomic<uint64_t> verified_hits_{0};
    mutable std::atomic<uint64_t> false_positives_{0};

    UTXOEntry parse_csv_line(const std::string& line, uint64_t min_satoshis) const {
        UTXOEntry entry;

        std::vector<std::string> fields;
        std::istringstream iss(line);
        std::string field;

        while (std::getline(iss, field, ',')) {
            // Trim
            field.erase(0, field.find_first_not_of(" \t"));
            field.erase(field.find_last_not_of(" \t\r\n") + 1);
            fields.push_back(field);
        }

        if (fields.size() >= 4) {
            // Full format: txid,vout,address,amount
            entry.address = fields[2];
            entry.satoshis = std::stoull(fields[3]);
        } else if (fields.size() >= 2) {
            // Simple format: address,amount
            entry.address = fields[0];
            entry.satoshis = std::stoull(fields[1]);
        } else {
            throw std::invalid_argument("Invalid CSV format");
        }

        // Decode address to H160
        entry.h160 = decode_address(entry.address);

        return entry;
    }

    // Base58 alphabet for Bitcoin address decoding
    static constexpr const char* BASE58_ALPHABET =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    utxo::H160 decode_address(const std::string& address) const {
        if (address.empty()) {
            throw std::invalid_argument("Empty address");
        }

        // Raw H160 hex
        if (address.size() == 40) {
            return utxo::H160::from_hex(address);
        }

        // P2PKH (1...) or P2SH (3...)
        if (address[0] == '1' || address[0] == '3') {
            return base58check_decode(address);
        }

        // P2WPKH (bc1q...)
        if (address.size() >= 4 && address.substr(0, 4) == "bc1q") {
            return bech32_decode(address);
        }

        throw std::invalid_argument("Unsupported address format: " + address);
    }

    utxo::H160 base58check_decode(const std::string& address) const {
        // Decode base58
        std::vector<uint8_t> num;
        for (char c : address) {
            const char* pos = std::strchr(BASE58_ALPHABET, c);
            if (!pos) {
                throw std::invalid_argument("Invalid base58 character");
            }

            int carry = static_cast<int>(pos - BASE58_ALPHABET);
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
        size_t zeros = 0;
        for (char c : address) {
            if (c != '1') break;
            zeros++;
        }

        // Build decoded bytes (reversed)
        std::vector<uint8_t> decoded(zeros, 0);
        for (auto it = num.rbegin(); it != num.rend(); ++it) {
            decoded.push_back(*it);
        }

        // Should be 25 bytes: version (1) + H160 (20) + checksum (4)
        if (decoded.size() != 25) {
            throw std::invalid_argument("Invalid address length");
        }

        // Extract H160 (skip version byte, ignore checksum)
        utxo::H160 result;
        std::memcpy(result.data, decoded.data() + 1, 20);
        return result;
    }

    utxo::H160 bech32_decode(const std::string& address) const {
        // Simplified bech32 decode for bc1q addresses
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
            data.push_back(static_cast<uint8_t>(pos - BECH32_ALPHABET));
        }

        // Convert from 5-bit to 8-bit groups
        std::vector<uint8_t> converted;
        uint32_t acc = 0;
        int bits = 0;

        for (size_t i = 1; i < data.size() - 6; i++) {  // Skip version and checksum
            acc = (acc << 5) | data[i];
            bits += 5;
            if (bits >= 8) {
                bits -= 8;
                converted.push_back((acc >> bits) & 0xff);
            }
        }

        if (converted.size() != 20) {
            throw std::invalid_argument("Invalid witness program length");
        }

        utxo::H160 result;
        std::memcpy(result.data, converted.data(), 20);
        return result;
    }
};

/**
 * Hit Handler - Processes verified hits.
 *
 * Logs hits, writes to potfile, and handles notifications.
 */
class HitHandler {
public:
    struct Config {
        std::string potfile_path = "collider.pot";
        std::string hits_log_path = "hits.log";
        bool console_output = true;
        bool json_output = false;

        Config() = default;
    };

    HitHandler() : config_() {
        potfile_.open(config_.potfile_path, std::ios::app);
        hits_log_.open(config_.hits_log_path, std::ios::app);
    }

    explicit HitHandler(const Config& config) : config_(config) {
        // Open potfile
        potfile_.open(config_.potfile_path, std::ios::app);
        hits_log_.open(config_.hits_log_path, std::ios::app);
    }

    /**
     * Handle a verified hit.
     */
    void handle(const VerifiedHit& hit) {
        std::lock_guard<std::mutex> lock(mutex_);

        hits_count_++;
        total_satoshis_ += hit.satoshis;

        // Write to potfile (hashcat format)
        if (potfile_) {
            potfile_ << hit.h160.to_hex() << ":" << hit.passphrase << "\n";
            potfile_.flush();
        }

        // Write to hits log (detailed)
        if (hits_log_) {
            auto time = std::chrono::system_clock::to_time_t(hit.timestamp);

            hits_log_ << "=== HIT FOUND ===\n"
                      << "Time: " << std::ctime(&time)
                      << "Address: " << hit.address << "\n"
                      << "H160: " << hit.h160.to_hex() << "\n"
                      << "Passphrase: " << hit.passphrase << "\n"
                      << "Private Key: ";

            for (int i = 0; i < 32; i++) {
                char hex[3];
                std::snprintf(hex, sizeof(hex), "%02x", hit.private_key[i]);
                hits_log_ << hex;
            }

            hits_log_ << "\n"
                      << "Balance: " << hit.satoshis << " satoshis ("
                      << (hit.satoshis / 100000000.0) << " BTC)\n"
                      << "================\n\n";
            hits_log_.flush();
        }

        // Console output
        if (config_.console_output) {
            printf("\n*** HIT FOUND ***\n");
            printf("Address: %s\n", hit.address.c_str());
            printf("Passphrase: %s\n", hit.passphrase.c_str());
            printf("Balance: %.8f BTC\n", hit.satoshis / 100000000.0);
            printf("*****************\n\n");
        }
    }

    /**
     * Get hit statistics.
     */
    struct Stats {
        uint64_t hits_count;
        uint64_t total_satoshis;
        double total_btc;
    };

    Stats get_stats() const {
        Stats s;
        s.hits_count = hits_count_;
        s.total_satoshis = total_satoshis_;
        s.total_btc = total_satoshis_ / 100000000.0;
        return s;
    }

private:
    Config config_;
    std::ofstream potfile_;
    std::ofstream hits_log_;
    std::mutex mutex_;

    std::atomic<uint64_t> hits_count_{0};
    std::atomic<uint64_t> total_satoshis_{0};
};

}  // namespace collider
