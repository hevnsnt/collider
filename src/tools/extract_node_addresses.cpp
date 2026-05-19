/**
 * extract_node_addresses -- parse a Bitcoin Core full-node blocks/ directory
 * and emit every h160-shaped address ever seen on the chain to a CSV that
 * build_bloom can consume directly.
 *
 * built for the --track-empty-hits workflow.
 *
 * Input:  one or more blk*.dat files (Bitcoin Core block storage format)
 * Output: one address per line on stdout (or to -o <file>):
 *           <h160_hex>,<script_type>
 *           1ec84ff80a8459e72ae04bfe6e5c01bd34a16f00,P2PKH
 *
 * Coverage: emits the three h160-shaped output types that the brain-wallet
 * h160 bloom can probe:
 *   - P2PKH   (legacy)
 *   - P2SH    (BIP-13 / BIP-16, also covers nested SegWit)
 *   - P2WPKH  (BIP-141 native SegWit v0, 20-byte witness)
 *
 * Skipped (32-byte witness programs, not h160-shaped):
 *   - P2WSH   (P2WPKH's 32-byte cousin)
 *   - P2TR    (taproot, witness v1)
 *   - bare P2PK / multisig (rare, ~0.5% of outputs; extracting them would
 *     require running SHA256+RIPEMD160 over the embedded pubkey)
 *
 * Performance: ~80-150 MB/s on commodity SSDs with the default 16 MiB
 * read buffer. 1 TB of blocks ~= 2-4 hours single-threaded. Output is
 * append-only stdout so you can `tee` and watch progress.
 *
 * Build:  produced by the COLLIDER_BUILD_TOOLS=ON CMake target.
 * Usage:  extract_node_addresses /mnt/hdd/bitcoin/bitcoin/blocks > seen.csv
 *         extract_node_addresses /path/blocks -o seen.csv
 *
 * The tool reads blk*.dat files in numerical order. Each file is a
 * sequence of frames:
 *     [4-byte network magic] [4-byte LE block size] [block bytes]
 * Tail garbage / partial frames at EOF are tolerated (a freshly-syncing
 * node may have written a block size but not the data yet).
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Bitcoin mainnet p2p magic (Pieter Wuille's network bytes).
constexpr uint32_t kMainnetMagic = 0xD9B4BEF9u;  // little-endian when read as uint32

// scriptPubKey opcodes we care about.
constexpr uint8_t OP_DUP         = 0x76;
constexpr uint8_t OP_HASH160     = 0xA9;
constexpr uint8_t OP_EQUAL       = 0x87;
constexpr uint8_t OP_EQUALVERIFY = 0x88;
constexpr uint8_t OP_CHECKSIG    = 0xAC;
constexpr uint8_t OP_0           = 0x00;
constexpr uint8_t PUSH_20        = 0x14;  // pushdata 20 bytes
constexpr uint8_t PUSH_32        = 0x20;  // pushdata 32 bytes

// Bitcoin "compact-size" varint decoder. Standard form: <=0xFC inline,
// 0xFD => uint16_t LE, 0xFE => uint32_t LE, 0xFF => uint64_t LE.
struct VarintResult {
    uint64_t value;
    size_t   bytes;
    bool     ok;
};

VarintResult read_varint(const uint8_t* buf, size_t len) {
    if (len < 1) return {0, 0, false};
    uint8_t first = buf[0];
    if (first <= 0xFC)        return {first, 1, true};
    if (first == 0xFD) {
        if (len < 3) return {0, 0, false};
        uint16_t v = static_cast<uint16_t>(buf[1]) |
                     (static_cast<uint16_t>(buf[2]) << 8);
        return {v, 3, true};
    }
    if (first == 0xFE) {
        if (len < 5) return {0, 0, false};
        uint32_t v = static_cast<uint32_t>(buf[1])        |
                     (static_cast<uint32_t>(buf[2]) <<  8) |
                     (static_cast<uint32_t>(buf[3]) << 16) |
                     (static_cast<uint32_t>(buf[4]) << 24);
        return {v, 5, true};
    }
    // 0xFF
    if (len < 9) return {0, 0, false};
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= static_cast<uint64_t>(buf[1 + i]) << (i * 8);
    return {v, 9, true};
}

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])        |
           (static_cast<uint32_t>(p[1]) <<  8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

// Tagged h160 result. type is one of: "P2PKH", "P2SH", "P2WPKH", or "" if
// not an h160-shaped script.
struct H160Hit {
    uint8_t     data[20];
    const char* type;  // string literal -- not owned
};

bool extract_h160(const uint8_t* script, size_t len, H160Hit& out) {
    // P2PKH: OP_DUP OP_HASH160 <20> <h160> OP_EQUALVERIFY OP_CHECKSIG (25 bytes)
    if (len == 25 &&
        script[0]  == OP_DUP &&
        script[1]  == OP_HASH160 &&
        script[2]  == PUSH_20 &&
        script[23] == OP_EQUALVERIFY &&
        script[24] == OP_CHECKSIG) {
        std::memcpy(out.data, script + 3, 20);
        out.type = "P2PKH";
        return true;
    }
    // P2SH: OP_HASH160 <20> <h160> OP_EQUAL (23 bytes)
    if (len == 23 &&
        script[0]  == OP_HASH160 &&
        script[1]  == PUSH_20 &&
        script[22] == OP_EQUAL) {
        std::memcpy(out.data, script + 2, 20);
        out.type = "P2SH";
        return true;
    }
    // P2WPKH: OP_0 <20> <h160> (22 bytes total). Witness program length 20.
    if (len == 22 &&
        script[0] == OP_0 &&
        script[1] == PUSH_20) {
        std::memcpy(out.data, script + 2, 20);
        out.type = "P2WPKH";
        return true;
    }
    return false;
}

void emit_h160(std::FILE* out, const H160Hit& h) {
    char buf[44];  // 40 hex + ',' + 5 (longest type "P2WPKH") + '\n' + NUL
    for (int i = 0; i < 20; i++) {
        std::snprintf(buf + i * 2, 3, "%02x", h.data[i]);
    }
    buf[40] = ',';
    size_t off = 41;
    const char* t = h.type;
    while (*t) buf[off++] = *t++;
    buf[off++] = '\n';
    std::fwrite(buf, 1, off, out);
}

// ---------------------------------------------------------------------------
// Block parsing: walk every transaction's outputs.
// ---------------------------------------------------------------------------

struct ParseStats {
    uint64_t blocks    = 0;
    uint64_t txs       = 0;
    uint64_t outputs   = 0;
    uint64_t p2pkh     = 0;
    uint64_t p2sh      = 0;
    uint64_t p2wpkh    = 0;
    uint64_t skipped   = 0;  // outputs that didn't match an h160 shape
    uint64_t parse_err = 0;  // tx/block parse errors (truncated, bad varint)
};

// Skip a script with a leading varint length. Returns false on truncation.
bool skip_script(const uint8_t*& p, const uint8_t* end) {
    auto v = read_varint(p, static_cast<size_t>(end - p));
    if (!v.ok) return false;
    p += v.bytes;
    if (static_cast<uint64_t>(end - p) < v.value) return false;
    p += v.value;
    return true;
}

// Parse one transaction's outputs and emit h160s. The full tx is parsed
// (we skip inputs, witness data, locktime) so we can move the cursor.
// Returns false on truncation / malformed tx.
bool parse_transaction(const uint8_t*& p, const uint8_t* end,
                       std::FILE* out, ParseStats& stats) {
    // 4-byte version (LE).
    if (end - p < 4) return false;
    p += 4;

    // SegWit marker: if next byte is 0x00 and the one after is non-zero,
    // this is a segwit tx (BIP-141). Consume the marker+flag and remember.
    bool is_segwit = false;
    if (end - p >= 2 && p[0] == 0x00 && p[1] != 0x00) {
        is_segwit = true;
        p += 2;
    }

    // Input count.
    auto vin_cnt = read_varint(p, static_cast<size_t>(end - p));
    if (!vin_cnt.ok) return false;
    p += vin_cnt.bytes;

    // Inputs. Each: 32-byte prev_hash + 4-byte vout + varint script + 4-byte sequence.
    for (uint64_t i = 0; i < vin_cnt.value; i++) {
        if (end - p < 36) return false;
        p += 36;  // prev_hash + vout
        if (!skip_script(p, end)) return false;  // scriptSig
        if (end - p < 4) return false;
        p += 4;  // sequence
    }

    // Output count.
    auto vout_cnt = read_varint(p, static_cast<size_t>(end - p));
    if (!vout_cnt.ok) return false;
    p += vout_cnt.bytes;

    // Outputs. Each: 8-byte amount + varint script + script bytes.
    for (uint64_t i = 0; i < vout_cnt.value; i++) {
        if (end - p < 8) return false;
        p += 8;  // amount
        auto sk = read_varint(p, static_cast<size_t>(end - p));
        if (!sk.ok) return false;
        p += sk.bytes;
        if (static_cast<uint64_t>(end - p) < sk.value) return false;
        const uint8_t* script = p;
        p += sk.value;
        ++stats.outputs;

        H160Hit hit;
        if (extract_h160(script, sk.value, hit)) {
            emit_h160(out, hit);
            if (std::strcmp(hit.type, "P2PKH")  == 0) ++stats.p2pkh;
            else if (std::strcmp(hit.type, "P2SH")   == 0) ++stats.p2sh;
            else if (std::strcmp(hit.type, "P2WPKH") == 0) ++stats.p2wpkh;
        } else {
            ++stats.skipped;
        }
    }

    // Witness data (segwit only). One stack per input; we skip it all.
    if (is_segwit) {
        for (uint64_t i = 0; i < vin_cnt.value; i++) {
            auto stack_cnt = read_varint(p, static_cast<size_t>(end - p));
            if (!stack_cnt.ok) return false;
            p += stack_cnt.bytes;
            for (uint64_t j = 0; j < stack_cnt.value; j++) {
                auto item = read_varint(p, static_cast<size_t>(end - p));
                if (!item.ok) return false;
                p += item.bytes;
                if (static_cast<uint64_t>(end - p) < item.value) return false;
                p += item.value;
            }
        }
    }

    // 4-byte locktime.
    if (end - p < 4) return false;
    p += 4;
    return true;
}

bool parse_block(const uint8_t* block, size_t len,
                 std::FILE* out, ParseStats& stats) {
    // 80-byte header: version(4) + prev_hash(32) + merkle(32) + time(4)
    //                 + bits(4) + nonce(4)
    if (len < 80) {
        ++stats.parse_err;
        return false;
    }
    const uint8_t* p   = block + 80;
    const uint8_t* end = block + len;

    auto tx_cnt = read_varint(p, static_cast<size_t>(end - p));
    if (!tx_cnt.ok) {
        ++stats.parse_err;
        return false;
    }
    p += tx_cnt.bytes;

    for (uint64_t i = 0; i < tx_cnt.value; i++) {
        if (!parse_transaction(p, end, out, stats)) {
            ++stats.parse_err;
            return false;
        }
        ++stats.txs;
    }
    ++stats.blocks;
    return true;
}

// ---------------------------------------------------------------------------
// blk*.dat framing: [4-byte magic][4-byte LE size][block bytes] repeat.
// ---------------------------------------------------------------------------

bool process_blk_file(const fs::path& path,
                      std::FILE* out, ParseStats& stats) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[!] cannot open %s\n", path.string().c_str());
        return false;
    }

    // Read in chunks; carry a small scratch buffer for cross-chunk frames.
    // 16 MiB buffer is large enough that >99% of blocks (largest mainnet
    // block is ~4 MiB at the post-segwit cap) fit cleanly.
    constexpr size_t kChunk = 16 * 1024 * 1024;
    std::vector<uint8_t> buf(kChunk);
    std::vector<uint8_t> carry;
    carry.reserve(kChunk);

    while (f) {
        f.read(reinterpret_cast<char*>(buf.data()), kChunk);
        std::streamsize n = f.gcount();
        if (n <= 0) break;

        // Combine carry + new chunk into a single scratch view.
        std::vector<uint8_t> scratch;
        scratch.reserve(carry.size() + static_cast<size_t>(n));
        scratch.insert(scratch.end(), carry.begin(), carry.end());
        scratch.insert(scratch.end(), buf.data(), buf.data() + n);

        const uint8_t* base = scratch.data();
        size_t pos = 0;
        size_t avail = scratch.size();

        while (pos + 8 <= avail) {
            uint32_t magic = read_u32_le(base + pos);
            // Tail of blk*.dat is sometimes zero-padded. Treat zero magic
            // as "end of useful data in this chunk"; everything past it is
            // dropped on the next iteration.
            if (magic == 0) {
                pos = avail;
                break;
            }
            if (magic != kMainnetMagic) {
                // Magic resync: advance one byte. Bitcoin Core has occasional
                // garbage between blocks on testnet; on mainnet this almost
                // never fires, but the cost is bounded.
                ++pos;
                continue;
            }
            uint32_t block_size = read_u32_le(base + pos + 4);
            if (pos + 8 + block_size > avail) {
                // Frame straddles the chunk boundary -- keep for next iter.
                break;
            }
            parse_block(base + pos + 8, block_size, out, stats);
            pos += 8 + block_size;
        }

        // Anything from pos onward becomes the carry for the next read.
        carry.assign(scratch.begin() + pos, scratch.end());
    }

    if (!carry.empty()) {
        // Could be the file's last partial frame from an in-progress sync.
        // Not an error; Bitcoin Core writes block headers atomically but
        // a frame in flight can be partial.
        std::fprintf(stderr,
                     "[*] %s: %zu trailing bytes (likely partial/in-flight "
                     "frame; safe to ignore)\n",
                     path.filename().string().c_str(), carry.size());
    }
    return true;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string blocks_dir = "/mnt/hdd/bitcoin/bitcoin/blocks";
    std::string output_path;          // empty => stdout
    bool quiet = false;
    size_t max_files = 0;             // 0 = unlimited

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-o" || a == "--output") {
            if (++i >= argc) { std::fprintf(stderr, "-o needs arg\n"); return 2; }
            output_path = argv[i];
        } else if (a == "-q" || a == "--quiet") {
            quiet = true;
        } else if (a == "--max-files") {
            if (++i >= argc) { std::fprintf(stderr, "--max-files needs arg\n"); return 2; }
            max_files = std::strtoull(argv[i], nullptr, 10);
        } else if (a == "-h" || a == "--help") {
            std::fprintf(stderr,
                "Usage: extract_node_addresses [<blocks-dir>] [-o <out.csv>] [-q] [--max-files N]\n"
                "  <blocks-dir>     default /mnt/hdd/bitcoin/bitcoin/blocks\n"
                "  -o <out.csv>     output file (default stdout)\n"
                "  -q               suppress progress lines on stderr\n"
                "  --max-files N    stop after N blk*.dat files (testing)\n");
            return 0;
        } else if (!a.empty() && a[0] != '-') {
            blocks_dir = a;
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    if (!fs::is_directory(blocks_dir)) {
        std::fprintf(stderr, "[!] not a directory: %s\n", blocks_dir.c_str());
        return 1;
    }

    // Collect all blk*.dat files and sort numerically.
    std::vector<fs::path> files;
    for (auto& ent : fs::directory_iterator(blocks_dir)) {
        auto name = ent.path().filename().string();
        if (name.size() == 12 &&
            name.rfind("blk", 0) == 0 &&
            name.substr(name.size() - 4) == ".dat") {
            files.push_back(ent.path());
        }
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::fprintf(stderr,
                     "[!] no blk*.dat files found in %s\n",
                     blocks_dir.c_str());
        return 1;
    }

    std::FILE* out = stdout;
    if (!output_path.empty()) {
        out = std::fopen(output_path.c_str(), "w");
        if (!out) {
            std::fprintf(stderr, "[!] cannot open output %s\n",
                         output_path.c_str());
            return 1;
        }
    }

    if (!quiet) {
        std::fprintf(stderr,
                     "[*] scanning %zu blk*.dat files in %s\n",
                     files.size(), blocks_dir.c_str());
        std::fprintf(stderr,
                     "[*] output: %s\n",
                     output_path.empty() ? "stdout" : output_path.c_str());
    }

    ParseStats stats;
    auto t0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < files.size(); i++) {
        if (max_files && i >= max_files) break;
        process_blk_file(files[i], out, stats);
        if (!quiet) {
            auto now = std::chrono::steady_clock::now();
            double sec = std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - t0).count() / 1000.0;
            std::fprintf(stderr,
                         "[*] %s  blocks=%llu  txs=%llu  outs=%llu  "
                         "h160=%llu  (%.0fs)\n",
                         files[i].filename().string().c_str(),
                         (unsigned long long)stats.blocks,
                         (unsigned long long)stats.txs,
                         (unsigned long long)stats.outputs,
                         (unsigned long long)(stats.p2pkh + stats.p2sh + stats.p2wpkh),
                         sec);
            std::fflush(stderr);
        }
    }

    if (out != stdout) std::fclose(out);

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration_cast<std::chrono::milliseconds>(
                     t1 - t0).count() / 1000.0;

    std::fprintf(stderr, "\n[*] Done in %.1fs\n", sec);
    std::fprintf(stderr,
                 "    blocks scanned : %llu\n"
                 "    transactions   : %llu\n"
                 "    outputs total  : %llu\n"
                 "    P2PKH          : %llu\n"
                 "    P2SH           : %llu\n"
                 "    P2WPKH         : %llu\n"
                 "    skipped (non-h160 scripts: P2WSH/P2TR/P2PK/multisig/...) : %llu\n"
                 "    parse errors   : %llu\n",
                 (unsigned long long)stats.blocks,
                 (unsigned long long)stats.txs,
                 (unsigned long long)stats.outputs,
                 (unsigned long long)stats.p2pkh,
                 (unsigned long long)stats.p2sh,
                 (unsigned long long)stats.p2wpkh,
                 (unsigned long long)stats.skipped,
                 (unsigned long long)stats.parse_err);
    std::fprintf(stderr,
                 "\n[*] Next steps:\n"
                 "    1. Dedupe / sort:    sort -u %s -o seen_sorted.csv\n"
                 "    2. Build bloom:      build_bloom -i seen_sorted.csv -o seen.blf\n"
                 "    3. Get funded UTXOs: bitcoin-cli dumptxoutset funded.dat\n"
                 "                         (then parse with the companion script)\n"
                 "    4. Build funded UVRF: build_bloom -i funded.csv -v funded.uvrf\n"
                 "    5. Scan:             collider_pro --brainwallet \\\n"
                 "                           --bloom seen.blf --verify-set funded.uvrf \\\n"
                 "                           --track-empty-hits --wordlist big.txt\n",
                 output_path.empty() ? "<output>" : output_path.c_str());

    return 0;
}
