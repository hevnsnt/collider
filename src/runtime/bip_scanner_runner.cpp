/**
 * bip_scanner_runner.cpp -- v1.5.x BIP brainwallet scanner runtime.
 * See bip_scanner_runner.hpp for the full design rationale.
 */
#include "runtime/bip_scanner_runner.hpp"

#ifdef COLLIDER_PRO

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include "core/bip32.hpp"
#include "core/bip39.hpp"
#include "runtime/bip_address.hpp"
#include "core/crypto_cpu.hpp"
#include "core/paths.hpp"
#include "core/secure_write.hpp"
#include "core/session_log.hpp"
#include "core/version.hpp"
#include "runtime/bloom_loader.hpp"
#include "runtime/bip_gpu_dispatcher.hpp"  // multi-GPU dispatch
#include "tools/utxo_bloom_builder.hpp"
#include "gpu/v2/brain_wallet_v2.hpp"  // AddressType / addr_bit
#if defined(COLLIDER_USE_CUDA)
#include "gpu/bip39_pbkdf2.cuh"  // batched PBKDF2-HMAC-SHA512
#include <cuda_runtime.h>
#endif
#include "ui/banner.hpp"
#include "ui/box_render.hpp"
#include "ui/tui/tui_app.hpp"
#include "ui/tui/stdio_capture.hpp"
#include "ui/tui/tui_launcher.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <condition_variable>
#include <deque>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace collider::runtime {

// Build the list of directories to walk when looking for project data
// or operator drops. Used by both BIP-39 wordlist resolution and
// candidate-phrases / bloom auto-detection so all three follow the
// same disk-walk policy. Exposed (non-anon) so interactive_ui can
// reuse the same auto-detection helpers and skip prompts when paths
// resolve.
//
// Order matters: closer-to-the-binary locations win first so an
// operator who shipped collider_pro with the data/ subdir always
// hits the bundled wordlist regardless of where they cd'd from. CWD
// is checked last because it's the most volatile (operator may run
// from anywhere).
std::vector<std::filesystem::path> data_search_roots() {
    namespace fs = std::filesystem;
    std::vector<fs::path> roots;
    std::error_code ec;

    // 1. exe-relative paths: data lives either next to the exe OR
    //    one level up (the collider-pro/ layout puts data/ alongside
    //    build_pro/collider_pro.exe so the exe's parent.parent is the
    //    project root).
#ifdef _WIN32
    char exe_buf[4096];
    DWORD len = GetModuleFileNameA(nullptr, exe_buf, sizeof(exe_buf));
    if (len > 0 && len < sizeof(exe_buf)) {
        fs::path exe = exe_buf;
        roots.push_back(exe.parent_path());
        roots.push_back(exe.parent_path().parent_path());
    }
#endif

    // 2. operator-home + ~/.collider drop dirs.
    roots.push_back(collider::paths::collider_home());
    roots.push_back(collider::paths::home());

    // 3. CWD (last).
    auto cwd = fs::current_path(ec);
    if (!ec) roots.push_back(cwd);

    return roots;
}

// BIP-39 English wordlist resolution. Walks data_search_roots()
// against the common subdir + filename conventions ("data/crypto/" or
// "bip/" prefix, "bip39_<lang>.txt" or "<lang>.txt" naming). First
// existing file wins.
std::string resolve_bip39_wordlist() {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto roots = data_search_roots();
    static const char* kSubdirs[]   = {"data/bip39", "data/crypto", "bip",
                                       "bip39", ""};
    static const char* kFilenames[] = {"bip39_english.txt", "english.txt"};
    for (const auto& root : roots) {
        for (const char* sub : kSubdirs) {
            for (const char* fname : kFilenames) {
                fs::path p =
                    sub[0] ? (root / sub / fname) : (root / fname);
                if (fs::exists(p, ec)) return p.string();
            }
        }
    }
    return std::string{};
}

// Bloom-filter auto-detect. Mirrors the brainwallet runner's
// "find funded_addresses.blf in CWD or ~/.collider/" pattern so the
// BIP scanner does not have to ask the operator for a path that the
// operator typed once already for brainwallet.
std::string resolve_bloom_filter() {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto roots = data_search_roots();
    static const char* kNames[] = {
        "funded_addresses.blf",
        "bloom.blf",
    };
    for (const auto& root : roots) {
        for (const char* name : kNames) {
            fs::path p = root / name;
            if (fs::exists(p, ec)) return p.string();
        }
    }
    return std::string{};
}

// Candidate phrases file auto-detect. Looks for an operator-curated
// file at known locations; first hit wins. If nothing is found the
// caller falls back to the interactive prompt.
std::string resolve_candidate_phrases() {
    namespace fs = std::filesystem;
    std::error_code ec;
    auto roots = data_search_roots();
    static const char* kSubdirs[]   = {"wordlists", ".collider/wordlists", ""};
    static const char* kFilenames[] = {
        "bip_phrases.txt",
        "mnemonic_phrases.txt",
        "phrases.txt",
    };
    for (const auto& root : roots) {
        for (const char* sub : kSubdirs) {
            for (const char* fname : kFilenames) {
                fs::path p =
                    sub[0] ? (root / sub / fname) : (root / fname);
                if (fs::exists(p, ec)) return p.string();
            }
        }
    }
    return std::string{};
}

namespace {  // (resolvers above are public; everything below stays
              //  file-local under this anon namespace)

// Single derivation profile: a logical "wallet family" (early, BIP-44,
// 84, etc.) and the path template the scanner walks for each phrase.
// {idx} is substituted with the per-address index (0..count-1). Hardened
// segments use the apostrophe convention.
struct DerivationProfile {
    std::string label;        // operator-facing name shown in hits + status
    std::string path_template; // e.g. "m/44'/0'/0'/0/{idx}"
    int count = 20;            // addresses scanned per phrase
    enum class AddrKind {
        P2PKH,           // hash160(pubkey) directly bloom-probed
        P2SH_P2WPKH,     // hash160(0x00 0x14 || hash160(pubkey))
        P2WPKH,          // hash160(pubkey) (same bytes as P2PKH; here
                         // for label/audit purposes)
    } addr_kind = AddrKind::P2PKH;
};

// Map a profile's AddrKind to the GPU dispatcher's per-item address
// mask. P2WPKH (BIP-84 native segwit) is bloom-probed against the same
// hash160(pubkey) bytes as P2PKH_COMPRESSED -- they have identical
// h160 output -- so both kinds use the P2PKH_COMPRESSED bit. The
// dispatcher's hit-callback disambiguates them via the AddressType
// the kernel returned. Pre-fix, every item had a hardcoded mask of
// (P2PKH_COMPRESSED | P2SH_P2WPKH), which forced the kernel to check
// both kinds for every priv key including BIP-44 profiles that only
// need P2PKH. With per-profile masks the dispatcher can (in the
// future) partition the batch and dispatch homogeneous groups,
// halving GPU work for the P2SH-only kernel pass.
inline int addr_mask_for(DerivationProfile::AddrKind k) {
    using ::collider::gpu::v2::AddressType;
    using ::collider::gpu::v2::addr_bit;
    switch (k) {
        case DerivationProfile::AddrKind::P2PKH:
            return static_cast<int>(addr_bit(AddressType::P2PKH_COMPRESSED));
        case DerivationProfile::AddrKind::P2SH_P2WPKH:
            return static_cast<int>(addr_bit(AddressType::P2SH_P2WPKH));
        case DerivationProfile::AddrKind::P2WPKH:
            return static_cast<int>(addr_bit(AddressType::P2PKH_COMPRESSED));
    }
    return 0;
}

// Operator-facing label for a profile's address kind. Used by the
// CPU-fallback hit-writing paths so their hit records use the same
// 5-field schema as the GPU dispatcher's on_hit callback (which
// gets its label from addr_type_label() in bip_gpu_dispatcher.cpp).
inline const char* addr_kind_label(DerivationProfile::AddrKind k) {
    switch (k) {
        case DerivationProfile::AddrKind::P2PKH:       return "P2PKH-compressed";
        case DerivationProfile::AddrKind::P2SH_P2WPKH: return "P2SH-P2WPKH";
        case DerivationProfile::AddrKind::P2WPKH:      return "P2WPKH-bech32";
    }
    return "?";
}

// The historical-to-modern profile bundle. Order matters for the
// status line: the operator reads "P2PKH legacy" first as it covers
// the largest share of mined-but-lost balances.
std::vector<DerivationProfile> default_profiles() {
    std::vector<DerivationProfile> p;
    // Pre-BIP-44: original Bitcoin Core HD (rare, pre-2013).
    p.push_back({"Early m/0/i (raw HD)",      "m/0/{idx}",         20, DerivationProfile::AddrKind::P2PKH});
    p.push_back({"Early m/0'/0/i",            "m/0'/0/{idx}",      20, DerivationProfile::AddrKind::P2PKH});
    p.push_back({"Early m/0'/1/i",            "m/0'/1/{idx}",      20, DerivationProfile::AddrKind::P2PKH});
    // Electrum 2.x deterministic-seed wallets (pre-segwit).
    p.push_back({"Electrum 2.x m/0/i",        "m/0/{idx}",         20, DerivationProfile::AddrKind::P2PKH});
    // MultiBit HD (m/0'/0/i + m/0'/1/i).
    p.push_back({"MultiBit HD m/0'/0/i",      "m/0'/0/{idx}",      20, DerivationProfile::AddrKind::P2PKH});
    // BIP-44 P2PKH (the bulk of Trezor / Ledger / Mycelium pre-segwit).
    p.push_back({"BIP-44 P2PKH legacy",       "m/44'/0'/0'/0/{idx}",   20, DerivationProfile::AddrKind::P2PKH});
    p.push_back({"BIP-44 P2PKH change",       "m/44'/0'/0'/1/{idx}",   10, DerivationProfile::AddrKind::P2PKH});
    // BIP-49 P2SH-wrapped segwit (Trezor / Ledger 2017+).
    p.push_back({"BIP-49 P2SH-P2WPKH",        "m/49'/0'/0'/0/{idx}",   20, DerivationProfile::AddrKind::P2SH_P2WPKH});
    p.push_back({"BIP-49 P2SH-P2WPKH change", "m/49'/0'/0'/1/{idx}",   10, DerivationProfile::AddrKind::P2SH_P2WPKH});
    // BIP-84 native segwit (modern default).
    p.push_back({"BIP-84 native P2WPKH",      "m/84'/0'/0'/0/{idx}",   20, DerivationProfile::AddrKind::P2WPKH});
    p.push_back({"BIP-84 native P2WPKH chg",  "m/84'/0'/0'/1/{idx}",   10, DerivationProfile::AddrKind::P2WPKH});
    // BIP-86 Taproot (bloom indexed by tweaked x-only; SKIP in first cut
    // because the bloom we ship today doesn't include taproot outputs).
    // Tracked for v1.5.1.
    return p;
}

// CPU-side bloom probe. Same MurmurHash3-128 double-hash scheme the
// brainwallet GPU pipeline uses; mirrors BrainWalletRunner::check_tight_bloom.
bool bloom_probe(const ::collider::utxo::BloomFilterHeader& header,
                 const uint8_t* bloom_bytes,
                 const uint8_t* h160) {
    auto [h1, h2] = ::collider::utxo::murmurhash3_128(h160, 20, header.seed);
    const uint64_t nbits = header.num_bits;
    const uint32_t k = header.num_hashes;
    for (uint32_t i = 0; i < k; ++i) {
        const uint64_t bit_idx = (h1 + static_cast<uint64_t>(i) * h2) % nbits;
        const uint64_t byte_idx = bit_idx >> 3;
        const uint8_t  bit_in   = static_cast<uint8_t>(bit_idx & 7);
        if ((bloom_bytes[byte_idx] & (1u << bit_in)) == 0) {
            return false;
        }
    }
    return true;
}

// Forward to the shared header so the KAT test calls the same code.
using ::collider::bip_address::hash160_pubkey;
using ::collider::bip_address::hash160_p2sh_p2wpkh;

// 32-byte big-endian hex.
std::string hex_lower(const uint8_t* bytes, size_t n) {
    static const char* h = "0123456789abcdef";
    std::string out;
    out.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(h[(bytes[i] >> 4) & 0xF]);
        out.push_back(h[(bytes[i] >> 0) & 0xF]);
    }
    return out;
}

// Write one hit record to the hits file under hits_mu. Format is the
// pipe-delimited "mnemonic | path | priv-hex | profile / addr-type |
// h160-hex" line both run_combinatorial_scan and run_bip_scan_mode
// emit. Pre-refactor this block was duplicated in five places (two
// on_hit lambdas + two legacy CPU fallbacks + the wordlist worker).
void write_hit_record(std::mutex& hits_mu,
                      std::ofstream& hits,
                      const std::string& mnemonic,
                      const std::string& path,
                      const std::string& profile_label,
                      const uint8_t      priv[32],
                      const char*        addr_type_label,
                      const uint8_t      h160[20]) {
    std::lock_guard<std::mutex> hl(hits_mu);
    hits << mnemonic << " | " << path << " | "
         << hex_lower(priv, 32) << " | "
         << profile_label << " / " << addr_type_label << " | "
         << hex_lower(h160, 20) << "\n";
    hits.flush();
}

// Build a BipGpuDispatcher::Config from the standard per-scan
// inputs. Extracted from the two run_*_scan dispatcher-init blocks
// (they constructed the exact same gcfg shape; only the per-mode
// on_hit milestone tag + bloom-hits counter differed). Returns by
// value; on_hit is moved in.
::collider::runtime::BipGpuDispatcher::Config
make_bip_dispatcher_config(
    const Arguments& args,
    const ::collider::runtime::BloomLoadResult& bloom,
    ::collider::runtime::BipGpuHitCallback on_hit) {
    ::collider::runtime::BipGpuDispatcher::Config gcfg;
    gcfg.gpu_ids       = args.gpu_ids;
    gcfg.bloom_data    = bloom.data.data();
    gcfg.bloom_bits    = bloom.header.num_bits;
    gcfg.bloom_hashes  = static_cast<int>(bloom.header.num_hashes);
    gcfg.bloom_seed    = bloom.header.seed;
    gcfg.batch_size    = 4096;
    gcfg.queue_max     = 65536;
    gcfg.on_hit        = std::move(on_hit);
    return gcfg;
}

// Populate the GPU-dispatcher portion of a BipScanInfo from the live
// dispatcher state. Extracted from the two run_*_scan TUI tick
// publish blocks (they had ~30 lines of character-identical
// gpu_init_message / gpu_count_requested / gpu_faulted_devices /
// gpu_count / gpu_shares plumbing each); the per-mode publish blocks
// now call this + populate the mode-specific fields (phrases_read,
// addresses_probed, mode_label, etc.) themselves.
void publish_dispatcher_dashboard(
    ::collider::ui::tui::BipScanInfo& bi,
    const ::collider::runtime::BipGpuDispatcher& dispatcher,
    bool gpu_active,
    bool pbkdf_gpu_active,
    bool gpu_disabled_by_flag) {
    bi.gpu_init_message     = dispatcher.last_error();
    bi.gpu_count_requested  =
        static_cast<int>(dispatcher.requested_device_count());
    bi.gpu_faulted_devices.clear();
    for (const auto& f : dispatcher.faulted_devices()) {
        ::collider::ui::tui::BipScanInfo::FaultedDevice fd;
        fd.device_id = f.device_id;
        fd.error     = f.error;
        bi.gpu_faulted_devices.push_back(std::move(fd));
    }
    bi.pbkdf_gpu_active     = pbkdf_gpu_active;
    bi.gpu_disabled_by_flag = gpu_disabled_by_flag;
    if (gpu_active) {
        const auto& dstats = dispatcher.device_stats();
        bi.gpu_count = static_cast<int>(dstats.size());
        bi.gpu_shares.clear();
        bi.gpu_shares.reserve(dstats.size());
        for (const auto& s : dstats) {
            ::collider::ui::tui::BipScanInfo::GpuShare gs;
            gs.device_id = s->device_id;
            gs.addresses_dispatched =
                s->addresses_dispatched.load(
                    std::memory_order_relaxed);
            gs.addresses_per_sec =
                s->addresses_per_sec.load(
                    std::memory_order_relaxed);
            bi.gpu_shares.push_back(gs);
        }
    } else {
        bi.gpu_count = 0;
    }
}

// Factory for the GPU dispatcher's on_hit callback. Captures the
// hits writer + tight-bloom params + per-scan bloom-hit counter and
// returns a BipGpuHitCallback ready for BipGpuDispatcher::Config.
// Eliminates the character-identical lambda copy-paste between
// run_combinatorial_scan and run_bip_scan_mode (only the milestone
// tag and the per-mode atomic bloom_hits counter differed). The
// callback enforces the tight-bloom gate before counting / writing
// the hit so false positives from the primary bloom never reach
// bip_hits.txt.
::collider::runtime::BipGpuHitCallback make_bip_hit_callback(
    std::mutex&                  hits_mu,
    std::ofstream&               hits,
    std::atomic<uint64_t>&       bloom_hits_counter,
    bool                         tight_bloom_enabled,
    const ::collider::runtime::BloomLoadResult& tight_bloom,
    const char*                  milestone_tag) {
    return [&hits_mu, &hits, &bloom_hits_counter, tight_bloom_enabled,
            &tight_bloom, milestone_tag](
        const std::string& mnemonic,
        const std::string& path,
        const std::string& profile_label,
        const uint8_t      priv[32],
        const char*        addr_type_label,
        const uint8_t      h160[20]) {
        if (tight_bloom_enabled &&
            !bloom_probe(tight_bloom.header,
                         tight_bloom.data.data(), h160)) {
            return;
        }
        bloom_hits_counter.fetch_add(1, std::memory_order_relaxed);
        write_hit_record(hits_mu, hits, mnemonic, path, profile_label,
                         priv, addr_type_label, h160);
        ::collider::log::milestone(
            milestone_tag,
            profile_label + " path=" + path);
    };
}

}  // namespace  (closes the file-local helpers opened at line 154)

// v1.5.0 combinatorial entropy enumeration.
//
// Iterates every BIP-39 entropy value of the chosen byte width (16 /
// 20 / 24 / 28 / 32 bytes -> 12 / 15 / 18 / 21 / 24 word phrases) as a
// 256-bit counter, derives each candidate against every default
// derivation profile, and probes each address against the loaded
// bloom filter. The search space is 2^(8*ent_bytes), unreachable in
// finite time; the scanner exists to provide a checkpoint-resumable
// long-running probe per the operator's request.
//
// Resume state lives at ~/.collider/bip_combinatorial.json with a
// single field `next_index_hex` (big-endian 32-byte counter). The
// counter is checkpointed every kCheckpointEvery probes; on Ctrl+C
// the operator loses at most that many probes of progress (negligible
// against the search space).
//
// Throughput: pure-CPU baseline is ~500 phrases/sec single-threaded.
// GPU path (PBKDF2 + EC + hash160 + bloom probe on device) lifts that
// to several thousand per second per GPU; the combinatorial path adds
// a 4-thread chain walker between the GPU PBKDF2 dispatch and the
// GPU multi-address dispatch so the inter-stage CPU work doesn't
// bottleneck the pipeline.

namespace {

constexpr uint64_t kCheckpointEvery = 4096;

void inc_be_counter(std::vector<uint8_t>& c) {
    for (size_t i = c.size(); i-- > 0;) {
        if (++c[i] != 0) return;
    }
    // overflow: counter wrapped (would require 2^(8*N) probes).
}

std::filesystem::path combinatorial_resume_path() {
    return ::collider::paths::collider_home() / "bip_combinatorial.json";
}

std::vector<uint8_t> load_resume_counter(size_t ent_bytes) {
    std::vector<uint8_t> c(ent_bytes, 0);
    std::ifstream f(combinatorial_resume_path());
    if (!f) return c;
    std::string body((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    auto pos = body.find("\"next_index_hex\"");
    if (pos == std::string::npos) return c;
    pos = body.find('"', pos + 16);
    if (pos == std::string::npos) return c;
    auto end = body.find('"', pos + 1);
    if (end == std::string::npos) return c;
    std::string hex = body.substr(pos + 1, end - pos - 1);
    // Only accept hex of exactly ent_bytes*2 (matches the saved width);
    // otherwise reset to zero so a width change starts clean.
    if (hex.size() != ent_bytes * 2) return c;
    for (size_t i = 0; i < ent_bytes; ++i) {
        unsigned v = 0;
        if (std::sscanf(hex.data() + i * 2, "%2x", &v) != 1) {
            return std::vector<uint8_t>(ent_bytes, 0);
        }
        c[i] = static_cast<uint8_t>(v);
    }
    return c;
}

void save_resume_counter(const std::vector<uint8_t>& c) {
    auto path = combinatorial_resume_path();
    auto tmp  = path;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return;
        f << "{\n  \"next_index_hex\": \"";
        for (uint8_t b : c) {
            char buf[3];
            std::snprintf(buf, sizeof(buf), "%02x", b);
            f << buf;
        }
        f << "\"\n}\n";
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
}

}  // namespace

int run_combinatorial_scan(const Arguments& args,
                           const std::function<void()>& release_capture) {
    namespace boxui = ::collider::ui::box;

    const int word_count = args.bip_combinatorial_word_count;
    size_t ent_bytes = 0;
    switch (word_count) {
        case 12: ent_bytes = 16; break;
        case 15: ent_bytes = 20; break;
        case 18: ent_bytes = 24; break;
        case 21: ent_bytes = 28; break;
        case 24: ent_bytes = 32; break;
        default:
            release_capture();
            std::cerr << "[!] --bip-combinatorial-words must be one of "
                         "12/15/18/21/24 (got " << word_count << ")\n";
            return 1;
    }

    // Auto-detect bloom + dictionary (same as the wordlist path).
    std::string bloom_path = args.bloom_file;
    if (bloom_path.empty()) bloom_path = resolve_bloom_filter();
    if (bloom_path.empty()) {
        release_capture();
        std::cerr << "[!] --bip-combinatorial needs a UTXO bloom "
                     "filter; drop one at ~/.collider/funded_addresses.blf "
                     "or pass --bloom <path>.\n";
        return 1;
    }
    std::string bip39_path = resolve_bip39_wordlist();
    if (bip39_path.empty()) {
        release_capture();
        std::cerr << "[!] BIP-39 English wordlist not found. Drop "
                     "english.txt at data/bip39/ next to the exe.\n";
        return 1;
    }

    bip39::WordlistEnglish wordlist;
    try {
        wordlist.load(bip39_path);
    } catch (const std::exception& e) {
        release_capture();
        std::cerr << "[!] BIP-39 wordlist load failed: " << e.what() << "\n";
        return 1;
    }

    BloomLoadResult bloom = load_bloom_file_into_memory(bloom_path);
    if (!bloom.ok) {
        release_capture();
        std::cerr << "[!] Bloom load failed: " << bloom.err_message << "\n";
        return 1;
    }
    BloomLoadResult tight_bloom;
    bool tight_bloom_enabled = false;
    if (!args.bloom_tight_file.empty()) {
        tight_bloom = load_bloom_file_into_memory(args.bloom_tight_file);
        tight_bloom_enabled = tight_bloom.ok;
    }

    // User feedback 2026-05-24: do not drop to host-terminal status
    // text between the interactive flow's confirm modal and the TUI
    // alt-screen taking over. The dictionary + bloom + derivation
    // profile counts are already shown in the COMBINATORIAL REALITY
    // CHECK confirm modal below, and the in-session TUI dashboard
    // surfaces them too. Removing the cout banners keeps the entire
    // experience inside FTXUI.
    auto profiles = default_profiles();

    // Hits sink: shared with the wordlist path so operator scripts that
    // tail bip_hits.txt work for both modes.
    auto exe_dir = std::filesystem::current_path();
    std::string hits_path = (exe_dir / "bip_hits.txt").string();
    std::ofstream hits = collider::secure_open_ofstream(
        hits_path,
        std::ios::out | std::ios::app,
        collider::SecureWriteOnFailure::FallbackLoud);
    if (!hits.is_open()) {
        release_capture();
        std::cerr << "[!] Could not open " << hits_path << " for hits.\n";
        return 1;
    }
    hits << "# bip_hits.txt - " << ::collider::kVersion
         << " combinatorial scan; words=" << word_count << "\n";
    hits.flush();

    std::vector<uint8_t> counter = load_resume_counter(ent_bytes);
    // flushed_counter mirrors counter EXCEPT during GPU PBKDF2
    // batching: counter advances on every push into the batch buffer,
    // but the actual chain-walk + bloom-probe doesn't run until the
    // batch flushes (or shutdown). flushed_counter only advances
    // AFTER work-done, so a checkpoint of flushed_counter resumes
    // from a known-probed offset (a counter-only checkpoint would
    // resume PAST up to 255 unprobed mnemonics still in the in-flight
    // batch on Ctrl+C).
    std::vector<uint8_t> flushed_counter = counter;

    // GPU dispatcher declared here; init deferred until after
    // bloom_hits is in scope (the on_hit callback captures it).
    BipGpuDispatcher gpu_dispatcher;
    bool gpu_active = false;
    std::mutex hits_mu;  // shared by the GPU hit callback + the
                        // legacy CPU path (when --no-bip-gpu is set).

    // Restore real stdout BEFORE launch_session. FTXUI's draw uses
    // std::cout (see ftxui/src/component/screen_interactive.cpp); if
    // tui_stdio_capture is still active the draw stream goes to a
    // ring buffer that flushes to a log file, leaving the alt-screen
    // visible but EMPTY. SILENT variant: captured boot text persists
    // to log file instead of scrolling on terminal right before the
    // alt-screen takes over.
    ::collider::ui::tui::StdioCapture::release_active_capture_silent();

    // TUI launch.
    ::collider::ui::tui::LaunchConfig launch_cfg;
    launch_cfg.mode_label              = "BIP-Scan";
    launch_cfg.version                 = std::string(::collider::kVersion);
    launch_cfg.tui_mode                = ::collider::ui::tui::TuiMode::BipScan;
    launch_cfg.gpu_ids                 = args.gpu_ids;
    launch_cfg.session_start           = std::chrono::steady_clock::now();
    launch_cfg.initial_phase_name      = "combinatorial scan";
    launch_cfg.initial_current_chunk   = 0;
    launch_cfg.initial_total_chunks    = 1;
    launch_cfg.render_cfg.refresh_hz   = 10;
    launch_cfg.render_cfg.alt_screen   = true;
    launch_cfg.guard_opts.alt_screen              = true;
    launch_cfg.guard_opts.hide_cursor             = true;
    launch_cfg.guard_opts.install_signal_handlers = true;
    auto session  = ::collider::ui::tui::launch_session(launch_cfg);
    auto* tui_app = session.app.get();

    // probed_this_run / addrs_probed / bloom_hits are bumped by
    // the chain-walker thread pool (4 walkers on the combinatorial
    // PBKDF2-GPU path), so they must be atomic. The TUI tick reads
    // via .load() (acquire / relaxed are both fine; only the
    // monotonic property matters for the dashboard).
    std::atomic<uint64_t> probed_this_run{0};
    std::atomic<uint64_t> addrs_probed{0};
    std::atomic<uint64_t> bloom_hits{0};
    auto start = std::chrono::steady_clock::now();
    auto last_tui_post = start;
    auto last_ckpt = start;

    // GPU dispatcher init. Captures bloom_hits + hits + hits_mu +
    // tight bloom by reference so the per-GPU worker threads route
    // matches through the same hits file the CPU path used. Failure
    // to init (no CUDA, etc.) leaves gpu_active=false and the per-
    // pubkey loop falls back to the CPU code path.
    if (!args.bip_no_gpu) {
        auto gcfg = make_bip_dispatcher_config(
            args, bloom,
            make_bip_hit_callback(
                hits_mu, hits, bloom_hits,
                tight_bloom_enabled, tight_bloom,
                "bip_combinatorial_hit"));
        if (gpu_dispatcher.init(gcfg) == 0) {
            gpu_active = true;
        }
    }
    // Dispatcher diagnostics are read live from gpu_dispatcher per
    // tick (last_error() + faulted_devices()) so runtime CUDA errors
    // surface to the dashboard, not just init-time ones.

    // ----------------------------------------------------------------
    // GPU PBKDF2 batching. Accumulate mnemonics into a fixed-size
    // batch; when full (or at shutdown) dispatch one GPU PBKDF2
    // kernel and walk the resulting seeds through the chain +
    // multi-address dispatcher. Default batch 256 mnemonics
    // (~0.3 sec of work on consumer Ampere PBKDF2 throughput) so
    // the TUI ticks every batch boundary stay responsive.
    // ----------------------------------------------------------------
#if defined(COLLIDER_USE_CUDA)
    constexpr size_t kPbkdfBatch = 256;
    constexpr size_t kMnemonicSlot = ::collider::gpu::bip39::kMaxMnemonicBytes;
    std::vector<uint8_t>  pbkdf_packed(kPbkdfBatch * kMnemonicSlot, 0);
    std::vector<uint32_t> pbkdf_lens(kPbkdfBatch, 0);
    std::vector<std::string> pbkdf_mnemonics;
    pbkdf_mnemonics.reserve(kPbkdfBatch);
    std::vector<uint8_t>  pbkdf_seeds(kPbkdfBatch * 64);
    // Multi-GPU PBKDF2: one stream per device in args.gpu_ids.
    // Round-robin batch dispatch across devices so a 2-GPU system
    // gets 2x PBKDF2 throughput. No silent {0} default -- if
    // detect_gpus() left args.gpu_ids empty (no CUDA), we honestly
    // skip GPU PBKDF2 and let the CPU fallback handle the work.
    const std::vector<int>&   pbkdf_devices = args.gpu_ids;
    std::vector<cudaStream_t> pbkdf_streams(pbkdf_devices.size(), nullptr);
    size_t pbkdf_next_device = 0;
    bool   pbkdf_gpu = false;
    if (!args.bip_no_gpu && !pbkdf_devices.empty()) {
        bool any_ok = false;
        for (size_t i = 0; i < pbkdf_devices.size(); ++i) {
            if (cudaSetDevice(pbkdf_devices[i]) != cudaSuccess) continue;
            if (cudaStreamCreate(&pbkdf_streams[i]) == cudaSuccess) {
                any_ok = true;
            }
        }
        pbkdf_gpu = any_ok;
    }
    // last_profile_label is read by the TUI thread and written by all
    // 4 walker threads in the chain-walker pool. Wordlist path uses
    // the same pattern (label_mu); combinatorial path was missing the
    // mutex (data race on std::string SSO/heap transitions, would
    // eventually corrupt the dashboard or crash).
    std::string last_profile_label;
    std::mutex  label_mu;
    auto publish_label = [&](const std::string& label) {
        std::lock_guard<std::mutex> lk(label_mu);
        last_profile_label = label;
    };

    auto walk_seed_and_dispatch = [&](const std::string& mnemonic,
                                      const uint8_t* seed64) {
        bip32::ExtKey master;
        try {
            master = bip32::master_from_seed(seed64, 64);
        } catch (const std::exception&) {
            return;
        }
        for (const auto& prof : profiles) {
            publish_label(prof.label);
            for (int i = 0; i < prof.count; ++i) {
                std::string path = prof.path_template;
                size_t pos = path.find("{idx}");
                if (pos != std::string::npos) {
                    path.replace(pos, 5, std::to_string(i));
                }
                bip32::ExtKey child;
                try {
                    auto parsed = bip32::parse_path(path);
                    child = bip32::derive_path(master, parsed);
                } catch (const std::exception&) {
                    continue;
                }
                if (gpu_active) {
                    BipGpuWorkItem item;
                    std::memcpy(item.priv.data(), child.key.data(), 32);
                    item.mnemonic        = mnemonic;
                    item.derivation_path = path;
                    item.profile_label   = prof.label;
                    item.addr_mask       = addr_mask_for(prof.addr_kind);
                    // enqueue() returns false on dispatcher shutdown.
                    // When that happens we bail out of THIS seed walk
                    // (no work was dispatched for this profile, so the
                    // addrs_probed counter is not bumped). The outer
                    // producer loop notices g_shutdown next iteration
                    // and exits cleanly.
                    if (!gpu_dispatcher.enqueue(std::move(item))) {
                        return;
                    }
                    addrs_probed.fetch_add(1, std::memory_order_relaxed);
                } else {
                    auto pub = bip32::detail::priv_to_pub(child.key.data());
                    std::array<uint8_t, 20> h160{};
                    switch (prof.addr_kind) {
                        case DerivationProfile::AddrKind::P2SH_P2WPKH:
                            h160 = hash160_p2sh_p2wpkh(pub.data());
                            break;
                        default:
                            h160 = hash160_pubkey(pub.data());
                            break;
                    }
                    const bool primary_hit =
                        bloom_probe(bloom.header, bloom.data.data(),
                                    h160.data());
                    // Count the address as probed AFTER the bloom check
                    // completes (success or no-hit). The counter is "work
                    // done", not "work enqueued".
                    addrs_probed.fetch_add(1, std::memory_order_relaxed);
                    if (primary_hit) {
                        if (tight_bloom_enabled &&
                            !bloom_probe(tight_bloom.header,
                                         tight_bloom.data.data(),
                                         h160.data())) {
                            continue;
                        }
                        bloom_hits.fetch_add(1, std::memory_order_relaxed);
                        write_hit_record(hits_mu, hits, mnemonic, path,
                                         prof.label, child.key.data(),
                                         addr_kind_label(prof.addr_kind),
                                         h160.data());
                        ::collider::log::milestone(
                            "bip_combinatorial_hit",
                            prof.label + " path=" + path);
                    }
                }
            }
        }
        probed_this_run.fetch_add(1, std::memory_order_relaxed);
    };

    // Persistent walker thread pool. Replaces the per-batch
    // std::thread spawn that was the previous structurally-wrong-but-
    // working pattern (4 fresh threads created + joined per 256-mnemonic
    // batch = ~16 thread creates/sec at steady state = ~1ms/sec pure
    // syscall overhead on Windows CreateThread). The pool spawns 4
    // workers ONCE at function entry; each waits on work_cv, drains
    // the shared next_idx counter, and signals done_cv when remaining
    // hits zero. flush_pbkdf_batch posts a new batch into the pool
    // and waits on done_cv. Shutdown is signaled at function exit.
    constexpr unsigned kWalkers = 4;
    struct WalkerPool {
        std::mutex mu;
        std::condition_variable work_cv;
        std::condition_variable done_cv;

        // Current-batch handoff (under mu).
        const std::vector<std::string>* mnemonics_ptr = nullptr;
        const uint8_t* seeds_base = nullptr;
        size_t batch_n = 0;
        unsigned generation = 0;
        bool shutdown = false;

        // Self-balancing index claim (lock-free).
        std::atomic<size_t> next_idx{0};
        std::atomic<unsigned> remaining{0};
    } walker_pool;

    auto walker_worker = [&]() {
        unsigned my_gen = 0;
        while (true) {
            // Snapshot the new-batch state under the pool mutex.
            const std::vector<std::string>* mnems = nullptr;
            const uint8_t* seeds = nullptr;
            {
                std::unique_lock<std::mutex> lk(walker_pool.mu);
                walker_pool.work_cv.wait(lk, [&] {
                    return walker_pool.shutdown ||
                           walker_pool.generation != my_gen;
                });
                if (walker_pool.shutdown) return;
                my_gen = walker_pool.generation;
                mnems  = walker_pool.mnemonics_ptr;
                seeds  = walker_pool.seeds_base;
            }
            // Drain the batch by claiming indices off the shared atomic.
            while (true) {
                const size_t i = walker_pool.next_idx.fetch_add(
                    1, std::memory_order_relaxed);
                if (i >= walker_pool.batch_n) break;
                walk_seed_and_dispatch((*mnems)[i], seeds + i * 64);
            }
            // Last worker out notifies the producer.
            if (walker_pool.remaining.fetch_sub(
                    1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> dlk(walker_pool.mu);
                walker_pool.done_cv.notify_one();
            }
        }
    };
    std::vector<std::thread> walker_threads;
    walker_threads.reserve(kWalkers);
    for (unsigned w = 0; w < kWalkers; ++w) {
        walker_threads.emplace_back(walker_worker);
    }

    auto flush_pbkdf_batch = [&]() {
        if (pbkdf_mnemonics.empty()) return;
        const size_t n = pbkdf_mnemonics.size();
        if (pbkdf_gpu) {
            // Round-robin device + stream. Sets the CUDA device for
            // this thread so the kernel + alloc lands on the right
            // GPU; subsequent set_bip_scan_info / dispatcher work
            // is unaffected because the multi-address dispatcher
            // workers each set their own device on their own thread.
            // Skip devices whose stream init failed (nullptr) so a
            // partial multi-GPU init doesn't NPE the kernel launch.
            size_t didx = pbkdf_next_device;
            size_t tries = 0;
            while (tries < pbkdf_devices.size() &&
                   pbkdf_streams[didx] == nullptr) {
                didx = (didx + 1) % pbkdf_devices.size();
                ++tries;
            }
            if (pbkdf_streams[didx] == nullptr) {
                // No working stream -- fall through to CPU path.
                goto pbkdf_cpu_fallback;
            }
            pbkdf_next_device = (didx + 1) % pbkdf_devices.size();
            cudaSetDevice(pbkdf_devices[didx]);
            ::collider::gpu::bip39::Pbkdf2Batch req{};
            req.mnemonic_bytes = pbkdf_packed.data();
            req.mnemonic_lens  = pbkdf_lens.data();
            req.count          = n;
            static const std::string salt = "mnemonic";  // empty passphrase
            req.salt_bytes = reinterpret_cast<const uint8_t*>(salt.data());
            req.salt_len   = static_cast<uint32_t>(salt.size());
            req.out_seeds  = pbkdf_seeds.data();
            cudaError_t rc = ::collider::gpu::bip39::run_pbkdf2_batch(
                req, pbkdf_streams[didx]);
            if (rc == cudaSuccess) {
                // Parallel chain walk via the PERSISTENT walker pool
                // declared at function entry. Pre-fix this code path
                // spawned 4 fresh std::thread per batch (~16 thread
                // creates/sec at steady state, ~1ms/sec syscall
                // overhead on Windows). Now: post the batch into
                // walker_pool + wake the workers + wait for them
                // to signal done. The 4 workers self-balance via the
                // shared next_idx atomic; no fixed per-thread slicing.
                {
                    std::lock_guard<std::mutex> lk(walker_pool.mu);
                    walker_pool.mnemonics_ptr = &pbkdf_mnemonics;
                    walker_pool.seeds_base    = pbkdf_seeds.data();
                    walker_pool.batch_n       = n;
                    walker_pool.next_idx.store(0,
                        std::memory_order_relaxed);
                    walker_pool.remaining.store(kWalkers,
                        std::memory_order_relaxed);
                    walker_pool.generation++;
                }
                walker_pool.work_cv.notify_all();
                {
                    std::unique_lock<std::mutex> lk(walker_pool.mu);
                    walker_pool.done_cv.wait(lk, [&] {
                        return walker_pool.remaining.load(
                            std::memory_order_acquire) == 0;
                    });
                }
                pbkdf_mnemonics.clear();
                std::fill(pbkdf_packed.begin(), pbkdf_packed.end(), 0);
                // Batch fully walked + dispatched. Snapshot the counter
                // so checkpoint saves the last-known-probed offset, not
                // the bumped-but-unprobed counter sitting at the next
                // batch boundary.
                flushed_counter = counter;
                return;
            }
            // Fall through to CPU path on GPU failure.
        }
pbkdf_cpu_fallback:
        // CPU fallback path: one seed at a time.
        for (size_t i = 0; i < n; ++i) {
            try {
                auto seed = bip32::mnemonic_to_seed(
                    pbkdf_mnemonics[i], std::string{});
                walk_seed_and_dispatch(pbkdf_mnemonics[i], seed.data());
            } catch (const std::exception&) {
                continue;
            }
        }
        pbkdf_mnemonics.clear();
        std::fill(pbkdf_packed.begin(), pbkdf_packed.end(), 0);
        // CPU fallback walks synchronously too -- snapshot for the
        // same reason as the GPU success path above.
        flushed_counter = counter;
    };
#endif

    while (!g_shutdown.load()) {
        // Each iteration: build mnemonic from counter, derive every
        // profile, probe addresses. Then increment counter.
        std::vector<std::string> words;
        try {
            words = bip39::entropy_to_mnemonic(counter, wordlist);
        } catch (const std::exception&) {
            inc_be_counter(counter);
            continue;
        }
        std::string mnemonic;
        for (size_t i = 0; i < words.size(); ++i) {
            if (i) mnemonic.push_back(' ');
            mnemonic += words[i];
        }

#if defined(COLLIDER_USE_CUDA)
        // Batched GPU PBKDF2 path: append to the pending batch +
        // flush when full. The flush helper walks the chain +
        // dispatches addresses per resolved seed.
        if (mnemonic.size() < kMnemonicSlot) {
            const size_t slot = pbkdf_mnemonics.size();
            std::memcpy(pbkdf_packed.data() + slot * kMnemonicSlot,
                        mnemonic.data(), mnemonic.size());
            pbkdf_lens[slot] = static_cast<uint32_t>(mnemonic.size());
            pbkdf_mnemonics.push_back(mnemonic);
            inc_be_counter(counter);
            if (pbkdf_mnemonics.size() >= kPbkdfBatch) {
                flush_pbkdf_batch();
            }
            // Skip the legacy per-iteration code below.
            goto bip_iter_tail;
        }
#endif
        {
        std::array<uint8_t, 64> seed;
        try {
            seed = bip32::mnemonic_to_seed(mnemonic, std::string{});
        } catch (const std::exception&) {
            inc_be_counter(counter);
            flushed_counter = counter;
            continue;
        }
        bip32::ExtKey master;
        try {
            master = bip32::master_from_seed(seed.data(), seed.size());
        } catch (const std::exception&) {
            inc_be_counter(counter);
            flushed_counter = counter;
            continue;
        }

        for (const auto& prof : profiles) {
            // Same publisher as the GPU-batched walker -- writes go
            // through label_mu so the TUI thread reads a consistent
            // value. Pre-fix this path had a SHADOW local of the same
            // name that was never seen by the dashboard.
            publish_label(prof.label);
            for (int i = 0; i < prof.count; ++i) {
                std::string path = prof.path_template;
                size_t pos = path.find("{idx}");
                if (pos != std::string::npos) {
                    path.replace(pos, 5, std::to_string(i));
                }
                bip32::ExtKey child;
                try {
                    auto parsed = bip32::parse_path(path);
                    child = bip32::derive_path(master, parsed);
                } catch (const std::exception&) {
                    continue;
                }
                // GPU dispatch: ship the priv key to the dispatcher.
                // The dispatcher's worker computes pub + hash160 +
                // bloom probe on GPU and calls our hit callback on
                // any match. Per-profile addr_mask (no longer the
                // hardcoded OR of both kinds; see addr_mask_for() at
                // the top of this file).
                if (gpu_active) {
                    BipGpuWorkItem item;
                    std::memcpy(item.priv.data(), child.key.data(), 32);
                    item.mnemonic        = mnemonic;
                    item.derivation_path = path;
                    item.profile_label   = prof.label;
                    item.addr_mask       = addr_mask_for(prof.addr_kind);
                    if (!gpu_dispatcher.enqueue(std::move(item))) {
                        // Dispatcher shutting down. Bail this seed walk;
                        // outer loop's g_shutdown check exits cleanly.
                        break;
                    }
                    addrs_probed.fetch_add(1, std::memory_order_relaxed);
                } else {
                    // Pure-CPU fallback (--no-bip-gpu).
                    auto pub = bip32::detail::priv_to_pub(child.key.data());
                    std::array<uint8_t, 20> h160{};
                    switch (prof.addr_kind) {
                        case DerivationProfile::AddrKind::P2SH_P2WPKH:
                            h160 = hash160_p2sh_p2wpkh(pub.data());
                            break;
                        default:
                            h160 = hash160_pubkey(pub.data());
                            break;
                    }
                    const bool primary_hit =
                        bloom_probe(bloom.header, bloom.data.data(),
                                    h160.data());
                    addrs_probed.fetch_add(1, std::memory_order_relaxed);
                    if (primary_hit) {
                        if (tight_bloom_enabled &&
                            !bloom_probe(tight_bloom.header,
                                         tight_bloom.data.data(),
                                         h160.data())) {
                            continue;
                        }
                        bloom_hits.fetch_add(1, std::memory_order_relaxed);
                        write_hit_record(hits_mu, hits, mnemonic, path,
                                         prof.label, child.key.data(),
                                         addr_kind_label(prof.addr_kind),
                                         h160.data());
                        ::collider::log::milestone(
                            "bip_combinatorial_hit",
                            prof.label + " path=" + path);
                    }
                }
            }
        }
        probed_this_run.fetch_add(1, std::memory_order_relaxed);
        inc_be_counter(counter);
        // Legacy CPU path: counter advance = mnemonic fully probed,
        // so flushed_counter tracks exactly.
        flushed_counter = counter;
        }  // end CPU fallback / legacy block

bip_iter_tail:
        auto now = std::chrono::steady_clock::now();
        if (now - last_tui_post > std::chrono::milliseconds(100)) {
            const double secs =
                std::chrono::duration<double>(now - start).count();
            if (tui_app) {
                tui_app->set_keys_per_sec_current(
                    secs > 0 ? addrs_probed.load(std::memory_order_relaxed) / secs : 0.0);
                ::collider::ui::tui::BipScanInfo bi;
                const uint64_t pr_snap   = probed_this_run.load(std::memory_order_relaxed);
                const uint64_t ap_snap   = addrs_probed.load(std::memory_order_relaxed);
                const uint64_t hits_snap = bloom_hits.load(std::memory_order_relaxed);
                bi.phrases_read     = pr_snap;
                bi.phrases_valid    = pr_snap;
                bi.addresses_probed = ap_snap;
                bi.bloom_hits       = hits_snap;
                {
                    std::lock_guard<std::mutex> ll(label_mu);
                    bi.current_profile = last_profile_label;
                }
                // Dashboard refactor 2026-05-24: surface mode + bloom
                // + worker counts + real throughput so the BIP scan
                // dashboard stops looking like a starved brainwallet.
                bi.mode_label       = "combinatorial";
                bi.word_count       = word_count;
                bi.bloom_elements   = bloom.header.num_elements;
                bi.derivation_profiles = static_cast<int>(profiles.size());
                {
                    int total_per_phrase = 0;
                    for (const auto& p : profiles) total_per_phrase += p.count;
                    bi.addresses_per_phrase = total_per_phrase;
                }
                // T1-C threading: 1 producer + N workers.
                {
                    const unsigned hw = std::thread::hardware_concurrency();
                    bi.worker_threads = hw > 1 ? hw - 1 : 1;
                }
                // GPU diagnostics: dispatcher init + PBKDF2 stream state.
                // gpu_init_message is the SUMMARY ("1 of 2 GPU online",
                // "all devices failed", etc.). Per-device fault detail
                // is in gpu_faulted_devices, rendered as warn-colored
                // rows next to the per-device dispatch breakdown.
                publish_dispatcher_dashboard(
                    bi, gpu_dispatcher, gpu_active,
                    /*pbkdf_gpu_active=*/pbkdf_gpu,
                    /*gpu_disabled_by_flag=*/args.bip_no_gpu);
                if (secs > 0) {
                    bi.phrases_per_sec   = pr_snap / secs;
                    bi.addresses_per_sec = ap_snap / secs;
                }
                tui_app->set_bip_scan_info(bi);
                tui_app->set_current_phase_name(
                    std::string("combinatorial " ) +
                    std::to_string(word_count) + "w");
                if (tui_app->requested_quit() && !g_shutdown.load()) {
                    g_shutdown.store(true);
                    break;
                }
            }
            last_tui_post = now;
        }
        if (probed_this_run.load(std::memory_order_relaxed) % kCheckpointEvery == 0 ||
            now - last_ckpt > std::chrono::seconds(5)) {
            // flushed_counter (NOT counter) -- counter may be ahead
            // of the in-flight PBKDF2 batch; flushed_counter is the
            // last KNOWN-PROBED offset so resume restarts from a
            // safe point and never silently skips mnemonics.
            save_resume_counter(flushed_counter);
            last_ckpt = now;
        }
    }

    // Flush any pending PBKDF2 batch BEFORE shutting down the
    // multi-address dispatcher so the addresses derived from the
    // last partial batch reach the GPU. flush_pbkdf_batch updates
    // flushed_counter on success so the final checkpoint below
    // reflects ALL work that completed.
#if defined(COLLIDER_USE_CUDA)
    flush_pbkdf_batch();
    for (size_t i = 0; i < pbkdf_streams.size(); ++i) {
        if (pbkdf_streams[i]) {
            cudaSetDevice(pbkdf_devices[i]);
            cudaStreamDestroy(pbkdf_streams[i]);
        }
    }
#endif

    // Always checkpoint on exit so a Ctrl+C does not lose progress.
    // Saved AFTER the final flush above so flushed_counter reflects
    // every probed mnemonic, not the post-push pre-flush state.
    save_resume_counter(flushed_counter);

    // Drain + shutdown the GPU dispatcher BEFORE we tear down the
    // tui_app / capture so any final in-flight matches land in
    // bip_hits.txt. addrs_probed now only counts items that
    // successfully entered the dispatcher's pipeline (we increment
    // AFTER enqueue() returned true, not before), so no max() vs
    // gpu_dispatcher reconciliation is needed -- both counters agree
    // on the final value after drain.
    if (gpu_active) {
        gpu_dispatcher.shutdown();
    }

    // Shut down the persistent walker pool. Signal shutdown under
    // the mutex, broadcast on work_cv so every worker wakes from
    // its wait and observes the flag, then join all threads. The
    // pool was idle by this point (last flush ran above + signaled
    // done_cv, then the producer loop exited), so the workers were
    // already parked on work_cv.
    {
        std::lock_guard<std::mutex> lk(walker_pool.mu);
        walker_pool.shutdown = true;
    }
    walker_pool.work_cv.notify_all();
    for (auto& t : walker_threads) {
        if (t.joinable()) t.join();
    }

    // Same UX rationale as the wordlist-driven path: tear the TUI down
    // immediately and emit a prominent cout banner so PowerShell shows
    // the summary in scrollback. Pause for Enter so the operator gets
    // to read it before the process exits.
    auto end = std::chrono::steady_clock::now();
    const double elapsed_s =
        std::chrono::duration<double>(end - start).count();
    release_capture();

    const uint64_t final_phrases = probed_this_run.load(std::memory_order_relaxed);
    const uint64_t final_addrs   = addrs_probed.load(std::memory_order_relaxed);
    const uint64_t final_hits    = bloom_hits.load(std::memory_order_relaxed);
    std::cout << "\n";
    std::cout << "================================================================\n";
    if (final_hits > 0) {
        std::cout << "  BIP COMBINATORIAL SCAN STOPPED -- " << final_hits
                  << " HIT(S) FOUND\n";
    } else {
        std::cout << "  BIP COMBINATORIAL SCAN STOPPED -- 0 hits\n";
    }
    std::cout << "================================================================\n";
    std::cout << "    Phrases this run:    " << final_phrases << "\n";
    std::cout << "    Addresses probed:    " << final_addrs << "\n";
    std::cout << "    Bloom hits:          " << final_hits << "\n";
    std::cout << "    Elapsed:             " << std::fixed
              << std::setprecision(2) << elapsed_s << " s\n";
    std::cout << "    Resume counter:      ";
    for (uint8_t b : counter) {
        char buf[3];
        std::snprintf(buf, sizeof(buf), "%02x", b);
        std::cout << buf;
    }
    std::cout << "\n";
    std::cout << "    State file:          "
              << combinatorial_resume_path().string() << "\n";
    if (final_hits > 0) {
        std::cout << "    Hits written:        " << hits_path << "\n";
    }
    std::cout << "================================================================\n";
    // The "Press Enter to exit" std::getline prompt previously here
    // hung the process forever on piped stdin (Task Scheduler, CI,
    // SSH-no-tty). The operator who pressed 'q' to exit the TUI
    // already signaled "I'm done"; making them press Enter twice was
    // just bad UX. Removed -- the cout summary above stays in the
    // terminal scrollback for review.
    return 0;
}

int run_bip_scan_mode(const Arguments& args) {
    namespace boxui = ::collider::ui::box;
    // main_impl installs a StdioCapture before dispatching us; it pipes
    // cout/cerr into a ring buffer so the TUI alt-screen stays clean.
    // We have to release it before printing any fatal "[!] ..." line so
    // the operator actually sees the error on the real terminal instead
    // of having it disappear into ~/.collider/logs/tui-boot-*.log.
    auto release_capture = [] {
        if (auto* cap = ::collider::ui::tui::StdioCapture::current()) {
            cap->release_to_stderr();
        }
    };
    // User feedback 2026-05-24: "make sure this stays in the TUI and
    // never drops out to text like this." The pre-fix banner + intro
    // cout printed to the host terminal after the interactive flow's
    // confirm modal closed, breaking the all-TUI experience. Banner +
    // intro removed; the TUI scan view itself surfaces the mode label
    // and the interactive flow's confirm modal already showed the
    // operator the full configuration before they hit Start.

    // v1.5.0 combinatorial mode: when --bip-combinatorial is passed,
    // skip the operator-supplied candidates file entirely and iterate
    // every BIP-39 entropy value of the chosen width. Each entropy
    // value maps to exactly one valid mnemonic by construction (no
    // checksum rejection), so this is the maximum-throughput scan.
    // The search space is 2^128 .. 2^256 depending on word count.
    // Exhausting it is not physically possible; the scanner makes
    // checkpoint-resumable progress and the operator stops with q.
    if (args.bip_combinatorial) {
        return run_combinatorial_scan(args, release_capture);
    }

    // Auto-detect candidate phrases file when the operator did not pass
    // --bip-scan-wordlist. data_search_roots() covers exe-relative,
    // ~/.collider, ~/, CWD with multiple subdir conventions so a typical
    // install (D:/collider/wordlists/ or ~/.collider/wordlists/) is picked
    // up without manual flags.
    std::string candidates_path = args.bip_scan_wordlist;
    if (candidates_path.empty()) {
        candidates_path = resolve_candidate_phrases();
    }
    if (candidates_path.empty()) {
        release_capture();
        std::cerr << "[!] --bip-scan needs a MNEMONIC CANDIDATES file:\n";
        std::cerr << "    a list of 12/15/18/21/24-word BIP-39 mnemonic\n";
        std::cerr << "    phrases (one per line, whitespace separated\n";
        std::cerr << "    words) to TRY against the bloom filter. This is\n";
        std::cerr << "    NOT the BIP-39 dictionary (english.txt) -- the\n";
        std::cerr << "    dictionary just validates checksums.\n\n";
        std::cerr << "    Drop a file named bip_phrases.txt at one of:\n";
        std::cerr << "      .\\wordlists\\bip_phrases.txt\n";
        std::cerr << "      .\\bip_phrases.txt\n";
        std::cerr << "      ~/.collider/wordlists/bip_phrases.txt\n";
        std::cerr << "    or pass --bip-scan-wordlist <path>.\n";
        return 1;
    }

    // Same treatment for the bloom: auto-detect funded_addresses.blf
    // (or bloom.blf) before complaining.
    std::string bloom_path = args.bloom_file;
    if (bloom_path.empty()) {
        bloom_path = resolve_bloom_filter();
    }
    if (bloom_path.empty()) {
        release_capture();
        std::cerr << "[!] --bip-scan needs a UTXO bloom filter; without\n";
        std::cerr << "    one every derived address would be probed against\n";
        std::cerr << "    nothing. Auto-detect looked for\n";
        std::cerr << "    funded_addresses.blf / bloom.blf under the exe\n";
        std::cerr << "    directory, ~/.collider/, and CWD. Pass\n";
        std::cerr << "    --bloom <path> to override.\n";
        return 1;
    }

    // Load the BIP-39 English wordlist for checksum validation.
    std::string bip39_path = resolve_bip39_wordlist();
    if (bip39_path.empty()) {
        release_capture();
        std::cerr << "[!] BIP-39 English wordlist not found. Auto-detect\n";
        std::cerr << "    looked for bip39_english.txt and english.txt under\n";
        std::cerr << "    data/crypto/, bip/, and bip39/ subdirs of the exe\n";
        std::cerr << "    directory, ~/.collider/, and CWD. Drop the file\n";
        std::cerr << "    at D:/collider/bip/english.txt (or similar) and\n";
        std::cerr << "    re-run.\n";
        return 1;
    }
    bip39::WordlistEnglish wordlist;
    try {
        wordlist.load(bip39_path);
    } catch (const std::exception& e) {
        release_capture();
        std::cerr << "[!] BIP-39 wordlist load failed: " << e.what() << "\n";
        return 1;
    }
    std::cout << "[*] BIP-39 wordlist: " << bip39_path << " ("
              << bip39::WordlistEnglish::kCount << " words)\n";

    // Load the primary bloom filter into host memory.
    BloomLoadResult bloom = load_bloom_file_into_memory(bloom_path);
    if (!bloom.ok) {
        release_capture();
        std::cerr << "[!] Bloom load failed: " << bloom.err_message << "\n";
        return 1;
    }
    std::cout << "[*] Bloom filter: " << bloom_path << " ("
              << bloom.header.num_elements << " elements, "
              << bloom.header.num_bits << " bits, k="
              << bloom.header.num_hashes << ")\n";

    // TP-4: optional tight bloom for secondary FP gating. When loaded,
    // an address must hit BOTH the primary AND the tight bloom before
    // we count it as a real hit. Mirrors brainwallet's dual-bloom path
    // so a misconfigured wordlist + giant tight bloom doesn't drown
    // bip_hits.txt in noise.
    BloomLoadResult tight_bloom;
    bool tight_bloom_enabled = false;
    if (!args.bloom_tight_file.empty()) {
        tight_bloom = load_bloom_file_into_memory(args.bloom_tight_file);
        if (!tight_bloom.ok) {
            std::cerr << "[!] WARNING: tight-bloom load failed: "
                      << tight_bloom.err_message
                      << " (continuing with primary bloom only)\n";
        } else {
            tight_bloom_enabled = true;
            std::cout << "[*] Tight bloom:  " << args.bloom_tight_file
                      << " ("
                      << tight_bloom.header.num_elements << " elements, "
                      << tight_bloom.header.num_bits << " bits, k="
                      << tight_bloom.header.num_hashes
                      << ") -- secondary gate enabled\n";
        }
    }

    auto profiles = default_profiles();
    std::cout << "[*] Derivation profiles: " << profiles.size()
              << " (early/Electrum/MultiBit/BIP-44/49/84)\n";
    int total_addr_per_phrase = 0;
    for (const auto& p : profiles) total_addr_per_phrase += p.count;
    std::cout << "[*] Addresses probed per mnemonic: "
              << total_addr_per_phrase << "\n\n";

    // Same release_active_capture_silent rationale as the
    // combinatorial path: FTXUI's draw uses std::cout and that's
    // rdbuf-redirected to a ring buffer while stdio_capture is
    // active, so the alt-screen would show empty without this. The
    // SILENT variant keeps the captured boot text in the log file
    // rather than scrolling it on the terminal right before the
    // alt-screen takes over.
    ::collider::ui::tui::StdioCapture::release_active_capture_silent();

    // Phase C TUI shell.
    ::collider::ui::tui::LaunchConfig launch_cfg;
    launch_cfg.mode_label              = "BIP-Scan";
    launch_cfg.version                 = std::string(::collider::kVersion);
    launch_cfg.tui_mode                = ::collider::ui::tui::TuiMode::BipScan;
    launch_cfg.gpu_ids                 = args.gpu_ids;
    launch_cfg.session_start           = std::chrono::steady_clock::now();
    launch_cfg.initial_phase_name      = "Loading mnemonic candidates";
    launch_cfg.initial_current_chunk   = 0;
    launch_cfg.initial_total_chunks    = 1;
    launch_cfg.render_cfg.refresh_hz   = 10;
    launch_cfg.render_cfg.alt_screen   = true;
    launch_cfg.guard_opts.alt_screen              = true;
    launch_cfg.guard_opts.hide_cursor             = true;
    launch_cfg.guard_opts.install_signal_handlers = true;
    auto session = ::collider::ui::tui::launch_session(launch_cfg);
    auto* tui_app = session.app.get();

    // Hit log: owner-only ofstream in CWD by convention.
    std::string hits_path = "bip_hits.txt";
    std::ofstream hits = collider::secure_open_ofstream(
        hits_path,
        std::ios::out | std::ios::app,
        collider::SecureWriteOnFailure::FallbackLoud);
    if (!hits.is_open()) {
        release_capture();
        std::cerr << "[!] Could not open " << hits_path << " for hits.\n";
        return 1;
    }
    hits << "# bip_hits.txt - " << ::collider::kVersion
         << "  mnemonic | derivation | private_key | address_kind | hash160\n";
    hits.flush();

    // Phrase reader: each non-blank, non-comment line is a candidate
    // mnemonic. Whitespace-separated; the BIP-39 validator does its
    // own checksum check + word-count gate.
    std::ifstream src(candidates_path);
    if (!src) {
        release_capture();
        std::cerr << "[!] Cannot read BIP candidate phrases file: "
                  << candidates_path << "\n";
        return 1;
    }
    // The "[*] BIP candidate phrases: <path>" cout previously here was
    // post-launch_session AND post-release_active_capture, so it
    // printed directly onto the alt-screen as garbled characters --
    // the "TUI drops to text" pattern the user complained about.
    // Deleted; the operator can see candidates_path in the run summary
    // or in ~/.collider/logs/.

    // Sanity peek: scan the first 64 non-blank lines and check the
    // word counts. BIP-39 mandates 12/15/18/21/24 words per phrase; if
    // the file is actually the BIP-39 DICTIONARY (english.txt, one word
    // per line) we would silently process 2048 lines, validate zero of
    // them, then exit clean -- the operator would never know what went
    // wrong. Releasing the capture here lets the warning reach stderr
    // even when StdioCapture is active.
    {
        std::ifstream peek(candidates_path);
        std::string pl;
        int lines_seen = 0;
        int multi_word_lines = 0;
        while (lines_seen < 64 && std::getline(peek, pl)) {
            if (pl.empty() || pl[0] == '#') continue;
            ++lines_seen;
            auto pw = bip39::split_words(pl);
            if (pw.size() >= 12) ++multi_word_lines;
        }
        if (lines_seen > 0 && multi_word_lines == 0) {
            release_capture();
            std::cerr << "\n[!] " << candidates_path
                      << " does not look like a mnemonic CANDIDATES\n"
                      << "    file. Every line in the first " << lines_seen
                      << " checked had fewer than\n"
                      << "    12 words. BIP-39 phrases are 12/15/18/21/24 "
                      << "words per line.\n\n"
                      << "    If you pointed --bip-scan-wordlist at the "
                      << "BIP-39 DICTIONARY\n"
                      << "    (english.txt with one word per line), that "
                      << "is the wrong file:\n"
                      << "    the dictionary is loaded automatically; you "
                      << "need a SEPARATE\n"
                      << "    file of mnemonic phrases to try. Aborting "
                      << "before wasting time.\n";
            return 2;
        }
    }
    // Re-open after the peek consumed bytes.
    src.close();
    src.open(candidates_path);
    if (!src) {
        release_capture();
        std::cerr << "[!] Lost BIP candidate phrases file after peek: "
                  << candidates_path << "\n";
        return 1;
    }

    uint64_t phrases_read = 0;
    uint64_t phrases_valid = 0;
    uint64_t addrs_probed = 0;
    uint64_t bloom_hits = 0;
    uint64_t phrases_invalid_checksum = 0;
    uint64_t seed_failures = 0;
    uint64_t master_failures = 0;
    uint64_t derive_failures = 0;
    // TP-4: per-error-class counters so the operator can tell silent
    // failures apart from "nothing was found." Each is non-zero only
    // when the corresponding exception path fires; the summary surfaces
    // them so a misconfigured run (e.g. wrong passphrase, wrong
    // wordlist language) does not look like a clean miss.
    // T1-C (2026-05-23 v1.5.0): multi-threaded fan-out. Per-phrase
    // derivation is embarrassingly parallel (no shared mutable state
    // except the hits writer + atomic counters). Atomics replace the
    // plain uint64_t accumulators so worker threads can publish stats
    // without a per-update mutex; the hits writer is mutex-guarded
    // since hits are rare and ordering doesn't matter. A bounded
    // queue (kQueueMax slots) backs the producer-consumer channel so
    // a fast disk reader can't OOM the box ahead of slow workers.
    std::atomic<uint64_t> a_phrases_read{phrases_read};
    std::atomic<uint64_t> a_phrases_valid{phrases_valid};
    std::atomic<uint64_t> a_addrs_probed{addrs_probed};
    std::atomic<uint64_t> a_bloom_hits{bloom_hits};
    std::atomic<uint64_t> a_invalid_checksum{0};
    std::atomic<uint64_t> a_seed_failures{0};
    std::atomic<uint64_t> a_master_failures{0};
    std::atomic<uint64_t> a_derive_failures{0};

    auto start = std::chrono::steady_clock::now();
    auto last_tui_post = start;

    // GPU dispatcher for the wordlist path. Same architecture as the
    // combinatorial path -- N per-device worker threads consume the
    // shared queue, process_batch into MultiAddressSession, route
    // hits through on_hit. Multi-threaded T1-C worker pool below
    // feeds the dispatcher via enqueue().
    BipGpuDispatcher gpu_dispatcher;
    bool gpu_active = false;
    std::mutex hits_mu;
    if (!args.bip_no_gpu) {
        auto gcfg = make_bip_dispatcher_config(
            args, bloom,
            make_bip_hit_callback(
                hits_mu, hits, a_bloom_hits,
                tight_bloom_enabled, tight_bloom,
                "bip_bloom_hit"));
        if (gpu_dispatcher.init(gcfg) == 0) {
            gpu_active = true;
        }
    }
    // Aggregate signal for "did per-worker PBKDF2 stream-create
    // succeed for AT LEAST one worker." Each worker bumps this once
    // on successful cudaStreamCreate; the publish block uses it
    // instead of inferring from gpu_init_diag (which only describes
    // the dispatcher's MultiAddressSession state, not the separate
    // per-worker PBKDF2 streams).
    std::atomic<unsigned> any_worker_pbkdf_gpu{0};

    // Last-walked profile label for TUI display. std::string is not
    // atomic; the publish uses a mutex-protected copy + a workers-
    // read-only path. Workers WRITE only their local copy and post
    // via a small label_mu lock at end of phrase; the TUI thread READS
    // under the same lock. Loose semantics are fine here (display only).
    std::string last_profile_label = "(starting)";
    std::mutex label_mu;

    constexpr size_t kQueueMax = 1024;
    std::deque<std::string> queue;
    std::mutex queue_mu;
    std::condition_variable queue_cv;
    std::atomic<bool> queue_closed{false};
    // hits_mu declared above with the GPU dispatcher init; reused
    // here for both the GPU on_hit callback and the legacy CPU
    // fallback path's hits-file write.

    // Per-worker GPU PBKDF2 batching. Each worker owns its own
    // cudaStream pinned to device (worker_idx % num_gpu_devices) so
    // both GPUs see traffic. The local batch fills, dispatches, and
    // the worker walks each resulting seed through the BIP-32 chain
    // + multi-address dispatcher exactly as the combinatorial path.
    // num_gpu_devices = max(1, args.gpu_ids.size()).
    auto worker_fn = [&](unsigned worker_idx) {
#if defined(COLLIDER_USE_CUDA)
        constexpr size_t kPbkdfBatchW = 128;
        constexpr size_t kMnemonicSlotW =
            ::collider::gpu::bip39::kMaxMnemonicBytes;
        std::vector<uint8_t>  pbkdf_packed_w(
            kPbkdfBatchW * kMnemonicSlotW, 0);
        std::vector<uint32_t> pbkdf_lens_w(kPbkdfBatchW, 0);
        std::vector<std::string> pbkdf_lines_w;
        pbkdf_lines_w.reserve(kPbkdfBatchW);
        std::vector<uint8_t>  pbkdf_seeds_w(kPbkdfBatchW * 64);
        // No silent {0} default -- detect_gpus() populated
        // args.gpu_ids before this runner was dispatched. If it's
        // empty here, the platform reported no devices and we should
        // honestly skip GPU PBKDF2 instead of pretending to use one.
        const auto& worker_devices = args.gpu_ids;
        const int my_device = worker_devices.empty()
            ? -1
            : worker_devices[worker_idx % worker_devices.size()];
        cudaStream_t my_stream = nullptr;
        bool worker_gpu = false;
        if (!args.bip_no_gpu && my_device >= 0) {
            if (cudaSetDevice(my_device) == cudaSuccess &&
                cudaStreamCreate(&my_stream) == cudaSuccess) {
                worker_gpu = true;
                any_worker_pbkdf_gpu.fetch_add(
                    1, std::memory_order_relaxed);
            }
        }

        auto walk_seed_w = [&](const std::string& line,
                               const uint8_t* seed64) {
            bip32::ExtKey master;
            try {
                master = bip32::master_from_seed(seed64, 64);
            } catch (const std::exception&) {
                a_master_failures.fetch_add(
                    1, std::memory_order_relaxed);
                return;
            }
            std::string local_label;
            for (const auto& prof : profiles) {
                local_label = prof.label;
                for (int i = 0; i < prof.count; ++i) {
                    std::string path = prof.path_template;
                    size_t pos = path.find("{idx}");
                    if (pos != std::string::npos) {
                        path.replace(pos, 5, std::to_string(i));
                    }
                    bip32::ExtKey child;
                    try {
                        auto parsed = bip32::parse_path(path);
                        child = bip32::derive_path(master, parsed);
                    } catch (const std::exception&) {
                        a_derive_failures.fetch_add(
                            1, std::memory_order_relaxed);
                        continue;
                    }
                    if (gpu_active) {
                        BipGpuWorkItem item;
                        std::memcpy(item.priv.data(),
                                    child.key.data(), 32);
                        item.mnemonic        = line;
                        item.derivation_path = path;
                        item.profile_label   = prof.label;
                        item.addr_mask = addr_mask_for(prof.addr_kind);
                        // enqueue() returns false on shutdown. Bail
                        // this seed walk; outer loop's g_shutdown
                        // check exits the worker cleanly.
                        if (!gpu_dispatcher.enqueue(std::move(item))) {
                            return;
                        }
                        a_addrs_probed.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
            }
            if (!local_label.empty()) {
                std::lock_guard<std::mutex> ll(label_mu);
                last_profile_label = local_label;
            }
        };

        auto flush_pbkdf_w = [&]() {
            if (pbkdf_lines_w.empty()) return;
            const size_t n = pbkdf_lines_w.size();
            if (worker_gpu) {
                cudaSetDevice(my_device);
                ::collider::gpu::bip39::Pbkdf2Batch req{};
                req.mnemonic_bytes = pbkdf_packed_w.data();
                req.mnemonic_lens  = pbkdf_lens_w.data();
                req.count          = n;
                static const std::string salt = "mnemonic";
                req.salt_bytes = reinterpret_cast<const uint8_t*>(
                    salt.data());
                req.salt_len   = static_cast<uint32_t>(salt.size());
                req.out_seeds  = pbkdf_seeds_w.data();
                cudaError_t rc = ::collider::gpu::bip39::
                    run_pbkdf2_batch(req, my_stream);
                if (rc == cudaSuccess) {
                    for (size_t i = 0; i < n; ++i) {
                        walk_seed_w(pbkdf_lines_w[i],
                                     pbkdf_seeds_w.data() + i * 64);
                    }
                    pbkdf_lines_w.clear();
                    std::fill(pbkdf_packed_w.begin(),
                              pbkdf_packed_w.end(), 0);
                    return;
                }
            }
            // CPU fallback per mnemonic.
            for (size_t i = 0; i < n; ++i) {
                try {
                    auto seed = bip32::mnemonic_to_seed(
                        pbkdf_lines_w[i], std::string{});
                    walk_seed_w(pbkdf_lines_w[i], seed.data());
                } catch (const std::exception&) {
                    a_seed_failures.fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
            pbkdf_lines_w.clear();
            std::fill(pbkdf_packed_w.begin(),
                      pbkdf_packed_w.end(), 0);
        };
#endif  // COLLIDER_USE_CUDA

        while (true) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(queue_mu);
                queue_cv.wait(lk, [&] {
                    return !queue.empty() || queue_closed.load()
                        || g_shutdown.load();
                });
                if (queue.empty()) {
#if defined(COLLIDER_USE_CUDA)
                    flush_pbkdf_w();
                    if (my_stream) {
                        cudaSetDevice(my_device);
                        cudaStreamDestroy(my_stream);
                    }
#endif
                    return;  // closed + drained
                }
                line = std::move(queue.front());
                queue.pop_front();
            }
            queue_cv.notify_all();  // wake producer waiting on space

            auto words = bip39::split_words(line);
            std::vector<uint8_t> entropy;
            if (!bip39::validate_mnemonic(words, wordlist, &entropy)) {
                a_invalid_checksum.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            a_phrases_valid.fetch_add(1, std::memory_order_relaxed);

#if defined(COLLIDER_USE_CUDA)
            // Append to this worker's PBKDF2 batch + dispatch on
            // fill. The walk_seed_w lambda above handles the chain
            // walk + multi-address dispatch.
            if (line.size() < kMnemonicSlotW) {
                const size_t slot = pbkdf_lines_w.size();
                std::memcpy(pbkdf_packed_w.data() + slot * kMnemonicSlotW,
                            line.data(), line.size());
                pbkdf_lens_w[slot] = static_cast<uint32_t>(line.size());
                pbkdf_lines_w.push_back(line);
                if (pbkdf_lines_w.size() >= kPbkdfBatchW) {
                    flush_pbkdf_w();
                }
                continue;  // skip the legacy single-mnemonic path
            }
#endif

            std::array<uint8_t, 64> seed;
            try {
                seed = bip32::mnemonic_to_seed(line, std::string{});
            } catch (const std::exception&) {
                a_seed_failures.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            bip32::ExtKey master;
            try {
                master = bip32::master_from_seed(seed.data(), seed.size());
            } catch (const std::exception&) {
                a_master_failures.fetch_add(1, std::memory_order_relaxed);
                continue;
            }

            std::string this_phrase_last_label;
            for (const auto& prof : profiles) {
                this_phrase_last_label = prof.label;
                for (int i = 0; i < prof.count; ++i) {
                    std::string path = prof.path_template;
                    size_t pos = path.find("{idx}");
                    if (pos != std::string::npos) {
                        path.replace(pos, 5, std::to_string(i));
                    }
                    bip32::ExtKey child;
                    try {
                        auto parsed = bip32::parse_path(path);
                        child = bip32::derive_path(master, parsed);
                    } catch (const std::exception&) {
                        a_derive_failures.fetch_add(
                            1, std::memory_order_relaxed);
                        continue;
                    }
                    if (gpu_active) {
                        // GPU path: enqueue priv key + context for the
                        // dispatcher's per-device worker to process.
                        BipGpuWorkItem item;
                        std::memcpy(item.priv.data(), child.key.data(), 32);
                        item.mnemonic        = line;
                        item.derivation_path = path;
                        item.profile_label   = prof.label;
                        item.addr_mask = addr_mask_for(prof.addr_kind);
                        if (!gpu_dispatcher.enqueue(std::move(item))) {
                            // Dispatcher shutting down. Stop walking
                            // this phrase; outer loop's queue_closed
                            // path exits cleanly.
                            break;
                        }
                        a_addrs_probed.fetch_add(
                            1, std::memory_order_relaxed);
                    } else {
                        // Pure-CPU fallback (--no-bip-gpu).
                        auto pub = bip32::detail::priv_to_pub(
                            child.key.data());
                        std::array<uint8_t, 20> h160{};
                        switch (prof.addr_kind) {
                            case DerivationProfile::AddrKind::P2SH_P2WPKH:
                                h160 = hash160_p2sh_p2wpkh(pub.data());
                                break;
                            default:
                                h160 = hash160_pubkey(pub.data());
                                break;
                        }
                        const bool primary_hit =
                            bloom_probe(bloom.header, bloom.data.data(),
                                        h160.data());
                        a_addrs_probed.fetch_add(
                            1, std::memory_order_relaxed);
                        if (primary_hit) {
                            if (tight_bloom_enabled &&
                                !bloom_probe(tight_bloom.header,
                                             tight_bloom.data.data(),
                                             h160.data())) {
                                continue;
                            }
                            a_bloom_hits.fetch_add(
                                1, std::memory_order_relaxed);
                            write_hit_record(hits_mu, hits, line, path,
                                             prof.label, child.key.data(),
                                             addr_kind_label(prof.addr_kind),
                                             h160.data());
                            ::collider::log::milestone(
                                "bip_bloom_hit",
                                prof.label + " path=" + path);
                        }
                    }
                }
            }
            if (!this_phrase_last_label.empty()) {
                std::lock_guard<std::mutex> ll(label_mu);
                last_profile_label = this_phrase_last_label;
            }
        }
    };

    // Spawn (hardware_concurrency - 1) workers; reserve one logical
    // core for the producer (file IO + TUI ticks). Clamp to >=1 so a
    // single-core box still produces work.
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned thread_count = hw > 1 ? hw - 1 : 1;
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (unsigned i = 0; i < thread_count; ++i) {
        // Worker index passed in so the per-worker PBKDF2 batcher
        // can pin its stream to device (i % num_devices) for
        // multi-GPU round-robin coverage.
        workers.emplace_back(worker_fn, i);
    }
    // The "[*] BIP scanner threads: N (T1-C)\n" cout previously here
    // was post-launch_session AND post-release_active_capture, so it
    // printed directly onto the alt-screen. Also "T1-C" was an
    // internal sprint task label leaked into operator-facing output.
    // Deleted; bi.worker_threads is published to the dashboard's
    // WORKERS row every tick.

    auto post_tui = [&](std::chrono::steady_clock::time_point now) {
        if (!tui_app) return;
        const double secs =
            std::chrono::duration<double>(now - start).count();
        const uint64_t probed = a_addrs_probed.load();
        const uint64_t pr_read = a_phrases_read.load();
        const uint64_t pr_valid = a_phrases_valid.load();
        const uint64_t hits_n = a_bloom_hits.load();
        tui_app->set_keys_per_sec_current(
            secs > 0 ? probed / secs : 0.0);
        tui_app->set_chunk_progress(
            static_cast<int>(std::min<uint64_t>(pr_valid, INT_MAX)),
            static_cast<int>(std::min<uint64_t>(pr_read + 1, INT_MAX)));
        ::collider::ui::tui::BipScanInfo bi;
        bi.phrases_read     = pr_read;
        bi.phrases_valid    = pr_valid;
        bi.addresses_probed = probed;
        bi.bloom_hits       = hits_n;
        {
            std::lock_guard<std::mutex> ll(label_mu);
            bi.current_profile = last_profile_label;
        }
        // Dashboard refactor 2026-05-24: mirror combinatorial path.
        bi.mode_label = "wordlist";
        bi.bloom_elements = bloom.header.num_elements;
        bi.derivation_profiles = static_cast<int>(profiles.size());
        {
            int total_per_phrase = 0;
            for (const auto& p : profiles) total_per_phrase += p.count;
            bi.addresses_per_phrase = total_per_phrase;
        }
        bi.worker_threads = thread_count;
        // Mirror combinatorial path: surface GPU diagnostics so the
        // dashboard reflects reality (init failure / partial init /
        // --no-bip-gpu / CUDA missing) instead of the stale "BIP-39
        // PBKDF2 is CPU-bound" placeholder.
        // GPU diagnostics via the shared helper. pbkdf_gpu_active is
        // aggregated from per-worker cudaStream-create success across
        // all of the wordlist worker pool's threads.
        publish_dispatcher_dashboard(
            bi, gpu_dispatcher, gpu_active,
            /*pbkdf_gpu_active=*/
            any_worker_pbkdf_gpu.load(std::memory_order_relaxed) > 0,
            /*gpu_disabled_by_flag=*/args.bip_no_gpu);
        if (secs > 0) {
            bi.phrases_per_sec   = pr_read / secs;
            bi.addresses_per_sec = probed / secs;
        }
        tui_app->set_bip_scan_info(bi);
        tui_app->set_current_phase_name("BIP scan");
        if (tui_app->requested_quit() && !g_shutdown.load()) {
            g_shutdown.store(true);
        }
    };

    // Producer loop: read lines from file, push into bounded queue.
    // Block when queue full so a fast disk + slow workers does not
    // OOM the host; back-pressure via the condition variable.
    std::string line;
    while (std::getline(src, line) && !g_shutdown.load()) {
        a_phrases_read.fetch_add(1, std::memory_order_relaxed);
        if (line.empty() || line[0] == '#') continue;
        {
            std::unique_lock<std::mutex> lk(queue_mu);
            queue_cv.wait(lk, [&] {
                return queue.size() < kQueueMax || g_shutdown.load();
            });
            if (g_shutdown.load()) break;
            queue.push_back(std::move(line));
        }
        queue_cv.notify_one();

        auto now = std::chrono::steady_clock::now();
        if (now - last_tui_post > std::chrono::milliseconds(100)) {
            post_tui(now);
            last_tui_post = now;
        }
    }
    queue_closed.store(true);
    queue_cv.notify_all();
    for (auto& t : workers) t.join();

    // Drain + shutdown the GPU dispatcher AFTER CPU workers exit so
    // every priv key they enqueued reaches the GPU; on_hit callbacks
    // may still run during shutdown() and append to bip_hits.txt.
    if (gpu_active) {
        gpu_dispatcher.shutdown();
        // a_addrs_probed now only counts items that successfully
        // entered the dispatcher (incremented AFTER enqueue() returned
        // true). No max() reconciliation needed -- the CPU counter and
        // gpu_dispatcher.total_addresses_dispatched() converge after
        // drain.
    }

    // Final stat snapshot from atomics back into the local counters
    // so the shutdown summary uses the actual end-of-run values.
    phrases_read = a_phrases_read.load();
    phrases_valid = a_phrases_valid.load();
    addrs_probed = a_addrs_probed.load();
    bloom_hits = a_bloom_hits.load();
    phrases_invalid_checksum = a_invalid_checksum.load();
    seed_failures = a_seed_failures.load();
    master_failures = a_master_failures.load();
    derive_failures = a_derive_failures.load();

    // Final TUI tick so the operator sees the post-scan numbers
    // before the scan-complete hold below takes over.
    post_tui(std::chrono::steady_clock::now());

    auto end = std::chrono::steady_clock::now();
    const double elapsed_s = std::chrono::duration<double>(end - start).count();

    // Tear the TUI down immediately on scan completion so the operator
    // sees the cout summary in their normal terminal scrollback. The
    // historical "hold TUI until q" path (2026-05-23) was supposed to
    // prevent the alt-screen from snapping shut on tiny inputs, but
    // user-reported UX (2026-05-24): "screen goes blank, hit Ctrl-C"
    // -- the BIP TUI panel doesn't render the "press q to exit" hint
    // anywhere obvious, so the user thinks the process hung. New flow:
    //   1. Drop the TUI alt-screen immediately (release_capture).
    //   2. Print the summary to cout where PowerShell will scroll-back
    //      preserve it.
    //   3. Wait on stdin Enter (one keystroke) before returning so the
    //      operator gets a chance to read before the process exits.
    //   4. Skip the prompt when --no-tui (non-interactive) was set or
    //      g_shutdown fired (Ctrl-C already explicitly chose exit).
    release_capture();

    std::cout << "\n";
    std::cout << "================================================================\n";
    if (bloom_hits > 0) {
        std::cout << "  BIP SCAN COMPLETE -- " << bloom_hits
                  << " HIT(S) FOUND\n";
    } else {
        std::cout << "  BIP SCAN COMPLETE -- 0 hits\n";
    }
    std::cout << "================================================================\n";
    std::cout << "    Phrases read:        " << phrases_read << "\n";
    std::cout << "    Phrases valid:       " << phrases_valid << "\n";
    std::cout << "    Bad checksum:        " << phrases_invalid_checksum
              << "\n";
    if (seed_failures || master_failures || derive_failures) {
        std::cout << "    Seed errors:         " << seed_failures << "\n";
        std::cout << "    Master errors:       " << master_failures << "\n";
        std::cout << "    Derive errors:       " << derive_failures << "\n";
    }
    std::cout << "    Addresses probed:    " << addrs_probed << "\n";
    std::cout << "    Bloom hits:          " << bloom_hits << "\n";
    std::cout << "    Elapsed:             " << std::fixed
              << std::setprecision(2) << elapsed_s << " s\n";
    if (bloom_hits > 0) {
        std::cout << "    Hits written:        " << hits_path << "\n";
    }
    std::cout << "================================================================\n";
    // The "Press Enter to exit" prompt was removed (combinatorial path
    // has the same change). Hung the process on piped stdin; the
    // operator who pressed 'q' to leave the TUI already said "done".
    return 0;
}

}  // namespace collider::runtime

#endif  // COLLIDER_PRO
