/**
 * warpwallet_runner.cpp - WarpWallet CPU scan loop implementation.
 *
 * T3.14 (2026-05-17): extracted from brain_wallet_runner.cpp. Mirrors the
 * exact behaviour of the inline run_warpwallet_scan helper that used to
 * live there: same status-line format, same hit log content, same dual-
 * bloom + UVRF gating, same empty-hit writer construction. Code is
 * byte-identical apart from the file-local enqueue_empty_hit helper that
 * was moved out alongside it.
 */
#ifdef COLLIDER_PRO

#include "runtime/warpwallet_runner.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>

#include "core/byte_codec.hpp"
#include "core/secure_buffer.hpp"
#include "core/secure_write.hpp"
#include "core/session_log.hpp"  // milestone() for phase / hit_verified /
                                 // state_saved (WarpWallet scan).
#include "core/warpwallet.hpp"
#include "generators/streaming_brain_wallet.hpp"
#include "runtime/empty_hit_writer.hpp"
#include "runtime/runtime_globals.hpp"  // g_shutdown
#include "ui/box_render.hpp"

#include <sstream>

namespace collider::runtime {

namespace {

// File-local mirror of brain_wallet_runner.cpp's enqueue_empty_hit. Kept
// in this TU so this scan path can be reviewed without cross-file context.
// Format produced is byte-identical to the brainwallet path's empty-hit
// records (same ISO-8601 UTC, same hex encoding, same field order in the
// background writer's serialization).
inline void enqueue_empty_hit(EmptyHitWriter& writer,
                              const std::string& passphrase,
                              const uint8_t privkey[32],
                              const uint8_t h160[20]) {
    EmptyHitRecord rec;

    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::strftime(rec.ts_iso.data(), rec.ts_iso.size(),
                  "%Y-%m-%dT%H:%M:%SZ", &tm);

    std::memcpy(rec.privkey.data(), privkey, 32);
    std::memcpy(rec.h160.data(), h160, 20);
    rec.passphrase = passphrase;

    writer.enqueue(std::move(rec));
}

}  // namespace

int run_warpwallet_scan(
    const Arguments& args,
    const std::string& effective_wordlist_path,
    const std::string& effective_pcfg_path,
    const ::collider::utxo::BloomFilterHeader& bloom_header,
    const std::vector<uint8_t>& bloom_data,
    const std::function<bool(const uint8_t*)>& tight_bloom_check,
    uint64_t& tight_bloom_filtered,
    const std::unique_ptr<::collider::HitVerifier>& hit_verifier)
{
    using namespace ::collider;

    {
        namespace boxui = ::collider::ui::box;
        std::cout << "\n";
        boxui::top(std::cout);
        boxui::centered(std::cout, "WARPWALLET CPU MODE");
        boxui::top(std::cout);
        boxui::kv(std::cout, "Salt", args.warpwallet_salt);
        boxui::kv(std::cout, "Derivation",
                  "scrypt(N=2^18,r=8,p=1) + PBKDF2(c=2^16) XOR");
        boxui::kv(std::cout, "Throughput",
                  "~1 candidate/sec/core (scrypt is memory-hard)");
        boxui::bottom(std::cout);
        std::cout << "\n";
    }

    // CPU bloom probe. Recomputes the same double-hash bit scheme used by
    // the GPU kernel (h160_bloom_filter.cu) so a hit reported here is
    // byte-identical to a hit the GPU path would have reported on the
    // same h160.
    const uint64_t bloom_num_bits = bloom_header.num_bits;
    const uint32_t bloom_num_hashes = bloom_header.num_hashes;
    const uint32_t bloom_seed = bloom_header.seed;
    const uint8_t* bloom_bytes = bloom_data.data();

    auto cpu_bloom_check = [&](const std::array<uint8_t, 20>& h160) -> bool {
        auto [h1, h2] = ::collider::utxo::murmurhash3_128(
            h160.data(), 20, bloom_seed);
        for (uint32_t i = 0; i < bloom_num_hashes; i++) {
            uint64_t bit_idx = (h1 + (uint64_t)i * h2) % bloom_num_bits;
            uint64_t byte_idx = bit_idx >> 3;
            uint8_t bit_in_byte = static_cast<uint8_t>(bit_idx & 7);
            if ((bloom_bytes[byte_idx] & (1u << bit_in_byte)) == 0) {
                return false;
            }
        }
        return true;
    };

    // Build the candidate generator.
    ::collider::generators::StreamingBrainWallet::Config gen_cfg;
    gen_cfg.base_wordlist = effective_wordlist_path;
    gen_cfg.batch_size = 1024;
    gen_cfg.enable_dedup = false;
    gen_cfg.verbose = args.debug;
    if (!effective_pcfg_path.empty()) gen_cfg.pcfg_model_path = effective_pcfg_path;

    ::collider::generators::StreamingBrainWallet generator(gen_cfg);
    if (!generator.init()) {
        std::cerr << "[!] WarpWallet: failed to initialize candidate generator.\n";
        return 1;
    }

    ::collider::warpwallet::WarpWalletConfig wpw_cfg;
    wpw_cfg.salt = args.warpwallet_salt;
    wpw_cfg.enabled = true;
    ::collider::warpwallet::WarpWalletProcessor wpw(wpw_cfg);

    auto start = std::chrono::steady_clock::now();
    uint64_t total_checked = 0;
    uint64_t bloom_hits = 0;
    uint64_t verified_hits = 0;
    uint64_t empty_hits = 0;

    // One empty-hit writer per scan run, joined on scope exit.
    // Constructing it unconditionally keeps the scan-loop branch shape
    // tight (no per-iteration "is writer constructed" test); if
    // track_empty_hits is off, the writer thread idles on its condvar
    // and the destructor still joins cleanly.
    EmptyHitWriter empty_hit_writer("found-empty.txt");

    // Session log: WarpWallet has a single, monolithic scan phase (no
    // rule-based mutation phases like brain_wallet_runner). Emit one
    // phase milestone at entry so downstream tooling can correlate the
    // start of the run against the bloom_loaded / wordlist_loaded
    // events upstream. Salt is logged because it is part of the
    // derivation contract and a misconfigured salt would produce a
    // run that cannot match any pre-computed target; salt is NOT a
    // secret in WarpWallet's threat model.
    {
        std::ostringstream d;
        d << "phase=\"warpwallet_scan\""
          << " salt=\"" << args.warpwallet_salt << "\""
          << " wordlist=" << effective_wordlist_path
          << " pcfg=" << (effective_pcfg_path.empty()
                              ? std::string("(none)")
                              : effective_pcfg_path)
          << " bloom_bits=" << bloom_num_bits
          << " bloom_hashes=" << bloom_num_hashes;
        ::collider::log::milestone("phase", d.str());
    }

    while (!::g_shutdown.load(std::memory_order_acquire)) {
        auto batch = generator.next_batch();
        if (batch.empty()) break;

        for (const auto& passphrase : batch) {
            if (::g_shutdown.load(std::memory_order_acquire)) break;

            auto h160 = wpw.process(passphrase);
            total_checked++;

            if (!cpu_bloom_check(h160)) continue;
            bloom_hits++;

            // Bloom positive: verify via UVRF when available.
            if (hit_verifier) {
                ::collider::utxo::H160 uh;
                std::memcpy(uh.data, h160.data(), 20);
                auto verified = hit_verifier->verify(uh);
                if (!verified.has_value()) {
                    // --track-empty-hits: real but empty.
                    if (args.track_empty_hits) {
                        // Dual-bloom gate: re-probe the tight CPU bloom
                        // before claiming this as a real-but-empty hit.
                        // Drops loose-bloom FPs.
                        if (!tight_bloom_check(h160.data())) {
                            tight_bloom_filtered++;
                            continue;
                        }
                        auto pk = ::collider::warpwallet::derive_key(
                            passphrase, args.warpwallet_salt);
                        // Enqueue on the writer thread instead of opening +
                        // closing the file inline.
                        enqueue_empty_hit(empty_hit_writer,
                                          passphrase, pk.data(), h160.data());
                        empty_hits++;
                    }
                    continue;  // not a funded hit; skip box
                }
            }
            verified_hits++;

            // Session log: bloom-positive + UVRF-verified hit.
            // Cardinality only; passphrase, salt, and the recovered
            // private key are intentionally NOT logged here. The
            // recovered material lands in puzzle_found.txt via
            // secure_open_ofstream (FailHard, owner-only DACL); the
            // session log is a less-restricted sink and duplicating
            // sensitive material here would widen the leak surface.
            ::collider::log::milestone(
                "hit_verified",
                "scan=warpwallet total_verified_hits=" +
                    std::to_string(verified_hits));

            auto privkey = ::collider::warpwallet::derive_key(passphrase, args.warpwallet_salt);
            char privkey_hex[65];
            for (int i = 0; i < 32; i++) {
                std::snprintf(privkey_hex + i * 2, 3, "%02x", privkey[i]);
            }
            privkey_hex[64] = '\0';

            {
                namespace boxui = ::collider::ui::box;
                std::cout << "\n\n";
                boxui::top(std::cout);
                boxui::centered(std::cout, "WARPWALLET HIT FOUND");
                boxui::top(std::cout);
                boxui::kv(std::cout, "Passphrase", passphrase);
                boxui::kv(std::cout, "Salt",       args.warpwallet_salt);
                boxui::kv(std::cout, "Privkey",    privkey_hex);
                char h160_hex[41];
                for (int i = 0; i < 20; i++) {
                    std::snprintf(h160_hex + i * 2, 3, "%02x", h160[i]);
                }
                h160_hex[40] = '\0';
                boxui::kv(std::cout, "H160",       h160_hex);
                boxui::bottom(std::cout);
                std::cout << "\n";
            }

            // Append to puzzle_found.txt for durability. Owner-only file
            // permissions (POSIX 0600 / Windows owner-only DACL) so other
            // local users cannot read the recovered private key from a
            // shared host. FailHard: the file contains a recovered Bitcoin
            // private key in plaintext; "no file" is the only acceptable
            // failure mode (a world-readable file would be worse than no
            // file at all).
            std::ofstream hit_file =
                ::collider::secure_open_ofstream(
                    "puzzle_found.txt",
                    std::ios::app,
                    ::collider::SecureWriteOnFailure::FailHard);
            if (hit_file) {
                hit_file << "WARPWALLET HIT\n";
                hit_file << "passphrase=" << passphrase << "\n";
                hit_file << "salt=" << args.warpwallet_salt << "\n";
                hit_file << "privkey=" << privkey_hex << "\n\n";
            }
        }

        // Status line ~ every 1s.
        auto now = std::chrono::steady_clock::now();
        double sec = std::chrono::duration_cast<std::chrono::milliseconds>(
                         now - start).count() / 1000.0;
        if (sec >= 1.0 && total_checked > 0) {
            if (args.track_empty_hits) {
                std::printf("\r[*] WarpWallet: checked=%llu  empty=%llu  "
                            "verified=%llu  rate=%.2f/s   ",
                            (unsigned long long)total_checked,
                            (unsigned long long)empty_hits,
                            (unsigned long long)verified_hits,
                            total_checked / sec);
            } else {
                std::printf("\r[*] WarpWallet: checked=%llu  bloom_hits=%llu  "
                            "verified=%llu  rate=%.2f/s   ",
                            (unsigned long long)total_checked,
                            (unsigned long long)bloom_hits,
                            (unsigned long long)verified_hits,
                            total_checked / sec);
            }
            std::fflush(stdout);
        }
    }

    std::cout << "\n[*] WarpWallet scan ended. Total checked: " << total_checked
              << ", bloom hits: " << bloom_hits
              << ", verified hits: " << verified_hits;
    if (args.track_empty_hits) {
        std::cout << ", real-but-empty: " << empty_hits
                  << " (see found-empty.txt)";
    }
    std::cout << "\n";

    // Session log: WarpWallet keeps no resumable scan-state file (each
    // batch is independent; no cursor), so the only persistent on-disk
    // sink for this run is the empty-hits log when --track-empty-hits
    // is on. Emit a single state_saved milestone at scan end so the
    // run-history has a deterministic terminator. kind=warpwallet_summary
    // distinguishes this from the kangaroo herd state_saved records.
    {
        std::ostringstream d;
        d << "kind=warpwallet_summary"
          << " total_checked=" << total_checked
          << " bloom_hits=" << bloom_hits
          << " verified_hits=" << verified_hits;
        if (args.track_empty_hits) {
            d << " empty_hits=" << empty_hits
              << " empty_hit_log=found-empty.txt";
        }
        ::collider::log::milestone("state_saved", d.str());
    }

    return 0;
}

}  // namespace collider::runtime

#endif  // COLLIDER_PRO
