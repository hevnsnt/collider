/**
 * warpwallet_runner.hpp - WarpWallet CPU scan loop (PRO-only).
 *
 * T3.14 (2026-05-17): Extracted from brain_wallet_runner.cpp. WarpWallet is
 * a CPU-only scan path that runs an independent generator + scrypt-based
 * Keybase derivation; it shares nothing with the GPU brainwallet pipeline
 * past the bloom header + UVRF inputs. Pulling it out of the dispatch
 * shape "mode within a mode within a function" puts it on the same
 * footing as the GPU brainwallet mode: top-level dispatch from main.cpp's
 * mode router or from the brain-wallet runner before any GPU work begins.
 */
#pragma once

#ifdef COLLIDER_PRO

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "cli/cli_parser.hpp"      // Arguments
#include "core/hit_verifier.hpp"   // HitVerifier; transitively brings in
                                   // ::collider::utxo::BloomFilterHeader
                                   // from tools/utxo_bloom_builder.hpp.

namespace collider::runtime {

/**
 * Run the WarpWallet CPU scan loop. Caller has already resolved the
 * effective wordlist + pcfg paths, loaded the bloom data + header, and
 * (optionally) built a HitVerifier + tight CPU-bloom probe. Returns the
 * process exit code (0 on clean scan end / shutdown; 1 on init failure).
 *
 * The tight_bloom_check + tight_bloom_filtered pair mirror the
 * dual-bloom contract used elsewhere in the runner: tight_bloom_check
 * returns true when the tight bloom is unloaded (no filtering) and when
 * the h160 hits every k positions in the loaded tight bloom; counts of
 * tight-bloom rejections are accumulated into tight_bloom_filtered.
 */
int run_warpwallet_scan(
    const Arguments& args,
    const std::string& effective_wordlist_path,
    const std::string& effective_pcfg_path,
    const ::collider::utxo::BloomFilterHeader& bloom_header,
    const std::vector<uint8_t>& bloom_data,
    const std::function<bool(const uint8_t*)>& tight_bloom_check,
    uint64_t& tight_bloom_filtered,
    const std::unique_ptr<::collider::HitVerifier>& hit_verifier);

}  // namespace collider::runtime

#endif  // COLLIDER_PRO
