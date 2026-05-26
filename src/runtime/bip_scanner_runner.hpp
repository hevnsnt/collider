/**
 * bip_scanner_runner.hpp -- v1.5.x BIP brainwallet scanner.
 *
 * Treats every line of an operator-supplied wordlist file as a BIP-39
 * mnemonic candidate (12/15/18/21/24 words separated by whitespace).
 * For each candidate that passes BIP-39 checksum validation, the runner:
 *
 *   1. Derives the 64-byte seed via PBKDF2-HMAC-SHA512(mnemonic,
 *      salt="mnemonic"+passphrase, c=2048).
 *   2. Builds the BIP-32 master extended key (HMAC-SHA512(key="Bitcoin
 *      seed", data=seed)).
 *   3. Walks every "historical or modern" derivation path the operator
 *      configured (see DerivationProfile). The default profile covers
 *      pre-BIP-44 wallets (Electrum 2.x, MultiBit HD, blockchain.info,
 *      early Bitcoin Core HD), BIP-44 P2PKH, BIP-49 P2SH-P2WPKH, and
 *      BIP-84 native segwit P2WPKH. For each derived child key, the
 *      runner computes hash160(compressed_pubkey) and (for P2SH-P2WPKH)
 *      the wrapped hash160; both are probed against the loaded bloom
 *      filter via the same MurmurHash3-128 double-hash scheme the GPU
 *      brainwallet pipeline uses.
 *
 * Per-phrase cost: ~5 PBKDF2 rounds (one for each base path's first
 * derivation level cache) + N CKDpriv calls + N ec_mul + N hash160.
 * BIP-39 + PBKDF2 dominates at ~1 ms per phrase on a modern CPU. The
 * scanner is single-threaded for clarity in the first cut; multi-
 * threaded fan-out is a v1.5.1 follow-up because BIP-39 candidates are
 * embarrassingly parallel (no shared state).
 *
 * Hit policy: on a bloom hit, the recovered mnemonic + derivation path
 * + private key (32 bytes hex) is appended to bip_hits.txt (owner-only
 * via secure_open_ofstream). If the bloom emits a false positive the
 * line still lands; downstream HitVerifier or the operator manually
 * cross-references against the funded-address UTXO set before acting.
 */
#pragma once

#ifdef COLLIDER_PRO

#include "cli/cli_parser.hpp"
#include "runtime/runtime_globals.hpp"

namespace collider::runtime {

// Entry point matching the convention used by run_brain_wallet_mode /
// run_pool_mode. Reads args.bip_scan_wordlist as the mnemonic source
// (one candidate per line, whitespace-separated words). Returns the
// process exit code: 0 on clean shutdown, non-zero on a fatal error
// (bloom-load failure, BIP-39 wordlist missing, etc.). Hit counts are
// surfaced via the unified TUI's chunk-progress + status panels.
int run_bip_scan_mode(const Arguments& args);

// Auto-detect helpers exported for the interactive_ui flow so the
// "ask the operator for a path" prompts can be skipped when the
// system already has the file. Empty string return = not found.
// All three walk the same data_search_roots() (exe-relative,
// ~/.collider, ~/, CWD) and accept multiple filename conventions.
std::string resolve_bip39_wordlist();
std::string resolve_bloom_filter();
std::string resolve_candidate_phrases();

}  // namespace collider::runtime

#endif  // COLLIDER_PRO
