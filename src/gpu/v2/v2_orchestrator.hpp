/**
 * Brain Wallet v2 host-side orchestrator (Phases 3, 4, 9; v1.4.0).
 *
 * Sits between the CLI and the v2 device kernels. Owns:
 *   - parsing of --schemes / --addr-types CSV flags into bitmasks
 *   - loading puzzle keys from data/puzzle_history.json
 *   - dispatching to v2_brain_wallet_batch() with the right masks
 *   - draining V2MatchRecord results back to stdout / hits file
 *
 * All public functions are pure host code; they do NOT include cuda_runtime.h
 * unless COLLIDER_USE_CUDA is set. This lets us unit-test scheme parsing,
 * puzzle-key loading, and the orchestrator wire-up without a GPU.
 */

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace collider {
namespace gpu {
namespace v2 {

// Forward decls -- the full PuzzleTarget definition lives in
// brain_wallet_v2.hpp and pulls in cuda_runtime.h on Pro builds.
struct PuzzleTarget;

// ---------------------------------------------------------------------------
// CSV mask parsers (Phase 3 + Phase 4)
// ---------------------------------------------------------------------------

/**
 * Parse a comma-separated list of derivation-scheme names into a
 * bitmask of `DerivationScheme` values.
 *
 * Names are case-insensitive. The special value "all" or "*" means
 * SCHEME_MASK_ALL; "stock" or empty input means SCHEME_MASK_STOCK.
 *
 * Recognized names:
 *   sha256_pw, sha256_sha256_pw, sha256_pw_newline, sha256_pw_pw,
 *   sha256_sha256_pw_pw, sha256_iter_16, hmac_sha512_pw, sha512_pw_half
 *
 * @param csv input string
 * @param mask_out output: bitmask
 * @param error_out output: empty on success; otherwise human-readable
 *                  description of the first invalid token
 * @return true on success, false on parse error
 */
bool parse_scheme_mask(std::string_view csv,
                       uint32_t& mask_out,
                       std::string& error_out);

/**
 * Parse a comma-separated list of address-type names into a bitmask
 * of `AddressType` values.
 *
 * "all" or "*" => ADDR_MASK_ALL.
 * "modern"     => ADDR_MASK_MODERN.
 * "stock" or empty input => ADDR_MASK_STOCK.
 *
 * Recognized names:
 *   p2pkh_uncompressed, p2pkh_compressed, p2sh_p2wpkh, p2wpkh_v0, p2tr_bip86
 *
 * @param csv input string
 * @param mask_out output: bitmask
 * @param error_out output: empty on success; description on error
 */
bool parse_addr_mask(std::string_view csv,
                     uint32_t& mask_out,
                     std::string& error_out);

// ---------------------------------------------------------------------------
// Puzzle target loader (Phase 9)
// ---------------------------------------------------------------------------

/**
 * Load known-solved puzzle keys from a `puzzle_history.json` file and
 * convert each into a PuzzleTarget. Skips entries whose `private_key_hex`
 * is missing or invalid; logs a warning for each skip.
 *
 * The expected JSON shape (matches the website's data file):
 *
 *   {
 *     "puzzles": [
 *       {"puzzle_n": 32,
 *        "private_key_hex": "00...3b9aca00",
 *        "solved": true},
 *       ...
 *     ]
 *   }
 *
 * Or the alternate shape:
 *
 *   {
 *     "solve_history": [
 *       {"puzzle_number": 32, "private_key": "...", ...},
 *       ...
 *     ]
 *   }
 *
 * @param path filesystem path to puzzle_history.json (or our data file)
 * @param targets_out output vector, sorted by puzzle_n ascending
 * @param error_out empty on success; description on failure
 * @return true on success, false on read/parse error
 */
bool load_puzzle_targets(const std::string& path,
                         std::vector<PuzzleTarget>& targets_out,
                         std::string& error_out);

// ---------------------------------------------------------------------------
// Orchestrator entry point (Phase 9)
// ---------------------------------------------------------------------------

struct OrchestratorOptions {
    std::string puzzle_keys_path;   // empty -> "./data/puzzle_history.json"
    std::string wordlist_path;      // empty -> stdin
    uint32_t scheme_mask = 0;       // 0 means "use stock"
    uint32_t addr_mask = 0;         // 0 means "puzzle-only short-circuit"
    std::string bloom_path;         // empty when addr_mask == 0
    std::string hits_out_path;      // empty -> stdout
    bool dry_run = false;           // run host setup, skip GPU dispatch
    bool show_summary = true;       // print loaded-targets summary
};

/**
 * Run the v2 puzzle-only / multi-scheme orchestrator end-to-end.
 *
 * Implementation status:
 *   * On Pro builds with COLLIDER_USE_CUDA: dispatches to v2_brain_wallet_batch.
 *   * On --dry-run: loads puzzle keys + prints the schemes/addr types
 *     that would be dispatched, then returns without touching the GPU.
 *   * On non-CUDA builds: prints a clear message and returns 64 (EX_USAGE).
 *
 * @return Unix-style exit code: 0 on success, non-zero on error.
 */
int run_v2_orchestrator(const OrchestratorOptions& opts);

// ---------------------------------------------------------------------------
// Phase 4 multi-address pipeline driver (host-side stitch).
//
// Given a batch of priv-key candidates (32 bytes each, big-endian, packed
// in `priv_batch`), runs:
//   1) ec_mul_batch (priv -> pub_x, pub_y) via secp256k1.cu
//   2) v2_multi_address_check (pub -> address fingerprints -> bloom probe)
//
// This is the integration point for Phase 4: callers (brain wallet
// generators, weak-PRNG enumerators, historical-key sweepers) all drop
// their priv-key batches here.
// ---------------------------------------------------------------------------

struct MultiAddressBatch {
    const uint8_t* priv_batch;   // host pointer, size = count * 32
    size_t         count;
    uint32_t       addr_mask;
    const uint8_t* bloom;        // host pointer to bloom filter bits (only
                                 // consulted by run_multi_address_batch's
                                 // one-shot wrapper; the session API below
                                 // takes the bloom once at init).
    uint64_t       bloom_bits;
    int            bloom_hashes;
    uint32_t       bloom_seed;   // MurmurHash3 seed the bloom was built with
};

// ---------------------------------------------------------------------------
// MultiAddressSession: bloom-and-buffer cache for high-throughput scanning.
//
// Bloom filters for funded-address sets are typically several megabytes;
// uploading them on every batch (the original run_multi_address_batch
// behavior) was the dominant cost on 2k-passphrase batches. The session
// object uploads the bloom ONCE in init(), allocates the reusable d_priv /
// d_pub_xy / d_matches / d_match_count buffers up to a configured peak
// batch size, and holds the CUDA stream. Each subsequent call to
// process_batch() just memcpy's the new privs and dispatches the kernels.
//
// Compatibility: the legacy `run_multi_address_batch(batch)` free function
// still works unchanged -- it now allocates a single-shot session
// internally, which gives the same behavior as before.
//
// Implementation lives in v2_orchestrator.cpp; the type is opaque here so
// the header doesn't need to drag in <cuda_runtime.h>.
// ---------------------------------------------------------------------------

class MultiAddressSession {
public:
    MultiAddressSession();
    ~MultiAddressSession();

    MultiAddressSession(const MultiAddressSession&) = delete;
    MultiAddressSession& operator=(const MultiAddressSession&) = delete;

    // One-time setup. Uploads the bloom (if non-null), allocates buffers
    // sized for `max_batch_count` privs. Returns 0 on success, non-zero
    // CUDA error code otherwise.
    int init(const uint8_t* bloom,
             uint64_t bloom_bits,
             int bloom_hashes,
             uint32_t bloom_seed,
             size_t max_batch_count);

    // Run one batch through the pipeline using the cached bloom + buffers.
    // `batch.priv_batch` and `batch.count` are required; `batch.bloom*`
    // fields are ignored (the session's cached bloom is used). Returns
    // 0 on success, sysexit-style codes (64 = bad-args, 70 = CUDA error)
    // otherwise.
    int process_batch(const MultiAddressBatch& batch);

    // Diagnostics for DP-buffer (V2_MAX_MATCHES_PER_BATCH) overflow. The
    // kernel atomically increments d_match_count for every (pp, addr_type,
    // kind) hit, but the records buffer is fixed-size. When more hits land
    // than the buffer can hold, the kernel still counts them but doesn't
    // record. process_batch() previously truncated silently; it now
    // accumulates the dropped count here so callers (and `--verbose`
    // operators) can detect under-sizing.
    uint64_t total_dropped_matches() const;
    uint64_t total_overflow_batches() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

// Legacy one-shot API. Returns 0 on success; non-zero on cuda error or
// invalid args. Internally constructs a single-batch session.
int run_multi_address_batch(const MultiAddressBatch& batch);

}  // namespace v2
}  // namespace gpu
}  // namespace collider
