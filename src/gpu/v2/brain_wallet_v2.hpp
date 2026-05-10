/**
 * Brain Wallet v2 GPU Pipeline — Public API
 *
 * Implements items 1, 2, 4, 5, 6, 9 from docs/BRAINWALLET-V2-SPEC.md.
 *
 * Stock pipeline (src/gpu/brain_wallet_gpu.hpp) checks one derivation
 * scheme (`priv = SHA256(pw)`) against one address type (uncompressed
 * P2PKH). v2 generalizes both, plus adds puzzle-mode and weak-PRNG
 * brute force.
 *
 * Status (2026-05-08): API frozen, kernels stubbed. Implementations
 * land in follow-up commits per the spec doc.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <string>

// CUDA types are pulled in only when this header is consumed by a CUDA TU
// or by host code that explicitly opts in (COLLIDER_USE_CUDA). The host
// orchestrator (v2_orchestrator.cpp / .hpp) needs the struct definitions
// (PuzzleTarget, V2MatchRecord, enums, masks, helper) but not the CUDA
// kernel signatures, so we shim the CUDA-specific types when the include
// isn't available. The kernel-launching APIs are themselves gated under
// COLLIDER_USE_CUDA below.
#if defined(COLLIDER_USE_CUDA) || defined(__CUDACC__)
#  include <cuda_runtime.h>
#  define COLLIDER_V2_HAVE_CUDA 1
#else
   // Provide enough type stand-ins for the prototype declarations to parse.
   typedef int cudaError_t;
   typedef struct CUstream_st* cudaStream_t;
#  define COLLIDER_V2_HAVE_CUDA 0
#endif

namespace collider {
namespace gpu {
namespace v2 {

// ---------------------------------------------------------------------------
// Item 1: derivation schemes
// ---------------------------------------------------------------------------
//
// Each value names a concrete recipe for `pw -> priv`.  The kernel evaluates
// every scheme whose bit is set in the per-launch `scheme_mask` argument.
// Add new schemes by appending here and extending the device-side switch.

enum class DerivationScheme : uint8_t {
    SHA256_PW           = 0,   // S1: priv = SHA256(pw)            -- stock parity
    SHA256_SHA256_PW    = 1,   // S2: priv = SHA256(SHA256(pw))
    SHA256_PW_NEWLINE   = 2,   // S3: priv = SHA256(pw || 0x0a)
    SHA256_PW_PW        = 3,   // S4: priv = SHA256(pw || pw)
    SHA256_SHA256_PW_PW = 4,   // S5: priv = SHA256(SHA256(pw) || pw)
    SHA256_ITER_16      = 5,   // S6: priv = SHA256^16(pw)
    HMAC_SHA512_PW      = 6,   // S7: priv = HMAC-SHA512(pw, "")[:32]
    SHA512_PW_HALF      = 7,   // S8: priv = SHA512(pw)[:32]
    // Reserved: 8..31
    SCHEME_COUNT = 8,
};

constexpr uint32_t scheme_bit(DerivationScheme s) { return 1u << static_cast<uint8_t>(s); }
constexpr uint32_t SCHEME_MASK_ALL = (1u << static_cast<uint8_t>(DerivationScheme::SCHEME_COUNT)) - 1u;
constexpr uint32_t SCHEME_MASK_STOCK = scheme_bit(DerivationScheme::SHA256_PW);

// ---------------------------------------------------------------------------
// Item 2: address type bitmask
// ---------------------------------------------------------------------------
//
// Per `priv` candidate, evaluate any subset of these address derivations.
// Each bit set adds ~one extra hash round + one bloom probe, much cheaper
// than re-running the EC multiply.

enum class AddressType : uint8_t {
    P2PKH_UNCOMPRESSED = 0,    // h160(SHA256(0x04 || X || Y))   -- stock parity
    P2PKH_COMPRESSED   = 1,    // h160(SHA256(02_or_03 || X))
    P2SH_P2WPKH        = 2,    // h160(SHA256(0x00 0x14 || compressed_h160))
    P2WPKH_V0          = 3,    // compressed_h160 used directly as witness program
    P2TR_BIP86         = 4,    // BIP-86 tweaked x-only output key
    // Reserved: 5..7
    ADDR_COUNT = 5,
};

constexpr uint32_t addr_bit(AddressType a) { return 1u << static_cast<uint8_t>(a); }
constexpr uint32_t ADDR_MASK_ALL = (1u << static_cast<uint8_t>(AddressType::ADDR_COUNT)) - 1u;
constexpr uint32_t ADDR_MASK_STOCK = addr_bit(AddressType::P2PKH_UNCOMPRESSED);
constexpr uint32_t ADDR_MASK_MODERN =
    addr_bit(AddressType::P2PKH_COMPRESSED) |
    addr_bit(AddressType::P2SH_P2WPKH) |
    addr_bit(AddressType::P2WPKH_V0) |
    addr_bit(AddressType::P2TR_BIP86);

// ---------------------------------------------------------------------------
// Item 6: puzzle target list (loaded into constant memory by set_puzzle_targets)
// ---------------------------------------------------------------------------

// PuzzleTarget stores the bottom (n-1) bits of a known puzzle's private key
// as a 4-limb little-endian-of-uint64 representation.  Limb 0 = bits 0..63,
// limb 1 = bits 64..127, limb 2 = bits 128..191, limb 3 = bits 192..255.
// We need 4 limbs to cover puzzles up to N=160, where (n-1) = 159 bits and
// the masked value can occupy bits 0..158.
//
// Layout uses NATURAL alignment with an explicit 4-byte padding word so the
// struct lays out the same on CUDA host, CUDA device, AND Metal Shading
// Language (MSL has no #pragma pack and aligns ulong to 8). Total is 72
// bytes; 160 entries * 72 = 11520 bytes fits comfortably in __constant__.
struct PuzzleTarget {
    uint16_t puzzle_n;            // bit-length of the puzzle (1..160)
    uint8_t  reserved[2];         // 2 bytes
    uint32_t _pad;                // 4-byte explicit pad to 8-byte align uint64s
    uint64_t low_mask[4];         // (1 << (n-1)) - 1 spread across 4 limbs
    uint64_t low_value[4];        // priv & low_mask, same layout
    // Total: 8 + 32 + 32 = 72 bytes.
};
static_assert(sizeof(PuzzleTarget) == 72, "PuzzleTarget must be 72 bytes");
static_assert(alignof(PuzzleTarget) == 8, "PuzzleTarget must be 8-byte aligned");

constexpr int PUZZLE_TARGET_MAX = 160;

// ---------------------------------------------------------------------------
// Item 4: weak-PRNG target families
// ---------------------------------------------------------------------------

enum class WeakPrngFamily : uint8_t {
    LIBBITCOIN_BX_SEED = 0,   // CVE-2023-39910 "Milk Sad" -- MT19937, time-seeded
    PROFANITY          = 1,   // CVE-2022-40769 -- MT19937
    TRUST_WALLET_EXT   = 2,   // CVE-2023-31290 -- MT19937, weak entropy
    GLIBC_RAND         = 3,   // glibc rand() -- Park-Miller LCG
    MSVC_RAND          = 4,   // MSVC rand() -- 32-bit LCG
    JAVA_RANDOM        = 5,   // java.util.Random -- 48-bit LCG
    // Reserved: 6..15
    FAMILY_COUNT = 6,
};

// ---------------------------------------------------------------------------
// Match record (returned via device buffer)
// ---------------------------------------------------------------------------

// Natural alignment: pp_idx[4] + 4 implicit pad + weak_seed[8] + puzzle_n[2]
// + scheme_id[1] + addr_type[1] + kind[1] + reserved[3] = 24, struct align 8.
struct V2MatchRecord {
    enum class Kind : uint8_t {
        ADDR_BLOOM_HIT = 0,    // h160 hit the bloom (item 2)
        PUZZLE_KEY_HIT = 1,    // priv matches a puzzle target (item 6)
        WEAK_PRNG_HIT  = 2,    // weak-PRNG candidate hit something (item 4)
    };

    uint32_t pp_idx;           // index into the host-side passphrase batch
    uint64_t weak_seed;        // for WEAK_PRNG_HIT: the seed; else 0
    uint16_t puzzle_n;         // for PUZZLE_KEY_HIT: which puzzle bit-length; else 0
    uint8_t  scheme_id;        // DerivationScheme value
    uint8_t  addr_type;        // AddressType value (for ADDR_BLOOM_HIT)
    uint8_t  kind;             // V2MatchRecord::Kind
    uint8_t  reserved[3];
    // Total 24 bytes (with implicit pad after pp_idx).
};
static_assert(sizeof(V2MatchRecord) == 24, "V2MatchRecord must be 24 bytes");
static_assert(alignof(V2MatchRecord) == 8, "V2MatchRecord must be 8-byte aligned");

constexpr int V2_MAX_MATCHES_PER_BATCH = 4096;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Initialize v2 pipeline: precompute EC table, allocate device match buffer.
 * Idempotent. Must be called before any v2 kernel.
 */
cudaError_t v2_init(cudaStream_t stream);

/**
 * Free v2 device-side state. Safe to call without prior init.
 */
void v2_shutdown();

/**
 * Item 6: install the puzzle target list into device constant memory.
 *
 * Pass an empty vector to disable puzzle-mode (kernel will skip the check).
 *
 * @param targets  At most PUZZLE_TARGET_MAX records, one per known puzzle key.
 *                 Must be sorted by puzzle_n ascending.
 */
cudaError_t v2_set_puzzle_targets(const std::vector<PuzzleTarget>& targets);

/**
 * Item 1+2+6 combined: run the multi-scheme, multi-address kernel against a
 * batch of passphrases. Optional puzzle check happens between SHA256 and
 * EC_MUL so puzzle-only mode skips the EC multiply entirely (huge speedup).
 *
 * @param d_passphrases   device pointer to packed passphrase bytes
 * @param d_offsets       device pointer to per-passphrase offset
 * @param d_lengths       device pointer to per-passphrase length
 * @param count           number of passphrases in batch
 * @param scheme_mask     bitmask of DerivationScheme values to evaluate
 * @param addr_mask       bitmask of AddressType values; 0 disables address bloom
 * @param d_bloom         device bloom filter (nullable when addr_mask == 0)
 * @param bloom_bits      bloom size in bits
 * @param bloom_hashes    bloom hash count (k)
 * @param d_matches       device output buffer (V2_MAX_MATCHES_PER_BATCH records)
 * @param d_match_count   device output counter (atomic)
 * @param stream          CUDA stream
 *
 * Throughput targets (Blackwell RTX 6000):
 *   - scheme_mask = 1, addr_mask = 1:  bit-identical to stock pipeline
 *   - scheme_mask = ALL (8), addr_mask = MODERN (4): ~6 % overhead vs stock
 *     for the addr-mask expansion; per-scheme cost is ~1x EC_MUL each, so 8x
 *     wall time but 8x breadth.
 *   - scheme_mask = ALL, puzzle-only (addr_mask = 0): ~20x faster than stock
 *     because every scheme short-circuits before EC_MUL when no puzzle hit.
 */
cudaError_t v2_brain_wallet_batch(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    size_t count,
    uint32_t scheme_mask,
    uint32_t addr_mask,
    const uint8_t* d_bloom,
    uint64_t bloom_bits,
    int bloom_hashes,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream
);

/**
 * Item 4: weak-PRNG seed brute force. For each seed in [seed_lo, seed_hi),
 * derive priv per the named PRNG family, run the same item-2 multi-address
 * check, and write hits to the match buffer.
 *
 * Independent of the brain-wallet path — different keyspace entirely.
 *
 * @param family       which weak PRNG to brute (see WeakPrngFamily)
 * @param seed_lo,hi   half-open [lo, hi) range of 64-bit seeds to enumerate
 * @param addr_mask    address types to derive per priv (item 2 reuse)
 * @param ...          (other args mirror v2_brain_wallet_batch)
 *
 * For 32-bit seeded targets (libbitcoin/Profanity/Trust Wallet/MSVC),
 * the upper bits of seed_lo/seed_hi are ignored.
 *
 * Throughput target: ~10^8 seeds/sec sustained on Blackwell (the EC_MUL
 * still dominates).
 */
cudaError_t v2_weak_prng_brute(
    WeakPrngFamily family,
    uint64_t seed_lo,
    uint64_t seed_hi,
    uint32_t addr_mask,
    const uint8_t* d_bloom,
    uint64_t bloom_bits,
    int bloom_hashes,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream
);

/**
 * Item 5: BIP-39 mnemonic mode. Decode each mnemonic (already validated for
 * checksum on host side), run PBKDF2-HMAC-SHA512(salt="mnemonic"+passphrase,
 * 2048 iterations) to seed, derive BIP-32 master, then evaluate fixed paths.
 *
 * @param d_mnemonics   packed mnemonic UTF-8 bytes (validated on host)
 * @param d_offsets     start of each mnemonic
 * @param d_lengths     length of each mnemonic
 * @param d_passphrase  optional BIP-39 passphrase ("" if none)
 * @param paths         BIP-32 derivation paths to evaluate per seed; e.g.
 *                      {"m/44'/0'/0'/0/0", "m/49'/0'/0'/0/0",
 *                       "m/84'/0'/0'/0/0", "m/86'/0'/0'/0/0"}
 *                      Each path is sent to the kernel as a parsed
 *                      sequence of 32-bit child indices.
 *
 * Throughput target: ~10^6 mnemonics/sec on Blackwell, dominated by the
 * 2048 PBKDF2 iterations.
 */
cudaError_t v2_bip39_brute(
    const uint8_t* d_mnemonics,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    size_t count,
    const std::string& bip39_passphrase,
    const std::vector<std::vector<uint32_t>>& parsed_paths,
    uint32_t addr_mask,
    const uint8_t* d_bloom,
    uint64_t bloom_bits,
    int bloom_hashes,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream
);

/**
 * Phase 6: encoding-anomaly + double-hash brute (CPU reference at
 * src/gpu/v2/encoding_munge_cpu.hpp). For each passphrase, mutate it
 * under each enabled encoding bit, optionally apply SHA-256d, and run
 * the puzzle-target mask compare.
 *
 * encoding_mask uses the layout from encoding_munge_kernel.cu's
 * DevEncoding enum (UTF-8=bit 0, UTF-16-LE=bit 1, ..., LOWER_ASCII=bit 8,
 * NULL_TERMINATED=bit 9).
 */
cudaError_t v2_encoding_brute(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    size_t count,
    uint32_t encoding_mask,
    bool double_hash,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream
);

/**
 * Phase 7: legacy KDF brute. The kdf_id selects from a registry of
 * obsolete KDFs (early web-wallet generators that combined MD5/SHA-1
 * iterations with static salts). New KDFs are registered host-side via
 * v2_register_legacy_kdf and dispatched on the device by id.
 *
 * See src/gpu/v2/legacy_kdf_kernel.cu for the dispatch table.
 */
cudaError_t v2_legacy_kdf_brute(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    size_t count,
    uint8_t kdf_id,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream
);

/**
 * Phase 4: multi-address bloom probe.
 * d_pub_xy is a packed array of (X, Y) per point, 64 bytes per point,
 * matching ECPointAffine produced by secp256k1_batch_mul. Caller computes
 * it from the priv batch via secp256k1.cu's ec_mul kernel, then dispatches
 * this for the address derivation + bloom probe step. Per the v1.4.0 spec,
 * supports four h160-keyed address types via addr_mask (P2PKH uncomp/comp,
 * P2SH-P2WPKH, P2WPKH-V0). P2TR-BIP86 is recognized in the mask but its
 * bloom probe is deferred until the kernel grows EC tweak-add support.
 *
 * Bloom probe uses MurmurHash3 to match h160_bloom_filter.cu (the same
 * code path the existing brain_wallet_gpu pipeline reads from). The
 * `bloom_seed` parameter MUST equal the seed the bloom file was built
 * with.
 */
cudaError_t v2_multi_address_check(
    const uint8_t* d_pub_xy,
    size_t count, uint32_t addr_mask,
    const uint8_t* d_bloom, uint64_t bloom_bits, int bloom_hashes,
    uint32_t bloom_seed,
    V2MatchRecord* d_matches, uint32_t* d_match_count,
    cudaStream_t stream
);

/**
 * Phase 8: Electrum v1 hex-seed 100k SHA-256 stretch.
 * One thread per passphrase; passphrase BYTES are the v1 raw hex seed
 * (the user-display 12 words decode to this on the host side).
 */
cudaError_t v2_electrum_v1_brute(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    size_t count,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream
);

/**
 * Phase 8: Electrum v2 mnemonic seed via PBKDF2-HMAC-SHA512(2048).
 * Salt = "electrum" + passphrase (host-supplied, applied to all entries).
 */
cudaError_t v2_electrum_v2_brute(
    const uint8_t* d_passphrases,
    const uint32_t* d_offsets,
    const uint32_t* d_lengths,
    size_t count,
    const std::string& bip39_passphrase,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream
);

// ---------------------------------------------------------------------------
// Helpers (host-side, header-only)
// ---------------------------------------------------------------------------

/**
 * Build a PuzzleTarget from a (puzzle_n, full private_key) pair.
 *
 * The mask retains the low (n-1) bits; the high bit (bit n-1) is implicit
 * in the puzzle's bit-length and not stored. priv_be is 32 bytes big-endian.
 *
 * The 4-limb mask layout: limb i covers bit positions [64*i, 64*(i+1)).
 * For puzzle N with low_bits = N-1, full limbs (those entirely within
 * low_bits) are all-ones; the limb containing bit (low_bits-1) is partial;
 * limbs entirely above are zero.
 */
inline PuzzleTarget make_puzzle_target(uint16_t puzzle_n, const uint8_t priv_be[32]) {
    PuzzleTarget t{};
    t.puzzle_n = puzzle_n;
    // puzzle_n == 0 is undefined; puzzle_n == 1 has zero "low bits" to
    // compare. In both cases the mask compare would produce a vacuous
    // match against EVERY input. Set a sentinel low_mask that NO finite
    // hash output can satisfy (mask=0 but value != 0) so the kernel
    // produces zero matches against this target. The puzzle_n field is
    // still preserved so the host can introspect.
    if (puzzle_n <= 1) {
        for (int i = 0; i < 4; ++i) {
            t.low_mask[i]  = 0;
            // Non-zero value with all-zero mask is impossible to match:
            // (any & 0) == 0, never == 1.
            t.low_value[i] = 1;
        }
        return t;
    }

    const int low_bits = puzzle_n - 1;

    // Build the 4-limb mask: full limbs in low_bits/64, partial limb at the boundary.
    const int full_limbs = low_bits / 64;
    const int partial_bits = low_bits % 64;
    for (int i = 0; i < 4; i++) {
        if (i < full_limbs) {
            t.low_mask[i] = ~0ull;
        } else if (i == full_limbs && partial_bits > 0) {
            t.low_mask[i] = (1ull << partial_bits) - 1ull;
        } else {
            t.low_mask[i] = 0ull;
        }
    }

    // Pack priv_be (32 bytes BE) into 4 little-endian-by-limb uint64s.
    // priv_be[24..31] = limb 0 (lowest 64 bits)
    // priv_be[16..23] = limb 1
    // priv_be[ 8..15] = limb 2
    // priv_be[ 0.. 7] = limb 3 (highest 64 bits)
    auto pack_be64 = [](const uint8_t* b) -> uint64_t {
        uint64_t x = 0;
        for (int i = 0; i < 8; i++) x = (x << 8) | b[i];
        return x;
    };
    const uint64_t priv_limb[4] = {
        pack_be64(priv_be + 24),
        pack_be64(priv_be + 16),
        pack_be64(priv_be +  8),
        pack_be64(priv_be +  0),
    };
    for (int i = 0; i < 4; i++) {
        t.low_value[i] = priv_limb[i] & t.low_mask[i];
    }
    return t;
}

}  // namespace v2
}  // namespace gpu
}  // namespace collider
