/**
 * Weak-PRNG CPU references (Phase 5, restructure plan v1.4.0).
 *
 * Header-only implementations of every PRNG family the brain wallet
 * v2 weak-PRNG kernel cares about. Each function takes a seed and
 * advances the PRNG by one 32-bit (or 64-bit, family-dependent)
 * output. The brain-wallet pipeline then runs SHA-256(prng_output_bytes)
 * to obtain a candidate priv key, and runs the existing address-mask
 * + bloom check.
 *
 * The CPU references here are the AUTHORITATIVE source. The eventual
 * GPU kernel must produce byte-for-byte identical output sequences.
 *
 * Why these PRNGs are interesting:
 *
 *   * MT19937             : the underlying PRNG behind libbitcoin bx,
 *     Profanity (vanity-address tool), and several Trust Wallet
 *     Extension generations. Time-seeded variants (CVE-2023-39910
 *     "Milk Sad", CVE-2022-40769) are the headline vulnerabilities.
 *   * glibc rand()        : Park-Miller / extended LCG. Apparent
 *     randomness, but seedable from a 32-bit value.
 *   * MSVC rand()         : 32-bit LCG with documented constants.
 *     Output is only 15 bits per call; brute force trivial.
 *   * java.util.Random    : 48-bit LCG. Spec-stable across JVM
 *     versions; recoverable from any two consecutive nextInt() calls.
 *
 * NB: These are AUDIT references for the v2 historical-key sweep.
 * Do not adapt them as primary RNGs anywhere -- pool sampling uses
 * `secrets.SystemRandom` (Phase 2). See src/work_manager.py.
 */

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace collider {
namespace gpu {
namespace v2 {
namespace prng {

// ---------------------------------------------------------------------------
// MT19937 (Mersenne Twister 32-bit)
// Reference: Matsumoto-Nishimura 1998. Output of `next()` is the i-th
// 32-bit value emitted after `seed_mt(s)`.
// ---------------------------------------------------------------------------

struct Mt19937 {
    uint32_t mt[624];
    int      idx;

    void seed(uint32_t s) {
        mt[0] = s;
        for (int i = 1; i < 624; ++i) {
            mt[i] = 1812433253u * (mt[i-1] ^ (mt[i-1] >> 30)) + static_cast<uint32_t>(i);
        }
        idx = 624;
    }

    uint32_t next() {
        if (idx >= 624) generate();
        uint32_t y = mt[idx++];
        y ^= (y >> 11);
        y ^= (y <<  7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= (y >> 18);
        return y;
    }

private:
    void generate() {
        for (int i = 0; i < 624; ++i) {
            uint32_t y = (mt[i] & 0x80000000u) | (mt[(i + 1) % 624] & 0x7fffffffu);
            mt[i] = mt[(i + 397) % 624] ^ (y >> 1);
            if (y & 1u) mt[i] ^= 0x9908b0dfu;
        }
        idx = 0;
    }
};

// ---------------------------------------------------------------------------
// Park-Miller 32-bit LCG (glibc rand_r style).
//
// Note: glibc's actual rand() is a more elaborate 31-bit additive feedback
// generator; rand_r() and many embedded glibc-derived RNGs use this simpler
// form. We expose the Park-Miller variant because it's what most weak
// brain-wallet generators ship.
// ---------------------------------------------------------------------------

struct GlibcRandR {
    uint32_t state;
    void seed(uint32_t s) { state = s ? s : 1u; }
    uint32_t next() {
        // x_{n+1} = x_n * 1103515245 + 12345 mod 2^31
        state = (state * 1103515245u + 12345u) & 0x7fffffffu;
        return state;
    }
};

// ---------------------------------------------------------------------------
// MSVC rand(): 32-bit LCG, output is bits 30..16 of the state.
//   state_{n+1} = state_n * 214013 + 2531011
//   rand()     = (state_{n+1} >> 16) & 0x7fff
// ---------------------------------------------------------------------------

struct MsvcRand {
    uint32_t state;
    void seed(uint32_t s) { state = s; }
    uint32_t next() {
        state = state * 214013u + 2531011u;
        return (state >> 16) & 0x7fffu;
    }
};

// ---------------------------------------------------------------------------
// java.util.Random: 48-bit LCG.
//   seed' = (seed ^ 0x5deece66dL) & ((1L<<48)-1)   -- Java seed scrambling
//   state_{n+1} = (state_n * 0x5deece66dL + 0xbL) & ((1L<<48)-1)
//   nextInt()   = top 32 bits of state
// ---------------------------------------------------------------------------

struct JavaRandom {
    uint64_t state;

    void seed(int64_t java_seed) {
        // Apply Java's seed scrambling exactly as java.util.Random does.
        state = (static_cast<uint64_t>(java_seed) ^ 0x5deece66dULL) &
                ((1ULL << 48) - 1ULL);
    }

    int32_t next_bits(int bits) {
        state = (state * 0x5deece66dULL + 0xbULL) & ((1ULL << 48) - 1ULL);
        return static_cast<int32_t>(state >> (48 - bits));
    }

    int32_t next_int()  { return next_bits(32); }
    int64_t next_long() {
        // Java's long is the concatenation of two next(32)s, treated signed.
        int64_t hi = static_cast<int64_t>(next_bits(32));
        int64_t lo = static_cast<int64_t>(next_bits(32));
        return (hi << 32) + lo;
    }
};

// ---------------------------------------------------------------------------
// libbitcoin bx seed (CVE-2023-39910 "Milk Sad")
//
// The vulnerable code does roughly:
//   std::mt19937 gen(time(NULL));
//   for (i=0; i<32; i++) seed[i] = gen() & 0xff;
//
// So the 32 priv-bytes are the LOW byte of 32 consecutive mt19937
// outputs after seeding with time(NULL). 32-bit unix time is the only
// entropy.
// ---------------------------------------------------------------------------

inline void libbitcoin_bx_seed(uint32_t unix_time, uint8_t out_priv[32]) {
    Mt19937 m{};
    m.seed(unix_time);
    for (int i = 0; i < 32; ++i) {
        out_priv[i] = static_cast<uint8_t>(m.next() & 0xffu);
    }
}

// ---------------------------------------------------------------------------
// Profanity (CVE-2022-40769): vanity-address generator that derived an
// initial private key from a 32-bit seed via mt19937, then walked it via
// scalar additions. The vulnerable construction is:
//
//   std::mt19937 gen(seed);
//   uint8_t priv[32];
//   for (i=0; i<32; i++) priv[i] = gen() & 0xff;
//
// (Identical to libbitcoin_bx for the priv-key-derivation step; the
// difference is what's done with priv afterward.)
// ---------------------------------------------------------------------------

inline void profanity_seed(uint32_t seed, uint8_t out_priv[32]) {
    libbitcoin_bx_seed(seed, out_priv);  // Same construction.
}

// ---------------------------------------------------------------------------
// Trust Wallet Extension (CVE-2023-31290): derives a 12-word mnemonic
// from a 32-bit seed via mt19937. The derivation differs (BIP-39
// path), but the seed entropy is still 32 bits. We expose the raw
// 32-byte output as a convenience -- callers run the same SHA-256(...)
// step the BIP-39 path uses on the entropy bytes.
// ---------------------------------------------------------------------------

inline void trust_wallet_ext_entropy(uint32_t seed, uint8_t out[16]) {
    Mt19937 m{};
    m.seed(seed);
    for (int i = 0; i < 16; ++i) {
        out[i] = static_cast<uint8_t>(m.next() & 0xffu);
    }
}

}  // namespace prng
}  // namespace v2
}  // namespace gpu
}  // namespace collider
