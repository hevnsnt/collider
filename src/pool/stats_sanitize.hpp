/**
 * stats_sanitize.hpp -- v1.4.2 B.6
 *
 * Float-from-wire sanitization helper for STATS_RSP. Pre-fix the
 * client read float fields from the wire and immediately did
 * `static_cast<uint64_t>(f)`, which is undefined behaviour for NaN,
 * +/-Inf, or values outside [INT_MIN, UINT64_MAX]. A buggy or hostile
 * server could trigger UB on every stats refresh.
 *
 * Factored out so unit tests can drive it directly with adversarial
 * inputs (NaN, +Inf, -Inf, -1.0, 1e30) without standing up a TLS
 * connection.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

namespace collider {
namespace pool {

// Clamps `v` to [lo, hi]. NaN and +/-Inf are normalized to 0.0f.
// Returns the new (sanitized) value.
inline float sanitize_wire_float(float v, float lo, float hi) {
    if (!std::isfinite(v)) return 0.0f;
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// Bounds appropriate for STATS_RSP fields. dps_per_second can scale to
// the pool aggregate (currently ~10s of millions/sec; 1e18 leaves headroom
// without overflowing the uint64 cast site). your_share is a probability
// fraction in [0, 1].
inline void sanitize_stats_rsp_floats(float& dps_per_second, float& your_share) {
    dps_per_second = sanitize_wire_float(dps_per_second, 0.0f, 1.0e18f);
    your_share     = sanitize_wire_float(your_share,     0.0f, 1.0f);
}

// uint64 wire fields land in UI consumers (status panel, console
// printout, JSON exports) without further validation. A hostile or
// buggy server could feed values that overflow downstream arithmetic
// (e.g. multiplied by a divisor when computing per-second rates) or
// just look absurd in the UI. The cap below is generous: at 1 trillion
// DPs across the whole pool, the panel still has six-decimal headroom
// for human-readable formatting.
inline uint64_t sanitize_wire_u64(uint64_t v, uint64_t cap) {
    return v > cap ? cap : v;
}

inline void sanitize_stats_rsp_uints(uint64_t& your_dps,
                                     uint64_t& total_dps) {
    // Cap below UINT64_MAX so any downstream signed-int interpretation
    // (e.g. int64_t-based JSON serializers) or x2 arithmetic
    // (rate-of-change diff) cannot overflow. The cap is high enough
    // that a long-lived pool aggregating trillions of DPs across all
    // workers still round-trips losslessly.
    constexpr uint64_t kSanityCap =
        std::numeric_limits<uint64_t>::max() / 2;
    your_dps  = sanitize_wire_u64(your_dps,  kSanityCap);
    total_dps = sanitize_wire_u64(total_dps, kSanityCap);
}

}  // namespace pool
}  // namespace collider
