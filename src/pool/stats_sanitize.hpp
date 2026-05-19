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

}  // namespace pool
}  // namespace collider
