/**
 * kangaroo_dp_table.hpp -- v1.4.2 A.4
 *
 * Distinguished-Point table key for Pollard's kangaroo. Pre-fix the
 * dp_table was a std::unordered_map<uint64_t, DPEntry> keyed only on
 * the low 64 bits of x. At scale (>2^32 DPs), x-low hash collisions
 * silently discarded second-comers - a real DP collision could be lost
 * if a prior x_low-colliding DP existed in the table. Worse, the
 * "x_matches but same type" fallthrough also discarded DPs even when
 * the second-comer was at a genuinely-different x.
 *
 * The full-256-bit key removes both failure modes. Tests:
 * tests/test_dp_table_full256.cpp.
 */

#pragma once

#include <array>
#include <cstdint>
#include <functional>

namespace collider {
namespace kangaroo {

// Full 256-bit Distinguished-Point key (X coordinate, 4 limbs little-endian).
struct DPKey {
    uint64_t x[4];

    bool operator==(const DPKey& o) const {
        return x[0] == o.x[0] && x[1] == o.x[1] && x[2] == o.x[2] && x[3] == o.x[3];
    }
};

}  // namespace kangaroo
}  // namespace collider

namespace std {

template <>
struct hash<collider::kangaroo::DPKey> {
    size_t operator()(const collider::kangaroo::DPKey& k) const noexcept {
        // FxHash-style mixing: cheap, well-distributed across uint64 keys.
        // Two 256-bit values with identical low 64 bits will reliably
        // produce different hashes because the upper limbs participate.
        size_t h = 0;
        for (int i = 0; i < 4; ++i) {
            h ^= static_cast<size_t>(k.x[i]) + 0x9E3779B97F4A7C15ULL +
                 (h << 12) + (h >> 4);
        }
        return h;
    }
};

}  // namespace std
