// Shared text-formatting helpers used by puzzle and brain-wallet runners.
// Header-only (small inline bodies); no .cpp companion.
#pragma once

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace collider::runtime {

// Format large counts with human-readable suffixes (K, M, B, T) at one
// decimal place. Values under 1000 are emitted as plain integers.
inline std::string format_number_human(uint64_t n) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    if (n >= 1000000000000ULL) {
        oss << (static_cast<double>(n) / 1e12) << "T";
    } else if (n >= 1000000000ULL) {
        oss << (static_cast<double>(n) / 1e9) << "B";
    } else if (n >= 1000000ULL) {
        oss << (static_cast<double>(n) / 1e6) << "M";
    } else if (n >= 1000ULL) {
        oss << (static_cast<double>(n) / 1e3) << "K";
    } else {
        oss << n;
    }
    return oss.str();
}

// Normalize path separators to the platform-native form (backslash on
// Windows, forward slash elsewhere) for consistent display.
inline std::string normalize_path(const std::string& path) {
    std::string result = path;
#ifdef _WIN32
    std::replace(result.begin(), result.end(), '/', '\\');
#else
    std::replace(result.begin(), result.end(), '\\', '/');
#endif
    return result;
}

}  // namespace collider::runtime
