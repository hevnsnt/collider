/**
 * edition.hpp - theCollider Feature Edition Control
 * 
 * Controls which features are available based on build configuration.
 * Free edition supports pool mining with any pool.
 * Pro edition requires a valid license for all features.
 */

#pragma once

// ============================================================================
// Version Information
// ============================================================================
#define COLLIDER_VERSION "1.0.0"
#define COLLIDER_MAJOR_VERSION 1

// ============================================================================
// Edition Feature Gates
// ============================================================================

    // --- Free Edition Features ---
    #define COLLIDER_HAS_SOLO           0
    #define COLLIDER_HAS_BRAINWALLET    0
    #define COLLIDER_HAS_BLOOM          0
    #define COLLIDER_HAS_GENERATORS     0
    #define COLLIDER_HAS_RULES          0
    #define COLLIDER_HAS_SCRAPERS       0
    #define COLLIDER_EDITION_NAME       "collider"

// ============================================================================
// Feature Check Macros
// ============================================================================

#define COLLIDER_REQUIRE_PRO(feature_name) \
    do { \
        if (!COLLIDER_HAS_##feature_name) { \
            std::cerr << "[*] " #feature_name " requires collider pro — collisionprotocol.com/pro\n"; \
            return 1; \
        } \
    } while(0)

#define COLLIDER_FEATURE_AVAILABLE(feature_name) (COLLIDER_HAS_##feature_name)

// ============================================================================
// Edition Information
// ============================================================================

namespace collider {
namespace edition {

/**
 * Check if we're running the Pro edition
 */
constexpr bool is_pro() {
    return false;
}

/**
 * Check if we're running the Free edition  
 */
constexpr bool is_free() {
    return !is_pro();
}

/**
 * Get edition name
 */
constexpr const char* name() {
    return COLLIDER_EDITION_NAME;
}

/**
 * Get version string
 */
constexpr const char* version() {
    return COLLIDER_VERSION;
}

/**
 * Get major version
 */
constexpr int major_version() {
    return COLLIDER_MAJOR_VERSION;
}

/**
 * Check if a specific feature is available
 */
constexpr bool has_solo() { return COLLIDER_HAS_SOLO; }
constexpr bool has_brainwallet() { return COLLIDER_HAS_BRAINWALLET; }
constexpr bool has_bloom() { return COLLIDER_HAS_BLOOM; }
constexpr bool has_generators() { return COLLIDER_HAS_GENERATORS; }
constexpr bool has_rules() { return COLLIDER_HAS_RULES; }
constexpr bool has_scrapers() { return COLLIDER_HAS_SCRAPERS; }

} // namespace edition
} // namespace collider