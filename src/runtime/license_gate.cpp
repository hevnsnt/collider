/**
 * license_gate.cpp - PRO-only license activation + startup validation
 * implementation.
 *
 * Extracted verbatim from src/main.cpp during the v1.4.1 A.3 (6/6)
 * refactor; no behavior changes. Compiled only when COLLIDER_PRO is
 * defined; the file is excluded from CMake's source list in the free
 * build.
 */
#ifdef COLLIDER_PRO

#include "runtime/license_gate.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "license/license_check.hpp"

namespace collider::runtime {

bool process_activate_flag(int argc, char* argv[], int& out_exit_code) {
    // --activate KEY: save and validate a new license key.
    // Usage: ./collider --activate CLLDR-XXXX-XXXX-XXXX-XXXX
    if (argc < 3 || std::string(argv[1]) != "--activate") {
        return false;  // not handled here, caller continues.
    }

    std::string key = argv[2];
    std::cout << "[LICENSE] Validating " << key << " ...\n";
    auto result = collider::license::validate(key);
    if (!result.valid) {
        std::cerr << "[LICENSE] Invalid key: "
                  << (result.error.empty() ? "key not found" : result.error) << "\n"
                  << "          Purchase at: https://collisionprotocol.com/pro\n";
        out_exit_code = 1;
        return true;
    }
    // Persist the key
    namespace fs = std::filesystem;
    std::string kpath = collider::license::cache_path();
    // store key next to cache as license_key file
    std::string kfile = fs::path(kpath).parent_path().string() + "/license_key";
    try { fs::create_directories(fs::path(kfile).parent_path()); } catch (...) {}
    std::ofstream kf(kfile, std::ios::trunc);
    if (kf) kf << key << "\n";
    std::cout << "[LICENSE] Activated for " << result.email << ". Enjoy theCollider Pro!\n";
    out_exit_code = 0;
    return true;
}

bool validate_startup_license(int& out_exit_code) {
    // Startup license check: read stored key and validate (cache-backed, 24h).
    namespace fs = std::filesystem;
    std::string kpath = collider::license::cache_path();
    std::string kfile = fs::path(kpath).parent_path().string() + "/license_key";
    std::string stored_key;
    {
        std::ifstream kf(kfile);
        if (kf) std::getline(kf, stored_key);
    }
    if (stored_key.empty()) {
        std::cerr << "[LICENSE] No license key found.\n"
                  << "          Run: ./collider --activate YOUR_KEY\n"
                  << "          Purchase at: https://collisionprotocol.com/pro\n";
        out_exit_code = 1;
        return false;
    }
    auto result = collider::license::validate(stored_key);
    if (!result.valid) {
        std::cerr << "[LICENSE] License invalid or expired.\n"
                  << "          Key: " << stored_key << "\n"
                  << "          Error: " << result.error << "\n"
                  << "          Visit: https://collisionprotocol.com/pro\n";
        out_exit_code = 1;
        return false;
    }
    out_exit_code = 0;
    return true;
}

}  // namespace collider::runtime

#endif  // COLLIDER_PRO
