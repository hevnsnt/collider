// phase 4 (builder-threading: runtime-yaml).
//
// Round-trip + edge-case coverage for collider::runtime::RuntimeConfig
// load_runtime_config / save_runtime_config:
//
//   1. Round-trip: build a fully-populated RuntimeConfig, save, load, assert
//      every field round-trips exactly.
//   2. Missing-file: load against a path that does not exist returns
//      std::nullopt (no crash, no error log to test output that we care
//      about beyond "no exception").
//   3. Malformed: write a file full of garbage; load returns std::nullopt
//      and does not crash.
//   4. Partial: write a file with only one valid key; load returns a
//      RuntimeConfig with that field set and all other fields std::nullopt.
//   5. Save with all fields unset: file is created but contains only the
//      header comments. Load against that file returns std::nullopt
//      (parsed_count == 0).
//
// Uses set_runtime_config_path_for_testing to direct load/save at a
// dedicated tempdir-scoped path so the user's real ~/.collider/runtime.yml
// is never touched.

#include "runtime/runtime_config_yaml.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>

namespace {

std::filesystem::path make_temp_dir() {
    auto base = std::filesystem::temp_directory_path();
    std::random_device rd;
    std::mt19937_64 gen(rd());
    for (int attempt = 0; attempt < 32; ++attempt) {
        auto candidate = base / ("collider_runtime_yaml_test_" + std::to_string(gen()));
        std::error_code ec;
        if (std::filesystem::create_directory(candidate, ec) && !ec) {
            return candidate;
        }
    }
    // Fall back to a deterministic path; if it already exists, reuse it.
    auto candidate = base / "collider_runtime_yaml_test_fallback";
    std::error_code ec;
    std::filesystem::create_directories(candidate, ec);
    return candidate;
}

struct Failures {
    int count = 0;
    void check(bool cond, const char* what) {
        if (!cond) {
            std::cerr << "[FAIL] " << what << "\n";
            ++count;
        }
    }
};

void write_file(const std::filesystem::path& p, const std::string& contents) {
    std::ofstream out(p, std::ios::trunc);
    out << contents;
}

}  // namespace

int main() {
    using collider::runtime::RuntimeConfig;
    using collider::runtime::load_runtime_config;
    using collider::runtime::save_runtime_config;
    using collider::runtime::set_runtime_config_path_for_testing;

    std::cout << "test_runtime_yaml (phase 4 builder-threading)\n";

    Failures f;
    const auto tmp = make_temp_dir();
    const auto path = tmp / "runtime.yml";

    // ===== 1. Round-trip ===============================================
    set_runtime_config_path_for_testing(path.string());
    {
        RuntimeConfig cfg;
        cfg.batch_size = 6'000'000ull;
        cfg.rule_chunk_size = 500ull;
        cfg.bloom_path = std::string("D:/data/funded.blf");
        cfg.theme_variant = 1;
        cfg.gpu_enable_mask = static_cast<uint8_t>(0x03);

        f.check(save_runtime_config(cfg), "save_runtime_config returned true");
        f.check(std::filesystem::exists(path), "runtime.yml exists after save");

        auto loaded = load_runtime_config();
        f.check(loaded.has_value(), "load_runtime_config returned a value after round-trip");
        if (loaded.has_value()) {
            f.check(loaded->batch_size.has_value() &&
                    *loaded->batch_size == 6'000'000ull,
                    "round-trip batch_size matches");
            f.check(loaded->rule_chunk_size.has_value() &&
                    *loaded->rule_chunk_size == 500ull,
                    "round-trip rule_chunk_size matches");
            f.check(loaded->bloom_path.has_value() &&
                    *loaded->bloom_path == "D:/data/funded.blf",
                    "round-trip bloom_path matches");
            f.check(loaded->theme_variant.has_value() &&
                    *loaded->theme_variant == 1,
                    "round-trip theme_variant matches");
            f.check(loaded->gpu_enable_mask.has_value() &&
                    *loaded->gpu_enable_mask == 0x03,
                    "round-trip gpu_enable_mask matches");
        }
    }

    // ===== 2. Missing file =============================================
    {
        const auto missing = tmp / "does_not_exist.yml";
        set_runtime_config_path_for_testing(missing.string());
        auto loaded = load_runtime_config();
        f.check(!loaded.has_value(),
                "load_runtime_config returns nullopt for missing file");
        // The file MUST NOT have been created as a side effect of load.
        f.check(!std::filesystem::exists(missing),
                "load_runtime_config did not create the missing file");
    }

    // ===== 3. Malformed ================================================
    {
        const auto bad = tmp / "malformed.yml";
        set_runtime_config_path_for_testing(bad.string());
        // No ':' anywhere; every line is a bare token. Plus a 'batch_size: notanumber'
        // case that should be caught by the per-field validator.
        write_file(bad,
            "this is not yaml\n"
            "nor is this\n"
            "batch_size: notanumber\n"
            "theme_variant: 99\n"
            "gpu_enable_mask: deadbeef\n"
        );
        auto loaded = load_runtime_config();
        f.check(!loaded.has_value(),
                "load_runtime_config returns nullopt for fully-malformed file");
    }

    // ===== 4. Partial ==================================================
    {
        const auto partial = tmp / "partial.yml";
        set_runtime_config_path_for_testing(partial.string());
        write_file(partial, "batch_size: 4000000\n");
        auto loaded = load_runtime_config();
        f.check(loaded.has_value(),
                "load_runtime_config returns a value for partial file");
        if (loaded.has_value()) {
            f.check(loaded->batch_size.has_value() &&
                    *loaded->batch_size == 4'000'000ull,
                    "partial batch_size loaded");
            f.check(!loaded->rule_chunk_size.has_value(),
                    "partial rule_chunk_size is nullopt");
            f.check(!loaded->bloom_path.has_value(),
                    "partial bloom_path is nullopt");
            f.check(!loaded->theme_variant.has_value(),
                    "partial theme_variant is nullopt");
            f.check(!loaded->gpu_enable_mask.has_value(),
                    "partial gpu_enable_mask is nullopt");
        }
    }

    // ===== 5. Empty save ===============================================
    {
        const auto empty_path = tmp / "empty.yml";
        set_runtime_config_path_for_testing(empty_path.string());
        RuntimeConfig empty;
        f.check(save_runtime_config(empty),
                "save_runtime_config returns true for an all-nullopt config");
        f.check(std::filesystem::exists(empty_path),
                "all-nullopt save still creates the file (header comments only)");
        auto loaded = load_runtime_config();
        f.check(!loaded.has_value(),
                "load against header-only file returns nullopt");
    }

    // ===== 6. Hex mask round-trip ======================================
    {
        const auto hexp = tmp / "hex.yml";
        set_runtime_config_path_for_testing(hexp.string());
        RuntimeConfig cfg;
        cfg.gpu_enable_mask = static_cast<uint8_t>(0xff);
        f.check(save_runtime_config(cfg), "save hex mask");
        auto loaded = load_runtime_config();
        f.check(loaded.has_value() && loaded->gpu_enable_mask.has_value() &&
                *loaded->gpu_enable_mask == 0xff,
                "hex mask round-trip 0xff");
    }

    // Cleanup the tempdir. Best-effort; failures here are not test
    // failures.
    set_runtime_config_path_for_testing("");
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);

    if (f.count == 0) {
        std::cout << "PASS\n";
        return 0;
    }
    std::cout << "FAIL (" << f.count << " failures)\n";
    return 1;
}
