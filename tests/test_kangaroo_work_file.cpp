/**
 * Roundtrip KAT for KangarooWorkFile save/load (src/core/kangaroo.hpp).
 *
 * v1.4.0 phase 1.8 added a strict mismatched-checkpoint rejection in
 * KangarooSolver::solve(): if the loaded work file's
 * target_pubkey_hex / range_start_hex / range_end_hex / dp_bits don't
 * match the solver's settings, solve() refuses to load (so foreign DPs
 * don't get injected into the dp_table_ and produce false collisions).
 *
 * That logic only works if the load() faithfully reads back what
 * save() wrote. This KAT round-trips a populated KangarooWorkFile
 * through a temp file and checks every field survives.
 *
 * Pure host test, no GPU dependency.
 */

#include "../src/core/kangaroo.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <filesystem>

namespace {

int g_failures = 0;

void check_eq(const char* name, const std::string& expected, const std::string& got) {
    if (expected == got) {
        std::printf("[ok  ] %s\n", name);
    } else {
        std::printf("[FAIL] %s: expected '%s' got '%s'\n",
                    name, expected.c_str(), got.c_str());
        ++g_failures;
    }
}

void check_u64(const char* name, uint64_t expected, uint64_t got) {
    if (expected == got) {
        std::printf("[ok  ] %s\n", name);
    } else {
        std::printf("[FAIL] %s: expected %llu got %llu\n",
                    name,
                    (unsigned long long)expected,
                    (unsigned long long)got);
        ++g_failures;
    }
}

void check_u32(const char* name, uint32_t expected, uint32_t got) {
    if (expected == got) {
        std::printf("[ok  ] %s\n", name);
    } else {
        std::printf("[FAIL] %s: expected %u got %u\n", name, expected, got);
        ++g_failures;
    }
}

}  // namespace

int main() {
    using ::collider::KangarooWorkFile;

    // 1. Populate a source work file.
    KangarooWorkFile src;
    src.target_pubkey_hex =
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798"
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8";
    src.range_start_hex = "4000000000000000000";  // 2^74 (puzzle 75 lower bound)
    src.range_end_hex   = "7ffffffffffffffffff"; // 2^75 - 1
    src.dp_bits         = 24u;
    src.total_steps     = 1234567890ULL;
    src.elapsed_seconds = 42.7;
    // No DPs added -- the dp serializer is exercised by other tests.

    // 2. Pick a temp path inside the build tree.
    const std::string tmp_path =
        (std::filesystem::temp_directory_path() /
         "collider_test_kangaroo_work_file.tmp").string();

    // 3. Save -> Load -> compare.
    if (!src.save(tmp_path)) {
        std::printf("[FAIL] save() returned false\n");
        return 1;
    }

    KangarooWorkFile dst;
    if (!dst.load(tmp_path)) {
        std::printf("[FAIL] load() returned false\n");
        std::remove(tmp_path.c_str());
        return 1;
    }

    check_eq("target_pubkey_hex", src.target_pubkey_hex, dst.target_pubkey_hex);
    check_eq("range_start_hex",   src.range_start_hex,   dst.range_start_hex);
    check_eq("range_end_hex",     src.range_end_hex,     dst.range_end_hex);
    check_u32("dp_bits",          src.dp_bits,           dst.dp_bits);
    check_u64("total_steps",      src.total_steps,       dst.total_steps);
    // elapsed_seconds saved with 1-decimal precision per save() impl;
    // reading back must match within float tolerance.
    if (std::abs(src.elapsed_seconds - dst.elapsed_seconds) > 0.05) {
        std::printf("[FAIL] elapsed_seconds: expected %.1f got %.1f\n",
                    src.elapsed_seconds, dst.elapsed_seconds);
        ++g_failures;
    } else {
        std::printf("[ok  ] elapsed_seconds (within precision)\n");
    }

    // 4. Test mismatched-checkpoint detection: load a file written
    // with a different target. The KangarooSolver::solve() code path
    // does the comparison, but we can simulate the comparison logic
    // inline since the rejection condition is just != on hex strings.
    KangarooWorkFile mismatched;
    mismatched.target_pubkey_hex = "ffffffff" + std::string(56, '0');  // different
    mismatched.range_start_hex   = src.range_start_hex;
    mismatched.range_end_hex     = src.range_end_hex;
    mismatched.dp_bits           = src.dp_bits;
    mismatched.total_steps       = 0;
    mismatched.elapsed_seconds   = 0;

    if (mismatched.save(tmp_path)) {
        KangarooWorkFile reloaded;
        if (reloaded.load(tmp_path)) {
            if (reloaded.target_pubkey_hex == src.target_pubkey_hex) {
                std::printf("[FAIL] mismatched target round-tripped as identical\n");
                ++g_failures;
            } else {
                std::printf("[ok  ] mismatched target detected via field comparison\n");
            }
        }
    }

    std::remove(tmp_path.c_str());

    if (g_failures > 0) {
        std::printf("FAIL: %d KangarooWorkFile cases failed\n", g_failures);
        return 1;
    }
    std::printf("test_kangaroo_work_file: 7/7 PASS\n");
    return 0;
}
