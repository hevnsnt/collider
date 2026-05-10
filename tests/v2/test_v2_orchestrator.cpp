/**
 * Brain Wallet v2 orchestrator unit tests.
 *
 * Pure host code -- no CUDA needed. Exercises:
 *   * parse_scheme_mask: empty / "stock" / "all" / explicit list / unknown name
 *   * parse_addr_mask: same variants + the "modern" / "puzzle_only" shortcuts
 *   * load_puzzle_targets: well-formed file, malformed entries skipped,
 *     missing file rejected, wrong shape rejected
 *
 * Plain-assert style to match the rest of tests/.
 */

#include "../../src/gpu/v2/v2_orchestrator.hpp"
#include "../../src/gpu/v2/brain_wallet_v2.hpp"

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

using namespace collider::gpu::v2;

static int failures = 0;
#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        std::fprintf(stderr, "FAIL: %s   (%s:%d)\n",                \
                     msg, __FILE__, __LINE__);                      \
        ++failures;                                                 \
    }                                                               \
} while (0)

static void test_parse_scheme_mask_defaults() {
    uint32_t mask = 0;
    std::string err;
    CHECK(parse_scheme_mask("", mask, err), "empty -> stock parse OK");
    CHECK(mask == SCHEME_MASK_STOCK, "empty -> SCHEME_MASK_STOCK");

    CHECK(parse_scheme_mask("stock", mask, err), "stock parse OK");
    CHECK(mask == SCHEME_MASK_STOCK, "stock literal -> SCHEME_MASK_STOCK");

    CHECK(parse_scheme_mask("all", mask, err), "all parse OK");
    CHECK(mask == SCHEME_MASK_ALL, "all -> SCHEME_MASK_ALL");

    CHECK(parse_scheme_mask("*", mask, err), "* parse OK");
    CHECK(mask == SCHEME_MASK_ALL, "* -> SCHEME_MASK_ALL");
}

static void test_parse_scheme_mask_explicit() {
    uint32_t mask = 0;
    std::string err;
    CHECK(parse_scheme_mask("sha256_pw,sha256_iter_16", mask, err),
          "explicit list parses");
    CHECK(mask == (scheme_bit(DerivationScheme::SHA256_PW) |
                   scheme_bit(DerivationScheme::SHA256_ITER_16)),
          "two-scheme mask");

    CHECK(parse_scheme_mask("HMAC_SHA512_PW", mask, err),
          "case-insensitive scheme name");
    CHECK(mask == scheme_bit(DerivationScheme::HMAC_SHA512_PW),
          "case-insensitive scheme value");
}

static void test_parse_scheme_mask_errors() {
    uint32_t mask = 0xDEADBEEF;
    std::string err;
    CHECK(!parse_scheme_mask("bogus_scheme", mask, err),
          "unknown name returns false");
    CHECK(!err.empty(), "unknown name populates error_out");
    CHECK(err.find("bogus_scheme") != std::string::npos,
          "error names the offending token");
}

static void test_parse_addr_mask_defaults() {
    uint32_t mask = 0;
    std::string err;
    CHECK(parse_addr_mask("", mask, err), "empty parses");
    CHECK(mask == ADDR_MASK_STOCK, "empty -> stock");

    CHECK(parse_addr_mask("modern", mask, err), "modern parses");
    CHECK(mask == ADDR_MASK_MODERN, "modern keyword maps");

    CHECK(parse_addr_mask("puzzle_only", mask, err), "puzzle_only parses");
    CHECK(mask == 0, "puzzle_only -> 0 (short-circuit)");

    CHECK(parse_addr_mask("none", mask, err), "none parses");
    CHECK(mask == 0, "none -> 0");
}

static void test_parse_addr_mask_explicit() {
    uint32_t mask = 0;
    std::string err;
    CHECK(parse_addr_mask("p2pkh_compressed,p2tr_bip86", mask, err),
          "explicit list parses");
    CHECK(mask == (addr_bit(AddressType::P2PKH_COMPRESSED) |
                   addr_bit(AddressType::P2TR_BIP86)),
          "two-type mask");
}

static std::string write_temp(const char* name, const std::string& body) {
    // Use std::filesystem so we don't shell out (and mismatch path
    // separators between cmd.exe and the C++ standard library).
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / "tmp_v2_orch";
    std::error_code ec;
    fs::create_directories(dir, ec);
    fs::path path = dir / name;
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(body.data(), static_cast<std::streamsize>(body.size()));
    f.close();
    return path.string();
}

static void test_load_puzzle_targets_happy_path() {
    // Two well-formed entries plus one malformed (skipped).
    const std::string body = R"({
        "puzzles": [
            {"puzzle_n": 32,
             "private_key_hex": "00000000000000000000000000000000000000000000000000000000b81a8d49"},
            {"puzzle_number": 40,
             "private_key": "0xff35b9ee85"},
            {"n": 999, "key": "deadbeef"}
        ]
    })";
    const std::string path = write_temp("happy.json", body);
    std::vector<PuzzleTarget> targets;
    std::string err;
    bool ok = load_puzzle_targets(path, targets, err);
    CHECK(ok, "load_puzzle_targets accepts well-formed file");
    CHECK(targets.size() == 2, "two valid entries loaded (puzzle_n>160 skipped)");
    CHECK(targets[0].puzzle_n == 32, "first target is puzzle 32");
    CHECK(targets[1].puzzle_n == 40, "second target is puzzle 40");
    // Verify mask math: puzzle_n=32 => low_bits=31, mask covers bits 0..30.
    CHECK(targets[0].low_mask[0] == ((1ull << 31) - 1ull),
          "puzzle_n=32 mask is (2^31 - 1) on limb 0");
    CHECK(targets[0].low_mask[1] == 0 &&
          targets[0].low_mask[2] == 0 &&
          targets[0].low_mask[3] == 0,
          "puzzle_n=32 mask zero on upper limbs");
}

static void test_load_puzzle_targets_solve_history_shape() {
    const std::string body = R"({
        "solve_history": [
            {"puzzle_number": 16, "private_key": "0x1d3"}
        ]
    })";
    const std::string path = write_temp("solve_history.json", body);
    std::vector<PuzzleTarget> targets;
    std::string err;
    CHECK(load_puzzle_targets(path, targets, err),
          "alternate solve_history shape parses");
    CHECK(targets.size() == 1, "single entry loaded");
    CHECK(targets[0].puzzle_n == 16, "puzzle_n via 'puzzle_number'");
}

static void test_load_puzzle_targets_missing_file() {
    std::vector<PuzzleTarget> targets;
    std::string err;
    bool ok = load_puzzle_targets("/no/such/path/v2.json", targets, err);
    CHECK(!ok, "missing file returns false");
    CHECK(!err.empty(), "missing file populates error");
    CHECK(err.find("could not open") != std::string::npos,
          "error message identifies the open failure");
}

static void test_load_puzzle_targets_wrong_shape() {
    const std::string body = R"({"completely": "wrong"})";
    const std::string path = write_temp("wrong_shape.json", body);
    std::vector<PuzzleTarget> targets;
    std::string err;
    CHECK(!load_puzzle_targets(path, targets, err),
          "wrong shape returns false");
    CHECK(err.find("'puzzles'") != std::string::npos ||
          err.find("solve_history") != std::string::npos,
          "error names the expected keys");
}

static void test_run_orchestrator_dry_run() {
    const std::string body = R"({"puzzles":[{"puzzle_n":1,"private_key_hex":"01"}]})";
    const std::string path = write_temp("dry.json", body);
    OrchestratorOptions opts;
    opts.puzzle_keys_path = path;
    opts.scheme_mask = SCHEME_MASK_ALL;
    opts.addr_mask = 0;  // puzzle-only
    opts.dry_run = true;
    opts.show_summary = false;
    int rc = run_v2_orchestrator(opts);
    CHECK(rc == 0, "dry-run returns success");
}

int main() {
    test_parse_scheme_mask_defaults();
    test_parse_scheme_mask_explicit();
    test_parse_scheme_mask_errors();
    test_parse_addr_mask_defaults();
    test_parse_addr_mask_explicit();
    test_load_puzzle_targets_happy_path();
    test_load_puzzle_targets_solve_history_shape();
    test_load_puzzle_targets_missing_file();
    test_load_puzzle_targets_wrong_shape();
    test_run_orchestrator_dry_run();

    if (failures != 0) {
        std::fprintf(stderr, "test_v2_orchestrator: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_v2_orchestrator: PASS\n");
    return 0;
}
