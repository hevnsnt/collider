/**
 * cli_parser.cpp - Implementation of theCollider's CLI argument parser.
 *
 * Argv is walked exactly once. Each token is matched against a flat
 * std::array<FlagSpec, N> by long_name / short_name; the matching FlagSpec's
 * apply() callback handles value consumption and Arguments / CLIFlags
 * mutation. The previous implementation was a 600-line if/else chain inline
 * in this function; the registry collapses every uniform flag onto a one-line
 * entry and keeps each special-case flag's logic local to its callback.
 *
 * Pro-only flags stay gated by COLLIDER_PRO so the Free build still compiles
 * cleanly. In the Free build the Pro entries either disappear from the
 * registry (most cases) or remain with a Free-build callback that prints the
 * Pro hint and discards the value (so scripts that pass --bloom under a Free
 * binary still parse, they just no-op the flag with a notice).
 */
#include "cli/cli_parser.hpp"

#include "cli/flag_spec.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
  #include <aclapi.h>
  #include <sddl.h>
#else
  #include <sys/stat.h>
  #include <unistd.h>
#endif

namespace {

// Reject password files that are world-readable. Mirrors the convention
// that SSH applies to identity files (chmod 0600 required). Returns true
// when the file's permissions are safe to read; populates err_msg and
// returns false otherwise. The fallback path returns true to avoid
// hard-failing on platforms whose permission model we cannot inspect.
bool password_file_permissions_safe(const std::string& path,
                                    std::string& err_msg) {
#ifndef _WIN32
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0) {
        // open() will fail with a clearer message a moment later; fall
        // through to that error path rather than emitting a duplicate.
        return true;
    }
    if (st.st_mode & (S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH)) {
        err_msg = "[!] --pool-password-file: " + path +
                  " has group/world-readable permissions ("
                  "mode=0" + std::to_string(st.st_mode & 0777) +
                  "). Refusing to read a password from a file other "
                  "users can see. Run: chmod 600 " + path + "\n";
        return false;
    }
    return true;
#else
    // Windows: check the file's DACL. We accept the file only when the
    // current user's SID is the only identity granted FILE_GENERIC_READ.
    // BUILTIN\Administrators and SYSTEM are always trusted (they can
    // read anything anyway), so their presence on the ACL is ignored.
    // If the API calls fail we fall through to allow open() (the existing
    // behavior); the goal is to catch the obvious "Everyone:Read" case,
    // not to be a full ACL auditor.
    PSECURITY_DESCRIPTOR psd = nullptr;
    PACL pdacl = nullptr;
    if (GetNamedSecurityInfoA(
            path.c_str(), SE_FILE_OBJECT, DACL_SECURITY_INFORMATION,
            nullptr, nullptr, &pdacl, nullptr, &psd) != ERROR_SUCCESS) {
        return true;
    }
    bool safe = true;
    if (pdacl != nullptr) {
        for (DWORD i = 0; i < pdacl->AceCount; ++i) {
            void* ace_raw = nullptr;
            if (!GetAce(pdacl, i, &ace_raw)) continue;
            auto* hdr = static_cast<ACE_HEADER*>(ace_raw);
            if (hdr->AceType != ACCESS_ALLOWED_ACE_TYPE) continue;
            auto* ace = static_cast<ACCESS_ALLOWED_ACE*>(ace_raw);
            if (!(ace->Mask & FILE_GENERIC_READ)) continue;
            PSID sid = reinterpret_cast<PSID>(&ace->SidStart);
            LPSTR sid_str = nullptr;
            if (!ConvertSidToStringSidA(sid, &sid_str)) continue;
            std::string sid_view = sid_str;
            LocalFree(sid_str);
            // S-1-1-0 = Everyone. S-1-5-7 = ANONYMOUS LOGON. S-1-5-11 =
            // Authenticated Users. S-1-5-32-545 = BUILTIN\Users. Any of
            // these granting read on a password file is unsafe.
            if (sid_view == "S-1-1-0" ||
                sid_view == "S-1-5-7" ||
                sid_view == "S-1-5-11" ||
                sid_view == "S-1-5-32-545") {
                safe = false;
                break;
            }
        }
    }
    if (psd != nullptr) LocalFree(psd);
    if (!safe) {
        err_msg = "[!] --pool-password-file: " + path +
                  " has a DACL that grants read access to Everyone / "
                  "Authenticated Users / BUILTIN\\Users. Refusing to "
                  "read a password from a file other accounts can see. "
                  "Right-click the file > Properties > Security and "
                  "remove those entries, leaving only your own user.\n";
    }
    return safe;
#endif
}

// Read the first line of `path` into `out`, stripping any trailing
// CR/LF/whitespace. Returns true on success, false on open failure or
// when the file's permissions allow other users to read it. Used by
// --pool-password-file so the secret never appears in argv (which is
// readable via ps / Task Manager / /proc).
bool read_password_file(const std::string& path, std::string& out,
                        std::string& err_msg) {
    if (!password_file_permissions_safe(path, err_msg)) {
        return false;
    }
    std::ifstream f(path);
    if (!f.is_open()) {
        err_msg = "[!] --pool-password-file: cannot open " + path + "\n";
        return false;
    }
    std::string line;
    if (!std::getline(f, line)) {
        err_msg = "[!] --pool-password-file: " + path + " is empty\n";
        return false;
    }
    // Strip trailing whitespace (CR for Windows line endings, plus any
    // trailing spaces that crept in from a text editor).
    while (!line.empty() &&
           (line.back() == '\r' || line.back() == '\n' ||
            line.back() == ' '  || line.back() == '\t')) {
        line.pop_back();
    }
    out = std::move(line);
    return true;
}

// Helper: advance i and return the value at the new slot, or set err_msg and
// return nullptr when the next slot is past the end of argv. Used by every
// VALUE-shaped flag to gate the "--flag without value" case uniformly.
const char* take_value(int& i, int argc, char* argv[],
                       std::string_view flag, std::string& err_msg) {
    if (i + 1 >= argc) {
        err_msg = std::string("[!] ") + std::string(flag) +
                  " requires a value.\n";
        return nullptr;
    }
    return argv[++i];
}

using ::Arguments;                  // global scope; predates collider namespace
using ::collider::CLIFlags;
using ::collider::cli::FlagSpec;

// ---------------------------------------------------------------------------
// Apply callbacks. One per flag. The body of each callback mirrors exactly
// what the old if/else chain did for that flag, with no behavior change.
// ---------------------------------------------------------------------------

int apply_help(int&, int, char* [], Arguments& args, CLIFlags&,
               std::string&) {
    args.help = true;
    return 0;
}

int apply_verbose(int&, int, char* [], Arguments& args, CLIFlags& cli,
                  std::string&) {
    args.verbose = true;
    cli.verbose_set = true;
    return 0;
}

// Checked numeric parse: returns true and sets `out` on success; on a bad
// value (non-numeric, trailing junk, overflow) sets err_msg and returns false
// so the caller returns -1 down the clean error path. Without this, std::stoi/
// stoull throw std::invalid_argument / std::out_of_range which were uncaught
// and aborted the whole process on bad CLI input (audit HIGH-2).
static bool cli_parse_ll(const char* s, const char* flag, long long& out,
                         std::string& err_msg) {
    if (!s || !*s) {
        err_msg = std::string("[!] ") + flag + ": missing or empty value";
        return false;
    }
    try {
        const std::string str(s);
        size_t consumed = 0;
        long long v = std::stoll(str, &consumed);
        if (consumed != str.size()) {  // e.g. "12x", "1,2"
            err_msg = std::string("[!] ") + flag + ": not a valid integer: '" + s + "'";
            return false;
        }
        out = v;
        return true;
    } catch (const std::exception&) {
        err_msg = std::string("[!] ") + flag + ": not a valid integer (or out of range): '" + s + "'";
        return false;
    }
}

int apply_gpus(int& i, int argc, char* argv[], Arguments& args,
               CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--gpus", err_msg);
    if (!v) return -1;
    args.gpu_ids.clear();
    std::string gpus = v;
    size_t pos = 0;
    while ((pos = gpus.find(',')) != std::string::npos) {
        long long id;
        if (!cli_parse_ll(gpus.substr(0, pos).c_str(), "--gpus", id, err_msg)) return -1;
        if (id < 0) { err_msg = "[!] --gpus: device ids must be >= 0"; return -1; }
        args.gpu_ids.push_back(static_cast<int>(id));
        gpus.erase(0, pos + 1);
    }
    long long id;
    if (!cli_parse_ll(gpus.c_str(), "--gpus", id, err_msg)) return -1;
    if (id < 0) { err_msg = "[!] --gpus: device ids must be >= 0"; return -1; }
    args.gpu_ids.push_back(static_cast<int>(id));
    cli.gpu_ids_set = true;
    return 0;
}

int apply_batch_size(int& i, int argc, char* argv[], Arguments& args,
                     CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--batch-size", err_msg);
    if (!v) return -1;
    long long n;
    if (!cli_parse_ll(v, "--batch-size", n, err_msg)) return -1;
    // Preserve legacy semantics: 0 is an accepted CLI override (see the
    // batch-size-zero-cli-wins test); the guard above only rejects
    // non-numeric input that previously crashed via std::stoull.
    args.batch_size = static_cast<uint64_t>(n);
    args.batch_size_auto = false;
    cli.batch_size_set = true;
    return 0;
}

// v1.5.x: kangaroo count override (K). Positive int, no upper bound at
// parse time (consumers clamp per-backend if needed). 0 / negative is
// rejected so an accidental --kangaroos 0 doesn't silently fall back to
// the default (which would mask a typo).
int apply_kangaroos(int& i, int argc, char* argv[], Arguments& args,
                    CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--kangaroos", err_msg);
    if (!v) return -1;
    try {
        int n = std::stoi(v);
        if (n <= 0) {
            err_msg = "[!] --kangaroos: must be a positive integer; "
                      "got '" + std::string(v) + "'\n";
            return -1;
        }
        args.num_kangaroos = n;
        cli.num_kangaroos_set = true;
        return 0;
    } catch (const std::exception&) {
        err_msg = "[!] --kangaroos: failed to parse '" +
                  std::string(v) + "' as an integer\n";
        return -1;
    }
}

// v1.5.x: runtime kangaroo backend selection. Accepts cpu / cuda / metal
// case-insensitively. The actual parse-into-BackendKind happens at the
// consumption site (pool_solver, puzzle_solver_kangaroo) so this file
// doesn't have to include core/kangaroo_backend.hpp (which pulls the
// pool client header chain).
int apply_backend(int& i, int argc, char* argv[], Arguments& args,
                  CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--backend", err_msg);
    if (!v) return -1;
    std::string label = v;
    // Lowercase for the equality check below; the consumer also
    // case-folds via parse_backend_kind so this is belt + suspenders.
    std::string lower;
    lower.reserve(label.size());
    for (char c : label) {
        lower.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower != "cpu" && lower != "cuda" && lower != "metal") {
        err_msg = "[!] --backend: unknown value '" + label +
                  "'. Expected one of: cpu, cuda, metal.\n";
        return -1;
    }
    args.backend_kind = lower;
    cli.backend_kind_set = true;
    return 0;
}

int apply_solver(int& i, int argc, char* argv[], Arguments& args,
                 CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--solver", err_msg);
    if (!v) return -1;
    std::string label = v;
    std::string lower;
    lower.reserve(label.size());
    for (char c : label) {
        lower.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))));
    }
    if (lower != "kangaroo" && lower != "bsgs") {
        err_msg = "[!] --solver: unknown value '" + label +
                  "'. Expected one of: kangaroo, bsgs.\n";
        return -1;
    }
    args.solver = (lower == "kangaroo") ? std::string{} : lower;
    cli.solver_set = true;
    return 0;
}

int apply_bip_scan(int&, int, char* [], Arguments& args, CLIFlags&,
                   std::string&) {
    args.bip_scan_mode = true;
    return 0;
}

int apply_bip_scan_wordlist(int& i, int argc, char* argv[], Arguments& args,
                            CLIFlags&, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--bip-scan-wordlist", err_msg);
    if (!v) return -1;
    args.bip_scan_wordlist = v;
    return 0;
}

// v1.5.0: enable exhaustive BIP-39 entropy-space enumeration. Implies
// --bip-scan; ignores --bip-scan-wordlist if both are set.
int apply_bip_combinatorial(int&, int, char* [], Arguments& args,
                            CLIFlags&, std::string&) {
    args.bip_scan_mode = true;
    args.bip_combinatorial = true;
    return 0;
}

int apply_bip_no_gpu(int&, int, char* [], Arguments& args,
                     CLIFlags&, std::string&) {
    args.bip_no_gpu = true;
    return 0;
}

int apply_bip_combinatorial_words(int& i, int argc, char* argv[],
                                  Arguments& args, CLIFlags&,
                                  std::string& err_msg) {
    const char* v =
        take_value(i, argc, argv, "--bip-combinatorial-words", err_msg);
    if (!v) return -1;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (!end || *end != '\0' ||
        (n != 12 && n != 15 && n != 18 && n != 21 && n != 24)) {
        err_msg = "--bip-combinatorial-words expects one of 12/15/18/21/24";
        return -1;
    }
    args.bip_combinatorial_word_count = static_cast<int>(n);
    return 0;
}

int apply_benchmark(int&, int, char* [], Arguments& args, CLIFlags&,
                    std::string&) {
    args.benchmark = true;
    return 0;
}

int apply_benchmark_time(int& i, int argc, char* argv[], Arguments& args,
                         CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--benchmark-time", err_msg);
    if (!v) return -1;
    long long bench_n;
    if (!cli_parse_ll(v, "--benchmark-time", bench_n, err_msg)) return -1;
    args.benchmark_seconds = static_cast<int>(bench_n);
    cli.benchmark_seconds_set = true;
    return 0;
}

int apply_puzzle(int& i, int argc, char* argv[], Arguments& args,
                 CLIFlags& cli, std::string& err_msg) {
    // --puzzle [N]: N is optional; only consumed when the next argv exists
    // and doesn't start with '-' (so "--puzzle" alone still selects puzzle
    // mode without pinning a number, which is how auto-select-easiest works).
    args.puzzle_mode = true;
    if (i + 1 < argc && argv[i + 1][0] != '-') {
        long long pn;
        if (!cli_parse_ll(argv[++i], "--puzzle", pn, err_msg)) return -1;
        args.puzzle_number = static_cast<int>(pn);
        cli.puzzle_number_set = true;
    }
    return 0;
}

int apply_puzzle_target(int& i, int argc, char* argv[], Arguments& args,
                        CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--puzzle-target", err_msg);
    if (!v) return -1;
    args.puzzle_target = v;
    cli.puzzle_target_set = true;
    return 0;
}

int apply_puzzle_start(int& i, int argc, char* argv[], Arguments& args,
                       CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--puzzle-start", err_msg);
    if (!v) return -1;
    args.puzzle_range_start = v;
    cli.puzzle_start_set = true;
    return 0;
}

int apply_puzzle_end(int& i, int argc, char* argv[], Arguments& args,
                     CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--puzzle-end", err_msg);
    if (!v) return -1;
    args.puzzle_range_end = v;
    cli.puzzle_end_set = true;
    return 0;
}

int apply_pubkey(int& i, int argc, char* argv[], Arguments& args,
                 CLIFlags& cli, std::string& err_msg) {
    // 33-byte compressed pubkey (02/03 + 32B hex) for targets whose pubkey
    // isn't in puzzle_history.json. See README for the canonical
    // kangaroo-able puzzle list.
    const char* v = take_value(i, argc, argv, "--pubkey", err_msg);
    if (!v) return -1;
    args.puzzle_pubkey = v;
    cli.puzzle_pubkey_set = true;
    return 0;
}

int apply_sequential(int&, int, char* [], Arguments& args, CLIFlags& cli,
                     std::string&) {
    args.puzzle_random = false;
    cli.puzzle_random_set = true;
    return 0;
}

int apply_random(int&, int, char* [], Arguments& args, CLIFlags& cli,
                 std::string&) {
    args.puzzle_random = true;
    cli.puzzle_random_set = true;
    return 0;
}

int apply_puzzle_checkpoint(int& i, int argc, char* argv[], Arguments& args,
                            CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--puzzle-checkpoint", err_msg);
    if (!v) return -1;
    args.puzzle_checkpoint = v;
    cli.puzzle_checkpoint_set = true;
    return 0;
}

int apply_resume_kangaroo(int&, int, char* [], Arguments& args, CLIFlags&,
                          std::string&) {
    // Opt in to loading the kangaroo herd checkpoint at the standalone path
    // (the pool path already does this unconditionally for any backend that
    // supports it). The SIGINT save side is always on; only the load is
    // gated so a user who wants a fresh herd doesn't get a stale one.
    args.resume_kangaroo = true;
    return 0;
}

int apply_auto_next(int&, int, char* [], Arguments& args, CLIFlags& cli,
                    std::string&) {
    args.puzzle_auto_next = true;
    cli.puzzle_auto_next_set = true;
    return 0;
}

int apply_all_unsolved(int&, int, char* [], Arguments& args, CLIFlags&,
                       std::string&) {
    args.puzzle_all_unsolved = true;
    return 0;
}

int apply_puzzle_min_bits(int& i, int argc, char* argv[], Arguments& args,
                          CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--puzzle-min-bits", err_msg);
    if (!v) return -1;
    long long minb;
    if (!cli_parse_ll(v, "--puzzle-min-bits", minb, err_msg)) return -1;
    args.puzzle_min_bits = static_cast<int>(minb);
    cli.puzzle_min_bits_set = true;
    return 0;
}

int apply_puzzle_max_bits(int& i, int argc, char* argv[], Arguments& args,
                          CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--puzzle-max-bits", err_msg);
    if (!v) return -1;
    long long maxb;
    if (!cli_parse_ll(v, "--puzzle-max-bits", maxb, err_msg)) return -1;
    args.puzzle_max_bits = static_cast<int>(maxb);
    cli.puzzle_max_bits_set = true;
    return 0;
}

int apply_kangaroo(int&, int, char* [], Arguments& args, CLIFlags& cli,
                   std::string&) {
    args.puzzle_kangaroo = true;
    cli.puzzle_kangaroo_set = true;
    return 0;
}

int apply_dp_bits(int& i, int argc, char* argv[], Arguments& args,
                  CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--dp-bits", err_msg);
    if (!v) return -1;
    long long dpb;
    if (!cli_parse_ll(v, "--dp-bits", dpb, err_msg)) return -1;
    args.dp_bits = static_cast<int>(dpb);
    cli.dp_bits_set = true;
    return 0;
}

int apply_bloom(int& i, int argc, char* argv[], Arguments& args,
                CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--bloom", err_msg);
    if (!v) return -1;
#ifdef COLLIDER_PRO
    args.bloom_file = v;
    cli.bloom_file_set = true;
#else
    (void)args; (void)cli;
    // Pro-only: opportunistic address scanning. Consume the path arg but
    // ignore it; one-time hint so the user knows why.
    std::cerr << "[Pro] --bloom '" << v
              << "' ignored; opportunistic address scanning is a "
                 "Pro feature. https://collisionprotocol.com/pro"
              << std::endl;
#endif
    return 0;
}

int apply_bloom_tight(int& i, int argc, char* argv[], Arguments& args,
                      CLIFlags&, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--bloom-tight", err_msg);
    if (!v) return -1;
#ifdef COLLIDER_PRO
    // dual-bloom pipeline. The --bloom file lives on the GPU and is
    // intentionally loose (small enough to fit alongside working buffers).
    // At its design fill ratio every bloom probe has a non-trivial FP rate
    // (e.g. ~0.02% on a 2.6 GB seen-ever bloom built with -f 1e-3). When
    // --track-empty-hits is also on, those FPs flood found-empty.txt with
    // noise. --bloom-tight points at a much tighter (.blf) built from the
    // same source (e.g. -f 1e-7); we mmap it CPU-side and re-probe every
    // empty candidate before logging. Only candidates that hit BOTH the
    // loose GPU bloom AND the tight CPU bloom get logged as real empty
    // hits. See docs/EMPTY-HITS-WORKFLOW.md.
    args.bloom_tight_file = v;
#else
    (void)args;
    std::cerr << "[Pro] --bloom-tight '" << v
              << "' ignored; dual-bloom is a Pro feature." << std::endl;
#endif
    return 0;
}

int apply_verify_set(int& i, int argc, char* argv[], Arguments& args,
                     CLIFlags&, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--verify-set", err_msg);
    if (!v) return -1;
#ifdef COLLIDER_PRO
    args.verify_set_file = v;
#else
    (void)args;
    std::cerr << "[Pro] --verify-set '" << v
              << "' ignored; bloom-hit verification is a Pro feature."
              << std::endl;
#endif
    return 0;
}

int apply_verify_set_csv(int& i, int argc, char* argv[], Arguments& args,
                         CLIFlags&, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--verify-set-csv", err_msg);
    if (!v) return -1;
#ifdef COLLIDER_PRO
    // route HitVerifier::load_from_csv() from the runner. Operators with
    // the raw UTXO CSV (and no pre-built .uvrf) can use it directly without
    // building a binary UVRF first.
    args.verify_set_csv = v;
#else
    (void)args;
    std::cerr << "[Pro] --verify-set-csv '" << v
              << "' ignored; Pro feature." << std::endl;
#endif
    return 0;
}

int apply_track_empty_hits(int&, int, char* [], Arguments& args, CLIFlags&,
                           std::string&) {
#ifdef COLLIDER_PRO
    // When set, every bloom hit that fails the UVRF funded-set lookup is
    // treated as a "real but empty" wallet (had on-chain activity at some
    // point, currently 0 balance) and logged to found-empty.txt. Only
    // meaningful when the operator's --bloom file is a SEEN-EVER bloom
    // (addresses that had any transaction history) and --verify-set is the
    // currently-funded subset. With a funded-only bloom this flag just
    // promotes bloom false-positives to log spam, so the runner
    // auto-enables it only when both seen_tight.blf and
    // funded_addresses.uvrf auto-resolve (the seen-ever operator
    // configuration). The CLI flag is for operators who want to force it
    // on against the auto-detection.
    args.track_empty_hits = true;
    args.track_empty_hits_user_set = true;
#else
    (void)args;
    std::cerr << "[Pro] --track-empty-hits ignored; this is a Pro feature."
              << std::endl;
#endif
    return 0;
}

int apply_no_track_empty_hits(int&, int, char* [], Arguments& args,
                              CLIFlags&, std::string&) {
#ifdef COLLIDER_PRO
    // Opt-out: pin to false even when auto-detect would have enabled
    // empty-hit tracking.
    args.track_empty_hits = false;
    args.track_empty_hits_user_set = true;
#else
    (void)args;
#endif
    return 0;
}

int apply_use_texture_bloom(int&, int, char* [], Arguments& args, CLIFlags&,
                            std::string&) {
#ifdef COLLIDER_PRO
    // Opt into the texture-memory bloom probe path. The runtime falls back
    // to global memory if more than one GPU is in use (defensive guard) or
    // if the device cannot allocate a texture object.
    args.use_texture_bloom = true;
#else
    (void)args;
    std::cerr << "[Pro] --use-texture-bloom ignored; texture bloom is a "
                 "Pro feature." << std::endl;
#endif
    return 0;
}

int apply_brainwallet_warpwallet(int& i, int argc, char* argv[],
                                 Arguments& args, CLIFlags& cli,
                                 std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--brainwallet-warpwallet",
                               err_msg);
    if (!v) return -1;
#ifdef COLLIDER_PRO
    // WarpWallet brainwallet derivation
    // (scrypt(N=2^18,r=8,p=1) + PBKDF2-SHA256(c=2^16); per Keybase spec).
    // Implies --brainwallet. The argument is the salt used in the Keybase
    // scheme (typically the user's email address).
    args.warpwallet_salt = v;
    args.brainwallet_mode = true;
    args.pool_mode = false;
    args.pool_url.clear();
    cli.brainwallet_set = true;
#else
    (void)args; (void)cli;
    std::cerr << "[Pro] --brainwallet-warpwallet '" << v
              << "' ignored; WarpWallet derivation is a Pro feature."
              << std::endl;
#endif
    return 0;
}

int apply_brainwallet(int&, int, char* [], Arguments& args, CLIFlags& cli,
                      std::string&) {
    args.brainwallet_mode = true;
    args.pool_mode = false;     // Brainwallet overrides pool mode from config
    args.pool_url.clear();      // Don't leak a pool URL once brainwallet is selected
    cli.brainwallet_set = true;
    return 0;
}

int apply_brainwallet_v2(int&, int, char* [], Arguments& args, CLIFlags& cli,
                         std::string&) {
    // Route the brainwallet scan through the v2 kernel chain (multi-scheme
    // derivation + multi-address bloom check per priv). Implies
    // --brainwallet; defaults scheme_mask=ALL unless --schemes is also
    // supplied. Multi-address scanning via --addr-types was removed in
    // Q10 cleanup; the v2 orchestrator -> kernel bridge for addr_mask != 0
    // is deferred to v1.5.0+. Use the legacy --brainwallet --bloom path
    // for multi-address scanning today.
    args.brainwallet_mode = true;
    args.brainwallet_v2_mode = true;
    args.pool_mode = false;
    args.pool_url.clear();
    cli.brainwallet_set = true;
    return 0;
}

int apply_brainwallet_setup(int&, int, char* [], Arguments& args, CLIFlags&,
                            std::string&) {
    args.brainwallet_setup = true;
    return 0;
}

int apply_brute(int& i, int argc, char* argv[], Arguments& args,
                CLIFlags& cli, std::string& err_msg) {
    // Consume one or more positive integer length arguments until the next
    // flag (-prefixed token) or end of argv. Each length must be in [1,16];
    // longer is rejected because 62^17 = 3.0e30 candidates exceeds any
    // plausible scan budget and is almost always a typo.
    args.brainwallet_mode = true;
    args.pool_mode = false;
    args.pool_url.clear();
    cli.brainwallet_set = true;

    int consumed = 0;
    while (i + 1 < argc && argv[i + 1][0] != '-') {
        const char* tok = argv[++i];
        try {
            int n = std::stoi(tok);
            if (n < 1 || n > 16) {
                err_msg = "[!] --brute length must be in [1, 16]; got " +
                          std::to_string(n) +
                          " (62^17 already exceeds 3e30 candidates).\n";
                return -1;
            }
            args.brute_lengths.push_back(n);
            consumed++;
        } catch (const std::exception&) {
            err_msg = std::string("[!] --brute expected a positive integer "
                                  "length, got '") +
                      tok + "'.\n";
            return -1;
        }
    }
    if (consumed == 0) {
        err_msg = "[!] --brute requires at least one length argument "
                  "(e.g. --brute 7 8 9).\n";
        return -1;
    }
    return 0;
}

int apply_puzzle_only_v2(int&, int, char* [], Arguments& args, CLIFlags&,
                         std::string&) {
#ifdef COLLIDER_PRO
    // Enable v2 puzzle-mode kernel + multi-scheme.
    args.puzzle_only_v2 = true;
    args.brainwallet_mode = true;   // Goes through the brainwallet pipeline
    args.pool_mode = false;
    args.pool_url.clear();
    return 0;
#else
    (void)args;
    // Exact text per v1.4.0 spec; do NOT add URLs or extra trailing text.
    std::cerr << "I'm sorry, but this is a pro function. If you'd like "
                 "to try pro, go and visit the website to purchase and "
                 "download." << std::endl;
    return 2;
#endif
}

int apply_puzzle_keys(int& i, int argc, char* argv[], Arguments& args,
                      CLIFlags&, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--puzzle-keys", err_msg);
    if (!v) return -1;
    args.puzzle_keys_file = v;
    return 0;
}

int apply_schemes(int& i, int argc, char* argv[], Arguments& args,
                  CLIFlags&, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--schemes", err_msg);
    if (!v) return -1;
    args.schemes_csv = v;
    return 0;
}

int apply_resume(int&, int, char* [], Arguments& args, CLIFlags& cli,
                 std::string&) {
    args.resume = true;
    cli.resume_set = true;
    return 0;
}

int apply_cpu_rules(int&, int, char* [], Arguments& args, CLIFlags&,
                    std::string&) {
    args.cpu_rules = true;
    return 0;
}

int apply_save_interval(int& i, int argc, char* argv[], Arguments& args,
                        CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--save-interval", err_msg);
    if (!v) return -1;
    long long si;
    if (!cli_parse_ll(v, "--save-interval", si, err_msg)) return -1;
    if (si < 0) { err_msg = "[!] --save-interval: must be >= 0"; return -1; }
    args.save_interval = static_cast<uint64_t>(si);
    cli.save_interval_set = true;
    return 0;
}

int apply_calibrate(int&, int, char* [], Arguments& args, CLIFlags&,
                    std::string&) {
    args.calibrate = true;
    return 0;
}

int apply_force_calibrate(int&, int, char* [], Arguments& args,
                          CLIFlags& cli, std::string&) {
    args.calibrate = true;
    args.force_calibrate = true;
    cli.force_calibrate_set = true;
    return 0;
}

int apply_debug(int&, int, char* [], Arguments& args, CLIFlags& cli,
                std::string&) {
    args.debug = true;
    cli.debug_set = true;
    return 0;
}

int apply_perf_instrument(int&, int, char* [], Arguments& args, CLIFlags&,
                          std::string&) {
    // Opt-in for the per-kernel CUDA-event timing collector. See
    // Arguments::perf_instrument in cli_parser.hpp for the off-by-default
    // rationale. The runner consults args.perf_instrument before calling
    // perf::set_enabled(true); when false, the hot path stays a single
    // relaxed load + predicted-not-taken branch on every kernel launch.
    args.perf_instrument = true;
    return 0;
}

int apply_analyze(int&, int, char* [], Arguments& args, CLIFlags&,
                  std::string&) {
    args.analyze_puzzles = true;
    return 0;
}

int apply_no_smart(int&, int, char* [], Arguments& args, CLIFlags& cli,
                   std::string&) {
    args.smart_select = false;
    cli.smart_select_set = true;
    return 0;
}

int apply_pool(int& i, int argc, char* argv[], Arguments& args,
               CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--pool", err_msg);
    if (!v) return -1;
    args.pool_mode = true;
    args.pool_url = v;
    cli.pool_url_set = true;
    return 0;
}

int apply_worker(int& i, int argc, char* argv[], Arguments& args,
                 CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--worker", err_msg);
    if (!v) return -1;
    // Pentest CLIENT-LIE-3 (Info) + CLIENT-CHEAT-2 (defense-in-depth)
    // worker-name format check (2026-05-23 deep audit). Mirrors the
    // server-side strict validator added under SRV-LIE-14. A typo'd
    // BTC address permanently misroutes payouts; refusing at parse
    // time saves the operator from a silent payout-to-the-wrong-place
    // surprise. Accept [A-Za-z0-9_.-]{1,64}; escape via
    // --worker-unsafe-allow-any for tests.
    std::string name = v;
    bool unsafe = false;
    for (int j = 1; j < argc; ++j) {
        if (std::string(argv[j]) == "--worker-unsafe-allow-any") {
            unsafe = true;
            break;
        }
    }
    if (!unsafe) {
        if (name.empty() || name.size() > 64) {
            err_msg = "--worker must be 1..64 characters";
            return -1;
        }
        for (char c : name) {
            const bool ok =
                (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') ||
                c == '_' || c == '.' || c == '-';
            if (!ok) {
                err_msg =
                    "--worker contains invalid character; allowed "
                    "[A-Za-z0-9_.-] only (use --worker-unsafe-allow-any "
                    "for tests)";
                return -1;
            }
        }
    }
    args.pool_worker = std::move(name);
    cli.pool_worker_set = true;
    return 0;
}

int apply_worker_key(int& i, int argc, char* argv[], Arguments& args,
                     CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--worker-key", err_msg);
    if (!v) return -1;
    std::string path = v;
    // Existence + readability is verified at pool startup, not here,
    // so that --worker-key in a config.yml that's loaded long before
    // any pool work doesn't fail dry-runs. We only do shape checks
    // (non-empty path) at parse time.
    if (path.empty()) {
        err_msg = "--worker-key path must be non-empty";
        return -1;
    }
    args.pool_worker_key_file = std::move(path);
    return 0;
}

int apply_pool_password(int& i, int argc, char* argv[], Arguments& args,
                        CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--pool-password", err_msg);
    if (!v) return -1;
    // SecureString::assign copies the bytes into wiped-on-free storage.
    // The argv byte buffer itself is still operator-visible via
    // ps / Task Manager; that is the entire reason --pool-password is
    // deprecated in favour of --pool-password-file. SecureString here
    // closes the heap-residue half of the leak (no plain std::string
    // copy carrying the password past Arguments destruction).
    args.pool_password.assign(v, std::strlen(v));
    cli.pool_password_set = true;
    // Argv is world-readable on every supported platform (ps aux on POSIX,
    // Process Explorer / Task Manager on Windows, /proc/<pid>/cmdline on
    // Linux). Warn loudly so operators migrate to --pool-password-file
    // before v1.6.0 removes this flag.
    std::cerr << "[CLI] WARNING: --pool-password leaks via "
                 "process listing (ps / Task Manager). Use "
                 "--pool-password-file <path> for production "
                 "deployments. The --pool-password flag is "
                 "scheduled for removal in v1.6.0."
              << std::endl;
    return 0;
}

int apply_pool_password_file(int& i, int argc, char* argv[],
                             Arguments& args, CLIFlags& cli,
                             std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--pool-password-file",
                               err_msg);
    if (!v) return -1;
    std::string path = v;
    std::string secret;
    std::string read_err;
    if (!read_password_file(path, secret, read_err)) {
        err_msg = read_err;
        return -1;
    }
    args.pool_password_file = path;
    // Copy bytes into the SecureString slot, then explicitly wipe the
    // transient std::string `secret` before its scope exit so the on-heap
    // bytes are zeroed regardless of std::string's destructor behaviour
    // (which makes no wipe guarantee and on SSO may not even free()).
    args.pool_password.assign(secret.data(), secret.size());
    if (!secret.empty()) {
        ::collider::secure_wipe(secret.data(), secret.size());
    }
    cli.pool_password_file_set = true;
    // Mark pool_password as user-set so the yaml override path does not
    // clobber the file-sourced value. --pool-password and
    // --pool-password-file are alternative sources for the same field.
    cli.pool_password_set = true;
    return 0;
}

int apply_pool_api_key(int& i, int argc, char* argv[], Arguments& args,
                       CLIFlags& cli, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--pool-api-key", err_msg);
    if (!v) return -1;
    args.pool_api_key = v;
    cli.pool_api_key_set = true;
    // HTTP pool path was removed (credential-leak review). The remaining
    // JLP/JLPS transports authenticate via worker/password in AUTH, not an
    // API key. Keep the flag accepting a value to avoid surprising old
    // configs / scripts, but emit a one-time deprecation notice so
    // operators migrate.
    std::cerr << "[CLI] --pool-api-key is deprecated and ignored "
                 "(HTTP pool transport was removed; use --pool-worker / "
                 "--pool-password with jlp:// or jlps://)." << std::endl;
    return 0;
}

int apply_config(int& i, int argc, char* argv[], Arguments& args,
                 CLIFlags&, std::string& err_msg) {
    const char* v = take_value(i, argc, argv, "--config", err_msg);
    if (!v) return -1;
    args.config_file = v;
    return 0;
}

int apply_no_tui(int&, int, char* [], Arguments& args, CLIFlags&,
                 std::string&) {
    // TUI flag pair. --no-tui forces the legacy flat-line status output
    // (suitable for log scrapers, CI, and operators that prefer a quiet
    // single-line update). --tui forces the multi-panel TUI even when
    // stdout is not a TTY (e.g. piping into a recording wrapper that
    // handles ANSI escapes). When neither flag is supplied the runner
    // auto-detects via isatty.
    args.no_tui = true;
    args.no_tui_user_set = true;
    return 0;
}

int apply_tui(int&, int, char* [], Arguments& args, CLIFlags&,
              std::string&) {
    args.no_tui = false;
    args.no_tui_user_set = true;
    return 0;
}

// --no-update: disable client self-update in pool mode. An explicit CLI
// opt-out always wins over config.yml pool.auto_update (we mark the
// user-set sentinel so apply_config_to_args does not re-enable it).
int apply_no_update(int&, int, char* [], Arguments& args, CLIFlags&,
                    std::string&) {
    args.no_update = true;
    args.pool_auto_update = false;
    args.pool_auto_update_user_set = true;
    return 0;
}

// ---------------------------------------------------------------------------
// Flag registry. Order is not load-bearing for dispatch (flat lookup), but
// the groupings track the layout of the old if/else chain so reviewers can
// follow the refactor line-by-line. Group labels also drive --help.
// ---------------------------------------------------------------------------
// Deduced size so adding a flag entry doesn't require touching the count.
constexpr FlagSpec kFlagsRaw[] = {
    // General
    {"--help",                  "-h", apply_help,                  "general"},
    {"--verbose",               "-v", apply_verbose,               "general"},
    {"--debug",                 nullptr, apply_debug,              "general"},
    {"--perf-instrument",       nullptr, apply_perf_instrument,    "general"},
    {"--config",                "-c", apply_config,                "general"},

    // GPU
    {"--gpus",                  "-g", apply_gpus,                  "gpu"},
    {"--batch-size",            nullptr, apply_batch_size,         "gpu"},
    {"--backend",               nullptr, apply_backend,            "gpu"},
    {"--solver",                nullptr, apply_solver,             "puzzle"},
    {"--bip-scan",              nullptr, apply_bip_scan,           "puzzle"},
    {"--bip-scan-wordlist",     nullptr, apply_bip_scan_wordlist,  "puzzle"},
    {"--bip-combinatorial",     nullptr, apply_bip_combinatorial,  "puzzle"},
    {"--no-bip-gpu",            nullptr, apply_bip_no_gpu,         "puzzle"},
    {"--bip-combinatorial-words", nullptr,
        apply_bip_combinatorial_words, "puzzle"},
    {"--kangaroos",             nullptr, apply_kangaroos,          "gpu"},

    // Benchmark
    {"--benchmark",             nullptr, apply_benchmark,          "benchmark"},
    {"--benchmark-time",        nullptr, apply_benchmark_time,     "benchmark"},

    // Puzzle
    {"--puzzle",                "-P", apply_puzzle,                "puzzle"},
    {"--puzzle-target",         nullptr, apply_puzzle_target,      "puzzle"},
    {"--puzzle-start",          nullptr, apply_puzzle_start,       "puzzle"},
    {"--puzzle-end",            nullptr, apply_puzzle_end,         "puzzle"},
    {"--pubkey",                nullptr, apply_pubkey,             "puzzle"},
    {"--sequential",            nullptr, apply_sequential,         "puzzle"},
    {"--random",                nullptr, apply_random,             "puzzle"},
    {"--puzzle-checkpoint",     nullptr, apply_puzzle_checkpoint,  "puzzle"},
    {"--resume-kangaroo",       nullptr, apply_resume_kangaroo,    "puzzle"},
    {"--auto-next",             nullptr, apply_auto_next,          "puzzle"},
    {"--all-unsolved",          nullptr, apply_all_unsolved,       "puzzle"},
    {"--puzzle-min-bits",       nullptr, apply_puzzle_min_bits,    "puzzle"},
    {"--puzzle-max-bits",       nullptr, apply_puzzle_max_bits,    "puzzle"},
    {"--kangaroo",              nullptr, apply_kangaroo,           "puzzle"},
    {"--dp-bits",               nullptr, apply_dp_bits,            "puzzle"},

    // Bloom / brainwallet (Pro flags keep their slot in the Free build but
    // their callbacks no-op + print the Pro hint).
    {"--bloom",                 nullptr, apply_bloom,              "brainwallet"},
    {"--bloom-tight",           nullptr, apply_bloom_tight,        "brainwallet"},
    {"--verify-set",            nullptr, apply_verify_set,         "brainwallet"},
    {"--verify-set-csv",        nullptr, apply_verify_set_csv,     "brainwallet"},
    {"--track-empty-hits",      nullptr, apply_track_empty_hits,   "brainwallet"},
    {"--no-track-empty-hits",   nullptr, apply_no_track_empty_hits,"brainwallet"},
    {"--use-texture-bloom",     nullptr, apply_use_texture_bloom,  "brainwallet"},
    {"--brainwallet-warpwallet",nullptr, apply_brainwallet_warpwallet,"brainwallet"},
    {"--brainwallet",           nullptr, apply_brainwallet,        "brainwallet"},
    {"--brainwallet-v2",        nullptr, apply_brainwallet_v2,     "brainwallet"},
    {"--brainwallet-setup",     nullptr, apply_brainwallet_setup,  "brainwallet"},
    {"--brute",                 nullptr, apply_brute,              "brainwallet"},
    {"--puzzle-only-v2",        nullptr, apply_puzzle_only_v2,     "brainwallet"},
    {"--puzzle-keys",           nullptr, apply_puzzle_keys,        "brainwallet"},
    {"--schemes",               nullptr, apply_schemes,            "brainwallet"},
    {"--resume",                nullptr, apply_resume,             "brainwallet"},
    {"--cpu-rules",             nullptr, apply_cpu_rules,          "brainwallet"},
    {"--save-interval",         nullptr, apply_save_interval,      "brainwallet"},

    // Calibration / analysis
    {"--calibrate",             nullptr, apply_calibrate,          "calibrate"},
    {"--force-calibrate",       nullptr, apply_force_calibrate,    "calibrate"},
    {"--analyze",               nullptr, apply_analyze,            "calibrate"},
    {"--no-smart",              nullptr, apply_no_smart,           "calibrate"},

    // Pool
    {"--pool",                  "-p", apply_pool,                  "pool"},
    {"--worker",                "-w", apply_worker,                "pool"},
    {"--pool-password",         nullptr, apply_pool_password,      "pool"},
    {"--pool-password-file",    nullptr, apply_pool_password_file, "pool"},
    {"--pool-api-key",          nullptr, apply_pool_api_key,       "pool"},
    {"--worker-key",            nullptr, apply_worker_key,         "pool"},
    {"--no-update",             nullptr, apply_no_update,          "pool"},

    // TUI
    {"--no-tui",                nullptr, apply_no_tui,             "tui"},
    {"--tui",                   nullptr, apply_tui,                "tui"},
};
constexpr std::size_t kFlagCount = sizeof(kFlagsRaw) / sizeof(kFlagsRaw[0]);

// Look up by long_name OR short_name. Returns nullptr if no match. Linear
// scan is fine: kFlagCount is small enough that std::map / hash table
// would lose to cache-line locality on this workload.
const FlagSpec* find_flag(std::string_view arg) {
    for (const auto& f : kFlagsRaw) {
        if (arg == f.long_name) return &f;
        if (f.short_name && f.short_name[0] != '\0' && arg == f.short_name) {
            return &f;
        }
    }
    return nullptr;
}

}  // namespace

int validate_mode_mutex(const Arguments& args, std::string& msg) {
    int active = 0;
    std::vector<std::string> chosen;
    if (args.brainwallet_mode) { active++; chosen.emplace_back("--brainwallet"); }
    if (args.pool_mode)        { active++; chosen.emplace_back("--pool <url>"); }
    // --kangaroo is a sub-mode of puzzle solving. Only count it as a top-level
    // mode claim when neither brainwallet nor pool is asserted (so combining
    // --kangaroo with either still flags the conflict via the != 1 check below).
    if (args.puzzle_kangaroo && (args.brainwallet_mode || args.pool_mode)) {
        active++;
        chosen.emplace_back("--kangaroo");
    }
    if ((args.puzzle_number > 0 || args.puzzle_all_unsolved) &&
        (args.brainwallet_mode || args.pool_mode)) {
        active++;
        if (args.puzzle_all_unsolved) chosen.emplace_back("--all-unsolved");
        else chosen.emplace_back("--puzzle " + std::to_string(args.puzzle_number));
    }

    if (active <= 1) return 0;

    msg = "[!] Conflicting search modes selected: ";
    for (size_t i = 0; i < chosen.size(); ++i) {
        msg += chosen[i];
        if (i + 1 < chosen.size()) msg += ", ";
    }
    msg += ".\n";
    msg += "    Pick exactly one of: --brainwallet, --pool <url>, --puzzle <N> [--kangaroo].\n";
    msg += "    --kangaroo by itself is allowed (it implies puzzle mode);\n";
    msg += "    --puzzle <N> + --kangaroo is the standard kangaroo-on-puzzle case.\n";
    return -1;
}

int parse_args_core(int argc, char* argv[], Arguments& args,
                    collider::CLIFlags& cli, std::string& err_msg) {
    args = Arguments{};
    cli = collider::CLIFlags{};

    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
        const FlagSpec* spec = find_flag(arg);
        if (!spec) {
            // Unrecognized token. A dash-prefixed token is almost certainly a
            // typo'd flag (e.g. --puzzel 71); warn loudly so a multi-day run is
            // not silently launched on the wrong target (audit MEDIUM-4).
            // Non-dash positional residue is skipped as before to keep the
            // legacy CLIParser invariants stable.
            if (!arg.empty() && arg[0] == '-') {
                std::cerr << "[!] WARNING: unrecognized option '" << arg
                          << "' ignored. Run --help for the list of valid flags.\n";
            }
            continue;
        }
        int rc = spec->apply(i, argc, argv, args, cli, err_msg);
        if (rc != 0) {
            // Negative: hard parse error; caller will print err_msg + exit.
            // Positive: Free-build early-exit code (e.g. --puzzle-only-v2
            // returns 2). Propagate either way without modifying err_msg.
            return rc;
        }
    }

    // --brute mutex: brute mode replaces the candidate-generation pipeline
    // entirely, so it can't co-exist with v2 (which expects the streaming
    // generator's multi-scheme expansion) or warpwallet (which routes
    // through a CPU-only scrypt+PBKDF2 path).
    if (!args.brute_lengths.empty()) {
        if (args.brainwallet_v2_mode) {
            err_msg = "[!] --brute is incompatible with --brainwallet-v2; the v2 multi-scheme\n"
                      "    pipeline expects the streaming generator. Pick one.\n";
            return -1;
        }
        if (!args.warpwallet_salt.empty()) {
            err_msg = "[!] --brute is incompatible with --brainwallet-warpwallet; WarpWallet\n"
                      "    routes through CPU-only scrypt and would take cosmic time on\n"
                      "    bruteforce keyspaces.\n";
            return -1;
        }
    }

    return validate_mode_mutex(args, err_msg);
}

Arguments parse_args(int argc, char* argv[], collider::CLIFlags* cli_out) {
    Arguments args;
    collider::CLIFlags cli;
    std::string err;
    int rc = parse_args_core(argc, argv, args, cli, err);
    if (rc < 0) {
        std::cerr << err;
        std::cerr << "    Run `collider --help` for usage details.\n";
        std::exit(1);
    }
    if (rc > 0) {
        // Free-build early-exit code (--puzzle-only-v2 in Free returns 2).
        std::exit(rc);
    }
    if (cli_out) *cli_out = cli;
    return args;
}

int parse_args_for_test(int argc, char* argv[], Arguments& args,
                        collider::CLIFlags& cli, std::string* err_out) {
    std::string err;
    int rc = parse_args_core(argc, argv, args, cli, err);
    if (err_out) *err_out = err;
    return rc;
}

void print_usage() {
    std::cout << R"(
theCollider - Bitcoin Puzzle Solver
===================================

GPU-accelerated solver for the Bitcoin Puzzle Challenge.

Usage:
  collider [options]
  collider --puzzle <number>

Puzzle Options:
  --puzzle, -P [N]            Target puzzle number N (default: auto-select easiest
                              unsolved). Auto-selects Kangaroo if pubkey is known
                              for that puzzle, else falls back to brute force.
  --kangaroo                  Force Pollard's Kangaroo algorithm. Only works on
                              puzzles with a revealed public key (see README).
  --dp-bits <N>               Distinguished point bits for Kangaroo (default:
                              auto-calculate). Manual override: 16-28; higher =
                              fewer DPs, lower = more DPs.
  --puzzle-target <addr>      Override target Bitcoin address.
  --puzzle-start <hex>        Override range start (hex, with 0x prefix).
  --puzzle-end <hex>          Override range end (hex, with 0x prefix).
  --pubkey <hex>              33-byte compressed public key (02/03 + 32B hex)
                              for the target. Only needed when scanning a
                              target whose pubkey isn't in puzzle_history.json
                              (rare; see README). The bundled file already
                              has every revealed pubkey for the canonical
                              Bitcoin Puzzle Challenge.
  --puzzle-checkpoint <file>  Checkpoint file for resume across runs.
  --resume-kangaroo           Resume the kangaroo herd from the auto-saved
                              checkpoint at
                              ~/.collider/state/kangaroo_herd_puzzle_<N>.kang.
                              The SIGINT save is always on; this flag controls
                              the load side.
  --puzzle-min-bits <N>       Lower bit-bound when iterating with --all-unsolved.
  --puzzle-max-bits <N>       Upper bit-bound when iterating with --all-unsolved.
  --sequential                Walk the search range from start to end.
  --random                    Random search within the range (default).
  --all-unsolved              Auto-progress through all unsolved puzzles.
  --auto-next                 After solving the current puzzle, advance to the
                              next one automatically.

GPU Options:
  --gpus, -g <ids>            GPU device IDs to use (default: auto-detect all).
  --batch-size <n>            Keys per batch (default: auto-sized from rule
                              count when running --brainwallet; otherwise
                              4000000). Override only when GPU memory is
                              constrained.
  --backend <kind>            Kangaroo backend: cpu | cuda | metal (default
                              is the most-capable available in this build:
                              cuda on Windows/Linux + NVIDIA, metal on
                              macOS, cpu as the portable fallback).
  --kangaroos <N>             Number of kangaroos (K) per GPU. 0 / unset =
                              backend default (CPU: thread-driven, MultiGPU:
                              262144, RCKangaroo: kernel-grid-driven so
                              this flag is informational there). Higher K
                              = more VRAM (linear) and faster DP rate up to
                              GPU saturation.

Output Options:
  --verbose, -v               Verbose output.

Benchmark Mode:
  --benchmark                 Run GPU performance benchmark.
  --benchmark-time <sec>      Benchmark duration in seconds (default: 30).

Calibration:
  --calibrate                 Run GPU batch size calibration (auto on first run).
  --force-calibrate           Force re-calibration even if already saved.

Smart Selection:
  --analyze                   Show puzzle analysis with ROI ranking (no search).
  --no-smart                  Disable ROI-based puzzle auto-selection; pick the
                              lowest-numbered unsolved puzzle first when multiple
                              options exist. Has no effect when --puzzle <N> is
                              set explicitly. Distinct from --sequential, which
                              controls within-range search direction.

)";
#ifdef COLLIDER_PRO
    std::cout << R"(License:
  --activate <KEY>        Activate your Pro license key (run once after purchase)
                          Example: ./collider --activate CLLDR-A1B2-C3D4-E5F6-G7H8
                          Purchase at: https://collisionprotocol.com/pro

Brainwallet Mode (PRO):
  --brainwallet           Brainwallet-only mode (requires --bloom)
                          Generates passphrases and checks against bloom filter
  --brainwallet-v2        enable multi-address bloom checking. Each
                          passphrase's priv is probed against compressed P2PKH
                          (also covers P2WPKH/BIP-84 by hash160 equivalence),
                          uncompressed P2PKH, and P2SH-P2WPKH (BIP-49). The
                          additional probes cost ~2 extra SHA256 + RIPEMD160 +
                          bloom queries per passphrase; ec_mul is reused. P2TR
                          (BIP-86) and multi-scheme priv derivation land in a
                          later release.
  --brainwallet-setup     Run the brainwallet setup wizard
                          Configure wordlists, deduplication, and PCFG training
  --bloom <file.blf>      Bloom filter file with funded addresses. Auto-detected
                          as seen.blf in CWD, ~/.collider/, or the directory
                          containing the executable when this flag is not
                          supplied.
  --bloom-tight <file.blf>  Optional second .blf used CPU-side to re-probe every
                          empty-hit candidate before logging. Pair a loose GPU
                          bloom (small, fast) with a tight CPU bloom (big,
                          accurate) so found-empty.txt isn't dominated by the
                          loose bloom's FP noise floor. Build the tight one
                          with -f 1e-7 from the same source CSV. Auto-detected
                          as seen_tight.blf in the same search paths.
  --verify-set <file.uvrf>  Companion UVRF that rejects bloom false positives
                          before they reach the user. Auto-detected as
                          <bloom>.uvrf or funded_addresses.uvrf next to the
                          bloom file when this flag is not supplied.
  --use-texture-bloom     Bind the bloom filter to a CUDA texture object for
                          the membership probe. Single-GPU only (silently
                          falls back to global memory on multi-GPU runs);
                          measurable speedup on RTX 4090+ for the
                          opportunistic-scan phase.
  --track-empty-hits      Log every bloom hit that fails the UVRF funded-set
                          lookup as a "real but empty" wallet (had on-chain
                          activity at some point, currently 0 balance) to
                          found-empty.txt. Status line shows running
                          "Empty: N" counter. Only useful when the --bloom
                          file is a SEEN-EVER bloom (addresses that had any
                          transaction history); with a funded-only bloom
                          this just logs bloom false-positives. Proof-of-life
                          for scans that find real keys after the funds
                          have already been spent. AUTO-ENABLED when
                          --bloom-tight and --verify-set both resolve
                          (CLI, config, or auto-detection); use
                          --no-track-empty-hits to force off.
  --no-track-empty-hits   Disable empty-hit logging even when auto-detection
                          would have enabled it.
  --brainwallet-warpwallet <salt>
                          Use the Keybase WarpWallet derivation
                          (scrypt+PBKDF2) instead of plain SHA-256. CPU-only
                          (scrypt is memory-hard); much slower than the
                          default brainwallet but the canonical key-stretch
                          attack on WarpWallet-style keys. <salt> is the
                          WarpWallet salt (typically the target email
                          address). Implies --brainwallet.
  --brute N [N ...]       deterministic alphanumeric bruteforce by
                          length. Replaces the wordlist + rules + PCFG +
                          Markov pipeline with a lex-ordered sweep of every
                          string of length N over [0-9A-Za-z] (62 chars).
                          Multiple lengths run shortest-first. Lengths above
                          7 are typically intractable; 62^8 alphanumeric
                          is 218 trillion candidates (~458 days at 5.5M/s).
                          Implies --brainwallet. Mutually exclusive with
                          --brainwallet-v2 and --brainwallet-warpwallet.
                          Examples:
                            --brute 6        (exhaustive 62^6 ~ 56.8B candidates)
                            --brute 5 6 7    (5, then 6, then 7 in order)
  --resume                Resume from last saved checkpoint
  --save-interval <n>     Save state every N passphrases (default: 1000000)
  --cpu-rules             Force CPU rule processing (enables multi-GPU parallelism)

)";
#else
    std::cout << R"(Brainwallet Mode:
  Brain wallet scanning is a Pro feature. This is the FREE build.
  Pro adds: --brainwallet, --brainwallet-setup, --bloom, --resume,
            --save-interval, --cpu-rules.
  Purchase at: https://collisionprotocol.com/pro

)";
#endif
    std::cout << R"(Pool Mode (Distributed Solving):
  --pool, -p <url>        Connect to pool for distributed Kangaroo solving
                          URL format: jlps://host:port (TLS, recommended)
                                      jlp://host:port (plaintext)
                          Example: jlps://pool.collisionprotocol.com:17403
  --worker, -w <address>  Worker name (your Bitcoin address for rewards)
  --pool-password-file <path>
                          Read the pool password from the first line of
                          <path>. Preferred over --pool-password because
                          argv is visible to other local users via
                          ps / Task Manager / /proc/<pid>/cmdline.
  --pool-password <pass>  Pool password supplied inline. DEPRECATED:
                          leaks via process listing; scheduled for
                          removal in v1.6.0. Use --pool-password-file
                          for production deployments.

Other:
  --help, -h              Show this help message
  --debug                 Show debug output for troubleshooting
  --perf-instrument       Enable per-kernel CUDA-event timing collector
                          for the TUI performance panel. Default off so
                          production scans pay zero overhead on the
                          kernel-launch hot path.
  --config, -c <file>     Use custom config file (default: ./config.yml)

Configuration:
  Settings can be stored in config.yml (current directory or ~/.collider/)
  Command-line arguments override config file settings.

Examples:
  # Auto-select easiest unsolved puzzle
  collider

  # Target specific puzzle
  collider --puzzle 71

  # Target puzzle with Kangaroo algorithm (if pubkey known)
  collider --puzzle 135 --kangaroo

  # Kangaroo with manual dp_bits override (for tuning)
  collider --puzzle 135 --kangaroo --dp-bits 25

  # Use random search pattern
  collider --puzzle 71 --random

  # Auto-progress through all unsolved puzzles
  collider --all-unsolved

  # Use specific GPUs
  collider --puzzle 71 --gpus 0,1

  # Run GPU benchmark
  collider --benchmark

)";
#ifdef COLLIDER_PRO
    std::cout << R"(  # Brainwallet mode - scan for compromised brainwallets (PRO)
  collider --brainwallet --bloom funded_addresses.blf

)";
#endif
    std::cout << R"(  # Custom address + range, brute force (no kangaroo because no
  # revealed pubkey for an arbitrary address; see README on which
  # puzzles ARE kangaroo-able).
  collider --puzzle-target 13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so \
           --puzzle-start 0x20000000000000000 \
           --puzzle-end   0x3ffffffffffffffff

  # Join Collision Protocol for distributed solving
  collider --pool jlps://pool.collisionprotocol.com:17403 --worker 1YourBitcoinAddress...

  # Pool mode with HTTP API
  collider --pool http://api.collisionprotocol.com --worker 1YourBitcoinAddress...

Performance: see the GitHub release notes for benchmarked numbers per
GPU architecture. The figures depend strongly on driver version, batch
size, and the specific kernel path; anything quoted here would go
stale within weeks of the next driver release.

)";
}
