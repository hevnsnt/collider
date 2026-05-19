/**
 * test_mega_fused_brainwallet_absence: link-table absence trap for the
 * deleted `mega_fused_brainwallet_batch` kernel (and the entire
 * src/gpu/mega_fused_kernel.{cu,hpp} TU). T0.3.b + T3.1 (A-tier wave 1,
 * 2026-05-17); reauthored 2026-05-17 to replace the always-pass stub
 * with a real symbol-absence scan over collider_gpu.lib.
 *
 * What was deleted and why
 * ========================
 * `mega_ec_mul_windowed` (inside the deleted mega_fused_kernel.cu)
 * carried the same windowed-mul double-shift bug as
 * `ec_mul_batch_optimized_kernel` (deleted in T0.3.a): it doubled R
 * by MEGA_EC_WINDOW=5 between iterations AND added entries from a
 * precomputed table whose w-th window already encoded the 2^(5*w)
 * factor. The full mega-fused chain (rule, SHA256, EC_MUL, SHA256,
 * RIPEMD160, bloom) ran inside a persistent kernel, so any bloom
 * probe that "hit" was a spurious match because the EC step had
 * emitted the wrong pubkey for any multi-window scalar.
 *
 * The mega_fused_kernel.cu file (entire 2348-line TU plus its
 * 203-line header) had already been unhooked from collider_gpu in
 * Repair R5+R8. T0.3.b deletes the file outright because (a) nothing
 * references the symbols, (b) the kernel was an unmaintained crypto
 * landmine, and (c) the "register-budget-tuned for the persistent
 * kernel" comment no longer matters since no caller exists.
 *
 * What this test does
 * ===================
 * It scans the compiled collider_gpu.lib archive for the deleted
 * symbol names. If any of them are still present (the source file got
 * accidentally re-added, a stale .o slipped into the lib, a contributor
 * copy-pasted the body under a different file name, etc.) the test
 * fails. The lib path is supplied at compile time via the
 * COLLIDER_GPU_LIB_PATH macro that the CMake registration passes through
 * as a quoted string literal.
 *
 * Why a byte-grep over the .lib instead of a link-time check:
 * a normal link-time absence assertion ("link should fail iff the
 * symbol exists") is exactly backwards in C++. The closest mechanism
 * (a weak undefined reference paired with a static_assert that the
 * pointer is null at link time) is not portable across MSVC link and
 * GNU ld. Reading the archive's symbol table as bytes works on every
 * platform and toolchain we support and needs no build-system glue
 * beyond passing the .lib path through.
 *
 * The MSVC C++ mangling for non-extern-"C" symbols always embeds the
 * undecorated function name verbatim somewhere in the mangled string
 * (e.g. `mega_fused_brainwallet_batch` appears literally inside
 * `?mega_fused_brainwallet_batch@@YA...`). For extern "C" symbols the
 * name is the C name with an underscore on x86 (none on x64). For
 * __global__ CUDA kernels the host-visible registration name also
 * contains the undecorated symbol. Substring matching on the raw lib
 * bytes therefore catches every plausible reintroduction shape.
 *
 * If a future contributor genuinely needs the kernel back, they MUST
 * EITHER:
 *
 *   (a) Replace this absence trap with a real KAT: build a tiny
 *       wordlist, run mega_fused_brainwallet_batch with a known
 *       (priv, pub, h160) target, assert d_match_count > 0. The
 *       reference implementation is
 *       `fused_brain_wallet_batch_fixed_stride` in
 *       src/gpu/fused_pipeline.cu, which the production brain-wallet
 *       runner uses. The mega_ec_mul_windowed body MUST NOT double
 *       R between windows when the precomputed table already encodes
 *       the per-window 2^(5*w) factor; see
 *       secp256k1.cu::ec_mul_windowed for the correct shape, OR
 *
 *   (b) Delete this test (and its CMakeLists.txt registration) and
 *       trust `test_multi_gpu_brain_wallet` and `test_gpu_hash160` to
 *       cover the resurrected pipeline's correctness.
 *
 * SKIP semantics: if COLLIDER_GPU_LIB_PATH is empty (the test was built
 * without the path injection) the test returns 77 (SKIP_RETURN_CODE in
 * the CMake registration). Failing to find the lib is treated as SKIP
 * for the same reason: a developer running the test out of a fresh
 * checkout that has not yet built collider_gpu should not see a red
 * test. The CI matrix builds collider_gpu before running ctest, so the
 * SKIP path never trips on the production pipeline.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef COLLIDER_GPU_LIB_PATH
    // CMake did not inject the path. Fall through to SKIP at runtime.
    #define COLLIDER_GPU_LIB_PATH ""
#endif

namespace {

// SKIP return code matches set_tests_properties(... SKIP_RETURN_CODE 77).
constexpr int kSkip = 77;
constexpr int kPass = 0;
constexpr int kFail = 1;

// Deleted symbol substrings. Any one of these appearing in the lib bytes
// indicates the kernel has been reintroduced (or a stale object file
// slipped in). The names are the undecorated C identifiers; MSVC C++
// mangling preserves the undecorated name as a substring, so a plain
// memmem-style scan catches both extern "C" and mangled forms.
const char* const kDeletedSymbols[] = {
    "mega_fused_brainwallet_batch",
    "mega_ec_mul_windowed",
    "mega_fused_brainwallet_kernel",
};

// Read the entire file into memory. Returns true on success.
bool slurp(const std::string& path, std::vector<unsigned char>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    std::streamsize sz = f.tellg();
    if (sz < 0) {
        return false;
    }
    out.resize(static_cast<std::size_t>(sz));
    f.seekg(0, std::ios::beg);
    if (sz > 0) {
        f.read(reinterpret_cast<char*>(out.data()), sz);
        if (!f) {
            return false;
        }
    }
    return true;
}

// Byte-substring search; std::string::find on a string built from
// arbitrary binary bytes works correctly because std::string is
// length-counted, but we want to avoid the cost of copying the whole
// archive into a std::string. A trivial two-pointer scan is fast enough
// for a one-shot test: the lib is a handful of megabytes at most.
bool contains(const std::vector<unsigned char>& hay, const char* needle) {
    const std::size_t nlen = std::strlen(needle);
    if (nlen == 0 || hay.size() < nlen) {
        return false;
    }
    const std::size_t last = hay.size() - nlen;
    for (std::size_t i = 0; i <= last; ++i) {
        if (std::memcmp(hay.data() + i, needle, nlen) == 0) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    const std::string lib_path = COLLIDER_GPU_LIB_PATH;
    if (lib_path.empty()) {
        std::fprintf(stdout,
            "SKIP: COLLIDER_GPU_LIB_PATH was not provided at compile time. "
            "This test must be invoked via the CMake registration that "
            "passes -DCOLLIDER_GPU_LIB_PATH=path/to/collider_gpu.lib.\n");
        return kSkip;
    }

    std::vector<unsigned char> bytes;
    if (!slurp(lib_path, bytes)) {
        std::fprintf(stdout,
            "SKIP: could not open collider_gpu archive at %s. Build "
            "collider_gpu before invoking the absence trap.\n",
            lib_path.c_str());
        return kSkip;
    }

    std::fprintf(stdout,
        "Scanning %s (%zu bytes) for deleted mega_fused_kernel symbols...\n",
        lib_path.c_str(), bytes.size());

    int found = 0;
    for (const char* sym : kDeletedSymbols) {
        if (contains(bytes, sym)) {
            std::fprintf(stderr,
                "FAIL: deleted symbol substring `%s' is still present "
                "in %s. The mega_fused_kernel TU has been reintroduced "
                "(intentionally or via copy-paste). See the docblock at "
                "the top of this test for the contract any reintroduction "
                "must honour.\n",
                sym, lib_path.c_str());
            ++found;
        } else {
            std::fprintf(stdout, "  absent: %s\n", sym);
        }
    }

    if (found > 0) {
        std::fprintf(stderr,
            "FAIL: %d deleted symbol(s) still present in collider_gpu.lib.\n",
            found);
        return kFail;
    }

    std::fprintf(stdout,
        "PASS: every deleted mega_fused_kernel symbol is absent from "
        "collider_gpu.lib. Live brain-wallet path is "
        "fused_brain_wallet_batch_fixed_stride; covered by "
        "test_multi_gpu_brain_wallet and test_gpu_hash160.\n");
    return kPass;
}
