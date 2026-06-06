// self_update.cpp - see self_update.hpp for the contract.

#include "self_update.hpp"

#include "../core/session_log.hpp"  // milestone()
#include "../core/version.hpp"      // kVersion (anti-rollback floor)
#include "signing_keys.hpp"         // kReleaseSigningPubKey

#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#ifdef COLLIDER_HAVE_CURL
#include <curl/curl.h>
#endif

// OpenSSL EVP for streaming SHA-256 and Ed25519 signature verification.
// collider_core links OpenSSL::Crypto and defines COLLIDER_HAS_OPENSSL when
// it is available.
#ifdef COLLIDER_HAS_OPENSSL
#include <openssl/evp.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#include <climits>
#endif

namespace collider {
namespace update {

namespace fs = std::filesystem;

namespace {

// Temp download file lives next to the running executable. The name carries
// a per-run random suffix (kTempPrefix + hex + kTempSuffix) so an attacker
// who can write the install directory cannot pre-plant a symlink at a
// predictable path to redirect our write. cleanup_stale_update_artifacts
// removes any leftover file matching the prefix+suffix. (The sha256 verify
// is the real RCE gate; this just closes a predictable-path / symlink game.)
constexpr const char* kTempPrefix = "collider_update.";
constexpr const char* kTempSuffix = ".tmp";
// Suffix appended to the running executable when we move it aside.
constexpr const char* kOldSuffix = ".old";

// Build a per-run temp file name with an unpredictable random component.
std::string make_temp_update_name() {
    std::random_device rd;
    uint64_t r = (static_cast<uint64_t>(rd()) << 32) ^ rd();
    char hex[17];
    std::snprintf(hex, sizeof(hex), "%016llx",
                  static_cast<unsigned long long>(r));
    return std::string(kTempPrefix) + hex + kTempSuffix;
}

// True when `name` is one of our temp download files (prefix + suffix).
bool is_temp_update_name(const std::string& name) {
    const std::string pfx = kTempPrefix;
    const std::string sfx = kTempSuffix;
    return name.size() >= pfx.size() + sfx.size()
        && name.compare(0, pfx.size(), pfx) == 0
        && name.compare(name.size() - sfx.size(), sfx.size(), sfx) == 0;
}

// Resolve the absolute path of the running executable. Returns an empty
// path on failure (the caller treats that as "cannot self-update").
fs::path running_executable_path(char** argv) {
#ifdef _WIN32
    std::wstring buf;
    buf.resize(MAX_PATH);
    for (;;) {
        DWORD n = GetModuleFileNameW(nullptr, buf.data(),
                                     static_cast<DWORD>(buf.size()));
        if (n == 0) return {};
        if (n < buf.size()) {
            buf.resize(n);
            return fs::path(buf);
        }
        // Truncated: grow and retry.
        buf.resize(buf.size() * 2);
    }
#else
    char buf[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return fs::path(buf);
    }
    // Fallback: argv[0]. Best-effort; may be relative.
    if (argv && argv[0]) {
        std::error_code ec;
        fs::path p = fs::absolute(fs::path(argv[0]), ec);
        if (!ec) return p;
    }
    return {};
#endif
}

// Hex-encode raw bytes (lowercase) for log messages.
std::string to_hex(const uint8_t* data, size_t len) {
    static const char* digits = "0123456789abcdef";
    std::string out;
    out.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out.push_back(digits[(data[i] >> 4) & 0xF]);
        out.push_back(digits[data[i] & 0xF]);
    }
    return out;
}

#ifdef COLLIDER_HAVE_CURL
// libcurl write callback: append received bytes to an open std::ofstream.
std::size_t curl_write_file(void* ptr, std::size_t size, std::size_t nmemb,
                            void* userdata) {
    auto* ofs = static_cast<std::ofstream*>(userdata);
    const std::size_t total = size * nmemb;
    ofs->write(static_cast<const char*>(ptr), static_cast<std::streamsize>(total));
    if (!ofs->good()) return 0;  // signal write error to libcurl
    return total;
}
#endif  // COLLIDER_HAVE_CURL

#ifdef COLLIDER_HAS_OPENSSL
// Stream-read `path` and compute its SHA-256. Returns true on success and
// fills `out`. Returns false on file-open or digest error.
bool sha256_file(const fs::path& path, std::array<uint8_t, 32>& out) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) return false;
    bool ok = (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) == 1);

    std::vector<char> buf(64 * 1024);
    while (ok && ifs.good()) {
        ifs.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        std::streamsize got = ifs.gcount();
        if (got > 0) {
            ok = (EVP_DigestUpdate(ctx, buf.data(),
                                   static_cast<size_t>(got)) == 1);
        }
    }
    if (ifs.bad()) ok = false;

    unsigned int outlen = 0;
    if (ok) {
        ok = (EVP_DigestFinal_ex(ctx, out.data(), &outlen) == 1) &&
             (outlen == out.size());
    }
    EVP_MD_CTX_free(ctx);
    return ok;
}
#endif  // COLLIDER_HAS_OPENSSL

// True when every byte of the expected hash is zero (advert says
// auto-update is disabled / no hash published).
bool sha256_is_all_zero(const std::array<uint8_t, 32>& h) {
    for (uint8_t b : h) {
        if (b != 0) return false;
    }
    return true;
}

// True when the 64-byte signature is all zero (the server's "unsigned"
// sentinel). A zero signature must be rejected, never verified.
bool sig_is_all_zero(const std::array<uint8_t, 64>& s) {
    for (uint8_t b : s) {
        if (b != 0) return false;
    }
    return true;
}

// Domain separator for the canonical update manifest. MUST match the server
// (collision-protocol/src/jlp_protocol.py::UPDATE_MANIFEST_DOMAIN) and the
// layout in protocol/jlp.yaml::AuthOkPayload. 19 ASCII bytes including the
// trailing newline.
constexpr char kManifestDomain[] = "COLLIDER-UPDATE-v1\n";
// sizeof includes the implicit NUL terminator, which is NOT part of the
// domain bytes; subtract it.
constexpr size_t kManifestDomainLen = sizeof(kManifestDomain) - 1;

// Append a little-endian uint16 length prefix followed by `s` to `out`. The
// canonical manifest length-prefixes every variable field so no two distinct
// (version, url, hash) tuples can collide on the same signed bytes.
void append_lp_field(std::vector<uint8_t>& out, const std::string& s) {
    const uint16_t n = static_cast<uint16_t>(s.size());
    out.push_back(static_cast<uint8_t>(n & 0xFF));
    out.push_back(static_cast<uint8_t>((n >> 8) & 0xFF));
    out.insert(out.end(), s.begin(), s.end());
}

}  // namespace

bool semver_less(const std::string& a, const std::string& b) {
    // Parse up to three numeric components from each string. Parsing of a
    // component stops at the first non-digit, so "1.5.4-rc1" yields
    // {1,5,4}. Missing components default to 0.
    const auto parse = [](const std::string& s, int out[3]) {
        out[0] = out[1] = out[2] = 0;
        size_t i = 0;
        for (int comp = 0; comp < 3; ++comp) {
            // Accumulate digits for this component.
            long value = 0;
            bool any = false;
            while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
                value = value * 10 + (s[i] - '0');
                if (value > 1000000000L) value = 1000000000L;  // clamp
                any = true;
                ++i;
            }
            out[comp] = any ? static_cast<int>(value) : 0;
            // Skip a single separator (the '.'); skip any other non-digit
            // suffix so the next component reads 0.
            while (i < s.size() && s[i] != '.' &&
                   !std::isdigit(static_cast<unsigned char>(s[i]))) {
                ++i;
            }
            if (i < s.size() && s[i] == '.') {
                ++i;
            } else {
                // No more separators: remaining components stay 0.
                // Keep scanning only if more digits follow directly, which
                // they will not after a non-dot, so break here.
                if (i >= s.size()) break;
            }
        }
    };

    int va[3];
    int vb[3];
    parse(a, va);
    parse(b, vb);
    for (int i = 0; i < 3; ++i) {
        if (va[i] != vb[i]) return va[i] < vb[i];
    }
    return false;  // equal
}

std::vector<uint8_t> build_update_manifest(
    const std::string& latest_version,
    const std::string& min_version,
    const std::string& download_url,
    const std::array<uint8_t, 32>& sha256) {
    std::vector<uint8_t> m;
    m.reserve(kManifestDomainLen + 6 + latest_version.size() +
              min_version.size() + download_url.size() + 32);
    m.insert(m.end(), kManifestDomain, kManifestDomain + kManifestDomainLen);
    // The server signer (jlp_protocol.build_update_manifest) caps these fields
    // before signing: version strings to 16 bytes, the URL to 256. The client
    // MUST apply the identical caps before verifying or an over-length field
    // would produce different canonical bytes here than the server signed,
    // breaking verification of an otherwise-valid manifest.
    append_lp_field(m, latest_version.substr(0, 16));
    append_lp_field(m, min_version.substr(0, 16));
    append_lp_field(m, download_url.substr(0, 256));
    m.insert(m.end(), sha256.begin(), sha256.end());
    return m;
}

bool verify_update_manifest_with_key(
    const std::array<uint8_t, 32>& pubkey,
    const std::string& latest_version,
    const std::string& min_version,
    const std::string& download_url,
    const std::array<uint8_t, 32>& sha256,
    const std::array<uint8_t, 64>& signature) {
    // A zero ("unsigned") signature never verifies. Reject before touching
    // OpenSSL so the all-zero advert is an unambiguous "no signature".
    if (sig_is_all_zero(signature)) {
        return false;
    }
#ifdef COLLIDER_HAS_OPENSSL
    const std::vector<uint8_t> manifest = build_update_manifest(
        latest_version, min_version, download_url, sha256);

    // Wrap the raw 32-byte Ed25519 public key.
    EVP_PKEY* pkey = EVP_PKEY_new_raw_public_key(
        EVP_PKEY_ED25519, nullptr, pubkey.data(), pubkey.size());
    if (!pkey) {
        return false;
    }

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        EVP_PKEY_free(pkey);
        return false;
    }

    bool ok = false;
    // Ed25519 is a single-shot (PureEdDSA) verify: DigestVerifyInit with a
    // null md, then one DigestVerify over the whole message.
    if (EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, pkey) == 1) {
        const int rc = EVP_DigestVerify(
            ctx, signature.data(), signature.size(),
            manifest.data(), manifest.size());
        ok = (rc == 1);
    }

    EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(pkey);
    return ok;
#else
    // Without OpenSSL we cannot verify a signature; refuse to update rather
    // than trusting an unverified manifest.
    (void)pubkey;
    (void)latest_version;
    (void)min_version;
    (void)download_url;
    (void)sha256;
    return false;
#endif  // COLLIDER_HAS_OPENSSL
}

bool verify_update_manifest(
    const std::string& latest_version,
    const std::string& min_version,
    const std::string& download_url,
    const std::array<uint8_t, 32>& sha256,
    const std::array<uint8_t, 64>& signature) {
    return verify_update_manifest_with_key(
        ::collider::keys::kReleaseSigningPubKey,
        latest_version, min_version, download_url, sha256, signature);
}

void cleanup_stale_update_artifacts() {
    std::error_code ec;
    fs::path exe = running_executable_path(nullptr);
    if (exe.empty()) return;

    fs::path dir = exe.parent_path();
    // Remove "<exe>.old" if present.
    fs::path old = exe;
    old += kOldSuffix;
    if (fs::exists(old, ec)) {
        fs::remove(old, ec);  // best-effort; ignore failure (may be locked)
    }
    // Remove any temp download file a prior (crashed) run left behind. Names
    // are randomized per run, so match by the known prefix+suffix.
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        if (is_temp_update_name(it->path().filename().string())) {
            fs::remove(it->path(), ec);
        }
    }
}

bool perform_self_update(const std::string& latest_version,
                         const std::string& min_version,
                         const std::string& download_url,
                         const std::array<uint8_t, 32>& expected_sha256,
                         const std::array<uint8_t, 64>& manifest_sig,
                         int /*argc*/, char** argv) {
    // ---- Security gates (run on EVERY build, before any fetch) ------------
    // These checks do not depend on libcurl: even a no-curl build must reject
    // a forged/stale manifest rather than fall through to a "skipped" log that
    // could be mistaken for "the advert was fine".
    if (download_url.empty() || sha256_is_all_zero(expected_sha256)) {
        ::collider::log::milestone("self_update_skipped",
                                   "no_url_or_hash");
        return false;
    }

    // Gate 1: signature MUST verify against the compiled-in release pubkey
    // over the exact canonical manifest. A missing (all-zero) or forged
    // signature is rejected here, BEFORE any network fetch (review C3).
    if (!verify_update_manifest(latest_version, min_version, download_url,
                                expected_sha256, manifest_sig)) {
        ::collider::log::milestone(
            "self_update_rejected",
            "manifest signature invalid or missing; refusing to fetch");
        return false;
    }

    // Gate 2: anti-rollback. Even a validly signed manifest may not downgrade
    // (or reinstall) the running version. Require latest_version strictly
    // newer than kVersion.
    if (!semver_less(std::string(::collider::kVersion), latest_version)) {
        ::collider::log::milestone(
            "self_update_rejected",
            std::string("anti-rollback: advertised ") + latest_version +
                " is not newer than running " + ::collider::kVersion);
        return false;
    }

#ifndef COLLIDER_HAVE_CURL
    (void)argv;
    ::collider::log::milestone("self_update_skipped", "no_curl");
    return false;
#else
    fs::path exe = running_executable_path(argv);
    if (exe.empty()) {
        ::collider::log::milestone("self_update_failed",
                                   "cannot resolve own executable path");
        return false;
    }
    fs::path dir = exe.parent_path();
    fs::path tmp = dir / make_temp_update_name();

    std::error_code ec;
    fs::remove(tmp, ec);  // clear any stale temp before downloading

    // ---- Download ---------------------------------------------------------
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs) {
            ::collider::log::milestone("self_update_failed",
                                       "cannot open temp file for write");
            return false;
        }

        CURL* curl = curl_easy_init();
        if (!curl) {
            ofs.close();
            fs::remove(tmp, ec);
            ::collider::log::milestone("self_update_failed", "curl_init");
            return false;
        }

        curl_easy_setopt(curl, CURLOPT_URL, download_url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_file);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ofs);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
        // Restrict the transfer (and any redirect) to HTTPS only. Without
        // this libcurl would happily follow a file://, scp://, or plaintext
        // http:// URL or redirect. The sha256 pin still gates execution, but
        // an unconstrained scheme needlessly widens the fetch surface (local
        // file exfil via file://, MITM-able plaintext). CURLOPT_PROTOCOLS_STR
        // is curl 7.85+; fall back to the integer bitmask on older libcurl.
#if defined(CURLOPT_PROTOCOLS_STR)
        curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "https");
        curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS_STR, "https");
#else
        curl_easy_setopt(curl, CURLOPT_PROTOCOLS, CURLPROTO_HTTPS);
        curl_easy_setopt(curl, CURLOPT_REDIR_PROTOCOLS, CURLPROTO_HTTPS);
#endif
        // TLS verification ON (peer + host). Never install over a binary
        // fetched from a server we could not authenticate.
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);  // 5 min for a binary
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "theCollider-selfupdate/1");
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 0L);  // we check code below

        const CURLcode rc = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        curl_easy_cleanup(curl);
        ofs.flush();
        ofs.close();

        if (rc != CURLE_OK) {
            fs::remove(tmp, ec);
            ::collider::log::milestone(
                "self_update_failed",
                std::string("download_error: ") + curl_easy_strerror(rc));
            return false;
        }
        if (http_code != 200) {
            fs::remove(tmp, ec);
            ::collider::log::milestone(
                "self_update_failed",
                std::string("http_status=") + std::to_string(http_code));
            return false;
        }
    }

    // ---- Verify SHA-256 ---------------------------------------------------
#ifdef COLLIDER_HAS_OPENSSL
    std::array<uint8_t, 32> got{};
    if (!sha256_file(tmp, got)) {
        fs::remove(tmp, ec);
        ::collider::log::milestone("self_update_failed", "sha256_compute");
        return false;
    }
    if (got != expected_sha256) {
        fs::remove(tmp, ec);
        ::collider::log::milestone(
            "self_update_verify_failed",
            "got=" + to_hex(got.data(), got.size()) +
                " want=" + to_hex(expected_sha256.data(),
                                  expected_sha256.size()));
        return false;
    }
#else
    // Without OpenSSL we cannot verify the binary. Refuse to install rather
    // than running an unverified executable.
    fs::remove(tmp, ec);
    ::collider::log::milestone("self_update_failed", "no_openssl_cannot_verify");
    return false;
#endif  // COLLIDER_HAS_OPENSSL

    // ---- Self-replace + relaunch -----------------------------------------
#ifdef _WIN32
    std::wstring wexe = exe.wstring();
    std::wstring wold = wexe + L".old";
    std::wstring wtmp = tmp.wstring();

    // Move the running executable aside (Windows allows renaming a running
    // image but not deleting / overwriting it in place).
    if (!MoveFileExW(wexe.c_str(), wold.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        fs::remove(tmp, ec);
        ::collider::log::milestone(
            "self_update_failed",
            "move_running_aside_failed err=" +
                std::to_string(GetLastError()));
        return false;
    }
    // Move the verified download into the executable's place.
    if (!MoveFileExW(wtmp.c_str(), wexe.c_str(), MOVEFILE_REPLACE_EXISTING)) {
        DWORD err = GetLastError();
        // Roll back: restore the original executable.
        MoveFileExW(wold.c_str(), wexe.c_str(), MOVEFILE_REPLACE_EXISTING);
        fs::remove(tmp, ec);
        ::collider::log::milestone(
            "self_update_failed",
            "install_new_failed err=" + std::to_string(err));
        return false;
    }

    // Relaunch with the original command line (preserves all args). The new
    // process starts fresh; this one exits so the OS releases the .old file
    // (cleaned up on the next startup by cleanup_stale_update_artifacts).
    STARTUPINFOW si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    // CreateProcessW may modify the command-line buffer, so copy it.
    std::wstring cmdline = GetCommandLineW();
    std::vector<wchar_t> cmd_buf(cmdline.begin(), cmdline.end());
    cmd_buf.push_back(L'\0');

    if (!CreateProcessW(wexe.c_str(), cmd_buf.data(), nullptr, nullptr, FALSE,
                        0, nullptr, nullptr, &si, &pi)) {
        DWORD err = GetLastError();
        // Roll back so the operator is left with a working binary.
        MoveFileExW(wexe.c_str(), wtmp.c_str(), MOVEFILE_REPLACE_EXISTING);
        MoveFileExW(wold.c_str(), wexe.c_str(), MOVEFILE_REPLACE_EXISTING);
        ::collider::log::milestone(
            "self_update_failed",
            "relaunch_failed err=" + std::to_string(err));
        return false;
    }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    ::collider::log::milestone("self_update_relaunch", "new binary launched");
    // Force-flush logs before we leave (milestone() already flushes the
    // session log line). Exit cleanly so the parent shell sees a normal exit.
    ExitProcess(0);
    return true;  // unreachable
#else
    // POSIX: rename self aside, move the verified file in, chmod +x, execv.
    fs::path old = exe;
    old += kOldSuffix;
    if (::rename(exe.c_str(), old.c_str()) != 0) {
        fs::remove(tmp, ec);
        ::collider::log::milestone("self_update_failed",
                                   "rename_running_aside_failed");
        return false;
    }
    if (::rename(tmp.c_str(), exe.c_str()) != 0) {
        // Roll back.
        ::rename(old.c_str(), exe.c_str());
        fs::remove(tmp, ec);
        ::collider::log::milestone("self_update_failed",
                                   "install_new_failed");
        return false;
    }
    if (::chmod(exe.c_str(), 0755) != 0) {
        // Not fatal to the swap, but the new binary may not be executable.
        // Try to roll back to leave a working state.
        ::rename(exe.c_str(), tmp.c_str());
        ::rename(old.c_str(), exe.c_str());
        ::collider::log::milestone("self_update_failed", "chmod_failed");
        return false;
    }
    ::collider::log::milestone("self_update_relaunch", "execv new binary");
    ::execv(exe.c_str(), argv);
    // execv only returns on failure.
    ::collider::log::milestone("self_update_failed", "execv_returned");
    return false;
#endif  // _WIN32
#endif  // COLLIDER_HAVE_CURL
}

}  // namespace update
}  // namespace collider
