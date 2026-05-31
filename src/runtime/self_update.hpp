// self_update.hpp - client self-update for theCollider pool mode (v1.5.4).
//
// The pool server advertises the latest available client version, a signed
// download URL, and the binary's SHA-256 in the AUTH_OK payload (see
// src/pool/jlp_wire_generated.hpp AuthOkPayload). After the first successful
// authentication, run_pool_mode consults the advert and, when auto-update is
// enabled and the advertised version is newer than collider::kVersion, calls
// perform_self_update(). That function downloads the binary, verifies its
// SHA-256 against the advert (NEVER installs an unverified binary), then
// self-replaces the running executable and relaunches with the same argv.
//
// Design notes:
//   * Self-contained and dependency-light: libcurl (gated on
//     COLLIDER_HAVE_CURL) for the download, OpenSSL EVP for the SHA-256, and
//     std::filesystem for path handling. No project-internal types leak into
//     this interface.
//   * Verify-before-install is the security boundary. A mismatched hash means
//     the download was corrupted or tampered; we delete the temp file and
//     refuse to install rather than running an unverified binary.
//   * On a build without libcurl the download path is compiled out and
//     perform_self_update logs a skip and returns false (mining continues on
//     the current version).

#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace collider {
namespace update {

// Compare two "MAJOR.MINOR.PATCH" semver strings numerically. Returns true
// when a < b. Any non-numeric suffix (e.g. "-rc1", "+build") on a component
// is ignored gracefully: parsing stops at the first non-digit. Missing
// components are treated as 0 (so "1.5" == "1.5.0"). This is intentionally
// permissive: the advert is operator-controlled and we only need a robust
// "is the advertised version strictly newer than ours" decision.
bool semver_less(const std::string& a, const std::string& b);

// Delete any leftover update artifacts ("<exe>.old" and the temp download
// file) next to the running executable. Best-effort: swallows every error
// (a locked .old from a still-exiting prior process, a permissions issue)
// and never throws. Call once at process startup.
void cleanup_stale_update_artifacts();

// Download download_url, verify its SHA-256 against expected_sha256, then
// self-replace the running executable and relaunch with the original argv.
//
// Returns true ONLY on the relaunch path having been initiated (on Windows
// the function does not actually return true: it ExitProcess(0) after
// CreateProcessW succeeds). Any failure (no curl, download error, non-200,
// hash mismatch, file-move failure) logs a milestone, cleans up the temp
// file, attempts rollback if the swap was partial, and returns false so the
// caller can continue mining on the current version.
//
// NEVER installs a binary whose SHA-256 does not match expected_sha256.
bool perform_self_update(const std::string& download_url,
                         const std::array<uint8_t, 32>& expected_sha256,
                         int argc, char** argv);

}  // namespace update
}  // namespace collider
