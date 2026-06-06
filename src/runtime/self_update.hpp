// self_update.hpp - client self-update for theCollider pool mode.
//
// The pool server advertises the latest available client version, a download
// URL, the binary's SHA-256, and (v1.5.5+) an Ed25519 signature over the
// canonical update manifest in the AUTH_OK payload (see
// src/pool/jlp_wire_generated.hpp AuthOkPayload). After the first successful
// authentication, run_pool_mode consults the advert and, when auto-update is
// enabled and the advertised version is newer than collider::kVersion, calls
// perform_self_update(). That function FIRST verifies the manifest signature
// against the release public key compiled into the binary
// (src/runtime/signing_keys.hpp); only if the signature is valid AND the
// advertised version is strictly newer does it download the binary, verify
// its SHA-256 against the advert (NEVER installs an unverified binary), then
// self-replace the running executable and relaunch with the same argv.
//
// SECURITY (review finding C3): the signature check closes a fleet-wide RCE.
// Without it a malicious or compromised pool advertises download_url + sha256
// of its OWN binary; the client fetches over valid TLS, the (attacker-chosen)
// hash matches, and the attacker binary is installed and relaunched with the
// operator's argv. The Ed25519 signature binds (version, url, sha256) to the
// release private key the pool does NOT hold, so a forged or unsigned advert
// is rejected BEFORE any byte is fetched.
//
// Design notes:
//   * Self-contained and dependency-light: libcurl (gated on
//     COLLIDER_HAVE_CURL) for the download, OpenSSL EVP for the SHA-256 and
//     the Ed25519 verify, and std::filesystem for path handling. No
//     project-internal types leak into this interface.
//   * Verify-the-signature-before-fetch and verify-the-hash-before-install
//     are the two security boundaries. A bad signature means we never touch
//     the network; a mismatched hash means we delete the temp file and refuse
//     to install rather than running an unverified binary.
//   * Anti-rollback: a manifest whose latest_version is not strictly newer
//     than the running kVersion is refused even with a valid signature, so a
//     signed-but-stale advert cannot downgrade the fleet.
//   * On a build without libcurl the download path is compiled out and
//     perform_self_update logs a skip and returns false (mining continues on
//     the current version).

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace collider {
namespace update {

// Build the CANONICAL UPDATE MANIFEST byte string that the release key signs
// and this client re-derives + verifies. MUST be byte-identical to the server
// builder (collision-protocol/src/jlp_protocol.py::build_update_manifest) and
// the layout documented in protocol/jlp.yaml::AuthOkPayload:
//
//   DOMAIN
//     || u16le(len(latest_version)) || latest_version
//     || u16le(len(min_version))    || min_version
//     || u16le(len(download_url))   || download_url
//     || sha256
//
// where DOMAIN is the 19 ASCII bytes "COLLIDER-UPDATE-v1\n", the lengths are
// the unpadded string lengths, and sha256 is the raw 32 bytes.
std::vector<uint8_t> build_update_manifest(
    const std::string& latest_version,
    const std::string& min_version,
    const std::string& download_url,
    const std::array<uint8_t, 32>& sha256);

// Verify an Ed25519 detached signature over the canonical update manifest
// against an explicit raw 32-byte Ed25519 public key. Returns true ONLY when
// the signature is a valid Ed25519 signature for these exact manifest fields.
// A 64-zero ("unsigned") signature returns false. Requires OpenSSL; without
// it the function returns false (cannot verify -> refuse to update).
//
// This key-parameterized form exists so tests can exercise the real verify
// path with a TEST keypair without the production private key. Production code
// uses verify_update_manifest() below, which pins the compiled-in release key.
bool verify_update_manifest_with_key(
    const std::array<uint8_t, 32>& pubkey,
    const std::string& latest_version,
    const std::string& min_version,
    const std::string& download_url,
    const std::array<uint8_t, 32>& sha256,
    const std::array<uint8_t, 64>& signature);

// Verify against the release public key compiled into the binary
// (collider::keys::kReleaseSigningPubKey). This is the production gate.
bool verify_update_manifest(
    const std::string& latest_version,
    const std::string& min_version,
    const std::string& download_url,
    const std::array<uint8_t, 32>& sha256,
    const std::array<uint8_t, 64>& signature);

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

// Verify the manifest signature + anti-rollback floor, then (only if both
// pass) download download_url, verify its SHA-256 against expected_sha256,
// self-replace the running executable, and relaunch with the original argv.
//
// SECURITY GATES, in order, BEFORE any network fetch (review finding C3):
//   1. Ed25519 signature over the canonical manifest (latest_version,
//      min_version, download_url, expected_sha256) MUST verify against the
//      release public key compiled into the binary. A missing (all-zero) or
//      forged signature is rejected with NO fetch.
//   2. Anti-rollback: latest_version MUST be strictly newer than the running
//      collider::kVersion. A signed-but-stale advert is rejected.
// Only after both gates pass is the binary fetched and its SHA-256 checked
// before install.
//
// Returns true ONLY on the relaunch path having been initiated (on Windows
// the function does not actually return true: it ExitProcess(0) after
// CreateProcessW succeeds). Any failure (bad signature, rollback floor, no
// curl, download error, non-200, hash mismatch, file-move failure) logs a
// milestone, cleans up the temp file, attempts rollback if the swap was
// partial, and returns false so the caller can continue mining on the current
// version.
//
// NEVER fetches on an unverified signature and NEVER installs a binary whose
// SHA-256 does not match expected_sha256.
bool perform_self_update(const std::string& latest_version,
                         const std::string& min_version,
                         const std::string& download_url,
                         const std::array<uint8_t, 32>& expected_sha256,
                         const std::array<uint8_t, 64>& manifest_sig,
                         int argc, char** argv);

}  // namespace update
}  // namespace collider
