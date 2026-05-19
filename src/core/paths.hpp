// paths.hpp - centralised HOME / USERPROFILE resolution and the canonical
// on-disk directory layout for theCollider.
//
// Pre-v1.4.2-quality there were 16+ open-coded copies of the
// `#ifdef _WIN32 / getenv("USERPROFILE") else getenv("HOME") endif` ladder
// scattered across src/core, src/runtime, src/pool, src/ui, src/plugins,
// src/license. The duplication was a known foot-gun: any change in the
// resolution policy (Windows fallback to HOMEDRIVE+HOMEPATH, POSIX fallback
// to getpwuid, alternative roots for sandboxed builds) had to be reapplied
// site-by-site, and divergence had already crept in (license_check.cpp had
// a getpwuid fallback that no other site used; jlp_pool_client.cpp tried
// HOME before USERPROFILE on Windows-as-MSYS hosts, while everyone else
// tried USERPROFILE first).
//
// This header is the single source of truth. New code MUST use these
// helpers; old sites should be migrated as they are touched.
//
// Layout convention (kept verbatim from the per-site copies):
//   ~/.collider/                <- collider_home()      state, config, logs
//   ~/.collider/state/          <- state_dir()          per-puzzle / per-work
//                                                       kangaroo herd snapshots
//                                                       and brain-wallet scan
//                                                       checkpoints
//   ~/.collider/config.yml      <- config_file()        user-managed YAML config
//   ~/.thecollider/             <- brainwallet_home()   brain-wallet wizard
//                                                       output dir (LEGACY name,
//                                                       kept for compat; v1.6.0
//                                                       will migrate to
//                                                       collider_home())
//
// Failure policy: when neither HOME nor USERPROFILE is set we return ".".
// This matches the pre-existing fallback used by ~13 of the 16 sites; the
// other 3 returned an empty string and treated empty as "skip this feature".
// Migrate those sites by checking `home()` against "." if a hard fail is
// desired, rather than introducing a second helper.

#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

namespace collider {
namespace paths {

// Resolves to $HOME on POSIX, %USERPROFILE% on Windows, with a
// HOMEDRIVE+HOMEPATH composite as a secondary Windows fallback (this is the
// pattern documented by Microsoft for headless service accounts that don't
// have USERPROFILE set). Returns "." as the last-resort fallback so callers
// always get a usable path; sites that prefer hard-fail semantics should
// compare the result against "." explicitly.
inline std::filesystem::path home() {
#ifdef _WIN32
    if (const char* p = std::getenv("USERPROFILE")) {
        if (p[0] != '\0') return p;
    }
    if (const char* drive = std::getenv("HOMEDRIVE")) {
        if (const char* dir = std::getenv("HOMEPATH")) {
            if (drive[0] != '\0' && dir[0] != '\0') {
                return std::string(drive) + dir;
            }
        }
    }
#else
    if (const char* p = std::getenv("HOME")) {
        if (p[0] != '\0') return p;
    }
#endif
    return ".";
}

// ~/.collider/  -- canonical state dir for solver checkpoints, calibration,
// pool DP-seq counter, recovered keys, logs.
inline std::filesystem::path collider_home() {
    return home() / ".collider";
}

// ~/.collider/state/  -- per-puzzle / per-work kangaroo herd snapshots and
// brain-wallet scan checkpoints.
inline std::filesystem::path state_dir() {
    return collider_home() / "state";
}

// ~/.collider/config.yml  -- user config (precedence: CLI > config > defaults).
inline std::filesystem::path config_file() {
    return collider_home() / "config.yml";
}

// ~/.thecollider/  -- brain-wallet wizard output dir (legacy name, kept for
// compat with the wizard's documented behaviour). Brain-wallet wizard TUs
// should consume from here. v1.6.0 will migrate to collider_home().
inline std::filesystem::path brainwallet_home() {
    return home() / ".thecollider";
}

}  // namespace paths
}  // namespace collider
