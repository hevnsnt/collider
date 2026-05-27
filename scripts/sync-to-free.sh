#!/usr/bin/env bash
#
# sync-to-free.sh — publish the Free distribution of theCollider to the
# public repository at github.com/hevnsnt/collider.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# This (private) repository at github.com/hevnsnt/collider-pro carries
# both Free and Pro sources for collider. The Pro source files (brain
# wallet pipeline, license check, mega-fused kernel, etc.) MUST NOT
# appear in the public Free repo. The path-exclusion list in PRO_PATHS
# below is the single source of truth for what stays private.
#
# Previously the two repos drifted because there was no automated path
# from a private commit to a public release. This script is that path.
#
# WHAT IT DOES (single deterministic operation)
# ---------------------------------------------
# 1. Clone the public Free repo (PUBLIC_URL) to a temp dir.
# 2. Snapshot the files the public repo owns and the private repo does
#    not (LICENSE, build_macos.sh, public docs, public CI workflow).
# 3. Wipe the temp tree (preserving .git).
# 4. Copy the private working tree over, EXCLUDING the Pro-only paths.
# 5. Restore the preserved public-only files.
# 6. Commit, tag, push -- on the public Free repo.
#
# USAGE
# -----
#   scripts/sync-to-free.sh <version-tag>
#
#   $ scripts/sync-to-free.sh v1.2.1-free
#
# Re-running with the same tag is a no-op if nothing changed; otherwise
# the existing tag is reused (no force-tag).
#
# REQUIREMENTS
#   - git (for ls-files + commit/push)
#   - SSH or HTTPS auth that can push to PUBLIC_URL.
#
# Cross-platform note: copy uses git ls-files + per-file `cp`, so the
# script works on Linux/macOS/Git-Bash-on-Windows without rsync.

set -euo pipefail

PUBLIC_URL="${COLLIDER_FREE_REPO_URL:-https://github.com/hevnsnt/collider.git}"
PRIVATE_DIR="${COLLIDER_PRIVATE_DIR:-$(git rev-parse --show-toplevel)}"

# -----------------------------------------------------------------------------
# Pro-only paths -- never appear in the public Free repo.
# -----------------------------------------------------------------------------
# Single source of truth. If you add a Pro source file, add it here in
# the SAME commit, otherwise the next sync will leak it to the public.
PRO_PATHS=(
    # Brain-wallet pipeline (host + GPU)
    "src/generators/"
    "src/license/"
    "src/rules/"            # gpu_rules.hpp lives here in our tree
    "src/scrapers/"         # lyrics/quotes scrapers for brain wallet

    # Brain-wallet state persistence (save/resume across runs). No
    # #ifdef guard but has no free equivalent and is only included by
    # brain_wallet_runner (excluded below) and brain_wallet_setup.
    "src/core/brainwallet_state.hpp"

    "src/runtime/brain_wallet_runner.hpp"
    "src/runtime/brain_wallet_runner.cpp"

    # First-run brain-wallet setup wizard (runtime variant). Pulls in
    # src/generators/streaming_brain_wallet.hpp (excluded via generators/)
    # and src/runtime/scan_state.hpp (excluded below). Shipping either
    # file to Free would break the Free build on its first inclusion.
    "src/runtime/brain_wallet_setup.hpp"
    "src/runtime/brain_wallet_setup.cpp"

    # License gate: pre-dispatch activation + startup validation. Whole
    # TU is COLLIDER_PRO-only (consumes src/license/license_check.hpp,
    # which is itself excluded via the src/license/ prefix above). Without
    # explicit entries here the next sync would publish the orchestration
    # source AND break the Free build (license_gate.cpp would reference
    # the missing license_check.hpp).
    "src/runtime/license_gate.hpp"
    "src/runtime/license_gate.cpp"

    # First-run brainwallet setup wizard. Pulls in src/generators/pcfg.hpp
    # (excluded via the src/generators/ prefix), so leaving this in the
    # Free tree would break Free compilation. Pro-only by content.
    "src/ui/brainwallet_setup.hpp"

    # Brain-wallet TUI infrastructure (added during brainwallet TUI
    # overhaul). These are consumed exclusively by brain_wallet_runner;
    # main.cpp, puzzle_solver.cpp, and pool_solver.cpp do not include
    # them. Verified by grep at sync-script extension time.
    #
    # NOTE: src/runtime/runtime_control.hpp was previously listed here
    # but it is ALSO included (hard #include, not just via the TUI
    # layer) by free-side files: pool_solver.cpp (rc.banner_text +
    # banner_mu for status messages), puzzle_solver_bsgs.cpp,
    # puzzle_solver_helpers.cpp, and main.cpp (state reset between
    # interactive-menu picks). Removing it broke the free build at
    # v1.5.0 first sync ("fatal error: runtime/runtime_control.hpp:
    # No such file or directory" in pool_solver.cpp). The header
    # itself only depends on standard library + gpu_caps.hpp (free)
    # so it is free-compatible by content.
    "src/runtime/scan_state.hpp"
    "src/runtime/runtime_config_yaml.hpp"
    "src/runtime/runtime_config_yaml.cpp"
    "src/runtime/perf_instrumentation.hpp"
    "src/runtime/perf_instrumentation.cpp"
    "src/runtime/empty_hit_writer.hpp"
    "src/runtime/empty_hit_writer.cpp"

    # Interactive menu + TUI settings sidecar (added v1.5.0).
    # interactive_ui.cpp hard-#includes ui/tui/menu/main_menu.hpp which
    # lives under the excluded src/ui/tui/ prefix; settings_sidecar.hpp
    # hard-#includes ui/tui/panels/settings_panel.hpp. Free's pool /
    # puzzle / benchmark code paths do not need either: in free, mode
    # selection comes from CLI flags (--puzzle N, --pool, --benchmark)
    # rather than an interactive menu, and the TUI settings sidecar
    # only persists state for the brain-wallet dashboard. Free-side
    # callers in main.cpp / pool_solver.cpp / puzzle_solver_kangaroo.cpp
    # are #ifdef COLLIDER_PRO-gated.
    "src/ui/interactive_ui.cpp"
    "src/ui/interactive_ui.hpp"
    "src/core/settings_sidecar.hpp"

    # Subprocess plugin runner: PRO-only feature wired into brain-wallet
    # pipeline (hit-feed plugins for Slack notify, balance enrich, etc.).
    "src/plugins/"

    # Brain-wallet multi-panel FTXUI interface
    "src/ui/tui/"

    # v2 multi-scheme weak-PRNG kernel (Milk Sad, Profanity, Trust Wallet)
    "src/gpu/v2/"

    # GPU telemetry: NVML wrapper + Metal IOReport reach-around. These
    # power the brain-wallet GPU panel (fan, power, temp, clock). Not
    # used by Free puzzle/pool modes.
    "src/platform/nvml_query.hpp"
    "src/platform/nvml_query.cpp"
    "src/platform/gpu_telemetry.hpp"
    "src/platform/gpu_telemetry_cuda.cpp"
    "src/platform/gpu_telemetry_metal.mm"

    "src/gpu/brain_wallet_gpu.cpp"
    "src/gpu/brain_wallet_gpu.hpp"
    "src/gpu/gpu_rules.cpp"
    "src/gpu/gpu_rules.cu"
    "src/gpu/gpu_rules.hpp"
    "src/gpu/gpu_rule_kernel.cu"
    "src/gpu/h160_bloom_filter.cu"
    "src/gpu/bloom_filter.cu"
    "src/gpu/fused_pipeline.cu"

    # v1.5.0: BIP scanner runtime + GPU dispatcher + BIP-32/39 helpers.
    # The runner iterates BIP-39 candidate phrases and derives BIP-32
    # children across historical + modern derivation paths (pre-BIP-44,
    # Electrum, MultiBit, BIP-44/49/84); the GPU dispatcher routes the
    # per-pubkey work through MultiAddressSession (secp256k1 + hash160
    # + bloom probe). Pro-only; main.cpp's run_bip_scan_mode dispatch
    # is already gated by #ifdef COLLIDER_PRO but the SOURCE files
    # would otherwise leak proprietary scan strategy + derivation
    # plumbing to the free repo and be re-enabled by any fork flipping
    # COLLIDER_PRO=ON. The supporting GPU primitives (bip39_pbkdf2,
    # hmac_sha512_device, sha512_device) are exclusively consumed by
    # BIP scanner code; verified by grep at sync-script extension time
    # (no bench_pipeline / warpwallet / kangaroo TU includes them).
    "src/core/bip32.hpp"
    "src/core/bip39.hpp"
    "src/gpu/bip39_pbkdf2.cu"
    "src/gpu/bip39_pbkdf2.cuh"
    "src/gpu/hmac_sha512_device.cuh"
    "src/gpu/sha512_device.cuh"
    "src/runtime/bip_address.hpp"
    "src/runtime/bip_gpu_dispatcher.cpp"
    "src/runtime/bip_gpu_dispatcher.hpp"
    "src/runtime/bip_scanner_runner.cpp"
    "src/runtime/bip_scanner_runner.hpp"

    # BIP scanner test corpus. Each test links collider_core which in
    # the free build does NOT have the BIP runtime, so leaving these
    # in would either fail to link or compile to empty tests. The
    # CMakeLists already guards each entry with if(EXISTS); shipping
    # only the harness without the targets just adds dead files.
    "tests/test_bip32_kat.cpp"
    "tests/test_bip39_validate.cpp"
    "tests/test_bip39_pbkdf2_kat.cpp"
    "tests/test_bip39_pbkdf2_gpu_kat.cpp"
    "tests/test_bip_gpu_dispatcher.cpp"
    "tests/test_bip49_p2sh_p2wpkh_kat.cpp"
    "tests/test_bip_scan_runner_smoke.cpp"
    "tests/test_device_hmac_sha512.cu"

    # Tests that #include Pro-only headers. Without this list the test
    # source file lands in free but the header it depends on does not,
    # so any `cmake -DCOLLIDER_BUILD_TESTS=ON` against the free tree
    # explodes with "Cannot open include file" on every one of these.
    # The CMakeLists.txt also gates most of them on COLLIDER_PRO, but
    # gating only helps when the file is also absent -- otherwise the
    # source still ships, an EXISTS-check guard still fires, and a
    # casual cmake -B build trips the same C1083 / fatal error.
    # See issue #6 (free repo) for the precise failure mode that
    # triggered this exclusion. Defense in depth: tests for Pro-only
    # subsystems live only in the Pro tree.
    "tests/test_priority_queue.cpp"
    "tests/test_kangaroo_mode_asymmetric.cu"
    "tests/test_brute_resume_state.cpp"
    "tests/test_bruteforce_generator.cpp"
    "tests/test_empty_hit_writer.cpp"
    "tests/test_fused_pipeline_oob.cu"
    "tests/test_generator_budget.cpp"
    "tests/test_generator_modes.cpp"
    "tests/test_input_handler.cpp"
    "tests/test_license_cache.cpp"
    "tests/test_nvml_wrapper.cpp"
    "tests/test_perf_instrumentation.cpp"
    "tests/test_phase_change_fault.cu"
    "tests/test_plugin_runner.cpp"
    "tests/test_resume_iteration_mode.cpp"
    "tests/test_runtime_yaml.cpp"
    "tests/test_scan_state_atomics.cpp"
    "tests/test_sparkline.cpp"
    "tests/test_tui_panels.cpp"

    # Brain-wallet rule files + scraper outputs at repo root
    "rules/"
    "scrapers/"

    # Brain-wallet wordlist data + processing tooling
    "data/"
    "tools/"
    "processed/"

    # Protocol IDL: source of truth for JLP wire format. The generated
    # C++ header at src/pool/jlp_wire_generated.hpp is the only artifact
    # Free needs; the IDL + codegen tool stay private.
    "protocol/"

    # Test trees that touch private generators / kernels. Keep the
    # protocol smoke tests though (added to PRESERVE_PATHS below).
    "tests/protocol/"

    # Commercial website (collisionprotocol.com -- Stripe, NextAuth,
    # license issuance, Firebase functions). Not part of the CLI tool.
    "website/"

    # All documentation under docs/ is intentionally PUBLIC. The Pro
    # repository itself is private, so there is no audience for "Pro-
    # internal" docs. Anything genuinely Pro-only as content belongs
    # in a paying-customer dashboard or release notes, not in a repo.
    # Historical Pro-internal docs (PRO-MIGRATION.md describing a
    # completed migration; review-2026-05-04/ adversarial review whose
    # findings shipped in v1.4.0 / v1.4.1) were deleted; CRYPTO-
    # VALIDATION.md is general engineering content and now ships to
    # the public Free repo unchanged.

    # Local-build helpers + scratch state, never publish
    "build/"
    "build-*/"
    "build_*/"
    "*.pot"
    "*_hits.txt"
    "utxodump.csv"
    "funded_addresses.blf"

    # Pro-side CI plumbing: these workflows orchestrate the pro->free
    # sync itself and the JLP protocol push to collision-protocol.
    # They have no place in the free repo (no source repo to pull from,
    # no COLLIDER_SYNC_DEPLOY_KEY secret), and leaving them on the free
    # side would also re-trigger sync-free.yml on every tag pushed to free.
    ".github/workflows/sync-free.yml"
    ".github/workflows/sync-protocol.yml"

    # Pro's build-release.yml builds Pro binaries (-DCOLLIDER_PRO=ON).
    # The free repo needs its own variant. It lives in scripts/templates/ and
    # is installed by the sync script below as .github/workflows/build-release.yml.
    # Exclude the pro version so it never lands in the free tree.
    ".github/workflows/build-release.yml"

    # Template directory: internal to the sync toolchain; never published.
    "scripts/templates/"

    # Internal pre-release audit + pen-test + team-plan markdown at repo
    # root. These are scratch artifacts from the v1.5.0 GA review wave
    # (and successors). They contain internal threat-model notes,
    # auditor names, and unredacted finding IDs that we do not surface
    # to the public free repo. Glob form so future passes (AUDIT_V1_6_*,
    # TEAM_PLAN_V1_5_1_*, PENTEST_*) are excluded automatically without
    # another sync-script edit.
    "AUDIT_*.md"
    "PENTEST_*.md"
    "TEAM_PLAN_*.md"
)

# -----------------------------------------------------------------------------
# Public-only paths -- the public Free repo owns these, do not overwrite.
# -----------------------------------------------------------------------------
# LICENSE was previously in this list because the Free repo had been
# bootstrapped with an MIT LICENSE and the Pro tree had no LICENSE at all,
# so preserving the public file was the only sane behavior at sync time.
# That state was a GPL violation: the Free distribution statically links
# the GPLv3-licensed third_party/RCKangaroo/ source, so the Free binary
# distribution (and therefore the Free source tree) must be GPLv3, not
# MIT. The Pro tree now carries a GPLv3 LICENSE at the root that names
# SixCyber LLC as the copyright holder of the original code and references
# THIRD_PARTY_LICENSES.md for the dependency inventory. Removing LICENSE
# from PRESERVE_PATHS lets that file overwrite Free's stale MIT LICENSE
# on the next sync, bringing Free into GPLv3 compliance.
PRESERVE_PATHS=(
    "build_macos.sh"
    "src/core/edition.hpp"

    # v1.4.1: README.md is now sync'd from this repo. Pre-1.4.1 it was
    # in PRESERVE_PATHS because the private README pitched Pro features
    # the free landing page should not advertise. The v1.4.1 README is
    # written as a unified document covering both editions with Pro
    # features clearly tagged "Pro-only", so blindly overwriting the
    # public's copy is correct now.
    #
    # example-config.yml is also synced (was preserved before but now
    # ships from this repo as the canonical schema reference).
    #
    # docs/BUILD-LINUX.md, docs/BUILD-MACOS.md, docs/BUILD-WINDOWS.md
    # were also removed from PRESERVE_PATHS in the v1.4.1 documentation
    # rewrite. The Pro tree now carries the canonical build references
    # for all three platforms (BUILD-MACOS.md kept; BUILD-LINUX.md and
    # BUILD-WINDOWS.md authored fresh). The Pro versions are unified
    # in style with the rest of the docs suite (TOC, nav-footer table,
    # Pro tagging, no em-dashes) and ship to the Free repo verbatim.
)

# -----------------------------------------------------------------------------
# Orphan paths -- files that EXIST in the public Free repo today but should
# NOT be there. Two reasons these survived:
#
#   1. The Free repo was bootstrapped in v1.2.0 by a manual upload that
#      shipped internal planning docs (TODO.md from "thePuzzler" era,
#      IMPLEMENTATION-PLAN.md from "Superflayer" era, research notes,
#      strategy memos). None of them are linked from README, and the
#      content references retired project codenames -- they cannot stay.
#
#   2. Some files were deleted from Pro (e.g. src/pool/http_pool_client.*
#      removed for silently leaking credentials when scheme was https://)
#      but the deletion never propagated because no prior sync ran.
#
# The wipe-and-copy at line ~238 SHOULD remove these implicitly (Free is
# wiped, only files present in Pro's `git ls-files` get re-copied). This
# list is defense-in-depth: an explicit kill list of known orphans so
# (a) the sync's intent is auditable, (b) we get a log line per orphan
# removed, and (c) any future regression in the wipe step still produces
# a clean Free tree.
#
# Entries are paths relative to the Free repo root. Add to this list
# rather than silently relying on the wipe; remove an entry only when
# the orphan no longer exists in Free's HEAD (i.e. a prior sync removed
# it cleanly).
# -----------------------------------------------------------------------------
ORPHAN_PATHS=(
    # Internal planning docs -- never belonged in public Free repo.
    # Leak old project codenames "thePuzzler" / "Superflayer" and pre-
    # launch strategy. Verified 2026-05-17: none are linked from README,
    # none are in Pro's docs/ tree.
    "docs/TODO.md"
    "docs/IMPLEMENTATION-PLAN.md"
    "docs/RESEARCH.md"
    "docs/COMPREHENSIVE-ANALYSIS.md"
    "docs/BITCOIN-PUZZLE-DEEP-RESEARCH.md"
    "docs/BITCOIN-PUZZLE-STRATEGY.md"
    "docs/LANGUAGE-DECISION.md"
    "docs/DEFCON-STRATEGIES.md"
    "docs/KANGAROO-IMPLEMENTATION-GUIDE.md"
    "docs/PERFORMANCE-TARGETS.md"
    "docs/POOL-ECONOMICS.md"
    "docs/pool-research.md"

    # USAGE.md opens with "thePuzzler" branding and is not part of the
    # current Pro docs suite (the v1.4.x README links INSTALL.md +
    # CONFIGURATION.md + POOL.md instead of a monolithic USAGE.md).
    "docs/USAGE.md"

    # PRO-FEATURES.md was the v1.4.0-era Pro pitch page. Superseded by
    # Pro's docs/PRO.md (which the sync will copy in). Free's README
    # already points at PRO.md after the next sync.
    "docs/PRO-FEATURES.md"

    # Root-level security audit report from a 2026-Q1 internal review.
    # Not appropriate for the public repo (references unfixed-at-time-of-
    # writing issues that have since been patched). Tracked in Free HEAD
    # but already removed from the working tree.
    "SECURITY-AUDIT-REPORT.md"

    # v1.5.0 GA audit + pen-test + team-plan markdown that leaked to the
    # free repo before the PRO_PATHS glob (AUDIT_*.md / PENTEST_*.md /
    # TEAM_PLAN_*.md) was added. Listed explicitly so the next sync
    # removes them with an auditable log line. The PRO_PATHS glob now
    # blocks future occurrences automatically.
    "AUDIT_V1_5_0.md"
    "AUDIT_V1_5_0_PASS_3.md"
    "TEAM_PLAN_V1_5_0_GA.md"

    # http_pool_client.{cpp,hpp} were deleted from Pro because the HTTP
    # transport silently downgraded https:// pool URLs (CHANGELOG entry
    # for that deletion is correct; the deletion just never reached Free
    # because no prior sync ran). Listing them explicitly so the kill is
    # visible in the sync log.
    "src/pool/http_pool_client.cpp"
    "src/pool/http_pool_client.hpp"
)

# -----------------------------------------------------------------------------

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <version-tag>" >&2
    echo "  e.g.: $0 v1.2.1-free" >&2
    exit 2
fi
TAG="$1"

if [[ ! -d "$PRIVATE_DIR/.git" ]]; then
    echo "error: PRIVATE_DIR=$PRIVATE_DIR is not a git repo" >&2
    exit 1
fi

PRIVATE_HEAD=$(git -C "$PRIVATE_DIR" rev-parse HEAD)
PRIVATE_HEAD_SHORT=$(git -C "$PRIVATE_DIR" rev-parse --short HEAD)
echo "[sync] private HEAD: $PRIVATE_HEAD"
echo "[sync] target tag:   $TAG"
echo "[sync] public repo:  $PUBLIC_URL"

WORK="$(mktemp -d -t collider-free-sync.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
echo "[sync] workdir: $WORK"

echo "[sync] cloning public free repo (depth 1)..."
git clone --depth 1 "$PUBLIC_URL" "$WORK/free" >/dev/null 2>&1

cd "$WORK/free"

git config user.email "hevnsnt@gmail.com"
git config user.name  "Bill Swearingen"

# -----------------------------------------------------------------------------
# Snapshot the public-only files we must preserve.
# -----------------------------------------------------------------------------
mkdir -p "$WORK/preserve"
preserved_count=0
for p in "${PRESERVE_PATHS[@]}"; do
    if [[ -e "$p" ]]; then
        mkdir -p "$WORK/preserve/$(dirname "$p")"
        cp -R "$p" "$WORK/preserve/$p"
        preserved_count=$((preserved_count + 1))
    fi
done
echo "[sync] preserved $preserved_count public-only path(s)"

# -----------------------------------------------------------------------------
# Wipe the working tree (keep .git). git's index gets cleaned up by the
# subsequent `git add -A` based on what files end up on disk.
# -----------------------------------------------------------------------------
find . -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +

# -----------------------------------------------------------------------------
# Copy from private, applying Pro exclusions. Uses git ls-files as the
# source-of-truth for "what files to consider" (so build/, vcpkg_installed/,
# .ninja_deps, etc. that aren't tracked never get copied at all).
#
# is_excluded() returns 0 (match, exclude) if the relative path matches any
# entry in PRO_PATHS. Trailing-slash entries match prefixes (directories),
# entries with a glob (* or ?) use shell glob semantics, exact paths match
# only themselves.
# -----------------------------------------------------------------------------
is_excluded() {
    local rel="$1"
    local pat
    for pat in "${PRO_PATHS[@]}"; do
        if [[ "$pat" == */ ]]; then
            # Directory prefix match
            if [[ "$rel" == "${pat%/}"/* || "$rel" == "${pat%/}" ]]; then
                return 0
            fi
        elif [[ "$pat" == *'*'* || "$pat" == *'?'* ]]; then
            # Glob (e.g. build-*/, *.pot)
            # shellcheck disable=SC2053
            if [[ "$rel" == $pat || "$rel" == ${pat%/}/* ]]; then
                return 0
            fi
        else
            # Exact path
            if [[ "$rel" == "$pat" ]]; then
                return 0
            fi
        fi
    done
    return 1
}

echo "[sync] copying tracked files from private tree minus ${#PRO_PATHS[@]} Pro path(s)..."
copied=0
excluded=0
while IFS= read -r rel; do
    [[ -z "$rel" ]] && continue
    if is_excluded "$rel"; then
        excluded=$((excluded + 1))
        continue
    fi
    src="$PRIVATE_DIR/$rel"
    if [[ ! -e "$src" ]]; then
        # File is in git index but missing on disk (rare); skip silently.
        continue
    fi
    mkdir -p "$(dirname "$rel")"
    cp "$src" "$rel"
    copied=$((copied + 1))
done < <(git -C "$PRIVATE_DIR" ls-files)
echo "[sync] copied $copied file(s); excluded $excluded Pro file(s)"

# -----------------------------------------------------------------------------
# Restore public-only files (overrides whatever rsync put there, if anything).
# -----------------------------------------------------------------------------
restored_count=0
for p in "${PRESERVE_PATHS[@]}"; do
    if [[ -e "$WORK/preserve/$p" ]]; then
        mkdir -p "$(dirname "$p")"
        cp -R "$WORK/preserve/$p" "$p"
        restored_count=$((restored_count + 1))
    fi
done
echo "[sync] restored $restored_count public-only path(s)"

# -----------------------------------------------------------------------------
# Orphan removal: explicitly delete files that should never appear in Free.
# The wipe-and-copy normally takes care of this implicitly (anything not in
# Pro's `git ls-files` does not get re-copied), but we run the explicit kill
# anyway so (1) the sync log shows what was removed by name, (2) we get a
# loud signal if a known orphan reappears, and (3) defense-in-depth against
# any future change to the wipe logic.
# -----------------------------------------------------------------------------
orphan_removed=0
for p in "${ORPHAN_PATHS[@]}"; do
    if [[ -e "$p" ]]; then
        rm -rf "$p"
        echo "[sync] removed orphan: $p"
        orphan_removed=$((orphan_removed + 1))
    fi
done
echo "[sync] removed $orphan_removed orphan path(s) of ${#ORPHAN_PATHS[@]} listed"

# -----------------------------------------------------------------------------
# Defense-in-depth: prove no Pro source file slipped through the copy.
# Run this BEFORE template installation so the check only covers files
# that came from the copy step; the template writes build-release.yml
# intentionally afterward.
# -----------------------------------------------------------------------------
LEAKED=()
for p in "${PRO_PATHS[@]}"; do
    # Strip trailing slash for directory checks
    pclean="${p%/}"
    if [[ -e "$pclean" ]]; then
        LEAKED+=("$pclean")
    fi
done
if (( ${#LEAKED[@]} > 0 )); then
    echo "[sync] FATAL: Pro paths present in staged free tree:" >&2
    printf '  %s\n' "${LEAKED[@]}" >&2
    exit 1
fi
echo "[sync] verified: 0 Pro paths in staged free tree"

# -----------------------------------------------------------------------------
# Install the free-repo CI workflow from the pro-side template.
# scripts/templates/free-build-release.yml is the canonical source of
# truth for the free build pipeline. It is listed in PRO_PATHS so the
# pro's build-release.yml never lands in free; the template is installed
# here under the correct name after the Pro-leak check completes.
# -----------------------------------------------------------------------------
FREE_WORKFLOW_TEMPLATE="$PRIVATE_DIR/scripts/templates/free-build-release.yml"
if [[ -f "$FREE_WORKFLOW_TEMPLATE" ]]; then
    mkdir -p ".github/workflows"
    cp "$FREE_WORKFLOW_TEMPLATE" ".github/workflows/build-release.yml"
    echo "[sync] installed free CI workflow from template"
else
    echo "[sync] WARNING: free-build-release.yml template not found at $FREE_WORKFLOW_TEMPLATE" >&2
fi

# -----------------------------------------------------------------------------
# Install free-side STUB headers at Pro paths.
#
# v1.5.0 introduced Pro TUI integration into formerly-free files:
# pool_solver.cpp, puzzle_solver.cpp, puzzle_solver_kangaroo.cpp,
# puzzle_solver_bsgs.cpp, puzzle_solver_bruteforce.cpp now reference
# ui/tui/* symbols (StdioCapture, TuiApp, LaunchConfig, etc.) through
# unguarded #include + method calls. Rather than #ifdef-gating ~60
# callsites across those free-shipped files, we install no-op stub
# headers at the same paths in the free tree. Each stub matches the
# Pro symbol surface (types + method signatures) so the includes
# resolve and call sites compile; every method body is a no-op so
# the free binary just discards TUI updates silently.
#
# The stubs live in scripts/templates/free_stubs/ with the same
# relative paths as the Pro originals. After the wipe + Pro-tree
# copy (which omits PRO_PATHS entries like src/ui/tui/), we walk
# the stubs directory and copy each file into the free tree.
# The Pro originals are NOT copied here (already excluded via
# PRO_PATHS); the stubs are the ONLY definitions free sees.
# -----------------------------------------------------------------------------
FREE_STUBS_ROOT="$PRIVATE_DIR/scripts/templates/free_stubs"
if [[ -d "$FREE_STUBS_ROOT" ]]; then
    stub_count=0
    while IFS= read -r -d '' stub; do
        rel="${stub#$FREE_STUBS_ROOT/}"
        dest_dir="$(dirname "$rel")"
        mkdir -p "$dest_dir"
        cp "$stub" "$rel"
        stub_count=$((stub_count + 1))
    done < <(find "$FREE_STUBS_ROOT" -type f -print0)
    echo "[sync] installed $stub_count free-side stub header(s) from scripts/templates/free_stubs/"
else
    echo "[sync] note: no free-side stubs dir at $FREE_STUBS_ROOT (skipping)"
fi

# -----------------------------------------------------------------------------
# Stage everything, commit, tag, push.
# -----------------------------------------------------------------------------
git add -A

if git diff --cached --quiet; then
    echo "[sync] no content changes vs current free HEAD; skipping commit"
else
    git commit -m "Sync free release $TAG (private $PRIVATE_HEAD_SHORT)

Generated by scripts/sync-to-free.sh from private commit
$PRIVATE_HEAD. Pro-only paths (${#PRO_PATHS[@]} entries) excluded
per the PRO_PATHS list in scripts/sync-to-free.sh."
    echo "[sync] committed sync commit"
fi

if git tag -l "$TAG" | grep -q .; then
    echo "[sync] tag $TAG already exists locally on free repo; reusing"
else
    git tag -a "$TAG" -m "Free release $TAG (synced from private $PRIVATE_HEAD_SHORT)"
    echo "[sync] tagged $TAG"
fi

echo "[sync] pushing main + $TAG to public repo..."
git push origin main
git push origin "$TAG"
echo "[sync] done"
