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
# appear in the public Free repo. See docs/PRO-MIGRATION.md.
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
    "src/gpu/mega_fused_kernel.cu"
    "src/gpu/mega_fused_kernel.hpp"
    "src/gpu/brain_wallet_gpu.cpp"
    "src/gpu/brain_wallet_gpu.hpp"
    "src/gpu/gpu_rules.cpp"
    "src/gpu/gpu_rules.cu"
    "src/gpu/gpu_rules.hpp"
    "src/gpu/gpu_rule_kernel.cu"
    "src/gpu/h160_bloom_filter.cu"
    "src/gpu/bloom_filter.cu"
    "src/gpu/fused_pipeline.cu"
    "src/gpu/pipeline.cu"

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

    # Pro-internal documentation
    "docs/PRO-MIGRATION.md"
    "docs/CRYPTO-VALIDATION.md"
    "docs/review-2026-05-04/"

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
    # no SYNC_DEPLOY_KEY secret), and leaving them on the free side
    # would also re-trigger sync-free.yml on every tag pushed to free.
    ".github/workflows/sync-free.yml"
    ".github/workflows/sync-protocol.yml"
)

# -----------------------------------------------------------------------------
# Public-only paths -- the public Free repo owns these, do not overwrite.
# -----------------------------------------------------------------------------
PRESERVE_PATHS=(
    "LICENSE"
    "build_macos.sh"
    "example-config.yml"
    "docs/BUILD-LINUX.md"
    "docs/BUILD-MACOS.md"
    "docs/BUILD-WINDOWS.md"
    "src/core/edition.hpp"
    ".github/workflows/build-release.yml"
    # README.md is public-only: the public Free repo's landing page is
    # written for the Free audience and does not advertise Pro features.
    # The private repo's README pitches the full Pro/Free product, so
    # blindly overwriting public's would re-introduce false claims.
    "README.md"
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
# Defense-in-depth: prove no Pro source file slipped through. If any
# survive, abort BEFORE pushing.
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
# Stage everything, commit, tag, push.
# -----------------------------------------------------------------------------
git add -A

if git diff --cached --quiet; then
    echo "[sync] no content changes vs current free HEAD; skipping commit"
else
    git -c user.email=hevnsnt@gmail.com \
        -c user.name="Bill Swearingen" \
        commit -m "Sync free release $TAG (private $PRIVATE_HEAD_SHORT)

Generated by scripts/sync-to-free.sh from private commit
$PRIVATE_HEAD. Pro-only paths (${#PRO_PATHS[@]} entries) excluded
per docs/PRO-MIGRATION.md."
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
