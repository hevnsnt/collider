#!/usr/bin/env bash
# activate-offline.sh - Seed the binary's local license cache so it
# trusts a valid license key WITHOUT making a network call.
#
# Why this exists: the collider_pro binary's `--activate <KEY>` flow
# POSTs to /api/license-check on collisionprotocol.com. When App Hosting
# is mid-deploy (or DNS hasn't cut over from the legacy VPS yet) the
# remote call can return 404/500/empty, blocking the operator from
# running the binary even though their license is genuinely valid.
#
# The cache file at ~/.collider/license.cache uses an HMAC keyed on the
# license key itself (collider-cache-v1:<key>) — see src/license/
# license_check.cpp::cache_hmac_key. Computing it requires only the key
# value and OpenSSL, both of which the operator has locally. This script
# does that computation and writes the cache file with a far-future
# expiry, so the binary's read_cache call short-circuits before
# remote_validate ever runs.
#
# Usage:
#   bash scripts/activate-offline.sh <LICENSE_KEY> [EMAIL]
#
# Defaults: EMAIL falls back to hevnsnt@gmail.com (the bootstrap admin)
# if not provided. The email is cosmetic — the binary just echoes it in
# the activated banner; the binary does NOT use it for authentication.
#
# Once the remote /api/license-check route is reachable again, the next
# scheduled cache refresh (24h after this seed) will re-validate over
# the network. To force an immediate re-check, delete the cache file:
#   rm ~/.collider/license.cache

set -euo pipefail

KEY="${1:-}"
EMAIL="${2:-hevnsnt@gmail.com}"

if [[ -z "$KEY" ]]; then
  echo "usage: $0 <LICENSE_KEY> [EMAIL]" >&2
  exit 2
fi

if [[ ${#KEY} -lt 16 || ${#KEY} -gt 256 ]]; then
  echo "[!] license key length out of range: ${#KEY} (expected 16..256)" >&2
  exit 2
fi

# Expiry: 2100-01-01 UTC. Matches the validate route's default for
# perpetual licenses. The binary re-validates at every restart anyway
# (cache TTL inside the binary is 24h; once that elapses it hits the
# server, picks up the real expiry, and refreshes the cache).
EXPIRY=4102444800

PAYLOAD="${KEY}|1|${EMAIL}|${EXPIRY}"
HMAC_KEY="collider-cache-v1:${KEY}"
HMAC=$(printf '%s' "$PAYLOAD" \
  | openssl dgst -sha256 -hmac "$HMAC_KEY" \
  | awk '{print $NF}')

if [[ -z "$HMAC" || ${#HMAC} -ne 64 ]]; then
  echo "[!] HMAC computation failed (got: '$HMAC')" >&2
  exit 1
fi

CACHE_DIR="${HOME}/.collider"
CACHE_FILE="${CACHE_DIR}/license.cache"
KEY_FILE="${CACHE_DIR}/license_key"

mkdir -p "$CACHE_DIR"

# Write the cache file in the exact format read_cache expects:
#   line 1: key
#   line 2: "1" if valid else "0"
#   line 3: email
#   line 4: expiry_epoch
#   line 5: hmac
printf '%s\n1\n%s\n%s\n%s\n' "$KEY" "$EMAIL" "$EXPIRY" "$HMAC" > "$CACHE_FILE"

# Also persist the key so validate_startup_license() finds it on
# subsequent runs (matches what --activate would have written).
printf '%s\n' "$KEY" > "$KEY_FILE"

echo "[*] Wrote $CACHE_FILE"
echo "[*] Wrote $KEY_FILE"
echo "[*] License now active offline. Run: ./build/collider --puzzle 135 --kangaroo"
