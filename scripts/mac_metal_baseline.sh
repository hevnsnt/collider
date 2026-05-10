#!/usr/bin/env bash
#
# Run the Metal kangaroo KAT + smoke tests on Mac and print a summary.
# Use to establish a baseline before/after Phase 5 Montgomery batch
# inversion changes.
#
# Usage (on Mac):
#   cd ~/dev/collider-pro                  # or wherever this repo lives
#   git pull
#   bash scripts/mac_metal_baseline.sh > /tmp/metal_baseline.log 2>&1
#   tail -40 /tmp/metal_baseline.log
#
# Then paste the last 40 lines back to the AI.

set -euo pipefail

# Apple Silicon Homebrew lives under /opt/homebrew; an SSH non-interactive
# shell on the user's Mac defaults to PATH=/usr/bin:/bin which omits it,
# so cmake/ninja from `brew install cmake ninja` aren't found unless we
# add the bin dir explicitly. Idempotent if already on PATH.
case ":$PATH:" in
  *":/opt/homebrew/bin:"*) ;;
  *) export PATH="/opt/homebrew/bin:$PATH" ;;
esac

cd "$(git rev-parse --show-toplevel)"
echo "==== git tip ===="
git log -1 --oneline

echo
echo "==== configure ===="
if [ ! -f build_mac/CMakeCache.txt ]; then
    cmake -B build_mac -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCOLLIDER_PRO=ON \
        -DCOLLIDER_USE_METAL=ON \
        -DCOLLIDER_USE_CUDA=OFF \
        -DCOLLIDER_BUILD_TESTS=ON \
        -DCOLLIDER_BUILD_BENCHMARKS=OFF \
        -DCOLLIDER_BUILD_TOOLS=OFF
fi

echo
echo "==== build ===="
cmake --build build_mac --parallel

echo
echo "==== test: Metal secp256k1 KAT ===="
if [ -x build_mac/test_metal_secp256k1 ]; then
    ./build_mac/test_metal_secp256k1 || echo "FAIL: test_metal_secp256k1 exit $?"
else
    echo "MISSING: build_mac/test_metal_secp256k1"
fi

echo
echo "==== test: Metal kangaroo smoke ===="
if [ -x build_mac/test_metal_kangaroo_smoke ]; then
    ./build_mac/test_metal_kangaroo_smoke || echo "FAIL: test_metal_kangaroo_smoke exit $?"
else
    echo "MISSING: build_mac/test_metal_kangaroo_smoke"
fi

echo
echo "==== summary ===="
echo "Tip: $(git rev-parse --short HEAD)"
echo "Date: $(date -u +%FT%TZ)"
echo "Done."
