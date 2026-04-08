#!/bin/bash
# Build collider-pro on macOS (Apple Silicon / Metal)
set -e

cd "$(dirname "$0")"

echo "=== Pulling latest ==="
git pull origin develop

echo "=== Configuring CMake (Metal) ==="
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCOLLIDER_EDITION_PRO=ON \
  -DCOLLIDER_USE_METAL=ON \
  -DCOLLIDER_USE_CUDA=OFF \
  -DCOLLIDER_BUILD_TESTS=OFF \
  -DCOLLIDER_BUILD_BENCHMARKS=OFF \
  -DCOLLIDER_BUILD_TOOLS=OFF

echo "=== Building ==="
cmake --build build --parallel $(sysctl -n hw.ncpu)

if [ -f build/collider_pro ]; then
  echo "=== SUCCESS ==="
  ls -lh build/collider_pro
  file build/collider_pro
else
  echo "=== FAILED ==="
  exit 1
fi
