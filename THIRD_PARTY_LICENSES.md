# Third-Party Licenses

This document enumerates third-party software incorporated into theCollider
and the license terms under which each component is distributed. The
combined source distribution at the root of this repository is licensed
under GPLv3 (see LICENSE) because at least one statically-linked component
(RCKangaroo) is GPLv3-licensed.

If you redistribute a binary built from this source, you are bound by the
license terms of every component listed below. Read each entry. If you
intend to ship a closed-source product derived from this tree, the GPLv3
status of RCKangaroo applies to the combined work; see the "GPL compliance
notes" section at the bottom.

## Dependency inventory

### RCKangaroo (Pollard kangaroo GPU solver)

- Source: `third_party/RCKangaroo/`
- Upstream: https://github.com/RetiredC
- License: GNU General Public License v3.0 (GPLv3)
- License text: `third_party/RCKangaroo/LICENSE.TXT`
- Copyright: (c) 2024 RetiredCoder (RC)

Modifications by SixCyber LLC in 2026 add a KangarooMode
field (KANG_MODE_BOTH, KANG_MODE_TAME_ONLY, KANG_MODE_WILD_ONLY) on
TKparams plus the corresponding seeding logic in RCGpuKang::Start so that
a worker can be assigned ONE half of the kangaroo walk. These modifications
support the v1.5 asymmetric tame/wild work assignment protocol that
removes the worker-side path to computing the puzzle private key locally.
Per GPLv3 section 5, the modifications are licensed under GPLv3 and the
source of the modifications is published in this same repository.

### xxHash

- Source: fetched at build time via CMake FetchContent (CMakeLists.txt
  top-level)
- Upstream: https://github.com/Cyan4973/xxHash
- License: BSD 2-Clause "Simplified" License
- Pinned version: v0.8.2
- Copyright: (c) 2012-2024 Yann Collet

Used as a header-only hash function inside the bloom filter pipeline.
BSD-2-Clause is GPL-compatible: BSD-licensed code may be combined with
GPL code in a single distribution without altering the GPL status of
the combined work.

### OpenSSL

- Source: resolved through vcpkg at build time (vcpkg.json)
- Upstream: https://www.openssl.org
- License: Apache License 2.0 (OpenSSL 3.0 and later)
- Copyright: (c) The OpenSSL Project Authors

Used for TLS in the JLP pool client. Apache 2.0 is GPL-compatible with
GPLv3 specifically (it is not compatible with GPLv2). The combined
distribution as a whole is fine under GPLv3.

If a vcpkg baseline upgrade pulls in an OpenSSL 1.1.x release instead of
3.0+, that older release is dual-licensed under the original OpenSSL
License and SSLeay License. That older licensing is GPLv2-incompatible
but considered acceptable for GPLv3-licensed binaries under the system
library exception in GPLv3 section 1. The recommended path is to stay
on OpenSSL 3.0+ to avoid the historical incompatibility entirely.

### libcurl

- Source: resolved through vcpkg at build time (vcpkg.json)
- Upstream: https://curl.se
- License: curl license (MIT-style)
- Copyright: (c) 1996-2024 Daniel Stenberg

Used for HTTP requests in the balance probe pipeline and (when the C++
HTTP pool client is enabled) for pool requests. The curl license is
MIT-style and GPL-compatible.

### CUDA Toolkit (runtime + libraries)

- Source: NVIDIA CUDA Toolkit installed on the build host
- Upstream: https://developer.nvidia.com/cuda-toolkit
- License: NVIDIA Software License Agreement (proprietary)
- Vendor: NVIDIA Corporation

CUDA runtime libraries (`cudart`) and headers are required to build the
GPU acceleration paths. The NVIDIA Software License Agreement explicitly
permits redistribution of CUDA Toolkit runtime libraries with binaries
that depend on them; see the CUDA Toolkit EULA, section 1.5 "Redistributable
Components". Binary distributions that ship CUDA DLLs must include the
NVIDIA license text alongside.

CUDA Toolkit is treated as a system library for GPLv3 section 1 purposes,
so its proprietary status does not affect the GPLv3 status of the combined
work.

### Bitcoin secp256k1 constants

- Source: `src/gpu/secp256k1.cu`
- Origin: derived from secp256k1 curve constants, which are public domain
  mathematical values. The implementation in this repository is original
  work by SixCyber LLC, not derived from libsecp256k1 (Bitcoin Core).

If a future port pulls in libsecp256k1 source, that library is licensed
under the MIT License (https://github.com/bitcoin-core/secp256k1) and
GPL-compatible.

## GPL compliance notes for distributors

If you distribute a compiled binary from this source tree, GPLv3 obliges
you to:

1. Provide recipients of the binary with access to the corresponding
   source code, on the same terms as the binary distribution. The
   simplest way to satisfy this is to publish the source tree alongside
   the binary distribution, or to include a written offer in the
   distribution pointing to a public source repository (this one, at
   https://github.com/hevnsnt/collider, is the canonical public source
   for the Free edition).
2. Preserve all copyright notices and license texts. Do not strip
   `LICENSE`, `THIRD_PARTY_LICENSES.md`, or `third_party/RCKangaroo/LICENSE.TXT`
   from a redistribution.
3. Mark any further modifications you make in a way that distinguishes
   your work from the unmodified source. Include the date and the
   modifier's identity. See `third_party/RCKangaroo/GpuKang.cpp` for an
   example of the modification-header convention used in this project.
4. Apply GPLv3 to any larger work into which you incorporate this
   source. In particular, you may NOT statically link this source (or
   the resulting libraries) into a proprietary closed-source product
   without releasing that combined work under GPLv3.

If you want to use the GPU kangaroo capabilities of this project in a
proprietary product, your options are (a) negotiate a commercial license
with the upstream RCKangaroo author, (b) replace RCKangaroo with a
non-copyleft alternative, or (c) run the RCKangaroo-derived component as
a separate process with its own GPLv3 source distribution, communicating
with your proprietary code via IPC such that the two programs are
genuinely "separate and independent works" under the GPLv3 mere-aggregation
clause.

## Verification

To verify the integrity of this inventory against the actual source tree:

```bash
# RCKangaroo files present
ls third_party/RCKangaroo/

# xxHash declared as a FetchContent dependency
grep -n "xxHash\|xxhash" CMakeLists.txt

# OpenSSL + libcurl declared via vcpkg
cat vcpkg.json
```

If a future commit adds a new third-party dependency, add it to this
file in the SAME commit. Drift between this file and the actual
dependency graph is a compliance risk.
