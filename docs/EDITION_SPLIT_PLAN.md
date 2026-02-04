# theCollider Edition Split Plan

## Overview

Split theCollider into two products from one source tree:

| | **collider** (Free) | **collider-pro** (Paid) |
|---|---|---|
| **Source** | Open source (public repo) | Private repo |
| **Pool mode** | ✅ Hardcoded to collisionprotocol.com | ✅ Any pool or custom |
| **Solo puzzle solving** | ❌ | ✅ |
| **Brain wallet scanner** | ❌ | ✅ |
| **Bloom filter** | ❌ | ✅ |
| **PCFG / Markov / Rules** | ❌ | ✅ |
| **WarpWallet / Scrypt** | ❌ | ✅ |
| **Multi-GPU** | ✅ | ✅ |
| **Benchmark** | ✅ | ✅ |
| **License required** | No | Yes (Ed25519 signed key) |
| **Banner** | "theCollider — Pool Edition" | "theCollider Pro — Licensed to: Name" |

---

## Build System

### CMake Changes (`CMakeLists.txt`)

Add one new option:
```cmake
option(COLLIDER_EDITION_PRO "Build Pro edition with all features" OFF)
```

**Two executable targets:**
- `collider` — always built, free edition (no `COLLIDER_PRO` define)
- `collider_pro` — only built when `COLLIDER_EDITION_PRO=ON`, gets `-DCOLLIDER_PRO=1`

**When `COLLIDER_EDITION_PRO=OFF` (free build):**
- Do NOT compile: `src/generators/*`, `src/scrapers/*`, `src/rules/*`, `src/tools/build_bloom.cpp`
- Do NOT compile: brain wallet GPU kernels (`brain_wallet_gpu.*`, `fused_pipeline.cu`, `gpu_rules.*`, `gpu_rule_kernel.cu`, `bloom_filter.cu`, `h160_bloom_filter.cu`)
- Do NOT link: `collider_core` library (contains rule engine, passphrase generators)
- DO compile: pool client, kangaroo GPU kernels, platform abstraction, main.cpp

**When `COLLIDER_EDITION_PRO=ON` (pro build):**
- Everything compiles as it does today
- Additionally compiles: `src/license/license.cpp`
- Additionally builds: `generate_license` tool

### New Source Lists

```cmake
# --- Shared sources (both editions) ---
SHARED_GPU_SOURCES:
  src/gpu/sha256.cu
  src/gpu/secp256k1.cu
  src/gpu/ripemd160.cu
  src/gpu/puzzle_gpu.cu
  src/gpu/puzzle_optimized.cu
  src/gpu/kangaroo_kernel.cu
  src/gpu/pipeline.cu
  src/gpu/rckangaroo_wrapper.cu

# --- Pro-only GPU sources ---
PRO_GPU_SOURCES:
  src/gpu/brain_wallet_gpu.cpp
  src/gpu/fused_pipeline.cu
  src/gpu/gpu_rule_kernel.cu
  src/gpu/gpu_rules.cu
  src/gpu/gpu_rules.cpp
  src/gpu/bloom_filter.cu
  src/gpu/h160_bloom_filter.cu

# --- Pro-only CPU sources ---
PRO_CORE_SOURCES:
  src/core/rule_engine.cpp
  src/generators/passphrase_generator.cpp
  src/generators/priority_queue.cpp

# --- Pro-only license ---
PRO_LICENSE_SOURCES:
  src/license/license.cpp
  src/license/ed25519/ed25519.c
```

---

## New Files to Create

### 1. `src/core/edition.hpp` — Feature gate header

Central header that all files include. Defines feature macros based on `COLLIDER_PRO`:

```
COLLIDER_PRO              — Master gate (set by CMake)
COLLIDER_HAS_BRAINWALLET  — Brain wallet scanner
COLLIDER_HAS_SOLO         — Standalone puzzle solving
COLLIDER_HAS_BLOOM        — Bloom filter integration
COLLIDER_HAS_RULES        — Rule engine, PCFG, Markov
COLLIDER_HAS_CUSTOM_POOL  — User-specified pool URLs
COLLIDER_HAS_LICENSE       — License verification
```

Free edition: none of these are defined.
Pro edition: all are defined.

### 2. `src/license/license.hpp` + `src/license/license.cpp` — License verification

**License key format:**
```
eyJuYW1lIjoiSm9obiBEb2UiLC...base64payload...
.
SIGNATURE_BASE64
```

Specifically: `base64url(json_payload)\n.\nbase64url(ed25519_signature)`

**Payload JSON:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "issued": "2026-02-03",
  "expires": "2027-02-03",
  "plan": "pro",
  "version": 1
}
```

**Public API:**
```cpp
namespace collider::license {
  struct LicenseInfo {
    std::string name;
    std::string email;
    std::string issued;
    std::string expires;
    std::string plan;
    bool valid;
    std::string error;
  };

  LicenseInfo load_and_verify();        // Checks ~/.collider/license.key, ./license.key
  bool is_valid(const LicenseInfo&);
  bool is_expired(const LicenseInfo&);
}
```

**Ed25519 implementation:** Vendor a minimal Ed25519 impl (~300 lines) into `src/license/ed25519/`. No external dependency. Public key is a `constexpr` array in `license.cpp`.

**Key search order:**
1. `./license.key` (current directory)
2. `~/.collider/license.key` (user home)
3. Environment variable `COLLIDER_LICENSE_KEY` (path to file)

### 3. `tools/generate_license.cpp` — License key generator (private, never distributed)

CLI tool:
```bash
./generate_license --name "John Doe" --email "john@example.com" --expires "2027-02-03" --key private_key.bin --output license.key
```

Also generates keypair:
```bash
./generate_license --keygen --output-public public_key.bin --output-private private_key.bin
```

### 4. `.gitignore` additions
```
# License keys
*.key
private_key.bin
tools/license_private_key.bin
```

---

## Files to Modify

### `src/main.cpp` (~4135 lines)

This is the biggest change. All modifications use `#ifdef COLLIDER_PRO` / `#endif` gates.

**Includes (top of file, ~line 51-75):**
```cpp
#include "core/edition.hpp"  // ADD — must be first project include

// Gate pro-only includes:
#ifdef COLLIDER_HAS_BRAINWALLET
#include "core/brainwallet_state.hpp"
#include "ui/brainwallet_setup.hpp"
#include "gpu/brain_wallet_gpu.hpp"
#include "generators/brain_wallet_engine.hpp"
#include "generators/streaming_brain_wallet.hpp"
#endif

#ifdef COLLIDER_HAS_BLOOM
#include "tools/utxo_bloom_builder.hpp"
#endif

#ifdef COLLIDER_HAS_RULES
#include "core/rule_engine.hpp"
#include "gpu/gpu_rules.hpp"
#endif

#ifdef COLLIDER_HAS_LICENSE
#include "license/license.hpp"
#endif
```

**Arguments struct (~line 480-530):**
Gate pro-only fields:
```cpp
struct Arguments {
    // Shared (both editions)
    bool benchmark = false;
    bool verbose = false;
    bool pool_mode = false;
    std::string pool_url;
    std::string pool_worker;
    // ... other shared fields

    #ifdef COLLIDER_PRO
    bool puzzle_mode = true;
    bool brainwallet_mode = false;
    bool brainwallet_setup = false;
    std::string wordlist_file;
    std::string bloom_file;
    // ... other pro fields
    #else
    bool puzzle_mode = false;  // Free: no standalone puzzle
    #endif
};
```

**Argument parsing (~line 555-620):**
Gate pro-only CLI flags. In free mode, `--brainwallet`, `--bloom`, `--puzzle`, `--rules`, etc. are unrecognized (print error + help).

In free mode, `--pool` is NOT exposed. Instead, `--worker <BTC_ADDRESS>` is the only required arg. Pool URL is hardcoded:
```cpp
#ifndef COLLIDER_PRO
static constexpr const char* DEFAULT_POOL_URL = "jlp://pool.collisionprotocol.com:17403";
#endif
```

**print_usage() (~line 630):**
Two versions — free shows pool-only usage, pro shows everything.

**Interactive menus (~line 1620-1670):**
Free menu:
```
[1] Join Collision Protocol Pool
[2] Run Benchmark
[3] Help
```
Pro menu (existing + license info):
```
Licensed to: John Doe
[1] Solve Bitcoin Puzzle
[2] Brain Wallet Scanner
[3] Join Pool
[4] Run Benchmark
[5] Help
```

**run_brainwallet_interactive() (~line 1206-1500):**
Entire function wrapped in `#ifdef COLLIDER_HAS_BRAINWALLET`.

**run_puzzle_interactive() (~line 1058-1200):**
Standalone puzzle selection wrapped in `#ifdef COLLIDER_HAS_SOLO`.
Pool puzzle mode stays available in free version.

**run_pool_mode() (~line 1696-1920):**
Stays in both editions. In free mode, pool URL is forced to `DEFAULT_POOL_URL`.

**Brainwallet main loop (~line 2200-2900):**
Entire brainwallet execution section wrapped in `#ifdef COLLIDER_HAS_BRAINWALLET`.

**main() (~line 1934):**
Add license check at top for pro edition:
```cpp
#ifdef COLLIDER_HAS_LICENSE
auto license = collider::license::load_and_verify();
if (!license.valid) {
    std::cerr << "Invalid or missing license. Purchase at collisionprotocol.com\n";
    return 1;
}
if (collider::license::is_expired(license)) {
    std::cerr << "License expired on " << license.expires << "\n";
    return 1;
}
#endif
```

In free mode, default to pool mode instead of puzzle mode:
```cpp
#ifndef COLLIDER_PRO
args.pool_mode = true;
args.pool_url = DEFAULT_POOL_URL;
#endif
```

### `src/ui/banner.hpp`

**Free edition banner:**
```
╔══════════════════════════════════════╗
║          theCollider v1.1            ║
║        ── Pool Edition ──            ║
╚══════════════════════════════════════╝
```

**Pro edition banner:**
```
╔══════════════════════════════════════╗
║        theCollider Pro v1.1          ║
║    Licensed to: John Doe             ║
╚══════════════════════════════════════╝
```

### `src/ui/interactive.hpp`

Gate `BRAINWALLET_MODE` menu option and `run_brainwallet_interactive` behind `#ifdef`.
Gate `PuzzleModeChoice::STANDALONE` behind `#ifdef`.
Free version: `display_main_menu()` returns simplified choices.

### `src/ui/brainwallet_setup.hpp`

Entire file contents wrapped in `#ifdef COLLIDER_HAS_BRAINWALLET`.

### `src/core/yaml_config.hpp`

In free mode, `apply_config_to_args()` ignores brainwallet/bloom/puzzle config sections and forces pool URL to the hardcoded default.

### `src/pool/pool_manager.hpp` / `.cpp`

No changes needed — pool code is shared. The URL is just set differently based on edition.

---

## Testing Plan

1. **Free build compiles:** `cmake -DCOLLIDER_EDITION_PRO=OFF .. && make`
   - No brain wallet, bloom, rule, or PCFG code linked
   - Binary starts, shows pool edition banner
   - `--worker` flag works, connects to hardcoded pool
   - `--brainwallet` flag prints "Pro feature" error
   - Interactive menu shows only pool + benchmark

2. **Pro build compiles:** `cmake -DCOLLIDER_EDITION_PRO=ON .. && make`
   - All existing features work as before
   - Without license.key: prints error and exits
   - With valid license.key: shows licensee name in banner, all features unlocked
   - With expired license.key: prints expiry error and exits

3. **License tool:** `./generate_license --keygen` creates keypair
   - Generated license verifies correctly
   - Tampered license fails verification
   - Wrong public key fails verification

---

## Distribution Plan

- **Public GitHub repo** (`collisionprotocol/collider`): Free edition source + build instructions
- **Private repo** (`hevnsnt/theCollider`): Full source with pro features
- **Releases**: Pre-built binaries for both editions (Windows, Linux, macOS)
- **License sales**: Generate keys with private `generate_license` tool, deliver to customers

---

## File Change Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `CMakeLists.txt` | Modify | Add `COLLIDER_EDITION_PRO` option, dual targets, conditional source lists |
| `src/core/edition.hpp` | **New** | Feature gate macros |
| `src/license/license.hpp` | **New** | License verification API |
| `src/license/license.cpp` | **New** | License verification impl |
| `src/license/ed25519/` | **New** | Vendored Ed25519 impl |
| `tools/generate_license.cpp` | **New** | License key generator |
| `src/main.cpp` | Modify | `#ifdef` gates on pro features (~15 gate blocks) |
| `src/ui/banner.hpp` | Modify | Edition-aware banner |
| `src/ui/interactive.hpp` | Modify | Edition-aware menus |
| `src/ui/brainwallet_setup.hpp` | Modify | Wrap in `#ifdef` |
| `src/core/yaml_config.hpp` | Modify | Ignore pro config in free mode |
| `.gitignore` | Modify | Add license key patterns |

**Total new code:** ~800 lines (license module + edition header + generate tool)
**Total gate modifications:** ~15 `#ifdef` blocks across 6 existing files
**No existing logic changed** — only wrapped in compile-time gates.
