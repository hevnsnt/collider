# theCollider v1.4.0 — Release Notes

Tag: `v1.4.0-pro` (private) / `v1.4.0-free` (public). Sync via
`scripts/sync-to-free.sh v1.4.0-free`.

This release focuses on three pillars:

1. **Single source of truth for the JLP wire format.** The C++ client
   and the Python pool server now derive their structs from one IDL
   (`protocol/jlp.yaml`) via codegen, with drift detection at every
   build.
2. **Anti-cheat and CSPRNG correctness in the pool server.** Twenty-two
   new tests, plus a full ladder-escalation state machine and
   cryptographically secure uniform work distribution.
3. **Brain Wallet v2.** Multi-scheme + multi-address-type derivation
   on the GPU, plus puzzle-mode bloom checking that short-circuits
   before EC_MUL when no puzzle target hits.

## What's new

### Client (collider-pro)

- **`--puzzle-only-v2` flag.** Drives the v2 brain-wallet kernel
  through the `v2_orchestrator` host driver. Loads puzzle keys from
  `data/puzzle_history.json`, validates `--schemes` and `--addr-types`
  CSV flags, dispatches to the GPU. With `addr_mask=0` (default), the
  EC multiply is skipped entirely on non-hits — measured ~20x speedup
  over the legacy stock pipeline.
- **Six SHA-256-only DerivationScheme implementations** in
  `src/gpu/v2/brain_wallet_v2.cu` (S1 SHA256_PW through S6 SHA256_ITER_16).
  Templated `v2_puzzle_only_kernel_scheme<S>` keeps register pressure
  manageable; per-scheme launches let nvcc inline each composition.
- **CPU references with KAT tests** for every kernel:
  - `address_derive_cpu.hpp`: hash160 + 5 address derivations
    (P2PKH uncompressed/compressed, P2SH-P2WPKH, P2WPKH-V0, P2TR
    BIP-86 tap tweak)
  - `weak_prng_cpu.hpp`: MT19937, Park-Miller LCG, MSVC rand,
    java.util.Random, libbitcoin bx (Milk Sad), Profanity
  - `encoding_munge_cpu.hpp`: 9 encodings (UTF-8, UTF-16-LE/BE,
    UTF-32-LE/BE, Latin-1, strip-non-ascii, upper/lower-ascii)
  - `sha512_cpu.hpp`: SHA-512 + HMAC-SHA256/512 + PBKDF2-HMAC-SHA256/512
  - `electrum_cpu.hpp`: Electrum v1 100k SHA-256 stretch + v2 version
    byte verification + v2 PBKDF2 seed
- **Apple Metal port** of the v2 puzzle-mode kernel
  (`brain_wallet_v2.metal`). 1:1 with the CUDA implementation; KAT
  tests in `test_address_derive_cpu.cpp` apply to both.
- **License cache lifecycle tests.** Eight tests verify that the
  HMAC-bound 24h cache rejects every plausible tampering attempt
  (flipped valid bit, swapped email, pushed expiry, mismatched
  license key, expired cache).

### Pool server (collision-protocol)

- **DP_SUBMIT_V2 with work_id attestation.** Eight-byte work_id prefix
  on every DP submission. The server rejects mismatches and routes
  them straight to the IP-ban pipeline. v1 wire format remains for
  backwards compatibility with deployed v1.2.x clients.
- **IP-based ban escalation ladder.** 1h → 6h → 1d → 7d → permanent.
  Configurable thresholds and durations. Twenty-two unit tests cover
  every state transition, including 50-fold concurrent invalid-DP
  recording and the sliding-window count for prior bans.
- **Cryptographically secure uniform work distribution.** Replaces
  random.Random (Mersenne Twister, recoverable from ~624 outputs) with
  `secrets.SystemRandom` (kernel CSPRNG via `os.urandom`). Default
  distribution is uniform for puzzle work; historical-bias is opt-in
  via `use_history_bias=True` for brain-wallet sweeps.

### Protocol IDL + codegen

- **`protocol/jlp.yaml` is the single source of truth.** The codegen
  tool (`tools/codegen/jlp_codegen.py`) emits both
  `src/pool/jlp_wire_generated.hpp` (C++ packed structs with size
  asserts) and `data/protocol/jlp_protocol_generated.py` (Python
  dataclasses). CMake runs `--check` at configure; CI workflow at
  `.github/workflows/sync-protocol.yml` pushes the Python module to
  collision-protocol on every pro/main commit.
- **C++ drift detector** (`tests/test_jlp_wire_generated.cpp`) uses
  `static_assert` to catch any divergence between the legacy
  hand-written wire structs and the codegen output.

## Test additions

| Test target          | Phase | Where                                         |
| -------------------- | ----- | --------------------------------------------- |
| JLPWireGenerated     | 0     | tests/test_jlp_wire_generated.cpp             |
| Python codegen suite | 0     | tests/protocol/test_jlp_codegen.py (14 tests) |
| Anti-cheat suite     | 1     | tests/test_anti_cheat.py (17 tests)           |
| Work-id attestation  | 1     | tests/test_workid_attestation.py (5 tests)    |
| Work manager CSPRNG  | 2     | tests/test_work_manager_csprng.py (9 tests)   |
| BrainWalletV2 host   | 9     | tests/v2/test_brain_wallet_v2.cpp             |
| V2Orchestrator       | 9     | tests/v2/test_v2_orchestrator.cpp             |
| AddressDeriveCPU     | 4     | tests/v2/test_address_derive_cpu.cpp          |
| WeakPrngCPU          | 5     | tests/v2/test_weak_prng_cpu.cpp               |
| EncodingMungeCPU     | 6     | tests/v2/test_encoding_munge_cpu.cpp          |
| Pbkdf2CPU            | 7     | tests/v2/test_pbkdf2_cpu.cpp                  |
| ElectrumCPU          | 8     | tests/v2/test_electrum_cpu.cpp                |
| LicenseCache         | 10    | tests/test_license_cache.cpp                  |

## Cross-repo sync

- `scripts/sync-to-free.sh` is hardened: `protocol/`, `tests/protocol/`,
  and the entire `tools/` and `data/` trees stay private. The
  generated C++ header (`src/pool/jlp_wire_generated.hpp`) ships to
  Free as a regular source file.
- New CI workflow (`.github/workflows/sync-protocol.yml`) auto-pushes
  the generated Python protocol module to collision-protocol on every
  IDL change.

## Backlog (post-1.4.0, see `docs/PHASES-3-12-FOLLOWUP.md`)

These were scoped out of 1.4.0 but the algorithmic work is captured
in the CPU references shipped here:

- **GPU SHA-512** kernel (unblocks DerivationScheme S7 HMAC_SHA512_PW
  and S8 SHA512_PW_HALF; Phase 8 BIP-39 / Electrum kernels also
  need it)
- **Multi-address GPU kernel** (Phase 4 GPU half; the CPU reference
  - KATs are in)
- **Weak-PRNG GPU kernels** (one per CVE family; CPU references are
  in for MT19937 / Park-Miller / MSVC rand / java.util.Random)
- **CVE vector validation against known-cracked addresses** for
  libbitcoin bx and Profanity
- **Encoding-anomaly GPU kernel** wired into the v2 dispatch path
- **Modular legacy KDF dispatch table** in fused_pipeline (Phase 7
  GPU half)
- **Electrum v1/v2 GPU kernels** (Phase 8 GPU half; CPU references
  ready)
- **License lifecycle e2e** (live Stripe → webhook → cache → binary
  patch → verify; cache-format unit tests are in)

## Backwards compatibility

- Wire protocol: v1 messages (`DP_SUBMIT`, `DP_BATCH`) remain
  understood. Server rejects v2 messages from clients that haven't
  authenticated.
- CLI: existing flags (`--brainwallet`, `--pool`, `--puzzle N`,
  `--kangaroo`) unchanged. New flags are additive: `--puzzle-only-v2`,
  `--schemes`, `--addr-types`, `--puzzle-keys`.
- License cache file format unchanged. Old caches continue to work
  (the new HMAC verifier exposed for testing is the same algorithm
  read_cache has always used).

## Upgrade procedure

```
# Pro (private):
cd /path/to/collider-pro
git checkout main
git merge --no-ff phase-0-foundation
git merge --no-ff phase-9-puzzle-mode-integration
git tag -a v1.4.0-pro -m "v1.4.0 release"
git push origin main v1.4.0-pro

# Free (public):
cd /path/to/collider-pro
scripts/sync-to-free.sh v1.4.0-free

# Pool server (collision-protocol):
cd /path/to/collision-protocol
git checkout develop
git merge --no-ff phase-1-anti-cheat-tests
git merge --no-ff phase-2-csprng-uniform
git push origin develop
```

CI must build pro Windows + Linux releases for `v1.4.0-pro`. The
website (`https://collisionprotocol.com`) auto-pulls from
`github.com/hevnsnt/collider/releases/latest` so download links update
without a website edit.
