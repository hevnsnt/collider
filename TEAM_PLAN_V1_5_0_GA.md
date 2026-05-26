# theCollider Pro v1.5.0 GA Sprint Plan (Final / 2026-05-23)

## Sprint progress (2026-05-23 update)

**Shipped this sprint:**

- ✅ T1-E BIP-39 trezor PBKDF2 KAT (6/6 vectors green)
- ✅ T1-F BIP-49 P2SH-P2WPKH KAT (3/3 spec vector + e2e)
- ✅ T1-D BIP scanner end-to-end smoke ctest (6/6)
- ✅ T1-C BIP scanner multi-threaded fan-out (hardware_concurrency-1 workers)
- ✅ B1 server-side wire-v4 scaffolding (PR #19; 13/13 KAT; bech32 + ECDSA verify + canonical message)
- ✅ Hoist hash160_pubkey + hash160_p2sh_p2wpkh to runtime/bip_address.hpp
- ✅ Pushed collider-pro 1.5.0 branch to origin (was local-only)
- ✅ ctest baseline 76 -> 79

**Remaining for production-ready:**

- B1 client-side wire-v4 in collider-pro (multi-file: bech32 + WIF + ECDSA-sign via OpenSSL + JLPClientHelloV4 + CLI --worker-key + tools/generate_worker_key)
- B1 cutover PR on collision-protocol (workers.pubkey schema migration + PROTOCOL_VERSION_MIN bump 3->4 + \_authenticate routing)
- T1-A run_brainwallet_interactive cout -> TUI modal port (430 lines, 5 modals)
- T1-B run_puzzle_interactive cout -> TUI modal port
- B2 release ship: tag v1.5.0 + signed binaries + GitHub release page
- B9 sweep.expected_destination_address pinned in VPS config.yaml
- B10 rotate C-1 leaked license keys

**Source of truth.** Drives the work from current state to production-ready v1.5.0.

## Current state

- collision-protocol: 12 PRs merged + deployed to VPS. Pool healthy. CRIT-1 architectural hole open (no identity binding; 2-connection attacker defeats v1.5 asymmetric).
- collider-pro: 1.5.0 branch local-only with audit fixes + features (BIP combinatorial, TUI main menu, banner v2, fingerprint perf). security/client-deep-batch-1 has CLIENT-LIC-1 (machine-id HMAC) + CLIENT-LIE-1 + CLIENT-LIE-3. 76/76 ctest green.
- Audit Pass 2 grades: Pool A, Brain wallet A-, BIP B+, TUI A-, Menus C+, Text A-, Build A-, Crypto A. Two areas below A bar.

## GA blockers (must close for production-ready)

### B1: CRIT-1 wire-v4 signed AUTH

Architectural close: workers prove possession of their BTC address by signing the AUTH frame with the address's private key. No password required (user-rejected). Wire format bump to PROTOCOL_VERSION=4. Servers reject <4 with UPGRADE_REQUIRED.

Scope:

- **collider-pro side (CLIENT-CHEAT-2):**
  - `--worker-key <wif>` CLI flag, loaded once at startup into SecureBuffer
  - `tools/generate_worker_key.cpp`: ECDSA keypair gen + bech32 P2WPKH address derivation, written to a 0600 wif file
  - AUTH frame extension: 33-byte compressed pubkey + 64-byte ECDSA signature
  - Canonical signed message: `"COLLIDER-WORKER-AUTH-v1\n" || PROTOCOL_VERSION_byte || timestamp_ms_u64 || nonce_16 || worker_name_pascal`
  - `--worker` must match bech32(hash160(pubkey)) or AUTH fails client-side
- **collision-protocol side (SRV-LIE-15):**
  - AUTH decoder reads pubkey + signature
  - Verify ECDSA over canonical message
  - Verify bech32(hash160(pubkey)) == worker_name
  - First-AUTH binds pubkey to workers table row; subsequent AUTHs must match
  - Wire PROTOCOL_VERSION_MIN bumped 3 -> 4

Acceptance:

- Worker can complete AUTH only by signing with the correct privkey
- Name hijack impossible (mismatched pubkey -> bech32 mismatch -> AUTH_FAIL)
- 2-connection attacker can still open 2 distinct pubkeys, BUT server refuses to assign opposite kangaroo types to two pubkeys that hash160 to the same /24 IP block within the puzzle lifetime (defense-in-depth)

### B2: Ship collider-pro 1.5.0

- Push `1.5.0` branch to origin
- Push `security/client-deep-batch-1` (or rebase into 1.5.0)
- Tag `v1.5.0`
- Build signed release binaries (Windows x64, Linux x64, macOS arm64)
- Upload to GitHub releases page
- Update download endpoint on website to serve v1.5.0

### B3: T1-A run_brainwallet_interactive -> TUI modals

430 lines of cout in src/ui/interactive_ui.cpp:365-796 ported to FTXUI modals:

- wordlist_picker_modal
- bloom_picker_modal
- resume_confirm_modal
- multi_address_picker_modal
- brainwallet_confirm_modal (uses existing confirm_config_modal)

### B4: T1-B run_puzzle_interactive -> TUI modals

- pool_config_modal (URL + worker entry; needs TUI text input widget)
- puzzle_number_picker_modal (1..256 + auto)
- puzzle_confirm_modal (uses confirm_config_modal)

### B5: T1-C BIP scanner multi-threaded fan-out

src/runtime/bip_scanner_runner.cpp: per-phrase derivation is embarrassingly parallel. Thread pool sized to hardware_concurrency(), lock-free phrase queue, single mutex-guarded hits writer.

### B6: T1-D BIP scanner smoke ctest

tests/test_bip_scan_runner_smoke.cpp: 5-phrase fixture + tiny .blf seeded with one known address. Assert exactly 1 hit.

### B7: T1-E BIP-39 trezor PBKDF2 KAT

tests/test_bip39_pbkdf2_kat.cpp: 24 trezor vectors through bip32::mnemonic_to_seed.

### B8: T1-F BIP-49 P2SH-P2WPKH KAT

tests/test_bip49_p2sh_p2wpkh_kat.cpp: known BIP-49 mnemonic + path -> expected P2SH address.

### B9: SWEEP-2 enforcement on VPS

Add `sweep.expected_destination_address` to /opt/collision-protocol/config.yaml on VPS. Operator runs `python3 -m tools.derive_sweep_address` to get the current bech32, pins it.

### B10: License key rotation (operator action)

Up to 20 keys returned by C-1 diag routes between launch and PR #5 merge. Operator must rotate in Firebase admin.

## Sprint execution order

Sequential where dependencies require it; parallel where not.

1. **B1 wire-v4 design + code** (largest, do first to unblock B2 release semantics)
2. **B7, B8** KATs (small, parallel with B1)
3. **B6** smoke ctest (small)
4. **B5** BIP threading
5. **B3, B4** TUI ports (largest after B1)
6. **B9** SWEEP-2 config (operator)
7. **B2** release ship
8. **B10** key rotation (operator)

## What only the operator can do

- B2 binary signing (operator has the signing key)
- B9 generate the sweep destination bech32 + edit /opt/collision-protocol/config.yaml on VPS
- B10 rotate license keys in Firebase admin

## Definition of done

- CRIT-1 closed via wire-v4 (B1 deployed both sides)
- collider-pro v1.5.0 tagged + binaries shipped (B2)
- 76 ctest baseline grown to >= 80 (B6, B7, B8)
- run_brainwallet_interactive + run_puzzle_interactive zero cout/cin (B3, B4)
- BIP scanner threads (B5)
- SWEEP-2 pinned in VPS config.yaml (B9)
- Up-to-20 leaked license keys rotated (B10)
- Pass-3 audit returns 9/9 A grades
