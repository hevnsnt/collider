# Ecosystem Restructure & Pro Cracking Upgrade — Implementation Plan

**Source of truth**: this document. Updated as phases land. PRs reference
the phase number in their titles.

> Sibling doc `IMPLEMENTATION-PLAN.md` covers the original Superflayer
> phased plan; this doc covers the May 2026 ecosystem restructure +
> brain-wallet/standard-puzzle upgrade work.

## Repository roles

| Repo                         | Visibility | Role                                                                                                         |
| ---------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------ |
| `hevnsnt/collider-pro`       | private    | source of truth for client + protocol IDL                                                                    |
| `hevnsnt/collider` (Free)    | public     | regenerated from pro via `scripts/sync-to-free.sh`, owns its own README + LICENSE + Free-only build workflow |
| `hevnsnt/collision-protocol` | private    | pool server (Python) + website (Next.js); consumes generated Python protocol bindings from pro               |

## Sync direction

```
collider-pro/main  --[scripts/sync-to-free.sh]-->  collider/main         (file mirror, Pro paths stripped)
collider-pro/main  --[CI extract step]-------->   collision-protocol     (generated jlp_protocol.py only)
```

No cross-repo submodules. No reverse syncs. Every cross-repo update
originates in pro/main.

## Phase status

| #   | Phase                                                    | Status                                                     | PR / Commit          |
| --- | -------------------------------------------------------- | ---------------------------------------------------------- | -------------------- |
| 0   | Protocol IDL + codegen + sync hardening + test harnesses | in progress                                                | `phase-0-foundation` |
| 1   | Anti-cheat unit tests vs current JLP server              | pending                                                    | —                    |
| 2   | Pollard's Kangaroo + crypto-secure uniform sweep starts  | pending                                                    | —                    |
| 3   | Multi-scheme derivation in `fused_pipeline.cu`           | pending                                                    | —                    |
| 4   | Multi-address-type + historical weak-key sweep           | pending                                                    | —                    |
| 5   | Weak-PRNG + starved-entropy kernels                      | pending                                                    | —                    |
| 6   | Encoding anomalies / data munging kernel                 | pending                                                    | —                    |
| 7   | Modular legacy KDF framework                             | pending                                                    | —                    |
| 8   | Electrum v1/v2 mnemonic kernels                          | pending                                                    | —                    |
| 9   | Puzzle-mode bloom check integration                      | pending (kernel exists on `feat/brainwallet-improvements`) | —                    |
| 10  | License lifecycle test suite                             | pending                                                    | —                    |
| 11  | Apple Metal kernel parity                                | pending                                                    | —                    |
| 12  | Future/staged work                                       | pending                                                    | —                    |

## Dependencies

- 0 blocks all others (IDL is the foundation)
- 4 blocks 5, 6, 7, 8 (multi-address layer is reused)
- 8 blocks 9 (Electrum kernels feed into puzzle-mode for completeness)
- 11 (Metal) parallels with any CUDA-only phase

## Per-phase exit criteria

Each phase is "done" only when:

1. All listed features implemented
2. CPU reference + GPU implementation byte-equal where applicable
3. Tests passing in local Catch2 / pytest harnesses
4. Tests passing in CI on Linux + Windows
5. Spec doc updated (if applicable)
6. PR merged to main

## Cryptographic test-vector sources

| Component                         | Source                         |
| --------------------------------- | ------------------------------ |
| SHA-256 / SHA-512                 | NIST CAVS                      |
| HMAC-SHA512                       | RFC 4231                       |
| BIP-32 derivation                 | BIP-32 spec test vectors       |
| BIP-39 mnemonic / PBKDF2          | BIP-39 spec test vectors       |
| BIP-49 (P2SH-P2WPKH)              | BIP-49 spec test vectors       |
| BIP-84 (P2WPKH)                   | BIP-84 spec test vectors       |
| BIP-86 (P2TR)                     | BIP-86 spec test vectors       |
| secp256k1 EC operations           | sipa/secp256k1 test vectors    |
| ECDSA RFC 6979 nonce              | RFC 6979                       |
| RIPEMD-160                        | published test vectors         |
| Electrum v1 seeds                 | electrum source test vectors   |
| Electrum v2 seeds                 | electrum source test vectors   |
| libbitcoin `bx seed` (Milk Sad)   | CVE-2023-39910 disclosure data |
| Profanity vanity tool             | CVE-2022-40769 disclosure data |
| Trust Wallet ext (CVE-2023-31290) | disclosure data                |
| Java Random                       | OpenJDK source                 |
| glibc `rand()`                    | glibc source                   |
| MSVC `rand()`                     | MSVC CRT documented constants  |
| Debian OpenSSL CVE-2008-0166      | DSA-1571-1 disclosure          |

## Branch / PR conventions

- One branch per phase: `phase-N-<short-name>`
- Each phase opens at most one PR against `main`
- PRs reference the phase number in the title
- No phase merges to `main` until the previous phase has merged
  (exception: phase 11 Metal can run in parallel)
