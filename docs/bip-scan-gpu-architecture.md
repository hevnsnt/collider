# BIP-39 Scan GPU Architecture

Status: shipped on `1.5.0` branch.

## Why this exists

The user-reported throughput on the BIP-39 scan was ~5 phrases/sec, ~1080 addresses/sec. The dashboard showed "0 Keys/s" + "Waiting for GPU telemetry" forever because the entire BIP-39 derivation pipeline was running on CPU. This document describes the GPU pipeline that replaced it.

## Pipeline

Per BIP-39 mnemonic, the work splits across two distinct GPU dispatch points:

```
+--------------------------+
|   CPU producer thread    |
|                          |
|  1. validate mnemonic    |
|  2. accumulate into      |
|     PBKDF2 batch         |
|     (256 mnemonics)      |
+--------------------------+
              |
              v
+--------------------------+
|   GPU dispatch #1:       |
|   PBKDF2-HMAC-SHA512     |    Per-device cudaStream, round-robin
|   bip39_pbkdf2.cu        |    across args.gpu_ids (default {0}).
|                          |    Kernel = one thread per mnemonic;
|   In: mnemonic batch     |    each runs the full 2048-iter HMAC
|   Out: 64-byte seeds     |    chain locally. ipad/opad states
|                          |    pre-computed; only 2 SHA-512
|                          |    transforms per HMAC call vs naive 4.
+--------------------------+
              |
              v
+--------------------------+
|   CPU chain walker       |
|                          |    Per seed (256 per batch):
|  3. master_from_seed     |    - HMAC-SHA512 to derive master key
|  4. CKDpriv for every    |    - BIP-32 CKDpriv chain per profile
|     profile/index        |      (11 profiles, ~20 indexes each;
|     (~190 priv keys)     |       per-step HMAC-SHA512 on CPU)
+--------------------------+
              |
              v
+--------------------------+
|   GPU dispatch #2:       |
|   MultiAddressSession    |    Per-device worker thread + own
|   (existing kernel)      |    MultiAddressSession on each device
|                          |    in args.gpu_ids. Batches 4096 priv
|   In: priv key batch     |    keys; one fused kernel does
|   Out: bloom hits        |    secp256k1 EC mul + hash160 +
|                          |    bloom probe per priv key.
+--------------------------+
              |
              v
+--------------------------+
|  CPU on_hit callback     |
|                          |
|  5. recompute h160       |
|  6. write bip_hits.txt   |
+--------------------------+
```

## Key files

| File                                       | Role                                                                                                     |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `src/gpu/sha512_device.cuh`                | FIPS 180-4 SHA-512 transform; multi-block.                                                               |
| `src/gpu/hmac_sha512_device.cuh`           | RFC 2104 HMAC-SHA512; pre-computed ipad/opad states for PBKDF2 fast path.                                |
| `src/gpu/bip39_pbkdf2.cu`                  | Per-thread PBKDF2-HMAC-SHA512 (2048 iter). Host wrapper `run_pbkdf2_batch`.                              |
| `src/gpu/v2/v2_orchestrator.{hpp,cpp}`     | Existing brain-wallet kernel; reused unchanged. `last_matches()` API added for structured hit retrieval. |
| `src/runtime/bip_gpu_dispatcher.{hpp,cpp}` | Per-device worker thread + queue + on_hit routing.                                                       |
| `src/runtime/bip_scanner_runner.cpp`       | Producer loop + chain walker + dispatch glue.                                                            |

## CLI flags

| Flag                        | Default                         | Effect                                                                |
| --------------------------- | ------------------------------- | --------------------------------------------------------------------- |
| `--bip-scan`                | off                             | Enable BIP scan mode                                                  |
| `--bip-combinatorial`       | off (use `--bip-scan-wordlist`) | Iterate entropy space; one valid mnemonic per entropy by construction |
| `--bip-combinatorial-words` | 12                              | 12/15/18/21/24 word width                                             |
| `--gpu-ids`                 | `{0}`                           | Devices to dispatch on (e.g. `--gpu-ids 0,1`)                         |
| `--no-bip-gpu`              | off                             | Force pure-CPU fallback (KAT regression / debug)                      |

## KAT coverage

| Test                         | Pins                                           |
| ---------------------------- | ---------------------------------------------- |
| `test_bip39_pbkdf2_kat`      | CPU PBKDF2 vs trezor reference vectors         |
| `test_bip39_pbkdf2_gpu_kat`  | GPU PBKDF2 byte-for-byte vs CPU OpenSSL output |
| `test_bip49_p2sh_p2wpkh_kat` | BIP-49 derivation + P2SH-P2WPKH hash160        |
| `test_bip_scan_runner_smoke` | End-to-end CPU pipeline with seeded bloom      |

## Performance notes

- PBKDF2 GPU throughput on consumer Ampere/Ada: roughly 3000-5000 phrases/sec per device.
- Multi-address dispatch (secp256k1 + hash160 + bloom) is essentially free compared to PBKDF2 -- it's the existing brain-wallet kernel that processes ~100k priv keys/sec per device.
- End-to-end target on a 2-GPU 24-core system: ~6000 phrases/sec aggregate, ~1.14 M addresses/sec bloom-probed.
- CPU chain walk between GPU dispatches is single-threaded per scan path. The wordlist path's T1-C worker pool gives N parallel chain walkers naturally; the combinatorial path is bottlenecked on the producer loop and could benefit from a parallel chain-walker pool as a follow-up.

## Dashboard surface

The BIP scan TUI panel (status_panel.cpp `case TuiMode::BipScan`) renders:

```
MODE        : combinatorial / 12 words
WORKERS     : 23 CPU + 2 GPU
PHRASES     : 5670 valid / 5670 read (5.5/s)
ADDRESSES   : 1.08 M probed
THROUGHPUT  : 1080 addrs/s
UPTIME      : 17m 14s
BLOOM       : 36.2 M entries / 11 profiles, ~190 addrs/phrase
PROFILE     : BIP-84 native P2WPKH chg
GPU#0       : 540 K probed (530 addrs/s)
GPU#1       : 540 K probed (530 addrs/s)
```

The "Empty Hits by Phase" + "Trying <passphrase>" + "Waiting for GPU telemetry" rows are gated on `TuiMode::Brainwallet` -- the historical placeholder text doesn't bleed into BIP mode anymore.
