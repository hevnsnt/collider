# Brain-Wallet Scanner v2 — Design Spec

**Status**: Draft, in-progress on `feat/brainwallet-improvements`.
**Owner**: pool ops.
**Deadline**: when shipped.

## Motivation

The stock pipeline only catches one shape of brain-wallet:
`priv = SHA256(passphrase)` → uncompressed-P2PKH only. Empirical
session work (`puzzle-attack-prng-brute.py`, `puzzle-attack-string-seeds.py`,
the brain-wallet brute against 2.5 M phrases × 8 schemes from
Collider's own `data/`) showed that the wild-attack literature exploits
at least 4 _additional_ derivation classes that this pipeline cannot
currently see. Drained-recently wallets in the Milk Sad / Phantom
Keysmith / Profanity / Trust-Wallet incidents fall in those classes.

This spec lists the 12 improvements ranked by impact-per-implementation-cost,
with concrete file-by-file plans. Items 1, 2, 6, 9, 10, 12 land in this
PR. Items 3, 4, 5, 7, 8, 11 are detailed specs to be done in follow-up
PRs.

## Scope and risk

Every kernel change must:

1. Pass the existing `tests/test_brain_wallet_*.cpp` oracle (16-passphrase
   suite, currently in `tests/`).
2. Add new oracle entries for each new derivation scheme.
3. Not regress fused-pipeline single-scheme throughput by more than 5%.

The new modes are opt-in via runtime flags so the existing fast path
stays bit-identical.

---

## Item 1 — Multi-scheme derivation per passphrase _[implementing]_

**Why**: The single dominant cost in the fused pipeline is `ec_mul_optimized`.
Single-block SHA-256 is essentially free relative to it. So we can derive
multiple candidate `priv` values per passphrase, but each one still costs
a full EC multiply. The honest win is **breadth × work**, not free
breadth: 6 schemes ≈ 6 × current compute, 6× addresses checked. The
infrastructure savings (one wordlist load, one queue fill, one bloom
filter on-GPU) are real but small.

**Schemes to support per passphrase**:

| Tag  | Derivation                          | Source attack class                    |
| ---- | ----------------------------------- | -------------------------------------- |
| `S1` | `priv = SHA256(pw)`                 | bitaddress.org brain wallet (existing) |
| `S2` | `priv = SHA256(SHA256(pw))`         | double-hash brain wallets              |
| `S3` | `priv = SHA256(pw \|\| 0x0a)`       | textfile-with-newline brain wallets    |
| `S4` | `priv = SHA256(pw \|\| pw)`         | bash `echo $pw$pw \| sha256sum`        |
| `S5` | `priv = SHA256(SHA256(pw) \|\| pw)` | salted variant                         |
| `S6` | `priv = SHA256^16(pw)`              | KDF-lite brain wallet                  |
| `S7` | `priv = HMAC-SHA512(pw, "")[:32]`   | crude HKDF                             |
| `S8` | `priv = SHA512(pw)[:32]`            | lazy 512-truncation                    |

**File changes**:

- `src/gpu/fused_pipeline.cu`:
  - Add `__global__ void brain_wallet_fused_multi_kernel(...)` that
    accepts a 32-bit `scheme_mask`. Each set bit means "evaluate that
    scheme too". Per scheme, do the full priv → pub → h160 → bloom path.
  - Output match records carry both `passphrase_idx` and `scheme_id`.
- `src/gpu/brain_wallet_gpu.hpp`: declare new entry point + struct
  `MultiSchemeBatchResult { uint32_t pp_idx; uint8_t scheme_id; uint8_t addr_type; }`.
- `src/gpu/brain_wallet_gpu.cpp`: dispatch helper, replays match record
  on host to get the actual priv (deterministic given pw + scheme_id).
- `tests/test_brain_wallet_multi.cpp`: oracle for each S1..S8 with
  hand-computed (pw, scheme, expected_priv, expected_h160) tuples.

**Impact**: 8× breadth at 8× compute. The _combined_ pipeline still
saves ~10–15 % of total time vs running 8 separate scans because of
amortized wordlist I/O.

---

## Item 2 — Per-priv multi-address-type check _[implementing]_

**Why**: Stock pipeline computes `h160(SHA256(uncompressed_pubkey))`,
which matches **only** legacy P2PKH using the uncompressed pubkey. Modern
brain wallets and weak-RNG wallets use:

- P2PKH compressed (different h160; ~100% of post-2013 wallets)
- P2SH-P2WPKH (BIP-49, common 2017+)
- P2WPKH (BIP-84, common 2019+)
- P2TR x-only (BIP-86, 2022+)

For one EC multiply you can produce all five h160-equivalents at the
cost of ~4 extra hash blocks each. Almost free relative to EC_MUL.

**Five address derivations from one `priv`**:

```
1. uncompressed P2PKH:   h160(SHA256(0x04 || X || Y))
2. compressed P2PKH:     h160(SHA256(02_or_03 || X))
3. P2SH-P2WPKH:          h160(SHA256(0x00 0x14 || compressed_h160))
4. P2WPKH:               compressed_h160 (used directly as witness program)
5. P2TR x-only:          tweaked_x (BIP-86 derivation, x-only)
```

For (5), the BIP-86 tweak is `t = TaggedHash("TapTweak", x)`; the
output key is `Q = P + tG`. That's an extra EC add per check.

**File changes**:

- `src/gpu/fused_pipeline.cu`: extend `brain_wallet_fused_multi_kernel`
  loop body to compute all 5 h160s per priv and bloom-check each.
  Match records add `addr_type ∈ [0..4]`.
- `src/gpu/h160_bloom_filter.cu`: support tagged bloom (separate slices
  for different address types) so we can skip-check by type when the
  bloom is sparse.
- Bloom build script must include all 5 address-type extractions per
  funded UTXO (separate change tracked in item 3).

**Impact**: 5× addresses per priv at ~10% extra cost (dominated by
the 4 extra SHA-256 + 1 EC add). Best ROI of any change.

---

## Item 3 — Bloom filter content audit _[spec, follow-up]_

**Why**: A 5-way address scheme (item 2) is useless if the bloom only
contains uncompressed-P2PKH h160s.

**Plan**:

- Inventory current bloom build pipeline (script lives in `tools/`).
  Currently: walks UTXO set, extracts P2PKH h160 from each scriptPubKey.
- Extend extraction to all 5 script template types:
  - P2PKH (existing)
  - P2SH (extract script hash directly)
  - P2WPKH v0 (extract witness program)
  - P2WSH v0 (script hash, separate slice)
  - P2TR (x-only output key, separate slice)
- Build separate bloom slices per address type so per-priv item-2 lookup
  can target only the relevant slice (saves k hash ops per check).
- Estimated bloom size growth: ~3.5× current (more UTXOs survive the
  filter when all script types are included). Memory budget on the
  Blackwell box: 96 GB → 240 MB bloom is trivial.

**Out of scope for this PR**: implementation. Only spec.

---

## Item 4 — Weak-PRNG mode (Milk Sad / Phantom Keysmith / Profanity / etc.) _[spec + stub]_

**Why**: Drained ~120 K BTC across these incidents. The vulnerable
generator class is fundamentally different from brain wallet — it's a
deterministic PRNG seeded by 32-bit (sometimes 31-bit) entropy.
Brute-force the seed space, derive candidate keys per seed, bloom-check.

**Targets**:

- libbitcoin `bx seed` (MT19937, time-seeded; CVE-2023-39910)
- Profanity vanity tool (MT19937; CVE-2022-40769)
- Trust Wallet browser (MT19937; CVE-2023-31290)
- glibc `rand()` (Park-Miller LCG)
- Microsoft `rand()` (LCG, 32-bit seed)
- Java `Random` (48-bit LCG; can be seeded by `currentTimeMillis()`)

**Plan**:

- Per target: one device-side function `derive_priv_from_seed_v(uint64_t seed)`.
- New kernel `weak_prng_seed_brute_kernel(target_id, seed_lo, seed_hi)`
  iterates `[seed_lo, seed_hi)`, computes priv per target, runs item-2
  multi-address bloom check.
- 2³² seeds × 2 targets ≈ 4 × 10⁹ checks. At Blackwell pipeline
  throughput ~10⁸ priv/s including EC_MUL, this is ~80 s per scan per
  target. Cheap.

**File changes (eventual)**:

- New `src/gpu/weak_prng_kernel.cu` (device math for each target)
- New `src/gpu/weak_prng_brute.cpp` (host dispatcher)
- New `src/cli/weak_prng_mode.cpp` (CLI wiring)

**Stub in this PR**: declare the entry point, write a TODO test, leave
implementation for follow-up.

---

## Item 5 — BIP-39 mnemonic mode _[spec + stub]_

**Why**: Many brain wallets use 6–12 BIP-39 words instead of free
text. Different keyspace. The PBKDF2-HMAC-SHA512 step (2048 iterations
salt="mnemonic"+passphrase) is the cost driver but very GPU-friendly.

**Plan**:

- Generate candidate phrases by combining `data/crypto/bip39_*.txt`
  word lists in 3..12-word combinations (subject to BIP-39 checksum
  constraints — only `(words % 3 == 0)` valid lengths qualify).
- New kernel `bip39_mnemonic_kernel`:
  ```
  mnemonic_str → PBKDF2-HMAC-SHA512(salt="mnemonic", iter=2048) → 64-byte seed
  seed → BIP-32 master_key, master_chain_code
  derive child at fixed paths {m/44'/0'/0'/0/0..N, m/49'/.../0..N, m/84'/.../0..N, m/86'/.../0..N}
  for each derived priv: full multi-address check (item 2)
  ```
- PBKDF2-HMAC-SHA512 GPU implementation: 2 SHA-512 per HMAC × 2 HMAC
  per PBKDF2 round × 2048 rounds = ~8 K SHA-512 blocks per mnemonic.
  At Blackwell SHA-512 throughput ~10¹⁰ blocks/s → ~10⁶ mnemonics/s.
  Acceptable.

**File changes (eventual)**:

- New `src/gpu/bip39_kernel.cu`
- New `src/generators/bip39_phrase_combinator.hpp`
- New `tests/test_bip39_oracle.cpp` with BIP-39 test vectors from the spec

**Stub in this PR**: spec only.

---

## Item 6 — Puzzle-mode bloom check _[implementing]_

**Why**: For the Bitcoin Puzzle Transaction (and any future puzzle-style
challenge with masked deterministic-wallet keys), the address bloom is
the wrong target. The right check is "does this passphrase derive a
priv whose **masked form for any of N=1..160** equals a known puzzle key?"

**Algorithm**:

- Load puzzle key list onto GPU as `(N, low_bits_mask, low_bits_value)`
  triples in constant memory. ~160 × 12 bytes = 1.9 KB. Fits.
- For each (passphrase, scheme) candidate, after computing the candidate
  priv (256-bit integer):
  ```
  for each puzzle (N, mask, low_value):
      if (priv & mask) == low_value:
          report match (pp_idx, scheme_id, puzzle_N)
  ```
- ~160 cheap bitwise comparisons per candidate. Free relative to EC_MUL.

**Note**: Puzzle mode and address-bloom mode can run **simultaneously**
in the same kernel pass. The puzzle check executes after `priv` is
computed and **before** EC_MUL — that lets us short-circuit the EC_MUL
when puzzle-mode is the only target (saves the EC step entirely for
puzzle scanning).

**File changes**:

- `src/gpu/fused_pipeline.cu`: add `puzzle_check_inline` device function;
  add puzzle constants array as kernel parameter.
- `src/gpu/brain_wallet_gpu.hpp`: declare `set_puzzle_targets(...)` API
  and a new `MatchType { ADDR_BLOOM, PUZZLE }`.
- `src/gpu/brain_wallet_gpu.cpp`: host-side loader for the puzzle-target
  array.
- `tests/test_puzzle_mode.cpp`: oracle using the 79 known puzzle keys
  from `data/puzzle_history.json`.

**Impact**: makes Collider the obvious tool for puzzle-style problems
without affecting the existing brain-wallet path.

---

## Item 7 — Near-miss feedback to PCFG/Markov _[spec, follow-up]_

**Why**: PCFG/Markov generators are blind — they emit candidates with no
feedback. If a candidate's h160 matches the bloom in `M < k` slots
(where k is the bloom hash count), the candidate is a "partial hit"
worth investigating. With current binary check we throw it away.

**Plan**:

- Bloom check kernel returns the `(M / k)` ratio per candidate via an
  optional output buffer.
- Host post-processor sorts candidates by ratio, top-N feed back into
  the PCFG state machine as "high-prior" tokens for the next batch.
- Empirically this is a 2-5 % real-hit-rate boost in published
  password-cracking research.

**File changes**: bloom_filter.cu add scoring path, generator side
add a feedback buffer.

**Stub in this PR**: none. Defer.

---

## Item 8 — Multi-host distributed brain-wallet pool _[spec, follow-up]_

**Why**: The kangaroo pool already supports distributed work via JLP
protocol. Brain-wallet scanning could reuse the same pool model:
coordinator hands out `(wordlist_offset, rule_chain_offset)` chunks,
workers report match candidates and progress.

**Plan**:

- Reuse `src/pool/jlp_pool_client.{hpp,cpp}` infrastructure.
- New `MessageType::BW_WORK_REQ`, `BW_WORK_ASN`, `BW_HIT` messages.
- Server-side: `src/generators/streaming_brain_wallet.hpp` already has
  the chunk-emission interface; needs hooking to a network distributor.

**Defer**: this is a 1-2 week task on its own.

---

## Item 9 — Length-bucket batching for SHA-256 efficiency _[implementing]_

**Why**: GPU SHA-256 throughput on Blackwell is highest when all inputs
in a warp are the same length (no divergent control flow on
`while (len > 0) { ... }`). Stock pipeline mixes lengths within batches.

**Plan**:

- In `src/gpu/brain_wallet_gpu.cpp` batch dispatcher, sort the pending
  passphrase list by length before submitting to GPU. Then submit one
  kernel launch per length-bucket (or use the fixed-stride variant).
- Length-bucket boundaries: `[1..16, 17..32, 33..64, 65..128, 129..256]`.
- Empirical 30–50 % SHA-256 speedup on real wordlists where lengths
  vary.

**File changes**:

- `src/gpu/brain_wallet_gpu.cpp`: add `batch_by_length` flag, default on.
- New `LengthBucket` type internal to the dispatcher.

**Impact**: 30 % faster overall pipeline for free, no kernel changes.

---

## Item 10 — Resumable scanning state _[implementing]_

**Why**: Multi-day scans currently lose progress on crash. Persist
state every N seconds so we can resume.

**Plan**:

- New CPU-side checkpoint file at `~/.collider/brainwallet_state.json`:
  ```json
  {
    "wordlist_path": "...",
    "wordlist_offset": 12345678,
    "rule_chain_state": "best64:rule_id_47",
    "phase": "RULE_STACKING",
    "iteration": 7,
    "phrases_tested": 12345678901,
    "matches_found": 3,
    "started_at": "2026-05-08T00:00:00Z",
    "last_checkpoint": "2026-05-08T03:14:15Z"
  }
  ```
- Periodic flush every 30 s during scan.
- `--resume` flag picks up from latest checkpoint.

**File changes**:

- `src/generators/streaming_brain_wallet.hpp`: add checkpoint/restore.
- `src/cli/brain_wallet_mode.cpp` (or wherever the CLI driver lives).

**Impact**: lose at most 30 s on crash. Multi-day runs become safe.

---

## Item 11 — Wordlist freshness _[spec, trivial]_

**Why**: `data/passwords/` looks 2020-vintage. Diminishing returns but
cheap.

**Plan**:

- Pull RockYou2024 (~10 GB compressed, ~80 GB unzipped). Subsample
  top 100 M.
- Add 2024–2025 breach corpora from Have-I-Been-Pwned and other public
  sources.
- Update `data/passwords/README.md` with provenance table.

**Defer**: trivial but sensitive (download and verify checksums).

---

## Item 12 — Per-source / per-rule telemetry _[implementing]_

**Why**: After a 24-hour scan you should know which `data/` subdirs
actually produced matches. Currently the only output is the total hit
count.

**Plan**:

- Add `BrainWalletStats` struct with per-source and per-rule counters:
  ```c++
  struct BrainWalletStats {
      std::map<std::string, uint64_t> phrases_per_source;
      std::map<std::string, uint64_t> hits_per_source;
      std::map<std::string, uint64_t> phrases_per_rule;
      std::map<std::string, uint64_t> hits_per_rule;
      std::map<int, uint64_t> hits_per_scheme;     // item 1
      std::map<int, uint64_t> hits_per_addr_type;  // item 2
  };
  ```
- Print summary on shutdown / Ctrl-C / checkpoint.
- JSON output via `--stats-json /path/to/file.json`.

**File changes**:

- New `src/generators/brainwallet_stats.hpp/.cpp`.
- Hook into source/rule callbacks already present in
  `passphrase_generator.hpp`.

**Impact**: future tuning work has data instead of vibes.

---

## Implementation order in this PR

1. Spec doc (this file) ✓
2. Item 6 puzzle-mode (smallest, most isolated change)
3. Item 1+2 multi-scheme + multi-address (the big kernel rework)
4. Item 9 length bucketing (CPU-side, easy)
5. Item 10 resumable state (CPU-side)
6. Item 12 telemetry (CPU-side)
7. Stubs for 4 + 5 (declare APIs, write TODO tests)
8. Specs for 3, 7, 8, 11 (above)

Build target: must `make` clean on `feat/brainwallet-improvements` and
the existing oracle tests must pass unchanged.
