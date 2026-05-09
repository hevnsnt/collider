# theCollider Pro

The Free version is the pool-mining client documented in the [main README](../README.md). Pro adds **brain-wallet attack capabilities** to the same binary — a license unlocks the Pro code paths; no separate download or installation.

This page describes what Pro does, how it differs from Free, and how to acquire it.

---

## Why Pro

The Bitcoin Puzzle pool gets you a piece of the prize when the pool solves a puzzle. That's a real-but-finite return: ~5% of one of the unsolved puzzles per ~year if you contribute your fair share.

Pro turns the same GPU into a brain-wallet attack engine. Three concrete revenue streams:

1. **Opportunistic brain-wallet detection.** Every passphrase your GPU evaluates during pool mining can ALSO be checked against a 50M+ funded-address bloom filter. If any candidate hits, you've found a (likely abandoned) brain wallet — a free bonus on top of pool credit.
2. **Dedicated brain-wallet cracker.** Targeted attacks against specific weak-RNG events: libbitcoin "Milk Sad", Profanity vanity tool, Trust Wallet Extension. Each disclosed CVE is a small enumerable space; once it's swept you can stop.
3. **Custom passphrase wordlists.** Run your own dictionary against the bloom. PCFG / Markov / rule-engine generation included.

Pro users typically run Pro mode on idle GPUs (the pool path is more efficient when the GPU is fully utilized; brain-wallet attacks fill in the gaps).

## What's in Pro

### 1. Brain Wallet v2 (multi-scheme + multi-address)

The same passphrase can produce different priv keys depending on the derivation scheme. Pro evaluates **all 8 schemes** in a single GPU pass:

| Scheme | Recipe |
|---|---|
| `sha256_pw` | `priv = SHA256(passphrase)` |
| `sha256_sha256_pw` | `priv = SHA256(SHA256(passphrase))` |
| `sha256_pw_newline` | `priv = SHA256(passphrase \|\| 0x0a)` |
| `sha256_pw_pw` | `priv = SHA256(passphrase \|\| passphrase)` |
| `sha256_sha256_pw_pw` | `priv = SHA256(SHA256(passphrase) \|\| passphrase)` |
| `sha256_iter_16` | `priv = SHA256^16(passphrase)` |
| `hmac_sha512_pw` | `priv = HMAC-SHA512("", passphrase)[:32]` |
| `sha512_pw_half` | `priv = SHA512(passphrase)[:32]` |

Each priv is then derived to all **5 modern Bitcoin address types** in parallel:

| Address type | Description |
|---|---|
| `p2pkh_uncompressed` | Legacy P2PKH (uncompressed pubkey, e.g. `1...`) |
| `p2pkh_compressed` | Legacy P2PKH (compressed pubkey, modern default) |
| `p2sh_p2wpkh` | BIP-49 wrapped SegWit (e.g. `3...`) |
| `p2wpkh_v0` | BIP-84 native SegWit (e.g. `bc1q...`) |
| `p2tr_bip86` | BIP-86 Taproot (e.g. `bc1p...`) |

Selectable via the CLI:

```bash
./collider --brainwallet --bloom funded.blf \
           --schemes all \
           --addr-types modern
```

`--schemes` accepts `all`, `stock`, or a comma list. `--addr-types` accepts `all`, `modern`, `stock`, `puzzle_only`, or a comma list.

### 2. Puzzle-only mode

When you're not running an address-bloom probe, Pro can sweep passphrases against the **known puzzle keys** (the 79 historically solved puzzles plus the 81 funded ones). The kernel short-circuits before the expensive EC multiply when no puzzle target hits, giving roughly **20x speedup** vs the legacy stock pipeline.

```bash
./collider --puzzle-only-v2 \
           --schemes all \
           --puzzle-keys ./data/puzzle_history.json
```

This mode is the cheapest way to validate a wordlist against the puzzle set. Hits are extremely unlikely (the puzzle set is small), but the cost per passphrase is so low that running it as a background task is essentially free.

### 3. Weak-PRNG kernels

Each disclosed weak-RNG vulnerability has a small, enumerable seed space. Pro ships dedicated GPU kernels for six families:

| Family | CVE / source | Seed entropy | Mapped derivation |
|---|---|---|---|
| `libbitcoin_bx_seed` | CVE-2023-39910 ("Milk Sad") | 32-bit time_t | Legacy P2PKH |
| `profanity` | CVE-2022-40769 | MT19937 with 32-bit seed | Legacy P2PKH |
| `trust_wallet_ext` | CVE-2023-31290 | MT19937 with weak seed | BIP-84 native SegWit |
| `glibc_rand` | n/a (defensive) | Park-Miller LCG, 32-bit | Legacy P2PKH |
| `msvc_rand` | n/a (defensive) | 32-bit LCG | Legacy P2PKH |
| `java_random` | n/a (defensive) | 48-bit LCG | Legacy P2PKH |

Each family has its own derivation path. The Trust Wallet kernel maps strictly to BIP-84 (per the CVE disclosure); the others map to legacy P2PKH because that's how the affected wallets historically dumped to chain. Per-family kernel launches keep register pressure low and let nvcc inline each PRNG state-advance.

A full 2^32 sweep on Profanity takes minutes on a single 4090. Milk Sad's time-window of `[disclosure_start, disclosure_end)` is roughly 2^31 candidates and finishes in similar time.

### 4. Encoding-anomaly kernel

Many real-world brain-wallet generators encode the user's passphrase via the language's default codec, not UTF-8. Pro mutates each passphrase through 10 encodings before hashing:

- UTF-8 (identity)
- UTF-16-LE (Windows / .NET default)
- UTF-16-BE (Java `getBytes()` no-charset)
- UTF-32-LE / UTF-32-BE
- Latin-1 (8-bit truncation)
- strip-non-ASCII (drop bytes >= 0x80)
- upper-ASCII / lower-ASCII (case fold)
- null-terminated (append `\0`)

Plus a `--double-hash` toggle that runs SHA-256d (`SHA256(SHA256(passphrase))`) on the chosen encoding instead of single SHA-256.

Surrogate-pair handling is correct. Inputs that can't be represented in the target encoding (e.g. emoji into Latin-1) are skipped without producing a false-positive hit.

### 5. Modular legacy KDF framework

`fused_pipeline.cu`'s initial hashing step is modular. Pro ships five legacy KDFs that defunct early web-wallet generators used:

| KDF id | Recipe |
|---|---|
| 0 | `SHA256(pw)` (baseline / parity with stock) |
| 1 | `MD5(pw) \|\| MD5(MD5(pw) \|\| pw)` (early PHP web wallets) |
| 2 | `SHA-1(pw) \|\| SHA-1(SHA-1(pw))` (early MultiBit Classic) |
| 3 | `SHA-256^1024(pw \|\| "BTC-SALT")` (stretching loops) |
| 4 | `MD5 -> SHA-256 -> RIPEMD-160` mixed chain |

Adding a new KDF takes ~20 lines of CUDA in `legacy_kdf_kernel.cu` plus a registry entry. We accept PRs that add CVE-disclosed KDFs.

### 6. Electrum v1 + v2 mnemonic kernels

Electrum predates BIP-39 and uses non-standard seed derivation:

- **Electrum v1**: 12 hex words decode to a 32-char hex seed; priv is derived via 100,000 SHA-256 chained iterations. CPU is the bottleneck (each candidate is ~100k SHA-256s). GPU brings it down by 100-200x.
- **Electrum v2**: 12 word mnemonic; version byte verified via HMAC-SHA512("Seed version", mnemonic). Valid mnemonics derive seed via PBKDF2-HMAC-SHA512(2048 iterations, salt=`"electrum"+passphrase`). Same PBKDF2 primitive used by BIP-39, but a different salt prefix.

```bash
./collider --electrum-v1 --wordlist seeds.txt --bloom funded.blf
./collider --electrum-v2 --wordlist mnemonics.txt --passphrase ""
```

### 7. Historical CVE sweep (Debian OpenSSL + Android SecureRandom)

Two specific historical events produced enumerable broken-RNG keyspaces that have ALREADY been partially swept by other parties, but new addresses are still being added to the funded set:

- **CVE-2008-0166 (Debian OpenSSL)**: a 17-bit (PID) × 16-bit (time_t) keyspace; ~32k candidate keys per second window.
- **Android SecureRandom 2013**: 64-bit Java seed, but in practice ~32 bits of entropy due to PID + low-resolution time on affected devices.

Pro ships enumerators for both. A modern address derivation (P2WPKH, BIP-49 wrapped) is run over each candidate; if a current funded address has been generated by a key that originated in either of these compromised pools, you find it. This is rare but high-EV: each hit is a fully-controlled wallet.

```bash
./collider --historical-sweep debian_openssl --addr-types modern --bloom funded.blf
./collider --historical-sweep android_securerandom_2013 --addr-types modern --bloom funded.blf
```

### 8. Bloom filter database

The Pro license includes a curated bloom filter at `data/funded.blf`:

- **50M+ funded Bitcoin addresses** (current UTXO set + historical never-emptied addresses)
- Configured for **0.0001 false-positive rate**
- 64-byte MurmurHash3 double-hashing scheme (matches the kernel probe)
- Updated monthly from a fresh UTXO snapshot; subscribers can opt in to delta updates

The Free version cannot use `--bloom` even with a self-built bloom file (the flag is gated). Self-built blooms work in Pro using `tools/build_bloom`.

### 9. Interactive menu

Pro launches into a guided setup wizard if no flags are passed. Walks the user through:

- Selecting a wordlist (built-in dictionaries or custom)
- Choosing scheme/address coverage (stock-only, modern, all)
- Loading a bloom filter
- Resume-from-checkpoint detection

The menu also surfaces pool stats and historical solve rates; helpful for tuning `--dp-bits` if you're solo-mining.

### 10. Resumable + checkpoint-aware

Long sweeps survive process death. Pro autosaves every `--save-interval` (default 1M passphrases) and resumes via `--resume`. Checkpoint files live in `./checkpoints/` and are compatible across versions.

## Free vs Pro feature matrix

| Capability | Free | Pro |
|---|---|---|
| Pool mining (JLP, TLS, anti-cheat) | ✓ | ✓ |
| Solo Kangaroo on small puzzles | ✓ | ✓ |
| Benchmark | ✓ | ✓ |
| Stock brain-wallet (`--brainwallet`) | – | ✓ |
| Multi-scheme + multi-address | – | ✓ |
| Bloom filter probe | – | ✓ |
| Weak-PRNG kernels | – | ✓ |
| Encoding-anomaly kernel | – | ✓ |
| Legacy KDF framework | – | ✓ |
| Electrum v1/v2 | – | ✓ |
| Historical CVE sweep | – | ✓ |
| PCFG / Markov / rule engine | – | ✓ |
| Interactive menu | – | ✓ |
| Resumable scanning state | – | ✓ |

The same binary handles both. Attempting a Pro flag in Free emits:

> *I'm sorry, but this is a pro function. If you'd like to try pro, go and visit the website to purchase and download.*

## Pricing

Pro is a **one-time license** valid for the major version (1.x). Free upgrades for the lifetime of the v1 series.

Current price: **$49.99** (one-time). License attaches to a Bitcoin payout address; no recurring fees, no telemetry phone-home, no DRM beyond the binary-embedded license slot.

Purchase + download flow: [collisionprotocol.com/pro](https://collisionprotocol.com/pro).

## License validation

The Pro binary embeds a license slot that signs against an on-disk `~/.collider/license.cache`. The cache is renewed once every 24 hours via a Cloud Function; once cached, the binary works offline indefinitely (until the cache is tampered with, at which point the HMAC verification fails and the Pro features stop working until you re-validate).

Validation checks (all must pass):

- License key matches what's embedded in the binary
- Cache HMAC-SHA256 over `(license_key | valid_bit | email | expiry_epoch)` matches the recorded value
- `now < expiry_epoch`

Tampering with any field invalidates the HMAC; the binary falls back to Free behavior.

If the Cloud Function is unreachable, the cache continues to validate against its existing 24-hour window. After expiry, the binary refuses to start in Pro mode until network is restored.

## License Q&A

**Q: Can I use Pro on multiple machines?**
A: Yes. The license is per-purchase, not per-machine. Run on as many GPUs as you own.

**Q: What if I lose access to the license file?**
A: Email `support@collisionprotocol.com` with the Bitcoin address you paid from; we re-issue.

**Q: Can I share a license with a friend?**
A: One purchase = one license. Sharing isn't technically prevented (the cache is just a file) but you're on the honor system.

**Q: What if a major version (2.x) ships?**
A: The 1.x license stays valid for 1.x forever. 2.x will be a separate purchase if it ships. We expect 2.x not to ship for a long time.

**Q: Do you sell to entities outside the US / under sanctions?**
A: We follow OFAC. Sales to sanctioned jurisdictions are blocked at checkout.

**Q: What does the bond mechanism look like?**
A: The pool operator's 25 BTC bond is documented in [collisionprotocol.com/whitepaper](https://collisionprotocol.com/whitepaper). It is independent of the Pro license — pool participation is free regardless of license status.

## Source-of-truth pointer

Pro feature implementations live in the private [`hevnsnt/collider-pro`](https://github.com/hevnsnt/collider-pro) repository. The public [`hevnsnt/collider`](https://github.com/hevnsnt/collider) repo is regenerated from the private source via `scripts/sync-to-free.sh` with all Pro paths stripped. So the public repo is genuinely the Free distribution; the Pro feature code does not appear there at all.

If you want to audit the Pro feature implementations before purchasing, we publish redacted previews of each kernel under `collisionprotocol.com/audit/`. The full source is available to contracted security researchers under NDA.
