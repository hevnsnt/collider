# --track-empty-hits Workflow

Demonstrate the brainwallet scanner is finding **real keys** — just keys
whose Bitcoin has already been spent by their original owner. With a
seen-ever address bloom + a currently-funded UVRF, every bloom-positive
candidate that fails the UVRF lookup is logged as a "real but empty"
hit. The running status line shows `Empty Wallets Found: N` and each hit is appended
to `found-empty.txt` with the passphrase, privkey, and h160.

This guide walks through extracting the two address sets from a local
Bitcoin Core full node.

## Inputs you need

- A fully-synced Bitcoin Core node (path to its data dir, e.g.
  `/mnt/hdd/bitcoin/bitcoin` on the user's box).
- Enough disk to hold ~30 GB of intermediate CSVs (compressible).
- The two binary tools shipped in the Pro build:
  - `extract_node_addresses` (C++) — scans `blk*.dat` for seen-ever.
  - `build_bloom` (C++) — produces both `.blf` and `.uvrf` files.
- `scripts/extract_funded_utxos.py` (Python 3, stdlib only) — parses
  `bitcoin-cli dumptxoutset` output.

## Step 1: extract seen-ever addresses

Scan every block file. **`bitcoind` does NOT need to be running** — the
tool reads `blk*.dat` directly. If your node IS running it's still safe;
Bitcoin Core writes block files atomically.

```bash
./extract_node_addresses /mnt/hdd/bitcoin/bitcoin/blocks -o seen_raw.csv
```

Time: ~2-4 hours single-threaded on a commodity SSD (1 TB of blocks at
80-150 MB/s). Output is one h160 per line, with duplicates.

Output format:

```
1ec84ff80a8459e72ae04bfe6e5c01bd34a16f00,P2PKH
4a3f7f1ab16f5cdc91e1f2a7b4d5e6c8d9e0f1a2,P2WPKH
...
```

Stats at the end on stderr:

```
[*] Done in 8421.3s
    blocks scanned : 856321
    transactions   : 1042157893
    outputs total  : 3128947128
    P2PKH          : 1287456321
    P2SH           : 487123498
    P2WPKH         : 1054872194
    skipped (non-h160 scripts: P2WSH/P2TR/P2PK/multisig/...) : 299495115
```

## Step 2: dedupe + sort

```bash
sort -u seen_raw.csv -o seen_sorted.csv
```

Expected: input ~30-40 GB (3B outputs × ~10 chars/line), deduped output
~25-30 GB (~1.2B unique h160 entries on mainnet as of 2026).

`sort -u` is happy with files larger than RAM (it spills to `$TMPDIR`).
Set `TMPDIR=/mnt/hdd/tmp` if your `/tmp` is small.

## Step 3: build the seen-ever bloom

```bash
./build_bloom -i seen_sorted.csv -o seen.blf
```

A ~3 GB bloom at 1% false-positive rate for 1.2B entries. `build_bloom`
also accepts `-f 0.001` for a tighter FP rate (~5 GB output) — useful
if you want to minimize empty-hit FPs.

## Step 4: extract the currently-funded UTXO set

```bash
bitcoin-cli dumptxoutset /tmp/utxos.dat
python3 scripts/extract_funded_utxos.py /tmp/utxos.dat -o funded.csv
```

- `dumptxoutset` takes ~5-15 minutes for the full UTXO set (~80M
  UTXOs in 2026).
- The Python parser is stdlib-only — no external deps.
- Output:
  ```
  1ec84ff80a8459e72ae04bfe6e5c01bd34a16f00,2500000,P2PKH
  ```
  (h160 hex, amount in satoshis, script type).

## Step 5: build the funded UVRF

```bash
./build_bloom -i funded.csv -v funded.uvrf
```

The `-v` flag tells `build_bloom` to emit a UVRF (UTXO Verification
file) instead of a bloom. UVRFs are exact-membership and carry the
amount metadata so a verified hit knows the wallet's balance.

## Step 6: scan

```bash
./collider_pro --brainwallet \
    --bloom seen.blf \
    --verify-set funded.uvrf \
    --track-empty-hits \
    --wordlist /path/to/big_passphrase_list.txt
```

Status line during the scan:

```
[*] Checked: 1.2M | Speed: 980K/s | Hits: 0 | Empty Wallets Found: 47
[*] Trying: Phase 2 | "satoshi2014"
```

Every "Empty" tick appends a line to `found-empty.txt` in the working
directory:

```
2026-05-11T18:42:07Z  privkey=a3b1...c92e  h160=1ec8...4f00  passphrase="i love bitcoin"
```

At end-of-scan, the summary box adds:

```
Real-but-empty hits: 1247  (see found-empty.txt)
```

## How does this differ from a funded-only bloom?

With **--bloom funded.blf --verify-set funded.uvrf**, every bloom hit
that fails UVRF is just a bloom FP (probability ~2e-7 per query).
`--track-empty-hits` would just log noise.

With **--bloom seen.blf --verify-set funded.uvrf** (this workflow),
bloom hits that fail UVRF are **real wallets that no longer hold
funds**. The bloom FP rate sets the noise floor; everything above is
real-but-empty hits.

## Sanity check before a real run

```bash
# Tiny smoke: parse only the first 10 blk*.dat files (~40 GB of blocks)
./extract_node_addresses /mnt/hdd/bitcoin/bitcoin/blocks \
    --max-files 10 -o smoke.csv

# Make sure you get sane counts; ratio P2PKH : P2SH : P2WPKH should be
# roughly 5 : 1 : 2 across the chain's lifetime.
wc -l smoke.csv
awk -F, '{print $2}' smoke.csv | sort | uniq -c
```

## What about P2PK / multisig / P2TR?

- **P2PK** (bare pubkey, pre-2012): ~0.4% of historic outputs. The
  pubkey is in the output script, but we'd need a SHA256+RIPEMD160
  pass to convert it to an h160 for the bloom. Not implemented; if
  you need P2PK coverage, extend `extract_h160()` in
  `extract_node_addresses.cpp`.
- **Multisig** (bare `m-of-n`): ~0.01% of outputs. Same as P2PK —
  each pubkey would need to be hashed. Not implemented.
- **P2WSH / P2TR** (32-byte witness programs): not h160-shaped, so
  they can't be probed by the h160 brainwallet bloom. The whole
  brainwallet scan family is h160-only by design; P2WSH/P2TR brain
  wallets are vanishingly rare.

The skip rate from `extract_node_addresses` (`skipped` line in the
final stats) is your audit. Mainnet 2026: ~9% of outputs skipped,
~91% covered.
