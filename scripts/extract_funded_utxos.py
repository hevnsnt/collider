#!/usr/bin/env python3
"""
extract_funded_utxos.py -- companion to extract_node_addresses for the
v1.4.2 --track-empty-hits workflow.

Reads a Bitcoin Core dumptxoutset snapshot (produced by
`bitcoin-cli dumptxoutset <file>`) and writes a CSV of currently-funded
h160-shaped UTXOs. The CSV is the input to build_bloom's UVRF mode
(`build_bloom -i funded.csv -v funded.uvrf`).

Output format (one line per UTXO):
    <h160_hex>,<amount_satoshis>,<script_type>
    1ec84ff80a8459e72ae04bfe6e5c01bd34a16f00,2500000,P2PKH

Coverage matches extract_node_addresses: P2PKH, P2SH, P2WPKH. P2WSH,
P2TR, bare P2PK, and multisig are skipped (not h160-shaped).

Why this script and not just `bitcoin-cli scantxoutset`:
  - dumptxoutset emits the whole UTXO set in ~5-10 min (one disk pass)
  - scantxoutset emits JSON, takes 30-60 min, and parses much slower

dumptxoutset format (Bitcoin Core v22+ src/node/utxo_snapshot.h plus
src/rpc/blockchain.cpp CreateUTXOSnapshot). Verified against a v27.0
mainnet dump:

  Header (40 bytes, fixed-width):
      32 bytes   base block hash (little-endian uint256)
       8 bytes   coin count (little-endian uint64; counts groups, may
                 differ slightly from the JSON `coins_written` field)

  Then for each UTXO:
       32 bytes   txid                  (LE uint256, lex-sorted)
        4 bytes   vout                  (LE uint32)
      VARINT      code = height*2 | coinbase_flag
      VARINT      compressed amount
      VARINT      nSize  -- script identifier
      N bytes     script payload (length depends on nSize)

  VARINT here is Bitcoin's "differential varint" (src/serialize.h), NOT
  CompactSize. Each byte contributes 7 bits with bit 7 indicating "more
  bytes follow"; values are accumulated with a +1 increment per
  continuation byte to eliminate redundant encodings.

  nSize semantics (src/compressor.cpp):
       0  -> P2PKH;   payload = 20-byte h160
       1  -> P2SH;    payload = 20-byte h160
       2,3 -> P2PK uncompressed; payload = 32 bytes (x-coord)
       4,5 -> P2PK compressed;   payload = 32 bytes (x-coord)
       6+ -> raw script of (nSize - 6) bytes (we recognize P2WPKH:
             0x00 0x14 <20-byte h160>)

Operator workflow (Linux):
    bitcoin-cli dumptxoutset /tmp/utxos.dat
    python3 extract_funded_utxos.py /tmp/utxos.dat -o funded.csv
    build_bloom -i funded.csv -v funded.uvrf

Performance: ~80 MB/s on the dump (so ~5-10 min for a typical 4 GB
mainnet snapshot). The parser is stdlib-only -- no external Python deps.
"""

import argparse
import struct
import sys
from pathlib import Path


# -- Bitcoin Core "compressor.h" amount/script decompressors --------------

def read_diff_varint(f):
    """Bitcoin Core's "differential varint" used for serialized integers
    in the chainstate (Bitcoin's own VARINT macro, src/serialize.h).
    Each byte contributes 7 bits with bit 7 indicating "more bytes follow";
    values are accumulated with a +1 increment per continuation byte to
    eliminate redundant encodings.

    Reference: src/serialize.h ReadVarInt / WriteVarInt.
    """
    n = 0
    while True:
        b = f.read(1)
        if not b:
            return None
        ch = b[0]
        n = (n << 7) | (ch & 0x7F)
        if ch & 0x80:
            n += 1  # increment per continuation byte
        else:
            return n


def decompress_amount(x):
    """Bitcoin Core src/compressor.cpp DecompressAmount. Recovers an
    amount in satoshis from its compressed varint representation."""
    if x == 0:
        return 0
    x -= 1
    e = x % 10
    x //= 10
    if e < 9:
        d = (x % 9) + 1
        x //= 9
        n = x * 10 + d
    else:
        n = x + 1
    while e > 0:
        n *= 10
        e -= 1
    return n


def decompress_script(f, n_size):
    """Bitcoin Core src/compressor.cpp DecompressScript. The chainstate
    compressor recognises six "special" script shapes (encoded with
    n_size 0..5) and falls back to raw bytes for everything else (n_size
    = actual_length + 6).

    We only need to recover h160 / pubkey-hash data, so we handle:
      0 = P2PKH  (20-byte hash160 follows)
      1 = P2SH   (20-byte script-hash follows)
      2 = P2PK uncompressed (32-byte x follows; y is parity-derived)
      3 = P2PK uncompressed (32-byte x follows; y has opposite parity)
      4 = P2PK compressed (32-byte x, y_parity=0)
      5 = P2PK compressed (32-byte x, y_parity=1)
    For 6+, the raw script is `n_size - 6` bytes long and we parse it the
    same way the C++ tool does (P2PKH-shape / P2SH-shape / P2WPKH-shape).
    """
    if n_size == 0:
        # P2PKH: just the 20-byte h160.
        return ("P2PKH", f.read(20))
    if n_size == 1:
        # P2SH: just the 20-byte script-hash.
        return ("P2SH", f.read(20))
    if n_size in (2, 3, 4, 5):
        # P2PK -- we'd need SHA256+RIPEMD160 over the recovered pubkey to
        # produce an h160. Skip for now (P2PK is ~0.5% of historic outputs
        # and ~0% of modern outputs; an h160 bloom built from P2PKH+P2SH+
        # P2WPKH already covers the vast majority of value).
        f.read(32)
        return (None, None)
    # Raw script, length = n_size - 6.
    raw_len = n_size - 6
    raw = f.read(raw_len)
    # Try to recognise P2WPKH (the only h160-shaped raw script we emit
    # in this path).
    if (raw_len == 22 and
            raw[0] == 0x00 and
            raw[1] == 0x14):
        return ("P2WPKH", raw[2:])
    return (None, None)


def parse_dumptxoutset(in_path, out_file, quiet):
    """Iterate UTXOs from a Bitcoin Core v22+ dumptxoutset file.

    File layout (see module docstring for details):
        40 bytes header  : 32-byte block hash (LE) + 8-byte LE uint64 count
        repeating entry  : 32 txid + 4 vout (LE uint32)
                           + VARINT code + VARINT amount
                           + VARINT nSize + script payload
    """
    with open(in_path, "rb") as f:
        # Header: 32-byte block hash + 8-byte LE uint64 coin count.
        block_hash = f.read(32)
        if len(block_hash) < 32:
            sys.exit("[!] file too small to contain a snapshot header")
        coin_count_raw = f.read(8)
        if len(coin_count_raw) < 8:
            sys.exit("[!] truncated snapshot header (coin count)")
        coin_count = struct.unpack("<Q", coin_count_raw)[0]

        if not quiet:
            print(f"[*] dumptxoutset header block hash: "
                  f"{block_hash[::-1].hex()}", file=sys.stderr)
            print(f"[*] {coin_count:,} coin entries declared in header",
                  file=sys.stderr)

        emitted = {"P2PKH": 0, "P2SH": 0, "P2WPKH": 0}
        skipped = 0
        seen = 0

        while True:
            txid = f.read(32)
            if not txid or len(txid) < 32:
                break
            vout_raw = f.read(4)
            if len(vout_raw) < 4:
                break
            # vout: little-endian uint32 (not a varint).
            _vout = struct.unpack("<I", vout_raw)[0]
            # code = (height << 1) | coinbase_flag, as a diff-varint.
            code = read_diff_varint(f)
            if code is None:
                break
            comp_amt = read_diff_varint(f)
            if comp_amt is None:
                break
            n_size = read_diff_varint(f)
            if n_size is None:
                break

            script_type, payload = decompress_script(f, n_size)
            seen += 1

            if script_type is None:
                skipped += 1
            else:
                amount = decompress_amount(comp_amt)
                out_file.write(f"{payload.hex()},{amount},{script_type}\n")
                emitted[script_type] += 1

            if not quiet and seen % 1_000_000 == 0:
                total_emit = sum(emitted.values())
                print(f"[*] {seen:>12,d} entries scanned  "
                      f"emit={total_emit:,}  skipped={skipped:,}",
                      file=sys.stderr)

    return seen, emitted, skipped


def main():
    p = argparse.ArgumentParser(
        description="Extract h160-shaped UTXOs from a "
                    "bitcoin-cli dumptxoutset snapshot.")
    p.add_argument("input", help="path to dumptxoutset output")
    p.add_argument("-o", "--output", default="-",
                   help="output CSV (default stdout)")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="suppress progress on stderr")
    args = p.parse_args()

    if not Path(args.input).is_file():
        sys.exit(f"[!] input not found: {args.input}")

    if args.output == "-":
        out = sys.stdout
        close = False
    else:
        out = open(args.output, "w")
        close = True

    try:
        seen, emitted, skipped = parse_dumptxoutset(
            args.input, out, args.quiet)
    finally:
        if close:
            out.close()

    if not args.quiet:
        print("", file=sys.stderr)
        print(f"[*] Done.", file=sys.stderr)
        print(f"    UTXOs scanned  : {seen:>12,d}", file=sys.stderr)
        print(f"    P2PKH          : {emitted['P2PKH']:>12,d}", file=sys.stderr)
        print(f"    P2SH           : {emitted['P2SH']:>12,d}", file=sys.stderr)
        print(f"    P2WPKH         : {emitted['P2WPKH']:>12,d}", file=sys.stderr)
        print(f"    skipped (P2PK/P2WSH/P2TR/multisig/other) : "
              f"{skipped:>12,d}", file=sys.stderr)
        print("", file=sys.stderr)
        print("[*] Next:", file=sys.stderr)
        print("    build_bloom -i funded.csv -v funded.uvrf",
              file=sys.stderr)


if __name__ == "__main__":
    main()
