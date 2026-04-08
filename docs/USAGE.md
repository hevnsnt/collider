# Usage Guide

This guide covers all thePuzzler commands, modes, and workflows for Bitcoin puzzle solving and brain wallet security research.

---

## Table of Contents

- [Quick Reference](#quick-reference)
- [Operating Modes](#operating-modes)
- [Bitcoin Puzzle Mode](#bitcoin-puzzle-mode)
- [Kangaroo Mode](#kangaroo-mode)
- [Pool Mode (Distributed Solving)](#pool-mode-distributed-solving)
- [Brain Wallet Mode](#brain-wallet-mode)
- [Bloom Filter Management](#bloom-filter-management)
- [Wordlist Preparation](#wordlist-preparation)
- [Multi-GPU Configuration](#multi-gpu-configuration)
- [Checkpointing and Resume](#checkpointing-and-resume)
- [Output and Logging](#output-and-logging)
- [Advanced Workflows](#advanced-workflows)

---

## Quick Reference

### Most Common Commands

```bash
# Solve Bitcoin puzzle with auto-selected method
./thepuzzler --puzzle 135

# Kangaroo solve with specific public key
./thepuzzler --kangaroo --pubkey 02abc... --range 135 --start 0x400...

# Join a pool for distributed solving (recommended for hard puzzles)
./thepuzzler --pool jlp://pool.example.com:17403 --worker 1YourBTCAddress

# Brain wallet scan with wordlist
./thepuzzler --bloom addresses.blf --wordlist rockyou.txt

# Benchmark all modes
./thepuzzler --benchmark --all

# List available GPUs
./thepuzzler --list-gpus
```

### Command Structure

```
./thepuzzler [MODE] [OPTIONS]
```

| Mode Flag | Description |
|-----------|-------------|
| `--puzzle N` | Bitcoin Puzzle Challenge mode |
| `--kangaroo` | Pollard's Kangaroo ECDLP solver |
| `--bloom FILE` | Brain wallet scanning mode |
| `--benchmark` | Performance benchmarking |

---

## Operating Modes

thePuzzler operates in one of four primary modes:

### 1. Puzzle Mode
Targets a specific Bitcoin Puzzle Challenge address. Automatically selects optimal algorithm based on whether the public key is known.

### 2. Kangaroo Mode
Uses Pollard's Kangaroo algorithm to solve ECDLP when the public key is known. Reduces complexity from O(n) to O(sqrt(n)).

### 3. Brain Wallet Mode
Scans passphrases against a bloom filter of funded Bitcoin addresses. Processes billions of keys per second.

### 4. Benchmark Mode
Measures performance across all subsystems. Use to validate your hardware configuration.

---

## Bitcoin Puzzle Mode

### Basic Usage

```bash
# Target specific puzzle (auto-selects algorithm)
./thepuzzler --puzzle 135

# Target puzzle with explicit algorithm
./thepuzzler --puzzle 71 --brute-force
./thepuzzler --puzzle 135 --kangaroo
```

### Puzzle Selection

thePuzzler automatically analyzes puzzles and recommends the best approach:

| Puzzle Range | Public Key Known | Recommended Method |
|--------------|------------------|-------------------|
| #71-74 | No | Brute Force (center-heavy) |
| #76-79 | No | Brute Force |
| #81-84 | No | Brute Force |
| #131-134 | No | Brute Force |
| #135, #140, #145... | Yes | Kangaroo |

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--puzzle N` | Target puzzle number | Required |
| `--puzzle-target ADDR` | Override target address | Auto |
| `--puzzle-start HEX` | Override range start | Auto |
| `--puzzle-end HEX` | Override range end | Auto |
| `--puzzle-sequential` | Use sequential scan | Random |
| `--puzzle-checkpoint FILE` | Checkpoint file | None |
| `--all-unsolved` | Progress through all unsolved | Off |

### Center-Heavy Scanning

For brute-force puzzles, thePuzzler implements center-heavy zone scanning based on historical key distribution:

```bash
# Use default center-heavy strategy
./thepuzzler --puzzle 71

# Customize zone priorities
./thepuzzler --puzzle 71 --zone-priority "0.6-0.85,0.3-0.5,0.0-0.3,0.85-1.0"
```

Zone analysis of solved puzzles:

| Zone | Range | Historical Hit Rate |
|------|-------|---------------------|
| Center-High | 60-85% | 38% |
| Center-Low | 30-50% | 32% |
| Upper Edge | 85-100% | 17% |
| Lower Edge | 0-30% | 13% |

### Example: Solving Puzzle #71

```bash
# Start with center-heavy optimization
./thepuzzler --puzzle 71 \
  --puzzle-checkpoint puzzle71.ckpt \
  --output puzzle71_found.txt \
  --verbose

# Resume after interruption
./thepuzzler --puzzle 71 \
  --puzzle-checkpoint puzzle71.ckpt \
  --resume
```

---

## Kangaroo Mode

### When to Use Kangaroo

Kangaroo mode is applicable when:
1. The target's **public key is known** (exposed via outgoing transaction)
2. The private key range is bounded

For puzzles without known public keys, brute force is required.

### Basic Usage

```bash
# Solve with public key
./thepuzzler --kangaroo \
  --pubkey 02abc123def456... \
  --range 135 \
  --start 0x4000000000000000000000000000000000

# Benchmark mode (random keys)
./thepuzzler --kangaroo --benchmark
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--kangaroo` | Enable Kangaroo mode | - |
| `--pubkey HEX` | Target compressed public key | Required |
| `--range N` | Key range in bits (32-170) | Required |
| `--start HEX` | Range start offset | Required |
| `--dp-bits N` | Distinguished point bits (14-60) | Auto |
| `--tames FILE` | Load/save tame kangaroos | None |
| `--max N` | Max operations multiplier | Unlimited |

### Distinguished Point Tuning

The `--dp-bits` parameter controls the tradeoff between memory and collision detection:

| DP Bits | Steps Between DPs | Memory Usage | Recommended For |
|---------|-------------------|--------------|-----------------|
| 14-16 | 16K-64K | Very High | Quick testing |
| 18-20 | 256K-1M | High | Short ranges (<80 bits) |
| 20-24 | 1M-16M | Medium | General use |
| 24-28 | 16M-256M | Low | Large ranges (>100 bits) |
| 28+ | 256M+ | Very Low | Massive parallelism |

```bash
# Smaller DP bits for faster collision detection
./thepuzzler --kangaroo --pubkey 02abc... --range 80 --dp-bits 16

# Larger DP bits for large-scale solving
./thepuzzler --kangaroo --pubkey 02abc... --range 135 --dp-bits 24
```

### Precomputed Tames

For repeated solving in the same range, precompute tame kangaroos:

```bash
# Generate tames (takes time but accelerates future solves)
./thepuzzler --kangaroo --range 76 --tames tames76.dat --max 0.5

# Use precomputed tames
./thepuzzler --kangaroo --pubkey 02abc... --range 76 --tames tames76.dat
```

### Opportunistic Bloom Filter Checking

Enable bloom filter checking during Kangaroo solving to discover funded wallets:

```bash
./thepuzzler --kangaroo \
  --pubkey 02abc... \
  --range 135 \
  --bloom addresses.blf \
  --bloom-check-dps
```

This checks each Distinguished Point against the bloom filter with less than 1% overhead.

### Example Output

```
[*] Using Pollard's Kangaroo Algorithm (O(sqrt(n)))
    Search complexity reduced from 2^134 to ~2^67

[*] GPU 0: NVIDIA GeForce RTX 4090 (128 SMs, 24576 MB)
[*] GPU 1: NVIDIA GeForce RTX 4090 (128 SMs, 24576 MB)
[*] Multi-GPU Kangaroo initialized with 2 GPU(s)

[*] Using dp_bits=22 (auto-calculated)
    Expected steps between DPs: ~4,194,304

[*] Starting GPU Kangaroo search...
    Press Ctrl+C to stop

[*] Progress: 1,523,417,088 ops | 8.12 GKeys/s | 14,523 DPs | K=1.15
[*] Progress: 3,046,834,176 ops | 8.09 GKeys/s | 29,047 DPs | K=1.15
...
[!] SOLUTION FOUND!
    Private Key: 0x4a5b6c7d8e9f...
    Verification: SUCCESS
    Total Operations: 147,573,952,589
    Elapsed Time: 5h 23m 17s
    Achieved K: 1.14
```

### Pool Mode (Distributed Solving)

For puzzles like #135 that would take ~195 years solo, pool mode enables collaborative solving:

#### Why Use a Pool?

| Configuration | Puzzle #135 Time |
|---------------|------------------|
| 4x RTX 4090 (solo) | ~195 years |
| 100x RTX 5090 (solo) | ~6 years |
| Pool with 10,000 GPUs | months |

Pools aggregate Distinguished Points from thousands of workers. When a DP from a tame kangaroo collides with a DP from a wild kangaroo, the private key is found.

#### Basic Usage

```bash
# Connect to a JLP-compatible pool
./thepuzzler --pool jlp://pool.example.com:17403 \
             --worker 1YourBitcoinAddressForRewards

# Connect to an HTTP-based pool
./thepuzzler --pool http://api.puzzlepool.io \
             --worker 1YourBitcoinAddress \
             --pool-api-key YOUR_API_KEY
```

#### Pool Options

| Option | Description |
|--------|-------------|
| `--pool, -p <url>` | Pool URL (jlp://host:port or http://host:port) |
| `--worker, -w <addr>` | Your Bitcoin address for reward distribution |
| `--pool-password` | Pool password if required |
| `--pool-api-key` | API key for HTTP pools |

#### How It Works

1. **Connect**: thePuzzler connects to the pool server
2. **Get Work**: Pool assigns target public key and parameters
3. **Solve**: RCKangaroo runs, submitting DPs to pool
4. **Reward**: If your DP causes a collision, you receive your share

#### Example Output

```
=============================================================
             POOL MODE - Distributed Kangaroo Solving
=============================================================

[*] Pool Configuration:
    Type:   jlp
    Host:   pool.example.com:17403
    Worker: 1ABC...xyz
    GPUs:   2 detected

[*] Connecting to pool...
[+] Connected to pool successfully!

[*] Requesting work from pool...
[+] Work assigned: Puzzle #135
    DP Bits: 24
    Work ID: 12345

[*] Starting pool solving...
    Press Ctrl+C to stop

[Pool] Speed: 8.12 MKeys/s | DPs Found: 1,234 | Submitted: 1,234 | Rate: 42.1 DP/s
```

#### Finding a Pool

Several community pools exist for Bitcoin puzzle solving. Search for:
- "Bitcoin puzzle Kangaroo pool"
- "BTC puzzle collaborative solving"
- Bitcointalk forums for active pool discussions

**Note**: thePuzzler implements the JLP (JeanLucPons) Kangaroo protocol which is widely used. Most puzzle pools are compatible.

---

## Brain Wallet Mode

### Concept

Brain wallets derive private keys from passphrases:

```
passphrase -> SHA256 -> private_key -> EC_multiply -> public_key -> address
```

thePuzzler tests billions of passphrases per second against funded addresses.

### First-Run Setup Wizard

On your first brainwallet scan, thePuzzler runs an interactive setup wizard to configure your wordlists:

```bash
# Run setup wizard directly
./thepuzzler --brainwallet-setup

# Or it runs automatically on first brainwallet use
./thepuzzler --brainwallet --bloom addresses.blf
```

**The wizard walks you through:**

1. **Wordlist Location** - Specify directories containing your wordlists
2. **Scanning** - Finds all `.txt`, `.lst`, `.dic`, `.wordlist` files
3. **Processing** - Combines, deduplicates, and normalizes all wordlists
4. **PCFG Training** - Optionally train a probability model

**What Processing Does:**

| Step | Description |
|------|-------------|
| Combine | Merges all wordlists into single file |
| Deduplicate | Removes exact duplicates (case-sensitive) |
| Normalize | Trims whitespace, removes control characters |
| Filter | Skips empty lines and entries > 256 chars |

**Configuration Persistence:**

After setup, configuration is saved to `~/.thepuzzler/`:

```
~/.thepuzzler/
├── brainwallet_config.txt    # Settings and paths
└── processed/
    └── combined_wordlist.txt # Deduplicated wordlist
```

On subsequent runs, thePuzzler loads your saved configuration automatically.

**Reconfigure:**

```bash
# Re-run setup wizard
./thepuzzler --brainwallet-setup

# Or when prompted in interactive mode, answer 'Y' to "Reconfigure wordlists?"
```

### Basic Usage

```bash
# Single wordlist
./thepuzzler --bloom addresses.blf --wordlist rockyou.txt

# Directory of wordlists
./thepuzzler --bloom addresses.blf --wordlist-dir processed/

# Recursive directory scan
./thepuzzler --bloom addresses.blf --wordlist-dir data/ --recursive

# Stdin input
cat passphrases.txt | ./thepuzzler --bloom addresses.blf --wordlist -
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--bloom FILE` | Bloom filter file (.blf) | Required |
| `--wordlist FILE` | Wordlist file (use `-` for stdin) | - |
| `--wordlist-dir DIR` | Directory of wordlists | - |
| `--recursive` | Recursively scan directories | Off |
| `--rules FILE` | Hashcat-compatible rule file | None |
| `--pcfg FILE` | PCFG grammar file | None |
| `--output FILE` | Output file for hits | found.txt |
| `--gpus LIST` | GPU device IDs | All |
| `--batch-size N` | Candidates per batch | Auto |

### Passphrase Generation Modes

**1. Direct Wordlist:**
```bash
./thepuzzler --bloom addresses.blf --wordlist passwords.txt
```

**2. With Mutation Rules:**
```bash
./thepuzzler --bloom addresses.blf \
  --wordlist rockyou.txt \
  --rules rules/best64.rule
```

**3. PCFG Generation:**
```bash
# Train PCFG on known passwords
./thepuzzler --train known_passwords.txt --train-output brain_wallet.pcfg

# Generate and test
./thepuzzler --bloom addresses.blf --pcfg brain_wallet.pcfg
```

#### Why PCFG for Brain Wallets?

**PCFG (Probabilistic Context-Free Grammar)** learns password structure patterns from real data and generates candidates in probability order—most likely passwords first.

**Key Benefits:**

| Benefit | Description |
|---------|-------------|
| **Probability-Ordered** | Tests `password123` before `xq7$mZpK` |
| **Learns Real Patterns** | Understands `[Word][Digits][Symbol]` is common |
| **Generative** | Creates passwords not in any wordlist |
| **Human-Predictable** | Exploits how humans choose passphrases |

**Patterns PCFG Learns:**

| Pattern | Example | Frequency |
|---------|---------|-----------|
| `Word + Digits` | bitcoin123 | Very high |
| `Name + Year + Symbol` | Satoshi2009! | High |
| `l33tspeak` | p@ssw0rd | Medium |
| `Phrase` | iloveyou | High |

**Efficiency Comparison:**

| Method | Coverage | Order | Speed to Success |
|--------|----------|-------|------------------|
| Brute force | 100% | Sequential | Slowest |
| Wordlist | Limited | Alphabetical | Fast for exact matches |
| **PCFG** | Generative | Probability | Fastest for human-chosen |

```
Brute force 8-char:  95^8 = 6.6 quadrillion candidates
Wordlist:            10 million candidates (limited coverage)
PCFG:                Infinite candidates, but "bitcoin123" tested in first 1000
```

For brain wallets (human-memorable phrases), PCFG dramatically outperforms brute force because humans are predictable.

**4. Combination Mode:**
```bash
# Multiple wordlists combined
./thepuzzler --bloom addresses.blf \
  --wordlist words1.txt \
  --wordlist words2.txt \
  --combination
```

### Rule Files

thePuzzler supports Hashcat-compatible rules. Example `rules/crypto.rule`:

```
# Append years
$2$0$2$4
$2$0$2$3
$2$0$2$5

# Common substitutions
sa@
se3
so0
si1
sl1

# Bitcoin-specific suffixes
$b$t$c
$B$T$C
$s$a$t$s
$s$a$t$o$s$h$i
```

### Example Output

```
[*] Loading bloom filter: addresses.blf
    Size: 5.7 GB
    Bits: 47,999,999,960
    Hashes: 17
    Expected FP Rate: 0.001%

[*] Loading wordlist: rockyou.txt
    Lines: 14,344,391

[*] Initializing GPU pipeline...
    GPU 0: RTX 4090 (24576 MB)

[*] Processing...
    Rate: 1,823,456,789 keys/sec
    Progress: 14,344,391 / 14,344,391 (100%)

*** HIT FOUND ***
Address: 1Abc123...
Passphrase: correct horse battery staple
Private Key: c4bbcb1f...
*****************

[*] Complete.
    Total Processed: 14,344,391
    Matches Found: 1
    Elapsed: 0.008s
```

---

## Bloom Filter Management

### Building Bloom Filters

**Step 1: Extract UTXO Data**

Use bitcoin-utxo-dump to extract addresses from Bitcoin Core:

```bash
# Install utxo-dump
go install github.com/in3rsha/bitcoin-utxo-dump@latest

# Extract from chainstate
bitcoin-utxo-dump \
  -db ~/.bitcoin/chainstate \
  -o utxo.csv \
  -f csv
```

**Step 2: Build Bloom Filter**

```bash
./build_bloom \
  -i utxo.csv \
  -o addresses.blf \
  -m 100000 \
  -f 0.00001 \
  -e 50000000
```

### Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `-i, --input` | Input CSV file | Required |
| `-o, --output` | Output .blf file | Required |
| `-m, --min-sats` | Minimum satoshis | 100000 |
| `-f, --fp-rate` | False positive rate | 0.00001 |
| `-e, --expected` | Expected elements | 50000000 |
| `-v, --verify` | Verification set output | None |
| `-s, --stats` | Show statistics | Off |

### Example

```bash
./build_bloom -i utxo.csv -o addresses.blf -m 100000 -s

# Output:
UTXO Bloom Filter Builder
=========================

Configuration:
  Input:            utxo.csv
  Output:           addresses.blf
  Min Satoshis:     100000 (0.001 BTC)
  Target FP Rate:   0.001%
  Expected:         50,000,000

Processing...
  Addresses found:  48,234,521
  Below threshold:  112,456,789

Filter Parameters:
  Bits:             47,999,999,960 (5.7 GB)
  Hash Functions:   17
  Fill Ratio:       48.2%
  Actual FP Rate:   0.00098%

Saved to: addresses.blf
```

---

## Wordlist Preparation

### Preprocessing Raw Data

thePuzzler includes a preprocessing script for cleaning wordlists:

```bash
python3 scripts/preprocess_data.py \
  --data-dir data \
  --output-dir processed \
  --verbose
```

### Input Directory Structure

```
data/
  passwords/       # Password corpuses (rockyou, darkc0de, etc.)
  lyrics/          # Song lyrics, music data
  quotes/          # Famous quotes, phrases
  literature/      # Books, poetry
  names/           # Usernames, celebrities
  wordlists/       # Dictionary words
  crypto/          # BIP39, crypto terms
  rules/           # Hashcat rules
```

### Output Structure

```
processed/
  passwords.txt    # 15M deduplicated passwords
  phrases.txt      # 1.1M phrases (lyrics, quotes)
  wordlists.txt    # 7.4M dictionary words
  names.txt        # 8.2M names
  crypto.txt       # 19K crypto-specific terms
  rules/           # Copied rule files
```

### What the Preprocessor Does

- Extracts text from JSON structures
- Parses CSV columns
- Strips headers/footers from literature
- Removes comments
- Normalizes Unicode (NFC)
- Deduplicates (case-insensitive)
- Removes empty lines and control characters

---

## Multi-GPU Configuration

### Automatic Detection

By default, thePuzzler uses all available GPUs:

```bash
./thepuzzler --list-gpus

# Output:
Detected GPUs:
  [0] NVIDIA GeForce RTX 4090 (24576 MB, SM 8.9, 128 SMs)
  [1] NVIDIA GeForce RTX 3090 (24576 MB, SM 8.6, 82 SMs)
  [2] NVIDIA GeForce RTX 3060 (12288 MB, SM 8.6, 28 SMs)
```

### Selecting Specific GPUs

```bash
# Use only GPU 0 and 1
./thepuzzler --bloom addresses.blf --wordlist rockyou.txt --gpus 0,1

# Use GPU 2 only
./thepuzzler --puzzle 135 --kangaroo --gpus 2
```

### Heterogeneous GPU Support

thePuzzler automatically balances work across GPUs with different capabilities:

```bash
# Mix RTX 4090 with RTX 3060 (work balanced by performance)
./thepuzzler --bloom addresses.blf --wordlist-dir processed/ --gpus 0,1,2
```

### Per-GPU Statistics

Use verbose mode to see per-GPU performance:

```bash
./thepuzzler --bloom addresses.blf --wordlist rockyou.txt --verbose

# Output includes:
[*] GPU 0: 1,234,567,890 keys/sec (67.7%)
[*] GPU 1:   589,123,456 keys/sec (32.3%)
[*] Total:  1,823,691,346 keys/sec
```

---

## Checkpointing and Resume

### Enabling Checkpoints

```bash
# Save progress every 100M keys
./thepuzzler --puzzle 71 \
  --puzzle-checkpoint puzzle71.ckpt \
  --checkpoint-interval 100000000
```

### Resuming from Checkpoint

```bash
./thepuzzler --puzzle 71 \
  --puzzle-checkpoint puzzle71.ckpt \
  --resume
```

### Checkpoint Contents

Checkpoints store:
- Current position in search space
- Total keys processed
- Timestamp
- Configuration hash (validates compatible resume)

### Kangaroo Checkpoints

For Kangaroo mode, save Distinguished Points:

```bash
./thepuzzler --kangaroo \
  --pubkey 02abc... \
  --range 135 \
  --dp-file puzzle135_dps.bin \
  --checkpoint puzzle135.ckpt
```

---

## Output and Logging

### Output Files

| File | Contents |
|------|----------|
| `found.txt` | Discovered private keys (default) |
| `hits.log` | Detailed hit information |
| `progress.log` | Session progress (if `--log` enabled) |

### Found.txt Format

```
# Private Key : Passphrase
c4bbcb1fbec99d65bf59d85c8cb62ee2db963f0fe106f483d9afa73bd4e39a8a:correct horse battery staple
```

### Hits.log Format

```
=== HIT FOUND ===
Time: 2025-01-04 12:34:56
Address: 1Abc123...
H160: 62e907b15cbf27d5425399ebf6f0fb50ebb88f18
Passphrase: correct horse battery staple
Private Key: c4bbcb1fbec99d65...
Balance: 6859000000 satoshis (68.59 BTC)
==================
```

### Verbosity Levels

```bash
# Normal output
./thepuzzler --bloom addresses.blf --wordlist rockyou.txt

# Verbose (per-GPU stats, detailed progress)
./thepuzzler --bloom addresses.blf --wordlist rockyou.txt --verbose

# Debug (internal timings, memory usage)
./thepuzzler --bloom addresses.blf --wordlist rockyou.txt --debug
```

---

## Advanced Workflows

### Workflow 1: Comprehensive Brain Wallet Audit

```bash
# Step 1: Build bloom filter from UTXO
./build_bloom -i utxo.csv -o addresses.blf -m 10000 -s

# Step 2: Preprocess wordlists
python3 scripts/preprocess_data.py --data-dir data --output-dir processed

# Step 3: Run with all wordlists and rules
./thepuzzler --bloom addresses.blf \
  --wordlist-dir processed/ \
  --rules rules/best64.rule \
  --output audit_results.txt \
  --verbose

# Step 4: Verify any hits
./thepuzzler --verify audit_results.txt
```

### Workflow 2: Puzzle #135 Campaign

```bash
# Step 1: Verify public key
./thepuzzler --puzzle 135 --info

# Step 2: Start Kangaroo solve with checkpointing
./thepuzzler --puzzle 135 \
  --kangaroo \
  --dp-bits 22 \
  --dp-file puzzle135_dps.bin \
  --checkpoint puzzle135.ckpt \
  --bloom addresses.blf \
  --bloom-check-dps

# Step 3: Monitor progress
tail -f progress.log

# Step 4: Resume after interruption
./thepuzzler --puzzle 135 \
  --kangaroo \
  --dp-file puzzle135_dps.bin \
  --checkpoint puzzle135.ckpt \
  --resume
```

### Workflow 3: Multi-Machine Distributed Solving

For large-scale solves across multiple machines:

**Machine 1 (Coordinator):**
```bash
./thepuzzler --kangaroo \
  --pubkey 02abc... \
  --range 135 \
  --distributed-server \
  --port 17403
```

**Machine 2-N (Workers):**
```bash
./thepuzzler --kangaroo \
  --distributed-worker \
  --server 192.168.1.100:17403
```

### Workflow 4: Continuous Background Scanning

For long-running background operation:

```bash
# Run in screen/tmux session
screen -S puzzler

./thepuzzler --puzzle 71 \
  --puzzle-checkpoint puzzle71.ckpt \
  --checkpoint-interval 10000000000 \
  --log progress.log \
  --output found.txt

# Detach: Ctrl+A, D
# Reattach: screen -r puzzler
```

---

## Performance Tuning

### Batch Size Optimization

```bash
# Larger batches for high-VRAM GPUs
./thepuzzler --bloom addresses.blf --wordlist rockyou.txt --batch-size 8000000

# Smaller batches for memory-constrained GPUs
./thepuzzler --bloom addresses.blf --wordlist rockyou.txt --batch-size 1000000
```

### Memory Management

```bash
# Monitor VRAM usage
watch nvidia-smi

# Reduce VRAM usage if needed
./thepuzzler --bloom addresses.blf --wordlist rockyou.txt \
  --batch-size 500000 \
  --no-double-buffer
```

### CPU Affinity (Linux)

```bash
# Pin to specific CPU cores
taskset -c 0-7 ./thepuzzler --bloom addresses.blf --wordlist rockyou.txt

# NUMA-aware execution
numactl --cpunodebind=0 --membind=0 ./thepuzzler ...
```

---

## Security Considerations

### Protecting Found Keys

When a key is found:

1. **Do not broadcast immediately** to public mempool
2. Contact a mining pool for private transaction inclusion
3. Use high transaction fees to ensure fast confirmation
4. Transfer to a new secure wallet immediately

### Mempool Sniping Protection

Attackers monitor the mempool for puzzle solution transactions and attempt to front-run with higher fees. See [BITCOIN-PUZZLE-STRATEGY.md](BITCOIN-PUZZLE-STRATEGY.md) for protection strategies.

### Secure Operation

- Run on isolated systems
- Do not share checkpoint files
- Clear found.txt after transferring funds
- Use encrypted storage for wordlists

---

## Command Reference

### Global Options

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version` | Show version |
| `--list-gpus` | List available GPUs |
| `--gpus LIST` | GPU device IDs (comma-separated) |
| `--verbose` | Verbose output |
| `--debug` | Debug output |
| `--output FILE` | Output file for hits |
| `--log FILE` | Log file for progress |

### Puzzle Mode Options

| Option | Description |
|--------|-------------|
| `--puzzle N` | Target puzzle number |
| `--puzzle-target ADDR` | Override target address |
| `--puzzle-start HEX` | Override range start |
| `--puzzle-end HEX` | Override range end |
| `--puzzle-sequential` | Sequential scan |
| `--puzzle-checkpoint FILE` | Checkpoint file |
| `--all-unsolved` | Auto-progress through puzzles |

### Kangaroo Mode Options

| Option | Description |
|--------|-------------|
| `--kangaroo` | Enable Kangaroo mode |
| `--pubkey HEX` | Target public key |
| `--range N` | Range in bits |
| `--start HEX` | Range start |
| `--dp-bits N` | Distinguished point bits |
| `--tames FILE` | Tames file |
| `--max N` | Max operations multiplier |
| `--dp-file FILE` | DP storage file |

### Brain Wallet Mode Options

| Option | Description |
|--------|-------------|
| `--brainwallet` | Enable brainwallet mode |
| `--brainwallet-setup` | Run setup wizard (configure wordlists, dedup, PCFG) |
| `--bloom FILE` | Bloom filter file |
| `--wordlist FILE` | Wordlist file |
| `--wordlist-dir DIR` | Wordlist directory |
| `--recursive` | Recursive directory scan |
| `--rules FILE` | Rule file |
| `--pcfg FILE` | PCFG grammar |
| `--batch-size N` | Batch size |
| `--train FILE` | Train PCFG |
| `--train-output FILE` | PCFG output |

---

*Ready to solve puzzles? Pick your target and start hunting.*
