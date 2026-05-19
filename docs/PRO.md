# theColliderPro

theCollider Free is one of the most complete open-source Bitcoin Puzzle Challenge solvers in the world. **theColliderPro turns the same binary into a treasure hunter.**

You are running a GPU. It is grinding millions, billions, eventually trillions of private keys. In Free, every one of those keys is checked against exactly one target: the puzzle you are solving. In Pro, **every distinguished-point pubkey is also handed to a CPU-side checker that probes it against a 100 million entry funded-address bloom filter**, alongside whatever puzzle work you were doing anyway.

If puzzle 135 is the moonshot, Pro adds a parallel scan of every recoverable wallet that human passphrase choice has ever leaked.

[Buy a Pro license](https://collisionprotocol.com/pro) | [Back to README](../README.md)

---

## The math nobody talks about

There are roughly **3 to 4 million Bitcoin permanently lost** to forgotten passwords, dead hard drives, garbage-truck'd laptops, and wallets whose owners died without leaving keys. That is somewhere between $200B and $400B at current prices, depending on the day.

Some of that is unrecoverable in principle (the keys are gone, full stop). But a measurable slice is recoverable in principle:

- **Brain wallets**: thousands of wallets where the private key is `SHA256(password)` of a passphrase a human picked. People used song lyrics, Bible verses, "correct horse battery staple", their dog's name. Most were drained by automated scanners within months. A long tail were not. **Some still hold real Bitcoin today.**
- **Lost-and-forgotten wallets**: addresses that received Bitcoin in 2011 and have never moved. The owner has the keys somewhere. Or used to.

Pro is built to find these.

---

## Opportunistic scanning while you solve puzzles

While your GPU is grinding puzzle 135 in pool or standalone mode, the distinguished points it produces are also handed to a CPU-side checker that derives a Bitcoin H160 from each one and probes the 100 million entry funded-address bloom filter. The GPU pipeline is the same one Free runs. The opportunistic check happens after the DP leaves the GPU, on the CPU; the only added work on the GPU is the small bookkeeping needed to keep the pubkey alongside the DP.

Here is the shape of the pipeline:

```
Free build, pool mode:
  GPU computes pubkey -> derives DP -> sends DP to pool server.
  (One target. Puzzle 135. That is the whole story.)

Pro build, pool mode:
  GPU computes pubkey -> derives DP -> sends DP to pool server.
  CPU also takes each DP's pubkey -> hashes to H160 -> probes the
    bloom for funded Bitcoin addresses.
  If a bloom hit shows up: candidate gets logged for verification.
```

Empirically, enabling the opportunistic CPU checker has a small effect on overall pool throughput (a few percent on a 4090, depending on how saturated the pool client is). The trade is a few percent of puzzle-135 odds for an entirely separate scan of every distinguished-point pubkey against the funded-address set.

Will you find an address? Almost certainly not. The chance on any given pubkey is roughly 1 in 2^229 even after collapsing the search to the funded subset. **The point is not "you will probably find one." The point is "the marginal cost of looking is small, so you might as well look."**

---

## Dedicated brain-wallet mode: your GPU is a key factory

When you are not pool-mining puzzle 135 (overnight, between work assignments, when the pool is quiet), Pro flips to **dedicated brain-wallet mode**: a purpose-built pipeline that systematically scans human-chosen passphrases against the same 100M+ funded-address bloom.

Bitcoin's earliest users protected coins with a passphrase that hashes directly to a private key. `SHA256("correct horse battery staple")` is a private key. No wallet file. No mnemonic. Just words. **Tens of thousands of these wallets were created between 2011 and 2014, and many of them still hold real Bitcoin today.**

The Pro brain-wallet pipeline runs a fully on-GPU fused kernel:

```
passphrase -> SHA-256 -> secp256k1 (private to public key) ->
RIPEMD-160(SHA-256(pubkey)) -> Bitcoin H160 ->
multi-address bloom probe -> hit?
```

Every stage stays on the GPU until a bloom hit needs definitive CPU verification.

### Measured throughput

The `--benchmark` flag drives the production fused kernel against a synthetic on-device passphrase buffer and reports per-stage and end-to-end rates. Sample measurement from `bench_gpu_pipeline --time 30 --batch 4000000` on an RTX 3060 (Ampere, sm_86):

```
Stage                          Rate              Window
-----                          ----              ------
SHA-256 (passphrase)             334.6 MH/s    30.01 s
secp256k1 mul                      1.33 MH/s   30.01 s
hash160 (SHA + RIPEMD)           318.7 MH/s    30.01 s
bloom probe                       11.92 GH/s   30.00 s
FUSED end-to-end                   7.48 MH/s   30.50 s
```

The end-to-end rate is the **production scan rate**. Per-stage rates are reported in isolation: SHA-256, hash160, and bloom probe all run at hundreds of millions to multiple billions of checks per second on this card. The secp256k1 scalar multiply standalone rate (~1 MH/s) is the per-stage cost of the EC kernel run in isolation; inside the fused kernel the inlined arithmetic and the SHA + RIPEMD + bloom probe pipelined across the same threads yields the ~7.5 MH/s end-to-end figure. The EC stage remains the dominant pipeline cost and is what the v1.5.0 crypto pipeline rewrite targets (32-bit limb PTX arithmetic, GLV decomposition).

Expected ranges by GPU generation: **tens to hundreds of millions of passphrase checks per second on Ampere through Blackwell GPUs under representative workloads**, scaling roughly with SM count and clock. Run the benchmark on your own card before planning a scan window. Numbers vary 2 to 3x across the supported architectures and a further 30 to 50% with driver version.

Reproduce on your hardware:

```bash
./collider --benchmark --benchmark-time 30
# OR for the standalone driver with stage-by-stage tables:
./bench_gpu_pipeline --time 30 --gpu 0
```

The reported rate **is** the rate the brain-wallet runner achieves in production. Do not extrapolate; run the benchmark.

### Generators that are smarter than dumb enumeration

You do not want to enumerate every 12-character string. There are too many. Pro ships three classes of generator that are smarter than brute force:

- **PCFG (Probabilistic Context-Free Grammar)**: trained on real password dumps. Generates `word + number + symbol` patterns in descending probability order. The high-probability cracks hit in seconds. The long tail descends predictably.
- **Markov chains**: character-level transition models trained on human-readable passphrases. Outputs that look like words, not random gibberish.
- **Hashcat-style rule engine**: take any wordlist (`rockyou.txt`, the Wikipedia corpus, your own custom list) and mutate it into **billions** of candidates via 200+ standard rules (case, leetspeak, append digits, prepend symbols, reverse, duplicate, capitalize-every-Nth, etc.).

`rockyou.txt` is 14 million words. `rockyou.txt × best64.rule` is roughly 9 billion candidates. At the measured rate on an RTX 3060 a full pass through `rockyou.txt × best64.rule` takes well over a day; on Ada/Blackwell-class silicon the rule-engine output rate grows in proportion to the per-card MH/s figure above. Measure your card, then plan your scan window.

---

## The secret weapon: the 100M+ funded-address bloom

The opportunistic check and the brain-wallet pipeline both sit on top of a single foundational dataset: **every Bitcoin address that has ever held a non-zero balance.**

Pro ships with `funded_addresses.blf`, a GPU-resident bloom filter built from:

- Every output address on every transaction on the Bitcoin blockchain.
- Three distinct H160 derivation paths per candidate pubkey: **compressed P2PKH** (which shares its H160 with **P2WPKH / BIP-84**), **uncompressed P2PKH**, and **P2SH-P2WPKH / BIP-49**. P2TR (BIP-86) requires a separate tweak-add EC computation and is in flight for a future release.
- Built fresh against recent UTXO snapshots; updates ship with each Pro release.

False positive rate at the shipped capacity: roughly **1 in 1 billion**. False negatives: zero by construction. Every bloom-positive hit triggers a definitive CPU-side check against the full address set, so spurious bloom positives never produce false alarms in your hits log.

The bloom is structured for **GPU-resident queries**: bit array on device memory, MurmurHash3-128 double-hashing per candidate, no CPU round-trip per check. Queries happen at memory-bandwidth speed inside the same kernel that computed the H160.

Build your own bloom from a custom address list with `build_bloom` (ships in `tools/`).

---

## What you get when you buy

| Capability                                            | Free       | Pro            |
| ----------------------------------------------------- | ---------- | -------------- |
| Solve puzzle 71 to 135 (kangaroo, brute force, pool)  | Yes        | Yes            |
| **Opportunistic bloom scan during pool / standalone** | **No**     | **Yes**        |
| **Dedicated brain-wallet mode (PCFG, Markov, rules)** | **No**     | **Yes**        |
| **100M+ funded-address bloom filter (shipped)**       | **No**     | **Yes**        |
| **Real-time hit logging**                             | **No**     | **Yes**        |
| Multi-address derivation (3 H160 paths per pubkey)    | Limited    | Yes            |
| Hashcat-style rule engine (GPU)                       | No         | Yes            |
| License                                               | MIT (Free) | Commercial Pro |

Pro is the same binary you already run. Features unlock when a valid HMAC-SHA256-signed license key is present. The license verifies **offline** (the shared key and license slot are embedded in the binary at build time). Once activated, your interactive UI gains the brain-wallet mode; pool and puzzle modes silently enable the opportunistic bloom scan; everything else behaves identically to Free.

---

## How to use it once you have a license

### Pool mode with opportunistic scanning (the default Pro experience)

```bash
./collider_pro --pool jlps://collisionprotocol.com:17403 \
               --worker bc1qYourBitcoinAddress \
               --bloom funded_addresses.blf
```

Once the pool client connects, the banner reads:

```
[*] Bloom filter loaded - opportunistic address checking enabled
```

From that moment, every DP your GPU produces also gets its pubkey turned into H160 and probed. Hits log to your configured hits file. You can tail it in another terminal and walk away.

### Dedicated brain-wallet mode

Brain-wallet mode is configured through the **setup wizard** plus `config.yml`, not via CLI flags. The setup wizard walks you through choosing wordlists, dedup behavior, PCFG training, and where to write hits.

```bash
# First run: configure wordlists, dedup, PCFG, hit log path
./collider_pro --brainwallet-setup

# Subsequent runs: scan
./collider_pro --brainwallet --bloom funded_addresses.blf
```

The setup wizard writes its decisions into `config.yml` (see [CONFIGURATION.md](CONFIGURATION.md) for the schema). Wordlists, rule files, generators, dedup, and hit log all live in that file. You can also enable v1.4.2's multi-address derivation explicitly:

```bash
./collider_pro --brainwallet-v2 --bloom funded_addresses.blf
```

`--brainwallet-v2` probes every pubkey against all three H160 paths (compressed P2PKH, uncompressed P2PKH, P2SH-P2WPKH/BIP-49) inside the fused kernel.

---

## Pricing

Pro licenses are tier-priced by use case. See [collisionprotocol.com/pro](https://collisionprotocol.com/pro) for current pricing, terms, and license keys.

Every license is delivered as a key string. On first activation the binary POSTs the key once over TLS to the issuer's license endpoint, receives a signed validation record, and caches it locally at `~/.collider/license.cache` (HMAC-SHA256 over the cache contents against an embedded shared key). Subsequent runs verify the cache HMAC offline; the cache is refreshed against the issuer endpoint after a 24-hour TTL. There is no per-run server check inside that 24-hour window, no usage telemetry, and no per-DP or per-solve reporting. If you need a fully air-gapped deployment, contact the issuer for an offline-validation license (separate product).

---

## FAQ

**Q: Is the opportunistic scan really free?**

A: It is cheap, not zero. The GPU still does the same kernel pass it would do in Free. The added cost is on the CPU: each distinguished point's pubkey gets one extra SHA-256 + RIPEMD-160 + bloom probe (and, with `--brainwallet-v2` enabled, two extra per-pubkey hashes for the uncompressed P2PKH and P2SH-P2WPKH paths). Empirically that costs a few percent of overall pool throughput on a 4090. You give up a few percent of your puzzle-135 odds in exchange for a separate scan against 100 million funded addresses on every DP your GPU produces. The math is clearly favorable.

**Q: What is the probability of an opportunistic hit?**

A: Astronomically low on any given key. Bitcoin's keyspace is 2^256; the funded-address set is roughly 2^27. The probability of any random key happening to produce an H160 in the funded set is roughly 1 in 2^229. But you are computing trillions of keys per day per machine. If you mine for a year, you compute somewhere around 2^57 to 2^60 keys (rough estimate; depends on hardware and uptime). The expected number of hits per year on a single machine is still vanishingly small. **The point is not "you will probably find one." The point is "the marginal cost of looking is small, so you might as well look."**

**Q: What happens if I do find a hit?**

A: Pro writes the candidate to your configured hits log with the H160, the candidate private key (in hex), and the timestamp. **Pro never spends the funds automatically.** You verify the hit manually (check the address on a block explorer; sweep the funds with a wallet you control if it is non-zero). Spending found funds is a separate workflow Pro does not touch; that is on purpose, because hits can be your own test addresses, dust, or honeypots.

**Q: Is this legal?**

A: In every jurisdiction we know of: yes, with the obvious caveats. Computing keys and checking against a public address list is not theft. Spending funds from an address whose key you happened to compute is jurisdiction-dependent and morally complicated; the funds belong to whoever originally controlled them, even if the bookkeeping is on a public ledger. Pro is a tool. You decide what to do with what it finds. **Consult a lawyer in your jurisdiction before sweeping any non-trivial balance from a found wallet.** This documentation is not legal advice.

**Q: Why not just open-source the bloom and the brain-wallet pipeline?**

A: Two reasons. First, the 100M+ funded-address bloom is built from datasets that take operational work to maintain (recent UTXO snapshots, address-type expansion, occasional manual filtering). Pro pays for that maintenance. Second, the PCFG and Markov generators are tuned against real password-cracking datasets that came from operational pentesting work. Releasing them under MIT would dilute the value of that tuning for the people who fund the project. The free build is intentionally a complete puzzle solver; Pro is intentionally a complete treasure hunter.

**Q: What if a bug in Pro burns through my electricity for nothing?**

A: Pro ships the same telemetry as Free (DP rate, accepted-by-pool counter, bloom hits if any). If your DP rate drops or your accrual on the pool stalls, you will see it in the same place you see it in Free. Open an issue on the [private Pro tracker](https://github.com/hevnsnt/collider-pro) (access included with your license) and we will triage. The free build's CI runs the same kernels; we have not shipped a Pro build that produces correct opportunistic scans but incorrect kangaroo DPs, and the design makes that hard to do accidentally.

---

## Buying a license

Head to [collisionprotocol.com/pro](https://collisionprotocol.com/pro).

Pick a tier. Pay. Receive your HMAC-SHA256-signed license file by email. Drop it next to your `collider_pro` binary. The next run unlocks Pro features and prints your license fingerprint at startup.

You bought the hardware. You pay the power bill. **Pro makes sure every key you compute is doing as much work for you as it can.**

---

## Where to go next

- [Back to README](../README.md) — overview and quick-start.
- [POOL.md](POOL.md) — pool mode, share-of-pool accrual, picking a worker address.
- [CONFIGURATION.md](CONFIGURATION.md) — `config.yml` schema, including the `bloom:` and brain-wallet blocks.
- [ARCHITECTURE.md](ARCHITECTURE.md) — source-tree map; the Pro brain-wallet pipeline lives in `src/gpu/fused_pipeline.cu` and `src/gpu/mega_fused_kernel.cu` (rule-engine fast path).
- [collisionprotocol.com/pro](https://collisionprotocol.com/pro) — buy a license.
