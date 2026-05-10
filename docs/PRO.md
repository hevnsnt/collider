# theColliderPro

> Every key your GPU computes is a lottery ticket. Pro just gives you more lotteries to win.

theCollider Free is one of the most complete open-source Bitcoin Puzzle Challenge solvers in the world. **theColliderPro turns the same binary into a treasure hunter.**

You are running a GPU. It is grinding millions, billions, eventually trillions of private keys. In Free, every one of those keys is checked against exactly one target: the puzzle you are solving. In Pro, **every single key is also checked against 100 million funded Bitcoin addresses, automatically, in the same GPU kernel pass, at zero marginal cost.**

If puzzle 135 is the moonshot, Pro is the lottery ticket that comes free with the moonshot.

[Buy a Pro license](https://collisionprotocol.com/pro) | [Back to README](../README.md)

---

## The math nobody talks about

There are roughly **3 to 4 million Bitcoin permanently lost** to forgotten passwords, dead hard drives, garbage-truck'd laptops, and wallets whose owners died without leaving keys. That is somewhere between $200B and $400B at current prices, depending on the day.

Some of that is unrecoverable in principle (the keys are gone, full stop). But a measurable slice is recoverable in principle:

- **Brain wallets**: thousands of wallets where the private key is `SHA256(password)` of a passphrase a human picked. People used song lyrics, Bible verses, "correct horse battery staple", their dog's name. Most were drained by automated scanners within months. A long tail were not. **Some still hold real Bitcoin today.**
- **Weak-PRNG wallets**: every few years a popular wallet ships with a broken random-number generator. Milk Sad. Profanity. Trust Wallet. Each of those CVEs left tens of thousands of wallets with enumerable private keys. Most got scanned and emptied within weeks of public disclosure. The ones that did not got forgotten.
- **Lost-and-forgotten wallets**: addresses that received Bitcoin in 2011 and have never moved. The owner has the keys somewhere. Or used to.

Pro is built to find these.

---

## The killer feature: opportunistic scanning while you solve

This is the part of Pro that nobody else does. It is the reason ColliderPro exists.

**While your GPU is solving puzzle 135 in pool or standalone mode, every key it computes is also being checked against a 100 million entry funded-address bloom filter.** Inside the same GPU kernel pass. With no additional GPU time, no additional electricity, no additional anything.

Here is what is happening, simplified:

```
Free build, pool mode:
  GPU computes pubkey -> derives DP -> sends DP to pool server.
  (One target. Puzzle 135. That is the whole story.)

Pro build, pool mode:
  GPU computes pubkey -> derives DP -> hashes to H160 (Bitcoin address) ->
    queries bloom filter for 100M+ funded addresses ->
    if hit: writes candidate to bloom_hits.txt for CPU verification ->
  Sends DP to pool server.
  (Still mining puzzle 135. But also scanning every key for any
   funded wallet, anywhere, ever, in history.)
```

The marginal cost is **zero**. The DP is already on the GPU. The H160 is already a few ops away. The bloom query is a handful of memory reads. The Pro kernel does all of it in the same pass that the Free kernel uses to just send the DP.

A 4090 doing 8 GKeys/s in pool mode is, in Pro, also doing 8 GKeys/s of opportunistic funded-address checking. **Per machine. Every machine. All the time.**

Will you find an address? Probably not. The chance on any given key is roughly 1 in 2^130. But you are computing trillions of keys. And if your DP X coordinate ever happens to be the X coordinate of a key behind a funded wallet, **the bloom filter catches it**, the hit gets logged, and you can spend the result.

It is the most expensive Bitcoin scratch-off in the world, and you were buying it anyway.

---

## Dedicated brain-wallet mode: your GPU is a key factory

When you are not pool-mining puzzle 135 (overnight, between work assignments, when the pool is quiet), Pro flips to **dedicated brain-wallet mode**: a purpose-built pipeline that systematically scans human-chosen passphrases against the same 100M+ funded-address bloom.

Bitcoin's earliest users protected coins with a passphrase that hashes directly to a private key. `SHA256("correct horse battery staple")` is a private key. No wallet file. No mnemonic. Just words. **Tens of thousands of these wallets were created between 2011 and 2014, and many of them still hold real Bitcoin today.**

The Pro brain-wallet pipeline runs at hundreds of millions of candidates per second on a 4090, through a fused GPU kernel:

```
passphrase -> SHA-256 -> secp256k1 (private to public key) ->
RIPEMD-160(SHA-256(pubkey)) -> Bitcoin address ->
bloom filter query -> hit?
```

Every stage stays on the GPU. Nothing rolls back to the CPU until a bloom hit needs definitive verification.

### Generators that are smarter than dumb enumeration

You do not want to enumerate every 12-character string. There are too many. Pro ships three classes of generator that are smarter than brute force:

- **PCFG (Probabilistic Context-Free Grammar)**: trained on real password dumps. Generates `word + number + symbol` patterns in descending probability order. The high-probability cracks hit in seconds. The long tail descends predictably.
- **Markov chains**: character-level transition models trained on human-readable passphrases. Outputs that look like words, not random gibberish.
- **Hashcat-style rule engine**: take any wordlist (`rockyou.txt`, the Wikipedia corpus, your own custom list) and mutate it into **billions** of candidates via 200+ standard rules (case, leetspeak, append digits, prepend symbols, reverse, duplicate, capitalize-every-Nth, etc.).

`rockyou.txt` is 14 million words. `rockyou.txt × best64.rule` is roughly 9 billion candidates. Pro can rip through it in hours on a single 4090.

---

## Weak-PRNG sweeps: catch the next Milk Sad

Every few years a popular wallet ships with a broken random-number generator and quietly drains user funds for months before anyone notices. The disclosures hit, the scanners come out, and the long tail of wallets that nobody noticed gets emptied. Some never do.

Pro's **v2 multi-scheme kernel** scans for keys generated by historically-broken generators. Each candidate seed is dispatched through every known vulnerable scheme in a single GPU pass:

| Scheme                  | CVE / Disclosure  | Notes                                                |
| ----------------------- | ----------------- | ---------------------------------------------------- |
| **libbitcoin Milk Sad** | CVE-2023-39910    | 32-bit entropy bug, 2014 to 2017. Millions affected. |
| **Profanity / 1inch**   | CVE-2022-40769    | Predictable vanity-address seed.                     |
| **Trust Wallet**        | 2018 to 2022      | Documented weak-entropy derivation paths.            |
| **glibc PRNG**          | Various           | Known seed ranges, cheap to enumerate.               |
| **MSVC rand()**         | Various           | Predictable seed pattern on Windows.                 |
| **Java SecureRandom**   | Pre-Java 8 issues | Documented predictability on early Android.          |

Five Bitcoin address types (P2PKH, P2SH, P2WPKH, P2WSH, P2TR) get derived from every candidate seed. Eight scheme variants. **Every derived address checked against the same 100M+ funded-address bloom.** One GPU pass per seed.

This is the kind of attack surface that paid private tools have charged $5000/year for. Pro ships it.

---

## The secret weapon: the 100M+ funded-address bloom

The opportunistic check, the brain-wallet pipeline, the weak-PRNG sweeps — all three sit on top of a single foundational dataset: **every Bitcoin address that has ever held a non-zero balance.**

Pro ships with `funded_addresses.blf`, a 142 MB GPU-resident bloom filter built from:

- Every output address on every transaction on the Bitcoin blockchain.
- All five address types: P2PKH, P2SH, P2WPKH, P2WSH, P2TR.
- Built fresh against recent UTXO snapshots; updates ship with each Pro release.

False positive rate at the shipped capacity: roughly **1 in 1 billion**. False negatives: zero by construction. Every bloom-positive hit triggers a definitive CPU-side check against the full address set, so spurious hits never produce false alarms in `bloom_hits.txt`.

The bloom is structured for **GPU-resident queries**: bit array on device memory, xxHash64 lookups per candidate, no CPU round-trip per check. That is what makes the opportunistic scanning during kangaroo work essentially free; queries happen at memory-bandwidth speed inside the same kernel that computed the address.

Build your own bloom from a custom address list with `build_bloom` (ships in `tools/`).

---

## What you get when you buy

| Capability                                            | Free       | Pro            |
| ----------------------------------------------------- | ---------- | -------------- |
| Solve puzzle 71 to 135 (kangaroo, brute force, pool)  | Yes        | Yes            |
| **Opportunistic bloom scan during pool / standalone** | **No**     | **Yes**        |
| **Dedicated brain-wallet mode (PCFG, Markov, rules)** | **No**     | **Yes**        |
| **Weak-PRNG sweeps (v2 multi-scheme kernel)**         | **No**     | **Yes**        |
| **100M+ funded-address bloom filter (shipped)**       | **No**     | **Yes**        |
| **Real-time hit logging to `bloom_hits.txt`**         | **No**     | **Yes**        |
| All five Bitcoin address types in derivation          | Limited    | Yes            |
| Hashcat-style rule engine (GPU)                       | No         | Yes            |
| License                                               | MIT (Free) | Commercial Pro |

Pro is the same binary you already run. Features unlock when a valid Ed25519-signed license key is present. The license verifies **offline** (the public key and license slot are embedded in the binary at build time). Once activated, your interactive UI gains the brain-wallet and weak-PRNG modes; pool and puzzle mode silently enable the opportunistic bloom scan; everything else behaves identically to Free.

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

From that moment, every DP your GPU produces also gets its address checked. Hits log to `bloom_hits.txt` in the working directory. You can `tail -f bloom_hits.txt` in another terminal and walk away.

### Dedicated brain-wallet mode

```bash
./collider_pro --brainwallet \
               --wordlist rockyou.txt \
               --rules rules/best64.rule \
               --bloom funded_addresses.blf
```

This is pure scanner mode. No puzzle work. Every passphrase from `rockyou.txt`, mutated through `best64.rule`, hashed to a Bitcoin address, checked against the bloom. 9 billion candidates on `rockyou × best64`. Hours, not days, on a 4090.

### v2 weak-PRNG sweep

```bash
./collider_pro --puzzle-only-v2 \
               --schemes all \
               --addr-types puzzle_only \
               --bloom funded_addresses.blf
```

Scans every supported broken-PRNG scheme across the supported seed ranges, against every Bitcoin address type, with the bloom filter as the result gate.

---

## Pricing

Pro licenses are tier-priced by use case. See [collisionprotocol.com/pro](https://collisionprotocol.com/pro) for current pricing, terms, and license keys.

Every license is delivered as an Ed25519-signed key file. Activation is offline. No phone-home, no telemetry, no per-run server check. The license file lives next to the binary; the binary verifies the signature against the embedded public key.

---

## FAQ

**Q: Is the opportunistic scan really free? It cannot be free.**

A: It is essentially free. The GPU is already computing the pubkey and hashing it to derive the DP for the pool. The H160 derivation is one extra RIPEMD-160 pass (which is cheap relative to the secp256k1 group operation that dominates the kernel). The bloom query is a few memory reads. Empirically, enabling the opportunistic scan costs roughly 1 to 3 percent of overall GPU throughput. You give up 1 to 3 percent of your puzzle-135 odds in exchange for an entirely separate lottery against 100 million funded addresses. The math is clearly favorable.

**Q: What is the probability of an opportunistic hit?**

A: Astronomically low on any given key. Bitcoin's keyspace is 2^256; the funded-address set is roughly 2^27. The probability of any random key happening to produce an H160 in the funded set is roughly 1 in 2^229. But you are computing trillions of keys per day per machine. If you mine for a year, you compute somewhere around 2^57 to 2^60 keys (rough estimate; depends on hardware and uptime). The expected number of hits per year on a single machine is still vanishingly small. **The point is not "you will probably find one." The point is "the marginal cost of looking is zero, so you might as well look."**

**Q: What happens if I do find a hit?**

A: Pro writes the candidate to `bloom_hits.txt` with the H160, the candidate private key (in hex), and the timestamp. **Pro never spends the funds automatically.** You verify the hit manually (check the address on a block explorer; sweep the funds with a wallet you control if it is non-zero). Spending found funds is a separate workflow Pro does not touch; that is on purpose, because hits can be your own test addresses, dust, or honeypots.

**Q: Is this legal?**

A: In every jurisdiction we know of: yes, with the obvious caveats. Computing keys and checking against a public address list is not theft. Spending funds from an address whose key you happened to compute is jurisdiction-dependent and morally complicated; the funds belong to whoever originally controlled them, even if the bookkeeping is on a public ledger. Pro is a tool. You decide what to do with what it finds. **Consult a lawyer in your jurisdiction before sweeping any non-trivial balance from a found wallet.** This documentation is not legal advice.

**Q: Why not just open-source the bloom and the brain-wallet pipeline?**

A: Two reasons. First, the 100M+ funded-address bloom is built from datasets that take operational work to maintain (recent UTXO snapshots, address-type expansion, occasional manual filtering). Pro pays for that maintenance. Second, the PCFG and Markov generators are tuned against real password-cracking datasets that came from operational pentesting work. Releasing them under MIT would dilute the value of that tuning for the people who fund the project. The free build is intentionally a complete puzzle solver; Pro is intentionally a complete treasure hunter.

**Q: What if a bug in Pro burns through my electricity for nothing?**

A: Pro ships the same telemetry as Free (DP rate, accepted-by-pool counter, bloom hits if any). If your DP rate drops or your accrual on the pool stalls, you will see it in the same place you see it in Free. Open an issue on the [private Pro tracker](https://github.com/hevnsnt/collider-pro) (access included with your license) and we will triage. The free build's CI runs the same kernels; we have not shipped a Pro build that produces correct opportunistic scans but incorrect kangaroo DPs, and the design makes that hard to do accidentally.

---

## Buying a license

Head to [collisionprotocol.com/pro](https://collisionprotocol.com/pro).

Pick a tier. Pay. Receive your Ed25519-signed license file by email. Drop it next to your `collider_pro` binary. The next run unlocks Pro features and prints your license fingerprint at startup.

You bought the hardware. You pay the power bill. **Pro makes sure every key you compute is doing as much work for you as it can.**

---

## Where to go next

- [Back to README](../README.md) — overview and quick-start.
- [POOL.md](POOL.md) — pool mode, share-of-pool accrual, picking a worker address.
- [CONFIGURATION.md](CONFIGURATION.md) — `config.yml` schema, including the `bloom:` block.
- [ARCHITECTURE.md](ARCHITECTURE.md) — source-tree map; the Pro pipeline lives in `src/gpu/mega_fused_kernel.cu` and `src/gpu/rckangaroo_wrapper.cu` (opportunistic path).
- [collisionprotocol.com/pro](https://collisionprotocol.com/pro) — buy a license.
