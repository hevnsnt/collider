# DEF CON Password Cracking Village: Strategies & Learnings

## Executive Summary

This document synthesizes winning strategies from DEF CON's "Crack Me If You Can" (CMIYC) competitions and Password Village techniques, adapted for brain wallet cracking in Superflayer.

**Key Insight**: GPU power alone doesn't win competitions. The winners consistently emphasize **wordlist quality**, **rule optimization**, and **contextual intelligence** over raw hashrate.

---

## Competition-Proven Strategies

### 1. Wordlist Quality Over Quantity

From CMIYC 2022 winner writeup:
> "The best password-cracking dictionary for this year's contest? The English dictionary."

**Strategy for Superflayer:**
```
Priority Order:
1. Leaked password databases (RockYou, LinkedIn, Adobe, etc.)
2. Previously cracked brain wallets (17,956 known compromised)
3. Contextual sources (song lyrics, quotes, Wikipedia)
4. Combinatorial generation (word + word + number)
5. Pure brute force (last resort)
```

### 2. Rule Efficiency Hierarchy

From Team Hashcat's decade of dominance:

| Rule Set | Crack Rate | Size | Efficiency |
|----------|------------|------|------------|
| best64.rule | 62% | 64 rules | Highest |
| OneRuleToRuleThemAll | 68.36% | 52,000 rules | Medium |
| OneRuleToRuleThemStill | 68.36% | 50,088 rules | Optimized |
| dive.rule | 71% | 99,000+ rules | Low |
| Pantagrule | 73% | 800,000+ rules | Very Low |

**Key Learning**: OneRuleToRuleThemAll cracks 68.36% of passwords with only 52K rules. Running massive rule sets has diminishing returns.

### 3. The Pivot Strategy

From CMIYC 2024 winner:
> "Being able to pivot is as critical as always"

**Implementation**: Build a feedback loop where cracked passwords inform next attack:
```
Cracked "bitcoin2024!"
  → Extract pattern: [word][year][symbol]
  → Generate: ethereum2024!, litecoin2024!, dogecoin2024!
  → Add to active wordlist
```

### 4. Potfile Management

> "Meticulous use of hashcat's `--potfile-path` made per-hash-type crack tracking way easier"

**Superflayer Adaptation**:
- Maintain separate potfiles per source type (lyrics, quotes, passwords)
- Track which wordlist sources are most productive
- Prune unproductive sources in real-time

### 5. Web Research for Context

From CMIYC 2022:
> "Search for contextual phrases found in cracked passwords... I discovered significant score improvements by locating articles about specific events, then scraping those sources"

**Brain Wallet Specific Sources**:
- Bitcoin/crypto forums (2009-2014 era)
- Early adopter communities
- Cypherpunk mailing lists
- Silk Road archives
- Tech conference proceedings

---

## PCFG (Probabilistic Context-Free Grammar)

### How It Works

PCFG analyzes password structure patterns:
```
"Password123!" → L8 D3 S1 (8 letters, 3 digits, 1 symbol)
"Summer2024"   → U1L5 D4   (1 upper, 5 lower, 4 digits)
```

Train on cracked passwords to learn probability distributions:
```
P(L8 D3 S1) = 0.023
P(U1L5 D4)  = 0.018
P(L6 D2)    = 0.031  ← Most common
```

### Training Requirements

| Dataset Size | Quality | Use Case |
|--------------|---------|----------|
| 10K-100K | Minimum viable | Single-domain attacks |
| 100K-1M | Good coverage | General purpose |
| 1M-50M | Excellent | Competition-grade |

**Brain Wallet Specific Training Data**:
1. Known compromised brain wallets (17,956 passwords)
2. Bitcoin Talk forum password leaks
3. Crypto exchange breaches
4. General password corpuses with crypto-related filters

### PCFG + Brainflayer Integration

```
[PCFG Trainer] → [Grammar Rules] → [Candidate Generator] → [Brainflayer/Superflayer]
     ↑                                    ↓
     └──────── Feedback Loop ─────────────┘
```

---

## Passphrase Sources (The Intelligence Layer)

### Tier 1: High-Value Targets

| Source | Est. Entries | Why |
|--------|--------------|-----|
| Song lyrics (1950-2015) | ~50M lines | Memorable, emotional |
| Movie quotes (IMDB top 10K) | ~500K | Cultural touchstones |
| Book first lines | ~100K | Literary references |
| Wikipedia article titles | ~6M | Common knowledge |
| Famous quotes (Wikiquote) | ~200K | "Inspirational" choices |

### Tier 2: Crypto-Specific

| Source | Est. Entries | Why |
|--------|--------------|-----|
| Bitcoin whitepaper phrases | ~1K | Ideological significance |
| Cypherpunk manifesto | ~500 | Early adopter culture |
| Crypto subreddit titles | ~5M | Community language |
| Bitcointalk.org phrases | ~10M | Historical forum |
| Satoshi's emails/posts | ~500 | Mythology |

### Tier 3: Personal Patterns

| Pattern | Example | Frequency |
|---------|---------|-----------|
| Name + birthdate | john19850315 | Very high |
| Pet + year | fluffy2012 | High |
| Sports team + number | yankees42 | Medium |
| Location + number | newyork10001 | Medium |

---

## Rule Engine Design

### Core Mutation Rules (from best64.rule)

```
:           # No change (passthrough)
l           # Lowercase all
u           # Uppercase all
c           # Capitalize first, lower rest
t           # Toggle case
r           # Reverse
d           # Duplicate
$1          # Append "1"
$!          # Append "!"
^1          # Prepend "1"
sa4         # Substitute a→4
se3         # Substitute e→3
si1         # Substitute i→1
so0         # Substitute o→0
ss$         # Substitute s→$
```

### Brain Wallet Specific Rules

```
# Bitcoin-themed appends
$b$t$c      # Append "btc"
$B$T$C      # Append "BTC"
$2$0$0$9    # Append "2009" (Bitcoin genesis year)
$2$0$1$3    # Append "2013" (first bubble)
$2$0$1$7    # Append "2017" (major bubble)

# Crypto substitutions
ss$         # s → $ (dollar sign)
sB8         # B → 8 (Bitcoin 8)
so0         # o → 0 (zero)

# Common brain wallet patterns
c $1        # Capitalize + append 1
c $!        # Capitalize + append !
c $1 $2 $3  # Capitalize + append 123
l $b$t$c    # lowercase + btc
```

### Combinator Attacks

Combine two wordlists:
```
[Adjectives] × [Nouns] × [Numbers]
  "correct"  × "horse"  × "2014"
  "happy"    × "bitcoin" × "!"
  "strong"   × "wallet"  × "123"
```

---

## Implementation Architecture

### Phase 1: Intelligence Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    SOURCE COLLECTORS                         │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│  Lyrics  │  Quotes  │  Wiki    │  Crypto  │  Leaked PWs    │
│ Scraper  │ Scraper  │ Scraper  │ Forums   │  Importer      │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┴───────┬────────┘
     │          │          │          │             │
     ▼          ▼          ▼          ▼             ▼
┌─────────────────────────────────────────────────────────────┐
│                  NORMALIZATION LAYER                         │
│  - Lowercase conversion    - Unicode normalization           │
│  - Whitespace handling     - Duplicate removal               │
│  - Length filtering        - Encoding standardization        │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PCFG TRAINER                              │
│  - Pattern extraction      - Probability assignment          │
│  - Grammar generation      - Keyboard pattern detection      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   CANDIDATE GENERATOR                        │
│  - Priority queue (by probability)                           │
│  - Rule application        - Combinator expansion            │
│  - Deduplication           - Rate limiting for GPU           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                    [GPU CRACKING PIPELINE]
```

### Phase 2: Feedback Loop

```python
class FeedbackLoop:
    def on_crack(self, passphrase: str, source: str):
        # 1. Analyze structure
        pattern = self.pcfg.analyze(passphrase)

        # 2. Update source weights
        self.source_weights[source] += 1

        # 3. Extract new rules
        if self.detect_mutation(passphrase):
            new_rule = self.extract_rule(passphrase)
            self.active_rules.append(new_rule)

        # 4. Generate variations
        variations = self.mutate(passphrase)
        self.priority_queue.add_high_priority(variations)
```

---

## Performance Expectations

### Wordlist Mode (No Rules)

| Wordlist Size | Keys/sec (4× RTX 5090) | Time to Exhaust |
|---------------|------------------------|-----------------|
| 1 Million | 10B | 0.0001 seconds |
| 1 Billion | 10B | 0.1 seconds |
| 1 Trillion | 10B | 100 seconds |

### With Rule Expansion

| Base Words | Rules | Effective Size | Time |
|------------|-------|----------------|------|
| 1M | best64 (64) | 64M | 0.006s |
| 1M | OneRule (50K) | 50B | 5s |
| 10M | OneRule (50K) | 500B | 50s |
| 100M | best64 (64) | 6.4B | 0.64s |

### Optimal Strategy

```
1. Quick wins first:
   - Known brain wallet passwords (17K)
   - Top 1M passwords + best64
   - Crypto-specific wordlist + best64

2. Medium effort:
   - Full password corpuses + OneRuleToRuleThemStill
   - Lyrics/quotes databases + light rules

3. Deep search:
   - PCFG-generated candidates
   - Combinator attacks
   - Full mutation chains
```

---

## Key Takeaways for Superflayer

1. **Intelligence > Compute**: A well-curated 10M wordlist beats a random 10B wordlist
2. **Feedback is Critical**: Cracked passwords inform next attacks
3. **Rules Have Diminishing Returns**: best64 → OneRule → stop
4. **Context Matters**: Crypto-specific sources outperform generic ones
5. **PCFG Bridges the Gap**: Learn patterns, generate smart candidates
6. **Prioritize by Probability**: Attack likely passwords before unlikely ones

---

## References

- [Team Hashcat GitHub](https://github.com/hashcat/team-hashcat)
- [CMIYC 2024 Winner Writeup](https://barelycompetent.dev/crackmeifyoucan-cmiyc-2024-@-def-con-32-results/)
- [OneRuleToRuleThemStill](https://in.security/2023/01/10/oneruletorulethemstill-new-and-improved/)
- [PCFG Cracker](https://github.com/lakiw/pcfg_cracker)
- [Cracking Cryptocurrency Brainwallets - Ryan Castellucci](https://rya.nc/files/cracking_cryptocurrency_brainwallets.pdf)
- [Bitcoin Brain Drain Paper](https://mvasek.com/static/papers/vasekfc16.pdf)
