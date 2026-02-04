# Wordlist Architecture: Superflayer Intelligence Layer

## Design Philosophy

Superflayer treats wordlist generation as an **intelligence problem**, not a brute-force problem. The goal is to generate candidates in **probability order**—most likely passphrases first.

```
Traditional:  Generate ALL candidates → Test ALL
Superflayer:  Generate LIKELY candidates → Test → Learn → Generate MORE LIKELY → Repeat
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SUPERFLAYER INTELLIGENCE LAYER                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   SCRAPERS   │  │   IMPORTERS  │  │  GENERATORS  │  │   FEEDBACK   │    │
│  │              │  │              │  │              │  │              │    │
│  │ • Lyrics     │  │ • RockYou    │  │ • PCFG       │  │ • Cracked    │    │
│  │ • Wikiquote  │  │ • LinkedIn   │  │ • Markov     │  │   passwords  │    │
│  │ • Wikipedia  │  │ • Adobe      │  │ • Combinator │  │ • Patterns   │    │
│  │ • Forums     │  │ • Brain PWs  │  │ • Mask       │  │ • Rules      │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                 │                 │             │
│         └────────────────┬┴─────────────────┴─────────────────┘             │
│                          │                                                   │
│                          ▼                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      NORMALIZATION PIPELINE                            │  │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │  │
│  │  │ Decode  │→ │ Clean   │→ │ Filter  │→ │ Dedup   │→ │ Score   │     │  │
│  │  │ UTF-8   │  │ Spaces  │  │ Length  │  │ Hash    │  │ Prob    │     │  │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        RULE ENGINE                                     │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │  │
│  │  │   best64    │  │  OneRule    │  │  Custom     │                    │  │
│  │  │   (fast)    │  │  (thorough) │  │  (crypto)   │                    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                   │
│                          ▼                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    PRIORITY QUEUE                                      │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │  │
│  │  │ Tier 1 (P > 0.001)  │ Tier 2 (P > 0.0001) │ Tier 3 (P > 1e-6)  │  │  │
│  │  │ Known brain wallets │ Common passwords    │ PCFG generated     │  │  │
│  │  │ Top 10K passwords   │ Lyrics + rules      │ Combinator output  │  │  │
│  │  └─────────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                          │                                                   │
│                          ▼                                                   │
│                  [GPU BATCH BUFFER - 4M candidates]                          │
│                          │                                                   │
│                          ▼                                                   │
│                  [4× RTX 5090 CRACKING PIPELINE]                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Source Collectors

### 1. Lyrics Scraper

**Target**: Song lyrics from 1950-2020

```python
class LyricsScraper:
    sources = [
        "genius.com",        # ~5M songs
        "azlyrics.com",      # ~1M songs
        "lyrics.com",        # ~1M songs
    ]

    def extract_phrases(self, lyrics: str) -> List[str]:
        """Extract memorable phrases from lyrics."""
        phrases = []

        # Full lines (most common brain wallet pattern)
        for line in lyrics.split('\n'):
            clean = self.normalize(line)
            if 8 <= len(clean) <= 64:
                phrases.append(clean)

        # First lines of verses (highly memorable)
        verses = self.split_verses(lyrics)
        for verse in verses:
            first_line = verse.split('\n')[0]
            phrases.append(self.normalize(first_line))

        # Chorus (repeated = memorable)
        chorus = self.detect_chorus(lyrics)
        if chorus:
            phrases.extend(self.normalize(line) for line in chorus)

        return phrases
```

**Output Format**:
```
# lyrics_phrases.txt
dont stop believin
hold on to that feeling
we are the champions my friends
imagine all the people
```

### 2. Quote Scraper

**Target**: Wikiquote, IMDB, Goodreads

```python
class QuoteScraper:
    sources = {
        "wikiquote": "https://en.wikiquote.org/wiki/",
        "imdb": "https://www.imdb.com/search/title/?title_type=feature",
        "goodreads": "https://www.goodreads.com/quotes",
    }

    categories = [
        "Philosophy",
        "Literature",
        "Film",
        "Technology",
        "Science",
        "Politics",
        "Religion",
        "Motivation",  # High brain wallet correlation
    ]

    def normalize_quote(self, quote: str) -> List[str]:
        """Generate variations of a quote."""
        variations = []

        # Original
        variations.append(quote.lower().strip())

        # Without punctuation
        clean = re.sub(r'[^\w\s]', '', quote.lower())
        variations.append(clean)

        # Without spaces
        variations.append(clean.replace(' ', ''))

        # First N words
        words = clean.split()
        for n in [3, 4, 5, 6]:
            if len(words) >= n:
                variations.append(' '.join(words[:n]))
                variations.append(''.join(words[:n]))

        return variations
```

### 3. Wikipedia Scraper

**Target**: Article titles, first sentences

```python
class WikipediaScraper:
    def scrape_titles(self) -> Iterator[str]:
        """Stream all Wikipedia article titles."""
        # Use Wikipedia dumps (~20GB compressed)
        # https://dumps.wikimedia.org/enwiki/latest/
        dump_url = "enwiki-latest-all-titles-in-ns0.gz"

        with gzip.open(dump_url, 'rt') as f:
            for line in f:
                title = line.strip().replace('_', ' ')
                yield title.lower()

    def scrape_first_sentences(self) -> Iterator[str]:
        """Extract first sentence of articles."""
        # Target: ~6M articles
        # Focus on: People, Places, Events, Concepts
        pass
```

### 4. Crypto Forum Scraper

**Target**: BitcoinTalk, Reddit r/bitcoin (2009-2014)

```python
class CryptoForumScraper:
    """
    Critical source: Early adopters likely used crypto-related phrases.
    """

    targets = [
        ("bitcointalk.org", 2009, 2016),
        ("reddit.com/r/bitcoin", 2010, 2016),
        ("reddit.com/r/cryptocurrency", 2013, 2016),
    ]

    keyword_boost = [
        "satoshi", "nakamoto", "bitcoin", "blockchain",
        "crypto", "wallet", "private key", "seed",
        "moon", "hodl", "lambo", "doge",
    ]

    def extract_memorable_phrases(self, post: str) -> List[str]:
        """Extract phrases that might be used as brain wallets."""
        # Thread titles
        # Signatures
        # Quoted text
        # Memes and catchphrases
        pass
```

### 5. Password Importer

**Target**: Known breached password lists

```python
class PasswordImporter:
    sources = {
        "rockyou": 14_344_391,      # Classic
        "linkedin": 117_000_000,    # Professional
        "adobe": 153_000_000,       # Tech-savvy
        "collection1": 773_000_000, # Mega breach
        "brain_wallets": 17_956,    # Known compromised (PRIORITY)
    }

    def import_with_frequency(self, source: str) -> Dict[str, int]:
        """Import passwords with occurrence counts."""
        passwords = defaultdict(int)

        with open(source) as f:
            for line in f:
                pw = line.strip()
                passwords[pw] += 1

        # Sort by frequency
        return dict(sorted(
            passwords.items(),
            key=lambda x: -x[1]
        ))
```

---

## Normalization Pipeline

### Stage 1: Decode

```python
def decode_utf8(raw: bytes) -> str:
    """Handle various encodings gracefully."""
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='ignore')
```

### Stage 2: Clean

```python
def clean_phrase(phrase: str) -> str:
    """Normalize whitespace and case."""
    # Collapse whitespace
    phrase = ' '.join(phrase.split())

    # Convert to lowercase (keep original as variant)
    phrase_lower = phrase.lower()

    # Remove leading/trailing whitespace
    return phrase_lower.strip()
```

### Stage 3: Filter

```python
def filter_candidates(phrase: str) -> bool:
    """Filter out unlikely brain wallet candidates."""

    # Length bounds (SHA256 input practical limits)
    if len(phrase) < 6 or len(phrase) > 128:
        return False

    # Must contain at least one letter
    if not re.search(r'[a-zA-Z]', phrase):
        return False

    # Skip pure hex (likely not a passphrase)
    if re.match(r'^[0-9a-fA-F]+$', phrase) and len(phrase) >= 32:
        return False

    return True
```

### Stage 4: Deduplicate

```python
class BloomDeduplicator:
    """Memory-efficient deduplication using Bloom filter."""

    def __init__(self, expected_items: int = 1_000_000_000):
        # 1% false positive rate
        self.bloom = BloomFilter(
            max_elements=expected_items,
            error_rate=0.01
        )

    def is_duplicate(self, phrase: str) -> bool:
        phrase_hash = hashlib.sha256(phrase.encode()).digest()

        if phrase_hash in self.bloom:
            return True  # Probably seen (1% FP)

        self.bloom.add(phrase_hash)
        return False
```

### Stage 5: Score

```python
class ProbabilityScorer:
    """Assign probability scores to candidates."""

    def __init__(self):
        self.password_freq = self.load_password_frequencies()
        self.ngram_model = self.load_ngram_model()
        self.source_weights = {
            "brain_wallets_known": 1000.0,  # Highest priority
            "rockyou_top10k": 100.0,
            "lyrics": 10.0,
            "quotes": 10.0,
            "wikipedia": 1.0,
            "pcfg_generated": 0.1,
        }

    def score(self, phrase: str, source: str) -> float:
        """Calculate probability score."""
        base_score = self.source_weights.get(source, 0.01)

        # Boost for known password patterns
        if phrase in self.password_freq:
            base_score *= (1 + self.password_freq[phrase])

        # Penalize unlikely character sequences
        ngram_score = self.ngram_model.score(phrase)
        base_score *= ngram_score

        return base_score
```

---

## Rule Engine

### Architecture

```python
class RuleEngine:
    """Apply hashcat-compatible rules to candidates."""

    def __init__(self):
        self.rule_sets = {
            "passthrough": [":"],
            "best64": self.load_rules("best64.rule"),
            "one_rule": self.load_rules("OneRuleToRuleThemStill.rule"),
            "crypto": self.load_crypto_rules(),
        }

    def apply_rules(
        self,
        base: str,
        rule_set: str = "best64"
    ) -> Iterator[str]:
        """Apply rule set to base candidate."""
        rules = self.rule_sets[rule_set]

        for rule in rules:
            try:
                yield self.apply_rule(base, rule)
            except RuleError:
                continue

    def apply_rule(self, word: str, rule: str) -> str:
        """Apply single hashcat rule."""
        result = word

        i = 0
        while i < len(rule):
            op = rule[i]

            if op == ':':
                pass  # No-op
            elif op == 'l':
                result = result.lower()
            elif op == 'u':
                result = result.upper()
            elif op == 'c':
                result = result.capitalize()
            elif op == 'r':
                result = result[::-1]
            elif op == 'd':
                result = result + result
            elif op == '$':
                result = result + rule[i+1]
                i += 1
            elif op == '^':
                result = rule[i+1] + result
                i += 1
            elif op == 's':
                result = result.replace(rule[i+1], rule[i+2])
                i += 2
            # ... more rules

            i += 1

        return result
```

### Crypto-Specific Rules

```python
def load_crypto_rules(self) -> List[str]:
    """Rules targeting crypto brain wallet patterns."""
    return [
        # Year appends (crypto boom years)
        "$2$0$0$9",  # 2009 - Bitcoin genesis
        "$2$0$1$0",  # 2010
        "$2$0$1$1",  # 2011
        "$2$0$1$3",  # 2013 - First major bubble
        "$2$0$1$7",  # 2017 - ATH bubble
        "$2$0$2$1",  # 2021 - Recent ATH

        # Crypto keywords
        "$b$t$c",
        "$B$T$C",
        "$e$t$h",
        "$c$o$i$n",

        # Common endings
        "$!",
        "$1",
        "$1$2$3",
        "$!$!$!",

        # Leet speak for crypto
        "so0",       # o → 0
        "sa4",       # a → 4
        "se3",       # e → 3
        "ss$",       # s → $
        "sB8",       # B → 8

        # Combinations
        "c$2$0$1$3",
        "c$b$t$c",
        "l$!",
        "u$1$2$3",
    ]
```

---

## Priority Queue

### Implementation

```python
import heapq
from dataclasses import dataclass, field

@dataclass(order=True)
class Candidate:
    priority: float
    phrase: str = field(compare=False)
    source: str = field(compare=False)
    rule_applied: str = field(compare=False)

class PriorityQueue:
    """Probability-ordered candidate queue."""

    def __init__(self, max_size: int = 100_000_000):
        self.heap = []
        self.max_size = max_size
        self.seen = BloomFilter(max_size, error_rate=0.001)

    def push(self, phrase: str, priority: float, source: str, rule: str = ":"):
        """Add candidate if not seen and queue not full."""
        if phrase in self.seen:
            return

        self.seen.add(phrase)

        # Negate priority for max-heap behavior
        candidate = Candidate(-priority, phrase, source, rule)

        if len(self.heap) < self.max_size:
            heapq.heappush(self.heap, candidate)
        elif candidate < self.heap[0]:
            heapq.heapreplace(self.heap, candidate)

    def pop_batch(self, batch_size: int = 4_000_000) -> List[str]:
        """Pop batch of highest-priority candidates."""
        batch = []

        for _ in range(min(batch_size, len(self.heap))):
            candidate = heapq.heappop(self.heap)
            batch.append(candidate.phrase)

        return batch
```

---

## GPU Batch Buffer

### Interface with Cracking Pipeline

```python
class GPUBatchBuffer:
    """Buffer candidates for GPU processing."""

    BATCH_SIZE = 4_000_000  # 4M candidates per GPU batch

    def __init__(self, priority_queue: PriorityQueue):
        self.queue = priority_queue
        self.current_batch = []

    def fill_batch(self) -> List[bytes]:
        """Fill batch buffer for GPU."""
        self.current_batch = self.queue.pop_batch(self.BATCH_SIZE)

        # Convert to bytes for GPU transfer
        return [p.encode('utf-8') for p in self.current_batch]

    def stream_to_gpu(self) -> Iterator[bytes]:
        """Stream candidates to GPU as they're generated."""
        while True:
            batch = self.fill_batch()
            if not batch:
                break

            yield from batch

    def feedback_crack(self, cracked: str, original_source: str):
        """Handle cracked password feedback."""
        # Log the crack
        logger.info(f"CRACKED: {cracked} (source: {original_source})")

        # Extract patterns for future attacks
        pattern = self.analyze_pattern(cracked)
        self.generate_variations(cracked, pattern)
```

---

## Disk-Backed Storage

For handling billions of candidates:

```python
class DiskBackedWordlist:
    """Sorted, disk-backed wordlist for massive scale."""

    def __init__(self, path: str):
        self.path = path
        self.db = rocksdb.DB(
            path,
            rocksdb.Options(create_if_missing=True)
        )

    def add(self, phrase: str, priority: float):
        """Add phrase with priority score."""
        # Key: priority (inverted for sorting) + phrase hash
        key = struct.pack('>d', -priority) + hashlib.md5(phrase.encode()).digest()
        self.db.put(key, phrase.encode())

    def iterate_by_priority(self) -> Iterator[str]:
        """Iterate phrases in priority order."""
        it = self.db.iterkeys()
        it.seek_to_first()

        for key in it:
            value = self.db.get(key)
            yield value.decode()
```

---

## Implementation Checklist

- [ ] Lyrics scraper (Genius API)
- [ ] Quote scraper (Wikiquote parser)
- [ ] Wikipedia title importer
- [ ] Crypto forum scraper (BitcoinTalk)
- [ ] Password list importer (with frequency)
- [ ] Normalization pipeline
- [ ] Bloom filter deduplicator
- [ ] Probability scorer
- [ ] Rule engine (hashcat-compatible)
- [ ] Crypto-specific rule set
- [ ] Priority queue (heap-based)
- [ ] GPU batch buffer
- [ ] Feedback loop handler
- [ ] Disk-backed storage (RocksDB)
- [ ] PCFG integration (next document)

---

## Data Volume Estimates

| Source | Raw Entries | After Normalization | With Rules (best64) |
|--------|-------------|---------------------|---------------------|
| Song lyrics | 50M lines | 30M | 1.9B |
| Quotes | 500K | 400K | 25.6M |
| Wikipedia titles | 6M | 5M | 320M |
| Crypto forums | 10M posts | 5M phrases | 320M |
| Password lists | 1B+ | 500M unique | 32B |
| **Total** | ~1.1B | ~540M | ~35B |

At 10B keys/sec: **~3.5 seconds** to test all candidates once.

With full rule expansion (OneRule): **~100 seconds**.

---

## Next Steps

1. **PCFG Integration** (see PCFG-INTEGRATION.md)
2. **Source code implementation** (see src/)
3. **GPU pipeline integration** (see ARCHITECTURE.md)
