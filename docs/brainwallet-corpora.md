# Brain-Wallet Corpora and Rule Sets

Operator guide for sourcing large external wordlists, sentence corpora, and
mutation rule sets to feed theCollider's brain-wallet scanner. These resources
are too large, too frequently updated, or too license-encumbered to bundle in
the repository, so this document tells you where to get each one and how to
point the scanner at it.

The two small curated lists that **are** bundled live in `data/`:

- `data/crypto_culture.txt`: high-probability crypto-culture and famous weak
  phrases (xkcd, Satoshi quotes, memes, coin/exchange names).
- `data/known_brainwallets.txt`: passphrases publicly documented in research
  as real funded brain wallets.

Everything below is an **external** corpus you fetch yourself and feed in via
the scanner's wordlist/corpus source (see "How the scanner consumes a corpus").

---

## ETHICS AND SCOPE (read first)

theCollider's brain-wallet mode exists for **defensive** security research:

- Auditing the operator's **own** funded test/demo wallets to confirm they are
  not derived from a guessable passphrase.
- Reproducing **published** brain-wallet research against targets the operator
  is authorized to test.
- Measuring how quickly a weak passphrase would be swept, to justify migrating
  any real funds to a properly random key (BIP39 with full entropy).

It does **not** exist to sweep funds from wallets you do not own or are not
explicitly authorized to test. Brain wallets on the public chain may hold other
people's money; deriving a key and moving those funds is theft regardless of how
weak the passphrase was. Stay inside your own wallets and authorized scope.

Several corpora below (leaked-password compilations especially) carry their own
legal and ethical constraints. Treat any breach-derived data as sensitive:
store it encrypted at rest, never redistribute it, and use it only for the
defensive purposes above. When in doubt, prefer the synthetic/cultural corpora
over breach dumps.

---

## How the scanner consumes a corpus

The brain-wallet runner reads passphrase candidates from a **corpus source**.
Each external resource below is fed in one of three shapes:

1. **Flat wordlist**: one candidate per line. Point a wordlist source at the
   file. Largest throughput, lowest cleverness.
2. **Sentence/n-gram corpus**: free text (books, subtitles, lyrics) that the
   corpus source slices into lines, sentences, and sliding n-grams so that, for
   example, a single memorable sentence from a novel becomes a candidate.
3. **Wordlist + rule set**: a base wordlist plus a hashcat-style `.rule` file.
   The rule engine mutates each base word (case folding, leet, appended digits,
   reversal) to expand a small list into millions of realistic variants.

For very large inputs, pre-sort and de-duplicate on disk first
(`sort -u big.txt > big.dedup.txt`) so you are not hashing the same candidate
twice. Keep the highest-probability sources early in the run; brain-wallet hits
are overwhelmingly short, memorable, English phrases and a handful of famous
quotes, so the curated lists plus a good rule set usually fire before you ever
reach a multi-gigabyte dump.

---

## Plain-text and sentence corpora

### Project Gutenberg (public-domain books)

Tens of thousands of public-domain books in plain UTF-8. Excellent source of
quotable opening lines, famous passages, poetry, and scripture that people reuse
as passphrases.

- Where: https://www.gutenberg.org/ and the mirror list at
  https://www.gutenberg.org/MIRRORS.ALL
- Bulk: the Gutenberg "robot" mirror lets you rsync the whole plain-text corpus;
  see https://www.gutenberg.org/help/mirroring.html
- Aggregated/cleaned: Standardized Project Gutenberg Corpus
  (https://github.com/pgcorpus/gutenberg).
- Feed as: sentence/n-gram corpus. Strip headers/footers first.

### Song-lyrics datasets

Lyrics are one of the highest-yield brain-wallet sources (chorus lines, famous
hooks). Note lyrics are usually copyrighted; use only for private defensive
research, do not redistribute.

- Genius / "5 Million Song Lyrics" style datasets on Kaggle
  (search Kaggle for "song lyrics dataset").
- MetroLyrics / "380,000+ lyrics" archived datasets.
- LyricsGenius scraper (https://github.com/johnwmillr/LyricsGenius) if you want
  to build your own from the Genius API.
- Feed as: sentence/n-gram corpus (chorus lines matter most).

### OpenSubtitles (movie and TV dialogue)

Millions of subtitle lines. Captures movie one-liners and catchphrases that
people pick as passphrases.

- Where: https://www.opensubtitles.org/ and the research-friendly OPUS
  collection at https://opus.nlpl.eu/OpenSubtitles.php
- Feed as: sentence corpus, one subtitle line per candidate.

### Wikipedia sentence dumps

Full-text Wikipedia, sliced into sentences, yields a broad sweep of phrases,
proper nouns, and definitions.

- Where: https://dumps.wikimedia.org/ (the `*-pages-articles.xml.bz2` dumps).
- Easier: pre-extracted text via WikiExtractor
  (https://github.com/attardi/wikiextractor) or the Hugging Face
  `wikipedia` / `wikimedia/wikipedia` datasets.
- Feed as: sentence/n-gram corpus. This is large; de-dupe aggressively.

### Famous-quotes datasets

Curated quote collections concentrate exactly the "memorable line" candidates
that brain wallets favor.

- Quotable API dataset (https://github.com/lukePeavey/quotable).
- Goodreads quotes datasets on Kaggle (search "goodreads quotes").
- Wikiquote dumps (https://dumps.wikimedia.org/, `*wikiquote*`).
- Feed as: flat wordlist (each quote is already one line) or sentence corpus.

---

## Leaked-password corpora (handle with care)

These are real human-chosen secrets and therefore extremely high-yield, but they
are breach-derived. See the ETHICS AND SCOPE note: encrypt at rest, never
redistribute, defensive use only.

### RockYou

The classic 14M-password leak. Small, fast, high hit rate for weak passwords.

- Where: ships with Kali at `/usr/share/wordlists/rockyou.txt.gz`; also widely
  mirrored (e.g. https://github.com/brannondorsey/naive-hashcat releases).
- Feed as: flat wordlist, ideally with a rule set on top.

### SecLists

Curated collection of many password and wordlist files (not a single breach),
maintained for security testing. The best general starting point.

- Where: https://github.com/danielmiessler/SecLists
  (`Passwords/` and `Passwords/Leaked-Databases/`).
- Feed as: flat wordlist(s).

### COMB / breach compilations

"Compilation of Many Breaches" style aggregates (billions of lines). Enormous,
noisy, and legally sensitive. Use only if smaller corpora are exhausted.

- Where: circulated on breach forums; no canonical safe link. Verify provenance
  and your authorization before touching it.
- Feed as: pre-sorted, de-duplicated flat wordlist. Expect heavy disk I/O.

### Have I Been Pwned (Pwned Passwords)

HIBP publishes ~850M real-world password **hashes** (SHA-1/NTLM), not plaintext.
Useful as a filter (is this candidate known-breached?) rather than a direct
source, but the k-anonymity range API and downloadable hash set are valuable for
prioritization.

- Where: https://haveibeenpwned.com/Passwords and the downloader at
  https://github.com/HaveIBeenPwned/PwnedPasswordsDownloader
- Feed as: not a plaintext source. Use to rank/triage other corpora.

---

## Multilingual wordlists

Brain wallets are not English-only. German, Russian, Spanish, and Chinese
(pinyin) phrases all appear. Add native-language corpora when auditing wallets
likely created by non-English speakers.

- **General**: aspell/hunspell dictionaries ship per-language word lists
  (https://ftp.gnu.org/gnu/aspell/dict/). The `wordlist` packages on most Linux
  distros (`wngerman`, `wfrench`, `wspanish`, etc.) are quick wins.
- **Frequency lists**: Hermit Dave's FrequencyWords
  (https://github.com/hermitdave/FrequencyWords) covers dozens of languages,
  ordered by real-world frequency, which is ideal candidate ordering.
- **German**: `wngerman`/`wogerman` packages; the German Wikipedia dump.
- **Russian**: OpenCorpora (https://opencorpora.org/) and the Russian
  FrequencyWords list. Remember to also try Latin transliteration.
- **Spanish**: `wspanish`, RAE-derived lists, Spanish FrequencyWords.
- **Chinese (pinyin)**: people rarely type Han characters as a passphrase, so
  prefer **pinyin** wordlists. The CC-CEDICT dictionary
  (https://www.mdbg.net/chinese/dictionary?page=cc-cedict) provides pinyin
  readings you can flatten into candidates (with and without tone numbers, with
  and without spaces between syllables).
- Feed as: flat wordlists, frequency-ordered where available, with a rule set
  for casing/digit variants.

---

## Hashcat-style rule sets

Rule files mutate a base wordlist into realistic variants (capitalize, leet,
append `123`/`!`, reverse, duplicate). A small curated list plus a strong rule
set typically outperforms a giant raw dump. theCollider's rule engine consumes
hashcat-syntax `.rule` files; the bundled examples live in `rules/`
(`best64.rule`, `crypto.rule`).

### OneRuleToRuleThemAll

The de-facto "if you only pick one" rule set. Broad coverage, good yield/size
balance.

- Where: https://github.com/stealthsploit/OneRuleToRuleThemAll
- Use with: any base wordlist as the default mutation pass.

### dive.rule

Large, aggressive rule set shipped with hashcat (`rules/dive.rule`). Much bigger
expansion factor than best64; use when you have GPU time to spare.

- Where: hashcat distribution `rules/` directory
  (https://github.com/hashcat/hashcat/tree/master/rules).
- Use with: smaller base lists (the expansion is huge).

### KoreLogic CMIYC rules

Rule sets published from the "Crack Me If You Can" DEF CON contests. Strong at
human password-construction patterns.

- Where: https://contest-data.korelogic.com/ and
  https://github.com/hashcat/hashcat/blob/master/rules/ (several `KoreLogic`
  and `T0XlC`-style rules are mirrored in community repos).
- Use with: leaked-password base lists for maximum realism.

### generated2.rule

A widely used machine-generated rule set (ships with hashcat as
`rules/generated2.rule`). Good middle ground between best64 and dive.

- Where: hashcat distribution `rules/` directory
  (https://github.com/hashcat/hashcat/tree/master/rules).
- Use with: medium base lists when best64 underperforms.

---

## Recommended starter set

For a first authorized run against your own funded test wallet, in priority
order:

1. `data/known_brainwallets.txt` (bundled): documented real brain wallets.
2. `data/crypto_culture.txt` (bundled): crypto memes, Satoshi quotes, coin
   names, famous weak phrases.
3. A famous-quotes dataset (Quotable / Wikiquote) as a sentence corpus.
4. RockYou (or SecLists `Passwords/`) **with OneRuleToRuleThemAll** applied.
5. A song-lyrics dataset as a sentence corpus (chorus lines).
6. Project Gutenberg opening lines / famous passages as a sentence corpus.

That ordering front-loads the candidates most likely to hit. Escalate to
Wikipedia sentence dumps, multilingual frequency lists, and COMB-scale dumps
only if the starter set comes up empty, and only within authorized scope.

---

## Quick reference table

| Resource                    | Type                   | Where                      | Feed as            |
| --------------------------- | ---------------------- | -------------------------- | ------------------ |
| Project Gutenberg           | Public-domain books    | gutenberg.org / pgcorpus   | Sentence corpus    |
| Song lyrics (Kaggle/Genius) | Lyrics                 | Kaggle, LyricsGenius       | Sentence corpus    |
| OpenSubtitles (OPUS)        | Movie/TV dialogue      | opus.nlpl.eu               | Sentence corpus    |
| Wikipedia dumps             | Encyclopedia text      | dumps.wikimedia.org        | Sentence corpus    |
| Famous quotes               | Curated quotes         | quotable, Wikiquote        | Wordlist/sentence  |
| RockYou                     | Leaked passwords       | Kali, naive-hashcat        | Wordlist + rules   |
| SecLists                    | Curated wordlists      | github danielmiessler      | Wordlist(s)        |
| COMB                        | Breach compilation     | breach forums (verify)     | Wordlist (sorted)  |
| HIBP Pwned Passwords        | Breached hashes        | haveibeenpwned.com         | Triage/filter only |
| FrequencyWords              | Multilingual frequency | github hermitdave          | Wordlist (ordered) |
| CC-CEDICT (pinyin)          | Chinese pinyin         | mdbg.net                   | Wordlist           |
| OneRuleToRuleThemAll        | Rule set               | github stealthsploit       | Rule file          |
| dive.rule                   | Rule set               | hashcat rules/             | Rule file          |
| KoreLogic CMIYC             | Rule set               | contest-data.korelogic.com | Rule file          |
| generated2.rule             | Rule set               | hashcat rules/             | Rule file          |
