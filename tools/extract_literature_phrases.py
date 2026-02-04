#!/usr/bin/env python3
"""
Smart Literature Phrase Extractor

Extracts passphrase-worthy lines from raw book text files.
Designed for brain wallet scanning - finds memorable quotes and phrases.

Usage:
    python extract_literature_phrases.py <input_dir> <output_file>

Example:
    python extract_literature_phrases.py D:\theCollider\data\literature D:\theCollider\data\literature_phrases.txt
"""

import os
import sys
import re
from pathlib import Path
from collections import Counter

# Configuration
MIN_LENGTH = 8          # Minimum characters
MAX_LENGTH = 120        # Maximum characters (reasonable passphrase)
MIN_WORDS = 2           # At least 2 words
MAX_WORDS = 20          # Not too many words

# Patterns that indicate a good quote/phrase
GOOD_PATTERNS = [
    r'^[A-Z][^.!?]*[.!?]$',           # Complete sentence
    r'^"[^"]+[.!?]?"$',                # Quoted speech
    r"^'[^']+[.!?]?'$",                # Single-quoted speech
    r'^[A-Z][a-z]+ said',              # Dialogue attribution
    r'^\w+ \w+ \w+$',                  # Simple 3-word phrase
]

# Patterns that indicate junk (skip these)
BAD_PATTERNS = [
    r'^\d+$',                          # Just numbers
    r'^[^a-zA-Z]*$',                   # No letters
    r'^\s*$',                          # Empty/whitespace
    r'^(chapter|part|book|section)\s+\d+',  # Chapter headers (case insensitive)
    r'^\*+',                           # Asterisk lines
    r'^_{3,}',                         # Underscores
    r'^-{3,}',                         # Dashes
    r'www\.|http|\.com|\.org',         # URLs
    r'@.*\.(com|org|net)',             # Emails
    r'^\[\d+\]',                       # Reference numbers
    r'^table of contents',             # TOC
    r'^copyright|^public domain|^project gutenberg',  # Legal
    r'â€|Ã|Â',                         # Encoding garbage
]

def fix_encoding(text):
    """Fix common UTF-8 mojibake issues."""
    replacements = {
        'â€œ': '"',
        'â€': '"',
        'â€™': "'",
        'â€˜': "'",
        'â€"': '—',
        'â€"': '–',
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ã ': 'à',
        'Ã¢': 'â',
        'Ã®': 'î',
        'Ã´': 'ô',
        'Ã»': 'û',
        'Ã§': 'ç',
        'Ã«': 'ë',
        'Ã¯': 'ï',
        'Ã¼': 'ü',
        'Ã¶': 'ö',
        'Ã¤': 'ä',
        'Ã±': 'ñ',
        'Â': '',
        '\x00': '',
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

def normalize_quotes(text):
    """Normalize various quote styles to simple ASCII."""
    text = re.sub(r'[""„‟]', '"', text)
    text = re.sub(r"[''‚‛]", "'", text)
    text = re.sub(r'[—–]', '-', text)
    text = re.sub(r'…', '...', text)
    return text

def is_valid_phrase(line):
    """Check if a line is a valid passphrase candidate."""
    # Length check
    if len(line) < MIN_LENGTH or len(line) > MAX_LENGTH:
        return False

    # Word count check
    words = line.split()
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False

    # Bad pattern check
    line_lower = line.lower()
    for pattern in BAD_PATTERNS:
        if re.search(pattern, line_lower, re.IGNORECASE):
            return False

    # Must have reasonable letter ratio
    letters = sum(1 for c in line if c.isalpha())
    if letters < len(line) * 0.5:
        return False

    # Must have at least one vowel (real words)
    if not re.search(r'[aeiouAEIOU]', line):
        return False

    return True

def extract_sentences(text):
    """Split text into individual sentences."""
    # Fix encoding first
    text = fix_encoding(text)
    text = normalize_quotes(text)

    # Split on sentence boundaries
    # Handle Mr. Mrs. Dr. etc. to avoid false splits
    text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.\s+', r'\1<PERIOD> ', text)

    # Split on .!? followed by space and capital letter or end
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$', text)

    # Restore periods
    sentences = [s.replace('<PERIOD>', '.') for s in sentences]

    return sentences

def extract_dialogue(text):
    """Extract quoted dialogue/speech."""
    # Match various quote patterns
    patterns = [
        r'"([^"]{10,100})"',      # Double quotes
        r"'([^']{10,100})'",      # Single quotes
        r'"([^"]{10,100})"',      # Smart quotes
        r''([^']{10,100})'',      # Smart single quotes
    ]

    dialogue = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        dialogue.extend(matches)

    return dialogue

def score_phrase(phrase):
    """Score a phrase by how likely it is to be used as a passphrase."""
    score = 0

    # Bonus for common/memorable patterns
    if re.match(r'^[A-Z][a-z]', phrase):  # Starts with capital
        score += 1
    if phrase.endswith(('.', '!', '?')):  # Complete sentence
        score += 1
    if len(phrase.split()) >= 3 and len(phrase.split()) <= 8:  # Good word count
        score += 2
    if re.search(r'\b(I|you|we|love|life|death|god|heaven|hell|truth|beauty)\b', phrase, re.I):
        score += 1  # Philosophical/emotional words
    if re.search(r'\b(be|not|or|to|is|was|are|have|has|will|shall)\b', phrase, re.I):
        score += 1  # Common verbs = readable

    # Penalties
    if re.search(r'\d{4,}', phrase):  # Long numbers
        score -= 2
    if phrase.count(',') > 3:  # Too many commas = run-on
        score -= 1
    if len(set(phrase.lower().split())) < len(phrase.split()) * 0.7:  # Too repetitive
        score -= 1

    return score

def process_file(filepath):
    """Process a single literature file and extract phrases."""
    phrases = set()

    try:
        # Try UTF-8 first, fall back to latin-1
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()

        # Extract sentences
        sentences = extract_sentences(content)
        for sentence in sentences:
            sentence = sentence.strip()
            if is_valid_phrase(sentence):
                phrases.add(sentence)

        # Extract dialogue separately
        dialogues = extract_dialogue(content)
        for dialogue in dialogues:
            dialogue = dialogue.strip()
            if is_valid_phrase(dialogue):
                phrases.add(dialogue)

        # Also process line by line for already-formatted files
        for line in content.split('\n'):
            line = fix_encoding(line.strip())
            line = normalize_quotes(line)
            if is_valid_phrase(line):
                phrases.add(line)

    except Exception as e:
        print(f"  Warning: Error processing {filepath}: {e}", file=sys.stderr)

    return phrases

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_literature_phrases.py <input_dir> <output_file>")
        print("Example: python extract_literature_phrases.py D:\\theCollider\\data\\literature literature_phrases.txt")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        sys.exit(1)

    print(f"[*] Extracting phrases from: {input_dir}")
    print(f"[*] Output file: {output_file}")
    print()

    all_phrases = set()
    file_count = 0

    # Process all text files
    for filepath in sorted(input_dir.glob('*.txt')):
        print(f"  Processing: {filepath.name}...", end=' ', flush=True)
        phrases = process_file(filepath)
        print(f"{len(phrases):,} phrases")
        all_phrases.update(phrases)
        file_count += 1

    print()
    print(f"[*] Processed {file_count} files")
    print(f"[*] Total unique phrases: {len(all_phrases):,}")

    # Score and sort phrases
    print("[*] Scoring phrases...")
    scored = [(score_phrase(p), p) for p in all_phrases]
    scored.sort(key=lambda x: (-x[0], x[1]))  # Best scores first, then alphabetical

    # Write output
    print(f"[*] Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for score, phrase in scored:
            f.write(phrase + '\n')

    # Stats
    lengths = [len(p) for _, p in scored]
    word_counts = [len(p.split()) for _, p in scored]

    print()
    print("=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Total phrases: {len(scored):,}")
    print(f"  Avg length: {sum(lengths)/len(lengths):.1f} chars")
    print(f"  Avg words: {sum(word_counts)/len(word_counts):.1f}")
    print(f"  Length range: {min(lengths)} - {max(lengths)} chars")
    print()
    print("Top 10 highest-scored phrases:")
    for score, phrase in scored[:10]:
        print(f"  [{score}] {phrase[:70]}{'...' if len(phrase) > 70 else ''}")
    print()
    print(f"Output: {output_file}")

if __name__ == '__main__':
    main()
