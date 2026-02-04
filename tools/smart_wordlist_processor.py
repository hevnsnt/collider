#!/usr/bin/env python3
"""
Smart Wordlist Processor

Intelligently processes wordlist directories, detecting file types and
extracting appropriate content:

- Password lists: Use as-is (one password per line)
- Quote files: Use as-is
- Raw book text: Extract sentences and dialogue
- Lyrics: Extract individual lines

Usage:
    python smart_wordlist_processor.py <input_dirs...> <output_file>

Example:
    python smart_wordlist_processor.py D:\theCollider\data D:\output\combined.txt
"""

import os
import sys
import re
from pathlib import Path
from collections import Counter
import hashlib

# Import the literature extractor functions
# (In production, these would be in a shared module)

# === Configuration ===
MIN_LENGTH = 4
MAX_LENGTH = 150
MIN_WORDS = 1
MAX_WORDS = 25

# Directories that contain raw books (need sentence extraction)
RAW_TEXT_DIRS = {'literature', 'books', 'texts', 'ebooks', 'novels'}

# Directories that are already formatted properly
FORMATTED_DIRS = {'passwords', 'wordlists', 'quotes', 'names', 'crypto', 'gaming', 'slang'}

# Directories with lyrics (line-by-line but may need cleaning)
LYRICS_DIRS = {'lyrics', 'songs', 'music'}

# Skip these directories entirely
SKIP_DIRS = {'rules', 'output', 'processed', '__pycache__', '.git'}

# === Encoding fixes ===
ENCODING_FIXES = {
    'â€œ': '"', 'â€': '"', 'â€™': "'", 'â€˜': "'",
    'â€"': '—', 'â€"': '–', 'Â': '', '\x00': '',
}

def fix_encoding(text):
    for bad, good in ENCODING_FIXES.items():
        text = text.replace(bad, good)
    # Normalize quotes
    text = re.sub(r'[""„‟]', '"', text)
    text = re.sub(r"[''‚‛]", "'", text)
    text = re.sub(r'[—–]', '-', text)
    return text

# === Validation ===
BAD_PATTERNS = [
    r'^\d+$', r'^[^a-zA-Z]*$', r'^\s*$',
    r'^(chapter|part|book|section|volume)\s+[ivxlcdm\d]+',
    r'^\*+$', r'^_{3,}$', r'^-{3,}$', r'^={3,}$',
    r'www\.|http|\.com|\.org|\.net',
    r'@.*\.\w{2,4}$',
    r'^\[\d+\]', r'^table of contents',
    r'^copyright|^public domain|^project gutenberg|^the end$',
    r'â€|Ã|Â',  # Encoding garbage
    r'^\d{4,}$',  # Just long numbers
]

def is_valid_phrase(line):
    """Check if a line is a valid passphrase candidate."""
    line = line.strip()

    if len(line) < MIN_LENGTH or len(line) > MAX_LENGTH:
        return False

    words = line.split()
    if len(words) > MAX_WORDS:
        return False

    line_lower = line.lower()
    for pattern in BAD_PATTERNS:
        if re.search(pattern, line_lower, re.IGNORECASE):
            return False

    # Must have letters
    letters = sum(1 for c in line if c.isalpha())
    if letters < max(3, len(line) * 0.3):
        return False

    return True

# === File type detection ===
def detect_file_type(filepath, content_sample):
    """Detect if a file is raw book text or formatted wordlist."""
    filename = filepath.name.lower()
    parent_dir = filepath.parent.name.lower()

    # Check directory name first
    if parent_dir in RAW_TEXT_DIRS:
        return 'book'
    if parent_dir in LYRICS_DIRS:
        return 'lyrics'
    if parent_dir in FORMATTED_DIRS:
        return 'wordlist'

    # Analyze content
    lines = content_sample.split('\n')[:100]
    avg_len = sum(len(l) for l in lines) / max(len(lines), 1)

    # Long average line length = probably raw text
    if avg_len > 80:
        return 'book'

    # Short lines = probably wordlist
    if avg_len < 30:
        return 'wordlist'

    # Check for paragraph-like content
    long_lines = sum(1 for l in lines if len(l) > 100)
    if long_lines > len(lines) * 0.3:
        return 'book'

    return 'wordlist'

# === Extractors ===
def extract_from_book(content):
    """Extract sentences and dialogue from raw book text."""
    phrases = set()
    content = fix_encoding(content)

    # Split into sentences
    content = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.\s+', r'\1<P> ', content)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"])|(?<=[.!?])$', content)

    for sentence in sentences:
        sentence = sentence.replace('<P>', '.').strip()
        if is_valid_phrase(sentence):
            phrases.add(sentence)

    # Extract dialogue
    for pattern in [r'"([^"]{8,100})"', r"'([^']{8,100})'", r'"([^"]{8,100})"']:
        for match in re.findall(pattern, content):
            match = match.strip()
            if is_valid_phrase(match):
                phrases.add(match)

    return phrases

def extract_from_lyrics(content):
    """Extract lines from lyrics files."""
    phrases = set()
    content = fix_encoding(content)

    for line in content.split('\n'):
        line = line.strip()
        # Skip section markers like [Verse 1], [Chorus]
        if re.match(r'^\[.*\]$', line):
            continue
        if is_valid_phrase(line):
            phrases.add(line)

    return phrases

def extract_from_wordlist(content):
    """Extract from pre-formatted wordlist."""
    phrases = set()
    content = fix_encoding(content)

    for line in content.split('\n'):
        line = line.strip()
        if is_valid_phrase(line):
            phrases.add(line)

    return phrases

# === Main processor ===
def process_file(filepath):
    """Process a single file and extract phrases."""
    try:
        # Try UTF-8, fall back to latin-1
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()

        # Detect file type
        file_type = detect_file_type(filepath, content[:5000])

        # Extract based on type
        if file_type == 'book':
            return extract_from_book(content), file_type
        elif file_type == 'lyrics':
            return extract_from_lyrics(content), file_type
        else:
            return extract_from_wordlist(content), file_type

    except Exception as e:
        print(f"  Warning: {filepath}: {e}", file=sys.stderr)
        return set(), 'error'

def process_directory(dir_path, all_phrases, stats):
    """Recursively process a directory."""
    dir_path = Path(dir_path)

    if not dir_path.exists():
        print(f"Warning: Directory not found: {dir_path}")
        return

    for item in sorted(dir_path.iterdir()):
        if item.is_dir():
            if item.name.lower() in SKIP_DIRS:
                print(f"  Skipping: {item.name}/")
                continue
            process_directory(item, all_phrases, stats)
        elif item.suffix.lower() in ('.txt', '.lst', '.dic', '.wordlist'):
            print(f"  {item.relative_to(dir_path.parent)}...", end=' ', flush=True)
            phrases, file_type = process_file(item)
            print(f"{len(phrases):,} ({file_type})")

            all_phrases.update(phrases)
            stats['files'] += 1
            stats['by_type'][file_type] = stats['by_type'].get(file_type, 0) + 1

def main():
    if len(sys.argv) < 3:
        print("Usage: python smart_wordlist_processor.py <input_dirs...> <output_file>")
        print("Example: python smart_wordlist_processor.py D:\\theCollider\\data combined.txt")
        sys.exit(1)

    input_dirs = sys.argv[1:-1]
    output_file = Path(sys.argv[-1])

    print("=" * 60)
    print("SMART WORDLIST PROCESSOR")
    print("=" * 60)
    print()

    all_phrases = set()
    stats = {'files': 0, 'by_type': {}}

    for input_dir in input_dirs:
        print(f"[*] Processing: {input_dir}")
        process_directory(input_dir, all_phrases, stats)
        print()

    print(f"[*] Total unique phrases: {len(all_phrases):,}")
    print(f"[*] Files processed: {stats['files']}")
    print(f"[*] By type: {stats['by_type']}")

    # Write output
    print(f"\n[*] Writing to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for phrase in sorted(all_phrases):
            f.write(phrase + '\n')

    # Final stats
    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Output: {output_file}")
    print(f"  Total phrases: {len(all_phrases):,}")

    # Sample
    sample = list(all_phrases)[:10]
    print("\nSample phrases:")
    for p in sample:
        print(f"  {p[:60]}{'...' if len(p) > 60 else ''}")

if __name__ == '__main__':
    main()
