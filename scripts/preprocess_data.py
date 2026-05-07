#!/usr/bin/env python3
"""
Superflayer Data Preprocessing Script

Robust preprocessing for brain wallet passphrase generation training data.
Handles .txt, .json, .csv files with various formats and structures.

Features:
- Multi-format support (TXT, JSON, CSV)
- Intelligent content extraction from nested structures
- Project Gutenberg header/footer removal
- Comment stripping (# lines)
- Line ending normalization (Unix \n)
- Encoding normalization (UTF-8)
- Duplicate removal (per-file and global)
- Empty line removal
- Whitespace normalization
- Progress reporting and statistics

Output Structure:
    processed/
    ├── passwords.txt      # Combined password corpus
    ├── phrases.txt        # Lyrics, quotes, literature phrases
    ├── wordlists.txt      # Combined wordlists
    ├── names.txt          # Usernames, names, entities
    ├── crypto.txt         # BIP39 and crypto terms
    └── rules/             # Hashcat rules (copied)
"""

import os
import sys
import json
import csv
import re
import argparse
import logging
from pathlib import Path
from typing import Set, List, Optional, Dict, Any, Iterator
from collections import defaultdict
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Main preprocessing engine for superflayer training data."""

    # Project Gutenberg markers
    GUTENBERG_START_MARKERS = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
        "***START OF THE PROJECT GUTENBERG",
    ]

    GUTENBERG_END_MARKERS = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
    ]

    # JSON keys that typically contain useful data
    JSON_VALUE_KEYS = [
        'name', 'title', 'word', 'phrase', 'text', 'content',
        'line', 'lyric', 'lyrics', 'quote', 'value', 'label',
        'identifier', 'username', 'first_name', 'last_name',
        'full_name', 'artist', 'song', 'album'
    ]

    # JSON array keys to extract from
    JSON_ARRAY_KEYS = [
        'adjs', 'adjectives', 'animals', 'celebrities', 'words',
        'names', 'nouns', 'verbs', 'elements', 'colors', 'fruits',
        'vegetables', 'countries', 'cities', 'rivers', 'planets',
        'drugs', 'moves', 'pokemon', 'gods', 'titans', 'archetypes',
        'languages', 'industries', 'companies', 'characters',
        'prepositions', 'pronouns', 'stopwords', 'data', 'items',
        'entries', 'results', 'list', 'values'
    ]

    # CSV columns to extract
    CSV_EXTRACT_COLUMNS = [
        'name', 'title', 'artist', 'artists', 'song', 'song_name',
        'line', 'lyric', 'lyrics', 'text', 'content', 'phrase',
        'word', 'username', 'first_name', 'last_name', 'full_name'
    ]

    def __init__(self, data_dir: str, output_dir: str,
                 min_length: int = 1, max_length: int = 500,
                 dedupe_global: bool = True):
        """
        Initialize preprocessor.

        Args:
            data_dir: Source data directory
            output_dir: Output directory for processed files
            min_length: Minimum line length to keep
            max_length: Maximum line length to keep
            dedupe_global: Whether to deduplicate across all files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.max_length = max_length
        self.dedupe_global = dedupe_global

        # Global deduplication sets
        self.seen_passwords: Set[str] = set()
        self.seen_phrases: Set[str] = set()
        self.seen_words: Set[str] = set()
        self.seen_names: Set[str] = set()
        self.seen_crypto: Set[str] = set()

        # Statistics
        self.stats = defaultdict(lambda: defaultdict(int))

    def normalize_line(self, line: str) -> Optional[str]:
        """
        Normalize a single line of text.

        - Strip whitespace
        - Normalize Unicode
        - Remove control characters
        - Check length constraints
        """
        if not line:
            return None

        # Strip whitespace
        line = line.strip()

        if not line:
            return None

        # Skip comment lines
        if line.startswith('#'):
            return None

        # Normalize Unicode (NFC form)
        try:
            line = unicodedata.normalize('NFC', line)
        except Exception:
            pass

        # Remove control characters except newline/tab
        line = ''.join(c for c in line if c == '\t' or c == '\n' or
                      (ord(c) >= 32 and ord(c) != 127))
        line = line.strip()

        if not line:
            return None

        # Length check
        if len(line) < self.min_length or len(line) > self.max_length:
            return None

        return line

    def strip_gutenberg_headers(self, lines: List[str]) -> List[str]:
        """Remove Project Gutenberg headers and footers."""
        start_idx = 0
        end_idx = len(lines)

        # Find start marker
        for i, line in enumerate(lines):
            for marker in self.GUTENBERG_START_MARKERS:
                if marker.lower() in line.lower():
                    start_idx = i + 1
                    break
            if start_idx > 0:
                break

        # Find end marker
        for i in range(len(lines) - 1, start_idx, -1):
            for marker in self.GUTENBERG_END_MARKERS:
                if marker.lower() in lines[i].lower():
                    end_idx = i
                    break
            if end_idx < len(lines):
                break

        return lines[start_idx:end_idx]

    def extract_from_json(self, data: Any, depth: int = 0) -> Iterator[str]:
        """
        Recursively extract string values from JSON data.

        Handles nested structures, arrays, and objects.
        """
        if depth > 10:  # Prevent infinite recursion
            return

        if isinstance(data, str):
            yield data

        elif isinstance(data, (int, float)):
            yield str(data)

        elif isinstance(data, list):
            for item in data:
                yield from self.extract_from_json(item, depth + 1)

        elif isinstance(data, dict):
            # First, try to extract from known keys
            for key in self.JSON_VALUE_KEYS:
                if key in data:
                    value = data[key]
                    if isinstance(value, str):
                        yield value
                    elif isinstance(value, (int, float)):
                        yield str(value)
                    else:
                        yield from self.extract_from_json(value, depth + 1)

            # Then check array keys
            for key in self.JSON_ARRAY_KEYS:
                if key in data:
                    yield from self.extract_from_json(data[key], depth + 1)

            # If no known keys found, extract all string values
            found_known = any(k in data for k in self.JSON_VALUE_KEYS + self.JSON_ARRAY_KEYS)
            if not found_known:
                for key, value in data.items():
                    if key not in ('description', 'source', 'url', 'link', 'id',
                                   'metadata', 'schema', 'version'):
                        yield from self.extract_from_json(value, depth + 1)

    def process_json_file(self, filepath: Path) -> List[str]:
        """Process a JSON file and extract all text values."""
        results = []

        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Handle JSON Lines format
            if filepath.suffix == '.jsonl':
                for line in content.strip().split('\n'):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            for value in self.extract_from_json(data):
                                normalized = self.normalize_line(value)
                                if normalized:
                                    results.append(normalized)
                        except json.JSONDecodeError:
                            continue
            else:
                # Regular JSON
                data = json.loads(content)
                for value in self.extract_from_json(data):
                    normalized = self.normalize_line(value)
                    if normalized:
                        results.append(normalized)

        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error in {filepath}: {e}")
        except Exception as e:
            logger.error(f"Error processing JSON {filepath}: {e}")

        return results

    def process_csv_file(self, filepath: Path) -> List[str]:
        """Process a CSV file and extract relevant columns."""
        results = []

        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                # Detect delimiter
                sample = f.read(4096)
                f.seek(0)

                try:
                    dialect = csv.Sniffer().sniff(sample)
                except csv.Error:
                    dialect = csv.excel

                reader = csv.DictReader(f, dialect=dialect)

                if not reader.fieldnames:
                    return results

                # Find columns to extract
                extract_cols = []
                for col in reader.fieldnames:
                    col_lower = col.lower().strip()
                    if any(target in col_lower for target in self.CSV_EXTRACT_COLUMNS):
                        extract_cols.append(col)

                # If no specific columns found, try first few columns
                if not extract_cols and reader.fieldnames:
                    extract_cols = list(reader.fieldnames)[:3]

                for row in reader:
                    for col in extract_cols:
                        if col in row and row[col]:
                            value = row[col]
                            # Handle list-like strings (e.g., "['Artist1', 'Artist2']")
                            if value.startswith('[') and value.endswith(']'):
                                try:
                                    # Parse as Python literal
                                    import ast
                                    items = ast.literal_eval(value)
                                    if isinstance(items, list):
                                        for item in items:
                                            normalized = self.normalize_line(str(item))
                                            if normalized:
                                                results.append(normalized)
                                        continue
                                except:
                                    pass

                            normalized = self.normalize_line(value)
                            if normalized:
                                results.append(normalized)

        except Exception as e:
            logger.error(f"Error processing CSV {filepath}: {e}")

        return results

    def process_txt_file(self, filepath: Path, is_literature: bool = False) -> List[str]:
        """Process a plain text file."""
        results = []

        try:
            # Try UTF-8 first, then fallback encodings
            content = None
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(filepath, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue

            if content is None:
                logger.warning(f"Could not decode {filepath}")
                return results

            # Normalize line endings
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            lines = content.split('\n')

            # Strip Gutenberg headers for literature files
            if is_literature:
                lines = self.strip_gutenberg_headers(lines)

            for line in lines:
                normalized = self.normalize_line(line)
                if normalized:
                    results.append(normalized)

        except Exception as e:
            logger.error(f"Error processing TXT {filepath}: {e}")

        return results

    def process_file(self, filepath: Path, category: str) -> List[str]:
        """Process a single file based on its extension."""
        suffix = filepath.suffix.lower()
        is_literature = category == 'literature'

        if suffix == '.json':
            return self.process_json_file(filepath)
        elif suffix == '.csv':
            return self.process_csv_file(filepath)
        elif suffix in ['.txt', '.text', '.lst', '.dic']:
            return self.process_txt_file(filepath, is_literature)
        elif suffix == '.rule':
            # Copy rules as-is
            return self.process_txt_file(filepath)
        else:
            logger.warning(f"Unknown file type: {filepath}")
            return []

    def deduplicate(self, items: List[str], seen_set: Set[str]) -> List[str]:
        """Remove duplicates using provided set."""
        results = []
        for item in items:
            # Use lowercase for deduplication but keep original case
            key = item.lower()
            if key not in seen_set:
                seen_set.add(key)
                results.append(item)
        return results

    def process_category(self, category: str, seen_set: Set[str]) -> List[str]:
        """Process all files in a category directory."""
        category_dir = self.data_dir / category
        if not category_dir.exists():
            logger.warning(f"Category directory not found: {category_dir}")
            return []

        all_items = []

        for filepath in sorted(category_dir.iterdir()):
            if filepath.is_file() and not filepath.name.startswith('.'):
                logger.info(f"Processing: {filepath.name}")

                items = self.process_file(filepath, category)

                # Track stats
                self.stats[category][filepath.name] = len(items)

                if self.dedupe_global:
                    items = self.deduplicate(items, seen_set)

                all_items.extend(items)

        # Final deduplication within category if not doing global
        if not self.dedupe_global:
            local_seen: Set[str] = set()
            all_items = self.deduplicate(all_items, local_seen)

        return all_items

    def run(self) -> Dict[str, int]:
        """Run the full preprocessing pipeline."""
        logger.info(f"Starting preprocessing...")
        logger.info(f"Source: {self.data_dir}")
        logger.info(f"Output: {self.output_dir}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # Process passwords
        logger.info("\n=== Processing Passwords ===")
        passwords = self.process_category('passwords', self.seen_passwords)
        if passwords:
            output_path = self.output_dir / 'passwords.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(passwords))
            results['passwords'] = len(passwords)
            logger.info(f"Wrote {len(passwords):,} passwords")

        # Process phrases (lyrics + quotes + literature)
        logger.info("\n=== Processing Phrases ===")
        phrases = []
        for category in ['lyrics', 'quotes', 'literature']:
            items = self.process_category(category, self.seen_phrases)
            phrases.extend(items)

        if phrases:
            output_path = self.output_dir / 'phrases.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(phrases))
            results['phrases'] = len(phrases)
            logger.info(f"Wrote {len(phrases):,} phrases")

        # Process wordlists
        logger.info("\n=== Processing Wordlists ===")
        words = self.process_category('wordlists', self.seen_words)
        if words:
            output_path = self.output_dir / 'wordlists.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(words))
            results['wordlists'] = len(words)
            logger.info(f"Wrote {len(words):,} words")

        # Process names
        logger.info("\n=== Processing Names ===")
        names = self.process_category('names', self.seen_names)
        if names:
            output_path = self.output_dir / 'names.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(names))
            results['names'] = len(names)
            logger.info(f"Wrote {len(names):,} names")

        # Process crypto
        logger.info("\n=== Processing Crypto ===")
        crypto = self.process_category('crypto', self.seen_crypto)
        if crypto:
            output_path = self.output_dir / 'crypto.txt'
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(crypto))
            results['crypto'] = len(crypto)
            logger.info(f"Wrote {len(crypto):,} crypto terms")

        # Copy rules
        logger.info("\n=== Copying Rules ===")
        rules_dir = self.data_dir / 'rules'
        if rules_dir.exists():
            output_rules = self.output_dir / 'rules'
            output_rules.mkdir(exist_ok=True)
            rule_count = 0
            for rulefile in rules_dir.glob('*.rule'):
                with open(rulefile, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                with open(output_rules / rulefile.name, 'w', encoding='utf-8') as f:
                    f.write(content)
                rule_count += 1
            results['rules'] = rule_count
            logger.info(f"Copied {rule_count} rule files")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("PREPROCESSING COMPLETE")
        logger.info("=" * 60)

        total = sum(results.values())
        for category, count in results.items():
            logger.info(f"  {category:15} {count:>12,}")
        logger.info("-" * 60)
        logger.info(f"  {'TOTAL':15} {total:>12,}")

        # Write stats file
        stats_path = self.output_dir / 'stats.json'
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': results,
                'per_file': dict(self.stats)
            }, f, indent=2)
        logger.info(f"\nStats written to: {stats_path}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess superflayer training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--data-dir', '-d',
        default='data',
        help='Source data directory (default: data)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='processed',
        help='Output directory (default: processed)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=1,
        help='Minimum line length (default: 1)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=500,
        help='Maximum line length (default: 500)'
    )
    parser.add_argument(
        '--no-global-dedupe',
        action='store_true',
        help='Disable global deduplication across files'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    preprocessor = DataPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        dedupe_global=not args.no_global_dedupe
    )

    try:
        results = preprocessor.run()
        sys.exit(0 if results else 1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
