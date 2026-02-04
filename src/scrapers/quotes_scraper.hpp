/**
 * Collider Quotes Scraper
 *
 * Extracts famous quotes for brain wallet wordlists.
 *
 * Sources:
 * - Wikiquote
 * - IMDB (movie quotes)
 * - Goodreads
 * - Local quote databases
 */

#pragma once

#include "../core/types.hpp"
#include <string>
#include <vector>
#include <functional>
#include <unordered_set>

namespace collider {
namespace scrapers {

/**
 * Represents a scraped quote.
 */
struct Quote {
    std::string text;
    std::string author;
    std::string source;     // Movie, book, speech, etc.
    std::string category;   // Philosophy, motivational, etc.
    float popularity_score; // Estimated popularity (0-1)
};

/**
 * Configuration for quote scraping.
 */
struct QuotesScraperConfig {
    // Filtering
    size_t min_length = 10;
    size_t max_length = 128;

    // Categories to include
    std::vector<std::string> categories = {
        "philosophy", "motivational", "technology",
        "science", "literature", "film"
    };

    // Normalization
    bool lowercase = true;
    bool remove_punctuation = true;
    bool generate_first_n_words = true;
    std::vector<size_t> word_counts = {3, 4, 5, 6};  // Generate N-word prefixes
};

/**
 * Quotes scraper for passphrase extraction.
 */
class QuotesScraper {
public:
    explicit QuotesScraper(const QuotesScraperConfig& config)
        : config_(config) {}

    /**
     * Scrape quotes from Wikiquote.
     *
     * @param category Category to scrape (e.g., "Philosophy")
     * @param callback Called for each extracted quote
     * @param max_quotes Maximum quotes to extract
     */
    void scrape_wikiquote(
        const std::string& category,
        std::function<void(Quote&&)> callback,
        size_t max_quotes = 10000
    );

    /**
     * Scrape movie quotes from IMDB.
     *
     * @param movie_id IMDB movie ID
     * @param callback Called for each quote
     */
    void scrape_imdb(
        const std::string& movie_id,
        std::function<void(Quote&&)> callback
    );

    /**
     * Scrape from local quote file.
     * Supports: JSON, CSV, plain text (one per line)
     *
     * @param path Path to quotes file
     * @param callback Called for each quote
     */
    void scrape_local(
        const std::string& path,
        std::function<void(Quote&&)> callback
    );

    /**
     * Process a quote and generate passphrase variants.
     *
     * @param quote The source quote
     * @param callback Called for each generated passphrase
     */
    void generate_variants(
        const Quote& quote,
        std::function<void(Candidate&&)> callback
    );

    /**
     * Normalize quote text.
     */
    std::string normalize(const std::string& text);

private:
    QuotesScraperConfig config_;

    /**
     * Extract first N words from a phrase.
     */
    std::string first_n_words(const std::string& text, size_t n);

    /**
     * Generate common abbreviations.
     * e.g., "I love you" -> "ily", "Be right back" -> "brb"
     */
    std::vector<std::string> generate_abbreviations(const std::string& text);
};

// Implementation
inline void QuotesScraper::generate_variants(
    const Quote& quote,
    std::function<void(Candidate&&)> callback
) {
    std::string normalized = normalize(quote.text);
    if (normalized.empty()) return;

    // Full quote (if within length limit)
    if (normalized.size() >= config_.min_length &&
        normalized.size() <= config_.max_length) {

        callback(Candidate{
            .phrase = normalized,
            .priority = quote.popularity_score,
            .source = CandidateSource::QUOTES,
            .rule_applied = ":"
        });

        // No-space variant
        std::string no_space = normalized;
        no_space.erase(std::remove(no_space.begin(), no_space.end(), ' '),
                      no_space.end());

        if (no_space.size() >= config_.min_length) {
            callback(Candidate{
                .phrase = no_space,
                .priority = quote.popularity_score * 0.9f,
                .source = CandidateSource::QUOTES,
                .rule_applied = ":"
            });
        }
    }

    // First N words variants
    if (config_.generate_first_n_words) {
        for (size_t n : config_.word_counts) {
            std::string prefix = first_n_words(normalized, n);
            if (prefix.size() >= config_.min_length) {
                callback(Candidate{
                    .phrase = prefix,
                    .priority = quote.popularity_score * 0.8f,
                    .source = CandidateSource::QUOTES,
                    .rule_applied = ":"
                });

                // No-space variant of prefix
                std::string no_space = prefix;
                no_space.erase(std::remove(no_space.begin(), no_space.end(), ' '),
                              no_space.end());

                if (no_space.size() >= config_.min_length) {
                    callback(Candidate{
                        .phrase = no_space,
                        .priority = quote.popularity_score * 0.7f,
                        .source = CandidateSource::QUOTES,
                        .rule_applied = ":"
                    });
                }
            }
        }
    }

    // Abbreviations
    auto abbrevs = generate_abbreviations(normalized);
    for (const auto& abbrev : abbrevs) {
        if (abbrev.size() >= 4) {  // Minimum 4 chars for abbreviations
            callback(Candidate{
                .phrase = abbrev,
                .priority = quote.popularity_score * 0.5f,
                .source = CandidateSource::QUOTES,
                .rule_applied = ":"
            });
        }
    }
}

inline std::string QuotesScraper::normalize(const std::string& text) {
    std::string result;
    result.reserve(text.size());

    bool last_was_space = true;

    for (char c : text) {
        if (config_.remove_punctuation && !std::isalnum(c) && c != ' ') {
            continue;
        }

        if (std::isspace(c)) {
            if (!last_was_space) {
                result += ' ';
                last_was_space = true;
            }
        } else {
            char out = config_.lowercase ? std::tolower(c) : c;
            result += out;
            last_was_space = false;
        }
    }

    // Trim trailing space
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }

    return result;
}

inline std::string QuotesScraper::first_n_words(const std::string& text, size_t n) {
    std::string result;
    size_t word_count = 0;
    size_t i = 0;

    while (i < text.size() && word_count < n) {
        // Skip leading spaces
        while (i < text.size() && text[i] == ' ') i++;

        if (i >= text.size()) break;

        // Read word
        size_t word_start = i;
        while (i < text.size() && text[i] != ' ') i++;

        if (word_count > 0) result += ' ';
        result += text.substr(word_start, i - word_start);
        word_count++;
    }

    return result;
}

inline std::vector<std::string> QuotesScraper::generate_abbreviations(const std::string& text) {
    std::vector<std::string> abbrevs;

    // First letter of each word
    std::string acronym;
    bool at_word_start = true;

    for (char c : text) {
        if (c == ' ') {
            at_word_start = true;
        } else if (at_word_start) {
            acronym += c;
            at_word_start = false;
        }
    }

    if (acronym.size() >= 3) {
        abbrevs.push_back(acronym);
    }

    return abbrevs;
}

inline void QuotesScraper::scrape_wikiquote(
    const std::string& /*category*/,
    std::function<void(Quote&&)> /*callback*/,
    size_t /*max_quotes*/
) {
    // Wikiquote scraping requires:
    // 1. HTTP client implementation (libcurl)
    // 2. HTML parsing (for MediaWiki markup)
    // 3. Rate limiting to respect server
    //
    // For production use, download Wikiquote dumps instead:
    // https://dumps.wikimedia.org/enwikiquote/
    std::cerr << "[QuotesScraper] Wikiquote scraping not implemented.\n";
    std::cerr << "[QuotesScraper] Use scrape_local() with pre-downloaded quotes instead.\n";
}

inline void QuotesScraper::scrape_imdb(
    const std::string& /*movie_id*/,
    std::function<void(Quote&&)> /*callback*/
) {
    // IMDB scraping requires:
    // 1. HTTP client implementation
    // 2. HTML parsing
    // 3. Compliance with IMDB terms of service
    //
    // Consider using IMDB datasets instead:
    // https://www.imdb.com/interfaces/
    std::cerr << "[QuotesScraper] IMDB scraping not implemented.\n";
    std::cerr << "[QuotesScraper] Use scrape_local() with pre-downloaded quotes instead.\n";
}

inline void QuotesScraper::scrape_local(
    const std::string& path,
    std::function<void(Quote&&)> callback
) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "[QuotesScraper] Cannot open: " << path << "\n";
        return;
    }

    std::string line;
    size_t line_number = 0;

    while (std::getline(file, line)) {
        line_number++;

        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Trim whitespace
        size_t start = line.find_first_not_of(" \t");
        size_t end = line.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        line = line.substr(start, end - start + 1);

        // Check length
        if (line.size() < config_.min_length || line.size() > config_.max_length) {
            continue;
        }

        // Create quote entry
        Quote quote{
            .text = line,
            .author = "",
            .source = path,
            .category = "",
            .popularity_score = 0.5f  // Default mid-range score
        };

        callback(std::move(quote));
    }

    std::cerr << "[QuotesScraper] Loaded " << line_number << " lines from " << path << "\n";
}

}  // namespace scrapers
}  // namespace collider
