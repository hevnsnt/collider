/**
 * Collider Lyrics Scraper
 *
 * Extracts memorable phrases from song lyrics for brain wallet wordlists.
 *
 * Sources:
 * - Genius.com (API)
 * - AZLyrics.com (HTML scraping)
 * - Offline lyrics databases
 */

#pragma once

#include "../core/types.hpp"
#include <string>
#include <vector>
#include <functional>
#include <regex>

namespace collider {
namespace scrapers {

/**
 * Configuration for lyrics scraping.
 */
struct LyricsScraperConfig {
    // API keys
    std::string genius_api_token;

    // Filtering
    size_t min_line_length = 8;
    size_t max_line_length = 64;
    bool include_first_lines = true;
    bool include_chorus = true;
    bool include_all_lines = true;

    // Output
    bool remove_punctuation = true;
    bool lowercase = true;
    bool generate_no_space_variant = true;
};

/**
 * Represents a scraped lyric phrase.
 */
struct LyricPhrase {
    std::string text;
    std::string song_title;
    std::string artist;
    std::string source;  // "genius", "azlyrics", "local"
    float memorability_score;  // Estimated memorability (0-1)
};

/**
 * Lyrics scraper for passphrase extraction.
 */
class LyricsScraper {
public:
    explicit LyricsScraper(const LyricsScraperConfig& config)
        : config_(config) {}

    /**
     * Scrape lyrics from Genius API.
     *
     * @param query Search query (artist, song, etc.)
     * @param callback Called for each extracted phrase
     * @param max_songs Maximum songs to process
     */
    void scrape_genius(
        const std::string& query,
        std::function<void(LyricPhrase&&)> callback,
        size_t max_songs = 1000
    );

    /**
     * Scrape lyrics from local file.
     * Supports multiple formats: plain text, JSON, LRC.
     *
     * @param path Path to lyrics file or directory
     * @param callback Called for each extracted phrase
     */
    void scrape_local(
        const std::string& path,
        std::function<void(LyricPhrase&&)> callback
    );

    /**
     * Extract memorable phrases from raw lyrics text.
     *
     * @param lyrics Full lyrics text
     * @param metadata Song metadata
     * @return Vector of extracted phrases
     */
    std::vector<LyricPhrase> extract_phrases(
        const std::string& lyrics,
        const std::string& song_title = "",
        const std::string& artist = ""
    );

    /**
     * Normalize a phrase for consistency.
     */
    std::string normalize(const std::string& phrase);

    /**
     * Estimate memorability of a phrase.
     * Based on: length, common words, repetition, structure.
     */
    float estimate_memorability(const std::string& phrase);

private:
    LyricsScraperConfig config_;

    /**
     * Detect chorus by finding repeated sections.
     */
    std::vector<std::string> detect_chorus(const std::string& lyrics);

    /**
     * Split lyrics into verses.
     */
    std::vector<std::string> split_verses(const std::string& lyrics);

    /**
     * HTTP request helper.
     */
    std::string http_get(const std::string& url);
};

// Implementation
inline std::vector<LyricPhrase> LyricsScraper::extract_phrases(
    const std::string& lyrics,
    const std::string& song_title,
    const std::string& artist
) {
    std::vector<LyricPhrase> phrases;

    // Split into lines
    std::istringstream stream(lyrics);
    std::string line;
    int line_number = 0;
    std::string prev_line;

    while (std::getline(stream, line)) {
        line_number++;

        // Skip empty lines
        if (line.empty()) continue;

        // Skip metadata lines (e.g., "[Verse 1]", "[Chorus]")
        if (line[0] == '[' && line.back() == ']') continue;

        // Normalize
        std::string normalized = normalize(line);
        if (normalized.empty()) continue;

        // Check length
        if (normalized.size() < config_.min_line_length ||
            normalized.size() > config_.max_line_length) {
            continue;
        }

        // Calculate memorability
        float score = estimate_memorability(normalized);

        // Boost first lines of verses
        if (config_.include_first_lines && prev_line.empty()) {
            score *= 1.2f;
        }

        // Add phrase
        LyricPhrase phrase{
            .text = normalized,
            .song_title = song_title,
            .artist = artist,
            .source = "local",
            .memorability_score = score
        };
        phrases.push_back(phrase);

        // Also add no-space variant if configured
        if (config_.generate_no_space_variant) {
            std::string no_space = normalized;
            no_space.erase(std::remove(no_space.begin(), no_space.end(), ' '),
                          no_space.end());

            if (no_space.size() >= config_.min_line_length) {
                phrase.text = no_space;
                phrase.memorability_score = score * 0.8f;  // Slightly lower
                phrases.push_back(phrase);
            }
        }

        prev_line = line;
    }

    // Detect and extract chorus
    if (config_.include_chorus) {
        auto chorus_lines = detect_chorus(lyrics);
        for (const auto& cl : chorus_lines) {
            std::string normalized = normalize(cl);
            if (normalized.size() >= config_.min_line_length &&
                normalized.size() <= config_.max_line_length) {

                LyricPhrase phrase{
                    .text = normalized,
                    .song_title = song_title,
                    .artist = artist,
                    .source = "local",
                    .memorability_score = estimate_memorability(normalized) * 1.5f  // Boost chorus
                };
                phrases.push_back(phrase);
            }
        }
    }

    return phrases;
}

inline std::string LyricsScraper::normalize(const std::string& phrase) {
    std::string result;
    result.reserve(phrase.size());

    bool last_was_space = true;

    for (char c : phrase) {
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

inline float LyricsScraper::estimate_memorability(const std::string& phrase) {
    float score = 0.5f;

    // Length score: prefer 15-40 characters
    if (phrase.size() >= 15 && phrase.size() <= 40) {
        score += 0.1f;
    }

    // Word count: prefer 3-8 words
    size_t word_count = std::count(phrase.begin(), phrase.end(), ' ') + 1;
    if (word_count >= 3 && word_count <= 8) {
        score += 0.1f;
    }

    // Common memorable patterns
    static const std::vector<std::string> memorable_patterns = {
        "love", "heart", "dream", "night", "forever",
        "never", "always", "believe", "world", "life"
    };

    for (const auto& pattern : memorable_patterns) {
        if (phrase.find(pattern) != std::string::npos) {
            score += 0.05f;
        }
    }

    // Penalize too short or too long
    if (phrase.size() < 10) score -= 0.2f;
    if (phrase.size() > 50) score -= 0.1f;

    // Clamp to [0, 1]
    return std::max(0.0f, std::min(1.0f, score));
}

inline std::vector<std::string> LyricsScraper::detect_chorus(const std::string& lyrics) {
    std::vector<std::string> chorus_lines;

    // Simple heuristic: find lines that appear multiple times
    std::unordered_map<std::string, int> line_counts;

    std::istringstream stream(lyrics);
    std::string line;

    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '[') continue;
        std::string normalized = normalize(line);
        if (!normalized.empty()) {
            line_counts[normalized]++;
        }
    }

    // Lines appearing 2+ times are likely chorus
    for (const auto& [text, count] : line_counts) {
        if (count >= 2) {
            chorus_lines.push_back(text);
        }
    }

    return chorus_lines;
}

inline std::vector<std::string> LyricsScraper::split_verses(const std::string& lyrics) {
    std::vector<std::string> verses;
    std::string current_verse;
    std::istringstream stream(lyrics);
    std::string line;

    while (std::getline(stream, line)) {
        if (line.empty()) {
            // Empty line marks verse boundary
            if (!current_verse.empty()) {
                verses.push_back(current_verse);
                current_verse.clear();
            }
        } else {
            if (!current_verse.empty()) current_verse += "\n";
            current_verse += line;
        }
    }

    // Don't forget the last verse
    if (!current_verse.empty()) {
        verses.push_back(current_verse);
    }

    return verses;
}

inline void LyricsScraper::scrape_genius(
    const std::string& /*query*/,
    std::function<void(LyricPhrase&&)> /*callback*/,
    size_t /*max_songs*/
) {
    // Genius API scraping requires:
    // 1. Valid API token in config_.genius_api_token
    // 2. HTTP client implementation
    // 3. JSON parsing for API responses
    // 4. HTML parsing for lyrics extraction
    //
    // For production use, implement with libcurl and a JSON library.
    // For now, use scrape_local() with pre-downloaded lyrics databases.
    std::cerr << "[LyricsScraper] Genius API scraping not implemented.\n";
    std::cerr << "[LyricsScraper] Use scrape_local() with lyrics files instead.\n";
}

inline void LyricsScraper::scrape_local(
    const std::string& path,
    std::function<void(LyricPhrase&&)> callback
) {
    // Check if path is a file or directory
    std::ifstream file(path);
    if (!file) {
        std::cerr << "[LyricsScraper] Cannot open: " << path << "\n";
        return;
    }

    // Read entire file
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    // Try to detect format and extract
    if (path.find(".json") != std::string::npos) {
        // JSON format: would need JSON parsing library
        // For now, treat as plain text
        std::cerr << "[LyricsScraper] JSON parsing not implemented, treating as plain text\n";
    }

    // Extract phrases from content
    auto phrases = extract_phrases(content, "", "");
    for (auto& phrase : phrases) {
        phrase.source = "local";
        callback(std::move(phrase));
    }
}

inline std::string LyricsScraper::http_get(const std::string& /*url*/) {
    // HTTP GET requires external library (libcurl, etc.)
    // Not implemented in header-only mode
    std::cerr << "[LyricsScraper] HTTP requests not implemented.\n";
    return "";
}

}  // namespace scrapers
}  // namespace collider
