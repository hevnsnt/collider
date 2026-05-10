/**
 * Brain Wallet v2 host-side orchestrator -- implementation.
 *
 * See v2_orchestrator.hpp for the API contract.
 *
 * Stays free of cuda_runtime.h on the parser / loader paths so that
 * tests can exercise them on any host. The single CUDA-touching helper
 * (`dispatch_to_kernel`) is gated under COLLIDER_USE_CUDA and stubs out
 * to a runtime-error otherwise.
 */

#include "v2_orchestrator.hpp"
#include "brain_wallet_v2.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace collider {
namespace gpu {
namespace v2 {

namespace {

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

std::string to_lower_trim(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
        out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

std::vector<std::string> split_csv(std::string_view csv) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : csv) {
        if (c == ',') {
            if (!cur.empty()) { out.push_back(to_lower_trim(cur)); cur.clear(); }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(to_lower_trim(cur));
    return out;
}

// ---------------------------------------------------------------------------
// Hex helpers
// ---------------------------------------------------------------------------

bool hex_to_bytes_be(std::string_view hex, uint8_t out[32]) {
    // Strip optional "0x"/"0X" prefix
    if (hex.size() >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) {
        hex.remove_prefix(2);
    }
    if (hex.size() > 64) return false;
    // Right-justify into 32 bytes (pad with zeros on the left)
    std::fill(out, out + 32, uint8_t{0});
    const size_t pad = 32 - (hex.size() + 1) / 2;
    for (size_t i = 0; i < hex.size(); ++i) {
        char c = hex[i];
        int digit;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') digit = 10 + (c - 'A');
        else return false;
        const size_t byte_idx = pad + (i + (hex.size() % 2)) / 2;
        if (byte_idx >= 32) return false;
        if (((i + (hex.size() % 2)) & 1) == 1) {
            out[byte_idx] |= static_cast<uint8_t>(digit);
        } else {
            out[byte_idx] |= static_cast<uint8_t>(digit << 4);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Tiny JSON parser (loader-grade, NOT a general-purpose JSON parser).
// Handles only the shapes we expect from puzzle_history.json:
//   - top-level object
//   - one array under "puzzles" or "solve_history"
//   - each entry an object with string/number fields
// Comments not allowed; supports the standard JSON string escapes
// including \uXXXX (BMP code points; surrogate pairs concatenated to
// form non-BMP characters before UTF-8 emission). v1.4.1 C.4 added the
// escape handling -- pre-1.4.1 the captured string body was returned
// verbatim, which silently mis-rendered any operator-authored entry
// that used a \\u escape (e.g. "\\u00ad" appearing in a key field).
// ---------------------------------------------------------------------------

class TinyJson {
public:
    explicit TinyJson(std::string_view src) : src_(src) {}

    // Find the first top-level array under one of the given keys.
    // Returns true and writes the array's contents (still as substring of src)
    // into `out_array_body`. Substring excludes the surrounding [ ].
    //
    // Robustness: a naive `src_.find("\"puzzles\"")` matches anywhere the
    // literal string appears, INCLUDING inside string values (e.g.
    // `"description": "all known puzzles"` would match before reaching the
    // actual `"puzzles":` key). The match is only valid when followed by
    // optional whitespace then a colon then optional whitespace then `[`.
    // When the colon-and-bracket check fails, advance past the false match
    // and keep searching the same key before falling back to the next
    // candidate.
    //
    // If a key is found but its value is NOT an array (e.g. {"puzzles":"oops"}),
    // we log a one-line warning and try the next candidate -- this surfaces
    // the typo instead of silently falling through.
    bool find_array(const std::vector<std::string>& keys,
                    std::string_view& out_array_body) const {
        for (const auto& key : keys) {
            const std::string needle = "\"" + key + "\"";
            size_t search_from = 0;
            size_t i = 0;
            bool found_real_key = false;
            while (true) {
                const size_t pos = src_.find(needle, search_from);
                if (pos == std::string::npos) break;
                i = pos + needle.size();
                // Validate this is a real key, not a string-value coincidence:
                // optional whitespace, then colon.
                size_t j = i;
                while (j < src_.size() && std::isspace(static_cast<unsigned char>(src_[j]))) ++j;
                if (j >= src_.size() || src_[j] != ':') {
                    // Not a key occurrence; advance past this match and retry.
                    search_from = pos + 1;
                    continue;
                }
                i = j + 1;
                while (i < src_.size() && std::isspace(static_cast<unsigned char>(src_[i]))) ++i;
                found_real_key = true;
                break;
            }
            if (!found_real_key) continue;

            if (i >= src_.size() || src_[i] != '[') {
                std::fprintf(stderr,
                    "[v2] warning: top-level key \"%s\" is present but not an "
                    "array; ignoring and trying next candidate\n", key.c_str());
                continue;
            }
            const size_t arr_start = i + 1;
            // Find matching ]
            int depth = 1;
            bool in_str = false;
            for (size_t j = arr_start; j < src_.size(); ++j) {
                char c = src_[j];
                if (in_str) {
                    if (c == '\\' && j + 1 < src_.size()) { ++j; continue; }
                    if (c == '"') in_str = false;
                    continue;
                }
                if (c == '"') { in_str = true; continue; }
                if (c == '[') ++depth;
                else if (c == ']') {
                    if (--depth == 0) {
                        out_array_body = src_.substr(arr_start, j - arr_start);
                        return true;
                    }
                }
            }
            return false;
        }
        return false;
    }

    // Iterate top-level objects in the array body.
    // For each, calls `cb(object_body)` where object_body excludes { }.
    template <typename F>
    static void for_each_object(std::string_view array_body, F&& cb) {
        size_t i = 0;
        while (i < array_body.size()) {
            while (i < array_body.size() && (std::isspace(static_cast<unsigned char>(array_body[i])) || array_body[i] == ',')) ++i;
            if (i >= array_body.size()) break;
            if (array_body[i] != '{') { ++i; continue; }
            const size_t obj_start = i + 1;
            int depth = 1;
            bool in_str = false;
            size_t j = obj_start;
            for (; j < array_body.size(); ++j) {
                char c = array_body[j];
                if (in_str) {
                    if (c == '\\' && j + 1 < array_body.size()) { ++j; continue; }
                    if (c == '"') in_str = false;
                    continue;
                }
                if (c == '"') { in_str = true; continue; }
                if (c == '{') ++depth;
                else if (c == '}') {
                    if (--depth == 0) {
                        cb(array_body.substr(obj_start, j - obj_start));
                        i = j + 1;
                        goto next_object;
                    }
                }
            }
            // Unterminated object; bail.
            return;
next_object:;
        }
    }

    // v1.4.1 C.4: decode the standard JSON string escapes.
    //
    // \\, \", \/  -> the literal char.
    // \b \f \n \r \t -> the corresponding control byte.
    // \uXXXX -> UTF-8 encoding of the BMP code point. A high-surrogate
    //           (0xD800-0xDBFF) immediately followed by \uXXXX with a
    //           low-surrogate (0xDC00-0xDFFF) is combined into a non-
    //           BMP code point and emitted as a 4-byte UTF-8 sequence.
    //           Lone surrogates emit their replacement character so we
    //           never produce invalid UTF-8.
    //
    // Returns false on any malformed escape (e.g. \\uXY with non-hex
    // digits) so callers can surface a parse error rather than silently
    // mis-render the string.
    static bool decode_string_escapes(std::string_view raw, std::string& out) {
        auto hex_nibble = [](char c, int& v) {
            if (c >= '0' && c <= '9') { v = c - '0'; return true; }
            if (c >= 'a' && c <= 'f') { v = 10 + (c - 'a'); return true; }
            if (c >= 'A' && c <= 'F') { v = 10 + (c - 'A'); return true; }
            return false;
        };
        auto emit_utf8 = [&](uint32_t cp) {
            if (cp < 0x80) {
                out.push_back(static_cast<char>(cp));
            } else if (cp < 0x800) {
                out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else if (cp < 0x10000) {
                out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            } else {
                out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
            }
        };
        auto parse_u4 = [&](size_t& i, uint32_t& cp_out) -> bool {
            if (i + 4 > raw.size()) return false;
            uint32_t cp = 0;
            for (int k = 0; k < 4; ++k) {
                int v;
                if (!hex_nibble(raw[i + k], v)) return false;
                cp = (cp << 4) | static_cast<uint32_t>(v);
            }
            i += 4;
            cp_out = cp;
            return true;
        };
        out.clear();
        out.reserve(raw.size());
        size_t i = 0;
        while (i < raw.size()) {
            const char c = raw[i];
            if (c != '\\') { out.push_back(c); ++i; continue; }
            if (i + 1 >= raw.size()) return false;
            const char esc = raw[i + 1];
            i += 2;
            switch (esc) {
                case '"':  out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case '/':  out.push_back('/');  break;
                case 'b':  out.push_back('\b'); break;
                case 'f':  out.push_back('\f'); break;
                case 'n':  out.push_back('\n'); break;
                case 'r':  out.push_back('\r'); break;
                case 't':  out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!parse_u4(i, cp)) return false;
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        // High surrogate; expect immediately following \\uXXXX low surrogate.
                        if (i + 2 <= raw.size() && raw[i] == '\\' && raw[i + 1] == 'u') {
                            i += 2;
                            uint32_t low = 0;
                            if (!parse_u4(i, low)) return false;
                            if (low >= 0xDC00 && low <= 0xDFFF) {
                                cp = 0x10000
                                   + ((cp - 0xD800) << 10)
                                   + (low - 0xDC00);
                                emit_utf8(cp);
                                break;
                            }
                            // Low not a low-surrogate: emit replacement
                            // for the high surrogate and re-emit the
                            // non-low \\u as itself.
                            emit_utf8(0xFFFD);
                            emit_utf8(low);
                            break;
                        }
                        emit_utf8(0xFFFD);  // lone high surrogate
                        break;
                    }
                    if (cp >= 0xDC00 && cp <= 0xDFFF) {
                        emit_utf8(0xFFFD);  // lone low surrogate
                        break;
                    }
                    emit_utf8(cp);
                    break;
                }
                default:
                    return false;  // unrecognized escape
            }
        }
        return true;
    }

    // Get a string-typed field's value (without surrounding quotes).
    // Returns false if not found, not a string, or contains a malformed
    // escape -- callers surface a parse error rather than silently
    // mis-render.
    static bool get_string(std::string_view obj, std::string_view key,
                           std::string& out) {
        const std::string needle = "\"" + std::string(key) + "\"";
        size_t pos = obj.find(needle);
        if (pos == std::string::npos) return false;
        size_t i = pos + needle.size();
        while (i < obj.size() && std::isspace(static_cast<unsigned char>(obj[i]))) ++i;
        if (i >= obj.size() || obj[i] != ':') return false;
        ++i;
        while (i < obj.size() && std::isspace(static_cast<unsigned char>(obj[i]))) ++i;
        if (i >= obj.size() || obj[i] != '"') return false;
        ++i;
        const size_t v_start = i;
        while (i < obj.size()) {
            if (obj[i] == '\\' && i + 1 < obj.size()) { i += 2; continue; }
            if (obj[i] == '"') break;
            ++i;
        }
        if (i >= obj.size()) return false;
        std::string_view raw(obj.data() + v_start, i - v_start);
        return decode_string_escapes(raw, out);
    }

    // Get a number-typed field. Accepts decimal int.
    static bool get_int(std::string_view obj, std::string_view key, int64_t& out) {
        const std::string needle = "\"" + std::string(key) + "\"";
        size_t pos = obj.find(needle);
        if (pos == std::string::npos) return false;
        size_t i = pos + needle.size();
        while (i < obj.size() && std::isspace(static_cast<unsigned char>(obj[i]))) ++i;
        if (i >= obj.size() || obj[i] != ':') return false;
        ++i;
        while (i < obj.size() && std::isspace(static_cast<unsigned char>(obj[i]))) ++i;
        const size_t v_start = i;
        if (i < obj.size() && (obj[i] == '-' || obj[i] == '+')) ++i;
        while (i < obj.size() && std::isdigit(static_cast<unsigned char>(obj[i]))) ++i;
        if (i == v_start) return false;
        try {
            out = std::stoll(std::string(obj.substr(v_start, i - v_start)));
        } catch (...) {
            return false;
        }
        return true;
    }

private:
    std::string_view src_;
};

}  // namespace

// ---------------------------------------------------------------------------
// parse_scheme_mask
// ---------------------------------------------------------------------------

bool parse_scheme_mask(std::string_view csv,
                       uint32_t& mask_out,
                       std::string& error_out) {
    error_out.clear();

    static const std::unordered_map<std::string, DerivationScheme> kMap = {
        {"sha256_pw",            DerivationScheme::SHA256_PW},
        {"sha256_sha256_pw",     DerivationScheme::SHA256_SHA256_PW},
        {"sha256_pw_newline",    DerivationScheme::SHA256_PW_NEWLINE},
        {"sha256_pw_pw",         DerivationScheme::SHA256_PW_PW},
        {"sha256_sha256_pw_pw",  DerivationScheme::SHA256_SHA256_PW_PW},
        {"sha256_iter_16",       DerivationScheme::SHA256_ITER_16},
        {"hmac_sha512_pw",       DerivationScheme::HMAC_SHA512_PW},
        {"sha512_pw_half",       DerivationScheme::SHA512_PW_HALF},
    };

    const std::string trimmed = to_lower_trim(csv);
    if (trimmed.empty() || trimmed == "stock") {
        mask_out = SCHEME_MASK_STOCK;
        return true;
    }
    if (trimmed == "all" || trimmed == "*") {
        mask_out = SCHEME_MASK_ALL;
        return true;
    }

    uint32_t mask = 0;
    for (const auto& tok : split_csv(csv)) {
        if (tok.empty()) continue;
        auto it = kMap.find(tok);
        if (it == kMap.end()) {
            error_out = "unknown scheme name: " + tok;
            return false;
        }
        mask |= scheme_bit(it->second);
    }
    if (mask == 0) {
        error_out = "scheme list expanded to empty mask";
        return false;
    }
    mask_out = mask;
    return true;
}

// ---------------------------------------------------------------------------
// parse_addr_mask
// ---------------------------------------------------------------------------

bool parse_addr_mask(std::string_view csv,
                     uint32_t& mask_out,
                     std::string& error_out) {
    error_out.clear();

    static const std::unordered_map<std::string, AddressType> kMap = {
        {"p2pkh_uncompressed", AddressType::P2PKH_UNCOMPRESSED},
        {"p2pkh_compressed",   AddressType::P2PKH_COMPRESSED},
        {"p2sh_p2wpkh",        AddressType::P2SH_P2WPKH},
        {"p2wpkh_v0",          AddressType::P2WPKH_V0},
        {"p2tr_bip86",         AddressType::P2TR_BIP86},
    };

    const std::string trimmed = to_lower_trim(csv);
    if (trimmed.empty() || trimmed == "stock") {
        mask_out = ADDR_MASK_STOCK;
        return true;
    }
    if (trimmed == "all" || trimmed == "*") {
        mask_out = ADDR_MASK_ALL;
        return true;
    }
    if (trimmed == "modern") {
        mask_out = ADDR_MASK_MODERN;
        return true;
    }
    if (trimmed == "none" || trimmed == "puzzle_only") {
        mask_out = 0;
        return true;
    }

    uint32_t mask = 0;
    for (const auto& tok : split_csv(csv)) {
        if (tok.empty()) continue;
        auto it = kMap.find(tok);
        if (it == kMap.end()) {
            error_out = "unknown address-type name: " + tok;
            return false;
        }
        mask |= addr_bit(it->second);
    }
    mask_out = mask;
    return true;
}

// ---------------------------------------------------------------------------
// load_puzzle_targets
// ---------------------------------------------------------------------------

bool load_puzzle_targets(const std::string& path,
                         std::vector<PuzzleTarget>& targets_out,
                         std::string& error_out) {
    error_out.clear();
    targets_out.clear();

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error_out = "could not open: " + path;
        return false;
    }
    std::ostringstream buf;
    buf << f.rdbuf();
    const std::string body = buf.str();
    if (body.empty()) {
        error_out = "file is empty: " + path;
        return false;
    }

    TinyJson tj(body);
    std::string_view arr;
    if (!tj.find_array({"puzzles", "solve_history"}, arr)) {
        error_out = "no top-level 'puzzles' or 'solve_history' array in " + path;
        return false;
    }

    int skipped = 0;
    TinyJson::for_each_object(arr, [&](std::string_view obj) {
        // Required fields: a number-typed puzzle index + a hex private key.
        int64_t puzzle_n = 0;
        if (!TinyJson::get_int(obj, "puzzle_n", puzzle_n) &&
            !TinyJson::get_int(obj, "puzzle_number", puzzle_n) &&
            !TinyJson::get_int(obj, "n", puzzle_n)) {
            ++skipped;
            return;
        }
        std::string priv_hex;
        if (!TinyJson::get_string(obj, "private_key_hex", priv_hex) &&
            !TinyJson::get_string(obj, "private_key", priv_hex) &&
            !TinyJson::get_string(obj, "key", priv_hex)) {
            ++skipped;
            return;
        }
        uint8_t priv_be[32];
        if (!hex_to_bytes_be(priv_hex, priv_be)) {
            ++skipped;
            return;
        }
        if (puzzle_n < 1 || puzzle_n > PUZZLE_TARGET_MAX) {
            ++skipped;
            return;
        }
        targets_out.push_back(make_puzzle_target(static_cast<uint16_t>(puzzle_n), priv_be));
    });

    std::sort(targets_out.begin(), targets_out.end(),
              [](const PuzzleTarget& a, const PuzzleTarget& b) {
                  return a.puzzle_n < b.puzzle_n;
              });

    // Deduplicate by puzzle_n: an input file with multiple entries for
    // the same puzzle (e.g., reconstructed from independent solves)
    // would otherwise make the GPU kernel check every passphrase
    // against each duplicate, wasting cycles. Sorted-then-unique is
    // O(n) on the already-sorted vector; std::unique keeps the first
    // entry for each puzzle_n, matching the order observed in the
    // input file (after the stable sort).
    const size_t before_dedup = targets_out.size();
    auto last = std::unique(targets_out.begin(), targets_out.end(),
                            [](const PuzzleTarget& a, const PuzzleTarget& b) {
                                return a.puzzle_n == b.puzzle_n;
                            });
    targets_out.erase(last, targets_out.end());
    const size_t deduped = before_dedup - targets_out.size();
    if (deduped > 0) {
        std::fprintf(stderr,
            "[v2] load_puzzle_targets: dropped %zu duplicate puzzle_n entries in %s\n",
            deduped, path.c_str());
    }

    if (targets_out.empty()) {
        error_out = "no usable puzzle entries found in " + path
                  + " (skipped " + std::to_string(skipped) + ")";
        return false;
    }
    if (skipped > 0) {
        std::fprintf(stderr,
            "[v2] load_puzzle_targets: skipped %d malformed entries in %s\n",
            skipped, path.c_str());
    }
    return true;
}

// ---------------------------------------------------------------------------
// Phase 4 multi-address pipeline driver.
// External entry points live in src/gpu/v2/multi_address_kernel.cu and
// src/gpu/secp256k1.cu. They are namespaced C++ functions (not extern "C")
// to match the brain_wallet_v2.hpp declarations and standard linkage rules.
// secp256k1_* are at namespace scope (no namespace) per secp256k1.cu's
// existing extern "C" definitions.
// ---------------------------------------------------------------------------

#if defined(COLLIDER_USE_CUDA) && defined(COLLIDER_PRO)
extern "C" cudaError_t secp256k1_batch_mul(
    const void* d_private_keys,
    void* d_public_keys,
    size_t count,
    cudaStream_t stream);
extern "C" cudaError_t secp256k1_init_table(cudaStream_t stream);
#endif

// ---------------------------------------------------------------------------
// MultiAddressSession::Impl -- holds the cached bloom + reusable device
// buffers across batches. See header for rationale (bloom upload was the
// dominant cost on small batches; Gemini PR #15 HIGH finding).
// ---------------------------------------------------------------------------
#if defined(COLLIDER_USE_CUDA) && defined(COLLIDER_PRO)
struct MultiAddressSession::Impl {
    cudaStream_t   stream         = nullptr;
    uint8_t*       d_priv         = nullptr;
    uint8_t*       d_pub_xy       = nullptr;
    uint8_t*       d_bloom        = nullptr;
    V2MatchRecord* d_matches      = nullptr;
    uint32_t*      d_match_count  = nullptr;
    uint64_t       bloom_bits     = 0;
    int            bloom_hashes   = 0;
    uint32_t       bloom_seed     = 0;
    size_t         max_batch_count = 0;

    // DP-buffer overflow diagnostics. dropped_matches is the running
    // total of records the kernel counted but had no slot for; non-zero
    // means V2_MAX_MATCHES_PER_BATCH is too small for the workload.
    // overflow_batches is the count of batches that triggered overflow.
    uint64_t dropped_matches  = 0;
    uint64_t overflow_batches = 0;

    ~Impl() {
        if (d_priv)        cudaFree(d_priv);
        if (d_pub_xy)      cudaFree(d_pub_xy);
        if (d_bloom)       cudaFree(d_bloom);
        if (d_matches)     cudaFree(d_matches);
        if (d_match_count) cudaFree(d_match_count);
        if (stream)        cudaStreamDestroy(stream);
    }
};
#else
struct MultiAddressSession::Impl {};   // empty stub for non-CUDA builds
#endif

MultiAddressSession::MultiAddressSession() : impl_(new Impl{}) {}
MultiAddressSession::~MultiAddressSession() { delete impl_; }

int MultiAddressSession::init(const uint8_t* bloom,
                              uint64_t bloom_bits,
                              int bloom_hashes,
                              uint32_t bloom_seed,
                              size_t max_batch_count)
{
    if (max_batch_count == 0) return 64;
#if defined(COLLIDER_USE_CUDA) && defined(COLLIDER_PRO)
    if (cudaStreamCreate(&impl_->stream) != cudaSuccess) return 70;

    cudaError_t rc = secp256k1_init_table(impl_->stream);
    if (rc) return 70;

    rc = cudaMalloc((void**)&impl_->d_priv,   max_batch_count * 32);
    if (rc) return 70;
    rc = cudaMalloc((void**)&impl_->d_pub_xy, max_batch_count * 64);
    if (rc) return 70;

    if (bloom && bloom_bits > 0) {
        const size_t bytes = (size_t)((bloom_bits + 7) / 8);
        rc = cudaMalloc((void**)&impl_->d_bloom, bytes);
        if (rc) return 70;
        // One-time synchronous-on-stream upload. cudaMemcpyAsync is fine
        // here because subsequent process_batch() calls run on the same
        // stream and so are ordered after the upload.
        rc = cudaMemcpyAsync(impl_->d_bloom, bloom, bytes,
                             cudaMemcpyHostToDevice, impl_->stream);
        if (rc) return 70;
    }

    rc = cudaMalloc((void**)&impl_->d_matches,
                    V2_MAX_MATCHES_PER_BATCH * sizeof(V2MatchRecord));
    if (rc) return 70;
    rc = cudaMalloc((void**)&impl_->d_match_count, sizeof(uint32_t));
    if (rc) return 70;

    impl_->bloom_bits      = bloom_bits;
    impl_->bloom_hashes    = bloom_hashes;
    impl_->bloom_seed      = bloom_seed;
    impl_->max_batch_count = max_batch_count;
    return 0;
#else
    (void)bloom; (void)bloom_bits; (void)bloom_hashes; (void)bloom_seed;
    return 64;  // CUDA-only feature
#endif
}

int MultiAddressSession::process_batch(const MultiAddressBatch& b) {
    if (b.count == 0 || b.addr_mask == 0) return 64;
#if defined(COLLIDER_USE_CUDA) && defined(COLLIDER_PRO)
    if (!impl_->stream) return 64;  // init() not called
    if (b.count > impl_->max_batch_count) return 64;  // batch too large

    // Reset the match counter; bloom + buffers are already on device.
    cudaError_t rc = cudaMemsetAsync(impl_->d_match_count, 0,
                                     sizeof(uint32_t), impl_->stream);
    if (rc) return 70;

    rc = cudaMemcpyAsync(impl_->d_priv, b.priv_batch, b.count * 32,
                         cudaMemcpyHostToDevice, impl_->stream);
    if (rc) return 70;

    // Phase 4 step 1: priv -> pub.
    rc = secp256k1_batch_mul(impl_->d_priv, impl_->d_pub_xy,
                             b.count, impl_->stream);
    if (rc) return 70;

    // Phase 4 step 2: multi-address derivation + bloom probe.
    // The session's cached bloom params override anything in `b`.
    rc = v2_multi_address_check(
        impl_->d_pub_xy, b.count, b.addr_mask,
        impl_->d_bloom, impl_->bloom_bits,
        impl_->bloom_hashes, impl_->bloom_seed,
        impl_->d_matches, impl_->d_match_count, impl_->stream);
    if (rc) return 70;

    rc = cudaStreamSynchronize(impl_->stream);
    if (rc) return 70;

    uint32_t hits = 0;
    cudaMemcpy(&hits, impl_->d_match_count, sizeof(uint32_t),
               cudaMemcpyDeviceToHost);
    if (hits > 0) {
        const uint32_t take = hits < V2_MAX_MATCHES_PER_BATCH
                              ? hits : V2_MAX_MATCHES_PER_BATCH;
        if (hits > V2_MAX_MATCHES_PER_BATCH) {
            const uint32_t dropped = hits - V2_MAX_MATCHES_PER_BATCH;
            impl_->dropped_matches  += dropped;
            impl_->overflow_batches += 1;
            // Surface the overflow on stderr so a `2>` redirect captures
            // it even when stdout is being parsed for HIT lines. The
            // running totals stay accessible via total_dropped_matches()
            // for programmatic callers.
            std::fprintf(stderr,
                "[v2:multi-addr] WARN match buffer overflow: "
                "kernel counted %u hits, buffer holds %u, dropped %u "
                "(running total dropped=%llu across %llu batch(es)). "
                "Increase V2_MAX_MATCHES_PER_BATCH or shrink batch count.\n",
                hits, V2_MAX_MATCHES_PER_BATCH, dropped,
                static_cast<unsigned long long>(impl_->dropped_matches),
                static_cast<unsigned long long>(impl_->overflow_batches));
        }
        std::vector<V2MatchRecord> recs(take);
        cudaMemcpy(recs.data(), impl_->d_matches,
                   take * sizeof(V2MatchRecord), cudaMemcpyDeviceToHost);
        for (auto& r : recs) {
            std::printf("[v2:multi-addr] HIT pp_idx=%u addr_type=%u kind=%u\n",
                        r.pp_idx, r.addr_type, r.kind);
        }
    }
    return 0;
#else
    return 64;
#endif
}

uint64_t MultiAddressSession::total_dropped_matches() const {
#if defined(COLLIDER_USE_CUDA) && defined(COLLIDER_PRO)
    return impl_ ? impl_->dropped_matches : 0;
#else
    return 0;
#endif
}

uint64_t MultiAddressSession::total_overflow_batches() const {
#if defined(COLLIDER_USE_CUDA) && defined(COLLIDER_PRO)
    return impl_ ? impl_->overflow_batches : 0;
#else
    return 0;
#endif
}

// Legacy single-shot API: build a session, run one batch, tear down.
// Equivalent to the pre-session implementation; preserved so existing
// callers don't change.
int run_multi_address_batch(const MultiAddressBatch& b) {
    if (b.count == 0 || b.addr_mask == 0) return 64;
#if defined(COLLIDER_USE_CUDA) && defined(COLLIDER_PRO)
    MultiAddressSession session;
    int rc = session.init(b.bloom, b.bloom_bits, b.bloom_hashes,
                          b.bloom_seed, b.count);
    if (rc != 0) return rc;
    return session.process_batch(b);
#else
    (void)b;
    std::fprintf(stderr,
        "[v2:multi-addr] requires Pro CUDA build (COLLIDER_PRO + COLLIDER_USE_CUDA)\n");
    return 64;
#endif
}

// ---------------------------------------------------------------------------
// run_v2_orchestrator
// ---------------------------------------------------------------------------

int run_v2_orchestrator(const OrchestratorOptions& opts) {
    // 1. Load puzzle targets.
    const std::string keys_path = opts.puzzle_keys_path.empty()
        ? std::string("./data/puzzle_history.json")
        : opts.puzzle_keys_path;
    std::vector<PuzzleTarget> targets;
    std::string err;
    if (!load_puzzle_targets(keys_path, targets, err)) {
        std::fprintf(stderr, "[v2] %s\n", err.c_str());
        return 65;  // EX_DATAERR
    }

    if (opts.show_summary) {
        std::printf("[v2] loaded %zu puzzle target(s) from %s\n",
                    targets.size(), keys_path.c_str());
        std::printf("[v2] scheme_mask=0x%08x  addr_mask=0x%08x\n",
                    opts.scheme_mask, opts.addr_mask);
        if (opts.addr_mask == 0) {
            std::printf("[v2] puzzle-only short-circuit ENABLED (no EC_MUL "
                        "for non-hits)\n");
        }
    }

    if (opts.dry_run) {
        std::printf("[v2] --dry-run: GPU dispatch skipped\n");
        return 0;
    }

#if defined(COLLIDER_USE_CUDA) && defined(COLLIDER_PRO)
    // Multi-address mode requires the Phase 4 priv-derivation pipeline
    // (passphrase -> scheme(...) -> priv -> MultiAddressSession). That
    // bridge isn't built yet; v2_brain_wallet_batch returns
    // cudaErrorNotSupported for addr_mask != 0. Refuse cleanly so
    // operators don't get a confusing CUDA error half-way through a
    // scan.
    if (opts.addr_mask != 0) {
        std::fprintf(stderr,
            "[v2] --addr-types selects multi-address scanning, which is "
            "not wired through the orchestrator yet (Phase 4 follow-up). "
            "Use the legacy brain-wallet pipeline (--brainwallet --bloom) "
            "for now.\n");
        return 64;  // EX_USAGE
    }

    // ------------------------------------------------------------------
    // Puzzle-only scan loop (CUDA).
    // ------------------------------------------------------------------
    cudaStream_t stream = nullptr;
    cudaError_t cerr = cudaStreamCreate(&stream);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr, "[v2] cudaStreamCreate failed: %s\n",
                     cudaGetErrorString(cerr));
        return 70;  // EX_SOFTWARE
    }
    cerr = v2_init(stream);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr, "[v2] v2_init failed: %s\n",
                     cudaGetErrorString(cerr));
        cudaStreamDestroy(stream);
        return 70;
    }
    cerr = v2_set_puzzle_targets(targets);
    if (cerr != cudaSuccess) {
        std::fprintf(stderr, "[v2] v2_set_puzzle_targets failed: %s\n",
                     cudaGetErrorString(cerr));
        v2_shutdown();
        cudaStreamDestroy(stream);
        return 70;
    }

    // Per-batch scan parameters. 64K passphrases per batch keeps the
    // packed buffer under 16 MiB at MAX_PP_LEN=256 -- comfortably below
    // any reasonable GPU memory budget while large enough that kernel
    // launch overhead is amortized. MAX_PP_LEN matches the kernel's
    // internal SHA buffer caps; longer lines are skipped.
    constexpr size_t BATCH_SIZE  = 65536;
    constexpr size_t MAX_PP_LEN  = 256;
    const size_t pp_buf_bytes    = BATCH_SIZE * MAX_PP_LEN;

    uint8_t*  d_passphrases  = nullptr;
    uint32_t* d_offsets      = nullptr;
    uint32_t* d_lengths      = nullptr;
    V2MatchRecord* d_matches = nullptr;
    uint32_t* d_match_count  = nullptr;

    auto cleanup_scan = [&]() {
        if (d_passphrases) cudaFree(d_passphrases);
        if (d_offsets)     cudaFree(d_offsets);
        if (d_lengths)     cudaFree(d_lengths);
        if (d_matches)     cudaFree(d_matches);
        if (d_match_count) cudaFree(d_match_count);
        v2_shutdown();
        cudaStreamDestroy(stream);
    };

    cerr = cudaMalloc(&d_passphrases, pp_buf_bytes);
    if (!cerr) cerr = cudaMalloc(&d_offsets, BATCH_SIZE * sizeof(uint32_t));
    if (!cerr) cerr = cudaMalloc(&d_lengths, BATCH_SIZE * sizeof(uint32_t));
    if (!cerr) cerr = cudaMalloc(&d_matches,
                                 V2_MAX_MATCHES_PER_BATCH * sizeof(V2MatchRecord));
    if (!cerr) cerr = cudaMalloc(&d_match_count, sizeof(uint32_t));
    if (cerr != cudaSuccess) {
        std::fprintf(stderr, "[v2] cudaMalloc failed: %s\n",
                     cudaGetErrorString(cerr));
        cleanup_scan();
        return 70;
    }

    // Host-side staging buffers reused across batches.
    std::vector<uint8_t>  pp_buf(pp_buf_bytes);
    std::vector<uint32_t> offsets;
    std::vector<uint32_t> lengths;
    offsets.reserve(BATCH_SIZE);
    lengths.reserve(BATCH_SIZE);

    // Wordlist input: empty path -> stdin.
    std::ifstream wl_file;
    std::istream* wl = nullptr;
    if (opts.wordlist_path.empty()) {
        wl = &std::cin;
    } else {
        wl_file.open(opts.wordlist_path);
        if (!wl_file) {
            std::fprintf(stderr, "[v2] cannot open wordlist '%s'\n",
                         opts.wordlist_path.c_str());
            cleanup_scan();
            return 66;  // EX_NOINPUT
        }
        wl = &wl_file;
    }

    // Hit output: empty path -> stdout. The hits sink stays open across
    // batches; flushing per-batch keeps the file useful even if the
    // process is killed mid-scan.
    std::ofstream hits_file;
    std::ostream* hits_out = nullptr;
    if (opts.hits_out_path.empty()) {
        hits_out = &std::cout;
    } else {
        hits_file.open(opts.hits_out_path, std::ios::out | std::ios::app);
        if (!hits_file) {
            std::fprintf(stderr, "[v2] cannot open hits output '%s'\n",
                         opts.hits_out_path.c_str());
            cleanup_scan();
            return 73;  // EX_CANTCREAT
        }
        hits_out = &hits_file;
    }

    if (opts.show_summary) {
        std::printf("[v2] starting scan (BATCH_SIZE=%zu, MAX_PP_LEN=%zu)\n",
                    BATCH_SIZE, MAX_PP_LEN);
    }

    uint64_t total_passphrases     = 0;
    uint64_t total_hits             = 0;
    uint64_t total_skipped          = 0;
    uint64_t total_dropped_hits     = 0;
    uint64_t total_overflow_batches = 0;
    const auto start = std::chrono::steady_clock::now();

    std::string line;
    while (true) {
        offsets.clear();
        lengths.clear();
        size_t buf_pos = 0;
        size_t batch_n = 0;
        while (batch_n < BATCH_SIZE && std::getline(*wl, line)) {
            // Strip trailing \r (Windows-format wordlists on Linux/Mac).
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) { ++total_skipped; continue; }
            if (line.size() > MAX_PP_LEN) { ++total_skipped; continue; }
            std::memcpy(pp_buf.data() + buf_pos, line.data(), line.size());
            offsets.push_back(static_cast<uint32_t>(buf_pos));
            lengths.push_back(static_cast<uint32_t>(line.size()));
            buf_pos += line.size();
            ++batch_n;
        }
        if (batch_n == 0) break;  // EOF on first read in batch.

        // Upload + dispatch.
        cerr = cudaMemcpyAsync(d_passphrases, pp_buf.data(), buf_pos,
                               cudaMemcpyHostToDevice, stream);
        if (!cerr) cerr = cudaMemcpyAsync(d_offsets, offsets.data(),
                                          offsets.size() * sizeof(uint32_t),
                                          cudaMemcpyHostToDevice, stream);
        if (!cerr) cerr = cudaMemcpyAsync(d_lengths, lengths.data(),
                                          lengths.size() * sizeof(uint32_t),
                                          cudaMemcpyHostToDevice, stream);
        if (cerr) {
            std::fprintf(stderr, "[v2] cudaMemcpyAsync failed: %s\n",
                         cudaGetErrorString(cerr));
            cleanup_scan();
            return 70;
        }

        cerr = v2_brain_wallet_batch(
            d_passphrases, d_offsets, d_lengths, batch_n,
            opts.scheme_mask, opts.addr_mask,
            nullptr, 0, 0,  // bloom unused on the puzzle-only path
            d_matches, d_match_count, stream);
        if (cerr) {
            std::fprintf(stderr, "[v2] v2_brain_wallet_batch failed: %s\n",
                         cudaGetErrorString(cerr));
            cleanup_scan();
            return 70;
        }
        cerr = cudaStreamSynchronize(stream);
        if (cerr) {
            std::fprintf(stderr, "[v2] cudaStreamSynchronize failed: %s\n",
                         cudaGetErrorString(cerr));
            cleanup_scan();
            return 70;
        }

        // Drain matches.
        uint32_t hits = 0;
        cudaMemcpy(&hits, d_match_count, sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);
        if (hits > V2_MAX_MATCHES_PER_BATCH) {
            const uint32_t dropped = hits - V2_MAX_MATCHES_PER_BATCH;
            total_dropped_hits  += dropped;
            ++total_overflow_batches;
            // Surface to stderr so the operator notices even when stdout
            // is being filtered/parsed for HIT lines. The summary at end
            // of run carries the running totals.
            std::fprintf(stderr,
                "[v2] WARN match buffer overflow: kernel counted %u, "
                "buffer holds %u, dropped %u (running dropped=%llu across "
                "%llu batch(es)). Increase V2_MAX_MATCHES_PER_BATCH or "
                "shrink BATCH_SIZE.\n",
                hits, V2_MAX_MATCHES_PER_BATCH, dropped,
                static_cast<unsigned long long>(total_dropped_hits),
                static_cast<unsigned long long>(total_overflow_batches));
            hits = V2_MAX_MATCHES_PER_BATCH;
        }
        if (hits > 0) {
            std::vector<V2MatchRecord> recs(hits);
            cudaMemcpy(recs.data(), d_matches, hits * sizeof(V2MatchRecord),
                       cudaMemcpyDeviceToHost);
            for (const auto& r : recs) {
                // Reconstruct the source passphrase for the hit. pp_idx
                // is the index within this batch; offsets[pp_idx] +
                // lengths[pp_idx] addresses pp_buf.
                if (r.pp_idx < batch_n) {
                    const uint32_t off = offsets[r.pp_idx];
                    const uint32_t len = lengths[r.pp_idx];
                    std::string_view pw(reinterpret_cast<const char*>(
                                            pp_buf.data() + off), len);
                    *hits_out << "[v2:hit] puzzle_n=" << (unsigned)r.puzzle_n
                              << " scheme=" << (unsigned)r.scheme_id
                              << " pp_idx=" << r.pp_idx
                              << " pw=" << pw << "\n";
                } else {
                    *hits_out << "[v2:hit] puzzle_n=" << (unsigned)r.puzzle_n
                              << " scheme=" << (unsigned)r.scheme_id
                              << " pp_idx=" << r.pp_idx
                              << " (pp_idx out of batch range)\n";
                }
            }
            hits_out->flush();
            total_hits += hits;
        }
        total_passphrases += batch_n;

        // Periodic progress: every 16 batches (= 1M passphrases at
        // BATCH_SIZE=64K) print a one-line status to stderr so the
        // operator sees throughput without flooding the hits sink.
        if ((total_passphrases % (BATCH_SIZE * 16)) == 0) {
            const auto now = std::chrono::steady_clock::now();
            const double secs = std::chrono::duration<double>(now - start).count();
            const double rate = secs > 0 ? total_passphrases / secs : 0.0;
            std::fprintf(stderr,
                "[v2] %llu passphrases scanned, %llu hits, %.0f pp/s\n",
                (unsigned long long)total_passphrases,
                (unsigned long long)total_hits,
                rate);
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const double total_secs = std::chrono::duration<double>(end - start).count();
    if (opts.show_summary) {
        std::printf("[v2] scan complete: %llu passphrases, %llu hits, %llu skipped, %.1fs\n",
                    (unsigned long long)total_passphrases,
                    (unsigned long long)total_hits,
                    (unsigned long long)total_skipped,
                    total_secs);
        if (total_dropped_hits > 0) {
            std::printf("[v2] WARN: dropped %llu match record(s) across %llu "
                        "overflow batch(es). Increase V2_MAX_MATCHES_PER_BATCH "
                        "to recover them on a re-scan.\n",
                        (unsigned long long)total_dropped_hits,
                        (unsigned long long)total_overflow_batches);
        }
    }
    cleanup_scan();
    return total_hits > 0 ? 0 : 0;
#elif defined(__APPLE__) && defined(COLLIDER_USE_METAL) && defined(COLLIDER_PRO)
    // Metal dispatch path. v1.4.0 ships the CUDA scan loop only;
    // --puzzle-only-v2 on Mac is not wired through this orchestrator
    // yet (the Metal kernels exist, but the per-batch dispatch loop
    // analogous to lines ~720-880 above is in
    // src/gpu/v2/v2_metal_dispatch.mm and is invoked by the legacy
    // --brainwallet path). Refuse cleanly with a clear error rather
    // than printing "init complete" and exiting 0, which previously
    // looked like a successful no-op scan to the operator.
    std::fprintf(stderr,
        "[v2] --puzzle-only-v2 is CUDA-only in v1.4.0. On macOS the same "
        "workflow runs via --brainwallet, which dispatches through Metal. "
        "Aborting before the no-op return to avoid the false-success exit.\n");
    return 64;  // EX_USAGE
#else
    std::fprintf(stderr,
        "[v2] --puzzle-only-v2 requires a Pro build with CUDA "
        "(COLLIDER_PRO=ON + COLLIDER_USE_CUDA=ON) on Linux/Windows or Metal "
        "(COLLIDER_PRO=ON + COLLIDER_USE_METAL=ON) on macOS. "
        "Re-run with --dry-run to validate config without a GPU.\n");
    return 64;  // EX_USAGE
#endif
}

}  // namespace v2
}  // namespace gpu
}  // namespace collider
