#include "btc_balance.hpp"

#include <iomanip>
#include <sstream>
#include <string>

#ifdef COLLIDER_HAVE_CURL
#include <curl/curl.h>
#endif

namespace collider {
namespace ui {

namespace {

#ifdef COLLIDER_HAVE_CURL

// libcurl write callback. Appends received bytes to a std::string.
std::size_t curl_write_string(void* ptr, std::size_t size,
                              std::size_t nmemb, void* userdata) {
    auto* buf = static_cast<std::string*>(userdata);
    buf->append(static_cast<const char*>(ptr), size * nmemb);
    return size * nmemb;
}

// Tiny single-purpose JSON extractor: finds the integer value
// associated with `key` somewhere in `body` (e.g. "funded_txo_sum").
// Returns -1 if not found / unparseable. The mempool.space response
// is small (<2KB) and well-formed; pulling in a real JSON parser
// just for two integers would be overkill.
long long extract_int_field(std::string_view body, std::string_view key) {
    const std::string needle = std::string("\"") + std::string(key) + "\":";
    auto pos = body.find(needle);
    if (pos == std::string::npos) return -1;
    pos += needle.size();
    while (pos < body.size() && (body[pos] == ' ' || body[pos] == '\t')) ++pos;

    bool negative = false;
    if (pos < body.size() && body[pos] == '-') { negative = true; ++pos; }

    long long value = 0;
    bool any_digits = false;
    while (pos < body.size() && body[pos] >= '0' && body[pos] <= '9') {
        any_digits = true;
        value = value * 10 + (body[pos] - '0');
        ++pos;
    }
    if (!any_digits) return -1;
    return negative ? -value : value;
}

#endif  // COLLIDER_HAVE_CURL

}  // namespace

std::optional<double> fetch_balance_btc(std::string_view address) {
#ifndef COLLIDER_HAVE_CURL
    (void)address;
    return std::nullopt;
#else
    if (address.empty()) return std::nullopt;

    // mempool.space returns:
    //   {
    //     "address": "...",
    //     "chain_stats": {
    //       "funded_txo_sum": <sats>,
    //       "spent_txo_sum":  <sats>,
    //       ...
    //     },
    //     "mempool_stats": { ... }
    //   }
    // Unspent balance = chain_stats.funded - chain_stats.spent. We
    // ignore mempool_stats: a mempool-only inflow isn't confirmed
    // and shouldn't claim the puzzle solved.
    std::string url = "https://mempool.space/api/address/";
    url.append(address);

    CURL* curl = curl_easy_init();
    if (!curl) return std::nullopt;

    std::string body;
    body.reserve(2048);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 3L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "collider/1.4.x");
    // mempool.space serves valid LE certs; default verify is correct.

    const CURLcode rc = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK || http_code != 200) return std::nullopt;

    // Drill into chain_stats first, then look up funded/spent within
    // its slice. Doing the global search would also match the same
    // keys inside mempool_stats which we deliberately ignore.
    const auto chain_pos = body.find("\"chain_stats\"");
    if (chain_pos == std::string::npos) return std::nullopt;
    std::string_view chain_slice = std::string_view(body).substr(chain_pos);
    // Limit slice to the closing brace of chain_stats so a stray
    // "funded_txo_sum" elsewhere can't bleed in.
    auto brace_close = chain_slice.find('}');
    if (brace_close != std::string_view::npos) {
        chain_slice = chain_slice.substr(0, brace_close);
    }

    const long long funded = extract_int_field(chain_slice, "funded_txo_sum");
    const long long spent  = extract_int_field(chain_slice, "spent_txo_sum");
    if (funded < 0 || spent < 0) return std::nullopt;

    const long long sats = funded - spent;
    if (sats < 0) return std::nullopt;
    return static_cast<double>(sats) / 1e8;
#endif
}

std::string format_balance(std::optional<double> balance) {
    if (!balance.has_value()) return "balance unavailable (offline)";

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(8) << *balance;
    if (*balance == 0.0) {
        oss << " BTC (claimed)";
    } else {
        oss << " BTC (unclaimed)";
    }
    return oss.str();
}

}  // namespace ui
}  // namespace collider
