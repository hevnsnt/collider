/**
 * Live BTC balance lookup against mempool.space.
 *
 * Used by the solved-puzzle banner to show whether the puzzle's
 * reward has been swept (e.g. a 0 balance for puzzle 66 means
 * someone already claimed the 6.6 BTC) or is still sitting on the
 * address (likely a reorg / partial-claim case).
 *
 * Compile-time gated on COLLIDER_HAVE_CURL: if libcurl wasn't found
 * at configure time, fetch_balance_btc() unconditionally returns
 * nullopt and the banner falls back to the bundled reward field.
 *
 * Network failures (timeout, DNS, HTTP error, malformed JSON) all
 * return nullopt; callers must handle that. 5-second timeout caps
 * the worst-case banner-render delay.
 */

#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace collider {
namespace ui {

// Returns the unspent balance (BTC) at `address` if reachable, else
// nullopt. Network call: ~150-500ms typical, 5s hard timeout.
std::optional<double> fetch_balance_btc(std::string_view address);

// Convenience: format balance for the banner.
//   nullopt          -> "balance unavailable (offline)"
//   0.0              -> "0.00000000 BTC (claimed)"
//   non-zero         -> "X.YYYYYYYY BTC (unclaimed)"
std::string format_balance(std::optional<double> balance);

}  // namespace ui
}  // namespace collider
