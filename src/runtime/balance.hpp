// Best-effort Bitcoin balance check used after a true brain-wallet or
// puzzle hit. Hands the probe to a process-local BalanceFetcher that
// owns every in-flight std::jthread; never blocks the scan thread.
#pragma once

#include <string>

namespace collider::runtime {

// Asynchronously query mempool.space for the funded/spent balance of
// `address`. Prints a celebratory box on a non-zero balance, or a dim
// "false positive" line otherwise. Errors are logged once to stderr.
// Internally owned by a process-local BalanceFetcher singleton whose
// destructor stops + joins every in-flight probe on process exit; safe
// to call from any thread.
void check_balance_async(const std::string& address,
                         const std::string& passphrase);

}  // namespace collider::runtime
