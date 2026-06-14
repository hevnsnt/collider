// pool_config.hpp - Shared compile-time pool client constants
// Centralizes constants that were previously duplicated between
// jlp_pool_client (the in-thread receiver-loop reconnect path) and
// pool_manager (the external supervisor reconnect path). Both used a
// 60s cap on the exponential backoff but each declared its own
// constant. If anyone touched one and not the other, one cap would
// drift longer than the other, breaking the operational contract that
// "the pool sees no worker pause longer than MAX_RECONNECT_BACKOFF_MS
// between connection attempts." Single source of truth here.
// Header is intentionally tiny (no <chrono>, no SDKs) so it can be
// pulled into both .hpp and .cpp without bloating compile times.

#pragma once

#include <cstdint>

namespace collider {
namespace pool {

// Maximum backoff between reconnect attempts. The exponential backoff
// in JLPPoolClient and the supervisor in PoolManager must agree on
// this cap or one will pause longer than the other. Capped at 5 minutes:
// the supervisor never gives up on connection failures, so during a long
// pool outage (maintenance, restart) the worker keeps probing every 5 min
// and reconnects on its own when service is restored -- no manual restart.
inline constexpr std::uint32_t MAX_RECONNECT_BACKOFF_MS = 300'000;

// Cap on consecutive AUTH_FAIL responses before the reconnect supervisor
// in PoolManager gives up. v1.4.2 Pool-B3: moved here from JLPPoolClient
// after the dead in-receiver-thread reconnect path was deleted. The
// supervisor in PoolManager is the only reconnect driver now; this
// constant lives next to MAX_RECONNECT_BACKOFF_MS so both reconnect
// policy values are in one place.
inline constexpr std::uint32_t MAX_AUTH_FAIL_ATTEMPTS = 3;

}  // namespace pool
}  // namespace collider
