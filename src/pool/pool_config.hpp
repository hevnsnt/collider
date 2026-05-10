// pool_config.hpp - Shared compile-time pool client constants
//
// v1.4.1: Centralizes constants that were previously duplicated between
// jlp_pool_client (the in-thread receiver-loop reconnect path) and
// pool_manager (the external supervisor reconnect path). Both used a
// 60s cap on the exponential backoff but each declared its own
// constant. If anyone touched one and not the other, one cap would
// drift longer than the other, breaking the operational contract that
// "the pool sees no worker pause longer than MAX_RECONNECT_BACKOFF_MS
// between connection attempts." Single source of truth here.
//
// Header is intentionally tiny (no <chrono>, no SDKs) so it can be
// pulled into both .hpp and .cpp without bloating compile times.

#pragma once

#include <cstdint>

namespace collider {
namespace pool {

// Maximum backoff between reconnect attempts. The exponential backoff
// in JLPPoolClient and the supervisor in PoolManager must agree on
// this cap or one will pause longer than the other.
inline constexpr std::uint32_t MAX_RECONNECT_BACKOFF_MS = 60'000;

}  // namespace pool
}  // namespace collider
