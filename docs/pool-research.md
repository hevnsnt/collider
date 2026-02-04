# Kangaroo Pool Research & Architecture

**Date**: January 2025
**Status**: Research complete, implementation pending
**Target**: Bitcoin Puzzle #135 (13.5 BTC prize)

---

## Current State of Bitcoin Puzzle Pools

### Active Pools (as of January 2025)

| Pool | Status | Algorithm | Target Puzzles | Kangaroo Support |
|------|--------|-----------|----------------|------------------|
| **btcpuzzle.info** | Active | Brute Force | #71, #72, #73 | No |
| **hyenasoft.com (bpc.hyenasoft.com)** | Active (May 2025 posts) | Brute Force | Non-5th puzzles | No |
| **kangaroo.network** | Defunct | - | - | - |
| **puzzlesearch.github.io** | Unknown | Kangaroo (claimed) | Unknown | No connection info |
| **bitcointalk #130 effort** | Abandoned | Kangaroo | #130 (solved) | Stalled |

### Key Finding

**Puzzle #130 was SOLVED in late 2024** (~$800k prize claimed).

**Puzzle #135 is now the next Kangaroo-solvable target:**
- Prize: **13.5 BTC** (~$1.4M+ at current prices)
- Public key: `02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16`
- Bit range: 135-bit (2^134 to 2^135)

### Critical Gap

**There are NO active public Kangaroo pools for puzzle #135.**

Existing pools use brute force (works only on puzzles WITHOUT exposed public keys). Kangaroo algorithm is required for #135 due to its exposed public key.

---

## Why Firebase is Wrong for Kangaroo Pools

| Requirement | Firebase | VPS |
|-------------|----------|-----|
| **DP Collision Detection** | High latency (~100-500ms per query) | Sub-millisecond with local DB |
| **Millions of DP Lookups** | Expensive ($0.06/100k reads) | Flat monthly cost |
| **Persistent TCP Connections** | Not supported | Native (JLP protocol) |
| **Real-time DP Streaming** | Polling or expensive listeners | Direct socket writes |
| **Cost at Scale** | Grows with usage | Fixed ~$5-20/month |

**Conclusion**: Firebase suitable for dashboards/auth, NOT for core DP collection.

---

## Recommended Architecture: VPS-Based Pool

```
┌─────────────────────────────────────────────────┐
│            Cheap VPS ($5-10/month)              │
│  ┌─────────────────────────────────────────┐    │
│  │  Kangaroo Pool Server                   │    │
│  │  - TCP listener (port 17403, JLP)       │    │
│  │  - DP storage (SQLite or PostgreSQL)    │    │
│  │  - Bloom filter for fast collision check│    │
│  │  - Work distribution                    │    │
│  │  - Collision detection & key recovery   │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
           ▲              ▲              ▲
           │              │              │
      ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
      │ Worker 1│    │ Worker 2│    │ Worker N│
      │ 2x 5090 │    │ 2x 5090 │    │  (any)  │
      └─────────┘    └─────────┘    └─────────┘
```

### VPS Provider Options

| Provider | Cost | Notes |
|----------|------|-------|
| **DigitalOcean** | $5-10/month | 1GB RAM, reliable |
| **Vultr** | $5-10/month | Good performance |
| **Linode** | $5-10/month | Solid option |
| **Hetzner** | ~€4/month | Cheapest, EU-based |
| **Oracle Cloud Free** | Free | 2 AMD instances, always free tier |

### Server Requirements

- **RAM**: 1-2GB minimum (bloom filter + active connections)
- **Storage**: 10-50GB (DP database grows over time)
- **CPU**: Single core sufficient (I/O bound, not CPU bound)
- **Network**: Low latency to workers preferred

---

## Implementation Plan

### Already Implemented (Client Side)

Located in `src/pool/`:
- `pool_client.hpp` - Abstract pool interface
- `jlp_pool_client.hpp/cpp` - JLP protocol client (port 17403)
- `http_pool_client.hpp/cpp` - REST API client
- `pool_manager.hpp/cpp` - High-level coordinator

Main integration in `src/main.cpp`:
- `--pool <url>` argument
- `--worker <name>` argument
- `run_pool_mode()` function

### To Implement (Server Side)

1. **TCP Server** (JLP Protocol)
   - Accept connections on port 17403
   - Parse JLP messages (KANG magic bytes)
   - Handle authentication, work requests, DP submissions

2. **DP Storage**
   - SQLite database for persistence
   - In-memory bloom filter for fast collision pre-check
   - Indexed by DP prefix for efficient lookups

3. **Collision Detection**
   - When new DP arrives, check bloom filter
   - If potential match, query DB for exact match
   - If tame/wild collision found, compute private key

4. **Work Distribution**
   - Divide 2^135 range into chunks
   - Track assigned ranges per worker
   - Reassign stale/disconnected worker ranges

5. **Key Recovery**
   - On collision: `private_key = (d_tame - d_wild) mod n`
   - Verify against known public key
   - Alert all workers, claim prize

### Language Options for Server

| Language | Pros | Cons |
|----------|------|------|
| **C++** | Matches client, fastest | More complex |
| **Python** | Fast development, easy | Slower (but I/O bound anyway) |
| **Rust** | Safe, fast | Learning curve |
| **Go** | Good concurrency | Different ecosystem |

**Recommendation**: Python for rapid prototyping, C++ if performance issues arise.

---

## Puzzle #135 Technical Details

- **Bit size**: 135 bits
- **Range**: `0x4000000000000000000000000000000000` to `0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF`
- **Public Key**: `02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16`
- **Expected Operations**: O(2^67.5) ≈ 2.1 × 10^20 group operations
- **With 4x RTX 5090 (~50 GKeys/s)**: ~130 years solo
- **With 100 GPUs pooled**: ~5 years
- **With 1000 GPUs pooled**: ~6 months

---

## Resources & References

- **JLP Kangaroo**: https://github.com/JeanLucPons/Kangaroo
- **Keyhunt CUDA**: https://github.com/WanderingPhilosopher/Keyhunt-cuda
- **btcpuzzle.info**: https://btcpuzzle.info/
- **Puzzle status**: https://privatekeys.pw/puzzles/bitcoin-puzzle-tx
- **BitcoinTalk thread**: https://bitcointalk.org/index.php?topic=1306983.0

---

## Next Steps

1. [ ] Choose VPS provider and provision server
2. [ ] Implement pool server (Python prototype)
3. [ ] Test with local workers
4. [ ] Deploy and connect remote workers
5. [ ] Monitor DP collection rate and collision progress
6. [ ] (Optional) Add web dashboard for stats
