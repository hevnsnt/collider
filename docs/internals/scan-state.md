# ScanState design rationale

`ScanState` is the atomic counter container for the brain-wallet scan loop
with a lock-free seq-lock snapshot pattern, so the TUI render thread (or
any other reader) can take a consistent multi-counter snapshot while the
scan loop writes counters in any order within a batch. See
`src/runtime/scan_state.hpp` for the public API.

## Memory ordering contract

### Writers (the scan loop, single producer thread per ScanState instance)

- At the START of every batch, call `begin_batch()`. This bumps `seq` to
  an ODD value with release semantics, signaling readers "writer is
  mid-batch, counters may be partially updated". The default-constructed
  `seq` is 0 (even / quiescent) so the very first snapshot before any
  `begin_batch()` call sees a consistent zero-everywhere state.
- During the batch, call `inc_xxx()` / `set_xxx()` in any order. These
  use `memory_order_relaxed` because intra-batch counter ordering does
  not matter; the snapshot only needs to be consistent at batch
  boundaries and the seq-odd state forces readers to retry past any
  relaxed-reorder window.
- At the END of every batch call `commit_batch()`. This bumps `seq` from
  ODD to the NEXT EVEN value with release semantics. Both bumps together
  bracket the batch's writes in a true seq-lock.

### Readers (the render thread, or any number of other reader threads)

- Call `snapshot()` to obtain a consistent `ScanSnapshot`. Internally,
  `snapshot()` spins until it observes an EVEN `seq` with no mid-read
  change: read `seq_before`, if odd retry; read each counter with
  relaxed; read `seq_after`, retry if `seq_after != seq_before`. The
  retry budget is bounded at `kMaxRetries`; on exhaustion, `snapshot()`
  returns the last-attempted read to guarantee bounded latency. A reader
  is never blocked.

### One-shot setters (pre-loop initialization)

- `set_total_checked()` is provided for the resume baseline path, which
  seeds the counter with the streaming generator's lifetime value before
  the scan loop begins. It is NOT safe to call concurrently with
  `snapshot()`; it must run before any reader thread starts.
- `set_current_phase()` is similarly safe to call before the loop starts
  to seed the published phase. See PHASE COUNTERS below.

The contract guarantees readers see snapshots that satisfy every
cross-counter invariant the writer establishes during a batch (e.g.
`bloom_collisions_filtered >= tight_bloom_filtered`,
`sum(phase_keys_processed) <= total_checked`,
`sum(empty_hits_by_phase) <= real_empty()`).

## Phase counters

Per-phase keys-processed totals and per-phase empty-hit totals live
alongside the global counters and are published by the same
`commit_batch()` seq bump. The current_phase index + phase_iteration are
also stored as atomics and read under the same seq-lock window so a
snapshot's `current_phase` ALWAYS matches the counter values committed in
the same batch.

Phase index domain: `0..kNumPhases-1` (5 phases: Quick Wins, Crypto Focus,
Extended, Combinator, Deep Dive). Out-of-range phase indices passed to the
per-phase writers are SILENTLY DROPPED (no crash, no exception); this
matches the relaxed contract of the rest of the surface and avoids
exception paths in the hot loop. A debug assert in development builds
catches obvious bugs.

## PhaseRateTracker

`PhaseRateTracker` is the host-side derived rolling-window keys-per-second
accessor for each phase. NOT thread-safe; owned by a single consumer (the
TUI render thread). The producer never touches this. It only writes raw
`phase_keys_processed` totals through `ScanState` and the render thread
samples those totals via `snapshot()` and feeds them here.

Ring buffer per phase. Each entry: `(timestamp_ms, phase_keys_processed_value)`
captured at `sample()` time. `keys_per_sec(idx, window_ms)` returns the
simple slope `(newest - oldest_within_window) / wall_dt` over the
requested window. A tracker holding fewer than 2 samples for that phase,
or seeing zero wall time across the window, returns 0.0.
