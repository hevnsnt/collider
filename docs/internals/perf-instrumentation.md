# PerfCollector design rationale

`PerfCollector` is the CUDA-event-based kernel timing collector for the
fused brain-wallet pipeline. It drains async `cudaEvent`s, computes
elapsed time per kernel launch, and aggregates into log-spaced histograms
suitable for the performance panel. See `src/runtime/perf_instrumentation.hpp`
for the public API.

## Zero-overhead-when-off contract

The `g_enabled` atomic flag gates every instrumentation entry point. When
false (default), the prologue of `record_start` / `record_stop` is a
single relaxed atomic load + branch; nothing else executes, no
`cudaEvent` is created, no allocation occurs, no lock is taken. The
kernel-launch hot path therefore pays one predicted-not-taken branch per
instrumented call site and nothing more. Verified by inspection in
`perf_instrumentation.cpp`.

When true (perf panel open), `record_start` picks the next free slot from
a fixed per-kernel ring of 256 `cudaEvent` pairs and records the start
event on the supplied stream. `record_stop` records the matching stop
event. `drain_pending()` polls completed pairs, converts to microseconds,
and feeds the result into the kernel's stats + log-spaced histogram. The
collector owns the `cudaEvent`s; they are created lazily on first enable
and reused.

## Concurrency model

### Producer side (record_start / record_stop)

MULTI-producer per kernel. `record_start` atomically reserves a ring slot
via `fetch_add` and returns a `PerfToken` holding `(KernelId, slot)`. The
caller threads the token through to the matching `record_stop`. There is
NO shared "last_reserved" field; every in-flight pair is identified by
the token, which lives in the caller's stack. This is naturally
thread-safe: thread A reserving slot N and thread B reserving slot N+1
cannot collide because each `record_stop` carries the exact slot to
release.

The token-based API replaced an earlier single-producer "remember
last_reserved on the kernel record" design. That design assumed the scan
loop launched kernels serially per `KernelId`; under multi-GPU the
dispatch loop launches the SAME `KernelId` from multiple worker threads
concurrently, which made `record_stop` on thread A read the slot that
thread B reserved. Result: slot leaks (the thread-A pair was never
released) and cross-stream `cudaEventElapsedTime` returning noise (the
start event came from thread A's stream, the stop from thread B's). The
token-based API eliminates the shared mutable state.

### Drain side (drain_pending)

The perf panel sampler. Runs at sub-Hz, acquires the collector mutex
once per pass, folds completed `cudaEvent` pairs into the per-kernel
aggregator (`sample_count`, `sum_us`, min/max, log histogram), and bumps
`seq` once.

### Snapshot side (snapshot)

Any number of readers. Acquires the collector mutex briefly. Lock-free
was considered but ruled out: the protected aggregator state is
non-atomic doubles and uint64 arrays, and a seq-lock over non-atomic
memory is UB under the C++ memory model. The reader is called at sub-Hz
from the perf panel; mutex contention is bounded. The `seq` counter
remains a publication marker so consumers can detect "no change since
last snapshot" cheaply without comparing every field.

## Stream parameter

`record_start` / `record_stop` accept a `void*` stream rather than
`cudaStream_t` so this header stays buildable in non-CUDA translation
units (e.g. the CPU-only or Metal backend builds, and the synthetic-feed
test). The `.cu` caller passes its `cudaStream_t` directly; the cast is
internal to the `.cpp`.

## Kernel identifiers

The order in the `KernelId` enum is the OUTPUT order of any snapshot.
`RuleApply` lives in `src/gpu/gpu_rule_kernel.cu` (not this file's
instrumentation scope); the slot is reserved so a later phase can
instrument it without breaking the snapshot shape.

## Ring overflow

On ring overflow the new sample is dropped and `skipped_count` is
incremented (no exception, no crash; one stderr warning is emitted once
per kernel ring as a hint).

## Process scoping

The collector is a process-wide singleton. Instrumentation is
process-scoped: every kernel launch from any thread / GPU funnels into
the same collector so the perf panel sees a unified view. Per-GPU
disaggregation lives at a higher layer (the GPU panel) and is out of
scope here.
