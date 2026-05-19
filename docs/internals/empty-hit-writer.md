# EmptyHitWriter design rationale

`EmptyHitWriter` is the background-thread writer for the `--track-empty-hits`
brain-wallet path. See `src/runtime/empty_hit_writer.hpp` for the public API.

## Why a background thread

The original inline `log_empty_hit` helper in `brain_wallet_runner.cpp` opened
`found-empty.txt` (std::ofstream, append mode), formatted the line via
`snprintf`, and let the destructor close the file: three syscalls per hit.
With a heavy bloom (millions of empty-positive candidates per minute) those
syscalls land on the scan thread and each one stalls the brain-wallet
dispatch loop for the duration of the kernel-mode round trip (10-50 us per
hit measured on Windows NTFS with a hot file handle in the OS cache).

`EmptyHitWriter` moves that work off the scan thread. The producer (the
brain-wallet runner's hit-handling path) calls `enqueue()`, which under a
single mutex pushes one record onto a bounded `std::deque` and notifies a
writer thread. The writer thread holds the `std::ofstream` open for the
lifetime of the scanner, drains the deque in 100 ms cycles (or sooner when
notified), formats the lines, writes them in one batch, and flushes once per
drain.

## Bounded queue policy

Capacity is 10000. On overflow, the OLDEST record in the queue is dropped
(so the producer is never blocked by a slow disk and the most recent N hits
are always preserved). A single one-line warning is emitted to `std::cerr`
the first time overflow happens per writer-lifetime to surface the condition
to the operator without spamming on sustained overflow.

## Lifetime

Construct one `EmptyHitWriter` alongside the scan-loop locals (after bloom
load + before the inner loop). Pass a reference to every hit site. The
destructor calls `stop()`, which signals the writer thread to drain
remaining records and then joins. Explicit `stop()` is exposed too for
callers that want to deterministically flush before any subsequent
reporting.

## Thread-safety contract

- `enqueue()` is safe to call from any thread.
- `stop()` must be called from at most one thread (it joins the writer
  thread). The destructor calls `stop()` so the typical "scoped at the
  scan loop" use is automatically safe.
- Concurrent `enqueue()` during `stop()` is allowed; any record enqueued
  before `stop()` observes `is_stopped_` is flushed. Records enqueued
  after that race is lost. (The brain-wallet scan path always drives
  `stop()` from the scan thread AFTER the last `enqueue`, so the race
  window does not occur in the production caller; the tests exercise it.)

## Why std::deque + mutex + condvar (not a lock-free queue)

- The producer's critical section is two pointer writes plus a
  `notify_one`, ~30-50 ns under contention. The scan-loop thread spent
  ~50 us per hit on the prior file-open path, so even a slow mutex
  acquisition is a 100-1000x improvement.
- Lock-free SPSC ring buffers would be marginally faster on the producer
  side but require a fixed-size pre-allocation that cannot drop-oldest
  cleanly without further machinery.
- `std::deque` allows arbitrary `push_back` / `pop_front` in O(1) so
  drop-oldest is a single `pop_front`. The bounded behaviour is enforced
  by checking `size()` before `push_back`.

## Stream failure handling

A one-shot warning latch is flipped to true the first time the bounded
queue drops a record so the warning is emitted once, not per drop.

Without further care, a transient EIO / ENOSPC that flips `out_`'s failbit
would leave the writer silently dropping every subsequent record because
the stream stays in fail state until `clear()` is called and the
queue-overflow warning never fires (only triggered by bounded-queue
overflow). The writer now clears the failbit and retries the line; if the
retry also fails we log ONCE so the operator sees the disk problem, then
continue dropping silently until either the bad-write retry succeeds
(clearing the latch) or `stop()` returns.

A step counter drives periodic re-attempt: re-clear + retry every
`kStreamRetryInterval` consecutive bad-write events so a recovered stream
(e.g. disk space freed) starts writing again without requiring an operator
restart.

## Output format

`format_line` matches the prior line format exactly:
`"<ts>  privkey=<hex>  h160=<hex>  passphrase=\"<raw>\"\n"`. The passphrase
is written raw (the prior `log_empty_hit` also did not escape it; preserving
that means any output-consuming tooling stays unchanged).

The ISO-8601 UTC timestamp is formatted at `enqueue()` time on the scan
thread. The writer thread does not call `gmtime` / `strftime`; doing so on
the writer would either re-introduce timing jitter on the file (multiple
records would share the same second) or push formatting back onto the
producer side anyway. Keeping the timestamp pre-formatted is also what the
existing `log_empty_hit` body did, so output is byte-identical.
