# RuntimeControlState design rationale

`RuntimeControlState` is the all-atomic snapshot of operator-requested
tuning state coming in from the TUI keyboard input thread. The scan loop
reads it at batch boundaries (and a few phase boundaries for the bloom
hot-swap path) and acts on each request at the next safe checkpoint. See
`src/runtime/runtime_control.hpp` for the public API.

## Threading model

- Writer side: `ui::tui::InputHandler` (one thread, processes FTXUI
  keyboard events; mutates the atomics + the `bloom_path` / `banner_text`
  mutex-protected strings).
- Reader side: `brain_wallet_runner.cpp` scan loop (single producer of
  counter state; consumes runtime control at every `BatchScope`
  iteration).
- Aux reader: `TuiApp` render thread reads `requested_theme_variant` +
  the `banner_text` / `banner_set_at` pair for status display only.

All numeric fields are `std::atomic` so cross-thread visibility is
guaranteed without further synchronization. Strings live behind small
mutexes because `std::string` is not atomically swappable on Windows
MSVC even at `-O2`.

## Request / applied contract

The contract for "requested vs applied" counter pairs is:

- `requested_*` fields are SET BY THE WRITER. A value of 0 (or empty
  string, or `-1` for theme) means "no change requested".
- `last_applied_*` fields are SET BY THE READER (scan loop) AFTER the
  requested change actually took effect. The scan loop also clears the
  `requested_*` field back to 0 so a single keypress yields exactly one
  applied change.
- For one-shot requests (`save_requested`, pause toggle) the convention
  is "writer sets to true; reader clears to false after acting".

## Singleton

There is exactly ONE `RuntimeControlState` instance per scan process,
accessed via `global_runtime_control()`. The lifetime is the entire
program. The accessor uses a function-local static so the first caller
initializes (C++11 thread-safe initialization) and all subsequent
callers see the same instance.

## Field-level notes

### Lifecycle (`quit_requested`)

Set by the `q` keypress (and the SIGINT handler path inside
`CookedModeGuard`). The scan loop polls this at every batch boundary
and flips `g_shutdown` when it sees true. The reader does NOT clear
this flag; it is a one-way latch from "running" to "shutting down".

### Pause / resume

`pause_requested` is toggled by `p`. While true, the scan loop drains
its in-flight GPU dispatches at the next batch boundary and then sets
`is_paused=true`. While paused, the scan loop spins reading
`RuntimeControlState` (so `save_requested` / `quit_requested` / GPU
toggles still work) until `pause_requested` goes false, at which point
it clears `is_paused` and resumes normal dispatch.

### Save now

Toggled by `s`. The scan loop bypasses its `save_interval` throttle on
the next batch boundary, persists the current streaming-gen state, and
clears the flag.

### GPU enable mask + per-GPU phase

The TUI keybind `g<N>` flips bit N of `gpu_enable_mask`. The scan loop
reads the mask and `gpu_phase[i]` at every batch boundary:

- mask bit cleared + `gpu_phase[i] == Active` -> transition to
  `Draining`, sync the GPU's stream, call `drain_and_free()` on the rule
  engine + brain-wallet context for that device, set
  `gpu_phase[i] = Disabled`.
- mask bit set + `gpu_phase[i] == Disabled` -> transition to
  `Initializing`, re-init the rule engine + brain-wallet context with
  the original config, set `gpu_phase[i] = Active`.

The dispatch path skips any GPU whose phase is not `Active`. Up to
`kMaxGpus` (8) devices supported; matches the existing GPU detection
ceiling in the rest of the codebase.

### Batch size tuning

The TUI `+` / `-` keys nudge `requested_batch_size`. The scan loop reads
it at every batch boundary; if it differs from the current batch size
it tries to reallocate the GPU buffers at the new size and, on success,
sets `last_applied_batch_size` + clears the request. On `cudaMalloc`
failure it logs a banner and clears the request without touching
`last_applied` (i.e. the previous size stays in effect).

### Rule chunk size cycle

The `r` key cycles through the chunk-size values `{200, 500, 1000}`.
The scan loop interprets a non-zero `requested_rule_chunk_size` as
"re-init the rule engines with this `max_rules` at the next batch
boundary"; on success it sets `last_applied` + clears the request.

### Bloom hot-swap (phase-boundary swap, not batch-boundary)

The bloom picker modal writes a target path into `requested_bloom_path`
under `bloom_mu`. The scan loop polls at every PHASE boundary (not
batch boundary; in-flight bloom probes must complete first) and
performs a full drain + per-GPU reload. On success it copies the path
into `last_applied_bloom_path` + clears `requested_bloom_path`; on
failure it logs a banner and clears the request.

### Wordlist hot-swap (phase-boundary swap)

The wordlist picker modal writes a target profile path into
`requested_wordlist_profile` under `profile_mu`. The scan loop polls at
every PHASE boundary (not batch boundary) and forwards the new profile
to `StreamingBrainWallet::queue_profile_swap`, which in turn applies
the swap at the next phase advance. On success the runner copies the
path into `last_applied_wordlist_profile` and clears
`requested_wordlist_profile`; on failure it banners the operator and
clears the request without touching `last_applied`.

Mirrors the bloom path pattern above so the input handler and the scan
loop can use the same one-shot writer / consumer convention.

### Theme cycle (consumed by TUI render thread, not scan loop)

The `t` key cycles `Default -> HighContrast -> Monochrome -> Light ->
Default`. `-1` means "no change"; `0..3` are the four variant indices
matching `collider::ui::tui::ThemeVariant`. The TUI render thread is
the only reader; the scan loop ignores this field.

### Focused-panel mode (render thread only)

The `Ctrl+1..4` keys focus a single panel; `Ctrl+0` / `Esc` clears.
Encoding:

- `-1`: no focus (default 2-column layout)
- `0`: status panel
- `1`: GPU panel
- `2`: performance panel
- `3`: plugins panel

The TUI render thread is the only reader; the scan loop ignores this
field. The `Ctrl+N` input dispatch is the writer; the render thread
reads at every frame to decide layout. Cleared back to `-1` when `Esc`
/ `Ctrl+0` is pressed.

### Banner message

The scan loop sets a short banner string after any user-action change
("Batch size set to 6M", "GPU 0 disabled", "Bloom reloaded: funded.blf").
The TUI status panel reads + dim-renders this for ~5 seconds after
`banner_set_at`, then clears.
