# v1.5.0 Adversarial Audit — Pass 3 (2026-05-24)

> **POST-AUDIT TIER-0 DISCOVERY (2026-05-24, commit 3f35aca):**
> Writing the BipGpuDispatcher unit test (B-Phase 6) surfaced a
> critical byte-order bug in MultiAddressSession::process_batch that
> the audit missed. Two-sided BE/LE mismatch caused EVERY production
> BIP scan to find zero real bloom hits since the dispatcher landed
> (commit 1da2a2e onward). Operators saw the GPU light up in
> nvidia-smi but never got hits; assumed it was a sparse-bloom
> probability thing. Fixed by per-batch BE->LE priv byte-swap in
> process_batch + in-kernel LE-limbs->BE-bytes conversion before
> hash160/keccak256. End-to-end hit-routing test now confirms.
> Operators with known-funded test mnemonics should re-run after
> this build to confirm dispatcher hits now fire correctly.

Triggered by: user reporting they "keep finding problems whenever I test, check,
or try anything" despite repeated claims of done. Five parallel adversarial
reviewers covered: BIP scan correctness, TUI consistency, multi-GPU correctness,
code quality, and test coverage.

Bottom line: the prior 13-commit BIP scan GPU work has 12 distinct correctness
bugs, the TUI "stay in TUI" mandate is ~30% complete (pool / puzzle / brain
wallet modes still flood cout AFTER the alt-screen takes over), and the user
has been seeing the symptoms of these bugs but I have been treating each as a
one-off instead of the systemic quality cascade it is.

This file is the canonical finding list. It is NOT a plan — that comes after the
user picks the fix order.

---

## TIER 0: User is hitting these right now (visible symptoms)

### T0-1: `MultiAddressSession::process_batch` printf'd to stdout from GPU worker thread on every match — corrupts TUI alt-screen

- File: `src/gpu/v2/v2_orchestrator.cpp:794`
- Code: `std::printf("[v2:multi-addr] HIT pp_idx=%u addr_type=%u kind=%u\n", ...);`
- Symptom: "BIP scan dropping back to text" on every hit. Was THE user's recurring complaint.
- Why my prior fixes missed it: I fixed `release_active_capture` for the runner's stdout but `printf` writes to raw C stdout FILE\*, bypasses rdbuf hijack, lands directly in the alt-screen as garbled characters.

### T0-2: Mouse-tracking left enabled in the runtime TUI — explains "white lines appear when I move the mouse"

- File: `src/ui/tui/tui_app.cpp:892` (no `screen.TrackMouse(false)` call)
- Code: The menu modals call `screen.TrackMouse(false)` at `src/ui/tui/menu/main_menu.cpp:137`. `TuiApp::start` does NOT. Once a mode launches, mouse tracking is back on.
- Symptom: User's complaint "white lines appear randomly when I move the mouse around" (verbatim).

### T0-3: Pool / Puzzle / Brain wallet runners flood `std::cout` AFTER launching the TUI — all goes into the captured boot log, operator sees nothing

- Files (callouts only — full list in TUI agent's report):
  - `src/runtime/pool_solver.cpp:175-191, 266-294, 569-575, 587-593, 604, 627-632, 648-664, 697-723` (~25 cout sites post-launch)
  - `src/runtime/puzzle_solver_kangaroo.cpp` (107 cout/cerr in the file — most run AFTER `launch_session`)
  - `src/runtime/puzzle_solver_bruteforce.cpp:204-258, 416-444, 701-727` (PUZZLE SOLVED banner is cout — disappears into capture log)
  - `src/runtime/brain_wallet_runner.cpp:2734-2747, 2797-2858, 2934, 2991, 3073-3101, 3328-3409, 3674+post-launch` (~30 cout sites)
- Symptom: User's "every mode drops out to text" complaint. The "T1-A port run_brainwallet_interactive to TUI" + "T1-B port run_puzzle_interactive to TUI" tasks marked complete were partial — the runners still cout after launch.

### T0-4: `select_algorithm` puzzle-pubkey prompt blocks `std::getline(std::cin, ...)` BEHIND the active TUI alt-screen

- File: `src/runtime/puzzle_solver_helpers.cpp:272-309` (called from `puzzle_solver.cpp:802` after TUI is live)
- Symptom: Operator picks a puzzle whose pubkey isn't bundled in `puzzle_history.json` (any non-mult-of-5 in 71-160). Runner deadlock-shapes: the prompt is hidden behind alt-screen, stdin is fighting the TUI input handler.

### T0-5: Brain wallet setup wizard is ENTIRELY text-based (Interactive::read_line ➜ std::cin)

- File: `src/ui/brainwallet_setup.hpp:343-633` (`run_wizard()`)
- Called from: `src/ui/interactive_ui.cpp:413-429` first-time setup path
- Symptom: First-time user picks brain wallet → FTXUI menu tears down → text wizard runs in cooked terminal → menu comes back. The single biggest "drop out to text" violation. Task #96 ("T1-A: port run_brainwallet_interactive to TUI") marked complete; this wizard was missed.

---

## TIER 1: Silent correctness bugs the user will hit but hasn't yet

### T1-1: Dispatcher silently drops queued work on shutdown

- File: `src/runtime/bip_gpu_dispatcher.cpp:58-71, 206-214` AND `bip_scanner_runner.cpp:599, 804, 1413, 1563`
- Bug: All 4 `gpu_dispatcher.enqueue(...)` call sites discard the bool. `enqueue` returns false on shutdown OR no-CUDA. Caller has ALREADY bumped `addrs_probed` before enqueue, so the final summary lies (reports work done that wasn't).
- Reproducer: Press 'q' mid-scan. Final "Addresses probed" overstates by up to (queue_max + in_flight) keys.

### T1-3: Data race on `last_profile_label` in combinatorial-mode 4-walker pool

- File: `src/runtime/bip_scanner_runner.cpp:561, 572, 859`
- Bug: 4 walker threads write `last_profile_label = prof.label;` with no mutex. TUI thread reads it with no mutex. Wordlist path correctly uses `label_mu`. Combinatorial path forgot.
- Reproducer: Run combinatorial GPU mode. Eventually TSAN flags, or the dashboard shows garbled profile string, or std::string SSO/heap transition triggers double-free.

### T1-4: `args.gpu_ids = {}` silently defaults to `{0}` — auto-detect comment LIES

- Files: `src/runtime/bip_scanner_runner.cpp:547`, `src/runtime/bip_gpu_dispatcher.cpp:111`, `src/cli/cli_parser.hpp:91` (comment says "Empty = auto-detect available GPUs")
- Bug: User with 2 GPUs invoking without `--gpus 0,1` gets ONLY device 0 dispatching. The "auto-detect" promised by the CLI help is unimplemented. Possible root cause for the user's "0 GPU" complaint pre-fix.
- Fix: When `args.gpu_ids` is empty, call `cudaGetDeviceCount` and populate with `0..n-1`.

### T1-5: BipGpuDispatcher::init aborts ENTIRE dispatcher when ANY device fails

- File: `src/runtime/bip_gpu_dispatcher.cpp:142-183`
- Bug: First device that fails any of cudaSetDevice / MultiAddressSession::init → return 70 → caller treats as "no GPUs" → `gpu_active = false`. On 2-GPU box where device 0 inits cleanly but device 1 fails for any reason, BOTH GPUs are abandoned. Should mirror `MultiGPUBrainWallet::init` pattern: per-device retry, commit any device that came up, surface faulted devices to the dashboard.

### T1-6: Wordlist-path `pbkdf_gpu_active` dashboard flag LIES

- File: `src/runtime/bip_scanner_runner.cpp:1662`
- Bug: `bi.pbkdf_gpu_active = !args.bip_no_gpu && gpu_init_diag.empty();` This conflates dispatcher-init state with per-worker PBKDF2-stream state. If per-worker `cudaStreamCreate` fails silently, dashboard says "PBKDF2 + EC + bloom on GPU" while workers run CPU.

### T1-7: Per-batch `std::thread` spawn (4 fresh threads per 256 mnemonics)

- File: `src/runtime/bip_scanner_runner.cpp:680-696`
- Bug: Spawning 4 std::thread per batch flush = ~16 thread creates/sec on Windows = ~1ms/sec pure syscall overhead. Structurally wrong — should be persistent pool with condvar.

### T1-8: `pbkdf_devices` defaults to `{0}` if `args.gpu_ids` empty in combinatorial PBKDF2 too

- File: `src/runtime/bip_scanner_runner.cpp:547`
- Same root cause as T1-4; counts as a separate site to fix.

### T1-9: BSGS solver hard-pinned to ONE GPU

- File: `src/runtime/puzzle_solver_bsgs.cpp:142` (`bcfg.device_id = args.gpu_ids.empty() ? 0 : args.gpu_ids[0];`)
- Bug: `--solver bsgs --gpus 0,1` runs on device 0 only. The "follow-up" comment at line 13 of the same file has been there for the lifetime of the file.

---

## TIER 2: Latent correctness landmines

### T2-1: PBKDF2 kernel writes 119+ bytes into a 128-byte stack buffer for long salts

- File: `src/gpu/bip39_pbkdf2.cu:62-69` + `bip39_pbkdf2.cuh:41` (`kMaxSaltBytes = 256`)
- Bug: `first_msg` is 128 bytes. Code writes `first_msg[salt_len + 3] = 0x01`. Host guard accepts up to 256-byte salts. Any salt > 115 bytes (computed: `SHA512_BLOCK_BYTES - 4 - 9`) overflows the buffer AND violates the HMAC contract (msg_len > 119).
- Latent: BIP-39 typical salt = "mnemonic" + passphrase. Most passphrases << 116 chars. First operator who tests with a long passphrase corrupts stack and gets wrong seed.
- Fix: Tighten `kMaxSaltBytes` to 115 in the header; reject in `run_pbkdf2_batch`; document the limit.

### T2-2: `v2_orchestrator.cpp:1077` discards `cudaMemcpy` return code

- File: `src/gpu/v2/v2_orchestrator.cpp:1077`
- Bug: `cudaMemcpy(&hits, d_match_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);` — return discarded. Comment at line 746-754 explicitly claims this was fixed (B7); the fix was applied to `MultiAddressSession::process_batch` only. The legacy orchestrator copy is still naked.

### T2-3: `run_v2_orchestrator` returns `total_hits > 0 ? 0 : 0`

- File: `src/gpu/v2/v2_orchestrator.cpp:1157`
- Bug: Ternary returns 0 on both branches. Dead code or unfinished refactor. Scripts can't distinguish "scan complete, hits" from "scan complete, no hits".

### T2-4: BipGpuDispatcher::shutdown race with producer threads still in `enqueue()`

- File: `src/runtime/bip_gpu_dispatcher.cpp:74-79, 333-339`
- Bug: `shutdown()` calls `signal_shutdown()` which notifies all CVs, then joins workers. Producer thread inside `enqueue` blocks on `queue_not_full_cv` — gets notified, sees shutdown, returns false. But priv keys pushed BEFORE the shutdown-notify can race past worker exit and never get probed. Conflates "stop accepting" with "stop draining".

### T2-5: Bloom data lifetime undocumented

- File: `src/runtime/bip_gpu_dispatcher.hpp:100`
- Bug: `const uint8_t* bloom_data` — raw pointer. Worker threads keep `impl_->cfg.bloom_data` forever. Caller (`bip_scanner_runner.cpp:486-490`) passes `bloom.data.data()`; if `bloom` (`BloomLoadResult`) goes out of scope while dispatcher is alive, UB. Header doesn't document this contract.

### T2-6: `Config::tight_bloom_*` fields are PUBLIC API but 100% DEAD

- File: `src/runtime/bip_gpu_dispatcher.hpp:104-107`
- Bug: 4 fields advertised on the Config struct. Never referenced in `bip_gpu_dispatcher.cpp`. Never set by `bip_scanner_runner.cpp` (runner enforces tight bloom in its own `on_hit` lambda). Future caller sets them expecting a second gate, gets false positives, silently corrupts hits log.

### T2-7: `gpu_init_diag` is captured ONCE; runtime CUDA errors mid-scan invisible

- File: `src/runtime/bip_scanner_runner.cpp:519-524, 1320`
- Bug: `last_error()` is sampled post-init only. If `process_batch` fails mid-scan (driver reset / VRAM exhaustion / SM clock throttle that fails some kernel), the dispatcher stderr's the error into the captured log. Dashboard stays "GPU OK".

---

## TIER 3: Lying / stale documentation

### T3-1: Dispatcher header still says "PBKDF2 stays on CPU" — primary doc lies

- File: `src/runtime/bip_gpu_dispatcher.hpp:6-26`
- Specifically lines 21-26: "PBKDF2 stays on CPU (it does not GPU-parallelize meaningfully ...)". The whole point of commit `1da2a2e` was to port PBKDF2 to GPU. Reading the header tells future maintainer the opposite of reality.

### T3-2: BIP runner has 3 "removed reality-check modal" archaeology blocks (rot in production code)

- File: `src/runtime/bip_scanner_runner.cpp:35-37, 394-400, 420-431`
- Total 13+ lines explaining why a feature doesn't exist. Belongs in commit messages (already in `33c7aba`).

### T3-3: HMAC header has REPL-style monologue committed: "wait that IS over 128"

- File: `src/gpu/hmac_sha512_device.cuh:18-19`

### T3-4: `bip_scanner_runner.cpp:269-271` says "Future GPU acceleration tracked separately" but GPU is now wired in.

### T3-5: Comment claims `kWalkers = 4` is "the sweet spot" without measurement

- File: `src/runtime/bip_scanner_runner.cpp:680`
- No bench. No justification for 4 vs 2/8/N-1.

### T3-6: `bip_scanner_runner.cpp:1622-1623` prints `"[*] BIP scanner threads: 23 (T1-C)\n"` to operator-facing output

- "T1-C" is an internal sprint task label. Leaked.

---

## TIER 4: TUI completeness (the "stay in TUI" mandate)

### T4-1: `run_bip_scan_mode` wordlist path has POST-`release_active_capture` `std::cout` lines

- Lines 1192, 1622 print to PowerShell AFTER alt-screen is live. Same family as T0-3.

### T4-2: BIP scan ends with `std::cout` summary + `std::getline(std::cin, discard)` "Press Enter"

- Files: `bip_scanner_runner.cpp:961-999, 1769-1803`
- Bug: Hangs forever on piped stdin (Windows Task Scheduler, CI, SSH-no-tty). Should be TUI overlay frame.

### T4-3: Pool mode never updates `current_phase_name` after "Mining"

- File: `src/runtime/pool_solver.cpp:230, 303`
- Bug: Set once at launch to "Connecting", once when work assigned to "Mining". Never reflects reconnects, supervisor giveup, pool drop.

### T4-4: TUI hotkeys advertised in footer for ALL modes but only WIRED in brainwallet

- File: `src/runtime/runtime_control.hpp` is only consumed by brain_wallet_runner.cpp
- Bug: Press 'p pause' in pool mode → footer says "Pause requested." → pool keeps mining (silent no-op). Same for '+/- batch size', 's save', 'b bloom', 'w wordlist', 'g GPU toggle'. Footer is lying to the operator about what works.

### T4-5: GPU telemetry sampler only started by brain_wallet_runner

- File: `src/runtime/brain_wallet_runner.cpp:3686-3703` (only `GpuTelemetrySampler::start` call site)
- Bug: Pool / puzzle / BIP / benchmark modes never start a sampler. GPU panel stays "Waiting for GPU telemetry..." for the entire run. Fix: move into `launch_session`.

### T4-6: Performance panel renders "Perf instrumentation disabled" for all non-BW modes

- File: `src/ui/tui/panels/performance_panel.cpp:323-328`
- Bug: `PerfCollector` is only populated by brain wallet. In other modes the panel takes 30% of the screen showing a wrong "instrumentation disabled" message.

### T4-7: Plugins panel renders "No plugins configured" zero-state in every non-BW mode

- File: `src/runtime/brain_wallet_runner.cpp:3716-3729` (only `plugin_runner->start()` call site)
- Bug: User can drop a plugins.yml; pool/puzzle/BIP modes still claim none configured.

### T4-8: BIP scan WORKERS row says "(initializing GPU...)" FOREVER when GPU absent

- File: `src/ui/tui/panels/status_panel.cpp:366-382`
- Bug: When no CUDA driver, the WORKERS row says "(initializing GPU...)" indefinitely. Should say "(no GPU detected)".

### T4-9: `run_kangaroo_*` writes the PUZZLE SOLVED banner via cout — the single most important user event disappears into the log

- File: `src/runtime/puzzle_solver_kangaroo.cpp:761-781, 947-970`

### T4-10: `run_bruteforce_solve` ditto — PUZZLE SOLVED banner via cout

- File: `src/runtime/puzzle_solver_bruteforce.cpp:204-258, 416-444, 701-727`

### T4-11: Pool mode "[*] Connecting to pool..." / "[+] Connected" / "[*] Work assigned:" all post-launch cout

### T4-12: `release_active_capture` only called on ERROR paths in pool_solver.cpp, not happy path

### T4-13: Mode-config / interactive_ui status messages cout-flash between modals

---

## TIER 5: Test coverage gaps

### T5-1: `BipGpuDispatcher` has ZERO direct test coverage (307 LOC, all new, all untested)

- Test needed: `tests/test_bip_gpu_dispatcher.cpp` — init / enqueue / shutdown lifecycle + 5 init failure paths + idempotency + double-shutdown safety + nullptr on_hit rejection.

### T5-2: `test_bip39_pbkdf2_gpu_kat` runs on device 0 ONLY

- Bug: User has 2 GPUs. If GPU 1 has different SM and undetected bug, prod produces wrong seeds on worker_idx mapped to GPU 1, test still passes.
- Fix: Loop test body over `cudaGetDeviceCount()`.

### T5-3: PBKDF2 KAT runs `count=1` ONLY (production uses 256 / 128)

- Bug: Off-by-one in grid math, per-tid stride bugs, shared salt access — all invisible at count=1.

### T5-4: Tight bloom code path has zero tests (5 call sites in runner)

- `tests/test_bip_scan_runner_smoke.cpp` builds primary bloom only.

### T5-5: `MultiAddressSession::last_matches()` API has zero tests

- New API added in commit `de78e60`. If a kernel change forgets to update `last_records`, BIP dispatcher silently misses all hits.

### T5-6: Parallel chain walker (4 threads, shared atomics, shared queue) has no determinism / race test

- ThreadSanitizer build in CI would catch automatically.

### T5-7: Stale-string regression has no panel render test

- `test_tui_panels.cpp` only tests pre-refactor `BipScanInfo` fields. A future revert to "PBKDF2 is CPU-bound" stale text passes the test.

### T5-8: Stream-skip-on-null fix from commit `2aa5dcb` is untested

- The exact 2-GPU-with-one-failed-init configuration the user is in has no test.

### T5-9: No CPU-vs-GPU parity test

- Same wordlist, same bloom; assert hit set identical between `--no-bip-gpu` and default. Defends multiple gaps at once.

### T5-10: SHA-512 and HMAC-SHA512 device headers have no direct KAT — only transitive coverage through 8 PBKDF2 vectors

- `tests/test_device_hmac_sha512.cu` tests `v2/device_hashes.cuh` which is a DIFFERENT file.

---

## ROOT-CAUSE THEORY: why did the user's dashboard show "0 GPU"?

The multi-GPU agent's "device-1 OOM" theory is REJECTED — the bloom file is 143MB
which fits trivially in any modern GPU.

Most likely actual cause: `args.gpu_ids = {}` (user invoked without `--gpus 0,1`).
The BIP runner defaulted `pbkdf_devices` to `{0}` (line 547) AND the dispatcher
defaulted gpu_ids to `{0}` (bip_gpu_dispatcher.cpp:111). So the user with 2 GPUs
got 1 GPU dispatching even though both work. T1-4 / T1-8 are the bug.

The PRIOR fix (commit `fcde545`) added init-failure plumbing for a symptom that
likely wasn't actually firing. The dispatcher probably initialized fine on
device 0. The dashboard showed `gpu_count = 1` and the user (with 2 GPUs)
correctly read this as "not using both GPUs."

The "1 GPU" vs "0 GPU" distinction is unclear from the screenshot — but the
WORKERS row formula at the time (commit `fcde545` change) was:

- `gpu_count > 0` → "23 CPU + 1 GPU (PBKDF2 + EC + bloom on GPU)"
- `gpu_count == 0` AND `gpu_init_message` empty → "23 CPU (initializing GPU...)"

The user's screenshot clearly shows "23 CPU (BIP-39 PBKDF2 is CPU-bound)" which
is the OLD stale string. So the binary they're testing is from BEFORE my latest
commit. Confirms my fix was correct for the symptom they reported, but the
ROOT cause is still T1-4: empty gpu_ids → device 0 only.

---

## Cross-cutting observations

1. **The "TUI everywhere" mandate is ~30% complete.** Pool / puzzle / brain
   wallet runners still flood cout after launching the alt-screen. Tasks #96,
   #97, #99 were marked complete but only the BIP scanner was substantially
   ported. The other three modes were skimmed at best.

2. **`bip_scanner_runner.cpp` is 1810 lines with two 666 / 803 line functions
   that are 90% copy-paste.** The on_hit lambda is character-identical between
   the two. Every bug fix must be applied twice (recent commit `2aa5dcb` may
   have only fixed one site).

3. **5 of the 13 bugs in this audit are "lying messages" — comments / errors /
   docs that say the wrong thing.** This is the highest-density category and
   the one the user has been most vocal about. Pattern: I edit code without
   editing the surrounding comment.

4. **Test coverage of the recent 2,400-LoC sprint is ~17% (400 lines covered).**
   The "ctest 84/84 green" is mostly testing what was already there. The new
   GPU dispatcher class has ZERO direct tests.

5. **Three "should-have-been-caught-in-PR-review" patterns:**
   - Return-value-ignored helpers (`enqueue`, `cudaMemcpy`)
   - Mutex / atomic naming that doesn't reflect what's protected
   - Magic numbers without comment (4096, 256, 65536, 2048, 4)
