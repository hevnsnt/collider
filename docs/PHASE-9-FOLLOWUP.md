# Phase 9 follow-up: wiring `--puzzle-only-v2` into the runtime

The Phase 9 branch (`phase-9-puzzle-mode-integration`) lands the
build-integration half of puzzle-mode bloom check:

- `src/gpu/v2/brain_wallet_v2.{cu,hpp}` — kernel + frozen public API
- `tests/v2/test_brain_wallet_v2.cpp` — host + GPU end-to-end tests
- `tools/brainwallet/puzzle_mode.py` — CPU reference / prototype
- `docs/BRAINWALLET-V2-SPEC.md` — design doc
- CMakeLists wiring: `src/gpu/v2/brain_wallet_v2.cu` is added to
  `collider_gpu` when `COLLIDER_PRO=ON`; `BrainWalletV2` test target
  is added to CTest with `SKIP_RETURN_CODE=77` for no-GPU CI runs.

What still needs to land before v1.4.0 ships:

1. **CLI flag**: add `--puzzle-only-v2` to the CLI args struct in
   `src/main.cpp`. Mutually exclusive with `--brainwallet`,
   `--pool`, and `--puzzle N`. When set, dispatch to a new
   orchestrator function instead of the legacy brain-wallet path.

2. **Orchestrator**: new file
   `src/gpu/v2/v2_orchestrator.cpp` exposing

   ```cpp
   int run_v2_puzzle_only(const CliArgs& args);
   ```

   which:
   - calls `v2_init(stream)`
   - loads the 79 historical solved puzzle keys (from the same
     JSON the website uses) and builds `PuzzleTarget` records
   - calls `v2_set_puzzle_targets(targets)`
   - drives the existing wordlist / passphrase generator into
     `v2_brain_wallet_batch(..., scheme_mask=ALL,
addr_mask=0, /*bloom*/ nullptr, ...)` (puzzle-only short
     circuit; kernel skips EC_MUL when no puzzle hit).
   - prints PUZZLE_KEY_HIT records to stdout.

3. **Smoke test against a known-hitting passphrase**: extend
   `tests/v2/test_brain_wallet_v2.cpp` to exercise the orchestrator
   end-to-end on a synthetic puzzle (puzzle_n = 32, key = SHA256("test")).

4. **Documentation**: add the flag to `docs/help.md` (or wherever
   the CLI help text is sourced) and the website features page.

The kernel proper, the 4-limb mask arithmetic, the puzzle target
host helper, and the unit tests all already exist on this branch.
The remaining work is plumbing, not algorithm.

Estimated effort: 1-2 hours. Owner: TBD.
