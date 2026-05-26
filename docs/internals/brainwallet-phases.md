# Brainwallet Phase Engine — State Machine Reference

theCollider's brainwallet runtime walks every passphrase candidate
through a **5-phase main loop** plus a secondary **iteration-mode
state machine**. New contributors (or future-you) need to know two
things:

1. The order phases run in (and why).
2. The order iteration modes advance through (and why).

This doc is the canonical reference. Code: `src/generators/streaming_brain_wallet.{hpp,cpp}`,
runtime driver: `src/runtime/brain_wallet_runner.cpp` (`scan_loop`).

## Main 5-Phase Cycle

```
       ┌────────────────┐
       │   Quick Wins   │  Phase 0  -- raw wordlist, 0 rules
       └────────┬───────┘
                ▼
       ┌────────────────┐
       │  Crypto Focus  │  Phase 1  -- crypto.rule (small, curated)
       └────────┬───────┘
                ▼
       ┌────────────────┐
       │    Extended    │  Phase 2  -- d3ad0ne.rule (sliding window)
       └────────┬───────┘
                ▼
       ┌────────────────┐
       │   Combinator   │  Phase 3  -- word1+word2 pair generation
       └────────┬───────┘                (vocab capped at 50k -> 2.5B pairs)
                ▼
       ┌────────────────┐
       │   Deep Dive    │  Phase 4  -- dive.rule (sliding window)
       └────────┬───────┘
                ▼
       ┌────────────────────────────────┐
       │ restart_with_next_iteration()  │
       │ advance IterationMode, loop    │
       └────────┬───────────────────────┘
                ▼
              (back to Quick Wins for next mode pass)
```

### Why this order

The phases are sorted **by hit probability per CPU/GPU cycle spent**.
Quick Wins is first because raw wordlist words match the broadest
slice of human-chosen brainwallets (the famous `password` -> `1A1zP1...`
discovery is a Quick Wins hit). Crypto Focus is second because
crypto-flavoured rules (caps, leetspeak) lift the same base wordlist
through patterns crypto-native operators try. Extended + Deep Dive
are third + fifth because they chew through hundreds of rules per
word; valuable but expensive. Combinator is fourth because two-word
pairs are a different _class_ of guess (not rule-mutation but
concatenation) — interleaving it between rule-stacking phases keeps
the cadence diverse for the operator watching the TUI.

### Phase skip rules

`advance_past_first_cycle_phases()` (called by `brain_wallet_runner.cpp`
once per IterationMode loop) skips phases whose **work is identical
to the previous pass**. Quick Wins + Crypto Focus only do useful new
work on the FIRST mode pass; after that the same wordlist + same
rules produce the same candidates. Extended + Combinator + Deep Dive
advance their internal sliding windows so they DO produce new work
on cycle 2+; `skip_on_repeat=true` for the first two phases lets the
generator skip straight to Extended on every cycle past the first.

## IterationMode State Machine

Each phase runs the candidates produced by the **current iteration
mode**. The modes advance sequentially after `restart_with_next_iteration()`:

```
PHASE_CYCLING --> RULE_STACKING --> HYBRID_MASK --> COMBINATOR
       ▲                                                │
       │                                                ▼
       └──────── PCFG <--- MARKOV <---- KEYBOARD_WALK ──┘
```

| Mode            | Generates                                                  | Why this priority                                                     |
| --------------- | ---------------------------------------------------------- | --------------------------------------------------------------------- |
| `PHASE_CYCLING` | wordlist × phase rules (per-phase windows)                 | Most-likely-to-hit first: human-chosen passwords skew to base words.  |
| `RULE_STACKING` | rule1(rule2(word)) — multiplicative expansion              | Catches stacked mutations (capslocked-leetspeak-suffixed).            |
| `HYBRID_MASK`   | word + mask (e.g. word + ?d?d?d?d)                         | Catches "year suffix" / "PIN suffix" patterns common in passphrases.  |
| `COMBINATOR`    | word1 + word2 pairs (capped vocab^2)                       | Two-word brainwallets ("correct horse" style); also runs as Phase 3.  |
| `MARKOV`        | per-trained-corpus probability-ordered char chains         | Catches "looks like a word" guesses without being in any wordlist.    |
| `PCFG`          | Probabilistic Context-Free Grammar (structure + terminals) | Most expensive; covers structure-aware patterns (NameYearSymbol etc). |
| `KEYBOARD_WALK` | qwerty/dvorak walks (`1qaz2wsx`, `asdfghjkl`)              | Last because hits are rare but the dataset is tiny so it's quick.     |

### Budget per iteration mode

`Config::generator_candidates_per_iteration` (default **4,000,000**)
caps how many candidates each generator yields before advancing to
the next sub-iteration. Without it MARKOV / PCFG would generate
infinitely many low-probability guesses and never advance to PHASE_CYCLING
again — the priority queue would happily emit every state in its
heap. The budget forces forward progress.

The budget is enforced inside `StreamingBrainWallet::next_batch()` by
tracking `mode_sub_iteration` (the COMBINATOR + PCFG / MARKOV outer
counters) and exiting when it crosses the per-mode threshold. See
`streaming_brain_wallet.cpp::restart_with_next_iteration()`.

## Save / Resume Invariants

A successful resume restores the full `StateSnapshot` (declared in
`streaming_brain_wallet.hpp`):

| Field                | What it tracks                                                 |
| -------------------- | -------------------------------------------------------------- |
| `current_phase`      | 0..4 main-cycle phase index                                    |
| `current_word_idx`   | word offset within the current wordlist pass                   |
| `current_rule_idx`   | rule offset within the current rule chunk                      |
| `phase_iteration`    | how many times the 5-phase cycle has restarted                 |
| `iteration_mode`     | which IterationMode is active                                  |
| `mode_sub_iteration` | outer counter for the COMBINATOR / MARKOV / PCFG inner loops   |
| `rule_window_index`  | for sliding-window phases, 1-based window position             |
| `is_brute_mode`      | true when `--brute` is active (mutually exclusive with phases) |

Clamping: if a restore lands with `current_word_idx >= wordlist.size()`
(e.g. operator shrunk the wordlist between runs), the next `next_batch()`
call clamps to `wordlist.size() - 1` and continues. The clamp happens
inside the generator, not the runner.

## Hot-Swap

`StreamingBrainWallet::queue_profile_swap()` deposits a new wordlist
path that takes effect at the **next phase boundary**. Mid-phase swap
would lose `current_word_idx` alignment; defer-to-boundary is the
sane trade. The runner triggers it from the Wordlist Picker modal
('w' key) and from the Wordlist Composer modal ('c' key) when a
recombine finishes.

## See Also

- `src/generators/streaming_brain_wallet.hpp` — Config / StateSnapshot / IterationMode enum
- `src/generators/streaming_brain_wallet.cpp` — phase setup + sub-iteration logic
- `src/runtime/brain_wallet_runner.cpp` — `scan_loop` + TUI integration
- `docs/v1.5.0-quality-grading.md` — TP-7 (this doc) + related TPs
