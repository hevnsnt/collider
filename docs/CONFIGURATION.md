# Configuration Guide

Complete reference for theCollider's `config.yml` and the precedence rules between CLI flags, config-file values, and built-in defaults.

The schema source of truth is [`src/core/yaml_config.hpp`](../src/core/yaml_config.hpp) (the `AppConfig` struct). A documented example with every section is at [`example-config.yml`](../example-config.yml) at the repo root.

---

## Precedence

```
CLI flags  >  config.yml  >  built-in defaults
```

Concretely (Wave 5 refactor, v1.4.0):

1. CLI flags explicitly supplied on `argv` are tracked via `CLIFlags::*_set` bits inside the parser, set the moment the flag is consumed (not inferred post-parse from the resulting value).
2. Config-file values overlay onto `Arguments` only when the matching `*_set` bit is unset.
3. Anything still unset falls through to the `Arguments` and `AppConfig` defaults.

This eliminates the "value silently overridden" bug class where CLI values that happened to equal the parser's sentinel default (`--batch-size 4000000`, `--puzzle 0`) were treated as unset and clobbered by config.

---

## File location

theCollider searches for `config.yml` in this order:

1. `./config.yml`
2. `./config.yaml`
3. `~/.collider/config.yml` (Linux / macOS)
4. `~/.collider/config.yaml`
5. `%USERPROFILE%\.collider\config.yml` (Windows)
6. `%USERPROFILE%\.collider\config.yaml`

Override with `--config <path>` (also `-c`).

---

## Schema

### `pool` section

Pool-mining (distributed solving) settings.

| Key        | Type   | Default | CLI override      | Description                                                                         |
| ---------- | ------ | ------- | ----------------- | ----------------------------------------------------------------------------------- |
| `url`      | string | `""`    | `--pool` / `-p`   | Pool URL. Schemes: `jlps://` (TLS), `jlp://` (plaintext), `http://` (HTTP variant). |
| `worker`   | string | `""`    | `--worker` / `-w` | Bitcoin address to credit pool rewards to.                                          |
| `password` | string | `""`    | `--pool-password` | Optional pool password. Collision Protocol's public pool ignores this.              |
| `api_key`  | string | `""`    | `--pool-api-key`  | Optional API key for HTTP-only pools.                                               |

Setting `pool.url` in the config (without a `--brainwallet` CLI flag) auto-enables pool mode.

### `puzzle` section

Standalone puzzle-solving settings (Bitcoin Puzzle Challenge).

| Key             | Type   | Default | CLI override                      | Description                                                                                   |
| --------------- | ------ | ------- | --------------------------------- | --------------------------------------------------------------------------------------------- |
| `number`        | int    | `0`     | `--puzzle N` / `-P N`             | Specific puzzle number. `0` selects automatically (ROI-ranked).                               |
| `smart_select`  | bool   | `true`  | `--no-smart`                      | ROI-rank when picking among unsolved puzzles. `--no-smart` picks the lowest-numbered instead. |
| `min_bits`      | int    | `0`     | `--puzzle-min-bits N`             | Lower bound for `--all-unsolved` iteration.                                                   |
| `max_bits`      | int    | `160`   | `--puzzle-max-bits N`             | Upper bound for `--all-unsolved` iteration.                                                   |
| `kangaroo`      | bool   | `true`  | `--kangaroo`                      | Use Pollard's Kangaroo when a pubkey is available. v1.4.1 demotes to brute force if not.      |
| `dp_bits`       | int    | `-1`    | `--dp-bits N`                     | Distinguished-point bits for kangaroo. `-1` auto-calculates. Manual range: 16 to 28.          |
| `random_search` | bool   | `true`  | `--random` / `--sequential`       | Random walk vs. sequential within the search range.                                           |
| `auto_next`     | bool   | `false` | `--auto-next`                     | Advance to the next unsolved puzzle after solving the current one.                            |
| `checkpoint`    | string | `""`    | `--puzzle-checkpoint <file>`      | Checkpoint file for save/resume across runs.                                                  |
| `target`        | string | `""`    | `--puzzle-target <addr>` (v1.4.1) | Override the target Bitcoin address. Independent of `number`.                                 |
| `start`         | string | `""`    | `--puzzle-start <hex>` (v1.4.1)   | Override the range start (hex, with `0x` prefix).                                             |
| `end`           | string | `""`    | `--puzzle-end <hex>` (v1.4.1)     | Override the range end.                                                                       |
| `pubkey`        | string | `""`    | `--pubkey <hex>` (v1.4.1)         | 33-byte compressed pubkey (`02...`/`03...`). Only needed for non-bundled targets.             |

**Kangaroo + pubkey rules.** The bundled `data/puzzle_history.json` ships every revealed pubkey for the canonical Bitcoin Puzzle Challenge (multiples of 5 in 71-160, plus all 82 confirmed-solved puzzles). For a target outside that set, supply `pubkey:` here. See README, "Which puzzles are kangaroo-able".

### `brainwallet` section **(PRO VERSION ONLY)**

Brain-wallet scanning. The free build silently ignores this section and prints a one-time hint if it sees `bloom.file`.

| Key             | Type   | Default   | CLI override        | Description                                             |
| --------------- | ------ | --------- | ------------------- | ------------------------------------------------------- |
| `enabled`       | bool   | `false`   | `--brainwallet`     | Enable brain-wallet mode. Mutually exclusive with pool. |
| `wordlist`      | string | `""`      | (none)              | Path to a wordlist file consumed by the generators.     |
| `save_interval` | uint64 | `1000000` | `--save-interval N` | Save state every N candidates.                          |
| `resume`        | bool   | `false`   | `--resume`          | Resume from the last checkpoint.                        |

### `bloom` section **(PRO VERSION ONLY)**

| Key    | Type   | Default | CLI override     | Description                                                           |
| ------ | ------ | ------- | ---------------- | --------------------------------------------------------------------- |
| `file` | string | `""`    | `--bloom <file>` | Bloom filter of funded addresses (built with the `build_bloom` tool). |

### `gpu` section

| Key               | Type   | Default | CLI override        | Description                                                     |
| ----------------- | ------ | ------- | ------------------- | --------------------------------------------------------------- |
| `devices`         | int[]  | `[]`    | `--gpus 0,1` / `-g` | Specific GPU IDs. Empty = auto-detect every visible device.     |
| `batch_size`      | uint64 | `0`     | `--batch-size N`    | Keys per batch. `0` = use the calibrated value (or 4M default). |
| `force_calibrate` | bool   | `false` | `--force-calibrate` | Re-run batch-size calibration even if a saved value exists.     |

### `settings` section

| Key                 | Type | Default | CLI override             | Description                       |
| ------------------- | ---- | ------- | ------------------------ | --------------------------------- |
| `verbose`           | bool | `false` | `--verbose` / `-v`       | Verbose output.                   |
| `debug`             | bool | `false` | `--debug`                | Debug output for troubleshooting. |
| `benchmark_seconds` | int  | `30`    | `--benchmark-time <sec>` | Benchmark duration.               |

### `paths` section

| Key              | Type   | Default         | Description                                 |
| ---------------- | ------ | --------------- | ------------------------------------------- |
| `data_dir`       | string | `./processed`   | Directory for processed data and wordlists. |
| `checkpoint_dir` | string | `./checkpoints` | Directory for save-state files.             |
| `log_dir`        | string | `./logs`        | Directory for output logs.                  |

These keys are parsed but not yet consumed by every subsystem (see `track-e` finding 7 in the codebase). They are documented as a known gap rather than dropped, so existing configs keep loading.

---

## Common configurations

### Solo puzzle solving with kangaroo

```yaml
puzzle:
  number: 75
  kangaroo: true
  dp_bits: -1
  random_search: true

gpu:
  devices: []
```

### Pool mining (Collision Protocol)

```yaml
pool:
  url: "jlps://pool.collisionprotocol.com:17403"
  worker: "1YourBitcoinAddressForRewards"
```

### Custom range, brute force

```yaml
puzzle:
  number: 71 # Used for record-keeping; range below overrides.
  kangaroo: false # No pubkey for arbitrary addresses, brute force only.
  target: "13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so"
  start: "0x40000000000000000"
  end: "0x7ffffffffffffffff"
```

### Multi-GPU with explicit selection

```yaml
gpu:
  devices: [0, 2] # Skip GPU 1.
  batch_size: 0 # Auto-calibrate.
```

### Brain wallet **(PRO VERSION ONLY)**

```yaml
brainwallet:
  enabled: true
  wordlist: "./processed/combined.txt"
  save_interval: 1000000
  resume: false

bloom:
  file: "./funded_addresses.blf"
```

---

## Advanced research flags **(PRO VERSION ONLY)**

The following flags are accepted by the parser but not listed in `--help`. They drive the v2 puzzle-mode brain-wallet kernel and are intended for research, not production scanning.

| Flag                   | Description                                                          |
| ---------------------- | -------------------------------------------------------------------- |
| `--puzzle-only-v2`     | Enable the v2 puzzle-mode kernel plus multi-scheme dispatch.         |
| `--puzzle-keys <file>` | Path to the puzzle keys file (typically `data/puzzle_history.json`). |
| `--schemes <csv>`      | Comma-separated scheme list (e.g. `all`, or specific names).         |

These flags imply `--brainwallet` (they go through the brain-wallet pipeline). Free builds reject them at the CLI with a Pro-feature message. Multi-address scanning (the prior `--addr-types` flag) is currently only available through the legacy `--brainwallet --bloom` path; v2 multi-address support is deferred to a later release.

---

## Validation

theCollider validates configuration on startup. Common errors:

| Error                                 | Cause                             | Fix                                                                                   |
| ------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------- |
| Conflicting search modes              | Multiple top-level modes selected | Pick one of `--brainwallet`, `--pool`, `--puzzle [N] [--kangaroo]`.                   |
| `[Pro] --bloom ... ignored`           | Free build, `bloom.file` set      | Brain wallet is **(PRO VERSION ONLY)**.                                               |
| `[!] Failed to decompress public key` | Malformed `pubkey:` value         | Provide 66 hex chars, leading byte `02`/`03` (or 130 hex with `04` for uncompressed). |
| `--pubkey accepts 33B compressed`     | Wrong-length pubkey               | Use the 33-byte compressed form.                                                      |

---

## Tips

- Keep secrets out of `config.yml`. Use the matching CLI flag instead.
- Use `--debug` to print the resolved configuration after merging CLI plus config.
- Auto-calibration is usually the right default. Only override `batch_size` when you have measured something.
- CLI flags always win. If a config value is being ignored, look for a matching CLI flag in the launch command (or a stray environment-set wrapper).

---

## Where to go next

| For                                             | See                                |
| ----------------------------------------------- | ---------------------------------- |
| Runtime usage examples and the full CLI surface | [README.md](../README.md)          |
| Building from source on each platform           | [INSTALL.md](INSTALL.md)           |
| Pool client setup, accrual, etiquette           | [POOL.md](POOL.md)                 |
| JLP wire format (third-party clients)           | [JLP-PROTOCOL.md](JLP-PROTOCOL.md) |
| Source-tree map for contributors                | [ARCHITECTURE.md](ARCHITECTURE.md) |
| Release history                                 | [CHANGELOG.md](CHANGELOG.md)       |
