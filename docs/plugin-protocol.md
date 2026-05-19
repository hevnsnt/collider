# theCollider Plugin Protocol

This document defines the wire protocol theCollider uses to drive subprocess plugins, the plugins.yml registry schema that selects which plugins run, and the operator workflow for writing and shipping a new plugin.

The protocol shipped with the multi-panel TUI. Future schema revisions bump the `schema_version` field; plugins must check this on startup and refuse to process events with a version they do not understand.

## At a Glance

- The collider binary launches each enabled plugin as a child process.
- The binary streams events to the plugin's stdin as JSON, one event per line, terminated by a single `\n`.
- The plugin writes free-form text to stdout. Every stdout line is forwarded verbatim to the TUI's Plugins panel.
- The plugin reads from stdin until EOF (which the binary sends at clean shutdown) and exits with status 0.

A plugin is just a subprocess. There is no shared library, no embedded interpreter, no plugin API to link against. Anything that can read JSON lines from stdin works (Python, Node, Go, a Bash one-liner, a compiled binary).

## Event Schema

Every event is a single JSON object on its own line:

```json
{
  "schema_version": 1,
  "event": "empty-hit",
  "ts": "2026-05-15T18:42:31Z",
  "data": {
    "passphrase": "correct horse battery staple",
    "privkey_hex": "c4bbcb1f...",
    "h160_hex": "1a1f5a6f...",
    "address": "1ABC..."
  }
}
```

Top-level fields, in order:

| Field            | Type    | Notes                                                                     |
| ---------------- | ------- | ------------------------------------------------------------------------- |
| `schema_version` | integer | Always 1 currently. Bumped on any breaking change.                       |
| `event`          | string  | One of `hit`, `empty-hit`, `phase-change`, `periodic`.                    |
| `ts`             | string  | ISO-8601 UTC timestamp at second precision (e.g. `2026-05-15T18:42:31Z`). |
| `data`           | object  | Event-specific payload (see below).                                       |

Unknown top-level keys are reserved and must be ignored by plugins. The binary will not emit them today but may add them later under the same schema_version when the addition is backward compatible.

### `hit` event

Emitted when the hit verifier (UVRF) confirms a candidate is a funded address.

```json
{
  "schema_version": 1,
  "event": "hit",
  "ts": "2026-05-15T18:42:31Z",
  "data": {
    "passphrase": "correct horse battery staple",
    "privkey_hex": "c4bbcb1f...",
    "h160_hex": "1a1f5a6f...",
    "address": "1ABC..."
  }
}
```

| Field         | Type   | Notes                                                                                               |
| ------------- | ------ | --------------------------------------------------------------------------------------------------- |
| `passphrase`  | string | The candidate text that hashed to the funded address.                                               |
| `privkey_hex` | string | 64 hex chars, the WIF-equivalent private key in raw form.                                           |
| `h160_hex`    | string | 40 hex chars, the HASH160 of the corresponding public key.                                          |
| `address`     | string | Optional. The Base58 (P2PKH) or Bech32 (P2WPKH) string when the binary rendered one. May be absent. |

### `empty-hit` event

Emitted when a candidate passes both bloom layers (loose + tight) but the UVRF confirms the address holds zero balance. These are the real-but-empty wallet collisions; they are valuable for off-binary tools that want to track empty-but-real BIP-39 brainwallets independent of the funded-only main hit feed.

The data shape is identical to `hit`. The `address` field follows the same optional rule.

### `phase-change` event

Emitted at every phase transition (Quick Wins to Crypto Focus, Crypto Focus to Deep Dive, etc.). Useful for plugins that want to chunk their work by phase or emit phase summaries.

```json
{
  "schema_version": 1,
  "event": "phase-change",
  "ts": "2026-05-15T19:01:08Z",
  "data": {
    "from": "Quick Wins",
    "to": "Crypto Focus",
    "phase_index": 1
  }
}
```

| Field         | Type    | Notes                                                |
| ------------- | ------- | ---------------------------------------------------- |
| `from`        | string  | Human-readable name of the phase that just finished. |
| `to`          | string  | Human-readable name of the phase about to start.     |
| `phase_index` | integer | Zero-based index of the new phase.                   |

### `periodic` event

Heartbeat with running stats. The binary chooses the cadence (suggested: 30 seconds). Plugins can use it to dashboard the run, log to disk, or detect "binary is stuck" when the heartbeat stops.

```json
{
  "schema_version": 1,
  "event": "periodic",
  "ts": "2026-05-15T18:42:31Z",
  "data": {
    "total_checked": 124851230,
    "bloom_hits": 7,
    "empty_wallets_real": 3,
    "empty_wallets_noise": 4,
    "keys_per_sec": 1820000.5,
    "current_phase": "Deep Dive"
  }
}
```

| Field                 | Type    | Notes                                                                      |
| --------------------- | ------- | -------------------------------------------------------------------------- |
| `total_checked`       | integer | Cumulative passphrases checked this run.                                   |
| `bloom_hits`          | integer | Loose bloom hits (includes both real and noise).                           |
| `empty_wallets_real`  | integer | UVRF-confirmed empty hits (passed both bloom layers, real address).        |
| `empty_wallets_noise` | integer | Loose-bloom hits the tight bloom rejected (false positives).               |
| `keys_per_sec`        | number  | Throughput averaged over the last sampling window. May be 0 during pauses. |
| `current_phase`       | string  | Human-readable current phase name.                                         |

Non-finite `keys_per_sec` values are encoded as `null` per JSON spec. Plugins should treat `null` as 0.

## Plugin Output Convention

Plugin stdout is free-form text. Every line written by the plugin is forwarded verbatim to the TUI's Plugins panel, where the last N lines per plugin are visible to the operator.

A plugin that wants structured output (for example, a balance lookup result you want another tool to consume) should still write one JSON object per line. The runner does not parse those lines today; it just displays them. A future v1.5 runner may grow a structured-output channel, but stdout is currently opaque.

Plugins should:

- Flush stdout after every line (`sys.stdout.flush()` in Python). Buffered output that never reaches the runner will not appear in the panel.
- Keep lines short (under ~120 characters). The panel truncates long lines.
- Avoid writing to stderr. The runner discards stderr in the current iteration; use stdout for everything you want the operator to see.

## Lifecycle

1. **Launch**: at scan start, the runner spawns one subprocess per enabled plugin via `execvp` (POSIX) or `CreateProcessW` (Windows). The `command` array becomes argv.
2. **Stream**: as the scan runs, every event whose kind matches the plugin's `events:` filter is serialized and written to the plugin's stdin, followed by a `\n`. Writes happen on a per-plugin writer thread so a slow plugin cannot block the scan loop.
3. **Back-pressure**: each plugin has a bounded send queue (1000 events). When the queue is full, the runner drops the OLDEST queued event and logs a one-line warning. This keeps the scan loop free of head-of-line blocking.
4. **Death detection**: the runner watches stdin for write failures. After three consecutive failed writes (broken pipe, plugin exited, etc.), the plugin is marked disabled for the rest of the run and the Plugins panel shows its status as `errored`.
5. **Clean shutdown**: at scan end, the runner closes the plugin's stdin (signalling EOF), waits up to 5 seconds for the plugin to exit, then sends SIGTERM (POSIX) or `TerminateProcess` (Windows) if needed.

A plugin should:

- Read its stdin in a loop until EOF.
- Parse each line as JSON.
- Validate `schema_version == 1`; refuse to process unknown versions (print a message to stdout and skip).
- Exit cleanly with status 0 when stdin reaches EOF.

## Registry: plugins.yml

The runner loads `~/.collider/plugins.yml` on Unix or `%USERPROFILE%\.collider\plugins.yml` on Windows. If the file does not exist, no plugins are launched (this is a normal state).

```yaml
plugins:
  - name: balance-scanner
    command: ["python", "tools/plugins/balance-scanner.py"]
    events: [empty-hit]
    enabled: true

  - name: webhook-poster
    command:
      [
        "python",
        "tools/plugins/webhook-poster.py",
        "--url",
        "https://example.com/hook",
      ]
    events: [hit, empty-hit]
    enabled: false

  - name: desktop-notifier
    command: ["python", "tools/plugins/desktop-notifier.py"]
    events: [hit]
    enabled: false
```

Per-plugin fields:

| Field     | Required | Type            | Notes                                                                                                                    |
| --------- | -------- | --------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `name`    | yes      | string          | Unique within the file. Used in the Plugins panel and for log messages.                                                  |
| `command` | yes      | list of strings | argv. First element is the executable. List-of-strings (NOT a single shell string) to avoid cross-platform quoting bugs. |
| `events`  | yes      | list of strings | Subset of `[hit, empty-hit, phase-change, periodic]`. Only matching events are delivered.                                |
| `enabled` | no       | bool            | Defaults to true. Set to false to leave the entry in the file but skip its subprocess.                                   |

Notes:

- Duplicate plugin names cause the registry to fail with an error; remove or rename the duplicate.
- The runner uses the host PATH (POSIX) or `CreateProcessW` search rules (Windows) to resolve the executable, so `python` finds whichever Python is first on PATH. Pin to an absolute path if that matters to you.
- Plugin working directory is the collider binary's CWD. Relative script paths resolve against that.

A starter file ships at `docs/plugins.yml.example`; copy it to `~/.collider/plugins.yml` and edit.

## Plugin Development Workflow

The reference plugins in `tools/plugins/` are the recommended starting point. The minimal Python skeleton:

```python
#!/usr/bin/env python3
import json
import sys

def handle(event):
    kind = event["event"]
    data = event["data"]
    # Do your thing. Anything you print() will show in the Plugins panel.
    print(f"got {kind}: {data}")
    sys.stdout.flush()

def main():
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"bad json: {e}")
            sys.stdout.flush()
            continue
        if event.get("schema_version") != 1:
            print(f"unsupported schema_version {event.get('schema_version')}")
            sys.stdout.flush()
            continue
        try:
            handle(event)
        except Exception as e:
            print(f"plugin error: {e}")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
```

Steps to ship a new plugin:

1. Drop a script in `tools/plugins/` (or anywhere; the registry takes any absolute or relative path the runner can resolve).
2. Add a `- name: ...` entry to `~/.collider/plugins.yml`.
3. Start a brainwallet scan. The Plugins panel shows the plugin as `active` once the subprocess is alive.
4. Trigger an event the plugin filters on (or wait for the next periodic heartbeat). Output appears in the panel.

## Error Handling

| Failure mode                      | Runner behavior                                                                                                          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Subprocess fails to spawn         | Plugin marked `errored`; warning logged once; scan continues.                                                            |
| Plugin closes stdin or exits      | After three consecutive stdin write failures, plugin marked `errored`; scan continues.                                   |
| Plugin write queue full           | Oldest queued event dropped; warning logged.                                                                             |
| Malformed JSON received by plugin | Plugin's choice. The runner does not validate plugin stdout.                                                             |
| Schema version mismatch           | Plugin's choice (recommended: print a message and skip the event). The runner does not check version on the plugin side. |
| plugins.yml parse error           | Registry empty for the run; warning logged. Fix the file and restart.                                                    |

## Versioning

`schema_version` starts at 1. The contract for bumping it:

- Adding a new event kind alongside existing ones is NOT a version bump (plugins filter on `events:` and ignore kinds they did not register for).
- Adding a new optional field inside an event's `data` object is NOT a version bump (existing plugins ignore unknown keys).
- Renaming a field, changing a field's type, or removing a field IS a version bump.

A v1.5 binary may emit `schema_version: 2` events. current plugins that strictly check the version will skip those events gracefully.
