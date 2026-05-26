# Migration Guide: v1.4.x to v1.5

This document is the upgrade procedure for **pool operators** and for **workers** moving from theCollider v1.4.x to v1.5.0.

v1.5 is a wire-breaking release. v1.4.x clients are rejected by v1.5 servers, and v1.5 clients cannot connect to v1.4.x servers. Both sides must upgrade. The break is intentional: v1.4.x is theft-vulnerable on mainnet by construction (see audit finding C1 history in the [v1.5 security audit report](../../collision-protocol/docs/v1.5-security-audit-report.md)), and the wire bump forces the network to upgrade together.

If you are running a private pool, plan the upgrade as a coordinated event with your workers. The recommended grace period is two weeks: announce the cutover, give workers time to download v1.5 binaries, then upgrade the pool server.

## Contents

- [Who needs to do what](#who-needs-to-do-what)
- [Worker upgrade procedure](#worker-upgrade-procedure)
- [Pool operator upgrade procedure](#pool-operator-upgrade-procedure)
- [Configuration changes summary](#configuration-changes-summary)
- [Database schema additions](#database-schema-additions)
- [Verifying the upgrade](#verifying-the-upgrade)
- [Payout lifecycle and operator invariants](#payout-lifecycle-and-operator-invariants)
- [Rollback procedure](#rollback-procedure)
- [Pre-Mainnet Operator Checklist (auditor verbatim)](#pre-mainnet-operator-checklist-auditor-verbatim)

## Who needs to do what

| Role                        | Action                                                                                                                                                                                                                                                                                       |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Worker**                  | Download v1.5 client binary. Replace v1.4.x binary in your run scripts. No config change required. (Type assignment is server-assigned at AUTH time; you do not pick TAME or WILD.)                                                                                                          |
| **Pool operator**           | Provision a hot wallet, configure two DISTINCT mempool API providers, seed Firebase Auth with at least one admin, deploy the new pool server and the website, then announce upgrade window to workers. The pool refuses to start if the wallet cannot decrypt; this fails fast on misconfig. |
| **Standalone solver user**  | No action required. `KangarooMode::BOTH` is preserved and unchanged for non-pool puzzle solving.                                                                                                                                                                                             |
| **Brain wallet (Pro) user** | No action required. The brain-wallet runner does not touch the pool path.                                                                                                                                                                                                                    |

## Worker upgrade procedure

### What changes for you as a worker

- You will be assigned **TAME** or **WILD** on each pool connect. You do not pick. The pool round-robins across connections.
- You no longer see the puzzle's private key. The pool server is the only entity that ever holds it, briefly and in memory only.
- Payouts are operator-triggered through the admin payout UI after a sweep clears. The pool operator broadcasts your share to the Bitcoin address you registered as `--worker`.
- The `recovered_keys/<ts>.json` files that v1.4.x created on solve are gone in v1.5. No private keys are ever written to your disk.

### Steps

1. Download the v1.5.0 client binary for your platform from the releases page:
   - Windows: `collider-1.5.0-windows-x64.zip`
   - Linux: `collider-1.5.0-linux-x64.tar.gz`
   - macOS (Intel/Apple Silicon): `collider-1.5.0-macos-universal.tar.gz`

2. Replace the v1.4.x binary in your run scripts. The CLI surface is unchanged; the same `--pool`, `--worker`, `--pool-password-file` flags continue to work.

3. Restart your worker. On connect you should see a new banner line:

   ```
   [*] Pool mode: asymmetric protocol v3 (assigned TAME or WILD by the pool; the puzzle key is computed by the pool, not here).
   ```

4. Verify the connection is accepted. If you see `AUTH_FAIL` with the reason `UPGRADE_REQUIRED`, you are still running v1.4.x or the binary swap did not take. Confirm `./collider --version` reports `1.5.0`.

### What happens if you do not upgrade

The v1.5 pool server rejects v1.4.x clients at AUTH with `AUTH_FAIL` and the reason `UPGRADE_REQUIRED`. Your worker will not be able to submit DPs and will not accrue any share. You will see the rejection in your client logs as a clear error. The grace period your operator announced is the window in which to upgrade.

## Pool operator upgrade procedure

The pool operator path is longer because v1.5 introduces a hot wallet, a payout UI, and stricter sweep-broadcast configuration. Do not skip any step. The pool service is intentionally fail-fast on misconfiguration; a half-configured v1.5 deployment will refuse to start rather than silently accept solves it cannot safely sweep.

### Prerequisites

- A clean working directory on the pool VPS with the v1.5 `collision-protocol` source.
- Python 3.12 with the v1.5 `requirements.txt` installed in a virtualenv (the v1.5 dependency surface includes argon2-cffi, coincurve, and cryptography that v1.4.x did not require).
- Firebase project with Authentication enabled (Email/Password or Google sign-in provider configured) and the Admin SDK service account JSON available.
- At least one operator email account that will be the first admin (`ADMIN_EMAIL_BOOTSTRAP`).
- Two distinct public Bitcoin mempool API providers reachable from the VPS (default: mempool.space and blockstream.info).

### Step 1: provision the hot wallet

Run the keypair generator once per puzzle (recommended) or once total (if you intentionally reuse a wallet across multiple puzzles):

```bash
ssh <operator>@<pool-vps>
cd /opt/collision-protocol
source .venv/bin/activate
python tools/build_keypair.py \
    --output /opt/collision-protocol/keys/sweep_wallet.enc
```

You will be prompted for a passphrase (12 character minimum, empty and whitespace-only rejected). The tool prints the public Bitcoin address that the sweep will send funds TO. Write it down; this is the address you will see in the admin UI after a solve.

Detailed workflow, file format, rotation policy, and recovery procedures are in [`collision-protocol/docs/HOT-WALLET.md`](../../collision-protocol/docs/HOT-WALLET.md).

### Step 2: configure environment variables

The pool service reads these at startup. Add them to your systemd unit's `[Service] Environment=` block, your `.env` file (if loaded by the unit), or your secrets manager.

| Env var                    | Purpose                                             | Required                        |
| -------------------------- | --------------------------------------------------- | ------------------------------- |
| `SWEEP_WALLET_PASSPHRASE`  | Decrypts the hot wallet at sweep time               | YES                             |
| `MEMPOOL_API_URL_OVERRIDE` | Primary mempool API base URL                        | Recommended                     |
| `MEMPOOL_API_FALLBACK_URL` | Fallback mempool API (MUST be a DIFFERENT provider) | YES                             |
| `ADMIN_EMAIL_BOOTSTRAP`    | First-sign-in admin auto-grant email                | YES (unset after first sign-in) |
| `POOL_DATA_DIR`            | Directory for SOLUTION.txt, ledger, sidecar files   | Recommended                     |
| `POOL_TESTNET`             | Set to `1` for testnet deployments                  | Optional                        |

The pool refuses to start with `MEMPOOL_API_FALLBACK_URL` unset or equal to `MEMPOOL_API_URL_OVERRIDE`. The cross-provider attestation gate requires two DISTINCT providers; setting them to the same value causes the sweep service to return `sweep_complete=False` for every solve, and the pool refuses to release SOLUTION. This is by design.

### Step 3: seed the first admin

The first user signing in with an email matching `ADMIN_EMAIL_BOOTSTRAP` gets the `isAdmin` Firebase custom claim auto-granted. Sign in once via the website at `/account/login`, confirm the `isAdmin` claim is set (visible in the Firebase console under Authentication > Users > the user > Custom claims), then UNSET `ADMIN_EMAIL_BOOTSTRAP` from the running process and from any persisted env config.

Why unset: while the env var is present, ANY future user signing in with that email gets re-promoted to admin on every sign-in. If the email address gets recycled, leaked, or mistyped into a collaborator's config, that account gains the ability to broadcast arbitrary payout transactions from the sweep wallet.

Future admins are promoted via the Firebase Console (Auth > Users > the user > Custom claims `{ "isAdmin": true }`) or via a one-shot admin script, NEVER by re-setting `ADMIN_EMAIL_BOOTSTRAP`.

### Step 4: deploy the new pool server and website

The deploy method depends on your existing setup. The repository ships a `docker/` directory with a `docker-compose.yml` that brings up the pool server, the website, and an nginx reverse proxy.

```bash
cd /opt/collision-protocol
git pull origin main
git checkout v1.5.0
cd docker
docker compose down
docker compose up -d --build
```

On startup the pool service will attempt to decrypt the sweep wallet (failing fast on a wrong passphrase or missing file). Check the logs:

```bash
docker compose logs pool-server | tail -50
docker compose logs website | tail -20
```

You should see:

```
[pool-server] Sweep wallet decrypted successfully. Destination address: bc1q...
[pool-server] Cross-provider attestation configured: primary=mempool.space fallback=blockstream.info
[pool-server] Listening on port 17403 (TLS)
```

If you see `Sweep wallet decryption failed` or `Cross-provider attestation MISCONFIGURED`, fix the env vars and re-deploy. The pool will not accept connections until the startup checks pass.

### Step 5: announce the upgrade to workers

Post the v1.5.0 cutover date to wherever you communicate with your workers (Discord, Telegram, the pool's status page, etc). Give workers at least two weeks to swap binaries. After the cutover, v1.4.x workers see `AUTH_FAIL` and can re-read your announcement to know they need to upgrade.

A reasonable announcement template:

> theCollider v1.5.0 ships [DATE]. This is a theft-resistance upgrade: the new protocol guarantees that no worker can compute the puzzle's private key on their own machine, closing the v1.4.x self-solve window. v1.4.x clients will be REJECTED by the pool starting [CUTOVER DATE]. Download v1.5.0 from [RELEASES URL] and replace your binary before the cutover. The CLI surface is unchanged; your existing run scripts continue to work without modification.

## Configuration changes summary

| What                            | v1.4.x                                             | v1.5.0                                                         |
| ------------------------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| Wire protocol version           | 2                                                  | 3                                                              |
| `WORK_ASN` payload              | `pubkey`, `range`, `dp_bits`, `work_id`            | adds `kangaroo_type`, `start_offset_a`, `start_offset_b`       |
| SOLUTION direction              | bidirectional (client uploads on local solve)      | server-to-client only; inbound client SOLUTION is dropped      |
| `report_solution()`             | present in `JLPPoolClient::report_solution`        | DELETED end-to-end                                             |
| `recovered_keys/*.json` on disk | written on every local solve                       | NEVER written; no v1.5 client code path holds a private key    |
| Sweep wallet                    | not used                                           | required: `tools/build_keypair.py` + `SWEEP_WALLET_PASSPHRASE` |
| Mempool API                     | optional, single endpoint                          | required, two DISTINCT providers                               |
| Admin UI                        | not present                                        | required for payouts: Firebase Auth + `isAdmin` custom claim   |
| SOLUTION.txt path               | hard-coded `./data/SOLUTION.txt` (relative to CWD) | `Path(config.data_dir) / "SOLUTION.txt"`                       |

## Database schema additions

The v1.5 pool server adds new tables and columns to the existing pool SQLite databases. Migrations are applied automatically on first startup; do not run them manually unless instructed by a release note.

| Database / Table                                    | Field                              | Purpose                                               |
| --------------------------------------------------- | ---------------------------------- | ----------------------------------------------------- |
| `work_store.db` / `work_assignments`                | `kangaroo_type: INTEGER NOT NULL`  | TAME=1, WILD=2, BOTH=0 (rejected in pool mode)        |
| `work_store.db` / `work_assignments`                | `start_offset_a: INTEGER NOT NULL` | Per-side offset window low bound                      |
| `work_store.db` / `work_assignments`                | `start_offset_b: INTEGER NOT NULL` | Per-side offset window high bound                     |
| `dp_store.db` / `dps`                               | `dp_type: INTEGER NOT NULL`        | Matches the worker's assigned type at submission time |
| `payout_ledger_<sweep_txid>.json` (file, mode 0600) | new                                | Frozen per-worker share snapshot                      |
| `payout_log_<sweep_txid>.json` (file, mode 0600)    | new                                | Append-only sidecar of paid events                    |
| `sweep_state_<puzzle>.json` (file, mode 0600)       | new                                | Crash-resume state for the sweep service              |

The v1.5 wire schema source of truth is `protocol/jlp.yaml` in both repos. The `WORK_ASN` message gains three fields (`kangaroo_type: u8`, `start_offset_a: u64`, `start_offset_b: u64`); regenerated codecs ship in `src/jlp_protocol.py` and `src/pool/jlp_wire_generated.hpp`. The `work_manager.py` round-robin counter plus offset pool drives the new assignment logic.

## Verifying the upgrade

### On the worker side

```
$ ./collider --version
collider 1.5.0
$ ./collider --pool jlps://your-pool.example.com:17403 --worker bc1q...
[*] Pool mode: asymmetric protocol v3 (assigned TAME or WILD by the pool; the puzzle key is computed by the pool, not here).
[*] AUTH OK
[*] Assigned: kangaroo_type=WILD, work_id=...
```

### On the pool operator side

```
$ curl https://your-pool.example.com/health
{
  "version": "1.5.0",
  "protocol_version": 3,
  "sweep_wallet_address": "bc1q...",
  "providers": {
    "primary": "https://mempool.space/api",
    "fallback": "https://blockstream.info/api"
  },
  "uptime_seconds": 12345
}
```

### Testnet rehearsal of the propagation gate (recommended before mainnet)

See `collision-protocol/docs/v1.5-phase-b-rehearsal-runbook.md` for the full step-by-step rehearsal procedure, including the audit-required Step 7.5 (firewall-block sub-rehearsal that exercises the C1 cross-provider attestation gate). That runbook is the operator-facing source of truth; the summary below is the high-level shape so readers of this migration guide know what the rehearsal accomplishes.

Operators MUST verify the cross-provider attestation gate by:

1. Standing up a testnet pool with two distinct API providers configured.
2. Firewall-blocking the fallback provider's outbound port at the VPS.
3. Triggering a testnet solve.
4. Confirming the sweep service logs `"Propagation timeout: fallback ... did NOT observe tx ..."` and `sweep_complete=False`, and that the pool server logs `"Sweep failed (...); REFUSING to broadcast SOLUTION"`.
5. Removing the firewall rule and re-triggering.
6. Confirming the sweep completes and SOLUTION is broadcast.

This rehearsal exercises the exact code path that closes the C1 audit finding (hostile-primary timing attack). Operators should not deploy to mainnet without running it at least once.

## Payout lifecycle and operator invariants

This section is the single most important operator-facing safety contract in v1.5. Read it before clicking `[Send Payout]` on any production sweep, even once.

### The order of artifacts on disk does NOT match the order of events on chain

In v1.5 the sweep flow writes files to disk and broadcasts transactions in an order chosen for **crash recovery**, not for visual operator interpretation. Specifically:

1. The pool detects a cross-collision and computes the puzzle private key in memory.
2. `sweep_service.py` signs the sweep transaction and broadcasts it via the primary mempool API provider.
3. The payout ledger snapshot `payout_ledger_<sweep_txid>.json` lands on disk **at this point**, BEFORE step 4. (See `sweep_service.py:1354`.) This is intentional: a crash mid-wait at step 4 still leaves the ledger on disk for operator audit and for the eventual rerun of the propagation check.
4. The sweep service polls the OTHER provider for `GET /tx/<sweep_txid>` returning 200, gated on a hard timeout.
5. If step 4 succeeds, the SOLUTION wire message is released to workers.
6. The sweep transaction (broadcast in step 2) confirms on chain at some later point (typically one block, ~10 minutes; sometimes longer at high mempool congestion).

**Step 3 happens BEFORE step 4, step 5, and step 6.** A `payout_ledger_<sweep_txid>.json` file existing on disk therefore does NOT mean the sweep has confirmed on chain. It does not even mean the sweep has propagated. It means the pool intended to sweep, signed the transaction, and broadcast it to one provider.

### The operator-facing invariant (CRITICAL)

> **A `payout_ledger_<sweep_txid>.json` snapshot file existing on disk does NOT mean the sweep transaction has confirmed on chain.** The ledger is written before propagation as a crash-recovery artifact.

Before paying any worker from a ledger snapshot:

1. **Verify the sweep transaction is confirmed on chain.** Take the `sweep_txid` field from the ledger JSON and look it up on a public block explorer (mempool.space, blockstream.info, or your own Bitcoin Core node). The transaction must have at least 1 confirmation. If the explorer reports "transaction not found" or "in mempool but not confirmed", DO NOT proceed with payouts.

2. **Click the explicit `[Send Payout]` button per worker in the `/admin/payouts` UI.** The admin UI is intentionally manual. There is no "pay all" sweep action, no auto-payout cron, no scheduled job. Each payout transaction is a deliberate admin click backed by a server-verified Firebase Auth session + isAdmin claim + per-call idempotency key.

The `PayoutService` is the last safety net: when an admin clicks `[Send Payout]`, the service queries UTXOs at the sweep wallet address before signing the payout transaction. If the sweep transaction never made it to chain (the operator ignored step 1 above, or the sweep was orphaned, or the broadcast actually failed in step 2 above but propagation step 4 also failed silently), `PayoutService.send_worker_payout` returns `ok=False` and **no payout transaction is signed**. The wallet has no funds to spend; the math refuses.

### Why the order is what it is

Step 3 (ledger snapshot) lands before step 4 (propagation confirmation) for one reason: **operator audit during a crash window**. If the pool process is killed between step 4 (signed and broadcast) and step 5 (SOLUTION released), the next operator action is to investigate. They need:

- The signed transaction hex (in `sweep_state_<puzzle>.json`, so they can re-broadcast it manually if the original broadcast failed).
- The per-worker DP contribution snapshot frozen at sweep-broadcast time (in `payout_ledger_<sweep_txid>.json`, so late DPs from the lingering pool runtime cannot retroactively change shares).

Writing the ledger AFTER propagation would mean a crash between steps 4 and 5 leaves the operator with a signed transaction on chain but no record of who is owed what. Writing the ledger BEFORE propagation is the safer choice for the crash-recovery case, AT THE COST of operators needing to understand that the ledger file's existence is not a confirmation signal. This document is that cost being paid in operator-facing words instead of in operator confusion at solve time.

### Quick decision tree

```
Is there a payout_ledger_<sweep_txid>.json file on disk?
├── Yes
│   └── Is the sweep_txid confirmed on chain (>= 1 confirmation)?
│       ├── Yes
│       │   └── Open /admin/payouts. Per-worker review.
│       │       Click [Send Payout] for each worker individually.
│       │       PayoutService double-checks UTXOs before signing.
│       └── No
│           ├── In mempool, unconfirmed
│           │   └── WAIT. Do not click [Send Payout].
│           │       The PayoutService will refuse to sign until
│           │       confirmation lands.
│           └── Not in mempool, not on chain
│               └── Investigate. Check sweep_state_<puzzle>.json for
│                   the signed_tx_hex; re-broadcast manually via any
│                   public broadcaster (mempool.space POST /tx,
│                   blockstream.info, or your own Bitcoin Core
│                   sendrawtransaction).
└── No
    └── No sweep has been attempted for this puzzle.
        Nothing to pay out.
```

### What the audit said

The auditor explicitly accepted this ordering and the operator-facing contract that comes with it. Audit finding O3 ("sweep-wallet matches-snapshot guard") confirms the PayoutService check that closes the only operationally-reachable failure mode: a planted snapshot file cannot bypass the wallet-derivation check, and a stale ledger from an un-confirmed sweep cannot produce a real payout because the wallet has no UTXOs to spend.

## Rollback procedure

If you need to roll back from v1.5.0 to v1.4.x, the wire protocol break makes a same-pool rollback nontrivial. Two paths:

### Safe rollback (recommended)

1. Stop the v1.5 pool server. The workers connected at the time will see their connections drop.
2. Inform workers to roll back to their v1.4.x binaries.
3. Restore the v1.4.x pool server from your pre-v1.5 backup. Restore the v1.4.x DB schema (the new columns added by v1.5 do not exist in v1.4.x and the v1.4.x server will not understand them).
4. Restart. Workers reconnect with their v1.4.x clients.

**Be aware**: any solves that happened during the v1.5 window were swept to the v1.5 hot wallet, which v1.4.x does not know about. You must manually move those funds (using the hot wallet's private key) before discarding the wallet file.

### Emergency rollback (not recommended; theft-vulnerable)

If you must roll back without coordinating with workers, you can flip the v1.5 server to v1.4.x compatibility by removing the `protocol_version < 3` check in `pool_server.py:_handle_authenticate` and the type-validation in `dp_store.py`. This re-opens the theft window that v1.5 closed and is **only acceptable on testnet**.

## Pre-Mainnet Operator Checklist (auditor verbatim)

The following six items are reproduced VERBATIM from the security auditor's final sign-off section "Operator Notes Before Pushing v1.5 to Mainnet" in [`v1.5-security-audit-report.md`](../../collision-protocol/docs/v1.5-security-audit-report.md). They are the formal checklist the auditor requires operators to complete before the v1.5.0 binary handles mainnet bounty funds. Do not skip any item. Item 5 is the production smoke check that catches misconfiguration regressions unit tests cannot reach; a misconfigured `MEMPOOL_API_URL_OVERRIDE` / `MEMPOOL_API_FALLBACK_URL` pair would slip past every other gate and be caught only here.

> Source: `collision-protocol/docs/v1.5-security-audit-report.md`, section "Operator Notes Before Pushing v1.5 to Mainnet", audit dated 2026-05-21.

> 1. **DO NOT deploy** until C1 is resolved. A 3-second sleep is not a propagation guarantee. The implementation must poll a second independent provider for `GET /tx/<txid>` returning 200 before the SOLUTION wire broadcast may fire. The cost is at most 30 extra seconds per solve, paid once per puzzle.

> 2. **Anchor the signing math** (S1, S2) against published Bitcoin Core / BIP-143 test vectors. The integration tests on testnet (Task #10) will catch a signing error only if you actually solve a testnet puzzle; the KAT tests catch it before deployment.

> 3. **Fix the integer-parse gap** (A1) and the hardcoded SOLUTION.txt path (O1). Both are one-line fixes.

> 4. **Set `ADMIN_EMAIL_BOOTSTRAP` only once** (A2). Remove it from the environment after the first admin sign-in. Document in HOT-WALLET.md.

**CRITICAL. DO NOT SKIP ITEM 5.** A misconfigured production `MEMPOOL_API_URL_OVERRIDE` / `MEMPOOL_API_FALLBACK_URL` pair (same provider on both, or fallback unreachable in prod) would be caught here and only here. The full step-by-step procedure (including the audit-required Step 7.5 firewall-block sub-rehearsal that exercises the C1 cross-provider attestation gate) lives in `collision-protocol/docs/v1.5-phase-b-rehearsal-runbook.md`; that runbook is the operator-facing source of truth for the rehearsal.

> 5. **Mainnet rehearsal**: before going live, the operator should do a testnet rehearsal that EXPLICITLY tests the C1 fix — block one of the two API providers at firewall level and verify the sweep refuses to release SOLUTION.

> 6. **The asymmetric design itself is sound**. P1 + P2 + the integration test in test_v15_asymmetric_integration.py establish that no v1.5 worker, even with binary modification, can self-solve under the protocol. The remaining risk is operational (sweep timing, signing correctness, admin auth) — all addressable before mainnet.

### Status of each item at v1.5.0 ship

| Item                     | Status at v1.5.0 ship                                                                                                                          | Operator action still required                                                                                                                                                                                                                                                                                                                                    |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. C1 resolved           | DONE in code (cross-provider attestation lands SOLUTION only after the second provider returns 200)                                            | Verify via Item 5 rehearsal on testnet                                                                                                                                                                                                                                                                                                                            |
| 2. Signing KATs          | DONE (legacy P2PKH KAT and BIP-143 P2WPKH KAT both pass)                                                                                       | None; covered by `pytest tests/`                                                                                                                                                                                                                                                                                                                                  |
| 3. A1 and O1             | DONE (regex `^\d+$` at the admin route gate; SOLUTION.txt routes through `ServerConfig.data_dir`)                                              | None; covered by `pytest tests/`                                                                                                                                                                                                                                                                                                                                  |
| 4. ADMIN_EMAIL_BOOTSTRAP | DOCUMENTED (PAYOUTS.md, HOT-WALLET.md). v1.5.1 will add a startup warning when the env var is present alongside an existing admin (A2 backlog) | UNSET the env var after first admin sign-in. Verify with `firebase auth:export admins.json` then remove from systemd unit / apphosting.yaml / Docker compose.                                                                                                                                                                                                     |
| 5. Mainnet rehearsal     | NOT possible in code; this is an operator action against a live testnet                                                                        | RUN the rehearsal before mainnet. The operator-facing source of truth is `collision-protocol/docs/v1.5-phase-b-rehearsal-runbook.md` (full step-by-step including the audit-required Step 7.5). The summary lives at [Testnet rehearsal of the propagation gate](#testnet-rehearsal-of-the-propagation-gate-recommended-before-mainnet) earlier in this document. |
| 6. Asymmetric design     | AUDITED and PASS (P1, P2, O2, O3; see audit report)                                                                                            | None; design is intrinsic to v1.5.0                                                                                                                                                                                                                                                                                                                               |

The auditor's Item 5 rehearsal IS the [Testnet rehearsal of the propagation gate](#testnet-rehearsal-of-the-propagation-gate-recommended-before-mainnet) section earlier in this document. The two are the same procedure described from two angles: the operator perspective (run these commands) and the auditor perspective (this is the regression contract). Both must be satisfied before mainnet.

## Where to go next

- [`HOT-WALLET.md`](../../collision-protocol/docs/HOT-WALLET.md). sweep wallet provisioning, rotation, recovery
- [`PAYOUTS.md`](../../collision-protocol/docs/PAYOUTS.md). admin payout UI workflow
- [`PAYOUT-DEPLOYMENT.md`](../../collision-protocol/docs/PAYOUT-DEPLOYMENT.md). HTTP-bridge deployment architecture
- [`v1.5-security-audit-report.md`](../../collision-protocol/docs/v1.5-security-audit-report.md). the full audit that cleared v1.5.0 for mainnet
- [`POOL.md`](POOL.md). operator-facing pool mode guide, including the v1.5 worker experience
