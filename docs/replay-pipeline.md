# Replay training pipeline

Human **completed match replay** ingest for BC/PPO training. Normative envelope fields live in portfolio-site [**Replay envelope contract (v1)**](https://github.com/enmaku/portfolio-site/blob/main/src/features/dungeon-runner/CONTRACT.md#replay-envelope-contract-v1) (sibling: `$PORTFOLIO_SITE_ROOT/src/features/dungeon-runner/CONTRACT.md`). This doc is the dungeon-runner maintainer runbook: staged CLIs, on-disk layout, env vars, manifests, and cross-repo release pointers.

**Pipeline doc issue:** [dungeon-runner #10](https://github.com/enmaku/dungeon-runner/issues/10). Cross-repo glossary: [`CONTEXT.md`](../CONTEXT.md). Portfolio epic tracker: [portfolio-site #128](https://github.com/enmaku/portfolio-site/issues/128) (CONTRACT, harness exports, golden fixture, TF.js sync via [#127](https://github.com/enmaku/portfolio-site/issues/127)).

## CLI entry point

```bash
python -m dungeon_runner.replay.cli <stage> [stage flags…]
```

| Stage | Status | Purpose |
|-------|--------|---------|
| `ingest` | **implemented** | Pull RTDB or `--from-export` into **raw envelope store** + **ingest manifest** |
| `verify` | **implemented** | **Replay verifier** — replay each pending envelope through **web game engine** to `match-over` |
| `eval_suite` | **implemented** | `eval_suite init` — freeze **eval suite artifact** from **verify manifest** `verified` (≥2 ids) |
| `eval_config` | **implemented** | `eval_config init` — create **eval config artifact** (sim seeds, ε; floor null until BC baseline) |
| `dataset` | **implemented** | **Dataset build** — **derived match artifact** per **verified replay** (Node labels → Parquet) |
| `bc` | **implemented** | **BC policy training** → **training run artifact** under `models/runs/` |
| `ppo` | **implemented** | **BC-anchored PPO policy training** (requires `--bc-run`) |
| `publish` | implemented | **Gated promotion** to `models/<promoted version>/` + **production latest** symlink |
| `run-all` | stub | **Manual pipeline run** — orchestrate stages in order (see [Run-all](#run-all)) |

Unimplemented stages exit code **`2`** with a short stderr message; no fake work.

Shared flag (all stages that touch replay data):

| Flag | Default | Notes |
|------|---------|--------|
| `--data-dir` | `data/replays` | **Training data root** (gitignored `data/`) |

Weight I/O for `bc` / `ppo` / `publish` uses repo-root `models/` (not under `--data-dir`). See [Models layout](#models-layout).

### Bootstrap order

Maintainers run stages in this order on a fresh machine (after `.env` is set):

1. `ingest` (live RTDB or `--from-export`)
2. `verify` (requires `PORTFOLIO_SITE_ROOT`)
3. `eval_suite init` then `eval_config init` (≥2 **verified replay** match ids)
4. `dataset` (requires both eval artifacts)
5. `bc` (optional `ppo` via `--with-ppo` on `run-all`)
6. `publish` only when opting in (`--with-publish` on `run-all` or explicit `publish --run`)

Domain terms: [`CONTEXT.md`](../CONTEXT.md).

---

## Environment variables

Copy [`.env.example`](../.env.example) to `.env` (gitignored). The CLI loads it on every invocation via `python-dotenv`.

| Variable | Required for | Notes |
|----------|--------------|-------|
| `FIREBASE_DATABASE_URL` | `ingest` (live RTDB) | Same URL as portfolio-site `VITE_FIREBASE_DATABASE_URL`. Not needed for `ingest --from-export`. v1 uses RTDB REST read only (no service account until rules tighten). |
| `PORTFOLIO_SITE_ROOT` | `verify`, `dataset`, Node harness paths | Absolute path to portfolio-site checkout (**web engine root**). Used by verify harness and (when implemented) dataset label harness. |

Portfolio-site mirror for sync after **gated promotion**: `DUNGEON_RUNNER_ROOT` — see [portfolio-site #127](https://github.com/enmaku/portfolio-site/issues/127) and [#128](https://github.com/enmaku/portfolio-site/issues/128).

---

## Ingest

```bash
# Live RTDB (requires FIREBASE_DATABASE_URL in .env)
python -m dungeon_runner.replay.cli ingest

# Offline export (same top-level map as RTDB)
python -m dungeon_runner.replay.cli ingest --from-export path/to/export.json
```

| Flag | Notes |
|------|--------|
| `--data-dir` | **Training data root** (default `data/replays`) |
| `--from-export` | Top-level JSON map: keys = match ids, values = envelope objects |

RTDB paths (v1):

- Shallow list: `{FIREBASE_DATABASE_URL}/dungeonRunnerCompletedMatches.json`
- Per match: `…/dungeonRunnerCompletedMatches/{matchId}.json`

Upload contract and browser dedup: portfolio-site [CONTRACT.md — Persistence](https://github.com/enmaku/portfolio-site/blob/main/src/features/dungeon-runner/CONTRACT.md#persistence-contract-v1) (`dungeonRunnerCompletedMatches`, `dungeonRunner:uploadedMatchIds`).

### Ingest eligibility

Python port of portfolio-site `importReplayEnvelope` / `hasValidTurnBoundaryHistory` (`debug/replay.js`). No Node on ingest. Empty `history: []` is eligible; non-terminal replays fail at **verify**, not ingest.

#### Intentional strictness

| Check | Ingest | Web `importReplayEnvelope` |
|-------|--------|------------------------------|
| `version` | Python `type(x) is int` and `== 1` | `version !== 1` (rejects string `"1"`, float `1.5`) |
| `seed` | `type(x) is int` (excludes `bool`) | `Number.isInteger` |
| v2+ envelopes | `unsupported_version` until supported | same coarse reject |

#### Skip reason codes

| Code | Trigger | Web import (coarse) |
|------|---------|---------------------|
| `unsupported_version` | missing / non-int / not `1` | `INVALID_REPLAY` |
| `missing_seed` | missing / non-int `seed` | `INVALID_REPLAY` |
| `missing_setup` | missing / non-object `setup` | `INVALID_REPLAY` |
| `missing_history` | missing / non-array `history` | `INVALID_REPLAY` |
| `invalid_presentation_speed` | key present and not `cinematic` \| `brisk` | `INVALID_REPLAY` |
| `invalid_history` | entry shape or RNG chain break | `INVALID_REPLAY_HISTORY` |

Unknown top-level keys and optional fields (`createdAt`, `rulesHash`, NN `__debug`, body `matchId`, etc.) are preserved when eligibility passes.

### Ingest manifest and atomicity

See [Training data root layout](#training-data-root-layout). One ingest run lists pending ids, writes all `raw/` files, then updates `manifest.json` once. Mid-run RTDB or disk failure leaves the prior manifest and partial `raw/` from that run cleaned up.

### Manual re-ingest

Remove the id from `manifest.json` (`ingested` / `skipped`) and delete `raw/{matchId}.json`, then re-run `ingest`. v1 does not auto-detect RTDB payload changes under the same key.

---

## Verify

**Replay verifier issue:** [dungeon-runner #3](https://github.com/enmaku/dungeon-runner/issues/3).

```bash
python -m dungeon_runner.replay.cli verify
```

| Flag | Notes |
|------|--------|
| `--data-dir` | **Training data root** |

Requires `PORTFOLIO_SITE_ROOT`. Processes **pending verify** ids only: in ingest `ingested`, `raw/{matchId}.json` present, absent from verify `verified` and `failed`. One Node process per match (`src/dungeon_runner/replay/harness/verify_match.mjs`). Outcomes land in `verify_manifest.json` atomically after all pending matches finish.

### Node harness

Entry: `src/dungeon_runner/replay/harness/verify_match.mjs` (dungeon-runner–owned; imports portfolio-site via `PORTFOLIO_SITE_ROOT`).

| Import | Path under `PORTFOLIO_SITE_ROOT` |
|--------|----------------------------------|
| `createInitialMatchState`, `applyAction`, `MATCH_PHASES` | `src/features/dungeon-runner/engine/kernel.js` |
| `encodeActionIndex` | `src/features/dungeon-runner/nn/policyAdapter.js` |

Per match: `node verify_match.mjs raw/{matchId}.json` → stdout JSON `{ "ok": true }` or `{ "ok": false, "failure": { "code", "step?", "detail?" } }`. Exit code `0` always (Python reads stdout).

Step loop: `createInitialMatchState(setup, { seed })` → for each history entry check `actorSeatId`, RNG chain (`rngStepBefore` / `rngStepAfter` vs `state.rng.step`), `encodeActionIndex` ≥ 0, `applyAction` ok → assert `phase === match-over`.

**RNG re-check:** after a successful `applyAction`, the harness compares `result.state.rng.step` to the entry’s `rngStepAfter` (same failure code `rng_chain_break` as a broken chain at the next step’s `rngStepBefore`). Ingest eligibility only checks recorded `rngStepBefore` / `rngStepAfter` shape and continuity; it does not run the engine.

Does **not** re-run ingest structural checks (`importReplayEnvelope`).

#### Verify failure codes

`actor_mismatch`, `illegal_action`, `match_not_over`, `unmapped_action_type`, `rng_chain_break`, `engine_error`.

### Manual re-verify

Remove the match id from `verify_manifest.json` (`verified` and `failed`), keep `raw/{matchId}.json`, then re-run `verify`.

### Verify test fixtures

Committed under `tests/fixtures/replay/`:

| Fixture | Expected verify outcome |
|---------|-------------------------|
| `valid-match-over-seed42.json` | verified |
| `match-not-over.json` | `match_not_over` |
| `actor-mismatch.json` | `actor_mismatch` |
| `rng-chain-break.json` | `rng_chain_break` |
| `unmapped-action-type.json` | `unmapped_action_type` |
| `illegal-action.json` | `illegal_action` |

With `PORTFOLIO_SITE_ROOT` set, tests also replay portfolio-site `engine/fixtures/golden-seed-4242-two-pass.json` as a v1 envelope (history from `expected.history`). That fixture ends before **match over**; verify must report `match_not_over` with no per-step failure.

```bash
PORTFOLIO_SITE_ROOT=/path/to/portfolio-site pytest tests/replay/
```

---

## Eval suite init

```bash
python -m dungeon_runner.replay.cli eval_suite init [--data-dir data/replays] [--sampling-seed 42]
```

| Flag | Notes |
|------|--------|
| `--data-dir` | **Training data root** |
| `--sampling-seed` | Seeded holdout sampling (default `42`) |

- Reads **verify manifest** `verified` match ids (sorted).
- Refuses if fewer than two verified ids (**exit `1`**, stderr message, no artifact written). Success prints val id count and lists each `val:` match id (**exit `0`**).
- Always overwrites **eval suite artifact** `eval_suite.json` (no skip-if-exists). Atomic write via `eval_suite.json.tmp` → rename.
- Holdout: `k = max(1, round(0.2 * n))` val ids sampled with `random.Random(sampling_seed)` from sorted verified ids.
- Run once before first `dataset` build; bump `suite_version` + re-init to change holdout.

### Eval suite artifact (`eval_suite.json`)

```json
{
  "suite_version": 1,
  "sampling_seed": 42,
  "created_from_match_ids": ["match-001", "match-002"],
  "val_match_ids": ["match-001"]
}
```

- **`suite_version`**: integer schema version (v1 = `1`).
- **`sampling_seed`**: seed passed to `--sampling-seed` at init.
- **`created_from_match_ids`**: sorted snapshot of verify `verified` at init time.
- **`val_match_ids`**: frozen holdout match ids (~20%, minimum one when n ≥ 2).

**Split tags:** `split_resolver.split_for(match_id, suite)` (and `split_for_match_id` with `--data-dir`) return `val` when the id is in `val_match_ids`, else `train`. Post-freeze verified ids not in `val_match_ids` are **train** (not re-sampled each dataset build).

---

## Eval config init

```bash
python -m dungeon_runner.replay.cli eval_config init [--data-dir data/replays] [--overwrite]
```

| Flag | Notes |
|------|--------|
| `--data-dir` | **Training data root** |
| `--overwrite` | Replace existing `eval_config.json` |

- Creates **eval config artifact** `eval_config.json` (**exit `0`**). Refuses if artifact already exists unless `--overwrite` (**exit `1`**). Success prints seed count, ε, and floor value (**exit `0`**).
- Atomic write via `eval_config.json.tmp` → rename (same helper as **eval suite artifact**).

### Eval config artifact (`eval_config.json`)

```json
{
  "sim_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "sim_regression_tolerance": 0.01,
  "replay_accuracy_floor": null
}
```

- **`sim_seeds`**: sixteen fixed **legacy Python sim benchmarks** seeds (`0`–`15`).
- **`sim_regression_tolerance`**: ε for **promotion gates** sim leg (default `0.01` at init).
- **`replay_accuracy_floor`**: `null` until the first **BC baseline run**; **floor recorder** (`record_floor_if_needed`) sets it once to `metrics.replay.val_masked_accuracy` and thereafter skips updates unless a maintainer resets the artifact.

---

## Eval metrics (issue #5)

Shared Python modules under `dungeon_runner.replay.eval` (no separate CLI stage):

| Module | Role |
|--------|------|
| `replay_metrics` | Val **human step** rows from **derived store** Parquet: forward candidate + `latest` on stored `obs`/`mask` vs **policy action index**; **disagreement rate** is report-only |
| `derived_store.load_derived_rows` | Read `derived/{matchId}/rows.parquet` (`split`, `is_human`, `obs`, `mask`, `policy_action_index`) |
| `sim_metrics` | Independent **Python training sim** rollouts vs RandomBot on **eval config artifact** seeds |
| `write_metrics` | Atomic `metrics.json` beside `policy.weights.h5` in **training run artifact** |
| `evaluate_gates` | Pass/fail from committed metrics + **eval config artifact** (preview in `bc`/`ppo`, real in `publish`) |
| `record_floor_if_needed` | First **BC baseline run** writes **replay accuracy floor** into **eval config artifact** |

**BC policy training** and **BC-anchored PPO policy training** call `replay_metrics` + `sim_metrics` after training, then `write_metrics`. **Publish CLI** trusts the committed **metrics artifact** (no second eval pass).

---

## Dataset

Tracked in [dungeon-runner #4](https://github.com/enmaku/dungeon-runner/issues/4).

```bash
python -m dungeon_runner.replay.cli dataset [--data-dir data/replays]
python -m dungeon_runner.replay.cli dataset --all [--data-dir data/replays]
```

| Flag | Notes |
|------|--------|
| `--data-dir` | **Training data root** |
| `--all` | Re-encode every **verify manifest** `verified` id (default: **pending dataset** only) |

- Requires `PORTFOLIO_SITE_ROOT` and **eval suite artifact** (exit `1` if missing).
- Does not require **eval config artifact** (BC/sim eval only).
- **Dataset build** invokes **web game engine** per verified match; writes `derived/{matchId}/rows.parquet` + `meta.json` (**dataset encoding version**, `row_count`, `built_at`).
- All history **action** steps → **derived training rows**; `is_human=true` only on **human step** rows (seat resolved at build time). See [CONTRACT.md](https://github.com/enmaku/portfolio-site/blob/main/src/features/dungeon-runner/CONTRACT.md#replay-envelope-contract-v1).
- Staging under `derived/.staging/` — all pending matches in the run succeed or none are committed (**dataset run atomicity**).

**`rows.parquet` columns (encoding v1):** `step`, `seat`, `obs` (87 floats), `mask` (26 ints), `policy_action_index`, `phase`, `subphase`, `is_human`, `model_id`, `nn_debug`, `match_id`, `split` (`train` / `val` from **eval suite artifact**).

**Manual re-dataset:** delete `derived/{matchId}/`, keep raw + verify `verified`, re-run `dataset`.

---

## BC policy training

```bash
python -m dungeon_runner.replay.cli bc [--data-dir data/replays] [--run-id ID] [--no-gate-preview]
```

| Flag | Default | Notes |
|------|---------|--------|
| `--data-dir` | `data/replays` | Replay/derived/eval artifacts only |
| `--run-id` | `bc-<UTC compact>` | **Training run id** under `models/runs/` |
| `--no-gate-preview` | preview on | Skip **gate evaluator preview** (non-mutating) |

Exit **`1`** on prerequisite or training failure (no committed run dir). Requires `pip install -e ".[train]"` (TensorFlow).

- **Training parent:** `models/latest/policy.weights.h5`.
- Prerequisites: **derived store** with train/val **human step** rows, eval artifacts, val ids ⊆ suite.
- Post-train: `replay_metrics` on val **human step** Parquet rows + `sim_metrics` on **eval config artifact** `sim_seeds`; `write_metrics` → `models/runs/<run_id>/metrics.json`.
- **Floor recorder** sets **replay accuracy floor** on first **BC baseline run**; **gate evaluator preview** by default (`--no-gate-preview` to skip).
- Does **not** promote; use `publish` or `run-all --with-publish`.

---

## BC-anchored PPO policy training

```bash
python -m dungeon_runner.replay.cli ppo --bc-run models/runs/bc-… \
  [--data-dir data/replays] [--run-id ID] \
  [--bc-anchor-lambda 0.1] [--bc-anchor-beta 0] \
  [--ray-workers 8] [--no-ray] [--no-gate-preview]
```

| Flag | Default | Notes |
|------|---------|--------|
| `--bc-run` | (required) | **PPO BC run** — init weights + **BC-only candidate** for **PPO BC regression check** |
| `--data-dir` | `data/replays` | Eval artifacts; **derived store** required when **BC anchor CE** `λ > 0` |
| `--run-id` | `ppo-<UTC compact>` | **Training run id** |
| `--bc-anchor-lambda` | `0.1` | **BC anchor CE** strength (`0` skips derived-store prerequisites) |
| `--bc-anchor-beta` | `0` | **BC anchor KL** strength |
| `--ray-workers` | `8` | Parallel **PPO rollout collection** (Ray path delegates to local in v1 pass 1) |
| `--no-ray` | Ray on | Single-process rollouts (debug / macOS) |
| `--no-gate-preview` | preview on | Skip preview vs `latest` |

Exit **`1`** on prerequisite/training failure (no committed run dir) or **PPO BC regression check** fail (artifact still committed). Requires `pip install -e ".[train]"` (TensorFlow).

- **Rollout match templates:** 20% learner vs **RandomBot**, 45% vs **BC-bot** (**frozen BC teacher**), 35% self-play.
- Post-train: same `replay_metrics` + `sim_metrics` + `write_metrics` path as `bc` (vs `latest` for preview gates).
- Records `ppo_bc_regression.pass` in **metrics artifact**; `publish` on `ppo-*` requires `true`.

---

## Publish (gated promotion)

```bash
python -m dungeon_runner.replay.cli publish --run models/runs/bc-… \
  [--data-dir data/replays] [--version v0.3]
```

| Flag | Notes |
|------|--------|
| `--run` | **Required.** Committed **training run artifact** dir (not `*.tmp`) |
| `--data-dir` | Loads **eval config artifact** for **publish gate evaluation** |
| `--version` | Optional minor line bump (e.g. `v0.3`); patch auto-bumps under current line |

- Trusts committed **metrics artifact** (no re-eval at publish).
- **Promotion gates (pre-floor):** fail closed while `replay_accuracy_floor` is null.
- On pass: `models/<promoted version>/`, **production latest** symlink, `models/promotions.jsonl`.
- First replay-pipeline promote: `v0.2`, then `v0.2.01`, `v0.2.02`, … — [ADR 0002](adr/0002-promoted-version-semver-and-latest-symlink.md).
- One-time migration: legacy duplicate `models/latest/` dir → symlink to `v0.1.30a` on first `publish`.
- Does **not** run **TF.js model sync**; see [Release to portfolio-site](#release-to-portfolio-site).

---

## Run-all

```bash
python -m dungeon_runner.replay.cli run-all [--data-dir data/replays] \
  [--with-ppo] [--with-publish]
```

**Not implemented** (exit `2` today — same stub behavior as other unimplemented stages). When shipped, the orchestrator invokes the same stage CLIs below in order, passing `--data-dir` through, and **fails fast**: the first child stage with a non-zero exit code aborts `run-all` without running later steps (no partial train/publish after a failed verify).

| Step | Stage | Env / prereqs | Notes |
|------|--------|---------------|--------|
| 1 | `ingest` | `FIREBASE_DATABASE_URL` for live RTDB | Default live pull; export-only workflows run `ingest --from-export` separately before `run-all` |
| 2 | `verify` | `PORTFOLIO_SITE_ROOT` | Pending verify only |
| 3 | `eval_suite init` | ≥2 verify `verified` ids | Always overwrites **eval suite artifact**; fails with exit `1` if holdout cannot be sampled |
| 4 | `eval_config init` | — | Creates artifact if missing; **skips** when `eval_config.json` exists (no `--overwrite` from `run-all` v1) |
| 5 | `dataset` | `PORTFOLIO_SITE_ROOT`, eval artifacts | Pending matches only |
| 6 | `bc` | derived + eval artifacts | **Gate evaluator preview** on by default |
| 7 | `ppo` | `--bc-run` from step 6 | Only with `--with-ppo` |
| 8 | `publish` | `--run` on last train artifact | Only with `--with-publish`; `ppo-*` if step 7 ran, else `bc-*` |

| Flag | Default | Notes |
|------|---------|--------|
| `--data-dir` | `data/replays` | Passed through to every child stage |
| `--with-ppo` | off | Chain **BC-anchored PPO policy training** after `bc` |
| `--with-publish` | off | Chain **gated promotion** on the last train artifact |

**Default `run-all` stops after `bc`.** PPO and promotion are opt-in so maintainers can review **metrics artifact** / preview before promoting.

**Today:** run steps 1–4 manually (`ingest` → `verify` → `eval_suite init` → `eval_config init`) before stub stages ship; `dataset` / `bc` / `ppo` / `publish` / `run-all` each exit `2` until their issues land.

Future: passthrough for `ingest --from-export` and `eval_config init --overwrite` on `run-all` may be added when the orchestrator ships (not in v1 CLI yet).

---

## Training data root layout

Default: `data/replays/` (entire `data/` is gitignored).

```
data/replays/
  manifest.json              # ingest manifest
  verify_manifest.json       # verify manifest
  eval_suite.json            # eval suite artifact (after eval_suite init)
  eval_config.json           # eval config artifact (after eval_config init)
  raw/{matchId}.json         # verbatim (RTDB) or normalized JSON (export)
  derived/{matchId}/
    rows.parquet             # derived training rows (after dataset)
    meta.json                # dataset encoding version, row_count, built_at
```

**Eval suite artifact** and **eval config artifact** are **frozen on disk** under this tree (local backup with replays; not committed to git).

### Ingest manifest (`manifest.json`)

```json
{
  "ingested": ["match-1778989461147"],
  "skipped": [{ "id": "match-old", "reason": "unsupported_version" }]
}
```

- **`ingested`**: match ids with `raw/{matchId}.json`.
- **`skipped`**: ingest eligibility failures; not retried until **manual re-ingest**.

Dedup is key-only (`ingested` ∪ `skipped`). v1 does not compare content hashes.

### Verify manifest (`verify_manifest.json`)

```json
{
  "verified": ["match-1778989461147"],
  "failed": [
    {
      "id": "match-bad",
      "reason": { "code": "illegal_action", "step": 12, "detail": "applyAction rejected" }
    }
  ]
}
```

- **`verified`**: full replay to `match-over` under web rules.
- **`failed`**: structured **verify failure**; not auto-retried.

Verify run atomicity matches ingest: all pending matches, then one manifest write.

### Raw envelope storage

- **RTDB**: response bytes as-is (plus trailing newline if omitted).
- **Export**: canonical minified JSON per match.

### Browser upload dedup (portfolio-site)

`dungeonRunner:uploadedMatchIds` in the browser avoids re-uploading to RTDB. Independent of **ingest manifest** dedup on the training machine.

---

## Models layout

Repo-root `models/` (not under `--data-dir`):

```
models/
  latest/                    # symlink → ../<promoted version>/ (production latest)
  runs/<run_id>/             # training run artifacts (bc-*, ppo-*)
    policy.weights.h5
    metrics.json
    tb/
  <promoted version>/        # gated promotion output (v0.2, v0.2.01, …)
  promotions.jsonl           # append-only promotion ledger
```

See [ADR 0002](adr/0002-promoted-version-semver-and-latest-symlink.md) and **gated promotion** terms in [`CONTEXT.md`](../CONTEXT.md).

---

## Release to portfolio-site

**Gated promotion** copies H5 into semver dirs and repoints **production latest** in dungeon-runner only. Players load TF.js from portfolio-site until maintainers run **TF.js model sync** ([#127](https://github.com/enmaku/portfolio-site/issues/127), umbrella [#128](https://github.com/enmaku/portfolio-site/issues/128)).

dungeon-runner [#11](https://github.com/enmaku/dungeon-runner/issues/11) tracks the epic **two-repo model release** checklist; full sync/smoke steps live in portfolio-site docs when that slice ships. This runbook does not duplicate the portfolio checklist (Pass 1 / #10 scope).

---

## Related

| Topic | Location |
|-------|----------|
| Replay envelope v1 (normative) | `$PORTFOLIO_SITE_ROOT/src/features/dungeon-runner/CONTRACT.md` · [GitHub](https://github.com/enmaku/portfolio-site/blob/main/src/features/dungeon-runner/CONTRACT.md) |
| Portfolio-site glossary | `$PORTFOLIO_SITE_ROOT/CONTEXT.md` · [GitHub](https://github.com/enmaku/portfolio-site/blob/main/CONTEXT.md) |
| Web engine authoritative | [ADR 0001](adr/0001-web-game-engine-authoritative.md) |
| Promoted version semver | [ADR 0002](adr/0002-promoted-version-semver-and-latest-symlink.md) |
| Portfolio epic v1 | [portfolio-site #128](https://github.com/enmaku/portfolio-site/issues/128) |
| Pipeline doc issue | [dungeon-runner #10](https://github.com/enmaku/dungeon-runner/issues/10) |
| Envelope / ingest parity | [dungeon-runner #9](https://github.com/enmaku/dungeon-runner/issues/9) |
| portfolio-site import tests | `src/features/dungeon-runner/debug/replay.test.js` |
| Canonical golden replay | `src/features/dungeon-runner/engine/fixtures/golden-seed-4242-two-pass.json` |
