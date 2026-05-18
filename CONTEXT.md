# Dungeon Runner (training)

Python repo for match simulation, RL training, and model artifacts. The playable game and replay source of truth live in **portfolio-site**; this context covers turning human **completed match replays** into improved policy weights and gated promotion.

## Language

**Completed match replay**:
A finished game exported as a **replay envelope** (seed, setup, history, version) keyed by match id in RTDB or Firebase export JSON.
_Avoid_: "game log", "save file"

**Replay envelope**:
The versioned contract for a completed match (portfolio-site authors the shape; dungeon-runner ingests and verifies it).
_Avoid_: "replay JSON" without version context

**Replay envelope version**:
Integer schema id on each envelope. v1 ingest accepts only `version === 1`; other or missing values are skipped (not coerced) with a recorded skip reason.
_Avoid_: "schema version" without the integer field name `version`

**Derived training row**:
One row per envelope `history` entry that has an `action`: **pre-action** `obs` (87) and `mask` (26) at that decision, plus **policy action index** for the action taken. Human and NN history steps both produce rows; `is_human` distinguishes **human step** entries. Row `step` is the 0-based **history index**; seat is **actor seat id** from that entry (not engine seat index). `phase` and `subphase` string columns copy the web session/kernel fields at that decision verbatim. Optional `model_id` from history `action.modelId` when present; optional `nn_debug` JSON blob on non-human rows when the web policy path returns debug info (null on **human step** rows).
_Avoid_: "sample", "datapoint"; post-action obs/mask; rows only for human steps; engine seat index as the stored seat key; Python **Match** phase names; `nn_debug` on human rows; deriving **policy action index** from Python `actions_codec`

**Policy action index**:
Integer 0–25 labeling the chosen action in the production policy layout (87-dim obs, 26-dim mask). Authored and mapped from replay `action.type` only in portfolio-site; Node replay emits it per step.
_Avoid_: "action id"; re-deriving indices in dungeon-runner Python

**Actor seat id**:
History field `actorSeatId` (`seat-1`…`seat-4`) naming which seat took the step. Maps to engine seat index only after `randomizeSeatsFromSetup` from envelope `seed` + `setup`; **replay verifier** requires it to match the engine’s current actor each step.
_Avoid_: Treating `seat-1` as the human player; ignoring `actorSeatId` during verify

**Human step**:
A history entry whose `action` has no `modelId` (human is not listed in setup; seat is seed-shuffled). Used to set `is_human` on **derived training rows**.
_Avoid_: Assuming `seat-1`, or requiring `humanSeatIds` in v1

**Gated promotion**:
Copying candidate weights from **training run artifact** to `models/<version>/` only after **promotion gates** pass, so production `latest` does not regress.
_Avoid_: "deploy", "release" (those include portfolio-site TF.js sync); semver paths for failed runs

**Training run artifact**:
Directory `models/runs/<run_id>/` with candidate `policy.weights.h5` and **metrics artifact** written atomically per BC/PPO run; **gated promotion** (#8) reads this bundle and copies to `models/<version>/` only on pass.
_Avoid_: Failed candidates under `models/<version>/`; metrics only under **training data root**

**Metrics artifact**:
`metrics.json` inside **training run artifact**: `run_id`, timestamp, `parent_weights`, **replay eval metrics**, **sim eval metrics**, training losses, and inputs for **promotion gates**.
_Avoid_: Metrics only on promoted versions; omitting `parent_weights`

**Promotion gates**:
Non-negotiable checks before **gated promotion**: **replay eval metrics** at or above **replay accuracy floor**, plus **sim eval metrics** non-regression vs `latest` on the frozen seed list in **eval config artifact** until PPO moves off **Python training sim**.
_Avoid_: "eval passed" without naming both bars; treating Python sim as co-equal with web rules truth; a second Node replay at eval time

**Replay eval metrics**:
Masked action accuracy on val-split **human step** **derived training rows**: candidate and `latest` forward on stored `obs`/`mask`, compared to stored **policy action index**; labels are **web-authoritative labels** from dataset build—no second Node replay at metrics time. **Disagreement rate** (report only) is the fraction of those rows where masked argmax(candidate) ≠ masked argmax(`latest`) on the same `obs`/`mask`.
_Avoid_: Re-stepping val envelopes through Node for promotion; using "replay" to mean live engine re-run at eval time; treating disagreement as label error rate

**Training parent**:
BC and PPO both start from `models/latest/policy.weights.h5`; each run still records that resolved path as `parent_weights` for audit, even though v1 does not expose alternate parents.
_Avoid_: "checkpoint" without saying latest vs semver

**Frozen eval suite**:
Fixed held-out match ids (~20% of ingested matches, chosen once) for **replay eval metrics** on **derived training rows**, plus a fixed seed list for **legacy Python sim benchmarks** vs `latest` and RandomBot. No separate disagreement slice or self-play matrix in v1.
_Avoid_: "validation set" without specifying match-level holdout vs step-level; implying metrics re-invoke the **web game engine**

**Eval suite artifact**:
Frozen file at `data/replays/eval_suite.json` (alongside ingest manifest), produced once by `eval_suite init`. Lists `val_match_ids`, suite version, and sampling seed; val ids chosen via seeded random ~20% (minimum one val match when at least two matches exist). Committed after first init so splits are stable across machines.
_Avoid_: Re-deriving the val split each time the dataset is built

**Eval config artifact**:
File at `data/replays/eval_config.json` under **training data root**: **replay accuracy floor**, fixed **legacy Python sim benchmarks** seed list, and **sim regression tolerance** (default `0.01` at **eval config init**). Separate from **eval suite artifact** so holdout re-init does not overwrite thresholds.
_Avoid_: Storing the floor or sim seeds only inside **eval suite artifact** or per-run `metrics.json`

**Sim regression tolerance**:
ε in **eval config artifact** for the sim leg of **promotion gates**: candidate win rate may be up to ε below `latest` on the frozen seed list (default `0.01`).
_Avoid_: Strict zero tolerance on small seed lists; conflating with replay accuracy floor

**Sim eval metrics**:
Win rate on **legacy Python sim benchmarks**: candidate and **training parent** (`latest`) each run the full frozen seed list independently vs RandomBot on **Python training sim**; **promotion gates** sim leg passes when candidate win rate ≥ latest win rate minus **sim regression tolerance**.
_Avoid_: Paired per-seed sign tests in v1; strict `>` with zero tolerance on small seed lists

**Replay accuracy floor**:
Minimum val **replay eval metrics** masked accuracy for **promotion gates**; set once to the first BC baseline run’s `replay.val_masked_accuracy` exactly (stored in **eval config artifact**), not overwritten by later runs unless a maintainer resets it.
_Avoid_: A fixed threshold before any baseline exists; a margin below baseline accuracy; recomputing the floor every train run

**BC baseline run**:
First BC training run that persists **replay accuracy floor** from its **metrics artifact** when the floor was previously null; that run may **gated promote** if all **promotion gates** pass after the floor is written.
_Avoid_: A separate calibration-only run that never promotes; an arbitrary hand-set floor

**Promotion gates (pre-floor)**:
While **replay accuracy floor** is null in **eval config artifact**, **gated promotion** fails closed (no promote); BC/PPO may still emit **training run artifact**. Replay leg applies only after the first **BC baseline run** writes the floor.
_Avoid_: Skipping the replay leg until a floor exists; blocking BC training until a floor exists

**Post-freeze ingest**:
Match ids ingested after the eval suite is frozen join the **train** split only; `val_match_ids` change only when the eval suite version is bumped and init is re-run.
_Avoid_: Automatically adding new matches to the held-out set

**Dataset split tag**:
Each **derived training row** carries `split: train | val` by match id (all steps stored for both splits). BC trains on train-split rows; replay eval metrics use val-split **human step** rows.
_Avoid_: "validation set" as a separate on-the-fly recompute from raw only

**Eval suite init**:
Required once before dataset build; creates **eval suite artifact** from **verify manifest** `verified` match ids. Refuses to run when fewer than two verified ids exist (no artifact written). Dataset build refuses to run if the artifact is absent.
_Avoid_: Implicit or auto-created val splits during dataset build; init with zero or one val match via empty or 100% holdout

**Eval config init**:
Creates **eval config artifact** with sixteen fixed sim seeds (default integers `0`–`15`) and **sim regression tolerance**; `replay_accuracy_floor` null until the first BC baseline run sets it. Committed alongside **eval suite artifact** after bootstrap. Metrics/gate stages fail fast if missing (no auto-create on train).
_Avoid_: Auto-generating config on first train; storing sim seeds only in per-run `metrics.json`

**Gated promotion path**:
A model may promote after BC alone if **promotion gates** pass; PPO is optional per run, not a prerequisite for promotion in v1.
_Avoid_: Implying every promoted artifact went through PPO

**Web game engine**:
The portfolio-site JavaScript kernel (`engine/kernel.js`), invoked via Node for replay verify, dataset labels, and any step that must match live play. Single authoritative rules runtime for this pipeline; no Python parity target.
_Avoid_: "the engine" without saying web vs Python; implying two implementations must stay aligned

**Web-authoritative labels**:
**Derived training rows** from human **completed match replays**, produced only by replaying through the **web game engine** (Node), never by Python `Match`.
_Avoid_: "Python verifier"; golden fixtures as a substitute for live replay (fixtures test the Node path only)

**Python training sim**:
Legacy in-repo `Match` for existing PPO / self-play until the training stack moves off it; not maintained for parity with the **web game engine** and expected to be deprecated after the replay pipeline ships.
_Avoid_: Treating Python sim as co-equal source of truth; investing in Python–JS alignment

**Replay verifier**:
Pipeline stage that stepwise replays each ingested **replay envelope** through the **web game engine**, requiring terminal phase `match-over`, **actor seat id** alignment, legal actions, and a valid **policy action index** for each history action (via web `encodeActionIndex`); each **pending verify** match is checked via one Node invocation; outcomes are recorded in **verify manifest**; failing matches do not reach dataset build.
_Avoid_: "valid replay" (ingest shape only); Python-based replay check; treating replay-without-`match-over` as verified; re-verifying the full corpus or auto-retrying `failed` every run

**Verified replay**:
A **completed match replay** that passed **replay verifier** (full replay to `match-over` with web rules); listed in **verify manifest** `verified`.
_Avoid_: Conflating with ingest-only eligibility or structural envelope checks alone

**Pending verify**:
An ingested match id with `raw/{matchId}.json` present, listed in **ingest manifest** `ingested`, and absent from both **verify manifest** `verified` and `failed`.
_Avoid_: Treating skipped ingest ids as verify input; re-running `failed` without **manual re-verify**

**Verify manifest**:
`verify_manifest.json` under **training data root** with `verified` (match ids that passed **replay verifier**) and `failed` (`id` + structured **verify failure**). Pass/fail only—no obs/mask sidecars at verify time; **dataset build** re-invokes the **web game engine** per verified match. Separate from **ingest manifest**.
_Avoid_: Merging verify state into **ingest manifest**; caching full replay traces at verify unless a later perf ADR adds **verified replay artifact**

**Verify failure**:
Structured record for a failed **replay verifier** run: stable `code` (e.g. `rng_chain_break`, `actor_mismatch`, `illegal_action`, `match_not_over`, `unmapped_action_type`, `engine_error`), optional history `step` index, optional `detail` string. Emitted by the Node verify harness (dungeon-runner–owned entrypoint importing web modules); stored on **verify manifest** `failed` entries.
_Avoid_: Free-text-only reasons; Python-invented failure taxonomy

**Manual re-verify**:
Re-running **replay verifier** on a match id after it appears in **verify manifest**; maintainer removes that id from `verified` and `failed`, then re-runs verify. Raw envelope must still exist under `raw/`.
_Avoid_: Expecting verify to refresh automatically when `raw/` is overwritten in place

**Verify run atomicity**:
A single verify run either completes all pending replays and then updates **verify manifest**, or aborts with no manifest changes (Node failure, per-match replay error before commit).
_Avoid_: Leaving a half-updated **verify manifest** after a failed run

**Replay test fixture**:
Committed export snippet envelopes under dungeon-runner `tests/fixtures/replay/` for verifier wiring and `action.type` coverage; not a substitute for portfolio-site’s canonical golden replay file.
_Avoid_: Duplicating the full golden envelope in dungeon-runner; running golden tests without **web engine root**

**Canonical golden replay**:
portfolio-site `engine/fixtures/golden-seed-4242-two-pass.json` (or successor); full-match regression for **replay verifier** when **web engine root** is set.
_Avoid_: Treating snippet fixtures as the only regression gate

**Manual pipeline run**:
A maintainer-run script (or script chain) that performs ingest → verify → dataset → train → gate; scheduling/cron is out of scope, not live data access. Implemented as staged CLIs plus a thin `run-all` orchestrator that calls them in order.
_Avoid_: "manual ingest" meaning export files only; a monolith with no stage boundaries

**Replay pipeline CLI**:
Shared package entry point with subcommands (`ingest`, later `verify`, `dataset`, …) invoked as `python -m dungeon_runner.replay.cli <stage>`; **run-all** orchestrator calls the same stages in order.
_Avoid_: A separate top-level script per pipeline stage with no shared module

**Live replay ingest**:
Ingest reads new matches directly from Firebase RTDB `dungeonRunnerCompletedMatches` (incremental by top-level match key), not only from dropped export JSON files.
_Avoid_: Conflating "no cron" with "no RTDB pull"

**Firebase database URL**:
The only required Firebase env var for **live replay ingest** v1: `FIREBASE_DATABASE_URL` in gitignored `.env` at repo root (loaded automatically on ingest; same value as portfolio-site `VITE_FIREBASE_DATABASE_URL`). API key and other web client vars are not loaded by ingest until authenticated RTDB access is needed.
_Avoid_: "Firebase config" when only the database URL is needed

**Web engine root**:
Filesystem path to the portfolio-site repo root, supplied as `PORTFOLIO_SITE_ROOT` in gitignored `.env` (loaded on verify/dataset stages). **Replay verifier** loads existing portfolio-site modules (`debug/replaySession.js`, `nn/policyAdapter.js`, …) from that path; verify fails fast if unset.
_Avoid_: Vendoring the kernel into dungeon-runner; submodule as the v1 default; large new portfolio-site APIs for training

**Pipeline consumer**:
dungeon-runner treats portfolio-site as producer of **completed match replay** envelopes and rules truth; v1 work consumes those outputs and orchestrates Node against existing web modules, with portfolio-site code changes kept minimal (e.g. exporting `encodeActionIndex` from `policyAdapter.js` when the harness needs it).
_Avoid_: Blocking dungeon-runner issues on portfolio-site feature PRs; duplicating envelope or replay logic in Python

**RTDB ingest access**:
Hand-run ingest uses the Realtime Database REST API with **Firebase database URL** from `.env`; optional `--from-export` for offline replay. Service account / Admin SDK is deferred until rules require it.
_Avoid_: "Firebase pull" implying export files only

**RTDB incremental fetch**:
**Live replay ingest** lists match ids with a shallow `GET …/dungeonRunnerCompletedMatches.json?shallow=true`, then fetches `…/{matchId}.json` only for keys absent from **ingest manifest**. **Offline export ingest** reads the full top-level map from one local file and filters by manifest in memory.
_Avoid_: Re-downloading the entire RTDB map on every ingest run

**Offline export ingest**:
Ingest path via `--from-export path.json` reading the same top-level match-id map from a local file; does not contact RTDB and does not require **Firebase database URL**.
_Avoid_: Treating export files as the only v1 input

**Training data root**:
Gitignored directory (default `data/replays/`, overridable via pipeline `--data-dir`) holding `raw/{matchId}.json` envelopes, `manifest.json` (**ingest manifest**), `verify_manifest.json` (**verify manifest**), **eval suite artifact**, **eval config artifact**, and per-match **derived match artifact** files under `derived/`.
_Avoid_: Storing replays under `models/` or the repo root

**Derived match artifact**:
Per-match directory `derived/{matchId}/` with `rows.parquet` (**derived training rows**) and `meta.json` (`match_id`, **dataset encoding version**, `row_count`, `built_at` UTC ISO-8601). Produced by **dataset build**; presence and version stamp determine **pending dataset**. Node harness emits row JSON on stdout per match; Python writes Parquet and metadata.
_Avoid_: One monolithic store with no per-match boundary; SQLite as v1 default; Node writing Parquet in v1; conflating with raw envelope or verify outcome

**Derived store**:
All **derived match artifact** trees under `derived/` in **training data root**; format v1 is Parquet per match, not SQLite.
_Avoid_: "dataset.db"; a single corpus-wide file as the only layout

**Dataset encoding version**:
Integer constant shared by the dataset CLI and Node harness (bump both when obs/mask/**policy action index** encoding changes). Stored in each `derived/{matchId}/meta.json`; **dataset build** skips a verified match when on-disk version matches current.
_Avoid_: Content-hash of raw envelope as the only skip signal (envelope unchanged but encoder fixed must rebuild)

**Dataset build (full refresh)**:
CLI flag `--all` re-encodes every **verify manifest** `verified` id, replacing **derived match artifact** trees on success. Default run processes **pending dataset** only.
_Avoid_: A separate `rebuild-all` subcommand for the same behavior; silent full rebuild as default

**Pending dataset**:
A **verified replay** match id with no **derived match artifact**, or whose `derived/{matchId}/meta.json` **dataset encoding version** is older than the current harness. Detected from the filesystem only—no **dataset manifest** in v1.
_Avoid_: Re-encoding every `verified` id on every run; treating `failed` verify ids as dataset input; a separate built-id manifest duplicating `derived/`

**Manual re-dataset**:
Re-running **dataset build** for a match after derived rows must be replaced; maintainer removes that match’s `derived/{matchId}/` tree, then re-runs dataset. Raw envelope must still exist and the match must remain in **verify manifest** `verified`.
_Avoid_: Expecting dataset to refresh when only `raw/` is overwritten without re-verify and re-dataset

**Dataset run atomicity**:
A single **dataset build** either completes all **pending dataset** matches in that run (each **derived match artifact** written fully) or aborts without leaving new or partial artifacts from that run.
_Avoid_: Committing per-match Parquet mid-run while later matches fail; treating a crashed run as success

**Ingest manifest**:
`manifest.json` under **training data root** with `ingested` (match id strings written to `raw/`) and `skipped` (`id` + machine-readable `reason`, e.g. `unsupported_version`). Dedup is by match id only—any id in either list is not re-fetched in v1.
_Avoid_: "processed ids file" without skip reasons

**Manual re-ingest**:
Pulling a match id again after it appears in **ingest manifest**; maintainer removes that id from the manifest and deletes its raw envelope first. RTDB payload edits under the same key are not detected automatically in v1.
_Avoid_: Expecting ingest to refresh changed RTDB rows for an existing key

**Ingest run atomicity**:
A single ingest run either completes all new pulls and eligibility writes and then updates **ingest manifest**, or aborts with no manifest changes (RTDB list failure, per-match fetch failure, or write error).
_Avoid_: Leaving a half-updated manifest after a failed pull

**Ingest eligibility**:
A match passes ingest only when it would pass portfolio-site `importReplayEnvelope` (version, seed, setup, history shape including per-entry RNG step rules; optional `presentationSpeedProfile` enum). Skipped matches are recorded in **ingest manifest** with a reason code. Structural checks only—**replay verifier** owns full replay to `match-over`.
_Avoid_: "valid replay" at ingest; expecting ingest to prove `match-over`

**Structural envelope check**:
Shape and static RNG-chain validation equivalent to `importReplayEnvelope`; performed at ingest only. **Replay verifier** does not repeat it for **pending verify** matches.
_Avoid_: Duplicating structural checks at verify; using structural pass as a substitute for **verified replay**

**Epic v1 success bar**:
At least one **gated promotion** plus a documented manual two-repo release path; **promotion gates** are guardrails inside that bar. PPO remains in the pipeline but is not required for a given promotion. Ingest is **live replay ingest** inside a **manual pipeline run**.
_Avoid_: "pipeline complete" without a promoted model

## Relationships

- A **completed match replay** is ingested raw, then verified, then yields many **derived training rows**
- **Gated promotion** in dungeon-runner precedes portfolio-site TF.js sync (separate repo step)
- **Web-authoritative labels** and **replay verifier** use the **web game engine** only; **Python training sim** is legacy PPO/self-play, not a parity target
- Only **human step** entries contribute human-labeled rows (unless a future envelope adds `humanSeatIds`)
- **Promotion gates** replay leg uses **replay eval metrics** on **frozen eval suite** val ids from **derived store** (no second Node replay at eval)
- **Training data root** holds raw envelopes, **ingest manifest**, and **verify manifest** before dataset build
- **Replay verifier** and dataset stages require **web engine root**
- **Eval suite init** draws match ids only from **verify manifest** `verified`
- **Eval suite artifact** must exist before dataset build; dataset builder fails fast if missing
- **Replay accuracy floor** lives in **eval config artifact**; first BC baseline sets it once
- **Eval config init** precedes metrics/gate runs; **eval config artifact** holds seeds and ε before the floor is set
- BC/PPO write **training run artifact**; **gated promotion** consumes **metrics artifact** + weights from that directory
- **Dataset build** reads **verify manifest** `verified` only; default scope is **pending dataset** (full refresh via CLI flag)

## Example dialogue

> **Dev:** "Can we promote if replay accuracy improved but we lose a bit vs RandomBot in sim?"
> **Domain expert:** "No — **Promotion gates** require both: sim must not regress vs `latest`, and replay accuracy must clear the held-out floor. Imitation alone isn't enough."

> **Dev:** "Is v1 done when ingest and verify work?"
> **Domain expert:** "No — **Epic v1 success bar** means a **gated promotion** actually shipped and we documented the manual sync to portfolio-site."

> **Dev:** "BC passed gates but PPO didn't — can we ship?"
> **Domain expert:** "Yes — **Gated promotion path** allows BC-only promotion when **promotion gates** pass."

> **Dev:** "Do I rerun the whole pipeline after fixing the verifier?"
> **Domain expert:** "Run the staged CLI from dataset onward, or use **manual pipeline run** `run-all` when you want the full chain including **live replay ingest**."

## Flagged ambiguities

- Issue #3 Python vs JS engine parity — resolved: **web game engine** via Node everywhere replay/rules matter; **Python training sim** is legacy, no alignment investment.
- Promotion gates sim leg vs engine deprecation — resolved: v1 keeps **legacy Python sim benchmarks** in **promotion gates**; Node sim gates are a later replacement when PPO leaves Python.
- Issue #3 verify outcomes storage — resolved: separate **verify manifest** (`verified` / `failed`), not merged into **ingest manifest**.
- Issue #3 locating portfolio-site — resolved: **web engine root** via `PORTFOLIO_SITE_ROOT` sibling checkout for v1, not submodule/npm package.
- Issue #3 verify batch scope — resolved: **pending verify** only; `failed` ids stay until **manual re-verify**.
- Issue #3 action.type → index 0–25 — resolved: mapping lives in portfolio-site only; Node emits **policy action index**; no Python replay codec.
- Issue #3 verify persistence — resolved: **verify manifest** pass/fail only; dataset stage replays verified matches for **derived training rows**.
- Issue #3 failed reasons — resolved: structured **verify failure** (`code`, optional `step`, optional `detail`).
- Issue #3 actorSeatId — resolved: strict **actor seat id** match each step (`actor_mismatch` on failure).
- Issue #3 test fixtures — resolved: **replay test fixture** snippets in dungeon-runner; **canonical golden replay** in portfolio-site when **web engine root** is set.
- Issue #3 Node invocation — resolved: one Node script run per match for v1.
- Issue #3 repo scope — resolved: **pipeline consumer**; issue #3 deliverable is dungeon-runner verify CLI/tests; portfolio-site changes minimal; Node harness in dungeon-runner imports existing web modules via **web engine root**.
- Issue #3 terminal phase — resolved: **verified replay** requires final `match-over`; else `match_not_over`.
- Issue #3 vs ingest structural checks — resolved: verify trusts ingest; no second **structural envelope check**.
- Issue #3 per-step policy index — resolved: stepwise verify; minimal web export of action→index encoder; `unmapped_action_type` on failure.
- Web vs Python engine authority — recorded in [ADR 0001](docs/adr/0001-web-game-engine-authoritative.md).
- Issue #1 "v1 out of automated scope: Firebase pull cron" — resolved: **live replay ingest** in hand-run scripts; automation of when to run is separate.
- Issue #2 "tolerate missing version" — resolved: missing **replay envelope version** is `unsupported_version` skip, same as web `importReplayEnvelope` (no coercion).
- Issue #2 / PRD "full `FIREBASE_*` in `.env`" — resolved: ingest v1 requires **`FIREBASE_DATABASE_URL` only**; `.env.example` documents that var.
- Issue #2 manifest content-hash — resolved: key-only dedup; **manual re-ingest** when a stored envelope must be replaced.
- Issue #2 acceptance "optional content hash" — superseded by key-only **ingest manifest** (see above).
- Issue #4 dataset run scope — resolved: default **pending dataset** only; **dataset encoding version** in `derived/{matchId}/meta.json`; **manual re-dataset** to force refresh; full corpus via explicit CLI flag.
- Issue #4 derived store format — resolved: **derived store** is Parquet per match (`rows.parquet` + `meta.json`), not SQLite.
- Issue #4 row grain — resolved: one **derived training row** per history entry with an `action`; pre-action obs/mask; all action steps stored (human + NN).
- Issue #4 row keys — resolved: `step` = history array index; seat column = **actor seat id** only.
- Issue #4 dataset run atomicity — resolved: same all-or-nothing rule as ingest/verify for artifacts created in that run.
- Issue #4 phase columns — resolved: `phase` + `subphase` strings from **web game engine** state at decision time (field names match portfolio-site session state).
- Issue #4 `nn_debug` — resolved: optional JSON on non-human rows only when web policy debug is available; omitted/null on **human step** rows.
- Issue #4 Python sim divergence metadata — resolved: omit in v1; **derived training rows** are **web-authoritative labels** only; PPO does not consume this store for labels.
- Issue #4 built-state tracking — resolved: **pending dataset** from `derived/{matchId}/` presence + `meta.json` version only; no dataset manifest.
- Issue #4 full refresh — resolved: `dataset --all` re-encodes all `verified` ids; default is **pending dataset** only.
- Issue #5 replay gate execution — resolved: **replay eval metrics** forward on stored obs/mask vs parquet labels; no second Node replay at eval time.
- Issue #5 eval suite init with fewer than two verified ids — resolved: **eval suite init** fail fast until at least two **verified replay** ids; no artifact.
- Issue #5 replay accuracy floor storage — resolved: **eval config artifact** sibling; floor set once from first BC baseline.
- Issue #5 sim regression rule — resolved: independent rollouts on same seed list; pass if candidate ≥ latest − **sim regression tolerance** (default ε in **eval config artifact**).
- Issue #5 disagreement rate — resolved: val **human step** rows; masked argmax candidate ≠ masked argmax `latest`; report only, not a gate in v1.
- Issue #5 eval config bootstrap — resolved: **eval config init** writes seeds + ε; floor null until first BC baseline; fail fast if missing at metrics time.
- Issue #5 metrics.json location — resolved: **metrics artifact** in `models/runs/<run_id>/` with candidate weights; semver dir only after **gated promotion**.
- Issue #5 replay accuracy floor value — resolved: exact first **BC baseline run** `replay.val_masked_accuracy`; baseline may promote if gates pass.
- Issue #5 promotion before floor — resolved: **fail closed** on promote while floor null; baseline run writes floor then gates apply.
- Issue #5 default sim seed count — resolved: sixteen seeds (`0`–`15`) at **eval config init** unless maintainer edits the artifact.
- Issue #5 default sim regression tolerance — resolved: **sim regression tolerance** `0.01` at **eval config init**.
- Issue #4 Node handoff — resolved: one Node process per match; JSON row array on stdout; Python owns Parquet + `meta.json`.
- Issue #4 `meta.json` — resolved: `match_id`, **dataset encoding version**, `row_count`, `built_at` (no raw content-hash in v1).
- portfolio-site `database.rules.json` is fully open today; **RTDB ingest access** may need Admin SDK or auth when rules tighten.
