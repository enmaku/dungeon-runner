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
A supervised transition `(observation, action_mask, action_index)` extracted from a verified replay, stored separately from raw envelopes so encoding can be rebuilt.
_Avoid_: "sample", "datapoint"

**Human step**:
A history entry whose `action` has no `modelId` (human is not listed in setup; seat is seed-shuffled). Used to set `is_human` on **derived training rows**.
_Avoid_: Assuming `seat-1`, or requiring `humanSeatIds` in v1

**Gated promotion**:
Writing `models/<version>/policy.weights.h5` only after eval gates pass, so production `latest` does not regress.
_Avoid_: "deploy", "release" (those include portfolio-site TF.js sync)

**Promotion gates**:
Non-negotiable checks before **gated promotion**: no regression vs current `latest` on Python sim benchmarks, plus held-out human **replay action accuracy** (masked, human steps only) at or above a floor set from the first BC baseline run.
_Avoid_: "eval passed" without naming both sim and replay bars

**Training parent**:
BC and PPO both start from `models/latest/policy.weights.h5`; each run still records that resolved path as `parent_weights` for audit, even though v1 does not expose alternate parents.
_Avoid_: "checkpoint" without saying latest vs semver

**Frozen eval suite**:
Fixed held-out match ids (~20% of ingested matches, chosen once) for replay metrics, plus a fixed seed list for Python sim benchmarks vs `latest` and RandomBot. No separate disagreement slice or self-play matrix in v1.
_Avoid_: "validation set" without specifying match-level holdout vs step-level

**Eval suite artifact**:
Frozen file at `data/replays/eval_suite.json` (alongside ingest manifest), produced once by `eval_suite init`. Lists `val_match_ids`, suite version, and sampling seed; val ids chosen via seeded random ~20% (minimum one val match when at least two matches exist). Committed after first init so splits are stable across machines.
_Avoid_: Re-deriving the val split each time the dataset is built

**Post-freeze ingest**:
Match ids ingested after the eval suite is frozen join the **train** split only; `val_match_ids` change only when the eval suite version is bumped and init is re-run.
_Avoid_: Automatically adding new matches to the held-out set

**Dataset split tag**:
Each **derived training row** carries `split: train | val` by match id (all steps stored for both splits). BC trains on train-split rows; replay eval metrics use val-split **human step** rows.
_Avoid_: "validation set" as a separate on-the-fly recompute from raw only

**Eval suite init**:
Required once before dataset build; creates **eval suite artifact** from verifier-passing match ids. Dataset build refuses to run if the artifact is absent.
_Avoid_: Implicit or auto-created val splits during dataset build

**Gated promotion path**:
A model may promote after BC alone if **promotion gates** pass; PPO is optional per run, not a prerequisite for promotion in v1.
_Avoid_: Implying every promoted artifact went through PPO

**Web-authoritative labels**:
Human replay-derived **derived training rows** are produced by replaying through portfolio-site engine semantics (Node kernel or shared golden contract), not assumed Python `Match` parity.
_Avoid_: "Python verifier" for human label paths

**Python training sim**:
The in-repo `Match` used for PPO / self-play against bots; may diverge from web until explicitly aligned. Not the source of truth for human imitation labels in v1.
_Avoid_: Using "the engine" without qualifying web vs Python

**Manual pipeline run**:
A maintainer-run script (or script chain) that performs ingest → verify → dataset → train → gate; scheduling/cron is out of scope, not live data access. Implemented as staged CLIs plus a thin `run-all` orchestrator that calls them in order.
_Avoid_: "manual ingest" meaning export files only; a monolith with no stage boundaries

**Live replay ingest**:
Ingest reads new matches directly from Firebase RTDB `dungeonRunnerCompletedMatches` (incremental by top-level match key), not only from dropped export JSON files.
_Avoid_: Conflating "no cron" with "no RTDB pull"

**Firebase database URL**:
RTDB endpoint for **live replay ingest**, from `FIREBASE_DATABASE_URL` in gitignored `.env` (same value as portfolio-site `VITE_FIREBASE_DATABASE_URL`). Other web config vars are not required for ingest v1.
_Avoid_: "Firebase config" when only the database URL is needed

**RTDB ingest access**:
Hand-run ingest uses the Realtime Database REST API at `{databaseURL}/dungeonRunnerCompletedMatches.json` (or per-child fetch), with **Firebase database URL** from `.env`; optional `--from-export` for offline replay. Service account / Admin SDK is deferred until rules require it.
_Avoid_: "Firebase pull" implying export files only

**Offline export ingest**:
Optional ingest path reading the same top-level match-id map from a local Firebase export JSON file (`--from-export`), for debug or when RTDB is unavailable.
_Avoid_: Treating export files as the only v1 input

**Training data root**:
Gitignored `data/replays/` tree: raw **completed match replay** JSON per match id under `raw/`, plus `manifest.json` for ingest dedup and skip reasons.
_Avoid_: Storing replays under `models/` or the repo root

**Ingest manifest**:
Lists which match ids were ingested to `raw/` and which were skipped with a reason (e.g. unsupported **replay envelope version**). No content-hash or overwrite tracking in v1.
_Avoid_: "processed ids file" without skip reasons

**Ingest eligibility**:
A match is ingested only when **replay envelope version** is `1` and `seed`, `setup`, and `history` are present with types matching web `importReplayEnvelope` (replay legality is verified later).
_Avoid_: "valid replay" at ingest (that is the verifier's job)

**Epic v1 success bar**:
At least one **gated promotion** plus a documented manual two-repo release path; **promotion gates** are guardrails inside that bar. PPO remains in the pipeline but is not required for a given promotion. Ingest is **live replay ingest** inside a **manual pipeline run**.
_Avoid_: "pipeline complete" without a promoted model

## Relationships

- A **completed match replay** is ingested raw, then verified, then yields many **derived training rows**
- **Gated promotion** in dungeon-runner precedes portfolio-site TF.js sync (separate repo step)
- **Web-authoritative labels** come from portfolio-site engine replay; **Python training sim** is separate and used for PPO/self-play
- Only **human step** entries contribute human-labeled rows (unless a future envelope adds `humanSeatIds`)
- **Promotion gates** replay leg uses the **frozen eval suite** match ids only
- **Training data root** holds raw envelopes before verify/dataset stages
- **Eval suite artifact** must exist before dataset build; dataset builder fails fast if missing

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

- Issue #1 "v1 out of automated scope: Firebase pull cron" — resolved: **live replay ingest** in hand-run scripts; automation of when to run is separate.
- portfolio-site `database.rules.json` is fully open today; **RTDB ingest access** may need Admin SDK or auth when rules tighten.
