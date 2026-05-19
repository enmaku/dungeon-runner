# Dungeon Runner (training)

Python repo for match simulation, RL training, and model artifacts. The playable game and replay source of truth live in **portfolio-site**; this context covers turning human **completed match replays** into improved policy weights and gated promotion.

## Language

**Completed match replay**:
A finished game exported as a **replay envelope** (seed, setup, history, version) keyed by match id in RTDB or Firebase export JSON.
_Avoid_: "game log", "save file"

**Replay envelope**:
The versioned contract for a completed match (portfolio-site authors the shape; dungeon-runner ingests and verifies it).
_Avoid_: "replay JSON" without version context

**Replay envelope contract (v1)**:
Normative field and eligibility rules for **replay envelope version** `1` in a dedicated portfolio-site `CONTRACT.md` section (`## Replay envelope contract (v1)`), with a v2 planned appendix in the same area. dungeon-runner adds a cross-linked pipeline doc for ingest-only extensions (skip reason codes, `raw/{matchId}.json` layout, RTDB map shape)—not a second full copy of the field list. Planned **replay envelope version** `2` fields are documented only in portfolio-site CONTRACT; ingest skips unsupported versions until a later issue implements them.
_Avoid_: Duplicating the full v1 spec or v2 roadmap in dungeon-runner `CONTEXT.md`; burying the spec as a single bullet under Debug only; treating dungeon-runner pipeline docs as the producer-owned rules source

**Replay envelope version**:
Integer schema id on each envelope. v1 ingest requires a true integer `1` (not string, bool, or float); missing or any other value is skipped as `unsupported_version` without coercion. portfolio-site `importReplayEnvelope` uses a looser `version !== 1` check—dungeon-runner may reject edge types web would reject anyway; pipeline doc notes any intentional divergence.
_Avoid_: "schema version" without the integer field name `version`; coercing missing version to `1`; accepting string `"1"` at ingest

**Derived training row**:
One row per envelope `history` entry that has an `action`: **pre-action** `obs` (87) and `mask` (26) at that decision, plus **policy action index** for the action taken. **Human step**, **Non-NN history step**, and NN steps can all produce rows; `is_human` is true only for **human step** entries. Row `step` is the 0-based **history index**; seat is **actor seat id** from that entry (not engine seat index). `phase` and `subphase` string columns copy the web session/kernel fields at that decision verbatim. Optional `model_id` from history `action.modelId` when present; optional `nn_debug` JSON blob on non-human rows when the web policy path returns debug info (null on **human step** rows).
_Avoid_: "sample", "datapoint"; post-action obs/mask; rows only for human steps; setting `is_human` from `modelId` absent alone; engine seat index as the stored seat key; Python **Match** phase names; `nn_debug` on human rows; deriving **policy action index** from Python `actions_codec`

**Policy action index**:
Integer 0–25 labeling the chosen action in the production policy layout (87-dim obs, 26-dim mask). Authored and mapped from replay `action.type` only in portfolio-site; Node replay emits it per step.
_Avoid_: "action id"; re-deriving indices in dungeon-runner Python

**Actor seat id**:
History field `actorSeatId` (`seat-1`…`seat-4`) naming which seat took the step. Maps to engine seat index only after `randomizeSeatsFromSetup` from envelope `seed` + `setup`; **replay verifier** requires it to match the engine’s current actor each step.
_Avoid_: Treating `seat-1` as the human player; ignoring `actorSeatId` during verify

**Non-NN history step**:
A history entry whose `action` has no `modelId` (human player and **randombot** steps both qualify). Envelope-level NN vs not-NN marker in v1 until **replay envelope version** 2 `humanSeatIds`.
_Avoid_: Calling these "human" in BC/training docs; inferring the human player seat from `modelId` alone

**Human step**:
A history entry whose **actor seat id** is the human player seat after setup seat shuffle (resolved at **dataset build** / **web game engine** replay—not from `modelId` absent alone). Sets `is_human: true` on **derived training rows**. Human is not listed in `setup.opponents`; seat assignment is seed-shuffled.
_Avoid_: Assuming `seat-1`; treating **Non-NN history step** as **human step**; requiring `humanSeatIds` in v1 envelope

**Gated promotion**:
Copying candidate weights from **training run artifact** to `models/<promoted version>/` only after **promotion gates** pass, then repointing **production latest** at that directory.
_Avoid_: "deploy", "release" (those include portfolio-site TF.js sync); semver paths for failed runs; copying weights into a non-symlink `models/latest/` tree

**Promoted version**:
Semver directory name under `models/` assigned on successful **gated promotion**. Legacy alpha dirs (`v0.1.29a`, `v0.1.30a`) used epoch numbers under the `v0.1` line with an `a` suffix; the replay-pipeline line starts at `v0.2` (first post-alpha promote), then auto-increments patch as two digits: `v0.2.01`, `v0.2.02`, … Minor line bumps (e.g. `v0.3`) are maintainer-only via `publish --version`. No letter suffixes on new promotes. Distinct from **training run id** under `models/runs/`.
_Avoid_: Using `bc-*` / `ppo-*` run ids as promoted dir names; continuing the `v0.1.*a` pattern for new promotes; unpadded patch segments like `v0.2.1` on the new line; allocator auto-advancing minor without an explicit flag

**Production latest**:
Symlink `models/latest/` → `../<promoted version>/`; holds `policy.weights.h5` via the target dir. Updated only by **gated promotion** (#8), never by `bc`/`ppo` or failed publish. Issue #8 includes a one-time migration replacing the legacy duplicate `latest/` directory with a symlink to `v0.1.30a` until the first `v0.2` **gated promotion** repoints it.
_Avoid_: A real directory that duplicates weights; manual copy into `latest` without publish; treating `latest` as a version string; conflating with **web deployed latest**

**Web deployed latest**:
portfolio-site directory `public/models/dungeon-runner/latest/` with TF.js `model.json` + shards; overwritten on each **TF.js model sync** from the new **promoted version** so default opponent `modelId: 'latest'` tracks production without editing setup. A copied deploy alias, not dungeon-runner’s **production latest** symlink and not semver history.
_Avoid_: Expecting portfolio `latest/` to symlink to `v0.2/`; skipping **web deployed latest** refresh when only semver was synced; using `models/runs/*` as the sync source id

**Dungeon-runner root**:
Filesystem path to the dungeon-runner repo for portfolio-site **TF.js model sync** (`DUNGEON_RUNNER_ROOT` in portfolio-site `.env.example`), mirroring **web engine root** (`PORTFOLIO_SITE_ROOT`) on the dungeon-runner side.
_Avoid_: "models repo path" without naming the env var; requiring GitHub clone when sibling checkout exists

**TF.js model sync**:
Manual portfolio-site step after **gated promotion**: one command converts `models/<promoted version>/policy.weights.h5` to semver TF.js, then regenerates **model catalog**. Overwrites **web deployed latest** only when the target id equals dungeon-runner **production latest** (including `--from-latest` resolution); otherwise semver TF.js only. Maintainer passes an explicit **promoted version** in the release checklist; optional `--from-latest` resolves from **production latest**. H5 source: local `DUNGEON_RUNNER_ROOT` when set and file exists; else shallow clone. Never a **training run id**.
_Avoid_: Bare `sync` as the only documented path; single-id sync of an old semver id overwriting **web deployed latest**; `sync … bc-*`; `sync --all` refreshing **web deployed latest** per directory in loop order; accepting explicit semver ids not in **promotion manifest** ledger

**Two-repo model release**:
Maintainer flow spanning **gated promotion** (dungeon-runner) then **TF.js model sync** (portfolio-site), documented in split checklists: dungeon-runner **Replay pipeline documentation** covers publish + pointer to portfolio-site; portfolio-site docs cover sync, **model catalog**, manual smoke play on `/projects/dungeon-runner`, and optional automated guard (e.g. `runtime.test.js` fails if **web deployed latest** artifacts missing after sync). Together they complete **Epic v1 success bar**. Excludes automated git commit, production Firebase deploy, and mandatory headless WebGL CI.
_Avoid_: Full checklist in one repo only; duplicate full steps in both repos; headless WebGL as the only smoke; conflating with **gated promotion** alone

**Portfolio-site coordination issue**:
Umbrella tracker in portfolio-site ([#128](https://github.com/enmaku/portfolio-site/issues/128)) for all epic v1 work in the playable repo: **replay envelope contract** in CONTRACT, minimal harness exports, golden fixture, env cross-links, and **TF.js model sync** / **release smoke** ([#127](https://github.com/enmaku/portfolio-site/issues/127)). May start in parallel with dungeon-runner #8 (**gated promotion**); first end-to-end promote → sync waits on a real **promoted version**. dungeon-runner #11 remains the epic cross-repo tracker.
_Avoid_: Blocking portfolio work on #8 merge; marking portfolio coordination done before CONTRACT + export + sync streams complete; conflating #127 alone with the full portfolio-site scope; all cross-repo work in one dungeon-runner PR

**Release smoke**:
Post-sync verification that TF.js loads and returns a legal action: required manual pass in the browser; optional fast Node test on **web deployed latest** artifacts. Not a second inference-quality eval.
_Avoid_: Calling **replay eval metrics** or **promotion gates** “smoke”; WebGL-only automation as v1 gate

**TF.js sync backfill**:
Rebuilding TF.js trees for older **promoted version** dirs (e.g. after converter fixes) without changing what players load as **web deployed latest**. Uses portfolio-site `sync --all` (or equivalent): convert every semver dir under dungeon-runner `models/`, refresh **web deployed latest** only from **production latest** target, then **model catalog**. Distinct from the post-promote single-id sync.
_Avoid_: Treating backfill like a new promote; `--all` overwriting **web deployed latest** with whichever dir sorts last

**Model catalog**:
portfolio-site `public/models/dungeon-runner/models.json` listing on-disk TF.js dirs (semver ids plus `latest`); regenerated after sync from directories that contain `model.json`. Append-only in normal releases: new **promoted version** ids are added, not removed. Re-syncing an existing semver id overwrites that folder only for **TF.js sync backfill** or converter fixes.
_Avoid_: Hand-editing the catalog; deleting historical semver dirs in routine promote; treating catalog ids as dungeon-runner **training run id**s

**Deployed model version**:
A semver-labeled TF.js tree under `public/models/dungeon-runner/<promoted version>/` mirroring a dungeon-runner **promoted version** H5 bundle. Immutable identity in git history; bytes may be replaced only by intentional re-sync of the same id.
_Avoid_: "model version" meaning **training run id**; expecting portfolio semver folders to track `models/runs/`

**Promotion manifest**:
Audit record written only on successful **gated promotion**: per **promoted version** dir (`promotion.json` with version, `parent_weights`, timestamp, **training run id**, pointer to metrics snapshot) plus one append-only line in repo-level `models/promotions.jsonl`. A full copy of the run **metrics artifact** lives in the promoted dir alongside weights.
_Avoid_: Manifest on failed publish; manifest only in the run dir under `models/runs/`; overwriting prior JSONL lines

**Publish gate evaluation**:
Stage `publish` applies **promotion gates** by reading the committed run **metrics artifact** and **eval config artifact** only—no second **replay eval metrics** or **sim eval metrics** pass. Same gate rules as **gate evaluator preview**, but may copy weights and write **promotion manifest** on pass. For `ppo-*` artifacts, also requires `ppo_bc_regression.pass: true` from stage `ppo`.
_Avoid_: Re-running eval at publish time; promoting from `.tmp` staging dirs; treating preview output as promotion; skipping **PPO BC regression check** at publish

**Publish CLI**:
**Replay pipeline CLI** stage `publish` (`python -m dungeon_runner.replay.cli publish`) with required `--run` (committed **training run artifact** dir, not `*.tmp`) and `--data-dir` (default **training data root**) to load **eval config artifact** for **publish gate evaluation**; exit 0 only when **gated promotion** completes. Each **training run id** may promote at most once (reject if `run_id` already in `models/promotions.jsonl`).
_Avoid_: Run-id-only resolution without an explicit path in v1; implicit promote from `bc`/`ppo`; second promote of the same run artifact; hard-coded eval config path ignoring `--data-dir`

**Publish run atomicity**:
A single `publish` stages `models/<promoted version>.tmp/` (weights, metrics copy, `promotion.json`), renames to final semver dir, then atomically repoints **production latest** and appends `models/promotions.jsonl`. Failure before the final steps leaves `latest` and the JSONL ledger unchanged.
_Avoid_: Appending JSONL before weights exist; repointing `latest` at a partial version dir; leaving a committed semver dir without a matching ledger line

**Training run artifact**:
Directory `models/runs/<run_id>/` with candidate `policy.weights.h5`, **metrics artifact**, and TensorBoard logs under `tb/` written atomically per BC/PPO run; **gated promotion** (#8) reads this bundle and copies to `models/<version>/` only on pass. v1 write: stage under `models/runs/<run_id>.tmp/` (TensorBoard `tb/`, weights, then `metrics.json` last), then rename to `models/runs/<run_id>/`; failed runs must not leave a committed final directory.
_Avoid_: Failed candidates under `models/<version>/`; metrics only under **training data root**; partial dirs with weights but no **metrics artifact**; TensorBoard only outside the staged run dir

**Training run id**:
Directory name under `models/runs/` and `run_id` field in **metrics artifact**. Default auto-generated as stage prefix + UTC compact time (`YYYYMMDDTHHMMSSZ`): `bc-…` for **BC policy training**, `ppo-…` for **BC-anchored PPO policy training** (e.g. `ppo-20260518T150000Z`); optional CLI `--run-id` override for tests and reruns.
_Avoid_: UUID-only ids with no time signal; requiring `--run-id` on every maintainer run; embedding parent BC id in the default `ppo-` name (use **metrics artifact** `parent_weights` / init path for lineage)

**Metrics artifact**:
`metrics.json` inside **training run artifact**: `run_id`, timestamp, `parent_weights`, **replay eval metrics**, **sim eval metrics**, training losses, and inputs for **promotion gates**. `train.bc_loss` is masked CE on all train-split **human step** rows evaluated on **BC best checkpoint** weights (post-restore), not last-epoch or loop-only estimates. **BC-anchored PPO policy training** adds `train.ppo_loss`, `train.bc_anchor_ce`, and `train.bc_anchor_kl` (when `β > 0`) on final saved weights, mirroring TensorBoard scalars. `parent_weights` is the absolute resolved path to weights at run start (**training parent** for `bc`; **PPO BC run** checkpoint for `ppo`).
_Avoid_: Metrics only on promoted versions; omitting `parent_weights`; `train.bc_loss` from final epoch or uncoupled from saved weights; repo-relative-only `parent_weights` when cwd varies; PPO training losses only in TensorBoard

**Promotion gates**:
Non-negotiable checks before **gated promotion**: **replay eval metrics** at or above **replay accuracy floor**, plus **sim eval metrics** non-regression vs `latest` on the frozen seed list in **eval config artifact** until PPO moves off **Python training sim**.
_Avoid_: "eval passed" without naming both bars; treating Python sim as co-equal with web rules truth; a second Node replay at eval time

**Replay eval metrics**:
Masked action accuracy on val-split **human step** **derived training rows**: candidate and `latest` forward on stored `obs`/`mask`, compared to stored **policy action index**; labels are **web-authoritative labels** from dataset build—no second Node replay at metrics time. **Disagreement rate** (report only) is the fraction of those rows where masked argmax(candidate) ≠ masked argmax(`latest`) on the same `obs`/`mask`. During **BC policy training**, per-epoch val scoring uses the same masked-accuracy rule as `replay.val_masked_accuracy` (candidate vs label only); the post-train **replay metrics** pass on **BC best checkpoint** is authoritative for **metrics artifact** and **replay accuracy floor**.
_Avoid_: Re-stepping val envelopes through Node for promotion; using "replay" to mean live engine re-run at eval time; treating disagreement as label error rate; a different metric for early-stop vs floor

**Training parent**:
BC and PPO both start from `models/latest/policy.weights.h5`; each run still records that resolved path as `parent_weights` for audit, even though v1 does not expose alternate parents. v1: **training data root** (`--data-dir`) is only for replays/derived/eval artifacts; weight I/O stays under repo-root `models/` (no `--models-dir` flag).
_Avoid_: "checkpoint" without saying latest vs semver; parent or run weights under **training data root**

**Frozen eval suite**:
Fixed held-out match ids (~20% of ingested matches, chosen once) for **replay eval metrics** on **derived training rows**, plus a fixed seed list for **legacy Python sim benchmarks** vs `latest` and RandomBot. No separate disagreement slice or self-play matrix in v1.
_Avoid_: "validation set" without specifying match-level holdout vs step-level; implying metrics re-invoke the **web game engine**

**Eval suite artifact**:
Frozen file at `data/replays/eval_suite.json` (alongside ingest manifest), produced once by `eval_suite init`. Lists `val_match_ids`, suite version, and sampling seed; val ids chosen via seeded random ~20% (minimum one val match when at least two matches exist). **Frozen on disk** after first init under gitignored **training data root** (not checked into git); backup with the replay tree; stable across pipeline runs on the same machine until suite version is bumped and init is re-run.
_Avoid_: Re-deriving the val split each time the dataset is built; treating “committed” as git-tracked; gitignore exceptions for eval files under `data/`

**Eval config artifact**:
File at `data/replays/eval_config.json` under **training data root**: **replay accuracy floor**, fixed **legacy Python sim benchmarks** seed list, and **sim regression tolerance** (default `0.01` at **eval config init**). Separate from **eval suite artifact** so holdout re-init does not overwrite thresholds. **Frozen on disk** like **eval suite artifact** (local-only under `data/`). **Floor recorder** updates use atomic replace (`eval_config.json.tmp` then rename).
_Avoid_: Storing the floor or sim seeds only inside **eval suite artifact** or per-run `metrics.json`; in-place partial writes of **eval config artifact**; checking eval config into the repo while raw replays stay gitignored

**Sim regression tolerance**:
ε in **eval config artifact** for the sim leg of **promotion gates**: candidate win rate may be up to ε below `latest` on the frozen seed list (default `0.01`).
_Avoid_: Strict zero tolerance on small seed lists; conflating with replay accuracy floor

**Sim eval metrics**:
Win rate on **legacy Python sim benchmarks**: candidate and **training parent** (`latest`) each run the full frozen seed list independently vs RandomBot on **Python training sim**; **promotion gates** sim leg passes when candidate win rate ≥ latest win rate minus **sim regression tolerance**.
_Avoid_: Paired per-seed sign tests in v1; strict `>` with zero tolerance on small seed lists

**Replay accuracy floor**:
Minimum val **replay eval metrics** masked accuracy for **promotion gates**; set once to the first BC baseline run’s `replay.val_masked_accuracy` exactly (stored in **eval config artifact**), not overwritten by later runs unless a maintainer resets it.
_Avoid_: A fixed threshold before any baseline exists; a margin below baseline accuracy; recomputing the floor every train run

**BC policy training**:
Fine-tuning **training parent** weights with masked softmax on **policy action index** on train-split **human step** rows only; shared trunk and policy head receive gradients; value head frozen with no loss in v1. Early-stops on patience against val-split **human step** masked accuracy; **BC best checkpoint** is restored before **training run artifact** write. Stage `bc` then runs **replay eval metrics**, **sim eval metrics**, atomic **training run artifact** write, **floor recorder** when floor was null, and **gate evaluator preview** by default (`--no-gate-preview` to skip); **gated promotion** copy stays in #8. v1 uses fixed hyperparameters in code (no BC CLI tuning flags). Train rows are globally shuffled with a fixed seed in code before batching. **BC start prerequisites** fail fast: **derived store** with ≥1 train and ≥1 val **human step** row, **eval suite artifact** + **eval config artifact**, **training parent** weights, and on-disk val match ids ⊆ suite `val_match_ids`. Every run writes TensorBoard scalars under `tb/` inside the staged run dir (`models/runs/<run_id>.tmp/tb/` during training, `models/runs/<run_id>/tb/` after commit) for live monitoring (per-epoch `train/bc_loss`, `val/masked_accuracy`; optional `lr` if scheduled).
_Avoid_: PPO-style value targets on human rows; training the value head with zero or dummy targets; saving last-epoch weights after early stopping; promoting from `bc`; skipping floor recorder on baseline runs; per-run LR/epoch flags in v1; starting BC with empty val or missing eval artifacts; BC without per-run TensorBoard logs; PPO rollout tags on BC runs

**BC best checkpoint**:
Candidate weights at the training epoch with highest val-split **human step** masked accuracy before early stopping; **training run artifact** and post-train **replay eval metrics** / **sim eval metrics** use this checkpoint, not the final epoch.
_Avoid_: Gating or **replay accuracy floor** from last-epoch or non-restored weights

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

**Gate evaluator preview**:
Pass/fail **promotion gates** printed after stage `bc` (metrics + **floor recorder**); does not copy weights to `models/<version>/`. Full **gated promotion** is a separate #8 stage.
_Avoid_: Treating preview as promote; running preview before **replay accuracy floor** is written on **BC baseline run**

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
A maintainer-run script (or script chain) that performs ingest → verify → dataset → train → gate; scheduling/cron is out of scope, not live data access. Implemented as staged CLIs plus a thin `run-all` orchestrator that calls them in order. Default **run-all** ends after **BC policy training**; `--with-ppo` chains **BC-anchored PPO policy training** with `--bc-run` set to that `bc-*` artifact; `--with-publish` chains **Publish CLI** on the artifact just written (`ppo-*` when both flags are set, else `bc-*`).
_Avoid_: "manual ingest" meaning export files only; a monolith with no stage boundaries; **run-all** always implying PPO or **gated promotion**; auto-publish without an explicit flag

**Replay pipeline CLI**:
Shared package entry point with subcommands (`ingest`, `verify`, `dataset`, `bc`, `ppo`, `publish`, …) invoked as `python -m dungeon_runner.replay.cli <stage>`; **run-all** orchestrator calls the same stages in order. **BC policy training** is stage `bc`; **BC-anchored PPO policy training** is stage `ppo`; **gated promotion** is stage `publish` — uses `--run` plus `--data-dir` for **eval config artifact**.
_Avoid_: A separate top-level script per pipeline stage with no shared module; a standalone BC script with a different data-root flag name

**Replay pipeline documentation**:
Maintainer runbook at `docs/replay-pipeline.md`: full staged CLI map (implemented + stub stages), per-stage flags, **`run-all`** (`--with-ppo`, `--with-publish`), env vars (`FIREBASE_DATABASE_URL`, `PORTFOLIO_SITE_ROOT`), **training data root** and `models/` layout, manifest JSON examples, full **ingest manifest** skip-reason table, ingest/verify deep sections, and cross-links to portfolio-site **Replay envelope contract (v1)**, [portfolio-site #128](https://github.com/enmaku/portfolio-site/issues/128), portfolio-site `CONTEXT.md`, and portfolio-site ADRs. Cross-repo links use **web engine root** / sibling checkout paths as primary and matching `github.com/enmaku/portfolio-site` URLs as fallback. Root `README.md` links here in a short **Replay training pipeline** section (plus this glossary). **Gated promotion** vs portfolio-site **TF.js model sync** (#11) is a release pointer only, not a full two-repo checklist. Domain terms live in this `CONTEXT.md` glossary only.
_Avoid_: Putting directory trees or flag lists in `CONTEXT.md`; duplicating CONTRACT in dungeon-runner; GitHub-only links that ignore `PORTFOLIO_SITE_ROOT`; skip codes only as a bullet list without triggers; duplicating full downstream issue PRDs in stub stage prose

**BC-anchored PPO policy training**:
Optional pipeline stage `ppo` that fine-tunes on **Python training sim** rollouts with a **BC anchor** and **rollout opponent roster** including **BC-bot**; writes the same **training run artifact** bundle as **BC policy training** (weights, **metrics artifact**, TensorBoard `tb/`, post-train eval + **gate evaluator preview** on by default, `--no-gate-preview` to skip). v1 deliverable is the **Replay pipeline CLI** stage, not hand-run `scripts/train*.py` alone (scripts may remain for experiments). Unlike **BC policy training**, PPO trains the value head from rollout returns; policy head + trunk receive PPO and **BC anchor** gradients. PPO rollout/update hyperparameters are fixed in code for v1 (like **BC policy training**); CLI tuning is limited to `--bc-run`, anchor strengths, `--ray-workers`, and `--no-ray`. **Gated promotion** remains #8 only.
_Avoid_: PPO-only script runs as the canonical pipeline step; skipping **metrics artifact** / frozen eval on PPO runs; freezing the value head through PPO as in BC; per-run PPO `--updates` / `--rollout` flags in v1

**PPO BC run**:
Required CLI path to a `bc-*` **training run artifact**. Stage `ppo` loads init weights from its `policy.weights.h5`, uses the same directory as **BC-only candidate** for **PPO BC regression check**, and records that path in **metrics artifact** `parent_weights` (the BC checkpoint at PPO start, not `latest` unless they coincide).
_Avoid_: Separate `--init-run` and `--bc-candidate` in v1; **ppo** without an explicit BC artifact path

**BC-only candidate**:
The **training run artifact** from **BC policy training** used as the baseline for **PPO BC regression check** — in v1 always the **PPO BC run** passed on the CLI.
_Avoid_: Comparing PPO only to promoted `latest`; “BC-only” meaning human-only dataset rows

**PPO BC regression check**:
After post-train **replay eval metrics** and **sim eval metrics**, stage `ppo` compares candidate to **BC-only candidate**. Replay leg: candidate **replay eval metrics** must be ≥ BC (strict). Sim leg: candidate win rate must be ≥ BC win rate minus **sim regression tolerance** ε from **eval config artifact** (same ε as **promotion gates** sim leg). Failure exits non-zero but still commits **training run artifact** with `ppo_bc_regression.pass: false` in **metrics artifact**. **Publish CLI** refuses `ppo-*` runs unless that flag is true. Separate from **gate evaluator preview** vs `latest` and **promotion gates**.
_Avoid_: Treating BC regression as warn-only; discarding artifact on BC regression failure; applying ε to replay accuracy; publishing `ppo-*` when BC regression failed

**PPO start prerequisites**:
Fail-fast checks before **BC-anchored PPO policy training** begins. Always: **PPO BC run** path, **eval suite artifact**, and **eval config artifact**. **Derived store** with train-split **human step** rows (and val-id sanity vs the suite) required only when **BC anchor CE** is active (`λ > 0`).
_Avoid_: Requiring a full derived corpus when `λ = 0`; **ppo** without eval artifacts or **PPO BC run**

**BC anchor**:
Extra imitation pressure during **BC-anchored PPO policy training** beyond rollout PPO loss. v1 implements both legs: **BC anchor CE** (masked cross-entropy on train-split **human step** rows from **derived store**, strength `λ`, default `0.1`) and **BC anchor KL** (divergence vs **frozen BC teacher**, strength `β`, default `0`). Setting `λ = 0` and `β = 0` disables anchoring. Stage `ppo` exposes `--bc-anchor-lambda` and `--bc-anchor-beta` (defaults `0.1` / `0`).
_Avoid_: “BC anchor” meaning only rollout opponents; conflating anchor with **BC-bot** seat assignment

**BC anchor CE**:
The human-example leg of **BC anchor**: periodic batches from **derived store** train-split **human step** rows, same masked label rule as **BC policy training**.
_Avoid_: Re-deriving labels from Python `actions_codec`; applying CE to val-split rows during PPO training

**BC anchor KL**:
The stay-close-to-BC leg of **BC anchor**: compares the training policy to **frozen BC teacher** action distributions; active only when `β > 0`.
_Avoid_: Updating the teacher during PPO; KL without a frozen snapshot

**Frozen BC teacher**:
A non-trainable copy of `policy.weights.h5` snapshotted at **BC-anchored PPO policy training** start from **PPO BC run**. Used for **BC anchor KL** and for **BC-bot** action selection in rollouts.
_Avoid_: “Frozen BC” meaning promoted `latest`; teacher weights drifting with the learner

**Rollout opponent roster**:
The opponent types used in **BC-anchored PPO policy training** on **Python training sim**: **RandomBot**, **BC-bot** (**frozen BC teacher**), and self-play (other seats use the learner). v1 samples one **rollout match template** per new match (not long contiguous blocks of a single template).
_Avoid_: “Roster” meaning eval opponents only; BC-bot on **web game engine** rollouts

**Rollout match template**:
A fixed lineup rule for one sim match. v1 templates (fixed probabilities in code, default **20% / 45% / 35%**): (1) one **learner** seat vs **RandomBot** opponents, (2) one **learner** vs **BC-bot** opponents, (3) full self-play (all seats **learner**). Each new match draws a template independently so TensorBoard rollout stats stay interleaved. Only **learner** seats contribute transitions to the PPO buffer except template (3), where all seats do. Training mix is separate from **sim eval metrics** (still vs RandomBot on frozen seeds).
_Avoid_: Long runs of a single template before switching; per-seat independent opponent sampling in v1; assuming rollout RandomBot share must match sim eval opponent

**BC-bot**:
An opponent seat in **Python training sim** that acts via **frozen BC teacher** forward passes (Keras **PolicyValueModel** + Python `actions_codec` for legal actions). Not the training policy and not **RandomBot**.
_Avoid_: BC-bot on replay-derived rows; BC-bot meaning the human-labeled **derived store**

**PPO rollout collection**:
**BC-anchored PPO policy training** gathers **Python training sim** games via Ray-parallel workers by default (`--ray-workers`, default **8** on maintainer dev hardware; counts below **1** clamp to one worker). CLI `--no-ray` forces single-process collection for debugging or when Ray fails to start or misbehaves (e.g. macOS). Ray init/collect failures raise **`RayRolloutError`** with a `--no-ray` fallback hint; the pool always tears down actors in `shutdown()` after training. Shared rollout logic backs both modes; `scripts/train_rllib.py` reuses the same local-cluster init and per-worker step split as the pipeline pool. TensorBoard under `tb/` logs PPO and **BC anchor** losses each update, plus per-**rollout match template** rollout stats (tagged by template so interleaved episodes stay readable).
_Avoid_: Two unrelated rollout implementations for pipeline vs scripts; unbounded default worker count on laptops; training losses only in **metrics artifact** or only in TensorBoard

**Live replay ingest**:
Ingest reads new matches directly from Firebase RTDB `dungeonRunnerCompletedMatches` (incremental by top-level match key), not only from dropped export JSON files.
_Avoid_: Conflating "no cron" with "no RTDB pull"

**Firebase database URL**:
The only required Firebase env var for **live replay ingest** v1: `FIREBASE_DATABASE_URL` in gitignored `.env` at repo root (loaded automatically on ingest; same value as portfolio-site `VITE_FIREBASE_DATABASE_URL`). `.env.example` documents this var and **web engine root** together with stage comments (ingest vs verify/dataset). API key and other web client vars are not loaded by ingest until authenticated RTDB access is needed.
_Avoid_: "Firebase config" when only the database URL is needed; Phase A `.env.example` omitting **web engine root** when the two-repo setup is documented upfront

**Web engine root**:
Filesystem path to the portfolio-site repo root, supplied as `PORTFOLIO_SITE_ROOT` in gitignored `.env` (loaded on verify/dataset stages). **Replay verifier** loads existing portfolio-site modules (`debug/replaySession.js`, `nn/policyAdapter.js`, …) from that path; verify fails fast if unset.
_Avoid_: Vendoring the kernel into dungeon-runner; submodule as the v1 default; large new portfolio-site APIs for training

**Pipeline consumer**:
dungeon-runner treats portfolio-site as producer of **completed match replay** envelopes and rules truth; v1 work consumes those outputs and orchestrates Node against existing web modules, with portfolio-site code changes kept minimal (e.g. exporting `encodeActionIndex` from `policyAdapter.js` when the harness needs it). **Replay envelope contract (v1)** field rules live in portfolio-site `CONTRACT.md`; ingest extensions live in dungeon-runner pipeline docs.
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
Gitignored directory (default `data/replays/`, overridable via pipeline `--data-dir`) holding `raw/{matchId}.json` envelopes, `manifest.json` (**ingest manifest**), `verify_manifest.json` (**verify manifest**), **eval suite artifact**, **eval config artifact**, and per-match **derived match artifact** files under `derived/`. Repo `.gitignore` covers all of `data/` (not only `data/replays/`), so nothing under that tree is committed.
_Avoid_: Storing replays under `models/` or the repo root; committing raw replays or manifests; gitignoring only `data/replays/` while leaving sibling `data/*` tracked by accident

**Raw envelope store**:
Per-match file `raw/{matchId}.json` under **training data root** containing the eligible envelope body only (match id in the path, not required inside JSON). Ingest writes the payload **verbatim** after **ingest eligibility** passes—including unknown top-level keys, optional nested `matchId` on the body, and full history entry objects (NN `__debug`, etc.). Canonical match id is always the RTDB key / filename, never body `matchId`.
_Avoid_: Stripping unknown envelope keys at ingest; rejecting ingest solely because body `matchId` exists; using body `matchId` when it disagrees with the path key

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
`manifest.json` under **training data root** with `ingested` (match id strings written to `raw/`) and `skipped` (`id` + machine-readable `reason`, e.g. `unsupported_version`). Skip reasons are granular for ingest audit (version, seed, setup, history, presentation speed, etc.); portfolio-site `importReplayEnvelope` may keep coarser web error codes—eligibility behavior must match, reason strings need not. Dedup is by match id only—any id in either list is not re-fetched in v1.
_Avoid_: "processed ids file" without skip reasons; requiring web and ingest to share identical reason code strings

**Manual re-ingest**:
Pulling a match id again after it appears in **ingest manifest**; maintainer removes that id from the manifest and deletes its raw envelope first. RTDB payload edits under the same key are not detected automatically in v1.
_Avoid_: Expecting ingest to refresh changed RTDB rows for an existing key

**Ingest run atomicity**:
A single ingest run either completes all new pulls and eligibility writes and then updates **ingest manifest**, or aborts with no manifest changes (RTDB list failure, per-match fetch failure, or write error).
_Avoid_: Leaving a half-updated manifest after a failed pull

**Ingest eligibility**:
A match passes ingest only when it would pass portfolio-site `importReplayEnvelope` (version, seed, setup, history shape including per-entry RNG step rules; optional `presentationSpeedProfile` enum). Empty `history` arrays are eligible if import would pass; **replay verifier** rejects non-terminal replays. Enforced at ingest without **web engine root**; dungeon-runner keeps behavior aligned via acceptance tests keyed to portfolio-site replay import cases. Skipped matches are recorded in **ingest manifest** with a granular reason code. Shape/RNG-chain only—**replay verifier** owns full replay to `match-over`.
_Avoid_: "valid replay" at ingest; expecting ingest to prove `match-over`; rejecting `history: []` at ingest when web import accepts it; Node or **web game engine** at ingest; eligibility logic drifting from web import without shared test coverage

**Structural envelope check**:
Same shape and static RNG-chain rules as **ingest eligibility**; performed at ingest only. **Replay verifier** does not repeat it for **pending verify** matches.
_Avoid_: Duplicating structural checks at verify; using structural pass as a substitute for **verified replay**; treating this as a separate rule set from **ingest eligibility**

**Epic v1 success bar**:
At least one **gated promotion** plus a documented manual two-repo release path; **promotion gates** are guardrails inside that bar. PPO remains in the pipeline but is not required for a given promotion. Ingest is **live replay ingest** inside a **manual pipeline run**.
_Avoid_: "pipeline complete" without a promoted model

## Relationships

- A **completed match replay** is ingested raw, then verified, then yields many **derived training rows**
- **Gated promotion** in dungeon-runner precedes **TF.js model sync** (separate portfolio-site step)
- Each **TF.js model sync** after promote writes semver TF.js and, when the id matches **production latest**, overwrites **web deployed latest**
- **Two-repo model release** follows **gated promotion** and is documented for **Epic v1 success bar**
- **Portfolio-site coordination issue** ([#128](https://github.com/enmaku/portfolio-site/issues/128)) owns portfolio-side deliverables; **TF.js model sync** implementation in [#127](https://github.com/enmaku/portfolio-site/issues/127); dungeon-runner #11 tracks epic closure
- **Web-authoritative labels** and **replay verifier** use the **web game engine** only; **Python training sim** is legacy PPO/self-play, not a parity target
- Only **human step** entries get `is_human` on **derived training rows** (BC trains on those); **Non-NN history step** without human seat (e.g. randombot) still produces rows but not human-labeled ones unless a future envelope adds `humanSeatIds`
- **Promotion gates** replay leg uses **replay eval metrics** on **frozen eval suite** val ids from **derived store** (no second Node replay at eval)
- **BC-anchored PPO policy training** requires **PPO BC run**; optional in **manual pipeline run** via `--with-ppo`
- **Training data root** holds raw envelopes, **ingest manifest**, and **verify manifest** before dataset build
- **Replay verifier** and dataset stages require **web engine root**
- **Eval suite init** draws match ids only from **verify manifest** `verified`
- **Eval suite artifact** must exist before dataset build; dataset builder fails fast if missing
- **Replay accuracy floor** lives in **eval config artifact**; first BC baseline sets it once
- **Eval config init** precedes metrics/gate runs; **eval config artifact** holds seeds and ε before the floor is set
- BC/PPO write **training run artifact**; **gated promotion** consumes **metrics artifact** + weights from that directory and updates **production latest** symlink
- **Promoted version** semver dirs are separate from **training run id** under `models/runs/`
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
- Issue #6 BC loss scope — resolved: **BC policy training** updates trunk + policy head only; value head frozen in v1.
- Issue #6 checkpoint selection — resolved: **BC best checkpoint** (best val human masked accuracy), not last epoch.
- Issue #6 early-stop vs replay metric — resolved: same masked-accuracy definition; post-train **replay eval metrics** on **BC best checkpoint** sets floor and **metrics artifact**.
- Issue #6 CLI entry — resolved: **BC policy training** via **Replay pipeline CLI** subcommand `bc` (`python -m dungeon_runner.replay.cli bc`).
- Issue #6 **training run id** — resolved: default `bc-YYYYMMDDTHHMMSSZ`; optional `--run-id` override.
- Issue #6 post-train chain — resolved: `bc` writes artifact, **floor recorder**, default **gate evaluator preview**; promote in #8 only.
- Issue #6 artifact atomicity — resolved: `.tmp` staging dir + rename; `metrics.json` last in staging.
- Issue #6 BC hyperparameters — resolved: fixed constants in code for v1; no BC CLI tuning flags.
- Issue #6 BC prerequisites — resolved: **BC start prerequisites** fail fast (train+val human rows, eval artifacts, parent weights, val id sanity).
- Issue #6 `train.bc_loss` — resolved: full train-split human CE on **BC best checkpoint** weights in **metrics artifact**.
- Issue #6 train row ordering — resolved: global shuffle with fixed seed in code before batching.
- Issue #6 weight paths — resolved: `--data-dir` for replay/derived/eval only; repo-root `models/` for parent and **training run artifact** in v1.
- Issue #6 `parent_weights` field — resolved: absolute resolved path in **metrics artifact**.
- Issue #6 TensorBoard — resolved: always per-run `tb/` (staged under `.tmp`, committed with artifact); per-epoch `train/bc_loss` and `val/masked_accuracy`.
- Issue #7 PPO CLI surface — resolved: **BC-anchored PPO policy training** is **Replay pipeline CLI** stage `ppo` with **training run artifact** parity to `bc`; `scripts/train*.py` are non-canonical dev entrypoints.
- Issue #7 PPO start weights — resolved: required **`--bc-run`** sets init weights and **BC-only candidate**; **parent_weights** records that BC path.
- Issue #7 BC anchor — resolved: **BC anchor CE** (`λ`, default on) + **BC anchor KL** (`β`, default off) vs **frozen BC teacher**; both knobs configurable.
- Issue #7 rollout opponents — resolved: **rollout match template** per new match (learner vs RandomBot, learner vs **BC-bot**, self-play); fixed template probabilities in code; per-template TensorBoard tags later; no contiguous template blocks.
- Issue #7 value head — resolved: PPO trains value head from sim returns; BC-style value freeze applies to **BC policy training** only.
- Issue #7 rollout collection — resolved: Ray-parallel `--ray-workers` default **8**; `--no-ray` single-process fallback on `replay.cli ppo`.
- Issue #7 PPO BC regression — resolved: **PPO BC regression check** fails stage (exit ≠ 0) on replay or sim regression vs **BC-only candidate**; artifact still written.
- Issue #7 PPO prerequisites (dataset) — resolved: **derived store** / BC-style row checks only when `λ > 0`; init weights always.
- Issue #7 PPO prerequisites (BC artifact) — resolved: **PPO BC run** (`--bc-run`) always required before **ppo** starts.
- Issue #7 PPO prerequisites (eval) — resolved: **eval suite artifact** and **eval config artifact** always required (same as **BC policy training** post-train eval).
- Issue #7 run-all — resolved: **run-all** stops after `bc` by default; `--with-ppo` runs `ppo` against that BC artifact.
- Issue #8 run-all publish — resolved: **gated promotion** never by default; `--with-publish` opt-in chains `publish --run` on the last train artifact.
- Issue #7 default λ — resolved: **BC anchor CE** default `λ = 0.1`; **BC anchor KL** default `β = 0`.
- Issue #7 template mix — resolved: default **20% / 45% / 35%** (RandomBot / BC-bot / self-play); RandomBot minority seasoning only.
- Issue #7 anchor CLI — resolved: `--bc-anchor-lambda` and `--bc-anchor-beta` on `replay.cli ppo`.
- Issue #7 training run id — resolved: default `ppo-` + UTC timestamp; optional `--run-id` (same pattern as `bc-`).
- Issue #7 gate preview — resolved: **gate evaluator preview** on by default for `ppo`; `--no-gate-preview` to skip.
- Issue #7 PPO BC regression strictness — resolved: replay strict vs **BC-only candidate**; sim allows ε from **eval config artifact**.
- Issue #7 PPO hyperparameters — resolved: fixed in code for v1; CLI limited to `--bc-run`, anchor flags, Ray/`--no-ray`.
- Issue #7 PPO metrics — resolved: `train.ppo_loss`, `train.bc_anchor_ce`, `train.bc_anchor_kl` in **metrics artifact** and matching TensorBoard scalars; per-template rollout tags in TensorBoard.
- Issue #8 promoted dir name — resolved: **promoted version** is semver under `models/<version>/`, not **training run id**.
- Issue #8 **production latest** — resolved: `models/latest/` is a symlink to the current **promoted version** dir, not a duplicate weights tree.
- Issue #8 semver line — resolved: first replay-pipeline promote is `v0.2`; then `v0.2.01`, `v0.2.02`, … (two-digit patch); legacy `v0.1.*a` epoch dirs are not the allocator template.
- Issue #8 **promotion manifest** — resolved: `promotion.json` per **promoted version** dir + append `models/promotions.jsonl`; metrics snapshot copied into promoted dir.
- Issue #8 **publish gate evaluation** — resolved: trust committed **metrics artifact** + **eval config artifact**; no re-eval at publish.
- Issue #8 **Publish CLI** — resolved: `replay.cli publish --run <dir> --data-dir …`; reject `*.tmp`.
- Issue #8 re-publish — resolved: same **training run id** cannot promote twice; JSONL is the source of truth.
- Issue #8 **publish run atomicity** — resolved: stage `.<version>.tmp/` → rename; symlink + JSONL last.
- Issue #8 minor semver bump — resolved: manual `publish --version` only; patch auto-bump under current minor line.
- Issue #8 PPO at publish — resolved: `publish` hard-fails `ppo-*` unless **metrics artifact** records `ppo_bc_regression.pass: true`.
- Issue #8 legacy `latest/` — resolved: #8 migrates duplicate `latest/` → symlink to `v0.1.30a`.
- Issue #8 semver + symlink — recorded in [ADR 0002](docs/adr/0002-promoted-version-semver-and-latest-symlink.md).
- Issue #9 normative v1 spec location — resolved: portfolio-site `CONTRACT.md` authoritative for envelope fields/eligibility; dungeon-runner pipeline doc for ingest-only extensions (cross-link, no duplicate full spec).
- Issue #9 ingest skip reason codes — resolved: granular codes in **ingest manifest** only; web `importReplayEnvelope` stays coarse; eligibility behavior must mirror web, reason strings need not match.
- Issue #9 ingest enforcement mechanism — resolved: dungeon-runner-local eligibility (no **web engine root** on ingest); parity guarded by acceptance tests aligned with portfolio-site replay import tests.
- Issue #9 unknown envelope fields — resolved: **raw envelope store** preserves full eligible payload verbatim; unknown top-level keys are not stripped at ingest.
- Issue #9 human step vs `modelId` — resolved: **Non-NN history step** = no `action.modelId`; **Human step** / `is_human` = resolved human seat at **dataset build**, not envelope-only inference.
- Epic #1 human labeling — resolved (2026-05-19): epic PRD uses **`is_human`** / **human step** (dataset seat resolution) everywhere; dropped `action.modelId` absent as the human definition.
- Epic #1 row grain vs BC — resolved (2026-05-19): epic user story 4 — all history **action** steps → **derived training rows**; BC + replay gates use `is_human=true` only.
- Epic #1 eval bootstrap order — resolved (2026-05-19): epic user story 7 — **`eval_suite init`** + **`eval_config init`** after verify, before first **`dataset` build** (≥2 verified ids).
- Epic #1 promotion pre-floor — resolved (2026-05-19): epic user story 13 — **gated promotion** blocked until first **BC baseline run** writes **replay accuracy floor**; train artifacts + preview still allowed.
- Epic #1 PPO publish — resolved (2026-05-19): epic user story 14 — `publish` on `ppo-*` requires **PPO BC regression check** pass plus **promotion gates** vs `latest`.
- Epic #1 artifact layout — resolved (2026-05-19): epic user stories 12/15 — candidates in `models/runs/`; **gated promotion** copies to `models/<promoted version>/` + **production latest** symlink.
- Epic #1 manifests — resolved (2026-05-19): epic user stories 22/26 — **ingest manifest** vs **verify manifest**; no dataset manifest in v1.
- Epic #1 post-freeze ingest — resolved (2026-05-19): epic **Frozen eval suite** row — new ingests after init → train only until suite version bump + re-init.
- Epic #1 gate preview vs publish — resolved (2026-05-19): epic user story 24 + **Manual pipeline run** row — **gate evaluator preview** after train; **`publish`** separate unless `--with-publish`.
- Issue #9 nested body `matchId` — resolved: allowed in **raw envelope store** if otherwise eligible; canonical id is path/RTDB key only.
- Issue #9 `version` type strictness — resolved: ingest requires integer `1`; document any gap vs web `!== 1` in pipeline doc; no portfolio-site code change in #9 scope.
- Issue #9 v2 documentation — resolved: v2 field roadmap in portfolio-site CONTRACT only; dungeon-runner pipeline doc links + states skip-until-supported ingest behavior.
- Issue #9 empty `history` — resolved: eligible at ingest when web import accepts; failure at **replay verifier**, not ingest skip.
- Issue #9 portfolio-site CONTRACT layout — resolved: dedicated `## Replay envelope contract (v1)` section (+ v2 appendix), not only a Debug bullet.
- Issue #6 floor write — resolved: **floor recorder** uses atomic replace for **eval config artifact**.
- Issue #10 pipeline doc location — resolved: single **Replay pipeline documentation** at `docs/replay-pipeline.md`; `CONTEXT.md` glossary-only for domain language.
- Issue #10 training data gitignore — resolved: ignore entire `data/`; default **training data root** path remains `data/replays/`.
- Issue #10 `.env.example` — resolved: `FIREBASE_DATABASE_URL` + `PORTFOLIO_SITE_ROOT` with stage comments; `.env` gitignored at repo root.
- Issue #10 ingest skip-reason table — resolved: full table in **Replay pipeline documentation** (code, trigger, web coarse mapping); not in `CONTEXT.md`.
- Issue #10 portfolio-site cross-links — resolved: sibling/`PORTFOLIO_SITE_ROOT` path primary, GitHub URL fallback (CONTRACT, CONTEXT, ADRs).
- Issue #10 README discoverability — resolved: short **Replay training pipeline** section in root `README.md` linking **Replay pipeline documentation** and this glossary.
- Issue #10 Phase A doc scope — resolved: full stage map stub + ingest detail only; #11 called out in table, not full release checklist.
- Issue #10 eval artifacts vs git — resolved: **eval suite artifact** and **eval config artifact** are **frozen on disk** under gitignored `data/`, not git-committed.
- Issue #10 version strictness doc — resolved: **Intentional strictness** subsection under ingest in **Replay pipeline documentation**.
- Issue #4 Node handoff — resolved: one Node process per match; JSON row array on stdout; Python owns Parquet + `meta.json`.
- Issue #4 `meta.json` — resolved: `match_id`, **dataset encoding version**, `row_count`, `built_at` (no raw content-hash in v1).
- portfolio-site `database.rules.json` is fully open today; **RTDB ingest access** may need Admin SDK or auth when rules tighten.
- Issue #11 semver vs portfolio `latest` — resolved: every promote syncs **promoted version** semver dir and refreshes **web deployed latest**; default `modelId: 'latest'` unchanged.
- Issue #11 sync UX — resolved: single `sync-dungeon-runner-model.mjs <promoted version>` performs semver + **web deployed latest** + catalog (no separate latest step).
- Issue #11 H5 source — resolved: **TF.js model sync** prefers local `DUNGEON_RUNNER_ROOT` sibling checkout, falls back to git clone.
- Issue #11 version argument — resolved: checklist uses explicit **promoted version**; optional `--from-latest` for convenience after a fresh promote.
- Issue #11 `sync --all` — resolved: **TF.js sync backfill** converts all semver dirs; **web deployed latest** updated once from **production latest**, not per directory in loop order.
- Issue #11 old semver single-id sync — resolved: writes semver TF.js only unless id matches **production latest** (then also refreshes **web deployed latest**).
- Issue #11 promote validation — resolved: explicit `sync <id>` fails without **promotion manifest** ledger entry; `--from-latest` and **TF.js sync backfill** exempt.
- Issue #11 checklist location — resolved: **two-repo model release** split per **Two-repo model release** (dungeon-runner publish + link; portfolio-site sync/smoke/catalog).
- Issue #11 repo split — resolved: **portfolio-site release issue** + feature branch for portfolio code/docs/tests; dungeon-runner #11 for glossary, pipeline publish section, cross-link (no Python sync implementation in dungeon-runner).
- Issue #11 portfolio timing — resolved: **portfolio-site release issue** can proceed in parallel with #8; E2E promote → sync after first real **gated promotion**.
- Issue #11 cross-repo links — resolved: [portfolio-site #128](https://github.com/enmaku/portfolio-site/issues/128) umbrella + [#127](https://github.com/enmaku/portfolio-site/issues/127) (sync) ↔ dungeon-runner #1/#8/#9/#11; **Replay pipeline documentation** release section links portfolio coordination when #11 doc slice ships.
- Issue #11 acceptance edits — resolved: dungeon-runner #11 PRD + [portfolio-site #127](https://github.com/enmaku/portfolio-site/issues/127) (2026-05-19).
- Issue #11 smoke scope — resolved: **release smoke** = manual browser play + optional Node artifact guard; no required headless WebGL script in v1.
- Issue #11 semver immutability — resolved: **deployed model version** dirs append-only on promote; same-id overwrite only for backfill/converter fix; **web deployed latest** is the moving alias.
