# Ubiquitous Language — Dungeon Runner (training repo)

Canonical vocabulary for the dungeon-runner repo and the shared **Dungeon Runner product chain** with [portfolio-site](https://github.com/enmaku/portfolio-site). Training-specific detail also lives in [`CONTEXT.md`](./CONTEXT.md). Sibling copy: [`portfolio-site/UBIQUITOUS_LANGUAGE.md`](https://github.com/enmaku/portfolio-site/blob/main/UBIQUITOUS_LANGUAGE.md).

## Match play (shared with portfolio-site)

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Match** | One full play from **setup** through **match over** | Game (ambiguous), session |
| **Match over** | **Web game engine** terminal `match-over` with a recorded winner; authoritative for play and replays | Sim forfeit, finished game |
| **Sim empty-pile forfeit** | **Python training sim** end with no winner when bidding ends on an empty pile; not **match over** | Match over, runner loss |
| **Dungeon run** | One runner’s lane attempt within a **match** | Run (ambiguous with training run) |
| **Setup** | Seat count and **opponent** types before a **match** starts | Config, lobby |
| **Opponent** | Non-human seat (`nn` or **Randombot** in setup) | Bot (vague), role badge |
| **Randombot** | **Opponent** with setup role `randombot`; non-NN actions without `modelId` | Random bot, **RandomBot** (code only) |
| **Neural opponent** | **Opponent** with setup role `nn` and `modelId` at runtime | AI (vague) |
| **Human player seat** | Sole `human` role after seat shuffle; its **actor seat id** anchors labeling | `seat-1`, human seat (vague) |
| **Seed** | Integer RNG root; same **setup** + **seed** ⇒ same outcomes | Random seed (casual) |
| **History** | Ordered canonical actions + per-step RNG metadata | Game log, action log, save file |

## Replay & archive

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Replay envelope** | Versioned serialized **match**: `version`, `seed`, `setup`, `history`, optional pace | Replay JSON (without version) |
| **Replay envelope version** | Integer schema id on the envelope (`1` in v1) | Schema version (vague) |
| **Replay envelope contract (v1)** | Normative field rules in portfolio-site `CONTRACT.md` | Duplicating full spec in this repo |
| **Presentation pace** | Optional envelope `presentationSpeedProfile` (`cinematic` \| `brisk`); playback hint only | Training label, rules field |
| **Completed match replay** | **Replay envelope** for a **match** that reached **match over** | Telemetry, training upload |
| **Completed match replay archive** | RTDB tree `dungeonRunnerCompletedMatches/{matchId}` | Firebase bucket (vague) |
| **Archive listing** | Shallow read of archive keys for incremental ingest | Service-account listing (v1 uses open read) |

## History steps & training labels

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Actor seat id** | History field `actorSeatId` (`seat-1`…`seat-4`) for who acted | Seat index, “seat 1 = human” |
| **Policy action index** | Integer 0–25 for the action taken in the policy layout | Action id; re-derive in Python |
| **Non-NN history step** | History step with no `action.modelId` (**human player seat** or **Randombot**) | Human step (wrong) |
| **Human step** | History step whose **actor seat id** equals the **human player seat** | **Non-NN history step**, `modelId` absent alone |
| **Derived training row** | One row per history step with `action`: pre-action obs/mask + **policy action index** | Sample, datapoint |
| **Web-authoritative labels** | **Derived training rows** from Node **web game engine** replay only | Python verifier, golden-only labels |

## Rules authority

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Web game engine** | portfolio-site JS kernel (`engine/kernel.js`); sole rules truth for replay pipeline | The engine (vague), Python `Match` |
| **Web engine root** | `PORTFOLIO_SITE_ROOT` path to portfolio-site checkout | Vendored kernel, submodule default |
| **Replay verifier** | Stage replaying each envelope to **match over** with legal actions + indices | Valid replay (ingest-only) |
| **Verified replay** | **Completed match replay** that reached **match over** via **replay verifier** | Ingest-eligible only |
| **Python training sim** | Legacy in-repo Python `Match` for PPO rollouts; not parity with web | Co-equal engine, JS alignment target |

## Pipeline gates

Stages people confuse; full pipeline vocabulary is in [`CONTEXT.md`](./CONTEXT.md).

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Ingest eligibility** | Shape and import rules aligned with `importReplayEnvelope`; no **web game engine** | Valid replay, verified |
| **Verified replay** | **Completed match replay** that passed **replay verifier** to **match over** | Ingest-eligible, import ok |
| **Ingest manifest** | `manifest.json`: `ingested` ids and `skipped` with reason codes | Processed list |
| **Verify manifest** | `verify_manifest.json`: `verified` and `failed` (+ **verify failure**) | Merged ingest state |
| **Ingest run atomicity** | Ingest updates **ingest manifest** only after the whole run succeeds | Partial manifest |
| **Verify run atomicity** | Verify updates **verify manifest** only after all pending replays finish | Partial verify state |

## Training data pipeline

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Training data root** | Gitignored tree (default `data/replays/`): raw, manifests, derived | `models/`, committed replays |
| **Raw envelope store** | `raw/{matchId}.json` verbatim eligible body | Stripped export |
| **Derived match artifact** | `derived/{matchId}/rows.parquet` + `meta.json` | `dataset.db`, monolithic DB |
| **Derived store** | All **derived match artifact** trees under `derived/` | Single corpus file only |
| **Pending verify** | Ingested id not yet in **verify manifest** | Re-verify `failed` automatically |
| **Pending dataset** | **Verified replay** missing or stale **derived match artifact** | Re-encode all every run |
| **Live replay ingest** | RTDB pull of **completed match replay archive** | Export-only ingest |
| **Replay pipeline CLI** | `python -m dungeon_runner.replay.cli <stage>` | One-off scripts per stage |
| **Manual pipeline run** | Maintainer chain ingest → … → train → optional publish | Cron, live Firebase Auth |

## Training runs & evaluation

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Training run id** | Dir name under `models/runs/` (`bc-*`, `ppo-*`) | Promoted semver, UUID-only |
| **Training run artifact** | `models/runs/<id>/`: weights, **metrics artifact**, `tb/` | Failed dir under `models/<semver>/` |
| **Metrics artifact** | `metrics.json` in run dir: eval + train losses + `parent_weights` | Metrics only on promoted dir |
| **Training parent** | Weights at run start (`models/latest/` in v1) | Checkpoint (vague) |
| **BC policy training** | Masked CE on **human step** rows; value head frozen in v1 | PPO on human rows only |
| **BC best checkpoint** | Epoch with best val human masked accuracy before early stop | Last epoch weights |
| **BC-anchored PPO policy training** | PPO on **Python training sim** with **BC anchor** + **BC-bot** | Script-only `train.py` as canonical |
| **Replay eval metrics** | Val **human step** masked accuracy on stored obs/mask (no re-replay) | Live engine re-run at eval |
| **Gate evaluator preview** | Applies **promotion gates** after train; does not promote | Preview = promote |

## PPO training opponents

**Python training sim** only—not play-surface roles.

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Learner** | Seat running the trainable policy during **BC-anchored PPO policy training** | Human player, player |
| **BC-bot** | Sim **opponent** seat acting via **frozen BC teacher** forward passes | **Neural opponent**, **Randombot** |
| **Frozen BC teacher** | Non-trainable snapshot of **PPO BC run** weights at PPO start | **Production latest**, drifting teacher |
| **Rollout match template** | Per-match lineup mix (**learner** vs **Randombot**, vs **BC-bot**, or self-play) | Eval suite, single-opponent block |

## Evaluation & promotion

Curated eval vocabulary; full detail in [`CONTEXT.md`](./CONTEXT.md).

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Frozen eval suite** | Fixed ~20% match-id holdout chosen once at **eval suite init**; val ids stable until suite version bump | Validation set (vague), re-split each run |
| **Eval suite artifact** | On-disk `eval_suite.json`: `val_match_ids`, suite version, sampling seed | Per-run random split |
| **Eval config artifact** | On-disk `eval_config.json`: **replay accuracy floor**, sim seed list, **sim regression tolerance** | Thresholds only in `metrics.json` |
| **Replay accuracy floor** | Minimum val **replay eval metrics** for promote; set once from first **BC baseline run** | Hand-tuned cutoff each train |
| **Sim regression tolerance** | ε allowing candidate sim win rate slightly below **production latest** on frozen seeds (default 0.01) | Zero tolerance, replay ε |
| **Sim eval metrics** | Win rate vs **Randombot** on frozen seeds in **Python training sim** (independent rollouts) | Web engine re-run at eval |
| **Promotion gates** | Both replay leg (floor on **human step** val accuracy) and sim leg (**sim eval metrics** vs `latest` − ε) | Replay-only promote, “eval passed” |

## Model release (two repos)

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Promoted version** | Semver dir `models/<version>/` after successful **gated promotion** | Training run id as dir name |
| **Gated promotion** | `publish` copies weights + **promotion manifest**; updates **production latest** | Deploy, release (includes TF.js) |
| **Production latest** | Symlink `models/latest/` → current **promoted version** (H5) | Real `latest/` directory tree |
| **Web deployed latest** | portfolio-site TF.js alias `public/.../latest/` for `modelId: 'latest'` | Same as **production latest** path |
| **Deployed model version** | Immutable semver TF.js tree mirroring a **promoted version** | Training run id folder |
| **Model catalog** | portfolio-site `models.json` listing TF.js dirs (not **game data catalog**) | Catalog (unqualified) |
| **TF.js model sync** | portfolio-site convert H5 → TF.js + catalog + maybe **web deployed latest** | `sync` without promote context |
| **Two-repo model release** | **Gated promotion** here, then **TF.js model sync** + **release smoke** there | Single-repo checklist |
| **Epic v1 success bar** | Shipped milestone: ≥1 **gated promotion** + documented **two-repo model release**; PPO optional | Ingest done, one `bc` run, sync alone |
| **Release smoke** | Manual play on `/projects/dungeon-runner` after sync | Headless WebGL-only gate |

## Relationships

- A **match** produces **history**; at **match over** the browser may upload a **completed match replay** to the **completed match replay archive**.
- **Live replay ingest** → **raw envelope store** → **replay verifier** (**web game engine**) → **derived store** → **BC policy training** / **BC-anchored PPO policy training** → **training run artifact** → optional **gated promotion** → **production latest**.
- **Promoted version** is the shared semver identity; **production latest** (H5) and **web deployed latest** (TF.js) are format-specific aliases for the same production pointer.
- **Epic v1 success bar** closes when promote + cross-repo release docs have been executed end-to-end at least once—not when ingest/verify/dataset alone work.
- **Replay envelope contract (v1)** is authoritative in portfolio-site; ingest skip extensions are in [`docs/replay-pipeline.md`](./docs/replay-pipeline.md).
- Each **match** has one **human player seat**; **Human step** rows drive BC and replay gates; **Non-NN history step** from **Randombot** may still yield **derived training rows** with `is_human: false`.
- **Randombot** setup role is `randombot`; **sim eval metrics** and PPO templates refer to the same opponent in domain language.
- **Learner** is sim-training vocabulary only—not the **human player seat** and not a portfolio **Neural opponent**.

## Example dialogue

> **Dev:** "Can we train labels from Python `Match`?"  
> **Domain expert:** "No — **web-authoritative labels** only. **Replay verifier** and dataset build call the **web game engine** via **web engine root**."

> **Dev:** "Steps without `modelId` — all human training rows?"  
> **Domain expert:** "No — those are **Non-NN history step** entries. Only steps whose **actor seat id** is the **human player seat** are **Human step** rows with `is_human: true`."

> **Dev:** "Ingest passed but verify failed with `match_not_over`."  
> **Domain expert:** "**Ingest eligibility** is shape-only. **Verified replay** requires **match over** in the **web game engine**, not import alone."

> **Dev:** "BC epoch accuracy is flat at 0.89 — broken?"  
> **Domain expert:** "Often the **training parent** already matches val argmax labels; **BC best checkpoint** may stay epoch 1. Check **replay eval metrics** on the restored weights, not plateau alone."

> **Dev:** "Replay accuracy beat the floor but promote failed — why?"  
> **Domain expert:** "**Promotion gates** need both legs. Check **sim eval metrics** on the **frozen eval suite** seeds in **Python training sim** against **production latest** within **sim regression tolerance**."

> **Dev:** "Is the **learner** the human in archived replays?"  
> **Domain expert:** "No — **learner** is which sim seat is training in PPO. The human is the **human player seat** in real **match** exports; **BC-bot** is a frozen teacher opponent, not the **learner**."

> **Dev:** "`publish` succeeded but players see old NN weights."  
> **Domain expert:** "**Gated promotion** updated **production latest** here. Run **TF.js model sync** in portfolio-site so **web deployed latest** matches the new **promoted version**."

## Flagged ambiguities

- **Catalog** — use **model catalog** (neural weights) vs portfolio-site **game data catalog** (equipment/monsters); never bare “catalog.”
- **Match** vs game — canonical term is **match** in all product-chain docs (see sibling glossary).
- **Match over** vs **sim empty-pile forfeit** — **match over** is web-only (always a winner). Sim forfeit is a legacy anti-pass-farming rule; early models exploited pass-heavy play; divergence persists as a training **local minima** until sim is retired or aligned.
- **Empty dungeon pile at bidding end** — web: immediate successful **dungeon run**; sim: **sim empty-pile forfeit** (not **match over**). Mid-run empty-pile clear is aligned between environments.
- **Production latest** vs **web deployed latest** — same production line, different artifact format; always name which side when debugging sync.
- **Randombot** vs `RandomBot` — product term **Randombot**; **RandomBot** only when citing Python class names in code-oriented maintainer notes.

## See also

- [`CONTEXT.md`](./CONTEXT.md) — exhaustive training glossary (all pipeline stages, eval, PPO, publish)  
- [`CROSS_REPO.md`](./CROSS_REPO.md) — sibling paths, env vars, doc map  
- [`portfolio-site/UBIQUITOUS_LANGUAGE.md`](https://github.com/enmaku/portfolio-site/blob/main/UBIQUITOUS_LANGUAGE.md) — play + site terms (duplicate shared tables)
