# dungeon-runner

Python repo for training neural opponents in a digital implementation of IELLO Games' [Welcome to the Dungeon](https://iellogames.com/games/welcome-to-the-dungeon/). This repo owns match simulation (legacy Python), BC/PPO training, replay ingest from human play, and gated promotion of Keras weights. The playable match UI, authoritative web game engine, replay export, and TensorFlow.js deployment live in the sibling [portfolio-site](https://github.com/enmaku/portfolio-site) repo.

## Tech stack

| Layer | Choice |
| --- | --- |
| Language | Python 3.10+ |
| Package | setuptools editable install (`pip install -e .`) |
| ML | TensorFlow / Keras (`PolicyValueModel`, custom PPO loop) |
| Replay labels | Node harness importing portfolio-site web engine |
| Training data | Parquet (`pyarrow`), gitignored under `data/` |
| Legacy sim | PettingZoo AEC env, optional Ray-parallel rollouts |
| Optional UI | pygame table view (`pip install -e ".[gui]"`) |
| Tests | pytest |
| Metrics | TensorBoard under `models/runs/<run_id>/tb/` |

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,train]"
```

For replay pipeline work you also need a sibling portfolio-site checkout and Node (verify/dataset harnesses invoke the web engine). Default layout:

```
../dungeon-runner
../portfolio-site
```

### Environment variables

Copy `.env.example` to `.env` at the repo root. The replay CLI loads it automatically.

| Variable | Required for | Notes |
| --- | --- | --- |
| `FIREBASE_DATABASE_URL` | `ingest` (live RTDB) | Same value as portfolio-site `VITE_FIREBASE_DATABASE_URL`. Not needed for `ingest --from-export`. |
| `PORTFOLIO_SITE_ROOT` | `verify`, `dataset`, Node harness tests | Absolute path to portfolio-site checkout (web engine root). |

Ingest uses RTDB REST read only (no Firebase Auth in v1). Verify and dataset fail fast if `PORTFOLIO_SITE_ROOT` is unset.

Optional dependency groups in [`pyproject.toml`](pyproject.toml):

| Group | Installs | Use when |
| --- | --- | --- |
| `dev` | pytest | Running tests |
| `gui` | pygame | `play_random_game.py --gui` |
| `train` | TensorFlow, PettingZoo, Ray, TensorBoard | BC/PPO stages and legacy `scripts/train*.py` |

Minimal dev setup: [`requirements.txt`](requirements.txt) (`pip install -e .` + pytest).

## What this repo does

| Track | Purpose |
| --- | --- |
| Replay pipeline | Ingest human completed match replays from portfolio-site, verify against the web game engine, build Parquet training rows, run BC/PPO, gated-promote weights |
| Legacy Python sim | In-repo `Match` + PettingZoo env for experiments and sim eval gates; not maintained for parity with the browser engine |
| Maintainer scripts | `scripts/train*.py`, `play_random_game.py`, `replay-train-and-release.sh` for ad hoc runs outside the staged CLI |

Human play happens in portfolio-site at `/projects/dungeon-runner`. Finished matches upload to RTDB `dungeonRunnerCompletedMatches`; this repo pulls them for imitation learning.

## Replay training pipeline

Maintainer runbook: [`docs/replay-pipeline.md`](docs/replay-pipeline.md). Domain terms: [`CONTEXT.md`](CONTEXT.md) (exhaustive) and [`UBIQUITOUS_LANGUAGE.md`](UBIQUITOUS_LANGUAGE.md) (consolidated).

CLI entry point:

```bash
python -m dungeon_runner.replay.cli <stage> [--data-dir data/replays]
```

| Stage | Purpose |
| --- | --- |
| `ingest` | Pull RTDB or `--from-export` JSON into raw envelope store + ingest manifest |
| `verify` | Replay each pending envelope through the web game engine to `match-over` |
| `eval_suite init` | Freeze held-out match ids (~20%) for replay eval metrics |
| `eval_config init` | Freeze sim seeds, regression tolerance; replay accuracy floor set by first BC baseline |
| `dataset` | Build derived training rows (Node labels → Parquet per match) |
| `bc` | BC policy training → training run artifact under `models/runs/` |
| `ppo` | BC-anchored PPO on Python sim (requires `--bc-run`) |
| `publish` | Gated promotion to `models/<promoted version>/` + production latest symlink |
| `run-all` | Orchestrate stages in order; PPO and publish are opt-in (`--with-ppo`, `--with-publish`) |

Bootstrap order on a fresh machine (after `.env`):

1. `ingest`
2. `verify`
3. `eval_suite init` then `eval_config init` (needs ≥2 verified match ids)
4. `dataset`
5. `bc` (optional `ppo`, optional `publish`)

Default `run-all` stops after `bc` so you can review metrics before promoting.

Weight I/O uses repo-root `models/` (not `--data-dir`). Training data (raw replays, manifests, derived Parquet, eval artifacts) defaults to gitignored `data/replays/`.

Full stage flags, manifest shapes, skip-reason tables, and release handoff: [`docs/replay-pipeline.md`](docs/replay-pipeline.md).

## Navigating the codebase

```
.
├── src/dungeon_runner/
│   ├── match.py, actions.py, catalog.py, types_core.py   # Legacy Python rules sim
│   ├── pettingzoo_aec.py                                 # WtdAECEnv (2–4 players)
│   ├── bots/                                             # RandomBot and helpers
│   ├── rl/                                               # Obs, action mask, PPO, PolicyValueModel
│   ├── ui/pygame_view.py                                 # Optional table UI
│   └── replay/
│       ├── cli.py                                        # Pipeline entry point
│       ├── ingest.py, verify.py, dataset.py              # Stages
│       ├── bc/, ppo/, publish/, eval/                    # Training, gates, promotion
│       ├── harness/                                      # Node scripts (verify, dataset build)
│       └── web_engine.py                                 # PORTFOLIO_SITE_ROOT resolution
├── scripts/                                              # Legacy trainers and training script
├── tests/                                                # pytest; replay/ mirrors pipeline stages
│   └── fixtures/replay/                                  # Verifier wiring fixtures
├── models/                                               # Promoted semver dirs, runs/
├── data/                                                 # Training data root
└── docs/
    ├── replay-pipeline.md
    ├── welcome-to-the-dungeon.md                         # Physical-game rules reference
    └── adr/
```

### Where to look by task

| Task | Start here |
| --- | --- |
| Add or change a pipeline stage | `src/dungeon_runner/replay/cli.py`, stage module under `replay/` |
| Ingest eligibility / skip reasons | `replay/eligibility.py`, parity tests in `tests/replay/ingest/` |
| Replay verification | `replay/verify.py`, `replay/harness/verify_match.mjs` |
| Dataset / derived rows | `replay/dataset.py`, `replay/harness/build_match_dataset.mjs` |
| BC or PPO training | `replay/bc/`, `replay/ppo/` |
| Promotion gates / publish | `replay/publish/`, `replay/eval/gate_evaluator.py` |
| Policy network architecture | `rl/model.py` (`DEFAULT_PPO_HIDDEN`, 87→26 layout) |
| Observation vector (Python sim) | `rl/observation.py` (`OBS_DIM = 87`) |
| Legacy self-play PPO scripts | `scripts/train.py`, `scripts/train_rllib.py`, `scripts/oscillate_train.py` |
| Cross-repo release | [`docs/replay-pipeline.md` § Release](docs/replay-pipeline.md#release-to-portfolio-site), portfolio-site [`scripts/MODEL_RELEASE.md`](https://github.com/enmaku/portfolio-site/blob/main/scripts/MODEL_RELEASE.md) |
| Replay envelope field rules | portfolio-site [`CONTRACT.md`](https://github.com/enmaku/portfolio-site/blob/main/src/features/dungeon-runner/CONTRACT.md) (normative); ingest extensions in pipeline doc |

### Domain language

Read before naming things or writing maintainer docs:

- [`UBIQUITOUS_LANGUAGE.md`](UBIQUITOUS_LANGUAGE.md) — shared product-chain terms with portfolio-site
- [`CONTEXT.md`](CONTEXT.md) — exhaustive training pipeline glossary
- [`CROSS_REPO.md`](CROSS_REPO.md) — sibling env vars, doc index, intentional divergences
- [`docs/welcome-to-the-dungeon.md`](docs/welcome-to-the-dungeon.md) — tabletop rules this sim encodes

Non-obvious architectural choices:

- [ADR 0001](docs/adr/0001-web-game-engine-authoritative.md) — web engine is rules truth; Python sim is legacy
- [ADR 0002](docs/adr/0002-promoted-version-semver-and-latest-symlink.md) — promoted semver dirs and `models/latest` symlink

## Web game engine vs Python sim

Replay verify, dataset labels, and replay eval metrics use the portfolio-site JavaScript kernel via Node (`engine/kernel.js`, `nn/policyAdapter.js`). Python `Match` is the Python training sim: still used for PPO rollouts and sim regression gates in v1, but not kept in parity with browser play. See [ADR 0001](docs/adr/0001-web-game-engine-authoritative.md).

Implication for contributors: rules changes belong in portfolio-site; this repo consumes replay envelopes and orchestrates Node harnesses—it does not fork game logic for the replay pipeline.

## Models layout

```
models/
  latest/                    # symlink → ../<promoted version>/ (production latest)
  runs/<run_id>/             # bc-* / ppo-* training run artifacts
    policy.weights.h5
    metrics.json
    tb/
  <promoted version>/        # gated promotion output (v0.2, v0.2.01, …)
  promotions.jsonl           # append-only promotion ledger
```

Gated promotion (`publish`) copies candidate weights here only when promotion gates pass (replay accuracy floor + sim non-regression vs latest). Each training run id may promote at most once.

After promote, sync TF.js weights in portfolio-site (`npm run sync-dungeon-runner-model`). That step is not implemented in this repo.

## portfolio-site sibling

| This repo | portfolio-site |
| --- | --- |
| Replay ingest, verify, dataset, BC/PPO, gated promotion | Playable match UI, web game engine, replay envelope export |
| Keras H5 under `models/<promoted version>/` | TF.js trees under `public/models/dungeon-runner/` |
| `models/latest` symlink (production latest) | `public/models/dungeon-runner/latest/` (web deployed latest) |
| `PORTFOLIO_SITE_ROOT` in `.env` | `DUNGEON_RUNNER_ROOT` in `.env` for model sync |

Expected sibling layout: `../portfolio-site` (set `PORTFOLIO_SITE_ROOT` here; set `DUNGEON_RUNNER_ROOT` there).

Two-repo release: promote here, then follow portfolio-site [`scripts/MODEL_RELEASE.md`](https://github.com/enmaku/portfolio-site/blob/main/scripts/MODEL_RELEASE.md). Cross-repo index: [`CROSS_REPO.md`](CROSS_REPO.md).

Play the shipped game: [Focus Disorder — Dungeon Runner](https://focusdisorder.com/#/projects/dungeon-runner) (or local `npm run dev` in portfolio-site).

## Testing

```bash
pytest                          # full suite
pytest tests/replay/ -q         # replay pipeline only
pytest tests/test_match_e2e.py  # legacy Python sim
```

Verify and dataset integration tests need `PORTFOLIO_SITE_ROOT` pointing at a portfolio-site checkout. With it set, verifier tests also replay the canonical golden fixture at `portfolio-site/src/features/dungeon-runner/engine/fixtures/golden-seed-4242-two-pass.json`.

Node harness unit test (no portfolio-site required):

```bash
node --test tests/replay/harness/replay_step_apply_seat.test.mjs
```

## Legacy training and pygame (optional)

Experimental paths outside the canonical replay CLI:

| Script | Purpose |
| --- | --- |
| `scripts/train.py` | In-process PPO vs random bot; writes `logdir/policy.weights.h5` |
| `scripts/train_rllib.py` | Same Keras PPO update with Ray-parallel self-play sampling |
| `scripts/oscillate_train.py` | Alternate the two trainers into numbered subdirs under a run root |
| `scripts/play_random_game.py` | CLI match; add `--gui` for pygame table view |
| `scripts/replay-train-and-release.sh` | Full pipeline + optional portfolio-site TF.js sync (maintainer convenience) |

Smoke after `pip install -e ".[train]"`:

```bash
python scripts/train.py --logdir runs/smoke --updates 5
tensorboard --logdir runs/smoke/scalars
```

Legacy sim notes:

- Bidding runs clockwise with partial observability per seat (`rl/observation.py`, 87 floats + 26-action mask).
- Training samples 2–4 players per episode for varied table sizes.
- Optimizer state is not saved across script restarts; H5 weights are.
- Reward shaping lives in `rl/rewards.py`.

## Official rulebook

Publisher PDF: [Welcome to the Dungeon — English rulebook](https://iellogames.com/wp-content/uploads/2016/08/Welcome-to-the-Dungeon-EN-Rulebook_web.pdf).

## License

Game rules and trademarks belong to their respective owners; this repository is only an independent implementation and research code. I'm doing this for my own education and because I love your game, please don't sue me IELLO.
