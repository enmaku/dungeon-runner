# Cross-repo vocabulary (dungeon-runner ↔ portfolio-site)

This repo owns **replay training**, **gated promotion**, and H5 **production latest**. [portfolio-site](https://github.com/enmaku/portfolio-site) owns the playable **match**, **replay envelope** export, and TF.js **web deployed latest**. Terms are **not translated** across repos—link to the other glossary and keep local names.

**Local glossary:** [`CONTEXT.md`](./CONTEXT.md)  
**Sibling glossary:** [`portfolio-site` Dungeon Runner `CONTEXT.md`](https://github.com/enmaku/portfolio-site/blob/main/src/features/dungeon-runner/CONTEXT.md) (checkout: `$PORTFOLIO_SITE_ROOT/src/features/dungeon-runner/CONTEXT.md`)

## Sibling checkout

| Env var | Points at |
| --- | --- |
| `PORTFOLIO_SITE_ROOT` | portfolio-site repo root (**web engine root**) |
| `DUNGEON_RUNNER_ROOT` | (set in portfolio-site) this repo |

Default sibling layout: `../portfolio-site` and `../dungeon-runner`.

## Shared documentation

| Topic | dungeon-runner | portfolio-site |
| --- | --- | --- |
| Play + envelope glossary | [`CONTEXT.md`](./CONTEXT.md) (training) | [`src/features/dungeon-runner/CONTEXT.md`](https://github.com/enmaku/portfolio-site/blob/main/src/features/dungeon-runner/CONTEXT.md) |
| Cross-repo index | this file | [`CROSS_REPO.md`](https://github.com/enmaku/portfolio-site/blob/main/CROSS_REPO.md) |
| Replay envelope v1 (normative fields) | cross-link in [`CONTEXT.md`](./CONTEXT.md) | [`src/features/dungeon-runner/CONTRACT.md`](https://github.com/enmaku/portfolio-site/blob/main/src/features/dungeon-runner/CONTRACT.md) |
| Ingest extensions (skip codes, RTDB layout) | [`docs/replay-pipeline.md`](./docs/replay-pipeline.md) | CONTRACT → pipeline doc |
| Maintainer runbooks | [`docs/replay-pipeline.md`](./docs/replay-pipeline.md) (promote) | [`scripts/MODEL_RELEASE.md`](https://github.com/enmaku/portfolio-site/blob/main/scripts/MODEL_RELEASE.md) (TF.js sync) |
| Web engine authority | [ADR 0001](./docs/adr/0001-web-game-engine-authoritative.md) | CONTRACT + kernel |
| Promoted semver + `latest` symlink | [ADR 0002](./docs/adr/0002-promoted-version-semver-and-latest-symlink.md) | MODEL_RELEASE + feature CONTEXT |
| Site-wide contexts | — | [`CONTEXT-MAP.md`](https://github.com/enmaku/portfolio-site/blob/main/CONTEXT-MAP.md) |

## Cross-linked terms (same concept, local name)

| dungeon-runner (`CONTEXT.md`) | portfolio-site | Notes |
| --- | --- | --- |
| **Completed match replay** | **Completed match replay** | Same term; producer uploads, consumer ingests. |
| **Replay envelope** | **Replay envelope** | portfolio-site authors shape; dungeon-runner ingests verbatim. |
| **Replay envelope contract (v1)** | CONTRACT § Replay envelope contract (v1) | Normative fields only in portfolio-site; ingest extensions in pipeline doc. |
| **Completed match replay archive** | **Completed match replay archive** | RTDB `dungeonRunnerCompletedMatches`; listing for ingest. |
| **Web game engine** | engine / `kernel.js` (no separate glossary entry) | Authoritative rules runtime; portfolio-site implements, dungeon-runner invokes via Node. |
| **Match over** | **Match over** | **Replay verifier** requires terminal phase `match-over`. |
| **TF.js model sync** | **TF.js model sync** | Runs in portfolio-site after **gated promotion** here. |
| **Two-repo model release** | `scripts/MODEL_RELEASE.md` | Split checklists; together = **Epic v1 success bar** tail. |
| **Canonical golden replay** | `engine/fixtures/golden-seed-4242-two-pass.json` | Lives in portfolio-site; verifier tests when **web engine root** set. |

## Cross-linked pairs (related concepts, different names)

| dungeon-runner | portfolio-site | Relationship |
| --- | --- | --- |
| **Production latest** | **Web deployed latest** | H5 symlink `models/latest/` here ↔ TF.js alias `public/models/dungeon-runner/latest/`. Same production pointer, different artifact format. |
| **Promoted version** | **Deployed model version** | Semver dir under `models/<id>/` here ↔ `public/models/dungeon-runner/<id>/` after sync. |
| **Model catalog** | **Model catalog** (`models.json`) | Neural TF.js ids only—not **game data catalog**. |
| **Gated promotion** | (no glossary term) | `publish` stage here; portfolio-site has no promote step. |
| **Training run id** | (no glossary term) | `bc-*` / `ppo-*` under `models/runs/`; never synced as player-facing ids. |
| **Derived training row** | **History** + dataset build | Rows are built from **history** via **web game engine**; **Human step** / `is_human` resolved at dataset build, not in envelope. |
| **Policy action index** | `encodeActionIndex` (CONTRACT export) | 0–25 labels; mapped only in portfolio-site; dungeon-runner stores, does not re-derive. |
| **Actor seat id** | `actorSeatId` (CONTRACT field) | `seat-1`…`seat-4`; not the human player by number alone. |
| **Python training sim** | (no glossary term) | Legacy `Match` here; playable **match** uses web engine only. |

## Intentional divergences (do not unify)

| Topic | dungeon-runner | portfolio-site | Cross-link |
| --- | --- | --- | --- |
| Casual “game” vs **match** | Avoid “game” in training docs | **Match** is the product term | portfolio-site [`CONTEXT.md` — Match](https://github.com/enmaku/portfolio-site/blob/main/src/features/dungeon-runner/CONTEXT.md) |
| **Empty dungeon run** | Python sim historically scored runner **loss** on empty pile | Table/web: runner **win** | portfolio-site Flagged ambiguities → dungeon-runner training note |
| **Catalog** without qualifier | **Model catalog** (weights) | **Game data catalog** (equipment/monsters) | Always use full term in either repo |
| `version` on envelope | Ingest: integer `1` only | Import: `version !== 1` (looser types) | Eligibility must match; type strictness may differ at edge types—see pipeline doc **Intentional strictness** |
| Replay envelope `version` | Skip unsupported at ingest | Export always integer `1` | v2 roadmap only in portfolio-site CONTRACT |

## Coordination issues

| Tracker | Repo |
| --- | --- |
| [#11](https://github.com/enmaku/dungeon-runner/issues/11) epic cross-repo | dungeon-runner |
| [#128](https://github.com/enmaku/portfolio-site/issues/128) umbrella | portfolio-site |
| [#127](https://github.com/enmaku/portfolio-site/issues/127) TF.js sync | portfolio-site |
