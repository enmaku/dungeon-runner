# Ubiquitous language (dungeon-runner)

Full definitions live in [`CONTEXT.md`](./CONTEXT.md). Cross-repo term links (no translation) live in [`CROSS_REPO.md`](./CROSS_REPO.md).

## Product chain

| Term | Definition | Sibling doc |
| --- | --- | --- |
| **Completed match replay** | Finished match exported as a versioned **replay envelope** for training ingest | portfolio-site [Dungeon Runner CONTEXT](https://github.com/enmaku/portfolio-site/blob/main/src/features/dungeon-runner/CONTEXT.md) |
| **Replay envelope** | Versioned contract (`seed`, `setup`, `history`, …) authored in portfolio-site | [CONTRACT.md](https://github.com/enmaku/portfolio-site/blob/main/src/features/dungeon-runner/CONTRACT.md) |
| **Gated promotion** | Copy promoted H5 weights after **promotion gates**; repoint **production latest** | portfolio-site sync: [MODEL_RELEASE.md](https://github.com/enmaku/portfolio-site/blob/main/scripts/MODEL_RELEASE.md) |
| **Production latest** | Symlink `models/latest/` → current **promoted version** H5 | ↔ portfolio-site **web deployed latest** in [CROSS_REPO.md](./CROSS_REPO.md) |

## Example dialogue

> **Dev:** "Where is the replay field list defined?"  
> **Domain expert:** "Normative **replay envelope contract (v1)** is in portfolio-site CONTRACT—we ingest and cross-link, we don't fork the spec in **CONTEXT.md**."

> **Dev:** "Players still see the old NN after `publish`."  
> **Domain expert:** "**Gated promotion** only updates **production latest** here. Run **TF.js model sync** in portfolio-site so **web deployed latest** catches up."

## Flagged ambiguities

- **Catalog** — always say **model catalog** (weights) vs portfolio-site **game data catalog**; see [CROSS_REPO.md](./CROSS_REPO.md).
- **Game** vs **match** — training docs use portfolio-site **match** vocabulary; see CROSS_REPO intentional divergences.
