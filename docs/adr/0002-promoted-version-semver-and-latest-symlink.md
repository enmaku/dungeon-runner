---
status: accepted
---

# Promoted model versions use v0.2+ semver; production latest is a symlink

Legacy promoted dirs (`v0.1.29a`, `v0.1.30a`) named training epochs under an alpha `v0.1` line. The replay pipeline (#6–#8) introduces **gated promotion** from **training run artifacts** under `models/runs/`, which must not reuse run ids as semver paths or keep duplicating weights in a real `models/latest/` directory.

We assign **promoted version** semver dirs on successful `publish` only: first replay-pipeline promote is `v0.2`, then auto patch bumps `v0.2.01`, `v0.2.02`, … (two-digit patch, no letter suffix). Minor line changes (e.g. `v0.3`) require an explicit `publish --version`. **Production latest** is always a symlink to the current promoted dir; issue #8 includes a one-time migration from the legacy duplicate `latest/` tree to `../v0.1.30a` until the first `v0.2` promote repoints it. Audit uses **promotion manifest** (`promotion.json` per version + append-only `models/promotions.jsonl`).

## Considered options

- **Promoted dir = training run id** (`bc-*`, `ppo-*`) — rejected: pollutes semver namespace and conflates failed runs with shipped versions.
- **Continue `v0.1.*a` epoch naming** — rejected: new pipeline is post-alpha; epoch numbers no longer match how models are trained.
- **Keep `latest/` as a copied weights directory** — rejected: drifts from promoted semver dirs and breaks a single pointer for **training parent** and gates.

## Consequences

- Implementers add a version allocator (scan existing promoted dirs + JSONL; patch auto-bump under current minor; `--version` for minor jumps).
- `publish` stages `models/<version>.tmp/`, renames, then updates symlink and JSONL last (**publish run atomicity**).
- portfolio-site TF.js sync (#11) keys off stable semver labels, not `models/runs/` ids.
