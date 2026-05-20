---
status: accepted
---

# Web game engine is authoritative; Python sim is legacy

The playable game and **completed match replay** exports live in portfolio-site. dungeon-runner’s replay pipeline (verify, dataset labels, replay **promotion gates**) must match what players actually ran, not a second rules implementation in Python.

We treat the portfolio-site JavaScript kernel as the **web game engine**, invoked via Node (`PORTFOLIO_SITE_ROOT`) from dungeon-runner-owned harnesses that import existing modules (`debug/replaySession.js`, `nn/policyAdapter.js`, etc.). We will not invest in Python `Match` parity with JS. The in-repo **Python training sim** remains only for legacy PPO/self-play and **legacy Python sim benchmarks** in v1 **promotion gates** until training moves off Python; it is expected to be deprecated after the replay pipeline ships.

## Considered options

- **Dual runtime with parity tests** — rejected: high ongoing cost; replays already prove web behavior.
- **Rewrite training entirely in Node first** — deferred: blocks shipping ingest/verify/BC; sim gates stay on Python temporarily.
- **Python verifier with copied rules** — rejected: duplicates truth and drifts from production.

## Consequences

- Issue #3 and dataset work land primarily in dungeon-runner; portfolio-site changes stay minimal (e.g. exporting `encodeActionIndex` when the harness needs it).
- Human **derived training rows** and **verified replay** are always **web-authoritative labels**.
- New rules work belongs in portfolio-site; dungeon-runner consumes envelopes and orchestrates Node, it does not fork game logic.
