"""Legacy Python sim benchmarks vs RandomBot on frozen seeds."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Protocol

from dungeon_runner.bots import RandomBot
from dungeon_runner.match import Match, MatchPhase
from dungeon_runner.types_core import AdventurerKind

ENGINE_NAME = "python_training_sim"
DEFAULT_MAX_STEPS = 20_000


class SimPolicy(Protocol):
    def select(self, m: Match, actions: set[object], rng: random.Random) -> object: ...


@dataclass(frozen=True)
class SimMetrics:
    candidate_win_rate_vs_randombot: float
    latest_win_rate_vs_randombot: float
    engine: str = ENGINE_NAME
    seed_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_win_rate_vs_randombot": self.candidate_win_rate_vs_randombot,
            "latest_win_rate_vs_randombot": self.latest_win_rate_vs_randombot,
            "engine": self.engine,
            "seed_count": self.seed_count,
        }


def _play_vs_randombot(
    policy: SimPolicy,
    seed: int,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    candidate_seat: int = 0,
) -> bool:
    rng = random.Random(seed)
    m = Match.new(2, rng, AdventurerKind.WARRIOR, 0)
    opponent = RandomBot()
    steps = 0
    while m.phase is not MatchPhase.ENDED and steps < max_steps:
        acts = m.legal_actions()
        if not acts:
            break
        seat = m.active_seat
        if seat == candidate_seat:
            action = policy.select(m, acts, rng)
        else:
            action = opponent.select(m, acts, rng)
        m.apply(action)
        steps += 1
    return m.winner_seat == candidate_seat


def win_rate_vs_randombot(
    policy: SimPolicy,
    seeds: list[int],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> float:
    if not seeds:
        return 0.0
    wins = sum(
        1 for seed in seeds if _play_vs_randombot(policy, seed, max_steps=max_steps)
    )
    return wins / len(seeds)


def sim_metrics(
    candidate: SimPolicy,
    latest: SimPolicy,
    seeds: list[int],
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> SimMetrics:
    cand_wr = win_rate_vs_randombot(candidate, seeds, max_steps=max_steps)
    latest_wr = win_rate_vs_randombot(latest, seeds, max_steps=max_steps)
    return SimMetrics(
        candidate_win_rate_vs_randombot=cand_wr,
        latest_win_rate_vs_randombot=latest_wr,
        seed_count=len(seeds),
    )


def sim_passes_regression(
    metrics: SimMetrics,
    tolerance: float,
) -> bool:
    return (
        metrics.candidate_win_rate_vs_randombot
        >= metrics.latest_win_rate_vs_randombot - tolerance
    )
