"""Per-match rollout template draw (fixed v1 probabilities)."""

from __future__ import annotations

import random

TEMPLATE_VS_RANDOMBOT = "vs_randombot"
TEMPLATE_BC_BOT = "vs_bc_bot"
TEMPLATE_SELF_PLAY = "self_play"

_TEMPLATES = (TEMPLATE_VS_RANDOMBOT, TEMPLATE_BC_BOT, TEMPLATE_SELF_PLAY)
_WEIGHTS = (0.20, 0.45, 0.35)


def sample_rollout_template(rng: random.Random) -> str:
    return rng.choices(_TEMPLATES, weights=_WEIGHTS, k=1)[0]


def learner_seat(n_players: int, rng: random.Random) -> int:
    return int(rng.randrange(n_players))


def roles_for_template(template: str, n_players: int, rng: random.Random) -> list[bool]:
    if template == TEMPLATE_SELF_PLAY:
        return [True] * n_players
    seat = learner_seat(n_players, rng)
    return [i == seat for i in range(n_players)]
