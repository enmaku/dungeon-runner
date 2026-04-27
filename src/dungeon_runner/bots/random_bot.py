"""Weighted random policy for simulation and RL opponent baselines."""

from __future__ import annotations

import random
from dataclasses import dataclass

import dungeon_runner.actions as A
from dungeon_runner.catalog import SPECIES_DATA
from dungeon_runner.match import BiddingState, Match, MatchPhase

_PASS_WEIGHT_GROWTH_PER_DUNGEON_CARD = 0.14
_SACRIFICE_WEIGHT_DECAY_PER_EQUIP_REMOVED = 0.58
_SACRIFICE_STRENGTH_MIN = 0.2
_SACRIFICE_STRENGTH_SCALE = 0.14


def _strength_driven_weight(strength: int) -> float:
    return max(_SACRIFICE_STRENGTH_MIN, _SACRIFICE_STRENGTH_SCALE * float(strength))


def _sim_pass_weight(m: Match, base: float) -> float:
    if m.phase is not MatchPhase.BIDDING:
        return base
    pile = len(m.dungeon_pile)
    return base * (1.0 + _PASS_WEIGHT_GROWTH_PER_DUNGEON_CARD * pile)


def _sim_sacrifice_weight(m: Match, base: float) -> float:
    if m.phase is not MatchPhase.BIDDING or m.bidding_sub is not BiddingState.PENDING:
        return base
    n = len(m.sacrifice_rows)
    return base * (_SACRIFICE_WEIGHT_DECAY_PER_EQUIP_REMOVED**n)


def _sacrifice_pending_strength_factor(m: Match) -> float:
    if m.phase is not MatchPhase.BIDDING or m.bidding_sub is not BiddingState.PENDING:
        return 1.0
    c = m.pending_card
    if c is None:
        return 1.0
    return _strength_driven_weight(c.strength)


def random_sim_action_subset(_m: Match, legal: set[object]) -> set[object]:
    """Reserved for narrowing the toy action set; currently returns ``legal`` unchanged."""
    return set(legal)


def _action_weight(
    a: object,
    m: Match,
    *,
    pass_weight: float,
    sacrifice_weight: float,
) -> float:
    if isinstance(a, A.PassBid):
        return _sim_pass_weight(m, pass_weight)
    if isinstance(a, A.SacrificeEquipment):
        w0 = _sim_sacrifice_weight(m, sacrifice_weight)
        w0 *= _sacrifice_pending_strength_factor(m)
        return max(1e-9, w0)
    if isinstance(a, A.DeclareVorpal):
        st0 = SPECIES_DATA[a.target_species][0]
        return max(1e-9, _strength_driven_weight(st0))
    return 1.0


@dataclass
class RandomBot:
    pass_weight: float = 0.2
    sacrifice_weight: float = 0.03
    apply_toy_subset: bool = True

    def select(self, m: Match, actions: set[object], rng: random.Random) -> object:
        acts = set(actions)
        if self.apply_toy_subset:
            narrowed = random_sim_action_subset(m, acts)
            if narrowed:
                acts = narrowed
        return pick_action(m, acts, rng, pass_weight=self.pass_weight, sacrifice_weight=self.sacrifice_weight)


def pick_action(
    m: Match,
    actions: set[object],
    rng: random.Random,
    *,
    pass_weight: float = 0.2,
    sacrifice_weight: float = 0.03,
) -> object:
    acts = list(actions)
    w = [
        _action_weight(a, m, pass_weight=pass_weight, sacrifice_weight=sacrifice_weight)
        for a in acts
    ]
    return rng.choices(acts, weights=w, k=1)[0]
