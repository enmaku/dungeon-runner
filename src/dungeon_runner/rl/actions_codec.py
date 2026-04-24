"""Fixed Discrete action index ↔ engine actions + legal mask (dim 26)."""

from __future__ import annotations

import numpy as np

import dungeon_runner.actions as A
from dungeon_runner.catalog import all_equipment_ids
from dungeon_runner.match import ALL_SPECIES, Match, MatchPhase
from dungeon_runner.types_core import AdventurerKind

# Layout (must stay stable for checkpoints)
IDX_PASS = 0
IDX_DRAW = 1
IDX_ADD = 2
# 3..8 sacrifice slots (hero loadout order)
N_SACRIFICE_SLOTS = 6
IDX_SACRIFICE_BASE = 3
# 9..12 pick next adventurer W,B,M,R
IDX_PICK_HERO_BASE = 9
N_HERO_CHOICES = 4
# 13..20 vorpal species
IDX_VORPAL_BASE = 13
N_VORPAL_SPECIES = 8
IDX_REVEAL = 21
IDX_AXE = 22
IDX_DECLINE_AXE = 23
IDX_POLY = 24
IDX_DECLINE_POLY = 25

N_ACTIONS = 26
MASK_DTYPE = np.float32

_ADVENTURER_LIST = (
    AdventurerKind.WARRIOR,
    AdventurerKind.BARBARIAN,
    AdventurerKind.MAGE,
    AdventurerKind.ROGUE,
)


def decode_index(m: Match, i: int) -> A.Action | None:
    if not (0 <= i < N_ACTIONS):
        return None
    if i == IDX_PASS:
        return A.PassBid()
    if i == IDX_DRAW:
        return A.DrawCard()
    if i == IDX_ADD:
        return A.AddToDungeon()
    if IDX_SACRIFICE_BASE <= i < IDX_SACRIFICE_BASE + N_SACRIFICE_SLOTS:
        k = i - IDX_SACRIFICE_BASE
        ids = all_equipment_ids(m.hero)
        if k >= len(ids):
            return None
        return A.SacrificeEquipment(ids[k])
    if IDX_PICK_HERO_BASE <= i < IDX_PICK_HERO_BASE + N_HERO_CHOICES:
        hi = i - IDX_PICK_HERO_BASE
        return A.ChooseNextAdventurer(_ADVENTURER_LIST[hi])
    if IDX_VORPAL_BASE <= i < IDX_VORPAL_BASE + N_VORPAL_SPECIES:
        si = i - IDX_VORPAL_BASE
        return A.DeclareVorpal(ALL_SPECIES[si])
    if i == IDX_REVEAL:
        return A.RevealOrContinue()
    if i == IDX_AXE:
        return A.UseFireAxe()
    if i == IDX_DECLINE_AXE:
        return A.DeclineFireAxe()
    if i == IDX_POLY:
        return A.UsePolymorph()
    if i == IDX_DECLINE_POLY:
        return A.DeclinePolymorph()
    return None


def _index_for_choose(hero: AdventurerKind) -> int:
    return IDX_PICK_HERO_BASE + int(hero)


def _index_for_sacrifice(eid: str, m: Match) -> int:
    ids = all_equipment_ids(m.hero)
    try:
        k = ids.index(eid)
    except ValueError:
        return -1
    return IDX_SACRIFICE_BASE + k


def encode_action(m: Match, a: A.Action) -> int:
    if isinstance(a, A.PassBid):
        return IDX_PASS
    if isinstance(a, A.DrawCard):
        return IDX_DRAW
    if isinstance(a, A.AddToDungeon):
        return IDX_ADD
    if isinstance(a, A.SacrificeEquipment):
        idx = _index_for_sacrifice(a.equipment_id, m)
        if idx < 0:
            msg = f"sacrifice id not in hero loadout: {a.equipment_id!r}"
            raise ValueError(msg)
        return idx
    if isinstance(a, A.ChooseNextAdventurer):
        return _index_for_choose(a.hero)
    if isinstance(a, A.DeclareVorpal):
        si = ALL_SPECIES.index(a.target_species)
        return IDX_VORPAL_BASE + si
    if isinstance(a, A.RevealOrContinue):
        return IDX_REVEAL
    if isinstance(a, A.UseFireAxe):
        return IDX_AXE
    if isinstance(a, A.DeclineFireAxe):
        return IDX_DECLINE_AXE
    if isinstance(a, A.UsePolymorph):
        return IDX_POLY
    if isinstance(a, A.DeclinePolymorph):
        return IDX_DECLINE_POLY
    msg = f"unencodable action: {a!r}"
    raise TypeError(msg)


def legal_mask(m: Match) -> np.ndarray:
    out = np.zeros((N_ACTIONS,), dtype=MASK_DTYPE)
    if m.phase is MatchPhase.ENDED:
        return out
    legal = m.legal_actions()
    for a in legal:
        try:
            i = encode_action(m, a)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if 0 <= i < N_ACTIONS:
            out[i] = 1.0
    return out


def assert_mask_matches_legal(m: Match) -> None:
    """Debug helper: full mask should match legality of decode."""
    mask = legal_mask(m)
    legal = m.legal_actions()
    for i in range(N_ACTIONS):
        dec = decode_index(m, i)
        ok = dec is not None and m.phase is not MatchPhase.ENDED
        in_legal = ok and (dec in legal)  # type: ignore[operator]
        if mask[i] > 0.5 and not in_legal:
            msg = f"mask 1 at {i} but {dec!r} not in legal {legal!r}, phase {m.phase}"
            raise AssertionError(msg)
        if in_legal and mask[i] < 0.5:
            msg = f"legal action {dec!r} at index {i} not in mask, phase {m.phase}"
            raise AssertionError(msg)
