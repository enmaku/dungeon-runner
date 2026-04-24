"""Action codec: legal mask and encode/decode for fixed states."""

from __future__ import annotations

import random

import dungeon_runner.actions as A
from dungeon_runner.rl import actions_codec as ac
from dungeon_runner.match import Match, MatchPhase
from dungeon_runner.types_core import AdventurerKind

pyr0 = random.Random(42)


def test_choose_hero() -> None:
    m = Match.new(2, pyr0, AdventurerKind.WARRIOR, 0)
    m.phase = MatchPhase.PICK_ADVENTURER
    m.active_seat = 0
    m.pick_next_seat = 0
    a = A.ChooseNextAdventurer(AdventurerKind.MAGE)
    i = ac.encode_action(m, a)
    o = ac.decode_index(m, i)
    assert o == a
    m.apply(a)


def test_legal_mask_matches_legal() -> None:
    m = Match.new(3, random.Random(1), AdventurerKind.WARRIOR, 0)
    for _ in range(80):
        if m.phase is MatchPhase.ENDED:
            break
        legal = m.legal_actions()
        mask = ac.legal_mask(m)
        for i in range(ac.N_ACTIONS):
            a = ac.decode_index(m, i)
            msk = bool(float(mask[i]) > 0.5)
            ok = a in legal
            if msk != ok:  # noqa: SIM201
                bs = getattr(m, "bidding_sub", None)  # noqa: SIM201
                raise AssertionError(  # noqa: SIM201
                    f"idx {i} dec {a!r} legal {ok} mask {msk} p {m.phase!r} sub {bs!r}"  # noqa: SIM201
                )
        m.apply(pyr0.choice(list(legal)))


def test_silence() -> None:
    assert ac.N_ACTIONS == 26
