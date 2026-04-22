import random

import pytest

import dungeon_runner.actions as A
from dungeon_runner.errors import IllegalAction
from dungeon_runner.match import BiddingState, Match, MatchPhase
from dungeon_runner.types_core import AdventurerKind, Species

from conftest import dungeon_only, make_monster


def test_wrong_equipment_sacrifice_raises():
    m = Match.new(2, random.Random(0), AdventurerKind.WARRIOR, 0)
    m.bidding_sub = BiddingState.PENDING
    m.pending_card = make_monster(Species.GOBLIN, 0)
    with pytest.raises(IllegalAction):
        m.apply(A.SacrificeEquipment("B_AXE"))


def test_legal_bidding_smoke():
    m = Match.new(2, random.Random(2), AdventurerKind.WARRIOR, 0)
    for _ in range(40):
        la = list(m.legal_actions())
        if not la:
            break
        m.apply(la[0])
        if m.phase is not MatchPhase.BIDDING:
            break


def test_dungeon_legal_warrior_goblin_chain():
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.GOBLIN, 0)], None)
    m.apply(A.DeclareVorpal(Species.GOBLIN))
    m.apply(A.RevealOrContinue())
    assert m.phase in (MatchPhase.DUNGEON, MatchPhase.PICK_ADVENTURER)
