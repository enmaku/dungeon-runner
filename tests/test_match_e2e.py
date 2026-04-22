import random

import dungeon_runner.actions as A
from dungeon_runner.match import DungeonSub, Match, MatchPhase
from dungeon_runner.types_core import AdventurerKind, Species


def test_two_pass_empty_pile_succeeds():
    m = Match.new(2, random.Random(0), AdventurerKind.WARRIOR, 0)
    m.apply(A.PassBid())
    assert m.phase is MatchPhase.DUNGEON
    r = m.runner_seat
    assert r is not None
    if m.dungeon_sub is DungeonSub.VORPAL:
        m.apply(A.DeclareVorpal(Species.DRAGON))
    m.apply(A.RevealOrContinue())
    assert m.phase is MatchPhase.PICK_ADVENTURER
    assert m.players[r].success_cards == 1
