import random

import dungeon_runner.actions as A
from dungeon_runner.match import Match, MatchPhase, MatchTerminalReason
from dungeon_runner.types_core import AdventurerKind


def test_one_pass_sole_bidder_empty_pile_forfeits_match():
    m = Match.new(2, random.Random(0), AdventurerKind.WARRIOR, 0)
    m.apply(A.PassBid())
    assert m.phase is MatchPhase.ENDED
    assert m.winner_seat is None
    assert m.terminal_reason is MatchTerminalReason.EMPTY_DUNGEON_FORFEIT
