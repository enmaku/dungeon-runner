import random

import pytest

import dungeon_runner.actions as A
from dungeon_runner.errors import IllegalAction
from dungeon_runner.match import BiddingState, Match, MatchPhase, MatchTerminalReason
from dungeon_runner.types_core import AdventurerKind


def test_empty_deck_cannot_draw():
    m = Match.new(2, random.Random(0), AdventurerKind.WARRIOR, 0)
    m.monster_deck = []
    la = m.legal_actions()
    assert la == {A.PassBid()}
    with pytest.raises(IllegalAction):
        m.apply(A.DrawCard())


def test_no_equipment_only_add_after_draw():
    m = Match.new(2, random.Random(0), AdventurerKind.WARRIOR, 0)
    m.center_equipment = set()
    c = m.monster_deck.pop(0)
    m.pending_card = c
    m.bidding_sub = BiddingState.PENDING
    assert m.legal_actions() == {A.AddToDungeon()}


def test_two_player_one_pass_sole_bidder_empty_pile_forfeits():
    m = Match.new(2, random.Random(0), AdventurerKind.WARRIOR, 0)
    m.apply(A.PassBid())
    assert m.phase is MatchPhase.ENDED
    assert m.winner_seat is None
    assert m.terminal_reason is MatchTerminalReason.EMPTY_DUNGEON_FORFEIT
    assert m.runner_seat == 1


def test_empty_deck_bidding_with_no_pile_ends_house_stale():
    m = Match.new(2, random.Random(0), AdventurerKind.WARRIOR, 0)
    m.monster_deck = []
    m.apply(A.PassBid())
    assert m.phase is MatchPhase.ENDED
    assert m.winner_seat is None
    assert m.terminal_reason is MatchTerminalReason.BIDDING_EMPTY_STALE
