"""End-to-end scripted games: bidding, equipment, dungeon outcomes, match termination."""

import random

import pytest

import dungeon_runner.actions as A
from dungeon_runner.catalog import EQUIP_POLY, all_equipment_ids
from dungeon_runner.errors import IllegalAction
from dungeon_runner.match import BiddingState, DungeonSub, Match, MatchPhase, MatchTerminalReason
from dungeon_runner.types_core import AdventurerKind, Species

from conftest import deck_in_draw_order, dungeon_only, make_monster, play_dungeon, play_pick


def test_sacrifice_removes_equipment_monster_not_on_pile():
    rng = random.Random(42)
    deck = deck_in_draw_order(Species.GOBLIN)
    m = Match.new(2, rng, AdventurerKind.WARRIOR, start_seat=0, monster_deck=deck)
    assert "W_PLATE" in m.center_equipment
    m.apply(A.DrawCard())
    assert m.bidding_sub is BiddingState.PENDING
    assert m.pending_card is not None
    m.apply(A.SacrificeEquipment("W_PLATE"))
    assert "W_PLATE" not in m.center_equipment
    assert len(m.sacrifice_rows) == 1
    assert m.sacrifice_rows[0].equipment_id == "W_PLATE"
    assert m.sacrifice_rows[0].seat == 0
    assert len(m.dungeon_pile) == 0
    m.apply(A.PassBid())
    assert m.phase is MatchPhase.DUNGEON
    assert m.runner_seat == 0
    assert "W_PLATE" not in m.d_in_play


def test_draw_and_add_puts_monster_on_pile_and_tracks_own_adds():
    rng = random.Random(1)
    deck = deck_in_draw_order(Species.SKELETON, Species.GOBLIN)
    m = Match.new(2, rng, AdventurerKind.BARBARIAN, start_seat=0, monster_deck=deck)
    m.apply(A.DrawCard())
    m.apply(A.AddToDungeon())
    assert len(m.dungeon_pile) == 1
    assert m.dungeon_pile[0].species is Species.SKELETON
    assert m.players[0].own_pile_adds == [Species.SKELETON]
    m.apply(A.PassBid())
    assert m.runner_seat == 0
    assert len(m.d_remaining) == 1


def test_healing_potion_explicit_hp_after_revive():
    m = dungeon_only(AdventurerKind.BARBARIAN, [make_monster(Species.ORC, 0)], ["B_HEAL", "B_CHAIN"])
    m.d_ad_base = 4
    m.d_hp = 3
    m.d_axe_spent = True
    m.d_poly_spent = True
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_hp == 4
    assert "B_HEAL" not in m.d_in_play


def test_warrior_holy_grail_defeats_even_strength():
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.VAMPIRE, 0)], ["W_HOLY", "W_TORCH"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_current is None
    assert m.phase is MatchPhase.PICK_ADVENTURER


def test_vorpal_first_goblin_second_defeated_by_torch():
    g1, g2 = make_monster(Species.GOBLIN, 0), make_monster(Species.GOBLIN, 1)
    m = dungeon_only(AdventurerKind.WARRIOR, [g1, g2], None)
    m.apply(A.DeclareVorpal(Species.GOBLIN))
    m.apply(A.RevealOrContinue())
    assert "W_VORPAL" not in m.d_in_play
    m.apply(A.RevealOrContinue())
    assert len(m.d_discard_run) == 2
    assert m.phase is MatchPhase.PICK_ADVENTURER


def test_polymorph_replaces_dragon_with_next_card():
    m = dungeon_only(
        AdventurerKind.MAGE,
        [make_monster(Species.DRAGON, 0), make_monster(Species.GOBLIN, 1)],
        list(all_equipment_ids(AdventurerKind.MAGE)),
    )
    m.d_hp = 30
    m.d_poly_spent = False
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.dungeon_sub is DungeonSub.PICK_POLYMORPH
    m.apply(A.UsePolymorph())
    assert EQUIP_POLY not in m.d_in_play
    assert m.phase is MatchPhase.PICK_ADVENTURER
    assert any(c.species is Species.GOBLIN for c in m.d_discard_run)


def test_omnipotence_fails_when_sacrifice_duplicates_species():
    m = dungeon_only(
        AdventurerKind.MAGE,
        [make_monster(Species.GOBLIN, 0)],
        ["M_OMNI", "M_BRACE"],
    )
    from dungeon_runner.types_core import SacrificeSetaside

    g2 = make_monster(Species.GOBLIN, 2)
    m.sacrifice_rows.append(SacrificeSetaside(g2, "M_WALL", 1))
    m.d_hp = 1
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.phase is MatchPhase.PICK_ADVENTURER
    assert m.players[0].aid_flips == 1


def test_three_player_two_dungeon_successes_end_match():
    """First dungeon has an empty pile (two passes before sole runner); that still awards a Success per rules."""
    rng = random.Random(0)
    deck = deck_in_draw_order(Species.GOBLIN)
    m = Match.new(3, rng, AdventurerKind.WARRIOR, start_seat=0, monster_deck=deck)

    m.apply(A.PassBid())
    m.apply(A.PassBid())
    assert m.phase is MatchPhase.DUNGEON
    assert m.runner_seat == 2
    play_dungeon(m)
    assert m.phase is MatchPhase.PICK_ADVENTURER
    assert m.players[2].success_cards == 1

    play_pick(m, AdventurerKind.WARRIOR)
    m.apply(A.DrawCard())
    m.apply(A.AddToDungeon())
    m.apply(A.PassBid())
    m.apply(A.PassBid())
    assert m.runner_seat == 2
    play_dungeon(m)
    assert m.phase is MatchPhase.ENDED
    assert m.terminal_reason is MatchTerminalReason.SECOND_SUCCESS
    assert m.winner_seat == 2


def test_two_dragon_dungeons_eliminate_runner():
    """Clears runner equipment and HP after bidding to force two lethal fails; not a full-fidelity run."""
    rng = random.Random(7)
    deck = deck_in_draw_order(Species.DRAGON)
    m = Match.new(3, rng, AdventurerKind.BARBARIAN, start_seat=2, monster_deck=deck)
    m.apply(A.DrawCard())
    m.apply(A.AddToDungeon())
    m.apply(A.PassBid())
    m.apply(A.PassBid())
    assert m.runner_seat == 2
    assert len(m.dungeon_pile) == 1
    m.d_in_play = set()
    m.d_hp = 1
    m.d_remaining = list(m.dungeon_pile)
    m.dungeon_pile.clear()
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.dungeon_sub = DungeonSub.REVEAL
    while m.phase is MatchPhase.DUNGEON and m.legal_actions():
        m.apply(list(m.legal_actions())[0])
    assert m.players[2].aid_flips == 1
    assert not m.players[2].eliminated

    play_pick(m, AdventurerKind.BARBARIAN)
    m.monster_deck = deck_in_draw_order(Species.DRAGON)
    m.apply(A.DrawCard())
    m.apply(A.AddToDungeon())
    m.apply(A.PassBid())
    m.apply(A.PassBid())
    assert m.runner_seat == 2
    m.d_in_play = set()
    m.d_hp = 1
    m.d_remaining = list(m.dungeon_pile)
    m.dungeon_pile.clear()
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.dungeon_sub = DungeonSub.REVEAL
    while m.phase is MatchPhase.DUNGEON and m.legal_actions():
        m.apply(list(m.legal_actions())[0])
    assert m.players[2].eliminated


def test_illegal_sacrifice_wrong_equipment():
    m = Match.new(2, random.Random(0), AdventurerKind.WARRIOR, 0, monster_deck=deck_in_draw_order(Species.GOBLIN))
    m.apply(A.DrawCard())
    with pytest.raises(IllegalAction):
        m.apply(A.SacrificeEquipment("B_AXE"))


def test_demonic_pact_pair_consumes_pact():
    m = dungeon_only(
        AdventurerKind.MAGE,
        [make_monster(Species.DEMON, 0), make_monster(Species.LICH, 1)],
        list(all_equipment_ids(AdventurerKind.MAGE)),
    )
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert "M_PACT" not in m.d_in_play
    assert m.phase is MatchPhase.PICK_ADVENTURER
