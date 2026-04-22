import random

import dungeon_runner.actions as A
from dungeon_runner.catalog import EQUIP_PACT, EQUIP_POLY, all_equipment_ids
from dungeon_runner.match import BiddingState, DungeonSub, Match, MatchPhase
from dungeon_runner.types_core import AdventurerKind, Species

from conftest import dungeon_only, make_monster


def test_vorpal_removes_tile_on_named_hit():
    m = dungeon_only(
        AdventurerKind.WARRIOR,
        [make_monster(Species.GOBLIN, 0), make_monster(Species.DRAGON, 1)],
        None,
    )
    m.apply(A.DeclareVorpal(Species.GOBLIN))
    m.apply(A.RevealOrContinue())
    assert "W_VORPAL" not in m.d_in_play


def test_rogue_ring_defeats_skeleton_and_heals():
    m = dungeon_only(AdventurerKind.ROGUE, [make_monster(Species.SKELETON, 0)], None)
    m.d_hp = 3
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_current is None
    assert m.d_hp == 5


def test_demonic_pact_solo_demon():
    m = dungeon_only(AdventurerKind.MAGE, [make_monster(Species.DEMON, 0)], None)
    m.d_poly_spent = EQUIP_POLY not in m.d_in_play
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert EQUIP_PACT not in m.d_in_play
    assert m.phase is MatchPhase.PICK_ADVENTURER


def test_omnipotence_saves_lethal_with_distinct_bag():
    """Simulates Polymorph already gone (e.g. sacrificed in bidding); remaining kit is legal."""
    eq = list(all_equipment_ids(AdventurerKind.MAGE))
    m = dungeon_only(AdventurerKind.MAGE, [make_monster(Species.GOBLIN, 0), make_monster(Species.SKELETON, 1)], eq)
    m.d_in_play = {e for e in m.d_in_play if e != EQUIP_POLY}
    m.d_poly_spent = True
    m.d_hp = 1
    m.dungeon_sub = DungeonSub.REVEAL
    m.d_axe_spent = True
    m.apply(A.RevealOrContinue())
    assert m.phase is MatchPhase.PICK_ADVENTURER


def test_barbarian_fire_axe_kills_reveal():
    m = dungeon_only(AdventurerKind.BARBARIAN, [make_monster(Species.DRAGON, 0)], None)
    m.dungeon_sub = DungeonSub.REVEAL
    m.d_poly_spent = True
    m.apply(A.RevealOrContinue())
    assert m.dungeon_sub is DungeonSub.PICK_FIRE_AXE
    m.apply(A.UseFireAxe())
    assert m.phase is MatchPhase.PICK_ADVENTURER


def test_legal_bidding_add_only_no_equipment():
    g = Match.new(2, random.Random(0), AdventurerKind.WARRIOR, 0)
    g.bidding_sub = BiddingState.PENDING
    g.pending_card = make_monster(Species.GOBLIN, 0)
    g.center_equipment = set()
    assert g.legal_actions() == {A.AddToDungeon()}
