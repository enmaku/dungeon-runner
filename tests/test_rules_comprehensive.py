"""Rule-by-rule coverage aligned with docs/welcome-to-the-dungeon.md (physical + spec)."""

import random

import pytest

import dungeon_runner.actions as A
from dungeon_runner.catalog import (
    EQUIP_FIRE_AXE,
    EQUIP_PACT,
    EQUIP_POLY,
    EQUIP_VORPAL_IDS,
    all_equipment_ids,
)
from dungeon_runner.errors import IllegalAction
from dungeon_runner.match import BiddingState, DungeonSub, Match, MatchPhase, MatchTerminalReason
from dungeon_runner.types_core import AdventurerKind, SacrificeSetaside, Species

from conftest import deck_in_draw_order, dungeon_only, make_monster, play_dungeon, play_pick


# --- Torch (Warrior / Barbarian): strength ≤ 3 ---


def test_warrior_torch_banishes_goblin():
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.GOBLIN, 0)], ["W_TORCH", "W_PLATE"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_current is None
    assert m.d_hp == hp0


def test_warrior_torch_banishes_orc_at_three():
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.ORC, 0)], ["W_TORCH", "W_PLATE"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_current is None


def test_warrior_torch_does_not_banish_vampire_four():
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.VAMPIRE, 0)], ["W_TORCH", "W_PLATE"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_hp == hp0 - 4


def test_warrior_even_skeleton_with_grail_and_torch_defeated_both_tiles_stay_in_play():
    m = dungeon_only(
        AdventurerKind.WARRIOR,
        [make_monster(Species.SKELETON, 0)],
        ["W_HOLY", "W_TORCH", "W_PLATE"],
    )
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_current is None
    assert m.d_hp == hp0
    assert "W_HOLY" in m.d_in_play and "W_TORCH" in m.d_in_play


def test_barbarian_torch_does_not_banish_vampire_four():
    m = dungeon_only(AdventurerKind.BARBARIAN, [make_monster(Species.VAMPIRE, 0)], ["B_TORCH", "B_CHAIN"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_hp == hp0 - 4


def test_barbarian_torch_banishes_skeleton():
    m = dungeon_only(AdventurerKind.BARBARIAN, [make_monster(Species.SKELETON, 0)], ["B_TORCH", "B_CHAIN"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_current is None


# --- Holy Grail (even strength): Warrior / Mage ---


def test_warrior_holy_grail_does_not_banish_odd_orc():
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.ORC, 0)], ["W_HOLY", "W_PLATE"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_hp == hp0 - 3


def test_mage_holy_grail_banishes_skeleton_two():
    m = dungeon_only(AdventurerKind.MAGE, [make_monster(Species.SKELETON, 0)], ["M_HOLY", "M_BRACE"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_current is None


def test_mage_holy_grail_odd_strength_takes_damage():
    m = dungeon_only(AdventurerKind.MAGE, [make_monster(Species.ORC, 0)], ["M_HOLY", "M_BRACE", "M_WALL"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_hp == hp0 - 3


# --- Dragon Spear / War Hammer ---


def test_warrior_spear_banishes_dragon_only():
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.DRAGON, 0)], ["W_SPEAR", "W_PLATE"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_current is None


def test_warrior_spear_does_not_banish_non_dragon():
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.GOLEM, 0)], ["W_SPEAR", "W_PLATE"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_hp == hp0 - 5


def test_warrior_dragon_defeated_by_spear_torch_tile_unused_because_strength_gt_three():
    """Dragon (9) is outside Torch range (≤3); only Spear defeats it. Both tiles stay."""
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.DRAGON, 0)], ["W_SPEAR", "W_TORCH", "W_PLATE"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_current is None
    assert "W_SPEAR" in m.d_in_play and "W_TORCH" in m.d_in_play


def test_barbarian_hammer_banishes_golem():
    m = dungeon_only(AdventurerKind.BARBARIAN, [make_monster(Species.GOLEM, 0)], ["B_HAMMER", "B_CHAIN"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_current is None


def test_barbarian_hammer_ignores_orc():
    m = dungeon_only(AdventurerKind.BARBARIAN, [make_monster(Species.ORC, 0)], ["B_HAMMER", "B_CHAIN"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_hp == hp0 - 3


# --- Rogue: Ring (≤2 heal) / Cloak (≥6) ---


def test_rogue_ring_heals_on_goblin():
    m = dungeon_only(AdventurerKind.ROGUE, [make_monster(Species.GOBLIN, 0)], ["R_RING", "R_ARMOR"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_hp == hp0 + 1


def test_rogue_ring_does_not_apply_to_orc_three():
    m = dungeon_only(AdventurerKind.ROGUE, [make_monster(Species.ORC, 0)], ["R_RING", "R_ARMOR"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_hp == hp0 - 3


def test_rogue_cloak_banishes_lich_six():
    m = dungeon_only(AdventurerKind.ROGUE, [make_monster(Species.LICH, 0)], ["R_CLOAK", "R_ARMOR"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_current is None


def test_rogue_cloak_does_not_banish_golem_five():
    m = dungeon_only(AdventurerKind.ROGUE, [make_monster(Species.GOLEM, 0)], ["R_CLOAK", "R_ARMOR"])
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_hp == hp0 - 5


# --- Vorpal: before other auto-defeats; tile spent; unused if species absent ---


def test_vorpal_resolves_before_torch_would_same_card():
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.GOBLIN, 0)], None)
    m.apply(A.DeclareVorpal(Species.GOBLIN))
    m.apply(A.RevealOrContinue())
    assert "W_VORPAL" not in m.d_in_play
    assert "W_TORCH" in m.d_in_play


def test_vorpal_resolves_before_ring_heal_on_rogue_skeleton():
    m = dungeon_only(AdventurerKind.ROGUE, [make_monster(Species.SKELETON, 0)], None)
    hp_before = m.d_hp
    m.apply(A.DeclareVorpal(Species.SKELETON))
    m.apply(A.RevealOrContinue())
    assert "R_VORP" not in m.d_in_play
    assert m.d_hp == hp_before


def test_vorpal_unused_when_named_species_never_revealed():
    m = dungeon_only(
        AdventurerKind.WARRIOR,
        [make_monster(Species.ORC, 0)],
        None,
    )
    m.apply(A.DeclareVorpal(Species.GOBLIN))
    m.apply(A.RevealOrContinue())
    assert "W_VORPAL" in m.d_in_play
    assert m.d_current is None


def test_warrior_without_vorpal_equipment_starts_in_reveal_not_vorpal_subphase():
    eq = [e for e in all_equipment_ids(AdventurerKind.WARRIOR) if e != "W_VORPAL"]
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.GOBLIN, 0)], eq)
    assert m.dungeon_sub is DungeonSub.REVEAL
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.apply(A.RevealOrContinue())
    assert m.d_current is None


# --- Demonic Pact ---


def test_demonic_pact_second_card_skips_normal_resolution():
    m = dungeon_only(
        AdventurerKind.MAGE,
        [make_monster(Species.DEMON, 0), make_monster(Species.DRAGON, 1)],
        list(all_equipment_ids(AdventurerKind.MAGE)),
    )
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_hp == hp0
    assert len(m.d_discard_run) == 2
    assert {c.species for c in m.d_discard_run} == {Species.DEMON, Species.DRAGON}


# --- Fire Axe / Polymorph choices ---


def test_barbarian_decline_fire_axe_then_takes_damage():
    m = dungeon_only(AdventurerKind.BARBARIAN, [make_monster(Species.VAMPIRE, 0)], ["B_AXE", "B_CHAIN"])
    m.d_poly_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.dungeon_sub is DungeonSub.PICK_FIRE_AXE
    m.apply(A.DeclineFireAxe())
    assert m.d_hp == hp0 - 4
    assert "B_AXE" in m.d_in_play


def test_mage_decline_polymorph_takes_damage():
    m = dungeon_only(
        AdventurerKind.MAGE,
        [make_monster(Species.DRAGON, 0), make_monster(Species.SKELETON, 1)],
        ["M_POLY", "M_BRACE", "M_WALL", "M_HOLY", "M_OMNI", "M_PACT"],
    )
    m.d_poly_spent = False
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    hp0 = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.dungeon_sub is DungeonSub.PICK_POLYMORPH
    m.apply(A.DeclinePolymorph())
    assert m.d_hp == hp0 - 9
    m.apply(A.RevealOrContinue())
    assert m.d_current is None
    assert m.phase is MatchPhase.PICK_ADVENTURER


def test_mage_single_card_dragon_no_polymorph_branch_poly_stays_unused():
    m = dungeon_only(
        AdventurerKind.MAGE,
        [make_monster(Species.DRAGON, 0)],
        ["M_POLY", "M_WALL", "M_BRACE"],
    )
    m.d_poly_spent = False
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.dungeon_sub is not DungeonSub.PICK_POLYMORPH
    assert EQUIP_POLY in m.d_in_play


def test_illegal_polymorph_when_not_in_poly_step():
    m = dungeon_only(AdventurerKind.MAGE, [make_monster(Species.DRAGON, 0)], list(all_equipment_ids(AdventurerKind.MAGE)))
    m.dungeon_sub = DungeonSub.REVEAL
    with pytest.raises(IllegalAction):
        m.apply(A.UsePolymorph())


def test_illegal_use_fire_axe_outside_axe_step():
    m = dungeon_only(
        AdventurerKind.BARBARIAN,
        [make_monster(Species.VAMPIRE, 0)],
        ["B_AXE", "B_CHAIN"],
    )
    m.d_poly_spent = True
    m.d_axe_spent = False
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    with pytest.raises(IllegalAction):
        m.apply(A.UseFireAxe())


# --- Healing Potion (once per dungeon) ---


def test_healing_potion_only_once_then_lethal_fails():
    m = dungeon_only(
        AdventurerKind.BARBARIAN,
        [make_monster(Species.ORC, 0), make_monster(Species.DRAGON, 1)],
        ["B_HEAL", "B_CHAIN"],
    )
    m.d_ad_base = 4
    m.d_hp = 1
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_hp == 4
    assert "B_HEAL" not in m.d_in_play
    m.apply(A.RevealOrContinue())
    assert m.players[0].aid_flips == 1


def test_rogue_healing_potion_revives_to_base_only():
    m = dungeon_only(
        AdventurerKind.ROGUE,
        [make_monster(Species.DRAGON, 0)],
        ["R_HEAL", "R_ARMOR", "R_BUCK"],
    )
    m.d_ad_base = 3
    m.d_hp = 2
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.d_hp == 3
    assert "R_HEAL" not in m.d_in_play


# --- Bidding / match flow ---


def test_three_player_two_passes_empty_pile_forfeits():
    rng = random.Random(0)
    deck = deck_in_draw_order(Species.GOBLIN, Species.ORC)
    m = Match.new(3, rng, AdventurerKind.WARRIOR, start_seat=0, monster_deck=deck)
    m.apply(A.PassBid())
    m.apply(A.PassBid())
    assert m.phase is MatchPhase.ENDED
    assert m.winner_seat is None
    assert m.terminal_reason is MatchTerminalReason.EMPTY_DUNGEON_FORFEIT
    assert m.runner_seat == 2


def test_bidding_sacrifice_removes_equipment_then_empty_pile_forfeits():
    rng = random.Random(0)
    deck = deck_in_draw_order(Species.GOBLIN, Species.ORC)
    m = Match.new(2, rng, AdventurerKind.WARRIOR, start_seat=0, monster_deck=deck)
    m.apply(A.DrawCard())
    m.apply(A.SacrificeEquipment("W_PLATE"))
    m.apply(A.PassBid())
    assert m.phase is MatchPhase.ENDED
    assert m.terminal_reason is MatchTerminalReason.EMPTY_DUNGEON_FORFEIT
    assert "W_PLATE" not in m.center_equipment


def test_draw_then_pass_advances_active_seat():
    rng = random.Random(1)
    m = Match.new(3, rng, AdventurerKind.WARRIOR, start_seat=0, monster_deck=deck_in_draw_order(Species.GOBLIN))
    s0 = m.active_seat
    m.apply(A.DrawCard())
    m.apply(A.AddToDungeon())
    assert m.active_seat != s0


def test_two_player_last_standing_when_opponent_double_fail():
    """Harness: after legal bidding, clears d_in_play and sets HP=1 to force lethal vs dragon.

    Asserts elimination / LAST_STANDING only; not a claim that a real runner would enter naked.
    """
    rng = random.Random(99)
    deck = deck_in_draw_order(Species.DRAGON, Species.GOBLIN)
    m = Match.new(2, rng, AdventurerKind.BARBARIAN, start_seat=1, monster_deck=deck)
    m.apply(A.DrawCard())
    m.apply(A.AddToDungeon())
    m.apply(A.PassBid())
    assert m.phase is MatchPhase.DUNGEON
    assert m.runner_seat == 1
    assert len(m.d_remaining) == 1
    m.d_in_play = set()
    m.d_hp = 1
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.dungeon_sub = DungeonSub.REVEAL
    play_dungeon(m)
    assert m.players[1].aid_flips == 1

    play_pick(m, AdventurerKind.BARBARIAN)
    m.monster_deck = deck_in_draw_order(Species.DRAGON, Species.GOBLIN)
    m.apply(A.DrawCard())
    m.apply(A.AddToDungeon())
    m.apply(A.PassBid())
    assert m.runner_seat == 1
    m.d_in_play = set()
    m.d_hp = 1
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.dungeon_sub = DungeonSub.REVEAL
    play_dungeon(m)
    assert m.players[1].eliminated
    assert m.phase is MatchPhase.ENDED
    assert m.terminal_reason is MatchTerminalReason.LAST_STANDING
    assert m.winner_seat == 0


# --- Illegal / guard rails ---


def test_illegal_reveal_while_vorpal_choice_unresolved():
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.GOBLIN, 0)], None)
    assert m.dungeon_sub is DungeonSub.VORPAL
    with pytest.raises(IllegalAction):
        m.apply(A.RevealOrContinue())


def test_illegal_vorpal_declare_without_vorpal_subphase():
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.GOBLIN, 0)], ["W_TORCH", "W_PLATE"])
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    with pytest.raises(IllegalAction):
        m.apply(A.DeclareVorpal(Species.GOBLIN))


def test_fire_axe_tile_removed_after_use_second_threat_takes_hp():
    m = dungeon_only(
        AdventurerKind.BARBARIAN,
        [make_monster(Species.VAMPIRE, 0), make_monster(Species.VAMPIRE, 1)],
        ["B_AXE", "B_CHAIN", "B_SHIELD"],
    )
    m.d_poly_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    m.apply(A.UseFireAxe())
    assert EQUIP_FIRE_AXE not in m.d_in_play
    hp_mid = m.d_hp
    m.apply(A.RevealOrContinue())
    assert m.d_hp == hp_mid - 4


def test_vorpal_names_dragon_then_dragon_still_slain_by_vorpal_before_spear_order():
    """Vorpal is spent first; Spear never fires. Distinct from Grail/Torch (non-consuming overlap)."""
    m = dungeon_only(AdventurerKind.WARRIOR, [make_monster(Species.DRAGON, 0)], None)
    m.apply(A.DeclareVorpal(Species.DRAGON))
    m.apply(A.RevealOrContinue())
    assert not ({"W_VORPAL", "R_VORP"} & m.d_in_play)
    assert "W_SPEAR" in m.d_in_play


# --- Omnipotence naming ---


def test_omnipotence_saves_when_all_species_distinct_including_sacrifice_row():
    """Matches Omnipotence gather: in-play pile + discard + sacrifice rows; all species unique → win."""
    m = dungeon_only(
        AdventurerKind.MAGE,
        [make_monster(Species.ORC, 0)],
        ["M_OMNI", "M_BRACE"],
    )
    m.sacrifice_rows.append(SacrificeSetaside(make_monster(Species.VAMPIRE, 9), "M_WALL", 0))
    m.d_hp = 1
    m.d_poly_spent = True
    m.d_axe_spent = True
    m.d_vorpal_target = None
    m.dungeon_sub = DungeonSub.REVEAL
    m.apply(A.RevealOrContinue())
    assert m.phase is MatchPhase.PICK_ADVENTURER
    assert m.players[0].success_cards == 1
