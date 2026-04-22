import random
from typing import Collection

import dungeon_runner.actions as A
from dungeon_runner.catalog import all_equipment_ids, default_monster_deck_list, hp_for_equip, make_deck_instance_ids
from dungeon_runner.match import DungeonSub, Match, MatchPhase
from dungeon_runner.types_core import AdventurerKind, MonsterInstance, Species, base_hp


def deck_in_draw_order(*species: Species) -> list[MonsterInstance]:
    """Build a 13-card deck: first species args in order, then fill from default deck."""
    want = list(species)
    pool = make_deck_instance_ids(default_monster_deck_list(), start_id=0)
    by_sp: dict[Species, list[MonsterInstance]] = {}
    for c in pool:
        by_sp.setdefault(c.species, []).append(c)
    out: list[MonsterInstance] = []
    used: set[int] = set()
    for sp in want:
        stack = by_sp.get(sp, [])
        if not stack:
            raise ValueError(f"no card for {sp}")
        c = stack.pop(0)
        out.append(c)
        used.add(c.def_id)
    for c in pool:
        if c.def_id not in used:
            out.append(c)
            used.add(c.def_id)
    assert len(out) == 13
    return out


def _action_priority(a: object) -> tuple[int, str]:
    if isinstance(a, A.DeclareVorpal):
        return (0, a.target_species.value)
    if isinstance(a, A.RevealOrContinue):
        return (1, "")
    if isinstance(a, A.UseFireAxe):
        return (2, "")
    if isinstance(a, A.UsePolymorph):
        return (2, "")
    if isinstance(a, A.DeclineFireAxe):
        return (3, "")
    if isinstance(a, A.DeclinePolymorph):
        return (3, "")
    return (9, str(a))


def play_dungeon(m: Match) -> None:
    """Apply legal dungeon actions until leaving DUNGEON phase (stable priority per step)."""
    while m.phase is MatchPhase.DUNGEON:
        acts = m.legal_actions()
        if not acts:
            break
        m.apply(sorted(acts, key=_action_priority)[0])


def play_pick(m: Match, hero: AdventurerKind) -> None:
    if m.phase is MatchPhase.PICK_ADVENTURER:
        m.apply(A.ChooseNextAdventurer(hero))


def make_monster(species: Species, i: int = 0) -> MonsterInstance:
    from dungeon_runner.catalog import SPECIES_DATA

    st, icons = SPECIES_DATA[species]
    return MonsterInstance(species=species, strength=st, icons=icons, def_id=1000 + i)


def dungeon_only(
    hero: AdventurerKind,
    pile: list[MonsterInstance],
    equipment: Collection[str] | None = None,
) -> Match:
    m = Match.new(2, random.Random(0), hero, 0)
    m.phase = MatchPhase.DUNGEON
    m.active_seat = 0
    m.runner_seat = 0
    m.d_remaining = list(pile)
    eq = set(equipment) if equipment is not None else set(all_equipment_ids(hero))
    m.d_in_play = set(eq)
    m.d_ad_base = base_hp(hero)
    m.d_hp = m.d_ad_base + sum(hp_for_equip(e) for e in m.d_in_play)
    m.d_discard_run = []
    m.d_current = None
    m.d_vorpal_target = None
    from dungeon_runner.catalog import EQUIP_POLY, EQUIP_FIRE_AXE

    m.d_poly_spent = EQUIP_POLY not in m.d_in_play
    m.d_axe_spent = EQUIP_FIRE_AXE not in m.d_in_play
    m.d_heal_used = set()
    m.sacrifice_rows = []
    from dungeon_runner.catalog import EQUIP_VORPAL_IDS

    if m.d_in_play & EQUIP_VORPAL_IDS:
        m.dungeon_sub = DungeonSub.VORPAL
    else:
        m.dungeon_sub = DungeonSub.REVEAL
    return m
