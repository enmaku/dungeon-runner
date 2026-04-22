from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from dungeon_runner.types_core import (
    AdventurerKind,
    EffectKey,
    Icon,
    MonsterDef,
    MonsterInstance,
    Species,
)

_monster_defs: list[MonsterDef] = [
    MonsterDef(Species.GOBLIN, 1, frozenset({Icon.TORCH})),
    MonsterDef(Species.SKELETON, 2, frozenset({Icon.TORCH, Icon.CHALICE})),
    MonsterDef(Species.ORC, 3, frozenset({Icon.TORCH})),
    MonsterDef(Species.VAMPIRE, 4, frozenset({Icon.CHALICE})),
    MonsterDef(Species.GOLEM, 5, frozenset({Icon.HAMMER})),
    MonsterDef(Species.LICH, 6, frozenset({Icon.CHALICE, Icon.CLOAK})),
    MonsterDef(Species.DEMON, 7, frozenset({Icon.PACT, Icon.CLOAK})),
    MonsterDef(Species.DRAGON, 9, frozenset({Icon.STAFF, Icon.CLOAK})),
]

SPECIES_DATA: dict[Species, tuple[int, frozenset[Icon]]] = {
    m.species: (m.strength, m.icons) for m in _monster_defs
}


def default_monster_deck_list() -> list[MonsterDef]:
    out: list[MonsterDef] = []
    for m in _monster_defs:
        count = 2
        if m.species in (Species.LICH, Species.DEMON, Species.DRAGON):
            count = 1
        out.extend([m] * count)
    assert len(out) == 13, len(out)
    return out


def make_deck_instance_ids(defs: Sequence[MonsterDef], start_id: int = 0) -> list[MonsterInstance]:
    return [
        MonsterInstance(species=d.species, strength=d.strength, icons=d.icons, def_id=start_id + i)
        for i, d in enumerate(defs)
    ]


def shuffled_deck(
    rng: random.Random, defs: list[MonsterDef] | None = None, start_id: int = 0
) -> list[MonsterInstance]:
    if defs is None:
        defs = default_monster_deck_list()
    inst = make_deck_instance_ids(list(defs), start_id=start_id)
    rng.shuffle(inst)
    return inst


@dataclass(frozen=True, slots=True)
class HeroLoadout:
    kind: AdventurerKind
    equipment_ids: tuple[str, ...]
    effect_priority: tuple[EffectKey, ...]


ALL_EQUIP_DB: dict[str, tuple[str, int, bool, Icon | None]] = {
    "W_PLATE": ("Plate Armor", 5, False, None),
    "W_SHIELD": ("Knight Shield", 3, False, None),
    "W_VORPAL": ("Vorpal Sword", 0, True, None),
    "W_TORCH": ("Torch", 0, True, Icon.TORCH),
    "W_HOLY": ("Holy Grail", 0, True, Icon.CHALICE),
    "W_SPEAR": ("Dragon Spear", 0, True, Icon.STAFF),
    "B_HEAL": ("Healing Potion", 0, True, None),
    "B_SHIELD": ("Leather Shield", 3, False, None),
    "B_CHAIN": ("Chainmail", 4, False, None),
    "B_AXE": ("Fire Axe", 0, True, None),
    "B_TORCH": ("Torch", 0, True, Icon.TORCH),
    "B_HAMMER": ("War Hammer", 0, True, Icon.HAMMER),
    "M_WALL": ("Wall of Fire", 6, False, None),
    "M_HOLY": ("Holy Grail", 0, True, Icon.CHALICE),
    "M_OMNI": ("Omnipotence", 0, True, None),
    "M_BRACE": ("Bracelet of Protection", 3, False, None),
    "M_POLY": ("Polymorph", 0, True, None),
    "M_PACT": ("Demonic Pact", 0, True, Icon.PACT),
    "R_ARMOR": ("Mithril Armor", 5, False, None),
    "R_HEAL": ("Healing Potion", 0, True, None),
    "R_RING": ("Ring of Power", 0, True, None),
    "R_BUCK": ("Buckler", 3, False, None),
    "R_VORP": ("Vorpal Dagger", 0, True, None),
    "R_CLOAK": ("Invisibility Cloak", 0, True, Icon.CLOAK),
}

HERO_LOADOUT: dict[AdventurerKind, HeroLoadout] = {
    AdventurerKind.WARRIOR: HeroLoadout(
        AdventurerKind.WARRIOR,
        ("W_PLATE", "W_SHIELD", "W_VORPAL", "W_TORCH", "W_HOLY", "W_SPEAR"),
        (EffectKey.HOLY_GRAIL, EffectKey.STAFF_DRAGON, EffectKey.TORCH),
    ),
    AdventurerKind.BARBARIAN: HeroLoadout(
        AdventurerKind.BARBARIAN,
        ("B_HEAL", "B_SHIELD", "B_CHAIN", "B_AXE", "B_TORCH", "B_HAMMER"),
        (EffectKey.WAR_HAMMER_GOLEM, EffectKey.TORCH),
    ),
    AdventurerKind.MAGE: HeroLoadout(
        AdventurerKind.MAGE,
        ("M_WALL", "M_HOLY", "M_OMNI", "M_BRACE", "M_POLY", "M_PACT"),
        (EffectKey.HOLY_GRAIL,),
    ),
    AdventurerKind.ROGUE: HeroLoadout(
        AdventurerKind.ROGUE,
        ("R_ARMOR", "R_HEAL", "R_RING", "R_BUCK", "R_VORP", "R_CLOAK"),
        (EffectKey.RING, EffectKey.CLOAK),
    ),
}


def hp_for_equip(eq_id: str) -> int:
    return ALL_EQUIP_DB[eq_id][1]


EQUIP_VORPAL_IDS: frozenset[str] = frozenset({"W_VORPAL", "R_VORP"})
EQUIP_HEAL_POT: frozenset[str] = frozenset({"B_HEAL", "R_HEAL"})
EQUIP_FIRE_AXE: str = "B_AXE"
EQUIP_POLY: str = "M_POLY"
EQUIP_PACT: str = "M_PACT"
EQUIP_OMNI: str = "M_OMNI"


def all_equipment_ids(hero: AdventurerKind) -> list[str]:
    return list(HERO_LOADOUT[hero].equipment_ids)
