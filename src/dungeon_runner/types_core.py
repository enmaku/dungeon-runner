from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto, Enum
from typing import FrozenSet


class Species(Enum):
    GOBLIN = "Goblin"
    SKELETON = "Skeleton"
    ORC = "Orc"
    VAMPIRE = "Vampire"
    GOLEM = "Golem"
    LICH = "Lich"
    DEMON = "Demon"
    DRAGON = "Dragon"


class Icon(Enum):
    TORCH = auto()
    CHALICE = auto()
    HAMMER = auto()
    CLOAK = auto()
    PACT = auto()
    STAFF = auto()


@dataclass(frozen=True, slots=True)
class MonsterDef:
    species: Species
    strength: int
    icons: FrozenSet[Icon]


@dataclass(frozen=True, slots=True)
class MonsterInstance:
    species: Species
    strength: int
    icons: FrozenSet[Icon]
    def_id: int  # unique id for this physical card in the sim


class AdventurerKind(IntEnum):
    WARRIOR = 0
    BARBARIAN = 1
    MAGE = 2
    ROGUE = 3


# Effect tags for static ordering; resolvers in match.py
class EffectKey(Enum):
    VORPAL_AUTO = auto()
    DEMONIC_PACT = auto()
    RING = auto()
    HOLY_GRAIL = auto()  # even
    WAR_HAMMER_GOLEM = auto()
    STAFF_DRAGON = auto()  # dragon spear
    CLOAK = auto()  # >=6
    TORCH = auto()  # le3
    HAMMER = auto()  # golem, generic icon
    PACT = auto()  # unused for direct icon


def base_hp(hero: AdventurerKind) -> int:
    return {AdventurerKind.WARRIOR: 3, AdventurerKind.BARBARIAN: 4, AdventurerKind.MAGE: 2, AdventurerKind.ROGUE: 3}[
        hero
    ]


@dataclass
class PlayerState:
    seat: int
    has_passed_bid: bool = False
    aid_flips: int = 0  # 0=white, 1=red, 2=eliminated
    success_cards: int = 0
    eliminated: bool = False
    # bidding: monsters added to dungeon (for observation later) — this seat's contribution list
    own_pile_adds: list[Species] = field(default_factory=list)
    # monsters facedown in front from sacrifice (kept in central sacrifice list with seat ref if needed; engine tracks globally)


@dataclass
class SacrificeSetaside:
    monster: MonsterInstance
    equipment_id: str
    seat: int  # who sacrificed
