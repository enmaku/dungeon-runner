from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from dungeon_runner.types_core import AdventurerKind, Species

# Bidding
@dataclass(frozen=True, slots=True)
class PassBid:
    pass


@dataclass(frozen=True, slots=True)
class DrawCard:
    pass


@dataclass(frozen=True, slots=True)
class AddToDungeon:
    pass


@dataclass(frozen=True, slots=True)
class SacrificeEquipment:
    equipment_id: str


# After bidding: choose adventurer for next round (from runner of previous)
@dataclass(frozen=True, slots=True)
class ChooseNextAdventurer:
    hero: AdventurerKind


# Dungeon
@dataclass(frozen=True, slots=True)
class DeclareVorpal:
    target_species: Species


@dataclass(frozen=True, slots=True)
class RevealOrContinue:
    """Advance dungeon step (reveal next / finish micro-step)."""
    pass


@dataclass(frozen=True, slots=True)
class UseFireAxe:
    pass


@dataclass(frozen=True, slots=True)
class DeclineFireAxe:
    pass


@dataclass(frozen=True, slots=True)
class UsePolymorph:
    pass


@dataclass(frozen=True, slots=True)
class DeclinePolymorph:
    pass


Action = Union[
    PassBid,
    DrawCard,
    AddToDungeon,
    SacrificeEquipment,
    ChooseNextAdventurer,
    DeclareVorpal,
    RevealOrContinue,
    UseFireAxe,
    DeclineFireAxe,
    UsePolymorph,
    DeclinePolymorph,
]
