from dungeon_runner.catalog import HERO_LOADOUT, default_monster_deck_list, all_equipment_ids
from dungeon_runner.types_core import AdventurerKind, Species


def test_deck_is_13_from_rules():
    d = default_monster_deck_list()
    assert len(d) == 13
    from collections import Counter

    c = Counter(m.species for m in d)
    assert c[Species.DRAGON] == 1
    assert c[Species.GOBLIN] == 2


def test_each_hero_has_six_equipment():
    for h in AdventurerKind:
        assert len(HERO_LOADOUT[h].equipment_ids) == 6
        assert len(set(all_equipment_ids(h))) == 6
