"""Fixed float vector observations per honest seat (README-faithful, no opponent leaks)."""

from __future__ import annotations

import numpy as np

from dungeon_runner.catalog import all_equipment_ids
from dungeon_runner.match import ALL_SPECIES, BiddingState, DungeonSub, Match, MatchPhase
from dungeon_runner.types_core import Icon, Species

# Species index in ALL_SPECIES
SPECIES_LIST = list(ALL_SPECIES)
N_SPECIES = len(SPECIES_LIST)  # 8
N_ICONS = 6
N_PHASE = 4
N_MAX_PLAYERS = 4
N_HERO = 4
# phase4, bid2, dsub4, npl1, act4, pl4x4, hero4, g3, eq6, pend15, own8, me2, dun18 = 4+2+4+1+4+16+4+3+6+15+8+2+18 = 87
OBS_DIM = 87


def _oh(n: int, i: int) -> list[float]:
    o = [0.0] * n
    if 0 <= i < n:
        o[i] = 1.0
    return o


def _species_oh(s: Species) -> list[float]:
    return _oh(N_SPECIES, SPECIES_LIST.index(s))


def _icons_oh(icons: frozenset) -> list[float]:
    o = [0.0] * N_ICONS
    order = (Icon.TORCH, Icon.CHALICE, Icon.HAMMER, Icon.CLOAK, Icon.PACT, Icon.STAFF)
    for j, k in enumerate(order):
        if k in icons:
            o[j] = 1.0
    return o


def _pending_features(m: Match, seat: int) -> list[float]:
    c = m.pending_card
    if m.phase is not MatchPhase.BIDDING or m.bidding_sub is not BiddingState.PENDING:
        return [0.0] * (N_SPECIES + 1 + N_ICONS)
    if c is None or m.active_seat != seat:
        return [0.0] * (N_SPECIES + 1 + N_ICONS)
    f = _species_oh(c.species)
    f.append(min(c.strength / 9.0, 1.0))
    f.extend(_icons_oh(c.icons))
    return f


def _player_block(m: Match) -> list[float]:
    out: list[float] = []
    for i in range(N_MAX_PLAYERS):
        if i >= m.n_players:
            out.extend([0.0, 0.0, 0.0, 0.0])
            continue
        p = m.players[i]
        out.append(min(p.success_cards / 2.0, 1.0))
        out.append(min(p.aid_flips / 2.0, 1.0))
        out.append(1.0 if p.eliminated else 0.0)
        o = 1.0 if (m.phase is MatchPhase.BIDDING and p.has_passed_bid) else 0.0
        out.append(o)
    return out


def _dungeon_block(m: Match, seat: int) -> list[float]:
    if m.phase is not MatchPhase.DUNGEON or m.runner_seat is None or seat != m.runner_seat:
        return [0.0] * 18
    dcur = m.d_current
    cur = _species_oh(dcur.species) if dcur is not None else [0.0] * N_SPECIES
    hpn = min(m.d_hp / 20.0, 1.0) if m.d_hp > 0 else 0.0
    rem = min(len(m.d_remaining) / 13.0, 1.0)
    sub = m.dungeon_sub
    sname, srev, saxe, spoly = 0.0, 0.0, 0.0, 0.0
    if sub is DungeonSub.VORPAL:
        sname = 1.0
    elif sub is DungeonSub.REVEAL:
        srev = 1.0
    elif sub is DungeonSub.PICK_FIRE_AXE:
        saxe = 1.0
    elif sub is DungeonSub.PICK_POLYMORPH:
        spoly = 1.0
    eids = all_equipment_ids(m.hero)
    inplay = [1.0 if e in m.d_in_play else 0.0 for e in eids]
    msum = sum(inplay) / 6.0
    mcnt = len(m.d_in_play) / 6.0
    return (
        cur
        + [hpn, rem, sname, srev, saxe, spoly, 1.0 if m.d_poly_spent else 0.0, 1.0 if m.d_axe_spent else 0.0]
        + [msum, mcnt]
    )


def build_observation(m: Match, seat: int) -> np.ndarray:
    v: list[float] = []

    pidx = 0
    if m.phase is MatchPhase.BIDDING:
        pidx = 0
    elif m.phase is MatchPhase.DUNGEON:
        pidx = 1
    elif m.phase is MatchPhase.PICK_ADVENTURER:
        pidx = 2
    else:
        pidx = 3
    v.extend(_oh(N_PHASE, pidx))

    if m.phase is MatchPhase.BIDDING and m.bidding_sub is BiddingState.TURN:
        v.extend([1.0, 0.0])
    elif m.phase is MatchPhase.BIDDING and m.bidding_sub is BiddingState.PENDING:
        v.extend([0.0, 1.0])
    else:
        v.extend([0.0, 0.0])

    ds = {DungeonSub.VORPAL: 0, DungeonSub.REVEAL: 1, DungeonSub.PICK_FIRE_AXE: 2, DungeonSub.PICK_POLYMORPH: 3}
    if m.phase is MatchPhase.DUNGEON and m.dungeon_sub is not None:
        v.extend(_oh(4, ds[m.dungeon_sub]))
    else:
        v.extend([0.0, 0.0, 0.0, 0.0])

    v.append(m.n_players / 4.0)
    v.extend(_oh(N_MAX_PLAYERS, m.active_seat if 0 <= m.active_seat < N_MAX_PLAYERS else 0))
    v.extend(_player_block(m))
    v.extend(_oh(N_HERO, int(m.hero)))
    v.append(len(m.dungeon_pile) / 13.0)
    v.append(len(m.monster_deck) / 13.0)
    v.append(m.success_cards_left / 5.0)

    ids = all_equipment_ids(m.hero)
    for eid in ids:
        v.append(1.0 if eid in m.center_equipment else 0.0)

    v.extend(_pending_features(m, seat))

    counts = [0] * N_SPECIES
    for sp in m.players[seat].own_pile_adds:
        idx = SPECIES_LIST.index(sp)
        counts[idx] = min(2, counts[idx] + 1)
    for c in counts:
        v.append(c / 2.0)

    v.append(1.0 if m.active_seat == seat else 0.0)
    rs = m.runner_seat
    v.append(1.0 if rs is not None and seat == rs else 0.0)
    v.extend(_dungeon_block(m, seat))

    out = np.asarray(v, dtype=np.float32)
    if out.shape[0] != OBS_DIM:
        msg = f"obs dim {out.shape[0]} != {OBS_DIM}"
        raise RuntimeError(msg)
    return out
