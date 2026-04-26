from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import ClassVar, Final

from dungeon_runner import actions as A
from dungeon_runner.catalog import (
    EQUIP_OMNI,
    EQUIP_PACT,
    EQUIP_POLY,
    HERO_LOADOUT,
    shuffled_deck,
    all_equipment_ids,
    default_monster_deck_list,
    make_deck_instance_ids,
    hp_for_equip,
    EQUIP_VORPAL_IDS,
    EQUIP_FIRE_AXE,
    EQUIP_HEAL_POT,
)
from dungeon_runner.errors import IllegalAction
from dungeon_runner.types_core import (
    AdventurerKind,
    EffectKey,
    MonsterInstance,
    PlayerState,
    SacrificeSetaside,
    Species,
    base_hp,
)

ALL_SPECIES: ClassVar[Final[tuple[Species, ...]]] = (
    Species.GOBLIN,
    Species.SKELETON,
    Species.ORC,
    Species.VAMPIRE,
    Species.GOLEM,
    Species.LICH,
    Species.DEMON,
    Species.DRAGON,
)


class MatchTerminalReason(Enum):
    SECOND_SUCCESS = auto()
    LAST_STANDING = auto()
    # House rule: all passed with nothing in the deck or dungeon pile; no one runs.
    BIDDING_EMPTY_STALE = auto()
    # Sole bidder would run an empty pile with no bidding sacrifices (pass-only bidding).
    EMPTY_DUNGEON_FORFEIT = auto()


class MatchPhase(Enum):
    BIDDING = auto()
    DUNGEON = auto()
    PICK_ADVENTURER = auto()
    ENDED = auto()


class BiddingState(Enum):
    TURN = auto()
    PENDING = auto()


class DungeonSub(Enum):
    VORPAL = auto()
    REVEAL = auto()
    PICK_FIRE_AXE = auto()
    PICK_POLYMORPH = auto()


@dataclass
class Match:
    n_players: int
    rng: random.Random
    phase: MatchPhase
    players: list[PlayerState]
    start_seat: int
    active_seat: int
    hero: AdventurerKind
    center_equipment: set[str]
    monster_deck: list[MonsterInstance]
    global_discard: list[MonsterInstance]
    bidding_sub: BiddingState
    pending_card: MonsterInstance | None
    dungeon_pile: list[MonsterInstance]
    runner_seat: int | None
    sacrifice_rows: list[SacrificeSetaside]
    dungeon_sub: DungeonSub | None
    d_remaining: list[MonsterInstance]
    d_in_play: set[str]
    d_hp: int
    d_ad_base: int
    d_discard_run: list[MonsterInstance]
    d_current: MonsterInstance | None
    d_vorpal_target: Species | None
    d_poly_spent: bool
    d_axe_spent: bool
    success_cards_left: int
    next_id: int
    terminal_reason: MatchTerminalReason | None
    winner_seat: int | None
    pick_next_seat: int | None
    d_heal_used: set[str] = field(default_factory=set)
    # pool of cards to shuffle when starting a round from PICK
    _round_pool: list[MonsterInstance] = field(default_factory=list)
    # Human-readable lines for the current dungeon only; cleared when a new dungeon starts
    dungeon_run_log: list[str] = field(default_factory=list)
    # RL dungeon-success reward: len(dungeon_pile) + bidding SacrificeEquipment count at run start
    dungeon_run_reward_difficulty: int = 0

    @classmethod
    def new(
        cls,
        n_players: int,
        rng: random.Random,
        first_hero: AdventurerKind = AdventurerKind.WARRIOR,
        start_seat: int = 0,
        *,
        monster_deck: list[MonsterInstance] | None = None,
    ) -> Match:
        """
        Create a new match. Pass ``monster_deck`` (exactly 13 cards) to skip shuffling for scripted tests.
        """
        if n_players < 2 or n_players > 4:
            raise ValueError("n_players must be 2-4")
        base_defs = default_monster_deck_list()
        if monster_deck is not None:
            if len(monster_deck) != 13:
                raise ValueError("monster_deck must have exactly 13 cards")
            deck = list(monster_deck)
        else:
            deck = shuffled_deck(rng, base_defs, start_id=0)
        return cls(
            n_players=n_players,
            rng=rng,
            phase=MatchPhase.BIDDING,
            players=[PlayerState(seat=i) for i in range(n_players)],
            start_seat=start_seat,
            active_seat=start_seat,
            hero=first_hero,
            center_equipment=set(all_equipment_ids(first_hero)),
            monster_deck=deck,
            global_discard=[],
            bidding_sub=BiddingState.TURN,
            pending_card=None,
            dungeon_pile=[],
            runner_seat=None,
            sacrifice_rows=[],
            dungeon_sub=None,
            d_remaining=[],
            d_in_play=set(),
            d_hp=0,
            d_ad_base=base_hp(first_hero),
            d_discard_run=[],
            d_current=None,
            d_vorpal_target=None,
            d_poly_spent=True,
            d_axe_spent=True,
            success_cards_left=5,
            next_id=len(base_defs),
            terminal_reason=None,
            winner_seat=None,
            pick_next_seat=None,
            dungeon_run_log=[],
            dungeon_run_reward_difficulty=0,
        )

    def _left_neighbor(self, s: int) -> int:
        return (s - 1) % self.n_players

    def _dlog(self, msg: str) -> None:
        self.dungeon_run_log.append(msg)

    def _dlog_axe_choice(self, m: MonsterInstance) -> None:
        self._dlog(
            f"  Choice — Fire Axe: destroy {m.species.value} (strength {m.strength}) for no HP loss, or decline."
        )

    def _dlog_poly_choice(self, m: MonsterInstance) -> None:
        nxt = len(self.d_remaining)
        self._dlog(
            f"  Choice — Polymorph: discard {m.species.value} (strength {m.strength}) facedown and face the "
            f"next card ({nxt} facedown left), or decline and fight this one."
        )

    def legal_actions(self) -> set[object]:
        if self.phase is MatchPhase.ENDED:
            return set()
        if self.phase is MatchPhase.PICK_ADVENTURER:
            if self.pick_next_seat is None or self.active_seat != self.pick_next_seat:
                return set()
            return {A.ChooseNextAdventurer(h) for h in AdventurerKind}  # type: ignore[return-value]
        if self.phase is MatchPhase.BIDDING:
            p = self.players[self.active_seat]
            if p.eliminated or p.has_passed_bid:
                return set()
            if self.bidding_sub is BiddingState.PENDING:
                if self.pending_card is None:
                    return set()
                a: set[object] = {A.AddToDungeon()}
                for e in sorted(self.center_equipment):
                    a.add(A.SacrificeEquipment(e))
                if not self.center_equipment:
                    a = {A.AddToDungeon()}
                return a
            if self.monster_deck:
                return {A.PassBid(), A.DrawCard()}
            return {A.PassBid()}
        if self.phase is MatchPhase.DUNGEON:
            if self.active_seat != self.runner_seat or self.runner_seat is None:
                return set()
            sub = self.dungeon_sub
            if sub is DungeonSub.VORPAL:
                if not any(v in self.d_in_play for v in EQUIP_VORPAL_IDS):
                    return {A.RevealOrContinue()}
                return {A.DeclareVorpal(sp) for sp in ALL_SPECIES}
            if sub is DungeonSub.PICK_FIRE_AXE:
                return {A.UseFireAxe(), A.DeclineFireAxe()}
            if sub is DungeonSub.PICK_POLYMORPH:
                return {A.UsePolymorph(), A.DeclinePolymorph()}
            if sub is DungeonSub.REVEAL:
                return {A.RevealOrContinue()}
        return set()

    def apply(self, action: A.Action) -> None:  # noqa: C901, PLR0911, PLR0912
        if self.phase is MatchPhase.ENDED:
            raise IllegalAction("match ended", action)
        if self.phase is MatchPhase.PICK_ADVENTURER:
            if not isinstance(action, A.ChooseNextAdventurer):
                raise IllegalAction("expected ChooseNextAdventurer", action)
            if self.pick_next_seat is None or self.active_seat != self.pick_next_seat:
                raise IllegalAction("not picker", action)
            self.hero = action.hero
            self.center_equipment = set(all_equipment_ids(action.hero))
            self.d_ad_base = base_hp(action.hero)
            if not self._round_pool:
                self._rebuild_pool_from_spare()
            self.rng.shuffle(self._round_pool)
            self.monster_deck = list(self._round_pool)
            self._round_pool = []
            self.global_discard = []
            for pl in self.players:
                if not pl.eliminated:
                    pl.has_passed_bid = False
            self.dungeon_pile = []
            self.sacrifice_rows = []
            self.bidding_sub = BiddingState.TURN
            self.pending_card = None
            self.phase = MatchPhase.BIDDING
            self.start_seat = self.active_seat
            for pl in self.players:
                pl.own_pile_adds = []
            self.pick_next_seat = None
            return
        if self.phase is MatchPhase.BIDDING:
            self._apply_bidding(action)
            return
        if self.phase is MatchPhase.DUNGEON:
            self._apply_dungeon(action)
            return
        raise IllegalAction("bad phase", action)

    def _rebuild_pool_from_spare(self) -> None:
        n = default_monster_deck_list()
        self._round_pool = make_deck_instance_ids(n, self.next_id)
        self.next_id += len(n)

    def _apply_bidding(self, action: A.Action) -> None:
        s, p = self.active_seat, self.players[self.active_seat]
        if p.eliminated or p.has_passed_bid:
            raise IllegalAction("bad turn", action)
        if self.bidding_sub is BiddingState.PENDING:
            if self.pending_card is None:
                raise IllegalAction("no pending", action)
            if isinstance(action, A.AddToDungeon):
                self.dungeon_pile.append(self.pending_card)
                p.own_pile_adds.append(self.pending_card.species)
            elif isinstance(action, A.SacrificeEquipment):
                eid = action.equipment_id
                if eid not in self.center_equipment:
                    raise IllegalAction("sac", action)
                self.sacrifice_rows.append(SacrificeSetaside(self.pending_card, eid, s))
                self.center_equipment.remove(eid)
            else:
                raise IllegalAction("add/sac", action)
            self.pending_card = None
            self.bidding_sub = BiddingState.TURN
            if self._count_active_bidders() == 1:
                self._end_bidding()
            else:
                self._advance_bid_seat()
            return
        if isinstance(action, A.PassBid):
            p.has_passed_bid = True
            c = self._count_active_bidders()
            if c == 0:
                self._end_m(MatchTerminalReason.BIDDING_EMPTY_STALE, None)
            elif c == 1 and self._bidding_stale_house():
                self._end_m(MatchTerminalReason.BIDDING_EMPTY_STALE, None)
            elif c == 1:
                self._end_bidding()
            else:
                self._advance_bid_seat()
            return
        if isinstance(action, A.DrawCard):
            if not self.monster_deck:
                raise IllegalAction("empty deck, must pass", action)
            self.pending_card = self.monster_deck.pop(0)
            self.bidding_sub = BiddingState.PENDING
            return
        raise IllegalAction("bidding", action)

    def _count_active_bidders(self) -> int:
        return sum(1 for pl in self.players if (not pl.eliminated) and (not pl.has_passed_bid))

    def _advance_bid_seat(self) -> None:
        a = (self.active_seat + 1) % self.n_players
        for _ in range(self.n_players + 1):
            pl = self.players[a]
            if (not pl.eliminated) and (not pl.has_passed_bid):
                self.active_seat = a
                return
            a = (a + 1) % self.n_players

    def _end_bidding(self) -> None:
        for i, pl in enumerate(self.players):
            if (not pl.eliminated) and (not pl.has_passed_bid):
                self.runner_seat = i
                break
        else:
            raise RuntimeError("no runner after bidding")
        if not self.dungeon_pile:
            self._dlog(
                "Forfeit — sole bidder with an empty dungeon pile (nothing to run); match ends with no winner."
            )
            self._end_m(MatchTerminalReason.EMPTY_DUNGEON_FORFEIT, None)
            return
        self.dungeon_run_log.clear()
        self.active_seat = self.runner_seat
        self.dungeon_run_reward_difficulty = len(self.dungeon_pile) + len(self.sacrifice_rows)
        self.d_remaining = list(self.dungeon_pile)
        self.d_in_play = set(self.center_equipment)
        self.d_ad_base = base_hp(self.hero)
        self.d_hp = self.d_ad_base + sum(hp_for_equip(e) for e in self.d_in_play)
        self.d_discard_run = []
        self.d_current = None
        self.d_vorpal_target = None
        self.d_poly_spent = EQUIP_POLY not in self.d_in_play
        self.d_axe_spent = EQUIP_FIRE_AXE not in self.d_in_play
        self.d_heal_used = set()
        if any(v in self.d_in_play for v in EQUIP_VORPAL_IDS):
            self.dungeon_sub = DungeonSub.VORPAL
        else:
            self.dungeon_sub = DungeonSub.REVEAL
        self.phase = MatchPhase.DUNGEON
        self._dlog(
            f"Start · seat {self.runner_seat} runs · {_hero_label(self.hero)} · "
            f"HP {self.d_hp} · {len(self.d_remaining)} monsters in the pile"
        )
        if self.dungeon_sub is DungeonSub.VORPAL:
            self._dlog(
                "Choice — Vorpal sword: name one monster species; the first card of that species "
                "revealed this run is slain at no HP cost, then the Vorpal tile is spent."
            )

    def _apply_dungeon(self, act: A.Action) -> None:  # noqa: C901, PLR0911, PLR0912
        if self.runner_seat is None or self.active_seat != self.runner_seat:
            raise IllegalAction("not runner", act)
        if self.dungeon_sub is DungeonSub.VORPAL:
            if isinstance(act, A.DeclareVorpal):
                self.d_vorpal_target = act.target_species
                self._dlog(f"Runner declares Vorpal prey: {act.target_species.value}.")
                self.dungeon_sub = DungeonSub.REVEAL
                return
            if isinstance(act, A.RevealOrContinue) and not any(
                v in self.d_in_play for v in EQUIP_VORPAL_IDS
            ):
                self.d_vorpal_target = None
                self._dlog("Runner continues without Vorpal naming (no Vorpal tile in play).")
                self.dungeon_sub = DungeonSub.REVEAL
                return
            raise IllegalAction("vorpal", act)
        if self.dungeon_sub is DungeonSub.REVEAL:
            if not isinstance(act, A.RevealOrContinue):
                raise IllegalAction("reveal", act)
            self._reveal_step()
            return
        if self.dungeon_sub is DungeonSub.PICK_FIRE_AXE:
            m = self.d_current
            if m is None:
                raise IllegalAction("no card", act)
            if isinstance(act, A.UseFireAxe):
                if self.d_axe_spent or EQUIP_FIRE_AXE not in self.d_in_play:
                    raise IllegalAction("no axe", act)
                self._dlog(f"  Runner uses Fire Axe — {m.species.value} destroyed (no HP loss).")
                self.d_in_play.discard(EQUIP_FIRE_AXE)
                self.d_axe_spent = True
                self.d_discard_run.append(m)
                self.d_current = None
                self._after_defeated()
                return
            if isinstance(act, A.DeclineFireAxe):
                self._dlog(f"  Runner declines Fire Axe on {m.species.value} ({m.strength}).")
                if self._poly_legal():
                    self._dlog_poly_choice(m)
                    self.dungeon_sub = DungeonSub.PICK_POLYMORPH
                else:
                    self._apply_hits()
                return
            raise IllegalAction("axe", act)
        if self.dungeon_sub is DungeonSub.PICK_POLYMORPH:
            m = self.d_current
            if m is None:
                raise IllegalAction("no card", act)
            if isinstance(act, A.UsePolymorph):
                if not self._poly_legal():
                    raise IllegalAction("poly", act)
                self.d_poly_spent = True
                self.d_in_play.discard(EQUIP_POLY)
                nxt = self.d_remaining.pop(0)
                self._dlog(
                    f"  Runner uses Polymorph — {m.species.value} discarded facedown; "
                    f"next reveal: {nxt.species.value} (strength {nxt.strength})."
                )
                self.d_discard_run.append(m)
                self.d_current = nxt
                self.dungeon_sub = DungeonSub.REVEAL
                self._process_auto_chain()
                return
            if isinstance(act, A.DeclinePolymorph):
                self._dlog(
                    f"  Runner declines Polymorph on {m.species.value} ({m.strength}); "
                    "resolving combat (HP loss if not defeated)."
                )
                self._apply_hits()
                return
            raise IllegalAction("poly2", act)
        raise IllegalAction("dungeon", act)

    def _poly_legal(self) -> bool:
        return (not self.d_poly_spent) and (EQUIP_POLY in self.d_in_play) and (len(self.d_remaining) > 0)

    def _reveal_step(self) -> None:
        if self.d_current is None:
            if not self.d_remaining:
                if self.d_hp > 0:
                    self._dlog("Pile empty — dungeon cleared with no further reveals.")
                    self._dungeon_success()
                return
            self.d_current = self.d_remaining.pop(0)
        self._process_auto_chain()

    def _process_auto_chain(self) -> None:
        m = self.d_current
        if m is None:
            return
        self._dlog(f"Reveal {m.species.value} (strength {m.strength})")
        if self._vorpal_kills(m):
            self._dlog(
                f"  Vorpal — first {m.species.value} this run is slain; vorpal tile spent (no HP loss)."
            )
            for vid in list(EQUIP_VORPAL_IDS):
                self.d_in_play.discard(vid)
            self.d_discard_run.append(m)
            self.d_current = None
            self.dungeon_sub = DungeonSub.REVEAL
            self._after_defeated()
            return
        dpair = self._demonic_pact(m)
        if dpair is not None:
            if len(dpair) == 2:
                self._dlog(
                    f"  Demonic Pact — Demon and {dpair[1].species.value} removed with no combat; pact spent."
                )
            else:
                self._dlog("  Demonic Pact — Demon removed (no next card); pact spent.")
            for c in dpair:
                if c is not None:
                    self.d_discard_run.append(c)
            self.d_in_play.discard(EQUIP_PACT)
            self.d_current = None
            self.dungeon_sub = DungeonSub.REVEAL
            self._after_defeated()
            return
        std = self._standard_defeat(m)
        if std is not None:
            self._dlog(f"  {std}")
            self.d_discard_run.append(m)
            self.d_current = None
            self.dungeon_sub = DungeonSub.REVEAL
            self._after_defeated()
            return
        if (not self.d_axe_spent) and (EQUIP_FIRE_AXE in self.d_in_play):
            self._dlog_axe_choice(m)
            self.dungeon_sub = DungeonSub.PICK_FIRE_AXE
            return
        if self._poly_legal():
            self._dlog_poly_choice(m)
            self.dungeon_sub = DungeonSub.PICK_POLYMORPH
            return
        self._apply_hits()

    def _demonic_pact(self, m: MonsterInstance) -> list[MonsterInstance] | None:
        if m.species is not Species.DEMON or EQUIP_PACT not in self.d_in_play:
            return None
        if not self.d_remaining:
            return [m]
        n2 = self.d_remaining.pop(0)
        return [m, n2]

    def _vorpal_kills(self, m: MonsterInstance) -> bool:
        if self.d_vorpal_target is None or m.species is not self.d_vorpal_target:
            return False
        return any(v in self.d_in_play for v in EQUIP_VORPAL_IDS)

    def _standard_defeat(self, m: MonsterInstance) -> str | None:
        h = self.hero
        for ek in HERO_LOADOUT[h].effect_priority:
            if ek is EffectKey.RING and "R_RING" in self.d_in_play and m.strength <= 2:
                self.d_hp += m.strength
                return f"Ring of Power — defeats strength ≤2; heal +{m.strength} HP (now {self.d_hp})."
            if ek is EffectKey.HOLY_GRAIL:
                gr = None
                if h is AdventurerKind.WARRIOR:
                    gr = "W_HOLY"
                elif h is AdventurerKind.MAGE:
                    gr = "M_HOLY"
                if gr and gr in self.d_in_play and m.strength % 2 == 0:
                    return "Holy Grail — even strength monster banished (no HP loss)."
            if ek is EffectKey.TORCH and _torch_ok(h, self.d_in_play) and m.strength <= 3:
                return "Torch — strength ≤3 banished (no HP loss)."
            if (
                ek is EffectKey.STAFF_DRAGON
                and "W_SPEAR" in self.d_in_play
                and m.species is Species.DRAGON
            ):
                return "Dragon Spear — Dragon banished (no HP loss)."
            if (
                ek is EffectKey.CLOAK
                and "R_CLOAK" in self.d_in_play
                and m.strength >= 6
            ):
                return "Invisibility Cloak — strength ≥6 banished (no HP loss)."
            if (
                ek is EffectKey.WAR_HAMMER_GOLEM
                and "B_HAMMER" in self.d_in_play
                and m.species is Species.GOLEM
            ):
                return "War Hammer — Golem banished (no HP loss)."
        return None

    def _after_defeated(self) -> None:
        if self.d_remaining:
            self.dungeon_sub = DungeonSub.REVEAL
            return
        if self.d_hp > 0:
            self._dungeon_success()

    def _apply_hits(self) -> None:  # noqa: C901
        m = self.d_current
        if m is None:
            return
        self.d_hp -= m.strength
        if self.d_hp > 0:
            self._dlog(f"  No automatic defeat — take {m.strength} damage → {self.d_hp} HP left.")
            self.d_discard_run.append(m)
            self.d_current = None
            self.dungeon_sub = DungeonSub.REVEAL
            self._after_defeated()
            return
        for pot in list(self.d_in_play):
            if pot in EQUIP_HEAL_POT and pot not in self.d_heal_used:
                self.d_hp = self.d_ad_base
                self.d_in_play.discard(pot)
                self.d_heal_used.add(pot)
                self._dlog(
                    f"  Would die on {m.species.value} ({m.strength}) — Healing Potion revives to "
                    f"{self.d_hp} HP (adventurer base only); potion spent."
                )
                self.d_discard_run.append(m)
                self.d_current = None
                self.dungeon_sub = DungeonSub.REVEAL
                self._after_defeated()
                return
        if self.hero is AdventurerKind.MAGE and EQUIP_OMNI in self.d_in_play and self._omni_saves():
            self._dlog(
                "  HP at 0 — Omnipotence: every species in this round's dungeon (including sacrifices) "
                "is unique; dungeon still counts as a win."
            )
            self.d_discard_run.append(m)
            self.d_current = None
            self.d_discard_run.extend(self.d_remaining)
            self.d_remaining.clear()
            self.dungeon_sub = DungeonSub.REVEAL
            self._dungeon_success()
            return
        self._dlog(
            f"  Lethal on {m.species.value} ({m.strength}) — no potion / no Omnipotence save; dungeon lost."
        )
        self.d_discard_run.append(m)
        self.d_current = None
        self._dungeon_fail()

    def _omni_saves(self) -> bool:
        allm: list[MonsterInstance] = list(self.d_discard_run)
        allm.extend(r.monster for r in self.sacrifice_rows)
        if self.d_current is not None:
            allm.append(self.d_current)
        allm.extend(self.d_remaining)
        sps = [x.species for x in allm]
        return bool(sps) and (len(sps) == len(set(sps)))

    def _build_round_pool(self) -> None:
        p = (
            self.monster_deck
            + self.global_discard
            + self.d_discard_run
            + [r.monster for r in self.sacrifice_rows]
            + self.d_remaining
        )
        if self.pending_card is not None:
            p.append(self.pending_card)
        if self.d_current is not None:
            p.append(self.d_current)
        dedup: dict[int, MonsterInstance] = {c.def_id: c for c in p}
        p = list(dedup.values())
        if len(p) != 13:
            self._rebuild_pool_from_spare()
        else:
            self._round_pool = p
        if len(self._round_pool) != 13:
            raise RuntimeError(f"expected 13 cards in round pool, got {len(self._round_pool)}")

    def _dungeon_success(self) -> None:
        r = self.runner_seat
        if r is None:
            return
        if self.success_cards_left <= 0:
            return
        pl = self.players[r]
        self.success_cards_left -= 1
        pl.success_cards += 1
        self._dlog(
            f"Result: Success — seat {r} survives; success cards for that seat now {pl.success_cards}."
        )
        if pl.success_cards >= 2:
            self._dlog("Match over — second success.")
            self._end_m(MatchTerminalReason.SECOND_SUCCESS, r)
            return
        self._build_round_pool()
        sp = r if (not pl.eliminated) else self._left_neighbor(r)
        self.start_seat = sp
        self.active_seat = sp
        self.pick_next_seat = sp
        self.phase = MatchPhase.PICK_ADVENTURER
        self.dungeon_sub = None

    def _dungeon_fail(self) -> None:  # noqa: C901
        r = self.runner_seat
        if r is None:
            return
        p = self.players[r]
        p.aid_flips += 1
        if p.aid_flips >= 2:
            p.eliminated = True
            self._dlog(f"Result: Failure — seat {r} is eliminated (second dungeon loss).")
        else:
            self._dlog(f"Result: Failure — seat {r} marks a dungeon loss on the aid ({p.aid_flips} of 2).")
        alive = [i for i, pl in enumerate(self.players) if not pl.eliminated]
        if len(alive) == 1:
            self._dlog(f"Match over — only seat {alive[0]} remains.")
            self._end_m(MatchTerminalReason.LAST_STANDING, alive[0])
            return
        if not p.eliminated:
            sp = r
        else:
            sp = self._left_neighbor(r)
        self._build_round_pool()
        self.start_seat = sp
        self.active_seat = sp
        self.pick_next_seat = sp
        self.phase = MatchPhase.PICK_ADVENTURER
        self.dungeon_sub = None

    def _bidding_stale_house(self) -> bool:
        return (
            self.bidding_sub is BiddingState.TURN
            and not self.monster_deck
            and self.pending_card is None
            and not self.dungeon_pile
        )

    def _end_m(self, reason: MatchTerminalReason, w: int | None) -> None:
        self.phase = MatchPhase.ENDED
        self.terminal_reason = reason
        self.winner_seat = w
        self.dungeon_sub = None

def _torch_ok(hero: AdventurerKind, inv: set[str]) -> bool:
    if hero is AdventurerKind.WARRIOR:
        return "W_TORCH" in inv
    if hero is AdventurerKind.BARBARIAN:
        return "B_TORCH" in inv
    return False


def _hero_label(h: AdventurerKind) -> str:
    return h.name.capitalize()
