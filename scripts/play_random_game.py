#!/usr/bin/env python3
"""Simulate a match with random legal moves (weighted sampling).

Policy knobs below (--pass-weight, --sacrifice-weight, min pile before pass) exist only
to make this script produce nicer toy runs. They are not enforced by ``Match`` and are
not part of the real Welcome to the Dungeon ruleset.

Bidding sim policy (on top of those bases): the effective pass weight scales up with
dungeon pile size; the effective sacrifice weight goes down for each equipment
piece already set aside to dodge cards this round; when a facedown card is pending,
sacrifice is weighted by that card's strength. In the dungeon, Vorpal declaration
is weighted by each species' base strength (see module constants).

With ``--gui`` / ``--visual``, opens a pygame window (install ``.[gui]``) in a
**table-style** layout: facedown cards, private draw, and equipment with pieces
X'd when sacrificed or used. Use ``--god`` to also show exact card faces from the
deck and pending draw (debug). Timing flags control pace (defaults are slow).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import dungeon_runner.actions as A
from dungeon_runner.bots import pick_action, random_sim_action_subset
from dungeon_runner.catalog import ALL_EQUIP_DB
from dungeon_runner.match import BiddingState, Match, MatchPhase
from dungeon_runner.types_core import AdventurerKind


def _eq_label(eid: str) -> str:
    return ALL_EQUIP_DB.get(eid, (eid, 0, False, None))[0]


def _hero(h: AdventurerKind) -> str:
    return h.name.capitalize()


def describe_move(m: Match, a: object) -> str:
    if m.phase is MatchPhase.DUNGEON and m.runner_seat is not None:
        who = f"Runner {m.runner_seat}"
    else:
        who = f"Seat {m.active_seat}"

    if isinstance(a, A.PassBid):
        return f"{who} passes"
    if isinstance(a, A.DrawCard):
        return f"{who} draws"
    if isinstance(a, A.AddToDungeon):
        return f"{who} adds to dungeon"
    if isinstance(a, A.SacrificeEquipment):
        return f"{who} sacrifices {_eq_label(a.equipment_id)}"
    if isinstance(a, A.ChooseNextAdventurer):
        p = m.pick_next_seat
        return f"Seat {p} picks {_hero(a.hero)}"
    if isinstance(a, A.DeclareVorpal):
        return f"{who} vorpal → {a.target_species.value}"
    if isinstance(a, A.RevealOrContinue):
        return f"{who} next card"
    if isinstance(a, A.UseFireAxe):
        return f"{who} Fire Axe"
    if isinstance(a, A.DeclineFireAxe):
        return f"{who} skips axe"
    if isinstance(a, A.UsePolymorph):
        return f"{who} Polymorph"
    if isinstance(a, A.DeclinePolymorph):
        return f"{who} skips poly"
    return str(a)


def state_line(m: Match) -> str:
    if m.phase is MatchPhase.BIDDING:
        mid = "add/sac?" if m.bidding_sub is BiddingState.PENDING else "—"
        return (
            f"{_hero(m.hero)} · pile {len(m.dungeon_pile)} · deck {len(m.monster_deck)} · "
            f"eq {len(m.center_equipment)} · seat {m.active_seat} · {mid}"
        )

    if m.phase is MatchPhase.DUNGEON:
        sub = (m.dungeon_sub.name if m.dungeon_sub else "—").lower()
        cur = m.d_current.species.value if m.d_current else "—"
        return f"{_hero(m.hero)} · HP {m.d_hp} · {len(m.d_remaining)} facedown · now {cur} · {sub}"

    if m.phase is MatchPhase.PICK_ADVENTURER:
        return f"Pick hero · seat {m.pick_next_seat} · {m.success_cards_left} successes in box"

    if m.phase is MatchPhase.ENDED:
        r = m.terminal_reason.name.replace("_", " ").lower() if m.terminal_reason else "?"
        return f"Winner seat {m.winner_seat} ({r})"

    return m.phase.name


def roster_line(m: Match) -> str:
    parts = []
    for i, pl in enumerate(m.players):
        bits = []
        if pl.eliminated:
            bits.append("out")
        if pl.success_cards:
            bits.append(f"{pl.success_cards} succ")
        if pl.aid_flips:
            bits.append(f"{pl.aid_flips} hurt")
        if pl.has_passed_bid and m.phase is MatchPhase.BIDDING:
            bits.append("passed")
        extra = (" " + " ".join(bits)) if bits else ""
        parts.append(f"{i}{extra}")
    return "Scores: " + " · ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--players", type=int, default=3, help="2–4 (default 3)")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (optional)")
    ap.add_argument(
        "--pass-weight",
        type=float,
        default=0.2,
        help="Relative weight for pass vs other bidding moves (simulation only)",
    )
    ap.add_argument(
        "--sacrifice-weight",
        type=float,
        default=0.03,
        help="Relative weight for sacrifice vs add in pending step (simulation only)",
    )
    ap.add_argument("--max-steps", type=int, default=8000, help="Safety cap")
    ap.add_argument(
        "--gui",
        "--visual",
        action="store_true",
        help="Open pygame observer view (pip install -e \".[gui]\")",
    )
    ap.add_argument(
        "--god",
        action="store_true",
        help="With --gui: show deck / pending card faces (debug, not a fair table view)",
    )
    ap.add_argument(
        "--step-ms",
        type=float,
        default=1400.0,
        help="With --gui: delay after each move outside dungeon phase (ms)",
    )
    ap.add_argument(
        "--dungeon-step-ms",
        type=float,
        default=2000.0,
        help="With --gui: delay after each move during dungeon phase (ms)",
    )
    ap.add_argument(
        "--end-screen-ms",
        type=float,
        default=8000.0,
        help="With --gui: time to show final state before exit (ms)",
    )
    ap.add_argument(
        "--banner-ms",
        type=float,
        default=5000.0,
        help="With --gui: how long the dungeon success/fail banner stays visible (ms)",
    )
    args = ap.parse_args()

    view = None
    if args.gui:
        try:
            from dungeon_runner.ui import MatchViewConfig, get_match_view
        except ImportError as err:
            print(
                "Pygame UI requires: pip install -e \".[gui]\"  (or: pip install pygame)",
                file=sys.stderr,
            )
            raise SystemExit(1) from err
        vcfg = MatchViewConfig(
            step_delay_ms=args.step_ms,
            dungeon_step_delay_ms=args.dungeon_step_ms,
            end_screen_ms=args.end_screen_ms,
            god_mode=args.god,
            run_outcome_banner_ms=args.banner_ms,
        )
        view = get_match_view(vcfg)

    rng = random.Random(args.seed)
    m = Match.new(args.players, rng, AdventurerKind.WARRIOR, start_seat=0)
    step = 0

    seed_note = f" · seed {args.seed}" if args.seed is not None else ""
    print(f"{args.players} players · Warrior first{seed_note}")
    print(roster_line(m), "—", state_line(m), end="\n\n")

    if view is not None:
        view.sync(m)
        if not view.pump(0):
            view.close()
            print("Window closed before start.", file=sys.stderr)
            return

    while m.phase is not MatchPhase.ENDED and step < args.max_steps:
        acts = m.legal_actions()
        if not acts:
            print("No legal moves; stopping.")
            break
        acts = random_sim_action_subset(m, acts)
        if not acts:
            print("Random-run policy left no moves; stopping.")
            break
        a = pick_action(
            m,
            acts,
            rng,
            pass_weight=args.pass_weight,
            sacrifice_weight=args.sacrifice_weight,
        )
        step += 1
        line = describe_move(m, a)
        phase_before = m.phase
        m.apply(a)
        if view is not None:
            view.sync(m)
            d_ms = (
                view.config.dungeon_step_delay_ms
                if m.phase is MatchPhase.DUNGEON
                or phase_before is MatchPhase.DUNGEON
                else view.config.step_delay_ms
            )
            if not view.pump(d_ms):
                print("Window closed; stopping.", file=sys.stderr)
                break
        if phase_before is MatchPhase.DUNGEON and m.phase is not MatchPhase.DUNGEON:
            if m.dungeon_run_log:
                print("  Dungeon report:")
                for ln in m.dungeon_run_log:
                    print(f"    {ln}")
        if m.phase != phase_before and m.phase is MatchPhase.BIDDING and phase_before is MatchPhase.PICK_ADVENTURER:
            print(roster_line(m))
        print(f"{step:4d}  {line}  →  {state_line(m)}")

    print()
    if m.phase is MatchPhase.ENDED:
        print(roster_line(m))
        print(state_line(m))
    elif step >= args.max_steps:
        print(f"(Cut off at {args.max_steps} moves.)")

    if view is not None:
        if view.is_open:
            view.sync(m)
            view.pump(view.config.end_screen_ms)
        view.close()


if __name__ == "__main__":
    main()
