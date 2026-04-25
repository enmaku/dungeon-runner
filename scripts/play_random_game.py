#!/usr/bin/env python3
"""Simulate a match with per-seat policies: weighted random and/or Keras checkpoints.

Use ``--p0`` … ``--p3`` to set each seat to ``random`` or a path to a training logdir
(containing ``policy.weights.h5``) or to an ``.h5`` file. Omitted seats default to
``random``. Model seats need ``pip install -e ".[train]"`` (TensorFlow).

Example: ``--players 2 --p0 runs/v0.1a/3 --p1 random`` loads seat 0 from that
logdir's ``policy.weights.h5``. Example: ``--p0 runs/a/1 --p2 runs/b/2 --players 3``
uses two different checkpoints on seats 0 and 2.

Policy knobs below (--pass-weight, --sacrifice-weight, min pile before pass) apply only
to **random** seats; they are not enforced by ``Match`` and are not part of the real
Welcome to the Dungeon ruleset.

Bidding sim policy (random seats): the effective pass weight scales up with dungeon
pile size; sacrifice weight decays with sacrifice rows; pending-card strength scales
sacrifice; Vorpal is weighted by species base strength (see ``dungeon_runner.bots``).

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
        if m.winner_seat is None:
            return f"No winner  ({r})"
        return f"Winner seat {m.winner_seat} ({r})"

    return m.phase.name


def _parse_player_spec(spec: str) -> tuple[str, Path | None]:
    s = spec.strip()
    if s.lower() == "random":
        return ("random", None)
    p = Path(s).expanduser()
    if not p.exists():
        msg = f"path does not exist: {spec!r}"
        raise ValueError(msg)
    p_res = p.resolve()
    if p_res.is_file() and p_res.suffix.lower() == ".h5":
        return ("model", p_res)
    if p_res.is_dir():
        return ("model", p_res / "policy.weights.h5")
    msg = f"expected 'random', a directory with policy.weights.h5, or a .h5 file: {spec!r}"
    raise ValueError(msg)


def _load_policy_models(weight_paths: set[Path]) -> dict[Path, object]:
    try:
        import tensorflow as tf  # noqa: PLC0415
        from dungeon_runner.rl import actions_codec, observation  # noqa: PLC0415
        from dungeon_runner.rl.model import DEFAULT_PPO_HIDDEN, PolicyValueModel  # noqa: PLC0415
    except ImportError as err:
        print("Install: pip install -e \".[train]\"", file=sys.stderr)
        raise SystemExit(1) from err

    out: dict[Path, PolicyValueModel] = {}
    for w in weight_paths:
        model = PolicyValueModel(hidden=DEFAULT_PPO_HIDDEN)
        _ = model(
            tf.zeros((1, observation.OBS_DIM), tf.float32),
            tf.zeros((1, actions_codec.N_ACTIONS), tf.float32),
        )
        model.load_weights(str(w))
        out[w] = model
    return out


def _nn_select_action(m: Match, seat: int, model: object, legal: set[object]) -> object:
    from dungeon_runner.rl import actions_codec, observation  # noqa: PLC0415
    from dungeon_runner.rl.ppo import sample_action  # noqa: PLC0415

    oa = observation.build_observation(m, seat)
    mk = actions_codec.legal_mask(m)
    ai, _, _ = sample_action(model, oa, mk)
    a = actions_codec.decode_index(m, ai)
    if a is None or a not in legal:
        msg = f"NN sampled illegal action idx={ai} → {a!r}; legal count={len(legal)}"
        raise SystemExit(msg)
    return a


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
    _ps = (
        "Seat policy: 'random' or logdir / policy.weights.h5 path "
        '(requires pip install -e ".[train]" for checkpoints). Default: random.'
    )
    ap.add_argument("--p0", metavar="SPEC", default=None, help=_ps)
    ap.add_argument("--p1", metavar="SPEC", default=None, help=_ps)
    ap.add_argument("--p2", metavar="SPEC", default=None, help=_ps)
    ap.add_argument("--p3", metavar="SPEC", default=None, help=_ps)
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
    n_pl = int(args.players)
    if not 2 <= n_pl <= 4:
        print(f"--players must be 2–4, got {n_pl}", file=sys.stderr)
        raise SystemExit(2)

    for i in range(4):
        if getattr(args, f"p{i}", None) is not None and i >= n_pl:
            print(f"warning: --p{i} ignored (--players {n_pl})", file=sys.stderr)

    seat_specs: list[tuple[str, Path | None]] = []
    try:
        for i in range(n_pl):
            raw = getattr(args, f"p{i}", None) or "random"
            seat_specs.append(_parse_player_spec(raw))
    except ValueError as e:
        print(e, file=sys.stderr)
        raise SystemExit(2) from e

    model_paths: set[Path] = {w for k, w in seat_specs if k == "model" and w is not None}
    for i, (k, w) in enumerate(seat_specs):
        if k == "model" and w is not None and not w.is_file():
            print(f"seat {i}: missing weights file {w}", file=sys.stderr)
            raise SystemExit(2)

    models_by_path: dict[Path, object] = {}
    if model_paths:
        try:
            import tensorflow as tf  # noqa: PLC0415
        except ImportError as err:
            print("Install: pip install -e \".[train]\"", file=sys.stderr)
            raise SystemExit(1) from err
        if args.seed is not None:
            tf.random.set_seed(int(args.seed))
        models_by_path = _load_policy_models(model_paths)

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
    m = Match.new(n_pl, rng, AdventurerKind.WARRIOR, start_seat=0)
    step = 0

    seed_note = f" · seed {args.seed}" if args.seed is not None else ""
    seat_desc = []
    for i, (k, w) in enumerate(seat_specs):
        seat_desc.append(f"p{i}={'random' if k == 'random' else str(w)}")
    print(f"{n_pl} players · Warrior first · {' · '.join(seat_desc)}{seed_note}")
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
        seat = m.active_seat
        sk, wpath = seat_specs[seat]
        if sk == "random":
            acts_n = random_sim_action_subset(m, acts)
            if not acts_n:
                print("Random-run policy left no moves; stopping.")
                break
            a = pick_action(
                m,
                acts_n,
                rng,
                pass_weight=args.pass_weight,
                sacrifice_weight=args.sacrifice_weight,
            )
        else:
            assert wpath is not None
            a = _nn_select_action(m, seat, models_by_path[wpath], acts)
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
