#!/usr/bin/env python3
"""PPO training: shared policy vs weighted-random bot, random 2–4p.

Requires: ``pip install -e ".[train]"``
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dungeon_runner.bots import RandomBot
from dungeon_runner.match import MatchPhase
from dungeon_runner.pettingzoo_aec import WtdAECEnv
from dungeon_runner.rl import actions_codec, observation
from dungeon_runner.rl.model import PolicyValueModel
from dungeon_runner.rl.ppo import PPOConfig, RolloutBatch, compute_gae, ppo_minibatch_update, sample_action
from dungeon_runner.types_core import AdventurerKind


def sample_episode(rng: np.random.Generator) -> tuple[int, list[bool], int, AdventurerKind]:
    n = int(rng.integers(2, 5))
    roles = [bool(rng.random() < 0.5) for _ in range(n)]
    if not any(roles):
        roles[int(rng.integers(0, n))] = True
    st = int(rng.integers(0, n))
    h = AdventurerKind(int(rng.integers(0, 4)))
    return n, roles, st, h


@dataclass
class RolloutGameStats:
    n_episodes: int = 0
    n_decided: int = 0
    n_truncated: int = 0
    episode_lengths: list[int] = field(default_factory=list)
    n_all_nn: int = 0
    nn_wins: int = 0
    nn_games: int = 0
    env_steps: int = 0


def _log_game_scalars(step: int, g: RolloutGameStats, mean_r: float) -> None:
    ne = int(g.n_episodes)
    tf.summary.scalar("rollout/mean_reward", float(mean_r), step=step)
    tf.summary.scalar("rollout/env_steps", float(g.env_steps), step=step)
    tf.summary.scalar("game/episodes_ended", float(ne), step=step)
    if ne:
        tf.summary.scalar("game/mean_episode_length", float(np.mean(g.episode_lengths)), step=step)
        tf.summary.scalar("game/fraction_all_nn", float(g.n_all_nn) / ne, step=step)
        tf.summary.scalar("game/fraction_mixed_bot", 1.0 - float(g.n_all_nn) / ne, step=step)
        tf.summary.scalar("game/fraction_natural_end", float(g.n_decided) / ne, step=step)
        wr = float(g.nn_wins) / float(g.nn_games) if g.nn_games else 0.0
        tf.summary.scalar("game/nn_win_rate", wr, step=step)
        tf.summary.scalar("game/truncation_rate", float(g.n_truncated) / float(ne), step=step)


def fill_rollout(
    env: WtdAECEnv,
    model: PolicyValueModel,
    bot: RandomBot,
    roles: list[bool],
    pyr: random.Random,
    np_r: np.random.Generator,
    target: int,
) -> tuple[RolloutBatch, list[bool], RolloutGameStats]:
    b = RolloutBatch()
    g = RolloutGameStats()
    safety = 0
    while len(b) < target and safety < 50_000:
        safety += 1
        m0 = env._m  # noqa: SLF001
        if m0 is None or m0.phase is MatchPhase.ENDED or (env.agents and all(  # type: ignore[union-attr, misc]
            env.terminations.get(ag) for ag in env.agents
        )):
            n, new_roles, st, h0 = sample_episode(np_r)
            env.reset(
                seed=int(np_r.integers(0, 2**30)),
                options={"n_players": n, "start_seat": st, "first_hero": h0, "max_episode_steps": 20_000},
            )
            roles = new_roles
            m0 = env._m
        m0 = env._m  # noqa: SLF001
        if m0 is None or m0.phase is MatchPhase.ENDED:  # type: ignore[union-attr]
            continue
        sel = env.agent_selection
        s = int(sel)
        o = env.observe(sel)
        if o is None or s >= m0.n_players or s >= len(roles):  # type: ignore[union-attr]
            break
        oa, mk = o["obs"], o["action_mask"]
        if not (mk > 0).any():
            break
        if roles[s] and s < len(roles):
            ai, nlp, val = sample_action(model, oa, mk)
            b.obs.append(oa.copy())
            b.mask.append(mk.copy())
            b.act.append(ai)
            b.logp.append(nlp)
            b.value.append(val)
        else:
            a_obj = bot.select(m0, m0.legal_actions(), pyr)  # type: ignore[union-attr]
            ai = actions_codec.encode_action(m0, a_obj)  # type: ignore[arg-type, union-attr]
        env.step(int(ai))
        g.env_steps += 1
        re = float((env.rewards or {}).get(sel) or 0.0)  # noqa: SIM201, SIM201, SIM201
        m1 = env._m  # noqa: SLF001
        done = bool(m1 and m1.phase is MatchPhase.ENDED)
        if roles[s] and s < len(roles):
            b.reward.append(re)
            b.done.append(done)
        if m1 is not None:
            tr = any(env.truncations.get(ag) for ag in env.agents)
            ended = m1.phase is MatchPhase.ENDED
            if ended or tr:
                n_pl = int(m1.n_players)
                rloc = list(roles[:n_pl]) if len(roles) >= n_pl else [False] * n_pl
                g.n_episodes += 1
                g.episode_lengths.append(int(env._step_i))  # noqa: SLF001
                if all(rloc):
                    g.n_all_nn += 1
                if tr and not ended:
                    g.n_truncated += 1
                if ended:
                    g.n_decided += 1
                    w = m1.winner_seat
                    if w is not None and 0 <= w < n_pl and any(rloc):
                        g.nn_games += 1
                        if rloc[int(w)]:
                            g.nn_wins += 1
    return b, roles, g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, default=Path("runs/ppo_wtd"))
    ap.add_argument(
        "--rollout",
        type=int,
        default=256,
        help="Target number of NN transitions to collect per PPO update (may end early on env limits).",
    )
    ap.add_argument(
        "--updates",
        type=int,
        default=10_000,
        help="Number of PPO update steps (one rollout + optimization each). Use a small value (e.g. 5) for a smoke test.",
    )
    ap.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print loss to stdout every N updates (also prints the first and last).",
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Write logdir/policy.weights.h5 every N updates; 0 = only at the end.",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tf.random.set_seed(args.seed)
    np_r = np.random.default_rng(args.seed)
    pyr = random.Random(args.seed)
    env = WtdAECEnv()
    n, roles, st, h0 = sample_episode(np_r)
    env.reset(seed=int(np_r.integers(0, 2**30)), options={"n_players": n, "start_seat": st, "first_hero": h0})
    model = PolicyValueModel(hidden=(512, 512, 512, 256))
    _ = model(  # build
        tf.zeros((1, observation.OBS_DIM), tf.float32),
        tf.zeros((1, actions_codec.N_ACTIONS), tf.float32),
    )
    opt = keras.optimizers.Adam(3e-4)
    bot = RandomBot()
    cfg = PPOConfig()
    args.logdir.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(args.logdir / "scalars"))  # noqa: SIM201
    t0 = time.time()
    ck = args.logdir / "policy.weights.h5"
    for _u in range(args.updates):
        b, roles, game = fill_rollout(env, model, bot, roles, pyr, np_r, args.rollout)
        n = min(len(b.obs), len(b.reward), len(b.act), len(b.value), len(b.logp), len(b.done))
        if n < 3:
            n2, roles, st, h0 = sample_episode(np_r)
            env.reset(
                seed=int(np_r.integers(0, 2**30)),
                options={"n_players": n2, "start_seat": st, "first_hero": h0, "max_episode_steps": 20_000},
            )
            with writer.as_default():
                tf.summary.scalar("rollout/skipped", 1.0, step=_u)
                r_sk = np.asarray(b.reward[:n], np.float32)
                _log_game_scalars(_u, game, float(np.mean(r_sk)) if n else 0.0)
            continue
        o = np.stack(b.obs[:n], 0)
        m = np.stack(b.mask[:n], 0)
        a_ = np.asarray(b.act[:n], np.int32)
        v = np.asarray(b.value[:n], np.float32)
        lp = np.asarray(b.logp[:n], np.float32)
        r = np.asarray(b.reward[:n], np.float32)
        d = np.asarray(b.done[:n], bool)
        mean_r = float(np.mean(r)) if n else 0.0
        _, lvv = model(tf.convert_to_tensor(o[-1:, :], tf.float32), tf.convert_to_tensor(m[-1:, :], tf.float32))
        lastv = float(lvv[0, 0].numpy())
        adv, rets = compute_gae(r, v, d, lastv, cfg.gamma, cfg.gae_lambda)
        idx = np.arange(n)
        pyr.shuffle(idx)
        acc: dict[str, float] = {"loss": 0.0, "pg": 0.0, "vl": 0.0, "en": 0.0}
        mb_n = 0
        for _e in range(cfg.n_epochs):
            pyr.shuffle(idx)
            for s0 in range(0, n, cfg.minibatch_size):
                sl = idx[s0 : s0 + cfg.minibatch_size]
                if sl.size == 0:
                    continue
                lg = ppo_minibatch_update(
                    model, opt, cfg, o[sl], m[sl], a_[sl], lp[sl], v[sl], adv[sl], rets[sl]
                )
                for k in acc:
                    acc[k] += float(lg[k])
                mb_n += 1
        if mb_n:
            for k in acc:
                acc[k] /= mb_n
        with writer.as_default():
            tf.summary.scalar("loss/loss", acc["loss"], step=_u)
            tf.summary.scalar("loss/policy", acc["pg"], step=_u)
            tf.summary.scalar("loss/value", acc["vl"], step=_u)
            tf.summary.scalar("loss/entropy", acc["en"], step=_u)
            tf.summary.scalar("rollout/nn_transitions", float(n), step=_u)
            _log_game_scalars(_u, game, mean_r)
        le = max(1, int(args.log_every))
        if _u % le == 0 or _u == args.updates - 1:
            ngm = int(game.nn_games)
            nn_wr = float(game.nn_wins) / float(ngm) if ngm else 0.0
            tf.print(
                "u",
                _u + 1,
                "/",
                args.updates,
                "n",
                n,
                "loss",
                acc["loss"],
                "nn_wr",
                nn_wr,
                output_stream=sys.stdout,
            )
        se = int(args.save_every)
        if se > 0 and (_u + 1) % se == 0:
            model.save_weights(ck)
            tf.print("checkpoint", str(ck), "step", _u + 1, output_stream=sys.stdout)
    tf.print("elapsed", time.time() - t0, "s", output_stream=sys.stdout)
    model.save_weights(ck)
    tf.print("wrote", str(ck), output_stream=sys.stdout)


if __name__ == "__main__":
    try:
        main()
    except ImportError as err:
        print("Install: pip install -e \".[train]\"", file=sys.stderr)
        raise SystemExit(1) from err
