"""Self-play PPO with **Keras** :class:`PolicyValueModel` under **Ray** parallel rollouts.

**Why not `ray.rllib` PPO for TF2 here:** Ray 2.5+ ships PPO on the *new* API stack for
`torch` only. Disabling the new stack to use the legacy ``tf2`` PPO still hits
incompatibilities with the current Keras/TF combination in this environment, and
:class:`PPOTfRLModule` is not offered as a drop-in for custom Keras weights. This
module (and :mod:`scripts.train_rllib`) therefore uses **Ray tasks/actors** only for
**distributed sampling**; optimization stays the same hand-rolled Keras PPO as
``train.py`` so :func:`Model.save_weights` / :func:`Model.load_weights` round-trip.

Weights are portable; optimizer (Adam) state is *not* shared when switching
``train_rllib`` ↔ ``train``.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from dungeon_runner.match import MatchPhase
from dungeon_runner.pettingzoo_aec import WtdAECEnv
from dungeon_runner.rl import actions_codec, observation
from dungeon_runner.rl.model import DEFAULT_PPO_HIDDEN, PolicyValueModel
from dungeon_runner.rl.ppo import RolloutBatch, RolloutGameStats, sample_action
from dungeon_runner.types_core import AdventurerKind


def default_policy_h5_path(logdir: Path) -> Path:
    return logdir / "policy.weights.h5"


def build_policy_value_model() -> PolicyValueModel:
    model = PolicyValueModel(hidden=DEFAULT_PPO_HIDDEN)
    _ = model(
        tf.zeros((1, observation.OBS_DIM), tf.float32),
        tf.zeros((1, actions_codec.N_ACTIONS), tf.float32),
    )
    return model


def model_weights_to_numpy(model: PolicyValueModel) -> list[np.ndarray]:
    w = model.get_weights()
    return [np.asarray(x) for x in w]


def set_model_weights_numpy(model: PolicyValueModel, weights: list[np.ndarray]) -> None:
    model.set_weights(weights)


def load_policy_weights_h5_if_present(model: PolicyValueModel, path: Path) -> bool:
    if not path.is_file():
        return False
    model.load_weights(path)
    return True


def save_policy_weights_h5(model: PolicyValueModel, path: Path) -> None:
    model.save_weights(path)


def sample_episode_config(
    np_r: np.random.Generator,
) -> tuple[int, list[bool], int, AdventurerKind]:
    n = int(np_r.integers(2, 5))
    roles = [True] * n
    st = int(np_r.integers(0, n))
    h0 = AdventurerKind(int(np_r.integers(0, 4)))
    return n, roles, st, h0


def fill_rollout_selfplay(
    env: WtdAECEnv,
    model: PolicyValueModel,
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
        if m0 is None or m0.phase is MatchPhase.ENDED or (env.agents and all(
            env.terminations.get(ag) for ag in env.agents
        )):
            n, new_roles, st, h0 = sample_episode_config(np_r)
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
        if o is None or s >= m0.n_players or s >= len(roles):  # type: ignore[union-attr, misc]
            break
        oa, mk = o["obs"], o["action_mask"]
        if not (mk > 0).any():
            break
        if s >= len(roles):
            break
        ai, nlp, val = sample_action(model, oa, mk)
        b.obs.append(oa.copy())
        b.mask.append(mk.copy())
        b.act.append(ai)
        b.logp.append(nlp)
        b.value.append(val)
        env.step(int(ai))
        g.env_steps += 1
        re = float((env.rewards or {}).get(sel) or 0.0)
        m1 = env._m  # noqa: SLF001
        done = bool(m1 and m1.phase is MatchPhase.ENDED)
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


def merge_batches(parts: list[RolloutBatch]) -> RolloutBatch:
    o = RolloutBatch()
    for p in parts:
        o.obs.extend(p.obs)
        o.mask.extend(p.mask)
        o.act.extend(p.act)
        o.reward.extend(p.reward)
        o.value.extend(p.value)
        o.logp.extend(p.logp)
        o.done.extend(p.done)
    return o


def merge_game_stats(parts: list[RolloutGameStats]) -> RolloutGameStats:
    g0 = RolloutGameStats()
    for g in parts:
        g0.n_episodes += g.n_episodes
        g0.n_decided += g.n_decided
        g0.n_truncated += g.n_truncated
        g0.episode_lengths.extend(g.episode_lengths)
        g0.n_all_nn += g.n_all_nn
        g0.nn_wins += g.nn_wins
        g0.nn_games += g.nn_games
        g0.env_steps += g.env_steps
    return g0


def pack_rollout_for_ray(b: RolloutBatch) -> dict[str, object]:
    if not b.obs:
        return {"n": 0}
    return {
        "n": int(len(b.obs)),
        "obs": np.stack(b.obs, 0).astype(np.float32, copy=False),
        "mask": np.stack(b.mask, 0).astype(np.float32, copy=False),
        "act": np.asarray(b.act, np.int32),
        "value": np.asarray(b.value, np.float32),
        "logp": np.asarray(b.logp, np.float32),
        "reward": np.asarray(b.reward, np.float32),
        "done": np.asarray(b.done, bool),
    }


def unpack_ray_rollout(packed: dict[str, object]) -> RolloutBatch:
    b = RolloutBatch()
    n = int(packed.get("n", 0))
    if n <= 0:
        return b
    o = packed["obs"]
    m = packed["mask"]
    for i in range(n):
        b.obs.append(o[i].copy())
        b.mask.append(m[i].copy())
    b.act.extend(list(packed["act"].tolist()))  # type: ignore[union-attr, index]
    b.value.extend([float(x) for x in packed["value"]])  # type: ignore[union-attr, index]
    b.logp.extend([float(x) for x in packed["logp"]])  # type: ignore[union-attr, index]
    b.reward.extend([float(x) for x in packed["reward"]])  # type: ignore[union-attr, index]
    b.done.extend([bool(x) for x in packed["done"]])  # type: ignore[union-attr, index]
    return b


def pack_game_stats_for_ray(g: RolloutGameStats) -> dict[str, int | list[int]]:
    return {
        "n_episodes": g.n_episodes,
        "n_decided": g.n_decided,
        "n_truncated": g.n_truncated,
        "episode_lengths": list(g.episode_lengths),
        "n_all_nn": g.n_all_nn,
        "nn_wins": g.nn_wins,
        "nn_games": g.nn_games,
        "env_steps": g.env_steps,
    }


def unpack_game_stats(packed: dict[str, int | list[int]]) -> RolloutGameStats:
    return RolloutGameStats(
        n_episodes=int(packed["n_episodes"]),
        n_decided=int(packed["n_decided"]),
        n_truncated=int(packed["n_truncated"]),
        episode_lengths=list(packed.get("episode_lengths", [])),  # type: ignore[arg-type]
        n_all_nn=int(packed["n_all_nn"]),
        nn_wins=int(packed["nn_wins"]),
        nn_games=int(packed["nn_games"]),
        env_steps=int(packed["env_steps"]),
    )
