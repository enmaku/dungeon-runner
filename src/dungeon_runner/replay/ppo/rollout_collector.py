"""Python training sim rollouts for pipeline PPO (template-aware)."""

from __future__ import annotations

import random

import numpy as np

from dungeon_runner.bots import RandomBot
from dungeon_runner.match import MatchPhase
from dungeon_runner.pettingzoo_aec import WtdAECEnv
from dungeon_runner.replay.ppo.frozen_teacher import FrozenBCTeacher
from dungeon_runner.replay.ppo.template_sampler import (
    TEMPLATE_BC_BOT,
    TEMPLATE_SELF_PLAY,
    TEMPLATE_VS_RANDOMBOT,
    roles_for_template,
    sample_rollout_template,
)
from dungeon_runner.rl import actions_codec
from dungeon_runner.rl.model import PolicyValueModel
from dungeon_runner.rl.ppo import RolloutBatch, RolloutGameStats, sample_action
from dungeon_runner.types_core import AdventurerKind


def sample_episode_config(
    np_r: np.random.Generator,
    pyr: random.Random,
) -> tuple[int, list[bool], int, AdventurerKind, str]:
    n = int(np_r.integers(2, 5))
    template = sample_rollout_template(pyr)
    roles = roles_for_template(template, n, pyr)
    st = int(np_r.integers(0, n))
    h0 = AdventurerKind(int(np_r.integers(0, 4)))
    return n, roles, st, h0, template


def fill_rollout(
    env: WtdAECEnv,
    model: PolicyValueModel,
    *,
    teacher: FrozenBCTeacher,
    random_bot: RandomBot,
    roles: list[bool],
    template: str,
    pyr: random.Random,
    np_r: np.random.Generator,
    target: int,
) -> tuple[RolloutBatch, list[bool], RolloutGameStats, str]:
    batch = RolloutBatch()
    stats = RolloutGameStats()
    current_template = template
    safety = 0
    while len(batch) < target and safety < 50_000:
        safety += 1
        match = env._m  # noqa: SLF001
        if match is None or match.phase is MatchPhase.ENDED or (
            env.agents and all(env.terminations.get(ag) for ag in env.agents)
        ):
            n, roles, st, h0, current_template = sample_episode_config(np_r, pyr)
            env.reset(
                seed=int(np_r.integers(0, 2**30)),
                options={
                    "n_players": n,
                    "start_seat": st,
                    "first_hero": h0,
                    "max_episode_steps": 20_000,
                },
            )
            match = env._m  # noqa: SLF001
        match = env._m  # noqa: SLF001
        if match is None or match.phase is MatchPhase.ENDED:
            continue
        sel = env.agent_selection
        seat = int(sel)
        obs_pack = env.observe(sel)
        if obs_pack is None or seat >= match.n_players or seat >= len(roles):
            break
        obs_arr, mask_arr = obs_pack["obs"], obs_pack["action_mask"]
        if not (mask_arr > 0).any():
            break
        if roles[seat] and seat < len(roles):
            action_idx, nlp, val = sample_action(model, obs_arr, mask_arr)
            batch.obs.append(obs_arr.copy())
            batch.mask.append(mask_arr.copy())
            batch.act.append(action_idx)
            batch.logp.append(nlp)
            batch.value.append(val)
        else:
            if current_template == TEMPLATE_BC_BOT:
                action_idx = teacher.select_masked(obs_arr, mask_arr)
            else:
                action_obj = random_bot.select(match, match.legal_actions(), pyr)
                action_idx = actions_codec.encode_action(match, action_obj)
        env.step(int(action_idx))
        stats.env_steps += 1
        reward = float((env.rewards or {}).get(sel) or 0.0)
        match_after = env._m  # noqa: SLF001
        done = bool(match_after and match_after.phase is MatchPhase.ENDED)
        if roles[seat] and seat < len(roles):
            batch.reward.append(reward)
            batch.done.append(done)
        if match_after is not None:
            truncated = any(env.truncations.get(ag) for ag in env.agents)
            ended = match_after.phase is MatchPhase.ENDED
            if ended or truncated:
                n_pl = int(match_after.n_players)
                seat_roles = list(roles[:n_pl]) if len(roles) >= n_pl else [False] * n_pl
                stats.n_episodes += 1
                stats.episode_lengths.append(int(env._step_i))  # noqa: SLF001
                if all(seat_roles):
                    stats.n_all_nn += 1
                if truncated and not ended:
                    stats.n_truncated += 1
                if ended:
                    stats.n_decided += 1
                    winner = match_after.winner_seat
                    if winner is not None and 0 <= winner < n_pl and any(seat_roles):
                        stats.nn_games += 1
                        if seat_roles[int(winner)]:
                            stats.nn_wins += 1
    return batch, roles, stats, current_template


def collect_rollouts_local(
    model: PolicyValueModel,
    teacher: FrozenBCTeacher,
    *,
    target_steps: int,
    seed: int = 0,
) -> tuple[RolloutBatch, RolloutGameStats, str]:
    env = WtdAECEnv()
    pyr = random.Random(seed)
    np_r = np.random.default_rng(seed)
    n, roles, st, h0, template = sample_episode_config(np_r, pyr)
    env.reset(
        seed=int(np_r.integers(0, 2**30)),
        options={"n_players": n, "start_seat": st, "first_hero": h0},
    )
    batch, _, stats, last_template = fill_rollout(
        env,
        model,
        teacher=teacher,
        random_bot=RandomBot(),
        roles=roles,
        template=template,
        pyr=pyr,
        np_r=np_r,
        target=target_steps,
    )
    return batch, stats, last_template
