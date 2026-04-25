"""PettingZoo AEC environment wrapping :class:`Match`."""

from __future__ import annotations

import random
import typing

import numpy as np
from gymnasium import spaces

import dungeon_runner.actions as A
from pettingzoo import AECEnv
from pettingzoo.utils import env as pz_env
from pettingzoo.utils import wrappers

from dungeon_runner.match import Match, MatchPhase, MatchTerminalReason
from dungeon_runner.rl import actions_codec, observation
from dungeon_runner.rl import rewards as R
from dungeon_runner.types_core import AdventurerKind

AGENT_ID = str
MAX_SEATS = 4


def env(**kwargs: typing.Any) -> pz_env.AECEnv:
    e: pz_env.AECEnv = raw_env(**kwargs)  # type: ignore[assignment, misc]
    e = wrappers.AssertOutOfBoundsWrapper(e)  # type: ignore[assignment, arg-type]
    e = wrappers.OrderEnforcingWrapper(e)  # type: ignore[assignment, arg-type]
    return e


def raw_env(
    max_episode_steps: int = 20_000,
) -> pz_env.AECEnv:
    return WtdAECEnv(max_episode_steps=max_episode_steps)


def _ob_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "obs": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(observation.OBS_DIM,),
                dtype=np.float32,
            ),
            "action_mask": spaces.Box(0.0, 1.0, shape=(actions_codec.N_ACTIONS,), dtype=np.float32),
        }
    )


def _act_space() -> spaces.Space[typing.SupportsInt]:
    return spaces.Discrete(actions_codec.N_ACTIONS)


class WtdAECEnv(
    AECEnv,
):
    """Variable 2–4 player match. Only ``agents[0..n)`` are in play; other ``possible`` seats are pre-terminated."""

    metadata: typing.ClassVar[dict[str, typing.Any]] = {
        "is_parallelizable": False,
        "name": "dungeon_wtd_v0",
        "render_modes": [],
    }

    def __init__(self, max_episode_steps: int = 20_000) -> None:
        super().__init__()
        self._max_episode_steps = max_episode_steps
        self.possible_agents: list[AGENT_ID] = [str(i) for i in range(MAX_SEATS)]
        self._obs_sp = _ob_space()
        self._act_sp = _act_space()
        self.observation_spaces: dict[AGENT_ID, spaces.Space[typing.Any]] = {
            a: self._obs_sp for a in self.possible_agents
        }
        self.action_spaces: dict[AGENT_ID, spaces.Space[typing.SupportsInt]] = {a: self._act_sp for a in self.possible_agents}
        self._m: Match | None = None
        self._step_i = 0
        self.agents: list[AGENT_ID] = []
        self.terminations: dict[AGENT_ID, bool] = {}
        self.truncations: dict[AGENT_ID, bool] = {}
        self.rewards: dict[AGENT_ID, float] = {}
        self._cumulative_rewards: dict[AGENT_ID, float] = {}
        self.infos: dict[AGENT_ID, dict[str, typing.Any]] = {}
        self.agent_selection: AGENT_ID = "0"

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> None:
        opt: dict = options or {}
        n_players = int(opt.get("n_players", 3))
        if not 2 <= n_players <= 4:
            msg = f"n_players must be 2-4, got {n_players}"
            raise ValueError(msg)
        pyr: random.Random = (
            random.Random(int(seed)) if seed is not None else random.Random()
        )
        self._m = Match.new(
            n_players,
            pyr,
            opt.get("first_hero", AdventurerKind.WARRIOR),
            int(opt.get("start_seat", 0)),
        )
        self._max_episode_steps = int(opt.get("max_episode_steps", self._max_episode_steps))
        self.agents = [str(i) for i in range(n_players)]
        self._step_i = 0
        for a in self.possible_agents:
            in_game = a in self.agents
            self.terminations[a] = not in_game
            self.truncations[a] = not in_game
        for a in self.agents:
            self.terminations[a] = False
            self.truncations[a] = False
        for a in self.possible_agents:
            self._cumulative_rewards[a] = 0.0
        self.rewards = {a: 0.0 for a in self.possible_agents}
        self.infos = {a: {} for a in self.possible_agents}
        m = self._m
        assert m is not None
        if m.phase is not MatchPhase.ENDED:
            self.agent_selection = str(m.active_seat)
        for a in self.possible_agents:
            oa = self.observe(a)
            if oa:
                self.infos[a] = {"action_mask": oa["action_mask"].copy()}

    def observe(self, agent: AGENT_ID) -> dict[str, np.ndarray] | None:
        m = self._m
        if m is None or int(agent) >= m.n_players or agent not in self.possible_agents:
            return {
                "obs": np.zeros((observation.OBS_DIM,), dtype=np.float32),
                "action_mask": np.zeros((actions_codec.N_ACTIONS,), dtype=actions_codec.MASK_DTYPE),
            }
        seat = int(agent)
        return {
            "obs": observation.build_observation(m, seat),
            "action_mask": actions_codec.legal_mask(m)
            if m.phase is not MatchPhase.ENDED
            else np.zeros((actions_codec.N_ACTIONS,), dtype=actions_codec.MASK_DTYPE),
        }

    def _accumulate_rewards(self) -> None:
        for a in self.agents:
            r = self.rewards.get(a, 0.0) or 0.0
            self._cumulative_rewards[a] = self._cumulative_rewards.get(a, 0.0) + r

    def step(self, action: int) -> None:
        m0 = self._m
        if m0 is None or m0.phase is MatchPhase.ENDED:
            return
        phase_b = m0.phase
        rnr0 = m0.runner_seat
        p_succ0 = m0.players[rnr0].success_cards if rnr0 is not None else 0
        p_aid0 = m0.players[rnr0].aid_flips if rnr0 is not None else 0
        adec = actions_codec.decode_index(m0, int(action))
        if adec is None or adec not in m0.legal_actions():
            err = f"invalid action {action!r} → {adec!r} legal {m0.legal_actions()!r}"
            raise ValueError(err)
        actor = self.agent_selection
        m0.apply(typing.cast(A.Action, adec))  # type: ignore[unused-ignore]
        self._step_i += 1
        m = self._m
        assert m is not None
        for a in self.possible_agents:
            in_game = a in self.agents
            if not in_game and a in self.terminations and a in self.truncations:
                self.terminations[a] = True
                self.truncations[a] = True
        for a in self.agents:
            if a in self.terminations and a in self.truncations:
                self.terminations[a] = False
                self.truncations[a] = False
        self.rewards = {a: 0.0 for a in self.possible_agents}
        if m.phase is MatchPhase.ENDED:
            w = m.winner_seat
            tr = m.terminal_reason
            for a in self.agents:
                s = int(a)
                if w is not None and s == w:
                    if tr is MatchTerminalReason.SECOND_SUCCESS:
                        self.rewards[a] = R.REWARD_MATCH_WIN_SUCCESS
                    else:
                        self.rewards[a] = R.REWARD_MATCH_WIN_STANDING
                else:
                    self.rewards[a] = R.REWARD_MATCH_LOSE
            if rnr0 is not None and phase_b is MatchPhase.DUNGEON and m.players[rnr0].success_cards > p_succ0:
                rk = str(rnr0)
                if rk in self.rewards:
                    self.rewards[rk] = (self.rewards.get(rk) or 0.0) + R.REWARD_DUNGEON_SUCCESS
        elif rnr0 is not None and phase_b is MatchPhase.DUNGEON and m.phase is not MatchPhase.DUNGEON:
            pr = m.players[rnr0]
            if pr.success_cards > p_succ0:
                self.rewards[str(rnr0)] = R.REWARD_DUNGEON_SUCCESS
            elif pr.aid_flips > p_aid0 or pr.eliminated:
                self.rewards[str(rnr0)] = R.REWARD_DUNGEON_FAIL
        if m.phase is not MatchPhase.ENDED and actor in self.agents:
            self.rewards[actor] = (self.rewards.get(actor) or 0.0) + R.REWARD_PER_ACTION
        self._accumulate_rewards()
        if m.phase is MatchPhase.ENDED:
            for a in self.agents:
                self.terminations[a] = True
        elif self._step_i >= self._max_episode_steps:
            for a in self.agents:
                self.truncations[a] = True
        if m.phase is not MatchPhase.ENDED and self._step_i < self._max_episode_steps:
            self.agent_selection = str(m.active_seat)
        m3 = self._m
        assert m3 is not None
        for a in self.possible_agents:
            o2 = self.observe(a)
            if o2 and int(a) < m3.n_players:
                self.infos[a] = {"action_mask": o2["action_mask"].copy()}

    def render(self) -> None:  # noqa: A003
        return None

    def state(self) -> np.ndarray:  # noqa: A003
        raise NotImplementedError

    def close(self) -> None:
        self._m = None
