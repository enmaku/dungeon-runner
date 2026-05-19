"""Model forward helpers for replay metrics and sim benchmarks."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import tensorflow as tf

from dungeon_runner.match import Match
from dungeon_runner.replay.eval.replay_metrics import PredictFn
from dungeon_runner.replay.eval.sim_metrics import SimPolicy
from dungeon_runner.rl import actions_codec, observation
from dungeon_runner.rl.model import PolicyValueModel


def make_replay_predict(model: PolicyValueModel) -> PredictFn:
    def predict(obs: Any, mask: Any) -> int:
        o = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        m = np.asarray(mask, dtype=np.float32).reshape(1, -1)
        logits, _ = model(
            tf.convert_to_tensor(o, tf.float32),
            tf.convert_to_tensor(m, tf.float32),
            training=False,
        )
        return int(tf.argmax(logits[0]).numpy())

    return predict


class KerasSimPolicy:
    def __init__(self, model: PolicyValueModel, *, seat: int = 0) -> None:
        self._model = model
        self._seat = seat

    def select(self, m: Match, actions: set[object], rng: random.Random) -> object:
        del rng
        obs = np.asarray(
            observation.build_observation(m, self._seat),
            dtype=np.float32,
        )
        mask = np.asarray(actions_codec.legal_mask(m), dtype=np.float32)
        predict = make_replay_predict(self._model)
        idx = predict(obs, mask)
        action = actions_codec.decode_index(m, idx)
        if action is None or action not in actions:
            legal = sorted(actions, key=repr)
            return legal[0]
        return action


def load_policy_model(parent_weights: Path) -> PolicyValueModel:
    from dungeon_runner.rl.model import DEFAULT_PPO_HIDDEN, PolicyValueModel

    parent_weights = parent_weights.resolve()
    model = PolicyValueModel(hidden=DEFAULT_PPO_HIDDEN)
    _ = model(
        tf.zeros((1, observation.OBS_DIM), tf.float32),
        tf.zeros((1, actions_codec.N_ACTIONS), tf.float32),
    )
    model.load_weights(str(parent_weights))
    return model
