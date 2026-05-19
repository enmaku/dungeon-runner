"""Frozen BC teacher for BC-bot seats and optional anchor KL."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf

from dungeon_runner.rl.model import PolicyValueModel

LoadModelFn = Callable[[Path], PolicyValueModel]


class FrozenBCTeacher:
    def __init__(self, model: PolicyValueModel) -> None:
        self._model = model

    @classmethod
    def from_weights(
        cls,
        weights: Path,
        *,
        load_model: LoadModelFn,
    ) -> FrozenBCTeacher:
        return cls(load_model(weights))

    def select_masked(self, obs: np.ndarray, mask: np.ndarray) -> int:
        o = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        m = np.asarray(mask, dtype=np.float32).reshape(1, -1)
        logits, _ = self._model(
            tf.convert_to_tensor(o, tf.float32),
            tf.convert_to_tensor(m, tf.float32),
            training=False,
        )
        lg = logits[0].numpy()
        mk = m[0] > 0
        lg = np.where(mk, lg, -1e9)
        return int(np.argmax(lg))
