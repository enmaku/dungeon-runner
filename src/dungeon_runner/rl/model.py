"""Keras policy–value network (shared trunk).

**Inputs / outputs (fixed in this codebase)**
* ``obs``: ``(batch, obs_dim)`` with ``obs_dim=87``—hand-built mix of one-hots, normalized
  scalars, and per-seat blocks. No separate embedding table; the net must learn a flat map
  from this vector to (policy, value).
* ``action_mask``: ``(batch, n_actions)`` with ``n_actions=26``; illegal actions get large
  negative logit.
* **Heads** share the same trunk: policy logits and scalar value. No dueling, no
  per-phase heads (those would be a larger design change).

**Tunable *architecture* knobs (below)** — width, depth, LayerNorm, and activation. These
affect *capacity* and *optimization* (gradient scale). They do *not* replace a richer
observation or structured encoder (e.g. attention over cards) if the task needs it.

**Training (not in this file)** is governed by PPO/optimizer settings in
``rl/ppo.PPOConfig`` and ``scripts/train.py`` (learning rate, PPO clip, GAE, rollout size,
``vf_coef`` / ``ent_coef``, number of SGD steps per update). Those are as important as
hidden width for sample efficiency.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dungeon_runner.rl import actions_codec, observation

LOGIT_MASK = -1.0e9


def _mlp_trunk(
    hidden: tuple[int, ...],
    use_layer_norm: bool,
    activation: str,
) -> keras.layers.Layer:
    layers: list[keras.layers.Layer] = []
    for i, w in enumerate(hidden):
        layers.append(keras.layers.Dense(w, activation=None, name=f"hidden_{i + 1}"))
        if use_layer_norm:
            layers.append(keras.layers.LayerNormalization(name=f"ln_{i + 1}"))
        if activation:
            layers.append(keras.layers.Activation(activation, name=f"act_{i + 1}"))
    return keras.Sequential(layers, name="trunk")


class PolicyValueModel(keras.Model):
    def __init__(
        self,
        obs_dim: int = observation.OBS_DIM,
        n_actions: int = actions_codec.N_ACTIONS,
        hidden: tuple[int, ...] = (512, 512, 256),
        use_layer_norm: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self._obs_dim = obs_dim
        self._n_actions = n_actions
        if not hidden:
            msg = "hidden must be a non-empty tuple of layer widths"
            raise ValueError(msg)
        self.trunk = _mlp_trunk(hidden, use_layer_norm, activation)
        self.logits = keras.layers.Dense(n_actions, name="logits")
        self.value = keras.layers.Dense(1, name="value")

    def call(
        self,
        obs: tf.Tensor,
        action_mask: tf.Tensor | None = None,
        training: bool = False,  # noqa: ARG002
    ) -> tuple[tf.Tensor, tf.Tensor]:
        h = self.trunk(obs)
        lg = self.logits(h)
        v = self.value(h)
        if action_mask is not None:
            am = tf.cast(action_mask, tf.float32)
            lg = lg + (1.0 - am) * float(LOGIT_MASK)
        return lg, v


def apply_mask_numpy(logits: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    return logits + (1.0 - action_mask) * float(LOGIT_MASK)
