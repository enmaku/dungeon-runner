"""GAE, PPO update, and rollout collection for :class:`WtdAECEnv`."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dungeon_runner.rl.model import PolicyValueModel


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


@dataclass
class RolloutBatch:
    obs: list[np.ndarray] = field(default_factory=list)
    mask: list[np.ndarray] = field(default_factory=list)
    act: list[int] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    done: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.obs)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    lr: float = 3e-4
    n_epochs: int = 4
    minibatch_size: int = 64


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_v: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """`dones[i]` = True if ``s'`` after step ``i`` is terminal (so ``V(s')=0``; match ended)."""
    n = int(rewards.shape[0])
    if n == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    adv = np.zeros(n, dtype=np.float32)
    lga = 0.0
    dnp = dones.astype(bool)
    for t in range(n - 1, -1, -1):
        if t < n - 1 and dnp[t]:
            nxtv = 0.0
        elif t < n - 1:
            nxtv = float(values[t + 1])
        else:
            nxtv = 0.0 if dnp[t] else float(last_v)
        d = float(rewards[t]) + gamma * nxtv - float(values[t])
        not_term_next_adv = 1.0
        if t < n - 1:
            not_term_next_adv = 0.0 if dnp[t + 1] else 1.0
        lga = d + gamma * lam * not_term_next_adv * lga
        adv[t] = lga
    return adv, adv + values


def ppo_minibatch_update(
    model: PolicyValueModel,
    opt: keras.optimizers.Optimizer,
    cfg: PPOConfig,
    o: np.ndarray,
    m: np.ndarray,
    a: np.ndarray,
    old_logp: np.ndarray,
    _old_v: np.ndarray,
    adv: np.ndarray,
    ret: np.ndarray,
) -> dict[str, float]:
    o_t = tf.constant(o, dtype=tf.float32)
    m_t = tf.constant(m, dtype=tf.float32)
    a_t = tf.constant(a, dtype=tf.int32)
    old_l_t = tf.constant(old_logp, dtype=tf.float32)
    adv_t = tf.constant(adv, dtype=tf.float32)
    ret_t = tf.constant(ret, dtype=tf.float32)
    adv_t = (adv_t - tf.math.reduce_mean(adv_t)) / (tf.math.reduce_std(adv_t) + 1e-8)

    with tf.GradientTape() as tape:
        lg, v_ = model(o_t, m_t, training=True)
        lns = tf.nn.log_softmax(lg, axis=-1)
        pr = tf.exp(lns)
        bidx = tf.range(tf.shape(lg)[0], dtype=a_t.dtype)
        gpi = tf.stack([bidx, a_t], 1)
        nlp = tf.gather_nd(lns, gpi)
        ent = -tf.reduce_sum(pr * lns, axis=-1)
        r_ = tf.exp(nlp - old_l_t)
        c = cfg.clip
        s1 = r_ * adv_t
        s2 = tf.clip_by_value(r_, 1.0 - c, 1.0 + c) * adv_t
        policy_loss = -tf.reduce_mean(tf.minimum(s1, s2))
        v_loss = 0.5 * tf.reduce_mean((tf.squeeze(v_, -1) - ret_t) ** 2)
        ent_m = tf.reduce_mean(ent)
        loss = policy_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_m
    g = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = [
        (h, v) for h, v in zip(g, model.trainable_variables) if h is not None
    ]
    if grads_and_vars:
        opt.apply_gradients(grads_and_vars)  # type: ignore[no-untyped-call, misc]
    return {
        "loss": float(loss),
        "pg": float(policy_loss),
        "vl": float(v_loss),
        "en": float(ent_m),
    }


def sample_action(
    model: PolicyValueModel,
    obs1: np.ndarray,
    mask1: np.ndarray,
) -> tuple[int, float, float]:
    o_t = tf.convert_to_tensor(obs1[None, :], tf.float32)
    m_t = tf.convert_to_tensor(mask1[None, :], tf.float32)
    lg, v_ = model(o_t, m_t, training=False)
    a_t = tf.random.categorical(lg, 1)[:, 0]
    a_i = int(a_t[0].numpy())
    lns = tf.nn.log_softmax(lg, axis=-1)
    nlp_ = float(lns[0, a_i].numpy())
    val_ = float(v_[0, 0].numpy())
    return a_i, nlp_, val_
