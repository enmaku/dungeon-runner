"""BC-anchored PPO training loop for replay pipeline stage."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

UpdateEndFn = Callable[[int, float], None]

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dungeon_runner.replay.bc.human_rows import load_human_rows
from dungeon_runner.replay.bc.predict import make_replay_predict
from dungeon_runner.replay.bc.trainer import masked_accuracy
from dungeon_runner.replay.eval.derived_store import ParquetDerivedRow
from dungeon_runner.replay.ppo.bc_anchor import anchor_ce_loss, anchor_kl_loss
from dungeon_runner.replay.ppo.frozen_teacher import FrozenBCTeacher
from dungeon_runner.replay.ppo.ray_collect import collect_rollouts
from dungeon_runner.replay.ppo.rollout_collector import collect_rollouts_local
from dungeon_runner.replay.ppo.ray_workers import RayRolloutPool
from dungeon_runner.replay.ppo.template_sampler import TEMPLATE_BC_BOT
from dungeon_runner.rl.model import PolicyValueModel
from dungeon_runner.rl.ppo import PPOConfig, compute_gae, ppo_minibatch_update

PPO_MAX_UPDATES = 16
PPO_ROLLOUT_STEPS = 256
PPO_SEED = 17
PPO_PPO_LR = 1e-4
PPO_ANCHOR_LR = 3e-4


@dataclass(frozen=True)
class PPOTrainResult:
    ppo_loss: float
    bc_anchor_ce: float
    bc_anchor_kl: float | None
    best_val_masked_accuracy: float | None = None
    best_update: int = 0


def _apply_anchor_step(
    model: PolicyValueModel,
    rows: list[ParquetDerivedRow],
    *,
    lam: float,
    teacher: PolicyValueModel | None,
    beta: float,
    opt: keras.optimizers.Optimizer,
) -> tuple[float, float | None]:
    ce = anchor_ce_loss(model, rows) if lam > 0 and rows else 0.0
    kl_val: float | None = None
    if beta > 0 and teacher is not None and rows:
        kl_val = anchor_kl_loss(model, teacher, rows, beta=beta)
    total = lam * ce + (kl_val or 0.0)
    if total <= 0:
        return ce, kl_val
    with tf.GradientTape() as tape:
        obs = np.stack([np.asarray(r.obs, np.float32) for r in rows], axis=0)
        masks = np.stack([np.asarray(r.mask, np.float32) for r in rows], axis=0)
        labels = np.array([int(r.policy_action_index) for r in rows], dtype=np.int32)
        logits, _ = model(
            tf.convert_to_tensor(obs, tf.float32),
            tf.convert_to_tensor(masks, tf.float32),
            training=True,
        )
        ce_t = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        )
        loss = lam * ce_t
        if beta > 0 and teacher is not None:
            t_logits, _ = teacher(
                tf.convert_to_tensor(obs, tf.float32),
                tf.convert_to_tensor(masks, tf.float32),
                training=False,
            )
            s_log = tf.nn.log_softmax(logits, axis=-1)
            t_log = tf.nn.log_softmax(t_logits, axis=-1)
            kl = tf.reduce_sum(tf.exp(t_log) * (t_log - s_log), axis=-1)
            loss = loss + beta * tf.reduce_mean(kl)
    grads = tape.gradient(loss, model.trainable_variables)
    pairs = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    if pairs:
        opt.apply_gradients(pairs)  # type: ignore[no-untyped-call, misc]
    return float(ce), kl_val


def _ppo_update_from_batch(
    model: PolicyValueModel,
    opt: keras.optimizers.Optimizer,
    cfg: PPOConfig,
    batch,
    pyr: random.Random,
) -> float:
    n = min(
        len(batch.obs),
        len(batch.reward),
        len(batch.act),
        len(batch.value),
        len(batch.logp),
        len(batch.done),
    )
    if n < 3:
        return 0.0
    o = np.stack(batch.obs[:n], 0)
    m = np.stack(batch.mask[:n], 0)
    actions = np.asarray(batch.act[:n], np.int32)
    values = np.asarray(batch.value[:n], np.float32)
    logp = np.asarray(batch.logp[:n], np.float32)
    rewards = np.asarray(batch.reward[:n], np.float32)
    dones = np.asarray(batch.done[:n], bool)
    _, last_v = model(
        tf.convert_to_tensor(o[-1:, :], tf.float32),
        tf.convert_to_tensor(m[-1:, :], tf.float32),
    )
    last_val = float(last_v[0, 0].numpy())
    adv, rets = compute_gae(rewards, values, dones, last_val, cfg.gamma, cfg.gae_lambda)
    idx = np.arange(n)
    losses: list[float] = []
    for _ in range(cfg.n_epochs):
        pyr.shuffle(idx)
        for start in range(0, n, cfg.minibatch_size):
            sl = idx[start : start + cfg.minibatch_size]
            if sl.size == 0:
                continue
            stats = ppo_minibatch_update(
                model,
                opt,
                cfg,
                o[sl],
                m[sl],
                actions[sl],
                logp[sl],
                values[sl],
                adv[sl],
                rets[sl],
            )
            losses.append(float(stats["loss"]))
    return float(np.mean(losses)) if losses else 0.0


def _log_template_scalar(tb_dir: Path, template: str, stats, step: int) -> None:
    writer = tf.summary.create_file_writer(str(tb_dir))
    suffix = template.replace("_", "-")
    with writer.as_default():
        tf.summary.scalar(f"rollout/{suffix}/episodes", float(stats.n_episodes), step=step)
        tf.summary.scalar(f"rollout/{suffix}/env_steps", float(stats.env_steps), step=step)


def _maybe_anchor(
    model: PolicyValueModel,
    train_rows: list[ParquetDerivedRow],
    *,
    lam: float,
    teacher: PolicyValueModel | None,
    beta: float,
    opt: keras.optimizers.Optimizer,
) -> tuple[float, float | None]:
    if lam <= 0 and beta <= 0:
        return 0.0, None
    return _apply_anchor_step(
        model,
        train_rows,
        lam=lam,
        teacher=teacher,
        beta=beta,
        opt=opt,
    )


def train_ppo(
    model: PolicyValueModel,
    teacher: FrozenBCTeacher,
    train_rows: list[ParquetDerivedRow],
    *,
    tb_dir: Path,
    val_rows: list[ParquetDerivedRow] | None = None,
    teacher_weights: Path | None = None,
    bc_anchor_lambda: float = 0.1,
    bc_anchor_beta: float = 0.0,
    use_ray: bool = True,
    ray_workers: int = 8,
    max_updates: int = PPO_MAX_UPDATES,
    rollout_steps: int = PPO_ROLLOUT_STEPS,
    on_update_end: UpdateEndFn | None = None,
) -> PPOTrainResult:
    pyr = random.Random(PPO_SEED)
    ppo_opt = keras.optimizers.Adam(PPO_PPO_LR)
    anchor_opt = keras.optimizers.Adam(PPO_ANCHOR_LR)
    cfg = PPOConfig()
    teacher_model = teacher._model  # noqa: SLF001
    ppo_losses: list[float] = []
    anchor_ce_vals: list[float] = []
    anchor_kl_vals: list[float] = []
    best_val = -1.0
    best_weights: list[np.ndarray] | None = None
    best_update = 0
    if val_rows:
        best_val = masked_accuracy(make_replay_predict(model), val_rows)
        best_weights = model.get_weights()

    def local_collect():
        batch, stats, template = collect_rollouts_local(
            model, teacher, target_steps=rollout_steps, seed=PPO_SEED
        )
        return batch, stats, template

    ray_pool: RayRolloutPool | None = None
    if use_ray:
        if teacher_weights is None:
            raise ValueError("teacher_weights is required when use_ray=True")
        ray_pool = RayRolloutPool(
            teacher_weights=teacher_weights,
            n_workers=ray_workers,
            seed=PPO_SEED,
        )

    def ray_collect(workers: int):
        del workers
        assert ray_pool is not None
        return ray_pool.collect(model, target_steps=rollout_steps, update_step=step)

    try:
        for step in range(max_updates):
            batch, stats, template = collect_rollouts(
                use_ray=use_ray,
                ray_workers=ray_workers,
                local_fn=local_collect,
                ray_fn=ray_collect,
            )
            ce_pre, kl_pre = _maybe_anchor(
                model,
                train_rows,
                lam=bc_anchor_lambda,
                teacher=teacher_model if bc_anchor_beta > 0 else None,
                beta=bc_anchor_beta,
                opt=anchor_opt,
            )
            loss = _ppo_update_from_batch(model, ppo_opt, cfg, batch, pyr)
            if loss > 0:
                ppo_losses.append(loss)
            ce_post, kl_post = _maybe_anchor(
                model,
                train_rows,
                lam=bc_anchor_lambda,
                teacher=teacher_model if bc_anchor_beta > 0 else None,
                beta=bc_anchor_beta,
                opt=anchor_opt,
            )
            ce = ce_post if bc_anchor_lambda > 0 else ce_pre
            kl = kl_post if bc_anchor_beta > 0 else kl_pre
            if bc_anchor_lambda > 0:
                anchor_ce_vals.append(ce)
            if bc_anchor_beta > 0 and kl is not None:
                anchor_kl_vals.append(kl)
            if val_rows:
                val_acc = masked_accuracy(make_replay_predict(model), val_rows)
                if val_acc > best_val:
                    best_val = val_acc
                    best_update = step + 1
                    best_weights = model.get_weights()
                if tb_dir is not None:
                    writer = tf.summary.create_file_writer(str(tb_dir))
                    with writer.as_default():
                        tf.summary.scalar("val/masked_accuracy", val_acc, step=step)
            if on_update_end is not None:
                on_update_end(step, loss)
            _log_template_scalar(tb_dir, template or TEMPLATE_BC_BOT, stats, step)
            writer = tf.summary.create_file_writer(str(tb_dir))
            with writer.as_default():
                tf.summary.scalar("train/ppo_loss", loss, step=step)
                if bc_anchor_lambda > 0:
                    tf.summary.scalar("train/bc_anchor_ce", ce, step=step)
                if bc_anchor_beta > 0 and kl is not None:
                    tf.summary.scalar("train/bc_anchor_kl", kl, step=step)
    finally:
        if ray_pool is not None:
            ray_pool.shutdown()

    if best_weights is not None:
        model.set_weights(best_weights)

    return PPOTrainResult(
        ppo_loss=float(np.mean(ppo_losses)) if ppo_losses else 0.0,
        bc_anchor_ce=float(np.mean(anchor_ce_vals)) if anchor_ce_vals else 0.0,
        bc_anchor_kl=(
            float(np.mean(anchor_kl_vals)) if anchor_kl_vals else None
        ),
        best_val_masked_accuracy=best_val if val_rows else None,
        best_update=best_update,
    )
