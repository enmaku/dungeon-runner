"""BC anchor CE on human rows and optional KL vs frozen teacher."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from dungeon_runner.replay.bc.trainer import _row_arrays
from dungeon_runner.replay.eval.derived_store import ParquetDerivedRow
from dungeon_runner.rl.model import PolicyValueModel


def anchor_ce_loss(model: PolicyValueModel, rows: list[ParquetDerivedRow]) -> float:
    if not rows:
        return 0.0
    obs, masks, labels = _row_arrays(rows)
    logits, _ = model(
        tf.convert_to_tensor(obs, tf.float32),
        tf.convert_to_tensor(masks, tf.float32),
        training=False,
    )
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    )
    return float(loss.numpy())


def anchor_kl_loss(
    model: PolicyValueModel,
    teacher: PolicyValueModel,
    rows: list[ParquetDerivedRow],
    *,
    beta: float,
) -> float:
    if beta <= 0 or not rows:
        return 0.0
    obs, masks, _ = _row_arrays(rows)
    student_logits, _ = model(
        tf.convert_to_tensor(obs, tf.float32),
        tf.convert_to_tensor(masks, tf.float32),
        training=False,
    )
    teacher_logits, _ = teacher(
        tf.convert_to_tensor(obs, tf.float32),
        tf.convert_to_tensor(masks, tf.float32),
        training=False,
    )
    s_log = tf.nn.log_softmax(student_logits, axis=-1)
    t_log = tf.nn.log_softmax(teacher_logits, axis=-1)
    t_prob = tf.exp(t_log)
    kl = tf.reduce_sum(t_prob * (t_log - s_log), axis=-1)
    return float(tf.reduce_mean(kl).numpy()) * beta
