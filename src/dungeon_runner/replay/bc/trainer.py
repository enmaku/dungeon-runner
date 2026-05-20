"""BC training loop: masked CE, frozen value head, best-checkpoint selection."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from dungeon_runner.replay.bc.predict import make_replay_predict
from dungeon_runner.replay.eval.derived_store import ParquetDerivedRow
from dungeon_runner.replay.eval.replay_metrics import PredictFn
from dungeon_runner.rl.model import PolicyValueModel

EpochEndFn = Callable[[int, float], None]

BC_BATCH_SIZE = 64
BC_LEARNING_RATE = 1e-4
BC_MAX_EPOCHS = 100
BC_EARLY_STOP_PATIENCE = 10
BC_SHUFFLE_SEED = 42


@dataclass(frozen=True)
class BCTrainHistory:
    epochs: int
    best_val_masked_accuracy: float
    best_epoch: int
    final_train_bc_loss: float


@dataclass(frozen=True)
class BCTrainResult:
    model: PolicyValueModel
    history: BCTrainHistory


def _row_arrays(
    rows: list[ParquetDerivedRow],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.stack([np.asarray(r.obs, dtype=np.float32) for r in rows], axis=0)
    masks = np.stack([np.asarray(r.mask, dtype=np.float32) for r in rows], axis=0)
    labels = np.array([int(r.policy_action_index) for r in rows], dtype=np.int32)
    return obs, masks, labels


def masked_accuracy(predict: PredictFn, rows: list[ParquetDerivedRow]) -> float:
    if not rows:
        return 0.0
    correct = sum(
        1
        for row in rows
        if predict(row.obs, row.mask) == int(row.policy_action_index)
    )
    return correct / len(rows)


def compute_bc_loss(model: PolicyValueModel, rows: list[ParquetDerivedRow]) -> float:
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


def _freeze_value_head(model: PolicyValueModel) -> None:
    model.value.trainable = False


def _epoch_batches(
    rows: list[ParquetDerivedRow],
    batch_size: int,
    rng: random.Random,
) -> list[list[ParquetDerivedRow]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def _train_step(
    model: PolicyValueModel,
    optimizer: keras.optimizers.Optimizer,
    batch: list[ParquetDerivedRow],
) -> float:
    obs, masks, labels = _row_arrays(batch)
    with tf.GradientTape() as tape:
        logits, _ = model(
            tf.convert_to_tensor(obs, tf.float32),
            tf.convert_to_tensor(masks, tf.float32),
            training=True,
        )
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits,
            )
        )
    trainable = [v for v in model.trainable_variables if v.trainable]
    grads = tape.gradient(loss, trainable)
    optimizer.apply_gradients(zip(grads, trainable, strict=False))
    return float(loss.numpy())


def _log_epoch(
    writer: tf.summary.SummaryWriter,
    *,
    epoch: int,
    train_loss: float,
    val_acc: float,
) -> None:
    with writer.as_default():
        tf.summary.scalar("train/bc_loss", train_loss, step=epoch)
        tf.summary.scalar("val/masked_accuracy", val_acc, step=epoch)


def train_bc(
    model: PolicyValueModel,
    train_rows: list[ParquetDerivedRow],
    val_rows: list[ParquetDerivedRow],
    *,
    predict_fn: PredictFn | None = None,
    tb_dir: Path | None = None,
    batch_size: int = BC_BATCH_SIZE,
    learning_rate: float = BC_LEARNING_RATE,
    max_epochs: int = BC_MAX_EPOCHS,
    patience: int = BC_EARLY_STOP_PATIENCE,
    shuffle_seed: int = BC_SHUFFLE_SEED,
    on_epoch_end: EpochEndFn | None = None,
) -> BCTrainResult:
    _freeze_value_head(model)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    rng = random.Random(shuffle_seed)
    val_predict = predict_fn or make_replay_predict(model)
    best_weights: list[np.ndarray] | None = None
    best_val = -1.0
    best_epoch = 0
    epochs_run = 0
    stale = 0
    writer = (
        tf.summary.create_file_writer(str(tb_dir))
        if tb_dir is not None
        else None
    )

    try:
        for epoch in range(1, max_epochs + 1):
            epochs_run = epoch
            batch_losses: list[float] = []
            for batch in _epoch_batches(train_rows, batch_size, rng):
                if not batch:
                    continue
                batch_losses.append(_train_step(model, optimizer, batch))
            train_loss = (
                float(np.mean(batch_losses)) if batch_losses else compute_bc_loss(model, train_rows)
            )
            val_acc = masked_accuracy(val_predict, val_rows)
            if writer is not None:
                _log_epoch(writer, epoch=epoch, train_loss=train_loss, val_acc=val_acc)
            if on_epoch_end is not None:
                on_epoch_end(epoch, val_acc)
            if val_acc > best_val:
                best_val = val_acc
                best_epoch = epoch
                best_weights = model.get_weights()
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    finally:
        if writer is not None:
            writer.close()

    if best_weights is not None:
        model.set_weights(best_weights)

    train_bc_loss = compute_bc_loss(model, train_rows)
    history = BCTrainHistory(
        epochs=epochs_run,
        best_val_masked_accuracy=best_val,
        best_epoch=best_epoch,
        final_train_bc_loss=train_bc_loss,
    )
    return BCTrainResult(model=model, history=history)
