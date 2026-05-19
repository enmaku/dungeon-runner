"""BC trainer acceptance: frozen value head, checkpoint, metric alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from dungeon_runner.replay.bc.predict import make_replay_predict
from dungeon_runner.replay.bc.trainer import (
    BC_BATCH_SIZE,
    BC_EARLY_STOP_PATIENCE,
    BC_LEARNING_RATE,
    BC_MAX_EPOCHS,
    BC_SHUFFLE_SEED,
    masked_accuracy,
    train_bc,
)
from dungeon_runner.replay.eval.replay_metrics import replay_metrics
from dungeon_runner.rl.model import PolicyValueModel
from tests.replay.bc.bc_fixtures import (
    SMOKE_N_ACTIONS,
    SMOKE_OBS_DIM,
    write_bc_derived_fixture,
)
from dungeon_runner.replay.bc.human_rows import load_human_rows


@pytest.fixture(autouse=True)
def _seed_rng():
    tf.random.set_seed(42)
    np.random.seed(42)


@pytest.fixture
def tiny_model() -> PolicyValueModel:
    model = PolicyValueModel(
        obs_dim=SMOKE_OBS_DIM,
        n_actions=SMOKE_N_ACTIONS,
        hidden=(16,),
        use_layer_norm=False,
    )
    _ = model(
        tf.zeros((1, SMOKE_OBS_DIM), tf.float32),
        tf.zeros((1, SMOKE_N_ACTIONS), tf.float32),
    )
    return model


@pytest.fixture
def human_rows(tmp_path: Path):
    write_bc_derived_fixture(tmp_path)
    train = load_human_rows(tmp_path, split="train")
    val = load_human_rows(tmp_path, split="val")
    return train, val


def test_bc_hyperparameters_are_fixed_constants():
    assert BC_BATCH_SIZE == 64
    assert BC_LEARNING_RATE == 1e-4
    assert BC_MAX_EPOCHS == 100
    assert BC_EARLY_STOP_PATIENCE == 10
    assert BC_SHUFFLE_SEED == 42


def test_value_head_weights_unchanged_after_train_bc(tiny_model, human_rows):
    train_rows, val_rows = human_rows
    value_before = [w.numpy().copy() for w in tiny_model.value.weights]
    train_bc(
        tiny_model,
        train_rows,
        val_rows,
        max_epochs=12,
        patience=4,
        batch_size=4,
    )
    value_after = [w.numpy().copy() for w in tiny_model.value.weights]
    for before, after in zip(value_before, value_after, strict=True):
        np.testing.assert_allclose(before, after)


def test_restored_weights_match_best_val_epoch(tiny_model, human_rows):
    train_rows, val_rows = human_rows
    snapshots: dict[int, list[list[np.ndarray]]] = {}

    def on_epoch(epoch: int, val_acc: float) -> None:
        del val_acc
        snapshots[epoch] = [np.copy(w) for w in tiny_model.get_weights()]

    result = train_bc(
        tiny_model,
        train_rows,
        val_rows,
        max_epochs=25,
        patience=3,
        batch_size=4,
        on_epoch_end=on_epoch,
    )
    assert result.history.best_epoch in snapshots
    best_weights = snapshots[result.history.best_epoch]
    for expected, actual in zip(best_weights, tiny_model.get_weights(), strict=True):
        np.testing.assert_allclose(expected, actual)


def test_masked_accuracy_matches_replay_metrics_definition(tiny_model, human_rows):
    _train_rows, val_rows = human_rows
    result = train_bc(
        tiny_model,
        _train_rows,
        val_rows,
        max_epochs=10,
        patience=3,
        batch_size=4,
    )
    predict = make_replay_predict(result.model)
    replay = replay_metrics(predict, predict, val_rows)
    assert replay.val_masked_accuracy == pytest.approx(masked_accuracy(predict, val_rows))
