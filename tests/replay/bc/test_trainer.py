"""BC trainer: masked accuracy, best checkpoint, loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from dungeon_runner.replay.bc.human_rows import load_human_rows
from dungeon_runner.replay.bc.predict import make_replay_predict
from dungeon_runner.replay.bc.trainer import compute_bc_loss, masked_accuracy, train_bc
from dungeon_runner.rl.model import PolicyValueModel
from tests.replay.bc.bc_fixtures import (
    SMOKE_N_ACTIONS,
    SMOKE_OBS_DIM,
    write_bc_derived_fixture,
)


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


def test_train_bc_reduces_train_loss_on_fixture(tiny_model, human_rows):
    train_rows, val_rows = human_rows
    initial_loss = compute_bc_loss(tiny_model, train_rows)
    result = train_bc(
        tiny_model,
        train_rows,
        val_rows,
        max_epochs=20,
        patience=5,
        batch_size=4,
    )
    assert result.history.epochs >= 1
    assert result.history.final_train_bc_loss < initial_loss
    assert np.isfinite(result.history.final_train_bc_loss)
    assert 0.0 <= result.history.best_val_masked_accuracy <= 1.0


def test_best_checkpoint_not_last_epoch_when_val_peaks_early(tiny_model, human_rows):
    train_rows, val_rows = human_rows
    predict_fn = make_replay_predict(tiny_model)
    result = train_bc(
        tiny_model,
        train_rows,
        val_rows,
        predict_fn=predict_fn,
        max_epochs=30,
        patience=2,
        batch_size=4,
    )
    assert result.history.best_epoch <= result.history.epochs
    assert compute_bc_loss(result.model, train_rows) == pytest.approx(
        result.history.final_train_bc_loss
    )
