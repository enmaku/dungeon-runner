"""BC anchor CE / KL on human rows."""

from __future__ import annotations

import numpy as np
import pytest

from dungeon_runner.replay.bc.human_rows import load_human_rows
import tensorflow as tf
from tensorflow import keras

from dungeon_runner.replay.ppo.bc_anchor import anchor_ce_loss, anchor_kl_loss
from dungeon_runner.replay.ppo.trainer import _apply_anchor_step
from dungeon_runner.rl.model import PolicyValueModel
from tests.replay.bc.bc_fixtures import (
    SMOKE_N_ACTIONS,
    SMOKE_OBS_DIM,
    smoke_load_model,
    write_bc_derived_fixture,
)
from tests.replay.ppo.ppo_fixtures import write_bc_run_artifact


def test_anchor_ce_decreases_after_step(tmp_path):
    data = tmp_path / "data"
    write_bc_derived_fixture(data)
    rows = load_human_rows(data, split="train")[:4]
    model = PolicyValueModel(
        obs_dim=SMOKE_OBS_DIM,
        n_actions=SMOKE_N_ACTIONS,
        hidden=(8,),
        use_layer_norm=False,
    )
    _ = model(
        tf.zeros((1, SMOKE_OBS_DIM), tf.float32),
        tf.zeros((1, SMOKE_N_ACTIONS), tf.float32),
    )
    opt = keras.optimizers.Adam(learning_rate=1e-2)
    ce_before = anchor_ce_loss(model, rows)
    _apply_anchor_step(model, rows, lam=1.0, teacher=None, beta=0.0, opt=opt)
    ce_after = anchor_ce_loss(model, rows)
    assert ce_after < ce_before


def test_anchor_ce_loss_non_negative(tmp_path):
    data = tmp_path / "data"
    write_bc_derived_fixture(data)
    rows = load_human_rows(data, split="train")
    repo = tmp_path / "repo"
    bc_run = write_bc_run_artifact(repo)
    model = smoke_load_model(bc_run / "policy.weights.h5")
    loss = anchor_ce_loss(model, rows)
    assert loss >= 0.0


def test_anchor_kl_zero_when_beta_zero(tmp_path):
    data = tmp_path / "data"
    write_bc_derived_fixture(data)
    rows = load_human_rows(data, split="train")
    repo = tmp_path / "repo"
    bc_run = write_bc_run_artifact(repo)
    model = smoke_load_model(bc_run / "policy.weights.h5")
    teacher = smoke_load_model(bc_run / "policy.weights.h5")
    assert anchor_kl_loss(model, teacher, rows, beta=0.0) == 0.0


@pytest.mark.parametrize("beta", [0.05])
def test_anchor_kl_non_negative_when_enabled(tmp_path, beta: float):
    data = tmp_path / "data"
    write_bc_derived_fixture(data)
    rows = load_human_rows(data, split="train")
    repo = tmp_path / "repo"
    bc_run = write_bc_run_artifact(repo)
    model = smoke_load_model(bc_run / "policy.weights.h5")
    teacher = smoke_load_model(bc_run / "policy.weights.h5")
    loss = anchor_kl_loss(model, teacher, rows, beta=beta)
    assert loss >= 0.0
