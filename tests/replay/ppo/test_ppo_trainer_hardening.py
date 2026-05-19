"""PPO trainer: fixed hyperparams, value-head updates, anchor metrics."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from dungeon_runner.replay.bc.human_rows import load_human_rows
from dungeon_runner.replay.ppo.frozen_teacher import FrozenBCTeacher
from dungeon_runner.replay.ppo.trainer import (
    PPO_MAX_UPDATES,
    PPO_ROLLOUT_STEPS,
    PPO_SEED,
    train_ppo,
)
from dungeon_runner.rl.model import PolicyValueModel
from dungeon_runner.rl.ppo import RolloutBatch, RolloutGameStats
from tests.replay.bc.bc_fixtures import SMOKE_N_ACTIONS, SMOKE_OBS_DIM, write_bc_derived_fixture
from tests.replay.ppo.ppo_fixtures import write_bc_run_artifact


@pytest.fixture(autouse=True)
def _seed():
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


def _fake_batch(n: int = 8) -> RolloutBatch:
    batch = RolloutBatch()
    for i in range(n):
        batch.obs.append(np.zeros(SMOKE_OBS_DIM, np.float32))
        batch.mask.append(np.ones(SMOKE_N_ACTIONS, np.float32))
        batch.act.append(0)
        batch.logp.append(-1.0)
        batch.value.append(0.0)
        batch.reward.append(0.01)
        batch.done.append(i == n - 1)
    return batch


def test_ppo_hyperparameters_are_fixed_constants():
    assert PPO_MAX_UPDATES == 32
    assert PPO_ROLLOUT_STEPS == 256
    assert PPO_SEED == 17


def test_value_head_updates_during_train_ppo(tiny_model, tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    bc_run = write_bc_run_artifact(repo)
    teacher = FrozenBCTeacher(tiny_model)
    data = tmp_path / "data"
    write_bc_derived_fixture(data)
    rows = load_human_rows(data, split="train")
    value_before = [w.numpy().copy() for w in tiny_model.value.weights]

    def fake_collect(*_a, **_k):
        return _fake_batch(), RolloutGameStats(env_steps=8), "vs_bc_bot"

    monkeypatch.setattr(
        "dungeon_runner.replay.ppo.trainer.collect_rollouts",
        fake_collect,
    )

    train_ppo(
        tiny_model,
        teacher,
        rows,
        tb_dir=tmp_path / "tb",
        bc_anchor_lambda=0.0,
        use_ray=False,
        max_updates=2,
    )
    value_after = [w.numpy().copy() for w in tiny_model.value.weights]
    changed = any(
        not np.allclose(before, after)
        for before, after in zip(value_before, value_after, strict=True)
    )
    assert changed


def test_train_ppo_ray_path_uses_rollout_pool(tiny_model, tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    bc_run = write_bc_run_artifact(repo)
    teacher = FrozenBCTeacher(tiny_model)
    data = tmp_path / "data"
    write_bc_derived_fixture(data)
    rows = load_human_rows(data, split="train")
    collect_steps: list[int] = []

    class FakePool:
        def __init__(self, **_kw) -> None:
            pass

        def collect(self, _model, *, target_steps: int, update_step: int):
            collect_steps.append(update_step)
            return _fake_batch(), RolloutGameStats(env_steps=target_steps), "vs_bc_bot"

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(
        "dungeon_runner.replay.ppo.trainer.RayRolloutPool",
        FakePool,
    )

    train_ppo(
        tiny_model,
        teacher,
        rows,
        tb_dir=tmp_path / "tb",
        teacher_weights=bc_run / "policy.weights.h5",
        bc_anchor_lambda=0.0,
        use_ray=True,
        ray_workers=4,
        max_updates=3,
    )
    assert collect_steps == [0, 1, 2]


def test_train_ppo_reports_anchor_kl_when_beta_positive(tiny_model, tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    write_bc_run_artifact(repo)
    teacher = FrozenBCTeacher(tiny_model)
    data = tmp_path / "data"
    write_bc_derived_fixture(data)
    rows = load_human_rows(data, split="train")

    monkeypatch.setattr(
        "dungeon_runner.replay.ppo.trainer.collect_rollouts",
        lambda **_k: (_fake_batch(), RolloutGameStats(), "vs_bc_bot"),
    )

    result = train_ppo(
        tiny_model,
        teacher,
        rows,
        tb_dir=tmp_path / "tb",
        bc_anchor_lambda=0.1,
        bc_anchor_beta=0.05,
        use_ray=False,
        max_updates=1,
    )
    assert result.bc_anchor_kl is not None
    assert result.bc_anchor_kl >= 0.0
