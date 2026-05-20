"""Frozen BC teacher masked inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dungeon_runner.replay.ppo.frozen_teacher import FrozenBCTeacher
from tests.replay.bc.bc_fixtures import SMOKE_N_ACTIONS, SMOKE_OBS_DIM, smoke_load_model, write_smoke_parent_weights


@pytest.fixture
def teacher_weights(tmp_path: Path) -> Path:
    path = tmp_path / "teacher.weights.h5"
    write_smoke_parent_weights(path)
    return path


def test_frozen_teacher_masked_select_always_legal(teacher_weights: Path):
    teacher = FrozenBCTeacher.from_weights(teacher_weights, load_model=smoke_load_model)
    obs = np.zeros(SMOKE_OBS_DIM, dtype=np.float32)
    mask = np.array([1, 1, 0, 0], dtype=np.float32)
    for _ in range(20):
        idx = teacher.select_masked(obs, mask)
        assert 0 <= idx < SMOKE_N_ACTIONS
        assert mask[idx] > 0


def test_frozen_teacher_weights_not_mutated_by_forward(teacher_weights: Path):
    teacher = FrozenBCTeacher.from_weights(teacher_weights, load_model=smoke_load_model)
    before = teacher_weights.read_bytes()
    obs = np.ones(SMOKE_OBS_DIM, dtype=np.float32)
    mask = np.ones(SMOKE_N_ACTIONS, dtype=np.float32)
    _ = teacher.select_masked(obs, mask)
    assert teacher_weights.read_bytes() == before
