"""Policy load + replay predict helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from dungeon_runner.replay.bc.predict import load_policy_model, make_replay_predict
from dungeon_runner.rl import actions_codec, observation
from tests.replay.bc.bc_fixtures import PRODUCTION_PARENT_WEIGHTS


@pytest.mark.skipif(
    not PRODUCTION_PARENT_WEIGHTS.is_file(),
    reason="models/latest/policy.weights.h5 not present",
)
def test_load_policy_model_production_weights():
    model = load_policy_model(PRODUCTION_PARENT_WEIGHTS)
    logits, value = model(
        tf.zeros((1, observation.OBS_DIM), tf.float32),
        tf.zeros((1, actions_codec.N_ACTIONS), tf.float32),
    )
    assert logits.shape == (1, actions_codec.N_ACTIONS)
    assert value.shape == (1, 1)
    predict = make_replay_predict(model)
    idx = predict(np.zeros(observation.OBS_DIM, np.float32), np.ones(actions_codec.N_ACTIONS, np.float32))
    assert 0 <= idx < actions_codec.N_ACTIONS


def test_load_policy_model_missing_file(tmp_path: Path):
    missing = tmp_path / "missing.weights.h5"
    with pytest.raises(OSError):
        load_policy_model(missing)
