"""Eval config init artifact defaults."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dungeon_runner.replay.eval.eval_config import (
    EvalConfigError,
    eval_config_path,
    init_eval_config,
    load_eval_config,
    require_eval_config,
)
from dungeon_runner.replay.eval.floor_recorder import record_floor_if_needed


def test_init_eval_config_defaults(tmp_path: Path):
    artifact = init_eval_config(tmp_path)
    assert artifact.sim_seeds == list(range(16))
    assert artifact.sim_regression_tolerance == 0.01
    assert artifact.replay_accuracy_floor is None

    on_disk = json.loads(eval_config_path(tmp_path).read_text(encoding="utf-8"))
    assert on_disk["replay_accuracy_floor"] is None
    assert load_eval_config(tmp_path) == artifact


def test_init_refuses_existing_without_overwrite(tmp_path: Path):
    init_eval_config(tmp_path)
    with pytest.raises(EvalConfigError, match="already exists"):
        init_eval_config(tmp_path)


def test_floor_recorder_sets_floor_once(tmp_path: Path):
    init_eval_config(tmp_path)
    metrics = {"replay": {"val_masked_accuracy": 0.8125}}
    assert record_floor_if_needed(metrics, tmp_path) == "updated"
    config = load_eval_config(tmp_path)
    assert config is not None
    assert config.replay_accuracy_floor == 0.8125
    assert record_floor_if_needed(metrics, tmp_path) == "skipped"
    assert not eval_config_path(tmp_path).with_suffix(".json.tmp").exists()


def test_require_eval_config_missing(tmp_path: Path):
    with pytest.raises(EvalConfigError, match="eval config artifact missing"):
        require_eval_config(tmp_path)


def test_floor_recorder_requires_eval_config(tmp_path: Path):
    with pytest.raises(EvalConfigError, match="eval config artifact missing"):
        record_floor_if_needed({"replay": {"val_masked_accuracy": 0.5}}, tmp_path)


def test_floor_recorder_requires_val_masked_accuracy(tmp_path: Path):
    init_eval_config(tmp_path)
    with pytest.raises(EvalConfigError, match="val_masked_accuracy"):
        record_floor_if_needed({"replay": {}}, tmp_path)


def test_cli_eval_config_init(tmp_path: Path):
    repo = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dungeon_runner.replay.cli",
            "eval_config",
            "init",
            "--data-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    assert proc.returncode == 0, proc.stderr
    config = load_eval_config(tmp_path)
    assert config is not None
    assert config.replay_accuracy_floor is None
    assert len(config.sim_seeds) == 16
