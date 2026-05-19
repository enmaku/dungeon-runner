"""BC stage acceptance: artifact order, floor, CLI, production model path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dungeon_runner.replay.bc.predict import load_policy_model
from dungeon_runner.replay.bc.stage import run_bc
from dungeon_runner.replay.bc.trainer import train_bc
from dungeon_runner.replay.eval.eval_config import load_eval_config
from dungeon_runner.replay.eval.metrics_writer import load_metrics
from tests.replay.bc.bc_fixtures import (
    PRODUCTION_PARENT_WEIGHTS,
    REPO_ROOT,
    smoke_load_model,
    stub_sim_metrics,
    write_bc_fixture_tree,
    write_bc_fixture_tree_production,
)


def _fast_train(*args, **kwargs):
    kwargs.setdefault("max_epochs", 8)
    kwargs.setdefault("patience", 3)
    kwargs.setdefault("batch_size", 4)
    return train_bc(*args, **kwargs)


def test_metrics_json_written_after_weights(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree(data, repo)

    summary = run_bc(
        data_dir=data,
        repo_root=repo,
        run_id="bc-artifact-order",
        gate_preview=False,
        train_bc_fn=_fast_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    weights = summary.run_dir / "policy.weights.h5"
    metrics = summary.metrics_path
    assert weights.is_file() and metrics.is_file()
    assert metrics.stat().st_mtime_ns >= weights.stat().st_mtime_ns


def test_floor_equals_post_train_val_masked_accuracy(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree(data, repo)

    summary = run_bc(
        data_dir=data,
        repo_root=repo,
        run_id="bc-floor-exact",
        gate_preview=False,
        train_bc_fn=_fast_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    metrics = load_metrics(summary.metrics_path)
    config = load_eval_config(data)
    assert config is not None
    assert config.replay_accuracy_floor == pytest.approx(
        metrics["replay"]["val_masked_accuracy"]
    )


@pytest.mark.skipif(
    not PRODUCTION_PARENT_WEIGHTS.is_file(),
    reason="models/latest/policy.weights.h5 not present",
)
def test_run_bc_with_production_policy_value_model(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree_production(data, repo)

    summary = run_bc(
        data_dir=data,
        repo_root=repo,
        run_id="bc-production-smoke",
        gate_preview=False,
        train_bc_fn=_fast_train,
        load_model_fn=load_policy_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    assert (summary.run_dir / "policy.weights.h5").is_file()
    metrics = load_metrics(summary.metrics_path)
    assert metrics["parent_weights"] == str(
        (repo / "models" / "latest" / "policy.weights.h5").resolve()
    )


@pytest.mark.skipif(
    not PRODUCTION_PARENT_WEIGHTS.is_file(),
    reason="models/latest/policy.weights.h5 not present",
)
def test_bc_cli_success_on_fixture_tree(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree_production(data, repo)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dungeon_runner.replay.cli",
            "bc",
            "--data-dir",
            str(data),
            "--run-id",
            "bc-cli-fixture",
            "--no-gate-preview",
        ],
        capture_output=True,
        text=True,
        cwd=repo,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert "bc run bc-cli-fixture" in proc.stdout
    assert (repo / "models" / "runs" / "bc-cli-fixture" / "metrics.json").is_file()


@pytest.mark.skipif(
    not PRODUCTION_PARENT_WEIGHTS.is_file(),
    reason="models/latest/policy.weights.h5 not present",
)
def test_bc_cli_uses_repo_root_models_not_data_dir(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree_production(data, repo)
    stray = data / "models" / "latest" / "policy.weights.h5"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"not-valid-weights")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dungeon_runner.replay.cli",
            "bc",
            "--data-dir",
            str(data),
            "--run-id",
            "bc-cli-root-models",
            "--no-gate-preview",
        ],
        capture_output=True,
        text=True,
        cwd=repo,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": str(REPO_ROOT / "src"),
        },
    )
    assert proc.returncode == 0, proc.stderr
    metrics = load_metrics(repo / "models" / "runs" / "bc-cli-root-models" / "metrics.json")
    assert metrics["parent_weights"] == str(
        (repo / "models" / "latest" / "policy.weights.h5").resolve()
    )
    assert not (data / "models" / "runs").exists()
