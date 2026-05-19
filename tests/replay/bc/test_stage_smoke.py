"""BC stage end-to-end smoke on tiny fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from dungeon_runner.replay.bc.prerequisites import BCPrerequisiteError
from dungeon_runner.replay.bc.stage import BCStageError, run_bc
from dungeon_runner.replay.bc.trainer import train_bc
from dungeon_runner.replay.eval.eval_config import load_eval_config
from dungeon_runner.replay.eval.metrics_writer import load_metrics
from tests.replay.bc.bc_fixtures import (
    smoke_load_model,
    stub_sim_metrics,
    write_bc_fixture_tree,
)


def _fast_train(*args, **kwargs):
    kwargs.setdefault("max_epochs", 8)
    kwargs.setdefault("patience", 3)
    kwargs.setdefault("batch_size", 4)
    return train_bc(*args, **kwargs)


def test_run_bc_writes_run_artifact_and_metrics(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree(data, repo)

    summary = run_bc(
        data_dir=data,
        repo_root=repo,
        run_id="bc-test-smoke",
        gate_preview=False,
        train_bc_fn=_fast_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )

    assert summary.run_dir.is_dir()
    assert (summary.run_dir / "policy.weights.h5").is_file()
    assert (summary.run_dir / "tb").is_dir()
    assert summary.metrics_path.is_file()
    assert not (repo / "models" / "runs" / "bc-test-smoke.tmp").exists()

    metrics = load_metrics(summary.metrics_path)
    assert metrics["run_id"] == "bc-test-smoke"
    assert Path(metrics["parent_weights"]).is_file()
    assert "replay" in metrics and "sim" in metrics
    assert "train" in metrics and metrics["train"]["bc_loss"] >= 0


def test_run_bc_records_floor_on_first_baseline(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree(data, repo)

    summary = run_bc(
        data_dir=data,
        repo_root=repo,
        run_id="bc-floor",
        gate_preview=False,
        train_bc_fn=_fast_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    assert summary.floor_outcome == "updated"
    metrics = load_metrics(summary.metrics_path)
    config = load_eval_config(data)
    assert config is not None
    assert config.replay_accuracy_floor == pytest.approx(
        metrics["replay"]["val_masked_accuracy"]
    )

    summary2 = run_bc(
        data_dir=data,
        repo_root=repo,
        run_id="bc-floor-2",
        gate_preview=False,
        train_bc_fn=_fast_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    assert summary2.floor_outcome == "skipped"
    floor_after = load_eval_config(data)
    assert floor_after is not None
    assert floor_after.replay_accuracy_floor == config.replay_accuracy_floor


def test_run_bc_gate_preview_when_floor_set(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree(data, repo)
    run_bc(
        data_dir=data,
        repo_root=repo,
        run_id="bc-gate-setup",
        gate_preview=False,
        train_bc_fn=_fast_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )

    summary = run_bc(
        data_dir=data,
        repo_root=repo,
        run_id="bc-gate",
        gate_preview=True,
        train_bc_fn=_fast_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    assert summary.gate_preview_passed is True
    assert summary.gate_preview_reasons == []


def test_run_bc_leaves_no_committed_dir_on_failure(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree(data, repo)

    def boom(*_a, **_k):
        raise RuntimeError("simulated train failure")

    with pytest.raises(BCStageError):
        run_bc(
            data_dir=data,
            repo_root=repo,
            run_id="bc-fail",
            train_bc_fn=boom,
            load_model_fn=smoke_load_model,
        )
    assert not (repo / "models" / "runs" / "bc-fail").exists()
    assert not (repo / "models" / "runs" / "bc-fail.tmp").exists()


def test_prerequisites_fail_before_staging(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    with pytest.raises(BCPrerequisiteError):
        run_bc(data_dir=data, repo_root=repo, run_id="bc-nodata")
