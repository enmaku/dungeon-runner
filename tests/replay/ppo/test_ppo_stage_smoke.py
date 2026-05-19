"""PPO stage end-to-end smoke on tiny fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from dungeon_runner.replay.bc.stage import run_bc
from dungeon_runner.replay.bc.trainer import train_bc
from dungeon_runner.replay.eval.metrics_writer import load_metrics
from dungeon_runner.replay.ppo.prerequisites import PPOPrerequisiteError
from dungeon_runner.replay.ppo.stage import PPOStageError, run_ppo
from tests.replay.bc.bc_fixtures import smoke_load_model, stub_sim_metrics, write_bc_fixture_tree
from tests.replay.ppo.ppo_fixtures import write_ppo_fixture_tree


@dataclass(frozen=True)
class _StubTrainResult:
    ppo_loss: float = 0.42
    bc_anchor_ce: float = 0.11
    bc_anchor_kl: float | None = None


def _stub_train(*_a, **_k):
    return _StubTrainResult()


def _fast_bc_train(*args, **kwargs):
    kwargs.setdefault("max_epochs", 8)
    kwargs.setdefault("patience", 3)
    kwargs.setdefault("batch_size", 4)
    return train_bc(*args, **kwargs)


def test_run_ppo_gate_preview_when_floor_set(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree(data, repo)
    bc_run = write_ppo_fixture_tree(data, repo)
    run_bc(
        data_dir=data,
        repo_root=repo,
        run_id="bc-gate-setup",
        gate_preview=False,
        train_bc_fn=_fast_bc_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )

    summary = run_ppo(
        data_dir=data,
        bc_run=bc_run,
        repo_root=repo,
        run_id="ppo-gate",
        train_ppo_fn=_stub_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    assert summary.gate_preview_passed is True
    assert summary.gate_preview_reasons == []


def test_run_ppo_writes_run_artifact_and_metrics(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(
        data,
        repo,
        bc_metrics={
            "replay": {"val_masked_accuracy": 0.0},
            "sim": {"candidate_win_rate_vs_randombot": 0.0},
        },
    )

    summary = run_ppo(
        data_dir=data,
        bc_run=bc_run,
        repo_root=repo,
        run_id="ppo-test-smoke",
        gate_preview=False,
        train_ppo_fn=_stub_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )

    assert summary.run_dir.is_dir()
    assert (summary.run_dir / "policy.weights.h5").is_file()
    assert (summary.run_dir / "tb").is_dir()
    assert summary.metrics_path.is_file()
    assert summary.regression_passed is True
    assert not (repo / "models" / "runs" / "ppo-test-smoke.tmp").exists()

    metrics = load_metrics(summary.metrics_path)
    assert metrics["run_id"] == "ppo-test-smoke"
    assert Path(metrics["parent_weights"]).resolve() == (bc_run / "policy.weights.h5").resolve()
    assert metrics["ppo_bc_regression"]["pass"] is True
    assert metrics["train"]["ppo_loss"] == 0.42
    assert metrics["train"]["bc_anchor_ce"] == 0.11
    assert metrics["train"].get("bc_anchor_kl") is None


def test_run_ppo_commits_artifact_on_regression_fail(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(
        data,
        repo,
        bc_metrics={
            "replay": {"val_masked_accuracy": 0.99},
            "sim": {"candidate_win_rate_vs_randombot": 0.99},
        },
    )

    summary = run_ppo(
        data_dir=data,
        bc_run=bc_run,
        repo_root=repo,
        run_id="ppo-regress-fail",
        gate_preview=False,
        train_ppo_fn=_stub_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )

    assert summary.run_dir.is_dir()
    assert summary.regression_passed is False
    metrics = load_metrics(summary.metrics_path)
    assert metrics["ppo_bc_regression"]["pass"] is False


def test_run_ppo_leaves_no_committed_dir_on_train_failure(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(data, repo)

    def boom(*_a, **_k):
        raise RuntimeError("simulated train failure")

    with pytest.raises(PPOStageError):
        run_ppo(
            data_dir=data,
            bc_run=bc_run,
            repo_root=repo,
            run_id="ppo-fail",
            train_ppo_fn=boom,
            load_model_fn=smoke_load_model,
        )
    assert not (repo / "models" / "runs" / "ppo-fail").exists()


def test_prerequisites_fail_before_staging(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(data, repo)
    with pytest.raises(PPOPrerequisiteError):
        run_ppo(
            data_dir=data,
            bc_run=repo / "models" / "runs" / "missing-bc",
            repo_root=repo,
            run_id="ppo-nobc",
        )
