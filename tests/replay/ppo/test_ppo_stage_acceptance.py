"""PPO stage acceptance: artifact order, metrics schema, production path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from dungeon_runner.replay.bc.predict import load_policy_model
from dungeon_runner.replay.eval.metrics_writer import load_metrics
from dungeon_runner.replay.ppo.stage import run_ppo
from tests.replay.bc.bc_fixtures import PRODUCTION_PARENT_WEIGHTS, smoke_load_model, stub_sim_metrics
from tests.replay.ppo.ppo_fixtures import write_ppo_fixture_tree, write_ppo_fixture_tree_production


@dataclass(frozen=True)
class _StubTrainResult:
    ppo_loss: float = 0.5
    bc_anchor_ce: float = 0.2
    bc_anchor_kl: float | None = None


def _stub_train(*_a, **k):
    kl = 0.03 if k.get("bc_anchor_beta", 0) > 0 else None
    return _StubTrainResult(bc_anchor_kl=kl)


def test_run_ppo_use_ray_true_uses_rollout_pool(tmp_path: Path, monkeypatch):
    import numpy as np
    import tensorflow as tf

    from dungeon_runner.replay.ppo.trainer import train_ppo
    from dungeon_runner.rl.ppo import RolloutBatch, RolloutGameStats
    from tests.replay.bc.bc_fixtures import SMOKE_N_ACTIONS, SMOKE_OBS_DIM

    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(data, repo)
    pool_inits: list[int] = []

    class FakePool:
        def __init__(self, **_kw) -> None:
            pool_inits.append(int(_kw.get("n_workers", 0)))

        def collect(self, _model, *, target_steps: int, update_step: int):
            del target_steps, update_step
            batch = RolloutBatch()
            for _ in range(8):
                batch.obs.append(np.zeros(SMOKE_OBS_DIM, np.float32))
                batch.mask.append(np.ones(SMOKE_N_ACTIONS, np.float32))
                batch.act.append(0)
                batch.logp.append(-1.0)
                batch.value.append(0.0)
                batch.reward.append(0.01)
                batch.done.append(False)
            return batch, RolloutGameStats(env_steps=8), "vs_bc_bot"

        def shutdown(self) -> None:
            pass

    def bounded_train(*args, **kwargs):
        kwargs["max_updates"] = 1
        kwargs["bc_anchor_lambda"] = 0.0
        return train_ppo(*args, **kwargs)

    monkeypatch.setattr(
        "dungeon_runner.replay.ppo.trainer.RayRolloutPool",
        FakePool,
    )
    tf.random.set_seed(42)
    np.random.seed(42)

    summary = run_ppo(
        data_dir=data,
        bc_run=bc_run,
        repo_root=repo,
        run_id="ppo-ray-stage",
        gate_preview=False,
        bc_anchor_lambda=0.0,
        use_ray=True,
        ray_workers=4,
        train_ppo_fn=bounded_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    assert pool_inits == [4]
    assert (summary.run_dir / "policy.weights.h5").is_file()


def test_metrics_json_written_after_weights(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(data, repo)

    summary = run_ppo(
        data_dir=data,
        bc_run=bc_run,
        repo_root=repo,
        run_id="ppo-artifact-order",
        gate_preview=False,
        train_ppo_fn=_stub_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    weights = summary.run_dir / "policy.weights.h5"
    metrics = summary.metrics_path
    assert weights.is_file() and metrics.is_file()
    assert metrics.stat().st_mtime_ns >= weights.stat().st_mtime_ns


def test_parent_weights_points_at_bc_run_not_latest(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(data, repo)
    latest = repo / "models" / "latest" / "policy.weights.h5"
    assert latest.is_file()

    summary = run_ppo(
        data_dir=data,
        bc_run=bc_run,
        repo_root=repo,
        run_id="ppo-parent-bc",
        gate_preview=False,
        train_ppo_fn=_stub_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    metrics = load_metrics(summary.metrics_path)
    assert Path(metrics["parent_weights"]).resolve() == (bc_run / "policy.weights.h5").resolve()
    assert Path(metrics["parent_weights"]).resolve() != latest.resolve()


def test_metrics_include_bc_run_id_and_regression_block(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(data, repo)

    summary = run_ppo(
        data_dir=data,
        bc_run=bc_run,
        repo_root=repo,
        run_id="ppo-regression-block",
        gate_preview=False,
        train_ppo_fn=_stub_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    metrics = load_metrics(summary.metrics_path)
    assert metrics["ppo_bc_regression"]["bc_run_id"] == "bc-parent"
    assert metrics["ppo_bc_regression"]["pass"] is summary.regression_passed


def test_metrics_include_bc_anchor_kl_when_beta_positive(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(data, repo)

    summary = run_ppo(
        data_dir=data,
        bc_run=bc_run,
        repo_root=repo,
        run_id="ppo-kl-metrics",
        bc_anchor_beta=0.05,
        gate_preview=False,
        train_ppo_fn=_stub_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    metrics = load_metrics(summary.metrics_path)
    assert metrics["train"]["bc_anchor_kl"] == pytest.approx(0.03)


@pytest.mark.skipif(
    not PRODUCTION_PARENT_WEIGHTS.is_file(),
    reason="models/latest/policy.weights.h5 not present",
)
def test_run_ppo_with_production_policy_value_model(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree_production(data, repo)

    summary = run_ppo(
        data_dir=data,
        bc_run=bc_run,
        repo_root=repo,
        run_id="ppo-production-smoke",
        gate_preview=False,
        train_ppo_fn=_stub_train,
        load_model_fn=load_policy_model,
        sim_metrics_fn=stub_sim_metrics,
    )
    assert (summary.run_dir / "policy.weights.h5").is_file()
    metrics = load_metrics(summary.metrics_path)
    assert Path(metrics["parent_weights"]).resolve() == (bc_run / "policy.weights.h5").resolve()


def test_ppo_cli_returns_one_when_regression_fails(monkeypatch, tmp_path: Path):
    from argparse import Namespace

    from dungeon_runner.replay import cli
    from dungeon_runner.replay.ppo.stage import PPORunSummary

    run_dir = tmp_path / "models" / "runs" / "ppo-cli-fail"
    run_dir.mkdir(parents=True)
    summary = PPORunSummary(
        run_id="ppo-cli-fail",
        run_dir=run_dir,
        metrics_path=run_dir / "metrics.json",
        regression_passed=False,
        gate_preview_passed=None,
        gate_preview_reasons=[],
    )
    monkeypatch.setattr(cli, "run_ppo", lambda **_k: summary)
    code = cli._cmd_ppo(
        Namespace(
            data_dir=str(tmp_path / "data"),
            bc_run="models/runs/bc-parent",
            run_id="ppo-cli-fail",
            bc_anchor_lambda=0.1,
            bc_anchor_beta=0.0,
            max_updates=16,
            no_ray=True,
            ray_workers=8,
            no_gate_preview=True,
        )
    )
    assert code == 1
