"""PPO BC regression check."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

from dungeon_runner.replay.eval.atomic_json import atomic_write_json
from dungeon_runner.replay.eval.eval_config import EvalConfigArtifact
from dungeon_runner.replay.eval.metrics_writer import load_metrics
from dungeon_runner.replay.eval.sim_metrics import SimMetrics
from dungeon_runner.replay.ppo.regression import check_ppo_bc_regression
from dungeon_runner.replay.ppo.stage import run_ppo
from tests.replay.bc.bc_fixtures import smoke_load_model
from tests.replay.ppo.ppo_fixtures import write_ppo_fixture_tree


def _metrics(replay_acc: float, sim_wr: float) -> dict:
    return {
        "replay": {"val_masked_accuracy": replay_acc},
        "sim": {"candidate_win_rate_vs_randombot": sim_wr},
    }


def test_regression_passes_when_replay_strict_and_sim_within_epsilon():
    bc = _metrics(0.80, 0.55)
    cand = _metrics(0.82, 0.54)
    assert check_ppo_bc_regression(cand, bc, epsilon=0.01)


def test_regression_fails_when_replay_below_bc():
    bc = _metrics(0.80, 0.50)
    cand = _metrics(0.79, 0.60)
    assert not check_ppo_bc_regression(cand, bc, epsilon=0.05)


def test_regression_fails_when_sim_below_bc_minus_epsilon():
    bc = _metrics(0.80, 0.55)
    cand = _metrics(0.80, 0.53)
    assert not check_ppo_bc_regression(cand, bc, epsilon=0.01)


@dataclass(frozen=True)
class _StubTrainResult:
    ppo_loss: float = 0.1
    bc_anchor_ce: float = 0.1
    bc_anchor_kl: float | None = None


def _stub_train(*_a, **_k):
    return _StubTrainResult()


def _sim_stub_drop_four_pct(_candidate, _latest, seeds: list[int]) -> SimMetrics:
    n = len(seeds)
    return SimMetrics(
        candidate_win_rate_vs_randombot=0.46,
        latest_win_rate_vs_randombot=0.50,
        seed_count=n,
    )


def test_run_ppo_passes_sim_regression_tolerance_from_eval_config(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(
        data,
        repo,
        bc_metrics={
            "replay": {"val_masked_accuracy": 0.0},
            "sim": {"candidate_win_rate_vs_randombot": 0.50},
        },
    )
    atomic_write_json(
        data / "eval_config.json",
        EvalConfigArtifact(
            sim_seeds=[0, 1],
            sim_regression_tolerance=0.05,
        ).to_dict(),
    )
    captured: list[float] = []
    real_check = check_ppo_bc_regression

    def recording_check(cand, bc, *, epsilon: float = 0.01):
        captured.append(epsilon)
        return real_check(cand, bc, epsilon=epsilon)

    with patch(
        "dungeon_runner.replay.ppo.stage.check_ppo_bc_regression",
        side_effect=recording_check,
    ):
        summary = run_ppo(
            data_dir=data,
            bc_run=bc_run,
            repo_root=repo,
            run_id="ppo-eps-loose",
            gate_preview=False,
            bc_anchor_lambda=0.0,
            train_ppo_fn=_stub_train,
            load_model_fn=smoke_load_model,
            sim_metrics_fn=_sim_stub_drop_four_pct,
        )
    assert captured == [0.05]
    assert summary.regression_passed is True

    atomic_write_json(
        data / "eval_config.json",
        EvalConfigArtifact(
            sim_seeds=[0, 1],
            sim_regression_tolerance=0.01,
        ).to_dict(),
    )
    summary_tight = run_ppo(
        data_dir=data,
        bc_run=bc_run,
        repo_root=repo,
        run_id="ppo-eps-tight",
        gate_preview=False,
        bc_anchor_lambda=0.0,
        train_ppo_fn=_stub_train,
        load_model_fn=smoke_load_model,
        sim_metrics_fn=_sim_stub_drop_four_pct,
    )
    assert summary_tight.regression_passed is False
    metrics = load_metrics(summary_tight.metrics_path)
    assert metrics["ppo_bc_regression"]["pass"] is False
