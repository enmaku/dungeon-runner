"""Fixtures for PPO stage tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dungeon_runner.replay.eval.metrics_writer import write_metrics
from tests.replay.bc.bc_fixtures import (
    write_bc_derived_fixture,
    write_bc_eval_artifacts,
    write_smoke_parent_weights,
)


def write_bc_run_artifact(
    repo_root: Path,
    *,
    run_id: str = "bc-parent",
    replay_acc: float = 0.75,
    sim_wr: float = 0.50,
) -> Path:
    run_dir = repo_root / "models" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = run_dir / "policy.weights.h5"
    write_smoke_parent_weights(weights)
    write_metrics(
        run_dir,
        run_id=run_id,
        parent_weights=str(weights.resolve()),
        replay={
            "val_masked_accuracy": replay_acc,
            "disagreement_rate": 0.0,
            "val_row_count": 4,
        },
        sim={
            "candidate_win_rate_vs_randombot": sim_wr,
            "latest_win_rate_vs_randombot": sim_wr,
            "engine": "python_training_sim",
            "seed_count": 2,
        },
        train={"bc_loss": 0.5},
    )
    return run_dir


def write_ppo_fixture_tree(
    data_dir: Path,
    repo_root: Path,
    *,
    bc_run_id: str = "bc-parent",
    bc_metrics: dict[str, Any] | None = None,
) -> Path:
    write_bc_derived_fixture(data_dir)
    write_bc_eval_artifacts(data_dir)
    latest = repo_root / "models" / "latest" / "policy.weights.h5"
    write_smoke_parent_weights(latest)
    replay_acc = 0.75
    sim_wr = 0.50
    if bc_metrics:
        replay_acc = float(bc_metrics.get("replay", {}).get("val_masked_accuracy", replay_acc))
        sim_wr = float(
            bc_metrics.get("sim", {}).get("candidate_win_rate_vs_randombot", sim_wr)
        )
    return write_bc_run_artifact(
        repo_root,
        run_id=bc_run_id,
        replay_acc=replay_acc,
        sim_wr=sim_wr,
    )
