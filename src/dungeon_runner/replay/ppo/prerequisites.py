"""PPO start prerequisites (fail fast before training)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dungeon_runner.replay.bc.human_rows import count_human_rows_by_split
from dungeon_runner.replay.bc.prerequisites import check_bc_prerequisites
from dungeon_runner.replay.eval.eval_config import require_eval_config
from dungeon_runner.replay.eval.eval_suite import require_eval_suite
from dungeon_runner.replay.eval.metrics_writer import load_metrics


class PPOPrerequisiteError(RuntimeError):
    """PPO cannot start; no training run artifact is written."""


@dataclass(frozen=True)
class PPOPrerequisites:
    data_dir: Path
    bc_run: Path
    bc_weights: Path
    bc_run_id: str
    train_human_rows: int


def check_ppo_prerequisites(
    data_dir: Path,
    bc_run: Path,
    *,
    bc_anchor_lambda: float,
    repo_root: Path | None = None,
) -> PPOPrerequisites:
    data_dir = data_dir.resolve()
    bc_run = bc_run.resolve()
    if not bc_run.is_dir():
        raise PPOPrerequisiteError(f"bc-run is not a directory: {bc_run}")

    weights = bc_run / "policy.weights.h5"
    if not weights.is_file():
        raise PPOPrerequisiteError(f"bc-run missing policy weights: {weights}")

    metrics_path = bc_run / "metrics.json"
    if not metrics_path.is_file():
        raise PPOPrerequisiteError(f"bc-run missing metrics artifact: {metrics_path}")

    require_eval_suite(data_dir)
    require_eval_config(data_dir)

    train_rows = 0
    if bc_anchor_lambda > 0:
        repo = (repo_root or Path.cwd()).resolve()
        bc_prereq = check_bc_prerequisites(
            data_dir,
            repo,
            parent_weights=weights,
        )
        train_rows = bc_prereq.train_human_rows

    metrics = load_metrics(metrics_path)
    run_id = str(metrics.get("run_id") or bc_run.name)

    return PPOPrerequisites(
        data_dir=data_dir,
        bc_run=bc_run,
        bc_weights=weights,
        bc_run_id=run_id,
        train_human_rows=train_rows,
    )
