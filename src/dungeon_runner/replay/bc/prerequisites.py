"""BC start prerequisites (fail fast before training)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dungeon_runner.replay.bc.human_rows import (
    count_human_rows_by_split,
    on_disk_val_match_ids,
)
from dungeon_runner.replay.eval.eval_config import require_eval_config
from dungeon_runner.replay.eval.eval_suite import require_eval_suite


class BCPrerequisiteError(RuntimeError):
    """BC cannot start; no training run artifact is written."""


@dataclass(frozen=True)
class BCPrerequisites:
    data_dir: Path
    parent_weights: Path
    train_human_rows: int
    val_human_rows: int


def default_parent_weights(repo_root: Path) -> Path:
    return repo_root / "models" / "latest" / "policy.weights.h5"


def check_bc_prerequisites(
    data_dir: Path,
    repo_root: Path,
    *,
    parent_weights: Path | None = None,
) -> BCPrerequisites:
    data_dir = data_dir.resolve()
    repo_root = repo_root.resolve()
    parent = (parent_weights or default_parent_weights(repo_root)).resolve()
    if not parent.is_file():
        raise BCPrerequisiteError(f"training parent weights missing: {parent}")

    eval_suite = require_eval_suite(data_dir)
    require_eval_config(data_dir)

    train_n, val_n = count_human_rows_by_split(data_dir, eval_suite)
    if train_n < 1:
        raise BCPrerequisiteError(
            "derived store has no train-split human step rows for BC"
        )
    if val_n < 1:
        raise BCPrerequisiteError(
            "derived store has no val-split human step rows in eval suite holdout"
        )

    suite_val = set(eval_suite.val_match_ids)
    disk_val = on_disk_val_match_ids(data_dir)
    extra = disk_val - suite_val
    if extra:
        ids = ", ".join(sorted(extra))
        raise BCPrerequisiteError(
            f"derived val match ids not in eval suite holdout: {ids}"
        )

    return BCPrerequisites(
        data_dir=data_dir,
        parent_weights=parent,
        train_human_rows=train_n,
        val_human_rows=val_n,
    )
