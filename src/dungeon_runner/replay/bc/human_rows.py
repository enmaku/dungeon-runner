"""Load train/val human-step rows from derived store."""

from __future__ import annotations

from pathlib import Path

from dungeon_runner.replay.eval.derived_store import ParquetDerivedRow, load_derived_rows
from dungeon_runner.replay.eval.eval_suite import EvalSuiteArtifact


def load_human_rows(
    data_dir: Path,
    *,
    split: str | None = None,
    val_match_ids: set[str] | None = None,
) -> list[ParquetDerivedRow]:
    rows: list[ParquetDerivedRow] = []
    for row in load_derived_rows(data_dir):
        if not row.is_human_step:
            continue
        if split is not None and row.split != split:
            continue
        if (
            row.split == "val"
            and val_match_ids is not None
            and row.match_id not in val_match_ids
        ):
            continue
        rows.append(row)
    return rows


def count_human_rows_by_split(
    data_dir: Path,
    eval_suite: EvalSuiteArtifact,
) -> tuple[int, int]:
    val_ids = set(eval_suite.val_match_ids)
    train_n = 0
    val_n = 0
    for row in load_derived_rows(data_dir):
        if not row.is_human_step:
            continue
        if row.split == "train":
            train_n += 1
        elif row.split == "val" and row.match_id in val_ids:
            val_n += 1
    return train_n, val_n


def on_disk_val_match_ids(data_dir: Path) -> set[str]:
    derived_root = data_dir / "derived"
    if not derived_root.is_dir():
        return set()
    val_ids: set[str] = set()
    for row in load_derived_rows(data_dir):
        if row.split == "val":
            val_ids.add(row.match_id)
    return val_ids
