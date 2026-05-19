"""Persist replay accuracy floor from BC baseline run (atomic eval config replace)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from dungeon_runner.replay.eval.eval_config import (
    EvalConfigError,
    load_eval_config,
    save_eval_config,
)

FloorRecordOutcome = Literal["updated", "skipped"]


def record_floor_if_needed(
    metrics: dict[str, Any],
    data_dir: Path,
) -> FloorRecordOutcome:
    config = load_eval_config(data_dir)
    if config is None:
        raise EvalConfigError("eval config artifact missing; run eval_config init")

    if config.replay_accuracy_floor is not None:
        return "skipped"

    replay = metrics.get("replay") or {}
    floor = replay.get("val_masked_accuracy")
    if floor is None:
        raise EvalConfigError(
            "metrics missing replay.val_masked_accuracy; cannot record floor"
        )

    config.replay_accuracy_floor = float(floor)
    save_eval_config(data_dir, config)
    return "updated"


def load_floor_from_config(data_dir: Path) -> float | None:
    config = load_eval_config(data_dir)
    if config is None:
        return None
    return config.replay_accuracy_floor
