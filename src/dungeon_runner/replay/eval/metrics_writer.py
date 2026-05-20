"""Assemble per-run metrics artifact beside candidate weights."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dungeon_runner.replay.eval.atomic_json import atomic_write_json


def write_metrics(
    run_dir: Path,
    *,
    run_id: str,
    parent_weights: str,
    replay: dict[str, Any],
    sim: dict[str, Any],
    train: dict[str, Any] | None = None,
    ppo_bc_regression: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "parent_weights": parent_weights,
        "replay": dict(replay),
        "sim": dict(sim),
    }
    if train:
        payload["train"] = dict(train)
    if ppo_bc_regression is not None:
        payload["ppo_bc_regression"] = dict(ppo_bc_regression)
    path = run_dir / "metrics.json"
    atomic_write_json(path, payload)
    return path


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
