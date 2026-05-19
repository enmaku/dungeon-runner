"""Promotion manifest writer and JSONL ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dungeon_runner.replay.eval.atomic_json import atomic_write_json


def load_promoted_run_ids(ledger_path: Path) -> set[str]:
    if not ledger_path.is_file():
        return set()
    ids: set[str] = set()
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ids.add(json.loads(line)["run_id"])
    return ids


def list_promoted_versions(models_dir: Path, ledger_path: Path) -> tuple[str, ...]:
    versions: set[str] = set()
    if models_dir.is_dir():
        for child in models_dir.iterdir():
            if (
                child.is_dir()
                and child.name.startswith("v")
                and not child.name.endswith(".tmp")
            ):
                versions.add(child.name)
    if ledger_path.is_file():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                versions.add(json.loads(line)["promoted_version"])
    return tuple(sorted(versions))


def write_promotion_manifest(
    version_dir: Path,
    *,
    promoted_version: str,
    run_id: str,
    parent_weights: str,
    promoted_at: str,
) -> Path:
    payload: dict[str, Any] = {
        "promoted_version": promoted_version,
        "run_id": run_id,
        "parent_weights": parent_weights,
        "promoted_at": promoted_at,
        "metrics_file": "metrics.json",
    }
    path = version_dir / "promotion.json"
    atomic_write_json(path, payload)
    return path


def append_promotion_ledger(
    ledger_path: Path,
    *,
    promoted_version: str,
    run_id: str,
    promoted_at: str,
) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "promoted_version": promoted_version,
        "run_id": run_id,
        "promoted_at": promoted_at,
    }
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
