"""Minimal training run artifact + eval config for publish tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from dungeon_runner.replay.eval.atomic_json import atomic_write_json
from dungeon_runner.replay.eval.eval_config import EvalConfigArtifact


def write_passing_eval_config(data_dir: Path, *, floor: float = 0.7) -> None:
    config = EvalConfigArtifact(
        sim_seeds=[0, 1],
        sim_regression_tolerance=0.01,
        replay_accuracy_floor=floor,
    )
    atomic_write_json(data_dir / "eval_config.json", config.to_dict())


def write_training_run_artifact(
    repo_root: Path,
    *,
    run_id: str = "bc-20260518T120000Z",
    val_acc: float = 0.85,
    cand_wr: float = 0.54,
    latest_wr: float = 0.55,
    ppo_pass: bool | None = None,
) -> Path:
    run_dir = repo_root / "models" / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "policy.weights.h5").write_bytes(b"weights")
    metrics: dict = {
        "run_id": run_id,
        "timestamp": "2026-05-18T12:00:00+00:00",
        "parent_weights": str((repo_root / "models" / "latest" / "policy.weights.h5").resolve()),
        "replay": {"val_masked_accuracy": val_acc},
        "sim": {
            "candidate_win_rate_vs_randombot": cand_wr,
            "latest_win_rate_vs_randombot": latest_wr,
        },
    }
    if ppo_pass is not None:
        metrics["ppo_bc_regression"] = {"pass": ppo_pass}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return run_dir


def seed_legacy_latest(repo_root: Path) -> None:
    legacy = repo_root / "models" / "v0.1.30a"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "policy.weights.h5").write_bytes(b"legacy")
    latest = repo_root / "models" / "latest"
    if latest.exists() and latest.is_symlink():
        latest.unlink()
    elif latest.is_dir():
        shutil.rmtree(latest)
    latest.mkdir(exist_ok=True)
    (latest / "policy.weights.h5").write_bytes(b"dup")
