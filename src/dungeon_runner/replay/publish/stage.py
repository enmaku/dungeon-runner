"""Publish stage: validate artifact, gates, gated promotion executor."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dungeon_runner.replay.eval.eval_config import require_eval_config
from dungeon_runner.replay.eval.metrics_writer import load_metrics
from dungeon_runner.replay.publish.latest_migrator import migrate_latest_symlink
from dungeon_runner.replay.publish.manifest import (
    append_promotion_ledger,
    list_promoted_versions,
    load_promoted_run_ids,
    write_promotion_manifest,
)
from dungeon_runner.replay.publish.publish_gates import run_publish_gates
from dungeon_runner.replay.publish.version_allocator import allocate_version


class PublishError(RuntimeError):
    def __init__(self, message: str, *, reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = list(reasons or [])


@dataclass(frozen=True)
class PublishSummary:
    run_id: str
    run_dir: Path
    promoted_version: str
    version_dir: Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_run_dir(run_dir: Path) -> None:
    resolved = run_dir.resolve()
    if resolved.name.endswith(".tmp") or ".tmp" in resolved.parts:
        raise PublishError(f"refusing staging training run artifact: {run_dir}")
    weights = run_dir / "policy.weights.h5"
    metrics_path = run_dir / "metrics.json"
    if not weights.is_file() or not metrics_path.is_file():
        raise PublishError(
            f"training run artifact missing policy.weights.h5 or metrics.json: {run_dir}"
        )


def run_publish(
    *,
    run_dir: Path,
    data_dir: Path,
    repo_root: Path | None = None,
    version_override: str | None = None,
    promoted_at: str | None = None,
) -> PublishSummary:
    repo_root = (repo_root or Path.cwd()).resolve()
    run_dir = run_dir.resolve()
    data_dir = data_dir.resolve()
    models_dir = repo_root / "models"
    ledger_path = models_dir / "promotions.jsonl"

    validate_run_dir(run_dir)
    migrate_latest_symlink(models_dir)

    metrics = load_metrics(run_dir / "metrics.json")
    run_id = str(metrics.get("run_id") or run_dir.name)
    eval_config = require_eval_config(data_dir)

    gate = run_publish_gates(
        metrics,
        eval_config,
        run_id,
        promoted_run_ids=load_promoted_run_ids(ledger_path),
    )
    if not gate.passed:
        raise PublishError(
            "publish gate evaluation failed: " + ", ".join(gate.reasons),
            reasons=gate.reasons,
        )

    version = allocate_version(
        existing_versions=list_promoted_versions(models_dir, ledger_path),
        override=version_override,
    )
    promoted_at_ts = promoted_at or _utc_now_iso()
    parent_weights = str(metrics.get("parent_weights", ""))

    staging = models_dir / f"{version}.tmp"
    final_dir = models_dir / version
    if staging.exists():
        shutil.rmtree(staging)
    if final_dir.exists():
        raise PublishError(f"promoted version directory already exists: {final_dir}")

    try:
        staging.mkdir(parents=True)
        shutil.copy2(run_dir / "policy.weights.h5", staging / "policy.weights.h5")
        shutil.copy2(run_dir / "metrics.json", staging / "metrics.json")
        write_promotion_manifest(
            staging,
            promoted_version=version,
            run_id=run_id,
            parent_weights=parent_weights,
            promoted_at=promoted_at_ts,
        )
        staging.rename(final_dir)

        latest = models_dir / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(version)

        append_promotion_ledger(
            ledger_path,
            promoted_version=version,
            run_id=run_id,
            promoted_at=promoted_at_ts,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        if final_dir.exists() and not ledger_path.is_file():
            shutil.rmtree(final_dir, ignore_errors=True)
        raise

    return PublishSummary(
        run_id=run_id,
        run_dir=run_dir,
        promoted_version=version,
        version_dir=final_dir,
    )
