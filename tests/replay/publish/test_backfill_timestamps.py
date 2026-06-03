"""Backfill promotion timestamps from portfolio models.json."""

from __future__ import annotations

import json
from pathlib import Path

from dungeon_runner.replay.publish.backfill_timestamps import run_backfill_timestamps


def _write_catalog(path: Path, models: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": models}, indent=2) + "\n")


def test_backfill_updates_ledger_and_promotion_json(tmp_path):
    repo = tmp_path / "repo"
    models = repo / "models"
    (models / "v0.2").mkdir(parents=True)
    (models / "v0.2" / "promotion.json").write_text(
        json.dumps(
            {
                "promoted_version": "v0.2",
                "run_id": "bc-a",
                "parent_weights": "x",
                "promoted_at": "2026-05-18T12:00:00+00:00",
                "metrics_file": "metrics.json",
            }
        )
        + "\n"
    )
    (models / "promotions.jsonl").write_text(
        '{"promoted_version":"v0.2","run_id":"bc-a","promoted_at":"2026-05-18T12:00:00+00:00"}\n'
    )

    catalog = tmp_path / "models.json"
    _write_catalog(
        catalog,
        [{"id": "v0.2", "publishedAt": "2026-06-01T00:00:00.000Z"}],
    )

    summary = run_backfill_timestamps(
        repo_root=repo,
        catalog_path=catalog,
        dry_run=False,
    )
    assert len(summary.changes) == 2

    manifest = json.loads((models / "v0.2" / "promotion.json").read_text())
    assert manifest["promoted_at"] == "2026-06-01T00:00:00.000Z"
    row = json.loads((models / "promotions.jsonl").read_text().strip())
    assert row["promoted_at"] == "2026-06-01T00:00:00.000Z"


def test_backfill_creates_legacy_promotion_json_without_ledger_line(tmp_path):
    repo = tmp_path / "repo"
    models = repo / "models"
    (models / "v0.1.30a").mkdir(parents=True)
    (models / "v0.1.30a" / "policy.weights.h5").write_bytes(b"w")

    catalog = tmp_path / "models.json"
    _write_catalog(
        catalog,
        [{"id": "v0.1.30a", "publishedAt": "2026-04-28T18:13:20.000Z"}],
    )

    summary = run_backfill_timestamps(
        repo_root=repo,
        catalog_path=catalog,
        dry_run=False,
    )
    assert len(summary.changes) == 1
    assert not (models / "promotions.jsonl").exists()

    manifest = json.loads((models / "v0.1.30a" / "promotion.json").read_text())
    assert manifest == {
        "promoted_version": "v0.1.30a",
        "promoted_at": "2026-04-28T18:13:20.000Z",
    }


def test_backfill_dry_run_writes_nothing(tmp_path):
    repo = tmp_path / "repo"
    models = repo / "models"
    (models / "v0.2").mkdir(parents=True)
    old = "2026-05-18T12:00:00+00:00"
    (models / "v0.2" / "promotion.json").write_text(
        json.dumps({"promoted_version": "v0.2", "promoted_at": old}) + "\n"
    )

    catalog = tmp_path / "models.json"
    _write_catalog(
        catalog,
        [{"id": "v0.2", "publishedAt": "2026-06-01T00:00:00.000Z"}],
    )

    summary = run_backfill_timestamps(
        repo_root=repo,
        catalog_path=catalog,
        dry_run=True,
    )
    assert summary.dry_run is True
    assert len(summary.changes) == 1
    manifest = json.loads((models / "v0.2" / "promotion.json").read_text())
    assert manifest["promoted_at"] == old
