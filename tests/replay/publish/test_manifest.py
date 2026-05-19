"""Promotion manifest: per-dir promotion.json + append-only JSONL."""

from __future__ import annotations

import json

from dungeon_runner.replay.publish.manifest import (
    append_promotion_ledger,
    load_promoted_run_ids,
    write_promotion_manifest,
)


def test_write_promotion_manifest_fields(tmp_path):
    version_dir = tmp_path / "v0.2"
    version_dir.mkdir()
    write_promotion_manifest(
        version_dir,
        promoted_version="v0.2",
        run_id="bc-20260518T120000Z",
        parent_weights="/abs/models/latest/policy.weights.h5",
        promoted_at="2026-05-18T12:00:00+00:00",
    )
    data = json.loads((version_dir / "promotion.json").read_text())
    assert data["promoted_version"] == "v0.2"
    assert data["run_id"] == "bc-20260518T120000Z"
    assert data["metrics_file"] == "metrics.json"


def test_append_promotion_ledger_creates_jsonl(tmp_path):
    ledger = tmp_path / "promotions.jsonl"
    append_promotion_ledger(
        ledger,
        promoted_version="v0.2",
        run_id="bc-20260518T120000Z",
        promoted_at="2026-05-18T12:00:00+00:00",
    )
    lines = ledger.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["promoted_version"] == "v0.2"
    assert row["run_id"] == "bc-20260518T120000Z"


def test_load_promoted_run_ids_from_ledger(tmp_path):
    ledger = tmp_path / "promotions.jsonl"
    ledger.write_text(
        '{"promoted_version":"v0.2","run_id":"bc-a","promoted_at":"t"}\n'
        '{"promoted_version":"v0.2.01","run_id":"bc-b","promoted_at":"t"}\n'
    )
    assert load_promoted_run_ids(ledger) == {"bc-a", "bc-b"}
