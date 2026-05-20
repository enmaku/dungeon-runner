"""Ingest from local export (--from-export)."""

import json
from pathlib import Path

import pytest

from dungeon_runner.replay.ingest import run_ingest


def _v1_envelope(**extra):
    body = {
        "createdAt": "2026-05-19T12:00:00.000Z",
        "seed": 42,
        "setup": {"totalSeats": 4, "opponents": []},
        "history": [],
        "version": 1,
    }
    body.update(extra)
    return body


def test_ingest_from_export_writes_raw_and_manifest(tmp_path: Path):
    export = {
        "match-100": _v1_envelope(),
        "match-200": _v1_envelope(seed=99),
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    summary = run_ingest(data_dir=data_dir, from_export=export_path)

    assert summary.ingested == ["match-100", "match-200"]
    assert summary.skipped == []
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ingested"] == ["match-100", "match-200"]
    assert manifest["skipped"] == []
    raw_100 = json.loads((data_dir / "raw" / "match-100.json").read_text(encoding="utf-8"))
    assert raw_100["version"] == 1
    assert raw_100["seed"] == 42
    assert "content_hashes" not in manifest


def test_ingest_skips_unsupported_version(tmp_path: Path):
    export = {
        "match-bad-missing": _v1_envelope(),
        "match-bad-string": {**_v1_envelope(), "version": "1"},
        "match-bad-v2": {**_v1_envelope(), "version": 2},
    }
    del export["match-bad-missing"]["version"]
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    summary = run_ingest(data_dir=data_dir, from_export=export_path)

    assert summary.ingested == []
    assert len(summary.skipped) == 3
    assert all(s["reason"] == "unsupported_version" for s in summary.skipped)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ingested"] == []
    skipped_ids = {s["id"] for s in manifest["skipped"]}
    assert skipped_ids == {"match-bad-missing", "match-bad-string", "match-bad-v2"}
    assert not (data_dir / "raw").exists() or list((data_dir / "raw").iterdir()) == []


def test_ingest_does_not_retry_skipped_ids(tmp_path: Path):
    export = {
        "match-ok": _v1_envelope(),
        "match-bad": {**_v1_envelope(), "version": 99},
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    run_ingest(data_dir=data_dir, from_export=export_path)
    second = run_ingest(data_dir=data_dir, from_export=export_path)

    assert second.ingested == []
    assert second.skipped == []
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["skipped"]) == 1


def test_ingest_preserves_unknown_top_level_keys_in_raw(tmp_path: Path):
    export = {
        "match-meta": {
            **_v1_envelope(),
            "rulesHash": "abc",
            "extra": {"nested": True},
        }
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    run_ingest(data_dir=data_dir, from_export=export_path)

    raw = json.loads((data_dir / "raw" / "match-meta.json").read_text(encoding="utf-8"))
    assert raw["rulesHash"] == "abc"
    assert raw["extra"] == {"nested": True}


def test_ingest_skips_invalid_seed_with_granular_reason(tmp_path: Path):
    export = {"match-bad-seed": {**_v1_envelope(), "seed": "42"}}
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    summary = run_ingest(data_dir=data_dir, from_export=export_path)

    assert summary.skipped == [{"id": "match-bad-seed", "reason": "missing_seed"}]


def test_ingest_allows_optional_body_match_id_uses_path_key(tmp_path: Path):
    export = {
        "match-canonical": {
            **_v1_envelope(),
            "matchId": "match-wrong-body-id",
        }
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    run_ingest(data_dir=data_dir, from_export=export_path)

    assert (data_dir / "raw" / "match-canonical.json").is_file()
    assert not (data_dir / "raw" / "match-wrong-body-id.json").exists()
    raw = json.loads((data_dir / "raw" / "match-canonical.json").read_text(encoding="utf-8"))
    assert raw["matchId"] == "match-wrong-body-id"


def test_ingest_preserves_history_debug_metadata(tmp_path: Path):
    export = {
        "match-nn": {
            **_v1_envelope(),
            "history": [
                {
                    "action": {"type": "PASS"},
                    "actorSeatId": "seat-1",
                    "rngStepBefore": 0,
                    "rngStepAfter": 1,
                    "__debug": {"policyLogits": [0.1, 0.9]},
                }
            ],
        }
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    run_ingest(data_dir=data_dir, from_export=export_path)

    raw = json.loads((data_dir / "raw" / "match-nn.json").read_text(encoding="utf-8"))
    assert raw["history"][0]["__debug"] == {"policyLogits": [0.1, 0.9]}


def test_manual_reingest_after_removing_skipped_entry(tmp_path: Path):
    export = {
        "match-was-bad": {**_v1_envelope(), "version": 99},
        "match-was-bad-fixed": _v1_envelope(),
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps({"match-was-bad": export["match-was-bad"]}), encoding="utf-8")
    data_dir = tmp_path / "replays"

    run_ingest(data_dir=data_dir, from_export=export_path)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert any(s["id"] == "match-was-bad" for s in manifest["skipped"])

    manifest["skipped"] = []
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    export_path.write_text(json.dumps(export), encoding="utf-8")

    second = run_ingest(data_dir=data_dir, from_export=export_path)

    assert second.ingested == ["match-was-bad-fixed"]
    assert (data_dir / "raw" / "match-was-bad-fixed.json").is_file()


def test_manual_reingest_after_removing_manifest_entry_and_raw(tmp_path: Path):
    export = {
        "match-re": _v1_envelope(),
        "match-re-v2": {**_v1_envelope(), "seed": 99},
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    run_ingest(data_dir=data_dir, from_export=export_path)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ingested"] = [mid for mid in manifest["ingested"] if mid != "match-re"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (data_dir / "raw" / "match-re.json").unlink()

    second = run_ingest(data_dir=data_dir, from_export=export_path)

    assert "match-re" in second.ingested
    assert "match-re-v2" not in second.ingested
    raw = json.loads((data_dir / "raw" / "match-re.json").read_text(encoding="utf-8"))
    assert raw["seed"] == 42


def test_ingest_second_run_is_incremental_noop(tmp_path: Path):
    export = {"match-1": _v1_envelope()}
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    first = run_ingest(data_dir=data_dir, from_export=export_path)
    second = run_ingest(data_dir=data_dir, from_export=export_path)

    assert first.ingested == ["match-1"]
    assert second.ingested == []
    assert second.skipped == []
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ingested"] == ["match-1"]
