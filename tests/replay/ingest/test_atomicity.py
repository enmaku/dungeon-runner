"""Ingest run atomicity: all-or-nothing manifest and raw writes."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dungeon_runner.replay.ingest import run_ingest
from dungeon_runner.replay.rtdb import RtdbClient


def _v1_envelope():
    return {
        "version": 1,
        "seed": 1,
        "setup": {"totalSeats": 4, "opponents": []},
        "history": [],
    }


def test_ingest_aborts_without_manifest_or_raw_on_fetch_failure(tmp_path: Path):
    data_dir = tmp_path / "replays"
    client = RtdbClient(database_url="https://test.firebaseio.com")

    with patch.object(client, "list_match_ids", return_value=["match-ok", "match-fail"]):
        with patch.object(
            client,
            "fetch_match_with_raw",
            side_effect=lambda mid: (
                (_v1_envelope(), b'{"version":1}\n')
                if mid == "match-ok"
                else (_ for _ in ()).throw(RuntimeError("network"))
            ),
        ):
            with pytest.raises(RuntimeError, match="network"):
                run_ingest(data_dir=data_dir, rtdb_client=client)

    assert not (data_dir / "manifest.json").exists()
    assert not (data_dir / "raw").exists()


def test_ingest_rolls_back_raw_on_mid_apply_write_failure(tmp_path: Path):
    export = {
        "match-a": _v1_envelope(),
        "match-b": _v1_envelope(),
        "match-c": _v1_envelope(),
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    from dungeon_runner.replay import ingest as ingest_mod
    from dungeon_runner.replay import store

    original = store.write_raw_envelope

    def fail_on_b(data_dir_arg: Path, match_id: str, envelope: dict) -> None:
        if match_id == "match-b":
            raise OSError("disk full")
        original(data_dir_arg, match_id, envelope)

    with patch.object(ingest_mod, "write_raw_envelope", side_effect=fail_on_b):
        with pytest.raises(OSError, match="disk full"):
            run_ingest(data_dir=data_dir, from_export=export_path)

    assert not (data_dir / "manifest.json").exists()
    assert not (data_dir / "raw").exists()


def test_ingest_mid_apply_failure_preserves_prior_ingested_raw(tmp_path: Path):
    export = {
        "match-ok": _v1_envelope(),
        "match-new-a": _v1_envelope(),
        "match-new-b": _v1_envelope(),
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    first_export = {"match-ok": export["match-ok"]}
    first_path = tmp_path / "first.json"
    first_path.write_text(json.dumps(first_export), encoding="utf-8")
    run_ingest(data_dir=data_dir, from_export=first_path)

    from dungeon_runner.replay import ingest as ingest_mod
    from dungeon_runner.replay import store

    original = store.write_raw_envelope
    seen: list[str] = []

    def fail_on_new_b(data_dir_arg: Path, match_id: str, envelope: dict) -> None:
        seen.append(match_id)
        if match_id == "match-new-b":
            raise OSError("disk full")
        original(data_dir_arg, match_id, envelope)

    with patch.object(ingest_mod, "write_raw_envelope", side_effect=fail_on_new_b):
        with pytest.raises(OSError, match="disk full"):
            run_ingest(data_dir=data_dir, from_export=export_path)

    assert seen == ["match-new-a", "match-new-b"]
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ingested"] == ["match-ok"]
    assert (data_dir / "raw" / "match-ok.json").is_file()
    assert not (data_dir / "raw" / "match-new-a.json").exists()
    assert not (data_dir / "raw" / "match-new-b.json").exists()


def test_ingest_commits_only_after_all_pending_processed(tmp_path: Path):
    export = {
        "match-a": _v1_envelope(),
        "match-b": {**_v1_envelope(), "seed": 2},
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    summary = run_ingest(data_dir=data_dir, from_export=export_path)

    assert summary.ingested == ["match-a", "match-b"]
    assert (data_dir / "manifest.json").is_file()
    assert (data_dir / "raw" / "match-a.json").is_file()
    assert (data_dir / "raw" / "match-b.json").is_file()


def test_ingest_rolls_back_raw_when_manifest_save_fails(tmp_path: Path):
    export = {
        "match-a": _v1_envelope(),
        "match-b": _v1_envelope(),
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    from dungeon_runner.replay import ingest as ingest_mod

    def fail_save(data_dir_arg: Path, manifest) -> None:
        raise OSError("manifest write failed")

    with patch.object(ingest_mod, "save_manifest", side_effect=fail_save):
        with pytest.raises(OSError, match="manifest write failed"):
            run_ingest(data_dir=data_dir, from_export=export_path)

    assert not (data_dir / "manifest.json").exists()
    assert not (data_dir / "raw").exists() or list((data_dir / "raw").iterdir()) == []
