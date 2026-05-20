"""Pending verify selection."""

from pathlib import Path

from dungeon_runner.replay.manifest import IngestManifest, save_manifest
from dungeon_runner.replay.store import write_raw_envelope
from dungeon_runner.replay.verify import pending_verify_ids
from dungeon_runner.replay.verify_manifest import VerifyManifest, save_verify_manifest

from tests.replay.helpers import load_fixture


def test_pending_verify_only_ingested_with_raw(tmp_path: Path):
    data_dir = tmp_path / "replays"
    envelope = load_fixture("valid-match-over-seed42.json")
    write_raw_envelope(data_dir, "match-ingested", envelope)
    save_manifest(
        data_dir,
        IngestManifest(
            ingested=["match-ingested"],
            skipped=[{"id": "match-skipped", "reason": "invalid_history"}],
        ),
    )
    write_raw_envelope(data_dir, "match-skipped", envelope)

    assert pending_verify_ids(data_dir) == ["match-ingested"]


def test_pending_verify_excludes_verified_and_failed(tmp_path: Path):
    data_dir = tmp_path / "replays"
    envelope = load_fixture("valid-match-over-seed42.json")
    for mid in ("match-a", "match-b", "match-c"):
        write_raw_envelope(data_dir, mid, envelope)
    save_manifest(data_dir, IngestManifest(ingested=["match-a", "match-b", "match-c"]))
    save_verify_manifest(
        data_dir,
        VerifyManifest(
            verified=["match-a"],
            failed=[{"id": "match-b", "reason": {"code": "illegal_action"}}],
        ),
    )

    assert pending_verify_ids(data_dir) == ["match-c"]


def test_manual_reverify_after_removing_failed_entry(tmp_path: Path):
    """US 11: drop id from verify manifest, keep raw/, re-run verify."""
    data_dir = tmp_path / "replays"
    envelope = load_fixture("valid-match-over-seed42.json")
    write_raw_envelope(data_dir, "match-x", envelope)
    save_manifest(data_dir, IngestManifest(ingested=["match-x"]))
    save_verify_manifest(
        data_dir,
        VerifyManifest(
            failed=[{"id": "match-x", "reason": {"code": "illegal_action", "step": 0}}],
        ),
    )

    assert pending_verify_ids(data_dir) == []

    save_verify_manifest(data_dir, VerifyManifest())
    assert pending_verify_ids(data_dir) == ["match-x"]
