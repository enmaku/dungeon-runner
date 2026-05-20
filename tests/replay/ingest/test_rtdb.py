"""Live RTDB ingest with mocked HTTP."""

import json
from pathlib import Path
from unittest.mock import patch

from dungeon_runner.replay.ingest import run_ingest
from dungeon_runner.replay.rtdb import RtdbClient


def _v1_envelope():
    return {
        "createdAt": "2026-05-19T12:00:00.000Z",
        "seed": 1,
        "setup": {"totalSeats": 4, "opponents": []},
        "history": [],
        "version": 1,
    }


def _wire(envelope: dict) -> tuple[dict, bytes]:
    body = json.dumps(envelope, separators=(",", ":")).encode("utf-8") + b"\n"
    return envelope, body


def test_rtdb_ingest_fetches_only_pending_ids(tmp_path: Path):
    data_dir = tmp_path / "replays"
    client = RtdbClient(database_url="https://test.firebaseio.com")

    with patch.object(client, "list_match_ids", return_value=["match-a", "match-b"]):
        with patch.object(
            client,
            "fetch_match_with_raw",
            side_effect=lambda mid: _wire({**_v1_envelope(), "seed": ord(mid[-1])}),
        ):
            first = run_ingest(data_dir=data_dir, rtdb_client=client)

    assert first.ingested == ["match-a", "match-b"]

    with patch.object(client, "list_match_ids", return_value=["match-a", "match-b", "match-c"]):
        with patch.object(
            client,
            "fetch_match_with_raw",
            side_effect=lambda mid: (
                _wire(_v1_envelope()) if mid == "match-c" else (_ for _ in ()).throw(AssertionError)
            ),
        ):
            second = run_ingest(data_dir=data_dir, rtdb_client=client)

    assert second.ingested == ["match-c"]
    assert second.skipped == []


def test_rtdb_client_builds_shallow_and_per_match_urls(tmp_path: Path):
    client = RtdbClient(database_url="https://proj-default-rtdb.firebaseio.com")
    seen: list[str] = []

    def fake_get_raw(url: str):
        seen.append(url)
        if "shallow=true" in url:
            return (b'{"match-x":true}', {"match-x": True})
        if url.endswith("/match-x.json"):
            env = _v1_envelope()
            raw = json.dumps(env, separators=(",", ":")).encode("utf-8")
            return raw, env
        return b"", None

    with patch.object(client, "_get_raw", side_effect=fake_get_raw):
        summary = run_ingest(
            data_dir=tmp_path / "replays",
            rtdb_client=client,
        )

    assert summary.ingested == ["match-x"]
    assert seen[0].endswith("/dungeonRunnerCompletedMatches.json?shallow=true")
    assert seen[1].endswith("/dungeonRunnerCompletedMatches/match-x.json")
