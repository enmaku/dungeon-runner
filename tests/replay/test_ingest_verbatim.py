"""Verbatim raw envelope storage where feasible."""

import json
from pathlib import Path
from unittest.mock import patch

from dungeon_runner.replay.ingest import run_ingest
from dungeon_runner.replay.rtdb import RtdbClient


def test_rtdb_ingest_stores_response_bytes_verbatim(tmp_path: Path):
    wire = (
        b'{"version":1,"seed":99,"setup":{"totalSeats":4,"opponents":[]},'
        b'"history":[],"presentationSpeedProfile":"brisk"}\n'
    )
    client = RtdbClient(database_url="https://test.firebaseio.com")

    with patch.object(client, "list_match_ids", return_value=["match-wire"]):
        with patch.object(
            client,
            "fetch_match_with_raw",
            return_value=(json.loads(wire), wire),
        ):
            run_ingest(data_dir=tmp_path / "replays", rtdb_client=client)

    stored = (tmp_path / "replays" / "raw" / "match-wire.json").read_bytes()
    assert stored == wire
