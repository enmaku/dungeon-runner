"""Default derive subprocess wiring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from dungeon_runner.replay.backfill_outcomes import run_backfill_outcomes
from dungeon_runner.replay.rtdb import RtdbClient


def test_default_derive_invokes_node_harness(tmp_path: Path):
    envelope = {
        "version": 1,
        "seed": 1,
        "setup": {"totalSeats": 4, "opponents": []},
        "history": [],
    }
    outcome = {"matchId": "match-x", "humanWon": True}

    client = RtdbClient(database_url="https://test.firebaseio.com")
    client.list_match_ids = lambda: ["match-x"]  # type: ignore[method-assign]
    client.fetch_match = lambda _mid: envelope  # type: ignore[method-assign]

    firestore = MagicMock()
    firestore.doc_exists.return_value = False

    proc = MagicMock()
    proc.returncode = 0
    proc.stdout = json.dumps(outcome)
    proc.stderr = ""

    with patch("dungeon_runner.replay.backfill_outcomes.subprocess.run", return_value=proc) as run:
        summary = run_backfill_outcomes(
            rtdb_client=client,
            firestore_client=firestore,
            portfolio_root=tmp_path,
            node_cmd=["node"],
        )

    assert summary.written == ["match-x"]
    firestore.create_outcome.assert_called_once_with("match-x", outcome)
    cmd = run.call_args[0][0]
    assert cmd[0] == "node"
    assert str(cmd[1]).endswith("derive_match_outcome.mjs")
    assert cmd[3] == "match-x"
