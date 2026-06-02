"""backfill-outcomes stage with faked RTDB, Firestore, and derive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dungeon_runner.replay.backfill_outcomes import (
    BackfillSummary,
    run_backfill_outcomes,
)
from dungeon_runner.replay.rtdb import RtdbClient


class FakeFirestore:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.existing = set(existing or ())
        self.created: list[tuple[str, dict[str, Any]]] = []

    def doc_exists(self, match_id: str) -> bool:
        return match_id in self.existing

    def create_outcome(self, match_id: str, outcome: dict[str, Any]) -> None:
        if match_id in self.existing:
            raise RuntimeError("doc already exists")
        self.existing.add(match_id)
        self.created.append((match_id, outcome))


def _envelope(match_id: str) -> dict[str, Any]:
    return {
        "version": 1,
        "createdAt": "2026-05-27T12:00:00.000Z",
        "seed": 1,
        "setup": {"totalSeats": 4, "opponents": []},
        "history": [],
    }


def _outcome(match_id: str) -> dict[str, Any]:
    return {"matchId": match_id, "humanWon": True, "createdAt": "2026-05-27T12:00:00.000Z"}


def _rtdb_with(*match_ids: str) -> RtdbClient:
    client = RtdbClient(database_url="https://test.firebaseio.com")
    envelopes = {mid: _envelope(mid) for mid in match_ids}

    def list_match_ids() -> list[str]:
        return sorted(envelopes.keys())

    def fetch_match(match_id: str) -> dict[str, Any]:
        return envelopes[match_id]

    client.list_match_ids = list_match_ids  # type: ignore[method-assign]
    client.fetch_match = fetch_match  # type: ignore[method-assign]
    return client


def test_skips_existing_firestore_docs():
    firestore = FakeFirestore(existing={"match-a"})
    summary = run_backfill_outcomes(
        rtdb_client=_rtdb_with("match-a", "match-b"),
        firestore_client=firestore,
        derive_fn=lambda _env, mid: (True, _outcome(mid), None),
        portfolio_root=Path("/portfolio"),
    )

    assert summary.written == ["match-b"]
    assert summary.skipped == [{"id": "match-a", "reason": "firestore_exists"}]
    assert summary.failed == []
    assert len(firestore.created) == 1


def test_writes_on_successful_derive():
    firestore = FakeFirestore()
    summary = run_backfill_outcomes(
        rtdb_client=_rtdb_with("match-a"),
        firestore_client=firestore,
        derive_fn=lambda _env, mid: (True, _outcome(mid), None),
        portfolio_root=Path("/portfolio"),
    )

    assert summary.written == ["match-a"]
    assert firestore.created == [("match-a", _outcome("match-a"))]


def test_dry_run_does_not_create_firestore_docs():
    firestore = FakeFirestore()
    summary = run_backfill_outcomes(
        dry_run=True,
        rtdb_client=_rtdb_with("match-a"),
        firestore_client=firestore,
        derive_fn=lambda _env, mid: (True, _outcome(mid), None),
        portfolio_root=Path("/portfolio"),
    )

    assert summary.written == ["match-a"]
    assert firestore.created == []


def test_derive_failure_increments_failed_without_write():
    firestore = FakeFirestore()
    summary = run_backfill_outcomes(
        rtdb_client=_rtdb_with("match-bad"),
        firestore_client=firestore,
        derive_fn=lambda _env, _mid: (False, None, "rng_chain_break"),
        portfolio_root=Path("/portfolio"),
    )

    assert summary.written == []
    assert summary.failed == [{"id": "match-bad", "reason": "rng_chain_break"}]
    assert firestore.created == []


def test_limit_caps_processed_match_ids():
    firestore = FakeFirestore()
    summary = run_backfill_outcomes(
        limit=1,
        rtdb_client=_rtdb_with("match-a", "match-b"),
        firestore_client=firestore,
        derive_fn=lambda _env, mid: (True, _outcome(mid), None),
        portfolio_root=Path("/portfolio"),
    )

    assert len(summary.written) + len(summary.skipped) + len(summary.failed) <= 1
    assert summary.written == ["match-a"]


def test_create_failure_surfaces_as_failed(tmp_path: Path):
    firestore = FakeFirestore()

    def boom(_match_id: str, _outcome: dict[str, Any]) -> None:
        raise RuntimeError("permission denied")

    firestore.create_outcome = boom  # type: ignore[method-assign]

    summary = run_backfill_outcomes(
        rtdb_client=_rtdb_with("match-a"),
        firestore_client=firestore,
        derive_fn=lambda _env, mid: (True, _outcome(mid), None),
        portfolio_root=Path("/portfolio"),
    )

    assert summary.written == []
    assert summary.failed == [{"id": "match-a", "reason": "permission denied"}]
