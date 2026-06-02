"""Firestore Admin helpers for backfill-outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dungeon_runner.replay.firestore_admin import (
    MATCH_OUTCOMES_COLLECTION,
    FirebaseOutcomeFirestoreClient,
    require_outcome_firestore,
)


def test_match_outcomes_collection_name():
    assert MATCH_OUTCOMES_COLLECTION == "dungeonRunnerMatchOutcomes"


def test_require_outcome_firestore_fails_without_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_APPLICATION_CREDENTIALS"):
        require_outcome_firestore()


def test_require_outcome_firestore_fails_when_credentials_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    missing = tmp_path / "missing-sa.json"
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(missing))
    with pytest.raises(RuntimeError, match="not found"):
        require_outcome_firestore()


def test_firebase_client_doc_exists_and_create(tmp_path: Path):
    creds = tmp_path / "sa.json"
    creds.write_text(json.dumps({"project_id": "test-proj"}), encoding="utf-8")

    mock_db = MagicMock()
    mock_snap = MagicMock()
    mock_snap.exists = True
    mock_db.collection.return_value.document.return_value.get.return_value = mock_snap

    with patch("firebase_admin.credentials.Certificate", return_value="cert"):
        with patch("firebase_admin.get_app", side_effect=ValueError("no app")):
            with patch("firebase_admin.initialize_app"):
                with patch("firebase_admin.firestore.client", return_value=mock_db):
                    client = FirebaseOutcomeFirestoreClient(credentials_path=creds)
                    assert client.doc_exists("match-1") is True

                    mock_snap.exists = False
                    outcome = {"matchId": "match-1", "humanWon": True}
                    client.create_outcome("match-1", outcome)

    mock_db.collection.assert_called_with(MATCH_OUTCOMES_COLLECTION)
    mock_db.collection.return_value.document.return_value.create.assert_called_once_with(
        outcome
    )
