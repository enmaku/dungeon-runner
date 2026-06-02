"""Firestore Admin SDK helpers for completed match outcomes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dungeon_runner.replay.env import load_dotenv

MATCH_OUTCOMES_COLLECTION = "dungeonRunnerMatchOutcomes"


@runtime_checkable
class OutcomeFirestoreClient(Protocol):
    def doc_exists(self, match_id: str) -> bool: ...

    def create_outcome(self, match_id: str, outcome: dict[str, Any]) -> None: ...


class FirebaseOutcomeFirestoreClient:
    def __init__(self, *, credentials_path: Path) -> None:
        from google.oauth2 import service_account
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred = credentials.Certificate(str(credentials_path))
        try:
            firebase_admin.get_app()
        except ValueError:
            firebase_admin.initialize_app(cred)
        self._db = firestore.client()

    def doc_exists(self, match_id: str) -> bool:
        snap = self._db.collection(MATCH_OUTCOMES_COLLECTION).document(match_id).get()
        return snap.exists

    def create_outcome(self, match_id: str, outcome: dict[str, Any]) -> None:
        ref = self._db.collection(MATCH_OUTCOMES_COLLECTION).document(match_id)
        ref.create(outcome)


def require_credentials_path() -> Path:
    load_dotenv()
    raw = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if not raw:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is required for backfill-outcomes; "
            "set it to a Firebase service account JSON path"
        )
    path = Path(raw).expanduser()
    if not path.is_file():
        raise RuntimeError(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {path}")
    return path


def require_outcome_firestore() -> FirebaseOutcomeFirestoreClient:
    return FirebaseOutcomeFirestoreClient(credentials_path=require_credentials_path())
