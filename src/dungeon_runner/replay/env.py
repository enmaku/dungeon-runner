"""Load pipeline environment from repo-root `.env`."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv as _load_dotenv


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (repo_root() / ".env")
    if env_path.is_file():
        _load_dotenv(env_path, override=False)


def require_database_url() -> str:
    load_dotenv()
    url = os.environ.get("FIREBASE_DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "FIREBASE_DATABASE_URL is required for live v1 RTDB ingest; "
            "set it in .env or use ingest --from-export for offline ingest"
        )
    return url
