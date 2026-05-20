"""Raw envelope store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def raw_path(data_dir: Path, match_id: str) -> Path:
    return data_dir / "raw" / f"{match_id}.json"


def write_raw_bytes(data_dir: Path, match_id: str, raw_bytes: bytes) -> None:
    path = raw_path(data_dir, match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_bytes)


def write_raw_envelope(data_dir: Path, match_id: str, envelope: dict[str, Any]) -> None:
    raw_bytes = (
        json.dumps(envelope, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    write_raw_bytes(data_dir, match_id, raw_bytes)
