"""Firebase RTDB REST client for completed matches."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dungeon_runner.replay.env import require_database_url

RTDB_MATCHES_PATH = "dungeonRunnerCompletedMatches"


class RtdbClient:
    def __init__(self, *, database_url: str | None = None) -> None:
        base = (database_url or require_database_url()).rstrip("/")
        self._base = base
        self._matches_base = f"{base}/{RTDB_MATCHES_PATH}"

    def list_match_ids(self) -> list[str]:
        url = f"{self._matches_base}.json?shallow=true"
        data = self._get_json(url)
        if data is None:
            return []
        if not isinstance(data, dict):
            raise ValueError("shallow listing must return an object")
        return sorted(data.keys())

    def fetch_match(self, match_id: str) -> dict[str, Any]:
        envelope, _raw = self.fetch_match_with_raw(match_id)
        return envelope

    def fetch_match_with_raw(self, match_id: str) -> tuple[dict[str, Any], bytes]:
        url = f"{self._matches_base}/{match_id}.json"
        raw_bytes, data = self._get_raw(url)
        if not isinstance(data, dict):
            raise ValueError(f"match {match_id!r} payload must be an object")
        return data, raw_bytes

    def _get_json(self, url: str) -> Any:
        _raw, data = self._get_raw(url)
        return data

    def _get_raw(self, url: str) -> tuple[bytes, Any]:
        request = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=60) as response:
                body = response.read()
        except HTTPError as exc:
            if exc.code == 404:
                return b"", None
            raise
        if body.strip() in (b"", b"null"):
            return body, None
        text = body.decode("utf-8")
        return body, json.loads(text)
