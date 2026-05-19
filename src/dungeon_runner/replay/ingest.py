"""Live and offline replay ingest."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dungeon_runner.replay.eligibility import eligibility_skip_reason
from dungeon_runner.replay.manifest import IngestManifest, load_manifest, save_manifest
from dungeon_runner.replay.rtdb import RtdbClient
from dungeon_runner.replay.store import raw_path, write_raw_bytes, write_raw_envelope


@dataclass
class IngestSummary:
    ingested: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)


@dataclass
class _PendingMatch:
    match_id: str
    envelope: dict[str, Any]
    raw_bytes: bytes | None


def _pending_ids(all_ids: list[str], manifest: IngestManifest) -> list[str]:
    known = manifest.known_ids()
    return [match_id for match_id in sorted(all_ids) if match_id not in known]


def _collect_export_pending(
    path: Path, manifest: IngestManifest
) -> list[_PendingMatch]:
    envelopes = _load_export(path)
    pending_ids = _pending_ids(list(envelopes.keys()), manifest)
    return [
        _PendingMatch(match_id=match_id, envelope=envelopes[match_id], raw_bytes=None)
        for match_id in pending_ids
    ]


def _collect_rtdb_pending(
    client: RtdbClient, manifest: IngestManifest
) -> list[_PendingMatch]:
    all_ids = client.list_match_ids()
    pending_ids = _pending_ids(all_ids, manifest)
    pending: list[_PendingMatch] = []
    for match_id in pending_ids:
        envelope, raw_bytes = client.fetch_match_with_raw(match_id)
        pending.append(
            _PendingMatch(match_id=match_id, envelope=envelope, raw_bytes=raw_bytes)
        )
    return pending


def _rollback_written_raw(paths: list[Path]) -> None:
    raw_dirs: set[Path] = set()
    for path in paths:
        raw_dirs.add(path.parent)
        path.unlink(missing_ok=True)
    for raw_dir in raw_dirs:
        if raw_dir.is_dir() and not any(raw_dir.iterdir()):
            raw_dir.rmdir()


def _apply_pending(
    pending: list[_PendingMatch],
    *,
    data_dir: Path,
    manifest: IngestManifest,
    summary: IngestSummary,
) -> list[Path]:
    manifest_before = (list(manifest.ingested), list(manifest.skipped))
    summary_before = (list(summary.ingested), list(summary.skipped))
    written_paths: list[Path] = []
    try:
        for item in pending:
            reason = eligibility_skip_reason(item.envelope)
            if reason is not None:
                manifest.skipped.append({"id": item.match_id, "reason": reason})
                summary.skipped.append({"id": item.match_id, "reason": reason})
                continue
            if item.raw_bytes:
                write_raw_bytes(data_dir, item.match_id, item.raw_bytes)
            else:
                write_raw_envelope(data_dir, item.match_id, item.envelope)
            written_paths.append(raw_path(data_dir, item.match_id))
            manifest.ingested.append(item.match_id)
            summary.ingested.append(item.match_id)
    except Exception:
        manifest.ingested, manifest.skipped = manifest_before
        summary.ingested, summary.skipped = summary_before
        _rollback_written_raw(written_paths)
        raise
    return written_paths


def run_ingest(
    *,
    data_dir: Path,
    from_export: Path | None = None,
    database_url: str | None = None,
    rtdb_client: RtdbClient | None = None,
) -> IngestSummary:
    data_dir = data_dir.resolve()
    manifest = load_manifest(data_dir)
    summary = IngestSummary()

    if from_export is not None:
        pending = _collect_export_pending(from_export, manifest)
    else:
        client = rtdb_client or RtdbClient(database_url=database_url)
        pending = _collect_rtdb_pending(client, manifest)

    written_paths: list[Path] = []
    try:
        written_paths = _apply_pending(
            pending, data_dir=data_dir, manifest=manifest, summary=summary
        )
        save_manifest(data_dir, manifest)
    except Exception:
        _rollback_written_raw(written_paths)
        raise
    return summary


def _load_export(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"export must be a JSON object, got {type(data).__name__}")
    out: dict[str, dict[str, Any]] = {}
    for match_id, envelope in data.items():
        if not isinstance(envelope, dict):
            raise ValueError(f"match {match_id!r} must be an object")
        out[str(match_id)] = envelope
    return out
