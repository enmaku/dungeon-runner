"""Backfill Firestore completed match outcomes from RTDB replay envelopes."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dungeon_runner.replay.firestore_admin import OutcomeFirestoreClient, require_outcome_firestore
from dungeon_runner.replay.rtdb import RtdbClient
from dungeon_runner.replay.web_engine import (
    default_node_command,
    require_portfolio_site_root,
)

_DERIVE_HARNESS = (
    Path(__file__).resolve().parent / "harness" / "derive_match_outcome.mjs"
)

DeriveFn = Callable[[dict[str, Any], str], tuple[bool, dict[str, Any] | None, str | None]]


@dataclass
class BackfillSummary:
    written: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)


def _parse_derive_failure(stderr: str) -> str:
    text = stderr.strip()
    if not text:
        return "derive_failed"
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        failure = payload.get("failure")
        if isinstance(failure, dict):
            code = failure.get("code")
            if isinstance(code, str) and code:
                step = failure.get("step")
                if step is not None:
                    return f"{code} step={step}"
                return code
    return text.splitlines()[-1][:500]


def _default_derive(
    envelope: dict[str, Any],
    match_id: str,
    *,
    portfolio_root: Path,
    node_cmd: list[str] | None = None,
    harness_path: Path | None = None,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    node = node_cmd or default_node_command()
    harness = harness_path or _DERIVE_HARNESS
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        json.dump(envelope, tmp, separators=(",", ":"))
        tmp_path = Path(tmp.name)

    try:
        env = os.environ.copy()
        env["PORTFOLIO_SITE_ROOT"] = str(portfolio_root)
        proc = subprocess.run(
            [*node, str(harness), str(tmp_path), match_id],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        return False, None, _parse_derive_failure(proc.stderr)

    stdout = proc.stdout.strip()
    if not stdout:
        return False, None, "derive_empty_stdout"
    try:
        outcome = json.loads(stdout)
    except json.JSONDecodeError as err:
        return False, None, f"derive_invalid_json: {err}"
    if not isinstance(outcome, dict):
        return False, None, "derive_invalid_outcome"
    return True, outcome, None


def run_backfill_outcomes(
    *,
    dry_run: bool = False,
    limit: int | None = None,
    database_url: str | None = None,
    rtdb_client: RtdbClient | None = None,
    firestore_client: OutcomeFirestoreClient | None = None,
    derive_fn: DeriveFn | None = None,
    portfolio_root: Path | None = None,
    node_cmd: list[str] | None = None,
    harness_path: Path | None = None,
) -> BackfillSummary:
    portfolio = portfolio_root or require_portfolio_site_root()
    client = rtdb_client or RtdbClient(database_url=database_url)
    firestore = firestore_client or require_outcome_firestore()

    if derive_fn is None:

        def derive_fn(envelope: dict[str, Any], match_id: str) -> tuple[bool, dict[str, Any] | None, str | None]:
            return _default_derive(
                envelope,
                match_id,
                portfolio_root=portfolio,
                node_cmd=node_cmd,
                harness_path=harness_path,
            )

    match_ids = client.list_match_ids()
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        match_ids = match_ids[:limit]

    summary = BackfillSummary()
    for match_id in match_ids:
        if firestore.doc_exists(match_id):
            summary.skipped.append({"id": match_id, "reason": "firestore_exists"})
            continue

        envelope = client.fetch_match(match_id)
        ok, outcome, reason = derive_fn(envelope, match_id)
        if not ok or outcome is None:
            summary.failed.append({"id": match_id, "reason": reason or "derive_failed"})
            continue

        if dry_run:
            summary.written.append(match_id)
            continue

        try:
            firestore.create_outcome(match_id, outcome)
        except Exception as exc:
            summary.failed.append({"id": match_id, "reason": str(exc)})
            continue

        summary.written.append(match_id)

    return summary
