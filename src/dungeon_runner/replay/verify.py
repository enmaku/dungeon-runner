"""Replay verifier: stepwise web game engine replay via Node harness."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dungeon_runner.replay.manifest import load_manifest
from dungeon_runner.replay.store import raw_path
from dungeon_runner.replay.verify_manifest import (
    VerifyFailure,
    VerifyManifest,
    load_verify_manifest,
    save_verify_manifest,
)
from dungeon_runner.replay.web_engine import (
    default_harness_path,
    default_node_command,
    require_portfolio_site_root,
)


@dataclass
class VerifySummary:
    verified: list[str] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _VerifyOutcome:
    match_id: str
    ok: bool
    failure: VerifyFailure | None = None


def pending_verify_ids(data_dir: Path) -> list[str]:
    ingest = load_manifest(data_dir)
    verify = load_verify_manifest(data_dir)
    known = verify.known_ids()
    pending: list[str] = []
    for match_id in ingest.ingested:
        if match_id in known:
            continue
        if not raw_path(data_dir, match_id).is_file():
            continue
        pending.append(match_id)
    return pending


def _parse_harness_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ValueError("harness produced empty stdout")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("harness stdout must be a JSON object")
    return payload


def _run_node_harness(
    *,
    envelope_path: Path,
    node_cmd: list[str],
    harness_path: Path,
    portfolio_root: Path,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["PORTFOLIO_SITE_ROOT"] = str(portfolio_root)
    proc = subprocess.run(
        [*node_cmd, str(harness_path), str(envelope_path)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"
        return {
            "ok": False,
            "failure": VerifyFailure(code="engine_error", detail=detail).to_dict(),
        }
    try:
        return _parse_harness_stdout(proc.stdout)
    except (json.JSONDecodeError, ValueError) as err:
        return {
            "ok": False,
            "failure": VerifyFailure(
                code="engine_error",
                detail=f"invalid harness JSON: {err}",
            ).to_dict(),
        }


def _verify_match(
    match_id: str,
    data_dir: Path,
    *,
    node_cmd: list[str],
    harness_path: Path,
    portfolio_root: Path,
) -> _VerifyOutcome:
    path = raw_path(data_dir, match_id)
    payload = _run_node_harness(
        envelope_path=path,
        node_cmd=node_cmd,
        harness_path=harness_path,
        portfolio_root=portfolio_root,
    )
    if payload.get("ok") is True:
        return _VerifyOutcome(match_id=match_id, ok=True)
    failure_raw = payload.get("failure")
    if isinstance(failure_raw, dict) and isinstance(failure_raw.get("code"), str):
        failure = VerifyFailure.from_dict(failure_raw)
    else:
        failure = VerifyFailure(code="engine_error", detail="harness returned ok=false without failure")
    return _VerifyOutcome(match_id=match_id, ok=False, failure=failure)


def run_verify(
    *,
    data_dir: Path,
    node_cmd: list[str] | None = None,
    harness_path: Path | None = None,
    portfolio_root: Path | None = None,
    verify_fn: Callable[..., _VerifyOutcome] | None = None,
) -> VerifySummary:
    data_dir = data_dir.resolve()
    portfolio_root = portfolio_root or require_portfolio_site_root()
    node_cmd = node_cmd or default_node_command()
    harness_path = harness_path or default_harness_path()
    if not harness_path.is_file():
        raise RuntimeError(f"verify harness not found: {harness_path}")

    pending = pending_verify_ids(data_dir)
    summary = VerifySummary()
    if not pending:
        return summary

    manifest = load_verify_manifest(data_dir)
    outcomes: list[_VerifyOutcome] = []
    do_verify = verify_fn or _verify_match

    for match_id in pending:
        outcome = do_verify(
            match_id,
            data_dir,
            node_cmd=node_cmd,
            harness_path=harness_path,
            portfolio_root=portfolio_root,
        )
        outcomes.append(outcome)

    for outcome in outcomes:
        if outcome.ok:
            manifest.verified.append(outcome.match_id)
            summary.verified.append(outcome.match_id)
        else:
            assert outcome.failure is not None
            entry = {
                "id": outcome.match_id,
                "reason": outcome.failure.to_dict(),
            }
            manifest.failed.append(entry)
            summary.failed.append(entry)

    save_verify_manifest(data_dir, manifest)
    return summary
