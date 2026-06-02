"""CLI entrypoint for backfill-outcomes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dungeon_runner.replay.backfill_outcomes import BackfillSummary


def _run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    cwd = Path(__file__).resolve().parents[3]
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "dungeon_runner.replay.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        env=run_env,
    )


def test_cli_backfill_outcomes_registered():
    proc = _run_cli("backfill-outcomes", "--help")
    assert proc.returncode == 0, proc.stderr
    assert "--dry-run" in proc.stdout
    assert "--limit" in proc.stdout


def test_cli_backfill_missing_firebase_url(tmp_path: Path):
    proc = _run_cli(
        "backfill-outcomes",
        "--dry-run",
        env={
            "FIREBASE_DATABASE_URL": "",
            "GOOGLE_APPLICATION_CREDENTIALS": "",
            "PORTFOLIO_SITE_ROOT": "/tmp/portfolio",
        },
    )
    assert proc.returncode == 1
    assert "FIREBASE_DATABASE_URL" in proc.stderr


def test_cli_backfill_missing_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    proc = _run_cli(
        "backfill-outcomes",
        env={
            "FIREBASE_DATABASE_URL": "https://test.firebaseio.com",
            "PORTFOLIO_SITE_ROOT": "/tmp/portfolio",
        },
    )
    assert proc.returncode == 1
    assert "GOOGLE_APPLICATION_CREDENTIALS" in proc.stderr


def test_cli_backfill_dry_run_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    creds = tmp_path / "sa.json"
    creds.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("FIREBASE_DATABASE_URL", "https://test.firebaseio.com")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(creds))

    summary = BackfillSummary(
        written=["match-a"],
        skipped=[{"id": "match-b", "reason": "firestore_exists"}],
        failed=[],
    )
    monkeypatch.setattr(
        "dungeon_runner.replay.cli.run_backfill_outcomes",
        lambda **_: summary,
    )

    from dungeon_runner.replay.cli import main
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(["backfill-outcomes", "--dry-run"])
    out = buf.getvalue()

    assert code == 0
    assert "would write 1" in out
    assert "match-a" in out
    assert "would skip 1" in out
    assert "firestore_exists" in out


def test_cli_backfill_help_has_no_force_overwrite():
    proc = _run_cli("backfill-outcomes", "--help")
    assert proc.returncode == 0
    assert "--force" not in proc.stdout
