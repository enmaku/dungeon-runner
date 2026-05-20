"""Environment loading for ingest (repo-root .env, FIREBASE_DATABASE_URL)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dungeon_runner.replay.env import load_dotenv, repo_root, require_database_url


def test_load_dotenv_reads_repo_root_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "FIREBASE_DATABASE_URL=https://from-dotenv.firebaseio.com\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FIREBASE_DATABASE_URL", raising=False)
    load_dotenv(env_file)
    assert os.environ.get("FIREBASE_DATABASE_URL") == "https://from-dotenv.firebaseio.com"


def test_require_database_url_fails_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("FIREBASE_DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="FIREBASE_DATABASE_URL"):
        require_database_url()


def test_repo_root_points_at_dungeon_runner_package_root():
    root = repo_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "dungeon_runner" / "replay" / "env.py").is_file()


def _run_ingest_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    cwd = repo_root()
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "dungeon_runner.replay.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        env=run_env,
    )


def test_cli_live_ingest_fails_without_firebase_url(tmp_path: Path):
    proc = _run_ingest_cli(
        "ingest",
        "--data-dir",
        str(tmp_path / "replays"),
        env={"FIREBASE_DATABASE_URL": ""},
    )
    assert proc.returncode == 1
    assert "FIREBASE_DATABASE_URL" in proc.stderr
    assert "Traceback" not in proc.stderr
