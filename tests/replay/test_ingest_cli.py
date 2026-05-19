"""CLI entrypoint smoke tests."""

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_ingest_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    cwd = Path(__file__).resolve().parents[2]
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "dungeon_runner.replay.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
        env=run_env,
    )


def test_cli_ingest_from_export_without_firebase_env(tmp_path: Path):
    export = {
        "match-offline": {
            "version": 1,
            "seed": 1,
            "setup": {"totalSeats": 4, "opponents": []},
            "history": [],
        }
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    proc = _run_ingest_cli(
        "ingest",
        "--data-dir",
        str(data_dir),
        "--from-export",
        str(export_path),
        env={"FIREBASE_DATABASE_URL": ""},
    )

    assert proc.returncode == 0, proc.stderr
    assert (data_dir / "raw" / "match-offline.json").is_file()


def test_cli_ingest_from_export(tmp_path: Path):
    export = {
        "match-cli": {
            "version": 1,
            "seed": 1,
            "setup": {"totalSeats": 4, "opponents": []},
            "history": [],
        }
    }
    export_path = tmp_path / "export.json"
    export_path.write_text(json.dumps(export), encoding="utf-8")
    data_dir = tmp_path / "replays"

    proc = _run_ingest_cli(
        "ingest",
        "--data-dir",
        str(data_dir),
        "--from-export",
        str(export_path),
    )

    assert proc.returncode == 0, proc.stderr
    assert "match-cli" in proc.stdout
    assert (data_dir / "raw" / "match-cli.json").is_file()
