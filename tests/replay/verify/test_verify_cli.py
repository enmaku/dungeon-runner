"""Verify CLI smoke tests."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from dungeon_runner.replay.env import repo_root
from dungeon_runner.replay.manifest import load_manifest

from tests.replay.helpers import FIXTURES


def _run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    run_env = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "dungeon_runner.replay.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root(),
        env=run_env,
    )


def _write_mock_node(tmp_path: Path) -> Path:
    script = tmp_path / "mock_node.py"
    script.write_text(
        "#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps({\"ok\": True}))\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_cli_verify_from_export(tmp_path: Path):
    export = json.loads((FIXTURES / "valid-match-over-seed42.json").read_text(encoding="utf-8"))
    export_path = tmp_path / "export.json"
    export_path.write_text(
        json.dumps({"match-cli": export}),
        encoding="utf-8",
    )
    data_dir = tmp_path / "replays"
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()
    (portfolio / "src/features/dungeon-runner/engine").mkdir(parents=True)
    (portfolio / "src/features/dungeon-runner/nn").mkdir(parents=True)
    (portfolio / "src/features/dungeon-runner/debug").mkdir(parents=True)
    for rel in (
        "engine/kernel.js",
        "nn/policyAdapter.js",
        "debug/replaySession.js",
    ):
        target = portfolio / "src/features/dungeon-runner" / rel
        target.write_text("// stub\n", encoding="utf-8")

    mock_node = _write_mock_node(tmp_path)
    node_link = tmp_path / "node"
    node_link.write_text(mock_node.read_text(encoding="utf-8"), encoding="utf-8")
    node_link.chmod(node_link.stat().st_mode | stat.S_IEXEC)
    env = {
        "PORTFOLIO_SITE_ROOT": str(portfolio),
        "PATH": f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}",
    }

    ingest = _run_cli(
        "ingest",
        "--data-dir",
        str(data_dir),
        "--from-export",
        str(export_path),
        env=env,
    )
    assert ingest.returncode == 0, ingest.stderr

    verify = _run_cli("verify", "--data-dir", str(data_dir), env=env)
    assert verify.returncode == 0, verify.stderr
    assert "verified" in verify.stdout
    manifest = load_manifest(data_dir)
    assert "match-cli" in manifest.ingested
    verify_manifest = json.loads((data_dir / "verify_manifest.json").read_text(encoding="utf-8"))
    assert "match-cli" in verify_manifest["verified"]


def test_cli_verify_fails_without_portfolio_root(tmp_path: Path):
    proc = _run_cli(
        "verify",
        "--data-dir",
        str(tmp_path / "replays"),
        env={"PORTFOLIO_SITE_ROOT": ""},
    )
    assert proc.returncode == 1
    assert "PORTFOLIO_SITE_ROOT" in proc.stderr
    assert "Traceback" not in proc.stderr
