"""Verify CLI smoke tests."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

from dungeon_runner.replay.manifest import load_manifest

from tests.replay.helpers import FIXTURES


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
    repo = Path(__file__).resolve().parents[2]
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
    env = os.environ.copy()
    env["PORTFOLIO_SITE_ROOT"] = str(portfolio)
    env["PATH"] = f"{tmp_path}:{env.get('PATH', '')}"
    # Replace node with mock via PATH prefix only works if harness uses 'node' from PATH.
    # Inject mock by shadowing: symlink mock as node
    node_link = tmp_path / "node"
    node_link.write_text(mock_node.read_text(encoding="utf-8"), encoding="utf-8")
    node_link.chmod(node_link.stat().st_mode | stat.S_IEXEC)
    env["PATH"] = f"{tmp_path}{os.pathsep}{env.get('PATH', '')}"

    ingest = subprocess.run(
        [
            sys.executable,
            "-m",
            "dungeon_runner.replay.cli",
            "ingest",
            "--data-dir",
            str(data_dir),
            "--from-export",
            str(export_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo,
    )
    assert ingest.returncode == 0, ingest.stderr

    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "dungeon_runner.replay.cli",
            "verify",
            "--data-dir",
            str(data_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo,
        env=env,
    )
    assert verify.returncode == 0, verify.stderr
    assert "verified" in verify.stdout
    manifest = load_manifest(data_dir)
    assert "match-cli" in manifest.ingested
    verify_manifest = json.loads((data_dir / "verify_manifest.json").read_text(encoding="utf-8"))
    assert "match-cli" in verify_manifest["verified"]
