"""PPO CLI stage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.replay.ppo.ppo_fixtures import REPO_ROOT


def test_ppo_missing_bc_run_exits_one():
    repo = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dungeon_runner.replay.cli",
            "ppo",
            "--bc-run",
            "models/runs/does-not-exist",
        ],
        capture_output=True,
        text=True,
        cwd=repo,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert proc.returncode == 1
    assert "bc-run" in proc.stderr.lower() or "prerequisite" in proc.stderr.lower()


def test_ppo_help_lists_bc_run_and_anchor_flags():
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dungeon_runner.replay.cli",
            "ppo",
            "--help",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")},
    )
    assert proc.returncode == 0
    assert "--bc-anchor-lambda" in proc.stdout
    assert "--bc-anchor-beta" in proc.stdout
    assert "--bc-run" in proc.stdout
    assert "--no-ray" in proc.stdout
