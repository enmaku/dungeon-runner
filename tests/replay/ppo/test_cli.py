"""PPO CLI stage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
    )
    assert proc.returncode == 1
    assert "bc-run" in proc.stderr.lower() or "prerequisite" in proc.stderr.lower()
