"""CLI stub stages mention eval metrics wiring (issue #5 pass 2)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_bc_missing_prerequisites_exits_one():
    repo = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-m", "dungeon_runner.replay.cli", "bc", "--no-gate-preview"],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    assert proc.returncode == 1
    assert "prerequisite" in proc.stderr.lower() or "eval suite" in proc.stderr.lower()
