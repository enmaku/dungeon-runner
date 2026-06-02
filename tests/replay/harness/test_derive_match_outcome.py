"""Node derive_match_outcome.test.mjs harness (requires PORTFOLIO_SITE_ROOT)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from dungeon_runner.replay.web_engine import default_node_command

_DERIVE_TEST = "tests/replay/harness/derive_match_outcome.test.mjs"
_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.skipif(
    not os.environ.get("PORTFOLIO_SITE_ROOT", "").strip(),
    reason="PORTFOLIO_SITE_ROOT not set",
)
def test_node_harness_derive_match_outcome(skip_without_portfolio: Path) -> None:
    env = os.environ.copy()
    env["PORTFOLIO_SITE_ROOT"] = str(skip_without_portfolio)
    proc = subprocess.run(
        [
            *default_node_command(),
            "--test",
            _DERIVE_TEST,
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
