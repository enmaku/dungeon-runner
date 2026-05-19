"""Pytest fixtures for replay tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.replay.helpers import FIXTURES, seed_verify_state

__all__ = ["FIXTURES", "seed_verify_state"]


@pytest.fixture
def replay_fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def portfolio_root() -> Path | None:
    raw = os.environ.get("PORTFOLIO_SITE_ROOT", "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser().resolve()
    kernel = root / "src/features/dungeon-runner/engine/kernel.js"
    if kernel.is_file():
        return root
    return None


@pytest.fixture
def skip_without_portfolio(portfolio_root: Path | None) -> Path:
    if portfolio_root is None:
        pytest.skip("PORTFOLIO_SITE_ROOT not set or invalid")
    return portfolio_root
