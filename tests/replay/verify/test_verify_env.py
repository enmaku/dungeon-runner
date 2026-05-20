"""PORTFOLIO_SITE_ROOT / web engine root resolution."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dungeon_runner.replay.env import load_dotenv
from dungeon_runner.replay.web_engine import require_portfolio_site_root


def test_require_portfolio_site_root_fails_when_unset(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("PORTFOLIO_SITE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="PORTFOLIO_SITE_ROOT"):
        require_portfolio_site_root()


def test_require_portfolio_site_root_fails_when_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "portfolio"
    (root / "src/features/dungeon-runner/engine").mkdir(parents=True)
    (root / "src/features/dungeon-runner/engine/kernel.js").write_text("// stub\n")
    monkeypatch.setenv("PORTFOLIO_SITE_ROOT", str(root))
    with pytest.raises(RuntimeError, match="missing required file"):
        require_portfolio_site_root()


def test_load_dotenv_sets_portfolio_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    skip_without_portfolio: Path,
):
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"PORTFOLIO_SITE_ROOT={skip_without_portfolio}\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PORTFOLIO_SITE_ROOT", raising=False)
    load_dotenv(env_file)
    assert os.environ.get("PORTFOLIO_SITE_ROOT") == str(skip_without_portfolio)
    assert require_portfolio_site_root() == skip_without_portfolio
