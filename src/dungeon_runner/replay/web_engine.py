"""Resolve portfolio-site web game engine root for Node harness."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dungeon_runner.replay.env import load_dotenv

_HARNESS = Path(__file__).resolve().parent / "harness" / "verify_match.mjs"
_REQUIRED_SUFFIXES = (
    "src/features/dungeon-runner/engine/kernel.js",
    "src/features/dungeon-runner/nn/policyAdapter.js",
    "src/features/dungeon-runner/debug/replaySession.js",
)


def default_harness_path() -> Path:
    return _HARNESS


def require_portfolio_site_root() -> Path:
    load_dotenv()
    raw = os.environ.get("PORTFOLIO_SITE_ROOT", "").strip()
    if not raw:
        raise RuntimeError(
            "PORTFOLIO_SITE_ROOT is required for verify; set it in .env to your "
            "portfolio-site checkout"
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"PORTFOLIO_SITE_ROOT is not a directory: {root}")
    for suffix in _REQUIRED_SUFFIXES:
        path = root / suffix
        if not path.is_file():
            raise RuntimeError(f"PORTFOLIO_SITE_ROOT missing required file: {suffix}")
    return root


def default_node_command() -> list[str]:
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node is required for verify but was not found on PATH")
    return [node]
