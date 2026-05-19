"""Replace legacy duplicate models/latest/ directory with symlink to v0.1.30a."""

from __future__ import annotations

import shutil
from pathlib import Path

LEGACY_LATEST_TARGET = "v0.1.30a"


def migrate_latest_symlink(models_dir: Path) -> None:
    latest = models_dir / "latest"
    target = models_dir / LEGACY_LATEST_TARGET
    if latest.is_symlink():
        return
    if latest.is_dir() and not latest.is_symlink():
        shutil.rmtree(latest)
    if not latest.exists():
        latest.symlink_to(LEGACY_LATEST_TARGET)
    if target.is_dir() and latest.is_symlink():
        return
