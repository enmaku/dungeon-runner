"""User-facing progress lines for the replay training pipeline."""

from __future__ import annotations

import sys
from pathlib import Path


def log(message: str) -> None:
    print(message, file=sys.stdout, flush=True)


def log_step(title: str, *, detail: str | None = None) -> None:
    log("")
    log(f"=== {title} ===")
    if detail:
        log(detail)


def log_done(summary: str | None = None) -> None:
    if summary:
        log(f"  done — {summary}")
    else:
        log("  done")


def log_failed(stage: str, exit_code: int) -> None:
    log(f"  FAILED at {stage} (exit {exit_code})")


def log_tensorboard(tb_dir: Path, *, run_label: str) -> None:
    resolved = tb_dir.resolve()
    parent = resolved.parent
    log(
        f"TensorBoard ({run_label}): {resolved}\n"
        "  Start in another terminal (training venv, not started automatically):\n"
        f"    tensorboard --logdir {parent}\n"
        "  Open http://localhost:6006 — scalars appear after the first epoch/update.\n"
        "  If TensorBoard fails with No module named 'distutils', run: pip install -e \".[train]\""
    )
