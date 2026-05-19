"""Shared Ray local-cluster setup for replay PPO and scripts/train_rllib."""

from __future__ import annotations

import os

try:
    import ray
except ImportError:  # pragma: no cover
    ray = None  # type: ignore[assignment, misc]


class RayRolloutError(RuntimeError):
    """Ray rollout pool failed; use --no-ray for single-process fallback."""


def init_ray_local_cluster() -> None:
    """Start a local Ray cluster if not already running."""
    if ray is None:
        raise RayRolloutError(
            "Ray is not installed; use --no-ray or pip install -e '.[train]'"
        )
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    if ray.is_initialized():  # type: ignore[union-attr]
        return
    try:
        ray.init(ignore_reinit_error=True, include_dashboard=False)  # type: ignore[union-attr]
    except Exception as exc:  # pragma: no cover - platform-specific
        raise RayRolloutError(
            f"Ray failed to start ({exc!r}); retry with --no-ray for single-process rollouts"
        ) from exc
