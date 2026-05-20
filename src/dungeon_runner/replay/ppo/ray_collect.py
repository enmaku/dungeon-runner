"""Dispatch rollout collection to local or Ray-backed worker pool."""

from __future__ import annotations

from typing import Any, Callable

RayCollectFn = Callable[[int], Any]
LocalCollectFn = Callable[[], Any]


def collect_rollouts(
    *,
    use_ray: bool,
    ray_workers: int,
    local_fn: LocalCollectFn,
    ray_fn: RayCollectFn,
) -> Any:
    if use_ray:
        return ray_fn(ray_workers)
    return local_fn()
