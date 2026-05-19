"""Ray rollout collection dispatches without starting a cluster in tests."""

from __future__ import annotations

from unittest.mock import MagicMock

from dungeon_runner.replay.ppo.ray_collect import collect_rollouts


def test_collect_rollouts_uses_local_when_no_ray():
    local = MagicMock(return_value={"steps": 3})
    ray_fn = MagicMock()
    out = collect_rollouts(use_ray=False, ray_workers=8, local_fn=local, ray_fn=ray_fn)
    local.assert_called_once()
    ray_fn.assert_not_called()
    assert out == {"steps": 3}


def test_collect_rollouts_uses_ray_fn_when_enabled():
    local = MagicMock()
    ray_fn = MagicMock(return_value={"steps": 8})
    out = collect_rollouts(use_ray=True, ray_workers=4, local_fn=local, ray_fn=ray_fn)
    ray_fn.assert_called_once_with(4)
    local.assert_not_called()
    assert out["steps"] == 8
