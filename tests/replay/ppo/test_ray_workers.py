"""Ray parallel replay PPO rollout pool (worker split + merge)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dungeon_runner.rl.ppo import RolloutBatch, RolloutGameStats
from dungeon_runner.rl import rllib_keras_module as rkm


def test_per_worker_rollout_target_splits_remainder():
    from dungeon_runner.replay.ppo.ray_workers import per_worker_rollout_target

    total, workers = 256, 8
    parts = [per_worker_rollout_target(total, workers, i) for i in range(workers)]
    assert sum(parts) == total
    assert parts == [32] * 8

    total, workers = 10, 3
    parts = [per_worker_rollout_target(total, workers, i) for i in range(workers)]
    assert sum(parts) == total
    assert sorted(parts, reverse=True) == [4, 3, 3]


def test_merge_rollout_payloads_combines_batches_and_dominant_template():
    from dungeon_runner.replay.ppo.ray_workers import merge_rollout_payloads

    b1 = RolloutBatch()
    for _ in range(3):
        b1.obs.append(np.zeros(4, np.float32))
        b1.mask.append(np.ones(2, np.float32))
        b1.act.append(0)
        b1.logp.append(-1.0)
        b1.value.append(0.0)
        b1.reward.append(0.0)
        b1.done.append(False)
    b2 = RolloutBatch()
    for _ in range(5):
        b2.obs.append(np.zeros(4, np.float32))
        b2.mask.append(np.ones(2, np.float32))
        b2.act.append(1)
        b2.logp.append(-2.0)
        b2.value.append(0.1)
        b2.reward.append(0.1)
        b2.done.append(False)

    payloads = [
        (rkm.pack_rollout_for_ray(b1), rkm.pack_game_stats_for_ray(RolloutGameStats(env_steps=3)), "vs_randombot"),
        (rkm.pack_rollout_for_ray(b2), rkm.pack_game_stats_for_ray(RolloutGameStats(env_steps=5)), "vs_bc_bot"),
    ]
    batch, stats, template = merge_rollout_payloads(payloads)
    assert len(batch) == 8
    assert stats.env_steps == 8
    assert template == "vs_bc_bot"


def test_per_worker_rollout_target_with_zero_workers_returns_full_target():
    from dungeon_runner.replay.ppo.ray_workers import per_worker_rollout_target

    assert per_worker_rollout_target(256, 0, 0) == 256


def test_ray_pool_clamps_zero_workers_to_one():
    pytest.importorskip("ray")
    from dungeon_runner.replay.ppo.ray_workers import RayRolloutPool

    with patch("dungeon_runner.replay.ppo.ray_workers.init_ray_local_cluster"):
        with patch("dungeon_runner.replay.ppo.ray_workers._ReplayRolloutActor") as actor_cls:
            actor_cls.remote.return_value = MagicMock()
            pool = RayRolloutPool(
                teacher_weights=Path("/tmp/teacher.h5"),
                n_workers=0,
                seed=1,
            )
    assert pool._n_workers == 1  # noqa: SLF001
    assert len(pool._actors) == 1  # noqa: SLF001


def test_ray_pool_init_failure_raises_ray_rollout_error():
    pytest.importorskip("ray")
    from dungeon_runner.rl.ray_local import RayRolloutError
    from dungeon_runner.replay.ppo.ray_workers import RayRolloutPool

    with patch(
        "dungeon_runner.replay.ppo.ray_workers.init_ray_local_cluster",
        side_effect=RayRolloutError("Ray failed to start"),
    ):
        with pytest.raises(RayRolloutError, match="Ray failed to start"):
            RayRolloutPool(
                teacher_weights=Path("/tmp/teacher.h5"),
                n_workers=2,
                seed=1,
            )


def test_ray_pool_shutdown_is_idempotent_and_kills_actors():
    pytest.importorskip("ray")
    from dungeon_runner.replay.ppo.ray_workers import RayRolloutPool

    actor = MagicMock()
    with patch("dungeon_runner.replay.ppo.ray_workers.ray") as mock_ray:
        pool = RayRolloutPool.__new__(RayRolloutPool)
        pool._actors = [actor, MagicMock()]  # noqa: SLF001
        pool.shutdown()
        assert pool._actors == []  # noqa: SLF001
        assert mock_ray.kill.call_count == 2
        pool.shutdown()
        assert mock_ray.kill.call_count == 2


def test_ray_pool_collect_wraps_ray_get_failure():
    pytest.importorskip("ray")
    from dungeon_runner.rl.ray_local import RayRolloutError
    from dungeon_runner.replay.ppo.ray_workers import RayRolloutPool

    actor = MagicMock()
    actor.collect.remote.return_value = "fut"
    with patch("dungeon_runner.replay.ppo.ray_workers.ray") as mock_ray:
        mock_ray.get.side_effect = [None, RuntimeError("worker died")]
        pool = RayRolloutPool.__new__(RayRolloutPool)
        pool._actors = [actor]  # noqa: SLF001
        pool._n_workers = 1
        model = MagicMock()
        model.get_weights.return_value = [np.array([1.0])]
        with pytest.raises(RayRolloutError, match="collect failed"):
            pool.collect(model, target_steps=8, update_step=0)


def test_ray_pool_collect_dispatches_one_remote_per_worker():
    pytest.importorskip("ray")
    from dungeon_runner.replay.ppo.ray_workers import RayRolloutPool

    actor_a = MagicMock()
    actor_b = MagicMock()
    actor_a.collect.remote.return_value = ("fut-a",)
    actor_b.collect.remote.return_value = ("fut-b",)

    packed = rkm.pack_rollout_for_ray(RolloutBatch())
    stats = rkm.pack_game_stats_for_ray(RolloutGameStats())
    ray_return = [(packed, stats, "self_play"), (packed, stats, "vs_bc_bot")]

    with patch("dungeon_runner.replay.ppo.ray_workers.ray") as mock_ray:
        mock_ray.get.side_effect = [None, ray_return]
        pool = RayRolloutPool.__new__(RayRolloutPool)
        pool._actors = [actor_a, actor_b]  # noqa: SLF001
        pool._n_workers = 2
        pool._teacher_weights = "/tmp/teacher.h5"

        model = MagicMock()
        model.get_weights.return_value = [np.array([1.0])]
        batch, _stats, _template = pool.collect(model, target_steps=10, update_step=3)

    actor_a.set_learner_weights.remote.assert_called_once()
    actor_b.set_learner_weights.remote.assert_called_once()
    actor_a.collect.remote.assert_called_once_with(5, 3)
    actor_b.collect.remote.assert_called_once_with(5, 3)
    assert len(batch) == 0
