"""Ray actor pool for parallel replay PPO rollout collection."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

from dungeon_runner.rl import rllib_keras_module as rkm
from dungeon_runner.rl.model import PolicyValueModel
from dungeon_runner.rl.ppo import RolloutBatch, RolloutGameStats
from dungeon_runner.rl.ray_local import RayRolloutError, init_ray_local_cluster

try:
    import ray
except ImportError:  # pragma: no cover
    ray = None  # type: ignore[assignment, misc]

_SRC = Path(__file__).resolve().parents[3]

_ReplayRolloutActor = None
if ray is not None:

    @ray.remote(num_cpus=1)  # type: ignore[union-attr, misc, attr-defined]
    class _ReplayRolloutActor:  # noqa: N801
        def __init__(
            self,
            src_dir: str,
            teacher_weights: str,
            worker_id: int,
            base_seed: int,
        ) -> None:
            if src_dir and src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import random

            import tensorflow as tf

            from dungeon_runner.bots import RandomBot
            from dungeon_runner.pettingzoo_aec import WtdAECEnv
            from dungeon_runner.replay.bc.predict import load_policy_model
            from dungeon_runner.replay.ppo.frozen_teacher import FrozenBCTeacher
            from dungeon_runner.replay.ppo.rollout_collector import (
                fill_rollout,
                sample_episode_config,
            )

            self._worker_id = worker_id
            self._base = int(base_seed) + worker_id * 1_000_003
            tf.random.set_seed(self._base)
            weights = Path(teacher_weights)
            self._model = load_policy_model(weights)
            self._teacher = FrozenBCTeacher.from_weights(
                weights, load_model=load_policy_model
            )
            self._env = WtdAECEnv()
            self._random_bot = RandomBot()
            np_r = np.random.default_rng(self._base)
            self._pyr = random.Random(self._base)
            n, roles, st, h0, template = sample_episode_config(np_r, self._pyr)
            self._roles = roles
            self._template = template
            self._env.reset(
                seed=int(np_r.integers(0, 2**30)),
                options={"n_players": n, "start_seat": st, "first_hero": h0},
            )

        def set_learner_weights(self, weights: list[object]) -> None:
            rkm.set_model_weights_numpy(self._model, [np.asarray(x) for x in weights])

        def collect(self, target: int, update_step: int) -> tuple[dict, dict, str]:
            import random

            mix = (
                self._base * 0x7F4A7C15 + update_step * 1_000_003 + self._worker_id
            ) & 0xFFFFFFFFFFFFFFFF
            pyr = random.Random(int(mix & 0x7FFFFFFF))
            np_r = np.random.default_rng(mix)
            batch, self._roles, stats, template = fill_rollout(
                self._env,
                self._model,
                teacher=self._teacher,
                random_bot=self._random_bot,
                roles=self._roles,
                template=self._template,
                pyr=pyr,
                np_r=np_r,
                target=target,
            )
            self._template = template
            return (
                rkm.pack_rollout_for_ray(batch),
                rkm.pack_game_stats_for_ray(stats),
                template,
            )


def per_worker_rollout_target(rollout_total: int, n_workers: int, index: int) -> int:
    if n_workers <= 0 or rollout_total <= 0:
        return max(1, rollout_total)
    base, rem = divmod(rollout_total, n_workers)
    return base + (1 if index < rem else 0)


def merge_rollout_payloads(
    payloads: list[tuple[dict, dict, str]],
) -> tuple[RolloutBatch, RolloutGameStats, str]:
    if not payloads:
        return RolloutBatch(), RolloutGameStats(), ""
    parts_b: list[RolloutBatch] = []
    stats_parts: list[RolloutGameStats] = []
    template_counts: dict[str, int] = {}
    for packed, stats_packed, template in payloads:
        n = int(packed.get("n", 0) or 0)
        if n > 0:
            parts_b.append(rkm.unpack_ray_rollout(packed))
            template_counts[template] = template_counts.get(template, 0) + n
        stats_parts.append(rkm.unpack_game_stats(stats_packed))
    batch = rkm.merge_batches(parts_b) if parts_b else RolloutBatch()
    stats = rkm.merge_game_stats(stats_parts) if stats_parts else RolloutGameStats()
    dominant = max(template_counts, key=template_counts.get) if template_counts else payloads[0][2]
    return batch, stats, dominant


def _kill_actor(actor: object) -> None:
    if ray is None:
        return
    try:
        ray.kill(actor)  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


class RayRolloutPool:
    """Persistent Ray actors for one PPO training run."""

    def __init__(
        self,
        *,
        teacher_weights: Path,
        n_workers: int,
        seed: int,
    ) -> None:
        if ray is None or _ReplayRolloutActor is None:
            raise RayRolloutError(
                "Ray is not installed; use --no-ray or pip install -e '.[train]'"
            )
        self._n_workers = max(1, int(n_workers))
        self._teacher_weights = str(teacher_weights.resolve())
        self._actors: list[object] = []
        init_ray_local_cluster()
        src_s = str(_SRC)
        try:
            self._actors = [
                _ReplayRolloutActor.remote(src_s, self._teacher_weights, i, seed)  # type: ignore[attr-defined]
                for i in range(self._n_workers)
            ]
        except Exception as exc:
            self.shutdown()
            raise RayRolloutError(
                f"Failed to start Ray rollout workers ({exc!r}); use --no-ray to fall back"
            ) from exc

    def collect(
        self,
        model: PolicyValueModel,
        *,
        target_steps: int,
        update_step: int,
    ) -> tuple[RolloutBatch, RolloutGameStats, str]:
        if not self._actors:
            raise RayRolloutError("Ray rollout pool has no workers; use --no-ray")
        w_np = rkm.model_weights_to_numpy(model)
        try:
            ray.get([a.set_learner_weights.remote(w_np) for a in self._actors])  # type: ignore[union-attr]
            targets = [
                per_worker_rollout_target(target_steps, self._n_workers, i)
                for i in range(self._n_workers)
            ]
            futs = [
                a.collect.remote(targets[i], update_step)  # type: ignore[attr-defined]
                for i, a in enumerate(self._actors)
            ]
            payload = ray.get(futs)  # type: ignore[union-attr]
        except RayRolloutError:
            raise
        except Exception as exc:
            raise RayRolloutError(
                f"Ray rollout collect failed ({exc!r}); use --no-ray for single-process rollouts"
            ) from exc
        return merge_rollout_payloads(payload)  # type: ignore[arg-type]

    def shutdown(self) -> None:
        for actor in self._actors:
            _kill_actor(actor)
        self._actors = []
