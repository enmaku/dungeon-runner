#!/usr/bin/env python3
"""PPO self-play: **Keras** policy, **Ray**-parallel rollouts, shared ``policy.weights.h5``.

Uses :mod:`ray` for distributed sampling; optimization matches ``train.py`` (not ``ray.rllib``'s
built-in PPO, which is not usable for a portable Keras ``PolicyValueModel`` on Ray 2.5+).

Requires: ``pip install -e ".[train]"``
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dungeon_runner.pettingzoo_aec import WtdAECEnv
from dungeon_runner.rl import rllib_keras_module as rkm
from dungeon_runner.rl.ppo import PPOConfig, RolloutBatch, RolloutGameStats, compute_gae, ppo_minibatch_update

try:
    import ray
except ImportError as err:  # pragma: no cover
    _RAY_ERR = err
    ray = None
else:  # pragma: no cover
    _RAY_ERR = None

_SelfPlayActor = None
if ray is not None:

    @ray.remote(num_cpus=1)  # type: ignore[union-attr, misc, attr-defined]
    class _SelfPlayActor:  # noqa: N801
        def __init__(self, src_dir: str, worker_id: int, base_seed: int) -> None:
            import os

            if src_dir and src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            self._worker_id = worker_id
            self._base = int(base_seed) + worker_id * 1_000_003
            tf.random.set_seed(self._base)
            self._env = WtdAECEnv()
            np0 = np.random.default_rng(self._base)
            n, self._roles, st, h0 = rkm.sample_episode_config(np0)
            self._env.reset(
                seed=int(np0.integers(0, 2**30)),
                options={"n_players": n, "start_seat": st, "first_hero": h0, "max_episode_steps": 20_000},
            )
            self._model = rkm.build_policy_value_model()

        def set_keras_weights(self, weights: list[object]) -> None:
            w2 = [np.asarray(x) for x in weights]
            rkm.set_model_weights_numpy(self._model, w2)

        def collect(self, target: int, ustep: int) -> tuple[dict, dict]:
            mix = (self._base * 0x7F4A7C15 + ustep * 1_000_003 + self._worker_id) & 0xFFFFFFFFFFFFFFFF
            np_r = np.random.default_rng(mix)
            pyr = random.Random(int(mix & 0x7FFFFFFF))
            b, self._roles, g = rkm.fill_rollout_selfplay(
                self._env, self._model, self._roles, pyr, np_r, target
            )
            return rkm.pack_rollout_for_ray(b), rkm.pack_game_stats_for_ray(g)


def _log_game_scalars_rl(step: int, g: RolloutGameStats, mean_r: float) -> None:
    ne = int(g.n_episodes)
    tf.summary.scalar("rollout/mean_reward", float(mean_r), step=step)
    tf.summary.scalar("rollout/env_steps", float(g.env_steps), step=step)
    tf.summary.scalar("game/episodes_ended", float(ne), step=step)
    if ne:
        tf.summary.scalar("game/mean_episode_length", float(np.mean(g.episode_lengths)), step=step)
        tf.summary.scalar("game/fraction_all_nn", float(g.n_all_nn) / ne, step=step)
        tf.summary.scalar("game/fraction_mixed_bot", 0.0, step=step)
        tf.summary.scalar("game/fraction_natural_end", float(g.n_decided) / ne, step=step)
        wr = float(g.nn_wins) / float(g.nn_games) if g.nn_games else 0.0
        tf.summary.scalar("game/nn_win_rate", wr, step=step)
        tf.summary.scalar("game/truncation_rate", float(g.n_truncated) / float(ne), step=step)


def _per_worker_rollout(rollout_total: int, n_workers: int, index: int) -> int:
    if n_workers <= 0 or rollout_total <= 0:
        return max(1, rollout_total)
    base, rem = divmod(rollout_total, n_workers)
    return base + (1 if index < rem else 0)


def _merge_worker_payloads(
    arrs: list[tuple[dict, dict]],
) -> tuple[RolloutBatch, RolloutGameStats]:
    parts_b = [rkm.unpack_ray_rollout(p[0]) for p in arrs if int(p[0].get("n", 0) or 0) > 0]
    b = rkm.merge_batches(parts_b) if parts_b else RolloutBatch()
    gparts = [rkm.unpack_game_stats(p[1]) for p in arrs]
    return b, rkm.merge_game_stats(gparts)


def main() -> None:
    if _RAY_ERR is not None or ray is None:
        print("Install: pip install -e \".[train]\"  (needs ray[rllib])", file=sys.stderr)
        raise SystemExit(1) from _RAY_ERR
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=Path, default=Path("runs/ppo_wtd_rllib"))
    ap.add_argument(
        "--rollout",
        type=int,
        default=256,
        help="Target NN transitions per PPO update (split across --num-workers).",
    )
    ap.add_argument(
        "--updates", type=int, default=4000, help="PPO update steps. Use 2–5 to smoke test."
    )
    ap.add_argument(
        "--log-every", type=int, default=100, help="Print loss to stdout every N updates."
    )
    ap.add_argument(
        "--save-every", type=int, default=500, help="H5 every N steps; 0 = only at end."
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Keras H5 to load on start; default: logdir/policy.weights.h5 if present.",
    )
    ap.add_argument(
        "--num-workers", type=int, default=1, help="Parallel Ray actor workers for collection."
    )
    args = ap.parse_args()
    n_workers = max(1, int(args.num_workers))
    if _SelfPlayActor is None:
        print("Ray actor is unavailable.", file=sys.stderr)
        raise SystemExit(1)
    wpath = args.weights if args.weights is not None else rkm.default_policy_h5_path(args.logdir)
    args.logdir.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(args.seed)
    np_r = np.random.default_rng(args.seed)
    pyr = random.Random(args.seed)
    # Quiets Ray 2.5+ FutureWarning about CUDA visible devices when num_gpus=0.
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(ignore_reinit_error=True, include_dashboard=False)  # type: ignore[union-attr, misc, attr-defined]
    cfg = PPOConfig()
    model = rkm.build_policy_value_model()
    if wpath.is_file():
        rkm.load_policy_weights_h5_if_present(model, wpath)
        print("loaded", wpath, file=sys.stderr)
    opt = keras.optimizers.Adam(cfg.lr)  # type: ignore[no-untyped-call, misc]
    src_s = str(_SRC)
    actors: list[object] = [  # Ray actor handles
        _SelfPlayActor.remote(src_s, i, args.seed)  # type: ignore[attr-defined]
        for i in range(n_workers)
    ]
    writer = tf.summary.create_file_writer(str(args.logdir / "scalars"))  # noqa: SIM201
    ck = rkm.default_policy_h5_path(args.logdir)
    t0 = time.time()
    for u in range(args.updates):
        w_np = rkm.model_weights_to_numpy(model)
        ray.get([a.set_keras_weights.remote(w_np) for a in actors])  # type: ignore[attr-defined]
        targets = [_per_worker_rollout(args.rollout, n_workers, i) for i in range(n_workers)]
        futs = [
            a.collect.remote(targets[i], u)  # type: ignore[attr-defined]
            for i, a in enumerate(actors)
        ]
        payload = ray.get(futs)  # type: ignore[attr-defined]
        b, game = _merge_worker_payloads(payload)  # type: ignore[assignment]
        n = min(len(b.obs), len(b.reward), len(b.act), len(b.value), len(b.logp), len(b.done))
        if n < 3:
            with writer.as_default():
                tf.summary.scalar("rollout/skipped", 1.0, step=u)
                r_sk = np.asarray(b.reward[:n], np.float32)
                _log_game_scalars_rl(u, game, float(np.mean(r_sk)) if n else 0.0)
            continue
        o = np.stack(b.obs[:n], 0)
        m = np.stack(b.mask[:n], 0)
        a_ = np.asarray(b.act[:n], np.int32)
        v = np.asarray(b.value[:n], np.float32)
        lp = np.asarray(b.logp[:n], np.float32)
        r_ = np.asarray(b.reward[:n], np.float32)
        d_ = np.asarray(b.done[:n], bool)
        mean_r = float(np.mean(r_)) if n else 0.0
        _, lvv = model(
            tf.convert_to_tensor(o[-1:, :], tf.float32),
            tf.convert_to_tensor(m[-1:, :], tf.float32),
        )
        lastv = float(lvv[0, 0].numpy())
        adv, rets = compute_gae(r_, v, d_, lastv, cfg.gamma, cfg.gae_lambda)
        idx = np.arange(n)
        pyr.shuffle(idx)
        acc: dict[str, float] = {"loss": 0.0, "pg": 0.0, "vl": 0.0, "en": 0.0}
        mb_n = 0
        for _e in range(cfg.n_epochs):
            pyr.shuffle(idx)
            for s0 in range(0, n, cfg.minibatch_size):
                sl = idx[s0 : s0 + cfg.minibatch_size]
                if sl.size == 0:
                    continue
                lg = ppo_minibatch_update(
                    model, opt, cfg, o[sl], m[sl], a_[sl], lp[sl], v[sl], adv[sl], rets[sl]
                )
                for k_ in acc:
                    acc[k_] += float(lg[k_])
                mb_n += 1
        if mb_n:
            for k_ in acc:
                acc[k_] /= mb_n
        with writer.as_default():
            tf.summary.scalar("loss/loss", acc["loss"], step=u)
            tf.summary.scalar("loss/policy", acc["pg"], step=u)
            tf.summary.scalar("loss/value", acc["vl"], step=u)
            tf.summary.scalar("loss/entropy", acc["en"], step=u)
            tf.summary.scalar("rollout/nn_transitions", float(n), step=u)
            _log_game_scalars_rl(u, game, mean_r)
        le = max(1, int(args.log_every))
        if u % le == 0 or u == args.updates - 1:
            ngm = int(game.nn_games)
            nn_wr = float(game.nn_wins) / float(ngm) if ngm else 0.0
            tf.print("u", u + 1, "/", args.updates, "n", n, "loss", acc["loss"], "nn_wr", nn_wr, output_stream=sys.stdout)
        se = int(args.save_every)
        if se > 0 and (u + 1) % se == 0:
            rkm.save_policy_weights_h5(model, ck)
            tf.print("checkpoint", str(ck), "step", u + 1, output_stream=sys.stdout)
    tf.print("elapsed", time.time() - t0, "s", output_stream=sys.stdout)
    rkm.save_policy_weights_h5(model, ck)
    tf.print("wrote", str(ck), output_stream=sys.stdout)
    ray.shutdown()  # type: ignore[union-attr, misc, attr-defined]


if __name__ == "__main__":
    try:
        main()
    except ImportError as err2:
        print("Install: pip install -e \".[train]\"", file=sys.stderr)
        raise SystemExit(1) from err2
