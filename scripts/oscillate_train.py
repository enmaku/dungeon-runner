#!/usr/bin/env python3
"""Alternate ``train_rllib.py`` and ``train.py`` forever using only numbered subfolders.

Each run uses ``--logdir <logdir>/<n>/`` (weights and TensorBoard live there, not under
``<logdir>/``). For ``n>1``, ``<logdir>/(n-1)/policy.weights.h5`` is copied to
``<logdir>/n/policy.weights.h5`` before training; subprocesses load that file by default
(no ``--weights`` path outside ``<logdir>/n/``).

On a new run, ``n`` starts at one past the largest segment that has
``<logdir>/n/policy.weights.h5`` (``1`` if none). Numeric empty folders do not
advance the counter. If ``n==1`` and ``1/policy.weights.h5`` is missing, it is created: copy from
``<logdir>/policy.weights.h5`` if that exists, else a fresh random inited model. The base
H5 is never read or written by the training subprocesses.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_NUM = re.compile(r"^\d+$")


def _max_completed_segment(root: Path) -> int:
    m = 0
    for p in root.iterdir() if root.is_dir() else []:
        if p.is_dir() and _NUM.match(p.name) and (p / "policy.weights.h5").is_file():
            m = max(m, int(p.name))
    return m


def _rllib_first_for_segment(seg: int, start: str) -> bool:
    if start == "rllib":
        return seg % 2 == 1
    return seg % 2 == 0


def _write_fresh_weights(h5: Path) -> None:
    import tensorflow as tf  # noqa: PLC0415
    from dungeon_runner.rl import actions_codec, observation  # noqa: PLC0415
    from dungeon_runner.rl.model import DEFAULT_PPO_HIDDEN, PolicyValueModel  # noqa: PLC0415

    h5.parent.mkdir(parents=True, exist_ok=True)
    model = PolicyValueModel(hidden=DEFAULT_PPO_HIDDEN)
    _ = model(  # build
        tf.zeros((1, observation.OBS_DIM), tf.float32),
        tf.zeros((1, actions_codec.N_ACTIONS), tf.float32),
    )
    model.save_weights(str(h5))


def _copy_prev_weights_into_segment(sub: Path, src: Path) -> None:
    sub.mkdir(parents=True, exist_ok=True)
    dst = sub / "policy.weights.h5"
    shutil.copy2(src, dst)
    print("seeded", dst, "from", src, file=sys.stderr)


def _ensure_segment_one(root: Path) -> None:
    w1 = root / "1" / "policy.weights.h5"
    if w1.is_file():
        return
    (root / "1").mkdir(parents=True, exist_ok=True)
    base = root / "policy.weights.h5"
    if base.is_file():
        shutil.copy2(base, w1)
        print("seeded", w1, "from", base, file=sys.stderr)
    else:
        _write_fresh_weights(w1)
        print("wrote fresh", w1, file=sys.stderr)


def _run_rllib(sub: Path, n_workers: int) -> None:
    ex: list[str | Path] = [
        sys.executable,
        str(_ROOT / "scripts" / "train_rllib.py"),
        "--logdir",
        str(sub),
        "--num-workers",
        str(n_workers),
    ]
    p = subprocess.run(ex, cwd=_ROOT)  # noqa: S603
    if p.returncode != 0:
        print("command failed:", p.returncode, ex, file=sys.stderr)
        raise SystemExit(p.returncode)


def _run_train(sub: Path) -> None:
    ex: list[str | Path] = [
        sys.executable,
        str(_ROOT / "scripts" / "train.py"),
        "--logdir",
        str(sub),
    ]
    p = subprocess.run(ex, cwd=_ROOT)  # noqa: S603
    if p.returncode != 0:
        print("command failed:", p.returncode, ex, file=sys.stderr)
        raise SystemExit(p.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logdir", type=Path, help="Run directory (e.g. runs/v0.1a)")
    ap.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="for train_rllib.py; parallel rollout workers (default: 8)",
    )
    ap.add_argument(
        "--start",
        choices=("rllib", "train"),
        default="rllib",
        help="Which script to run first (default: rllib).",
    )
    args = ap.parse_args()
    logdir = args.logdir.expanduser().resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    seg = _max_completed_segment(logdir) + 1
    if seg == 1:
        _ensure_segment_one(logdir)
    n_workers = max(1, int(args.num_workers))
    rllib_next = _rllib_first_for_segment(seg, args.start)
    while True:
        sub = logdir / str(seg)
        if seg > 1:
            src_w = logdir / str(seg - 1) / "policy.weights.h5"
            if not src_w.is_file():
                msg = f"missing {src_w} (cannot start segment {seg})"
                raise SystemExit(msg)
            _copy_prev_weights_into_segment(sub, src_w)
        if rllib_next:
            _run_rllib(sub, n_workers)
        else:
            _run_train(sub)
        w_out = sub / "policy.weights.h5"
        if not w_out.is_file():
            print("expected", w_out, "after training", file=sys.stderr)
            raise SystemExit(1)
        print("finished segment", seg, "→", w_out, file=sys.stderr)
        seg += 1
        rllib_next = not rllib_next


if __name__ == "__main__":
    main()
