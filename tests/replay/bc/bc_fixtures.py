"""Minimal derived store + models layout for BC stage smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dungeon_runner.replay.eval.atomic_json import atomic_write_json
from dungeon_runner.replay.eval.eval_config import EvalConfigArtifact
from dungeon_runner.replay.eval.eval_suite import EvalSuiteArtifact
from dungeon_runner.rl import actions_codec, observation

REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_PARENT_WEIGHTS = REPO_ROOT / "models" / "latest" / "policy.weights.h5"
PRODUCTION_OBS_DIM = observation.OBS_DIM
PRODUCTION_N_ACTIONS = actions_codec.N_ACTIONS

FIXTURE_MATCH_VAL = "match-val"
FIXTURE_MATCH_TRAIN = "match-train"
SMOKE_OBS_DIM = 4
SMOKE_N_ACTIONS = 4


def _obs(label: float) -> list[float]:
    v = [0.0] * SMOKE_OBS_DIM
    v[0] = float(label)
    return v


def _mask() -> list[int]:
    return [1] * SMOKE_N_ACTIONS


def _production_obs(label: float) -> list[float]:
    v = [0.0] * PRODUCTION_OBS_DIM
    v[0] = float(label)
    return v


def _production_mask() -> list[int]:
    return [1] * PRODUCTION_N_ACTIONS


def write_bc_derived_fixture_production(data_dir: Path) -> None:
    val_dir = data_dir / "derived" / FIXTURE_MATCH_VAL
    val_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "match_id": [FIXTURE_MATCH_VAL] * 4,
                "split": ["val"] * 4,
                "is_human": [True, True, True, False],
                "policy_action_index": [0, 1, 2, 0],
                "obs": [_production_obs(0.0), _production_obs(1.0), _production_obs(2.0), _production_obs(0.0)],
                "mask": [_production_mask()] * 4,
            }
        ),
        val_dir / "rows.parquet",
    )

    train_dir = data_dir / "derived" / FIXTURE_MATCH_TRAIN
    train_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "match_id": [FIXTURE_MATCH_TRAIN] * 6,
                "split": ["train"] * 6,
                "is_human": [True] * 6,
                "policy_action_index": [0, 1, 0, 1, 0, 1],
                "obs": [_production_obs(float(i)) for i in range(6)],
                "mask": [_production_mask()] * 6,
            }
        ),
        train_dir / "rows.parquet",
    )


def write_bc_derived_fixture(data_dir: Path) -> None:
    val_dir = data_dir / "derived" / FIXTURE_MATCH_VAL
    val_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "match_id": [FIXTURE_MATCH_VAL] * 4,
                "split": ["val"] * 4,
                "is_human": [True, True, True, False],
                "policy_action_index": [0, 1, 2, 0],
                "obs": [_obs(0.0), _obs(1.0), _obs(2.0), _obs(0.0)],
                "mask": [_mask(), _mask(), _mask(), _mask()],
            }
        ),
        val_dir / "rows.parquet",
    )

    train_dir = data_dir / "derived" / FIXTURE_MATCH_TRAIN
    train_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "match_id": [FIXTURE_MATCH_TRAIN] * 6,
                "split": ["train"] * 6,
                "is_human": [True] * 6,
                "policy_action_index": [0, 1, 0, 1, 0, 1],
                "obs": [_obs(float(i)) for i in range(6)],
                "mask": [_mask()] * 6,
            }
        ),
        train_dir / "rows.parquet",
    )


def write_bc_eval_artifacts(data_dir: Path) -> None:
    suite = EvalSuiteArtifact(
        suite_version=1,
        sampling_seed=42,
        created_from_match_ids=[FIXTURE_MATCH_VAL, FIXTURE_MATCH_TRAIN],
        val_match_ids=[FIXTURE_MATCH_VAL],
    )
    atomic_write_json(data_dir / "eval_suite.json", suite.to_dict())
    config = EvalConfigArtifact(
        sim_seeds=[0, 1],
        sim_regression_tolerance=0.01,
        replay_accuracy_floor=None,
    )
    atomic_write_json(data_dir / "eval_config.json", config.to_dict())


def write_smoke_parent_weights(
    path: Path,
    *,
    hidden: tuple[int, ...] = (8,),
) -> None:
    import tensorflow as tf

    from dungeon_runner.rl.model import PolicyValueModel

    path.parent.mkdir(parents=True, exist_ok=True)
    model = PolicyValueModel(
        obs_dim=SMOKE_OBS_DIM,
        n_actions=SMOKE_N_ACTIONS,
        hidden=hidden,
        use_layer_norm=False,
    )
    _ = model(
        tf.zeros((1, SMOKE_OBS_DIM), tf.float32),
        tf.zeros((1, SMOKE_N_ACTIONS), tf.float32),
    )
    model.save_weights(str(path))


def write_bc_fixture_tree(data_dir: Path, repo_root: Path) -> Path:
    write_bc_derived_fixture(data_dir)
    write_bc_eval_artifacts(data_dir)
    parent = repo_root / "models" / "latest" / "policy.weights.h5"
    write_smoke_parent_weights(parent)
    return parent


def write_bc_fixture_tree_production(data_dir: Path, repo_root: Path) -> Path:
    write_bc_derived_fixture_production(data_dir)
    write_bc_eval_artifacts(data_dir)
    parent = repo_root / "models" / "latest" / "policy.weights.h5"
    parent.parent.mkdir(parents=True, exist_ok=True)
    if PRODUCTION_PARENT_WEIGHTS.is_file():
        parent.write_bytes(PRODUCTION_PARENT_WEIGHTS.read_bytes())
    else:
        raise FileNotFoundError(f"missing production parent weights: {PRODUCTION_PARENT_WEIGHTS}")
    return parent


def rows_to_arrays(rows: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = np.stack([np.asarray(r.obs, dtype=np.float32) for r in rows], axis=0)
    masks = np.stack([np.asarray(r.mask, dtype=np.float32) for r in rows], axis=0)
    labels = np.array([int(r.policy_action_index) for r in rows], dtype=np.int32)
    return obs, masks, labels


def smoke_load_model(path: Path):
    import tensorflow as tf

    from dungeon_runner.rl.model import PolicyValueModel

    model = PolicyValueModel(
        obs_dim=SMOKE_OBS_DIM,
        n_actions=SMOKE_N_ACTIONS,
        hidden=(8,),
        use_layer_norm=False,
    )
    _ = model(
        tf.zeros((1, SMOKE_OBS_DIM), tf.float32),
        tf.zeros((1, SMOKE_N_ACTIONS), tf.float32),
    )
    model.load_weights(str(path))
    return model


def stub_sim_metrics(_candidate, _latest, seeds: list[int]):
    from dungeon_runner.replay.eval.sim_metrics import SimMetrics

    n = len(seeds)
    wr = 0.5 if n else 0.0
    return SimMetrics(
        candidate_win_rate_vs_randombot=wr,
        latest_win_rate_vs_randombot=wr,
        seed_count=n,
    )
