"""Minimal derived-store Parquet rows for replay metrics integration tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

FIXTURE_MATCH_VAL = "match-val"
FIXTURE_MATCH_TRAIN = "match-train"


def write_replay_metrics_fixture(data_dir: Path) -> None:
    """Two val human rows + one val NN row + one train human row."""
    val_dir = data_dir / "derived" / FIXTURE_MATCH_VAL
    val_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "match_id": [FIXTURE_MATCH_VAL] * 3,
                "split": ["val", "val", "val"],
                "is_human": [True, True, False],
                "policy_action_index": [1, 2, 0],
                "obs": [[1.0], [2.0], [0.0]],
                "mask": [[1], [1], [1]],
            }
        ),
        val_dir / "rows.parquet",
    )

    train_dir = data_dir / "derived" / FIXTURE_MATCH_TRAIN
    train_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "match_id": [FIXTURE_MATCH_TRAIN],
                "split": ["train"],
                "is_human": [True],
                "policy_action_index": [3],
                "obs": [[3.0]],
                "mask": [[1]],
            }
        ),
        train_dir / "rows.parquet",
    )


def obs_label_predict(obs: list[float], _mask: list[int]) -> int:
    return int(obs[0])


def obs_half_predict(obs: list[float], _mask: list[int]) -> int:
    return int(obs[0]) // 2
