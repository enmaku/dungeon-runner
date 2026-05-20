"""BC start prerequisites."""

from __future__ import annotations

from pathlib import Path

import pytest

from dungeon_runner.replay.bc.prerequisites import BCPrerequisiteError, check_bc_prerequisites
from tests.replay.bc.bc_fixtures import write_bc_eval_artifacts, write_bc_fixture_tree


def test_prerequisites_pass_with_fixture_tree(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree(data, repo)
    prereq = check_bc_prerequisites(data, repo)
    assert prereq.train_human_rows >= 1
    assert prereq.val_human_rows >= 1
    assert prereq.parent_weights.is_file()


def test_prerequisites_fail_without_parent_weights(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_eval_artifacts(data)
    with pytest.raises(BCPrerequisiteError, match="training parent"):
        check_bc_prerequisites(data, repo)


def test_prerequisites_fail_without_train_human_rows(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_eval_artifacts(data)
    (repo / "models" / "latest").mkdir(parents=True)
    (repo / "models" / "latest" / "policy.weights.h5").write_bytes(b"x")
    val_dir = data / "derived" / "match-val"
    val_dir.mkdir(parents=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table(
            {
                "match_id": ["match-val"] * 2,
                "split": ["val", "val"],
                "is_human": [True, True],
                "policy_action_index": [0, 1],
                "obs": [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                "mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
            }
        ),
        val_dir / "rows.parquet",
    )
    with pytest.raises(BCPrerequisiteError, match="train-split human"):
        check_bc_prerequisites(data, repo)


def test_prerequisites_fail_when_val_match_not_in_suite(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_fixture_tree(data, repo)
    rogue = data / "derived" / "match-rogue-val"
    rogue.mkdir(parents=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(
        pa.table(
            {
                "match_id": ["match-rogue-val"] * 2,
                "split": ["val", "val"],
                "is_human": [True, True],
                "policy_action_index": [0, 1],
                "obs": [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                "mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
            }
        ),
        rogue / "rows.parquet",
    )
    with pytest.raises(BCPrerequisiteError, match="not in eval suite holdout"):
        check_bc_prerequisites(data, repo)


def test_prerequisites_fail_without_eval_suite(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    (repo / "models" / "latest").mkdir(parents=True)
    (repo / "models" / "latest" / "policy.weights.h5").write_bytes(b"x")
    with pytest.raises(Exception, match="eval suite"):
        check_bc_prerequisites(data, repo)
