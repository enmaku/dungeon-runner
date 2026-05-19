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


def test_prerequisites_fail_without_eval_suite(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    (repo / "models" / "latest").mkdir(parents=True)
    (repo / "models" / "latest" / "policy.weights.h5").write_bytes(b"x")
    with pytest.raises(Exception, match="eval suite"):
        check_bc_prerequisites(data, repo)
