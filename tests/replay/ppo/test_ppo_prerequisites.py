"""PPO start prerequisites."""

from __future__ import annotations

from pathlib import Path

import pytest

from dungeon_runner.replay.ppo.prerequisites import (
    PPOPrerequisiteError,
    check_ppo_prerequisites,
)
from tests.replay.ppo.ppo_fixtures import write_ppo_fixture_tree


def test_prerequisites_pass_with_bc_run_and_lambda(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(data, repo)
    prereq = check_ppo_prerequisites(data, bc_run, bc_anchor_lambda=0.1)
    assert prereq.bc_run == bc_run.resolve()
    assert prereq.bc_weights.is_file()
    assert prereq.train_human_rows >= 1


def test_prerequisites_fail_without_bc_run(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_ppo_fixture_tree(data, repo)
    missing = repo / "models" / "runs" / "nope"
    with pytest.raises(PPOPrerequisiteError, match="bc-run"):
        check_ppo_prerequisites(data, missing, bc_anchor_lambda=0.1)


def test_prerequisites_skip_derived_when_lambda_zero(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(data, repo)
    prereq = check_ppo_prerequisites(data, bc_run, bc_anchor_lambda=0.0)
    assert prereq.train_human_rows == 0
