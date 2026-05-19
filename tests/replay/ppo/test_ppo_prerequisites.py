"""PPO start prerequisites."""

from __future__ import annotations

from pathlib import Path

import pytest

from dungeon_runner.replay.ppo.prerequisites import (
    PPOPrerequisiteError,
    check_ppo_prerequisites,
)
from tests.replay.bc.bc_fixtures import write_bc_derived_fixture, write_bc_eval_artifacts
from tests.replay.ppo.ppo_fixtures import write_bc_run_artifact, write_ppo_fixture_tree


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


def test_prerequisites_fail_without_bc_weights(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_ppo_fixture_tree(data, repo)
    empty = repo / "models" / "runs" / "bc-empty"
    empty.mkdir(parents=True)
    (empty / "metrics.json").write_text('{"run_id":"bc-empty"}', encoding="utf-8")
    with pytest.raises(PPOPrerequisiteError, match="policy weights"):
        check_ppo_prerequisites(data, empty, bc_anchor_lambda=0.0)


def test_prerequisites_fail_without_eval_suite(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_derived_fixture(data)
    bc_run = write_bc_run_artifact(repo)
    with pytest.raises(Exception, match="eval suite"):
        check_ppo_prerequisites(data, bc_run, bc_anchor_lambda=0.0)


def test_prerequisites_fail_without_eval_config(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    write_bc_derived_fixture(data)
    write_bc_eval_artifacts(data)
    (data / "eval_config.json").unlink()
    bc_run = write_bc_run_artifact(repo)
    with pytest.raises(Exception, match="eval config"):
        check_ppo_prerequisites(data, bc_run, bc_anchor_lambda=0.0)


def test_prerequisites_fail_without_bc_metrics(tmp_path: Path):
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    bc_run = write_ppo_fixture_tree(data, repo)
    (bc_run / "metrics.json").unlink()
    with pytest.raises(PPOPrerequisiteError, match="metrics"):
        check_ppo_prerequisites(data, bc_run, bc_anchor_lambda=0.0)
