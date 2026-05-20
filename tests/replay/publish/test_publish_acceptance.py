"""Publish acceptance: atomicity, semver bump, PPO, symlink targets, CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from dungeon_runner.replay.eval.atomic_json import atomic_write_json
from dungeon_runner.replay.eval.eval_config import EvalConfigArtifact
from dungeon_runner.replay.publish.manifest import list_promoted_versions
from dungeon_runner.replay.publish.stage import PublishError, run_publish, validate_run_dir
from tests.replay.publish.publish_fixtures import (
    seed_legacy_latest,
    write_passing_eval_config,
    write_training_run_artifact,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_latest_symlink_uses_sibling_target_not_parent_relative(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_passing_eval_config(data_dir)
    seed_legacy_latest(tmp_path)
    run_dir = write_training_run_artifact(tmp_path)

    summary = run_publish(
        run_dir=run_dir,
        data_dir=data_dir,
        repo_root=tmp_path,
        promoted_at="2026-05-18T12:00:00+00:00",
    )

    latest = tmp_path / "models" / "latest"
    assert latest.is_symlink()
    assert os.readlink(latest) == summary.promoted_version
    assert not os.readlink(latest).startswith("../")


def test_list_promoted_versions_ignores_staging_tmp_dirs(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "v0.2").mkdir()
    (models / "v0.3.tmp").mkdir()
    ledger = models / "promotions.jsonl"

    versions = list_promoted_versions(models, ledger)

    assert versions == ("v0.2",)


def test_publish_failure_before_symlink_leaves_latest_and_ledger_unchanged(
    tmp_path,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_passing_eval_config(data_dir)
    seed_legacy_latest(tmp_path)
    run_dir = write_training_run_artifact(tmp_path)
    legacy_latest = (tmp_path / "models" / "latest").resolve()

    with patch.object(Path, "symlink_to", side_effect=OSError("symlink blocked")):
        with pytest.raises(OSError, match="symlink blocked"):
            run_publish(run_dir=run_dir, data_dir=data_dir, repo_root=tmp_path)

    assert not (tmp_path / "models" / "promotions.jsonl").exists()
    assert not (tmp_path / "models" / "v0.2").exists()
    latest = tmp_path / "models" / "latest"
    assert latest.resolve() == legacy_latest


def test_second_distinct_run_allocates_v0_2_01(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_passing_eval_config(data_dir)
    seed_legacy_latest(tmp_path)

    first = write_training_run_artifact(tmp_path, run_id="bc-20260518T120000Z")
    run_publish(run_dir=first, data_dir=data_dir, repo_root=tmp_path)

    second = write_training_run_artifact(
        tmp_path,
        run_id="bc-20260518T130000Z",
    )
    summary = run_publish(run_dir=second, data_dir=data_dir, repo_root=tmp_path)

    assert summary.promoted_version == "v0.2.01"
    latest = tmp_path / "models" / "latest"
    assert latest.resolve() == (tmp_path / "models" / "v0.2.01").resolve()
    rows = [
        json.loads(line)
        for line in (tmp_path / "models" / "promotions.jsonl")
        .read_text()
        .strip()
        .splitlines()
    ]
    assert len(rows) == 2


def test_ppo_run_promotes_when_bc_regression_passes(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_passing_eval_config(data_dir)
    seed_legacy_latest(tmp_path)
    run_dir = write_training_run_artifact(
        tmp_path,
        run_id="ppo-20260518T120000Z",
        ppo_pass=True,
    )

    summary = run_publish(run_dir=run_dir, data_dir=data_dir, repo_root=tmp_path)

    assert summary.promoted_version == "v0.2"
    manifest = json.loads(
        (summary.version_dir / "promotion.json").read_text(),
    )
    assert manifest["run_id"] == "ppo-20260518T120000Z"


def test_publish_fails_prefloor_when_eval_config_floor_null(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = EvalConfigArtifact(
        sim_seeds=[0],
        sim_regression_tolerance=0.01,
        replay_accuracy_floor=None,
    )
    atomic_write_json(data_dir / "eval_config.json", config.to_dict())
    seed_legacy_latest(tmp_path)
    run_dir = write_training_run_artifact(tmp_path)

    with pytest.raises(PublishError) as exc:
        run_publish(run_dir=run_dir, data_dir=data_dir, repo_root=tmp_path)

    assert "replay_accuracy_floor_not_set" in exc.value.reasons


def test_validate_run_dir_rejects_missing_weights(tmp_path):
    run_dir = tmp_path / "models" / "runs" / "bc-incomplete"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text('{"run_id":"bc-incomplete"}\n')

    with pytest.raises(PublishError, match="missing policy.weights.h5"):
        validate_run_dir(run_dir)


def test_publish_cli_stderr_names_failing_gate_legs(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_passing_eval_config(data_dir, floor=0.9)
    seed_legacy_latest(repo)
    run_dir = write_training_run_artifact(repo, val_acc=0.5)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dungeon_runner.replay.cli",
            "publish",
            "--run",
            str(run_dir),
            "--data-dir",
            str(data_dir),
        ],
        capture_output=True,
        text=True,
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(_repo_root() / "src")},
    )
    assert proc.returncode == 1
    assert "replay_below_floor" in proc.stderr


def test_publish_cli_success_prints_promoted_version(tmp_path):
    repo = tmp_path / "repo"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_passing_eval_config(data_dir)
    seed_legacy_latest(repo)
    run_dir = write_training_run_artifact(repo)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dungeon_runner.replay.cli",
            "publish",
            "--run",
            str(run_dir),
            "--data-dir",
            str(data_dir),
        ],
        capture_output=True,
        text=True,
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(_repo_root() / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    assert "v0.2" in proc.stdout
    assert (repo / "models" / "v0.2").is_dir()
