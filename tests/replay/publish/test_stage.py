"""Publish stage integration: gates, promote layout, symlink, JSONL."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dungeon_runner.replay.publish.stage import PublishError, run_publish
from tests.replay.publish.publish_fixtures import (
    seed_legacy_latest,
    write_passing_eval_config,
    write_training_run_artifact,
)


def test_publish_success_creates_version_symlink_and_ledger(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_passing_eval_config(data_dir)
    seed_legacy_latest(tmp_path)
    run_dir = write_training_run_artifact(tmp_path)

    summary = run_publish(
        run_dir=run_dir,
        data_dir=data_dir,
        repo_root=tmp_path,
    )

    version_dir = tmp_path / "models" / summary.promoted_version
    assert version_dir.is_dir()
    assert (version_dir / "policy.weights.h5").read_bytes() == b"weights"
    assert (version_dir / "metrics.json").is_file()
    assert (version_dir / "promotion.json").is_file()

    latest = tmp_path / "models" / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == version_dir.resolve()

    ledger = tmp_path / "models" / "promotions.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().strip().splitlines()]
    assert rows[-1]["run_id"] == "bc-20260518T120000Z"
    assert rows[-1]["promoted_version"] == "v0.2"


def test_publish_fails_gates_without_touching_latest_or_ledger(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_passing_eval_config(data_dir, floor=0.9)
    seed_legacy_latest(tmp_path)
    run_dir = write_training_run_artifact(tmp_path, val_acc=0.5)

    with pytest.raises(PublishError) as exc:
        run_publish(run_dir=run_dir, data_dir=data_dir, repo_root=tmp_path)

    assert "replay_below_floor" in exc.value.reasons
    assert not (tmp_path / "models" / "promotions.jsonl").exists()
    latest = tmp_path / "models" / "latest"
    assert latest.is_symlink()
    assert latest.resolve() == (tmp_path / "models" / "v0.1.30a").resolve()


def test_publish_fails_sim_regression_without_promoting(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_passing_eval_config(data_dir)
    seed_legacy_latest(tmp_path)
    run_dir = write_training_run_artifact(
        tmp_path,
        cand_wr=0.5,
        latest_wr=0.55,
    )

    with pytest.raises(PublishError) as exc:
        run_publish(run_dir=run_dir, data_dir=data_dir, repo_root=tmp_path)

    assert "sim_regression" in exc.value.reasons
    assert not (tmp_path / "models" / "v0.2").exists()


def test_second_publish_same_run_id_rejected(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_passing_eval_config(data_dir)
    seed_legacy_latest(tmp_path)
    run_dir = write_training_run_artifact(tmp_path)
    run_publish(run_dir=run_dir, data_dir=data_dir, repo_root=tmp_path)

    with pytest.raises(PublishError) as exc:
        run_publish(run_dir=run_dir, data_dir=data_dir, repo_root=tmp_path)

    assert "already_promoted" in exc.value.reasons


def test_publish_cli_rejects_tmp_staging_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_passing_eval_config(data_dir)
    staging = tmp_path / "models" / "runs" / "bc-x.tmp"
    staging.mkdir(parents=True)
    (staging / "policy.weights.h5").write_bytes(b"w")
    (staging / "metrics.json").write_text('{"run_id":"bc-x"}\n')

    repo = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dungeon_runner.replay.cli",
            "publish",
            "--run",
            str(staging),
            "--data-dir",
            str(data_dir),
        ],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    assert proc.returncode == 1
    assert "tmp" in proc.stderr.lower() or ".tmp" in proc.stderr
