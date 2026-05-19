"""Eval suite init: sampling, artifacts, verify manifest prerequisite."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dungeon_runner.replay.eval.eval_suite import (
    EvalSuiteError,
    eval_suite_path,
    init_eval_suite,
    load_eval_suite,
    sample_val_match_ids,
)
from tests.replay.helpers import seed_verify_state


def test_sample_val_match_ids_n16_seed42():
    ids = [f"match-{i:03d}" for i in range(1, 17)]
    assert sample_val_match_ids(ids, 42) == ["match-001", "match-004", "match-012"]


def test_sample_val_match_ids_n2_seed42():
    assert sample_val_match_ids(["match-x", "match-y"], 42) == ["match-x"]


def test_sample_val_match_ids_n5_seed42():
    ids = ["match-a", "match-b", "match-c", "match-d", "match-e"]
    assert sample_val_match_ids(ids, 42) == ["match-a"]


def test_sample_val_match_ids_rejects_n1():
    with pytest.raises(EvalSuiteError, match="at least two"):
        sample_val_match_ids(["only-one"], 42)


def test_init_writes_artifact_from_verify_manifest(tmp_path: Path):
    verified = [f"match-{i:03d}" for i in range(1, 17)]
    seed_verify_state(tmp_path, verified=verified)
    artifact = init_eval_suite(tmp_path, sampling_seed=42)
    assert artifact.val_match_ids == ["match-001", "match-004", "match-012"]
    assert artifact.created_from_match_ids == sorted(verified)
    assert artifact.sampling_seed == 42
    assert artifact.suite_version == 1

    on_disk = json.loads(eval_suite_path(tmp_path).read_text(encoding="utf-8"))
    assert on_disk == artifact.to_dict()
    assert load_eval_suite(tmp_path) == artifact


def test_init_fails_with_one_verified_id_no_artifact(tmp_path: Path):
    seed_verify_state(tmp_path, verified=["match-solo"])
    with pytest.raises(EvalSuiteError):
        init_eval_suite(tmp_path)
    assert not eval_suite_path(tmp_path).is_file()


def test_init_fails_with_zero_verified(tmp_path: Path):
    seed_verify_state(tmp_path, verified=[])
    with pytest.raises(EvalSuiteError):
        init_eval_suite(tmp_path)


def test_cli_eval_suite_init(tmp_path: Path):
    verified = ["match-a", "match-b", "match-c"]
    seed_verify_state(tmp_path, verified=verified)
    repo = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "dungeon_runner.replay.cli",
            "eval_suite",
            "init",
            "--data-dir",
            str(tmp_path),
            "--sampling-seed",
            "42",
        ],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    assert proc.returncode == 0, proc.stderr
    artifact = load_eval_suite(tmp_path)
    assert artifact is not None
    assert len(artifact.val_match_ids) == 1
