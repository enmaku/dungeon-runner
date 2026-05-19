"""Split resolver: val holdout vs train (incl. post-freeze ids)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dungeon_runner.replay.eval.eval_suite import (
    EvalSuiteArtifact,
    EvalSuiteError,
    init_eval_suite,
    require_eval_suite,
)
from dungeon_runner.replay.eval.split_resolver import split_for, split_for_match_id
from tests.replay.helpers import seed_verify_state


@pytest.fixture
def frozen_suite(tmp_path: Path) -> EvalSuiteArtifact:
    seed_verify_state(
        tmp_path,
        verified=["match-a", "match-b", "match-c", "match-d"],
    )
    return init_eval_suite(tmp_path, sampling_seed=42)


def test_val_id_resolves_to_val(frozen_suite: EvalSuiteArtifact):
    for match_id in frozen_suite.val_match_ids:
        assert split_for(match_id, frozen_suite) == "val"


def test_non_val_verified_id_is_train(frozen_suite: EvalSuiteArtifact):
    val_set = set(frozen_suite.val_match_ids)
    for match_id in frozen_suite.created_from_match_ids:
        if match_id in val_set:
            continue
        assert split_for(match_id, frozen_suite) == "train"


def test_post_freeze_id_is_train(frozen_suite: EvalSuiteArtifact):
    assert split_for("match-new-after-freeze", frozen_suite) == "train"


def test_split_for_match_id_requires_eval_suite(tmp_path: Path):
    with pytest.raises(EvalSuiteError, match="eval suite artifact missing"):
        split_for_match_id(tmp_path, "match-a")


def test_split_for_match_id_loads_artifact(tmp_path: Path, frozen_suite: EvalSuiteArtifact):
    val_id = frozen_suite.val_match_ids[0]
    assert split_for_match_id(tmp_path, val_id) == "val"
    assert split_for_match_id(tmp_path, "match-new-after-freeze") == "train"


def test_require_eval_suite(tmp_path: Path, frozen_suite: EvalSuiteArtifact):
    assert require_eval_suite(tmp_path) == frozen_suite
