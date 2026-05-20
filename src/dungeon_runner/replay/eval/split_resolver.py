"""Match id → dataset split tag (train | val)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dungeon_runner.replay.eval.eval_suite import (
    EvalSuiteArtifact,
    EvalSuiteError,
    require_eval_suite,
)

SplitTag = Literal["train", "val"]


def split_for(match_id: str, eval_suite: EvalSuiteArtifact) -> SplitTag:
    if match_id in eval_suite.val_match_ids:
        return "val"
    return "train"


def split_for_match_id(data_dir: Path, match_id: str) -> SplitTag:
    eval_suite = require_eval_suite(data_dir)
    return split_for(match_id, eval_suite)


def require_split_for(match_id: str, eval_suite: EvalSuiteArtifact | None) -> SplitTag:
    if eval_suite is None:
        raise EvalSuiteError("eval suite artifact is required for split resolution")
    return split_for(match_id, eval_suite)
