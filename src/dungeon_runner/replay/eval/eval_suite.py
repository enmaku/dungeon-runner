"""Frozen eval suite artifact: holdout match ids sampled from verify manifest."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dungeon_runner.replay.eval.atomic_json import atomic_write_json
from dungeon_runner.replay.verify_manifest import load_verify_manifest

DEFAULT_SAMPLING_SEED = 42
SUITE_VERSION = 1


class EvalSuiteError(ValueError):
    """Eval suite init or load failed."""


@dataclass(frozen=True)
class EvalSuiteArtifact:
    suite_version: int
    sampling_seed: int
    created_from_match_ids: list[str]
    val_match_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_version": self.suite_version,
            "sampling_seed": self.sampling_seed,
            "created_from_match_ids": list(self.created_from_match_ids),
            "val_match_ids": list(self.val_match_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalSuiteArtifact:
        return cls(
            suite_version=int(data["suite_version"]),
            sampling_seed=int(data["sampling_seed"]),
            created_from_match_ids=[str(x) for x in data["created_from_match_ids"]],
            val_match_ids=[str(x) for x in data["val_match_ids"]],
        )


def eval_suite_path(data_dir: Path) -> Path:
    return data_dir / "eval_suite.json"


def sample_val_match_ids(verified_ids: list[str], sampling_seed: int) -> list[str]:
    """Seeded ~20% holdout: k = max(1, round(0.2 * n)) when n >= 2."""
    sorted_ids = sorted(verified_ids)
    n = len(sorted_ids)
    if n < 2:
        raise EvalSuiteError(
            f"eval suite init requires at least two verified match ids, got {n}"
        )
    k = max(1, round(0.2 * n))
    rng = random.Random(sampling_seed)
    return sorted(rng.sample(sorted_ids, k))


def init_eval_suite(
    data_dir: Path,
    *,
    sampling_seed: int = DEFAULT_SAMPLING_SEED,
    suite_version: int = SUITE_VERSION,
) -> EvalSuiteArtifact:
    verify = load_verify_manifest(data_dir)
    verified = sorted(verify.verified)
    val_match_ids = sample_val_match_ids(verified, sampling_seed)
    artifact = EvalSuiteArtifact(
        suite_version=suite_version,
        sampling_seed=sampling_seed,
        created_from_match_ids=list(verified),
        val_match_ids=val_match_ids,
    )
    atomic_write_json(eval_suite_path(data_dir), artifact.to_dict())
    return artifact


def load_eval_suite(data_dir: Path) -> EvalSuiteArtifact | None:
    path = eval_suite_path(data_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalSuiteArtifact.from_dict(data)


def require_eval_suite(data_dir: Path) -> EvalSuiteArtifact:
    artifact = load_eval_suite(data_dir)
    if artifact is None:
        raise EvalSuiteError(
            f"eval suite artifact missing at {eval_suite_path(data_dir)}; "
            "run eval_suite init after verify"
        )
    return artifact
