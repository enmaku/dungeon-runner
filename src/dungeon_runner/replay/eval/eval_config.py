"""Eval config artifact: sim seeds, regression tolerance, replay accuracy floor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dungeon_runner.replay.eval.atomic_json import atomic_write_json

DEFAULT_SIM_SEEDS: tuple[int, ...] = tuple(range(16))
DEFAULT_SIM_REGRESSION_TOLERANCE = 0.01


class EvalConfigError(ValueError):
    """Eval config init or load failed."""


@dataclass
class EvalConfigArtifact:
    sim_seeds: list[int]
    sim_regression_tolerance: float
    replay_accuracy_floor: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sim_seeds": list(self.sim_seeds),
            "sim_regression_tolerance": self.sim_regression_tolerance,
            "replay_accuracy_floor": self.replay_accuracy_floor,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalConfigArtifact:
        floor = data.get("replay_accuracy_floor")
        return cls(
            sim_seeds=[int(x) for x in data["sim_seeds"]],
            sim_regression_tolerance=float(data["sim_regression_tolerance"]),
            replay_accuracy_floor=float(floor) if floor is not None else None,
        )


def eval_config_path(data_dir: Path) -> Path:
    return data_dir / "eval_config.json"


def init_eval_config(
    data_dir: Path,
    *,
    overwrite: bool = False,
) -> EvalConfigArtifact:
    path = eval_config_path(data_dir)
    if path.is_file() and not overwrite:
        raise EvalConfigError(
            f"eval config artifact already exists at {path}; "
            "remove it first or pass overwrite=True"
        )
    artifact = EvalConfigArtifact(
        sim_seeds=list(DEFAULT_SIM_SEEDS),
        sim_regression_tolerance=DEFAULT_SIM_REGRESSION_TOLERANCE,
        replay_accuracy_floor=None,
    )
    save_eval_config(data_dir, artifact)
    return artifact


def load_eval_config(data_dir: Path) -> EvalConfigArtifact | None:
    path = eval_config_path(data_dir)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return EvalConfigArtifact.from_dict(data)


def save_eval_config(data_dir: Path, artifact: EvalConfigArtifact) -> None:
    atomic_write_json(eval_config_path(data_dir), artifact.to_dict())


def require_eval_config(data_dir: Path) -> EvalConfigArtifact:
    artifact = load_eval_config(data_dir)
    if artifact is None:
        raise EvalConfigError(
            f"eval config artifact missing at {eval_config_path(data_dir)}; "
            "run eval_config init"
        )
    return artifact
