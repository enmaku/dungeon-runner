"""Promotion gate pass/fail from metrics artifact + eval config."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dungeon_runner.replay.eval.eval_config import EvalConfigArtifact


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]


def evaluate_gates(
    metrics: dict[str, Any],
    eval_config: EvalConfigArtifact,
) -> GateResult:
    reasons: list[str] = []

    if eval_config.replay_accuracy_floor is None:
        return GateResult(passed=False, reasons=["replay_accuracy_floor_not_set"])

    replay = metrics.get("replay") or {}
    val_acc = replay.get("val_masked_accuracy")
    if val_acc is None:
        reasons.append("missing_replay_val_masked_accuracy")
    elif float(val_acc) < eval_config.replay_accuracy_floor:
        reasons.append("replay_below_floor")

    sim = metrics.get("sim") or {}
    candidate_wr = sim.get("candidate_win_rate_vs_randombot")
    latest_wr = sim.get("latest_win_rate_vs_randombot")
    if candidate_wr is None or latest_wr is None:
        reasons.append("missing_sim_win_rates")
    else:
        margin = float(latest_wr) - eval_config.sim_regression_tolerance
        if float(candidate_wr) < margin:
            reasons.append("sim_regression")

    return GateResult(passed=len(reasons) == 0, reasons=reasons)
