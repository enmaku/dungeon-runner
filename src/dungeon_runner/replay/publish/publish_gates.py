"""Publish gate evaluation: dedup, PPO check, then shared promotion gates."""

from __future__ import annotations

from typing import Any

from dungeon_runner.replay.eval.eval_config import EvalConfigArtifact
from dungeon_runner.replay.eval.gate_evaluator import GateResult, evaluate_gates


def run_publish_gates(
    metrics: dict[str, Any],
    eval_config: EvalConfigArtifact,
    run_id: str,
    *,
    promoted_run_ids: set[str],
) -> GateResult:
    reasons: list[str] = []
    if run_id in promoted_run_ids:
        reasons.append("already_promoted")
    if run_id.startswith("ppo-"):
        ppo = metrics.get("ppo_bc_regression") or {}
        if not ppo.get("pass"):
            reasons.append("ppo_bc_regression_failed")
    gate = evaluate_gates(metrics, eval_config)
    reasons.extend(gate.reasons)
    return GateResult(passed=len(reasons) == 0, reasons=reasons)
