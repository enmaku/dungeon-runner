"""PPO BC regression check vs BC-only candidate metrics artifact."""

from __future__ import annotations

from typing import Any


def check_ppo_bc_regression(
    candidate_metrics: dict[str, Any],
    bc_metrics: dict[str, Any],
    *,
    epsilon: float,
) -> bool:
    cand_replay = float(
        (candidate_metrics.get("replay") or {})["val_masked_accuracy"]
    )
    bc_replay = float((bc_metrics.get("replay") or {})["val_masked_accuracy"])
    if cand_replay < bc_replay:
        return False

    cand_sim = float(
        (candidate_metrics.get("sim") or {})["candidate_win_rate_vs_randombot"]
    )
    bc_sim = float(
        (bc_metrics.get("sim") or {})["candidate_win_rate_vs_randombot"]
    )
    return cand_sim >= bc_sim - epsilon
