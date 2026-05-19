"""Replay eval metrics on val human-step rows (no second Node replay)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

PredictFn = Callable[[Any, Any], int]


class DerivedRow(Protocol):
    match_id: str
    split: str
    is_human_step: bool
    obs: Any
    mask: Any
    policy_action_index: int


@dataclass(frozen=True)
class ReplayMetrics:
    val_masked_accuracy: float
    disagreement_rate: float
    val_row_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "val_masked_accuracy": self.val_masked_accuracy,
            "disagreement_rate": self.disagreement_rate,
            "val_row_count": self.val_row_count,
        }


def _masked_argmax(predict: PredictFn, obs: Any, mask: Any) -> int:
    return int(predict(obs, mask))


def replay_metrics(
    candidate_predict: PredictFn,
    latest_predict: PredictFn,
    rows: Iterable[DerivedRow],
    *,
    val_match_ids: set[str] | None = None,
) -> ReplayMetrics:
    correct = 0
    disagree = 0
    total = 0
    for row in rows:
        if row.split != "val" or not row.is_human_step:
            continue
        if val_match_ids is not None and row.match_id not in val_match_ids:
            continue
        label = int(row.policy_action_index)
        cand_idx = _masked_argmax(candidate_predict, row.obs, row.mask)
        latest_idx = _masked_argmax(latest_predict, row.obs, row.mask)
        total += 1
        if cand_idx == label:
            correct += 1
        if cand_idx != latest_idx:
            disagree += 1
    if total == 0:
        return ReplayMetrics(
            val_masked_accuracy=0.0,
            disagreement_rate=0.0,
            val_row_count=0,
        )
    return ReplayMetrics(
        val_masked_accuracy=correct / total,
        disagreement_rate=disagree / total,
        val_row_count=total,
    )
