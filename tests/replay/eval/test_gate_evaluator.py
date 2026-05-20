"""Gate evaluator: pre-floor fail closed, replay floor, sim regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from dungeon_runner.replay.eval.eval_config import EvalConfigArtifact
from dungeon_runner.replay.eval.gate_evaluator import evaluate_gates


@dataclass
class _ConfigFactory:
    floor: float | None = 0.75
    tolerance: float = 0.01

    def build(self) -> EvalConfigArtifact:
        return EvalConfigArtifact(
            sim_seeds=list(range(4)),
            sim_regression_tolerance=self.tolerance,
            replay_accuracy_floor=self.floor,
        )


def _metrics(
    *,
    val_acc: float | None = 0.8,
    cand_wr: float | None = 0.6,
    latest_wr: float | None = 0.55,
) -> dict[str, Any]:
    replay: dict[str, Any] = {}
    if val_acc is not None:
        replay["val_masked_accuracy"] = val_acc
    sim: dict[str, Any] = {}
    if cand_wr is not None:
        sim["candidate_win_rate_vs_randombot"] = cand_wr
    if latest_wr is not None:
        sim["latest_win_rate_vs_randombot"] = latest_wr
    return {"replay": replay, "sim": sim}


def test_pre_floor_fails_closed():
    config = _ConfigFactory(floor=None).build()
    result = evaluate_gates(_metrics(), config)
    assert not result.passed
    assert result.reasons == ["replay_accuracy_floor_not_set"]


def test_replay_below_floor_fails():
    config = _ConfigFactory(floor=0.8).build()
    result = evaluate_gates(_metrics(val_acc=0.79), config)
    assert not result.passed
    assert "replay_below_floor" in result.reasons


def test_sim_regression_fails():
    config = _ConfigFactory(floor=0.5, tolerance=0.01).build()
    result = evaluate_gates(_metrics(val_acc=0.9, cand_wr=0.50, latest_wr=0.55), config)
    assert not result.passed
    assert "sim_regression" in result.reasons


def test_passes_when_replay_and_sim_ok():
    config = _ConfigFactory(floor=0.7, tolerance=0.01).build()
    result = evaluate_gates(_metrics(val_acc=0.85, cand_wr=0.54, latest_wr=0.55), config)
    assert result.passed
    assert result.reasons == []


def test_sim_passes_at_tolerance_margin():
    config = _ConfigFactory(floor=0.5, tolerance=0.01).build()
    result = evaluate_gates(_metrics(val_acc=0.9, cand_wr=0.54, latest_wr=0.55), config)
    assert result.passed


def test_missing_replay_metric_fails():
    config = _ConfigFactory(floor=0.5).build()
    result = evaluate_gates(_metrics(val_acc=None), config)
    assert not result.passed
    assert "missing_replay_val_masked_accuracy" in result.reasons


def test_missing_sim_win_rates_fails():
    config = _ConfigFactory(floor=0.5).build()
    result = evaluate_gates(_metrics(val_acc=0.9, cand_wr=None, latest_wr=None), config)
    assert not result.passed
    assert "missing_sim_win_rates" in result.reasons


def test_replay_at_floor_passes():
    config = _ConfigFactory(floor=0.8).build()
    result = evaluate_gates(_metrics(val_acc=0.8, cand_wr=0.6, latest_wr=0.55), config)
    assert result.passed


def test_multiple_gate_failures_reported():
    config = _ConfigFactory(floor=0.9, tolerance=0.01).build()
    result = evaluate_gates(_metrics(val_acc=0.5, cand_wr=0.4, latest_wr=0.6), config)
    assert not result.passed
    assert "replay_below_floor" in result.reasons
    assert "sim_regression" in result.reasons
