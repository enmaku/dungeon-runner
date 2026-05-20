"""Publish gate runner: dedup, PPO regression, shared gate evaluator."""

from __future__ import annotations

from typing import Any

from dungeon_runner.replay.eval.eval_config import EvalConfigArtifact
from dungeon_runner.replay.publish.publish_gates import run_publish_gates


def _config(*, floor: float | None = 0.75) -> EvalConfigArtifact:
    return EvalConfigArtifact(
        sim_seeds=[0],
        sim_regression_tolerance=0.01,
        replay_accuracy_floor=floor,
    )


def _metrics(
    *,
    val_acc: float = 0.8,
    cand_wr: float = 0.6,
    latest_wr: float = 0.55,
    ppo_pass: bool | None = None,
) -> dict[str, Any]:
    m: dict[str, Any] = {
        "replay": {"val_masked_accuracy": val_acc},
        "sim": {
            "candidate_win_rate_vs_randombot": cand_wr,
            "latest_win_rate_vs_randombot": latest_wr,
        },
    }
    if ppo_pass is not None:
        m["ppo_bc_regression"] = {"pass": ppo_pass}
    return m


def test_passes_bc_run_when_gates_ok():
    result = run_publish_gates(
        _metrics(),
        _config(),
        run_id="bc-20260518T120000Z",
        promoted_run_ids=set(),
    )
    assert result.passed
    assert result.reasons == []


def test_rejects_already_promoted_run_id():
    result = run_publish_gates(
        _metrics(),
        _config(),
        run_id="bc-20260518T120000Z",
        promoted_run_ids={"bc-20260518T120000Z"},
    )
    assert not result.passed
    assert "already_promoted" in result.reasons


def test_ppo_run_requires_bc_regression_pass():
    result = run_publish_gates(
        _metrics(ppo_pass=False),
        _config(),
        run_id="ppo-20260518T120000Z",
        promoted_run_ids=set(),
    )
    assert not result.passed
    assert "ppo_bc_regression_failed" in result.reasons


def test_ppo_run_passes_with_regression_flag():
    result = run_publish_gates(
        _metrics(ppo_pass=True),
        _config(),
        run_id="ppo-20260518T120000Z",
        promoted_run_ids=set(),
    )
    assert result.passed


def test_replay_below_floor_reason():
    result = run_publish_gates(
        _metrics(val_acc=0.5),
        _config(floor=0.75),
        run_id="bc-20260518T120000Z",
        promoted_run_ids=set(),
    )
    assert not result.passed
    assert "replay_below_floor" in result.reasons


def test_sim_regression_reason():
    result = run_publish_gates(
        _metrics(cand_wr=0.5, latest_wr=0.55),
        _config(),
        run_id="bc-20260518T120000Z",
        promoted_run_ids=set(),
    )
    assert not result.passed
    assert "sim_regression" in result.reasons
