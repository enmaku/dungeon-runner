"""PPO BC regression check."""

from __future__ import annotations

from dungeon_runner.replay.ppo.regression import check_ppo_bc_regression


def _metrics(replay_acc: float, sim_wr: float) -> dict:
    return {
        "replay": {"val_masked_accuracy": replay_acc},
        "sim": {"candidate_win_rate_vs_randombot": sim_wr},
    }


def test_regression_passes_when_replay_strict_and_sim_within_epsilon():
    bc = _metrics(0.80, 0.55)
    cand = _metrics(0.82, 0.54)
    assert check_ppo_bc_regression(cand, bc, epsilon=0.01)


def test_regression_fails_when_replay_below_bc():
    bc = _metrics(0.80, 0.50)
    cand = _metrics(0.79, 0.60)
    assert not check_ppo_bc_regression(cand, bc, epsilon=0.05)


def test_regression_fails_when_sim_below_bc_minus_epsilon():
    bc = _metrics(0.80, 0.55)
    cand = _metrics(0.80, 0.53)
    assert not check_ppo_bc_regression(cand, bc, epsilon=0.01)
