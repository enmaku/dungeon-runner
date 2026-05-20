"""Sim eval metrics smoke on Python training sim."""

from __future__ import annotations

from dungeon_runner.bots import RandomBot
from dungeon_runner.replay.eval.sim_metrics import (
    ENGINE_NAME,
    sim_metrics,
    sim_passes_regression,
    win_rate_vs_randombot,
)


def test_win_rate_vs_randombot_empty_seeds():
    assert win_rate_vs_randombot(RandomBot(), []) == 0.0


def test_sim_metrics_smoke_tiny_seed_list():
    bot = RandomBot()
    metrics = sim_metrics(bot, bot, [0, 1, 2])
    assert metrics.engine == ENGINE_NAME
    assert metrics.seed_count == 3
    assert 0.0 <= metrics.candidate_win_rate_vs_randombot <= 1.0
    assert metrics.candidate_win_rate_vs_randombot == metrics.latest_win_rate_vs_randombot


def test_sim_passes_regression_at_tolerance_margin():
    from dungeon_runner.replay.eval.sim_metrics import SimMetrics

    metrics = SimMetrics(
        candidate_win_rate_vs_randombot=0.54,
        latest_win_rate_vs_randombot=0.55,
        seed_count=2,
    )
    assert sim_passes_regression(metrics, 0.01)
    assert not sim_passes_regression(
        SimMetrics(0.53, 0.55, seed_count=2),
        0.01,
    )
