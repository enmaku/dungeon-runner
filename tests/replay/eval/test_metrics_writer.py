"""Metrics artifact writer round-trip."""

from __future__ import annotations

from pathlib import Path

from dungeon_runner.replay.eval.metrics_writer import load_metrics, write_metrics
from dungeon_runner.replay.eval.replay_metrics import ReplayMetrics
from dungeon_runner.replay.eval.sim_metrics import SimMetrics


def test_write_metrics_round_trip(tmp_path: Path):
    run_dir = tmp_path / "bc-20260519T000000Z"
    path = write_metrics(
        run_dir,
        run_id="bc-20260519T000000Z",
        parent_weights="/abs/models/latest/policy.weights.h5",
        replay=ReplayMetrics(0.8125, 0.1, 32).to_dict(),
        sim=SimMetrics(0.5, 0.48, seed_count=4).to_dict(),
        train={"bc_loss": 0.42},
        timestamp="2026-05-19T00:00:00+00:00",
    )
    assert path == run_dir / "metrics.json"
    payload = load_metrics(path)
    assert payload["run_id"] == "bc-20260519T000000Z"
    assert payload["timestamp"] == "2026-05-19T00:00:00+00:00"
    assert payload["parent_weights"] == "/abs/models/latest/policy.weights.h5"
    assert payload["replay"]["val_masked_accuracy"] == 0.8125
    assert payload["sim"]["candidate_win_rate_vs_randombot"] == 0.5
    assert payload["sim"]["engine"] == "python_training_sim"
    assert payload["train"]["bc_loss"] == 0.42
