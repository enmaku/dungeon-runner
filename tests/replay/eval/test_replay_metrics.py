"""Replay eval metrics on derived-store Parquet fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from dungeon_runner.replay.eval.derived_store import load_derived_rows, load_match_rows
from dungeon_runner.replay.eval.replay_metrics import replay_metrics
from tests.replay.eval.parquet_fixtures import (
    FIXTURE_MATCH_TRAIN,
    FIXTURE_MATCH_VAL,
    obs_half_predict,
    obs_label_predict,
    write_replay_metrics_fixture,
)


@pytest.fixture
def derived_store(tmp_path: Path) -> Path:
    write_replay_metrics_fixture(tmp_path)
    return tmp_path


def test_replay_metrics_from_parquet_fixture(derived_store: Path):
    rows = list(load_derived_rows(derived_store))
    metrics = replay_metrics(
        obs_label_predict,
        obs_half_predict,
        rows,
        val_match_ids={FIXTURE_MATCH_VAL},
    )
    assert metrics.val_row_count == 2
    assert metrics.val_masked_accuracy == 1.0
    assert metrics.disagreement_rate == 1.0


def test_replay_metrics_ignores_train_and_nn_rows(derived_store: Path):
    rows = list(load_derived_rows(derived_store, match_ids=[FIXTURE_MATCH_TRAIN]))
    metrics = replay_metrics(obs_label_predict, obs_label_predict, rows)
    assert metrics.val_row_count == 0


def test_replay_metrics_ignores_nn_rows_on_val_match(derived_store: Path):
    rows = list(load_derived_rows(derived_store, match_ids=[FIXTURE_MATCH_VAL]))
    metrics = replay_metrics(
        obs_label_predict,
        obs_label_predict,
        rows,
        val_match_ids={FIXTURE_MATCH_VAL},
    )
    assert metrics.val_row_count == 2


def test_replay_metrics_without_val_match_id_filter(derived_store: Path):
    rows = list(load_derived_rows(derived_store))
    metrics = replay_metrics(obs_label_predict, obs_label_predict, rows)
    assert metrics.val_row_count == 2
    assert metrics.val_masked_accuracy == 1.0
    assert metrics.disagreement_rate == 0.0


COMMITTED_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "replay" / "derived"
)


def test_load_committed_parquet_fixture():
    rows = load_match_rows(COMMITTED_FIXTURE / "match-val" / "rows.parquet")
    assert len(rows) == 3
    human_val = [r for r in rows if r.split == "val" and r.is_human_step]
    assert len(human_val) == 2
