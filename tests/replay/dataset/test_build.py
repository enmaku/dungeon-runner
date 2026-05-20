"""Dataset build: derived store from verified replays."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from dungeon_runner.replay.dataset import (
    DATASET_ENCODING_VERSION,
    DatasetBuildError,
    derived_rows_path,
    load_derived_meta,
    pending_dataset_ids,
    run_dataset,
)
from dungeon_runner.replay.eval.derived_store import load_match_rows
from dungeon_runner.replay.eval.eval_suite import init_eval_suite
from dungeon_runner.replay.web_engine import default_harness_path
from tests.replay.helpers import seed_ingested, seed_verify_state


def _mock_dataset_harness(tmp_path: Path, responses: dict[str, dict]) -> Path:
    script = tmp_path / "mock_dataset.py"
    mapping = json.dumps(responses)
    script.write_text(
        f"""#!/usr/bin/env python3
import json
import sys
from pathlib import Path

responses = json.loads({mapping!r})
match_id = Path(sys.argv[2]).stem
payload = responses.get(match_id, {{"ok": False, "failure": {{"code": "engine_error", "detail": "unknown match"}}}})
print(json.dumps(payload))
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _sample_rows(human_seat: str = "seat-1") -> dict:
    return {
        "ok": True,
        "encoding_version": DATASET_ENCODING_VERSION,
        "human_seat_id": human_seat,
        "rows": [
            {
                "step": 0,
                "seat": human_seat,
                "obs": [0.0] * 87,
                "mask": [1] + [0] * 25,
                "policy_action_index": 0,
                "phase": "bidding",
                "subphase": "turn",
                "is_human": True,
                "model_id": None,
                "nn_debug": None,
            }
        ],
    }


def test_run_dataset_fails_without_eval_suite(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_verify_state(data_dir, verified=["match-a", "match-b"])
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")

    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()

    with pytest.raises(DatasetBuildError, match="eval suite"):
        run_dataset(data_dir=data_dir, portfolio_root=portfolio)


def test_run_dataset_fails_without_portfolio_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_dir = tmp_path / "replays"
    seed_verify_state(data_dir, verified=["match-a", "match-b"])
    init_eval_suite(data_dir, sampling_seed=42)
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")
    monkeypatch.delenv("PORTFOLIO_SITE_ROOT", raising=False)

    with pytest.raises(RuntimeError, match="PORTFOLIO_SITE_ROOT"):
        run_dataset(data_dir=data_dir)


def test_pending_dataset_ids_encode_all_includes_built(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_verify_state(data_dir, verified=["match-a", "match-b"])
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")
    seed_ingested(data_dir, "match-b", "valid-match-over-seed42.json")
    match_dir = data_dir / "derived" / "match-a"
    match_dir.mkdir(parents=True)
    meta = {
        "match_id": "match-a",
        "encoding_version": DATASET_ENCODING_VERSION,
        "row_count": 1,
        "built_at": "2026-05-19T12:00:00Z",
    }
    (match_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (match_dir / "rows.parquet").write_bytes(b"stub")

    assert pending_dataset_ids(data_dir) == ["match-b"]
    assert pending_dataset_ids(data_dir, encode_all=True) == ["match-a", "match-b"]


def test_pending_dataset_ids_requeues_stale_encoding_version(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_verify_state(data_dir, verified=["match-a", "match-b"])
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")
    match_dir = data_dir / "derived" / "match-a"
    match_dir.mkdir(parents=True)
    meta = {
        "match_id": "match-a",
        "encoding_version": DATASET_ENCODING_VERSION - 1,
        "row_count": 1,
        "built_at": "2026-01-01T00:00:00Z",
    }
    (match_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (match_dir / "rows.parquet").write_bytes(b"stub")

    assert pending_dataset_ids(data_dir) == ["match-a"]


def test_pending_dataset_ids_only_verified_with_raw(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_verify_state(
        data_dir,
        ingested=["match-a", "match-b", "match-failed", "match-unverified"],
        verified=["match-a", "match-b"],
        failed=[{"id": "match-failed", "reason": {"code": "illegal_action", "step": 0}}],
    )
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")
    seed_ingested(data_dir, "match-b", "valid-match-over-seed42.json")
    seed_ingested(data_dir, "match-failed", "valid-match-over-seed42.json")
    seed_ingested(data_dir, "match-unverified", "valid-match-over-seed42.json")

    assert pending_dataset_ids(data_dir) == ["match-a", "match-b"]


def test_manual_redataset_after_delete_derived(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_verify_state(data_dir, verified=["match-a", "match-b"])
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")
    init_eval_suite(data_dir, sampling_seed=42)
    match_dir = data_dir / "derived" / "match-a"
    match_dir.mkdir(parents=True)
    meta = {
        "match_id": "match-a",
        "encoding_version": DATASET_ENCODING_VERSION,
        "row_count": 1,
        "built_at": "2026-05-19T12:00:00Z",
    }
    (match_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (match_dir / "rows.parquet").write_bytes(b"stub")

    assert pending_dataset_ids(data_dir) == []

    import shutil

    shutil.rmtree(match_dir)
    assert pending_dataset_ids(data_dir) == ["match-a"]


def test_run_dataset_one_harness_call_per_match(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_verify_state(data_dir, verified=["match-a", "match-b"])
    init_eval_suite(data_dir, sampling_seed=42)
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")
    seed_ingested(data_dir, "match-b", "valid-match-over-seed42.json")

    calls: list[str] = []

    def counting_build(*, envelope_path: Path, **_kwargs: object) -> dict:
        calls.append(envelope_path.stem)
        return _sample_rows()

    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()
    run_dataset(
        data_dir=data_dir,
        portfolio_root=portfolio,
        build_fn=counting_build,
    )
    assert calls == ["match-a", "match-b"]


def test_pending_dataset_ids_skips_current_encoding_version(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_verify_state(data_dir, verified=["match-a", "match-b"])
    seed_ingested(data_dir, "match-b", "valid-match-over-seed42.json")
    match_dir = data_dir / "derived" / "match-a"
    match_dir.mkdir(parents=True)
    meta = {
        "match_id": "match-a",
        "encoding_version": DATASET_ENCODING_VERSION,
        "row_count": 3,
        "built_at": "2026-05-19T12:00:00Z",
    }
    (match_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (match_dir / "rows.parquet").write_bytes(b"stub")

    assert pending_dataset_ids(data_dir) == ["match-b"]


def test_run_dataset_writes_parquet_with_mock_harness(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_verify_state(data_dir, verified=["match-a", "match-b"])
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")
    seed_ingested(data_dir, "match-b", "valid-match-over-seed42.json")

    suite = init_eval_suite(data_dir, sampling_seed=42)
    val_id = suite.val_match_ids[0]
    train_id = next(m for m in ["match-a", "match-b"] if m != val_id)

    mock = _mock_dataset_harness(
        tmp_path,
        {
            train_id: _sample_rows(),
            val_id: _sample_rows(),
        },
    )
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()

    summary = run_dataset(
        data_dir=data_dir,
        node_cmd=[sys.executable, str(mock)],
        harness_path=default_harness_path().parent / "build_match_dataset.mjs",
        portfolio_root=portfolio,
    )

    assert set(summary.built) == {train_id, val_id}
    for match_id in summary.built:
        meta = load_derived_meta(data_dir, match_id)
        assert meta is not None
        assert meta.encoding_version == DATASET_ENCODING_VERSION
        assert meta.row_count == 1
        assert derived_rows_path(data_dir, match_id).is_file()
        table = pq.read_table(derived_rows_path(data_dir, match_id))
        assert table.num_rows == 1
        assert table.column("split")[0].as_py() in ("train", "val")
        assert "policy_action_index" in table.column_names


def test_run_dataset_parquet_loads_via_derived_store(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_verify_state(data_dir, verified=["match-a", "match-b"])
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")
    init_eval_suite(data_dir, sampling_seed=42)

    mock = _mock_dataset_harness(tmp_path, {"match-a": _sample_rows()})
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()

    run_dataset(
        data_dir=data_dir,
        node_cmd=[sys.executable, str(mock)],
        harness_path=default_harness_path().parent / "build_match_dataset.mjs",
        portfolio_root=portfolio,
        match_ids=["match-a"],
    )

    rows = load_match_rows(derived_rows_path(data_dir, "match-a"))
    assert len(rows) == 1
    assert rows[0].policy_action_index == 0
    assert rows[0].is_human_step is True


def test_run_dataset_aborts_without_partial_artifacts(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_verify_state(data_dir, verified=["match-ok", "match-fail"])
    init_eval_suite(data_dir, sampling_seed=42)
    seed_ingested(data_dir, "match-ok", "valid-match-over-seed42.json")
    seed_ingested(data_dir, "match-fail", "valid-match-over-seed42.json")

    mock = _mock_dataset_harness(
        tmp_path,
        {
            "match-ok": _sample_rows(),
            "match-fail": {
                "ok": False,
                "failure": {"code": "engine_error", "detail": "boom"},
            },
        },
    )
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()

    with pytest.raises(DatasetBuildError):
        run_dataset(
            data_dir=data_dir,
            node_cmd=[sys.executable, str(mock)],
            harness_path=default_harness_path().parent / "build_match_dataset.mjs",
            portfolio_root=portfolio,
        )

    assert not (data_dir / "derived" / "match-ok").exists()
    assert not (data_dir / "derived" / "match-fail").exists()


def test_cli_dataset_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    data_dir = tmp_path / "replays"
    seed_verify_state(data_dir, verified=["match-a", "match-b"])
    init_eval_suite(data_dir, sampling_seed=42)
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")

    mock = _mock_dataset_harness(tmp_path, {"match-a": _sample_rows()})
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()

    def _fake_run_dataset(**kwargs: object) -> object:
        from dungeon_runner.replay.dataset import run_dataset as real_run

        return real_run(
            data_dir=kwargs["data_dir"],
            node_cmd=[sys.executable, str(mock)],
            harness_path=default_harness_path().parent / "build_match_dataset.mjs",
            portfolio_root=portfolio,
        )

    monkeypatch.setattr("dungeon_runner.replay.cli.run_dataset", _fake_run_dataset)

    from dungeon_runner.replay.cli import main

    assert main(["dataset", "--data-dir", str(data_dir)]) == 0
    assert load_derived_meta(data_dir, "match-a") is not None


@pytest.mark.skipif(
    not os.environ.get("PORTFOLIO_SITE_ROOT", "").strip(),
    reason="PORTFOLIO_SITE_ROOT not set",
)
def test_node_harness_valid_match_row_count(tmp_path: Path, skip_without_portfolio: Path):
    data_dir = tmp_path / "replays"
    match_id = "match-valid-seed42"
    seed_verify_state(data_dir, verified=["match-valid-seed42", "match-other"])
    seed_ingested(data_dir, "match-valid-seed42", "valid-match-over-seed42.json")
    seed_ingested(data_dir, "match-other", "valid-match-over-seed42.json")
    init_eval_suite(data_dir, sampling_seed=42)

    summary = run_dataset(
        data_dir=data_dir,
        portfolio_root=skip_without_portfolio,
        match_ids=[match_id],
    )
    assert match_id in summary.built
    meta = load_derived_meta(data_dir, match_id)
    assert meta is not None
    assert meta.row_count == 5
    table = pq.read_table(derived_rows_path(data_dir, match_id))
    assert table.num_rows == 5
    assert len(table.column("obs")[0].as_py()) == 87
    assert len(table.column("mask")[0].as_py()) == 26
    assert table.column("is_human").to_pylist() == [True, False, False, True, True]
    assert "policy_action_index" in table.column_names

    rows = load_match_rows(derived_rows_path(data_dir, match_id))
    assert len(rows) == 5
    assert all(r.split in ("train", "val") for r in rows)
