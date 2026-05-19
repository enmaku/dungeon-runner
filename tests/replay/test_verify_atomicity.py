"""Verify run atomicity."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from dungeon_runner.replay.verify import _VerifyOutcome, run_verify
from dungeon_runner.replay.verify_manifest import load_verify_manifest
from dungeon_runner.replay.web_engine import default_harness_path

from tests.replay.helpers import seed_ingested


def test_verify_aborts_without_manifest_on_mid_run_failure(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")
    seed_ingested(data_dir, "match-b", "valid-match-over-seed42.json")
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()

    calls: list[str] = []

    def flaky_verify(match_id: str, *_args, **_kwargs) -> _VerifyOutcome:
        calls.append(match_id)
        if match_id == "match-b":
            raise RuntimeError("node crashed")
        return _VerifyOutcome(match_id=match_id, ok=True)

    with pytest.raises(RuntimeError, match="node crashed"):
        run_verify(
            data_dir=data_dir,
            node_cmd=[sys.executable, "-c", "pass"],
            harness_path=default_harness_path(),
            portfolio_root=portfolio,
            verify_fn=flaky_verify,
        )

    assert calls == ["match-a", "match-b"]
    assert not (data_dir / "verify_manifest.json").exists()
