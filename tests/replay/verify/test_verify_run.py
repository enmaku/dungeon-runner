"""run_verify orchestration (mocked Node)."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from dungeon_runner.replay.store import raw_path
from dungeon_runner.replay.verify import _VerifyOutcome, run_verify
from dungeon_runner.replay.verify_manifest import (
    VerifyFailure,
    VerifyManifest,
    load_verify_manifest,
    save_verify_manifest,
)
from dungeon_runner.replay.web_engine import default_harness_path

from tests.replay.helpers import seed_ingested


def _mock_harness_script(tmp_path: Path, responses: dict[str, dict]) -> Path:
    script = tmp_path / "mock_verify.py"
    mapping = json.dumps(responses)
    script.write_text(
        f"""#!/usr/bin/env python3
import json
import sys
from pathlib import Path

responses = json.loads({mapping!r})
match_id = Path(sys.argv[2]).stem
payload = responses.get(match_id, {{"ok": True}})
print(json.dumps(payload))
""",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_run_verify_updates_manifest_with_mock_node(tmp_path: Path):
    data_dir = tmp_path / "replays"
    seed_ingested(data_dir, "match-ok", "valid-match-over-seed42.json")
    seed_ingested(data_dir, "match-bad", "actor-mismatch.json")

    mock = _mock_harness_script(
        tmp_path,
        {
            "match-ok": {"ok": True},
            "match-bad": {
                "ok": False,
                "failure": {"code": "actor_mismatch", "step": 0},
            },
        },
    )
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()

    summary = run_verify(
        data_dir=data_dir,
        node_cmd=[sys.executable, str(mock)],
        harness_path=default_harness_path(),
        portfolio_root=portfolio,
    )

    assert summary.verified == ["match-ok"]
    assert len(summary.failed) == 1
    assert summary.failed[0]["id"] == "match-bad"
    assert summary.failed[0]["reason"]["code"] == "actor_mismatch"

    manifest = load_verify_manifest(data_dir)
    assert manifest.verified == ["match-ok"]
    assert manifest.failed[0]["id"] == "match-bad"


def test_run_verify_noop_when_nothing_pending(tmp_path: Path):
    data_dir = tmp_path / "replays"
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()
    summary = run_verify(
        data_dir=data_dir,
        node_cmd=[sys.executable, "-c", "pass"],
        harness_path=default_harness_path(),
        portfolio_root=portfolio,
    )
    assert summary.verified == []
    assert summary.failed == []
    assert not (data_dir / "verify_manifest.json").exists()


def test_run_verify_requires_portfolio_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data_dir = tmp_path / "replays"
    seed_ingested(data_dir, "match-ok", "valid-match-over-seed42.json")
    monkeypatch.delenv("PORTFOLIO_SITE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="PORTFOLIO_SITE_ROOT"):
        run_verify(data_dir=data_dir)


def test_run_verify_one_harness_call_per_pending_match(tmp_path: Path):
    """US 3: one Node/harness invocation per pending match."""
    data_dir = tmp_path / "replays"
    seed_ingested(data_dir, "match-a", "valid-match-over-seed42.json")
    seed_ingested(data_dir, "match-b", "valid-match-over-seed42.json")
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()
    calls: list[str] = []

    def counting_verify(match_id: str, *_args, **_kwargs) -> _VerifyOutcome:
        calls.append(match_id)
        return _VerifyOutcome(match_id=match_id, ok=True)

    run_verify(
        data_dir=data_dir,
        node_cmd=[sys.executable, "-c", "pass"],
        harness_path=default_harness_path(),
        portfolio_root=portfolio,
        verify_fn=counting_verify,
    )

    assert calls == ["match-a", "match-b"]


def test_verify_failure_preserves_raw_envelope(tmp_path: Path):
    """US 13: failed verify does not delete or rewrite raw/{matchId}.json."""
    data_dir = tmp_path / "replays"
    seed_ingested(data_dir, "match-bad", "actor-mismatch.json")
    raw = raw_path(data_dir, "match-bad")
    before = raw.read_bytes()
    portfolio = tmp_path / "portfolio"
    portfolio.mkdir()

    run_verify(
        data_dir=data_dir,
        node_cmd=[sys.executable, "-c", "pass"],
        harness_path=default_harness_path(),
        portfolio_root=portfolio,
        verify_fn=lambda match_id, *_a, **_k: _VerifyOutcome(
            match_id=match_id,
            ok=False,
            failure=VerifyFailure(code="actor_mismatch", step=0),
        ),
    )

    assert raw.read_bytes() == before


def test_verify_manifest_round_trip(tmp_path: Path):
    data_dir = tmp_path / "replays"
    manifest = VerifyManifest(
        verified=["match-a"],
        failed=[{"id": "match-b", "reason": {"code": "illegal_action", "step": 1}}],
    )
    save_verify_manifest(data_dir, manifest)
    loaded = load_verify_manifest(data_dir)
    assert loaded.verified == ["match-a"]
    assert loaded.failed == manifest.failed
    assert loaded.known_ids() == {"match-a", "match-b"}
