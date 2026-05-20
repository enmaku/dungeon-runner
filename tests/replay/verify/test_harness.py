"""Node verify_match.mjs harness (requires PORTFOLIO_SITE_ROOT)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from dungeon_runner.replay.verify import run_verify
from dungeon_runner.replay.verify_manifest import load_verify_manifest
from dungeon_runner.replay.web_engine import (
    default_harness_path,
    default_node_command,
)

from tests.replay.helpers import (
    FIXTURES,
    GOLDEN_KERNEL_FIXTURE,
    VERIFY_FIXTURE_OUTCOMES,
    golden_kernel_envelope,
    seed_ingested,
    seed_ingested_envelope,
)


@pytest.mark.parametrize(
    ("fixture_name", "expected_code"),
    list(VERIFY_FIXTURE_OUTCOMES.items()),
)
def test_node_harness_fixture_outcomes(
    tmp_path: Path,
    skip_without_portfolio: Path,
    fixture_name: str,
    expected_code: str | None,
):
    data_dir = tmp_path / "replays"
    match_id = f"match-{fixture_name.replace('.json', '')}"
    seed_ingested(data_dir, match_id, fixture_name)

    summary = run_verify(data_dir=data_dir, portfolio_root=skip_without_portfolio)

    manifest = load_verify_manifest(data_dir)
    if expected_code is None:
        assert match_id in summary.verified
        assert match_id in manifest.verified
    else:
        assert any(e["id"] == match_id for e in summary.failed)
        failed = next(e for e in manifest.failed if e["id"] == match_id)
        assert failed["reason"]["code"] == expected_code


def test_golden_kernel_fixture_replays_without_step_failures(
    tmp_path: Path,
    skip_without_portfolio: Path,
):
    golden_path = skip_without_portfolio / GOLDEN_KERNEL_FIXTURE
    if not golden_path.is_file():
        pytest.skip(f"missing portfolio-site golden fixture: {golden_path}")

    data_dir = tmp_path / "replays"
    match_id = "match-golden-seed-4242"
    seed_ingested_envelope(
        data_dir, match_id, golden_kernel_envelope(skip_without_portfolio)
    )

    summary = run_verify(data_dir=data_dir, portfolio_root=skip_without_portfolio)

    assert match_id not in summary.verified
    failed = next(e for e in summary.failed if e["id"] == match_id)
    assert failed["reason"]["code"] == "match_not_over"
    assert "step" not in failed["reason"]


def test_verify_harness_illegal_action_stdout(
    skip_without_portfolio: Path,
):
    envelope = FIXTURES / "illegal-action.json"
    proc = subprocess.run(
        [
            *default_node_command(),
            str(default_harness_path()),
            str(envelope),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PORTFOLIO_SITE_ROOT": str(skip_without_portfolio)},
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())
    assert payload == {
        "ok": False,
        "failure": {"code": "illegal_action", "step": 1, "detail": "INVALID_ACTION"},
    }


def test_fixture_action_types_encode_when_portfolio_set(
    skip_without_portfolio: Path,
    replay_fixtures: Path,
):
    types: set[str] = set()
    for path in replay_fixtures.glob("*.json"):
        for entry in json.loads(path.read_text(encoding="utf-8")).get("history", []):
            action = entry.get("action", {})
            if isinstance(action.get("type"), str):
                types.add(action["type"])

    assert types, "expected action types from replay fixtures"

    node = default_node_command()
    env = os.environ.copy()
    env["PORTFOLIO_SITE_ROOT"] = str(skip_without_portfolio)
    probe = f"""
import {{ createInitialMatchState, getLegalActions }} from './src/features/dungeon-runner/engine/kernel.js';
import {{ encodeActionIndex }} from './src/features/dungeon-runner/nn/policyAdapter.js';
const types = {json.dumps(sorted(types))};
const state = createInitialMatchState({{ totalSeats: 2, opponents: [{{ type: 'randombot' }}] }}, {{ seed: 1 }});
const seatId = state.turn.activeSeatId;
const actor = {{ seatId }};
let failed = [];
for (const type of types) {{
  const legal = getLegalActions(state, actor);
  const sample = legal.find((a) => a.type === type);
  if (!sample) continue;
  if (encodeActionIndex(state, actor, sample) < 0) failed.push(type);
}}
if (failed.length) {{
  console.error('unmapped:', failed.join(','));
  process.exit(1);
}}
"""
    proc = subprocess.run(
        [*node, "--input-type=module", "-e", probe],
        cwd=skip_without_portfolio,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
