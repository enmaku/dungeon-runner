"""Shared replay test helpers (importable from test modules)."""

from __future__ import annotations

import json
from pathlib import Path

from dungeon_runner.replay.manifest import IngestManifest, load_manifest, save_manifest
from dungeon_runner.replay.store import write_raw_envelope
from dungeon_runner.replay.verify_manifest import VerifyManifest, save_verify_manifest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "replay"

# Committed verify harness outcomes (None = verified).
VERIFY_FIXTURE_OUTCOMES: dict[str, str | None] = {
    "valid-match-over-seed42.json": None,
    "match-not-over.json": "match_not_over",
    "actor-mismatch.json": "actor_mismatch",
    "rng-chain-break.json": "rng_chain_break",
    "unmapped-action-type.json": "unmapped_action_type",
    "illegal-action.json": "illegal_action",
}

GOLDEN_KERNEL_FIXTURE = (
    "src/features/dungeon-runner/engine/fixtures/golden-seed-4242-two-pass.json"
)


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def golden_kernel_envelope(portfolio_root: Path) -> dict:
    """v1 envelope built from portfolio-site kernel golden fixture history."""
    path = portfolio_root / GOLDEN_KERNEL_FIXTURE
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "version": 1,
        "seed": data["seed"],
        "setup": data["setup"],
        "history": [
            {
                "action": entry["action"],
                "actorSeatId": entry["actorSeatId"],
                "rngStepBefore": entry["rngStepBefore"],
                "rngStepAfter": entry["rngStepAfter"],
            }
            for entry in data["expected"]["history"]
        ],
    }


def seed_ingested(data_dir: Path, match_id: str, fixture_name: str) -> None:
    seed_ingested_envelope(data_dir, match_id, load_fixture(fixture_name))


def seed_ingested_envelope(data_dir: Path, match_id: str, envelope: dict) -> None:
    write_raw_envelope(data_dir, match_id, envelope)
    manifest = load_manifest(data_dir)
    if match_id not in manifest.ingested:
        manifest.ingested.append(match_id)
    save_manifest(data_dir, manifest)


def seed_verify_state(
    data_dir: Path,
    *,
    ingested: list[str] | None = None,
    verified: list[str] | None = None,
    failed: list[dict] | None = None,
) -> None:
    save_manifest(
        data_dir,
        IngestManifest(ingested=list(ingested or [])),
    )
    save_verify_manifest(
        data_dir,
        VerifyManifest(verified=list(verified or []), failed=list(failed or [])),
    )
