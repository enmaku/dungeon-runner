"""Align promotion manifests with portfolio-site model catalog publishedAt values."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dungeon_runner.replay.eval.atomic_json import atomic_write_json
from dungeon_runner.replay.publish.stage import validate_promoted_at


class BackfillTimestampsError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackfillChange:
    model_id: str
    field: str
    old_value: str | None
    new_value: str


@dataclass(frozen=True)
class BackfillSummary:
    changes: tuple[BackfillChange, ...]
    dry_run: bool


def parse_catalog_published_at_by_id(catalog_path: Path) -> dict[str, str]:
    if not catalog_path.is_file():
        raise BackfillTimestampsError(f"catalog not found: {catalog_path}")
    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BackfillTimestampsError(f"invalid catalog JSON: {catalog_path}") from exc

    models = raw.get("models")
    if not isinstance(models, list):
        raise BackfillTimestampsError("catalog.models must be an array")

    dates: dict[str, str] = {}
    for entry in models:
        if isinstance(entry, str):
            entry_id = entry.strip()
            if entry_id and entry_id != "latest":
                dates.setdefault(entry_id, dates.get(entry_id, ""))
            continue
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id")
        published_at = entry.get("publishedAt")
        if not isinstance(entry_id, str) or not entry_id.strip():
            continue
        entry_id = entry_id.strip()
        if entry_id == "latest":
            continue
        if isinstance(published_at, str) and published_at.strip():
            dates[entry_id] = validate_promoted_at(published_at)
    return dates


def default_catalog_path(portfolio_site_root: Path) -> Path:
    return portfolio_site_root / "public" / "models" / "dungeon-runner" / "models.json"


def _plan_ledger_changes(
    ledger_path: Path,
    dates: dict[str, str],
) -> tuple[list[dict[str, object]], list[BackfillChange]]:
    if not ledger_path.is_file():
        return [], []

    rows: list[dict[str, object]] = []
    changes: list[BackfillChange] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue
        row = json.loads(trimmed)
        version = row.get("promoted_version")
        if not isinstance(version, str):
            rows.append(row)
            continue
        target = dates.get(version)
        if target is None:
            rows.append(row)
            continue
        old = row.get("promoted_at")
        old_str = old if isinstance(old, str) else None
        if old_str == target:
            rows.append(row)
            continue
        updated = dict(row)
        updated["promoted_at"] = target
        rows.append(updated)
        changes.append(
            BackfillChange(
                model_id=version,
                field="promotions.jsonl",
                old_value=old_str,
                new_value=target,
            )
        )
    return rows, changes


def _plan_promotion_json_change(
    version_dir: Path,
    model_id: str,
    published_at: str,
) -> BackfillChange | None:
    manifest_path = version_dir / "promotion.json"
    if manifest_path.is_file():
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        old = record.get("promoted_at")
        old_str = old if isinstance(old, str) else None
        if old_str == published_at:
            return None
        return BackfillChange(
            model_id=model_id,
            field="promotion.json",
            old_value=old_str,
            new_value=published_at,
        )

    if not version_dir.is_dir():
        return None
    return BackfillChange(
        model_id=model_id,
        field="promotion.json (create)",
        old_value=None,
        new_value=published_at,
    )


def run_backfill_timestamps(
    *,
    repo_root: Path,
    catalog_path: Path,
    dry_run: bool = False,
) -> BackfillSummary:
    repo_root = repo_root.resolve()
    catalog_path = catalog_path.resolve()
    models_dir = repo_root / "models"
    dates = parse_catalog_published_at_by_id(catalog_path)

    changes: list[BackfillChange] = []
    ledger_path = models_dir / "promotions.jsonl"
    ledger_rows, ledger_changes = _plan_ledger_changes(ledger_path, dates)
    changes.extend(ledger_changes)

    promotion_patches: list[tuple[Path, dict[str, object] | None, str]] = []
    for model_id, published_at in sorted(dates.items()):
        version_dir = models_dir / model_id
        change = _plan_promotion_json_change(version_dir, model_id, published_at)
        if change is None:
            continue
        changes.append(change)
        manifest_path = version_dir / "promotion.json"
        if manifest_path.is_file():
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["promoted_at"] = published_at
            promotion_patches.append((manifest_path, record, published_at))
        elif version_dir.is_dir():
            promotion_patches.append(
                (
                    manifest_path,
                    {
                        "promoted_version": model_id,
                        "promoted_at": published_at,
                    },
                    published_at,
                )
            )

    if dry_run:
        return BackfillSummary(changes=tuple(changes), dry_run=True)

    if ledger_changes:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(json.dumps(row) + "\n" for row in ledger_rows)
        tmp = ledger_path.with_suffix(".jsonl.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(ledger_path)

    for manifest_path, record, _ in promotion_patches:
        if record is not None:
            atomic_write_json(manifest_path, record)

    return BackfillSummary(changes=tuple(changes), dry_run=False)
