"""Ingest manifest load/save."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IngestManifest:
    ingested: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def known_ids(self) -> set[str]:
        skipped_ids = {entry["id"] for entry in self.skipped}
        return set(self.ingested) | skipped_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "ingested": list(self.ingested),
            "skipped": list(self.skipped),
        }


def manifest_path(data_dir: Path) -> Path:
    return data_dir / "manifest.json"


def load_manifest(data_dir: Path) -> IngestManifest:
    path = manifest_path(data_dir)
    if not path.is_file():
        return IngestManifest()
    data = json.loads(path.read_text(encoding="utf-8"))
    return IngestManifest(
        ingested=list(data.get("ingested", [])),
        skipped=list(data.get("skipped", [])),
    )


def save_manifest(data_dir: Path, manifest: IngestManifest) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path(data_dir).with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(manifest_path(data_dir))
