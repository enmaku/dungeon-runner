"""Verify manifest load/save."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VerifyFailure:
    code: str
    step: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"code": self.code}
        if self.step is not None:
            out["step"] = self.step
        if self.detail is not None:
            out["detail"] = self.detail
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifyFailure:
        return cls(
            code=str(data["code"]),
            step=data.get("step") if isinstance(data.get("step"), int) else None,
            detail=data.get("detail") if isinstance(data.get("detail"), str) else None,
        )


@dataclass
class VerifyManifest:
    verified: list[str] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)

    def known_ids(self) -> set[str]:
        failed_ids = {entry["id"] for entry in self.failed}
        return set(self.verified) | failed_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": list(self.verified),
            "failed": list(self.failed),
        }


def verify_manifest_path(data_dir: Path) -> Path:
    return data_dir / "verify_manifest.json"


def load_verify_manifest(data_dir: Path) -> VerifyManifest:
    path = verify_manifest_path(data_dir)
    if not path.is_file():
        return VerifyManifest()
    data = json.loads(path.read_text(encoding="utf-8"))
    return VerifyManifest(
        verified=list(data.get("verified", [])),
        failed=list(data.get("failed", [])),
    )


def save_verify_manifest(data_dir: Path, manifest: VerifyManifest) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = verify_manifest_path(data_dir).with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(verify_manifest_path(data_dir))
