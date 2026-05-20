"""Next promoted semver under ADR 0002 (v0.2+ line, two-digit patch)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_LEGACY_EPOCH = re.compile(r"^v0\.1\.\d+a$")
_BASE = re.compile(r"^v(\d+)\.(\d+)$")
_PATCH = re.compile(r"^v(\d+)\.(\d+)\.(\d{2})$")


@dataclass(frozen=True)
class _ParsedVersion:
    major: int
    minor: int
    patch: int | None

    @property
    def line_key(self) -> tuple[int, int]:
        return (self.major, self.minor)


def _parse_replay_version(version: str) -> _ParsedVersion | None:
    if _LEGACY_EPOCH.match(version):
        return None
    m = _PATCH.match(version)
    if m:
        return _ParsedVersion(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _BASE.match(version)
    if m:
        return _ParsedVersion(int(m.group(1)), int(m.group(2)), None)
    return None


def _format_version(major: int, minor: int, patch: int | None) -> str:
    if patch is None:
        return f"v{major}.{minor}"
    return f"v{major}.{minor}.{patch:02d}"


def _replay_versions(existing: tuple[str, ...]) -> list[_ParsedVersion]:
    out: list[_ParsedVersion] = []
    for v in existing:
        parsed = _parse_replay_version(v)
        if parsed is not None:
            out.append(parsed)
    return out


def allocate_version(
    *,
    existing_versions: tuple[str, ...],
    override: str | None,
) -> str:
    used = {v for v in existing_versions if _parse_replay_version(v) is not None}
    if override is not None:
        if _parse_replay_version(override) is None:
            raise ValueError(f"invalid promoted version: {override!r}")
        if override in used:
            raise ValueError(f"promoted version already exists: {override}")
        return override

    replay = _replay_versions(existing_versions)
    if not replay:
        return "v0.2"

    by_line: dict[tuple[int, int], list[_ParsedVersion]] = {}
    for v in replay:
        by_line.setdefault(v.line_key, []).append(v)

    line = max(by_line)
    versions_on_line = by_line[line]
    max_patch = max((v.patch or 0) for v in versions_on_line)
    has_base = any(v.patch is None for v in versions_on_line)

    if not has_base:
        return _format_version(line[0], line[1], max_patch + 1)
    if max_patch == 0:
        return _format_version(line[0], line[1], 1)
    return _format_version(line[0], line[1], max_patch + 1)
