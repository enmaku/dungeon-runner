"""promoted_at validation."""

from __future__ import annotations

import pytest

from dungeon_runner.replay.publish.stage import PublishError, validate_promoted_at


def test_validate_promoted_at_accepts_z_suffix():
    assert validate_promoted_at("2026-06-01T00:00:00Z") == "2026-06-01T00:00:00Z"


def test_validate_promoted_at_rejects_empty():
    with pytest.raises(PublishError, match="non-empty"):
        validate_promoted_at("   ")


def test_validate_promoted_at_rejects_invalid():
    with pytest.raises(PublishError, match="invalid promoted_at"):
        validate_promoted_at("not-a-date")
