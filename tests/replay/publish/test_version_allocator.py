"""Promoted version semver allocator (ADR 0002)."""

from __future__ import annotations

import pytest

from dungeon_runner.replay.publish.version_allocator import allocate_version


def test_empty_promoted_line_returns_v0_2():
    assert allocate_version(existing_versions=(), override=None) == "v0.2"


def test_ignores_legacy_v0_1_epoch_dirs():
    assert allocate_version(
        existing_versions=("v0.1.29a", "v0.1.30a"),
        override=None,
    ) == "v0.2"


def test_after_v0_2_returns_v0_2_01():
    assert allocate_version(existing_versions=("v0.2",), override=None) == "v0.2.01"


def test_after_v0_2_01_returns_v0_2_02():
    assert allocate_version(
        existing_versions=("v0.2", "v0.2.01"),
        override=None,
    ) == "v0.2.02"


def test_manual_minor_override_honored():
    assert allocate_version(
        existing_versions=("v0.2", "v0.2.01"),
        override="v0.3",
    ) == "v0.3"


def test_manual_override_rejects_used_version():
    with pytest.raises(ValueError, match="already exists"):
        allocate_version(existing_versions=("v0.3",), override="v0.3")


def test_after_manual_v0_3_returns_v0_3_01():
    assert allocate_version(existing_versions=("v0.3",), override=None) == "v0.3.01"
