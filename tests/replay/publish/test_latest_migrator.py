"""One-time migration: legacy duplicate latest/ dir → symlink to v0.1.30a."""

from __future__ import annotations

from dungeon_runner.replay.publish.latest_migrator import migrate_latest_symlink


def test_migrates_duplicate_latest_dir_to_v0_1_30a(tmp_path):
    legacy = tmp_path / "v0.1.30a"
    legacy.mkdir()
    (legacy / "policy.weights.h5").write_bytes(b"legacy")

    latest = tmp_path / "latest"
    latest.mkdir()
    (latest / "policy.weights.h5").write_bytes(b"dup")

    migrate_latest_symlink(tmp_path)

    assert latest.is_symlink()
    assert latest.resolve() == legacy.resolve()


def test_migrator_idempotent_when_symlink_already_points_at_legacy(tmp_path):
    legacy = tmp_path / "v0.1.30a"
    legacy.mkdir()
    latest = tmp_path / "latest"
    latest.symlink_to("v0.1.30a")

    migrate_latest_symlink(tmp_path)

    assert latest.is_symlink()
    assert latest.resolve() == legacy.resolve()
