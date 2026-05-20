"""CONTEXT.md glossary terms for cross-repo release (issue #11)."""

from pathlib import Path

CONTEXT = Path(__file__).resolve().parents[1] / "CONTEXT.md"

RELEASE_GLOSSARY_HEADINGS = (
    "Gated promotion",
    "Promoted version",
    "Production latest",
    "Web deployed latest",
    "Dungeon-runner root",
    "TF.js model sync",
    "Two-repo model release",
    "Release smoke",
    "Deployed model version",
    "Promotion manifest",
    "Publish CLI",
)


def test_context_release_glossary_headings_exist() -> None:
    text = CONTEXT.read_text(encoding="utf-8")
    for heading in RELEASE_GLOSSARY_HEADINGS:
        assert f"**{heading}**:" in text, f"missing glossary entry: {heading}"


def test_context_dungeon_runner_root_mentions_env_var() -> None:
    text = CONTEXT.read_text(encoding="utf-8")
    section = text.split("**Dungeon-runner root**:", 1)[1].split("\n\n", 1)[0]
    assert "DUNGEON_RUNNER_ROOT" in section


def test_context_tfjs_sync_distinguishes_production_and_web_latest() -> None:
    text = CONTEXT.read_text(encoding="utf-8")
    sync = text.split("**TF.js model sync**:", 1)[1].split("\n\n", 1)[0]
    assert "production latest" in sync.lower()
    assert "web deployed latest" in sync.lower()
