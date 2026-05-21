"""Cross-repo vocabulary docs stay paired with portfolio-site."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cross_repo_doc_exists_and_links_portfolio() -> None:
    path = ROOT / "CROSS_REPO.md"
    text = path.read_text(encoding="utf-8")
    assert "portfolio-site" in text
    assert "CROSS_REPO.md" in text
    assert "do not translate" in text.lower() or "not translated" in text.lower()


def test_ubiquitous_language_points_at_context_and_cross_repo() -> None:
    text = (ROOT / "UBIQUITOUS_LANGUAGE.md").read_text(encoding="utf-8")
    assert "CONTEXT.md" in text
    assert "CROSS_REPO.md" in text
