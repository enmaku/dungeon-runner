"""Cross-repo vocabulary docs stay paired with portfolio-site."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_cross_repo_doc_exists_and_links_portfolio() -> None:
    path = ROOT / "CROSS_REPO.md"
    text = path.read_text(encoding="utf-8")
    assert "portfolio-site" in text
    assert "UBIQUITOUS_LANGUAGE.md" in text


def test_ubiquitous_language_consolidated_glossary() -> None:
    text = (ROOT / "UBIQUITOUS_LANGUAGE.md").read_text(encoding="utf-8")
    assert "portfolio-site/UBIQUITOUS_LANGUAGE.md" in text
    assert "**Gated promotion**" in text
    assert "## Match play" in text


def test_cross_repo_points_at_ubiquitous_language() -> None:
    text = (ROOT / "CROSS_REPO.md").read_text(encoding="utf-8")
    assert "UBIQUITOUS_LANGUAGE.md" in text
