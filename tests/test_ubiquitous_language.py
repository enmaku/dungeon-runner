"""UBIQUITOUS_LANGUAGE.md stays consolidated and cross-linked."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UL = ROOT / "UBIQUITOUS_LANGUAGE.md"


def test_ubiquitous_language_exists_and_links_sibling() -> None:
    text = UL.read_text(encoding="utf-8")
    assert "portfolio-site/UBIQUITOUS_LANGUAGE.md" in text
    assert "**Match**" in text
    assert "**Web game engine**" in text
    assert "**Production latest**" in text
    assert "**Web deployed latest**" in text


def test_empty_pile_divergence_documented() -> None:
    text = UL.read_text(encoding="utf-8")
    assert "EMPTY_DUNGEON_FORFEIT" in text or "forfeit" in text.lower()
    assert "web game engine" in text.lower()


def test_catalog_ambiguity_resolved() -> None:
    text = UL.read_text(encoding="utf-8")
    assert "game data catalog" in text.lower()
    assert "model catalog" in text.lower()


def test_human_player_seat_defined() -> None:
    text = UL.read_text(encoding="utf-8")
    assert "**Human player seat**" in text
    assert "**Human step**" in text
    assert "**Non-NN history step**" in text


def test_pipeline_gates_section() -> None:
    text = UL.read_text(encoding="utf-8")
    assert "## Pipeline gates" in text
    assert "**Ingest eligibility**" in text
    assert "**Verified replay**" in text
    assert "exhaustive training glossary" in text


def test_epic_v1_success_bar_defined() -> None:
    text = UL.read_text(encoding="utf-8")
    assert "**Epic v1 success bar**" in text
    assert "gated promotion" in text.lower()


def test_presentation_pace_in_replay_section() -> None:
    text = UL.read_text(encoding="utf-8")
    assert "**Presentation pace**" in text
    assert "presentationSpeedProfile" in text


def test_ppo_training_opponents_section() -> None:
    text = UL.read_text(encoding="utf-8")
    assert "## PPO training opponents" in text
    assert "**Learner**" in text
    assert "**BC-bot**" in text
    assert "**Frozen BC teacher**" in text


def test_evaluation_and_promotion_section() -> None:
    text = UL.read_text(encoding="utf-8")
    assert "## Evaluation & promotion" in text
    assert "**Frozen eval suite**" in text
    assert "**Promotion gates**" in text
    assert "**Sim eval metrics**" in text


def test_randombot_canonical_term() -> None:
    text = UL.read_text(encoding="utf-8")
    assert "**Randombot**" in text
    assert "| **Randombot** |" in text


def test_match_over_distinct_from_sim_forfeit() -> None:
    text = UL.read_text(encoding="utf-8")
    assert "**Match over**" in text
    assert "**Sim empty-pile forfeit**" in text
    assert "match over" in text.lower()
