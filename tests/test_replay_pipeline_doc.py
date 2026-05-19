"""Doc link tests for docs/replay-pipeline.md (issue #10)."""

from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "replay-pipeline.md"


@pytest.fixture(scope="module")
def md() -> str:
    return DOC.read_text(encoding="utf-8")


def test_replay_pipeline_doc_exists() -> None:
    assert DOC.is_file()


def test_links_portfolio_contract_and_issue_128(md: str) -> None:
    assert "CONTRACT.md#replay-envelope-contract-v1" in md
    assert "github.com/enmaku/portfolio-site/issues/128" in md
    assert "$PORTFOLIO_SITE_ROOT/src/features/dungeon-runner/CONTRACT.md" in md


def test_documents_env_vars(md: str) -> None:
    assert "FIREBASE_DATABASE_URL" in md
    assert "PORTFOLIO_SITE_ROOT" in md


def test_documents_all_cli_stages(md: str) -> None:
    for stage in (
        "ingest",
        "verify",
        "eval_suite",
        "eval_config",
        "dataset",
        "bc",
        "ppo",
        "publish",
        "run-all",
    ):
        assert f"`{stage}`" in md


def test_documents_run_all_flags(md: str) -> None:
    assert "--with-ppo" in md
    assert "--with-publish" in md


def test_links_dungeon_runner_issue_10(md: str) -> None:
    assert "github.com/enmaku/dungeon-runner/issues/10" in md


def test_documents_eval_metrics_modules(md: str) -> None:
    assert "## Eval metrics" in md
    assert "replay_metrics" in md
    assert "sim_metrics" in md
    assert "write_metrics" in md


def test_verify_section_links_issue_3(md: str) -> None:
    verify = md.split("## Verify", 1)[1].split("\n## ", 1)[0]
    assert "github.com/enmaku/dungeon-runner/issues/3" in verify


def test_verify_documents_rng_recheck_after_apply_action(md: str) -> None:
    verify = md.split("## Verify", 1)[1].split("\n## ", 1)[0]
    assert "RNG re-check" in verify
    assert "result.state.rng.step" in verify


def test_verify_fixture_table_covers_committed_fixtures(md: str) -> None:
    from tests.replay.helpers import VERIFY_FIXTURE_OUTCOMES

    verify = md.split("## Verify", 1)[1].split("\n## ", 1)[0]
    for fixture, code in VERIFY_FIXTURE_OUTCOMES.items():
        assert fixture in verify
        if code is None:
            assert f"| `{fixture}` | verified |" in verify
        else:
            assert f"| `{fixture}` | `{code}` |" in verify


def test_links_dungeon_runner_issue_10_in_header(md: str) -> None:
    header, _, _ = md.partition("\n---\n")
    assert "github.com/enmaku/dungeon-runner/issues/10" in header


def test_documents_eval_suite_artifact_schema(md: str) -> None:
    start = md.index("## Eval suite init")
    end = md.index("## Eval config init")
    section = md[start:end]
    for field in (
        "suite_version",
        "sampling_seed",
        "created_from_match_ids",
        "val_match_ids",
    ):
        assert field in section
    assert "exit `1`" in section


def test_documents_eval_config_artifact_schema(md: str) -> None:
    start = md.index("## Eval config init")
    end = md.index("## Dataset")
    section = md[start:end]
    for field in (
        "sim_seeds",
        "sim_regression_tolerance",
        "replay_accuracy_floor",
    ):
        assert field in section
    assert "--overwrite" in section
    assert "floor recorder" in section.lower()


def test_documents_eval_split_resolver(md: str) -> None:
    assert "split_resolver" in md or "split tag" in md.lower()
    assert "train" in md and "val" in md


def test_dataset_section_links_issue_4(md: str) -> None:
    dataset = md.split("## Dataset", 1)[1].split("\n## ", 1)[0]
    assert "github.com/enmaku/dungeon-runner/issues/4" in dataset


def test_dataset_documents_parquet_schema(md: str) -> None:
    dataset = md.split("## Dataset", 1)[1].split("\n## ", 1)[0]
    for field in (
        "policy_action_index",
        "is_human",
        "dataset run atomicity",
        "derived/.staging/",
    ):
        assert field in dataset


def test_documents_run_all_orchestration_detail(md: str) -> None:
    section = md.split("## Run-all", 1)[1].split("\n## ", 1)[0]
    assert "fail" in section.lower()
    assert "exit `2`" in section
    assert "FIREBASE_DATABASE_URL" in section
    assert "PORTFOLIO_SITE_ROOT" in section
    assert "eval_config init" in section
    assert "overwrite" in section.lower()
