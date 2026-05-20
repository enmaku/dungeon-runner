"""Doc link tests for docs/replay-pipeline.md (issue #10)."""

from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "replay-pipeline.md"
ENV_EXAMPLE = DOC.parents[1] / ".env.example"
README = DOC.parents[1] / "README.md"

INGEST_SKIP_REASONS = (
    "unsupported_version",
    "missing_seed",
    "missing_setup",
    "missing_history",
    "invalid_presentation_speed",
    "invalid_history",
)


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


def test_ppo_section_links_issue_7(md: str) -> None:
    ppo = md.split("## BC-anchored PPO", 1)[1].split("\n## ", 1)[0]
    assert "github.com/enmaku/dungeon-runner/issues/7" in ppo


def test_ppo_documents_ray_and_legacy_scripts(md: str) -> None:
    ppo = md.split("## BC-anchored PPO", 1)[1].split("\n## ", 1)[0]
    assert "Ray rollout pool" in ppo
    assert "scripts/train.py" in ppo
    assert "scripts/train_rllib.py" in ppo
    assert "rollout_collector" in ppo
    assert "ray_workers" in ppo
    assert "--no-ray" in ppo


def test_documents_run_all_orchestration_detail(md: str) -> None:
    section = md.split("## Run-all", 1)[1].split("\n## ", 1)[0]
    assert "fail" in section.lower()
    assert "FIREBASE_DATABASE_URL" in section
    assert "PORTFOLIO_SITE_ROOT" in section
    assert "eval_config init" in section
    assert "overwrite" in section.lower()
    assert "exit `2`" not in section


def test_ingest_documents_rtdb_shallow_query(md: str) -> None:
    ingest = md.split("## Ingest", 1)[1].split("\n## ", 1)[0]
    assert "dungeonRunnerCompletedMatches.json?shallow=true" in ingest


def test_ingest_skip_reason_table_covers_all_codes(md: str) -> None:
    ingest = md.split("## Ingest", 1)[1].split("\n## ", 1)[0]
    table = ingest.split("#### Skip reason codes", 1)[1].split("\n### ", 1)[0]
    for code in INGEST_SKIP_REASONS:
        assert f"`{code}`" in table


def test_ingest_documents_intentional_strictness(md: str) -> None:
    ingest = md.split("## Ingest", 1)[1].split("\n## ", 1)[0]
    assert "#### Intentional strictness" in ingest
    assert "type(x) is int" in ingest


def test_eval_metrics_section_links_issue_5(md: str) -> None:
    section = md.split("## Eval metrics", 1)[1].split("\n## ", 1)[0]
    assert "github.com/enmaku/dungeon-runner/issues/5" in section


def test_bc_section_links_issue_6(md: str) -> None:
    bc = md.split("## BC policy training", 1)[1].split("\n## ", 1)[0]
    assert "github.com/enmaku/dungeon-runner/issues/6" in bc


def test_publish_section_links_issue_8(md: str) -> None:
    publish = md.split("## Publish (gated promotion)", 1)[1].split("\n## ", 1)[0]
    assert "github.com/enmaku/dungeon-runner/issues/8" in publish


def test_related_links_portfolio_context_and_adr_0001(md: str) -> None:
    related = md.split("## Related", 1)[1]
    assert "$PORTFOLIO_SITE_ROOT/CONTEXT.md" in related
    assert "github.com/enmaku/portfolio-site/blob/main/CONTEXT.md" in related
    assert "adr/0001-web-game-engine-authoritative.md" in related


def test_env_example_documents_both_pipeline_vars() -> None:
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert "FIREBASE_DATABASE_URL" in text
    assert "PORTFOLIO_SITE_ROOT" in text
    assert "ingest" in text.lower()
    assert "verify" in text.lower() or "dataset" in text.lower()


def test_readme_has_replay_training_pipeline_section() -> None:
    text = README.read_text(encoding="utf-8")
    assert "## Replay training pipeline" in text
    assert "docs/replay-pipeline.md" in text
    assert "CONTEXT.md" in text


def test_stage_map_references_web_sync_issue_11(md: str) -> None:
    header, _, _ = md.partition("\n---\n")
    assert "Web sync" in header
    assert "github.com/enmaku/dungeon-runner/issues/11" in header
    assert "TF.js" in header


def test_ingest_documents_manifest_atomicity(md: str) -> None:
    ingest = md.split("## Ingest", 1)[1].split("\n## ", 1)[0]
    assert "### Ingest manifest and atomicity" in ingest
    assert "manifest.json" in ingest


def _release_section(md: str) -> str:
    return md.split("## Release to portfolio-site", 1)[1].split("\n## ", 1)[0]


def test_release_section_links_issue_11(md: str) -> None:
    release = _release_section(md)
    assert "github.com/enmaku/dungeon-runner/issues/11" in release


def test_release_section_documents_two_repo_handoff(md: str) -> None:
    release = _release_section(md)
    assert "two-repo model release" in release
    assert "TF.js model sync" in release
    assert "promotion manifest" in release


def test_release_section_links_portfolio_model_release_doc(md: str) -> None:
    release = _release_section(md)
    assert "$PORTFOLIO_SITE_ROOT/scripts/MODEL_RELEASE.md" in release
    assert (
        "github.com/enmaku/portfolio-site/blob/main/scripts/MODEL_RELEASE.md"
        in release
    )


def test_release_section_links_portfolio_issue_127(md: str) -> None:
    release = _release_section(md)
    assert "github.com/enmaku/portfolio-site/issues/127" in release


def test_release_section_points_to_context_glossary(md: str) -> None:
    release = _release_section(md)
    assert "CONTEXT.md" in release
    assert "web deployed latest" in release
    assert "production latest" in release
    assert "DUNGEON_RUNNER_ROOT" in release


def test_release_section_does_not_defer_sync_to_future(md: str) -> None:
    release = _release_section(md)
    assert "when that slice ships" not in release
