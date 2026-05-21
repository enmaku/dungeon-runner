"""run-all orchestrator: stage order, flags, fail-fast."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from dungeon_runner.replay.run_all import RunAllStages, run_all

BcFn = Callable[[Path], tuple[int, Path | None]]
PpoFn = Callable[[Path, Path], tuple[int, Path | None]]
PublishFn = Callable[[Path, Path], int]
StageFn = Callable[[Path], int]


def _noop(_data_dir: Path) -> int:
    return 0


def _bc_ok(run_dir: Path) -> BcFn:
    def _fn(_data_dir: Path) -> tuple[int, Path | None]:
        return 0, run_dir

    return _fn


def _ppo_ok(run_dir: Path) -> PpoFn:
    def _fn(_data_dir: Path, bc_run: Path) -> tuple[int, Path | None]:
        return 0, run_dir

    return _fn


def _stages(**overrides: object) -> RunAllStages:
    defaults: dict[str, object] = {
        "ingest": _noop,
        "verify": _noop,
        "eval_suite_init": _noop,
        "eval_config_init": _noop,
        "dataset": _noop,
        "bc": lambda _d: (0, None),
        "ppo": lambda _d, _b: (0, None),
        "publish": lambda _d, _r: 0,
    }
    defaults.update(overrides)
    return RunAllStages(**defaults)  # type: ignore[arg-type]


def test_run_all_default_chain_stops_after_bc(tmp_path: Path) -> None:
    order: list[str] = []

    def track(name: str) -> StageFn:
        def _fn(_data_dir: Path) -> int:
            order.append(name)
            return 0

        return _fn

    code = run_all(
        data_dir=tmp_path,
        stages=_stages(
            ingest=track("ingest"),
            verify=track("verify"),
            eval_suite_init=track("eval_suite_init"),
            eval_config_init=track("eval_config_init"),
            dataset=track("dataset"),
            bc=lambda _d: (order.append("bc"), (0, tmp_path / "bc-run"))[1],
        ),
    )
    assert code == 0
    assert order == [
        "ingest",
        "verify",
        "eval_suite_init",
        "eval_config_init",
        "dataset",
        "bc",
    ]


def test_run_all_with_ppo_and_publish(tmp_path: Path) -> None:
    order: list[str] = []
    bc_dir = tmp_path / "models" / "runs" / "bc-fixture"
    ppo_dir = tmp_path / "models" / "runs" / "ppo-fixture"
    bc_dir.mkdir(parents=True)
    ppo_dir.mkdir(parents=True)

    def track(name: str) -> StageFn:
        def _fn(_data_dir: Path) -> int:
            order.append(name)
            return 0

        return _fn

    def ppo(data_dir: Path, bc_run: Path) -> tuple[int, Path | None]:
        order.append(f"ppo:{bc_run.name}")
        assert bc_run == bc_dir
        return 0, ppo_dir

    def publish(data_dir: Path, run_dir: Path) -> int:
        order.append(f"publish:{run_dir.name}")
        assert run_dir == ppo_dir
        return 0

    code = run_all(
        data_dir=tmp_path,
        with_ppo=True,
        with_publish=True,
        stages=_stages(
            ingest=track("ingest"),
            verify=track("verify"),
            eval_suite_init=track("eval_suite_init"),
            eval_config_init=track("eval_config_init"),
            dataset=track("dataset"),
            bc=_bc_ok(bc_dir),
            ppo=ppo,
            publish=publish,
        ),
    )
    assert code == 0
    assert order[-2:] == ["ppo:bc-fixture", "publish:ppo-fixture"]


def test_run_all_publish_uses_bc_when_no_ppo(tmp_path: Path) -> None:
    bc_dir = tmp_path / "bc-run"
    bc_dir.mkdir()
    published: list[Path] = []

    def publish(_data_dir: Path, run_dir: Path) -> int:
        published.append(run_dir)
        return 0

    run_all(
        data_dir=tmp_path,
        with_publish=True,
        stages=_stages(bc=_bc_ok(bc_dir), publish=publish),
    )
    assert published == [bc_dir]


def test_run_all_fail_fast(tmp_path: Path) -> None:
    order: list[str] = []

    code = run_all(
        data_dir=tmp_path,
        stages=_stages(
            ingest=lambda _: (order.append("ingest") or 0),
            verify=lambda _: (order.append("verify") or 1),
            dataset=lambda _: (order.append("dataset") or 0),
        ),
    )
    assert code == 1
    assert order == ["ingest", "verify"]


def test_run_all_skips_eval_suite_init_when_artifact_exists(
    tmp_path: Path,
) -> None:
    from dungeon_runner.replay.eval.eval_suite import init_eval_suite
    from tests.replay.helpers import seed_verify_state

    seed_verify_state(tmp_path, verified=["match-a", "match-b", "match-c"])
    init_eval_suite(tmp_path, sampling_seed=42)
    called = False

    def eval_suite_init(_data_dir: Path) -> int:
        nonlocal called
        called = True
        return 0

    run_all(data_dir=tmp_path, stages=_stages(eval_suite_init=eval_suite_init))
    assert called is False


def test_run_all_skips_eval_config_init_when_artifact_exists(
    tmp_path: Path,
) -> None:
    from dungeon_runner.replay.eval.eval_config import eval_config_path

    eval_config_path(tmp_path).write_text(
        json.dumps(
            {
                "sim_seeds": [0],
                "sim_regression_tolerance": 0.01,
                "replay_accuracy_floor": None,
            }
        ),
        encoding="utf-8",
    )
    called = False

    def eval_config_init(_data_dir: Path) -> int:
        nonlocal called
        called = True
        return 0

    run_all(data_dir=tmp_path, stages=_stages(eval_config_init=eval_config_init))
    assert called is False


def test_cli_run_all_passes_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run_all(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr("dungeon_runner.replay.cli.run_all", fake_run_all)
    from dungeon_runner.replay.cli import main

    assert (
        main(
            [
                "run-all",
                "--data-dir",
                str(tmp_path),
                "--with-ppo",
                "--with-publish",
            ]
        )
        == 0
    )
    assert captured["data_dir"] == tmp_path.resolve()
    assert captured["with_ppo"] is True
    assert captured["with_publish"] is True
