"""run-all orchestrator: chained replay pipeline stages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dungeon_runner.replay.bc import BCPrerequisiteError, BCStageError, run_bc
from dungeon_runner.replay.dataset import DatasetBuildError, run_dataset
from dungeon_runner.replay.env import require_database_url
from dungeon_runner.replay.eval.eval_config import (
    EvalConfigError,
    init_eval_config,
    load_eval_config,
)
from dungeon_runner.replay.eval.eval_suite import EvalSuiteError, init_eval_suite
from dungeon_runner.replay.ingest import run_ingest
from dungeon_runner.replay.ppo import PPOPrerequisiteError, PPOStageError, run_ppo
from dungeon_runner.replay.publish import PublishError, run_publish
from dungeon_runner.replay.verify import run_verify

StageFn = Callable[[Path], int]
BcFn = Callable[[Path], tuple[int, Path | None]]
PpoFn = Callable[[Path, Path], tuple[int, Path | None]]
PublishFn = Callable[[Path, Path], int]


@dataclass(frozen=True)
class RunAllStages:
    ingest: StageFn
    verify: StageFn
    eval_suite_init: StageFn
    eval_config_init: StageFn
    dataset: StageFn
    bc: BcFn
    ppo: PpoFn
    publish: PublishFn


def _default_ingest(data_dir: Path) -> int:
    database_url = require_database_url()
    run_ingest(data_dir=data_dir, from_export=None, database_url=database_url)
    return 0


def _default_verify(data_dir: Path) -> int:
    run_verify(data_dir=data_dir)
    return 0


def _default_eval_suite_init(data_dir: Path) -> int:
    try:
        init_eval_suite(data_dir)
    except EvalSuiteError:
        return 1
    return 0


def _default_eval_config_init(data_dir: Path) -> int:
    if load_eval_config(data_dir) is not None:
        return 0
    try:
        init_eval_config(data_dir)
    except EvalConfigError:
        return 1
    return 0


def _default_dataset(data_dir: Path) -> int:
    try:
        run_dataset(data_dir=data_dir, encode_all=False)
    except (DatasetBuildError, RuntimeError):
        return 1
    return 0


def _default_bc(data_dir: Path) -> tuple[int, Path | None]:
    try:
        summary = run_bc(data_dir=data_dir, gate_preview=True)
    except (BCPrerequisiteError, BCStageError, ImportError):
        return 1, None
    return 0, summary.run_dir


def _default_ppo(data_dir: Path, bc_run: Path) -> tuple[int, Path | None]:
    try:
        summary = run_ppo(data_dir=data_dir, bc_run=bc_run, gate_preview=True)
    except (PPOPrerequisiteError, PPOStageError, ImportError):
        return 1, None
    if not summary.regression_passed:
        return 1, None
    return 0, summary.run_dir


def _default_publish(data_dir: Path, run_dir: Path) -> int:
    try:
        run_publish(run_dir=run_dir, data_dir=data_dir)
    except PublishError:
        return 1
    return 0


def default_run_all_stages() -> RunAllStages:
    return RunAllStages(
        ingest=_default_ingest,
        verify=_default_verify,
        eval_suite_init=_default_eval_suite_init,
        eval_config_init=_default_eval_config_init,
        dataset=_default_dataset,
        bc=_default_bc,
        ppo=_default_ppo,
        publish=_default_publish,
    )


def run_all(
    *,
    data_dir: Path,
    with_ppo: bool = False,
    with_publish: bool = False,
    stages: RunAllStages | None = None,
) -> int:
    data_dir = data_dir.resolve()
    stages = stages or default_run_all_stages()

    for step in (
        stages.ingest,
        stages.verify,
        stages.eval_suite_init,
    ):
        code = step(data_dir)
        if code != 0:
            return code

    if load_eval_config(data_dir) is None:
        code = stages.eval_config_init(data_dir)
        if code != 0:
            return code

    code = stages.dataset(data_dir)
    if code != 0:
        return code

    bc_code, train_run_dir = stages.bc(data_dir)
    if bc_code != 0:
        return bc_code
    if train_run_dir is None:
        return 1

    if with_ppo:
        ppo_code, ppo_run_dir = stages.ppo(data_dir, train_run_dir)
        if ppo_code != 0:
            return ppo_code
        if ppo_run_dir is None:
            return 1
        train_run_dir = ppo_run_dir

    if with_publish:
        return stages.publish(data_dir, train_run_dir)

    return 0
