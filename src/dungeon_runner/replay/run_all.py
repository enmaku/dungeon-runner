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
from dungeon_runner.replay import progress
from dungeon_runner.replay.ppo import PPOPrerequisiteError, PPOStageError, run_ppo
from dungeon_runner.replay.publish import PublishError, run_publish
from dungeon_runner.replay.verify import pending_verify_ids, run_verify

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
    summary = run_ingest(data_dir=data_dir, from_export=None, database_url=database_url)
    if summary.ingested:
        progress.log(f"  ingested {len(summary.ingested)}: {', '.join(summary.ingested)}")
    if summary.skipped:
        progress.log(f"  skipped {len(summary.skipped)}")
    if not summary.ingested and not summary.skipped:
        progress.log("  no new matches")
    return 0


def _default_verify(data_dir: Path) -> int:
    pending = pending_verify_ids(data_dir)
    if pending:
        progress.log(f"  verifying {len(pending)} pending match(es)…")
    else:
        progress.log("  no pending verify matches")
    summary = run_verify(data_dir=data_dir)
    if summary.verified:
        progress.log(f"  verified {len(summary.verified)}: {', '.join(summary.verified)}")
    if summary.failed:
        progress.log(f"  failed {len(summary.failed)}")
        for entry in summary.failed:
            reason = entry["reason"]
            code = reason.get("code", "?")
            progress.log(f"    {entry['id']}: {code}")
    return 0


def _default_eval_suite_init(data_dir: Path) -> int:
    try:
        artifact = init_eval_suite(data_dir)
    except EvalSuiteError as exc:
        progress.log(f"  {exc}")
        return 1
    progress.log(
        f"  suite v{artifact.suite_version}: "
        f"{len(artifact.val_match_ids)} val / "
        f"{len(artifact.created_from_match_ids)} verified"
    )
    return 0


def _default_eval_config_init(data_dir: Path) -> int:
    if load_eval_config(data_dir) is not None:
        progress.log("  skipped (eval_config.json already exists)")
        return 0
    try:
        artifact = init_eval_config(data_dir)
    except EvalConfigError as exc:
        progress.log(f"  {exc}")
        return 1
    progress.log(
        f"  {len(artifact.sim_seeds)} sim seeds, "
        f"ε={artifact.sim_regression_tolerance}, floor={artifact.replay_accuracy_floor!r}"
    )
    return 0


def _default_dataset(data_dir: Path) -> int:
    try:
        summary = run_dataset(data_dir=data_dir, encode_all=False)
    except (DatasetBuildError, RuntimeError) as exc:
        progress.log(f"  {exc}")
        return 1
    if summary.built:
        progress.log(f"  built {len(summary.built)}: {', '.join(summary.built)}")
    else:
        progress.log("  no pending dataset matches")
    return 0


def _default_bc(data_dir: Path) -> tuple[int, Path | None]:
    try:
        summary = run_bc(data_dir=data_dir, gate_preview=True)
    except BCPrerequisiteError as exc:
        progress.log(f"  {exc}")
        return 1, None
    except BCStageError as exc:
        progress.log(f"  {exc}")
        return 1, None
    except ImportError as exc:
        progress.log(f"  BC requires TensorFlow: pip install -e \".[train]\" ({exc})")
        return 1, None
    progress.log(f"  artifact: {summary.run_dir}")
    if summary.floor_outcome:
        progress.log(f"  floor recorder: {summary.floor_outcome}")
    if summary.gate_preview_passed is not None:
        status = "pass" if summary.gate_preview_passed else "fail"
        progress.log(f"  gate preview: {status}")
    return 0, summary.run_dir


def _default_ppo(data_dir: Path, bc_run: Path) -> tuple[int, Path | None]:
    progress.log(f"  bc-run: {bc_run}")
    try:
        summary = run_ppo(data_dir=data_dir, bc_run=bc_run, gate_preview=True)
    except PPOPrerequisiteError as exc:
        progress.log(f"  {exc}")
        return 1, None
    except PPOStageError as exc:
        progress.log(f"  {exc}")
        return 1, None
    except ImportError as exc:
        progress.log(f"  PPO requires TensorFlow: pip install -e \".[train]\" ({exc})")
        return 1, None
    progress.log(f"  artifact: {summary.run_dir}")
    reg = "pass" if summary.regression_passed else "fail"
    progress.log(f"  ppo bc regression: {reg}")
    if summary.gate_preview_passed is not None:
        status = "pass" if summary.gate_preview_passed else "fail"
        progress.log(f"  gate preview: {status}")
    if not summary.regression_passed:
        return 1, summary.run_dir
    return 0, summary.run_dir


def _default_publish(data_dir: Path, run_dir: Path) -> int:
    progress.log(f"  run: {run_dir}")
    try:
        summary = run_publish(run_dir=run_dir, data_dir=data_dir)
    except PublishError as exc:
        if exc.reasons:
            progress.log(f"  publish failed: {', '.join(exc.reasons)}")
        else:
            progress.log(f"  {exc}")
        return 1
    progress.log(f"  promoted → {summary.promoted_version} ({summary.version_dir})")
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

    train_tail = "bc"
    if with_ppo:
        train_tail += " → ppo"
    if with_publish:
        train_tail += " → publish"

    progress.log("Replay pipeline run-all")
    progress.log(f"  data-dir: {data_dir}")
    progress.log(f"  train path: {train_tail}")

    step_labels = [
        "ingest",
        "verify",
        "eval_suite init",
        "eval_config init",
        "dataset",
        "bc (behavioral cloning)",
    ]
    if with_ppo:
        step_labels.append("ppo (BC-anchored fine-tuning)")
    if with_publish:
        step_labels.append("publish (gated promotion)")
    total = len(step_labels)
    step_i = 0

    def _header(label: str) -> None:
        nonlocal step_i
        step_i += 1
        progress.log_step(f"[{step_i}/{total}] {label}")

    _header(step_labels[0])
    if stages.ingest(data_dir) != 0:
        progress.log_failed(step_labels[0], 1)
        return 1
    progress.log_done()

    _header(step_labels[1])
    if stages.verify(data_dir) != 0:
        progress.log_failed(step_labels[1], 1)
        return 1
    progress.log_done()

    _header(step_labels[2])
    if stages.eval_suite_init(data_dir) != 0:
        progress.log_failed(step_labels[2], 1)
        return 1
    progress.log_done()

    _header(step_labels[3])
    if load_eval_config(data_dir) is None:
        if stages.eval_config_init(data_dir) != 0:
            progress.log_failed(step_labels[3], 1)
            return 1
    else:
        progress.log("  skipped (eval_config.json already exists)")
    progress.log_done()

    _header(step_labels[4])
    if stages.dataset(data_dir) != 0:
        progress.log_failed(step_labels[4], 1)
        return 1
    progress.log_done()

    _header(step_labels[5])
    bc_code, train_run_dir = stages.bc(data_dir)
    if bc_code != 0 or train_run_dir is None:
        progress.log_failed(step_labels[5], bc_code if bc_code != 0 else 1)
        return bc_code if bc_code != 0 else 1
    progress.log_done()

    if with_ppo:
        _header(step_labels[6])
        ppo_code, ppo_run_dir = stages.ppo(data_dir, train_run_dir)
        if ppo_code != 0 or ppo_run_dir is None:
            progress.log_failed(step_labels[6], ppo_code if ppo_code != 0 else 1)
            return ppo_code if ppo_code != 0 else 1
        train_run_dir = ppo_run_dir
        progress.log_done()

    if with_publish:
        publish_label = step_labels[-1]
        _header(publish_label)
        code = stages.publish(data_dir, train_run_dir)
        if code != 0:
            progress.log_failed(publish_label, code)
            return code
        progress.log_done()

    progress.log("")
    progress.log("run-all finished successfully.")
    if with_publish:
        progress.log("  Next: sync TF.js weights from portfolio-site (see docs/replay-pipeline.md).")
    return 0
