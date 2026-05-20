"""BC stage orchestration: train → metrics → artifact → floor → gate preview."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dungeon_runner.replay.bc.human_rows import load_human_rows
from dungeon_runner.replay.bc.predict import (
    KerasSimPolicy,
    load_policy_model,
    make_replay_predict,
)
from dungeon_runner.replay.bc.prerequisites import (
    BCPrerequisiteError,
    check_bc_prerequisites,
)
from dungeon_runner.replay.bc.trainer import BCTrainResult, compute_bc_loss, train_bc
from dungeon_runner.replay.eval.eval_config import require_eval_config
from dungeon_runner.replay.eval.eval_suite import require_eval_suite
from dungeon_runner.replay.eval.floor_recorder import record_floor_if_needed
from dungeon_runner.replay.eval.gate_evaluator import evaluate_gates
from dungeon_runner.replay.eval.metrics_writer import write_metrics
from dungeon_runner.replay.eval.replay_metrics import replay_metrics
from dungeon_runner.replay.eval.sim_metrics import SimMetrics, sim_metrics
from dungeon_runner.replay import progress
from dungeon_runner.rl.model import PolicyValueModel

LoadModelFn = Callable[[Path], PolicyValueModel]
SimMetricsFn = Callable[[Any, Any, list[int]], SimMetrics]


class BCStageError(RuntimeError):
    """BC stage failed after prerequisites passed."""


@dataclass(frozen=True)
class BCRunSummary:
    run_id: str
    run_dir: Path
    metrics_path: Path
    floor_outcome: str | None
    gate_preview_passed: bool | None
    gate_preview_reasons: list[str]


def default_bc_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"bc-{ts}"


def run_bc(
    *,
    data_dir: Path,
    repo_root: Path | None = None,
    run_id: str | None = None,
    gate_preview: bool = True,
    parent_weights: Path | None = None,
    train_bc_fn: Callable[..., BCTrainResult] = train_bc,
    load_model_fn: LoadModelFn | None = None,
    sim_metrics_fn: SimMetricsFn | None = None,
) -> BCRunSummary:
    repo_root = (repo_root or Path.cwd()).resolve()
    data_dir = data_dir.resolve()
    run_id = run_id or default_bc_run_id()
    load_model = load_model_fn or load_policy_model
    run_sim_metrics = sim_metrics_fn or sim_metrics

    prereq = check_bc_prerequisites(data_dir, repo_root, parent_weights=parent_weights)
    eval_suite = require_eval_suite(data_dir)
    eval_config = require_eval_config(data_dir)
    val_ids = set(eval_suite.val_match_ids)

    train_rows = load_human_rows(data_dir, split="train")
    val_rows = load_human_rows(data_dir, split="val", val_match_ids=val_ids)

    staging = repo_root / "models" / "runs" / f"{run_id}.tmp"
    final_dir = repo_root / "models" / "runs" / run_id
    if final_dir.exists():
        raise BCStageError(f"training run artifact already exists: {final_dir}")
    if staging.exists():
        shutil.rmtree(staging)

    tb_dir = staging / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    weights_path = staging / "policy.weights.h5"

    progress.log(
        f"BC {run_id}: {len(train_rows)} train rows, {len(val_rows)} val rows, "
        f"parent={prereq.parent_weights}"
    )
    progress.log_tensorboard(tb_dir, run_label=run_id)

    def _on_epoch_end(epoch: int, val_acc: float) -> None:
        progress.log(f"  epoch {epoch}: val masked accuracy={val_acc:.4f}")

    try:
        model = load_model(prereq.parent_weights)
        train_result = train_bc_fn(
            model,
            train_rows,
            val_rows,
            tb_dir=tb_dir,
            max_epochs=100,
            on_epoch_end=_on_epoch_end,
        )
        progress.log(
            f"  training finished: {train_result.history.epochs} epoch(s), "
            f"best val accuracy={train_result.history.best_val_masked_accuracy:.4f} "
            f"@ epoch {train_result.history.best_epoch}"
        )
        model.save_weights(str(weights_path))

        progress.log("  replay + sim eval…")

        latest_model = load_model(prereq.parent_weights)
        candidate_predict = make_replay_predict(model)
        latest_predict = make_replay_predict(latest_model)

        val_human_rows = load_human_rows(data_dir, split="val", val_match_ids=val_ids)
        replay = replay_metrics(
            candidate_predict,
            latest_predict,
            val_human_rows,
            val_match_ids=val_ids,
        )
        sim = run_sim_metrics(
            KerasSimPolicy(model),
            KerasSimPolicy(latest_model),
            eval_config.sim_seeds,
        )

        write_metrics(
            staging,
            run_id=run_id,
            parent_weights=str(prereq.parent_weights.resolve()),
            replay=replay.to_dict(),
            sim=sim.to_dict(),
            train={"bc_loss": compute_bc_loss(model, train_rows)},
        )

        if final_dir.exists():
            shutil.rmtree(final_dir)
        staging.rename(final_dir)

        floor_outcome: str | None = None
        try:
            floor_outcome = record_floor_if_needed(
                {
                    "replay": replay.to_dict(),
                    "sim": sim.to_dict(),
                },
                data_dir,
            )
        except Exception as exc:
            raise BCStageError(f"floor recorder failed: {exc}") from exc

        gate_passed: bool | None = None
        gate_reasons: list[str] = []
        if gate_preview:
            gate_eval_config = require_eval_config(data_dir)
            result = evaluate_gates(
                {
                    "replay": replay.to_dict(),
                    "sim": sim.to_dict(),
                },
                gate_eval_config,
            )
            gate_passed = result.passed
            gate_reasons = list(result.reasons)

        return BCRunSummary(
            run_id=run_id,
            run_dir=final_dir,
            metrics_path=final_dir / "metrics.json",
            floor_outcome=floor_outcome,
            gate_preview_passed=gate_passed,
            gate_preview_reasons=gate_reasons,
        )
    except BCPrerequisiteError:
        raise
    except Exception as exc:
        if staging.exists():
            shutil.rmtree(staging)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        if isinstance(exc, BCStageError):
            raise
        raise BCStageError(str(exc)) from exc
