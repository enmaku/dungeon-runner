"""PPO stage orchestration: train → metrics → regression → artifact → gate preview."""

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
from dungeon_runner.replay.bc.stage import LoadModelFn, SimMetricsFn
from dungeon_runner.replay.eval.eval_config import require_eval_config
from dungeon_runner.replay.eval.eval_suite import require_eval_suite
from dungeon_runner.replay.eval.gate_evaluator import evaluate_gates
from dungeon_runner.replay.eval.metrics_writer import load_metrics, write_metrics
from dungeon_runner.replay.eval.replay_metrics import replay_metrics
from dungeon_runner.replay.eval.sim_metrics import sim_metrics
from dungeon_runner.replay.ppo.frozen_teacher import FrozenBCTeacher
from dungeon_runner.replay.ppo.prerequisites import (
    PPOPrerequisiteError,
    check_ppo_prerequisites,
)
from dungeon_runner.replay.ppo.regression import check_ppo_bc_regression
from dungeon_runner.replay.ppo.trainer import PPOTrainResult, train_ppo
from dungeon_runner.rl.model import PolicyValueModel

TrainPPOFn = Callable[..., PPOTrainResult]


class PPOStageError(RuntimeError):
    """PPO stage failed after prerequisites passed."""


@dataclass(frozen=True)
class PPORunSummary:
    run_id: str
    run_dir: Path
    metrics_path: Path
    regression_passed: bool
    gate_preview_passed: bool | None
    gate_preview_reasons: list[str]


def default_ppo_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"ppo-{ts}"


def run_ppo(
    *,
    data_dir: Path,
    bc_run: Path,
    repo_root: Path | None = None,
    run_id: str | None = None,
    bc_anchor_lambda: float = 0.1,
    bc_anchor_beta: float = 0.0,
    use_ray: bool = True,
    ray_workers: int = 8,
    gate_preview: bool = True,
    train_ppo_fn: TrainPPOFn = train_ppo,
    load_model_fn: LoadModelFn | None = None,
    sim_metrics_fn: SimMetricsFn | None = None,
) -> PPORunSummary:
    repo_root = (repo_root or Path.cwd()).resolve()
    data_dir = data_dir.resolve()
    bc_run = bc_run.resolve()
    run_id = run_id or default_ppo_run_id()
    load_model = load_model_fn or load_policy_model
    run_sim_metrics = sim_metrics_fn or sim_metrics

    prereq = check_ppo_prerequisites(
        data_dir,
        bc_run,
        bc_anchor_lambda=bc_anchor_lambda,
        repo_root=repo_root,
    )
    eval_suite = require_eval_suite(data_dir)
    eval_config = require_eval_config(data_dir)
    val_ids = set(eval_suite.val_match_ids)
    bc_metrics = load_metrics(bc_run / "metrics.json")

    train_rows = []
    if bc_anchor_lambda > 0:
        train_rows = load_human_rows(data_dir, split="train")
    val_rows = load_human_rows(data_dir, split="val", val_match_ids=val_ids)

    staging = repo_root / "models" / "runs" / f"{run_id}.tmp"
    final_dir = repo_root / "models" / "runs" / run_id
    if final_dir.exists():
        raise PPOStageError(f"training run artifact already exists: {final_dir}")
    if staging.exists():
        shutil.rmtree(staging)

    tb_dir = staging / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    weights_path = staging / "policy.weights.h5"

    try:
        model = load_model(prereq.bc_weights)
        teacher = FrozenBCTeacher.from_weights(prereq.bc_weights, load_model=load_model)

        train_result = train_ppo_fn(
            model,
            teacher,
            train_rows,
            tb_dir=tb_dir,
            teacher_weights=prereq.bc_weights,
            bc_anchor_lambda=bc_anchor_lambda,
            bc_anchor_beta=bc_anchor_beta,
            use_ray=use_ray,
            ray_workers=ray_workers,
        )
        model.save_weights(str(weights_path))

        latest_weights = repo_root / "models" / "latest" / "policy.weights.h5"
        latest_model = load_model(latest_weights) if latest_weights.is_file() else model
        candidate_predict = make_replay_predict(model)
        latest_predict = make_replay_predict(latest_model)

        replay = replay_metrics(
            candidate_predict,
            latest_predict,
            val_rows,
            val_match_ids=val_ids,
        )
        sim = run_sim_metrics(
            KerasSimPolicy(model),
            KerasSimPolicy(latest_model),
            eval_config.sim_seeds,
        )

        candidate_metrics: dict[str, Any] = {
            "replay": replay.to_dict(),
            "sim": sim.to_dict(),
        }
        regression_passed = check_ppo_bc_regression(
            candidate_metrics,
            bc_metrics,
            epsilon=eval_config.sim_regression_tolerance,
        )

        train_block: dict[str, Any] = {
            "ppo_loss": train_result.ppo_loss,
            "bc_anchor_ce": train_result.bc_anchor_ce,
        }
        if train_result.bc_anchor_kl is not None:
            train_block["bc_anchor_kl"] = train_result.bc_anchor_kl

        write_metrics(
            staging,
            run_id=run_id,
            parent_weights=str(prereq.bc_weights.resolve()),
            replay=replay.to_dict(),
            sim=sim.to_dict(),
            train=train_block,
            ppo_bc_regression={"pass": regression_passed, "bc_run_id": prereq.bc_run_id},
        )

        if final_dir.exists():
            shutil.rmtree(final_dir)
        staging.rename(final_dir)

        gate_passed: bool | None = None
        gate_reasons: list[str] = []
        if gate_preview:
            result = evaluate_gates(
                {
                    "replay": replay.to_dict(),
                    "sim": sim.to_dict(),
                },
                eval_config,
            )
            gate_passed = result.passed
            gate_reasons = list(result.reasons)

        return PPORunSummary(
            run_id=run_id,
            run_dir=final_dir,
            metrics_path=final_dir / "metrics.json",
            regression_passed=regression_passed,
            gate_preview_passed=gate_passed,
            gate_preview_reasons=gate_reasons,
        )
    except PPOPrerequisiteError:
        raise
    except Exception as exc:
        if staging.exists():
            shutil.rmtree(staging)
        if final_dir.exists():
            shutil.rmtree(final_dir)
        if isinstance(exc, PPOStageError):
            raise
        raise PPOStageError(str(exc)) from exc
