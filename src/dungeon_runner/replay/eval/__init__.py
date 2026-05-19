"""Eval suite, metrics, and promotion gate contract (issue #5)."""

from dungeon_runner.replay.eval.eval_config import (
    EvalConfigArtifact,
    init_eval_config,
    load_eval_config,
    require_eval_config,
)
from dungeon_runner.replay.eval.eval_suite import (
    EvalSuiteArtifact,
    init_eval_suite,
    load_eval_suite,
    require_eval_suite,
)
from dungeon_runner.replay.eval.derived_store import (
    DerivedStoreError,
    ParquetDerivedRow,
    load_derived_rows,
    load_match_rows,
)
from dungeon_runner.replay.eval.gate_evaluator import GateResult, evaluate_gates
from dungeon_runner.replay.eval.metrics_writer import load_metrics, write_metrics
from dungeon_runner.replay.eval.replay_metrics import ReplayMetrics, replay_metrics
from dungeon_runner.replay.eval.sim_metrics import (
    SimMetrics,
    sim_metrics,
    sim_passes_regression,
)
from dungeon_runner.replay.eval.split_resolver import split_for

__all__ = [
    "DerivedStoreError",
    "EvalConfigArtifact",
    "EvalSuiteArtifact",
    "GateResult",
    "ParquetDerivedRow",
    "ReplayMetrics",
    "SimMetrics",
    "evaluate_gates",
    "load_derived_rows",
    "load_match_rows",
    "load_metrics",
    "replay_metrics",
    "sim_metrics",
    "sim_passes_regression",
    "write_metrics",
    "init_eval_config",
    "init_eval_suite",
    "load_eval_config",
    "load_eval_suite",
    "require_eval_config",
    "require_eval_suite",
    "split_for",
]
